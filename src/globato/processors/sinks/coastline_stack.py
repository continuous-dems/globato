#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.sinks.coastline_stack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Accumulates disparate sources into a binary Land/Water mask using weighted voting.
"""

import os
import logging
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from rasterio.warp import reproject
from fetchez.hooks import FetchHook
from fetchez.utils import str2inc

logger = logging.getLogger(__name__)

class CoastlineStack(FetchHook):
    """Weighted Voting Stacker for Coastline Generation."""

    name = "coastline_stack"
    stage = "post"
    category = "sink"

    def __init__(self, output, res, region, weights=None, polygonize=True, **kwargs):
        super().__init__(**kwargs)
        self.output = output
        self.res = str2inc(res)
        self.region = region
        self.polygonize = polygonize

        self.weights = {
            'nhd': -10.0,
            'hydrolakes': -10.0,
            'copernicus': 5.0,
            'nasadem': 5.0,
            'wsf': 5.0,
            'osm_landmask': 5.0,
            'gmrt': 0.1,
            'gebco': 0.1
        }
        if weights:
            self.weights.update(weights)

    def _init_grid(self):
        w, e, s, n = self.region
        self.width = int((e - w) / self.res)
        self.height = int((n - s) / self.res)
        self.transform = from_origin(w, n, self.res, self.res)
        self.grid = np.zeros((self.height, self.width), dtype=np.float32)

    def _process_raster(self, src_path, weight):
        """Warp raster to grid and apply voting logic."""

        with rasterio.open(src_path) as src:
            buffer = np.full((self.height, self.width), np.nan, dtype=np.float32)

            reproject(
                source=rasterio.band(src, 1),
                destination=buffer,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=self.transform,
                dst_crs=rasterio.crs.CRS.from_epsg(4326),
                src_nodata=src.nodata,
                dst_nodata=np.nan, # Crucial
                resampling=Resampling.nearest
            )

            valid_mask = ~np.isnan(buffer)

            if not np.any(valid_mask):
                return

            # Z > 0 is Land (+1), Z <= 0 is Water (-1)
            vote_grid = np.zeros_like(buffer)

            # Pixels that are Land
            land_mask = valid_mask & (buffer > 0)
            vote_grid[land_mask] = 1.0

            # Pixels that are Water (Valid but <= 0)
            water_mask = valid_mask & (buffer <= 0)
            vote_grid[water_mask] = -1.0

            if weight > 0:
                self.grid[valid_mask] += (vote_grid[valid_mask] * weight)
            else:
                # Negative weight usually implies "This is Water" (e.g. Lakes)
                # We subtract the absolute weight wherever data exists
                # But typically for 'lake' rasters, they might be 1=Lake, 0=Background
                # If buffer has 1s for lakes, we subtract weight there.

                # Check if this is a binary mask raster (0/1) or elevation
                # If elevation (min < -50), treat normally.
                # If mask (min >= 0), assume 1 = Feature.
                if np.nanmin(buffer) >= 0:
                    # Binary mask assumption (Lake=1)
                    feature_mask = valid_mask & (buffer > 0)
                    self.grid[feature_mask] -= abs(weight)
                else:
                    # Elevation assumption
                    self.grid[valid_mask] += (vote_grid[valid_mask] * weight)

    def _process_vector(self, src_path, weight):
        """Rasterize vector to grid and apply voting logic."""

        import fiona
        try:
            with fiona.open(src_path) as src:
                geoms = [f['geometry'] for f in src]

            if not geoms: return

            # Rasterize: 1 where polygon exists, 0 otherwise
            mask = rasterize(
                geoms,
                out_shape=(self.height, self.width),
                transform=self.transform,
                default_value=1,
                dtype=np.uint8
            )

            # Apply Vote
            if weight > 0:
                self.grid[mask == 1] += weight
            else:
                self.grid[mask == 1] -= abs(weight)

        except Exception as e:
            logger.warning(f"Vector processing failed for {src_path}: {e}")

    def run(self, entries):
        self._init_grid()

        for mod, entry in entries:
            dst_fn = entry.get('dst_fn')
            if not dst_fn or not os.path.exists(dst_fn): continue

            mod_name = entry.get('data_type', '').lower()
            weight = self.weights.get(mod_name, 0.1)

            logger.info(f"Voting: {os.path.basename(dst_fn)} as '{mod_name}' (Weight: {weight})")

            ext = os.path.splitext(dst_fn)[1].lower()
            if ext in ['.tif', '.nc', '.vrt']:
                self._process_raster(dst_fn, weight)
            elif ext in ['.shp', '.gpkg', '.geojson', '.json']:
                self._process_vector(dst_fn, weight)

        self._finalize()
        return entries

    def _finalize(self):
        """Convert voting grid to binary mask and save."""
        # Binary Rule: Vote > 0 is Land (1), Vote <= 0 is Water (0)
        final_mask = (self.grid > 0).astype(np.uint8)

        profile = {
            'driver': 'GTiff',
            'height': self.height,
            'width': self.width,
            'count': 1,
            'dtype': 'uint8',
            'crs': 'EPSG:4326',
            'transform': self.transform,
            'compress': 'lzw',
            'nodata': None
        }

        with rasterio.open(self.output, 'w', **profile) as dst:
            dst.write(final_mask, 1)

        if self.polygonize:
            self._write_vectors(final_mask)

    def _write_vectors(self, mask):
        from rasterio.features import shapes
        import fiona

        out_vec = self.output.replace('.tif', '.gpkg')
        schema = {'geometry': 'Polygon', 'properties': {'val': 'int'}}

        with fiona.open(out_vec, 'w', driver='GPKG', crs='EPSG:4326', schema=schema) as dst:
            for geom, val in shapes(mask, transform=self.transform):
                if val == 1:
                    dst.write({
                        'geometry': geom,
                        'properties': {'val': 1}
                    })
