#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.modules.glob_coast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Super-Module that generates a high-quality Coastline Mask.
Merges Vectors (NHD, OSM) and Rasters (Copernicus, GMRT) into a unified product using weighted voting.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from rasterio.warp import reproject
from pyogrio.raw import read
from shapely import from_wkb

from fetchez import core, cli, utils
from fetchez.hooks.extract import Extract
from fetchez.hooks.fn_filter import FilenameFilter
from fetchez.registry import ModuleRegistry
from fetchez.modules import FetchModule

# from globato.hooks.tools.osm_landmask import OSMLandmask
from globato.hooks.rasters.polygonize import RasterPolygonizeHook

logger = logging.getLogger(__name__)

_VECTOR_EXTS = {".shp", ".gpkg", ".geojson", ".json"}
_RASTER_EXTS = {".tif", ".nc", ".vrt"}


@cli.cli_opts(
    help_text="Generate a High-Resolution Coastline Mask raster.",
    res="Target resolution (e.g. '1s', '30m')",
    sources="Comma-separated sources (default: copernicus,nhd,osm_landmask,hydrolakes). "
    "Use 'osm_water' in place of 'osm_landmask' to also carve OSM lakes/rivers out of the land vote.",
)
class GlobCoast(FetchModule):
    """Synthesizes a coastline raster from multiple sources.

    Uses 'Weighted Voting' to resolve conflicts (e.g. NHD water overrides Copernicus land).
    """

    name = "glob_coast"
    meta_desc = "Fetch and glob a coastline"
    meta_agency = "Globato"
    meta_tags = ["global", "globato", "coastline", "landmask"]
    meta_category = "Tools"
    meta_resolution = "Varies"
    meta_license = "N/A"

    def __init__(
        self, res="1s", sources=None, weights=None, fill_inland_holes=False, **kwargs
    ):
        super().__init__(name="glob_coast", **kwargs)

        # Default Hierarchy:
        # High-Res Hydrography (NHD, HydroLakes)
        # High-Res DEM (Copernicus/NASADEM)
        # Vector Coastline (OSM)
        # Background (GMRT)
        if not sources:
            self.source_list = ["nhd", "osm_landmask", "copernicus", "gmrt"]
        else:
            self.source_list = sources.split(",")

        self.res_val = utils.str2inc(res)

        w, e, s, n = self.region if self.region else (0, 0, 0, 0)
        self.out_fn = os.path.join(self._outdir, f"coastline_{w}_{s}_{res}.tif")

        # Voting Weights
        self.weights = {
            "nhd": -10.0,
            "hydrolakes": -10.0,
            "copernicus": 5.0,
            "nasadem": 5.0,
            "wsf": 5.0,
            "osm_landmask": 5.0,
            # osm_water is osm_landmask with include_water=True: lake/river areas are
            # carved out of the land polygons, so they receive no land vote instead of +5.
            "osm_water": 5.0,
            "gmrt": 0.1,
            "gebco": 0.1,
        }
        if weights:
            self.weights.update(weights)

        # We need to fix this hook, it creates too many artifacts currently...
        # Sieve the raster to remove salt & pepper ocean noise
        # sieve_cls = self.add_hook(RasterSieveHook())
        # if sieve_cls:
        #     self.add_hook(sieve_cls(chunk="full", size=2))

        # Convert the cleaned raster to vector polygons
        self.add_hook(RasterPolygonizeHook(target_value=1))

        if fill_inland_holes:
            try:
                from globato.hooks.vectors.fill_holes import VectorFillHoles

                self.add_hook(VectorFillHoles(min_area=10.0))
            except ImportError:
                logger.warning(
                    "VectorFillHoles hook not found. Ponds will not be filled."
                )

    def _init_grid(self):
        """Initialize the empty voting grid based on the region and resolution."""

        w, e, s, n = self.region
        self.width = int((e - w) / self.res_val)
        self.height = int((n - s) / self.res_val)
        self.transform = from_origin(w, n, self.res_val, self.res_val)
        self.grid = np.zeros((self.height, self.width), dtype=np.float32)

    def _apply_source_votes(self, fetched_files, weight):
        """Apply voting for all files belonging to one source.

        Multiple overlapping tiles from the same source would each apply the
        full weight to shared pixels if processed individually, inflating the
        effective vote.  This method merges all tiles into a single per-source
        vote before modifying self.grid, so each pixel is counted exactly once
        per source regardless of how many tiles cover it.

        Vectors: all geometries from all files are collected and rasterized
        together, producing a single binary coverage mask.

        Rasters: tiles are merged using first-valid-wins so that overlapping
        regions are counted only once.
        """

        valid_files = [f for f in fetched_files if f and os.path.exists(f)]
        if not valid_files:
            return

        w, e, s, n = self.region

        all_geoms = []
        raster_files = []
        for f_path in valid_files:
            ext = os.path.splitext(f_path)[1].lower()
            if ext in _VECTOR_EXTS or f_path.lower().endswith(".gdb"):
                try:
                    region_geom = self.region.to_shapely()
                    meta, fids, geometry_wkb, fields = read(f_path)
                    raw_geoms = from_wkb(geometry_wkb)
                    all_geoms.extend(
                        [g for g in raw_geoms if region_geom.intersects(g)]
                    )
                except Exception as exc:
                    logger.warning("Vector read failed for %s: %s", f_path, exc)
            elif ext in _RASTER_EXTS:
                raster_files.append(f_path)

        # --- Vectors: rasterize all geometries at once ---
        if all_geoms:
            try:
                mask = rasterize(
                    all_geoms,
                    out_shape=(self.height, self.width),
                    transform=self.transform,
                    default_value=1,
                    dtype=np.uint8,
                )
                if weight > 0:
                    self.grid[mask == 1] += weight
                else:
                    self.grid[mask == 1] -= abs(weight)
            except Exception as exc:
                logger.warning("Vector rasterization failed: %s", exc)

        # --- Rasters: merge tiles (first-valid-wins), apply weight once ---
        if raster_files:
            # source_vote: +1 = land, -1 = water, 0 = not yet covered
            source_vote = np.zeros((self.height, self.width), dtype=np.float32)
            covered = np.zeros((self.height, self.width), dtype=bool)

            for f_path in raster_files:
                try:
                    buffer = np.full(
                        (self.height, self.width), np.nan, dtype=np.float32
                    )

                    with rasterio.open(f_path) as src:
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=buffer,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=self.transform,
                            dst_crs=rasterio.crs.CRS.from_user_input(
                                self.region.srs
                            ),  # rasterio.crs.CRS.from_epsg(4326),
                            src_nodata=src.nodata,
                            dst_nodata=np.nan,
                            resampling=Resampling.nearest,
                        )
                    new_pixels = ~np.isnan(buffer) & ~covered
                    if not np.any(new_pixels):
                        continue
                    # Z > 0 is Land (+1), Z <= 0 is Water (-1)
                    source_vote[new_pixels & (buffer > 0)] = 1.0
                    source_vote[new_pixels & (buffer <= 0)] = -1.0
                    covered |= new_pixels
                except Exception as exc:
                    logger.warning("Raster processing failed for %s: %s", f_path, exc)

            if np.any(covered):
                self.grid[covered] += source_vote[covered] * weight

    def _finalize(self):
        """Convert voting grid to binary mask and save to TIFF."""

        # Vote > 0 is Land (1), Vote <= 0 is Water (0)
        final_mask = (self.grid > 0).astype(np.uint8)

        profile = {
            "driver": "GTiff",
            "height": self.height,
            "width": self.width,
            "count": 1,
            "dtype": "uint8",
            "crs": self.region.srs,  # "EPSG:4326",
            "transform": self.transform,
            "compress": "lzw",
            "nodata": None,
        }

        with rasterio.open(self.out_fn, "w", **profile) as dst:
            dst.write(final_mask, 1)

    def run(self):
        """Fetch sources and generate the coastline mask."""

        if not self.region:
            logger.error("GlobCoast requires a region.")
            return

        self._init_grid()

        fetch_region = self.region.copy().buffer(pct=5)

        for mod_name in self.source_list:
            fetched_files = []
            weight = self.weights.get(mod_name, 0.1)

            # if mod_name == "osm_landmask":
            #     landmask_fn = os.path.join(
            #         self._outdir, f"temp_landmask_{w}_{s}.geojson"
            #     )
            #     osm_hook = OSMLandmask(filename=landmask_fn)

            #     mock_entries = [(self, {"dst_fn": "dummy"})]
            #     osm_hook.run(mock_entries)

            #     if os.path.exists(landmask_fn):
            #         fetched_files.append(landmask_fn)

            if mod_name == "nhd":
                mod_cls = ModuleRegistry.get_class("tnm")
                mod_instance = mod_cls(
                    src_region=fetch_region,
                    datasets="14",
                    extents="'HU-8 Subbasin,HU-4 Subregion'",
                    outdir=os.path.join(self._outdir, "sources", mod_name),
                )
                mod_instance.add_hook(FilenameFilter(match="GDB", stage="pre"))
                mod_instance.add_hook(Extract())

                try:
                    logger.info(
                        "[NHD] Querying USGS National Map for NHD hydrography data..."
                    )
                    mod_instance.run()
                    n_found = len(mod_instance.results)
                    logger.info(f"[NHD] Found {n_found} dataset(s). Fetching...")
                    final = core.run_fetchez([mod_instance])
                    if final:
                        fetched_files.extend(
                            [
                                e.get("dst_fn")
                                for _, e in final
                                if isinstance(e, dict) and e.get("dst_fn")
                            ]
                        )
                    else:
                        fetched_files.extend(
                            [
                                entry.get("dst_fn")
                                if isinstance(entry, dict)
                                else entry[1]
                                for entry in mod_instance.results
                            ]
                        )
                    logger.info(f"[NHD] {n_found} dataset(s) ready.")
                except Exception as e:
                    logger.error(f"[NHD] Failed to fetch NHD data: {e}")

            else:
                # osm_water reuses the osm_landmask module with include_water=True.
                # It gets its own outdir so its cache doesn't collide with osm_landmask.
                registry_name = "osm_landmask" if mod_name == "osm_water" else mod_name
                mod_cls = ModuleRegistry.get_class(registry_name)
                if not mod_cls:
                    logger.warning(f"Unknown source: {mod_name}")
                    continue

                extra_kwargs = (
                    {"include_water": True} if mod_name == "osm_water" else {}
                )
                mod_instance = mod_cls(
                    src_region=fetch_region,
                    outdir=os.path.join(self._outdir, "sources", mod_name),
                    **extra_kwargs,
                )
                mod_instance.add_hook(Extract())

                # osm_landmask and osm_water print their own [OSM] status lines; skip here.
                tag = mod_name.upper()
                log_progress = mod_name not in ("osm_landmask", "osm_water")

                try:
                    mod_instance.run()
                    n_found = len(mod_instance.results)

                    if log_progress:
                        if n_found == 0:
                            logger.info(f"[{tag}] No datasets found for region.")
                        else:
                            logger.info(
                                f"[{tag}] Found {n_found} dataset(s). Fetching..."
                            )

                    final = core.run_fetchez([mod_instance])
                    if final:
                        fetched_files.extend(
                            [
                                e.get("dst_fn")
                                for _, e in final
                                if isinstance(e, dict) and e.get("dst_fn")
                            ]
                        )
                    else:
                        fetched_files.extend(
                            [
                                entry.get("dst_fn")
                                if isinstance(entry, dict)
                                else entry[1]
                                for entry in mod_instance.results
                            ]
                        )

                    if log_progress and n_found > 0:
                        logger.info(f"[{tag}] {n_found} dataset(s) ready.")
                except Exception as e:
                    logger.error(f"[{tag}] Failed to fetch {mod_name} data: {e}")

            logger.debug(
                f"Voting: {len(fetched_files)} file(s) as '{mod_name}' (Weight: {weight})"
            )
            self._apply_source_votes(fetched_files, weight)

        self._finalize()

        self.results = []
        if os.path.exists(self.out_fn):
            self.add_entry_to_results(
                url=f"file://{self.out_fn}",
                dst_fn=self.out_fn,
                data_type="coastline_mask",
            )

        return self
