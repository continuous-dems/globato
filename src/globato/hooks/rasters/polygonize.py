#!/usr/bin/env python

"""
globato.hooks.rasters.polygonize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Polygonize the raster.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import os

import numpy as np
import rasterio
import shapely
from fetchez.hooks import FetchHook
from pyogrio.raw import write
from rasterio.features import shapes

logger = logging.getLogger(__name__)


# Update to convert a 'stream' from 'raster-stream'?
class RasterPolygonizeHook(FetchHook):
    """Converts a raster (like a binary mask or classified grid) into vector polygons."""

    name = "raster_polygonize"
    meta_desc = "Convert a raster into vector polygons."
    meta_category = "raster-op"
    meta_stage = "post"

    def __init__(self, target_value=None, output=None, format="GPKG", **kwargs):
        super().__init__(**kwargs)

        self.target_value = float(target_value) if target_value is not None else None

        self.output = output
        self.format = format.upper()

    def run(self, entries):
        """Iterate through the pipeline entries and polygonize valid rasters."""

        new_entries = []
        for mod, entry in entries:
            src_fn = entry.get("dst_fn")

            if (
                not src_fn
                or not os.path.exists(src_fn)
                or not src_fn.lower().endswith(".tif")
            ):
                new_entries.append((mod, entry))
                continue

            if self.output:
                dst_fn = self.output
            else:
                base = os.path.splitext(src_fn)[0]
                ext = ".gpkg" if self.format == "GPKG" else ".shp"
                dst_fn = f"{base}_poly{ext}"

            logger.debug(
                f"Polygonizing {os.path.basename(src_fn)} -> {os.path.basename(dst_fn)}"
            )

            try:
                success = self._polygonize(src_fn, dst_fn)
                if success:
                    # Update the entry to point to the new vector file
                    entry["src_fn"] = src_fn
                    entry["dst_fn"] = dst_fn
                    entry.setdefault("artifacts", {})[self.name] = dst_fn
            except Exception as e:
                logger.error(f"Failed to polygonize {src_fn}: {e}")

            new_entries.append((mod, entry))

        return new_entries

    def _polygonize(self, src_path, dst_path):
        """Performs the actual rasterio -> pyogrio extraction."""

        with rasterio.open(src_path) as src:
            image = src.read(1)
            transform = src.transform
            crs_str = src.crs.to_string() if src.crs else "EPSG:4326"

            if self.target_value is not None:
                mask = image == self.target_value
            else:
                if src.nodata is not None:
                    mask = image != src.nodata
                else:
                    mask = ~np.isnan(image)

            if not np.any(mask):
                logger.debug(f"No matching pixels found to polygonize in {src_path}")
                return False

            geoms = []
            vals = []

            # Extract the raw GeoJSON dictionaries and values from rasterio
            for s, v in shapes(image, mask=mask, transform=transform):
                geoms.append(shapely.geometry.shape(s))
                vals.append(v)

            if not geoms:
                return False

            geometry_wkb = shapely.to_wkb(geoms)
            dtype = "float64" if image.dtype.kind == "f" else "int32"
            field_data = [np.array(vals, dtype=dtype)]
            driver = "GPKG" if self.format == "GPKG" else "ESRI Shapefile"
            write(
                dst_path,
                geometry_wkb,
                field_data,
                fields=["val"],
                geometry_type="Polygon",
                crs=crs_str,
                driver=driver,
            )

        return True
