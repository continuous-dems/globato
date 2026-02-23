#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.filters.vector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Filters points using OGR Vector Data (Shapefile/GeoPackage).
Optimized by rasterizing vectors to a boolean mask on-the-fly.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
import numpy as np
from fetchez import utils
from .base import GlobatoFilter
from .reference import RasterSampling

try:
    from osgeo import gdal
    from osgeo import ogr

    HAS_OSGEO = True
except ImportError:
    HAS_OSGEO = False

logger = logging.getLogger(__name__)


class VectorMask(GlobatoFilter, RasterSampling):
    """
    Filters points based on a Vector Polygon mask (e.g. Landmask).

    Strategy:
        1. Rasterizes the vector onto a temp grid matching the data region.
        2. Uses fast raster sampling to filter the stream.

    Args:
        vector (str): Path to vector file (.shp, .gpkg).
        burn_value (int): Value to burn into mask (default 1).
        invert (bool): If True, keep points OUTSIDE the polygons.
        res (float): Resolution of the internal mask (default 0.0001 ~10m).
    """

    name = "vector_mask"

    def __init__(self, vector=None, invert=False, res=0.0001, **kwargs):
        super().__init__(invert=invert, **kwargs)
        self.vector_fn = vector
        self.res = float(res)
        self.mask_fn = None

    def setup(self, mod, entry):
        """Prepare the raster mask from the vector file."""

        if not HAS_OSGEO:
            logger.error("OSGEO required for vector processing.")
            return False

        if not self.vector_fn or not os.path.exists(self.vector_fn):
            logger.warning(f"[VectorMask] File not found: {self.vector_fn}")
            return False

        if not getattr(mod, "region", None):
            return False

        # Hash the region to allow reuse if multiple modules share the same area
        r_hash = hash(tuple(mod.region))
        self.mask_fn = os.path.join(
            mod._outdir,
            f"mask_{os.path.basename(self.vector_fn)}_{r_hash}.tif"
        )

        if os.path.exists(self.mask_fn):
            return True

        # --- Rasterize Vector to TIF ---
        logger.info(f"[VectorMask] Rasterizing {self.vector_fn} to {self.mask_fn}...")

        w, e, s, n = mod.region
        width = int((e - w) / self.res)
        height = int((n - s) / self.res)

        if width * height > 1e9:
            logger.warning("[VectorMask] Mask too large! Increasing resolution step.")
            width //= 2
            height //= 2

        try:
            driver = gdal.GetDriverByName("GTiff")
            ds = driver.Create(self.mask_fn, width, height, 1, gdal.GDT_Byte,
                               options=["COMPRESS=LZW", "TILED=YES"])
            ds.SetGeoTransform([w, self.res, 0, n, 0, -self.res])
            ds.SetProjection("EPSG:4326") # Assuming WGS84 for now

            vec_ds = ogr.Open(self.vector_fn)
            lyr = vec_ds.GetLayer()

            gdal.RasterizeLayer(ds, [1], lyr, burn_values=[1])
            ds = None # Save/Close
            return True

        except Exception as e:
            logger.error(f"[VectorMask] Rasterization failed: {e}")
            return False

    def filter_chunk(self, chunk):
        if not self.mask_fn:
            return None

        vals = self.sample_raster(self.mask_fn, chunk, default_val=0)
        is_inside = (vals == 1)

        return is_inside

    def teardown(self):
        if self.mask_fn and os.path.exists(self.mask_fn):
            try:
                os.remove(self.mask_fn)
            except Exception:
                pass
