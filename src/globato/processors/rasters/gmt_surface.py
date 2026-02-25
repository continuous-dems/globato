#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.interpolators.gmt_surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Uses GMT's 'surface' algorithm (Continuous Curvature Splines in Tension)
via PyGMT to interpolate sparse grids. Essential for deep water/large gaps.
"""

import os
import logging
import numpy as np
import rasterio
from rasterio.transform import xy

from ..rasters.base import RasterHook

try:
    import pygmt
    HAS_PYGMT = True
except ImportError:
    HAS_PYGMT = False

logger = logging.getLogger(__name__)


class GmtSurface(RasterHook):
    """Interpolates a sparse raster using GMT Surface (Splines in Tension).

    This is a Global Operator (process_raster), not a Chunk Operator,
    because splines require global context to resolve tension correctly.

    Args:
        tension (float): Spline tension [0-1]. 0=Minimum Curvature (Smooth), 1=Harmonic (Sharp). Default 0.35.
        convergence (float): Convergence limit. Default 1e-4.
        radius (str/float): Search radius for valid data.
    """

    name = "interp_gmt"
    default_suffix = "_gmt"

    def __init__(self, tension=0.35, convergence=1e-4, radius=None, **kwargs):
        super().__init__(**kwargs)
        self.tension = float(tension)
        self.convergence = float(convergence)
        self.radius = radius

    def process_raster(self, src_path, dst_path, entry):
        if not HAS_PYGMT:
            logger.error("[GmtSurface] PyGMT not installed. Cannot run surface.")
            return False

        with rasterio.open(src_path) as src:
            data = src.read(1)
            nodata = src.nodata
            if nodata is None: nodata = -9999

            valid_mask = (data != nodata) & (~np.isnan(data))

            if not np.any(valid_mask):
                logger.warning(f"[GmtSurface] No valid data in {src_path}. Skipping.")
                return False

            rows, cols = np.where(valid_mask)
            z_vals = data[rows, cols]
            x_vals, y_vals = xy(src.transform, rows, cols)
            # maybe use x/y values directly from src (if a multi-stack).

            w, s, e, n = src.bounds
            x_inc = src.res[0]
            y_inc = src.res[1]

            region_str = f"{w}/{e}/{s}/{n}"
            spacing_str = f"{x_inc}/{y_inc}"

            logger.info(f"[GmtSurface] Gridding {len(z_vals)} points via PyGMT...")

            try:
                grid = pygmt.surface(
                    x=np.array(x_vals),
                    y=np.array(y_vals),
                    z=z_vals,
                    region=region_str,
                    spacing=spacing_str,
                    tension=self.tension,
                    convergence=self.convergence,
                    # Optional: lower/upper limits if bathy constraints known
                    # verbose="q"
                )
                result_arr = grid.values
                result_arr = np.flipud(result_arr)

                profile = src.profile.copy()
                profile.update(dtype=rasterio.float32, nodata=nodata)

                with rasterio.open(dst_path, "w", **profile) as dst:
                    dst.write(result_arr.astype(rasterio.float32), 1)

                return True

            except Exception as e:
                logger.error(f"[GmtSurface] PyGMT failed: {e}")
                return False
