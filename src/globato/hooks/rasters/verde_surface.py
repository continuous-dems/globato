#!/usr/bin/env python

"""
globato.hooks.rasters.verde_surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uses Verde's Biharmonic Spline algorithm to interpolate sparse grids.
An excellent, pure-Python alternative to GMT's continuous curvature splines.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging

import numpy as np
import rasterio
from rasterio.transform import xy

from ..rasters.base import RasterGlobalHook

# Lazy import verde so it remains an optional dependency
try:
    import verde as vd

    HAS_VERDE = True
except ImportError:
    HAS_VERDE = False

logger = logging.getLogger(__name__)


class VerdeSurface(RasterGlobalHook):
    """Interpolates a sparse raster using Verde's Biharmonic Splines.

    This is a Global Operator (overrides process_raster), as splines
    require global context to resolve tension/damping correctly.

    Args:
        damping (float): Controls the smoothness/tension. 0=sharp, higher=smoother.
                         (Acts similarly to GMT's tension parameter).
        mindist (float): Minimum distance between data points. Helps condition the matrix.
    """

    name = "interp_verde"
    default_suffix = "_verde"
    meta_desc = "Intpolate NoData voids using Verde."
    meta_tags = ["globato", "interpolation", "multi-stack"]

    def __init__(self, damping=1e-4, mindist=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.damping = float(damping)
        self.mindist = float(mindist)

    def process_raster(self, src_path, dst_path, entry):
        if not HAS_VERDE:
            logger.error(
                "[VerdeSurface] 'verde' package not installed. Cannot interpolate."
            )
            return False

        with rasterio.open(src_path) as src:
            data = src.read(1)
            nodata = src.nodata
            if nodata is None:
                nodata = -9999

            valid_mask = (data != nodata) & (~np.isnan(data))

            if not np.any(valid_mask):
                logger.warning(f"[VerdeSurface] No valid data in {src_path}. Skipping.")
                return False

            rows, cols = np.where(valid_mask)
            z_vals = data[rows, cols]

            x_vals, y_vals = xy(src.transform, rows, cols)
            x_vals = np.array(x_vals)
            y_vals = np.array(y_vals)

            w, s, e, n = src.bounds
            x_inc = src.res[0]
            y_inc = src.res[1]

            logger.info(
                f"[VerdeSurface] Loaded {len(z_vals)} points. Conditioning data..."
            )

            reducer = vd.BlockReduce(reduction=np.median, spacing=(y_inc, x_inc))
            coords, reduced_z = reducer.filter((x_vals, y_vals), z_vals)

            logger.info(
                f"[VerdeSurface] Fitting Biharmonic Spline to {len(reduced_z)} decimated points..."
            )

            try:
                spline = vd.Spline(damping=self.damping, mindist=self.mindist)
                spline.fit(coords, reduced_z)

                grid = spline.grid(
                    region=(w, e, s, n), spacing=(y_inc, x_inc), pixel_register=True
                )

                result_arr = grid.scalars.values[: src.height, : src.width]
                result_arr = np.flipud(result_arr)

                profile = src.profile.copy()
                profile.update(dtype=rasterio.float32, nodata=nodata)

                with rasterio.open(dst_path, "w", **profile) as dst:
                    dst.write(result_arr.astype(rasterio.float32), 1)

                logger.info(f"[VerdeSurface] Interpolation complete: {dst_path}")
                return True

            except Exception as e:
                logger.error(f"[VerdeSurface] Spline fitting failed: {e}")
                return False
