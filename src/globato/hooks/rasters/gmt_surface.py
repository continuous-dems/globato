#!/usr/bin/env python

"""
globato.hooks.rasters.gmt_surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uses GMT's 'surface' algorithm (Continuous Curvature Splines in Tension)
via PyGMT to interpolate sparse grids. Essential for deep water/large gaps.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging

import numpy as np
import rasterio
from rasterio.transform import xy

from ..rasters.base import RasterGlobalHook

try:
    import pygmt

    HAS_PYGMT = True
except (ImportError, OSError):
    HAS_PYGMT = False

logger = logging.getLogger(__name__)


class GmtSurface(RasterGlobalHook):
    """Interpolates a sparse raster using GMT Surface (Splines in Tension).

    This is a Global Operator (process_raster), not a Chunk Operator,
    because splines require global context to resolve tension correctly.

    Args:
        tension (float): Spline tension [0-1]. 0=Minimum Curvature (Smooth), 1=Harmonic (Sharp). Default 0.35.
        convergence (float): Convergence limit. Default 1e-4.
        radius (str/float): Search radius for valid data.
        upper (str/float): Upper limit of ouput solution.
        verbose (bool): Add verbosity to pygmt
    """

    name = "interp_gmt"
    default_suffix = "_gmt"
    meta_desc = "Intpolate NoData voids using GMT surface."
    meta_tags = ["globato", "interpolation", "multi-stack"]

    def __init__(
        self,
        tension=0.35,
        convergence=1e-4,
        radius=None,
        gmt_upper=None,
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tension = float(tension)
        self.convergence = float(convergence)
        self.radius = radius
        self.gmt_upper = gmt_upper
        self.verbose = verbose

    def _validate_deps(self):
        if not HAS_PYGMT:
            return (
                False,
                "PyGMT is not installed. Please run: conda install -c conda-forge pygmt",
            )
        return True, ""

    def process_raster(self, src_path, dst_path, entry):
        if not HAS_PYGMT:
            logger.error("[GmtSurface] PyGMT not installed. Cannot run surface.")
            return False

        with rasterio.open(src_path) as src:
            data = src.read(1)
            nodata = src.nodata
            if nodata is None:
                nodata = -9999

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

            # Shift the GMT region bounds inward by half a cell.
            # This forces GMT's native gridline nodes to perfectly align with our pixel centers.
            w_shift = w + (x_inc / 2.0)
            e_shift = e - (x_inc / 2.0)
            s_shift = s + (y_inc / 2.0)
            n_shift = n - (y_inc / 2.0)

            region_str = f"{w_shift}/{e_shift}/{s_shift}/{n_shift}"
            spacing_str = f"{x_inc}/{y_inc}"

            logger.info(
                f"[GmtSurface] Gridding {len(z_vals)} points via PyGMT (Gridline workaround)..."
            )

            try:
                grid = pygmt.surface(
                    x=np.array(x_vals),
                    y=np.array(y_vals),
                    z=z_vals,
                    region=region_str,
                    spacing=spacing_str,
                    tension=self.tension,
                    convergence=self.convergence,
                    upper=self.gmt_upper,
                    # Run natively in gridline to avoid GMT pixel-registration bugs
                    registration="gridline",
                    verbose=self.verbose,
                )

                result_arr = grid.values
                result_arr = np.flipud(result_arr)

                # Because we shifted the GMT bounds by half a cell, the output array dimensions
                # perfectly match our original pixel-registered rasterio profile!
                profile = src.profile.copy()
                profile.update(dtype=rasterio.float32, nodata=nodata, count=1)

                with rasterio.open(dst_path, "w", **profile) as dst:
                    dst.write(result_arr.astype(rasterio.float32), 1)

                return True

            except Exception as e:
                logger.error(f"[GmtSurface] PyGMT failed: {e}")
                return False
