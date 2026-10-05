#!/usr/bin/env python

"""
globato.hooks.rasters.kriging_surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uses pykrige to interpolate sparse grids.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging

import numpy as np
import rasterio
from rasterio.transform import xy

from .base import RasterGlobalHook

try:
    from pykrige.ok import OrdinaryKriging

    HAS_PYKRIGE = True
except ImportError:
    HAS_PYKRIGE = False

logger = logging.getLogger(__name__)


class KrigingSurface(RasterGlobalHook):
    """Interpolates gaps using Ordinary Kriging (PyKrige)."""

    name = "interp_krige"
    default_suffix = "_krige"
    meta_desc = "Intpolate NoData voids using Kriging."
    meta_tags = ["globato", "interpolation", "multi-stack"]

    def __init__(self, model="linear", nlags=6, max_points=10000, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.nlags = int(nlags)
        self.max_points = int(max_points)

    def process_raster(self, src_path, dst_path, entry):
        if not HAS_PYKRIGE:
            logger.error("PyKrige not installed. Cannot run Kriging interpolation.")
            return False

        with rasterio.open(src_path) as src:
            data = src.read(1)
            nodata = src.nodata if src.nodata is not None else -9999
            valid_mask = (data != nodata) & (~np.isnan(data))

            if not np.any(valid_mask):
                return False

            rows, cols = np.where(valid_mask)
            z_vals = data[rows, cols]
            x_vals, y_vals = xy(src.transform, rows, cols)

            if len(z_vals) > self.max_points:
                logger.info(
                    f"Decimating {len(z_vals)} training points down to {self.max_points}..."
                )
                indices = np.random.choice(len(z_vals), self.max_points, replace=False)
                x_vals = np.array(x_vals)[indices]
                y_vals = np.array(y_vals)[indices]
                z_vals = z_vals[indices]

            logger.info(f"Training Ordinary Kriging ({self.model})...")
            OK = OrdinaryKriging(
                x_vals,
                y_vals,
                z_vals,
                variogram_model=self.model,
                nlags=self.nlags,
                enable_plotting=False,
            )

            # Generate grid coordinates
            grid_cols, grid_rows = np.meshgrid(
                np.arange(src.width), np.arange(src.height)
            )
            grid_x, grid_y = xy(src.transform, grid_rows.flatten(), grid_cols.flatten())

            # Predict
            z_pred, _ = OK.execute(
                "points", grid_x, grid_y, backend="loop", n_closest_points=10
            )
            result_arr = z_pred.reshape(src.height, src.width)
            result_arr = np.nan_to_num(result_arr, nan=nodata)

            profile = src.profile.copy()
            profile.update(dtype=rasterio.float32, nodata=nodata, count=1)

            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.write(result_arr.astype(rasterio.float32), 1)

            return True
