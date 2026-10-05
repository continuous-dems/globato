#!/usr/bin/env python

"""
globato.hooks.rasters.natural_neighbor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uses naturalneighbor to interpolate sparse grids.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging

import numpy as np
import rasterio
from rasterio.transform import xy

from .base import RasterGlobalHook

try:
    import naturalneighbor

    HAS_NATURALNEIGHBOR = True
except ImportError:
    HAS_NATURALNEIGHBOR = False

logger = logging.getLogger(__name__)


class NaturalNeighborSurface(RasterGlobalHook):
    """Interpolates gaps using Natural Neighbor (Discrete Sibson)."""

    name = "interp_nn"
    default_suffix = "_nn"
    meta_desc = "Intpolate NoData voids using Natural Neighbor."
    meta_tags = ["globato", "interpolation", "multi-stack"]

    def __init__(self, max_points=50000, **kwargs):
        super().__init__(**kwargs)
        self.max_points = int(max_points)

    def process_raster(self, src_path, dst_path, entry):
        if not HAS_NATURALNEIGHBOR:
            logger.error("'naturalneighbor' package not installed.")
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
                indices = np.random.choice(len(z_vals), self.max_points, replace=False)
                x_vals = np.array(x_vals)[indices]
                y_vals = np.array(y_vals)[indices]
                z_vals = z_vals[indices]

            input_points = np.column_stack((x_vals, y_vals, z_vals))
            w, s, e, n = src.bounds

            grid_ranges = [[w, e, src.res[0]], [s, n, src.res[1]]]

            try:
                logger.info("Executing Natural Neighbor interpolation...")
                z_chunk = naturalneighbor.griddata(input_points, grid_ranges)
                z_chunk = np.flipud(z_chunk)  # Correct Y orientation

                # Ensure dimensions match exactly
                if z_chunk.shape[0] != src.height or z_chunk.shape[1] != src.width:
                    z_chunk = z_chunk[: src.height, : src.width]

                z_chunk = np.nan_to_num(z_chunk, nan=nodata)

                profile = src.profile.copy()
                profile.update(dtype=rasterio.float32, nodata=nodata, count=1)

                with rasterio.open(dst_path, "w", **profile) as dst:
                    dst.write(z_chunk.astype(rasterio.float32), 1)

                return True
            except Exception as e:
                logger.error(f"[NaturalNeighbor] Failed: {e}")
                return False
