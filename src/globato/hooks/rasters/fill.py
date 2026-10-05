#!/usr/bin/env python

"""
globato.hooks.rasters.fill
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Fill NoData voids using Rasterio's discrete Poisson PDE solver (GDALFillNodata).
Requires global context to smoothly interpolate large gaps and extrapolate to edges.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging

import numpy as np
import rasterio
from rasterio.fill import fillnodata

from .base import RasterGlobalHook

logger = logging.getLogger(__name__)


class RasterFill(RasterGlobalHook):
    """Fill NoData voids using a discrete Poisson solver.

    Usage: --hook raster_fill:max_dist=1000:smoothing=3
    """

    name = "raster_fill"
    default_suffix = "_filled"
    meta_desc = "Fill NoData voids in a raster."
    meta_tags = ["globato", "interpolation", "multi-stack"]

    def __init__(self, max_dist=1000, smoothing=3, **kwargs):
        super().__init__(**kwargs)
        self.max_dist = float(max_dist)
        self.smoothing = int(smoothing)

    def process_raster(self, src_path, dst_path, entry):
        with rasterio.open(src_path) as src:
            data = src.read(1)
            nodata = src.nodata
            if nodata is None:
                nodata = -9999

            # 1 = Valid, 0 = Nodata
            is_float = data.dtype.kind == "f"
            if is_float:
                valid_mask = (data != nodata) & (~np.isnan(data))
            else:
                valid_mask = data != nodata

            if not np.any(valid_mask):
                logger.warning(f"[RasterFill] No valid data in {src_path}. Skipping.")
                return False

            logger.info(
                f"[RasterFill] Interpolating/extrapolating voids using Poisson solver (Max Dist: {self.max_dist})..."
            )

            try:
                filled_arr = fillnodata(
                    image=data,
                    mask=valid_mask,
                    max_search_distance=self.max_dist,
                    smoothing_iterations=self.smoothing,
                )

                profile = src.profile.copy()
                profile.update(dtype=rasterio.float32, nodata=nodata, count=1)

                with rasterio.open(dst_path, "w", **profile) as dst:
                    dst.write(filled_arr.astype(rasterio.float32), 1)

                return True

            except Exception as e:
                logger.error(f"[RasterFill] Poisson interpolation failed: {e}")
                return False
