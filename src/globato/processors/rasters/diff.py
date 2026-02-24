#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.rasters.diff
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Calculates the difference between a Source and Auxiliary DEM.
Handles on-the-fly resampling and alignment automatically.
"""

import os
import logging
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from .base import RasterHook

logger = logging.getLogger(__name__)

class RasterDiff(RasterHook):
    """Calculates pixel-wise difference (Source - Aux).

    Modes:
    - 'filter': Masks Source pixels where abs(Diff) > threshold.
    - 'difference': Outputs the difference grid (Source - Aux).

    Features:
    - Auto-Alignment: Automatically warps/resamples Aux to match Source.
    - safe-nan: Handles NoData correctly.

    Usage:
        --hook raster_diff:aux_path=ref.tif:threshold=10.0:mode=filter
        --hook raster_diff:aux_path=ref.tif:mode=difference
    """

    name = "raster_diff"
    default_suffix = "_diff"

    def __init__(self, aux_path=None, threshold=None, mode='filter', resample='bilinear', **kwargs):
        super().__init__(**kwargs)
        self.aux_path = aux_path
        self.threshold = float(threshold) if threshold is not None else None
        self.mode = mode.lower()

        # Resampling method for the Aux grid
        self.resample_alg = getattr(Resampling, resample, Resampling.bilinear)

    def process_raster(self, src_path, dst_path, entry):
        if not self.aux_path or not os.path.exists(self.aux_path):
            logger.error(f"[Diff] Aux path not found: {self.aux_path}")
            return False

        if self.mode == 'difference' and '_processed' in dst_path:
            dst_path = dst_path.replace('_processed', '_diff')

        with rasterio.open(src_path) as src:
            profile = src.profile.copy()

            if self.mode == 'difference':
                profile.update(dtype=rasterio.float32, nodata=np.nan)

            with rasterio.open(self.aux_path) as aux:
                with WarpedVRT(aux,
                               crs=src.crs,
                               transform=src.transform,
                               width=src.width,
                               height=src.height,
                               resampling=self.resample_alg) as vrt_aux:

                    with rasterio.open(dst_path, 'w', **profile) as dst:
                        for window, buff_win in self.yield_buffered_windows(src, buffer_size=0):

                            src_data = src.read(1, window=window)
                            src_ndv = src.nodata

                            aux_data = vrt_aux.read(1, window=window)
                            aux_ndv = vrt_aux.nodata # Usually inherited or set in VRT

                            s_arr = src_data.astype(np.float32)
                            s_arr[src_data == src_ndv] = np.nan

                            a_arr = aux_data.astype(np.float32)
                            if aux_ndv is not None:
                                a_arr[aux_data == aux_ndv] = np.nan

                            diff = s_arr - a_arr

                            if self.mode == 'difference':
                                diff[np.isnan(diff)] = profile['nodata']
                                dst.write(diff, 1, window=window)

                            elif self.mode == 'filter':
                                if self.threshold is not None:
                                    with np.errstate(invalid='ignore'):
                                        mask = np.abs(diff) > self.threshold

                                    src_data[mask] = src_ndv

                                dst.write(src_data, 1, window=window)

        return True
