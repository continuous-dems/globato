#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.rasters.zscore
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Filters data based on local Z-Score (Standard Score).
Z = (Value - LocalMean) / LocalStdDev

Based on cudem.grits.zscore

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import numpy as np
import scipy.ndimage

from .base import RasterHook

logger = logging.getLogger(__name__)


class RasterZScore(RasterHook):
    """Local Z-Score Filter.

    Args:
      threshold : float
        The Z-Score threshold. Pixels with |Z| > threshold are masked.
        Typical values: 2.0 (95%), 3.0 (99.7%).
      kernel_size : int
        Size of the neighborhood window (pixels).
    """

    name = "raster_zscore"
    default_suffix = "_zscore"

    def __init__(self, threshold=3.0, kernel_size=5, **kwargs):
        super().__init__(**kwargs)

        self.threshold = float(threshold)
        self.kernel_size = int(kernel_size)

    def process_chunk(self, data, ndv, entry, transform=None, window=None):
        """data: (Bands, Rows, Cols)"""

        src_arr = data
        valid_mask = (src_arr != ndv) & (~np.isnan(src_arr))
        if not np.any(valid_mask):
            return src_arr

        src_arr[src_arr[~valid_mask]] = np.nan
        filled_data = src_arr.copy()
        filled_data[np.isnan(src_arr)] = np.nanmean(src_arr)

        local_mean = scipy.ndimage.uniform_filter(
            filled_data, size=self.kernel_size, mode="reflect"
        )

        local_sq_mean = scipy.ndimage.uniform_filter(
            filled_data**2, size=self.kernel_size, mode="reflect"
        )

        local_var = local_sq_mean - local_mean**2
        local_std = np.sqrt(np.maximum(0, local_var)) # Ensure non-negative

        ## Avoid div by zero
        local_std[local_std == 0] = 1e-6

        # Calculate Z
        z_score = np.abs((filled_data - local_mean) / local_std)

        mask = (z_score > self.threshold) & (~np.isnan(src_arr))
        src_arr[np.isnan(src_arr)] = ndv
        src_arr[mask] = ndv

        return src_arr
