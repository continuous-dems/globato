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
        valid_mask = (data != ndv) & (~np.isnan(data))
        if not np.any(valid_mask):
            return data

        data[~valid_mask] = np.nan
        filled_data = data.copy()
        filled_data[np.isnan(filled_data)] = np.nanmean(filled_data)

        local_mean = scipy.ndimage.uniform_filter(
            filled_data, size=self.kernel_size, mode="reflect"
        )

        local_sq_mean = scipy.ndimage.uniform_filter(
            filled_data**2, size=self.kernel_size, mode="reflect"
        )

        local_var = local_sq_mean - local_mean**2
        local_std = np.sqrt(np.maximum(0, local_var)) # Ensure non-negative

        # Avoid div by zero
        local_std[local_std == 0] = 1e-6

        # Calculate Z
        z_score = np.abs((filled_data - local_mean) / local_std)

        mask = (z_score > self.threshold) & (~np.isnan(data))
        data[np.isnan(data)] = ndv
        data[mask] = ndv
        logger.info(mask)
        return data
