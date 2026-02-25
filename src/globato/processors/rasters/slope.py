#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.rasters.slope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Filters data based on calculated Slope (Rise/Run).
"""

import numpy as np
from .base import RasterHook

class RasterSlopeFilter(RasterHook):
    """Filters pixels based on local slope.

    Calculates slope using Numpy gradients (Run/Rise).
    Masks pixels where slope is outside [min_val, max_val].

    Usage: --hook raster_slope:max_val=1.0 (Filters slopes > 45 degrees if Z/XY are same units)
    """

    name = "raster_slope"
    default_suffix = "_slope_filtered"

    def __init__(self, min_val=None, max_val=None, **kwargs):
        super().__init__(**kwargs)
        self.min_val = float(min_val) if min_val is not None else None
        self.max_val = float(max_val) if max_val is not None else None

        if self.buffer == 0:
            self.buffer = 2

    def process_chunk(self, data, ndv, entry, transform=None, window=None):
        valid_mask = (data != ndv) & ~np.isnan(data)
        if not np.any(valid_mask):
            return data

        work_data = data.copy()
        work_data[~valid_mask] = np.nanmean(data) if np.any(valid_mask) else 0

        dx = abs(transform[0])
        dy = abs(transform[4])

        gy, gx = np.gradient(work_data, dy, dx)

        slope_arr = np.sqrt(gx**2 + gy**2)
        remove_mask = np.zeros(slope_arr.shape, dtype=bool)

        if self.min_val is not None:
            remove_mask |= (slope_arr < self.min_val)

        if self.max_val is not None:
            remove_mask |= (slope_arr > self.max_val)

        final_mask = remove_mask & valid_mask
        data[final_mask] = ndv

        return data
