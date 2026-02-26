#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.rasters.flats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Removes flat areas from a DEM by identifying contiguous areas of identical values
that exceed a specified size threshold.

Based on cudem.grits.flats

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import numpy as np

from .base import RasterHook

logger = logging.getLogger(__name__)

class RasterFlats(RasterHook):
    """Remove flat areas from the input DEM.
    Identified flat areas are set to NoData.

    Args:
      size_threshold : int, optional
        The minimum number of pixels required to define a "flat" area to be removed.
        If None, it is auto-calculated using outlier detection on value counts.
    """

    name = "raster_flats"
    default_suffix = "_deflat"

    def __init__(self, size_threshold=1.0, **kwargs):
        super().__init__(**kwargs)

        self.size_threshold = int(size_threshold)

    def process_chunk(self, data, ndv, entry, transform=None, window=None):
        """data: (Bands, Rows, Cols)"""
        src_arr = data
        valid_mask = (src_arr != ndv) & (~np.isnan(src_arr))
        if not np.any(valid_mask):
            return src_arr

        # Identify Unique Values and their Counts
        uv, uv_counts = np.unique(src_arr, return_counts=True)

        # Determine Threshold
        threshold = self.size_threshold
        if threshold is None:
            # Auto-detect: Assume flats are statistical outliers in terms of frequency
            outlier_val, _ = self.get_outliers(uv_counts, percentile=99)
            threshold = outlier_val if not np.isnan(outlier_val) else 100

            flat_values = uv[uv_counts > threshold]

            # Create Mask
            if ndv is not None:
                flat_values = flat_values[flat_values != ndv]

            if flat_values.size > 0:
                mask = np.isin(src_arr, flat_values)

                # Count and Apply
                n_removed = np.count_nonzero(mask)
                if n_removed > 0:
                    count_removed += n_removed
                    src_arr[mask] = ndv
                    logger.info(f"[Flats] Removed {count_removed} flat pixels.")
        return src_arr
