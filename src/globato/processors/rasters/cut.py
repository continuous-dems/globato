#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.rasters.cut
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Masks data outside the defined pipeline region.
"""

import numpy as np
import rasterio
from rasterio.windows import from_bounds, intersection
from .base import RasterHook

class RasterCut(RasterHook):
    """Cuts (masks) the raster to the project region.

    Usage: --hook raster_cut
    """
    name = "raster_cut"
    default_suffix = "_cut"

    def process_raster(self, src_path, dst_path, entry):
        """Override process_raster to pre-calculate the Cut Window."""

        if not self.region:
            return False

        with rasterio.open(src_path) as src:
            self.cut_window = from_bounds(
                *self.region.to_bbox(),
                transform=src.transform
            )

            return super().process_raster(src_path, dst_path, entry)

    def process_chunk(self, data, ndv, entry, transform=None, window=None):
        """Masks pixels that fall outside the pre-calculated cut_window."""

        try:
            overlap = intersection(window, self.cut_window)
        except ValueError:
            data.fill(ndv)
            return data

        if overlap == window:
            return data

        row_start = int(max(0, overlap.row_off - window.row_off))
        row_stop  = int(min(window.height, (overlap.row_off - window.row_off) + overlap.height))
        col_start = int(max(0, overlap.col_off - window.col_off))
        col_stop  = int(min(window.width, (overlap.col_off - window.col_off) + overlap.width))

        valid_mask = np.zeros(data.shape, dtype=bool)
        valid_mask[row_start:row_stop, col_start:col_stop] = True

        data[~valid_mask] = ndv

        return data
