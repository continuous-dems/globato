#!/usr/bin/env python

"""
globato.hooks.rasters.cut
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Masks data outside the defined pipeline region.
"""

import logging

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds, intersection

from .base import RasterStreamHook

logger = logging.getLogger(__name__)


class RasterCut(RasterStreamHook):
    """Cuts (masks) the raster to the project region.

    Usage: --hook raster_cut:region=-120/-119/34/35
    """

    name = "raster_cut"
    default_suffix = "_cut"
    meta_desc = "Cut a raster to a region"

    def process_chunk(self, data, ndv, entry, transform=None, window=None):
        """Masks pixels that fall outside the target region using local chunk coordinates."""

        target_region = self.region[0] or getattr(self.current_mod, "region", None)
        if not target_region:
            logger.error(
                "RasterCut requires a region (passed via args or attached to the module)."
            )
            return data

        if transform is None:
            logger.error("RasterCut requires a valid chunk transform.")
            return data

        local_cut_window = from_bounds(*target_region.to_bbox(), transform=transform)

        chunk_h, chunk_w = data.shape[-2:]
        local_chunk_window = Window(0, 0, chunk_w, chunk_h)

        try:
            overlap = intersection(local_chunk_window, local_cut_window)
        except (ValueError, rasterio.errors.WindowError):
            data[...] = ndv
            return data

        if overlap == local_chunk_window:
            return data

        row_start = int(round(overlap.row_off))
        row_stop = int(round(overlap.row_off + overlap.height))
        col_start = int(round(overlap.col_off))
        col_stop = int(round(overlap.col_off + overlap.width))

        # row_start = int(max(0, overlap.row_off - window.row_off))
        # row_stop  = int(min(window.height, (overlap.row_off - window.row_off) + overlap.height))
        # col_start = int(max(0, overlap.col_off - window.col_off))
        # col_stop  = int(min(window.width, (overlap.col_off - window.col_off) + overlap.width))

        valid_mask = np.zeros(data.shape, dtype=bool)
        valid_mask[..., row_start:row_stop, col_start:col_stop] = True
        # valid_mask[row_start:row_stop, col_start:col_stop] = True

        data[~valid_mask] = ndv

        return data
