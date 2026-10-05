#!/usr/bin/env python

"""
globato.hooks.rasters.seive
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove small regions from a raster.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging

import numpy as np
from rasterio.features import sieve

from .base import RasterStreamHook

logger = logging.getLogger(__name__)


class RasterSieveHook(RasterStreamHook):
    """Removes small contiguous regions (noise, holes, puddles) from a raster."""

    name = "raster_sieve"
    meta_desc = "Removes small contiguous regions from a raster."
    default_suffix = "_sieve"

    def __init__(self, size=100, connectivity=8, **kwargs):
        super().__init__(**kwargs)
        # Minimum size (in pixels) of a polygon to keep it.
        self.size = int(size)

        # (horizontal/vertical) or 8 (diagonal included)
        self.connectivity = int(connectivity)

    def process_chunk(self, data, ndv, entry, transform=None, window=None):
        working_data = data[0] if data.ndim == 3 else data
        is_float = working_data.dtype.kind == "f"

        if is_float:
            valid_mask = ~np.isnan(working_data)
            int_data = np.where(valid_mask, working_data, 0).astype(np.int32)
            mask_arg = valid_mask
        else:
            valid_mask = (
                (working_data != ndv)
                if ndv is not None
                else np.ones_like(working_data, dtype=bool)
            )
            int_data = working_data.copy()
            mask_arg = valid_mask

        sieved = sieve(
            int_data, size=self.size, connectivity=self.connectivity, mask=mask_arg
        )

        result = sieved.astype(data.dtype)
        result = np.where(valid_mask, result, ndv)

        if data.ndim == 3:
            result = result[np.newaxis, ...]

        return result
