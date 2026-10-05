#!/usr/bin/env python

"""
globato.hooks.rasters.limits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Apply limits to a raster z/w/u/etc.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging

import numpy as np
from fetchez.utils import float_or

from globato.hooks.rasters.base import RasterStreamHook

logger = logging.getLogger(__name__)


class RasterLimitsHook(RasterStreamHook):
    """Applies Z-limits and other constraints to a chunked raster stream.
    Replaces values outside the limits with NoData.
    """

    name = "raster_limits"
    meta_stage = "stream"
    meta_category = "raster-filter"
    meta_desc = "Applies limits to a raster stream."

    def __init__(self, min_z=None, max_z=None, **kwargs):
        super().__init__(**kwargs)
        self.min_z = float_or(min_z)
        self.max_z = float_or(max_z)

    def process_chunk(self, data, ndv, entry, transform, window):
        is_3d = data.ndim == 3
        z_data = data[0] if is_3d else data

        valid_mask = (z_data != ndv) & (~np.isnan(z_data))

        if not np.any(valid_mask):
            return data

        if self.min_z is not None:
            valid_mask &= z_data >= self.min_z
        if self.max_z is not None:
            valid_mask &= z_data <= self.max_z

        z_data[~valid_mask] = ndv

        if is_3d:
            data[0] = z_data
            return data
        else:
            return z_data
