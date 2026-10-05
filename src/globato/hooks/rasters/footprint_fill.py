#!/usr/bin/env python

"""
globato.hooks.rasters.footprint_fill
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fill a raster and clip to a morphological boundary (footprint)

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging

import numpy as np
import scipy.ndimage
from rasterio.fill import fillnodata

from globato.hooks.rasters.base import RasterStreamHook

logger = logging.getLogger(__name__)


class RasterFootprintFill(RasterStreamHook):
    """Interpolates internal gaps within a dataset's footprint.

    Requires a chunk buffer >= max_gap to prevent seamlines.
    """

    name = "raster-footprint-fill"
    meta_stage = "stream"
    meta_category = "raster-stream"
    meta_aliases = ["raster_footprint_fill"]
    meta_desc = "Interpolates gaps within a dataset's footprint."

    def __init__(self, max_gap=10, **kwargs):
        super().__init__(**kwargs)
        self.max_gap = int(max_gap)

    def process_chunk(self, data, ndv, entry, transform, window):
        """Processes a single buffered chunk of the raster."""

        is_3d = data.ndim == 3
        z_data = data[0] if is_3d else data

        valid_mask = (z_data != ndv) & (~np.isnan(z_data))
        if not np.any(valid_mask) or np.all(valid_mask):
            return data

        struct = scipy.ndimage.generate_binary_structure(2, 2)
        iterations = max(1, self.max_gap // 2)

        footprint_mask = scipy.ndimage.binary_closing(
            valid_mask, structure=struct, iterations=iterations
        )

        filled_z = fillnodata(
            z_data,
            mask=valid_mask,
            max_search_distance=self.max_gap * 1.5,
            smoothing_iterations=1,
        )

        filled_z[~footprint_mask] = ndv

        if is_3d:
            data[0] = filled_z
            return data
        else:
            return filled_z
