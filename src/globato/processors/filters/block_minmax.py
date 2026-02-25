#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.filters.block_minmax
~~~~~~~~~~~~~

These filters are destructive by nature.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import numpy as np
from fetchez import utils
from .base import GlobatoFilter

logger = logging.getLogger(__name__)


class BlockMinMax(GlobatoFilter):
    """Thin point cloud by keeping only the Min or Max Z point per grid block.
    Commonly used in hydrography for "shoal-biased" thinning.

    res=10: The resolution of the internal grid.
    mode=min: Which mode decides the thinning.
    soft=True: Marks thinned points as Class 12 (Overlap/Reserved).
    soft=False: Drops points immediately (Default).
    """

    name = "block_minmax"
    desc = "Thin the data by min/max"

    def __init__(self, res=10, mode='min', soft=True, **kwargs):
        super().__init__(**kwargs)
        self.res = utils.float_or(res, 10)
        self.mode = mode.lower() if mode else 'min'
        self.soft = utils.str2bool(soft)

    def filter_chunk(self, chunk):
        x_idx = np.floor(chunk['x'] / self.res).astype(np.int64)
        y_idx = np.floor(chunk['y'] / self.res).astype(np.int64)
        z_vals = chunk['z']

        if self.mode == 'max':
            sort_order = np.lexsort((-z_vals, y_idx, x_idx))
        else:
            sort_order = np.lexsort((z_vals, y_idx, x_idx))

        sorted_x = x_idx[sort_order]
        sorted_y = y_idx[sort_order]

        # Find Unique Blocks
        change_mask = np.concatenate(
            ([True], (sorted_x[1:] != sorted_x[:-1]) | (sorted_y[1:] != sorted_y[:-1]))
        )

        # Indices in the sorted array that we want to keep (min/max points)
        keep_sorted_indices = np.nonzero(change_mask)[0]

        # Map back to original array indices
        keep_indices = sort_order[keep_sorted_indices]

        if self.soft:
            mask = np.ones(len(chunk), dtype=bool)
            mask[keep_indices] = False
            return mask
        else:
            return chunk[keep_indices]
