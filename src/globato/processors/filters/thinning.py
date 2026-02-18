#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.filters.thinning
~~~~~~~~~~~~~

These filters are destructive by nature.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import numpy as np
from fetchez.hooks import FetchHook
from fetchez import utils
from .base import GlobatoFilter

logger = logging.getLogger(__name__)


class BlockThin(GlobatoFilter):
    """Spatial thinning.

    Keep one point per grid cell (Min/Max/Median/Mean/Random).

    res=10: The resolution of the internal grid.
    mode=min: Which mode decides the thinning.
    soft=True: Marks thinned points as Class 12 (Overlap/Reserved).
    soft=False: Drops points immediately (Default).
    """

    name = "block_thin"
    desc = "Thin the data"

    def __init__(self, res=10, mode='min', soft=True, **kwargs):
        super().__init__(**kwargs)
        self.res = utils.float_or(res, 10)
        self.mode = mode
        self.soft = utils.str2bool(soft)

    def filter_chunk(self, chunk):
        x_idx = np.floor((chunk['x'] - np.min(chunk['x'])) / self.res).astype(np.int64)
        y_idx = np.floor((chunk['y'] - np.min(chunk['y'])) / self.res).astype(np.int64)

        width = (np.max(x_idx) - np.min(x_idx)) + 1
        grid_ids = y_idx * width + x_idx

        sort_idx = np.argsort(grid_ids)
        sorted_ids = grid_ids[sort_idx]

        _, start_indices = np.unique(sorted_ids, return_index=True)

        keep_indices = []
        for i in range(len(start_indices)):
            start = start_indices[i]
            end = start_indices[i+1] if i+1 < len(start_indices) else len(sorted_ids)
            block_indices = sort_idx[start:end]

            if self.mode == 'min':
                keep_indices.append(block_indices[np.argmin(chunk['z'][block_indices])])
            elif self.mode == 'max':
                keep_indices.append(block_indices[np.argmax(chunk['z'][block_indices])])
            else:
                keep_indices.append(block_indices[0])

        if self.soft:
            mask = np.ones(len(chunk), dtype=bool)
            mask[keep_indices] = False
            return mask
        else:
            return chunk[keep_indices]


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
