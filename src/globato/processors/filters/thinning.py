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

logger = logging.getLogger(__name__)


class BlockThin(FetchHook):
    """Spatial thinning.

    Keep one point per grid cell (Min/Max/Median/Mean/Random).

    res=10: The resolution of the internal grid.
    mode=min: Which mode decides the thinning.
    soft=True: Marks thinned points as Class 12 (Overlap/Reserved).
    soft=False: Drops points immediately (Default).
    """

    name = "block_thin"
    stage = "file"
    desc = "Thin the data"
    category = "stream-filter"

    def __init__(self, res=10, mode='min', soft=False, set_class="12", **kwargs):
        super().__init__(**kwargs)
        self.res = utils.float_or(res, 10)
        self.mode = mode
        self.soft = utils.str2bool(soft)
        self.set_class = int(set_class) # 7 = LAS overlap

    def run(self, entries):
        for mod, entry in entries:
            stream = entry.get("stream")
            if not stream: continue

            entry["stream"] = self._process_stream(stream)
        return entries

    def _process_stream(self, stream):
        for chunk in stream:
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

                # Selection
                if self.mode == 'min':
                    keep_indices.append(block_indices[np.argmin(chunk['z'][block_indices])])
                elif self.mode == 'max':
                    keep_indices.append(block_indices[np.argmax(chunk['z'][block_indices])])
                else:
                    keep_indices.append(block_indices[0])

            if self.soft:
                if "classification" not in chunk.dtype.names:
                    chunk = utils.add_field_to_recarray(chunk, "classification", np.uint8, 0)

                mask = np.ones(len(chunk), dtype=bool)
                mask[keep_indices] = False
                chunk['classification'][mask] = self.set_class
                yield chunk
            else:
                yield chunk[keep_indices]
