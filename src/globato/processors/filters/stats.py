#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.filters.stats
~~~~~~~~~~~~~

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import numpy as np
from fetchez.hooks import FetchHook
from fetchez import utils

logger = logging.getLogger(__name__)


class OutlierZ(FetchHook):
    """Statistical Outlier Removal (Z-Score/Percentile)."""

    def __init__(self, threshold=3.0, set_class=7, **kwargs):
        super().__init__(**kwargs)
        self.threshold = float_or(threshold, 3.0)
        self.set_class = set_class

    def run(self, entries):
        for mod, entry in entries:
            stream = entry.get("stream")
            if not stream: continue

            entry["stream"] = self._process_stream(stream)
        return entries

    def _process_stream(self, stream):
        for chunk in stream:
            if "classification" not in chunk.dtype.names:
                chunk = utils.add_field_to_recarray(chunk, "classification", np.uint8, 0)

            z = self.chunk['z']
            if len(z) < 3: continue

            mean = np.mean(z)
            std = np.std(z)

            if std == 0: continue

            z_score = np.abs((z - mean) / std)
            mask = z_score > self.threshold
            if np.any(mask):
                chunk["classification"][mask] = self.set_class

            yield chunk
