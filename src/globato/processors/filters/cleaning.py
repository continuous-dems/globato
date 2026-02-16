#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.filters.cleaning
~~~~~~~~~~~~~

These filters are destructive

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import numpy as np
from fetchez.hooks import FetchHook
from fetchez import utils

logger = logging.getLogger(__name__)

class DropClass(FetchHook):
    """Destructive filter: Removes points with specific classifications.

    Usage: --hook drop_class:classes=7/18
    """

    name = "drop_class"
    stage = "file"
    desc = "Drop specified classes from the point stream"
    category = "stream-filter"

    def __init__(self, classes="7", invert=False, **kwargs):
        super().__init__(**kwargs)
        self.classes = [int(x) for x in str(classes).split('/')]
        self.invert = utils.str2bool(invert)

    def run(self, entries):
        for mod, entry in entries:
            stream = entry.get("stream")
            if not stream: continue

            entry["stream"] = self._process_stream(stream)
        return entries

    def _process_stream(self, stream):
        for chunk in stream:
            if "classification" not in chunk.dtype.names:
                yield chunk
                continue

            # Identify points to drop
            mask = np.isin(chunk["classification"], self.classes)

            if self.invert:
                # Keep these classes
                keep_mask = mask
            else:
                # Drop these classes
                keep_mask = ~mask

            # Yield valid points only
            if np.count_nonzero(keep_mask) > 0:
                yield chunk[keep_mask]
