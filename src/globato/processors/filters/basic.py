#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.filters.basic
~~~~~~~~~~~~~

Basic Filters for point clouds.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import numpy as np
from fetchez.hooks import FetchHook
from fetchez import utils

logger = logging.getLogger(__name__)


class RangeZ(FetchHook):
    """Classify points outside a Z range as Noise (or specified class).
    Does NOT remove points; only reclassifies them.

    Usage: --hook range_z:min_z=-50:max_z=0:set_class=7
    """

    name = "range_z"
    stage = "file"
    desc = "Classify points by Z range"
    category = "stream-filter"

    def __init__(self, min_z=None, max_z=None, set_class=7, **kwargs):
        super().__init__(**kwargs)
        self.min_z = utils.float_or(min_z)
        self.max_z = utils.float_or(max_z)
        self.set_class = int(set_class) # 7 = LAS Noise standard

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

            mask = np.zeros(len(chunk), dtype=bool)
            if self.min_z is not None: mask |= (chunk["z"] < self.min_z)
            if self.max_z is not None: mask |= (chunk["z"] > self.max_z)

            if np.any(mask):
                chunk["classification"][mask] = self.set_class

            yield chunk
