#!/usr/bin/env python

"""
globato.hooks.filters.field_filter
~~~~~~~~~~~~~

Filter data by field (z, weight, confidence, classification)

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging

import numpy as np
from fetchez.hooks import FetchHook

logger = logging.getLogger(__name__)


class StreamFieldFilter(FetchHook):
    """Filters a point stream, keeping only points where a specific field
    falls within the defined min_val and max_val thresholds.
    """

    name = "filter_field"
    meta_stage = "stream"
    meta_category = "stream-filter"
    meta_desc = "Filter stream chunks based on a specific field's value."

    def __init__(self, field="confidence", min_val=None, max_val=None, **kwargs):
        super().__init__(**kwargs)
        self.field = field
        self.min_val = float(min_val) if min_val is not None else None
        self.max_val = float(max_val) if max_val is not None else None

    def _process_stream(self, stream):
        for chunk in stream:
            if chunk is None or len(chunk) == 0:
                continue

            # If the chunk doesn't have this field, pass it through untouched
            if self.field not in chunk.dtype.names:
                yield chunk
                continue

            mask = np.ones(len(chunk), dtype=bool)
            if self.min_val is not None:
                mask &= chunk[self.field] >= self.min_val

            if self.max_val is not None:
                mask &= chunk[self.field] <= self.max_val

            valid_chunk = chunk[mask]
            if len(valid_chunk) > 0:
                yield valid_chunk

    def run(self, entries):
        for mod, entry in entries:
            if self.is_point_stream(entry):
                stream = entry.get("stream")
                entry["stream"] = self._process_stream(stream)
        return entries
