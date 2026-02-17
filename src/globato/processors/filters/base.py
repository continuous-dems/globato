#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.filters.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Base class for all Globato stream filters.
Handles stream iteration, schema enforcement, and classification logic.
"""

import logging
import numpy as np
from fetchez.hooks import FetchHook
from fetchez import utils

logger = logging.getLogger(__name__)

class GlobatoFilter(FetchHook):
    """Base class for Point Stream Filters/Classifiers.

    Subclasses should implement `filter_chunk(chunk)`.
    """

    stage = "file"
    category = "stream-filter"

    def __init__(self, set_class=7, exclude_classes=None, invert=False, **kwargs):
        super().__init__(**kwargs)
        self.set_class = int(set_class)
        self.invert = utils.str2bool(invert)

        if exclude_classes:
            self.exclude_classes = [int(x) for x in str(exclude_classes).split('/')]
        else:
            self.exclude_classes = []

    def run(self, entries):
        """Standard function to hook into the stream pipeline."""

        for mod, entry in entries:
            stream = entry.get("stream")
            if not stream: continue

            # `setup` allows subclass to prepare resources based on region/module
            if hasattr(self, 'setup'):
                if self.setup(mod, entry) is False:
                    continue # Skip if setup fails

            entry["stream"] = self._process_stream(stream)
        return entries

    def _process_stream(self, stream):
        """Iterates stream, handles schema, and calls filter_chunk."""

        try:
            for chunk in stream:
                if "classification" not in chunk.dtype.names:
                    chunk = utils.add_field_to_recarray(chunk, "classification", np.uint8, 0)

                if self.exclude_classes:
                    # True = Available to filter.
                    eligible_mask = ~np.isin(chunk['classification'], self.exclude_classes)

                    if not np.any(eligible_mask):
                        yield chunk
                        continue
                else:
                    eligible_mask = np.ones(len(chunk), dtype=bool)

                # subclass returns a boolean mask (True = Outlier/Target)
                # OR returns a modified chunk (for destructive filters)
                result = self.filter_chunk(chunk)

                if result is None:
                    yield chunk
                    continue

                if isinstance(result, np.ndarray) and result.dtype == bool:
                    # It's a Classification Mask (True = Change Class)
                    # Apply Invert Logic (Global)
                    if self.invert:
                        result = ~result

                    # Don't touch excluded classes even if filter said so
                    final_mask = result & eligible_mask

                    if np.any(final_mask):
                        chunk['classification'][final_mask] = self.set_class

                    yield chunk

                else:
                    # It's a New Chunk (Destructive Filter like Drop or Thin)
                    # We assume the subclass handled everything
                    yield result

        finally:
            # Teardown hook if needed
            if hasattr(self, 'teardown'):
                self.teardown()

    def filter_chunk(self, chunk):
        """Override this method.

        Args:
            chunk (recarray): The point data.

        Returns:
            np.array (bool): Mask of points to classify as `self.set_class`.
            OR
            np.recarray: A new (smaller) chunk if destructive.
        """

        raise NotImplementedError
