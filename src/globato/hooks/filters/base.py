#!/usr/bin/env python

"""
globato.hooks.filters.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Base class for all Globato stream filters/classifiers.
Handles stream iteration, schema enforcement, and classification.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging

import numpy as np
from fetchez import utils
from fetchez.hooks import FetchHook

from globato.utils import add_field_to_recarray

logger = logging.getLogger(__name__)


class GlobatoFilter(FetchHook):
    """Base class for Point Stream Filters/Classifiers.

    Subclasses should implement `filter_chunk(chunk)`.
    """

    meta_stage = "stream"
    meta_category = "stream-filter"

    def __init__(
        self,
        set_class=7,
        exclude_classes=None,
        invert=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.set_class = int(set_class)
        self.invert = utils.str2bool(invert)

        if exclude_classes:
            self.exclude_classes = [int(x) for x in str(exclude_classes).split("/")]
        else:
            self.exclude_classes = []

    def run(self, entries):
        """Standard function to hook into the stream pipeline."""

        for mod, entry in entries:
            if not self.is_point_stream(entry):
                logger.debug(f"[{self.name}] {entry} data has no stream!")
                continue

            stream = entry.get("stream")
            # `setup` allows subclass to prepare resources based on region/module
            if hasattr(self, "setup"):
                if self.setup(mod, entry) is False:
                    continue

            entry["stream"] = self._process_stream(stream)
        return entries

    def _process_stream(self, stream):
        """Iterates stream, handles schema, and calls filter_chunk."""

        try:
            for chunk in stream:
                if "classification" not in chunk.dtype.names:
                    chunk = add_field_to_recarray(chunk, "classification", np.uint8, 0)

                if self.exclude_classes:
                    # True = Available to filter.
                    eligible_mask = ~np.isin(
                        chunk["classification"], self.exclude_classes
                    )

                    if not np.any(eligible_mask):
                        yield chunk
                        continue
                else:
                    eligible_mask = np.ones(len(chunk), dtype=bool)

                # subclass returns a boolean mask (True = Outlier/Target)
                # or returns a modified chunk (for destructive filters)
                result = self.filter_chunk(chunk)

                if result is None:
                    yield chunk
                    continue

                if isinstance(result, np.ndarray) and result.dtype == bool:
                    if self.invert:
                        result = ~result

                    final_mask = result & eligible_mask

                    if np.any(final_mask):
                        chunk["classification"][final_mask] = self.set_class

                    yield chunk

                else:
                    yield result

        finally:
            if hasattr(self, "teardown"):
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
