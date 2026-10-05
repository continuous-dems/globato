#!/usr/bin/env python

"""
globato.streams.readers.base
~~~~~~~~~~~

:copyright: (c) 2025 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging

import numpy as np
from fetchez.streams.readers import BaseReader

from ..schema import ensure_schema

logger = logging.getLogger(__name__)


class BaseGlobatoReader(BaseReader):
    """The base class for Globato data readers.
    Ensures all output streams conform to the strict xyz_recarray schema.
    """

    def __init__(self, path, weight=1.0, uncertainty=0.0, **kwargs):
        super().__init__(path, **kwargs)
        self.module_weight = float(weight)
        self.module_unc = float(uncertainty)

    def _yield_raw_chunks(self):
        """Child globato classes MUST implement this to yield their raw format data."""

        raise NotImplementedError

    def _read_chunks(self):
        """Automatically intercepts the raw chunks and enforces the Globato schema."""

        raw_stream = self._yield_raw_chunks()

        yield from ensure_schema(
            raw_stream, module_weight=self.module_weight, module_unc=self.module_unc
        )

    def _extract_bounds(self, chunk):
        """Required for automatic .inf generation."""

        xmin, xmax = np.min(chunk["x"]), np.max(chunk["x"])
        ymin, ymax = np.min(chunk["y"]), np.max(chunk["y"])
        zmin, zmax = np.min(chunk["z"]), np.max(chunk["z"])
        return xmin, xmax, ymin, ymax, zmin, zmax, len(chunk)
