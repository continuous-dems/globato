#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.streams.readers.base
~~~~~~~~~~~

:copyright: (c) 2025 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import json
import math
import numpy as np
import logging
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

    def generate_inf(self, out_path=None):
        out_path = out_path or f"{self.path}.inf"

        minmax = [math.inf, -math.inf, math.inf, -math.inf, math.inf, -math.inf]
        total_pts = 0

        # Consume the stream directly (do not yield!)
        for chunk in self.yield_chunks():
            if chunk is None or len(chunk) == 0:
                continue

            try:
                xmin, xmax, ymin, ymax, zmin, zmax, count = self._extract_bounds(chunk)
                # minmax = [xmin, xmax, ymin, ymax, zmin, zmax]
                minmax[0] = min(minmax[0], xmin)
                minmax[1] = max(minmax[1], xmax)
                minmax[2] = min(minmax[2], ymin)
                minmax[3] = max(minmax[3], ymax)
                minmax[4] = min(minmax[4], zmin)
                minmax[5] = max(minmax[5], zmax)
                total_pts += count
            except Exception:
                pass  # Silently fail bounds extraction if chunk is weird

        meta = {}

        # Write the .inf file once the stream is completely exhausted
        if total_pts > 0 and minmax[0] != math.inf:
            w, e, s, n = minmax[0], minmax[1], minmax[2], minmax[3]
            wkt = f"POLYGON (({w} {n}, {e} {n}, {e} {s}, {w} {s}, {w} {n}))"

            meta = {
                "numpts": int(total_pts),
                "minmax": [float(x) for x in minmax],
                "wkt": wkt,
            }

            try:
                with open(out_path, "w") as f:
                    json.dump(meta, f, indent=4)
            except Exception:
                logger.debug(f"Could not write inf file {out_path}")
                pass

        return meta
