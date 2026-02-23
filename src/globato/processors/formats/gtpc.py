#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.formats.gtpc
~~~~~~~~~~~~~

Globato Point Cloud files.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import h5py
import numpy as np

logger = logging.getLogger(__name__)

class GTPCReader:
    """Reads .gtpc (Globato Point Cloud) files."""

    def __init__(self, fn, chunk_size=50000, **kwargs):
        self.fn = fn
        self.chunk_size = chunk_size

    def yield_chunks(self):
        try:
            with h5py.File(self.fn, "r") as f:
                if "points" not in f:
                    return

                grp = f["points"]

                if "x" not in grp:
                    return

                total_pts = grp["x"].shape[0]
                fields = list(grp.keys())

                for i in range(0, total_pts, self.chunk_size):
                    end = min(i + self.chunk_size, total_pts)

                    arrays = [grp[field][i:end] for field in fields]
                    dtypes = [(field, grp[field].dtype) for field in fields]

                    chunk = np.rec.fromarrays(arrays, dtype=dtypes)
                    yield chunk

        except Exception:
            pass
