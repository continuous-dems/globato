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
from .base import GlobatoFilter

logger = logging.getLogger(__name__)


class RangeZ(GlobatoFilter):
    """Classify points outside a Z range as Noise (or specified class).
    Does NOT remove points; only reclassifies them.

    Usage: --hook range_z:min_z=-50:max_z=0:set_class=7
    """

    name = "range_z"
    desc = "Classify points by Z range"

    def __init__(self, min_z=None, max_z=None, **kwargs):
        super().__init__(**kwargs)
        self.min_z = utils.float_or(min_z)
        self.max_z = utils.float_or(max_z)

    def filter_chunk(self, chunk):
        mask = np.zeros(len(chunk), dtype=bool)
        if self.min_z is not None: mask |= (chunk["z"] < self.min_z)
        if self.max_z is not None: mask |= (chunk["z"] > self.max_z)

        return mask
