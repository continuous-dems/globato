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
from .base import GlobatoFilter

logger = logging.getLogger(__name__)


class DropClass(GlobatoFilter):
    """Destructive filter: Removes points with specific classifications.

    Usage: --hook drop_class:classes=7/18
    """

    name = "drop_class"
    desc = "Drop specified classes from the point stream"

    def __init__(self, classes="7/12", **kwargs):
        super().__init__(**kwargs)
        self.target_classes = [int(x) for x in str(classes).split('/')]

    def filter_chunk(self, chunk):
        # Identify points to drop
        mask = np.isin(chunk["classification"], self.target_classes)

        if self.invert:
            # Keep these classes
            keep_mask = mask
        else:
            # Drop these classes
            keep_mask = ~mask

        #logger.info(f"Dropped {np.count_nonzero(~keep_mask)} points")
        if np.count_nonzero(keep_mask) > 0:
            return chunk[keep_mask]
        else:
            # All dropped
            return chunk[0:0]
