#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.filters.outlierz
~~~~~~~~~~~~~

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import numpy as np
from fetchez.utils import float_or
from .base import GlobatoFilter

logger = logging.getLogger(__name__)


class OutlierZ(GlobatoFilter):
    """Statistical Outlier Removal (Z-Score/Percentile).

    threshold=3
    set_class=7
    """

    name = "outlierz"
    desc = "filter outliers based on percentile"

    def __init__(self, threshold=3.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = float_or(threshold, 3.0)

    def filter_chunk(self, chunk):
        z = chunk['z']
        if len(z) < 3:
            return None

        mean = np.mean(z)
        std = np.std(z)
        if std == 0:
            return None

        z_score = np.abs((z - mean) / std)
        return z_score > self.threshold
