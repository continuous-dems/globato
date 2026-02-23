#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.interpolators.scipy_griddata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interpolates gaps in a stacked DEM using SciPy's griddata.
Methods: linear, cubic, nearest

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
import numpy as np
from scipy import interpolate
import rasterio
from rasterio.windows import Window

from .base import RasterHook

logger = logging.getLogger(__name__)


class ScipyInterp(RasterHook):
    name = "interp_scipy"
    default_suffix = "_interp"

    def __init__(self, method='linear', **kwargs):
        super().__init__(**kwargs)
        self.method = method.lower()
        # Default buffer needed for interpolation continuity
        if self.buffer == 0: self.buffer = 20

    def process_chunk(self, data, ndv, entry):
        valid_mask = (data != ndv) & ~np.isnan(data)

        if np.all(valid_mask) or not np.any(valid_mask):
            return data

        points = np.column_stack(np.where(valid_mask))
        values = data[valid_mask]

        grid_y, grid_x = np.mgrid[0:data.shape[0], 0:data.shape[1]]

        try:
            interp = interpolate.griddata(
                points, values, (grid_y, grid_x), method=self.method
            )
            interp[np.isnan(interp)] = ndv
            return interp.astype(data.dtype)
        except:
            return data
