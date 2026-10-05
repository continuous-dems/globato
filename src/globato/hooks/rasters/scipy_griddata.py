#!/usr/bin/env python

"""
globato.hooks.rasters.scipy_griddata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interpolates gaps in a stacked DEM using SciPy's griddata.
Methods: linear, cubic, nearest

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging

import numpy as np
from scipy import interpolate

from .base import RasterStreamHook

logger = logging.getLogger(__name__)


class ScipyInterp(RasterStreamHook):
    name = "interp_scipy"
    default_suffix = "_interp"
    meta_desc = "Intpolate NoData voids using SciPy."
    meta_tags = ["globato", "interpolation", "multi-stack"]

    def __init__(self, method="linear", **kwargs):
        super().__init__(**kwargs)
        self.method = method.lower()
        # Default buffer needed for interpolation continuity
        if self.buffer == 0:
            self.buffer = 20

    def process_chunk(self, data, ndv, entry, transform=None, window=None):
        is_3d = data.ndim == 3
        work_data = data[0] if is_3d else data

        valid_mask = (work_data != ndv) & ~np.isnan(work_data)

        if np.all(valid_mask) or not np.any(valid_mask):
            return data

        points = np.column_stack(np.where(valid_mask))
        values = work_data[valid_mask]

        grid_y, grid_x = np.mgrid[0 : work_data.shape[0], 0 : work_data.shape[1]]

        try:
            interp = interpolate.griddata(
                points, values, (grid_y, grid_x), method=self.method
            )
            interp[np.isnan(interp)] = ndv

            if is_3d:
                result = data.copy()
                result[0] = interp
            else:
                result = interp

            return result.astype(data.dtype)
        except Exception:
            return data
