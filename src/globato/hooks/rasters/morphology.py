#!/usr/bin/env python

"""
globato.hooks.rasters.morphology
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mophology operations on the raster.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import numpy as np
import scipy.ndimage

from .base import RasterStreamHook


class RasterMorphology(RasterStreamHook):
    """Apply morphological operations to the raster.

    Usage: --hook raster_morphology:op=closing:kernel=3
    """

    name = "raster_morphology"
    default_suffix = "_morph"
    meta_desc = "Apply morphological operations to a raster."

    def __init__(self, op="erosion", kernel=3, **kwargs):
        super().__init__(**kwargs)
        self.op = op.lower()
        self.kernel = int(kernel)

    def process_chunk(self, data, ndv, entry, transform=None, window=None):
        footprint = np.ones((self.kernel, self.kernel), dtype=bool)

        is_float = data.dtype.kind == "f"
        if is_float:
            valid_mask = (data != ndv) & ~np.isnan(data)
        else:
            valid_mask = (
                (data != ndv) if ndv is not None else np.ones_like(data, dtype=bool)
            )

        if not np.any(valid_mask):
            return data

        data_min, data_max = np.nanmin(data[valid_mask]), np.nanmax(data[valid_mask])
        fill_val = data_max if self.op in ["erosion", "opening"] else data_min

        working_data = data.copy()
        working_data[~valid_mask] = fill_val

        if self.op == "erosion":
            result = scipy.ndimage.grey_erosion(working_data, footprint=footprint)
        elif self.op == "dilation":
            result = scipy.ndimage.grey_dilation(working_data, footprint=footprint)
        elif self.op == "opening":
            result = scipy.ndimage.grey_opening(working_data, footprint=footprint)
        elif self.op == "closing":
            result = scipy.ndimage.grey_closing(working_data, footprint=footprint)
        else:
            result = working_data

        if ndv is not None:
            result[~valid_mask] = ndv

        return result
