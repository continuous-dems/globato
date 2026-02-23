#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.rasters.morphology
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Mophology operations on the raster.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import numpy as np
import scipy.ndimage
import rasterio
from .base import RasterHook

class RasterMorphology(RasterHook):
    """Apply morphological operations to the DEM.

    Usage: --hook raster_morphology:op=closing:kernel=3
    """

    name = "raster_morphology"
    default_suffix = "_morph"

    def __init__(self, op='erosion', kernel=3, **kwargs):
        super().__init__(**kwargs)
        self.op = op.lower()
        self.kernel = int(kernel)

    def process_raster(self, src_path, dst_path, entry):
        structure = np.ones((self.kernel, self.kernel))

        with rasterio.open(src_path) as src:
            profile = src.profile.copy()
            ndv = src.nodata

            with rasterio.open(dst_path, 'w', **profile) as dst:
                for window, buffered_window in self.yield_buffered_windows(src, buffer_size=self.kernel):

                    data = src.read(1, window=buffered_window)
                    valid_mask = (data != ndv) & ~np.isnan(data)

                    if not np.any(valid_mask):
                        dst.write(data, 1, window=buffered_window)
                        continue

                    data_min, data_max = np.nanmin(data[valid_mask]), np.nanmax(data[valid_mask])
                    fill_val = data_max if self.op in ["erosion", "opening"] else data_min
                    data[~valid_mask] = fill_val

                    if self.op == "erosion":
                        result = scipy.ndimage.grey_erosion(data, structure=structure)
                    elif self.op == "dilation":
                        result = scipy.ndimage.grey_dilation(data, structure=structure)
                    elif self.op == "opening":
                        result = scipy.ndimage.grey_opening(data, structure=structure)
                    elif self.op == "closing":
                        result = scipy.ndimage.grey_closing(data, structure=structure)
                    else:
                        result = data

                    result[~valid_mask] = ndv

                    y_off = window.row_off - buffered_window.row_off
                    x_off = window.col_off - buffered_window.col_off
                    out_arr = result[y_off:y_off+window.height, x_off:x_off+window.width]

                    dst.write(out_arr, 1, window=window)

        return True
