#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.rasters.fill
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Fill nodata using gdal (idw)

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import rasterio
from rasterio.fill import fillnodata
from .base import RasterHook

class RasterFill(RasterHook):
    """Fill NoData voids using Inverse Distance Weighting (IDW).

    Usage: --hook raster_fill:max_dist=100
    """

    name = "raster_fill"
    default_suffix = "_filled"

    def __init__(self, max_dist=100, smoothing=0, **kwargs):
        super().__init__(**kwargs)
        self.max_dist = float(max_dist)
        self.smoothing = int(smoothing)

    def process_raster(self, src_path, dst_path, entry):
        with rasterio.open(src_path) as src:
            profile = src.profile.copy()
            data = src.read(1)
            mask = src.dataset_mask() # 0 for NoData, 255 for Valid

            filled_data = fillnodata(
                data,
                mask=mask,
                max_search_distance=self.max_dist,
                smoothing_iterations=self.smoothing
            )

            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.write(filled_data, 1)

        return True
