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

    def process_chunk(self, data, ndv, entry):
        mask = (data != ndv) & ~np.isnan(data)

        filled_data = fillnodata(
            data,
            mask=mask,
            max_search_distance=self.max_dist,
            smoothing_iterations=self.smoothing
        )

        return filled_data
