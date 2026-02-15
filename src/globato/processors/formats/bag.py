#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.formats.bag
~~~~~~~~~~~~~~~~~~~

Dedicated BAG (Bathymetric Attributed Grid) Reader.
Handles VR-BAGs, standard BAGs, uncertainty bands, and corrupt XML metadata.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from .rio import RasterioReader
from fetchez.utils import float_or

logger = logging.getLogger(__name__)


class BAGReader(RasterioReader):
    """Specialized Reader for BAG files.

    - Automatically handles Variable Resolution (VR) via GDAL Open Options.
    - Reads Band 2 as Uncertainty ('u').
    - Calculates weight based on resolution.
    """

    def __init__(self, src_fn, mode="resampled", min_weight=0, **kwargs):
        super().__init__(src_fn, **kwargs)
        self.mode = mode
        self.min_weight = float_or(min_weight, 0)


    def _calculate_bag_weight(self, transform):
        """Weight = (3 * (10 if res <=3 else 1)) / res"""

        x_res = transform.a
        if x_res == 0: return 1.0

        base_mult = 10 if x_res <= 3.0 else 1
        calc_weight = (3 * base_mult) / x_res

        return max(calc_weight, self.min_weight)


    def _process_bag_dataset(self, src):
        """Internal generator that reads chunks from an open rasterio dataset."""

        bag_weight = self._calculate_bag_weight(src.transform)
        has_unc = (src.count >= 2)

        for ji, window in src.block_windows(1):
            z = src.read(1, window=window)

            if src.nodata is not None:
                mask = (z != src.nodata)
            else:
                mask = ~np.isnan(z)

            if not np.any(mask): continue

            if has_unc:
                u = src.read(2, window=window)
            else:
                u = np.zeros_like(z)

            z_valid = z[mask]
            u_valid = u[mask]

            rows, cols = np.where(mask)
            global_rows = rows + window.row_off
            global_cols = cols + window.col_off

            xs, ys = rasterio.transform.xy(src.transform, global_rows, global_cols, offset='center')

            count = len(z_valid)
            chunk = np.zeros(count, dtype=[
                ('x', 'f8'), ('y', 'f8'), ('z', 'f4'),
                ('w', 'f4'), ('u', 'f4')
            ])

            chunk['x'] = xs
            chunk['y'] = ys
            chunk['z'] = z_valid
            chunk['u'] = u_valid
            chunk['w'] = np.full(count, bag_weight, dtype='float32')

            yield chunk


    def yield_chunks(self):
        """Try opening as VR, fallback to Standard."""

        env_opts = {
            'GDAL_IGNORE_BAG_XML_METADATA': 'YES',
            'OGR_BAG_MIN_VERSION': '1.0'
        }

        # Attempt VR (Variable Resolution)
        vr_opts = {'MODE': 'RESAMPLED_GRID', 'RES_STRATEGY': 'MIN'}

        try:
            with rasterio.Env(**env_opts):
                with rasterio.open(self.src_fn, **vr_opts) as src:
                    # If this succeeds, yield and return
                    yield from self._process_bag_dataset(src)
                    return

        except RasterioIOError as e:
            err_str = str(e)
            if 'No supergrids' in err_str or 'RESAMPLED_GRID mode not available' in err_str:
                logger.debug(f'File is standard BAG (not VR): {self.src_fn}')
            else:
                logger.error(f'Error reading BAG {self.src_fn}: {e}')
                raise e

        # Just open it with rasterio
        try:
            with rasterio.Env(**env_opts):
                with rasterio.open(self.src_fn) as src:
                    yield from self._process_bag_dataset(src)

        except Exception as e:
            logger.error(f'Failed to read Standard BAG {self.src_fn}: {e}')
            return
