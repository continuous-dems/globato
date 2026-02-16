#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.sinks.multi_stack
~~~~~~~~~~~~~~~~~~~~~~~

Multi-band Statistical Gridder (The "Heavy" Stacker).
Generates Z, Count, Weight, Uncertainty, etc.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import sys
import logging
import threading
import numpy as np

import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin
from rasterio.crs import CRS
from rasterio.enums import ColorInterp

from transformez.spatial import TransRegion as Region
from fetchez.hooks import FetchHook
from fetchez.utils import float_or, parse_fmod
from fetchez import utils

from ..transforms.pointz import PointPixels

logger = logging.getLogger(__name__)

# MULTI_STACK ACCUMULATOR
class MultiStackAccumulator:
    """Stateful engine for building a multi-band statistical grid.

    Bands:
      1: Z (Weighted Sum / Value)
      2: Count
      3: Weights (Sum)
      4: Uncertainty (Sum Sq / Variance)
      5: Src_Uncertainty (Sum Sq)
      6: X (Weighted Sum)
      7: Y (Weighted Sum)
    """

    BAND_MAP = {
        'z': 1, 'count': 2, 'weights': 3, 'uncertainty': 4,
        'src_uncertainty': 5, 'x': 6, 'y': 7
    }

    def __init__(self, region, x_inc, y_inc, output_fn, mode='mean', crs=None, verbose=False):
        self.region = Region.from_list(region)
        self.x_inc = float(x_inc)
        self.y_inc = float(y_inc)
        self.output_fn = output_fn
        self.mode = mode
        self.crs = crs
        self.verbose = verbose
        self.lock = threading.Lock()

        self.xcount, self.ycount, self.dst_gt = self.region.geo_transform(
            x_inc=self.x_inc, y_inc=self.y_inc, node='grid'
        )

        self.transform = rasterio.transform.from_origin(
            self.dst_gt[0], self.dst_gt[3], self.dst_gt[1], abs(self.dst_gt[5])
        )

        self._init_raster()

        self.pixel_binner = PointPixels(
            src_region=self.region,
            x_size=self.xcount,
            y_size=self.ycount
        )


    def _init_raster(self):
        """Create the zero-filled accumulation file."""

        if not os.path.exists(os.path.dirname(os.path.abspath(self.output_fn))):
            os.makedirs(os.path.dirname(os.path.abspath(self.output_fn)))

        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': -9999,
            'width': self.xcount,
            'height': self.ycount,
            'count': 7,
            'crs': CRS.from_string(self.crs) if self.crs else None,
            'transform': self.transform,
            'tiled': True,
            'compress': 'lzw',
            'predictor': 2,
            'bigtiff': 'YES'
        }

        with rasterio.open(self.output_fn, 'w', **profile) as dst:
            for key, idx in self.BAND_MAP.items():
                dst.set_band_description(idx, key)

            pass


    def update(self, points):
        """Process a chunk of points: Bin in memory -> Update Disk."""

        if points is None or len(points) == 0: return

        # Use 'sums' mode to get weighted sums for Z, X, Y
        arrays, sub_win, _ = self.pixel_binner(points, mode='sums')
        if arrays['z'] is None: return

        col_off, row_off, width, height = sub_win
        window = Window(col_off, row_off, width, height)

        with self.lock:
            with rasterio.open(self.output_fn, 'r+') as dst:
                current_data = dst.read(window=window)

                def get_band(name): return current_data[self.BAND_MAP[name]-1]

                valid_new = arrays['count'] > 0

                current_data[current_data == -9999] = 0
                current_data[np.isnan(current_data)] = 0

                if self.mode in ['mean', 'weighted_mean']:
                    get_band('z')[valid_new] += arrays['z'][valid_new]
                    get_band('weights')[valid_new] += arrays['weight'][valid_new]
                    get_band('count')[valid_new] += arrays['count'][valid_new]
                    get_band('uncertainty')[valid_new] += np.square(arrays['uncertainty'][valid_new])

                    if 'src_uncertainty' in arrays and arrays['src_uncertainty'] is not None:
                         get_band('src_uncertainty')[valid_new] += arrays['src_uncertainty'][valid_new]

                    get_band('x')[valid_new] += arrays['x'][valid_new]
                    get_band('y')[valid_new] += arrays['y'][valid_new]

                elif self.mode == 'min':
                    cur_z = get_band('z')
                    cur_z[~valid_new & (cur_z == 0)] = 999999

                    update_mask = valid_new & (arrays['z'] < cur_z)

                    get_band('z')[update_mask] = arrays['z'][update_mask]
                    get_band('count')[update_mask] = 1

                elif self.mode == 'max':
                    cur_z = get_band('z')
                    cur_z[~valid_new & (cur_z == 0)] = -999999

                    update_mask = valid_new & (arrays['z'] > cur_z)
                    get_band('z')[update_mask] = arrays['z'][update_mask]
                    get_band('count')[update_mask] = 1

                dst.write(current_data, window=window)


    def finalize(self, ndv=-9999):
        """Convert accumulated sums to final values and write metadata."""

        if self.verbose: logger.info(f"Finalizing Multi Stack Grid: {self.output_fn}")

        with rasterio.open(self.output_fn, 'r+') as dst:
            dst.colorinterp = [ColorInterp.undefined] * dst.count

            for _, window in dst.block_windows(1):
                data = dst.read(window=window)

                z = data[self.BAND_MAP['z']-1]
                cnt = data[self.BAND_MAP['count']-1]
                w = data[self.BAND_MAP['weights']-1]
                unc = data[self.BAND_MAP['uncertainty']-1]
                src_u = data[self.BAND_MAP['src_uncertainty']-1]
                x = data[self.BAND_MAP['x']-1]
                y = data[self.BAND_MAP['y']-1]

                valid = cnt > 0
                data[:, ~valid] = ndv

                if self.mode in ['mean', 'weighted_mean']:
                    # Z = Sum_Z / Sum_W
                    with np.errstate(divide='ignore', invalid='ignore'):
                        # Z = Sum_Z / Sum_W
                        z[valid] = z[valid] / w[valid]

                        # X = Sum_X / Sum_W
                        x[valid] = x[valid] / w[valid]

                        # Y = Sum_Y / Sum_W
                        y[valid] = y[valid] / w[valid]

                        # Src_U = Sum_SrcU / Sum_W (Average Source Error)
                        src_u[valid] = src_u[valid] / w[valid]

                        # Unc = Mean Uncertainty
                        unc[valid] = np.sqrt(unc[valid]) / cnt[valid]


                dst.write(data, window=window)
            dst.nodata = ndv

            # Calculate Statistics
            stats_dict = {}
            for idx in range(1, dst.count + 1):
                stats_dict[idx] = dst.statistics(idx, approx=False, clear_cache=True)

            for idx, stats in stats_dict.items():
                desc = [k for k, v in self.BAND_MAP.items() if v == idx][0]
                dst.update_tags(bidx=idx,
                    STATISTICS_MINIMUM=str(stats.min),
                    STATISTICS_MAXIMUM=str(stats.max),
                    STATISTICS_MEAN=str(stats.mean),
                    STATISTICS_STDDEV=str(stats.std),
                    DESCRIPTION=desc
                )

        return self.output_fn


# MULTI_STACK HOOK
class MultiStackHook(FetchHook):
    """Multi_Stack Gridding Hook.
    accumulates streaming data into a multi-band statistical grid.

    Args:
      res (float/str): Resolution (e.g. '1s', '30', '0.0001'). Default '1s'.
      mode (str): Aggregation mode ('mean', 'min', 'max'). Default 'mean'.
      output (str): Output filename. Default 'output.tif'.

    Usage:
      --hook multi_stack:res=1s,mode=mean,output=dem.tif
    """

    name = 'multi_stack'
    category = 'stream sink'
    stage = 'file'

    def __init__(self, res='1s', output='multi_stack_output.tif', mode='mean', crs=None, **kwargs):
        super().__init__(**kwargs)
        self.res = res
        self.output = output
        self.mode = mode
        self.crs = crs
        self._accumulator = None


    def _init_accumulator(self, region):
        """Initialize the accumulator once we know the region."""

        if self._accumulator: return

        if isinstance(self.res, str) and self.res.endswith('s'):
            inc = float(self.res[:-1]) / 3600.0
            x_inc, y_inc = inc, inc
        elif '/' in str(self.res):
            x_inc, y_inc = map(float, self.res.split('/'))
        else:
            inc = float(self.res)
            x_inc, y_inc = inc, inc

        logger.info(f'Initializing Multi_Stack: {self.output} @ {x_inc},{y_inc} ({self.mode})')
        self._accumulator = MultiStackAccumulator(
            region=region,
            x_inc=x_inc,
            y_inc=y_inc,
            output_fn=self.output,
            mode=self.mode,
            crs=self.crs,
            verbose=True
        )

    def run(self, entries):
        if not self._accumulator:
            region = next((mod.region for mod, _ in entries if getattr(mod, 'region', None)), None)
            if region:
                self._init_accumulator(region)
            else:
                return entries

        for mod, entry in entries:
            stream = entry.get('stream')
            if stream:
                entry['stream'] = self._intercept(stream)

        return entries


    def _intercept(self, stream):
        """Generator wrapper to feed the accumulator."""

        for chunk in stream:
            if self._accumulator:
                self._accumulator.update(chunk)
            yield chunk


    def teardown(self):
        """Finalize the grid after all streams are exhausted."""

        if self._accumulator:
            logger.info('Stream finished. Finalizing Multi Stack grid...')
            self._accumulator.finalize()
            self._accumulator = None
