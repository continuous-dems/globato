#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.simple_stack
~~~~~~~~~~~~~

This is the grid engine utility for combining data into a grid.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
import numpy as np
import threading
import rasterio
from rasterio.windows import Window
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage

from fetchez.hooks import FetchHook
from fetchez import utils
from fetchez import spatial

logger = logging.getLogger(__name__)

class PointAccumulator:
    """A lightweight streaming gridder.

    Accumulates Weighted Mean (Sum_Z / Sum_W) into a temporary 2-band GeoTIFF.
    """
    
    def __init__(self, filename, region, x_inc, y_inc, crs="EPSG:4326", verbose=False):
        self.filename = filename
        self.region = region
        self.x_inc = float(x_inc)
        self.y_inc = float(y_inc)
        self.crs = crs
        self.verbose = verbose
        self.lock = threading.Lock()
        
        self.nx = int(self.region.width / self.x_inc)
        self.ny = int(self.region.height / self.y_inc)
        
        self.transform = rasterio.transform.from_origin(
            self.region.xmin, self.region.ymax, self.x_inc, self.y_inc
        )
        
        self.acc_fn = f"{os.path.splitext(filename)[0]}_acc.tif"
        self._init_accumulator()

        
    def _init_accumulator(self):
        """Create the temporary zero-filled accumulation raster."""
        
        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'count': 2,
            'width': self.nx,
            'height': self.ny,
            'crs': self.crs,
            'transform': self.transform,
            'tiled': True,
            'compress': 'lzw',
            'nodata': 0
        }
        
        with rasterio.open(self.acc_fn, 'w', **profile) as dst:
            dst.set_band_description(1, 'Weighted_Sum_Z')
            dst.set_band_description(2, 'Sum_Weights')

            
    def add_points(self, points):
        """Bin points and update the accumulator grid."""

        if points is None or len(points) == 0: return

        cols = np.floor((points['x'] - self.region.xmin) / self.x_inc).astype(int)
        rows = np.floor((self.region.ymax - points['y']) / self.y_inc).astype(int) # Top-down
        
        mask = (cols >= 0) & (cols < self.nx) & (rows >= 0) & (rows < self.ny)
        if not np.any(mask): return
        
        valid_cols = cols[mask]
        valid_rows = rows[mask]
        valid_z = points['z'][mask]
        
        if 'w' in points.dtype.names:
            valid_w = points['w'][mask]
        else:
            valid_w = np.ones_like(valid_z)

        flat_idx = valid_rows * self.nx + valid_cols
        
        unique_flat, inverse = np.unique(flat_idx, return_inverse=True)
        
        pixel_sum_z = np.bincount(inverse, weights=(valid_z * valid_w))
        pixel_sum_w = np.bincount(inverse, weights=valid_w)
        
        u_rows = unique_flat // self.nx
        u_cols = unique_flat % self.nx
        
        r_min, r_max = u_rows.min(), u_rows.max()
        c_min, c_max = u_cols.min(), u_cols.max()
        
        win_w = c_max - c_min + 1
        win_h = r_max - r_min + 1
        window = Window(c_min, r_min, win_w, win_h)
        
        with self.lock:
            with rasterio.open(self.acc_fn, 'r+') as dst:
                current_sum = dst.read(1, window=window)
                current_w = dst.read(2, window=window)
                
                rel_r = u_rows - r_min
                rel_c = u_cols - c_min
                
                current_sum[rel_r, rel_c] += pixel_sum_z
                current_w[rel_r, rel_c] += pixel_sum_w
                
                dst.write(current_sum, 1, window=window)
                dst.write(current_w, 2, window=window)

                
    def finalize(self, ndv=-9999):
        """Divide Sums by Weights to produce final Z grid."""
        
        if self.verbose: logger.info('Finalizing grid...')
        
        with rasterio.open(self.acc_fn) as src:
            profile = src.profile.copy()
            profile.update(count=1, nodata=ndv, dtype='float32')
            
            with rasterio.open(self.filename, 'w', **profile) as dst:
                for _, window in src.block_windows(1):
                    sums = src.read(1, window=window)
                    weights = src.read(2, window=window)
                    
                    out_z = np.full(sums.shape, ndv, dtype='float32')
                    
                    valid = weights > 0
                    out_z[valid] = sums[valid] / weights[valid]

                    dst.write(out_z, 1, window=window)
                    
        if os.path.exists(self.acc_fn):
            try: os.remove(self.acc_fn)
            except: pass
            
        return self.filename


    @staticmethod
    def finalize_from_file(acc_fn, out_fn, ndv=-9999):
        """Static finalizer to convert accumulation raster to result."""
        
        if not os.path.exists(acc_fn): return
        
        with rasterio.open(acc_fn) as src:
            profile = src.profile.copy()
            profile.update(count=1, nodata=ndv, dtype='float32')
            
            with rasterio.open(out_fn, 'w', **profile) as dst:
                for _, window in src.block_windows(1):
                    sums = src.read(1, window=window)
                    weights = src.read(2, window=window)
                    
                    out_z = np.full(sums.shape, ndv, dtype='float32')
                    valid = weights > 0
                    out_z[valid] = sums[valid] / weights[valid]
                    
                    dst.write(out_z, 1, window=window)
        
        try: os.remove(acc_fn)
        except: pass
        logger.info(f'Generated grid: {out_fn}')
    

class SimpleStack(FetchHook):
    name = "simple_stack"
    category = "sink"
    stage = "file"

    def __init__(self, res="1s", output="output.tif", **kwargs):
        super().__init__(**kwargs)
        self.res = res
        self.output = output
        self._accumulator = None

        
    def _init_accumulator(self, region):
        """Initialize the single accumulator for this run."""
        
        if self._accumulator: return

        if isinstance(self.res, str) and self.res.endswith('s'):
            inc = float(self.res[:-1]) / 3600.0
            x_inc, y_inc = inc, inc
        elif '/' in str(self.res):
            x_inc, y_inc = map(float, self.res.split('/'))
        else:
            inc = float(self.res)
            x_inc, y_inc = inc, inc

        out_dir = os.path.dirname(os.path.abspath(self.output))
        if not os.path.exists(out_dir) and out_dir:
            os.makedirs(out_dir)

        logger.info(f"Initializing Simple Gridder: {self.output} @ {x_inc},{y_inc}")
        
        self._accumulator = PointAccumulator(
            filename=self.output,
            region=region,
            x_inc=x_inc,
            y_inc=y_inc,
            verbose=True
        )

        
    def run(self, entries):
        if not self._accumulator:
            region = next((mod.region for mod, _ in entries if getattr(mod, 'region', None)), None)
            if region:
                self._init_accumulator(region)
            else:
                logger.error('SimpleStack: No region found.')
                return entries

        for mod, entry in entries:
            stream = entry.get('stream')
            if stream:
                entry['stream'] = self._intercept(stream)
                
        return entries

    def _intercept(self, stream):
        """Pass-through generator: updates grid, then yields data downstream."""
        
        #count = 0
        for chunk in stream:
            #count += chunk.size
            if self._accumulator:
                self._accumulator.add_points(chunk)
            yield chunk
        #logger.info(f'processed {count} points from ')

        
    def teardown(self):
        """Called by Core after the stream is fully exhausted."""
        
        if self._accumulator:
            logger.info(f'Finalizing grid: {self.output}')
            self._accumulator.finalize()
            self._accumulator = None
