#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.reproject
~~~~~~~~~~~~~

reproject the data stream. Hook for fetchez.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import os
import numpy as np
from fetchez.hooks import FetchHook

from transformez.grid_engine import GridEngine

from transformez.srs import SRSParser
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)
    
class StreamReproject(FetchHook):
    """Reprojection Hook.

    Process: 
      - Source(X,Y) -> Hub(X,Y) [PROJ]
      - Hub(Z) + Grid(Hub_X, Hub_Y) -> Hub(Z_new) [NumPy/SciPy]
      - Hub(X,Y) -> Dest(X,Y) [PROJ]
    """
    
    name = "stream_reproject"
    stage = "file"

    def __init__(self, dst_srs, src_srs=None, vert_grid=None, **kwargs):
        super().__init__(**kwargs)
        self.dst_srs = dst_srs
        self.forced_src_srs = src_srs
        self.vert_grid = vert_grid
        
        # Cache key: src_srs -> (t_to_hub, t_from_hub, grid_interpolator)
        self._cache = {}

        
    def _load_grid_interpolator(self, grid_fn):
        """Create a safe interpolator that returns 0.0 for off-grid points."""
        
        if not grid_fn or not os.path.exists(grid_fn):
            return None
            
        try:
            # Use GridEngine to read (handles .tif/.gtx/etc)
            # update this to use RasterioReader
            lons, lats, data = GridEngine._read_raster(grid_fn)
            if data is None: return None

            # SciPy RegularGridInterpolator requires strictly ascending coords
            if lons[0] > lons[-1]:
                lons = np.flip(lons)
                data = np.flip(data, axis=1)
            if lats[0] > lats[-1]:
                lats = np.flip(lats)
                data = np.flip(data, axis=0)

            return RegularGridInterpolator(
                (lats, lons), data, 
                bounds_error=False, fill_value=0.0, method='linear'
            )
        except Exception as e:
            logger.error(f"Failed to load grid {grid_fn}: {e}")
            return None

        
    def _get_pipeline(self, entry_src_srs, region=None):
        if not SRSParser: return None

        actual_src = self.forced_src_srs or entry_src_srs or 'EPSG:4326'        
        if not actual_src: return None
        
        if actual_src in self._cache:
            return self._cache[actual_src]

        parser = SRSParser(actual_src, self.dst_srs, region=region, vert_grid=self.vert_grid)
        t_in, t_out, grid_fn = parser.get_components()
        
        interpolator = self._load_grid_interpolator(grid_fn) if grid_fn else None
        
        self._cache[actual_src] = (t_in, t_out, interpolator)
        return self._cache[actual_src]

    
    def run(self, entries):
        for mod, entry in entries:
            stream = entry.get('stream')
            stream_type = entry.get('stream_type')
            if not stream or stream_type != 'xyz_recarray': continue
            #if not stream: continue

            src_srs = entry.get('src_srs', 'EPSG:4326')
            pipeline = self._get_pipeline(src_srs, region=mod.region)

            if pipeline:
                entry['stream'] = self._apply_transform(stream, pipeline)
                entry['src_srs'] = self.dst_srs

        return entries

    
    def _apply_transform(self, stream, pipeline):
        t_to_hub, t_from_hub, grid_interp = pipeline
        
        for chunk in stream:
            if chunk['x'][0] > 360 and t_to_hub.source_crs.is_geographic:
                logger.warning("Coordinate/CRS Mismatch! Input X > 360 but Source CRS is Geographic. Result will be INF.")
            
            # Source -> Hub (Horizontal Only)
            h_x, h_y = t_to_hub.transform(chunk['x'], chunk['y'])
            
            # Vertical Shift (Manual)
            if grid_interp and chunk['z'] is not None:
                query_points = np.column_stack((h_y, h_x))
                shifts = grid_interp(query_points)
                
                chunk['z'] += shifts

            # Hub -> Dest (Horizontal Only)
            d_x, d_y = t_from_hub.transform(h_x, h_y)
            
            chunk['x'] = d_x
            chunk['y'] = d_y
            
            yield chunk
