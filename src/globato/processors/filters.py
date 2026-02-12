#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.filters
~~~~~~~~~~~~~

pointz filters and hook.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
from fetchez.hooks import FetchHook
from . import pointz

logger = logging.getLogger(__name__)

class StreamFilter(FetchHook):
    """Apply PointZ filters to a data stream.
    
    Usage:
      ... --hook filter:method=outlierz,percentile=95
      ... --hook filter:method=block_thin,res=10
      ... --hook filter:method=rangez,min_z=-50,max_z=0
    """
    
    name = 'filter'
    stage = 'file'
    desc = 'filter a point stream'

    def __init__(self, method=None, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.kwargs = kwargs

    def run(self, entries):
        for mod, entry in entries:
            stream = entry.get('stream')
            if not stream: continue
            
            filter_obj = self._init_filter(mod.region)
            
            if filter_obj:
                entry['stream'] = self._apply_filter(stream, filter_obj)
        
        return entries

    def _init_filter(self, region):
        if not self.method: return None
        
        try:
            return pointz.PointFilterFactory.create(
                self.method, 
                points=None, # just init
                region=region,
                verbose=False,
                **self.kwargs
            )
        except Exception as e:
            logger.error(f'Failed to initialize filter {self.method}: {e}')
            return None

        
    def _apply_filter(self, stream, filter_obj):
        for chunk in stream:
            filter_obj.points = chunk        
            filtered_chunk = filter_obj()
            
            if filtered_chunk is not None and len(filtered_chunk) > 0:
                yield filtered_chunk
