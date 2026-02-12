#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.cog
~~~~~~~~~~~~~

This hook intercepts the url, determines if it's a cog and if so
subsets it to region using Rasterio (GDAL /vsicurl/).

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import ColorInterp
from fetchez.hooks import FetchHook

logger = logging.getLogger(__name__)

class COGSubset(FetchHook):
    """Intercepts COG entries and performs a smart windowed fetch 
    using Rasterio (HTTP Range Requests).
    """
    
    name = "cog_subset"
    stage = "pre"
    desc = "Pass the download url through a cog download."

    def run(self, entries):
        new_entries = []
        
        for mod, entry in entries:
            # Only run if explicitly marked as cog and a region is defined
            if entry.get('cog') and mod.region:                
                src_url = entry['url']
                dst_fn = entry['dst_fn']

                if not os.path.exists(dst_fn):
                    if not os.path.exists(os.path.dirname(dst_fn)):
                        os.makedirs(os.path.dirname(dst_fn))
                    try:
                        self._fetch_subset(src_url, dst_fn, mod.region)
                        
                        # Update entry to point to the local subset
                        entry['original_url'] = entry['url']
                        entry['url'] = f'file://{dst_fn}'
                        entry['status'] = 0

                    except Exception as e:
                        logger.error(f"COG Subset failed for {dst_fn}: {e}")
                        pass
            
            new_entries.append((mod, entry))

        return new_entries

    
    def _fetch_subset(self, url, dst_fn, region):
        """Perform windowed read and write using Rasterio."""
        
        logger.info(f"Subsetting COG: {os.path.basename(dst_fn)}")
        
        west, east, south, north = region[0], region[1], region[2], region[3]
        with rasterio.open(url) as src:
            window = from_bounds(west, south, east, north, transform=src.transform)
            data = src.read(window=window, boundless=True)
            new_transform = src.window_transform(window)
            
            profile = src.profile.copy()
            profile.update({
                'driver': 'GTiff',
                'height': window.height,
                'width': window.width,
                'transform': new_transform,
                'compress': 'deflate',
                'tiled': True
            })
            
            with rasterio.open(dst_fn, 'w', **profile) as dst:
                dst.write(data)
