#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.provenance
~~~~~~~~~~~~~~~~~~~~~~~

Generate bitmap data mask

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
import threading
import numpy as np
import rasterio
from fetchez.hooks import FetchHook
from rasterio.windows import Window
from dlim.processors.pointz import PointPixels

logger = logging.getLogger(__name__)

class ProvenanceHook(FetchHook):
    """Generates a 'Provenance' mask raster.
    Each module is assigned a Bit ID. Pixel value = Bitmask of contributing modules.
    
    Usage:
      fetchez ... --hook provenance:res=1s,output=mask.tif
    """
    
    name = "provenance"
    category = "sink"
    stage = "file"

    def __init__(self, res="1s", output="provenance.tif", **kwargs):
        super().__init__(**kwargs)
        self.res = res
        self.output = output
        self._initialized = False
        self.lock = threading.Lock()
        
        # { 'srtm': 1, 'multibeam': 2, ... }
        self.module_bits = {} 
        self.next_bit = 0

        
    def _init_raster(self, region):
        """Create the zero-filled UInt32 raster."""
        
        if self._initialized: return

        if isinstance(self.res, str) and self.res.endswith('s'):
            inc = float(self.res[:-1]) / 3600.0
            x_inc, y_inc = inc, inc
        else:
            inc = float(self.res)
            x_inc, y_inc = inc, inc

        self.xcount = int(region.width / x_inc)
        self.ycount = int(region.height / y_inc)
        self.transform = rasterio.transform.from_origin(
            region.xmin, region.ymax, x_inc, y_inc
        )
        
        self.pixel_binner = PointPixels(
            src_region=region, x_size=self.xcount, y_size=self.ycount
        )

        profile = {
            'driver': 'GTiff',
            'dtype': 'uint32', # Supports up to 32 modules
            'count': 1,
            'width': self.xcount,
            'height': self.ycount,
            'crs': 'EPSG:4326',
            'transform': self.transform,
            'compress': 'lzw',
            'nodata': 0
        }
        
        with rasterio.open(self.output, 'w', **profile) as dst:
            dst.set_band_description(1, "Module_Bitmask")
            
        self._initialized = True
        logger.info(f"Initialized Provenance Mask: {self.output}")

        
    def _get_module_bit(self, mod_name):
        """Assign a unique bit (power of 2) to this module."""
        
        with self.lock:
            if mod_name not in self.module_bits:
                if self.next_bit > 31:
                    logger.warning("Provenance: >32 modules! Mask overflow.")
                    return 0
                
                bit_val = 1 << self.next_bit
                self.module_bits[mod_name] = bit_val
                self.next_bit += 1
                
                logger.info(f"Provenance Map: {mod_name} -> Bit {self.next_bit} (Val {bit_val})")
                
            return self.module_bits[mod_name]

        
    def run(self, entries):
        if not self._initialized and entries:
            region = next((mod.region for mod, _ in entries if getattr(mod, 'region', None)), None)
            if region: self._init_raster(region)

            
        for mod, entry in entries:
            stream = entry.get('stream')
            if stream:
                bit_val = self._get_module_bit(mod.name)
                entry['stream'] = self._intercept(stream, bit_val)
                
        return entries

    
    def _intercept(self, stream, bit_val):
        """Pass-through stream to update mask."""
        
        for chunk in stream:
            self._update_mask(chunk, bit_val)
            yield chunk

            
    def _update_mask(self, points, bit_val):
        """Bin points and OR the bitmask into the raster."""
        
        if not self._initialized or len(points) == 0: return
        
        arrays, sub_win, _ = self.pixel_binner(points, mode='count')
        
        if arrays['count'] is None: return
        
        has_data = arrays['count'] > 0
        
        col_off, row_off, w, h = sub_win
        window = Window(col_off, row_off, w, h)
        
        with self.lock:
            with rasterio.open(self.output, 'r+') as dst:
                mask_data = dst.read(1, window=window)
                
                # Bitwise OR to add this module's presence
                mask_data[has_data] |= bit_val
                
                dst.write(mask_data, 1, window=window)

                
    def teardown(self):
        """Write the legend to metadata on exit."""
        
        if self._initialized:
            with rasterio.open(self.output, 'r+') as dst:
                for name, idx in self.module_bits.items():
                    tags = {f"MOD_{name}": str(bit) for name, bit in self.module_bits.items()}
                dst.update_tags(bidx=1, **tags)
            logger.info("Finalized Provenance Mask.")
