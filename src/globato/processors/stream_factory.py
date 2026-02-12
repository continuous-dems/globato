#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.stream_factory
~~~~~~~~~~~~~

This turns files into point streams.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging

import numpy as np
import numpy.lib.recfunctions as rfn

#from .gdal_proc import GDALReader
from .rio import RasterioReader
from .bag import BAGReader
from .ogr_proc import OGRReader
from .lidar import LASReader
from .multibeam import MBSReader
from .xyz import XYZReader
from transformez.spatial import TransRegion

from fetchez.hooks import FetchHook

logger = logging.getLogger(__name__)

def apply_metadata(chunk, module_weight=1.0, module_unc=0.0):
    """Ensure chunk has 'w' and 'u' fields, merging module-level defaults."""
    
    if chunk is None or len(chunk) == 0:
        return chunk

    if not isinstance(chunk, np.ndarray):
        return chunk

    names = chunk.dtype.names
    if not names: return chunk
    
    new_fields = {}
    
    if 'w' not in names:
        new_fields['w'] = np.full(len(chunk), module_weight, dtype=np.float32)
    
    if 'u' not in names:
        new_fields['u'] = np.full(len(chunk), module_unc, dtype=np.float32)
        
    if new_fields:
        chunk = rfn.append_fields(
            chunk, 
            names=list(new_fields.keys()), 
            data=list(new_fields.values()), 
            usemask=False, 
            asrecarray=True
        )
    
    if 'w' in names:
        chunk['w'] *= module_weight
        
    if 'u' in names:
        # sqrt(point_u^2 + module_u^2)
        if module_unc > 0:
            chunk['u'] = np.sqrt(np.square(chunk['u']) + np.square(module_unc))
        
    return chunk


class StreamFactory:
    """Auto-detects file type and returns the appropriate streaming iterator."""
    
    @staticmethod
    def get_stream(src_fn, **kwargs):
        """Returns a generator (yield_chunks) for the given file."""

        if not os.path.exists(src_fn):
            return None
            
        ext = os.path.splitext(src_fn)[1].lower()
        
        # LiDAR (LAS/LAZ)
        if ext in ['.las', '.laz']:
            return LASReader(src_fn, **kwargs).yield_chunks()

        # Vector Data (OGR)
        # .shp, .000 (S-57), .gdb, .geojson
        # if ext in ['.shp', '.000', '.json', '.geojson', '.kml'] or \
        #    (ext == '.gdb' and os.path.isdir(src_fn)):
        #     return OGRReader(src_fn, **kwargs).yield_chunks()

        # 3. ASCII / XYZ
        if ext in ['.xyz', '.txt', '.csv', '.dat']:
            return XYZReader(src_fn, **kwargs).yield_chunks()

        # Raster Data (Rasterio)
        if ext in ['.tif', '.tiff', '.nc', '.vrt', '.dt0', '.dt1', '.dt2']:
            return RasterioReader(src_fn, **kwargs).yield_chunks()

        # BAG (Rasterio)
        if ext in ['.bag']:
            return BAGReader(src_fn, **kwargs).yield_chunks()

        # Multibeam (MB-System)
        if ext in ['.fbt']:
            return MBSReader(src_fn, **kwargs).yield_chunks()
            
        # If unknown extension, try to open with GDAL. 
        try:
            from osgeo import gdal
            ds = gdal.Open(src_fn)
            if ds:
                #return GDALReader(src_fn, **kwargs).yield_chunks()
                #return RasterioReader(src_fn, **kwargs).yield_chunks()
                ds = None
        except:
            pass

        logger.warning(f'Could not detect stream type for {src_fn}')
        return None

    
    @staticmethod
    def get_reader(src_fn, **kwargs):
        """Returns a generator (yield_chunks) for the given file."""

        if not os.path.exists(src_fn):
            return None
            
        ext = os.path.splitext(src_fn)[1].lower()
        
        # LiDAR (LAS/LAZ)
        if ext in ['.las', '.laz']:
            return LASReader(src_fn, **kwargs)

        # # Vector Data (OGR)
        # # .shp, .000 (S-57), .gdb, .geojson
        # if ext in ['.shp', '.000', '.json', '.geojson', '.kml'] or \
        #    (ext == '.gdb' and os.path.isdir(src_fn)):
        #     return OGRReader(src_fn, **kwargs)

        # ASCII / XYZ
        if ext in ['.xyz', '.txt', '.csv', '.dat']:
            # XYZReader needs to be updated to yield recarrays like the others
            # For now, we assume it does.
            return XYZReader(src_fn, **kwargs)

        # Raster Data (Rasterio)
        if ext in ['.tif', '.tiff', '.nc', '.vrt', '.dt0', '.dt1', '.dt2']:
            return RasterioReader(src_fn, **kwargs)

        if ext in ['.bag']:
            return BAGReader(src_fn, **kwargs)

        if ext in ['.fbt']:
            return MBSReader(src_fn, **kwargs)
            
        # If unknown extension, try to open with GDAL. 
        try:
            from osgeo import gdal
            ds = gdal.Open(src_fn)
            if ds:
                #return GDALReader(src_fn, **kwargs)
                ds = None
        except:
            pass

        logger.warning(f'Could not detect stream type for {src_fn}')
        return None


class DataStream(FetchHook):
    """Auto-detects file type and attaches a stream.
    
    Usage:
      --hook stream_data
      --hook stream_data:chunk_size=10000
    """
    
    name = "stream_data"
    stage = "file"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reader_kwargs = kwargs

        
    def run(self, entries):
        for mod, entry in entries:
            if entry.get('stream'): continue
            
            src = entry.get('dst_fn')
            if not src: continue
            
            reader = StreamFactory.get_reader(src, **self.reader_kwargs)
            if not reader: continue

            w = getattr(mod, 'weight', 1.0)
            u = getattr(mod, 'uncertainty', 0.0)
            
            raw_stream = reader.yield_chunks()
            mod.region = TransRegion.from_list(mod.region)
            if raw_stream:
                if hasattr(reader, 'get_srs'):
                    entry['src_srs'] = reader.get_srs() or 'EPSG:4326'

                #entry['stream'] = stream
                entry['stream'] = self._inject_metadata(raw_stream, w, u)
                entry['stream_type'] = 'xyz_recarray'

        return entries    

    def _inject_metadata(self, stream, w, u):
        """Generator wrapper to apply metadata to every chunk."""
        
        for chunk in stream:
            yield apply_metadata(chunk, w, u)    
