#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.formats.stream_factory
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
#from .ogr_proc import OGRReader
from .lidar import LASReader
from .multibeam import MBSReader
from .xyz import XYZReader
from .gtpc import GTPCReader
from .schema import ensure_schema
from transformez.spatial import TransRegion

from fetchez.hooks import FetchHook

logger = logging.getLogger(__name__)


class StreamFactory:
    """Auto-detects file type and returns the appropriate streaming iterator."""

    @staticmethod
    def get_stream(src_fn, **kwargs):
        """Returns a generator (yield_chunks) for the given file."""

        logger.info(src_fn)
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

        if ext == '.gtpc':
            return GTPCReader(src_fn, **kwargs).yield_chunks()

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

        if ext == '.gtpc':
            return GTPCReader(src_fn, **kwargs)

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
    category = "streams"
    priority = 40

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reader_kwargs = kwargs


    def run(self, entries):
        for mod, entry in entries:
            if entry.get('stream'):
                continue

            src = entry.get('dst_fn')
            if not src:
                continue

            reader = StreamFactory.get_reader(src, **self.reader_kwargs)
            if not reader:
                continue

            w = getattr(mod, 'weight', 1.0)
            u = getattr(mod, 'uncertainty', 0.0)

            raw_stream = reader.yield_chunks()
            mod.region = TransRegion.from_list(mod.region)
            if raw_stream:
                if hasattr(reader, 'get_srs'):
                    entry['src_srs'] = reader.get_srs() or 'EPSG:4326'

                #entry['stream'] = stream
                entry['stream'] = ensure_schema(raw_stream, module_weight=w, module_unc=u)
                entry['stream_type'] = 'xyz_recarray'

        return entries
