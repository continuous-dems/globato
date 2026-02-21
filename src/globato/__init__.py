#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato
~~~~~~~~~~~~~

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

# import the fetchez hook registry
from fetchez.hooks.registry import HookRegistry
from fetchez.registry import FetchezRegistry

# --- Import Hooks from processors. ---

# stream factory / formats
from .processors.formats.stream_factory import DataStream
from .processors.formats.fred import FredGenerator
from .processors.formats.cog import COGSubset

# metadata
from .processors.metadata.provenance import ProvenanceHook, SourceMasks
from .processors.metadata.globato_inf import GlobatoInfo

# transforms
from .processors.transforms.reproject import StreamReproject
from .processors.transforms.point_pixels import Point2PixelStream

# filters
from .processors.filters.basic import RangeZ, SpatialCrop
from .processors.filters.reference import RasterMask, DiffZ
from .processors.filters.stats import OutlierZ
from .processors.filters.thinning import BlockThin
from .processors.filters.thinning import BlockMinMax
from .processors.filters.rq import ReferenceQuality
from .processors.filters.cleaning import DropClass

# sinks
from .processors.sinks.pipe import XYZPrinter
from .processors.sinks.multi_stack import MultiStackHook
from .processors.sinks.simple_stack import SimpleStack
from .processors.sinks.gtpc_writer import WriteGTPC

# --- Custom fetchez modules ---

# from .modules.multibeam import MultibeamXYZ
from .modules.gebco import GEBCO_COG
from .modules.glob_dem import GlobDEM
from .modules.sources import GlobCopernicus, GlobFabDEM, GlobMultibeam, GlobBAG

def setup_fetchez(registry_cls):
    """Register All globato capabilities with Fetchez."""

    # --- Register Modules  ---
    #registry_cls.register_module('multibeam_xyz', MultibeamXYZ, metadata={'desc': 'Fetch & Convert MB', 'tags': ['multibeam', 'mb', 'xyz', 'bathymetry']})
    registry_cls.register_module(
        'gebco_cog',
        GEBCO_COG,
        metadata={
            "inherits": "gebco",
            'desc': 'Fetch GEBCO as a COG subset',
            'tags': ['gebco', 'bathymetry', 'global', 'tid', 'cog'],
            'category': 'Globato',
        }
    )
    registry_cls.register_module(
        'glob_dem',
        GlobDEM,
        metadata={
            'desc': 'Fetch and glob the best available DEMs',
            'tags': ['gebco', 'bathymetry', 'global', 'etopo', 'globato'],
            'category': 'Tools',
        }
    )
    registry_cls.register_module(
        'copernicus_glob',
        GlobCopernicus,
        metadata={
            "inherits": "copernicus",
            "desc": "Copernicus Global/European Digital Elevation Models (COP-30/10)",
            "tags": ["satellite", "dsm", "radar", "global", "europe", "clean", "globato"],
            'category': 'Globato',
        }
    )
    registry_cls.register_module(
        'fabdem_glob',
        GlobFabDEM,
        metadata={
            "inherits": "fabdem",
            "tags": ["fabdem", "dem", "dtm", "copernicus", "global", "30m", "clean", "globato"],
            'category': 'Globato',
        }
    )
    registry_cls.register_module(
        'multibeam_glob',
        GlobMultibeam,
        metadata={
            "inherits": "multibeam",
            "tags": ["bathymetry", "multibeam", "ocean", "sonar", "noaa", "ncei", "globato"],
            'category': 'Globato',
        }
    )
    registry_cls.register_module(
        'bag_glob',
        GlobBAG,
        metadata={
            "inherits": "nos_hydro",
            "tags": ["bathymetry", "hydrography", "nos", "noaa", "bag", "soundings", "globato"],
            'category': 'Globato',
        }
    )

    # --- Register Hooks  ---
    HookRegistry.register_hook(FredGenerator)

    # --- Streams (recarrays) ---

    # open stream
    HookRegistry.register_hook(DataStream)

    # metadata
    HookRegistry.register_hook(ProvenanceHook)
    HookRegistry.register_hook(SourceMasks)
    HookRegistry.register_hook(GlobatoInfo)

    # transforms
    HookRegistry.register_hook(StreamReproject)
    HookRegistry.register_hook(Point2PixelStream)

    # filters
    #HookRegistry.register_hook(StreamFilter)
    HookRegistry.register_hook(RangeZ)
    HookRegistry.register_hook(SpatialCrop)
    HookRegistry.register_hook(RasterMask)
    HookRegistry.register_hook(DiffZ)
    HookRegistry.register_hook(OutlierZ)
    HookRegistry.register_hook(BlockThin)
    HookRegistry.register_hook(BlockMinMax)
    HookRegistry.register_hook(ReferenceQuality)
    HookRegistry.register_hook(DropClass)

    # sinks
    HookRegistry.register_hook(XYZPrinter)
    HookRegistry.register_hook(MultiStackHook)
    HookRegistry.register_hook(SimpleStack)
    HookRegistry.register_hook(WriteGTPC)


    # --- Register Presets ---
    #register_multibeam_presets()

setup_fetchez(FetchezRegistry)

# # --- PRESET: MAKE DEM ---
#     register_global_preset(
#         name="make-dem",
#         help_text="Download, clean, vertically transform, and grid data into a DEM.",
#         hooks=[
#             {
#                 "name": "stream_reproject",
#                 "args": {
#                     "dst_srs": "EPSG:6319+5703", # Standardize to NAVD88?
#                     "vert_grid": "auto"
#                 }
#             },
#             {
#                 "name": "filter",
#                 "args": {"method": "outlierz", "threshold": "3.0"}
#             },
#             {
#                 "name": "stack",
#                 "args": {"res": "1s", "mode": "mean", "output": "output_dem.tif"}
#             }
#         ]
#     )

#     # --- PRESET: RAW DATA CLEANUP ---
#     register_global_preset(
#         name="clean-points",
#         help_text="Download and clean point cloud (no gridding).",
#         hooks=[
#             {"name": "stream_reproject", "args": {"dst_srs": "EPSG:4326"}},
#             {"name": "filter", "args": {"method": "block_thin", "res": "0.0001"}}
#         ]
#     )
