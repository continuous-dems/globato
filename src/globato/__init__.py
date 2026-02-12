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

# Import Hooks from processors.
from .processors.pipe import XYZPrinter
from .processors.stream_factory import DataStream
from .processors.provenance import ProvenanceHook
from .processors.reproject import StreamReproject
from .processors.pointz import Point2PixelStream
from .processors.filters import StreamFilter
from .processors.multi_stack import MultiStackHook
from .processors.simple_stack import SimpleStack
from .processors.cog import COGSubset
from .processors.fred import FredGenerator

# Custom fetchez modules
#from .modules.multibeam import MultibeamXYZ
from .modules.gebco import GEBCO_COG

def setup_fetchez(registry_cls):
    """Register All globato capabilities with Fetchez."""
    
    # --- Register Modules  ---
    #registry_cls.register_module('multibeam_xyz', MultibeamXYZ, metadata={'desc': 'Fetch & Convert MB', 'tags': ['multibeam', 'mb', 'xyz', 'bathymetry']})
    registry_cls.register_module('gebco_cog', GEBCO_COG, metadata={'desc': 'Fetch GEBCO as a COG subset', 'tags': ['gebco', 'bathymetry', 'global', 'tid', 'cog']})

    # --- Register Hooks  ---    
    HookRegistry.register_hook(FredGenerator)
    
    # --- Streams (recarrays) ---
    HookRegistry.register_hook(XYZPrinter)
    HookRegistry.register_hook(DataStream)
    HookRegistry.register_hook(StreamReproject)
    HookRegistry.register_hook(StreamFilter)

    # --- Streams (pointz) ---
    HookRegistry.register_hook(Point2PixelStream)
    HookRegistry.register_hook(MultiStackHook)
    HookRegistry.register_hook(SimpleStack)
    HookRegistry.register_hook(ProvenanceHook)
    
    
    # --- Register Presets ---
    #register_multibeam_presets()
    
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
