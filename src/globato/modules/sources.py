#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.modules.sources
~~~~~~~~~~~~~~~~~~~~~~~

"Smart" wrappers around standard Fetchez modules.
These ensure data is unzipped, filtered, and ready for merging (GlobDEM).
"""

from fetchez.registry import FetchezRegistry
from fetchez.hooks.file_ops import Unzip
from fetchez.hooks import FetchHook
from fetchez.hooks.basic import FilenameFilter
from globato.processors.formats.stream_factory import DataStream
from globato.processors.filters.rq import ReferenceQuality
from globato.processors.filters.basic import RangeZ
from globato.processors.filters.cleaning import DropClass
from globato.processors.sinks.simple_stack import SimpleStack

# Dynamically load base classes to avoid hard imports if plugins are missing
BaseFabDEM = FetchezRegistry.load_module('fabdem') or object
BaseCopernicus = FetchezRegistry.load_module('copernicus') or object

class RegisterCleanRaster(FetchHook):
    """Registers the output of SimpleStack as a formal module result.
    Optionally hides the original (raw) file from downstream tools.
    """

    name = "register_clean"
    stage = "file" # run before run_fetchez starts downloading

    def __init__(self, clean_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.clean_fn = clean_fn

    def run(self, entries):
        for mod, entry in entries:
            orig_fn = entry.get('dst_fn')
            base, ext = orig_fn

            clean_fn = f"{base}_clean{ext}"
            if clean_fn and clean_fn.endswith('.tif'):
                entry['dst_fn'] = clean_fn
                entry['data_type'] = 'raster'
                entry['unclean_fn'] = orig_fn

        return entries

class CleanFabDEM(BaseFabDEM):
    """Cleaned FABDEM Module.

    Pipeline:
      1. Fetch Zip
      2. Unzip
      3. Stream (Load Points)
      4. RQ Filter (Flag Coastal Creep)
      5. Drop Class (Remove Noise)
      6. Stack (Save Clean Raster)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Unzip (FABDEM comes as .zip)
        self.add_hook(Unzip())

        self.add_hook(DataStream())
        self.add_hook(
            ReferenceQuality(
                reference="gebco_cog",
                threshold=50,       # Large deviations from GEBCO = Creep
                mode="diff",
                set_class=7         # Mark as Noise
            )
        )
        self.add_hook(DropClass(classes="7"))
        #self.hooks.append(SimpleStack())
        self.add_hook(
            SimpleStack(
                output="_clean.tif", # Hooks often support simple templating
                res="1s",                  # Keep original resolution
                mode="mean"
            )
        )
        #self.add_hook(RegisterCleanRaster())

class CleanCopernicus(BaseCopernicus):
    """Analysis-Ready Copernicus DEM.

    - Automatically Unzips.
    - Filters out water (Copernicus is often valid over ocean as 0 or noisy).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_hook(Unzip())
        self.add_hook(FilenameFilter(match=".tif"))

        self.add_hook(DataStream())
        self.add_hook(RangeZ(min_z=0.01))
        self.add_hook(DropClass(classes="7"))
        self.add_hook(
            SimpleStack(
                output="{base}_clean.tif", # Hooks often support simple templating
                res="1s",                  # Keep original resolution
                mode="mean"
            )
        )
        #self.add_hook(RegisterCleanRaster())
