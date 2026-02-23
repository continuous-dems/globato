#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.modules.sources
~~~~~~~~~~~~~~~~~~~~~~~

"Smart" wrappers around standard Fetchez modules.
These ensure data is unzipped, filtered, and ready for merging (GlobDEM).
"""

import os
import logging
import rasterio

from fetchez.registry import FetchezRegistry
from fetchez.hooks import FetchHook
from fetchez.hooks.builtins.file_ops.unzip import Unzip
from fetchez.hooks.builtins.metadata.datatype import SetDataType
from fetchez.hooks.builtins.pipeline.fn_filter import FilenameFilter

from globato.processors.formats.stream_factory import DataStream
from globato.processors.filters.rq import ReferenceQuality
from globato.processors.filters.rangez import RangeZ
from globato.processors.filters.dropclass import DropClass
from globato.processors.sinks.simple_stack import SimpleStack

logger = logging.getLogger(__name__)

BaseFabDEM = FetchezRegistry.load_module('fabdem') or object


class GlobFabDEM(BaseFabDEM):
    """Cleaned FABDEM Module.

      - Fetch Zip
      - Unzip
      - Stream (Load Points)
      - RQ Filter (Flag Coastal Creep)
      - Drop Class (Remove Noise)
      - Stack (Save Clean Raster)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.weight = 1

        self.add_hook(Unzip())

        self.add_hook(DataStream())
        self.add_hook(
            ReferenceQuality(
                reference="gebco_cog",
                threshold=50,       # Large deviations from GEBCO = Creep
                mode="diff",
                set_class=7
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

BaseCopernicus = FetchezRegistry.load_module('copernicus') or object

class GlobCopernicus(BaseCopernicus):
    """Cleaned Copernicus DEM.

      - Automatically Unzips.
      - Filters out water (Copernicus is often valid over ocean as 0 or noisy).
      - Drop Class (Remove Noise)
      - Stack (Save Clean Raster)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.weight = 1
        self.add_hook(Unzip())
        self.add_hook(FilenameFilter(match=".tif"))

        self.add_hook(DataStream())
        self.add_hook(RangeZ(min_z=0.01))
        self.add_hook(DropClass(classes="7"))
        self.add_hook(
            SimpleStack(
                output="{base}_clean.tif",
                res="1s",
                mode="mean"
            )
        )


BaseMultibeam = FetchezRegistry.load_module('multibeam') or object

class GlobMultibeam(BaseMultibeam):
    """Cleaned Multibeam

      - Filename_filter
      - Steam (Load Points)
      - RQ Filter
      - drop class
      - Stack (Save Clean Raster)
    """

    def __init__(self, res="1s", **kwargs):
        super().__init__(**kwargs)

        self.add_hook(FilenameFilter(exclude=".inf", stage="pre"))
        self.add_hook(DataStream())
        self.add_hook(
            ReferenceQuality(
                reference="gmrt",
                threshold=5,
                mode="percent",
                builder="grid",
                set_class=7,
            )
        )
        self.add_hook(DropClass(classes="7"))
        self.add_hook(
            SimpleStack(
                output="{base}_clean.tif",
                res=res,
                mode="mean"
            )
        )


# Base Class
BaseHydroNOS = FetchezRegistry.load_module('nos_hydro') or object

class ValidateBAG(FetchHook):
    """Checks if a BAG file is valid HDF5.
    If invalid, deletes it so Fetchez will retry the download next time.
    """

    name = "validate_bag"
    stage = "file"

    def run(self, entries):
        for mod, entry in entries:
            fn = entry['dst_fn']
            if not os.path.exists(fn) or not fn.endswith('.bag'):
                continue

            is_valid = False
            try:
                # 1. Check Magic Number
                with open(fn, 'rb') as f:
                    header = f.read(4)
                    if header == b'\x89HDF':
                        is_valid = True

                if is_valid:
                    with rasterio.Env(CPL_LOG='/dev/null'):
                        with rasterio.open(fn) as src:
                            pass
            except Exception as e:
                is_valid = False

            if not is_valid:
                logger.warning(f"Corrupt file detected: {fn}. Deleting.")
                os.remove(fn)
                entry['status'] = -1

        return entries

class GlobBAG(BaseHydroNOS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.datatype = 'bag'

        self.add_hook(ValidateBAG())
        self.add_hook(FilenameFilter(exclude="_Ellipsoid_", stage="pre"))

class GlobNOSXYZ(BaseHydroNOS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.datatype = "xyz"
        self.src_srs = "EPSG:4326+1089"

        self.add_hook(Unzip())
        self.add_hook(SetDataType(data_type='nox_xyz'))
