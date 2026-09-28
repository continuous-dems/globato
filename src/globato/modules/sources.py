#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.modules.sources
~~~~~~~~~~~~~~~~~~~~~~~

"Smart" wrappers around standard Fetchez modules.
These ensure data is unzipped, filtered, and ready for streaming into multi_stack.
"""

import os
import logging

from fetchez.hooks import FetchHook
from fetchez.hooks.extract import Extract
from fetchez.hooks.set_datatype import SetDatatype
from fetchez.hooks.set_srs import SetSrs
from fetchez.hooks.fn_filter import FilenameFilter
from fetchez.hooks.stream_init import DataStream
from fetchez import cli

from globato.hooks.filters.rq import ReferenceQuality
from globato.hooks.filters.rangez import RangeZ
from globato.hooks.filters.outlierz import OutlierZ

# from globato.hooks.filters.dropclass import DropClass
from globato.hooks.filters.spatial_crop import SpatialCrop
from globato.hooks.filters.point_raster_mask import PointRasterMask

logger = logging.getLogger(__name__)

try:
    from fetchez.modules.fabdem import FABDEM as BaseFABDEM
except ImportError:
    BaseFABDEM = object

try:
    from fetchez.modules.copernicus import CopernicusDEM as BaseCopernicus
except ImportError:
    BaseCopernicus = object

try:
    from fetchez.modules.multibeam import MBDB as BaseMultibeam
except ImportError:
    BaseMultibeam = object

try:
    from fetchez.modules.hydronos import HydroNOS as BaseHydroNOS
except ImportError:
    BaseHydroNOS = object


@cli.cli_opts(
    help_text="Forest and Building (removed) Copernicus DEMs",
)
class GlobFABDEM(BaseFABDEM):
    """Cleaned FABDEM Module.

    - Fetch Zip
    - Extract
    - Stream (Load Points)
    - RQ Filter (Flag Coastal Creep)
    """

    name = "glob_fabdem"
    meta_tags = [
        "fabdem",
        "dem",
        "dtm",
        "copernicus",
        "global",
        "30m",
        "clean",
        "globato",
        "glob-stream",
    ]
    meta_category = "Globato"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.weight = 1
        self.add_hook(Extract())


@cli.cli_opts(
    help_text="Copernicus Elevation Data",
)
class GlobCopernicus(BaseCopernicus):
    """Cleaned Copernicus DEM.

    Set `datatype` to 3 for COP-10 or 1 for COP-30

    Extracts, filters for .tif, sets rio datatype,
    initiates stream, and drops anomalous 0-valuesx.
    """

    name = "glob_copernicus"
    meta_desc = "Copernicus Global/European Digital Elevation Models (COP-30/10)"
    meta_tags = [
        "satellite",
        "dsm",
        "radar",
        "global",
        "europe",
        "clean",
        "globato",
        "glob-stream",
    ]
    meta_category = "Globato"

    def __init__(self, datatype=3, weight=1.0, **kwargs):
        # Pass datatype into the base class to control COP-30 vs COP-90
        super().__init__(datatype=datatype, **kwargs)

        self.weight = weight

        self.add_hook(Extract())
        self.add_hook(FilenameFilter(match=".tif"))
        self.add_hook(SetDatatype(data_type="rio"))
        self.add_hook(PointRasterMask(barrier="coastline", invert=False, res="3s"))
        self.add_hook(RangeZ(min_z=0.01))


@cli.cli_opts(
    help_text="NCEI Multibeam data",
)
class GlobMultibeam(BaseMultibeam):
    """Cleaned NCEI Multibeam (MBDB).

    Mirrors global_bato.yaml: Sets SRS, crops spatially, and runs
    the Reference Quality filter against GMRT to flag outliers.
    """

    name = "glob_multibeam"
    meta_tags = [
        "bathymetry",
        "multibeam",
        "ocean",
        "sonar",
        "noaa",
        "ncei",
        "globato",
        "glob-stream",
    ]
    meta_category = "Globato"

    def __init__(self, weight=1.0, want_inf=True, **kwargs):
        super().__init__(want_inf=want_inf, **kwargs)

        self.weight = weight

        self.add_hook(FilenameFilter(match=".tif"))
        self.add_hook(SetSrs(srs="EPSG:4326+5866"))
        self.add_hook(DataStream())
        self.add_hook(SpatialCrop())
        self.add_hook(
            ReferenceQuality(
                reference="gmrt/gebco",
                threshold=5,
                mode="percent",
                builder="grid",
                res="1s",
                # set_class=7,
            )
        )


class ValidateBAG(FetchHook):
    """Checks if a BAG file is valid HDF5."""

    name = "validate_bag"
    meta_stage = "file"

    def run(self, entries):
        for mod, entry in entries:
            fn = entry["dst_fn"]
            if not os.path.exists(fn) or not fn.endswith(".bag"):
                continue

            is_valid = False
            try:
                with open(fn, "rb") as f:
                    if f.read(4) == b"\x89HDF":
                        is_valid = True
            except Exception:
                pass

            if not is_valid:
                logger.warning(f"Corrupt file detected: {fn}. Deleting.")
                os.remove(fn)
                entry["status"] = -1

        return entries


@cli.cli_opts(
    help_text="NOAA NOS Hydrographic Surveys (BAG)",
)
class GlobBAG(BaseHydroNOS):
    name = "glob_bag"
    meta_tags = [
        "bathymetry",
        "hydrography",
        "nos",
        "noaa",
        "bag",
        "globato",
        "glob-stream",
    ]
    meta_category = "Globato"
    meta_desc = "NOAA NOS Hydrographic Surveys (BAG)"

    def __init__(self, weight=3.0, **kwargs):
        kwargs.pop("datatype")
        super().__init__(datatype="bag", weight=weight, **kwargs)
        # self.datatype = "bag"
        self.weight = weight

        self.add_hook(ValidateBAG())
        self.add_hook(FilenameFilter(exclude="Ellipsoid", stage="pre"))
        self.add_hook(SetDatatype(data_type="bag"))
        self.add_hook(SpatialCrop())


@cli.cli_opts(
    help_text="NOAA NOS Hydrographic Surveys (XYZ)",
)
class GlobNOSXYZ(BaseHydroNOS):
    name = "glob_nos"
    meta_tags = ["bathymetry", "nos", "noaa", "xyz", "legacy", "globato", "glob-stream"]
    meta_category = "Globato"
    meta_desc = "NOAA NOS Hydrographic Surveys (XYZ soundings)"

    def __init__(self, weight=0.35, **kwargs):
        kwargs.pop("datatype", None)
        super().__init__(datatype="xyz", **kwargs)

        self.weight = weight

        self.add_hook(Extract())
        self.add_hook(SetDatatype(data_type="nox-xyz"))
        self.add_hook(SetSrs(srs="EPSG:4326+5866"))
        self.add_hook(SpatialCrop)
        self.add_hook(OutlierZ())
