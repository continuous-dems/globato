#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.formats.glob_coast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Super-Module that generates a high-quality Coastline Mask.
Merges Vectors (NHD, OSM) and Rasters (Copernicus, GMRT) into a unified product.
"""

import os
import logging
from fetchez import core, cli, utils
from fetchez.hooks.builtins.file_ops.unzip import Unzip
from fetchez.hooks.builtins.pipeline.fn_filter import FilenameFilter
from fetchez.registry import FetchezRegistry
from globato.processors.sinks.coastline_stack import CoastlineStack
from globato.processors.hooks.osm_landmask import OSMLandmask

logger = logging.getLogger(__name__)

@cli.cli_opts(
    help_text="Generate a High-Resolution Coastline Mask.",
    res="Target resolution (e.g. '1s', '30m')",
    sources="Comma-separated sources (default: copernicus,nhd,osm_landmask,hydrolakes)",
    polygonize="Output vector polygons (bool, default: True)"
)
class GlobCoast(core.FetchModule):
    """Synthesizes a coastline from multiple sources.
    Uses 'Weighted Voting' to resolve conflicts (e.g. NHD water overrides Copernicus land).
    """

    def __init__(self, res="1s", sources=None, polygonize=True, **kwargs):
        super().__init__(name="glob_coast", **kwargs)

        # Default Hierarchy:
        # High-Res Hydrography (NHD, HydroLakes)
        # High-Res DEM (Copernicus/NASADEM)
        # Vector Coastline (OSM)
        # Background (GMRT)
        if not sources:
            self.source_list = ['nhd', 'osm_landmask', 'copernicus', 'gmrt']
        else:
            self.source_list = sources.split(',')

        self.res = res
        self.polygonize = utils.str2bool(polygonize)

        w, e, s, n = self.region
        self.out_fn = os.path.join(self._outdir, f"coastline_{w}_{s}_{res}.tif")

        self.add_hook(
            CoastlineStack(
                output=self.out_fn,
                res=self.res,
                region=self.region,
                polygonize=self.polygonize
            )
        )

    def run(self):
        """Fetch sources and feed them to the CoastlineStack."""

        if not self.region:
            logger.error("GlobCoast requires a region.")
            return

        initialized_mods = []

        for mod_name in self.source_list:

            w, e, s, n = self.region
            pad = 0.1
            fetch_region = [w - pad, e + pad, s - pad, n + pad]

            # Special handling for OSM Landmask (It's a Hook, not a Module in Registry)
            # let's add this as a module soon!
            if mod_name == 'osm_landmask':
                landmask_fn = f"temp_landmask_{self.region[0]}.geojson"
                osm_hook = OSMLandmask(filename=landmask_fn)

                mock_entries = [(self, {'dst_fn': 'dummy'})]
                osm_hook.run(mock_entries)

                # Register result
                if os.path.exists(landmask_fn):
                    self.add_entry_to_results(
                        url=f"file://{landmask_fn}",
                        dst_fn=landmask_fn,
                        data_type='osm_landmask' # Important for weight lookup
                    )
                continue

            if mod_name == 'nhd':
                mod_cls = FetchezRegistry.load_module('tnm')

                mod_instance = mod_cls(
                    src_region=fetch_region,
                    datasets="14",
                    extents="'HU-8 Subbasin,HU-4 Subregion'",
                    outdir=os.path.join(self._outdir, "sources", mod_name),
                )
                mod_instance.add_hook(FilenameFilter(match='GDB', stage='pre'))
                mod_instance.add_hook(Unzip())

            else:
                # Standard Modules
                mod_cls = FetchezRegistry.load_module(mod_name)
                if not mod_cls:
                    logger.warning(f"Unknown source: {mod_name}")
                    continue

                mod_instance = mod_cls(
                    src_region=fetch_region,
                    outdir=os.path.join(self._outdir, "sources", mod_name)
                )

            # Run fetch
            try:
                mod_instance.run()
                core.run_fetchez([mod_instance])

                for entry in mod_instance.results:
                    dst = entry.get('dst_fn') if isinstance(entry, dict) else entry[1]

                    self.add_entry_to_results(
                        url=f"file://{dst}",
                        dst_fn=dst,
                        data_type=mod_name # Important for weight lookup
                    )
            except Exception as e:
                logger.error(f"Failed to fetch {mod_name}: {e}")

        return self
