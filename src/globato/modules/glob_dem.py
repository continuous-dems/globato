#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.formats.glob_dem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A "Super-Module" that fetches, processes, and merges the best available DEMs
for a given region into a single, analysis-ready GeoTIFF.

Useful for:
- Reference Surfaces (RQ Filter)
- Quick Visualization (VizDEM)
- Base layers for modeling

:copyright: (c) 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
import numpy as np
from fetchez import core, cli, utils, spatial
from fetchez.registry import FetchezRegistry

from globato.processors.formats.stream_factory import DataStream
from globato.processors.filters.dropclass import DropClass
from globato.processors.sinks.simple_stack import SimpleStack

try:
    from transformez.grid_engine import GridEngine, GridWriter
    HAS_GRID_ENGINE = True
except ImportError:
    HAS_GRID_ENGINE = False

logger = logging.getLogger(__name__)


@cli.cli_opts(
    help_text="Fetch and merge best available DEMs into a single grid.",
    res="Target resolution (e.g. '1s', '30m', 0.000277)",
    sources="Comma-separated list of modules to use (default: fabdem,gebco_cog)",
    crs="Target CRS (default: EPSG:4326)",
    blend="Blending mode for overlaps (mean, first, last, min, max)",
    fill="Fill gaps/NaNs (bool, default: True)"
)
class GlobDEM(core.FetchModule):
    """Fetches, crops, and merges data from multiple sources into a single DEM."""

    def __init__(self, res="3s", sources=None, crs="EPSG:4326",
                 blend="mean", fill=True, **kwargs):
        super().__init__(name="glob_dem", **kwargs)
        self.res_str = res
        self.target_res = utils.str2inc(res)
        self.target_crs = crs
        self.blend_mode = blend
        self.fill_gaps = utils.str2bool(fill)

        if sources:
            self.source_list = sources.split(',')
        else:
            self.source_list = ['copernicus_glob', 'etopo']

        w, e, s, n = self.region
        self.out_fn = os.path.join(self._outdir, f"glob_dem_{w}_{s}_{self.res_str}.tif")

        self.add_hook(DataStream())
        #self.add_hook(DropClass(classes="7"))
        self.add_hook(
            SimpleStack(
                output=self.out_fn,
                res=res,
                mode="mean",
            )
        )

    def run(self):
        """Orchestrates the fetch -> merge pipeline.
        Results in a single entry pointing to the generated GeoTIFF.
        """

        if not HAS_GRID_ENGINE:
            logger.error("GlobDEM requires 'transformez' package.")
            return

        if not self.region:
            logger.error("Region required for GlobDEM.")
            return

        downloaded_files = []

        w, e, s, n = self.region
        pad = self.target_res * 10
        fetch_region = [w - pad, e + pad, s - pad, n + pad]

        logger.info(f"Fetching sources: {', '.join(self.source_list)}")

        initialized_mods = []
        for mod_name in self.source_list:
            mod_cls = FetchezRegistry.load_module(mod_name)
            if not mod_cls:
                logger.warning(f"Unknown module: {mod_name}")
                continue

            sub_outdir = os.path.join(self._outdir, "sources", mod_name)
            mod_instance = mod_cls(
                src_region=fetch_region,
                outdir=sub_outdir,
            )

            #try:
            mod_instance.run()
            initialized_mods.append(mod_instance)

        core.run_fetchez(initialized_mods)

        for mod in initialized_mods:
            for entry in mod.results:
                self.add_entry_to_results(
                    url=f"file://{entry['dst_fn']}",
                    dst_fn=entry['dst_fn'],
                    data_type="raster"
                )

                #except Exception as e:
                #    logger.error(f"Gridding failed: {e}", exc_info=True)
