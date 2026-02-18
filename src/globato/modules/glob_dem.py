#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.formats.best_dem
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
        super().__init__(**kwargs)
        self.res_str = res
        self.target_res = utils.str2inc(res)
        self.target_crs = crs
        self.blend_mode = blend
        self.fill_gaps = utils.str2bool(fill)

        if sources:
            self.source_list = sources.split(',')
        else:
            self.source_list = ['copernicus', 'gebco_cog']

    def run(self):
        """Orchestrates the fetch -> merge pipeline.
        Results in a single entry pointing to the generated GeoTIFF.
        """

        if not HAS_GRID_ENGINE:
            logger.error("GlobDEM requires 'transformez' package.")
            return

        if not self.region:
            logger.error("Region required for BestDEM.")
            return

        # Fetch Data from Sub-Modules
        downloaded_files = []

        # Buffer region slightly to avoid edge artifacts during resampling
        w, e, s, n = self.region
        pad = self.target_res * 10
        fetch_region = [w - pad, e + pad, s - pad, n + pad]

        logger.info(f"Fetching sources: {', '.join(self.source_list)}")

        for mod_name in self.source_list:
            mod_cls = FetchezRegistry.load_module(mod_name)
            if not mod_cls:
                logger.warning(f"Unknown module: {mod_name}")
                continue

            # Initialize and Run Module
            # We use a sub-directory to keep things clean
            sub_outdir = os.path.join(self._outdir, "sources", mod_name)
            mod_instance = mod_cls(
                src_region=fetch_region,
                outdir=sub_outdir,
            )

            try:
                mod_instance.run()
                core.run_fetchez([mod_instance])

                for entry in mod_instance.results:
                    status = entry.get('status')
                    if status is not None and status == 0:
                        downloaded_files.append(entry['dst_fn'])

            except Exception as e:
                logger.error(f"Failed to fetch {mod_name}: {e}")

        if not downloaded_files:
            logger.error("No source data found.")
            return

        # 2. Configure Output Grid
        # Ensure we snap to resolution
        res_val = self._parse_res(self.res_str)

        # Calculate Dimensions
        width = int(np.ceil((e - w) / res_val))
        height = int(np.ceil((n - s) / res_val))

        out_fn = os.path.join(self._outdir, f"best_dem_{w}_{s}_{self.res_str}.tif")

        # 3. Grid Engine Merge
        logger.info(f"Merging {len(downloaded_files)} files into {width}x{height} grid...")

        try:
            # A. Load & Interpolate (Mosaic)
            # GridEngine handles reprojection and resampling if needed
            grid_data = GridEngine.load_and_interpolate(
                downloaded_files,
                target_region=[w, e, s, n], # Pass raw list, GridEngine handles it
                nx=width,
                ny=height
            )

            # B. Gap Filling (Optional)
            if self.fill_gaps and np.isnan(grid_data).any():
                logger.info("Filling gaps...")
                # Use a reasonable decay based on resolution (e.g. fill 50 pixels ~ 1km-5km)
                grid_data = GridEngine.fill_nans(grid_data, decay_pixels=50)

            # C. Write Output
            # Ensure directory exists
            if not os.path.exists(self._outdir):
                os.makedirs(self._outdir)

            GridWriter.write(out_fn, grid_data, [w, e, s, n])

            logger.info(f"Generated: {out_fn}")

            # 4. Register Result
            # We add the generated file as the single result of this module.
            # This allows downstream hooks (like stream_data) to use it if desired.
            self.add_entry_to_results(
                url=f"file://{out_fn}",
                dst_fn=out_fn,
                data_type="raster"
            )

        except Exception as e:
            logger.error(f"Gridding failed: {e}", exc_info=True)
