#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.filters.rq
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reference Quality (RQ) Filter.
Fetches a reference raster (e.g. GEBCO) and filters points that deviate from it.

:copyright: (c) 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
import numpy as np
import rasterio
from fetchez.hooks import FetchHook
from fetchez.registry import FetchezRegistry
from fetchez.core import run_fetchez
from fetchez import utils
from .base import GlobatoFilter

# Optional Imports
try:
    from osgeo import gdal
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False

try:
    from transformez.grid_engine import GridEngine, GridWriter
    HAS_GRID_ENGINE = True
except ImportError:
    HAS_GRID_ENGINE = False

logger = logging.getLogger(__name__)

class ReferenceQuality(GlobatoFilter):
    """Filters points by comparing Z values to a Reference Raster (RQ).

    Builder Modes:
      - 'vrt': Uses GDAL to build a Virtual Raster. Fast, low disk usage.
      - 'grid': Uses GridEngine to mosaic/interpolate/fill a solid GeoTIFF.
                Slower, but handles gaps and overlaps better.

    Args:
        reference (str): Fetchez Module Name (default: 'gebco_cog').
        threshold (float): Max allowed difference.
        mode (str): 'diff' (absolute) or 'percent' (relative).
        builder (str): 'vrt' or 'grid'.
        res (float): Resolution for 'grid' builder (default: 0.004 ~400m).
    """

    name = "rq"

    def __init__(self, reference="gebco_cog", threshold=10, mode="diff",
                 builder="vrt", res=0.00416, **kwargs):
        super().__init__(**kwargs)
        self.ref_source = reference
        self.threshold = float(threshold)
        self.mode = mode.lower()
        self.builder = builder.lower()
        self.res = float(res) # Only used for 'grid' mode
        self.ref_fn = None

    def setup(self, mod, entry):
        """Called once before stream processing starts."""

        if not getattr(mod, 'region', None): return False
        region = getattr(mod, 'region')
        if not self.ref_fn:

            files = self._fetch_reference_files(mod.region)
            if not files:
                logger.warning("[RQ] No reference data found. Skipping.")
                #return entries
                return False

            if self.builder == 'grid' and HAS_GRID_ENGINE:
                self.ref_fn = self._build_grid(files, region)
            else:
                self.ref_fn = self._build_vrt(files, region)

        self.src = rasterio.open(self.ref_fn)
        return True

    def _fetch_reference_files(self, region):
        """Downloads reference data and returns list of paths."""

        if os.path.exists(self.ref_source) and os.path.isfile(self.ref_source):
            return [self.ref_source]

        logger.info(f"[RQ] Fetching reference data: {self.ref_source}...")
        mod_cls = FetchezRegistry.load_module(self.ref_source)

        if not mod_cls:
            return None

        buffered_region = region.buffer(0.1)
        fetcher = mod_cls(src_region=buffered_region)
        fetcher.run()
        run_fetchez([fetcher])

        files = []
        for entry in fetcher.results:
            if fetcher.fetch_entry(entry, check_size=True, verbose=False) == 0:
                files.append(entry['dst_fn'])
        return files

    def _build_vrt(self, files, region):
        """Builds a VRT using GDAL."""

        if not HAS_GDAL:
            logger.error("[RQ] GDAL required for 'vrt' builder.")
            return files[0] if files else None

        vrt_path = os.path.join(os.path.dirname(files[0]), f"rq_ref_{self.name}.vrt")
        try:
            vrt_options = gdal.BuildVRTOptions(resampleAlg='bilinear')
            gdal.BuildVRT(vrt_path, files, options=vrt_options)
            return vrt_path
        except Exception as e:
            logger.warning(f"[RQ] VRT Build failed: {e}. Using first file.")
            return files[0]

    def _build_grid(self, files, region):
        """Builds a mosaicked GeoTIFF using GridEngine."""

        if not HAS_GRID_ENGINE:
            logger.error("[RQ] transformez.grid_engine required for 'grid' builder.")
            return None

        out_path = os.path.join(os.path.dirname(files[0]), f"rq_ref_{self.name}.tif")

        target_region = region.buffer(0.05)
        nx = int(np.ceil((target_region[1] - target_region[0]) / self.res))
        ny = int(np.ceil((target_region[3] - target_region[2]) / self.res))

        logger.info(f"[RQ] Gridding reference surface ({nx}x{ny}) from {len(files)} files...")

        grid_data = GridEngine.load_and_interpolate(files, target_region, nx, ny)
        #grid_data = GridEngine.fill_nans(grid_data, decay_pixels=50)

        GridWriter.write(out_path, grid_data, target_region)
        return out_path

    def filter_chunk(self, chunk):
        nodata = self.src.nodata if self.src.nodata is not None else -9999

        # Sample Reference
        coords = list(zip(chunk['x'], chunk['y']))
        ref_vals = np.fromiter((val[0] for val in self.src.sample(coords)), dtype=np.float32)

        valid_ref = (ref_vals != nodata) & (~np.isnan(ref_vals))

        z = chunk['z']
        diff = np.abs(z - ref_vals)
        is_outlier = np.zeros(len(chunk), dtype=bool)

        if self.mode == 'percent':
            with np.errstate(divide='ignore', invalid='ignore'):
                pct_diff = (diff / np.abs(ref_vals)) * 100
                is_outlier = (pct_diff > self.threshold) & valid_ref
        else:
            is_outlier = (diff > self.threshold) & valid_ref

        return is_outlier

    def teardown(self):
        if hasattr(self, 'src'): self.src.close()
        if self.ref_fn and os.path.exists(self.ref_fn):
            if self.ref_fn.endswith('.vrt'):
                try: os.remove(self.ref_fn)
                except: pass
