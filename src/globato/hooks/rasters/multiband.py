#!/usr/bin/env python

"""
globato.hooks.rasters.multiband
~~~~~~~~~~~~~

Build a multi-band raster from collection of single-bands

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import os

import rasterio
from fetchez.hooks import FetchHook

logger = logging.getLogger(__name__)


class BuildMultiBandHook(FetchHook):
    """Combines multiple single-band rasters into a single multi-band GeoTIFF.
    Automatically labels the bands based on the originating module names.
    """

    name = "build_multiband"
    meta_desc = "Combines multiple rasters into a multi-band raster."
    meta_stage = "collection"
    meta_category = "raster-op"

    def __init__(self, output="stacked_bands.tif", **kwargs):
        super().__init__(**kwargs)
        self.output = os.path.abspath(output)

    def run(self, entries):
        tifs = []
        band_names = []

        for mod, entry in entries:
            path = entry.get("dst_fn") or entry.get("src_fn")
            if (
                path
                and os.path.exists(path)
                and path.endswith((".tif", ".tiff", ".img", ".nc"))
            ):
                with rasterio.open(path) as src0:
                    _stats = src0.stats()

                if _stats[0].max == 0.0:
                    continue

                tifs.append(os.path.abspath(path))
                # Grab the module name (e.g., 'coned', 'copernicus') for the band label!
                # band_names.append(getattr(mod, "name", f"Band_{len(tifs)+1}"))
                band_names.append(os.path.basename(path))

        if not tifs:
            logger.warning(f"[{self.name}] No valid rasters found to stack.")
            return entries

        logger.info(f"[{self.name}] Stacking {len(tifs)} files into {self.output}")

        with rasterio.open(tifs[0]) as src0:
            profile = src0.profile

        profile.update(count=len(tifs))

        with rasterio.open(self.output, "w", **profile) as dst:
            for i, tif in enumerate(tifs, start=1):
                with rasterio.open(tif) as src:
                    dst.write(src.read(1), i)

                dst.set_band_description(i, band_names[i - 1])

        if entries:
            entries[0][1].setdefault("artifacts", {})[self.name] = self.output

        return entries
