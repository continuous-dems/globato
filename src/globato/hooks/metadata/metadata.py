#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.hooks.metadata.metadata
~~~~~~~~~~~~~~~~~~~~~~~

Inject metadata into a raster

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
import rasterio

from fetchez.hooks import FetchHook

from globato.hooks.rasters.base import is_cog, update_cog_metadata

logger = logging.getLogger(__name__)


class RasterMetadataHook(FetchHook):
    """Injects global tags and band descriptions into an existing GeoTIFF.

    Usage:
      --hook raster_metadata:tags="Project=CRM,Author=NOAA",bands="Elevation,Uncertainty"
    """

    name = "raster_metadata"
    meta_stage = "collection"
    meta_desc = "Inject global metadata tags into a raster."
    meta_category = "metadata"

    def __init__(self, tags=None, bands=None, **kwargs):
        super().__init__(**kwargs)

        # Parse the tags string into a dictionary: "Project=CRM,Author=NOAA" -> {'Project': 'CRM', 'Author': 'NOAA'}
        self.tags = {}
        if tags:
            for kv in tags.split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    self.tags[k.strip()] = v.strip()

        # Parse the bands string into a list: "Elevation,Uncertainty" -> ['Elevation', 'Uncertainty']
        self.bands = [b.strip() for b in bands.split(",")] if bands else []

    def run(self, entries):
        for mod, entry in entries:
            path = entry.get("dst_fn") or entry.get("src_fn")
            if not path or not os.path.exists(path):
                continue

            try:
                if is_cog(path):
                    # GDAL will not edit a COG in place: it would break the layout.
                    logger.warning(
                        f"[{self.name}] {os.path.basename(path)} is a COG, which cannot be edited in place. "
                        "Rewriting it to add the metadata; put raster_metadata before format_cog to avoid the extra write."
                    )
                    update_cog_metadata(path, self.tags, self.bands)
                else:
                    # Open in 'r+' mode to inject metadata without rewriting the data array!
                    with rasterio.open(path, "r+") as dst:
                        if self.tags:
                            dst.update_tags(**self.tags)

                        if self.bands:
                            for i, name in enumerate(self.bands, start=1):
                                if i <= dst.count:
                                    dst.set_band_description(i, name)

                logger.debug(
                    f"[{self.name}] Injected metadata into {os.path.basename(path)}"
                )
            except Exception as e:
                logger.error(f"[{self.name}] Failed to update metadata for {path}: {e}")

        return entries
