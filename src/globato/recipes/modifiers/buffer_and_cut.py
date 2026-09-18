#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.recipes.modifiers.buffer_and_cut
~~~~~~~~~~~~~~

Modifies the recipes region value by buffering it and adding
cut and crop hooks at the end to return the output to the
desired region.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
from fetchez.utils import str2inc, str2bool, float_or, str_or
from fetchez.spatial import parse_region
from fetchez.recipes.modifiers import BaseModifier

logger = logging.getLogger(__name__)


class RegionBufferModifier(BaseModifier):
    name = "buffer-and-cut"
    meta_desc = "Expands the target region by a specified amount or percentage and appends a cut hook."
    meta_category = "Globato"
    meta_aliases = ["buffer_and_cut"]

    def __init__(
        self, cells=None, pct=None, inc=None, outname=None, force=False, **kwargs
    ):
        # None means "not given", which apply() tells apart from an explicit 0.
        self.cells = float_or(cells)
        self.pct = float_or(pct)
        self.inc = str2inc(str_or(inc, "1"))
        # 'outname' must match the basename of the DEM the recipe writes, so the
        # cropped DEM replaces the buffered one (see apply()). Placeholders are
        # resolved per tile, after modifiers are applied.
        self.outname = str_or(outname) or "%name%_%batch_name%"
        self.force = str2bool(force)

        if "increment" in kwargs.keys():
            self.inc = str2inc(str_or(kwargs["increment"], "1"))

    def apply(self, config):
        region = config.get("region")
        if not region:
            return config

        parsed_region = parse_region(region)[
            0
        ]  # update this to handle multiple regions.

        if self.cells is None and self.pct is None:
            logger.warning(
                f"[{self.name}] No buffer provided. Defaulting to 5% buffer."
            )
            self.pct = 5.0

        cells = self.cells or 0
        pct = self.pct or 0

        valid = True
        global_hooks = config.get("global_hooks", [])
        insert_idx = None

        for i, hook in enumerate(global_hooks):
            hook_name = hook.get("name", "").replace("-", "_")
            # Only the first format_cog, so every later hook sees the cropped DEM.
            # Keep scanning past it: a raster_cut may come later in the recipe.
            if hook_name == "format_cog" and insert_idx is None:
                insert_idx = i
            if hook_name == "raster_cut":
                valid = False

        if insert_idx is None:
            insert_idx = len(global_hooks)

        if not valid and not self.force:
            logger.warning(
                "A raster_cut hook is already present in the recipe, skipping the modification, use 'force=True' to inject it anyway."
            )

        else:
            buffer_region = parsed_region.copy().buffer(
                pct=pct, x_inc=self.inc, y_inc=self.inc
            )
            delivery_region = parsed_region.copy().buffer(
                x_bv=cells * self.inc, y_bv=cells * self.inc
            )
            config["region"] = buffer_region.to_list()
            if pct:
                logger.info(
                    f"[{self.name}] Expanded processing region to {buffer_region}."
                )

            # No suffix: the cropped DEM is written over the buffered one, so that
            # later hooks (format_cog, copy_artifact) pick up the delivery-sized DEM
            # under the name the recipe already expects.
            global_hooks.insert(
                insert_idx,
                {
                    "name": "raster_crop",
                    "args": {"output": f"{self.outname}.tif"},
                },
            )
            global_hooks.insert(
                insert_idx,
                {
                    "name": "raster_cut",
                    "args": {
                        "region": delivery_region.to_list(),
                    },
                },
            )
            logger.info(
                f"[{self.name}] Injected 'raster-cut' in global hooks with region: {delivery_region}."
            )

        return config
