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
from fetchez.registry import HookRegistry

logger = logging.getLogger(__name__)


class RegionBufferModifier(BaseModifier):
    name = "buffer-and-cut"
    meta_desc = "Expands the target region by a specified amount or percentage and appends a cut hook."
    meta_category = "Globato"
    meta_aliases = ["buffer_and_cut"]

    # The hooks that turn the buffered region into the DEM: anything carrying the
    # 'interpolation' meta_tag (ms_binary_cudem, ms_cudem, interp_*, raster_fill),
    # plus ms_blend, which blends rather than interpolates and so has no such tag.
    # The cut and crop have to run after the last of them.
    #
    # STAND-IN: this tag-plus-name lookup is temporary. Once hooks declare what they
    # provide (a 'provides' meta_* attribute is planned in the fetchez registry),
    # replace both of these with a lookup of that.
    dem_producer_tag = "interpolation"
    dem_producer_names = ("ms_blend",)

    # Hooks that use the finished DEM, plus anything named viz_*. The cut and crop
    # have to run before the first of them that follows the DEM.
    dem_consumers = ("format_cog", "cleanup_tmp", "copy_artifact")

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

    def _produces_dem(self, hook_name):
        # Stand-in for a 'provides' lookup; see dem_producer_tag above.
        if hook_name in self.dem_producer_names:
            return True
        hook_cls = HookRegistry.get_class(hook_name)
        return bool(hook_cls) and self.dem_producer_tag in (
            getattr(hook_cls, "meta_tags", None) or []
        )

    def _consumes_dem(self, hook_name):
        return hook_name in self.dem_consumers or hook_name.startswith("viz_")

    def apply(self, config):
        region = config.get("region")
        if not region:
            logger.warning(
                f"[{self.name}] No region set in the recipe, so there is nothing to buffer. Skipping the modification."
            )
            return config

        parsed_region = parse_region(region)[
            0
        ]  # update this to handle multiple regions.

        HookRegistry.load_all()
        global_hooks = config.get("global_hooks") or []
        hook_names = [h.get("name", "").replace("-", "_") for h in global_hooks]

        # The buffer only makes sense if something grids the buffered region into a
        # DEM that can be cut back afterwards. Without that, leave the recipe alone.
        dem_idx = max(
            (i for i, n in enumerate(hook_names) if self._produces_dem(n)),
            default=None,
        )
        if dem_idx is None:
            logger.warning(
                f"[{self.name}] No hook in the recipe produces a DEM (an interpolation hook such as "
                "ms_binary_cudem, or ms_blend), so there is nothing to cut back. Skipping the modification."
            )
            return config

        if self.cells is None and self.pct is None:
            logger.warning(
                f"[{self.name}] No buffer provided. Defaulting to 5% buffer."
            )
            self.pct = 5.0

        cells = self.cells or 0
        pct = self.pct or 0

        # After the last hook that produces the DEM, and before the first hook after
        # it that uses the DEM, so that one and everything later see the cropped DEM.
        # If nothing uses it afterwards, the end of the recipe is the right place.
        insert_idx = next(
            (
                i
                for i in range(dem_idx + 1, len(hook_names))
                if self._consumes_dem(hook_names[i])
            ),
            len(global_hooks),
        )
        valid = "raster_cut" not in hook_names

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
