#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.recipes.modifiers.ensure_spatial_claim
~~~~~~~~~~~~~~

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
from fetchez.recipes.modifiers import BaseModifier

logger = logging.getLogger(__name__)


class EnsureSpatialClaim(BaseModifier):
    name = "ensure-spatial-claim"
    meta_desc = (
        "Ensure the spatial-claim global hook is present when "
        "claim-grid-filter is used."
    )
    meta_category = "Globato"
    meta_aliases = ["ensure_spatial_claim"]

    @staticmethod
    def _hook_name(hook):
        return str(hook.get("name", "")).replace("_", "-")

    def apply(self, config):
        modules = config.get("modules") or []

        global_hooks = config.get("global_hooks")
        if global_hooks is None:
            global_hooks = []
        config["global_hooks"] = global_hooks

        claim_grid_enabled = any(
            self._hook_name(hook) == "claim-grid-filter"
            for module in modules
            if isinstance(module, dict)
            for hook in module.get("hooks", [])
            if isinstance(hook, dict)
        )

        spatial_claim_enabled = any(
            isinstance(hook, dict) and self._hook_name(hook) == "spatial-claim"
            for hook in global_hooks
        )

        if claim_grid_enabled and not spatial_claim_enabled:
            global_hooks.insert(
                0,
                {"name": "spatial-claim"},
            )

        return config
