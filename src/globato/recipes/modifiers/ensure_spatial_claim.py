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

    def __init__(self, res=None, **kwargs):
        super().__init__(**kwargs)
        self.res = res

    @staticmethod
    def _hook_name(hook):
        return str(hook.get("name", "")).replace("_", "-")

    def apply(self, config):
        modules = config.get("modules") or []

        global_hooks = config.get("global_hooks")
        if global_hooks is None:
            global_hooks = []
        config["global_hooks"] = global_hooks

        claim_grid_enabled = False

        for module in modules:
            if not isinstance(module, dict):
                continue

            for hook in module.get("hooks", []):
                if not isinstance(hook, dict):
                    continue

                if self._hook_name(hook) != "claim-grid-filter":
                    continue

                claim_grid_enabled = True

                if self.res is not None:
                    hook.setdefault("args", {}).setdefault(
                        "res",
                        self.res,
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
