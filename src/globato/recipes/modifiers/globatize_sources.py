#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.recipes.modifiers.globatize_sources
~~~~~~~~~~~~~~
"""

import logging
from fetchez.recipes.modifiers import BaseModifier

logger = logging.getLogger(__name__)


class GlobatizeSources(BaseModifier):
    name = "globatize_sources"

    def __init__(self, dst_srs=None, res=None, **kwargs):
        super().__init__(**kwargs)
        self.dst_srs = dst_srs
        self.res = res

    def apply(self, config):
        for module in config.get("modules", []):
            if not isinstance(module, dict):
                continue

            hooks = module.setdefault("hooks", [])

            reprojections = [
                hook
                for hook in hooks
                if isinstance(hook, dict)
                and hook.get("name")
                in (
                    "stream_reproject",
                    "stream-reproject",
                )
            ]

            if self.dst_srs is not None:
                reprojections = [
                    hook
                    for hook in hooks
                    if isinstance(hook, dict)
                    and hook.get("name") in ("stream_reproject", "stream-reproject")
                ]

                if reprojections:
                    # Update the existing transformation configuration.
                    for hook in reprojections:
                        hook.setdefault("args", {})["dst_srs"] = self.dst_srs

                    # Move the primary reprojection hook to the front.
                    primary = reprojections[0]
                    hooks.remove(primary)
                    hooks.insert(0, primary)

                else:
                    # run_fetchez will place stream-init before this.
                    hooks.insert(
                        0,
                        {
                            "name": "stream_reproject",
                            "args": {
                                "dst_srs": self.dst_srs,
                                "cache_dir": "%shared_cache%",
                            },
                        },
                    )

        return config
