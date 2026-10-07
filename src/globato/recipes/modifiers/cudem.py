#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.schemas.cudem
~~~~~~~~~~~~~~~

Registers CUDEM DEM-specific schema into the Fetchez engine.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

from fetchez.recipes.schemas import BaseSchema  # , SchemaRegistry
from fetchez.spatial import parse_region

# from globato.utils import globatize_modules


class CUDEMSchema(BaseSchema):
    """The "cudem" schema.

    Enforces:
      1/9 arc-second resolution
      1/4 degree tiles
      6 cell delivery overlap
      Buffer region for fetching
      Crop output to delivery spec
      Add cudem project metadata
    """

    name = "cudem"

    @classmethod
    def apply(cls, config):
        recipe_region = config.get("region")
        dist_region = (parse_region(recipe_region) if recipe_region else [None])[0]
        if not dist_region:
            return config

        dist_region.warp("EPSG:4269")
        dist_region.buffer(6)  # 6 cell buffer
        proc_region = dist_region.copy()
        proc_region.buffer(pct=5)  # 5 percent buffer

        config["region"] = proc_region.to_list()

        # config["modules"] = globatize_modules(
        #     config.get("modules"), crs="EPSG:4269+5703"
        # )

        global_hooks = config.get("global_hooks", [])
        insert_idx = len(global_hooks)

        # for i, hook in enumerate(global_hooks):
        #     if hook.get("name", "").replace("-", "_") == "ms_binary_cudem":
        #         insert_idx = i
        #         break

        # blend_hooks = [
        #     {
        #         "name": "ms_blend",
        #         "args": {
        #             "weight_threshold": "2.0",
        #             "blend_dist": 60,
        #         },
        #     },
        #     {
        #         "name": "ms_blend",
        #         "args": {
        #             "weight_threshold": "1.0",
        #             "blend_dist": 60,
        #         },
        #     },
        #     {
        #         "name": "ms_blend",
        #         "args": {
        #             "weight_threshold": "0.5",
        #             "blend_dist": 60,
        #         },
        #     },
        #     {
        #         "name": "ms_blend",
        #         "args": {
        #             "weight_threshold": "0.25",
        #             "blend_dist": 60,
        #         },
        #     },
        #     {
        #         "name": "raster_write",
        #         "args": {
        #             "suffix": "_final_blend",
        #             "artifact_id": "blended_checkpoint",
        #         },
        #     },
        #     {
        #         "name": "focus_sink",
        #         "args": {
        #             "target": "blended_checkpoint",
        #         },
        #     },
        # ]
        # global_hooks[insert_idx:insert_idx] = blend_hooks

        for i, hook in enumerate(global_hooks):
            if hook.get("name", "").replace("-", "_") == "raster_metadata":
                insert_idx = i
                break

        proj_name = config.get("project", {}).get("name", "globato")
        base_proj_name = proj_name.split("_")[0]
        delivery_fn = f"{base_proj_name}_{dist_region.format('delivery')}.tif"
        global_hooks.insert(
            insert_idx, {"name": "raster_crop", "args": {"output": delivery_fn}}
        )
        global_hooks.insert(
            insert_idx,
            {
                "name": "raster_cut",
                "args": {"region": dist_region.to_list()},
            },
        )
        config["global_hooks"] = global_hooks
        return config
