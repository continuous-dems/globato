#!/usr/bin/env python

"""
globato.schemas.crm
~~~~~~~~~~~~~~~~~~~

Registers CRM DEM-specific schema into the Fetchez engine.
Enforces 1 arc-second resolution and target CRS across both module-level
processing hooks and global pipeline assembly hooks.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging

from fetchez.recipes.schemas import BaseSchema
from fetchez.spatial import parse_region

logger = logging.getLogger(__name__)


class CRMSchema(BaseSchema):
    """The "CRM" schema.

    Enforces:
      - 1 arc-second resolution (0.0002777777777777778 deg)
      - Standard compound CRS: EPSG:4326+3855 (WGS84 horizontal + NAVD88 vertical)
      - Buffer region for fetch/grid coverage
      - Crop/Cut output to delivery spec
      - Inject resolution & CRS across module hooks (warps, masks, reprojectors)
      - Align global hooks (multi_stack, provenance, source_masks, viz)
    """

    name = "crm"
    meta_desc = "Coastal Relief Model Schema (1 arc-second)"
    meta_category = "NCEI"

    TARGET_SRS = "EPSG:4326+3855"
    RES_DEG = 0.0002777777777777778  # 1 arc-second in decimal degrees

    @classmethod
    def apply(cls, config):
        recipe_region = config.get("region")
        dist_region = (parse_region(recipe_region) if recipe_region else [None])[0]
        if not dist_region:
            return config

        # Expand processing region with a slight buffer to avoid edge artifacts
        proc_region = dist_region.copy()
        proc_region.buffer(10)  # 10-cell buffer
        config["region"] = proc_region.to_list()
        logger.debug(
            f"[Schema: {cls.name}] Expanded processing region to {proc_region}."
        )

        global_hooks = config.get("global_hooks", [])
        modules = config.get("modules", [])
        # Mutate Expanded Modules & Module-Level Hooks
        for module in modules:
            hooks = module.get("hooks", [])

            for hook in hooks:
                hook_name = hook.get("name", "").replace("-", "_")
                hook_args = hook.setdefault("args", {})

                # Reprojection Hooks
                if hook_name in ["stream_reproject", "reproject"]:
                    hook_args["dst_srs"] = cls.TARGET_SRS
                    logger.debug(
                        f"[Schema: {cls.name}] Enforced dst_srs='{cls.TARGET_SRS}' in {hook_name}."
                    )

                # Raster Warping / Resampling Hooks
                elif hook_name in ["raster_warp", "rio_warp", "warp"]:
                    hook_args["res"] = cls.RES_DEG
                    hook_args["dst_srs"] = cls.TARGET_SRS
                    logger.debug(
                        f"[Schema: {cls.name}] Enforced res={cls.RES_DEG} and srs in {hook_name}."
                    )

                # Point / Raster Masking
                elif hook_name == "point_raster_mask":
                    hook_args["res"] = cls.RES_DEG
                    logger.debug(
                        f"[Schema: {cls.name}] Enforced res={cls.RES_DEG} in point_raster_mask."
                    )

                # Resampling / Quality Filter Hooks
                elif hook_name == "rq":
                    hook_args["res"] = cls.RES_DEG
                    logger.debug(
                        f"[Schema: {cls.name}] Enforced res={cls.RES_DEG} in rq hook."
                    )

            # Insert safety range_z check right after data streaming starts
            insert_idx = len(hooks)
            for i, hook in enumerate(hooks):
                h_name = hook.get("name", "").replace("-", "_")
                if h_name in ["stream_init", "stream_data"]:
                    insert_idx = i + 1
                    break

            hooks.insert(
                insert_idx,
                {"name": "range_z", "args": {"min_z": -11000, "max_z": 9000}},
            )
            module["hooks"] = hooks

        # 3. Mutate Global Hooks
        for hook in global_hooks:
            hook_name = hook.get("name", "").replace("-", "_")
            hook_args = hook.setdefault("args", {})

            # Stacker alignment
            if hook_name == "multi_stack":
                hook_args.update(
                    {
                        "resolution": cls.RES_DEG,
                        "registration": "grid",
                        "srs": cls.TARGET_SRS,
                        "weight_threshold": "1/.5/.25",
                    }
                )
                logger.debug(
                    f"[Schema: {cls.name}] Aligned multi_stack: res={cls.RES_DEG}, srs={cls.TARGET_SRS}."
                )

            # Provenance map alignment
            elif hook_name == "provenance":
                hook_args["res"] = cls.RES_DEG
                hook_args["crs"] = cls.TARGET_SRS
                logger.debug(
                    f"[Schema: {cls.name}] Aligned provenance: res={cls.RES_DEG}."
                )

            # Source masks alignment
            elif hook_name == "source_masks":
                hook_args["res"] = cls.RES_DEG
                logger.debug(
                    f"[Schema: {cls.name}] Aligned source_masks: res={cls.RES_DEG}."
                )

            # Shading z-range alignment
            elif hook_name == "viz_geoshade":
                hook_args.update({"z_min": -3500, "z_max": 1500})
                logger.debug(
                    f"[Schema: {cls.name}] Aligned viz_geoshade z-range to [-3500, 1500]."
                )

            elif hook_name == "ms_binary_cudem":
                hook_args.update(
                    {
                        "resolutions": "1s/3s/9s/15s",
                        "weights": "1/.5/.25/0",
                        "steps": 3,
                    }
                )
                logger.debug(
                    f"[Schema: {cls.name}] Aligned ms_binary_cudem: res={cls.RES_DEG}."
                )

        # 4. Inject Crop / Cut / Delivery Hooks
        insert_idx = len(global_hooks)
        for i, hook in enumerate(global_hooks):
            if hook.get("name", "").replace("-", "_") == "viz_geoshade":
                insert_idx = i
                break

        proj_name = config.get("project", {}).get("name", "crm_dem")
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

        global_hooks.append(
            {
                # "name": "format_cog",
                # "args": {
                #     "overviews": "2/4/8/16/32",
                #     "resampling": "average",
                # },
                "name": "copy_artifact",
                "args": {
                    "target_dir": "../_crm_deliverables",
                    "match": [delivery_fn, "hillshade.tif"],
                },
            }
        )

        config["global_hooks"] = global_hooks
        return config
