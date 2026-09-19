#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.api
~~~~~~~~~~~

High-level Python API for Globato.
Provides interface for streaming, processing, and accessing geospatial data.

:copyright: (c) 2025 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import yaml
import logging
from typing import Union, List, Optional, Generator

from fetchez.recipe import Recipe
from fetchez.registry import HookRegistry
from fetchez.utils import int_or, str2inc, parse_hook_string, compile_sources
from fetchez.api import _compile_modules
from fetchez.spatial import parse_region

from globato.streams.base import GlobatoStream
from globato.utils import globatize_modules, make_recipe_config

logger = logging.getLogger(__name__)


def read(
    sources: Union[str, List[str]],
    region: Optional[Union[str, List[float]]] = None,
    shared_cache: Optional[str] = None,
    target_srs: Optional[str] = None,
    **kwargs,
) -> GlobatoStream:
    """The unified entry point for the Globato streaming API.

    Handles local file paths, directories, fetchez modules, and recipes.
    All reader options (data_type, classes, vertical_datum, etc.) are
    forwarded via kwargs.
    """

    modules = _compile_modules(
        sources, region=region, shared_cache=shared_cache, **kwargs
    )

    parsed_region = parse_region(region)[0] if region else None

    return GlobatoStream(modules=modules, region=parsed_region, target_srs=target_srs)


def build(
    sources: Union[str, List[str]],
    region: Union[str, List[float]],
    increment: str,
    format: str = "GTiff",
    outname: str = "globato_dem",
    outdir: Optional[str] = None,
    t_srs: str = "EPSG:4326",
    nodata: float = -9999.0,
    algo: str = "ms_binary_cudem:barrier=osm",
    stack_mode: str = "mixed",
    filters: Optional[List[str]] = None,
    clip: Optional[str] = None,
    extend: str = "0:0",
    limits: Optional[str] = None,
    weights: str = "auto",
    blend: Optional[str] = None,
    modifier: Optional[List[str]] = None,
    schema: Optional[List[str]] = None,
    shared_cache: Optional[str] = None,
    metadata: Optional[str] = None,
    export: bool = False,
    refresh: bool = False,
    fail_fast: bool = False,
    **kwargs,
) -> Generator:
    """Build a Digital Elevation Model recipe and execute it programmatically."""

    HookRegistry.load_all()

    if isinstance(sources, str):
        sources = [sources]

    filters = filters or []
    parsed_modifiers = [parse_hook_string(m) for m in (modifier or [])]
    parsed_schemas = [s for s in (schema or [])]

    compiled_modules = globatize_modules(
        compile_sources(sources),
        shared_cache=shared_cache,
        crs=t_srs,
        res=increment,
    )

    base_outdir = os.path.abspath(outdir) if outdir else os.path.abspath(".")

    # --- Parse Extend ---
    ext_parts = str(extend).split(":")
    ext_cells = int(ext_parts[0]) if len(ext_parts) > 0 else 0
    ext_pct = float(ext_parts[1]) if len(ext_parts) > 1 else 0.0

    # --- Weight Tiers ---
    base_res = str2inc(increment)
    if str(weights).lower() == "auto":
        target_max_res = (
            str2inc("15s") if (base_res < 1 or str(increment).endswith("s")) else 500
        )
        auto_res_list = [base_res]
        current_res = base_res

        while current_res < target_max_res and len(auto_res_list) < 6:
            current_res *= 3.0
            auto_res_list.append(current_res)

        num_steps = max(1, len(auto_res_list) - 1)
        master_weights = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0]
        weight_list = master_weights[:num_steps][::-1]
        master_blends = [1, 2, 5, 15, 45, 135, 405]
        blend_list = master_blends[: len(auto_res_list)][::-1]
    else:
        weight_list = sorted([float(w) for w in str(weights).split("/")], reverse=True)
        auto_res_list = [base_res * (3**i) for i in range(len(weight_list) + 1)]
        blend_list = [int_or(b, 10) for b in str(blend).split("/")] if blend else []

    batch_outname = "%name%_%batch_name%"

    # --- Base Hooks ---
    global_hooks = [
        {"name": "spatial-crop"},
        {"name": "audit"},
        {"name": "enrich"},
        {"name": "transfer_log"},
        {"name": "drop_class"},
        {
            "name": "provenance",
            "args": {"res": increment, "output": f"{batch_outname}_provenance.tif"},
        },
        {
            "name": "source_masks",
            "args": {
                "res": increment,
                "output": f"{batch_outname}_sources.vrt",
                "vector_output": f"{batch_outname}_sm.gpkg",
            },
        },
    ]

    # --- Multi Stack ---
    global_hooks.append(
        {
            "name": "multi_stack",
            "args": {
                "res": increment,
                "crs": t_srs,
                "mode": stack_mode,
                "nodata": nodata,
                "weight_threshold": "/".join([str(x) for x in weight_list]),
                "output": f"{batch_outname}_stack.tif",
            },
        }
    )
    global_hooks.append({"name": "focus_sink", "args": {"target": "multi_stack"}})
    global_hooks.append(
        {
            "name": "raster_stream",
            "args": {
                "stream_type": "raster",
                "chunk_size": 2048,
                "stage": "collection",
            },
        }
    )
    # --- Interpolation Algorithm ---
    algo_hook = parse_hook_string(algo)
    if algo_hook["name"] in ["ms_cudem", "ms_binary_cudem"]:
        args = algo_hook.setdefault("args", {})
        args["resolutions"] = "/".join([str(r) for r in auto_res_list])
        args["weights"] = weight_list
        args["steps"] = len(weight_list)
        if "blend_dists" not in args and blend_list:
            args["blend_dists"] = "/".join(map(str, blend_list))
        # if "barrier" not in args:
        args["barrier"] = "osm"
        args["bathy_max_z"] = "ocean:-0.01,river:0,lake:None,wetland:0,estuary:0"

    algo_hook.setdefault("args", {})["output"] = f"{batch_outname}.tif"
    global_hooks.append(algo_hook)

    # --- Format & Hillshade ---
    global_hooks.append(
        {
            "name": "format_cog",
            "args": {"overviews": "2/4/8/16/32", "resampling": "average"},
        }
    )
    global_hooks.append(
        {
            "name": "viz_geoshade",
            "args": {
                "output": f"{batch_outname}_hs.tif",
                "cmap": "coastal_relief",
                "cog": True,
            },
        }
    )
    global_hooks.append({"name": "cleanup_tmp", "args": {"target_dir": "tmp"}})

    # --- Build Config ---
    config = make_recipe_config(
        outname, region, compiled_modules, global_hooks, crs=t_srs
    )

    if ext_cells > 0 or ext_pct > 0:
        config["modifiers"] = [
            {
                "name": "buffer_and_cut",
                "args": {
                    "cells": ext_cells,
                    "pct": ext_pct,
                    "inc": increment,
                    "outname": batch_outname,
                },
            }
        ]

    config["schemas"] = [{"name": "validate-recipe"}]
    if parsed_modifiers:
        config["modifiers"].extend(parsed_modifiers)
    if parsed_schemas:
        config["schemas"].extend(parsed_schemas)

    if export:
        os.makedirs(base_outdir, exist_ok=True)
        out_yaml = os.path.join(os.getcwd(), f"{outname}_recipe.yaml")
        with open(out_yaml, "w") as f:
            yaml.dump(config, f, sort_keys=False)
        logger.info(f"Globato recipe exported to {out_yaml}.")

    else:
        recipe = Recipe.from_dict(config)
        iterations = recipe.run(
            outdir=outdir,
            shared_cache=shared_cache,
            refresh=refresh,
            ignore_failures=not fail_fast,
        )
        yield from iterations
