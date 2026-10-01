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
from fetchez.utils import str2inc, parse_hook_string, compile_sources
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


DEFAULT_STACK_WEIGHTS = [
    4.0,
    3.0,
    2.0,
    1.0,
    0.75,
    0.5,
    0.4,
    0.3,
    0.25,
    0.2,
    0.1,
]


# Binary CUDEM uses a deliberately coarser quality vocabulary than mixed-mode
# MultiStack.  These thresholds describe the resolution at which surviving
# observations become direct constraints, not source-to-source competition.
DEM_WEIGHT_SCALE = [2.0, 1.0, 0.5, 0.25, 0.1]


def _parse_weights(value):
    """Normalize slash-delimited or iterable weight thresholds."""
    if value is None:
        return []
    if isinstance(value, str):
        values = [item for item in value.split("/") if item.strip()]
    else:
        values = list(value)
    return sorted((float(item) for item in values), reverse=True)


def _format_inc(value, template=None):
    """Format a numeric increment as a parseable Fetchez resolution string.

    ``inc2str`` is intentionally not used here: it produces compact filename
    identifiers (for example, ``1/9`` arc-second becomes ``19``), which are not
    safe to feed back through ``str2inc``.

    When the requested increment uses arc-second syntax, preserve that syntax
    throughout the generated Binary CUDEM resolution ladder.
    """
    value = float(value)

    if isinstance(template, str):
        units = template.strip().lower()[-1:]

        if units == "s":
            arcseconds = value * 3600.0
            nearest_integer = round(arcseconds)
            if abs(arcseconds - nearest_integer) < 1e-6:
                arcseconds = float(nearest_integer)
            return f"{arcseconds:.12g}s"
        if units == "t":
            return f"{value * 111320.0:.12g}t"

    return f"{value:.12g}"


def _auto_resolutions(increment, max_tiers=6):
    """Build a factor-of-three resolution ladder through the coarse tier."""
    base_res = str2inc(increment)
    is_arcseconds = isinstance(increment, str) and increment.lower().endswith("s")
    target_max_res = str2inc("15s") if is_arcseconds or base_res < 1 else 500.0

    resolutions = [base_res]
    while resolutions[-1] < target_max_res and len(resolutions) < max_tiers:
        next_res = resolutions[-1] * 3.0
        if next_res >= target_max_res:
            if abs(resolutions[-1] - target_max_res) > 1e-12:
                resolutions.append(target_max_res)
            break
        resolutions.append(next_res)

    return resolutions


def _auto_dem_weights(resolutions):
    """Choose DEM-admission tiers independently from MultiStack mixed tiers."""
    num_steps = max(0, len(resolutions) - 1)
    if num_steps == 0:
        return []

    if num_steps <= len(DEM_WEIGHT_SCALE):
        return DEM_WEIGHT_SCALE[-num_steps:]

    # Extremely fine/custom products can extend the high end without changing
    # the familiar 0.5/0.25/0.1 coarse admission tiers.
    weights = list(DEM_WEIGHT_SCALE)
    next_weight = weights[0] * 2.0
    while len(weights) < num_steps:
        weights.insert(0, next_weight)
        next_weight *= 2.0
    return weights


def _auto_stack_weights():
    """Fine-grained competition tiers for mixed-mode source reduction."""
    return list(DEFAULT_STACK_WEIGHTS)


def _default_blend_distances(resolutions, previous_tier_mode):
    """Return conservative strategy-specific Binary CUDEM blend defaults."""
    if previous_tier_mode == "points":
        # Point handoff already blends through the tier interpolation itself.
        return [0] * len(resolutions)

    # Raster handoff may still benefit from reopening a small transition zone.
    count = len(resolutions)
    return [2 * (2**power) for power in reversed(range(count))]


def build(
    sources,
    region,
    increment,
    format="GTiff",
    outname="globato_dem",
    outdir=None,
    t_srs="EPSG:4326",
    nodata=-9999.0,
    algo="ms_binary_cudem:barrier=osm",
    stack_strategy="mixed",
    stack_weights="auto",
    dem_weights="auto",
    previous_tier_mode="points",
    previous_tier_resampling="bilinear",
    filters=None,
    clip=None,
    extend="0:0",
    limits=None,
    blend=None,
    modifier=None,
    schema=None,
    shared_cache=None,
    metadata=None,
    export=False,
    refresh=False,
    fail_fast=False,
    **kwargs,
) -> Generator:
    """Build and execute a Globato DEM recipe programmatically.

    ``stack_weights`` controls mixed-mode MultiStack source competition, while
    ``dem_weights`` controls the resolution tiers at which observations become
    direct Binary CUDEM constraints.  They intentionally use separate defaults.

    The dependency-light default step-down uses previous-tier points with
    ``raster_fill``.  Explicit options embedded in ``algo`` always take
    precedence over these API defaults.
    """

    HookRegistry.load_all()

    if isinstance(sources, str):
        sources = [sources]

    filters = filters or []
    parsed_modifiers = [parse_hook_string(m) for m in (modifier or [])]
    parsed_schemas = [s for s in (schema or [])]

    # Backward compatibility for callers that still pass weights=... through
    # **kwargs.  Historically one list drove both stacking and interpolation.
    legacy_weights = kwargs.pop("weights", None)
    if legacy_weights is not None:
        logger.warning(
            "build(weights=...) is deprecated; use stack_weights=... and "
            "dem_weights=... separately"
        )
        if str(stack_weights).lower() == "auto":
            stack_weights = legacy_weights
        if str(dem_weights).lower() == "auto":
            dem_weights = legacy_weights

    compiled_modules = globatize_modules(
        compile_sources(sources),
        shared_cache=shared_cache,
        crs=t_srs,
        res=increment,
    )

    # Opt in to collection-wide spatial claiming only when a module requests
    # the generic claim-grid filter; non-TNM builds keep upstream #239 behavior.
    spatial_claim_enabled = any(
        any(
            hook.get("name") in {"claim-grid-filter", "claim_grid_filter"}
            for hook in module.get("hooks", [])
            if isinstance(hook, dict)
        )
        for module in compiled_modules
        if isinstance(module, dict)
    )

    base_outdir = os.path.abspath(outdir) if outdir else os.path.abspath(".")

    # --- Parse Extend ---
    ext_parts = str(extend).split(":")
    ext_cells = int(ext_parts[0]) if len(ext_parts) > 0 and ext_parts[0] else 0
    ext_pct = float(ext_parts[1]) if len(ext_parts) > 1 and ext_parts[1] else 0.0

    # --- Resolution and Weight Tiers ---
    resolutions = _auto_resolutions(increment)

    if str(stack_weights).lower() == "auto":
        stack_weight_list = _auto_stack_weights()
    else:
        stack_weight_list = _parse_weights(stack_weights)

    if str(dem_weights).lower() == "auto":
        dem_weight_list = _auto_dem_weights(resolutions)
    else:
        dem_weight_list = _parse_weights(dem_weights)

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

    if spatial_claim_enabled:
        global_hooks.insert(
            0,
            {
                "name": "spatial-claim",
                "args": {"audit_output": f"{batch_outname}_spatial_claim.geojson"},
            },
        )

    # --- MultiStack ---
    stack_args = {
        "res": increment,
        "crs": t_srs,
        "strategy": stack_strategy,
        "nodata": nodata,
        "output": f"{batch_outname}_stack.tif",
    }
    if stack_weight_list:
        stack_args["weight_threshold"] = "/".join(map(str, stack_weight_list))

    global_hooks.append({"name": "multi_stack", "args": stack_args})
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
    if algo_hook["name"] == "ms_binary_cudem":
        args = algo_hook.setdefault("args", {})

        args.setdefault(
            "resolutions",
            "/".join(_format_inc(value, increment) for value in resolutions),
        )
        if dem_weight_list:
            args.setdefault("weights", dem_weight_list)
        args.setdefault("steps", max(0, len(resolutions) - 1))
        args.setdefault("previous_tier_mode", previous_tier_mode)
        args.setdefault("previous_tier_resampling", previous_tier_resampling)
        args.setdefault("algos", "raster_fill")
        args.setdefault("barrier", "osm")
        args.setdefault(
            "bathy_max_z",
            "ocean:-0.01,river:0,lake:None,wetland:0,estuary:0",
        )

        if "blend_dists" not in args:
            if blend is not None:
                if isinstance(blend, str):
                    blend_value = blend
                else:
                    blend_value = "/".join(str(v) for v in blend)
                args["blend_dists"] = blend_value
            else:
                effective_mode = str(
                    args.get("previous_tier_mode", previous_tier_mode)
                ).lower()
                blend_list = _default_blend_distances(resolutions, effective_mode)
                args["blend_dists"] = "/".join(map(str, blend_list))

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
        config.setdefault("modifiers", []).append(
            {
                "name": "buffer_and_cut",
                "args": {
                    "cells": ext_cells,
                    "pct": ext_pct,
                    "inc": increment,
                    "outname": batch_outname,
                },
            }
        )

    config["schemas"] = [{"name": "validate-recipe"}]
    if parsed_modifiers:
        config.setdefault("modifiers", []).extend(parsed_modifiers)
    if parsed_schemas:
        config["schemas"].extend(parsed_schemas)

    if export:
        os.makedirs(base_outdir, exist_ok=True)
        out_yaml = os.path.join(base_outdir, f"{outname}_recipe.yaml")
        with open(out_yaml, "w") as f:
            yaml.dump(config, f, sort_keys=False)
        logger.info("Globato recipe exported to %s.", out_yaml)

    else:
        recipe = Recipe.from_dict(config)
        iterations = recipe.run(
            outdir=outdir,
            shared_cache=shared_cache,
            refresh=refresh,
            ignore_failures=not (fail_fast or spatial_claim_enabled),
        )
        yield from iterations
