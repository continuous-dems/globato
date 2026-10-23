#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.cli.run
~~~~~~~~~~~~~~~~~~

The globato run command to mutate and run a globato/fetchez
recipe.

:copyright: (c) 2025 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import sys
import click
import copy
import json
import yaml
import logging

from fetchez.recipe import Recipe
from fetchez.registry import RecipeRegistry
from fetchez.utils import (
    str2inc,
    FetchezMainCommand,
)
from globato.utils import globatize_modules
from fetchez.spatial import yield_parsed_regions

logger = logging.getLogger(__name__)


def _load_yaml(target):
    base_config = None
    if os.path.exists(target) and not os.path.isdir(target):
        with open(target, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)
    else:
        recipe_meta = RecipeRegistry.get_recipe(target)
        if recipe_meta:
            base_config = recipe_meta["config"]
            click.secho(f"Loaded recipe: {target}", fg="cyan")

    return base_config


def _absolutize_local_sources(config, base_dir):
    """Converts relative local paths in a recipe to absolute paths."""

    for mod in config.get("modules", []):
        mod_name = mod.get("module")
        if mod_name in ["file", "local_fs"]:
            args = mod.setdefault("args", {})

            # Handle local_fs 'path'
            if "path" in args:
                args["path"] = os.path.normpath(
                    os.path.join(base_dir, str(args["path"]))
                )

            # Handle file 'paths' (which could be comma-separated)
            if "paths" in args:
                paths = str(args["paths"]).split(",")
                abs_paths = [
                    os.path.normpath(os.path.join(base_dir, p.strip())) for p in paths
                ]
                args["paths"] = ",".join(abs_paths)

    return config


@click.command("run", cls=FetchezMainCommand)
@click.argument("target")
@click.option(
    "-R",
    "--region",
    help="Override region. Can be a bounding box, loc string, or geojson file to trigger batch mode.",
)
@click.option(
    "-E", "--increment", help="Override gridding increment/resolution (e.g., 3s, 10)."
)
@click.option("-P", "--t-srs", help="Override target SRS (e.g., EPSG:3857).")
@click.option("-O", "--outname", help="Override project name / output basename.")
@click.option(
    "--outdir",
    type=click.Path(resolve_path=True),
    default=None,
    help="Base output directory for the tiles.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Force rebuild of already completed tiles in a batch run.",
)
@click.option(
    "--shared-cache",
    type=click.Path(resolve_path=True),
    help="Centralized directory to cache fetched data across all tiles.",
)
def run_cmd(target, region, increment, t_srs, outname, outdir, overwrite, shared_cache):
    """Execute a Globato recipe."""

    RecipeRegistry.load_all()

    base_config = _load_yaml(target)
    if not base_config:
        click.secho(
            f"Error: Recipe '{target}' not found locally or in the registry.", fg="red"
        )
        sys.exit(1)

    if outname:
        base_config.setdefault("project", {})["name"] = outname

    recipe_modules = globatize_modules(
        base_config.get("modules"), crs=t_srs, res=increment
    )
    base_config["modules"] = recipe_modules

    if increment or t_srs or outname:
        increment = str2inc(increment)
        for module in base_config.get("modules", []):
            for hook in module.get("hooks", []):
                if hook.get("name") == "stream_reproject":
                    if t_srs:
                        hook.setdefault("args", {})["dst_srs"] = t_srs

        for hook in base_config.get("global_hooks", []):
            hook_name = hook.get("name")
            if hook_name == "provenance":
                if increment:
                    hook.setdefault("args", {})["res"] = increment
                if outname:
                    hook.setdefault("args", {})["output"] = f"{outname}_provenance.tif"
            if hook_name == "multi_stack":
                if increment:
                    hook.setdefault("args", {})["res"] = increment
                if t_srs:
                    hook.setdefault("args", {})["crs"] = t_srs
                if outname:
                    hook.setdefault("args", {})["output"] = f"{outname}_stack.tif"

            if (
                hook_name == "ms_cudem"
                or hook_name == "interp_gmt"
                or hook_name == "raster_fill"
            ):
                if outname:
                    hook.setdefault("args", {})["output"] = f"{outname}_dem.tif"

                if increment and hook_name == "ms_cudem":
                    args = hook.setdefault("args", {})
                    # Fallback to the new increment if none exist
                    old_res = args.get("resolutions", [increment])

                    if isinstance(old_res, str):
                        old_res_list = [str2inc(x) for x in old_res.split("/")]
                    else:
                        old_res_list = [str2inc(str(x)) for x in old_res]

                    num_steps = len(old_res_list)
                    old_base_res = old_res_list[0] if num_steps > 0 else increment
                    new_base_res = increment

                    args["resolutions"] = [
                        new_base_res * (3**i) for i in range(num_steps)
                    ]

                    if "blend_dist" in args:
                        old_blend = args["blend_dist"]
                        if isinstance(old_blend, str):
                            old_blend_list = [int(x) for x in old_blend.split("/")]
                        elif isinstance(old_blend, list):
                            old_blend_list = [int(x) for x in old_blend]
                        else:
                            old_blend_list = [int(old_blend)]

                        ratio = old_base_res / new_base_res
                        args["blend_dist"] = [
                            int(round(b * ratio)) for b in old_blend_list
                        ]
            # if (
            #     hook_name == "ms_cudem"
            #     or hook_name == "interp_gmt"
            #     or hook_name == "raster_fill"
            # ):
            #     if increment and hook_name == "ms_cudem":
            #         hook.setdefault("args", {})["resolutions"] = increment
            #     if outname:
            #         hook.setdefault("args", {})["output"] = f"{outname}_dem.tif"

            if hook_name == "viz_geoshade":
                if outname:
                    hook.setdefault("args", {})["output"] = f"{outname}_hillshade.tif"

    if outdir is None:
        base_outdir = os.path.abspath(".")
    else:
        base_outdir = os.path.abspath(outdir)
    os.makedirs(base_outdir, exist_ok=True)
    original_cwd = os.getcwd()
    base_config = _absolutize_local_sources(base_config, original_cwd)

    state_file = os.path.join(original_cwd, ".globato_batch_state.json")
    completed_tiles = []

    if os.path.exists(state_file) and not overwrite:
        try:
            with open(state_file, "r") as f:
                completed_tiles = json.load(f)
        except Exception:
            pass  # If the state file is corrupted, we just ignore it

    for t_reg, feat_name in yield_parsed_regions(region):
        try:
            _is_batch = False
            config = copy.deepcopy(base_config)
            if t_reg:
                config["region"] = (
                    f"{t_reg.xmin}/{t_reg.xmax}/{t_reg.ymin}/{t_reg.ymax}"
                )
                if t_srs:
                    config["region_srs"] = t_srs

            if feat_name:
                _is_batch = True
                orig_name = config.get("project", {}).get("name", "globato_dem")
                batch_name = f"{orig_name}_{feat_name}"
                config.setdefault("project", {})["name"] = batch_name
                click.secho(
                    f"\n--- Running Batch Tile: {batch_name} ({config['region']}) ---",
                    fg="cyan",
                    bold=True,
                )
            elif outname:
                batch_name = outname
                click.secho(
                    f"\n--- Running Recipe with Override: {batch_name} ---",
                    fg="cyan",
                    bold=True,
                )
            else:
                batch_name = config.get("project", {}).get("name", "globato_dem")

            if batch_name in completed_tiles and not overwrite:
                click.secho(
                    f"  Skipping completed tile: {batch_name} (use --overwrite to force)",
                    fg="yellow",
                    bold=True,
                )
                continue

            for hook in config.get("global_hooks", []):
                hook_name = hook.get("name")
                if hook_name == "provenance":
                    hook.setdefault("args", {})["output"] = (
                        f"{batch_name}_provenance.tif"
                    )
                if hook_name == "multi_stack":
                    hook.setdefault("args", {})["output"] = f"{batch_name}_stack.tif"
                if (
                    hook_name == "ms_cudem"
                    or hook_name == "interp_gmt"
                    or hook_name == "raster_fill"
                ):
                    hook.setdefault("args", {})["output"] = f"{batch_name}_dem.tif"
                if hook_name == "viz_geoshade":
                    hook.setdefault("args", {})["output"] = (
                        f"{batch_name}_hillshade.tif"
                    )

            if _is_batch or not outdir:
                tile_dir = os.path.join(base_outdir, batch_name)
                os.makedirs(tile_dir, exist_ok=True)
                os.chdir(tile_dir)

            if shared_cache:
                abs_cache = os.path.abspath(shared_cache)
                os.makedirs(abs_cache, exist_ok=True)

                for mod in config.get("modules", []):
                    if mod.get("module") not in ["file", "local_fs", "stdin"]:
                        mod.setdefault("args", {})["outdir"] = abs_cache
                    for hook in mod.get("hooks", []):
                        if hook.get("name") == "stream_reproject":
                            if not hook.get("args", None):
                                hook.setdefault("args", {})
                            hook["args"].update({"cache_dir": abs_cache})

            batch_config_fn = f"{batch_name}_recipe.yaml"
            with open(batch_config_fn, "w") as f:
                yaml.dump(config, f, sort_keys=False, default_flow_style=False)

            try:
                Recipe.from_file(config).run()

                completed_tiles.append(batch_name)
                with open(state_file, "w") as f:
                    json.dump(completed_tiles, f, indent=2)

                click.secho(
                    f"✨ Successfully completed globato build for {batch_name}!",
                    fg="green",
                    bold=True,
                )

            except Exception as e:
                click.secho(f"\n Tile {batch_name} failed: {e}", fg="red", bold=True)
                click.secho(
                    "Batch processing halted. Re-run command to resume from this tile.",
                    fg="yellow",
                )
                sys.exit(1)

        except ValueError as e:
            click.secho(str(e), fg="red")
        finally:
            os.chdir(original_cwd)
