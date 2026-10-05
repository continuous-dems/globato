#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.cli.build
~~~~~~~~~~~~~~~~~~

The globato build command to build a fetchez recipe and execute it.

:copyright: (c) 2025 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import yaml
import click
import logging

from fetchez.utils import (
    FetchezMainCommand,
    parse_hook_string,
)
from fetchez.recipe import Recipe
from fetchez.spatial import parse_region
from fetchez.registry import PresetRegistry, HookRegistry
from fetchez.cli.pipeline import (
    PipelineExecutor,
    organize_pipeline_commands,
    make_pipeline_config,
)

from globato.utils import is_globato_source, is_globato_build_preset


logger = logging.getLogger(__name__)


# --- Build command ---
CONTEXT_SETTINGS = dict(max_content_width=220)


def add_component_options(options):
    def decorator(func):
        for option in reversed(options):
            config = dict(option)
            param_decls = config.pop("param_decls")
            func = click.option(*param_decls, **config)(func)
        return func

    return decorator


class MRGlobatoAdapter:
    preset = "mr-globato"
    description = "Multi Resolution GLOBATO DEM Generation"

    defaults = {
        "stack_strategy": "mixed",
        "stack_weights": "auto",
        "dem_weights": "auto",
        "previous_tier_mode": "points",
        "previous_tier_resampling": "bilinear",
        "bathy_max_z": -0.01,
    }

    cli_options = [
        {
            "param_decls": ["--stack-strategy"],
            "type": click.Choice(
                ["mean", "weighted_mean", "mixed", "supercede"],
                case_sensitive=False,
            ),
            "default": "mixed",
            "show_default": True,
            "help": "MultiStack source accumulation strategy.",
        },
        {
            "param_decls": ["--stack-weights"],
            "default": "auto",
            "help": "MultiStack source comptetion thresholds",
        },
        {
            "param_decls": ["--dem-weights"],
            "default": "auto",
            "help": "Binary CUDEM resolution-admission thresholds",
        },
        {
            "param_decls": ["--previous-tier-mode"],
            "default": "points",
            "type": click.Choice(
                ["points", "raster"],
                case_sensitive=False,
            ),
            "help": "Binary CUDEM previous-tier handoff mode.",
        },
        {
            "param_decls": ["--previous-tier-resampling"],
            "type": click.Choice(
                ["bilinear", "cubic", "near", "cubicspline", "lanczos", "average"],
                case_sensitive=False,
            ),
            "default": "bilinear",
            "help": "Binary CUDEM previous-tier resampling method..",
        },
        {
            "param_decls": ["--bathy-max-z"],
            "type": float,
            "default": -0.01,
            "help": "Binary CUDEM max bathymetry interpolation value.",
        },
    ]

    @classmethod
    def make_command(cls):
        help_text = cls.description

        @click.command(
            name=cls.preset,
            help=help_text,
            hidden=False,
            cls=FetchezMainCommand,
        )
        # @add_options(cls.cli_args)
        @add_component_options(cls.cli_options)
        def dynamic_preset_cmd(**kwargs):
            return {
                "type": "build_preset",
                "preset": cls.preset,
                "args": kwargs,
            }

        return dynamic_preset_cmd

    @classmethod
    def compile(cls, *, increment, target_srs, nodata, options):
        """Translate public DEM controls into hook-preset overrides."""

        options = {
            **cls.defaults,
            **(options or {}),
        }

        from globato.api import (
            _format_inc,
            _auto_resolutions,
            _auto_stack_weights,
            _parse_weights,
            _auto_dem_weights,
        )

        overrides = {}

        def update(name, **args):
            overrides.setdefault(name, {}).update(args)

        if increment is not None:
            for hook_name in (
                "provenance",
                "source_masks",
                "multi_stack",
            ):
                update(hook_name, res=increment)

        if target_srs is not None:
            update("multi_stack", crs=target_srs)

        if nodata is not None:
            update("multi_stack", nodata=nodata)

        # --- Resolution and Weight Tiers ---
        resolutions = _auto_resolutions(increment)

        if str(options.get("stack_weights")).lower() == "auto":
            stack_weight_list = _auto_stack_weights()
        else:
            stack_weight_list = _parse_weights(options.get("stack_weights"))

        if stack_weight_list:
            stack_weight_list = "/".join(map(str, stack_weight_list))

        if str(options.get("dem_weights")).lower() == "auto":
            dem_weight_list = _auto_dem_weights(resolutions)
        else:
            dem_weight_list = _parse_weights(options.get("dem_weights"))

        resolutions = "/".join(
            _format_inc(value, increment) for value in _auto_resolutions(increment)
        )
        update(
            "ms_binary_cudem",
            resolutions=resolutions,
            steps=len(resolutions.split("/")) - 1,
        )

        if options.get("stack_strategy") is not None:
            update(
                "multi_stack",
                strategy=options.get("stack_strategy"),
            )

        if options.get("stack_weights") is not None:
            update(
                "multi_stack",
                weight_threshold=stack_weight_list,
            )

        if options.get("dem_weights") is not None:
            update(
                "ms_binary_cudem",
                weights=dem_weight_list,
            )

        if options.get("previous_tier_mode") is not None:
            update(
                "ms_binary_cudem",
                previous_tier_mode=options.get("previous_tier_mode"),
            )

        if options.get("previous_tier_resampling") is not None:
            update(
                "ms_binary_cudem",
                previous_tier_resampling=options.get("previous_tier_resampling"),
            )

        if options.get("bathy_max_z") is not None:
            update(
                "ms_binary_cudem",
                bathy_max_z=options.get("bathy_max_z"),
            )

        return [{"name": name, "args": args} for name, args in overrides.items()]


GLOBATO_PRESET_ADAPTERS = {
    "mr-globato": MRGlobatoAdapter,
}


class GlobatoPipelineExecutor(PipelineExecutor):
    # PresetRegistry.load_all()

    def module_allowed(self, name, meta):
        return is_globato_source(meta)

    def bundle_allowed(self, name, bundle):
        return is_globato_source(bundle)

    def preset_allowed(self, name, preset):
        return True

    def hook_allowed(self, name, meta):
        return True

    def get_command(self, ctx, name):
        preset_def = PresetRegistry.get_yaml(name)

        if preset_def and is_globato_build_preset(preset_def):
            adapter = GLOBATO_PRESET_ADAPTERS.get(name)
            if adapter is None:
                return None

            return adapter.make_command()

        return super().get_command(ctx, name)

    def format_commands(self, ctx, formatter):
        PresetRegistry.load_all()

        presets = []

        for name, preset in PresetRegistry.get_registry().items():
            if not is_globato_build_preset(preset):
                continue

            cmd = self.get_command(ctx, name)
            if cmd is not None:
                presets.append((name, cmd))

        if presets:
            with formatter.section("DEM Build Methods"):
                formatter.write_dl(
                    [
                        (name, cmd.get_short_help_str(limit=80))
                        for name, cmd in sorted(presets)
                    ]
                )


@click.command("build", cls=GlobatoPipelineExecutor, chain=True)
@click.option("-R", "--region", help="Bounding box: W/E/S/N.")
@click.option(
    "-E",
    "--increment",
    help="Target DEM increment (e.g. 1s, 30m).",
)
@click.option(
    "-O", "--outname", default="globato_dem", show_default=True, help="Output basename."
)
@click.option(
    "-D",
    "--outdir",
    type=click.Path(),
    default=None,
    help="Base output directory.",
)
# @click.option(
#     "-F", "--format", default="GTiff", show_default=True, help="Output format."
# )
@click.option(
    "-P",
    "--t-srs",
    default="EPSG:4326",
    show_default=True,
    help="Target CRS.",
)
@click.option(
    "-N",
    "--nodata",
    type=float,
    default=-9999.0,
    show_default=True,
    help="NoData value.",
)
# @click.option("-T", "--filter", "filters", multiple=True, help="Apply a Grits filter.")
# @click.option("-C", "--clip", help="Clip output to polygon file.")
@click.option(
    "-X",
    "--extend",
    type=str,
    default="0:0",
    show_default=True,
    help="Extend region (cells[:percent]).",
)
# @click.option("-L", "--limits", type=str, default=None, help="Set global DEM limits.")
@click.option(
    "--region-srs",
    default="EPSG:4326",
    help="Set the SRS of the input bounding box (default: EPSG:4326).",
)
@click.option(
    "--modifier",
    multiple=True,
    help="Apply a recipe modifier at runtime (e.g. exclude_module:modules=csb/tnm).",
)
@click.option(
    "--schema",
    multiple=True,
    help="Apply a domain schema validation to the recipe.",
)
@click.option(
    "--threads", default=1, help="Number of parallel download threads (default: 1)."
)
@click.option(
    "--shared-cache",
    type=click.Path(),
    help="Centralized cache directory.",
)
# @click.option("--metadata", help="Global tags to inject.")
@click.option(
    "--export", type=click.Path(), help="Export to YAML instead of executing."
)
@click.option(
    "--refresh",
    is_flag=True,
    help="Force fresh API fetch, bypassing local cache.",
)
@click.option(
    "--fail-fast",
    is_flag=True,
    help="Raise on the first failure instead of continuing through failures.",
)
@click.pass_context
def build_cmd(
    ctx,
    region,
    increment,
    outname,
    outdir,
    t_srs,
    nodata,
    extend,
    region_srs,
    modifier,
    schema,
    threads,
    shared_cache,
    export,
    refresh,
    fail_fast,
):
    """Build and optionally execute a reproducible coastal DEM workflow.

    Components are chained from left to right.

    \b
    A Globato DEM build consists of:
      * A DEM build method, such as `mr-globato`, which defines the global
        stacking, interpolation, provenance, and output workflow.
      * One or more elevation modules or bundles. Only sources registered as
        Globato-compatible elevation streams are available through this command.
      * Optional Fetchez hooks or presets placed after a source. These are
        attached only to the preceding module or bundle and can be used to
        customize how that source is processed before it enters the DEM workflow.

    Use `globato build <DEM-method> --help` to inspect the options exposed by a
    particular DEM build method.

    Use the Fetchez module, bundle, hook, and preset discovery commands to inspect
    the components that can be composed into a workflow.

    Use `--export` to write the generated recipe to YAML instead of executing it.
    The exported recipe can later be reproduced with `fetchez run`.
    """

    ctx.ensure_object(dict)
    src_region = parse_region(region) if region else None
    ctx.obj["region"] = src_region
    ctx.obj["export"] = export


@build_cmd.result_callback()
def process_build(
    commands,
    region,
    increment,
    outname,
    outdir,
    t_srs,
    nodata,
    extend,
    region_srs,
    modifier,
    schema,
    threads,
    shared_cache,
    export,
    refresh,
    fail_fast,
):

    HookRegistry.load_all()
    PresetRegistry.load_all()

    build_presets = [cmd for cmd in commands if cmd.get("type") == "build_preset"]

    pipeline_commands = [cmd for cmd in commands if cmd.get("type") != "build_preset"]

    modules, command_global_hooks = organize_pipeline_commands(pipeline_commands)

    if command_global_hooks:
        names = [
            hook.get("name") or hook.get("preset") for hook in command_global_hooks
        ]
        raise click.UsageError(
            "Globato processing hooks and presets must follow a source "
            f"module or bundle. Global components are provided by the DEM "
            f"build preset: {', '.join(str(name) for name in names)}"
        )

    if len(build_presets) > 1:
        raise click.UsageError("Only one Globato DEM build preset may be selected.")

    if build_presets:
        build_preset = build_presets[0]
    else:
        build_preset = {
            "type": "build_preset",
            "preset": "mr-globato",
            "args": {},
        }

    adapter_cls = GLOBATO_PRESET_ADAPTERS.get(build_preset.get("preset"))
    if adapter_cls is None:
        raise click.UsageError(
            f"No Globato build adapter is registered for "
            f"'{build_preset.get('preset')}'."
        )

    preset_overrides = adapter_cls.compile(
        increment=increment,
        target_srs=t_srs,
        nodata=nodata,
        options=build_preset.get("args", {}),
    )

    global_hooks = [
        {
            "preset": build_preset["preset"],
            "args": preset_overrides,
        },
    ]

    # modules = [cmd for cmd in commands if cmd.pop("type", None) == "module"]
    # presets = [cmd for cmd in commands if cmd.get("type") == "preset"]
    parsed_modifiers = [parse_hook_string(m) for m in modifier]
    parsed_schemas = [s for s in schema]

    config = make_pipeline_config(
        modules,
        name=outname,
        region=region,
        region_srs=region_srs,
        global_hooks=global_hooks,
        modifiers=parsed_modifiers,
        schemas=parsed_schemas,
        threads=threads,
    )

    batch_outname = "%name%_%batch_name%"

    ext_parts = str(extend).split(":")
    ext_cells = int(ext_parts[0]) if len(ext_parts) > 0 and ext_parts[0] else 0
    ext_pct = float(ext_parts[1]) if len(ext_parts) > 1 and ext_parts[1] else 0.0

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

    if export:
        with open(export, "w", encoding="utf-8") as f:
            yaml.dump(config, f, sort_keys=False)
        click.secho(f"Pipeline recipe exported to {export}", fg="green", bold=True)
    else:
        click.secho("Executing dynamic pipeline...", fg="cyan", bold=True, err=True)
        [
            r
            for r in Recipe.from_dict(config).run(
                shared_cache=shared_cache,
                outdir=outdir,
                refresh=refresh,
                ignore_failures=not fail_fast,
            )
        ]
