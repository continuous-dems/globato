#!/usr/bin/env python

"""
globato.cli.wafflez
~~~~~~~~~~~~~~~~~~

The command-line interface for the wafflez group.

list, dump, copy, validate, run, build

:copyright: (c) 2025 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import copy
import json
import logging
import os
import sys

import click
import yaml
from fetchez.recipe import Recipe
from fetchez.registry import BundleRegistry, RecipeRegistry
from fetchez.spatial import yield_parsed_regions
from fetchez.utils import (
    FetchezMainCommand,
    FetchezMainGroup,
    compile_sources,
    int_or,
    parse_hook_string,
    str2inc,
)

from globato.utils import globatize_modules, make_recipe_config

logger = logging.getLogger(__name__)

WAFFLEZ_COMMANDS = {
    "Commmands": ["run", "build"],
}


@click.version_option(package_name="globato")
@click.group(
    cls=FetchezMainGroup,
    name="wafflez",
    fetchez_commands=WAFFLEZ_COMMANDS,
)
def wafflez_group():
    """Build and execute Digital Elevation Models Recipes.

    \b
      This is the GLOBATO automated DEM compilation engine. It takes overlapping
      streams of geospatial data, seamlessly stacks them based on quality weights,
      and interpolates the gaps to build continuous Digital Elevation Models (DEMs).

    \b
    Core Commands:
      run   : Execute a pre-configured YAML recipe (supports batch generation).
      build : Dynamically generate a pipeline using Globato's curated data bundles.

    \b
    Examples:
      # Build a DEM using the curated global-bathy-topo bundle
      $ globato wafflez build -R -120/-119/33/34 -E 1s -P EPSG:3857 global-bathy-topo

    \b
      # Run an existing recipe over a batch of geometries from a Shapefile
      $ globato wafflez run -R ./coastal_tiles.shp my_custom_recipe.yaml
    """


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


@wafflez_group.command("run", cls=FetchezMainCommand)
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
def wafflez_run(
    target, region, increment, t_srs, outname, outdir, overwrite, shared_cache
):
    """Execute a YAML recipe. Supports single runs, batch execution, and config overrides."""

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


# --- Build command ---
def _parse_source(src_str):
    """Parses 'module:key=val+hook:k=v' or local paths into a dictionary for the recipe."""

    # Split the module definition from any appended hooks using '+'
    components = src_str.split("+")
    mod_part = components[0]
    hook_parts = components[1:]

    parts = mod_part.split(":", 1)
    mod_name = parts[0]
    args = {}

    if len(parts) > 1:
        for kv in parts[1].split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                try:
                    v = float(v) if "." in v else int(v)
                except ValueError:
                    if v.lower() in ["true", "yes"]:
                        v = True
                    elif v.lower() in ["false", "no"]:
                        v = False
                args[k] = v

    if os.path.exists(mod_name):
        if os.path.isfile(mod_name):
            args["paths"] = mod_name
            mod_name = "file"
        elif os.path.isdir(mod_name):
            args["path"] = mod_name
            mod_name = "local_fs"

    mod_dict = {
        "module": mod_name,
    }
    if args:
        mod_dict["args"] = args

    mod_dict["hooks"] = [{"name": "stream_data"}]

    for h_str in hook_parts:
        h_parts = h_str.split(":", 1)
        h_name = h_parts[0]
        h_args = {}

        if len(h_parts) > 1:
            for kv in h_parts[1].split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    try:
                        v = float(v) if "." in v else int(v)
                    except ValueError:
                        if v.lower() in ["true", "yes"]:
                            v = True
                        elif v.lower() in ["false", "no"]:
                            v = False
                    h_args[k] = v

        hook_dict = {"name": h_name}
        if h_args:
            hook_dict["args"] = h_args

        mod_dict["hooks"].append(hook_dict)

    return mod_dict


def _list_sources(ctx, param, value):
    """List curated data sources and exit."""

    if not value or ctx.resilient_parsing:
        return

    from fetchez.registry import ModuleRegistry

    ModuleRegistry.load_all()
    BundleRegistry.load_all()
    registry = ModuleRegistry.get_registry()
    registry.update(BundleRegistry.get_registry())

    click.secho("\nCurated Globato Data Sources & Bundles:", fg="cyan", bold=True)
    click.echo("=" * 60)

    count = 0
    for name, meta in sorted(registry.items()):
        tags = meta.get("tags", [])
        # category = meta.get("category", "")
        # mod_path = meta.get("mod", "")
        # is_globato = (
        #     mod_path.startswith("globato.modules")
        #     or category.lower() == "globato"
        #     or "globato" in tags
        #     or "bundle" in tags
        # )
        is_globato = "glob-stream" in tags

        if is_globato and name not in meta.get("aliases", []):
            desc = meta.get("desc", "No description provided.").strip().split("\n")[0]
            click.echo(f"  {click.style(name, bold=True, fg='yellow'):<25} : {desc}")
            count += 1

    click.echo("-" * 60)
    click.secho("\nLocal File Support:", fg="cyan", bold=True)
    click.echo("=" * 60)
    click.echo("  You can also pass local files and directories directly!")
    click.echo("  Files will be wrapped in the 'file' module.")
    click.echo("  Directories will be crawled using the 'local_fs' module.")
    click.echo(
        "  Example: globato cudem build -R ... ./my_data.tif ./my_folder:ext=.xyz"
    )

    click.echo(
        f"\nTry 'globato cudem build --info-source <name>' for details. Total: {count}\n"
    )
    ctx.exit()


def _info_source(ctx, param, value):
    """inspect a specific data source and exit."""

    if not value or ctx.resilient_parsing:
        return

    from fetchez.registry import ModuleRegistry

    ModuleRegistry.load_all()
    registry = ModuleRegistry.get_registry()

    source_name = value
    if source_name not in registry:
        click.secho(f"Error: '{source_name}' is not a recognized source.", fg="red")
        ctx.exit(1)

    meta = registry[source_name]

    if not (
        meta.get("mod", "").startswith("globato.modules")
        or meta.get("category") == "Globato"
    ):
        click.secho(
            f" Note: '{source_name}' is a core Fetchez module, not a curated Globato DEM source.",
            fg="yellow",
        )

    click.secho(f"\nSOURCE: {source_name.upper()}", fg="cyan", bold=True)
    click.echo("=" * 60)
    click.echo(f"  Description : {meta.get('desc', 'N/A')}")
    click.echo(f"  Tags        : {', '.join(meta.get('tags', []))}")

    mod_cls = ModuleRegistry.get_class(source_name)
    if mod_cls:
        import inspect

        sig = inspect.signature(mod_cls.__init__)
        params = []
        for p_name, param in sig.parameters.items():
            if p_name not in [
                "self",
                "kwargs",
                "src_region",
                "callback",
                "outdir",
                "name",
            ]:
                default = (
                    param.default
                    if param.default is not inspect.Parameter.empty
                    else "None"
                )
                params.append(f"{p_name}={default}")
        if params:
            click.echo(f"  Arguments   : {', '.join(params)}")
    click.echo("\n")
    ctx.exit()


CONTEXT_SETTINGS = dict(max_content_width=120)


# @click.command(context_settings=CONTEXT_SETTINGS)
@wafflez_group.command(
    "build", cls=FetchezMainCommand, context_settings=CONTEXT_SETTINGS
)
@click.option("-R", "--region", required=True, help="Bounding box: W/E/S/N")
@click.option(
    "-E", "--increment", required=True, help="Gridding Increment (e.g., 1s, 30m)"
)
@click.option(
    "-O",
    "--outname",
    default="globato_dem",
    help="Output Basename (default: globato_dem)",
)
@click.option(
    "-D",
    "--outdir",
    type=click.Path(resolve_path=True),
    default=None,
    help="Base output directory for the DEM(s).",
)
@click.option(
    "-F",
    "--format",
    default="GTiff",
    help="Output Format (GTiff, NetCDF, etc.). Default: GTiff.",
)
@click.option(
    "-P", "--t-srs", default="EPSG:4326", help="Target Projection (default: EPSG:4326)"
)
@click.option(
    "-N", "--nodata", type=float, default=-9999.0, help="NoData Value. Default: -9999."
)
@click.option(
    "-M",
    "--algo",
    default="ms_binary_cudem:barrier=coastline",
    help="Interpolation algorithm and options (e.g., interp_gmt:tension=0.35)",
)
@click.option(
    "-A",
    "--stack-mode",
    type=click.Choice(["mean", "min", "max", "mixed", "supercede"]),
    default="mixed",
    help="Stacking mode",
)
@click.option(
    "-T",
    "--filter",
    "filters",
    multiple=True,
    help="Apply Grits Filter (e.g. 'blur:radius=3'). May be set multiple times.",
)
@click.option("-C", "--clip", help="Clip output to polygon file. e.g. 'clip_ply.shp'")
# @click.option(
#     "-B",
#     "--buffer",
#     type=int,
#     default=0,
#     help="Buffer the processing region by N cells to prevent edge artifacts. The final DEM will be cropped back to the strict -R region.",
# )
@click.option(
    "-B",
    "--blend",
    type=str,
    default=None,
    help="Blend between weighted data in the generated MultiStack (e.g. 10/20/60).",
)
@click.option(
    "-X",
    "--extend",
    type=str,
    default="0:0",
    help="Extend region (cells[:percent]). e.g. '6:10'. "
    "'cells' extends the final output region, 'percent' extends the internal processing region.",
)
@click.option(
    "-L",
    "--limits",
    type=str,
    default=None,
    help="Set global DEM limits (e.g., min_z=-11000,max_z=8850,expand=3).",
)
@click.option(
    "-W",
    "--weights",
    default="1.0/0.5",
    help="Weight thresholds for stacking, blending, and interpolation tiers (e.g. 1.0/0.5/0.1).",
)
@click.option(
    "--shared-cache",
    type=click.Path(resolve_path=True),
    help="Centralized directory to cache fetched data. Injects 'outdir' into all modules.",
)
@click.option(
    "--metadata",
    help="Global tags to inject into the final DEM (e.g., 'Project=CRM,Author=NOAA').",
)
@click.option(
    "--export",
    is_flag=True,
    help="Save the generated YAML recipe to disk without running it.",
)
@click.option(
    "--list-sources",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_list_sources,
    help="List available data sources and exit.",
)
@click.option(
    "--info-source",
    metavar="NAME",
    is_eager=True,
    expose_value=False,
    callback=_info_source,
    help="Show details for a specific data source and exit.",
)
@click.argument("sources", nargs=-1)
def wafflez_build(
    region,
    increment,
    outname,
    outdir,
    format,
    t_srs,
    nodata,
    algo,
    stack_mode,
    filters,
    clip,
    extend,
    limits,
    weights,
    blend,
    shared_cache,
    metadata,
    export,
    sources,
):
    """Build and run a Digital Elevation Model."""

    from fetchez.registry import HookRegistry

    HookRegistry.load_all()

    if not sources:
        click.secho(
            "Error: You must provide at least one data source or a modules.yaml file.",
            fg="red",
        )
        sys.exit(1)

    compiled_modules = globatize_modules(
        compile_sources(sources),
        shared_cache=shared_cache,
        crs=t_srs,
        res=increment,
    )

    if outdir is None:
        base_outdir = os.path.abspath(".")
    else:
        base_outdir = os.path.abspath(outdir)
    os.makedirs(base_outdir, exist_ok=True)
    original_cwd = os.getcwd()

    try:
        from fetchez.utils import str2inc

        inc_val = str2inc(increment)

        for t_reg, feat_name in yield_parsed_regions(region):
            # Parse the Extend argument
            ext_parts = str(extend).split(":")
            ext_cells = int(ext_parts[0]) if len(ext_parts) > 0 else 0
            ext_pct = float(ext_parts[1]) if len(ext_parts) > 1 else 0.0

            # Calculate the Delivery Region (Original Region + Cell Overlap)
            delivery_reg = t_reg.copy()
            if ext_cells > 0:
                # Buffer adds absolute values, so cells * increment
                delivery_reg.buffer(
                    pct=0, x_bv=(inc_val * ext_cells), y_bv=(inc_val * ext_cells)
                )

            delivery_r_str = f"{delivery_reg.xmin}/{delivery_reg.xmax}/{delivery_reg.ymin}/{delivery_reg.ymax}"

            # Calculate the Processing Region (Delivery Region + Percentage)
            proc_reg = delivery_reg.copy()
            if ext_pct > 0:
                proc_reg.buffer(pct=ext_pct, x_inc=inc_val, y_inc=inc_val)

            proc_r_str = (
                f"{proc_reg.xmin}/{proc_reg.xmax}/{proc_reg.ymin}/{proc_reg.ymax}"
            )

            tile_outname = f"{outname}_{feat_name}" if feat_name else outname

            if feat_name:
                click.secho(
                    f"\n--- Building Batch Tile: {feat_name} ---", fg="cyan", bold=True
                )
                click.secho(
                    f"  Delivery Region: {delivery_r_str} (+{ext_cells} cells)",
                    fg="blue",
                )
                if ext_pct > 0:
                    click.secho(
                        f"  Processing Region: {proc_r_str} (+{ext_pct}%)", fg="yellow"
                    )

            # --- Base Pipeline Standard Hooks ---
            global_hooks = [
                {"name": "spatial-crop"},
                {"name": "audit"},
                {"name": "enrich"},
                {"name": "transfer_log"},
                {"name": "drop_class"},
                {
                    "name": "provenance",
                    "args": {
                        "res": increment,
                        "output": f"{tile_outname}_provenance.tif",
                    },
                },
                {
                    "name": "source_masks",
                    "args": {"res": increment, "output": f"{tile_outname}_sources.vrt"},
                },
            ]

            # --- Parse Limits ---
            limit_args = {}
            if limits:
                for kv in limits.split(","):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        limit_args[k.strip()] = float(v.strip())

            # --- Multi Stack and Raster Stream ---
            global_hooks.append(
                {
                    "name": "multi_stack",
                    "args": {
                        "res": increment,
                        "crs": t_srs,
                        "mode": stack_mode,
                        "nodata": nodata,
                        "weight_threshold": weights,
                        "output": f"{tile_outname}_stack.tif",
                    },
                }
            )
            # todo
            # if "expand" in limit_args:
            #     ms_args["expand_limit"] = int(limit_args["expand"])

            global_hooks.append(
                {"name": "focus_sink", "args": {"target": "multi_stack"}}
            )
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

            if "min_z" in limit_args or "max_z" in limit_args:
                global_hooks.append(
                    {
                        "name": "raster_limits",
                        "args": {
                            "min_z": limit_args.get("min_z"),
                            "max_z": limit_args.get("max_z"),
                        },
                    }
                )

            # --- Dynamic Blending Tiers ---
            # Parse the weights to generate the correct number of blend/cudem steps
            weight_list = sorted(
                [float(w) for w in str(weights).split("/")], reverse=True
            )
            if weight_list[-1] > 0:
                weight_list.append(0)

            if blend:
                blend_list = [int_or(b, 10) for b in str(blend).split("/")]
                while len(blend_list) <= len(weight_list):
                    blend_list.append(blend_list[-1])
                for i, w in enumerate(weight_list):
                    if w > 0:
                        global_hooks.append(
                            {
                                "name": "ms_blend",
                                "args": {
                                    "weight_threshold": w,
                                    "blend_dist": blend_list[i],
                                    "random_scale": 0.5,
                                    "barrier": "osm",
                                },
                            }
                        )
                # Writes the blended data to disk without breaking the stream
                global_hooks.append(
                    {
                        "name": "raster_write",
                        "args": {
                            "suffix": "_final_blend",
                            "artifact_id": "blended_checkpoint",
                        },
                    }
                )
                global_hooks.append(
                    {
                        "name": "focus_sink",
                        "args": {"target": "blended_checkpoint"},
                    }
                )

            # --- Add requested Filters (-T) ---
            for f in filters:
                global_hooks.append(parse_hook_string(f))

            # --- Interpolation Algorithm (-M) ---
            algo_hook = parse_hook_string(algo)
            if (
                algo_hook["name"] == "ms_cudem"
                or algo_hook["name"] == "ms_binary_cudem"
            ):
                from fetchez.utils import str2inc

                base_res = str2inc(increment)
                args = algo_hook.setdefault("args", {})

                if "resolutions" not in args:
                    step_resolutions = [
                        base_res * (3**i) for i in range(len(weight_list))
                    ]
                    args["resolutions"] = step_resolutions

                args["weights"] = weight_list
                args["steps"] = len(weight_list) - 1

                if "barrier" not in args:
                    args["barrier"] = "osm"

            algo_hook.setdefault("args", {})["output"] = f"{tile_outname}.tif"
            global_hooks.append(algo_hook)

            if ext_pct > 0:
                global_hooks.append(
                    {
                        "name": "raster_cut",
                        "args": {
                            "region": delivery_r_str,  # Cut back to the expanded cell delivery bounds
                            # "output": f"{tile_outname}_final.tif",
                        },
                    }
                )
                global_hooks.append(
                    {
                        "name": "raster_crop",
                        "args": {"output": f"{tile_outname}_final.tif"},
                    }
                )

            # --- Add Clipping (-C) ---
            if clip:
                clip_hook = parse_hook_string(clip, default_name="raster_clip")
                if clip_hook["name"] != "raster_clip":
                    clip_hook["args"]["barrier"] = clip_hook.pop("name")
                    clip_hook["name"] = "raster_clip"
                global_hooks.append(clip_hook)

            if metadata:
                global_hooks.append(
                    {
                        "name": "raster_metadata",
                        "args": {"tags": metadata, "bands": "Elevation (meters)"},
                    }
                )

            # Re-format the tif for COG compatibility
            global_hooks.append(
                {
                    "name": "format_cog",
                    "args": {"overviews": "2/4/8/16/32", "resampling": "average"},
                }
            )

            # Generate a hillshade
            global_hooks.append(
                {
                    "name": "viz_geoshade",
                    "args": {
                        "output": f"{tile_outname}_hs.tif",
                        "cmap": "coastal_relief",
                    },
                }
            )

            if outdir:
                global_hooks.append(
                    {
                        "name": "copy_artifact",
                        "args": {
                            "target_dir": base_outdir,
                            "match": [
                                f"{tile_outname}_final.tif",
                                f"{tile_outname}_hs.tif",
                            ],
                        },
                    }
                )

            # Add some hook descriptions for the recipe yaml
            for hook in global_hooks:
                hook_name = hook.get("name")
                hook_names = hook.get("meta_aliases", [])
                hook_names.append(hook_name)  #  = [hook_name].extend(hook_aliases)
                hook_cls = HookRegistry.get_class(hook_name)

                if hook_cls:
                    desc = getattr(
                        hook_cls, "meta_desc", f"Executes the {hook_name} process."
                    )

                    if (
                        "multi_stack" in hook_names
                    ):  # in ["multi_stack", "multi-stack"]:
                        desc += " Args define the target resolution and grid math (e.g., mean, idw)."
                    elif "ms_cudem" in hook_names:  # == "ms_cudem":
                        desc += " Args define interpolation resolutions and blending distances."

                    hook["description"] = desc

            # --- Build the recipe ---
            config = make_recipe_config(
                tile_outname,
                proc_r_str,
                compiled_modules,
                global_hooks,
                crs=t_srs,
            )

            tile_dir = os.path.join(base_outdir, tile_outname)
            os.makedirs(tile_dir, exist_ok=True)
            os.chdir(tile_dir)

            yaml_str = yaml.dump(config, sort_keys=False)
            out_yaml = f"{tile_outname}_recipe.yaml"
            with open(out_yaml, "w") as f:
                f.write(yaml_str)
            click.secho(f"Globato recipe saved to {out_yaml}.", fg="green", bold=True)

            if not export:
                click.secho(
                    f"Executing dynamic recipe: {tile_outname}", fg="cyan", bold=True
                )
                recipe = Recipe.from_file(config)
                valid, errors = recipe.validate()
                if valid:
                    recipe.run()
                    click.secho(
                        f"✨ Successfully completed Globato build for {tile_outname}!",
                        fg="green",
                        bold=True,
                    )
                else:
                    click.secho(f"Recipe is invalid: {errors}", fg="red", bold=True)

    except ValueError as e:
        click.secho(str(e), fg="red")
    finally:
        os.chdir(original_cwd)
