#!/usr/bin/env python

"""
globato.cli.process
~~~~~~~~~~~~~~~~~~~
Unified pipeline execution for points and rasters.
"""

import logging
import os
import sys

import click
import yaml
from fetchez.recipe import Recipe
from fetchez.registry import HookRegistry
from fetchez.utils import (
    FetchezMainCommand,
    compile_sources,
    parse_hook_string,
)

from globato.utils import globatize_modules, make_recipe_config

logger = logging.getLogger(__name__)


@click.command(name="process", cls=FetchezMainCommand)
@click.option("-R", "--region", help="Spatial crop (W/E/S/N).")
@click.option("-P", "--t-srs", help="Target SRS for reprojection (e.g., EPSG:4326).")
@click.option(
    "--hook",
    "global_hooks_input",
    multiple=True,
    help="Apply a processing hook (e.g., --hook 'raster_slope:max_val=45'). Can be chained.",
)
@click.option(
    "-o", "--output", required=True, help="Output file path (e.g., out.tif or out.xyz)."
)
@click.option(
    "--shared-cache",
    type=click.Path(resolve_path=True),
    help="Centralized directory to cache fetched data.",
)
@click.option(
    "--save-recipe",
    is_flag=True,
    help="Save the pipeline as a YAML recipe instead of running it.",
)
@click.option("--list-hooks", is_flag=True, help="List available stream-hooks")
@click.argument("sources", nargs=-1, required=True)
def process_cmd(
    sources, region, t_srs, global_hooks_input, output, shared_cache, save_recipe
):
    """Process, filter, and stream geospatial data seamlessly.

    Accepts any combination of point clouds, DEMs, or Fetchez modules,
    pipes them through dynamic hooks, and dumps the result.
    """

    HookRegistry.load_all()

    if not sources:
        click.secho("Error: You must provide at least one data source.", fg="red")
        sys.exit(1)

    compiled_modules = globatize_modules(
        compile_sources(sources), shared_cache=shared_cache, crs=t_srs
    )

    global_hooks = []

    if region:
        global_hooks.append({"name": "spatial_crop", "args": {}})

    for h_str in global_hooks_input:
        hook_dict = parse_hook_string(h_str)
        mod_cls = HookRegistry.get_class(hook_dict["name"])
        if not mod_cls:
            click.secho(
                f"Error: Unknown hook '{hook_dict['name']}'", fg="red", err=True
            )
            sys.exit(1)
        global_hooks.append(hook_dict)

    ext = os.path.splitext(output)[1].lower()
    if ext in [".tif", ".tiff", ".geotiff"]:
        # Maybe enforce points2pixels here if we're in a point-stream
        global_hooks.append(
            {"name": "raster_write", "args": {"output_path": output, "stage": "stream"}}
        )
    elif ext in [".xyz", ".csv", ".txt", ".laz", ".las"]:
        # Ensure 'drop_class' is applied for point clouds like pointz used to
        global_hooks.append({"name": "drop_class", "args": {}})
        global_hooks.append({"name": "xyz_write", "args": {"output_path": output}})
    else:
        click.secho(
            f"Warning: Unknown extension '{ext}'. Defaulting to point cloud xyz_write.",
            fg="yellow",
        )
        global_hooks.append({"name": "xyz_write", "args": {"output_path": output}})

    config = make_recipe_config(
        "globato_process", region, compiled_modules, global_hooks, crs=t_srs
    )

    if save_recipe:
        out_yaml = "process_recipe.yaml"
        with open(out_yaml, "w") as f:
            yaml.dump(config, f, sort_keys=False)
        click.secho(f"Recipe saved to {out_yaml}", fg="green", bold=True, err=True)
    else:
        click.secho("🚀 Executing Globato Pipeline...", fg="cyan", bold=True, err=True)
        for _ in Recipe.from_dict(config).run():
            pass
