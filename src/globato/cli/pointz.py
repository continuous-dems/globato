#!/usr/bin/env python

"""
globato.cli.pointz
~~~~~~~~~~~~~~~~~~
Point cloud filtering and manipulation.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import sys

import click
import yaml
from fetchez.recipe import Recipe
from fetchez.registry import (
    HookRegistry,
    ProfileRegistry,
    # ModuleRegistry,
    ReaderRegistry,
)
from fetchez.spatial import Region
from fetchez.utils import (
    FetchezMainCommand,
    FetchezMainGroup,
    compile_sources,
    parse_hook_string,
    str2inc,
)

from globato.utils import globatize_modules, make_recipe_config

logger = logging.getLogger(__name__)

POINTZ_COMMANDS = ["info", "region", "dump", "list-filters", "pipeline"]


@click.version_option(package_name="globato")
@click.group(
    cls=FetchezMainGroup,
    name="pointz",
    fetchez_commands=POINTZ_COMMANDS,
)
def pointz_group():
    """Filter, transform, and stream point cloud data."""


def _yield_stdin_chunks(chunk_size=100000):
    """A lightweight generator to stream XYZ data from standard input."""

    import pandas as pd

    try:
        for chunk in pd.read_csv(
            sys.stdin,
            delim_whitespace=True,
            header=None,
            names=["x", "y", "z", "w", "u"],
            usecols=[0, 1, 2],
            chunksize=chunk_size,
            engine="c",
        ):
            yield chunk.to_records(index=False)
    except Exception as e:
        logger.error(f"Error reading from stdin: {e}")


@pointz_group.command(name="dump", hidden=False, cls=FetchezMainCommand)
@click.option("-R", "--region", help="Spatial crop (W/E/S/N).")
@click.option(
    "-E", "--inc", help="Grid increment (e.g., 1s, 0.0001). Triggers stacking!"
)
@click.option("-P", "--t-srs", help="Target SRS for reprojection (e.g., EPSG:4326).")
@click.option(
    "-F",
    "--filter",
    "global_filters",
    multiple=True,
    help="Apply a global filter (e.g., rq:threshold=10) to all sources.",
)
@click.option("-o", "--output", help="Output file (Default: stdout).")
@click.option(
    "--shared-cache",
    type=click.Path(resolve_path=True),
    help="Centralized directory to cache fetched data.",
)
@click.option("--save-recipe", is_flag=True, help="Save the pipeline as YAML recipe.")
@click.argument("sources", nargs=-1, required=True)
def dump(
    sources, region, inc, t_srs, global_filters, output, shared_cache, save_recipe
):
    """Process and dump elevation data."""

    HookRegistry.load_all()

    if not sources:
        click.secho(
            "Error: You must provide at least one data source or a modules.yaml file.",
            fg="red",
        )
        sys.exit(1)

    compiled_modules = globatize_modules(
        compile_sources(sources), shared_cache=shared_cache, crs=t_srs
    )

    global_hooks = []

    if region:
        global_hooks.append({"name": "spatial_crop", "args": {}})

    for f_str in global_filters:
        hook_dict = parse_hook_string(f_str)
        mod_cls = HookRegistry.get_class(hook_dict["name"])
        if not mod_cls:
            click.secho(
                f"Error: Unknown global filter '{hook_dict['name']}'",
                fg="red",
                err=True,
            )
            sys.exit(1)
        global_hooks.append(hook_dict)

    if inc and region:
        global_hooks.append(
            {
                "name": "points2pixels",
                "args": {"x_inc": str2inc(inc), "y_inc": str2inc(inc)},
            }
        )
        global_hooks.append({"name": "pixels2points", "args": {}})

    global_hooks.append({"name": "drop_class", "args": {}})
    global_hooks.append(
        {
            "name": "xyz_write",
            "args": {"output_path": output, "columns": ["x", "y", "z", "w"]},
        }
    )

    config = make_recipe_config(
        "pointz_dump", region, compiled_modules, global_hooks, crs=t_srs
    )

    if save_recipe:
        out_yaml = "pointz_recipe.yaml"
        with open(out_yaml, "w") as f:
            yaml.dump(config, f, sort_keys=False)
        click.secho(f"Recipe saved to {out_yaml}", fg="green", bold=True, err=True)
    else:
        click.secho("Executing PointZ Pipeline...", fg="cyan", err=True)
        [x for x in Recipe.from_file(config).run()]


@pointz_group.command("list-filters", cls=FetchezMainCommand)
def pointz_list_filters():
    """List all available point cloud filters."""

    HookRegistry.load_all()
    registry = HookRegistry.get_registry()

    click.secho("\n🌪️  Available `point-stream` Filters:\n", fg="cyan", bold=True)
    click.echo("=" * 50)
    for name, meta in sorted(registry.items()):
        if meta.get("category") in ["stream-filter"]:  # , "point-stream"]:
            if name in meta.get("aliases", ""):
                continue
            desc = meta.get("desc", "No description provided.")
            click.echo(f"  {click.style(name, bold=True, fg='yellow'):<15} : {desc}")
    click.echo("=" * 50 + "\n")


@pointz_group.command("info", cls=FetchezMainCommand)
@click.option("--format", "inf_format", is_flag=True, help="Format metadata")
@click.argument("source")
def pointz_info(source, inf_format):
    """Scan a point cloud and return its spatial statistics."""
    from globato.hooks.metadata.globato_inf import generate_stream_inf

    ReaderRegistry.load_all()
    ProfileRegistry.load_all()

    term = source.split(".")[-1]
    reader = ReaderRegistry.get_reader(source, term)

    if not reader:
        click.secho(
            f"Error: Could not determine format for {source}", fg="red", err=True
        )
        sys.exit(1)

    inf = generate_stream_inf(reader.yield_chunks())

    while True:
        try:
            next(inf)
        except StopIteration as e:
            meta = e.value
            break

    if inf_format:
        region = meta.get("minmax", None)
        click.secho(f"\n--- Point Cloud Info: {source} ---", fg="cyan", bold=True)
        click.echo(f"Format Reader: {reader.name}")
        click.echo(f"Total Points : {meta.get('numpts', 0):,}")

        if region is not None:
            click.echo(f"Bounds (X)   : {region[0]:.6f} to {region[1]:.6f}")
            click.echo(f"Bounds (Y)   : {region[2]:.6f} to {region[3]:.6f}")
            click.echo(f"Elevation (Z): {region[4]:.3f} to {region[5]:.3f}")
    else:
        click.echo(meta)


@pointz_group.command("region", cls=FetchezMainCommand)
@click.argument("source")
def pointz_region(source):
    """Scan a point cloud and return its region."""

    from globato.hooks.metadata.globato_inf import generate_stream_inf

    ReaderRegistry.load_all()
    ProfileRegistry.load_all()

    term = source.split(".")[-1]
    reader = ReaderRegistry.get_reader(source, term)

    if not reader:
        click.secho(
            f"Error: Could not determine format for {source}", fg="red", err=True
        )
        sys.exit(1)

    inf = generate_stream_inf(reader.yield_chunks())

    while True:
        try:
            next(inf)
        except StopIteration as e:
            meta = e.value
            break

    region = Region.from_list([*meta.get("minmax", None)])
    click.echo(region.format("gmt"))
