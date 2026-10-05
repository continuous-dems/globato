#!/usr/bin/env python

"""
globato.cli.fetch
~~~~~~~~~~~~~~~~~
Direct data discovery and downloading for curated Globato DEM sources.
"""

import logging
import os
import sys

import click
from fetchez.core import run_fetchez
from fetchez.registry import ModuleRegistry
from fetchez.utils import yield_parsed_regions

logger = logging.getLogger(__name__)


@click.group(name="fetch")
def fetch_group():
    """Discover and download curated elevation/bathymetry data."""


@fetch_group.command("list")
@click.option("--search", "-s", help="Filter modules by name or keyword.")
def fetch_list(search):
    """List all curated Globato DEM data modules."""

    ModuleRegistry.load_all()
    registry = ModuleRegistry.get_registry()

    click.secho("\nAvailable Curated Globato Sources:", fg="cyan", bold=True)
    click.echo("=" * 60)

    count = 0
    for name, cls in sorted(registry.items()):
        meta = ModuleRegistry.get_info(name)

        if meta.get("category") != "Globato" and "globato" not in meta.get("tags"):
            continue

        if search and search.lower() not in name.lower():
            continue

        desc = meta.get("desc", "No description available")

        click.secho(f"  {name:<15}", fg="green", bold=True, nl=False)
        click.echo(f" - {desc}")
        count += 1

    click.echo("=" * 60)
    click.echo(f"Total curated sources found: {count}")
    click.echo(
        "(Note: For raw, uncurated data access, use the 'fetchez' CLI directly)\n"
    )


@fetch_group.command("run")
@click.argument("module_name")
@click.option(
    "-R",
    "--region",
    required=True,
    help='Spatial bounding box (W/E/S/N or loc:"Name").',
)
@click.option(
    "-O",
    "--outdir",
    default=".",
    help="Output directory to save data (default: current directory).",
)
@click.argument("extra_args", nargs=-1)
def fetch_run(module_name, region, outdir, extra_args):
    """Download data from a curated Globato source.

    MODULE_NAME: The curated source name (e.g., glob_bag, glob_fabdem)
    EXTRA_ARGS: Optional key=value pairs to pass to the module.

    Example: globato fetch run glob_fabdem -R loc:"Denver, CO" -O ./denver_data
    """

    ModuleRegistry.load_all()
    mod_cls = ModuleRegistry.get_class(module_name)

    if not mod_cls:
        click.secho(
            f"Error: Unknown module '{module_name}'. Run 'globato fetch list' to see available options.",
            fg="red",
        )
        sys.exit(1)

    meta = ModuleRegistry.get_info(module_name)
    if meta.get("category") != "Globato":
        click.secho(
            f"Error: '{module_name}' is a raw Fetchez module, not a curated Globato source.",
            fg="red",
        )
        click.secho(
            "Please use the 'fetchez' CLI to download raw data, or select a curated source from 'globato fetch list'.",
            fg="yellow",
        )
        sys.exit(1)

    try:
        for parsed_region, feat_name in yield_parsed_regions(region):
            # prefix = f"{feat_name}: " if feat_name else ""

            click.secho(
                f"Target Region: [{parsed_region.xmin:.4f}, {parsed_region.xmax:.4f}, {parsed_region.ymin:.4f}, {parsed_region.ymax:.4f}]",
                fg="blue",
            )

            kwargs = {}
            for arg in extra_args:
                if "=" in arg:
                    k, v = arg.split("=", 1)
                    try:
                        v = float(v) if "." in v else int(v)
                    except ValueError:
                        pass
                    kwargs[k] = v

            outdir = os.path.abspath(outdir)
            os.makedirs(outdir, exist_ok=True)
            original_cwd = os.getcwd()
            os.chdir(outdir)

            click.secho(f"Initializing {module_name} fetcher...", fg="cyan", bold=True)
            if kwargs:
                click.echo(f"   Using custom arguments: {kwargs}")

            try:
                fetcher = mod_cls(src_region=parsed_region, **kwargs)
                fetcher.run()
                run_fetchez([fetcher])
                click.secho(
                    f"\nFetch complete! Data saved to: {outdir}", fg="green", bold=True
                )
            except Exception as e:
                click.secho(f"\nFetch failed: {e}", fg="red", bold=True)
            finally:
                os.chdir(original_cwd)

    except ValueError as e:
        click.secho(str(e), fg="red")
        sys.exit(1)
