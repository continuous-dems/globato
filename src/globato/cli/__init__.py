#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.cli
~~~~~~~~~~~

The main command-line interface for the Globato framework.

:copyright: (c) 2025 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import click
import logging

# Globato CLI
from .build import build_cmd
from .sources import sources_group
from .pointz import dump
from .perspecto import perspecto_hillshade
from .recipes import recipes_group
from .gui import gui_cmd
from .process import process_cmd

try:
    from fetchez.cli.run import run_recipe
except ImportError:
    from fetchez.cli.recipes import run_recipe

from fetchez.cli import setup_logging
from fetchez.utils import FetchezMainGroup

logger = logging.getLogger(__name__)


class GlobatoMainGroup(FetchezMainGroup):
    """A custom Click Group that handles deprecated aliases."""

    def get_command(self, ctx, cmd_name):
        if cmd_name == "cudem":
            click.secho(
                " DEPRECATION WARNING: 'globato cudem' is deprecated and will be removed in a future release.\n"
                "Please use 'globato build' to generate DEM or the `fetchez` CLI to discover and run recipes..",
                fg="yellow",
                err=True,
            )
            return click.Group.get_command(self, ctx, "build")

        elif cmd_name == "persepcto":
            click.secho(
                " DEPRECATION WARNING: 'globato perspecto' is deprecated and will be removed in a future release.\n"
                "Please use 'globato hillshade' to generate Hillshade images..",
                fg="yellow",
                err=True,
            )
            return click.Group.get_command(self, ctx, "hillshade")

        elif cmd_name == "dlim":
            click.secho(
                " DEPRECATION WARNING: 'globato dlim' is deprecated and will be removed in a future release.\n"
                "Please use 'globato dump' to process point cloud data..",
                fg="yellow",
                err=True,
            )
            return click.Group.get_command(self, ctx, "dump")

        return click.Group.get_command(self, ctx, cmd_name)


@click.group(
    cls=GlobatoMainGroup,
    fetchez_commands={
        "Commands": [
            "run",
            "build",
            "dump",
            "hillshade",
            "gui",
            "process",
        ],
        "Discovery": [
            "recipes",
            "sources",
        ],
    },
)
@click.version_option(package_name="globato")
@click.option("--verbose", is_flag=True, help="Enable verbose debug logging.")
@click.option("--quiet", is_flag=True, help="Suppress non-error output.")
def cli(verbose, quiet):
    """Globato: The ContinUous-DEM Generation Framework.

    \b
    This is the GLOBATO automated DEM compilation engine. It takes overlapping
    streams of geospatial data, seamlessly stacks them based on quality weights,
    and interpolates the gaps to build continuous Digital Elevation Models (DEMs).

    \b
    Build a recipe to generate a custom DEM from scratch with the build command,
    or use a curated builtin or custom recipe to construct a DEM for your region
    and specifications with the make command.
    """

    # logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    setup_logging(name="fetchez", quiet=quiet, verbose=verbose)
    setup_logging(name="transformez", quiet=quiet, verbose=verbose)
    setup_logging(name="globato", quiet=quiet, verbose=verbose)


cli.add_command(build_cmd, name="build")
cli.add_command(sources_group, name="sources")
cli.add_command(run_recipe, name="run")
cli.add_command(dump, name="dump")
cli.add_command(perspecto_hillshade, name="hillshade")
cli.add_command(recipes_group, name="recipes")
cli.add_command(gui_cmd, name="gui")
cli.add_command(process_cmd, name="process")


if __name__ == "__main__":
    cli()
