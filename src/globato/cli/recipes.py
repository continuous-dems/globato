#!/usr/bin/env python

"""
globato.cli.recipes
~~~~~~~~~~~~~~~~

Discoverability and documentation for globato dem recipes.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import click
from fetchez.cli.recipes import info_recipe, list_recipes, run_recipe
from fetchez.utils import FetchezMainGroup

for param in list_recipes.params:
    if param.name == "search":
        param.default = "globato"


@click.group(
    cls=FetchezMainGroup, name="recipes", fetchez_commands=["list", "info", "run"]
)
def recipes_group():
    """Discover, search, and learn about globato recipes.

    \b
    Globato curates and provides a number of fetchez pipeline recipes. Find them here.
    """


recipes_group.add_command(run_recipe, name="run")
recipes_group.add_command(list_recipes, name="list")
recipes_group.add_command(info_recipe, name="info")
