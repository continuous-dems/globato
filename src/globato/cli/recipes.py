#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.cli.recipes
~~~~~~~~~~~~~~~~

Discoverability and documentation for globato dem recipes.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import click

from fetchez.utils import FetchezMainGroup
from fetchez.cli.recipes import list_recipes, info_recipe
from fetchez.cli.run import run_recipe

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

    pass


recipes_group.add_command(run_recipe, name="run")
recipes_group.add_command(list_recipes, name="list")
recipes_group.add_command(info_recipe, name="info")
