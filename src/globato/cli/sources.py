#!/usr/bin/env python

"""
globato.cli.sources
~~~~~~~~~~~~~~~~

Discoverability and documentation for globato dem sources.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import sys

import click
from fetchez.api import list_bundles, list_modules, search_bundles, search_modules
from fetchez.recipe import Recipe
from fetchez.registry import ModuleRegistry
from fetchez.utils import FetchezMainCommand, FetchezMainGroup, truncate_string


@click.group(cls=FetchezMainGroup, name="sources", fetchez_commands=["list", "info"])
def sources_group():
    """Discover, search, and learn about globato sources.

    \b
    Globato curates and provides a number of fetchez module sources. Find them here.
    """


@sources_group.command("list", cls=FetchezMainCommand)
@click.option("--search", "-s", help="Filter sources by name or keyword.")
def sources_list(search):
    """List curated data sources and exit."""

    registry = search_modules(search)
    registry.update(search_bundles(search))

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
            desc = (
                meta.get("description")
                or meta.get("desc", "No Description provided").strip().split("\n")[0]
            ).strip()

            truncated_desc = truncate_string(desc, 40)
            click.echo(
                f"  {click.style(name, bold=True, fg='yellow'):<35} : {truncated_desc}"
            )
            count += 1

    click.echo("-" * 60)
    click.secho("\nLocal File Support:", fg="cyan", bold=True)
    click.echo("=" * 60)
    click.echo("  You can also pass local files and directories directly!")
    click.echo("  Files will be wrapped in the 'file' module.")
    click.echo("  Directories will be crawled using the 'local_fs' module.")
    click.echo("  Example: globato build -R ... ./my_data.tif ./my_folder:ext=.xyz")

    click.echo(f"\nTry 'globato sources info <name>' for details. Total: {count}\n")


@sources_group.command("info", cls=FetchezMainCommand)
@click.argument("name")
def sources_info(name):
    """inspect a specific data source and exit."""

    registry = list_modules()
    bundle_registry = list_bundles()

    source_name = name
    is_bundle = source_name in bundle_registry
    is_module = source_name in registry

    if not is_bundle and not is_module:
        click.secho(f"Error: '{source_name}' is not a recognized source.", fg="red")
        sys.exit(1)

    if is_module:
        meta = registry[source_name]
    elif is_bundle:
        meta = bundle_registry[source_name]

    if not (
        meta.get("mod", "").startswith("globato.modules")
        or meta.get("category") == "Globato"
    ):
        click.secho(
            f" Note: '{source_name}' is a core Fetchez module, not a curated Globato DEM source.",
            fg="yellow",
        )

    if is_bundle:
        modules = Recipe({})._expand_modules(meta.get("modules", []))
    else:
        mod_class = ModuleRegistry.get_class(source_name)()
        mod_hooks = [x.name for x in mod_class.hooks]
        skip_keys = [
            "external_hooks",
            "run",
            "headers",
            "stream_kwargs",
            "wgs_region",
            "region",
        ]
        mod_keys = mod_class.__dict__

        args = {}
        for key in mod_keys:
            if key.startswith("_") or key in skip_keys:
                continue

            key_val = getattr(mod_class, key)
            if key_val:
                args[key] = getattr(mod_class, key)

        meta["module"] = mod_class.name
        meta["args"] = args
        meta["hooks"] = mod_hooks
        modules = [meta]

    desc = meta.get("description") or meta.get("desc", "N/A")
    click.secho(f"\n📜 SOURCE SUMMARY: {source_name}", fg="cyan", bold=True)
    click.echo("=" * 60)
    click.echo(f"  Description : {desc.strip()}")

    if modules:
        click.echo(f"\n  Data Sources ({len(modules)}):")
        for mod in modules:
            mod_name = (
                mod.get("module") or mod.get("bundle") or mod.get("mod") or "Unknown"
            )
            click.echo(f"    + {click.style(mod_name, fg='green')}")
            for arg in mod.get("args", []):
                click.echo(
                    f"     ⤷ {click.style(arg, fg='cyan')}: {mod.get('args').get(arg)}"
                )
            for hook in mod.get("hooks", []):
                if is_bundle:
                    hook = hook.get("name")
                click.echo(f"     ⤷ {click.style(hook, fg='magenta')}")
