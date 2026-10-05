#!/usr/bin/env python

"""
globato.cli.hook
~~~~~~~~~~~~~~~~
Discoverability and documentation for processing hooks.
"""

import inspect
import sys

import click
from fetchez.registry import HookRegistry


@click.group(name="hook")
def hook_group():
    """Discover and inspect data processing hooks.

    Hooks are modular processing steps (filters, transforms, algorithms)
    that manipulate data streams or files in a pipeline.

    \b
    How to use Hooks:
      1. In YAML Recipes: Defined under `hooks` (per-module) or `global_hooks`.
      2. In CLI Commands: Appended directly to data sources using a plus (+).

    \b
    CLI String Syntax (Source + Hooks):
      <source>:arg=val+<hook_name>:arg=val,arg2=val

    \b
    CLI Examples:
      globato pointz run my_data.laz+rq:threshold=50,mode=percent
      globato recipe build -R loc:Miami copernicus:datatype=3+range_z:min_z=0
    """


@hook_group.command("list")
@click.option("--search", "-s", help="Filter hooks by name or keyword.")
def hook_list(search):
    """List all available processing hooks grouped by category."""

    HookRegistry.load_all()
    registry = HookRegistry.get_registry()

    grouped_hooks = {}
    for name, meta in registry.items():
        if name in meta.get("aliases", []):
            continue

        if (
            search
            and search.lower() not in name.lower()
            and search.lower() not in meta.get("desc", "").lower()
        ):
            continue

        cat = meta.get("category", "uncategorized").title()
        grouped_hooks.setdefault(cat, []).append((name, meta))

    click.secho("\nAvailable Hooks by Category:", fg="cyan", bold=True)
    click.echo("=" * 60)

    for cat in sorted(grouped_hooks.keys()):
        click.secho(f"\n[ {cat} ]", fg="yellow", bold=True)
        for name, meta in sorted(grouped_hooks[cat], key=lambda x: x[0]):
            stage = meta.get("stage", "unknown")
            desc = meta.get("desc", "No description provided.")

            name_padded = f"{name:<20}"
            stage_padded = f"[{stage:<12}]"

            click.echo(
                f"  {click.style(name_padded, bold=True, fg='green')} {click.style(stage_padded, fg='blue')} : {desc}"
            )

    click.echo("\nRun 'globato hook info <name>' for arguments and recipe examples.\n")


@hook_group.command("info")
@click.argument("name")
def hook_info(name):
    """Show arguments and YAML recipe examples for a specific hook."""

    HookRegistry.load_all()
    hook_cls = HookRegistry.get_class(name)
    meta = HookRegistry.get_info(name)

    if not hook_cls:
        click.secho(f"Error: Hook '{name}' not found.", fg="red")
        sys.exit(1)

    click.secho(f"\n🪝 HOOK: {name}", fg="cyan", bold=True)
    click.echo("=" * 60)
    click.echo(f"  Description : {meta.get('desc', 'N/A')}")
    click.echo(f"  Stage       : {meta.get('stage', 'N/A')}")
    click.echo(f"  Category    : {meta.get('category', 'N/A')}\n")

    # Inspect the __init__ signature to find arguments
    sig = inspect.signature(hook_cls.__init__)

    args_dict = {}
    click.secho("  Arguments:", fg="yellow", bold=True)

    has_args = False
    for param_name, param in sig.parameters.items():
        if param_name in ["self", "kwargs", "args"]:
            continue

        has_args = True
        default = (
            param.default
            if param.default is not inspect.Parameter.empty
            else "REQUIRED"
        )
        args_dict[param_name] = default

        req_str = (
            click.style("(Required)", fg="red")
            if default == "REQUIRED"
            else f"(Default: {default})"
        )
        click.echo(f"    - {click.style(param_name, bold=True)} {req_str}")

    if not has_args:
        click.echo("    (No configuration arguments required)")

    # Generate the YAML Snippet
    click.secho("\n  YAML Recipe Example:", fg="green", bold=True)
    click.echo("-" * 40)

    if meta.get("stage") in ["pre", "file"]:
        click.echo("  # Attached to a specific module:")
        click.echo("  modules:")
        click.echo("    - module: example_source")
        click.echo("      hooks:")
        click.echo(f"        - name: {name}")
    else:
        click.echo("  # Placed in the global pipeline:")
        click.echo("  global_hooks:")
        click.echo(f"    - name: {name}")

    if args_dict:
        click.echo("      args:")
        for k, v in args_dict.items():
            val_str = f'"{v}"' if isinstance(v, str) and v != "REQUIRED" else v
            click.echo(f"        {k}: {val_str}")

    click.echo("-" * 40 + "\n")
