#!/usr/bin/env python

"""
globato.cli.bundles
~~~~~~~~~~~~~~~~
Discoverability and documentation for module bundles.
"""

import os
import sys

import click
import yaml
from fetchez.registry import BundleRegistry


@click.group(name="bundle")
def bundle_group():
    """Discover and manage Globato Module Bundles.

    Bundles are collections of fetchez modules, pre-set with with weight hierarchy.

    Use a bundle in your recipe:

    project:
    name: "my_harbor"
    region: [-120.5, -120.0, 34.0, 34.5]
    modules:
    - bundle: us_coastal_streaming
      args: {weight: 1.0}
    """


@bundle_group.command("list")
def bundle_list():
    """List all available curated Data Bundles."""

    BundleRegistry.load_all()
    registry = BundleRegistry.get_registry()

    click.secho("\n📦 Available Curated Data Bundles:", fg="cyan", bold=True)
    click.echo("=" * 60)
    for name, meta in sorted(registry.items()):
        desc = meta.get("desc", meta.get("description", "No description provided."))
        click.secho(f"  {name:<25}", fg="green", bold=True, nl=False)
        click.echo(f" - {desc}")
    click.echo("=" * 60 + "\n")


@bundle_group.command("copy")
@click.argument("target")
@click.option(
    "-O",
    "--outdir",
    default="~/.fetchez/bundles",
    help="Where to save the bundle yaml.",
)
def macro_copy(target, outdir):
    """Copy a macro to your local environment for editing."""

    BundleRegistry.load_all()
    bundle_meta = BundleRegistry.get_yaml(target)

    if not bundle_meta:
        click.secho(f"Error: Bundle '{target}' not found.", fg="red")
        sys.exit(1)

    abs_outdir = os.path.expanduser(outdir)
    os.makedirs(abs_outdir, exist_ok=True)
    out_path = os.path.join(abs_outdir, f"{target}.yaml")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(yaml.dump(bundle_meta, sort_keys=False))

    click.secho(f"Copied '{target}' to {out_path}", fg="green", bold=True)
    click.echo("Globato will now prioritize this local version when executing recipes!")
