#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.cli.build
~~~~~~~~~~~~~~~~~~

The globato build command to build a fetchez recipe and execute it.

:copyright: (c) 2025 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import sys
import click
import logging

from fetchez.utils import FetchezMainCommand

import globato.api


logger = logging.getLogger(__name__)


# --- Build command ---
CONTEXT_SETTINGS = dict(max_content_width=220)


@click.command("build", cls=FetchezMainCommand, context_settings=CONTEXT_SETTINGS)
@click.option("-R", "--region", required=True, help="Bounding box: W/E/S/N.")
@click.option(
    "-E",
    "--increment",
    required=True,
    help="Target DEM increment (e.g. 1s, 30m).",
)
@click.option(
    "-O", "--outname", default="globato_dem", show_default=True, help="Output basename."
)
@click.option(
    "-D",
    "--outdir",
    type=click.Path(),
    default=None,
    help="Base output directory.",
)
@click.option(
    "-F", "--format", default="GTiff", show_default=True, help="Output format."
)
@click.option(
    "-P",
    "--t-srs",
    default="EPSG:4326",
    show_default=True,
    help="Target CRS.",
)
@click.option(
    "-N",
    "--nodata",
    type=float,
    default=-9999.0,
    show_default=True,
    help="NoData value.",
)
@click.option(
    "-M",
    "--algo",
    default="ms_binary_cudem",
    show_default=True,
    help=(
        "DEM/interpolation hook specification. The API supplies Binary CUDEM "
        "defaults such as OSM barriers, point-tier handoff, and raster_fill "
        "unless explicitly overridden here."
    ),
)
@click.option(
    "-A",
    "--stack-strategy",
    "stack_strategy",
    type=click.Choice(
        ["mean", "weighted_mean", "mixed", "supercede"],
        case_sensitive=False,
    ),
    default="mixed",
    show_default=True,
    help=(
        "MultiStack source-to-source accumulation strategy. "
        "'--stack-strategty' is retained as a compatibility alias."
    ),
)
@click.option(
    "--stack-weights",
    default="auto",
    show_default=True,
    help=(
        "MultiStack mixed-mode competition thresholds "
        "('auto' or e.g. '.1/.2/.3/.4/.5/.75/1')."
    ),
)
@click.option(
    "-W",
    "--dem-weights",
    "dem_weights",
    default="auto",
    show_default=True,
    help=("Binary CUDEM resolution-admission thresholds ('auto' or e.g. '.5/.25/.1')."),
)
@click.option(
    "--previous-tier-mode",
    type=click.Choice(["points", "raster"], case_sensitive=False),
    default="points",
    show_default=True,
    help=(
        "How Binary CUDEM passes the completed coarser tier into the next tier. "
        "'points' reuses sparse guide constraints; 'raster' resamples the surface."
    ),
)
@click.option(
    "--previous-tier-resampling",
    type=click.Choice(
        ["nearest", "bilinear", "cubic", "cubic_spline", "lanczos"],
        case_sensitive=False,
    ),
    default="bilinear",
    show_default=True,
    help="Raster resampling method used when --previous-tier-mode=raster.",
)
@click.option(
    "-B",
    "--blend",
    type=str,
    default=None,
    help=(
        "Binary CUDEM blend distances. If omitted, the API chooses strategy-"
        "appropriate defaults (zero for point-tier handoff)."
    ),
)
@click.option("-T", "--filter", "filters", multiple=True, help="Apply a Grits filter.")
@click.option("-C", "--clip", help="Clip output to polygon file.")
@click.option(
    "-X",
    "--extend",
    type=str,
    default="0:0",
    show_default=True,
    help="Extend region (cells[:percent]).",
)
@click.option("-L", "--limits", type=str, default=None, help="Set global DEM limits.")
@click.option(
    "--modifier",
    multiple=True,
    help="Apply a recipe modifier at runtime (e.g. exclude_module:modules=csb/tnm).",
)
@click.option(
    "--schema",
    multiple=True,
    help="Apply a domain schema validation to the recipe.",
)
@click.option(
    "--shared-cache",
    type=click.Path(),
    help="Centralized cache directory.",
)
@click.option("--metadata", help="Global tags to inject.")
@click.option("--export", is_flag=True, help="Save the generated YAML recipe to disk.")
@click.option(
    "--refresh",
    is_flag=True,
    help="Force fresh API fetch, bypassing local cache.",
)
@click.option(
    "--fail-fast",
    is_flag=True,
    help="Raise on the first failure instead of continuing through failures.",
)
@click.argument("sources", nargs=-1)
def build_cmd(sources, **kwargs):
    """Build a Digital Elevation Model recipe and execute it."""

    if not sources:
        click.secho(
            "Error: You must provide at least one data source or a modules.yaml file.",
            fg="red",
        )
        sys.exit(1)

    try:
        if not kwargs.get("export"):
            click.secho(
                f"Executing dynamic recipe for {kwargs.get('outname')}...",
                fg="cyan",
                bold=True,
            )

        # The CLI intentionally delegates recipe construction and default policy
        # to globato.api.build so Python and command-line builds stay aligned.
        for _ in globato.api.build(sources=sources, **kwargs):
            pass

        if kwargs.get("export"):
            click.secho(
                f"Globato recipe exported to {kwargs.get('outname')}_recipe.yaml.",
                fg="green",
                bold=True,
            )
        else:
            click.secho(
                "Successfully completed Globato build pipeline!",
                fg="green",
                bold=True,
            )

    except Exception as e:
        click.secho(
            f"Failed to execute Globato pipeline: {e}",
            fg="red",
            bold=True,
        )
        sys.exit(1)
