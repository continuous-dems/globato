#!/usr/bin/env python

"""
globato.cli.transform
~~~~~~~~~~~~~~~~~~~~~
Vertical Datum transformation using the Transformez API.
"""

import logging
import sys

import click
from transformez import api

logger = logging.getLogger(__name__)


@click.group(name="transform")
def transform_group():
    """Apply vertical datum transformations and generate shift grids."""


@transform_group.command("run")
@click.argument("input_file", required=False)
@click.option(
    "-R", "--region", help="Bounding box or location string (if no input file)."
)
@click.option(
    "-E", "--increment", help="Resolution (e.g., 1s, 30m) (if no input file)."
)
@click.option(
    "-I", "--input-datum", required=True, help="Source Datum (e.g., 'mllw', '5703')."
)
@click.option(
    "-O",
    "--output-datum",
    required=True,
    help="Target Datum (e.g., '4979', '5703:g2012b').",
)
@click.option("--out", "-o", help="Output filename (default: auto-named).")
@click.option(
    "--decay-pixels",
    type=int,
    default=100,
    help="Number of pixels to decay tidal shifts inland.",
)
def transform_run(
    input_file, region, increment, input_datum, output_datum, out, decay_pixels
):
    """Transform a raster's vertical datum or generate a standalone shift grid.

    If an INPUT_FILE is provided, that specific raster is transformed in place.
    If no INPUT_FILE is provided, -R and -E must be used to generate a shift grid.

    Examples:
      Transform a DEM : globato transform run my_dem.tif -I mllw -O 5703
      Generate a Grid : globato transform run -R loc:"Miami" -E 1s -I mllw -O 4979
    """

    if input_file:
        click.secho(f"Transforming raster: {input_file}", fg="cyan", bold=True)
        click.echo(f"   Shift: {input_datum} ➔ {output_datum}")

        result = api.transform_raster(
            input_raster=input_file,
            datum_in=input_datum,
            datum_out=output_datum,
            decay_pixels=decay_pixels,
            output_raster=out,
            verbose=True,
        )

        if result:
            click.secho(
                f"Successfully transformed raster: {result}", fg="green", bold=True
            )
        else:
            click.secho("Failed to transform raster.", fg="red")
            sys.exit(1)

    elif region and increment:
        click.secho(
            "Generating vertical shift grid for region...", fg="cyan", bold=True
        )
        click.echo(f"   Shift: {input_datum} ➔ {output_datum} @ {increment}")

        # Auto-generate an output name if one wasn't provided
        out_fn = out or f"shift_{input_datum}_to_{output_datum.replace(':', '_')}.tif"

        result = api.generate_grid(
            region=region,
            increment=increment,
            datum_in=input_datum,
            datum_out=output_datum,
            decay_pixels=decay_pixels,
            out_fn=out_fn,
            verbose=True,
        )

        if result is not None:
            click.secho(
                f"Successfully generated shift grid: {out_fn}", fg="green", bold=True
            )
        else:
            click.secho("Failed to generate shift grid.", fg="red")
            sys.exit(1)

    else:
        click.secho(
            "Error: You must provide either an INPUT_FILE or both --region and --increment.",
            fg="red",
        )
        sys.exit(1)


@transform_group.command("list")
def transform_list():
    """List all supported vertical datums, surfaces, and geoids."""

    try:
        from transformez.definitions import Datums

        click.secho("\n🌊 Supported Tidal Surfaces:", fg="cyan", bold=True)
        for k, v in Datums.SURFACES.items():
            region_str = v.get("region", "global").upper()
            click.echo(f"  {v['name']:<10} : {v['description']} [{region_str}]")

        click.secho("\n🌐 Ellipsoidal / Frame Datums:", fg="cyan", bold=True)
        click.echo(f"  {'NAD83':<10} : North American Datum 1983 (EPSG:6319)")
        click.echo(f"  {'WGS84':<10} : World Geodetic System 1984 (EPSG:4979)")

        click.secho("\n🏔️  Orthometric / Geoid-Based:", fg="cyan", bold=True)
        for k, v in Datums.CDN.items():
            geoid_str = v.get("default_geoid", "None")
            click.echo(f"  {v['name']:<20} (Default Geoid: {geoid_str})")

        click.secho("\n🌍 Available Geoids:", fg="cyan", bold=True)
        click.echo(f"  {', '.join(Datums.GEOIDS.keys())}\n")

    except ImportError:
        click.secho("Error: Could not load Transformez datum definitions.", fg="red")
