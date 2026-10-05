#!/usr/bin/env python

"""
globato.cli.gritz
~~~~~~~~~~~~~~~~

The command-line interface for the raster/grits group.
"""

import os
import sys
import time

import click
import numpy as np
from fetchez.utils import FetchezMainCommand, FetchezMainGroup


def generate_gritz_receipt(src_path, dst_path, op_name, elapsed):
    """Calculates before/after statistics and prints a receipt."""

    try:
        import rasterio
    except ImportError:
        return

    click.echo("\n" + "=" * 60)
    click.secho(f"GRITZ OPERATION COMPLETE: {op_name.upper()}", bold=True, fg="green")
    click.echo("=" * 60)
    click.echo(f"  Source : {os.path.basename(src_path)}")
    click.echo(f"  Output : {os.path.basename(dst_path)}")
    click.echo(f"  Time   : {elapsed:.2f} seconds")

    try:
        with rasterio.open(src_path) as s, rasterio.open(dst_path) as d:
            if s.shape == d.shape:
                s_data = s.read(1)
                d_data = d.read(1)
                s_ndv = s.nodata if s.nodata is not None else -9999
                d_ndv = d.nodata if d.nodata is not None else -9999

                s_valid = (s_data != s_ndv) & ~np.isnan(s_data)
                d_valid = (d_data != d_ndv) & ~np.isnan(d_data)

                s_count = np.sum(s_valid)
                d_count = np.sum(d_valid)

                modified = np.sum(s_valid & d_valid & (s_data != d_data))
                removed = np.sum(s_valid & ~d_valid)
                added = np.sum(~s_valid & d_valid)

                click.echo("-" * 60)
                click.secho("  Pixel Statistics (Band 1):", bold=True)
                click.echo(f"    Total Pixels    : {s.width * s.height:,}")
                click.echo(f"    Valid Before    : {s_count:,}")
                click.echo(f"    Valid After     : {d_count:,}")
                click.secho(f"    Pixels Modified : {modified:,}", fg="cyan")
                click.secho(
                    f"    Pixels Removed  : {removed:,} (Set to NoData)", fg="red"
                )
                click.secho(
                    f"    Pixels Added    : {added:,} (Filled/Interpolated)",
                    fg="yellow",
                )
    except Exception as e:
        click.secho(f"  [Could not compute pixel stats: {e}]", fg="yellow")

    click.echo("=" * 60 + "\n")


def run_raster_hook(hook_instance, src, dst, strip_bands=False):  # , region=None):
    """Execution wrapper for standalone raster commands."""

    # if region:
    #     try:
    #         r_vals = [float(x) for x in region.replace(',', '/').split('/')]
    #         if len(r_vals) == 4:
    #             hook_instance.region = TransRegion(r_vals)
    #         else:
    #             click.secho("Error: Region must be W/E/S/N", fg="red")
    #             sys.exit(1)
    #     except Exception as e:
    #         click.secho(f"Invalid region format: {e}", fg="red")
    #         sys.exit(1)

    if strip_bands:
        hook_instance.strip_bands = True

    entry = {"src_fn": src, "dst_fn": dst, "weight": 1.0}

    click.secho(
        f"\nStarting {hook_instance.name} on {os.path.basename(src)}...",
        fg="cyan",
        bold=True,
    )
    start_time = time.time()

    # try:
    success = hook_instance.process_raster(src, dst, entry)
    elapsed = time.time() - start_time

    if success:
        generate_gritz_receipt(src, dst, hook_instance.name, elapsed)
    else:
        click.secho("Operation failed (hook returned False)", fg="red")
        sys.exit(1)
    # except Exception as e:
    #    click.secho(f"Error during processing: {e}", fg="red")
    #    sys.exit(1)


def raster_io(f):
    """Click Decorator to share standard IO arguments across all raster commands."""

    f = click.option(
        "--strip-bands", is_flag=True, help="Strip extra bands in the output."
    )(f)
    f = click.argument("dst")(f)
    f = click.argument("src")(f)
    return f


GRITZ_COMMANDS = [
    "diff",
    "slope",
    "clip",
    "crop",
    "cut",
    "flats",
    "fill",
    "morph",
    "interp",
    "blend",
    "zscore",
]


# =============================================================================
# GRITS (RASTER TOOLS)
# =============================================================================
@click.version_option(package_name="globato")
@click.group(
    cls=FetchezMainGroup,
    name="gritz",
    fetchez_commands=GRITZ_COMMANDS,
)
def gritz_group():
    """Raster manipulation tools."""


@gritz_group.command("diff", cls=FetchezMainCommand)
@raster_io
@click.option("--aux", required=True, help="Auxiliary/Reference Raster")
@click.option(
    "--mode", type=click.Choice(["difference", "filter"]), default="difference"
)
@click.option("--threshold", type=float, help="Filter threshold")
def raster_diff(src, dst, strip_bands, aux, mode, threshold):
    """Calculate difference (Src - Aux)."""

    from globato.hooks.rasters.diff import RasterDiff

    hook = RasterDiff(aux_path=aux, mode=mode, threshold=threshold)
    run_raster_hook(hook, src, dst, strip_bands)


@gritz_group.command("slope", cls=FetchezMainCommand)
@raster_io
@click.option("--min", "min_val", type=float, help="Min Slope")
@click.option("--max", "max_val", type=float, help="Max Slope")
def raster_slope(src, dst, strip_bands, min_val, max_val):
    """Filter by Slope."""

    from globato.hooks.rasters.slope import RasterSlopeFilter

    hook = RasterSlopeFilter(min_val=min_val, max_val=max_val)
    run_raster_hook(hook, src, dst, strip_bands)


@gritz_group.command("clip", cls=FetchezMainCommand)
@raster_io
@click.option("-B", "--barrier", required=True, help="Vector to use for clipping.")
@click.option(
    "-i", "--invert", is_flag=True, default=False, help="Invert the vector mask"
)
def raster_clip(src, dst, strip_bands, barrier, invert):
    """Cut/Mask to Region."""

    from globato.hooks.rasters.clip import RasterClipHook

    hook = RasterClipHook(barrier=barrier, invert=invert)
    run_raster_hook(hook, src, dst, strip_bands)


@gritz_group.command("cut", cls=FetchezMainCommand)
@raster_io
@click.option("-R", "--region", required=True, help="Region W/E/S/N")
def raster_cut(src, dst, strip_bands, region):
    """Cut/Mask to Region."""

    from globato.hooks.rasters.cut import RasterCut

    hook = RasterCut(region=region)
    run_raster_hook(hook, src, dst, strip_bands)


@gritz_group.command("crop", cls=FetchezMainCommand)
@raster_io
def raster_crop(src, dst, strip_bands):
    """Crop a raster to its valid data bounds (removes NoData moat)."""

    from globato.hooks.rasters.crop import RasterCrop

    hook = RasterCrop()
    run_raster_hook(hook, src, dst, strip_bands)


@gritz_group.command("flats", cls=FetchezMainCommand)
@raster_io
@click.option(
    "--threshold", type=float, default=1.0, help="Minimum size of a flat-zone"
)
def raster_flats(src, dst, strip_bands, threshold):
    """Remove Flat-Zones."""

    from globato.hooks.rasters.flats import RasterFlats

    hook = RasterFlats(size_threshold=threshold)
    run_raster_hook(hook, src, dst, strip_bands)


@gritz_group.command("fill", cls=FetchezMainCommand)
@raster_io
@click.option("--dist", type=float, default=100.0, help="Max search distance")
@click.option("--smooth", type=int, default=0, help="Smoothing iterations")
def raster_fill(src, dst, strip_bands, dist, smooth):
    """Fill NoData using IDW."""

    from globato.hooks.rasters.fill import RasterFill

    hook = RasterFill(max_dist=dist, smoothing=smooth)
    run_raster_hook(hook, src, dst, strip_bands)


@gritz_group.command("morph", cls=FetchezMainCommand)
@raster_io
@click.option(
    "--op",
    type=click.Choice(["erosion", "dilation", "opening", "closing"]),
    default="erosion",
)
@click.option("--kernel", type=int, default=3, help="Kernel size")
def raster_morph(src, dst, strip_bands, op, kernel):
    """Morphology Operations."""

    from globato.hooks.rasters.morphology import RasterMorphology

    hook = RasterMorphology(op=op, kernel=kernel)
    run_raster_hook(hook, src, dst, strip_bands)


@gritz_group.command("interp", cls=FetchezMainCommand)
@raster_io
@click.option(
    "--method", type=click.Choice(["linear", "cubic", "nearest"]), default="linear"
)
def raster_interp(src, dst, strip_bands, method):
    """Interpolate Gaps."""

    from globato.hooks.rasters.scipy_griddata import ScipyInterp

    hook = ScipyInterp(method=method)
    run_raster_hook(hook, src, dst, strip_bands)


@gritz_group.command("blend", cls=FetchezMainCommand)
@raster_io
@click.option("--aux", required=True, help="Auxiliary/Reference Raster")
@click.option("--blend-dist", type=float, default=20.0, help="Max blend distance")
@click.option("--core-dist", type=float, default=5.0, help="Max core blend distance")
@click.option("--slope-scale", type=float, default=0.5, help="Normalize the slope-gate")
@click.option(
    "--random-scale", type=float, default=0.05, help="Density of random points"
)
def raster_blend(
    src, dst, strip_bands, aux, blend_dist, core_dist, slope_scale, random_scale
):
    """Blend rasters (Src -> Aux)."""

    from globato.hooks.rasters.blend import RasterBlend

    hook = RasterBlend(
        aux_path=aux,
        blend_dist=blend_dist,
        core_dist=core_dist,
        slope_scale=slope_scale,
        random_scale=random_scale,
    )
    run_raster_hook(hook, src, dst, strip_bands)


@gritz_group.command("zscore", cls=FetchezMainCommand)
@raster_io
@click.option(
    "--threshold", type=float, default=3.0, help="Mask zscore over this threshold"
)
@click.option(
    "--size", type=int, default=5, help="The size of the neighborhood window."
)
def raster_zscore(src, dst, strip_bands, threshold, size):
    """Filter based on neighborhood z-score."""

    from globato.hooks.rasters.zscore import RasterZScore

    hook = RasterZScore(threshold=threshold, kernel_size=size)
    run_raster_hook(hook, src, dst, strip_bands)
