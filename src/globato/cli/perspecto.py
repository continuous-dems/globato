#!/usr/bin/env python

"""
globato.cli.perspecto
~~~~~~~~~~~~~~~
Visualization tools for DEMs and Point Clouds.
"""

import logging
import os
import sys
import time

import click
from fetchez.registry import HookRegistry
from fetchez.spatial import Region
from fetchez.utils import FetchezMainCommand, FetchezMainGroup, parse_hook_string

from globato.utils import add_field_to_recarray

logger = logging.getLogger(__name__)

PERSPECTO_COMMANDS = ["colorbar", "hillshade", "points"]


def _list_cmaps(ctx, param, value):
    """List available colormaps and exit."""
    if not value or ctx.resilient_parsing:
        return

    click.secho("\n🎨 Globato Built-In Colormaps:", fg="cyan", bold=True)
    click.echo("  etopo           : Standard ETOPO1 elevation/bathymetry palette.")
    click.echo("  coastal_relief  : Custom QGIS Coastal Topobathy palette.")

    click.secho("\n🌍 CPT-City Integration:", fg="cyan", bold=True)
    click.echo("  Fetchez will automatically download valid CPT-City queries.")
    click.echo("  Examples: 'grass/haxby', 'cmocean/bathy', 'gmt/globe'")

    click.secho("\n📊 Matplotlib Native Colormaps:", fg="cyan", bold=True)
    try:
        import matplotlib.pyplot as plt

        cmaps = plt.colormaps()
        click.echo(f"  {', '.join(cmaps[:25])} ... and {len(cmaps) - 25} more.")
        click.echo(
            "  (Full list: https://matplotlib.org/stable/gallery/color/colormap_reference.html)"
        )
    except ImportError:
        click.echo("  Matplotlib is not installed.")

    click.echo("\n")
    ctx.exit()


@click.version_option(package_name="globato")
@click.group(
    cls=FetchezMainGroup,
    name="perspecto",
    fetchez_commands=PERSPECTO_COMMANDS,
)
def perspecto_group():
    """Visualize DEMs and Point Clouds."""


@perspecto_group.command("hillshade", cls=FetchezMainCommand)
@click.argument("src")
@click.argument("dst")
@click.option(
    "--azimuth",
    type=float,
    default=315.0,
    help="Sun azimuth angle in degrees (default: 315).",
)
@click.option(
    "--altitude",
    type=float,
    default=45.0,
    help="Sun altitude angle in degrees (default: 45).",
)
@click.option(
    "--exag",
    type=float,
    default=1.0,
    help="Vertical exaggeration factor (default: 1.0).",
)
@click.option(
    "--cmap",
    default="coastal_relief",
    help="Matplotlib colormap or CPT file/name (default: 'etopo').",
)
@click.option(
    "--blend",
    type=click.Choice(["multiply", "screen", "overlay", "hard_light", "soft_light"]),
    default="multiply",
    help="Blending mode (default: soft_light).",
)
@click.option("--alpha", is_flag=True, help="Add an alpha channel to mask NoData.")
@click.option("--gamma", type=float, help="Gamma correction factor.")
@click.option("--z-min", type=float, help="Force minimum Z value for the colormap.")
@click.option("--z-max", type=float, help="Force maximum Z value for the colormap.")
@click.option(
    "--list-cmaps",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_list_cmaps,
    help="List available native and custom colormaps and exit.",
)
@click.option(
    "--split-cpt",
    type=float,
    default=0.0,
    help="Hinge point for divergent colormaps (default: 0.0).",
)
def perspecto_hillshade(
    src,
    dst,
    azimuth,
    altitude,
    exag,
    cmap,
    blend,
    alpha,
    gamma,
    z_min,
    z_max,
    split_cpt,
):
    """Generate a georeferenced colored hillshade.

    SRC: Input DEM (GeoTIFF or NetCDF)
    DST: Output colored hillshade (GeoTIFF)

    Example: globato perspecto hillshade my_dem.tif my_hillshade.tif --cmap etopo --exag 3 --blend soft_light
    """

    try:
        from globato.hooks.viz.geohillshade import GeoHillshade
    except ImportError as e:
        click.secho(f"Error importing GeoHillshade: {e}", fg="red")
        sys.exit(1)

    if not os.path.exists(src):
        click.secho(f"Error: Input file not found: {src}", fg="red")
        sys.exit(1)

    hook = GeoHillshade(
        azimuth=azimuth,
        altitude=altitude,
        vert_exag=exag,
        cmap=cmap,
        blend_mode=blend,
        alpha=alpha,
        gamma=gamma,
        z_min=z_min,
        z_max=z_max,
        split_cpt=split_cpt,
    )

    click.secho(
        f"\nGenerating Colored Hillshade: {os.path.basename(src)}...",
        fg="cyan",
        bold=True,
    )
    click.echo(f"   Colormap: {cmap} | Exag: {exag}x | Blend: {blend}")

    start_time = time.time()

    entry = {"src_fn": src, "dst_fn": dst}
    success = hook._process_file_fallback(src, dst, entry)

    elapsed = time.time() - start_time

    if success:
        click.secho(f"\nVisualization Complete! Saved to: {dst}", fg="green", bold=True)
        click.echo(f"   Time: {elapsed:.2f} seconds\n")
    else:
        click.secho("\nFailed to generate hillshade.", fg="red")
        sys.exit(1)


@perspecto_group.command("colorbar", cls=FetchezMainCommand)
@click.argument("src")
@click.argument("dst")
@click.option(
    "--cmap",
    default="etopo",
    help="Matplotlib colormap or CPT file/name (default: 'etopo').",
)
@click.option("--z-min", type=float, help="Force minimum Z value for the colormap.")
@click.option("--z-max", type=float, help="Force maximum Z value for the colormap.")
@click.option(
    "--list-cmaps",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_list_cmaps,
    help="List available native and custom colormaps and exit.",
)
@click.option(
    "--split-cpt",
    type=float,
    default=0.0,
    help="Hinge point for divergent colormaps (default: 0.0).",
)
def perspecto_colorbar(
    src,
    dst,
    cmap,
    z_min,
    z_max,
    split_cpt,
):
    """Generate a colorbar.

    SRC: Input DEM
    DST: Output colorbar (PNG)

    Example: globato perspecto colorbar my_dem.tif my_colorbar.png
    """

    try:
        from globato.hooks.viz.colorbar import ColorBar
    except ImportError as e:
        click.secho(f"Error importing ColorBar: {e}", fg="red")
        sys.exit(1)

    if not os.path.exists(src):
        click.secho(f"Error: Input file not found: {src}", fg="red")
        sys.exit(1)

    hook = ColorBar(
        cmap=cmap,
        z_min=z_min,
        z_max=z_max,
        split_cpt=split_cpt,
    )

    click.secho(
        f"\nGenerating Color Bar: {os.path.basename(src)}...",
        fg="cyan",
        bold=True,
    )
    click.echo(f"   Colormap: {cmap}")

    start_time = time.time()

    entry = {"src_fn": src, "dst_fn": dst}
    success = hook.process_raster(src, dst, entry)

    elapsed = time.time() - start_time

    if success:
        click.secho(f"\nVisualization Complete! Saved to: {dst}", fg="green", bold=True)
        click.echo(f"   Time: {elapsed:.2f} seconds\n")
    else:
        click.secho("\nFailed to generate colorbar.", fg="red")
        sys.exit(1)


def _prepare_stream(gen):
    """Injects required schema fields (w, u) into raw point streams."""
    for chunk in gen:
        chunk = add_field_to_recarray(chunk, "w", float, 1.0)
        chunk = add_field_to_recarray(chunk, "u", float, 0.0)
        yield chunk


@perspecto_group.command("points", cls=FetchezMainCommand)
@click.argument("src")
@click.option(
    "-F",
    "--filter",
    "filters",
    multiple=True,
    help="Apply filters on-the-fly to classify outliers.",
)
@click.option(
    "-R", "--region", help="Spatial crop (required if using spatial filters like rq)."
)
@click.option(
    "--3d",
    "is_3d",
    is_flag=True,
    help="Render an interactive 3D plot (auto-decimates large data).",
)
@click.option(
    "--outliers", is_flag=True, help="Highlight rejected points (Class 7) in red."
)
@click.option("--out", default="{base}_persp.png", help="Output image filename.")
def perspecto_points(src, filters, region, is_3d, outliers, out):
    """Visualize a point cloud for quick sanity checks and filter tuning.

    Examples:
      globato perspecto points data.xyz --3d
      globato perspecto points mbdb -R loc:"Miami" -F rq:threshold=5 --outliers
    """

    from fetchez.core import run_fetchez
    from fetchez.registry import ReaderRegistry

    try:
        from globato.hooks.viz.pc import PointCloudViz
    except ImportError as e:
        click.secho(f"Error importing PointCloudViz: {e}", fg="red")
        sys.exit(1)

    HookRegistry.load_all()
    ReaderRegistry.load_all()
    active_filters = []
    parsed_region = Region.from_string(region) if region else None

    dummy_mod = type("Dummy", (), {"region": parsed_region, "name": "cli_perspecto"})()

    for f_str in filters:
        # Use the dictionary returned by parse_hook_string
        hook_dict = parse_hook_string(f_str)
        f_name = hook_dict["name"]
        f_kwargs = hook_dict.get("args", {})

        mod_cls = HookRegistry.get_class(f_name)
        if not mod_cls:
            click.secho(f"Error: Unknown filter '{f_name}'", fg="red")
            sys.exit(1)

        f = mod_cls(**f_kwargs)
        if hasattr(f, "setup") and f.setup(dummy_mod, {}) is False:
            click.secho(
                f"Error: Filter '{f.name}' failed to initialize. Did you forget a --region (-R)?",
                fg="red",
            )
            sys.exit(1)
        active_filters.append(f)

    click.secho(f"Loading point cloud: {src}...", fg="cyan")

    entries = []
    if os.path.exists(src):
        reader = ReaderRegistry.get_reader(src, src.split(".")[-1])
        # Apply _prepare_stream to guarantee schema!
        stream = _prepare_stream(reader.yield_chunks()) if reader else []
        entries.append((dummy_mod, {"dst_fn": src, "stream": stream}))
    else:
        mod_cls = HookRegistry.get_class(src)
        if not mod_cls or not parsed_region:
            click.secho(
                "Error: Invalid file, or missing -R for module streaming.", fg="red"
            )
            sys.exit(1)
        fetcher = mod_cls(src_region=parsed_region)
        fetcher.run()
        run_fetchez([fetcher])
        for entry in fetcher.results:
            if entry.get("dst_fn"):
                r = ReaderRegistry.get_reader(
                    entry["dst_fn"], entry["dst_fn"].split(".")[-1]
                )
                if r:
                    entries.append(
                        (
                            dummy_mod,
                            {
                                "dst_fn": entry["dst_fn"],
                                "stream": _prepare_stream(r.yield_chunks()),
                            },
                        )
                    )

    if not entries:
        click.secho("Error: No valid data streams found.", fg="red")
        sys.exit(1)

    for f in active_filters:
        entries = f.run(entries)

    if not dummy_mod.region:
        dummy_mod.region = "Global"

    perspecto_hook = PointCloudViz(output=out, outliers=outliers, is_3d=is_3d)
    entries = perspecto_hook.run(entries)

    total_pts = 0
    for mod, entry in entries:
        stream = entry.get("stream")
        if stream:
            for chunk in stream:
                total_pts += len(chunk)

    if total_pts == 0:
        click.secho("Error: No points survived the filter pipeline.", fg="red")

    for f in active_filters:
        if hasattr(f, "teardown"):
            f.teardown()
