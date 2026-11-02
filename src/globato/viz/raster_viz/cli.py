"""Simple standalone geospatial raster viewer; no scientific producer imports."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from .review import Review


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="raster-viz",
        description="View GeoTIFFs in a directory or a producer products.json",
    )
    p.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Finished raster directory or products.json (default: current directory)",
    )
    p.add_argument(
        "--recursive", action="store_true", help="Scan subdirectories in directory mode"
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=250,
        help="Maximum discovered files in directory mode (0 = unlimited)",
    )
    p.add_argument(
        "--manifest", type=Path, help="Explicit products.json; overrides directory scan"
    )
    p.add_argument(
        "--out",
        type=Path,
        help="Output folder (default: viz beside the catalog, or inside the input directory)",
    )
    p.add_argument(
        "--3d",
        dest="terrain_3d",
        action="store_true",
        help="Generate 3D terrain previews for elevation rasters",
    )
    p.add_argument(
        "--watch",
        action="store_true",
        help="Watch the input or catalog for completed new files",
    )
    p.add_argument(
        "--no-open",
        action="store_true",
        help="Do not try to open the HTML viewer automatically",
    )
    p.add_argument(
        "--water-level",
        type=float,
        default=0.0,
        help="Color water blue below this elevation (default 0; source vertical units). Catalog entries override.",
    )
    p.add_argument(
        "--vertical-unit-to-m",
        type=float,
        default=1.0,
        help="3D display-only vertical conversion to meters (default 1)",
    )
    p.add_argument(
        "--mesh-side",
        type=int,
        default=500,
        help="3D mesh maximum dimension (default 500)",
    )
    p.add_argument("--interval", type=float, default=15.0, help=argparse.SUPPRESS)
    from . import __version__

    p.add_argument("--version", action="version", version="raster-viz " + __version__)
    args = p.parse_args(argv)
    if args.manifest is None and args.input.suffix.lower() == ".json":
        args.manifest = args.input
    if args.input.is_file() and args.manifest is None:
        p.error(
            "For a single raster, publish it with raster-viz-catalog; input must be a directory or products.json"
        )
    if args.out is None:
        args.out = (args.manifest.parent if args.manifest else args.input) / "viz"
    if not np.isfinite(args.interval) or args.interval <= 0:
        p.error("--interval must be positive")
    if not np.isfinite(args.vertical_unit_to_m) or args.vertical_unit_to_m <= 0:
        p.error("--vertical-unit-to-m must be positive and finite")
    if not np.isfinite(args.water_level):
        p.error("--water-level must be finite")
    if args.max_files < 0:
        p.error("--max-files must be >= 0")
    if args.mesh_side < 2:
        p.error("--mesh-side must be at least 2")
    rv = Review(
        args.out,
        title="Raster Viz",
        render_3d=args.terrain_3d,
        mesh_side=args.mesh_side,
        open_browser=not args.no_open,
        vertical_unit_to_m=args.vertical_unit_to_m,
        water_level=args.water_level,
    )
    with rv.lock:
        state = rv._load()
        state["refresh_interval_seconds"] = args.interval
        # A new standalone invocation owns its viewing mode. Reusing --out
        # after a 3D run must not leave an obsolete 3D toggle enabled.
        # Explicit producer views=['2d','3d'] can re-enable it on publication.
        state["enable_3d"] = args.terrain_3d
        rv._save(state)
    failed = False
    try:
        if args.manifest:
            from .manifest import sync_catalog, watch_catalog

            rv.start()
            if args.watch:
                result = watch_catalog(
                    rv,
                    args.manifest,
                    interval=args.interval,
                    on_scan=lambda r: print("CATALOG", json.dumps(r), flush=True),
                )
                if result:
                    failed = bool(result["failed"] or result["pending"])
            else:
                result = sync_catalog(rv, args.manifest)
                print("CATALOG", json.dumps(result), flush=True)
                if result["failed"] or result["pending"]:
                    print(
                        "WARNING: Some declared products are unavailable; see CATALOG above",
                        file=sys.stderr,
                    )
                    failed = bool(result["failed"] or not result["producer_running"])
        else:
            from .discovery import discover_into, watch_into

            opts = dict(
                recursive=args.recursive,
                max_files=args.max_files,
                includes=[],
                excludes=[],
                exclude_processing=True,
                extensions={".tif", ".tiff", ".vrt", ".gpkg", ".geojson", ".shp"},
                area="default",
                include_vectors=True,
                style=None,
                skip_existing=True,
            )
            if args.watch:
                rv.start()
                watch_into(
                    rv,
                    args.input,
                    interval=args.interval,
                    on_scan=lambda r: print("DISCOVERY", json.dumps(r), flush=True),
                    **opts,
                )
            else:
                result = discover_into(rv, args.input, **opts)
                print("DISCOVERY", json.dumps(result), flush=True)
                failed = bool(result["skipped"])
    except KeyboardInterrupt:
        pass
    except (ValueError, OSError) as exc:
        p.exit(2, "raster-viz: " + str(exc) + "\n")
    finally:
        rv.finish()
    print("VIZ", rv.output / "viz.html")
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
