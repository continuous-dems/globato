"""Standard command-line publisher: paths and display labels, no DEM processing.

This is deliberately a thin wrapper around the stdlib-only ProductCatalog.
All source paths are interpreted relative to the shell's working directory.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

from .catalog import ProductCatalog

STYLES = ("elevation", "difference", "scalar", "grayscale", "categorical", "rgb")


def assignment(value, label):
    key, delimiter, text = value.partition("=")
    if not delimiter or not key.strip() or not text.strip():
        raise ValueError(f"{label} must be KEY=VALUE: {value!r}")
    return key.strip(), text.strip()


def named_options(items, label):
    result = {}
    for text in items:
        key, value = assignment(text, label)
        if key in result:
            raise ValueError(f"duplicate {label}: {key}")
        result[key] = value
    return result


def file_key(path):
    """Stable, human-readable key for a relative input path."""
    text = re.sub(r"[^a-z0-9]+", "_", str(path.with_suffix("")).lower()).strip("_")
    return text or "product"


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="raster-viz-catalog",
        description="Publish a simple products.json from finished geospatial files (no processing)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("products.json"),
        help="Catalog path (default: ./products.json)",
    )
    parser.add_argument("--title", default="Raster visualization")
    parser.add_argument(
        "--area", default="default", help="Common study area/tile identifier"
    )
    for name in (
        "raster",
        "rgb",
        "difference",
        "scalar",
        "grayscale",
        "categorical",
        "vector",
    ):
        parser.add_argument(
            "--" + name,
            action="append",
            default=[],
            metavar="KEY=FILE",
            help="Add completed " + name + " file (repeatable)",
        )
    for name in ("label", "group", "stage", "band", "layer"):
        parser.add_argument(
            "--" + name,
            action="append",
            default=[],
            metavar="KEY=TEXT",
            help="Set product " + name + " (repeatable)",
        )
    parser.add_argument(
        "--water-level",
        type=float,
        help="Elevation palette transition in TIFF elevation units (default: 0)",
    )
    parser.add_argument(
        "--visible",
        action="append",
        default=[],
        metavar="KEY",
        help="Show this vector initially (repeatable)",
    )
    parser.add_argument(
        "--hidden",
        action="append",
        default=[],
        metavar="KEY",
        help="Hide this vector initially (repeatable)",
    )
    parser.add_argument(
        "--compare", metavar="LEFT,RIGHT", help="Select initial comparison pair by key"
    )
    parser.add_argument(
        "--scan",
        type=Path,
        help="Find finished GeoTIFFs/vector files in this directory (automatic display classification)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also scan subdirectories; use --max-files to bound work",
    )
    parser.add_argument(
        "--max-files", type=int, default=250, help="Maximum scan inputs (default: 250)"
    )
    parser.add_argument(
        "--running",
        action="store_true",
        help="Producer is still publishing; viewer --watch will continue",
    )
    parser.add_argument("--status", help="Human-readable status text")
    args = parser.parse_args(argv)
    try:
        if args.max_files < 1:
            raise ValueError("--max-files must be >= 1")
        if args.water_level is not None and not math.isfinite(args.water_level):
            raise ValueError("--water-level must be a finite elevation value")
        overrides = {
            name: named_options(getattr(args, name), name)
            for name in ("label", "group", "stage", "band", "layer")
        }
        catalog = ProductCatalog(args.out, title=args.title, area=args.area)
        selected = []
        if args.scan:
            from .discovery import products_in_directory, raster_kind, VECTORS

            base = args.scan.resolve()
            for path in products_in_directory(
                base,
                recursive=args.recursive,
                max_files=args.max_files,
                exclude=args.out.parent / "viz",
            ):
                key = file_key(path.relative_to(base))
                if path.suffix.lower() in VECTORS:
                    style = "vector"
                else:
                    style, _ = raster_kind(path)
                selected.append((key, path, style))
        for name, style in (
            ("raster", "elevation"),
            ("rgb", "rgb"),
            ("difference", "difference"),
            ("scalar", "scalar"),
            ("grayscale", "grayscale"),
            ("categorical", "categorical"),
            ("vector", "vector"),
        ):
            for declaration in getattr(args, name):
                key, source = assignment(declaration, "--" + name)
                selected.append((key, Path(source), style))
        if not selected:
            raise ValueError(
                "Supply at least one --raster/--rgb/--vector or --scan directory"
            )
        keys = [key for key, _, _ in selected]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "Duplicate keys; use explicit KEY=FILE entries or scan a smaller directory"
            )
        if set().union(*(set(x) for x in overrides.values())) - set(keys):
            raise ValueError("Product option mentions an undeclared key")
        vectors = {key for key, _, style in selected if style == "vector"}
        if (set(args.visible) | set(args.hidden) | set(overrides["layer"])) - vectors:
            raise ValueError(
                "--visible/--hidden/--layer requires a declared vector key"
            )
        if set(args.visible) & set(args.hidden):
            raise ValueError("A vector cannot be both --visible and --hidden")
        if set(overrides["band"]) & vectors:
            raise ValueError("--band requires a raster key")
        rasters = []
        for key, source, style in selected:
            opts = dict(
                label=overrides["label"].get(key, source.stem.replace("_", " ")),
                stage=overrides["stage"].get(key),
                area=args.area,
            )
            if style == "vector":
                opts["layer"] = overrides["layer"].get(key)
                if key in args.visible or key in args.hidden:
                    opts["display"] = {"visible": key in args.visible}
                catalog.add_vector(key, source, **opts)
            else:
                opts["band"] = int(overrides["band"].get(key, "1"))
                opts["group"] = overrides["group"].get(
                    key, "imagery" if style == "rgb" else style
                )
                if style == "elevation" and args.water_level is not None:
                    opts["water_level"] = args.water_level
                catalog.add_raster(key, source, style=style, **opts)
                if style != "rgb":
                    rasters.append(key)
        if args.compare:
            pair = [x.strip() for x in args.compare.split(",")]
            if len(pair) != 2 or not all(pair):
                raise ValueError("--compare must be LEFT,RIGHT")
            catalog.compare(*pair)
        elif rasters:
            catalog.compare(rasters[0], rasters[1] if len(rasters) > 1 else rasters[0])
        print(catalog.publish(running=args.running, status=args.status))
        return 0
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(2, "raster-viz-catalog: " + str(exc) + "\n")


if __name__ == "__main__":
    sys.exit(main())
