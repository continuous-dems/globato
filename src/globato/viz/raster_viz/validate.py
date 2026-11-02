"""Non-rendering schema-v1 and GeoTIFF preflight for independent producers.

No Globato/Fetchez imports, raster mutations, interpolations, or color rendering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .manifest import read_catalog

_RASTER_SUFFIXES = {".tif", ".tiff", ".vrt"}
_VECTOR_SUFFIXES = {".gpkg", ".geojson", ".shp"}


def check_catalog(path, *, allow_pending=False):
    """Return a machine-readable result; errors prevent reliable visualization."""
    path = Path(path).expanduser().resolve()
    result = {
        "catalog": str(path),
        "products": 0,
        "errors": [],
        "warnings": [],
        "raster_products": 0,
        "vector_products": 0,
    }
    try:
        data, groups = read_catalog(path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        result["errors"].append("Invalid schema-v1 catalog: " + str(exc))
        result["ok"] = False
        return result
    result["products"] = len(data["products"])
    import rasterio

    for item in data["products"]:
        key = f"{item.get('area', 'default')}/{item['key']}"
        source = Path(item["path"]).expanduser()
        if not source.is_absolute():
            source = path.parent / source
        if not source.is_file():
            message = f"{key}: missing source: {source}"
            (
                result["warnings"]
                if allow_pending and data.get("running")
                else result["errors"]
            ).append(message)
            continue
        vector = item.get("type", "raster") == "vector"
        if vector:
            result["vector_products"] += 1
            if source.suffix.lower() not in _VECTOR_SUFFIXES:
                result["errors"].append(
                    f"{key}: unsupported vector extension: {source.suffix}"
                )
            if source.suffix.lower() == ".shp":
                for suffix in (".shx", ".dbf"):
                    if not source.with_suffix(suffix).is_file():
                        result["errors"].append(
                            f"{key}: shapefile missing {suffix} companion"
                        )
            try:
                from .vectors import inspect_vector, vector_layer

                crs, count = inspect_vector(source, layer=vector_layer(item))
                if not crs:
                    result["errors"].append(f"{key}: vector lacks a georeferenced CRS")
                if count == 0:
                    result["warnings"].append(f"{key}: vector layer is empty")
                if count > 150:
                    result["errors"].append(
                        f"{key}: vector exceeds viewer limit of 150 features; select/simplify first"
                    )
            except Exception as exc:
                result["errors"].append(f"{key}: cannot read vector layer: {exc}")
            continue
        result["raster_products"] += 1
        if source.suffix.lower() not in _RASTER_SUFFIXES:
            result["errors"].append(
                f"{key}: raster must be a GeoTIFF or VRT, not {source.suffix}"
            )
            continue
        try:
            with rasterio.open(source) as ds:
                if ds.width <= 0 or ds.height <= 0 or ds.count < 1:
                    result["errors"].append(
                        f"{key}: raster has empty dimensions or bands"
                    )
                if ds.crs is None:
                    result["errors"].append(f"{key}: raster lacks a georeferenced CRS")
                if item.get("band", 1) > ds.count:
                    result["errors"].append(
                        f"{key}: band {item['band']} exceeds source band count {ds.count}"
                    )
                if item.get("style") == "rgb" and ds.count < 3:
                    result["errors"].append(f"{key}: RGB requires three source bands")
                if ds.tags().get("GLOBATO_DATATYPE") == "FUSION_STATE":
                    result["errors"].append(
                        f"{key}: native FusionState is not a display elevation; finalize it first"
                    )
        except (OSError, rasterio.errors.RasterioError, ValueError) as exc:
            result["errors"].append(f"{key}: cannot read georeferenced raster: {exc}")
    result["areas"] = sorted({p.get("area", "default") for p in data["products"]})
    result["comparison_groups"] = len(groups)
    result["ok"] = not result["errors"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="raster-viz-check",
        description="Validate a producer products.json and its referenced geospatial files without rendering",
    )
    parser.add_argument("catalog", type=Path, help="Path to products.json")
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Missing inputs in a running catalog are warnings; completed catalogs always require all files",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON report"
    )
    args = parser.parse_args(argv)
    report = check_catalog(args.catalog, allow_pending=args.allow_pending)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for key in (
            "products",
            "raster_products",
            "vector_products",
            "comparison_groups",
        ):
            print(f"{key}: {report.get(key, 0)}")
        for level in ("errors", "warnings"):
            for message in report[level]:
                print(f"{level[:-1].upper()}: {message}")
        print("VALID" if report["ok"] else "INVALID", report["catalog"])
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
