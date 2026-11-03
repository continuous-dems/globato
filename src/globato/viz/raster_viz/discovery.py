"""Shallow, selectable geospatial discovery; no producer-stage assumptions."""

from __future__ import annotations
import fnmatch
import hashlib
import logging
import os
import re
import time
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling

logger = logging.getLogger(__name__)
RASTERS = {".tif", ".tiff", ".vrt"}
VECTORS = {".gpkg", ".geojson", ".shp"}
SUPPORTED = RASTERS | VECTORS
DEFAULT_EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".cache",
    "tmp",
    "temp",
    "stack",
    "input_sources_temp_masks",
}


class DiscoveryLimitError(ValueError):
    pass


def readable_label(path):
    return path.stem.replace("_", " ").replace("-", " ").strip() or path.name


def raster_kind(path, band=1):
    """Metadata/sample/name hints, explicitly overridable; never a science claim."""
    path = Path(path)
    with rasterio.open(path) as src:
        if not src.crs:
            raise ValueError("Missing CRS")
        if not 1 <= band <= src.count:
            raise ValueError("Band out of range")
        tags = {**src.tags(), **src.tags(band)}
        # Generic provenance JSON often mentions source IDs/masks even for a DEM.
        # Only semantic tags and the band/file label are classification hints.
        semantic = [
            tags.get(k, "")
            for k in [
                "description",
                "DESCRIPTION",
                "long_name",
                "standard_name",
                "BANDNAME",
                "ROLE",
            ]
        ]
        description = " ".join(
            str(x) for x in [path.stem, src.descriptions[band - 1] or "", *semantic]
        ).lower()
        note = {
            "raster_dtype": src.dtypes[band - 1],
            "classification_note": "Visualization heuristic; override with --style, add or a manifest when needed",
        }
        if any(str(v).upper() == "FUSION_STATE" for v in tags.values()):
            return "scalar", {
                **note,
                "type_inference": "additive_statistics_not_final_elevation",
                "terrain_eligible": False,
            }
        if "hillshade" in description or path.stem.lower().endswith(("_hs", "_shade")):
            return "grayscale", {
                **note,
                "type_inference": "hillshade_hint",
                "terrain_eligible": False,
            }
        categorical_hint = any(
            x in description
            for x in [
                "provenance",
                "source_id",
                "source id",
                "source_index",
                "source index",
                "rejected",
                "_mask",
                "mask_",
                "_flags",
            ]
        )
        if categorical_hint:
            return "categorical", {
                **note,
                "type_inference": "mask_or_provenance_hint",
                "terrain_eligible": False,
            }
        if src.count >= 3 and (
            any(
                w in description
                for w in ("rgb", "truecolor", "true_color", "sentinel", "s2_")
            )
            or (
                src.colorinterp[:3]
                == (
                    rasterio.enums.ColorInterp.red,
                    rasterio.enums.ColorInterp.green,
                    rasterio.enums.ColorInterp.blue,
                )
            )
        ):
            return "rgb", {
                **note,
                "type_inference": "georeferenced_rgb_hint",
                "terrain_eligible": False,
            }
        if path.suffix.lower() == ".vrt" and path.stem.lower().endswith("_sources"):
            return "scalar", {
                **note,
                "type_inference": "ambiguous_sources_container",
                "terrain_eligible": False,
                "band_note": "Could be elevation inputs or identifiers; producer/manifest must confirm semantics",
            }
        # A height difference is not absolute terrain. Recognize only isolated
        # filename tokens, never generic substrings (e.g. 'changeover').
        # Producers must still declare the intended style for ambiguous names.
        if re.search(
            r"(^|[_\-. ])(difference|diff|delta)(?=$|[_\-. ])", path.stem.lower()
        ):
            return "difference", {
                **note,
                "type_inference": "filename_elevation_difference_hint",
                "terrain_eligible": False,
            }
        sample = src.read(
            band,
            masked=True,
            out_shape=(min(src.height, 192), min(src.width, 192)),
            resampling=Resampling.nearest,
        )
        values = np.ma.compressed(sample)
        unique = np.unique(values[np.isfinite(values)])
        integer = np.issubdtype(np.dtype(src.dtypes[band - 1]), np.integer)
        # Floating-point elevation can be flat at 0/1; do not interpret this
        # as a mask solely because a small geographic sample is constant.
        if (
            unique.size
            and integer
            and (np.all(np.isin(unique, [0, 1])) or unique.size <= 16)
        ):
            return "categorical", {
                **note,
                "type_inference": "sampled_discrete_values",
                "sample_values": unique.tolist(),
                "terrain_eligible": False,
            }
        if src.count in {3, 4} or src.dtypes[band - 1] == "uint8":
            return "grayscale", {
                **note,
                "type_inference": "byte_or_multiband_image_preview",
                "terrain_eligible": False,
                "band_note": "First/selected band only; explicit manifest controls interpretation",
            }
        if any(
            x in description
            for x in [
                "uncertainty",
                "standard_deviation",
                "stddev",
                "variance",
                "_count",
                "_density",
            ]
        ):
            return "scalar", {
                **note,
                "type_inference": "auxiliary_scalar_hint",
                "terrain_eligible": False,
            }
        return "elevation", {
            **note,
            "type_inference": "continuous_raster_candidate",
            "terrain_eligible": True,
        }


def matches(relative, patterns):
    return any(
        fnmatch.fnmatchcase(relative, p) or fnmatch.fnmatchcase(Path(relative).name, p)
        for p in patterns
    )


def products_in_directory(
    root,
    *,
    recursive=False,
    max_files=250,
    exclude=None,
    includes=(),
    excludes=(),
    exclude_processing=True,
    extensions=None,
):
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if max_files is not None and max_files < 0:
        raise ValueError("max_files must be >=0 (0 means no explicit limit)")
    excluded = Path(exclude).resolve() if exclude else None
    selected_extensions = (
        SUPPORTED if extensions is None else set(extensions) & SUPPORTED
    )

    def blocked(path):
        resolved = path.resolve()
        return (
            excluded is not None
            and (resolved == excluded or excluded in resolved.parents)
        ) or matches(str(path.relative_to(root)), excludes)

    found = []
    for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
        dirs[:] = (
            sorted(
                d
                for d in dirs
                if not (Path(current) / d).is_symlink()
                and not blocked(Path(current) / d)
                and not (exclude_processing and d.lower() in DEFAULT_EXCLUDED_DIRS)
            )
            if recursive
            else []
        )
        for name in sorted(files):
            item = Path(current) / name
            if (
                item.suffix.lower() not in selected_extensions
                or item.is_symlink()
                or blocked(item)
            ):
                continue
            relative = str(item.relative_to(root))
            if includes and not matches(relative, includes):
                continue
            found.append(item.resolve())
            if max_files and len(found) > max_files:
                raise DiscoveryLimitError(
                    f"More than {max_files} selected geospatial files; no rendering started. Discovery is shallow by default. Narrow --include/--exclude, use --list, raise --max-files, or explicitly use --max-files 0 for no limit."
                )
    return sorted(found, key=lambda p: str(p.relative_to(root)).lower())


def product_id(root, path):
    # Namespace roots too, so adding another directory never replaces same-named
    # products from an earlier scan. v4 keys are migrated when source path matches.
    return (
        "auto_"
        + hashlib.sha256(
            (
                str(Path(root).resolve())
                + "\0"
                + str(Path(path).resolve().relative_to(Path(root).resolve()))
            ).encode()
        ).hexdigest()[:20]
    )


def catalog(root, paths, style=None):
    rows = []
    root = Path(root).resolve()
    for path in paths:
        try:
            kind, inference = (
                raster_kind(path)
                if path.suffix.lower() in RASTERS
                else ("vector", {"terrain_eligible": False})
            )
            if style and kind != "vector":
                kind = style
                inference["terrain_eligible"] = kind in {"elevation", "difference"}
            rows.append(
                {
                    "key": product_id(root, path),
                    "path": str(path.relative_to(root)),
                    "style": kind,
                    "3d_eligible": inference.get("terrain_eligible", False),
                    "reason": inference.get("type_inference", "vector"),
                }
            )
        except Exception as exc:
            rows.append({"path": str(path.relative_to(root)), "error": str(exc)})
    return rows


def discover_into(
    review,
    root,
    *,
    recursive=False,
    max_files=250,
    area="default",
    skip_existing=True,
    include_vectors=True,
    includes=(),
    excludes=(),
    exclude_processing=True,
    style=None,
    ready_signatures=None,
    extensions=None,
):
    root = Path(root).resolve()
    found = products_in_directory(
        root,
        recursive=recursive,
        max_files=max_files,
        exclude=review.output,
        includes=includes,
        excludes=excludes,
        exclude_processing=exclude_processing,
        extensions=extensions,
    )
    added = []
    unchanged = []
    skipped = []
    pending = []
    removed = []
    # Directory mode is an inventory of current files, not an append-only log.
    # Prune only entries owned by a scan of this root, leaving manually added
    # items and other discovery roots untouched. Old inventories predate the
    # discovery_root field: use their actual source path for compatibility.
    present = {str(f) for f in found}
    with review.lock:
        state = review._load()
        area_state = state.get("areas", {}).get(area, {})
        for collection in ("products", "vectors"):
            records = area_state.get(collection, {})
            for key, record in list(records.items()):
                meta = record.get("metadata", {})
                if meta.get("discovery") != "directory_scan":
                    continue
                original = record.get("source", {}).get("path")
                if not original:
                    continue
                source = Path(original).resolve()
                source_root = meta.get("discovery_root")
                owned = (
                    (source_root == str(root))
                    if source_root
                    else source.is_relative_to(root)
                )
                if owned and str(source) not in present:
                    del records[key]
                    removed.append(str(source))
        if removed:
            keys = area_state.get("products", {})
            if any(k not in keys for k in area_state.get("defaults", [])):
                area_state.pop("defaults", None)
            groups_in_use = {v["group"] for v in keys.values()}
            for group in list(area_state.get("groups", {})):
                if group not in groups_in_use:
                    del area_state["groups"][group]
            review._save(state)
    for path in found:
        if not include_vectors and path.suffix.lower() in VECTORS:
            continue
        key = product_id(root, path)
        stat = path.stat()
        signature = (str(path), stat.st_size, stat.st_mtime_ns)
        if (
            ready_signatures is not None
            and ready_signatures.get(str(path)) != signature
        ):
            pending.append(path)
            continue
        try:
            inferred, inference = (
                raster_kind(path) if path.suffix.lower() in RASTERS else ("vector", {})
            )
            if style and inferred != "vector":
                inferred = style
            with review.lock:
                data = review._load()
                a = data.get("areas", {}).get(area, {})
                collection = (
                    a.get("products", {})
                    if path.suffix.lower() in RASTERS
                    else a.get("vectors", {})
                )
                existing = next(
                    (
                        (k, v)
                        for k, v in collection.items()
                        if v.get("metadata", {}).get("discovery") == "directory_scan"
                        and v.get("source", {}).get("path") == str(path)
                    ),
                    None,
                )
                if existing:
                    key, record = existing
                else:
                    record = None
            source = record.get("source", {}) if record else {}
            settings = (
                review.render_settings(
                    review.render_3d and record.get("style") == "elevation"
                )
                if record and path.suffix.lower() in RASTERS
                else {"vector_renderer_version": 5}
            )
            assets = (
                not record
                or path.suffix.lower() in VECTORS
                or (review.output / record["image"]).is_file()
            )
            if record and record.get("terrain_scene"):
                assets = (
                    assets
                    and (review.output / record["terrain_scene"]["script"]).is_file()
                )
            if (
                skip_existing
                and record
                and assets
                and (source.get("path"), source.get("size"), source.get("mtime_ns"))
                == signature
                and record.get("render_settings") == settings
                and (inferred == "vector" or record.get("style") == inferred)
            ):
                unchanged.append(path)
                continue
            meta = {
                "discovery": "directory_scan",
                "discovery_root": str(root),
                "relative_path": str(path.relative_to(root)),
                "stage_inference": "none; producer supplies stage",
            }
            if path.suffix.lower() in RASTERS:
                meta.update(inference)
                meta["explicit_style_override"] = style
                review.add_raster(
                    key,
                    path,
                    area=area,
                    label=readable_label(path),
                    group="discovered_" + inferred,
                    stage="Unspecified",
                    style=inferred,
                    views=("2d",) if inferred != "elevation" else None,
                    metadata=meta,
                )
            else:
                review.add_vector(
                    key,
                    path,
                    area=area,
                    label=readable_label(path),
                    stage="Unspecified",
                    metadata=meta,
                )
            added.append(path)
        except Exception as exc:
            skipped.append((path, str(exc)))
            logger.warning("Skipping %s: %s", path, exc)
            review.event(f"Product skipped: {path.name}: {exc}", status="skipped")
    with review.lock:
        data = review._load()
        a = data.get("areas", {}).get(area, {})
        if a and not a.get("defaults"):
            groups = {}
            for k, v in a.get("products", {}).items():
                groups.setdefault(v["group"], []).append(k)
            pair = next((keys[:2] for keys in groups.values() if len(keys) >= 2), None)
            if pair:
                a["defaults"] = pair
                review._save(data)
    return {
        "removed": removed,
        "added": [str(p.relative_to(root)) for p in added],
        "unchanged": [str(p.relative_to(root)) for p in unchanged],
        "skipped": [
            {"path": str(p.relative_to(root)), "reason": why} for p, why in skipped
        ],
        "pending": [str(p.relative_to(root)) for p in pending],
    }


def watch_into(review, root, *, interval=15, stop=None, on_scan=None, **options):
    """Rescan until Ctrl-C, finish(), or stop(). Require stable source signatures."""
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("Watch interval must be positive")
    previous = {}
    review.start()
    try:
        while not (stop and stop()):
            if not review._load()["running"]:
                break
            paths = products_in_directory(
                root,
                exclude=review.output,
                **{
                    k: v
                    for k, v in options.items()
                    if k
                    in {
                        "recursive",
                        "max_files",
                        "includes",
                        "excludes",
                        "exclude_processing",
                        "extensions",
                    }
                },
            )
            current = {
                str(p): (str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths
            }
            result = discover_into(review, root, ready_signatures=previous, **options)
            if on_scan:
                on_scan(result)
            # --force applies to the initial stable scan only. Otherwise a
            # static directory is rerendered repeatedly at every watch tick.
            options["skip_existing"] = True
            previous = current
            deadline = time.monotonic() + interval
            while time.monotonic() < deadline and not (stop and stop()):
                time.sleep(min(0.25, max(0, deadline - time.monotonic())))
    finally:
        review.finish()
