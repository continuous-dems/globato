"""Generic product catalog consumption. No producer or filtering dependencies."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .review import MAX_EVENTS

import numpy as np

from .schema import product_group, validate_document


def read_catalog(path):
    """Read and validate the complete catalog before publishing any products."""
    return validate_document(json.loads(Path(path).resolve().read_text()))


def shared_limits(path, data, groups, default_water_level=0.0):
    """Resolve unconfigured palettes from ALL available raster inputs in a group.

    Keeps comparisons fair independent of product order. Uses small overviews,
    not full-resolution raster reads. As pending files arrive, cached previews
    are invalidated through their group-range fingerprint.
    """
    import rasterio
    from rasterio.enums import Resampling

    for (area, group), config in groups.items():
        if config["limits"] is not None:
            continue
        if config["style"] in {"categorical", "rgb"}:
            continue
        samples = []
        for item in data["products"]:
            if (
                item.get("type", "raster") != "raster"
                or item.get("area", "default") != area
                or product_group(item) != group
            ):
                continue
            source = Path(item["path"])
            source = source if source.is_absolute() else path.parent / source
            if not source.is_file():
                continue
            try:
                with rasterio.open(source) as ds:
                    width, height = ds.width, ds.height
                    factor = max(1, width / 320, height / 320)
                    w, h = max(1, round(width / factor)), max(1, round(height / factor))
                    values = (
                        ds.read(
                            item.get("band", 1),
                            out_shape=(h, w),
                            masked=True,
                            resampling=Resampling.average,
                        )
                        .astype(float)
                        .filled(np.nan)
                    )
                    good = values[np.isfinite(values)]
                    if good.size:
                        samples.append(good)
            except (OSError, ValueError, IndexError, rasterio.errors.RasterioError):
                # Later add_raster reports the precise per-file error.
                continue
        if samples:
            values = np.concatenate(samples)
            lo, hi = np.percentile(values, [1, 99]).tolist()
            if config["style"] == "elevation":
                level = float(
                    config["water_level"]
                    if config["water_level"] is not None
                    else default_water_level
                )
                if float(np.min(values)) < level and lo >= level:
                    lo = float(np.min(values))
                if float(np.max(values)) >= level and hi < level:
                    hi = float(np.max(values))
            if config["style"] == "difference":
                m = max(abs(lo), abs(hi), 1e-8)
                lo, hi = -m, m
            if hi <= lo:
                lo, hi = lo - 0.5, hi + 0.5
            config["limits"] = [float(lo), float(hi)]
    return groups


def shared_camera_bounds(path, data, groups):
    """Use the union of current group footprints for consistent 3D navigation.

    This is viewer-only geometry: no source file is reprojected or changed.
    Sources that are pending/invalid remain the responsibility of add_raster.
    """
    import rasterio
    from rasterio.warp import transform_bounds

    for item in data["products"]:
        if item.get("type", "raster") != "raster" or item.get("style") == "rgb":
            continue
        identity = (item.get("area", "default"), product_group(item))
        source = Path(item["path"])
        source = source if source.is_absolute() else path.parent / source
        if not source.is_file():
            continue
        try:
            with rasterio.open(source) as src:
                if src.crs is None:
                    continue
                ext = transform_bounds(
                    src.crs, "EPSG:3857", *src.bounds, densify_pts=21
                )
            if not all(np.isfinite(ext)) or ext[2] <= ext[0] or ext[3] <= ext[1]:
                continue
            config = groups[identity]
            if "camera_bounds_3857" not in config:
                config["camera_bounds_3857"] = list(ext)
            else:
                old = config["camera_bounds_3857"]
                config["camera_bounds_3857"] = [
                    min(old[0], ext[0]),
                    min(old[1], ext[1]),
                    max(old[2], ext[2]),
                    max(old[3], ext[3]),
                ]
        except (OSError, ValueError, rasterio.errors.RasterioError):
            continue
    return groups


def sync_catalog(review, path, *, force=False):
    """Publish completed file references; never generate scientific rasters.

    Catalog producers promise that a declared product is complete. Missing files
    are pending and retried. A benchmark producer can pause all raster access by
    publishing rendering_enabled=false while retaining live text updates.
    """
    path = Path(path).resolve()
    data, groups = read_catalog(path)
    enabled = data.get("rendering_enabled", True)
    if enabled:
        groups = shared_camera_bounds(
            path, data, shared_limits(path, data, groups, review.water_level)
        )
    for config in groups.values():
        if config["style"] == "elevation" and config["water_level"] is None:
            config["water_level"] = review.water_level
    owner = str(path)
    result = {
        "added": [],
        "unchanged": [],
        "pending": [],
        "failed": [],
        "rendering_paused": not enabled,
        "producer_running": data.get("running", False),
    }
    with review.lock:
        state = review._load()
        # A catalog can be refreshed for the same output directory. Replace only
        # its own transient information, not another producer's registrations.
        previous_status = state.get("catalog_status", {})
        if previous_status.get("path") == owner:
            for key in previous_status.get("metric_keys", []):
                state["metrics"].pop(key, None)
            for key in previous_status.get("table_keys", []):
                state.setdefault("tables", {}).pop(key, None)
        if enabled:
            desired_kinds = {
                (p.get("area", "default"), p["key"]): "vectors"
                if p.get("type", "raster") == "vector"
                else "products"
                for p in data["products"]
            }
            # Do not silently replace a manually registered or other-catalog
            # product with a coincidentally equal area/key.
            for (area, key), kind in desired_kinds.items():
                a = state["areas"].get(area, {})
                for collection in ("products", "vectors"):
                    record = a.get(collection, {}).get(key)
                    if (
                        record is not None
                        and record.get("metadata", {}).get("catalog_source") != owner
                    ):
                        raise ValueError(
                            f"Catalog key conflicts with existing product: {area}/{key}"
                        )
            for area, values in state["areas"].items():
                for collection in ("products", "vectors"):
                    records = values.get(collection, {})
                    for key, record in list(records.items()):
                        if (
                            record.get("metadata", {}).get("catalog_source") == owner
                            and desired_kinds.get((area, key)) != collection
                        ):
                            del records[key]
                if values.get("catalog_defaults_source") == owner:
                    values.pop("defaults", None)
                    values.pop("catalog_defaults_source", None)
                elif any(
                    k not in values.get("products", {})
                    for k in values.get("defaults", [])
                ):
                    values.pop("defaults", None)
                # Remove abandoned camera/color groups; never discard a group
                # still used by manually registered or other catalog products.
                referenced = {p["group"] for p in values.get("products", {}).values()}
                for group in list(values.get("groups", {})):
                    if group not in referenced and (area, group) not in groups:
                        del values["groups"][group]
            for (area, group), config in groups.items():
                a = state["areas"].setdefault(area, {"products": {}, "groups": {}})
                current = a["groups"].get(group)
                explicit = config["limits"] is not None
                if current is None or current["style"] != config["style"]:
                    a["groups"][group] = dict(config, catalog_declared_limits=explicit)
                else:
                    # A refreshed catalog is authoritative for its comparison
                    # palette.  Old auto or explicit limits must not pin a
                    # changed group or cause an "already fixed" error.  The
                    # product fingerprint below forces affected previews to
                    # be regenerated with this exact shared range.
                    current["limits"] = (
                        list(config["limits"]) if config["limits"] is not None else None
                    )
                    current["catalog_declared_limits"] = explicit
                    current["water_level"] = config.get("water_level")
                    if config.get("camera_bounds_3857") is None:
                        current.pop("camera_bounds_3857", None)
                    else:
                        current["camera_bounds_3857"] = list(
                            config["camera_bounds_3857"]
                        )
        state["title"] = str(data.get("title") or "Raster visualization")
        status = data.get(
            "status",
            "Waiting for products" if not data["products"] else "Products available",
        )
        state["events"] = [
            e for e in state["events"] if e.get("catalog_source") != owner
        ]
        state["events"].append(
            {
                "text": str(status),
                "status": "info",
                "epoch": time.time(),
                "catalog_source": owner,
            }
        )
        state["events"] = state["events"][-MAX_EVENTS:]
        state["metrics"].update(data.get("metrics", {}))
        state.setdefault("tables", {}).update(data.get("tables", {}))
        state["catalog_status"] = {
            "path": owner,
            "producer_running": data.get("running", False),
            "rendering_enabled": enabled,
            "metric_keys": list(data.get("metrics", {})),
            "table_keys": list(data.get("tables", {})),
        }
        review._save(state)
    if not enabled:
        return result
    for item in data["products"]:
        area, key = item.get("area", "default"), item["key"]
        raster = Path(item["path"])
        if not raster.is_absolute():
            raster = path.parent / raster
        raster = raster.resolve()
        if not raster.is_file():
            # Do not keep showing a previously published but now missing
            # source as if it were still the current run's product.
            with review.lock:
                state = review._load()
                collection = (
                    "vectors" if item.get("type", "raster") == "vector" else "products"
                )
                records = state.get("areas", {}).get(area, {}).get(collection, {})
                record = records.get(key)
                if record and record.get("metadata", {}).get("catalog_source") == owner:
                    del records[key]
                    review._save(state)
            result["pending"].append(item["path"])
            continue
        vector = item.get("type", "raster") == "vector"
        config = None if vector else groups[(area, product_group(item))]
        stat = raster.stat()
        signature = {
            "item": item,
            "path": str(raster),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "group": config,
            "renderer": {"vector_renderer_version": 7}
            if vector
            else review.render_settings(
                review.render_3d and item.get("style", "elevation") == "elevation"
            ),
        }
        fingerprint = hashlib.sha256(
            json.dumps(signature, sort_keys=True).encode()
        ).hexdigest()
        with review.lock:
            record = (
                review._load()["areas"]
                .get(area, {})
                .get("vectors" if vector else "products", {})
                .get(key)
            )
        assets = record is not None and (
            vector or (review.output / record["image"]).is_file()
        )
        if assets and record.get("terrain_scene"):
            assets = (review.output / record["terrain_scene"]["script"]).is_file()
        if (
            not force
            and assets
            and record.get("metadata", {}).get("catalog_fingerprint") == fingerprint
        ):
            result["unchanged"].append(key)
            continue
        attributes = {
            k: v
            for k, v in item.items()
            if k not in {"key", "path", "type", "method", "layer"}
        }
        if not vector:
            attributes.setdefault("group", product_group(item))
        metadata = dict(
            attributes.pop("metadata", {}),
            catalog_source=owner,
            catalog_fingerprint=fingerprint,
        )
        metadata.update(
            {k: item[k] for k in ("method", "layer") if k in item}
        )  # opaque legacy metadata
        attributes["metadata"] = metadata
        try:
            if vector:
                attributes = {
                    k: v
                    for k, v in attributes.items()
                    if k in {"label", "area", "stage", "metadata", "display"}
                }
                from .vectors import vector_layer

                attributes["layer"] = vector_layer(item)
                review.add_vector(key, raster, **attributes)
            else:
                if config["limits"] is not None:
                    attributes["limits"] = config["limits"]
                if item.get("style", "elevation") == "elevation":
                    attributes["water_level"] = config["water_level"]
                if item.get("style", "elevation") != "elevation":
                    attributes.setdefault("views", ("2d",))
                review.add_raster(key, raster, **attributes)
            result["added"].append(key)
        except Exception as exc:
            # Do not present a preview from the previous version of this source
            # as if it were the new declared product after a failed refresh.
            with review.lock:
                state = review._load()
                collection = "vectors" if vector else "products"
                records = state.get("areas", {}).get(area, {}).get(collection, {})
                old = records.get(key)
                if old and old.get("metadata", {}).get("catalog_source") == owner:
                    del records[key]
                    review._save(state)
            result["failed"].append({"key": key, "reason": str(exc)})
    with review.lock:
        state = review._load()
        for area, pair in data.get("defaults", {}).items():
            a = state["areas"].get(area, {})
            if all(k in a.get("products", {}) for k in pair):
                a["defaults"] = pair
                a["catalog_defaults_source"] = owner
        state["catalog_result"] = result
        review._save(state)
    return result


def watch_catalog(review, path, *, interval=15, force=False, stop=None, on_scan=None):
    """Watch an atomic producer catalog, including an initially empty/missing one."""
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("Watch interval must be positive")
    review.start()
    previous_error = None
    try:
        while not (stop and stop()):
            if not review._load()["running"]:
                break
            try:
                result = sync_catalog(review, path, force=force)
                if not result["rendering_paused"]:
                    force = False
                previous_error = None
                if on_scan:
                    on_scan(result)
                # A finished producer is authoritative. Report missing/failed
                # products and return, instead of waiting forever on a bad path.
                if not result["producer_running"]:
                    return result
            except (OSError, ValueError) as exc:
                message = str(exc)
                if message != previous_error:
                    review.event(
                        "Catalog awaiting a valid update: " + message, status="pending"
                    )
                    previous_error = message
            deadline = time.monotonic() + interval
            while time.monotonic() < deadline and not (stop and stop()):
                time.sleep(min(0.25, max(0, deadline - time.monotonic())))
    finally:
        review.finish()
