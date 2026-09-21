#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""globato.hooks.sinks.side_stack

Per-entry FusionState cache for Globato.

A side-stack stores the exact additive FusionState produced after an entry has
passed through its point-processing pipeline. Cache hits bypass raw source
reading and point processing and provide the cached FusionState directly as a
raster stream for ``multi_stack``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import rasterio
from rasterio.crs import CRS

from fetchez.hooks import FetchHook
from fetchez.utils import float_or, str2bool, str2inc

from globato.hooks.transforms.point_pixels import (
    FUSION_BAND_MAP,
    FUSION_BANDS,
    FUSION_STATE_VERSION,
    Point2PixelStream,
)

logger = logging.getLogger(__name__)

SIDESTACK_DATATYPE = "FUSION_STATE"
SIDESTACK_HASH_TAG = "GLOBATO_SIDESTACK_HASH"
SIDESTACK_IDENTITY_TAG = "GLOBATO_SIDESTACK_IDENTITY"

NON_FUSION_HOOKS = {
    "side_stack_check",
    "side-stack-check",
    "side_stack",
    "side-stack",
    "stream-init",
    "stream_init",
    "stream_data",
    "multi_stack",
    "multi-stack",
    "provenance",
    "source-masks",
    "source_masks",
    "ms_binary_cudem",
    "format_cog",
    "format-cog",
    "copy_artifact",
    "raster_metadata",
    "viz_geoshade",
}

VOLATILE_HOOK_FIELDS = {
    "logger",
    "results",
    "stream",
    "raster_stream",
    "session",
    "dataset",
    "_accumulator",
}


def _stable_value(value: Any) -> Any:
    """Convert configuration values to deterministic JSON-compatible values."""

    if value is None or isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        return value if np.isfinite(value) else repr(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, CRS):
        return value.to_wkt()

    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }

    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]

    if isinstance(value, set):
        return sorted(_stable_value(item) for item in value)

    if hasattr(value, "export_as_list"):
        try:
            return _stable_value(value.export_as_list())
        except Exception:
            pass

    if hasattr(value, "format"):
        try:
            return str(value.format("fn"))
        except Exception:
            pass

    return str(value)


def _hook_name(hook: Any) -> str:
    if isinstance(hook, Mapping):
        return str(hook.get("name", ""))
    return str(getattr(hook, "name", hook.__class__.__name__))


def _hook_affects_fusion_state(hook: Any) -> bool:
    """Return whether a hook should participate in side-stack invalidation."""

    if isinstance(hook, Mapping):
        explicit = hook.get("affects_fusion_state")
    else:
        explicit = getattr(hook, "affects_fusion_state", None)

    if explicit is not None:
        return bool(explicit)

    return _hook_name(hook) not in NON_FUSION_HOOKS


def _hook_identity(hook: Any) -> dict[str, Any]:
    """Return deterministic configuration identity for one upstream hook."""

    name = _hook_name(hook)

    if isinstance(hook, Mapping):
        config = {
            str(key): value
            for key, value in hook.items()
            if key not in {"name", "affects_fusion_state"}
        }
        return {"name": name, "config": _stable_value(config)}

    explicit = getattr(hook, "sidestack_identity", None)
    if callable(explicit):
        explicit = explicit()
    if explicit is not None:
        return {"name": name, "config": _stable_value(explicit)}

    config = {}
    for key, value in vars(hook).items():
        if key.startswith("_") or key in VOLATILE_HOOK_FIELDS:
            continue
        config[key] = value

    return {"name": name, "config": _stable_value(config)}


def processing_identity(mod: Any, entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Describe ordered processing that can change the cached FusionState."""

    module_hooks = list(getattr(mod, "hooks", []) or [])
    entry_hooks = list(entry.get("hooks", []) or [])

    hooks = [
        _hook_identity(hook)
        for hook in module_hooks + entry_hooks
        if _hook_affects_fusion_state(hook)
    ]

    hooks.append(
        {
            "name": "__fusion_context__",
            "config": _stable_value(
                {
                    "module_weight": getattr(mod, "weight", None),
                    "module_uncertainty": getattr(mod, "uncertainty", None),
                    "entry_weight": entry.get("weight"),
                    "entry_uncertainty": entry.get("uncertainty"),
                }
            ),
        }
    )

    return hooks


def source_identity(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return the best available identity for the underlying source."""

    checksum = entry.get("checksum")
    if checksum:
        return {"checksum": str(checksum)}

    identity: dict[str, Any] = {}

    for key in ("url", "etag", "e_tag", "last_modified", "version", "revision"):
        value = entry.get(key)
        if value not in (None, ""):
            identity[key] = _stable_value(value)

    dst_fn = entry.get("dst_fn")
    if dst_fn:
        path = Path(dst_fn)
        if path.exists():
            stat = path.stat()
            identity.setdefault("basename", path.name)
            identity.setdefault("size", stat.st_size)
            identity.setdefault("mtime_ns", stat.st_mtime_ns)
        elif not identity:
            identity["basename"] = path.name

    if not identity:
        identity["source"] = "unknown"

    return identity


def _resolve_grid(region: Any, res: Any, crs: Any) -> dict[str, Any]:
    """Resolve the exact raster grid used by Point2PixelStream."""

    inc = float_or(str2inc(res))
    if inc is None:
        raise ValueError(f"Unable to resolve side-stack resolution {res!r}")

    xcount, ycount, gt = region.geo_transform(
        x_inc=inc,
        y_inc=inc,
        node="pixel",
    )

    transform = rasterio.transform.from_origin(
        gt[0],
        gt[3],
        gt[1],
        abs(gt[5]),
    )

    canonical_crs = CRS.from_user_input(crs).to_wkt() if crs else None

    return {
        "width": int(xcount),
        "height": int(ycount),
        "transform": [
            float(transform.a),
            float(transform.b),
            float(transform.c),
            float(transform.d),
            float(transform.e),
            float(transform.f),
        ],
        "crs": canonical_crs,
        "resolution": [abs(float(transform.a)), abs(float(transform.e))],
    }


@dataclass(frozen=True)
class SideStackIdentity:
    """Canonical identity of one cached per-entry FusionState."""

    source: Mapping[str, Any]
    processing: tuple[Mapping[str, Any], ...]
    grid: Mapping[str, Any]
    fusion_version: int = FUSION_STATE_VERSION

    @classmethod
    def from_entry(
        cls,
        mod: Any,
        entry: Mapping[str, Any],
        *,
        region: Any,
        res: Any,
        crs: Any,
    ) -> "SideStackIdentity":
        return cls(
            source=source_identity(entry),
            processing=tuple(processing_identity(mod, entry)),
            grid=_resolve_grid(region, res, crs),
            fusion_version=FUSION_STATE_VERSION,
        )

    def canonical(self) -> dict[str, Any]:
        return {
            "source": _stable_value(self.source),
            "processing": _stable_value(self.processing),
            "grid": _stable_value(self.grid),
            "fusion": {
                "version": self.fusion_version,
                "bands": list(FUSION_BANDS),
            },
        }

    def json(self) -> str:
        return json.dumps(
            self.canonical(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    def digest(self) -> str:
        return hashlib.sha256(self.json().encode("utf-8")).hexdigest()


def _safe_basename(entry: Mapping[str, Any]) -> str:
    original = entry.get("_sidestack_source_dst_fn") or entry.get("dst_fn") or "stream"
    base = os.path.splitext(os.path.basename(str(original)))[0] or "stream"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)


def sidestack_path(
    entry: Mapping[str, Any],
    cache_dir: str | os.PathLike[str] | None,
    digest: str,
) -> str:
    base_dir = Path(cache_dir or os.path.dirname(str(entry.get("dst_fn", "."))) or ".")
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir / f"{_safe_basename(entry)}_{digest[:16]}_sidestack.tif")


def _identity_transform(identity: SideStackIdentity):
    a, b, c, d, e, f = identity.grid["transform"]
    return rasterio.Affine(a, b, c, d, e, f)


def validate_sidestack(
    path: str,
    identity: SideStackIdentity,
    digest: str,
) -> tuple[bool, str | None]:
    """Validate both the FusionState schema and the exact cache identity."""

    if not os.path.exists(path):
        return False, "missing"

    try:
        with rasterio.open(path) as src:
            tags = src.tags()

            if tags.get("GLOBATO_DATATYPE") != SIDESTACK_DATATYPE:
                return False, "not a FusionState"

            if int(tags.get("GLOBATO_FUSION_VERSION", -1)) != FUSION_STATE_VERSION:
                return False, "FusionState version mismatch"

            if tags.get(SIDESTACK_HASH_TAG) != digest:
                return False, "identity hash mismatch"

            if tuple(src.descriptions) != tuple(FUSION_BANDS):
                return False, "band schema mismatch"

            if src.count != len(FUSION_BANDS):
                return False, "band count mismatch"

            if (
                src.width != identity.grid["width"]
                or src.height != identity.grid["height"]
            ):
                return False, "grid dimensions mismatch"

            if src.transform != _identity_transform(identity):
                return False, "grid transform mismatch"

            expected_crs = (
                CRS.from_wkt(identity.grid["crs"]) if identity.grid["crs"] else None
            )
            if src.crs != expected_crs:
                return False, "CRS mismatch"

            stored_identity = tags.get(SIDESTACK_IDENTITY_TAG)
            if stored_identity and stored_identity != identity.json():
                return False, "identity manifest mismatch"

    except (OSError, ValueError, rasterio.errors.RasterioError) as exc:
        return False, f"unreadable cache: {exc}"

    return True, None


def read_fusion_state_stream(path: str):
    """Yield a cached FusionState TIFF as a standard raster stream."""

    with rasterio.open(path, "r") as src:
        profile = src.profile.copy()
        yield profile

        for _, window in src.block_windows(1):
            data = src.read(window=window).astype(np.float64, copy=False)
            transform = src.window_transform(window)
            yield window, window, data, None, transform


class SideStackCheck(FetchHook):
    """Manifest-stage side-stack lookup.

    A valid hit installs a lazy FusionState raster stream directly on the entry.
    The original ``dst_fn`` is retained for source identity/provenance.
    """

    name = "side_stack_check"
    meta_stage = "manifest"

    def __init__(self, res="1s", crs="EPSG:4326", cache_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.res = res
        self.crs = crs
        self.cache_dir = cache_dir

    def run(self, entries):
        for mod, entry in entries:
            region = getattr(mod, "region", None)
            if not region:
                continue

            entry.setdefault("_sidestack_source_dst_fn", entry.get("dst_fn"))

            cache_dir = self.cache_dir or getattr(
                mod,
                "_outdir",
                getattr(mod, "outdir", None),
            )

            identity = SideStackIdentity.from_entry(
                mod,
                entry,
                region=region,
                res=self.res,
                crs=self.crs,
            )
            digest = identity.digest()
            cache_path = sidestack_path(entry, cache_dir, digest)

            entry["_sidestack_hash"] = digest
            entry["_sidestack_identity"] = identity.json()
            entry["_sidestack_path"] = cache_path

            valid, reason = validate_sidestack(cache_path, identity, digest)

            if valid:
                logger.debug("Side-stack hit: %s", os.path.basename(cache_path))
                entry["_sidestack_hit"] = True
                entry["stream"] = read_fusion_state_stream(cache_path)
                entry["stream_type"] = "raster-stream"
                entry["data_type"] = "fusion-state"
                entry.setdefault("artifacts", {})["side-stack"] = cache_path
            else:
                entry["_sidestack_hit"] = False
                if os.path.exists(cache_path):
                    logger.debug(
                        "Ignoring invalid side-stack %s (%s)",
                        os.path.basename(cache_path),
                        reason,
                    )

        return entries


class SideStackGenerate(FetchHook):
    """Convert cache misses to FusionState, persist them, and pass state onward."""

    name = "side_stack"
    meta_stage = "stream"

    def __init__(
        self,
        res="1s",
        crs="EPSG:4326",
        cache_dir=None,
        compress=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.res = res
        self.crs = crs
        self.cache_dir = cache_dir
        self.compress = str2bool(compress)

    @staticmethod
    def _creation_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {
            "driver",
            "dtype",
            "nodata",
            "width",
            "height",
            "count",
            "crs",
            "transform",
        }

        out = {key: profile[key] for key in allowed if key in profile}
        out.update(
            tiled=True,
            blockxsize=256,
            blockysize=256,
            bigtiff="YES",
            dtype="float64",
            nodata=None,
        )
        return out

    def _cache_stream(
        self,
        raster_stream,
        cache_path: str,
        *,
        identity: SideStackIdentity,
        digest: str,
    ):
        """Write FusionState atomically while yielding identical chunks onward."""

        profile = next(raster_stream)
        yield profile

        cache_dir = os.path.dirname(os.path.abspath(cache_path))
        os.makedirs(cache_dir, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(cache_path)}.",
            suffix=".tmp.tif",
            dir=cache_dir,
        )
        os.close(fd)
        os.remove(tmp_path)

        creation_profile = self._creation_profile(profile)
        if self.compress:
            creation_profile["compress"] = "lzw"
            creation_profile["predictor"] = 3

        completed = False
        try:
            with rasterio.open(tmp_path, "w", **creation_profile) as dst:
                for name, idx in FUSION_BAND_MAP.items():
                    dst.set_band_description(idx, name)

                dst.update_tags(
                    GLOBATO_DATATYPE=SIDESTACK_DATATYPE,
                    GLOBATO_FUSION_VERSION=str(FUSION_STATE_VERSION),
                    **{
                        SIDESTACK_HASH_TAG: digest,
                        SIDESTACK_IDENTITY_TAG: identity.json(),
                    },
                )

                for chunk in raster_stream:
                    window, buff_win, data, ndv, transform = chunk

                    data = np.asarray(data, dtype=np.float64)
                    if data.ndim != 3 or data.shape[0] != len(FUSION_BANDS):
                        raise ValueError(
                            "Side-stack expected FusionState chunk with "
                            f"{len(FUSION_BANDS)} bands; got {data.shape!r}"
                        )

                    dst.write(data, window=window)
                    yield window, buff_win, data, ndv, transform

            os.replace(tmp_path, cache_path)
            completed = True
            logger.debug("Side-stack cached: %s", os.path.basename(cache_path))

        finally:
            if not completed and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _identity_for_entry(self, mod, entry, region):
        cached_json = entry.get("_sidestack_identity")
        if cached_json:
            payload = json.loads(cached_json)
            return SideStackIdentity(
                source=payload["source"],
                processing=tuple(payload["processing"]),
                grid=payload["grid"],
                fusion_version=int(payload["fusion"]["version"]),
            )

        return SideStackIdentity.from_entry(
            mod,
            entry,
            region=region,
            res=self.res,
            crs=self.crs,
        )

    def run(self, entries):
        point_to_state = Point2PixelStream(
            x_inc=self.res,
            y_inc=self.res,
        )

        for mod, entry in entries:
            if entry.get("_sidestack_hit"):
                continue

            region = getattr(mod, "region", None)
            if not region or not self.is_point_stream(entry):
                continue

            cache_dir = self.cache_dir or getattr(
                mod,
                "_outdir",
                getattr(mod, "outdir", None),
            )

            identity = self._identity_for_entry(mod, entry, region)
            digest = entry.get("_sidestack_hash") or identity.digest()
            cache_path = entry.get("_sidestack_path") or sidestack_path(
                entry,
                cache_dir,
                digest,
            )

            logger.debug(
                "Side-stack miss; generating %s",
                os.path.basename(cache_path),
            )

            fusion_stream = point_to_state._stream_wrapper(
                entry["stream"],
                entry=entry,
                region=region,
            )

            entry["stream"] = self._cache_stream(
                fusion_stream,
                cache_path,
                identity=identity,
                digest=digest,
            )
            entry["stream_type"] = "raster-stream"
            entry["data_type"] = "fusion-state"
            entry.setdefault("artifacts", {})["side-stack"] = cache_path

        return entries
