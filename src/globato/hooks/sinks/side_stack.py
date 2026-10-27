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
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import rasterio
from filelock import FileLock
from rasterio.crs import CRS

from fetchez.hooks import FetchHook
from fetchez.utils import float_or, str2bool
from fetchez.spatial import parse_increments
from fetchez.entry import source_id, dataset_id

from globato.hooks.transforms.point_pixels import (
    FUSION_BAND_MAP,
    FUSION_BANDS,
    FUSION_STATE_VERSION,
    Point2PixelStream,
)

logger = logging.getLogger(__name__)

SIDESTACK_DATATYPE = "FUSION_STATE"
SIDESTACK_HASH_TAG = "GLOBATO_SIDESTACK_HASH"
# 2: point chunks add into the cache instead of overwriting it.
SIDESTACK_CACHE_VERSION = 2


def _resolve_grid(region: Any, res: Any, crs: Any) -> dict[str, Any]:
    """Resolve the exact raster grid used by Point2PixelStream."""

    inc = float_or(parse_increments(res))
    if inc is None:
        raise ValueError(f"Unable to resolve side-stack resolution {res!r}")

    xcount, ycount, gt = region.geo_transform(
        x_inc=inc,
        y_inc=inc,
        node="grid",
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


def sidestack_provenance(entry):
    """Descriptive source tags; intentionally excluded from the cache digest."""
    metadata = entry.get("metadata") or {}
    survey_id = entry.get("survey_id") or metadata.get("survey_id")

    tags = {}
    identifier = source_id(entry)
    if identifier is not None:
        tags["GLOBATO_SOURCE_ID"] = str(identifier)

    if survey_id is not None and str(survey_id).strip():
        tags["GLOBATO_SURVEY_ID"] = str(survey_id)

    return tags


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


def validate_sidestack(path, grid, digest):
    """Validate a cached FusionState against its expected grid and identity."""

    if not os.path.isfile(path):
        return False, "missing"

    try:
        with rasterio.open(path) as src:
            tags = src.tags()

            if tags.get("GLOBATO_DATATYPE") != SIDESTACK_DATATYPE:
                return False, "not a FusionState"

            if tags.get("GLOBATO_FUSION_VERSION") != str(FUSION_STATE_VERSION):
                return False, "FusionState version mismatch"

            if tags.get(SIDESTACK_HASH_TAG) != digest:
                return False, "cache identity mismatch"

            if src.count != len(FUSION_BANDS):
                return False, "band count mismatch"

            if tuple(src.descriptions) != tuple(FUSION_BANDS):
                return False, "band schema mismatch"

            if any(dtype != "float64" for dtype in src.dtypes):
                return False, "dtype mismatch"

            if (src.width, src.height) != (
                grid["width"],
                grid["height"],
            ):
                return False, "grid dimensions mismatch"

            expected_transform = rasterio.Affine(*grid["transform"])
            if src.transform != expected_transform:
                return False, "grid transform mismatch"

            expected_crs = CRS.from_wkt(grid["crs"]) if grid["crs"] else None

            if src.crs != expected_crs:
                return False, "CRS mismatch"

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


def sidestack_cache_key(entry, grid):
    identity = dataset_id(
        entry,
        grid=grid,
        fusion_version=FUSION_STATE_VERSION,
        side_stack_version=SIDESTACK_CACHE_VERSION,
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


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

            grid = _resolve_grid(region, self.res, self.crs)
            digest = sidestack_cache_key(entry, grid)

            cache_path = sidestack_path(entry, cache_dir, digest)

            entry["_sidestack_hash"] = digest
            entry["_sidestack_path"] = cache_path

            valid, reason = validate_sidestack(cache_path, grid, digest)

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
            # Blocks no source chunk touches are never stored and read back as
            # zeros, the same additive FusionState; readers can skip them.
            sparse_ok=True,
        )
        return out

    def _cache_stream(
        self,
        raster_stream,
        cache_path: str,
        *,
        digest: str,
        grid: dict,
        entry: dict,
        provenance: dict | None = None,
    ):
        """Write FusionState atomically while yielding identical chunks onward."""

        profile = next(raster_stream)
        expected_crs = CRS.from_wkt(grid["crs"]) if grid["crs"] else None
        if CRS.from_user_input(profile["crs"]) != expected_crs:
            raise ValueError(
                "Side-stack source CRS differs from the build CRS. Apply "
                "stream_reproject before side_stack or stack_qc; changing a CRS "
                "label is not a transformation."
            )
        if (
            profile["width"],
            profile["height"],
            profile["transform"],
            profile["count"],
        ) != (
            grid["width"],
            grid["height"],
            rasterio.Affine(*grid["transform"]),
            len(FUSION_BANDS),
        ):
            raise ValueError(
                "Side-stack input must use the build's exact native FusionState grid"
            )
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
            with rasterio.open(tmp_path, "w+", **creation_profile) as dst:
                for name, idx in FUSION_BAND_MAP.items():
                    dst.set_band_description(idx, name)

                dst.update_tags(
                    GLOBATO_DATATYPE=SIDESTACK_DATATYPE,
                    GLOBATO_FUSION_VERSION=str(FUSION_STATE_VERSION),
                    GLOBATO_SIDESTACK_VERSION=str(SIDESTACK_CACHE_VERSION),
                    **{
                        SIDESTACK_HASH_TAG: digest,
                        **sidestack_provenance(entry),
                    },
                )

                point_count = 0
                for chunk in raster_stream:
                    window, buff_win, data, ndv, transform = chunk

                    data = np.asarray(data, dtype=np.float64)
                    if data.ndim != 3 or data.shape[0] != len(FUSION_BANDS):
                        raise ValueError(
                            "Side-stack expected FusionState chunk with "
                            f"{len(FUSION_BANDS)} bands; got {data.shape!r}"
                        )

                    # Point chunks can revisit the same cells. FusionState is
                    # additive; overwriting would silently discard earlier
                    # soundings on cache reuse.
                    dst.write(dst.read(window=window) + data, window=window)
                    point_count += int(np.sum(data[FUSION_BAND_MAP["count"] - 1]))
                    yield window, buff_win, data, ndv, transform
                dst.update_tags(GLOBATO_POINT_COUNT=str(point_count))

            # Concurrent misses may have produced this same cache entry.
            # Publish once so another build's copy stays stable.
            with FileLock(cache_path + ".lock"):
                valid, _ = validate_sidestack(cache_path, grid, digest)
                if valid:
                    os.remove(tmp_path)
                else:
                    os.replace(tmp_path, cache_path)
            completed = True
            logger.debug("Side-stack cached: %s", os.path.basename(cache_path))

        finally:
            if not completed and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def run(self, entries):
        point_to_state = Point2PixelStream(
            x_inc=self.res,
            y_inc=self.res,
        )

        for mod, entry in entries:
            if entry.get("_sidestack_hit"):
                continue

            region = getattr(mod, "region", None)
            if not region:
                continue

            if self.is_point_stream(entry):
                fusion_stream = None
            elif (
                self.is_raster_stream(entry)
                and entry.get("data_type") == "fusion-state"
            ):
                fusion_stream = entry["stream"]
            else:
                continue

            cache_dir = self.cache_dir or getattr(
                mod,
                "_outdir",
                getattr(mod, "outdir", None),
            )

            grid = _resolve_grid(region, self.res, self.crs)
            digest = sidestack_cache_key(entry, grid)
            cache_path = entry.get("_sidestack_path") or sidestack_path(
                entry,
                cache_dir,
                digest,
            )

            valid, reason = validate_sidestack(cache_path, grid, digest)

            if valid:
                entry.setdefault("artifacts", {})["side-stack"] = cache_path
                continue

            logger.debug(
                "Side-stack miss; generating %s",
                os.path.basename(cache_path),
            )

            if fusion_stream is None:
                fusion_stream = point_to_state._stream_wrapper(
                    entry["stream"],
                    entry=entry,
                    region=region,
                )

            entry["stream"] = self._cache_stream(
                fusion_stream,
                cache_path,
                digest=digest,
                grid=grid,
                entry=entry,
                provenance=sidestack_provenance(entry),
            )
            entry["stream_type"] = "raster-stream"
            entry["data_type"] = "fusion-state"
            entry.setdefault("artifacts", {})["side-stack"] = cache_path

        return entries
