#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""globato.hooks.sinks.side_stack

Side-stack generation and cache management for Globato.

A side-stack stores the exact additive FusionState produced after an entry has
passed through its point-processing pipeline. Cache hits bypass raw source
reading and point processing and provide the cached FusionState directly as a
raster stream for ``multi_stack``.


Side-stacks accumulate elevation observations into per-source
FusionState rasters, which can be reused or combined downstream.

Entry metadata
--------------
Side-stack hooks use internal ``_sidestack_*`` entry fields
documented in ``SIDESTACK_ENTRY_KEYS``.

These fields describe cache lookup, artifact registration,
processing identity, completion status, and source aliases.

Fresh generation
----------------
When ``fresh=True``, the writer uses an isolated namespace and
publishes a completion receipt only after the input stream has
been fully consumed and the resulting raster finalized.

A registered side-stack is not necessarily completed. Consumers
requiring completed artifacts must validate the receipt rather
than relying on registration metadata alone.

Completion receipts currently exist as entry metadata and are
not persisted as independent sidecar files.

The ``_sidestack_*`` vocabulary is internal to Globato and may
evolve as identity handling is standardized with Fetchez.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import rasterio
from rasterio.crs import CRS

from fetchez.hooks import FetchHook
from fetchez.utils import float_or, str2bool, str2inc
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
# Authoritative M2D additive writer; reject caches from overwrite-only writers.
SIDESTACK_CACHE_VERSION = 2


# Internal entry metadata used by side-stack pipeline(s).
#
# These keys support side-stack caching, artifact generation, completion
# validation, and downstream consumers such as Coast Mask.

SIDESTACK_ENTRY_KEYS = {
    "_sidestack_hit": "Whether an existing side-stack cache was found.",
    "_sidestack_path": "Path to the selected or registered side-stack.",
    "_sidestack_hash": "Identity token used for side-stack caching.",
    "_sidestack_identity": "Canonical side-stack identity description.",
    "_sidestack_namespace": "Isolated directory for fresh generation.",
    "_sidestack_registered": "Whether fresh generation was registered.",
    "_sidestack_context": "Optional caller-provided receipt metadata.",
    "_sidestack_processing_identity": (
        "Source and processing identity used for artifact binding "
        "and contribution equivalence."
    ),
    "_sidestack_receipt": (
        "Completion metadata published after successful generation."
    ),
    "_sidestack_aliases": (
        "Equivalent contributions recorded during source selection."
    ),
}


SIDESTACK_RECEIPT_FIELDS = {
    "schema": "Receipt schema identifier and version.",
    "sink": "Component that generated the artifact.",
    "namespace": "Fresh-generation namespace.",
    "source_file": "Original source file reference.",
    "processing_identity": "Recorded source processing identity.",
    "context": "Caller-provided metadata frozen at registration.",
    "registered_claims": "Weight and uncertainty values at registration.",
    "path": "Published artifact path.",
    "sha256": "SHA-256 checksum of the completed artifact.",
}


def _resolve_grid(region: Any, res: Any, crs: Any) -> dict[str, Any]:
    """Resolve the exact raster grid used by Point2PixelStream."""

    inc = float_or(str2inc(res))
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


def file_digest(path):
    """Content checksum for an opt-in materialization receipt.

    The full-file hash supports downstream artifact-integrity validation,
    not dataset identity itself.
    """
    value = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _stable(value, depth=0):
    if depth > 32:
        raise TypeError(
            "Processing settings are cyclic or too deeply nested; "
            "provide a finite sidestack_identity on the hook"
        )
    if value is None or type(value) in (str, int, bool):
        return value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return [_stable(v, depth + 1) for v in value]
    if isinstance(value, dict):
        return {
            str(k): _stable(v, depth + 1)
            for k, v in sorted(value.items())
            if not str(k).startswith("_") and k not in ("stream", "raster_stream")
        }
    if hasattr(value, "wkb_hex"):
        return {"wkb": value.normalize().wkb_hex}
    if hasattr(value, "sidestack_identity"):
        identity = getattr(value, "sidestack_identity")
        return _stable(identity() if callable(identity) else identity, depth + 1)
    try:
        data = vars(value)
    except TypeError:
        return str(value)
    return {
        "hook": getattr(value, "name", type(value).__name__),
        "config": {
            str(k): _stable(v, depth + 1)
            for k, v in sorted(data.items())
            if not str(k).startswith("_")
            and k
            not in (
                "logger",
                "stream",
                "raster_stream",
                "dataset",
                "results",
                "session",
                # Runtime execution context may reference the complete pipeline,
                # including modules and hooks, creating recursive object graphs.
                # It is not part of a hook's processing identity.
                "context",
            )
        },
    }


def source_processing_identity(module, entry):
    """Reuse native dataset_id with explicit processing selectors; no new ID hash."""
    hooks = list(getattr(module, "hooks", ()) or ()) + list(
        entry.get("hooks", ()) or ()
    )
    selectors = dict(
        module=module.name,
        data_type=_stable(entry.get("data_type")),
        profile=_stable(entry.get("profile")),
        source_crs=_stable(entry.get("src_srs")),
        stack_partition=entry.get("stack_partition"),
        module_weight=_stable(getattr(module, "weight", None)),
        module_uncertainty=_stable(getattr(module, "uncertainty", None)),
        entry_weight=_stable(entry.get("weight")),
        entry_uncertainty=_stable(entry.get("uncertainty")),
        processing=_stable(hooks),
        source_claims=_stable(
            {
                k: entry.get(k)
                for k in (
                    "claim_required",
                    "claim_geometry",
                    "accepted_geometry",
                    "excluded_geometry",
                    "claim_priority",
                )
            }
        ),
    )
    native = dict(entry)
    path = Path(entry.get("dst_fn") or "")
    asset_hash = file_digest(path) if path.is_file() else None
    selectors["asset_sha256"] = asset_hash
    if not native.get("url") and entry.get("dst_fn"):
        native["url"] = path.resolve().as_uri()
    if not native.get("checksum") and asset_hash:
        native["checksum"] = asset_hash
    return dict(
        native_dataset_id=dataset_id(native, **selectors),
        identity_checksum=native.get("checksum"),
        asset_sha256=asset_hash,
        upstream_contribution_id=_stable(entry.get("contribution_id")),
        selectors=selectors,
        identity_owner="fetchez.entry.dataset_id",
    )


def unique_source_contributions(entries):
    """Select completed native source contributions once, retaining URL aliases.

    Only the opt-in native processing identity participates. Different class,
    weight and partition selectors remain distinct. Equal identities must have
    exactly equal grid geometry and all seven statistics; conflicts fail before
    reduction. Legacy entries without that identity retain their normal behavior.
    """
    selected = []
    seen = {}
    for module, entry in entries:
        processing = entry.get("_sidestack_processing_identity") or {}
        key = processing.get("native_dataset_id")
        if not key:
            selected.append((module, entry))
            continue
        if key not in seen:
            seen[key] = entry
            selected.append((module, entry))
            continue
        original = seen[key]
        if original is entry:
            continue
        paths = []
        for candidate in (original, entry):
            path = Path((candidate.get("artifacts") or {}).get("side-stack") or "")
            receipt = candidate.get("_sidestack_receipt") or {}
            if (
                not path.is_file()
                or receipt.get("schema") != "globato.side_stack.receipt.v1"
                or receipt.get("processing_identity")
                != candidate.get("_sidestack_processing_identity")
                or Path(receipt.get("path") or "").resolve() != path.resolve()
                or receipt.get("sha256") != file_digest(path)
            ):
                raise ValueError(
                    "Duplicate source lacks a valid completed native side-stack receipt"
                )
            paths.append(path)
        with rasterio.open(paths[0]) as a, rasterio.open(paths[1]) as b:
            if (
                (a.crs, a.transform, a.shape, a.count, a.dtypes, a.descriptions)
                != (b.crs, b.transform, b.shape, b.count, b.dtypes, b.descriptions)
                or a.count != len(FUSION_BANDS)
                or a.descriptions != tuple(FUSION_BANDS)
            ):
                raise ValueError(
                    "Conflicting grids for one native processing identity; declare distinct source partitions"
                )
            for _, window in a.block_windows(1):
                if a.read(window=window).tobytes() != b.read(window=window).tobytes():
                    raise ValueError(
                        "Conflicting statistics for one native processing identity; declare distinct source partitions"
                    )
        aliases = original.setdefault("_sidestack_aliases", [])
        aliases.append(
            dict(
                source_file=str(entry.get("dst_fn") or ""),
                source_url=entry.get("url"),
                side_stack=str(paths[1].resolve()),
                sha256=entry["_sidestack_receipt"]["sha256"],
                native_dataset_id=key,
                comparison="exact_seven_band_state",
            )
        )
        aliases.extend(entry.get("_sidestack_aliases", []))
        logger.info("Native source alias selected once: %s", entry.get("dst_fn"))
    return selected


def sidestack_cache_key(entry, grid, *, processing=None):
    modifiers = {} if processing is None else {"processing": processing}
    identity = dataset_id(
        entry,
        grid=grid,
        fusion_version=FUSION_STATE_VERSION,
        sidestack_version=SIDESTACK_CACHE_VERSION,
        **modifiers,
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
    """Generate per-source FusionState grids from incoming point streams.

    By default, existing side-stack caches may be reused.

    When fresh=True, generate a new side-stack without reusing an
    existing cached result. Processing distinctions recorded in the
    entry's contribution metadata are reflected through dataset_id.

    A completion receipt (_sidestack_receipt) is published only
    after the input stream has been fully consumed and the output
    raster has been successfully finalized. Optional JSON-compatible
    entry['_sidestack_context'] metadata is preserved in the receipt
    for downstream consumers.

    Overlapping input chunks are accumulated rather than overwritten,
    ensuring all measurements contribute to the resulting FusionState.
    The writer cache version is incremented to invalidate older
    side-stacks created with the previous overwrite behavior.
    """

    name = "side_stack"
    meta_stage = "stream"

    def __init__(
        self,
        res="1s",
        crs="EPSG:4326",
        cache_dir=None,
        compress=True,
        skip_entry=None,
        *,
        fresh=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.res = res
        self.crs = crs
        self.cache_dir = cache_dir
        self.compress = str2bool(compress)
        self.fresh = str2bool(fresh)
        self.skip_entry = skip_entry

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
        digest: str,
        entry: dict,
        provenance: dict | None = None,
        receipt: dict | None = None,
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
            with rasterio.open(tmp_path, "w+", **creation_profile) as dst:
                for name, idx in FUSION_BAND_MAP.items():
                    dst.set_band_description(idx, name)

                dst.update_tags(
                    GLOBATO_DATATYPE=SIDESTACK_DATATYPE,
                    GLOBATO_FUSION_VERSION=str(FUSION_STATE_VERSION),
                    **{
                        SIDESTACK_HASH_TAG: digest,
                        **sidestack_provenance(entry),
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

                    # Native FusionState is additive even across overlapping chunks.
                    dst.write(dst.read(window=window) + data, window=window)
                    yield window, buff_win, data, ndv, transform

            os.replace(tmp_path, cache_path)
            completed = True
            if receipt is not None:
                entry["_sidestack_receipt"] = {
                    **receipt,
                    "path": str(Path(cache_path).resolve()),
                    "sha256": file_digest(cache_path),
                }
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
            if self.skip_entry and entry.get(self.skip_entry):
                continue
            if self.fresh and (
                entry.get("_sidestack_hit") or entry.get("_sidestack_path")
            ):
                raise ValueError("Fresh side_stack requires an unbound source stream")
            if entry.get("_sidestack_hit"):
                continue

            region = getattr(mod, "region", None)
            if not region or not self.is_point_stream(entry):
                continue

            stream_crs = CRS.from_user_input(entry.get("src_srs", "EPSG:4326"))
            target_crs = CRS.from_user_input(self.crs) if self.crs else None
            if stream_crs != target_crs:
                raise ValueError(
                    f"side_stack input CRS {stream_crs} differs from grid CRS "
                    f"{target_crs}; add stream_reproject before side_stack"
                )

            cache_dir = self.cache_dir or getattr(
                mod,
                "_outdir",
                getattr(mod, "outdir", None),
            )

            grid = _resolve_grid(region, self.res, self.crs)
            receipt = None
            processing = None
            if self.fresh:
                # Opt-in branch identity uses Fetchez's existing dataset_id;
                # no separate source registry or alternate rasterizer.
                processing = source_processing_identity(mod, entry)
                entry["_sidestack_processing_identity"] = processing
                base = Path(
                    cache_dir or os.path.dirname(str(entry.get("dst_fn", "."))) or "."
                )
                base.mkdir(parents=True, exist_ok=True)
                cache_dir = tempfile.mkdtemp(prefix="sidestack_", dir=base)
                entry["_sidestack_namespace"] = Path(cache_dir).name
                entry["_sidestack_registered"] = True

                # Build a completion receipt for a freshly generated side-stack.

                # The receipt records the originating source, processing identity,
                # registered source claims, caller-supplied context, output artifact,
                # and its SHA-256 checksum.

                # A receipt is attached to the entry only after the input stream has
                # completed successfully and the output raster has been finalized
                # and published.

                # Receipts are currently in-memory entry metadata, not persistent
                # sidecar files. They support downstream artifact validation but do
                # not guarantee that the source or output remains unchanged afterward.

                receipt = {
                    "schema": "globato.side_stack.receipt.v1",
                    "created_in_fresh_namespace": True,
                    "sink": "globato.side_stack",
                    "namespace": entry["_sidestack_namespace"],
                    "source_file": str(entry.get("dst_fn") or ""),
                    "processing_identity": processing,
                    "context": json.loads(
                        json.dumps(entry.get("_sidestack_context", {}), sort_keys=True)
                    ),
                    "registered_claims": {
                        "entry_weight": entry.get("weight"),
                        "entry_uncertainty": entry.get("uncertainty"),
                        "module_weight": getattr(mod, "weight", None),
                        "module_uncertainty": getattr(mod, "uncertainty", None),
                    },
                }
                receipt = json.loads(json.dumps(receipt, sort_keys=True))
            digest = sidestack_cache_key(entry, grid, processing=processing)
            cache_path = entry.get("_sidestack_path") or sidestack_path(
                entry,
                cache_dir,
                digest,
            )
            if self.fresh:
                entry["_sidestack_hash"] = digest
                entry["_sidestack_path"] = cache_path
                entry.pop("_sidestack_receipt", None)

            valid, reason = validate_sidestack(cache_path, grid, digest)

            if valid:
                entry.setdefault("artifacts", {})["side-stack"] = cache_path
                continue

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
                digest=digest,
                entry=entry,
                provenance=sidestack_provenance(entry),
                receipt=receipt,
            )
            entry["stream_type"] = "raster-stream"
            entry["data_type"] = "fusion-state"
            entry.setdefault("artifacts", {})["side-stack"] = cache_path

        return entries
