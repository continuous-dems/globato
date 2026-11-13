"""Native hook integration: saved exact points, classification, native masks.

No provider discovery, hook cloning, lifecycle replay or private Recipe runner.
The regular stream phase applies transforms/claims once. A lossless GTPC
artifact retains subcell information until collection-stage classification.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
import tempfile
from contextlib import nullcontext
from time import perf_counter

import geopandas as gpd
import numpy as np
from pyproj import CRS, Transformer
import rasterio
from rasterio.windows import Window
from shapely.geometry import box
from filelock import FileLock

from fetchez.entry import set_contribution_id
from fetchez.hooks import FetchHook
from fetchez.spatial import parse_increments
from fetchez.utils import str2bool
from globato.hooks.sinks.multi_stack import MultiStackAccumulator

from ._io import digest, identity, json_write, serializable, vector_files
from .artifact import MODES, MaskArtifact
from .masking import publish_masked_source
from .selection import horizontal_crs
from .provenance import source_snapshot, preparation_scope_with_tier_support
from .support import (
    pad_evidence_for_native_tiers,
    publish_preparation_evidence,
    _require_valid_evidence,
)

log = logging.getLogger(__name__)


def _is_raster_source(entry, path):
    """Use the declared data type, then the native GDAL reader, not a TIFF suffix."""
    datatype = str(entry.get("data_type") or "").lower()
    if datatype in {"lidar", "xyz", "point", "point-stream", "gtpc"}:
        return False
    try:
        with rasterio.open(path):
            return True
    except rasterio.errors.RasterioError:
        return False


def check_source(module, entry, request):
    native = dict(entry)
    # stream-init/reproject legitimately set this field during the normal pass.
    native["src_srs"] = request["source"]["entry"]["src_srs"]
    native["contribution_id"] = request["source"]["entry"]["contribution_id"]
    if (
        "stream_contribution_id" in request
        and serializable(entry.get("contribution_id"))
        != request["stream_contribution_id"]
    ):
        raise ValueError("Processing contribution changed after point capture")
    if (
        source_snapshot(
            module,
            native,
            Path(request["source"]["source"]),
            point_source=request.get("point_source", False),
        )
        != request["source"]
    ):
        raise ValueError(
            "Original source, claims or weights changed after registration"
        )


def _source_policy(group, mask, polygon_class, remove, outlet_lonlat):
    """Shared source/geometry validation, with no interpolation parameters."""
    if not isinstance(group, str) or not group.strip():
        raise ValueError("group must be a nonempty recipe label")
    is_osm = str(mask).lower() == "osm"
    if is_osm and polygon_class is not None:
        raise ValueError("OSM supplies its classification; omit polygon_class")
    if not is_osm and polygon_class not in (*MODES, "osm"):
        raise ValueError(
            "Local polygon_class is required: " + ", ".join((*MODES, "osm"))
        )
    if remove not in MODES:
        raise ValueError(
            "Explicit remove is required; choose one of: " + ", ".join(MODES)
        )
    if outlet_lonlat is not None:
        values = (
            outlet_lonlat.replace(",", "/").split("/")
            if isinstance(outlet_lonlat, str)
            else outlet_lonlat
        )
        try:
            longitude, latitude = map(float, values)
        except (TypeError, ValueError) as exc:
            raise ValueError("outlet_lonlat must be lon/lat in EPSG:4326") from exc
        if not (
            math.isfinite(longitude)
            and math.isfinite(latitude)
            and -180 <= longitude <= 180
            and -90 <= latitude <= 90
        ):
            raise ValueError("Invalid outlet_lonlat coordinates")
        outlet_lonlat = [longitude, latitude]
    return dict(
        group=group.strip(),
        mask="osm" if is_osm else str(mask),
        polygon_class=polygon_class,
        remove=remove,
        outlet_lonlat=outlet_lonlat,
    )


def _register_sources(entries, settings, output_dir):
    for module, entry in entries:
        if (
            "coast_refine_request" in entry
            or "coast_refine_consumer" in entry
            or entry.get("coast_processing_id")
        ):
            raise ValueError("Coast Refine source is already registered or masked")
        path = Path(entry["dst_fn"]).expanduser()
        if not path.is_absolute():
            path = Path(getattr(module, "_outdir", None) or Path.cwd()) / path
        path = path.resolve(strict=True)
        if not settings["grid_native"]:
            if path.suffix.lower() not in (".tif", ".tiff"):
                raise ValueError("Legacy Coast Refine requires original TIFF")
            if entry.get("stream") is not None or entry.get("_sidestack_hit"):
                raise ValueError("Legacy Coast Refine requires original stream")
            from ._science.input_contracts import check_encoding

            with rasterio.open(path) as src:
                check_encoding(src)
        elif entry.get("stream") is not None and not entry.get("_sidestack_hit"):
            raise ValueError("Grid Coast Mask must register before stream processing")
        root = Path(output_dir or Path.cwd() / "coast_refine").resolve()
        point_source = settings["grid_native"] and not _is_raster_source(entry, path)
        if (
            settings["grid_native"]
            and not point_source
            and settings.get("evidence", True)
        ):
            # Native gridding must not hide an unsupported producer encoding.
            # The authoritative terrain science expects a decoded, single,
            # floating metre-elevation band; it is not a raster decoder.
            from ._science.input_contracts import check_encoding

            with rasterio.open(path) as src:
                check_encoding(src)
        entry["coast_refine_request"] = {
            **settings,
            "output_dir": str(root),
            "source": source_snapshot(
                module,
                entry,
                path,
                point_source=point_source,
            ),
            "point_source": point_source,
        }
        entry["coast_refine_deferred"] = True
    return entries


class CoastRefine(FetchHook):
    """Select a raster source group; normal stream/collection hooks do the work.

    Attach after extraction and before stream initialization. ``group`` is an
    explicit recipe label, shared by the TNM tiers when they form one authority.
    """

    name = "coast_refine"
    meta_stage = "file"
    meta_category = "raster-filter"
    meta_desc = (
        "Register native rasters for Coast Refine capture and collection processing."
    )
    meta_input_paths = ("mask",)
    meta_output_paths = ("output_dir",)
    meta_named_inputs = {"mask": ("osm",), "barrier": ("coast_refine",)}

    def __init__(
        self,
        group,
        mask="osm",
        polygon_class=None,
        remove=None,
        barrier=None,
        review=False,
        viz=False,
        output_dir=None,
        barrier_extension_m=0,
        barrier_extension_min_z=0.25,
        barrier_extension_method="fixed",
        barrier_slope_window_m=30,
        river_guidance="off",
        outlet_lonlat=None,
        grid_native=False,
        **kwargs,
    ):
        unknown = set(kwargs) - {"stage"}
        if unknown:
            raise TypeError(
                "Unsupported coast_refine arguments: " + ", ".join(sorted(unknown))
            )
        super().__init__(**kwargs)
        if not str2bool(grid_native):
            from globato.streams.readers import gtpc

            if not hasattr(gtpc, "iter_gtpc_chunks"):
                raise ValueError(
                    "Legacy point capture needs the prepared-point runtime; use grid_native: true on current upstream"
                )
        if self.stage != "file":
            raise ValueError("coast_refine must run at file stage")
        policy = _source_policy(group, mask, polygon_class, remove, outlet_lonlat)
        group, mask, polygon_class, remove, outlet_lonlat = (
            policy[k]
            for k in ("group", "mask", "polygon_class", "remove", "outlet_lonlat")
        )
        is_osm = mask == "osm"
        if barrier not in (None, "coast_refine"):
            raise ValueError("barrier must be coast_refine or omitted")
        from .extension import settings

        extension = settings(
            barrier_extension_m,
            barrier_extension_min_z,
            barrier_extension_method,
            barrier_slope_window_m,
        )
        if extension["barrier_extension_m"] and barrier != "coast_refine":
            raise ValueError("barrier_extension_m requires barrier: coast_refine")
        if river_guidance not in ("off", "bank_smooth"):
            raise ValueError("river_guidance must be off or bank_smooth")
        if river_guidance != "off" and barrier != "coast_refine":
            raise ValueError("river_guidance requires barrier: coast_refine")
        self.settings = dict(
            river_guidance=river_guidance,
            outlet_lonlat=outlet_lonlat,
            group=group,
            mask="osm" if is_osm else str(mask),
            polygon_class=polygon_class,
            remove=remove,
            barrier=barrier,
            grid_native=str2bool(grid_native),
            review=str2bool(review),
            viz=str2bool(viz),
            **extension,
        )
        self.output_dir = output_dir

    def run(self, entries):
        return _register_sources(entries, self.settings, self.output_dir)


class CoastMask(FetchHook):
    """Source-specific mask and geometry refinement; no interpolation policy.

    The ordinary Coast Refine capture/finalization machinery remains shared:
    authoritative source data are processed only once, and the publication is
    reusable independently of any interpolation hook. Existing `coast_refine`
    recipes remain supported during migration.
    """

    name = "coast_mask"
    meta_stage = "file"
    meta_category = "raster-filter"
    meta_desc = "Mask selected raster sources and publish refined water geometry."
    meta_input_paths = ("mask",)
    meta_output_paths = ("output_dir",)
    meta_named_inputs = {"mask": ("osm",)}

    def __init__(
        self,
        group,
        mask="osm",
        polygon_class=None,
        remove=None,
        review=False,
        viz=False,
        output_dir=None,
        outlet_lonlat=None,
        grid_native=True,
        **kwargs,
    ):
        unknown = set(kwargs) - {"stage"}
        if unknown:
            raise TypeError(
                "coast_mask accepts only source filtering and geometry arguments; "
                "unsupported: " + ", ".join(sorted(unknown))
            )
        super().__init__(**kwargs)
        if self.stage != "file":
            raise ValueError("coast_mask must run at file stage")
        self.settings = _source_policy(
            group, mask, polygon_class, remove, outlet_lonlat
        )
        if not str2bool(grid_native):
            raise ValueError(
                "coast_mask uses grid_native: true; legacy prepared-point recipes use coast_refine"
            )
        self.settings.update(
            barrier="coast_refine",
            grid_native=str2bool(grid_native),
            review=str2bool(review),
            viz=str2bool(viz) or str2bool(review),
        )
        self.output_dir = output_dir

    def run(self, entries):
        selected = _register_sources(entries, self.settings, self.output_dir)
        for _, entry in selected:
            # Preserve the existing stream and collection lifecycle contracts
            # while publishing the source hook identity for downstream routing.
            entry["coast_refine_request"]["source_hook"] = "coast_mask"
        for module, entry in selected:
            metadata, claims = _consumer_registered_class_metadata(module, entry)
            entry["coast_refine_request"].update(
                registered_source_metadata=serializable(metadata),
                class_branch_claims=claims,
            )
        return selected


def _consumer_registered_class_metadata(module, entry):
    """Freeze an explicitly configured class branch before stream hooks run.

    `entry["metadata"]` is not a reliable *final* source of the branch:
    stream-init/readers can replace it. At FILE registration, only explicit
    `class_branch` metadata is a class claim. `drop-class.stack_partition` is
    an independent source-partition identifier, NOT a class label: it can be
    an arbitrary string such as "crescent-dav-5022-bathy-29". Keep that ID
    separately for provenance without interpreting its contents. Never derive
    a class from remove, elevation, or a partition-name substring.
    """
    from collections.abc import Mapping

    metadata = entry.get("metadata")
    frozen = dict(metadata) if isinstance(metadata, Mapping) else {}
    declared = []
    partitions = []
    initial = frozen.get("class_branch")
    if initial is not None:
        declared.append(("entry.metadata", initial))

    for hook in list(getattr(module, "hooks", ()) or ()) + list(
        entry.get("hooks") or ()
    ):
        if isinstance(hook, Mapping):
            name = hook.get("name", "")
            config = hook.get("args", {}) or {}
        else:
            name = getattr(hook, "name", "")
            config = hook
        name = str(name).replace("_", "-").lower()

        def arg(key):
            if isinstance(config, Mapping):
                return config.get(key)
            return getattr(config, key, None)

        if name == "set-entry" and arg("key") == "metadata":
            value = arg("value")
            if isinstance(value, Mapping):
                if value.get("class_branch") is not None:
                    declared.append(("set-entry.metadata", value["class_branch"]))
                # Only retain the configured metadata that was actually
                # declared, never imply it was observed in the final stream.
                for key, val in value.items():
                    frozen.setdefault(key, val)
        if name == "drop-class" and arg("stack_partition") is not None:
            partitions.append(str(arg("stack_partition")))

    # Preserve the partition identifier independently of the class-branch
    # claims. It scopes source-grid identity but says nothing about class.
    unique_partitions = {p for p in partitions if p}
    if len(unique_partitions) > 1:
        raise ValueError(
            "Conflicting Coast Mask consumer stack partitions: "
            + ", ".join(sorted(unique_partitions))
        )
    if unique_partitions:
        partition = next(iter(unique_partitions))
        if (
            frozen.get("stack_partition") is not None
            and str(frozen["stack_partition"]) != partition
        ):
            raise ValueError(
                "Conflicting Coast Mask consumer stack partitions: "
                f"entry.metadata={frozen['stack_partition']!r}, "
                f"drop-class.stack_partition={partition!r}"
            )
        frozen["stack_partition"] = partition

    labels = {str(value) for _, value in declared if value is not None and str(value)}
    if len(labels) > 1:
        raise ValueError(
            "Conflicting Coast Mask consumer class claims: "
            + ", ".join(f"{origin}={value!r}" for origin, value in declared)
        )
    label = next(iter(labels), None)
    if label is not None:
        frozen["class_branch"] = label
    return frozen, [origin for origin, value in declared if str(value) == label]


def _consumer_receipt_metadata(request, entry):
    """Merge live metadata with immutable, explicitly registered source claims."""
    from collections.abc import Mapping

    live = entry.get("metadata")
    live = dict(live) if isinstance(live, Mapping) else {}
    registered = request.get("registered_source_metadata") or {}
    branch = registered.get("class_branch")
    if (
        branch is not None
        and live.get("class_branch") is not None
        and str(live["class_branch"]) != str(branch)
    ):
        raise ValueError(
            "Consumer class branch changed between file registration and finalization"
        )
    return {**registered, **live}


class CoastMaskConsumer(FetchHook):
    """Register a grid-native consumer at the FILE stage, before side_stack.

    Unlike point_raster_mask, this hook never filters a live point stream.
    The ordinary source-side-stack captures an unfilled, per-entry FusionState;
    CoastRefineFinalize._finish_grid_consumers applies the group's geometry
    to that exact state AFTER all producer evidence has been prepared.
    """

    name = "coast_mask_consumer"
    meta_stage = "file"
    meta_category = "raster-filter"
    meta_desc = "Register an independent grid-native Coast Mask consumer."
    meta_output_paths = ("output_dir",)

    def __init__(self, authority, remove, output_dir=None, grid_native=True, **kwargs):
        unknown = set(kwargs) - {"stage"}
        if unknown:
            raise TypeError(
                "Unsupported coast_mask_consumer arguments: "
                + ", ".join(sorted(unknown))
            )
        super().__init__(**kwargs)
        if self.stage != "file":
            raise ValueError("coast_mask_consumer must run at file stage")
        if not isinstance(authority, str) or not authority.strip():
            raise ValueError("coast_mask_consumer requires a named authority group")
        if remove not in MODES:
            raise ValueError("Grid-native consumers require remove: " + "|".join(MODES))
        if not str2bool(grid_native):
            raise ValueError("coast_mask_consumer requires grid_native: true")
        self.authority = authority.strip()
        self.remove = remove
        self.output_dir = output_dir

    def run(self, entries):
        for module, entry in entries:
            if (
                entry.get("coast_refine_consumer")
                or entry.get("coast_refine_request")
                or entry.get("coast_processing_id")
            ):
                raise ValueError("Coast Mask entry is already registered")
            if entry.get("stream") is not None or entry.get("_sidestack_hit"):
                raise ValueError(
                    "Register grid-native consumer before streaming or cached side-stack binding"
                )
            if not entry.get("dst_fn"):
                raise ValueError("Coast Mask consumer lacks an original source file")
            path = Path(entry["dst_fn"]).expanduser()
            if not path.is_absolute():
                path = Path(getattr(module, "_outdir", None) or Path.cwd()) / path
            path = path.resolve(strict=True)
            root = Path(self.output_dir or Path.cwd() / "coast_refine").resolve()
            # Freeze configured branch provenance before downstream stream
            # readers can replace the general-purpose `metadata` mapping.
            branch_metadata, branch_claims = _consumer_registered_class_metadata(
                module, entry
            )
            point_source = not _is_raster_source(entry, path)
            entry["coast_refine_consumer"] = {
                "authority": self.authority,
                "remove": self.remove,
                "output_dir": str(root),
                "grid_native": True,
                "point_source": point_source,
                "registered_source_metadata": serializable(branch_metadata),
                "class_branch_claims": branch_claims,
                "source": source_snapshot(
                    module, entry, path, point_source=point_source
                ),
            }
        return entries


class CoastRefineCapture(FetchHook):
    """Lossless point checkpoint after normal transforms and before diagnostics.

    Only registered entries are consumed. Other entries continue to the
    ordinary side_stack sink. Captured/cached samples continue through ordinary
    diagnostic taps. No transformations are rediscovered or replayed.
    """

    name = "coast_refine_capture"
    meta_stage = "stream"
    meta_category = "stream-sink"

    def __init__(self, compression="lzf", grid_native=False, **kwargs):
        super().__init__(**kwargs)
        self.grid_native = str2bool(grid_native)
        if compression not in ("gzip", "lzf", None):
            raise ValueError("capture compression must be lzf, gzip or null")
        self.compression = compression

    def run(self, entries):
        from .capture import capture_identity

        for module, entry in entries:
            request = entry.get("coast_refine_request") or entry.get(
                "coast_refine_consumer"
            )
            if request is None:
                continue
            if self.grid_native:
                # Grid-native never needs an additional capture stage.  The
                # normal side_stack sink is responsible for source identity,
                # lazy raster materialization and entry-bound completion.
                raise ValueError(
                    "Remove coast_refine_capture from grid-native recipes; "
                    "use the standard side_stack lifecycle"
                )
            if not self.is_point_stream(entry):
                raise ValueError(
                    "Place coast_refine_capture after point transforms/claims, before raster sinks"
                )
            started = perf_counter()
            check_source(module, entry, request)
            timing = {"source_validation_seconds": perf_counter() - started}
            entry["coast_refine_capture_timing"] = timing
            root = Path(request["output_dir"]) / "points"
            root.mkdir(parents=True, exist_ok=True)
            cache, reason = capture_identity(module, entry, request, self.compression)
            if reason:
                log.info(
                    "Capture reuse disabled for %s: %s", entry.get("dst_fn"), reason
                )
            entry["stream"] = self._capture_stream(
                module, entry, entry["stream"], root, cache, timing, request=request
            )
        return entries

    def _capture_stream(
        self, module, entry, upstream, root, cache, timing, *, request=None
    ):
        from .capture import inventory, read_capture
        from globato.hooks.sinks.gtpc_writer import WriteGTPC
        from globato.streams.readers.gtpc import iter_gtpc_chunks

        started = perf_counter()
        request = request if request is not None else entry["coast_refine_request"]
        key, inputs = cache if cache else (None, None)
        lock = FileLock(str(root / (key + ".lock"))) if key else nullcontext()
        with lock:
            lookup = perf_counter()
            hit = read_capture(root, key, inputs) if key else None
            timing["lookup_and_integrity_seconds"] = perf_counter() - lookup
            if hit:
                target, saved = hit
                close = getattr(upstream, "close", None)
                if close:
                    close()
                checksum = saved["sha256"]
                timing.update(
                    cache_hit=True,
                    points=saved["inventory"]["points"],
                    upstream_seconds=0.0,
                    write_seconds=0.0,
                )
                log.info(
                    "Capture HIT %s: %s points; verify %.2fs; no source-point replay or recompression",
                    Path(entry["dst_fn"]).name,
                    timing["points"],
                    timing["lookup_and_integrity_seconds"],
                )
                cached = iter(iter_gtpc_chunks(target))
                read_seconds = downstream_seconds = 0.0
                while True:
                    read_started = perf_counter()
                    try:
                        chunk = next(cached)
                    except StopIteration:
                        read_seconds += perf_counter() - read_started
                        break
                    read_seconds += perf_counter() - read_started
                    downstream_started = perf_counter()
                    yield chunk
                    downstream_seconds += perf_counter() - downstream_started
                timing.update(
                    cached_read_seconds=read_seconds,
                    downstream_seconds=downstream_seconds,
                )
            else:
                log.info(
                    "Capture MISS %s: writing lossless %s points",
                    Path(entry["dst_fn"]).name,
                    self.compression,
                )
                fd, temporary = tempfile.mkstemp(suffix=".gtpc", dir=root)
                os.close(fd)
                temporary = Path(temporary)
                try:
                    local = dict(entry, stream=upstream)
                    WriteGTPC(
                        output_path=str(temporary),
                        drain=False,
                        compression=self.compression,
                        report_timing=True,
                    ).run([(module, local)])
                    yield from local["stream"]
                    timing.update(local["gtpc_timing"])
                    validation_started = perf_counter()
                    check_source(module, entry, request)
                    timing["source_validation_seconds"] += (
                        perf_counter() - validation_started
                    )
                    hash_started = perf_counter()
                    checksum = digest(temporary)
                    timing["hash_seconds"] = perf_counter() - hash_started
                    target = root / (checksum + ".gtpc")
                    if target.is_symlink():
                        raise ValueError("Point capture cache contains a symlink")
                    if target.exists():
                        if digest(target) != checksum:
                            raise ValueError("Saved point artifact is corrupt")
                    else:
                        os.replace(temporary, target)
                    saved = {
                        "inputs": inputs,
                        "sha256": checksum,
                        "inventory": inventory(target),
                    }
                    if key:
                        receipts = root / "receipts"
                        receipts.mkdir(exist_ok=True)
                        fd, pending = tempfile.mkstemp(suffix=".json", dir=receipts)
                        os.close(fd)
                        try:
                            json_write(pending, saved)
                            os.replace(pending, receipts / (key + ".json"))
                        finally:
                            Path(pending).unlink(missing_ok=True)
                    timing["cache_hit"] = False
                finally:
                    temporary.unlink(missing_ok=True)
            request.update(
                points=str(target),
                points_sha256=checksum,
                stream_srs=entry.get("src_srs"),
                stream_contribution_id=serializable(entry.get("contribution_id")),
            )
            entry.setdefault("artifacts", {})["coast_refine_points"] = str(target)
        timing["capture_elapsed_seconds"] = perf_counter() - started
        log.info(
            "Capture complete %s: hit=%s; upstream %.2fs; write %.2fs; flush/close %.2fs; hash %.2fs; cache-read %.2fs; diagnostics %.2fs; total %.2fs",
            Path(entry["dst_fn"]).name,
            timing["cache_hit"],
            timing.get("upstream_seconds", 0),
            timing.get("write_seconds", 0),
            timing.get("flush_close_seconds", 0),
            timing.get("hash_seconds", 0),
            timing.get("cached_read_seconds", 0),
            timing.get("downstream_seconds", 0),
            timing["capture_elapsed_seconds"],
        )


def retained_points(path, masked_source, stream_srs):
    """Select saved transformed samples by the exact native TIFF validity.

    Elevations/weights/uncertainties are never regenerated or transformed again.
    Inverse horizontal mapping locates the original pixel; a center check stops
    incompatible streams rather than accepting a resampled/relocated surface.
    """
    from globato.streams.readers.gtpc import iter_gtpc_chunks

    with rasterio.open(masked_source) as native:
        transform = Transformer.from_crs(
            horizontal_crs(stream_srs), horizontal_crs(native.crs), always_xy=True
        )
        inv = ~native.transform
        for chunk in iter_gtpc_chunks(path):
            x, y = transform.transform(chunk["x"], chunk["y"])
            col_f = inv.a * x + inv.b * y + inv.c
            row_f = inv.d * x + inv.e * y + inv.f
            if not np.isfinite(col_f).all() or not np.isfinite(row_f).all():
                raise ValueError("Nonfinite transformed source coordinates")
            if np.any(np.abs(col_f - np.floor(col_f) - 0.5) > 1e-3) or np.any(
                np.abs(row_f - np.floor(row_f) - 0.5) > 1e-3
            ):
                raise ValueError(
                    "Captured points no longer represent original native pixel centers"
                )
            cols, rows = (
                np.floor(col_f).astype("int64"),
                np.floor(row_f).astype("int64"),
            )
            inside = (
                (cols >= 0)
                & (cols < native.width)
                & (rows >= 0)
                & (rows < native.height)
            )
            if not inside.all():
                raise ValueError(
                    "Captured points fall outside their original native raster"
                )
            keep = np.zeros(len(chunk), dtype=bool)
            blocks = np.column_stack((rows // 512, cols // 512))
            unique, inverse = np.unique(blocks, axis=0, return_inverse=True)
            for index, (row, col) in enumerate(unique):
                win = Window(
                    int(col * 512),
                    int(row * 512),
                    min(512, native.width - int(col * 512)),
                    min(512, native.height - int(row * 512)),
                )
                selected = np.flatnonzero(inverse == index)
                values = native.read(1, window=win)
                valid = (native.read_masks(1, window=win) > 0) & np.isfinite(values)
                keep[selected] = valid[
                    rows[selected] - int(win.row_off), cols[selected] - int(win.col_off)
                ]
            yield chunk[keep]


def accumulator(path, region, res, crs, *, strategy="mean", weights="1"):
    return MultiStackAccumulator(
        region,
        res,
        res,
        str(path.with_suffix(".tif")),
        state_fn=str(path),
        resume=False,
        crs=crs,
        strategy=strategy,
        weight_threshold=weights,
    )


# Geometry policy is shared by producers of one authority group. Source-specific
# removal is NOT a geometry setting: two contributors may refine one coastline
# while retaining opposite sides in their own downstream source grids.
# The old Coast Refine extension settings remain group-wide until the separate
# coast_guidance/ms_binary_cudem migration is implemented.
_GROUP_GEOMETRY_KEYS = (
    "mask",
    "polygon_class",
    "barrier",
    "outlet_lonlat",
    "source_hook",
    "barrier_extension_m",
    "barrier_extension_min_z",
    "barrier_extension_method",
    "barrier_slope_window_m",
    "river_guidance",
)


def group_geometry_policy(request):
    """Return only fields that must agree to share one refined polygon."""
    return {key: request.get(key) for key in _GROUP_GEOMETRY_KEYS}


class CoastRefineFinalize(FetchHook):
    """Collection operation on completed point artifacts and original TIFFs."""

    name = "coast_refine_finalize"
    meta_stage = "collection"
    meta_category = "raster-filter"

    def __init__(
        self,
        res="0.1111111s",
        crs="EPSG:4269+5703",
        resolutions=None,
        require_barrier=False,
        strategy="mixed",
        weight_threshold="0.1/0.25/0.5/0.75/1/2/3/4",
        grid_native_compare=False,
        grid_native=False,
        binding_group=None,
        audit_overlaps=False,
        defer_barrier=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if binding_group is not None and (
            not isinstance(binding_group, str) or not binding_group.strip()
        ):
            raise ValueError("binding_group must be a nonempty group label")
        self.binding_group = (
            binding_group.strip() if binding_group is not None else None
        )
        self.defer_barrier = str2bool(defer_barrier)
        self.audit_overlaps = str2bool(audit_overlaps)
        self.res = float(parse_increments(res, crs=crs))
        self.crs = crs
        self.resolutions = tuple(
            float(parse_increments(x, crs=crs))
            for x in (
                resolutions.split("/")
                if isinstance(resolutions, str)
                else resolutions or ()
            )
        )
        self.require_barrier = str2bool(require_barrier)
        self.strategy = strategy
        self.weight_threshold = weight_threshold
        # Comparison-only: never alters the source path chosen for stack_combine.
        self.grid_native_compare = str2bool(grid_native_compare)
        self.grid_native = str2bool(grid_native)
        if self.grid_native and self.grid_native_compare:
            raise ValueError("Choose either grid_native or grid_native_compare")
        if not horizontal_crs(crs).is_geographic:
            raise ValueError(
                "Frozen Coast Refine science requires a geographic evidence grid"
            )

    def _source_grid(self, entry):
        """Validate an existing *per-entry* upstream side-stack, never a fused DEM."""
        from .source_grid_binding import bind_source_grid

        binding = bind_source_grid(entry, expected_crs=self.crs)
        entry["_coast_mask_source_grid_binding"] = binding
        return Path(binding["source_path"])

    @staticmethod
    def _feed_source_grid(acc, source_path):
        """Reduce source-specific FusionState with Globato's own stacking rules."""
        with rasterio.open(source_path) as src:
            if (
                src.crs != acc.dataset.crs
                or src.transform != acc.dataset.transform
                or (src.height, src.width) != (acc.dataset.height, acc.dataset.width)
            ):
                raise ValueError(
                    "Source grid and authority reducer must share exact alignment"
                )
            for _, window in src.block_windows(1):
                acc.update_raster_state(
                    (
                        window,
                        window,
                        src.read(window=window),
                        None,
                        src.window_transform(window),
                    )
                )

    def _publish_unmasked(self, module, entry, artifact):
        source = self._source_grid(entry)
        with rasterio.open(source) as ds:
            cells = 0
            observations = 0.0
            for _, window in ds.block_windows(2):
                count = ds.read(2, window=window)
                cells += int((count > 0).sum())
                observations += float(count.sum())
        counts = dict(
            before_cells=cells,
            after_cells=cells,
            removed_cells=0,
            before_observations=observations,
            after_observations=observations,
            removed_observations=0.0,
        )
        from .source_receipt import publish_accepted_source

        publish_accepted_source(module, entry, source, dict(counts=counts), artifact)

    def _finish_grid_consumers(self, entries, prepared):
        """Mask requested consumer side-stacks, keeping native identities separate."""
        from .grid_native_fusion import mask_fusion_state

        seen_source_grids = {}
        for module, entry in entries:
            request = entry.get("coast_refine_consumer")
            if not request:
                continue
            group = request.get("authority") or request.get("group")
            if group not in prepared:
                raise ValueError(
                    "Grid consumer has no completed authority group: " + str(group)
                )
            remove = request.get("remove")
            if remove not in MODES:
                raise ValueError("Grid consumer has an invalid removal policy")
            source = self._source_grid(entry)
            # Separate class/weight branches must never accidentally share a
            # single pre-mask side-stack, even when the input filename matches.
            # A shared state would mix source filtering policies before masking.
            if source in seen_source_grids:
                raise ValueError(
                    "Two Coast Mask consumer entries share the same raw side-stack "
                    "(possibly different classes/weights): %s; earlier entry=%s, "
                    "current entry=%s"
                    % (
                        source,
                        seen_source_grids[source],
                        entry.get("contribution_id") or entry.get("dst_fn"),
                    )
                )
            seen_source_grids[source] = entry.get("contribution_id") or entry.get(
                "dst_fn"
            )
            artifact = prepared[group]
            # The producer, native TIFF writer and every consumer share the
            # sealed SOURCE retention policy. Interpolation topology may have
            # signed additions and a different cell lattice; never use it to
            # decide whether another source observation survives.
            keep = artifact.keep_geometry(remove, artifact.scope_for(module))
            from ._io import identity, json_write

            token = identity(
                {
                    "source": request.get("source")
                    or {
                        "original_file": str(entry.get("dst_fn")),
                        "registered_metadata": request.get(
                            "registered_source_metadata"
                        ),
                    },
                    "hash": digest(source),
                    "geometry": artifact.manifest_hash,
                    "remove": remove,
                    "original": request.get("source"),
                    "contribution": serializable(entry.get("contribution_id")),
                }
            )
            root = Path(request["output_dir"]) / "grid_native_states"
            output = root / (token + ".fusion.tif")
            receipt = mask_fusion_state(
                source,
                output,
                keep,
                "EPSG:4326",
                source_id=str(entry.get("dst_fn")),
                group=group,
                validated_binding=entry["_coast_mask_source_grid_binding"],
            )
            # Live entry metadata may change during stream-init and LAS reads.
            # This reports the registration-stage branch, and rejects conflicts.
            receipt.update(
                status="completed",
                remove=remove,
                authority=group,
                source_entry_metadata=serializable(
                    _consumer_receipt_metadata(request, entry)
                ),
                class_branch_claims=list(request.get("class_branch_claims") or []),
                coast_mask_manifest=str(artifact.path),
                application_geometry_selection="source_retention_native_centres",
            )
            json_write(output.with_suffix(".json"), receipt)
            entry.setdefault("artifacts", {})["side-stack"] = str(output)
            entry["coast_processing_id"] = token
            set_contribution_id(entry, "coast_mask_grid", token)
            entry["coast_refine_receipt"] = str(output.with_suffix(".json"))
            from .source_receipt import publish_accepted_source

            publish_accepted_source(module, entry, output, receipt, artifact)
            entry.pop("stream", None)
            entry.pop("raster_stream", None)
            log.info(
                "Grid Coast Mask consumer %s remove=%s retained=%d/%d cells",
                str(entry.get("dst_fn")),
                remove,
                receipt["counts"]["after_cells"],
                receipt["counts"]["before_cells"],
            )

    def _prepare(self, selected, cache):
        from .prepare import prepare_from_baseline
        from ._science.extract_raw_surface import copy_band_one

        first = selected[0][1]["coast_refine_request"]
        for module, entry in selected:
            request = entry["coast_refine_request"]
            check_source(module, entry, request)
            if self.grid_native:
                if not request.get("grid_native") or request.get("points"):
                    raise ValueError(
                        "Grid-native producer must not have a GTPC checkpoint"
                    )
                self._source_grid(entry)
            else:
                if (
                    not request.get("points")
                    or digest(request["points"]) != request["points_sha256"]
                ):
                    raise ValueError("Missing/incomplete point capture")
                if CRS.from_user_input(request["stream_srs"]) != CRS.from_user_input(
                    self.crs
                ):
                    raise ValueError("Captured stream CRS differs from evidence CRS")
        scopes = [list(module.region) for module, _ in selected]
        if not all(scope == scopes[0] for scope in scopes):
            raise ValueError("One Coast Refine group requires one recipe region")
        # The evidence cache manifest is derived data. Never add its grid
        # digests to the registered request dictionaries: their identity was
        # sealed by the per-entry source-grid sink earlier in this run.
        from copy import deepcopy

        from .geometry_crs import operation_record, geometry_environment

        region = selected[0][0].region
        geometry_operation = operation_record(
            self.crs,
            4326,
            ((region.xmin + region.xmax) / 2, (region.ymin + region.ymax) / 2),
        )
        request = {
            "geometry_operation": geometry_operation,
            "geometry_environment": geometry_environment(),
            # Separate named groups never share a preparation cache even when
            # their source lists and polygons happen to be numerically equal.
            "group": first["group"],
            "sources": [deepcopy(item["coast_refine_request"]) for _, item in selected],
            "grid_native": self.grid_native,
            "res": self.res,
            "crs": self.crs,
            "tiers": self.resolutions,
            "region": scopes[0],
            "connected_outlet_lonlat": first.get("outlet_lonlat"),
            "mask_files": None
            if first["mask"] == "osm"
            else vector_files(first["mask"]),
            "code": {
                str(p.relative_to(Path(__file__).parent)): digest(p)
                for p in sorted(Path(__file__).parent.rglob("*.py"))
                if p.name not in {"extension.py", "shore_profile.py"}
            },
        }
        # Output/review locations do not influence classification or cache reuse.
        if self.grid_native:
            for (_, entry), snapshot in zip(selected, request["sources"]):
                source_state = Path(entry["artifacts"]["side-stack"])
                snapshot["source_grid_sha256"] = digest(source_state)
                snapshot["source_grid_identity"] = entry.get("_sidestack_hash")
        request["sources"] = [
            {
                k: v
                for k, v in s.items()
                if k
                not in (
                    "output_dir",
                    "review",
                    "viz",
                    "points",
                    "remove",
                    "barrier",
                    "group",
                    "barrier_extension_m",
                    "barrier_extension_min_z",
                    "barrier_extension_method",
                    "barrier_slope_window_m",
                    "outlet_lonlat",
                )
            }
            for s in request["sources"]
        ]
        root = cache / "coast_refine" / "native_preparations"
        root.mkdir(parents=True, exist_ok=True)
        target = root / identity(request)
        with FileLock(str(target) + ".lock"):
            if target.exists():
                return MaskArtifact(target)
            # A validated artifact is committed outside this scratch directory.
            # Busy/late log files on shared filesystems must not replace the
            # actual worker outcome with a scratch-cleanup exception.
            with tempfile.TemporaryDirectory(
                prefix="prepare_", dir=root, ignore_cleanup_errors=True
            ) as temporary:
                work = Path(temporary)
                weights = "/".join(
                    map(
                        str,
                        sorted(
                            {
                                float(m.weight) * float(e.get("weight", 1))
                                for m, e in selected
                            }
                        ),
                    )
                )
                reducer = accumulator(
                    work / "evidence.fusion.tif",
                    selected[0][0].region,
                    self.res,
                    self.crs,
                    strategy="mixed",
                    weights=weights,
                )
                try:
                    seen_original_states = set()
                    for _, entry in selected:
                        if self.grid_native:
                            source_state = self._source_grid(entry)
                            if source_state in seen_original_states:
                                raise ValueError(
                                    "Duplicate original source grid in authority: "
                                    + str(source_state)
                                )
                            seen_original_states.add(source_state)
                            self._feed_source_grid(reducer, source_state)
                        else:
                            from globato.streams.readers.gtpc import iter_gtpc_chunks

                            for chunk in iter_gtpc_chunks(
                                entry["coast_refine_request"]["points"]
                            ):
                                reducer.update(chunk)
                    reducer.finalize()
                finally:
                    reducer.close()
                original = work / "unmasked.tif"
                copy_band_one(reducer.output_fn, original)
                _require_valid_evidence(original)
                evidence, _ = pad_evidence_for_native_tiers(original, self.resolutions)
                module = selected[0][0]
                scope = (
                    gpd.GeoSeries(
                        [
                            box(
                                module.region.xmin,
                                module.region.ymin,
                                module.region.xmax,
                                module.region.ymax,
                            )
                        ],
                        crs=horizontal_crs(module.region.srs or 4326),
                    )
                    .to_crs(4326)
                    .iloc[0]
                )
                native_roi = (
                    gpd.GeoSeries(
                        [
                            box(
                                module.region.xmin,
                                module.region.ymin,
                                module.region.xmax,
                                module.region.ymax,
                            )
                        ],
                        crs=horizontal_crs(module.region.srs or 4326),
                    )
                    .to_crs(horizontal_crs(self.crs))
                    .total_bounds
                )
                with rasterio.open(evidence) as grid:
                    native_bounds = [
                        min(native_roi[0], grid.bounds.left),
                        min(native_roi[1], grid.bounds.bottom),
                        max(native_roi[2], grid.bounds.right),
                        max(native_roi[3], grid.bounds.top),
                    ]
                    scope = preparation_scope_with_tier_support(
                        scope, grid.bounds, grid.crs
                    )
                area = work / "area.gpkg"
                gpd.GeoDataFrame(geometry=[scope], crs=4326).to_file(area)
                try:
                    artifact = prepare_from_baseline(
                        evidence,
                        area,
                        work / "artifact",
                        cache,
                        mask=first["mask"],
                        polygon_class=first["polygon_class"],
                        provenance=request,
                        native_bounds=native_bounds,
                    )
                except Exception as error:
                    from .review import preserve_failed_preparation

                    preserve_failed_preparation(
                        work, first["output_dir"], request, error
                    )
                    raise
                # Re-publish the independent connected-water companion if the
                # caller provides an unambiguous outlet. This never changes
                # existing refined_water, masks, or classed topology.
                from .connected_water import publish_connected_water

                connected_path = artifact.root / "connected_water.json"
                if first.get("outlet_lonlat") is None and connected_path.exists():
                    import json

                    connected_receipt = json.loads(connected_path.read_text())
                else:
                    connected_receipt = publish_connected_water(
                        artifact.root, outlet_lonlat=first.get("outlet_lonlat")
                    )
                evidence_hashes = publish_preparation_evidence(
                    original, evidence, artifact.root
                )
                json_write(artifact.root / "frozen_sources.json", request)
                record = artifact.record
                record["preparation_inputs"].update(evidence_hashes)
                record["connected_water_selection"] = connected_receipt
                record["preparation_inputs"]["frozen_sources_sha256"] = digest(
                    artifact.root / "frozen_sources.json"
                )
                record["preparation_identity"] = identity(record["preparation_inputs"])
                json_write(artifact.path, record)
                MaskArtifact(artifact.root)
                artifact.root.rename(target)
        return MaskArtifact(target)

    def _compare_grid_entry(
        self,
        module,
        entry,
        *,
        artifact,
        keep_geometry,
        reference_state,
        output_dir,
        group,
        source_label,
    ):
        """Compare the *same* normalized observations gridded before masking.

        The existing GTPC state remains the authority for production. This
        per-entry original state is made with Globato's existing accumulator,
        not a new gridding kernel or fake point reconstruction.
        """
        from globato.streams.readers.gtpc import iter_gtpc_chunks
        from .grid_native_fusion import (
            mask_fusion_state,
            compare_fusion_states,
            write_comparison_report,
        )

        request = entry.get("coast_refine_request") or entry.get(
            "coast_refine_consumer"
        )
        if not request or not request.get("points"):
            raise ValueError(
                "Comparison requires already captured original observations"
            )
        if not request.get("stream_srs"):
            raise ValueError("Original observation CRS not recorded")
        if CRS.from_user_input(request["stream_srs"]) != CRS.from_user_input(self.crs):
            raise ValueError(
                "Native grid comparison CRS differs from captured source stream"
            )
        from ._io import identity

        root = Path(output_dir) / "grid_native_compare"
        key = identity(
            {
                "label": source_label,
                "source": request.get("source"),
                "points_sha256": request.get("points_sha256"),
                "group": group,
                "res": self.res,
                "crs": self.crs,
                "artifact": artifact.manifest_hash,
                "remove": request.get("remove"),
            }
        )
        source_state = root / "original" / (key + ".fusion.tif")
        masked_state = root / "masked" / (key + ".fusion.tif")
        report_path = root / "reports" / (key + ".json")
        source_state.parent.mkdir(parents=True, exist_ok=True)
        reducer = accumulator(
            source_state,
            module.region,
            self.res,
            self.crs,
            strategy=self.strategy,
            weights=self.weight_threshold,
        )
        try:
            for chunk in iter_gtpc_chunks(request["points"]):
                reducer.update(chunk)
            reducer.dataset.update_tags(
                COAST_MASK_GRID_ROLE="source_unfilled",
                COAST_MASK_SOURCE_ID=source_label,
                COAST_MASK_GROUP=group,
            )
        finally:
            reducer.close()
        receipt = mask_fusion_state(
            source_state,
            masked_state,
            keep_geometry,
            "EPSG:4326",
            source_id=source_label,
            group=group,
        )
        result = compare_fusion_states(masked_state, reference_state)
        result.update(
            group=group,
            source_label=source_label,
            masked_grid_receipt=receipt,
            mode="diagnostic_only",
            production_changed=False,
            science_note=(
                "Pixel-center grid selection is not guaranteed to match "
                "point/native-raster-centre masking in mixed shoreline cells."
            ),
        )
        write_comparison_report(report_path, result)
        entry.setdefault("artifacts", {})["coast_grid_native_comparison"] = str(
            report_path
        )
        log.info(
            "Coast Mask grid compare %s: same_support=%s exact_bands=%s; %s",
            source_label,
            result["exact_support"],
            result["matched_within_tolerance"],
            report_path,
        )
        return result

    def run(self, entries):
        groups = {}
        for module, entry in entries:
            if request := entry.get("coast_refine_request"):
                groups.setdefault(request["group"], []).append((module, entry))
        from .groups import select_barrier_group, validate_consumer_groups

        group_requests = {}
        for group, selected in groups.items():
            configs = {
                identity(group_geometry_policy(e["coast_refine_request"]))
                for _, e in selected
            }
            if len(configs) != 1:
                raise ValueError(
                    "Coast Refine group %r has inconsistent geometry configuration"
                    % group
                )
            group_requests[group] = [
                entry["coast_refine_request"] for _, entry in selected
            ]
        binding_group = (
            None
            if self.defer_barrier
            else select_barrier_group(
                group_requests,
                binding_group=self.binding_group,
                require_barrier=self.require_barrier,
            )
        )
        # Both legacy consumers and native FusionState consumers must identify
        # their own group; the selected legacy *barrier* is not a default mask.
        validate_consumer_groups(groups, [entry for _, entry in entries])
        authorities = [binding_group] if binding_group is not None else []
        mask_groups = [
            group
            for group, selected in groups.items()
            if selected[0][1]["coast_refine_request"].get("source_hook") == "coast_mask"
        ]
        exports = {}
        prepared = {}
        source_overlap_reports = {}
        for group, selected in groups.items():
            # Preserve acquisition completion order. Priority ties follow the
            # same order as the native stream sink in this execution.
            caches = {
                str(Path(getattr(m, "shared_cache", None) or m._outdir).resolve())
                for m, _ in selected
            }
            if len(caches) != 1:
                raise ValueError("A refinement group requires one shared cache")
            phase_started = perf_counter()
            log.info(
                "Coast Refine %s: prepare/verify evidence for %d sources",
                group,
                len(selected),
            )
            artifact = self._prepare(selected, Path(next(iter(caches))))
            prepared[group] = artifact
            if self.grid_native and self.audit_overlaps:
                # Read-only evidence disagreement reporting. No new weighting
                # or precedence rule; the reducer above remains authoritative.
                from .source_overlap import measure_overlaps

                original_states = [self._source_grid(e) for _, e in selected]
                source_overlap_reports[group] = measure_overlaps(original_states)
            if group in mask_groups and group == binding_group:
                # Source masking and refinement are independent of ms_binary_cudem.
                # Publish reusable geometry plus the existing authority binding
                # needed by point_raster_mask on DAV and other modules.
                exports["coast_mask_manifest"] = str(artifact.path)
                exports["coast_mask_refined_water"] = str(
                    artifact.root / "refined_water.gpkg"
                )
                connected = artifact.record.get("connected_water_selection", {})
                if connected.get("status") == "selected":
                    exports["coast_mask_polygon"] = str(
                        artifact.root / "connected_water.gpkg"
                    )
                    exports["coast_mask_polygon_receipt"] = str(
                        artifact.root / "connected_water.json"
                    )
                else:
                    log.warning(
                        "coast_mask polygon unresolved for %s: %s; "
                        "no coast_mask_polygon binding was published",
                        group,
                        connected.get("reason", "unresolved"),
                    )
            log.info(
                "Coast Refine %s: evidence preparation/verification %.2fs",
                group,
                perf_counter() - phase_started,
            )
            if group == binding_group:
                # Keep the coast_mask artifact bindings published above: replacing
                # this dict would drop the public polygon and its manifest.
                exports.update(
                    {
                        "coast_refine_barrier": str(
                            artifact.products["refined_topology"]
                        ),
                        "coast_refine_manifest": str(artifact.path),
                    }
                )
                # Phase 1 companion: valid only when the connected component
                # was selected unambiguously; no implicit largest-part choice.
                connected = artifact.record.get("connected_water_selection", {})
                if connected.get("status") == "selected":
                    exports["coast_refine_connected_water"] = str(
                        artifact.root / "connected_water.gpkg"
                    )
                    exports["coast_refine_connected_receipt"] = str(
                        artifact.root / "connected_water.json"
                    )
                else:
                    log.warning(
                        "Coast Refine connected-water guidance unavailable: %s",
                        connected.get("reason", "no_selection"),
                    )
                policy = selected[0][1]["coast_refine_request"]
                exports["coast_refine_river_guidance"] = "off"
                if policy.get("source_hook") != "coast_mask":
                    from .extension import build, validate

                    policy = selected[0][1]["coast_refine_request"]
                    exports["coast_refine_river_guidance"] = policy.get(
                        "river_guidance", "off"
                    )
                    if (
                        policy.get("barrier_extension_method")
                        in ("slope", "slope_smooth")
                        and policy.get("barrier_extension_m", 0) > 0
                    ):
                        # Bathymetric support exists only after native stack_combine.
                        # Register its policy here; do not replay source processing.
                        exports["coast_refine_barrier_options"] = {
                            "distance": policy["barrier_extension_m"],
                            "minimum_z": policy["barrier_extension_min_z"],
                            "method": policy["barrier_extension_method"],
                            "slope_window_m": policy["barrier_slope_window_m"],
                        }
                        extension = None
                    else:
                        extension = build(
                            artifact,
                            policy.get("barrier_extension_m", 0),
                            policy.get("barrier_extension_min_z", 0.25),
                        )
                    if extension:
                        exports["coast_refine_barrier"] = str(
                            validate(artifact, extension)
                        )
                        exports["coast_refine_barrier_extension"] = str(extension)
                        log.info(
                            "Coast Refine interpolation-only extension: %s", extension
                        )
            for module, entry in selected:
                phase_started = perf_counter()
                request = entry["coast_refine_request"]
                check_source(module, entry, request)
                source = Path(request["source"]["source"])
                required = (
                    artifact.scope_for(module, None)
                    if self.grid_native
                    else artifact.check_source(source, artifact.scope_for(module, None))
                )
                if required is None:
                    raise ValueError(
                        "Selected source does not overlap the prepared region"
                    )
                provenance = {
                    "source_identity": {
                        "url": entry.get("url"),
                        "source_id": entry.get("source_id"),
                        "module": module.name,
                        "native_name": source.name,
                    },
                    "source_snapshot": request["source"],
                    "source_weight": entry.get("weight", 1),
                    "module_weight": module.weight,
                    "claims": serializable(
                        {
                            k: entry.get(k)
                            for k in (
                                "claim_geometry",
                                "accepted_geometry",
                                "excluded_geometry",
                                "claim_priority",
                            )
                        }
                    ),
                    "remove": request["remove"],
                    "polygon_class": request["polygon_class"],
                    "mask_sha256": artifact.manifest_hash,
                    "application_geometry_selection": "source_retention_native_centres",
                    "adapter_sha256": digest(__file__),
                    "scope_wkb": required.normalize().wkb_hex,
                }
                if (
                    self.grid_native
                    and request.get("global_refinement")
                    and request["remove"] is None
                ):
                    self._publish_unmasked(module, entry, artifact)
                    continue
                if self.grid_native:
                    from .grid_native_fusion import mask_fusion_state

                    source_state = self._source_grid(entry)
                    token = identity(
                        {
                            "source": request["source"],
                            "raw_grid_hash": digest(source_state),
                            "geometry": artifact.manifest_hash,
                            "remove": request["remove"],
                            "group": group,
                        }
                    )
                    output = (
                        Path(request["output_dir"])
                        / "grid_native_states"
                        / (token + ".fusion.tif")
                    )
                    grid_receipt = mask_fusion_state(
                        source_state,
                        output,
                        artifact.keep_geometry(request["remove"], required),
                        "EPSG:4326",
                        source_id=source.name,
                        group=group,
                        validated_binding=entry["_coast_mask_source_grid_binding"],
                    )
                    grid_receipt.update(
                        status="completed",
                        remove=request["remove"],
                        coast_mask_manifest=str(artifact.path),
                        source_snapshot=request["source"],
                        processing_id=token,
                        application_geometry_selection="source_retention_native_centres",
                    )
                    json_write(output.with_suffix(".json"), grid_receipt)
                    entry["coast_processing_id"] = token
                    set_contribution_id(entry, "coast_mask_grid", token)
                    entry.setdefault("artifacts", {})["side-stack"] = str(output)
                    entry["artifacts"]["coast_refine"] = str(output)
                    entry["coast_refine_receipt"] = str(output.with_suffix(".json"))
                    from .source_receipt import publish_accepted_source

                    publish_accepted_source(
                        module, entry, output, grid_receipt, artifact
                    )
                    entry.pop("stream", None)
                    log.info(
                        "Grid Coast Mask producer %s retained=%d/%d cells; no GTPC replay",
                        source.name,
                        grid_receipt["counts"]["after_cells"],
                        grid_receipt["counts"]["before_cells"],
                    )
                    continue
                receipt = publish_masked_source(
                    source,
                    artifact.keep_geometry(request["remove"], required),
                    Path(request["output_dir"]) / "derived_sources",
                    provenance=provenance,
                    mask_manifest=artifact.path,
                    preparation_id=artifact.preparation_id,
                    source_files=request["source"]["files"],
                )
                log.info(
                    "Coast Refine %s: native mask publication/validation %.2fs",
                    source.name,
                    perf_counter() - phase_started,
                )
                entry["coast_original_dst_fn"] = str(source)
                entry["dst_fn"] = receipt["masked"]
                entry["coast_refine_receipt"] = str(
                    Path(receipt["masked"]).with_suffix(".json")
                )
                entry["coast_processing_id"] = receipt["processing_id"]
                set_contribution_id(entry, "coast_refine", receipt["processing_id"])
                entry.setdefault("artifacts", {})["coast_refine"] = receipt["masked"]
                entry["coast_refine_review"] = dict(
                    manifest=str(artifact.path),
                    receipt=receipt,
                    enabled=request["review"],
                    root=request["output_dir"],
                )
                state = (
                    Path(request["output_dir"])
                    / "states"
                    / (
                        identity(
                            {
                                "processing": receipt["processing_id"],
                                "points": request["points_sha256"],
                                "res": self.res,
                                "crs": self.crs,
                            }
                        )
                        + ".fusion.tif"
                    )
                )
                state.parent.mkdir(parents=True, exist_ok=True)
                # Rebuild from exact saved samples, never mask coarse statistics.
                state_started = perf_counter()
                log.info(
                    "Coast Refine %s: rebuild retained-sample FusionState", source.name
                )
                reducer = accumulator(
                    state,
                    module.region,
                    self.res,
                    self.crs,
                    strategy=self.strategy,
                    weights=self.weight_threshold,
                )
                try:
                    for chunk in retained_points(
                        request["points"], receipt["masked"], request["stream_srs"]
                    ):
                        reducer.update(chunk)
                    reducer.dataset.update_tags(
                        GLOBATO_QC_HASH=receipt["processing_id"]
                    )
                finally:
                    reducer.close()
                # Do not expose an interrupted or unpersisted retained state to
                # stack_combine. This checks only metadata, not all raster cells.
                with rasterio.open(state) as completed:
                    if (
                        completed.tags().get("GLOBATO_QC_HASH")
                        != receipt["processing_id"]
                    ):
                        raise ValueError(
                            "Retained-source FusionState is incomplete: " + str(state)
                        )
                if self.grid_native_compare and not any(
                    "coast_grid_native_comparison" in e.get("artifacts", {})
                    for _, e in entries
                ):
                    try:
                        self._compare_grid_entry(
                            module,
                            entry,
                            artifact=artifact,
                            keep_geometry=artifact.keep_geometry(
                                request["remove"], required
                            ),
                            reference_state=state,
                            output_dir=request["output_dir"],
                            group=group,
                            source_label=source.name,
                        )
                    except Exception as exc:
                        # Diagnostics must never replace or invalidate the
                        # already published reference science. Record failure.
                        from .grid_native_fusion import write_comparison_report

                        log.exception("Grid-native producer comparison failed: %s", exc)
                        write_comparison_report(
                            Path(request["output_dir"])
                            / "grid_native_compare"
                            / "reports"
                            / "producer_not_comparable.json",
                            {
                                "status": "not_comparable",
                                "group": group,
                                "source": source.name,
                                "reason": str(exc),
                                "production_changed": False,
                            },
                        )
                entry["artifacts"]["side-stack"] = str(state)
                entry.pop("stream", None)
                entry["coast_refine_processing_timing"] = {
                    "mask_and_state_seconds": perf_counter() - phase_started,
                    "retained_state_seconds": perf_counter() - state_started,
                }
                log.info(
                    "Coast Refine %s: retained-sample FusionState %.2fs",
                    source.name,
                    perf_counter() - state_started,
                )
                log.info("Coast Refine group %s: %s", group, receipt["masked"])
        if self.grid_native:
            self._finish_grid_consumers(entries, prepared)
        else:
            from .consumer import finish_consumers

            finish_consumers(entries, prepared, authorities, res=self.res, crs=self.crs)
        if self.grid_native_compare and prepared:
            # One DAV/other point consumer: intentionally compare diagnostics
            # against the REAL accepted side-stack, never replace it.
            for module, entry in entries:
                consumer = entry.get("coast_refine_consumer")
                if not consumer or not consumer.get("points"):
                    continue
                group = consumer.get("authority") or consumer.get("group")
                if group not in prepared:
                    continue
                state = entry.get("artifacts", {}).get("side-stack")
                if not state or not Path(state).exists():
                    continue
                remove = consumer.get("remove")
                if remove not in ("land", "water"):
                    continue
                artifact = prepared[group]
                polygon = artifact.root / "refined_water.gpkg"
                # Compare against the authoritative refined water geometry.
                # It is intentionally independent of individual LAZ returns.
                from .grid_native_fusion import write_comparison_report

                source_label = str(entry.get("dst_fn") or "consumer")
                report_root = Path(consumer["output_dir"]) / "grid_native_compare"
                try:
                    water = gpd.read_file(polygon).to_crs(4326).geometry.union_all()
                    if remove == "land":
                        keep = water
                    else:
                        from shapely.geometry import box as geom_box

                        area = geom_box(
                            module.region.xmin,
                            module.region.ymin,
                            module.region.xmax,
                            module.region.ymax,
                        )
                        keep = area.difference(water)
                    self._compare_grid_entry(
                        module,
                        entry,
                        artifact=artifact,
                        keep_geometry=keep,
                        reference_state=state,
                        output_dir=consumer["output_dir"],
                        group=group,
                        source_label=source_label,
                    )
                    break
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    # A diagnostic should not invalidate a scientifically
                    # successful production run. Make the omission explicit.
                    log.warning("DAV grid-native comparison unavailable: %s", exc)
                    report_root.mkdir(parents=True, exist_ok=True)
                    write_comparison_report(
                        report_root / "consumer_not_comparable.json",
                        {
                            "status": "not_comparable",
                            "source": source_label,
                            "reason": str(exc),
                            "production_changed": False,
                        },
                    )
                    break
        # Publication is committed only AFTER source masks and native consumers
        # successfully complete. The receipt makes the reusable geometry easy
        # to find in the run directory, even when the vectors live in the cache.
        from .groups import group_token, publication_record

        group_publications = {}
        for group in mask_groups:
            artifact = prepared[group]
            source_entry = groups[group][0][1]
            out_dir = Path(source_entry["coast_refine_request"]["output_dir"])
            publication = (
                out_dir / "groups" / group_token(group) / "coast_mask_publication.json"
            )
            record = publication_record(group, artifact, binding_group=binding_group)
            if self.grid_native:
                record["source_evidence_overlap"] = source_overlap_reports.get(
                    group, []
                )
                record["source_evidence_overlap_status"] = (
                    "completed" if self.audit_overlaps else "not_requested"
                )
            publication.parent.mkdir(parents=True, exist_ok=True)
            json_write(publication, record)
            group_publications[group] = {**record, "publication": str(publication)}
            if len(mask_groups) == 1:
                # Backward-compatible single-group alias only. Never use a
                # global filename for multiple independent groups.
                old_publication = out_dir / "coast_mask_publication.json"
                json_write(
                    old_publication,
                    {
                        "schema": "coast_mask_publication.v1",
                        "status": "completed",
                        "group": group,
                        "manifest": record["manifest"],
                        "refined_water": record["refined_water"],
                        "connected_water": record["connected_water"],
                        "connection_status": record["connection_status"],
                        "connection_reason": record["connection_reason"],
                        "native_consumer_binding": "coast_refine",
                    },
                )
                exports["coast_mask_publication"] = str(old_publication)
        if group_publications:
            # The index is *not* a default mask. Consumers select their own
            # authority by name. Write only after all groups/consumers succeed.
            index = (
                Path(groups[mask_groups[0]][0][1]["coast_refine_request"]["output_dir"])
                / "coast_mask_groups_index.json"
            )
            json_write(
                index,
                {
                    "schema": "coast_mask_group_index.v1",
                    "status": "completed",
                    "selected_legacy_barrier_group": binding_group,
                    "groups": group_publications,
                },
            )
            # Only publish file paths through entry["artifacts"]. The index
            # already contains each group's publication path; exposing the
            # group->path dictionary here breaks file-oriented collection
            # consumers such as Fetchez copy_artifact (Path(dict) TypeError).
            exports["coast_mask_groups_index"] = str(index)
        for _, entry in entries:
            entry.setdefault("artifacts", {}).update(exports)
        # Optional presentation-only declarations: completed files after the
        # existing science/consumer validation. Never affect source selection,
        # fused samples, interpolation, or barriers.
        for group, selected in groups.items():
            if not any(
                e["coast_refine_request"].get("viz", False) for _, e in selected
            ):
                continue
            if self.grid_native:
                from .source_products import publish_grid_products

                catalog = publish_grid_products(
                    group,
                    entries,
                    res=self.res,
                    crs=self.crs,
                    strategy=self.strategy,
                    weight_threshold=self.weight_threshold,
                    artifact=prepared[group],
                )
                for _, entry in selected:
                    entry.setdefault("artifacts", {})["coast_mask_products"] = str(
                        catalog
                    )
                continue
            try:
                from .viz_products import publish_source_group

                catalog = publish_source_group(
                    group, selected, prepared[group], deferred=(group in authorities)
                )
                for _, entry in selected:
                    entry["coast_refine_viz_catalog"] = str(catalog)
                log.info("COAST_REFINE_PRODUCTS %s", catalog)
            except Exception:
                log.exception(
                    "Optional Coast Refine visualization publication failed; source refinement remains valid"
                )
        return entries


class CoastRefineReview(FetchHook):
    """Optional, separately runnable rendering from completed artifacts."""

    name = "coast_refine_review"
    meta_stage = "collection"
    meta_category = "visualization"

    def run(self, entries):
        from .review import render_source_review, write_review_index

        groups = {}
        for _, entry in entries:
            record = entry.get("coast_refine_review")
            if record and record["enabled"]:
                groups.setdefault((record["manifest"], record["root"]), []).append(
                    record["receipt"]
                )
        for (manifest, output), receipts in groups.items():
            artifact = MaskArtifact(manifest)
            root = Path(output) / "review"
            receipt_id = identity(sorted(r["processing_id"] for r in receipts))[:8]
            target = (
                root
                / ("coast_refine_" + artifact.preparation_id[:12] + "_" + receipt_id)
                / "sources"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            json_write(
                target.parent / "sources.json",
                dict(mask_manifest=manifest, receipts=receipts),
            )
            render_source_review(artifact, receipts, target)
            write_review_index(root)
            log.info("Coast Refine spatial review: %s", root / "review.html")
        return entries
