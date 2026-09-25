#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""globato.hooks.sinks.multi_stack

Persistent, provenance-aware reduction of Globato FusionState.

The important boundary in this module is:

    elevation points -> PointPixels -> local FusionState
    FusionState      -> MultiStackAccumulator -> global FusionState
    global FusionState -> finalize -> MultiStack

The on-disk state is *not* a finalized MultiStack. It contains the same
associative sufficient statistics emitted by ``PointPixels`` and can therefore
be resumed, merged, cached, or supplied directly by side-stack without changing
results merely because data were chunked or serialized.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from typing import Mapping

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
from rasterio.windows import Window

from fetchez.hooks import FetchHook
from fetchez.spatial import Region
from fetchez.utils import BOLD, CYAN, colorize, format_dataset_id, str2bool

from ..transforms.point_pixels import (
    FUSION_BAND_MAP,
    FUSION_BANDS,
    FUSION_STATE_VERSION,
    PointPixels,
    finalize_fusion_state,
)
from globato import __version__

logger = logging.getLogger(__name__)

INIT_LOCK = threading.Lock()


# -----------------------------------------------------------------------------
# Finalized MultiStack schema
# -----------------------------------------------------------------------------

# The persisted state uses FUSION_BANDS. The final, user-facing MultiStack uses
# derived values. Keeping these schemas distinct prevents a finalized raster
# from being mistaken for an updateable accumulation state.
MULTISTACK_BANDS = (
    "z",
    "count",
    "weight",
    "uncertainty",
    "stddev",
    "x",
    "y",
)
MULTISTACK_BAND_MAP = {name: index + 1 for index, name in enumerate(MULTISTACK_BANDS)}

VALID_STRATEGIES = {"mean", "weighted_mean", "mixed", "supercede"}


class MultiStackAccumulator:
    """Reduce local FusionState into a persistent global FusionState.

    Parameters
    ----------
    strategy
        Controls *source-to-source* stacking policy. This is deliberately
        distinct from point aggregation, which is always handled by
        :class:`PointPixels` as associative FusionState accumulation.

        ``mean`` / ``weighted_mean``
            Add every incoming FusionState.
        ``supercede``
            Replace existing state where incoming mean point weight is higher.
        ``mixed``
            Compare source weight tiers. Replace lower tiers and merge equal
            tiers.

    state_fn
        Optional persistent FusionState TIFF. If omitted, a temporary state
        file is used for a one-shot build.

    resume
        If True and ``state_fn`` already exists, validate and reuse it.
        Otherwise initialize a fresh state.
    """

    STATE_TAG = "GLOBATO_DATATYPE"
    STATE_TYPE = "FUSION_STATE"
    PROVENANCE_TAG = "GLOBATO_PROVENANCE"

    def __init__(
        self,
        region,
        x_inc,
        y_inc,
        output_fn,
        *,
        strategy="mean",
        weight_threshold="1",
        crs="EPSG:4326",
        reset_masks=False,
        compress_state=True,
        verbose=False,
        state_fn=None,
        resume=True,
    ):
        strategy = strategy.lower()
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Unknown MultiStack strategy {strategy!r}; "
                f"expected one of {sorted(VALID_STRATEGIES)}"
            )

        self.region = Region.from_list(region)
        self.x_inc = abs(float(x_inc))
        self.y_inc = abs(float(y_inc))
        self.output_fn = output_fn
        self.strategy = strategy
        self.crs = crs
        self.verbose = verbose
        self.reset_masks = reset_masks
        self.compress_state = str2bool(compress_state)
        self.resume = bool(resume)

        self.lock = threading.Lock()
        self.mask_registry = {}
        self.wts = np.sort([float(x) for x in str(weight_threshold).split("/")])

        self.xcount, self.ycount, self.dst_gt = self.region.geo_transform(
            x_inc=self.x_inc,
            y_inc=self.y_inc,
            node="grid",
        )
        self.transform = rasterio.transform.from_origin(
            self.dst_gt[0],
            self.dst_gt[3],
            self.dst_gt[1],
            abs(self.dst_gt[5]),
        )

        # A caller-supplied state filename means the state is an intentional,
        # persistent artifact. Otherwise keep one-shot behavior by using a
        # temporary path.
        self._persistent_state = state_fn is not None
        if state_fn is None:
            base = os.path.splitext(os.path.basename(output_fn))[0]
            tmp_dir = os.path.abspath("tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            fd, state_fn = tempfile.mkstemp(
                prefix=f"{base}_",
                suffix=".fusion.tif",
                dir=tmp_dir,
            )
            os.close(fd)
            # rasterio must create the file itself.
            os.remove(state_fn)

        self.state_fn = os.path.abspath(state_fn)

        self.pixel_binner = PointPixels(
            src_region=self.region,
            x_size=self.xcount,
            y_size=self.ycount,
        )

        self._init_state_raster()
        self.dataset = rasterio.open(self.state_fn, "r+")

        logger.info(
            "Initialized MultiStack FusionState at %s/%s for %s",
            self.xcount,
            self.ycount,
            self.region,
        )

    # ------------------------------------------------------------------
    # State-file lifecycle and validation
    # ------------------------------------------------------------------

    def _state_profile(self):
        profile = {
            "driver": "GTiff",
            "dtype": "float64",
            "nodata": None,
            "width": self.xcount,
            "height": self.ycount,
            "count": len(FUSION_BANDS),
            "crs": CRS.from_string(self.crs) if self.crs else None,
            "transform": self.transform,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "bigtiff": "YES",
            "interleave": "band",
        }
        if self.compress_state:
            profile["compress"] = "lzw"
            profile["predictor"] = 3  # floating-point predictor
        return profile

    def _validate_existing_state(self, src):
        problems = []

        if src.width != self.xcount or src.height != self.ycount:
            problems.append(
                f"dimensions {src.width}x{src.height} != {self.xcount}x{self.ycount}"
            )
        if src.count != len(FUSION_BANDS):
            problems.append(f"band count {src.count} != {len(FUSION_BANDS)}")
        if src.transform != self.transform:
            problems.append("geotransform differs")

        expected_crs = CRS.from_string(self.crs) if self.crs else None
        if src.crs != expected_crs:
            problems.append(f"CRS {src.crs!s} != {expected_crs!s}")

        tags = src.tags()
        if tags.get(self.STATE_TAG) != self.STATE_TYPE:
            problems.append(
                f"{self.STATE_TAG}={tags.get(self.STATE_TAG)!r}, "
                f"expected {self.STATE_TYPE!r}"
            )
        if int(tags.get("GLOBATO_FUSION_VERSION", -1)) != FUSION_STATE_VERSION:
            problems.append(
                "FusionState schema version differs "
                f"({tags.get('GLOBATO_FUSION_VERSION')!r} != "
                f"{FUSION_STATE_VERSION})"
            )

        descriptions = tuple(src.descriptions)
        if descriptions != FUSION_BANDS:
            problems.append(f"band descriptions {descriptions!r} != {FUSION_BANDS!r}")

        if problems:
            raise ValueError(
                f"Existing FusionState {self.state_fn!r} is incompatible: "
                + "; ".join(problems)
            )

    def _init_state_raster(self):
        os.makedirs(os.path.dirname(self.state_fn), exist_ok=True)

        if os.path.exists(self.state_fn):
            if self.resume:
                with rasterio.open(self.state_fn, "r") as existing:
                    self._validate_existing_state(existing)
                logger.info("Resuming FusionState: %s", self.state_fn)
                return

            os.remove(self.state_fn)

        with rasterio.open(self.state_fn, "w", **self._state_profile()) as dst:
            # Zero is the identity value for every additive state band.
            # GeoTIFF creation commonly initializes sparse/tiled data to zero,
            # but write explicit blocks so the contract is unambiguous.
            for _, window in dst.block_windows(1):
                zeros = np.zeros(
                    (len(FUSION_BANDS), int(window.height), int(window.width)),
                    dtype=np.float64,
                )
                dst.write(zeros, window=window)

            for name, idx in FUSION_BAND_MAP.items():
                dst.set_band_description(idx, name)

            dst.update_tags(
                **{
                    self.STATE_TAG: self.STATE_TYPE,
                    "GLOBATO_FUSION_VERSION": str(FUSION_STATE_VERSION),
                    "GLOBATO_VERSION": __version__,
                    self.PROVENANCE_TAG: "[]",
                }
            )

    # ------------------------------------------------------------------
    # Provenance registry
    # ------------------------------------------------------------------

    def is_registered(self, dataset_id):
        if not dataset_id or not hasattr(self, "dataset") or self.dataset.closed:
            return False

        with self.lock:
            registry = json.loads(self.dataset.tags().get(self.PROVENANCE_TAG, "[]"))
            return dataset_id in registry

    def mark_registered(self, dataset_id):
        if not dataset_id:
            return

        with self.lock:
            registry = json.loads(self.dataset.tags().get(self.PROVENANCE_TAG, "[]"))
            if dataset_id not in registry:
                registry.append(dataset_id)
                self.dataset.update_tags(**{self.PROVENANCE_TAG: json.dumps(registry)})

    # ------------------------------------------------------------------
    # Local state creation and global-state reduction
    # ------------------------------------------------------------------

    def update(
        self,
        points,
        dataset_id=None,
        *,
        source_weight=1.0,
        source_uncertainty=0.0,
    ):
        """Convert one point chunk to FusionState and reduce it into the stack."""

        if points is None or len(points) == 0:
            return

        state, sub_win, _ = self.pixel_binner.accumulate(
            points,
            source_weight=source_weight,
            source_uncertainty=source_uncertainty,
        )
        if sub_win is None or state["count"] is None:
            return

        col_off, row_off, width, height = sub_win
        self.update_state(
            state,
            Window(col_off, row_off, width, height),
            dataset_id=dataset_id,
        )

    def update_raster_state(self, chunk, dataset_id=None):
        """Reduce one FusionState raster-stream chunk without re-pixelizing it."""

        if not isinstance(chunk, tuple) or len(chunk) < 3:
            raise TypeError("FusionState raster chunks must be raster-stream tuples")

        window = chunk[0]
        data = np.asarray(chunk[2])
        if data.ndim != 3 or data.shape[0] != len(FUSION_BANDS):
            raise ValueError(
                f"Expected {len(FUSION_BANDS)}-band FusionState chunk; "
                f"got {data.shape!r}"
            )

        state = {
            name: np.asarray(data[index], dtype=np.float64)
            for index, name in enumerate(FUSION_BANDS)
        }
        self.update_state(state, window, dataset_id=dataset_id)

    @staticmethod
    def _state_to_array(state: Mapping[str, np.ndarray]) -> np.ndarray:
        missing = [name for name in FUSION_BANDS if name not in state]
        if missing:
            raise ValueError("Incomplete FusionState; missing " + ", ".join(missing))
        return np.stack([state[name] for name in FUSION_BANDS]).astype(
            np.float64,
            copy=False,
        )

    @staticmethod
    def _mean_weight(data):
        count = data[FUSION_BAND_MAP["count"] - 1]
        weight_sum = data[FUSION_BAND_MAP["weight_sum"] - 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(count > 0, weight_sum / count, 0.0)

    def update_state(self, state, window, dataset_id=None):
        """Reduce an already-binned FusionState into the global state.

        This is the core cache/resume API. Side-stack should call this method
        directly with deserialized FusionState rather than round-tripping the
        state through representative point records.
        """

        incoming = self._state_to_array(state)
        incoming_count = incoming[FUSION_BAND_MAP["count"] - 1]
        valid_new = incoming_count > 0
        if not np.any(valid_new):
            return

        with self.lock:
            current = self.dataset.read(window=window).astype(np.float64, copy=False)

            if self.strategy in {"mean", "weighted_mean"}:
                # Every FusionState band is additive by contract.
                current[:, valid_new] += incoming[:, valid_new]

            else:
                incoming_mean_weight = self._mean_weight(incoming)
                current_mean_weight = self._mean_weight(current)
                current_count = current[FUSION_BAND_MAP["count"] - 1]

                if self.strategy == "supercede":
                    replace = valid_new & (
                        (current_count == 0)
                        | (incoming_mean_weight > current_mean_weight)
                    )
                    merge = np.zeros_like(replace, dtype=bool)
                else:  # mixed
                    incoming_tier = np.digitize(incoming_mean_weight, self.wts)
                    current_tier = np.digitize(current_mean_weight, self.wts)
                    current_tier[current_count == 0] = -1

                    replace = valid_new & (incoming_tier > current_tier)
                    merge = valid_new & (incoming_tier == current_tier)

                if np.any(replace):
                    current[:, replace] = incoming[:, replace]
                    if self.reset_masks and dataset_id:
                        self._reset_older_masks(dataset_id, window, replace)

                if np.any(merge):
                    current[:, merge] += incoming[:, merge]

            self.dataset.write(current, window=window)

    # ------------------------------------------------------------------
    # Source-mask bookkeeping
    # ------------------------------------------------------------------

    def _reset_older_masks(self, current_dataset_id, window, superseded):
        if not np.any(superseded):
            return

        for old_id, old_tif in self.mask_registry.items():
            if (
                old_id == current_dataset_id
                or not old_tif
                or not os.path.exists(old_tif)
            ):
                continue

            with rasterio.open(old_tif, "r+") as mask_ds:
                mask_data = mask_ds.read(1, window=window)
                overlap = (mask_data == 1) & superseded
                if np.any(overlap):
                    mask_data[overlap] = 0
                    mask_ds.write(mask_data, 1, window=window)

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    @staticmethod
    def _state_from_array(data):
        return {name: data[index] for index, name in enumerate(FUSION_BANDS)}

    def finalize(self, ndv=-9999):
        """Derive a finalized MultiStack without mutating persisted state."""

        if self.dataset and not self.dataset.closed:
            self.dataset.close()

        if self.verbose:
            logger.debug(
                "Finalizing FusionState: %s -> %s",
                os.path.basename(self.state_fn),
                os.path.basename(self.output_fn),
            )

        with rasterio.open(self.state_fn, "r") as src:
            profile = src.profile.copy()
            profile.update(
                dtype="float32",
                nodata=ndv,
                count=len(MULTISTACK_BANDS),
                compress="lzw",
                predictor=2,
            )

            with rasterio.open(self.output_fn, "w", **profile) as dst:
                dst.colorinterp = [ColorInterp.undefined] * dst.count
                for name, idx in MULTISTACK_BAND_MAP.items():
                    dst.set_band_description(idx, name)

                for _, window in src.block_windows(1):
                    state_data = src.read(window=window)
                    final = finalize_fusion_state(self._state_from_array(state_data))
                    valid = final.valid

                    out = np.full(
                        (len(MULTISTACK_BANDS),) + final.z.shape,
                        ndv,
                        dtype=np.float32,
                    )

                    out[MULTISTACK_BAND_MAP["z"] - 1][valid] = final.z[valid]
                    out[MULTISTACK_BAND_MAP["count"] - 1][valid] = final.count[valid]
                    out[MULTISTACK_BAND_MAP["weight"] - 1][valid] = final.mean_weight[
                        valid
                    ]
                    out[MULTISTACK_BAND_MAP["uncertainty"] - 1][valid] = (
                        final.uncertainty[valid]
                    )
                    out[MULTISTACK_BAND_MAP["stddev"] - 1][valid] = final.stddev[valid]
                    out[MULTISTACK_BAND_MAP["x"] - 1][valid] = final.x[valid]
                    out[MULTISTACK_BAND_MAP["y"] - 1][valid] = final.y[valid]

                    dst.write(out, window=window)

                dst.update_tags(**src.tags())
                dst.update_tags(
                    GLOBATO_DATATYPE="MULTI_STACK",
                    GLOBATO_FUSION_VERSION=str(FUSION_STATE_VERSION),
                    VERSION=__version__,
                )

        # Keep statistics behavior from the existing implementation.
        with rasterio.open(self.output_fn, "r+") as dst:
            all_stats = dst.stats(approx=False)
            for index, stats in enumerate(all_stats, start=1):
                desc = MULTISTACK_BANDS[index - 1]
                dst.update_tags(
                    bidx=index,
                    STATISTICS_MINIMUM=str(stats.min),
                    STATISTICS_MAXIMUM=str(stats.max),
                    STATISTICS_MEAN=str(stats.mean),
                    STATISTICS_STDDEV=str(stats.std),
                    DESCRIPTION=desc,
                    GLOBATO_DATATYPE="MULTI_STACK",
                    GLOBATO_FUSION_VERSION=str(FUSION_STATE_VERSION),
                    GLOBATO_STACK_STRATEGY=self.strategy,
                    GLOBATO_WEIGHT_TIERS=json.dumps(self.wts.tolist()),
                    VERSION=__version__,
                )

        return self.output_fn

    def close(self):
        if hasattr(self, "dataset") and self.dataset and not self.dataset.closed:
            self.dataset.close()


class MultiStackHook(FetchHook):
    """Accumulate point or FusionState streams into a MultiStack.

    ``strategy`` describes source-to-source stacking policy. ``mode`` is
    accepted as a temporary compatibility alias but should not be used in new
    recipes.
    """

    name = "multi_stack"
    meta_stage = "stream"
    meta_category = "stream-sink"
    meta_desc = "Reduce elevation sources into a persistent Globato FusionState."
    meta_aliases = ["multi-stack"]

    def __init__(
        self,
        res="1s",
        output="multi_stack_output.tif",
        *,
        strategy=None,
        mode=None,
        weight_threshold="1",
        crs=None,
        drop_classes=None,
        reset_masks=True,
        state=None,
        resume=True,
        compress_state=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Backward-compatible migration path. New recipes should say strategy.
        if strategy is None:
            strategy = mode or "mean"
        elif mode is not None and mode.lower() != strategy.lower():
            raise ValueError(
                f"Conflicting MultiStack strategy={strategy!r} and mode={mode!r}"
            )

        self.res = res
        self.output = output
        self.strategy = strategy.lower()
        self.weight_threshold = weight_threshold
        self.crs = crs
        self.drop_classes = (
            [int(x) for x in str(drop_classes).split("/")] if drop_classes else []
        )
        self.reset_masks = str2bool(reset_masks)
        self.state = state
        self.resume = str2bool(resume)
        self.compress_state = str2bool(compress_state)
        self._accumulator = None

    def _init_accumulator(self, region):
        with INIT_LOCK:
            if self._accumulator:
                return

            if isinstance(self.res, str) and self.res.endswith("s"):
                inc = float(self.res[:-1]) / 3600.0
                x_inc, y_inc = inc, inc
            elif "/" in str(self.res):
                x_inc, y_inc = map(float, self.res.split("/"))
            else:
                inc = float(self.res)
                x_inc, y_inc = inc, inc

            logger.info(
                "Initializing MultiStack: %s @ %s,%s (%s)",
                self.output,
                x_inc,
                y_inc,
                self.strategy,
            )

            self._accumulator = MultiStackAccumulator(
                region=region,
                x_inc=x_inc,
                y_inc=y_inc,
                output_fn=self.output,
                strategy=self.strategy,
                weight_threshold=self.weight_threshold,
                crs=self.crs,
                reset_masks=self.reset_masks,
                compress_state=self.compress_state,
                verbose=True,
                state_fn=self.state,
                resume=self.resume,
            )

    @staticmethod
    def _dataset_id(entry):
        dataset_id = entry.get("checksum")
        if dataset_id:
            return dataset_id

        url = entry.get("url", "")
        dst_fn = entry.get("dst_fn")

        if url and not url.startswith("file://"):
            return url
        if dst_fn and os.path.exists(dst_fn):
            size = os.path.getsize(dst_fn)
            return f"{os.path.basename(dst_fn)}|{size}B"
        return os.path.basename(dst_fn or url or "unknown_dataset")

    @staticmethod
    def _is_fusion_state_chunk(chunk):
        return (
            isinstance(chunk, tuple)
            and len(chunk) >= 3
            and isinstance(chunk[2], np.ndarray)
            and chunk[2].ndim == 3
            and chunk[2].shape[0] == len(FUSION_BANDS)
        )

    def run(self, entries):
        if not self._accumulator:
            region = next(
                (mod.region for mod, _ in entries if getattr(mod, "region", None)),
                None,
            )
            if not region:
                return entries

            region_str = region.format("fn")
            base, ext = os.path.splitext(self.output)
            if region_str not in base:
                self.output = f"{base}_{region_str}{ext}"
            self._init_accumulator(region)

        for mod, entry in entries:
            dataset_id = self._dataset_id(entry)
            mask_path = entry.get("artifacts", {}).get("source-masks")

            if self._accumulator and self._accumulator.is_registered(dataset_id):
                logger.debug("Dataset %r already inside stack. Skipping.", dataset_id)
                entry.pop("stream", None)
                entry.pop("raster_stream", None)
            elif self.has_stream(entry):
                entry["stream"] = self._intercept(
                    entry.get("stream"),
                    dataset_id,
                    mask_path,
                )

            entry.setdefault("artifacts", {})[self.name] = self.output
            if self._accumulator:
                entry["artifacts"]["fusion-state"] = self._accumulator.state_fn

        return entries

    def _intercept(self, stream, dataset_id, mask_path):
        """Feed point or FusionState chunks into the accumulator and pass through."""

        count = 0
        z_min = float("inf")
        z_max = float("-inf")
        w_min = float("inf")
        w_max = float("-inf")
        u_min = float("inf")
        u_max = float("-inf")

        if self._accumulator and mask_path:
            self._accumulator.mask_registry[dataset_id] = mask_path

        dataset_str = format_dataset_id(dataset_id)
        logger.debug("Streaming data from: %s", dataset_str)

        for chunk in stream:
            chunk_to_stack = chunk

            if self._is_fusion_state_chunk(chunk):
                data = chunk[2]
                state = {name: data[index] for index, name in enumerate(FUSION_BANDS)}
                final = finalize_fusion_state(state)
                valid = final.valid

                if np.any(valid):
                    count += int(np.sum(final.count[valid]))
                    z_min = min(z_min, float(np.nanmin(final.z[valid])))
                    z_max = max(z_max, float(np.nanmax(final.z[valid])))
                    w_min = min(w_min, float(np.nanmin(final.mean_weight[valid])))
                    w_max = max(w_max, float(np.nanmax(final.mean_weight[valid])))
                    u_min = min(u_min, float(np.nanmin(final.uncertainty[valid])))
                    u_max = max(u_max, float(np.nanmax(final.uncertainty[valid])))

                if self._accumulator:
                    self._accumulator.update_raster_state(chunk, dataset_id)

            elif (
                isinstance(chunk, np.ndarray)
                and chunk.dtype.names
                and "z" in chunk.dtype.names
            ):
                if self.drop_classes and "classification" in chunk.dtype.names:
                    keep = ~np.isin(chunk["classification"], self.drop_classes)
                    chunk_to_stack = chunk[keep]
                    if len(chunk_to_stack) == 0:
                        continue

                finite_z = np.isfinite(chunk_to_stack["z"])
                valid_z = chunk_to_stack["z"][finite_z]
                count += int(valid_z.size)

                if valid_z.size:
                    z_min = min(z_min, float(np.min(valid_z)))
                    z_max = max(z_max, float(np.max(valid_z)))

                if "w" in chunk_to_stack.dtype.names:
                    valid_w = chunk_to_stack["w"][np.isfinite(chunk_to_stack["w"])]
                    if valid_w.size:
                        w_min = min(w_min, float(np.min(valid_w)))
                        w_max = max(w_max, float(np.max(valid_w)))

                if "u" in chunk_to_stack.dtype.names:
                    valid_u = chunk_to_stack["u"][np.isfinite(chunk_to_stack["u"])]
                    if valid_u.size:
                        u_min = min(u_min, float(np.min(valid_u)))
                        u_max = max(u_max, float(np.max(valid_u)))

                if self._accumulator:
                    self._accumulator.update(chunk_to_stack, dataset_id)

            else:
                logger.debug(
                    "Ignoring unsupported chunk type from %s: %s",
                    dataset_str,
                    type(chunk).__name__,
                )

            yield chunk_to_stack

        if z_min == float("inf"):
            stats_str = "No valid Z data"
        else:
            z_str = (
                f"Z: [{z_min:.2e} to {z_max:.2e}]"
                if abs(z_min) > 1e10 or abs(z_max) > 1e10
                else f"Z: [{z_min:,.2f} to {z_max:,.2f}]"
            )

            w_str = ""
            if w_min != float("inf"):
                w_str = (
                    f" | W: [{w_min:.2f}]"
                    if w_min == w_max
                    else f" | W: [{w_min:.2f} to {w_max:.2f}]"
                )

            u_str = ""
            if u_min != float("inf"):
                u_str = (
                    f" | U: [{u_min:.2f}]"
                    if u_min == u_max
                    else f" | U: [{u_min:.2f} to {u_max:.2f}]"
                )

            stats_str = f"{z_str}{w_str}{u_str}"

        logger.info(
            "Stacked %s -> %s %s",
            dataset_str,
            colorize(f"{count:,}", BOLD) + " pts",
            colorize(f"({stats_str})", CYAN),
        )

        if self._accumulator and dataset_id:
            self._accumulator.mark_registered(dataset_id)

    def teardown(self):
        if self._accumulator:
            logger.debug("Streams finished. Finalizing MultiStack...")
            self._accumulator.finalize()
            self._accumulator = None
