#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Post-stack source provenance with interchangeable tier-state storage.

This hook is intentionally separate from ``multi_stack`` and ``source_masks``:

* ``source_masks`` records where a source produced valid parsed observations.
* ``multi_stack`` performs numerical FusionState reduction only.
* ``stack_provenance`` records enough per-source state to determine which
  sources are represented in the final stack, then materializes final masks at
  teardown.

For the ``mixed`` strategy the only per-source state required is the highest
weight tier seen at each pixel. Equal-tier inputs merge, and higher tiers
replace lower tiers, so a source survives exactly where its recorded tier
matches the final winning tier. State codes are UInt8; 0 means "source absent"
and tier N is stored as N+1.

Two storage backends are available:

* ``disk`` (default) persists one compact UInt8 tier raster per source. Memory
  usage stays bounded, at the cost of temporary raster I/O.
* ``memory`` keeps one dense UInt8 tier grid per source in RAM. It avoids state
  raster I/O and is intended for high-throughput systems with ample memory.

Both backends implement the same tier semantics and share the same final mask
materialization path.
"""

from __future__ import annotations

import json
import logging
import os
import threading

import numpy as np
import rasterio
from rasterio.windows import Window

from fetchez.hooks import FetchHook
from fetchez.spatial import Region
from fetchez.utils import str2bool

from ..transforms.point_pixels import FUSION_BAND_MAP, FUSION_BANDS, PointPixels
from .provenance import MaskSet, source_id, source_token

logger = logging.getLogger(__name__)


class StackTierStateBase:
    """Common stack-provenance tier semantics aligned to a FusionState grid.

    Disk storage uses one uint8 provenance-state raster per source.
    Pixel values encode the source’s highest stack weight tier at that location;
    they do not encode source identity. 0 means the source is absent,
    and positive values encode weight tiers. Because the state uses uint8,
    mixed stack provenance supports at most 254 configured weight thresholds
    (255 tier codes including the base tier).
    """

    TYPE_TAG = "GLOBATO_DATATYPE"
    STATE_TYPE = "STACK_PROVENANCE_STATE"
    SOURCE_TAG = "GLOBATO_SOURCE_ID"

    def __init__(self, fusion_state):
        self.fusion_state = os.path.abspath(fusion_state)

        with rasterio.open(self.fusion_state) as src:
            self.width = src.width
            self.height = src.height
            self.transform = src.transform
            self.crs = src.crs
            tags = src.tags()

        self.strategy = tags.get("GLOBATO_STACK_STRATEGY", "").lower()
        try:
            self.weight_tiers = np.sort(
                np.asarray(json.loads(tags["GLOBATO_WEIGHT_TIERS"]), dtype=np.float64)
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self.weight_tiers = np.asarray([], dtype=np.float64)

        if self.strategy == "mixed" and self.weight_tiers.size >= 255:
            raise ValueError(
                "stack_provenance supports at most 254 mixed weight thresholds "
                "when using uint8 tier state"
            )

    def encode(self, count, weight_sum):
        """Return UInt8 source-state codes for one local FusionState window."""
        valid = count > 0
        out = np.zeros(count.shape, dtype=np.uint8)
        if not np.any(valid):
            return out

        if self.strategy in {"mean", "weighted_mean"}:
            out[valid] = 1
            return out

        if self.strategy != "mixed":
            return out

        with np.errstate(divide="ignore", invalid="ignore"):
            mean_weight = np.where(valid, weight_sum / count, 0.0)

        # 0 is reserved for absent; digitize tier 0 becomes code 1.
        out[valid] = (
            np.digitize(mean_weight[valid], self.weight_tiers).astype(np.uint16) + 1
        ).astype(np.uint8)
        return out

    def build_winner_state(self, output_dir, *, compress=True):
        """Build a one-band UInt8 raster containing the final winning tier."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "_winning_tier.tif")

        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 1,
            "width": self.width,
            "height": self.height,
            "crs": self.crs,
            "transform": self.transform,
            "nodata": 0,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "bigtiff": "YES",
        }
        if compress:
            profile["compress"] = "lzw"

        with (
            rasterio.open(self.fusion_state) as src,
            rasterio.open(path, "w", **profile) as dst,
        ):
            dst.set_band_description(1, "Winning stack tier")
            dst.update_tags(
                **{
                    self.TYPE_TAG: "STACK_WINNING_TIER",
                    "GLOBATO_STACK_STRATEGY": self.strategy,
                    "GLOBATO_WEIGHT_TIERS": json.dumps(self.weight_tiers.tolist()),
                }
            )

            count_idx = FUSION_BAND_MAP["count"]
            weight_idx = FUSION_BAND_MAP["weight_sum"]

            for _, window in src.block_windows(count_idx):
                count = src.read(count_idx, window=window)
                weight_sum = src.read(weight_idx, window=window)
                dst.write(self.encode(count, weight_sum), 1, window=window)

        return path

    def register(self, dataset_id, *, description=None, tags=None):
        raise NotImplementedError

    def update(self, handle, window, codes):
        raise NotImplementedError

    def iter_sources(self):
        """Yield ``(dataset_id, description, tags, source)`` records."""
        raise NotImplementedError

    def iter_blocks(self, source):
        """Yield ``(window, source_code)`` for one registered source."""
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class DiskStackTierState(StackTierStateBase):
    """Persistent per-source UInt8 tier rasters with bounded memory use."""

    def __init__(self, fusion_state, output_dir, *, resume=True, compress=True):
        super().__init__(fusion_state)
        self.output_dir = os.path.abspath(output_dir)
        self.resume = bool(resume)
        self.compress = bool(compress)
        self.lock = threading.Lock()
        self.states = {}  # dataset_id -> path
        os.makedirs(self.output_dir, exist_ok=True)

        self.profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 1,
            "width": self.width,
            "height": self.height,
            "crs": self.crs,
            "transform": self.transform,
            "nodata": 0,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "bigtiff": "YES",
        }
        if self.compress:
            self.profile["compress"] = "lzw"

    def _path(self, dataset_id):
        return os.path.join(
            self.output_dir,
            f"{source_token(dataset_id)}_stack_state.tif",
        )

    def _valid_existing(self, path, dataset_id):
        if not os.path.exists(path):
            return False
        with rasterio.open(path) as src:
            return (
                src.width == self.width
                and src.height == self.height
                and src.transform == self.transform
                and src.crs == self.crs
                and src.count == 1
                and src.dtypes[0] == "uint8"
                and src.tags().get(self.TYPE_TAG) == self.STATE_TYPE
                and src.tags().get(self.SOURCE_TAG) == dataset_id
            )

    def register(self, dataset_id, *, description=None, tags=None):
        path = self._path(dataset_id)

        if not (self.resume and self._valid_existing(path, dataset_id)):
            with rasterio.open(path, "w", **self.profile) as dst:
                if description:
                    dst.set_band_description(1, description)
                metadata = {
                    self.TYPE_TAG: self.STATE_TYPE,
                    self.SOURCE_TAG: dataset_id,
                    "GLOBATO_STACK_STRATEGY": self.strategy,
                    "GLOBATO_WEIGHT_TIERS": json.dumps(self.weight_tiers.tolist()),
                    "GLOBATO_STORAGE": "disk",
                }
                if tags:
                    metadata.update(
                        {
                            str(k): str(v)
                            for k, v in tags.items()
                            if v not in (None, "", "None", "Unknown")
                        }
                    )
                dst.update_tags(**metadata)

        self.states[dataset_id] = path
        return path

    def update(self, path, window, codes):
        if codes is None or not np.any(codes):
            return

        with self.lock:
            with rasterio.open(path, "r+") as dst:
                current = dst.read(1, window=window)
                np.maximum(current, codes, out=current)
                dst.write(current, 1, window=window)

    def iter_sources(self):
        for dataset_id, path in self.states.items():
            if not os.path.exists(path):
                continue
            with rasterio.open(path) as src:
                yield dataset_id, src.descriptions[0], src.tags(), path

    def iter_blocks(self, path):
        with rasterio.open(path) as src:
            for _, window in src.block_windows(1):
                yield window, src.read(1, window=window)

    def __len__(self):
        return len(self.states)


class MemoryStackTierState(StackTierStateBase):
    """Dense per-source UInt8 tier grids optimized for high-RAM systems."""

    def __init__(self, fusion_state):
        super().__init__(fusion_state)
        self.lock = threading.Lock()
        self.states = {}  # dataset_id -> {data, description, tags}

    def register(self, dataset_id, *, description=None, tags=None):
        if dataset_id not in self.states:
            metadata = {
                self.TYPE_TAG: self.STATE_TYPE,
                self.SOURCE_TAG: dataset_id,
                "GLOBATO_STACK_STRATEGY": self.strategy,
                "GLOBATO_WEIGHT_TIERS": json.dumps(self.weight_tiers.tolist()),
                "GLOBATO_STORAGE": "memory",
            }
            if tags:
                metadata.update(
                    {
                        str(k): str(v)
                        for k, v in tags.items()
                        if v not in (None, "", "None", "Unknown")
                    }
                )
            self.states[dataset_id] = {
                "data": np.zeros((self.height, self.width), dtype=np.uint8),
                "description": description,
                "tags": metadata,
            }
        return dataset_id

    def update(self, dataset_id, window, codes):
        if codes is None or not np.any(codes):
            return

        row0 = int(window.row_off)
        col0 = int(window.col_off)
        row1 = row0 + int(window.height)
        col1 = col0 + int(window.width)

        with self.lock:
            current = self.states[dataset_id]["data"][row0:row1, col0:col1]
            np.maximum(current, codes, out=current)

    def iter_sources(self):
        for dataset_id, state in self.states.items():
            yield dataset_id, state["description"], state["tags"], state["data"]

    def iter_blocks(self, data):
        block = 256
        for row0 in range(0, self.height, block):
            height = min(block, self.height - row0)
            for col0 in range(0, self.width, block):
                width = min(block, self.width - col0)
                window = Window(col0, row0, width, height)
                yield window, data[row0 : row0 + height, col0 : col0 + width]

    def __len__(self):
        return len(self.states)

    @property
    def bytes_used(self):
        return sum(state["data"].nbytes for state in self.states.values())


class StackProvenance(FetchHook):
    """Materialize source masks representing data that survived ``multi_stack``.

    Place this hook immediately after ``multi_stack``. It observes the same
    pass-through stream and records UInt8 per-source tier state. ``storage=disk``
    persists that state in temporary rasters with bounded memory use;
    ``storage=memory`` keeps dense source grids in RAM for maximum throughput.
    Final masks are derived at teardown from the completed FusionState.

    ``mixed`` is the primary strategy. ``mean`` and ``weighted_mean`` are also
    supported because every valid source contribution survives those reducers.
    ``supercede`` is intentionally unsupported until reducer decisions can be
    emitted directly by ``multi_stack``; equal-weight replacement is order
    sensitive and cannot be reconstructed from tier state alone.
    """

    name = "stack_provenance"
    meta_aliases = ["stack-provenance", "stack_masks", "stack-masks"]
    meta_stage = "stream"
    meta_category = "metadata"
    meta_desc = "Record sources represented in the final MultiStack."

    def __init__(
        self,
        output="stack_sources.vrt",
        output_dir=None,
        state_dir=None,
        vector_output=None,
        resume=True,
        compress_state=True,
        storage="disk",
        vector_max_size=2048,
        vector_simplify=0.0,
        qgis_style=True,
        qgis_style_field="SOURCE_ID",
        group_by=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output = output
        self.output_dir = output_dir
        self.state_dir = state_dir
        self.vector_output = vector_output
        self.resume = str2bool(resume)
        self.compress_state = str2bool(compress_state)
        self.storage = str(storage).strip().lower()
        if self.storage not in {"disk", "memory"}:
            raise ValueError("stack_provenance storage must be 'disk' or 'memory'")
        self.vector_max_size = int(vector_max_size)
        self.vector_simplify = max(0.0, float(vector_simplify))
        self.qgis_style = str2bool(qgis_style)
        self.qgis_style_field = str(qgis_style_field)
        self.group_by = group_by

        self._tier_state = None
        self._final_masks = None
        self._fusion_state = None
        self._binner = None
        self._region = None

    def _init(self, fusion_state):
        if self._tier_state is not None:
            if os.path.abspath(fusion_state) != self._fusion_state:
                raise ValueError("stack_provenance received multiple FusionState grids")
            return

        self._fusion_state = os.path.abspath(fusion_state)

        with rasterio.open(self._fusion_state) as src:
            bounds = src.bounds
            self._region = Region(bounds.left, bounds.right, bounds.bottom, bounds.top)
            if src.crs:
                self._region.srs = src.crs.to_string()
            width, height = src.width, src.height
            res = abs(src.transform.a)

        base = os.path.splitext(self.output)[0]
        state_dir = self.state_dir or f"{base}_state"

        if self.storage == "memory":
            self._tier_state = MemoryStackTierState(self._fusion_state)
        else:
            self._tier_state = DiskStackTierState(
                self._fusion_state,
                state_dir,
                resume=self.resume,
                compress=self.compress_state,
            )

        if self._tier_state.strategy == "supercede":
            logger.warning(
                "[%s] supercede cannot be reconstructed from tier state; "
                "no stack-provenance masks will be generated",
                self.name,
            )
        elif self._tier_state.strategy not in {"mixed", "mean", "weighted_mean"}:
            logger.warning(
                "[%s] unsupported stack strategy %r",
                self.name,
                self._tier_state.strategy,
            )

        self._final_masks = MaskSet(
            self._region,
            res,
            self.output,
            output_dir=self.output_dir,
            vector_output=self.vector_output,
            mask_type="STACK_MASK",
            resume=False,
            vector_max_size=self.vector_max_size,
            vector_simplify=self.vector_simplify,
            qgis_style=self.qgis_style,
            qgis_style_field=self.qgis_style_field,
            group_by=self.group_by,
        )

        self._binner = PointPixels(
            src_region=self._region,
            x_size=width,
            y_size=height,
        )

    def run(self, entries):
        for mod, entry in entries:
            fusion_state = entry.get("artifacts", {}).get("fusion-state")
            stream = entry.get("stream")
            if not fusion_state or stream is None:
                continue

            self._init(fusion_state)
            if self._tier_state.strategy == "supercede":
                continue

            dataset_id = source_id(entry)
            src_name = os.path.basename(entry.get("dst_fn", dataset_id))
            tags = {
                "MODULE": getattr(mod, "name", None),
                "DATASET": getattr(mod, "title", getattr(mod, "name", None)),
                "CATEGORY": getattr(mod, "meta_category", None),
                "AGENCY": getattr(mod, "meta_agency", None),
                "DATATYPE": entry.get("data_type"),
                "URL": entry.get("url"),
                "WEIGHT": getattr(mod, "weight", 1.0),
            }
            if isinstance(entry.get("metadata"), dict):
                tags.update({str(k).upper(): v for k, v in entry["metadata"].items()})

            state_handle = self._tier_state.register(
                dataset_id,
                description=os.path.splitext(src_name)[0],
                tags=tags,
            )
            if self.storage == "disk":
                entry.setdefault("artifacts", {})["stack-provenance-state"] = (
                    state_handle
                )
            entry["stream"] = self._intercept(stream, state_handle)

        return entries

    def _record_state(self, path, state, window):
        count = np.asarray(state["count"])
        weight_sum = np.asarray(state["weight_sum"])
        codes = self._tier_state.encode(count, weight_sum)
        self._tier_state.update(path, window, codes)

    def _intercept(self, stream, state_path):
        for chunk in stream:
            if (
                isinstance(chunk, np.ndarray)
                and chunk.dtype.names
                and "z" in chunk.dtype.names
            ):
                state, sub_win, _ = self._binner.accumulate(chunk)
                if sub_win is not None and state["count"] is not None:
                    col, row, width, height = sub_win
                    self._record_state(
                        state_path,
                        state,
                        Window(col, row, width, height),
                    )

            elif (
                isinstance(chunk, tuple)
                and len(chunk) >= 3
                and isinstance(chunk[2], np.ndarray)
                and chunk[2].ndim == 3
                and chunk[2].shape[0] == len(FUSION_BANDS)
            ):
                data = chunk[2]
                state = {name: data[index] for index, name in enumerate(FUSION_BANDS)}
                self._record_state(state_path, state, chunk[0])

            yield chunk

    def _materialize_masks(self, winner_path):
        with rasterio.open(winner_path) as winners:
            for (
                dataset_id,
                description,
                tags,
                source,
            ) in self._tier_state.iter_sources():
                final_path = self._final_masks.register(
                    dataset_id,
                    description=description,
                    tags=tags,
                )

                any_valid = False
                with rasterio.open(final_path, "r+") as dst:
                    for window, source_code in self._tier_state.iter_blocks(source):
                        winning_code = winners.read(1, window=window)

                        if self._tier_state.strategy in {"mean", "weighted_mean"}:
                            keep = source_code > 0
                        else:
                            keep = (source_code > 0) & (source_code == winning_code)

                        if np.any(keep):
                            dst.write(keep.astype(np.uint8), 1, window=window)
                            any_valid = True

                if not any_valid and os.path.exists(final_path):
                    os.remove(final_path)

    def teardown(self):
        if self._tier_state is None or len(self._tier_state) == 0:
            return

        if self._tier_state.strategy == "supercede":
            return

        base = os.path.splitext(self.output)[0]
        state_dir = self.state_dir or f"{base}_state"
        winner_path = self._tier_state.build_winner_state(
            state_dir,
            compress=self.compress_state,
        )
        self._materialize_masks(winner_path)
        self._final_masks.finalize()

        logger.info(
            "[%s] finalized %d stack-provenance source states (%s storage) -> %s",
            self.name,
            len(self._tier_state),
            self.storage,
            self.output,
        )

        if self.storage == "memory":
            logger.info(
                "[%s] memory provenance state used %.1f MiB",
                self.name,
                self._tier_state.bytes_used / (1024 * 1024),
            )
