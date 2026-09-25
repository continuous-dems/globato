#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""globato.hooks.transforms.point_pixels

Associative point-to-pixel fusion state for Globato.

The core contract in this module is intentionally narrow:

    elevation points -> additive FusionState

FusionState is safe to split, merge, cache, serialize, and resume without
changing the final weighted statistics.  Derived values such as weighted mean,
standard deviation, and propagated uncertainty are calculated only when the
state is finalized.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.windows import Window

from fetchez.hooks import FetchHook
from fetchez.spatial import Region
from fetchez.utils import float_or, int_or, str2inc

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Fusion-state schema
# -----------------------------------------------------------------------------

FUSION_STATE_VERSION = 1

FUSION_BANDS = (
    "z_weighted_sum",  # sum(z * w)
    "count",  # N
    "weight_sum",  # sum(w)
    "z2_weighted_sum",  # sum(z^2 * w)
    "weighted_uncertainty_sq",  # sum((w * u)^2)
    "x_weighted_sum",  # sum(x * w)
    "y_weighted_sum",  # sum(y * w)
)

FUSION_BAND_MAP = {name: index + 1 for index, name in enumerate(FUSION_BANDS)}


@dataclass(frozen=True)
class FinalizedFusion:
    """Derived values from an additive fusion state.

    ``uncertainty`` is the propagated uncertainty of the weighted mean from
    input point uncertainties. ``stddev`` describes observed elevation spread
    within the cell. They remain distinct on purpose.
    """

    z: np.ndarray
    x: np.ndarray
    y: np.ndarray
    count: np.ndarray
    weight_sum: np.ndarray
    mean_weight: np.ndarray
    uncertainty: np.ndarray
    stddev: np.ndarray
    valid: np.ndarray


@dataclass(slots=True)
class PreparedPoints:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    w: np.ndarray
    u: np.ndarray

    pixel_x: np.ndarray
    pixel_y: np.ndarray

    local_x: np.ndarray
    local_y: np.ndarray
    flat_indices: np.ndarray

    window: tuple[int, int, int, int]
    shape: tuple[int, int]

    @property
    def count(self) -> int:
        return self.z.size

    @property
    def size(self) -> int:
        return self.z.size

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def cols(self) -> int:
        return self.shape[1]

    @property
    def raster_size(self) -> int:
        return self.rows * self.cols


class PointPixels:
    """Reduce elevation points into an additive per-pixel FusionState.

    Incoming data must provide ``x``, ``y`` and ``z``. ``w`` and ``u`` are
    optional and default to 1 and 0 respectively.

    The returned arrays are sufficient statistics. They are *not* finalized
    raster values. Every field is additive, making this state associative and
    therefore safe for chunking, caching, and later merging.
    """

    def __init__(self, src_region=None, x_size=None, y_size=None, **kwargs):
        self.src_region = src_region
        self.x_size = int_or(x_size, 10)
        self.y_size = int_or(y_size, 10)
        self.dst_gt = None

    @staticmethod
    def empty_state():
        return {name: None for name in FUSION_BANDS}

    def init_region_from_points(self, points):
        if self.src_region is None:
            self.src_region = Region.from_list(
                [
                    np.min(points["x"]),
                    np.max(points["x"]),
                    np.min(points["y"]),
                    np.max(points["y"]),
                ]
            )
        else:
            self.src_region = Region(*self.src_region)

        if not self.src_region.valid_p():
            # Preserve the existing behavior for degenerate one-point/line
            # regions, but keep it isolated from the accumulation math.
            self.src_region.buffer(2)
            if not self.src_region.valid_p():
                self.src_region.buffer(10)

        self.init_gt()

    def init_gt(self):
        if self.src_region is not None:
            self.dst_gt = self.src_region.geo_transform_from_count(
                x_count=self.x_size,
                y_count=self.y_size,
            )

    def __call__(
        self,
        points,
        *,
        source_weight=1.0,
        source_uncertainty=0.0,
    ):
        return self.accumulate(
            points,
            source_weight=source_weight,
            source_uncertainty=source_uncertainty,
        )

    def _prepare(self, points) -> PreparedPoints | None:
        if points is None or len(points) == 0:
            return None

        if hasattr(points, "to_records"):
            points = points.to_records(index=False)

        if self.src_region is None:
            self.init_region_from_points(points)
        elif self.dst_gt is None:
            self.init_gt()

        x = np.asarray(points["x"], dtype=np.float64)
        y = np.asarray(points["y"], dtype=np.float64)
        z = np.asarray(points["z"], dtype=np.float64)

        w = (
            np.asarray(points["w"], dtype=np.float64)
            if "w" in points.dtype.names
            else np.ones_like(z)
        )
        u = (
            np.asarray(points["u"], dtype=np.float64)
            if "u" in points.dtype.names
            else np.zeros_like(z)
        )

        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

        if not np.any(valid):
            return None

        x = x[valid]
        y = y[valid]
        z = z[valid]
        w = w[valid]
        u = u[valid]

        w[~np.isfinite(w)] = 1.0
        u[~np.isfinite(u)] = 0.0

        pixel_x = np.floor((x - self.dst_gt[0]) / self.dst_gt[1]).astype(np.int64)

        pixel_y = np.floor((y - self.dst_gt[3]) / self.dst_gt[5]).astype(np.int64)

        inside = (
            (pixel_x >= 0)
            & (pixel_x < self.x_size)
            & (pixel_y >= 0)
            & (pixel_y < self.y_size)
        )

        if not np.any(inside):
            return None

        x = x[inside]
        y = y[inside]
        z = z[inside]
        w = w[inside]
        u = u[inside]
        pixel_x = pixel_x[inside]
        pixel_y = pixel_y[inside]

        min_x = int(pixel_x.min())
        max_x = int(pixel_x.max())
        min_y = int(pixel_y.min())
        max_y = int(pixel_y.max())

        cols = max_x - min_x + 1
        rows = max_y - min_y + 1

        local_x = pixel_x - min_x
        local_y = pixel_y - min_y

        flat_indices = local_y * cols + local_x

        return PreparedPoints(
            x=x,
            y=y,
            z=z,
            w=w,
            u=u,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            local_x=local_x,
            local_y=local_y,
            flat_indices=flat_indices,
            window=(min_x, min_y, cols, rows),
            shape=(rows, cols),
        )

    def count(self, points):
        prepared = self._prepare(points)
        if prepared is None:
            return None, None, None

        rows, cols = prepared.shape

        count = np.bincount(
            prepared.flat_indices,
            minlength=rows * cols,
        ).reshape(prepared.shape)

        return count, prepared.window, self.dst_gt

    def coverage(self, points):
        count, window, gt = self.count(points)

        if count is None:
            return None, None, None

        return count > 0, window, gt

    def accumulate(
        self,
        points,
        source_weight=1.0,
        source_uncertainty=0.0,
    ):
        prepared = self._prepare(points)
        if prepared is None:
            return None, None, None

        out = self.empty_state()

        rows, cols = prepared.shape

        min_px, _max_px = int(np.min(prepared.pixel_x)), int(np.max(prepared.pixel_x))
        min_py, _max_py = int(np.min(prepared.pixel_y)), int(np.max(prepared.pixel_y))

        srcwin = (min_px, min_py, cols, rows)

        source_weight = float_or(source_weight, 1.0)
        source_uncertainty = float_or(source_uncertainty, 0.0)

        effective_w = prepared.w * source_weight
        effective_u2 = np.square(prepared.u) + source_uncertainty**2

        # Every statistic below is additive. Do not finalize here.
        out["count"] = (
            np.bincount(prepared.flat_indices, minlength=prepared.raster_size)
            .reshape(prepared.shape)
            .astype(np.float64)
        )
        out["weight_sum"] = np.bincount(
            prepared.flat_indices, weights=effective_w, minlength=prepared.raster_size
        ).reshape(prepared.shape)
        out["z_weighted_sum"] = np.bincount(
            prepared.flat_indices,
            weights=prepared.z * effective_w,
            minlength=prepared.raster_size,
        ).reshape(prepared.shape)
        out["z2_weighted_sum"] = np.bincount(
            prepared.flat_indices,
            weights=np.square(prepared.z) * effective_w,
            minlength=prepared.raster_size,
        ).reshape(prepared.shape)

        # For a weighted mean mu=sum(w_i*z_i)/sum(w_i), independent input
        # standard uncertainties propagate as:
        #     u(mu)^2 = sum((w_i*u_i)^2) / sum(w_i)^2
        # Store only the additive numerator here.
        out["weighted_uncertainty_sq"] = np.bincount(
            prepared.flat_indices,
            weights=np.square(effective_w) * effective_u2,
            minlength=prepared.raster_size,
        ).reshape(prepared.shape)

        out["x_weighted_sum"] = np.bincount(
            prepared.flat_indices,
            weights=prepared.x * effective_w,
            minlength=prepared.raster_size,
        ).reshape(prepared.shape)
        out["y_weighted_sum"] = np.bincount(
            prepared.flat_indices,
            weights=prepared.y * effective_w,
            minlength=prepared.raster_size,
        ).reshape(prepared.shape)

        return out, srcwin, self.dst_gt


def merge_fusion_states(*states):
    """Element-wise merge compatible FusionState dictionaries."""

    states = [state for state in states if state is not None]
    if not states:
        return PointPixels.empty_state()

    merged = {}
    for name in FUSION_BANDS:
        arrays = [state.get(name) for state in states if state.get(name) is not None]
        merged[name] = None if not arrays else np.add.reduce(arrays)
    return merged


def finalize_fusion_state(state) -> FinalizedFusion:
    """Derive weighted values from additive FusionState arrays.

    This function does not choose a policy for combining measurement
    uncertainty with observed cell dispersion. Both are returned separately.
    """

    count = np.asarray(state["count"], dtype=np.float64)
    weight_sum = np.asarray(state["weight_sum"], dtype=np.float64)
    z_sum = np.asarray(state["z_weighted_sum"], dtype=np.float64)
    z2_sum = np.asarray(state["z2_weighted_sum"], dtype=np.float64)
    wu2_sum = np.asarray(state["weighted_uncertainty_sq"], dtype=np.float64)
    x_sum = np.asarray(state["x_weighted_sum"], dtype=np.float64)
    y_sum = np.asarray(state["y_weighted_sum"], dtype=np.float64)

    valid = (count > 0) & np.isfinite(weight_sum) & (weight_sum != 0)
    shape = count.shape

    z = np.full(shape, np.nan, dtype=np.float64)
    x = np.full(shape, np.nan, dtype=np.float64)
    y = np.full(shape, np.nan, dtype=np.float64)
    mean_weight = np.zeros(shape, dtype=np.float64)
    uncertainty = np.zeros(shape, dtype=np.float64)
    stddev = np.zeros(shape, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        z[valid] = z_sum[valid] / weight_sum[valid]
        x[valid] = x_sum[valid] / weight_sum[valid]
        y[valid] = y_sum[valid] / weight_sum[valid]
        mean_weight[valid] = weight_sum[valid] / count[valid]

        second_moment = z2_sum[valid] / weight_sum[valid]
        variance = second_moment - np.square(z[valid])
        stddev[valid] = np.sqrt(np.clip(variance, 0.0, None))

        uncertainty[valid] = np.sqrt(np.clip(wu2_sum[valid], 0.0, None)) / np.abs(
            weight_sum[valid]
        )

    return FinalizedFusion(
        z=z,
        x=x,
        y=y,
        count=count,
        weight_sum=weight_sum,
        mean_weight=mean_weight,
        uncertainty=uncertainty,
        stddev=stddev,
        valid=valid,
    )


class PixelsToPoints(FetchHook):
    """Finalize FusionState raster chunks into representative point records.

    This operation is intentionally lossy: count and higher-order sufficient
    statistics are collapsed into one representative point per populated cell.
    It must therefore not be used to restore a side-stack cache intended for
    exact accumulation parity.
    """

    name = "pixels2points"
    meta_stage = "stream"
    meta_category = "stream-transform"
    meta_aliases = ["pixels_to_points"]
    meta_desc = "Finalize fusion-state raster chunks into representative points."

    def _raster_to_xyz(self, raster_stream):
        _profile = next(raster_stream)

        for window, buff_win, data, ndv, transform in raster_stream:
            if data.ndim != 3 or data.shape[0] != len(FUSION_BANDS):
                raise ValueError(
                    f"Expected {len(FUSION_BANDS)}-band FusionState; got {data.shape}"
                )

            state = {name: data[index] for index, name in enumerate(FUSION_BANDS)}
            final = finalize_fusion_state(state)
            valid = final.valid

            if not np.any(valid):
                continue

            chunk = np.rec.fromarrays(
                [
                    final.x[valid],
                    final.y[valid],
                    final.z[valid],
                    final.weight_sum[valid],
                    final.uncertainty[valid],
                ],
                names=["x", "y", "z", "w", "u"],
            )
            yield chunk

    def run(self, entries):
        for mod, entry in entries:
            if self.is_raster_stream(entry):
                entry["stream"] = self._raster_to_xyz(entry["stream"])
                entry["stream_type"] = "point-stream"
        return entries


class Point2PixelStream(FetchHook):
    """Convert a point stream into a raster stream of additive FusionState."""

    name = "points2pixels"
    meta_stage = "stream"
    meta_category = "stream-transform"
    meta_aliases = ["point2pixel", "points_to_pixel"]
    meta_desc = "Convert elevation points into associative Globato fusion state."

    def __init__(self, x_inc=None, y_inc=None, **kwargs):
        super().__init__(**kwargs)
        self.x_inc = float_or(str2inc(x_inc))
        self.y_inc = float_or(str2inc(y_inc))

    def process_chunk(self, chunk, region=None):
        if not region:
            return None, None, None

        xcount, ycount, _ = region.geo_transform(
            x_inc=self.x_inc,
            y_inc=self.y_inc,
            node="grid",
        )
        reducer = PointPixels(src_region=region, x_size=xcount, y_size=ycount)
        return reducer.accumulate(chunk)

    def _stream_wrapper(self, input_stream, entry=None, region=None):
        count = 0
        entry = entry or {}

        if not region:
            return

        xcount, ycount, gt = region.geo_transform(
            x_inc=self.x_inc,
            y_inc=self.y_inc,
            node="grid",
        )
        transform = rasterio.transform.from_origin(gt[0], gt[3], gt[1], abs(gt[5]))

        profile = {
            "driver": "GTiff",
            "dtype": "float64",
            "nodata": None,
            "width": xcount,
            "height": ycount,
            "count": len(FUSION_BANDS),
            "crs": entry.get("src_srs", "EPSG:4326"),
            "transform": transform,
            "GLOBATO_DATATYPE": "FUSION_STATE",
            "GLOBATO_FUSION_VERSION": FUSION_STATE_VERSION,
        }
        yield profile

        for chunk in input_stream:
            count += chunk.size
            state, srcwin, chunk_gt = self.process_chunk(chunk, region=region)
            if srcwin is None or state["count"] is None:
                continue

            col_off, row_off, width, height = srcwin
            window = Window(col_off, row_off, width, height)
            chunk_transform = rasterio.transform.from_origin(
                chunk_gt[0], chunk_gt[3], chunk_gt[1], abs(chunk_gt[5])
            )

            data = np.stack([state[name] for name in FUSION_BANDS]).astype(
                np.float64,
                copy=False,
            )
            yield window, window, data, None, chunk_transform

        logger.info(
            "Parsed %s data records from %s",
            f"{count:,}",
            entry.get("dst_fn", "point stream"),
        )

    def run(self, entries):
        for mod, entry in entries:
            if self.is_point_stream(entry):
                entry["stream"] = self._stream_wrapper(
                    entry["stream"],
                    entry=entry,
                    region=getattr(mod, "region", None),
                )
                entry["stream_type"] = "raster-stream"
                entry["data_type"] = "fusion-state"
        return entries
