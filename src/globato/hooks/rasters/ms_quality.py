#!/usr/bin/env python

"""
globato.hooks.rasters.ms_quality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quality-aware weighting of Multi-Stack cells based on local spatial
support and elevation consistency.

These hooks modify only the Multi-Stack weight band. They do not alter
elevation values or other accumulated statistics.

Hooks
-----
ms_support_weight
    Softly reduces weight for cells with unusually low local support.

ms_consistency_weight
    Softly reduces weight for low-count cells whose elevation is a
    robust local outlier.
"""

import logging
import warnings

import numpy as np
import scipy.ndimage
from numpy.lib.stride_tricks import sliding_window_view

from .base import RasterStreamHook

logger = logging.getLogger(__name__)


class MultiStackQuality(RasterStreamHook):
    """Common utilities for Multi-Stack quality hooks."""

    meta_category = "multi-stack"

    def __init__(
        self,
        radius=5,
        floor=0.5,
        min_neighbors=4,
        count_band=2,
        weight_band=3,
        **kwargs,
    ):
        super().__init__(buffer=int(radius), **kwargs)

        self.radius = max(1, int(radius))
        self.floor = float(np.clip(floor, 0.0, 1.0))
        self.min_neighbors = max(1, int(min_neighbors))
        self.count_band = int(count_band)
        self.weight_band = int(weight_band)

    def _validate_bands(self, data):
        required = max(self.count_band, self.weight_band)

        if data.ndim < 3:
            return False

        if data.shape[0] < required:
            logger.warning(
                "[%s] Expected at least %d bands, received %d.",
                self.name,
                required,
                data.shape[0],
            )
            return False

        return True

    def _valid_mask(self, data, ndv):
        z = data[0].astype(np.float64)
        count = data[self.count_band - 1].astype(np.float64)
        weight = data[self.weight_band - 1].astype(np.float64)

        valid = (
            (z != ndv)
            & np.isfinite(z)
            & np.isfinite(count)
            & np.isfinite(weight)
            & (count > 0)
            & (weight > 0)
        )

        return z, count, weight, valid

    def _local_valid_count(self, valid):
        """Return the number of valid cells in each local neighborhood."""

        size = (self.radius * 2) + 1

        return scipy.ndimage.uniform_filter(
            valid.astype(np.float64),
            size=size,
            mode="nearest",
        ) * (size * size)


class MultiStackSupportWeight(MultiStackQuality):
    """
    Softly reduce Multi-Stack weights for cells with unusually low
    local support.

    Support is measured relative to surrounding Multi-Stack cells rather
    than by an absolute threshold. Uniformly sparse regions therefore
    receive little or no adjustment, while isolated low-count cells in
    otherwise dense coverage are reduced.

    Args:
        radius (int): Neighborhood radius in pixels.
        floor (float): Minimum weight multiplier.
        power (float): Controls the strength of the relative support
            response. Values below 1 soften the penalty.
        min_neighbors (int): Minimum number of valid neighboring cells
            required before applying the adjustment.
        count_band (int): One-based Multi-Stack count band.
        weight_band (int): One-based Multi-Stack weight band.
    """

    name = "ms_support_weight"
    default_suffix = "_support"

    meta_desc = "Adjust Multi-Stack weights based on relative local data support."

    def __init__(
        self,
        radius=5,
        floor=0.5,
        power=0.5,
        min_neighbors=4,
        count_band=2,
        weight_band=3,
        **kwargs,
    ):
        super().__init__(
            radius=radius,
            floor=floor,
            min_neighbors=min_neighbors,
            count_band=count_band,
            weight_band=weight_band,
            **kwargs,
        )

        self.power = max(float(power), 0.01)

    def process_chunk(
        self,
        data,
        ndv,
        entry,
        transform=None,
        window=None,
    ):
        if not self._validate_bands(data):
            return data

        z, count, weight, valid = self._valid_mask(data, ndv)

        if not np.any(valid):
            return data

        size = (self.radius * 2) + 1

        count_values = np.where(valid, count, 0.0)

        local_count_sum = scipy.ndimage.uniform_filter(
            count_values,
            size=size,
            mode="nearest",
        ) * (size * size)

        local_valid_count = self._local_valid_count(valid)
        sufficient = local_valid_count >= self.min_neighbors

        local_mean_count = np.divide(
            local_count_sum,
            local_valid_count,
            out=np.full_like(local_count_sum, np.nan),
            where=local_valid_count > 0,
        )

        relative_support = np.divide(
            count,
            local_mean_count,
            out=np.ones_like(count),
            where=np.isfinite(local_mean_count) & (local_mean_count > 0),
        )

        weak_support = np.clip(
            np.where(np.isfinite(relative_support), relative_support, 1.0),
            0.0,
            1.0,
        )

        support_factor = self.floor + (
            (1.0 - self.floor) * np.power(weak_support, self.power)
        )

        support_factor = np.where(
            valid & sufficient,
            support_factor,
            1.0,
        )

        weight *= support_factor

        data[self.weight_band - 1] = weight.astype(
            data[self.weight_band - 1].dtype,
        )

        return data


class MultiStackConsistencyWeight(MultiStackQuality):
    """
    Softly reduce Multi-Stack weights for poorly supported cells whose
    elevations are robust local outliers.

    Only low-count cells are subjected to the neighborhood statistic.
    Cells at or above ``max_count`` are left untouched by the consistency
    calculation.

    The target cell is excluded from the neighborhood statistic, so an
    extreme observation cannot weaken the test against itself.

    ``method="mad"`` is the preferred statistic. ``method="iqr"`` is also
    available.

    Args:
        radius (int): Neighborhood radius in pixels.
        floor (float): Minimum weight multiplier.
        method (str): Robust statistic: ``"mad"`` or ``"iqr"``.
        sigma (float): Robust deviation threshold before penalty begins.
        min_count (float): Counts at or below this value receive full
            consistency scrutiny.
        max_count (float): Counts at or above this value are protected
            from consistency adjustment.
        max_residual (float): Absolute elevation residual corresponding
            to maximum penalty.
        min_neighbors (int): Minimum valid neighboring cells required
            before consistency adjustment is applied.
        count_band (int): One-based Multi-Stack count band.
        weight_band (int): One-based Multi-Stack weight band.
    """

    name = "ms_consistency_weight"
    default_suffix = "_consistent"

    meta_desc = (
        "Adjust Multi-Stack weights based on robust local elevation consistency."
    )

    def __init__(
        self,
        radius=5,
        floor=0.5,
        method="mad",
        sigma=3.0,
        min_count=1,
        max_count=4,
        max_residual=20.0,
        min_neighbors=8,
        count_band=2,
        weight_band=3,
        **kwargs,
    ):
        super().__init__(
            radius=radius,
            floor=floor,
            min_neighbors=min_neighbors,
            count_band=count_band,
            weight_band=weight_band,
            **kwargs,
        )

        method = str(method).lower()
        if method not in {"mad", "iqr"}:
            raise ValueError(
                f"Unsupported consistency method: {method!r}. Expected 'mad' or 'iqr'."
            )

        self.method = method
        self.sigma = max(float(sigma), 0.1)
        self.min_count = max(float(min_count), 0.0)
        self.max_count = max(
            float(max_count),
            self.min_count + 1.0,
        )
        self.max_residual = max(float(max_residual), 1e-6)

    def _candidate_statistics(self, z, valid, candidate_rows, candidate_cols):
        """
        Calculate robust local statistics only for candidate cells.

        The neighborhood windows are materialized only at candidate cells,
        avoiding a callback for every pixel in the chunk.

        Returns:
            tuple of ``(center, robust_scale, neighbor_count)`` arrays,
            each with one value per candidate cell.
        """

        size = (self.radius * 2) + 1
        radius = self.radius

        masked = np.where(valid, z, np.nan)
        padded = np.pad(
            masked,
            radius,
            mode="edge",
        )

        windows = sliding_window_view(
            padded,
            (size, size),
        )
        windows = windows[candidate_rows, candidate_cols]

        # Explicitly exclude the target cell from its own neighborhood.
        center = radius
        windows = windows.copy()
        windows[:, center, center] = np.nan

        valid_count = np.sum(
            np.isfinite(windows),
            axis=(1, 2),
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="All-NaN slice encountered",
                category=RuntimeWarning,
            )
            local_median = np.nanmedian(
                windows,
                axis=(1, 2),
            )

        if self.method == "iqr":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="All-NaN slice encountered",
                    category=RuntimeWarning,
                )
                q25, q75 = np.nanpercentile(
                    windows,
                    [25.0, 75.0],
                    axis=(1, 2),
                )
            robust_scale = (q75 - q25) / 1.349
        else:
            deviations = np.abs(windows - local_median[:, None, None])
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="All-NaN slice encountered",
                    category=RuntimeWarning,
                )
                mad = np.nanmedian(
                    deviations,
                    axis=(1, 2),
                )
            robust_scale = 1.4826 * mad

        return local_median, robust_scale, valid_count

    def _count_scrutiny_factor(self, count):
        """Return 0-1 describing how strongly a cell should be scrutinized."""

        return np.clip(
            (self.max_count - count) / (self.max_count - self.min_count),
            0.0,
            1.0,
        )

    def process_chunk(
        self,
        data,
        ndv,
        entry,
        transform=None,
        window=None,
    ):
        if not self._validate_bands(data):
            return data

        z, count, weight, valid = self._valid_mask(data, ndv)

        if not np.any(valid):
            return data

        # Identify the only cells that can possibly be changed.
        # Cells at/above max_count never enter the neighborhood
        # calculation at all.
        candidate_mask = valid & (count < self.max_count) & (count >= self.min_count)

        if not np.any(candidate_mask):
            return data

        candidate_rows, candidate_cols = np.where(candidate_mask)

        local_center, robust_scale, neighbor_count = self._candidate_statistics(
            z,
            valid,
            candidate_rows,
            candidate_cols,
        )

        sufficient = neighbor_count >= self.min_neighbors

        target_z = z[candidate_rows, candidate_cols]
        residual = np.abs(target_z - local_center)

        # Avoid zero/near-zero robust scales while retaining absolute
        # residual sensitivity in very flat neighborhoods.
        scale_floor = self.max_residual / self.sigma
        robust_scale = np.maximum(
            robust_scale,
            scale_floor,
        )

        robust_score = residual / robust_scale

        excess = np.maximum(
            robust_score - self.sigma,
            0.0,
        )

        sigma_strength = np.clip(
            excess / self.sigma,
            0.0,
            1.0,
        )

        residual_strength = np.clip(
            residual / self.max_residual,
            0.0,
            1.0,
        )

        outlier_strength = np.maximum(
            sigma_strength,
            residual_strength,
        )

        scrutiny = self._count_scrutiny_factor(
            count[candidate_rows, candidate_cols],
        )

        penalty_strength = outlier_strength * scrutiny

        consistency_factor = 1.0 - ((1.0 - self.floor) * penalty_strength)

        apply = sufficient & np.isfinite(consistency_factor)

        candidate_weights = weight[candidate_rows, candidate_cols]
        candidate_weights = np.where(
            apply,
            candidate_weights * consistency_factor,
            candidate_weights,
        )

        weight[candidate_rows, candidate_cols] = candidate_weights

        data[self.weight_band - 1] = weight.astype(
            data[self.weight_band - 1].dtype,
        )

        return data
