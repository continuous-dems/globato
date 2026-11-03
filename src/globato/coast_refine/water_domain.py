"""Shared, conservative water-domain rasterization for coastline and channel guidance.

The data source is the *published refined topology* from Coast Mask, whether
its source was OSM or a user-supplied attribute-free water polygon. Classes are
optional river hints, never prerequisites for water geometry. No buffering,
growth, depth inference, or automatic polygon editing is performed.
"""

from __future__ import annotations

import numpy as np
import pyogrio
import rasterio
from rasterio.features import rasterize
from .baseline import LAND, CLASSES
from .grid_metrics import cell_spacing_m

LAND_CLASSES = frozenset(LAND)
WATER_CLASSES = frozenset(CLASSES) - LAND_CLASSES
RIVER_CLASSES = frozenset(("river", "estuary"))


def topology_water_grids(path, shape, transform, crs):
    """Read the same sealed topology for both qualifier and solver.

    Fail closed when classes are missing: an unclassified local *input* belongs
    in coast_mask(mask=..., polygon_class=water), which publishes a classed
    refined topology. Unclassified barrier vectors must not silently become
    land or water. CRS mismatch is also a hard error.
    """
    g = pyogrio.read_dataframe(path)
    if (
        g.crs is None
        or crs is None
        or rasterio.crs.CRS.from_user_input(g.crs)
        != rasterio.crs.CRS.from_user_input(crs)
    ):
        raise ValueError(
            "Published Coast Mask topology and interpolation grid CRS differ or are missing"
        )
    if "class" not in g.columns:
        raise ValueError(
            "Expected Coast Mask classed *publication*, not a raw local polygon"
        )
    if g["class"].isna().any():
        raise ValueError("Coast Mask topology contains null classes")
    values = g["class"].astype(str).str.lower()
    unknown = set(values) - LAND_CLASSES - WATER_CLASSES
    if unknown:
        raise ValueError(
            "Unknown Coast Mask topology classes: " + repr(sorted(unknown))
        )

    def grid(classes):
        shapes = [
            (geom, 1)
            for geom, c in zip(g.geometry, values)
            if c in classes and geom is not None and not geom.is_empty
        ]
        return (
            rasterize(
                shapes,
                out_shape=shape,
                transform=transform,
                fill=0,
                dtype="uint8",
                all_touched=False,
            ).astype(bool)
            if shapes
            else np.zeros(shape, bool)
        )

    water = grid(WATER_CLASSES)
    land = grid(LAND_CLASSES)
    explicit = grid(RIVER_CLASSES)
    if np.any(water & land):
        raise ValueError("Overlapping water and land topology on interpolation grid")
    return water, land, explicit


def opposite_bank_channel_candidates(
    water, transform, crs, *, land=None, max_width_m=80.0
):
    """Conservative narrow-water screening from geometry alone.

    A water center must have land on both sides along one of four axes within
    max_width_m/2. This excludes one-sided open shoreline while allowing rivers
    physically connected to ocean water to be identified without attributes.
    This is *potential channel geometry*, not a river classification or soundings.
    Later bank_smooth_surface independently requires an elongated, two-bank-
    supported reach before making any prediction.
    """
    wet = np.asarray(water, dtype=bool)
    if wet.ndim != 2 or not 0 < max_width_m <= 1000:
        raise ValueError("Expected a 2D water grid and positive channel width")
    dy, dx = cell_spacing_m(transform, crs, wet.shape)
    if dy <= 0 or dx <= 0:
        raise ValueError("Invalid georeferenced cell spacing")
    # Published land cells, not the complement of water: pixels outside the
    # publication/AOI must never masquerade as observed opposite banks.
    if land is None:
        raise ValueError("Published land geometry is required for opposite-bank test")
    land = np.asarray(land, dtype=bool)
    if land.shape != wet.shape or np.any(land & wet):
        raise ValueError("Published land/water masks must be aligned and disjoint")
    candidate = np.zeros(wet.shape, bool)
    h, w = wet.shape
    # A four-direction sampled line-of-sight test: avoid EDT-style isotropic
    # proximity, which incorrectly tags an ordinary one-sided ocean shoreline.
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        step_m = np.hypot(dr * dy, dc * dx)
        max_step = min(int((max_width_m / 2) / step_m), max(h, w) - 1)
        if max_step < 1:
            continue

        def shift_land(rr, cc):
            p = np.zeros(wet.shape, bool)
            r0 = max(0, -rr)
            r1 = min(h, h - rr)
            c0 = max(0, -cc)
            c1 = min(w, w - cc)
            if r0 < r1 and c0 < c1:
                p[r0:r1, c0:c1] = land[r0 + rr : r1 + rr, c0 + cc : c1 + cc]
            return p

        plus = np.zeros_like(wet)
        minus = np.zeros_like(wet)
        for i in range(1, max_step + 1):
            plus |= shift_land(dr * i, dc * i)
            minus |= shift_land(-dr * i, -dc * i)
        candidate |= plus & minus
    # A one-pixel sliver cannot supply a meaningful bank-to-bank profile.
    return candidate & wet


def channel_guidance_grids(path, shape, transform, crs, *, max_width_m=80.0):
    water, land, explicit = topology_water_grids(path, shape, transform, crs)
    inferred = opposite_bank_channel_candidates(
        water, transform, crs, land=land, max_width_m=max_width_m
    )
    # Explicit labels remain useful where OSM provides them. They never cause
    # water outside the refined domain to be added.
    channel = water & (explicit | inferred)
    return {
        "water": water,
        "land": land,
        "explicit": explicit,
        "inferred": inferred,
        "channel": channel,
    }
