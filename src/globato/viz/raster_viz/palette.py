"""Scientifically keyed display color maps; source raster samples are never altered.

The default elevation ramp reserves blues for values below a configurable
water_level (default 0, in source elevation units), and land colors at/above
that level. Each side stretches independently without altering source values.
The same implementation is used by 2D, 3D and legends.
"""

from __future__ import annotations

import numpy as np

ELEVATION_PALETTE = "water_level_anchored_bathy_land_v1"

# A muted, legible bathymetric/land hypsometric palette. Colors at the
# shoreline are deliberately different on opposite sides of zero.
_WATER = (
    (0.00, "#132d5b"),
    (0.24, "#1d518c"),
    (0.52, "#3189bb"),
    (0.78, "#69bad6"),
    (1.00, "#b1e0e9"),
)
_LAND = (
    (0.00, "#307d5c"),
    (0.16, "#77a96a"),
    (0.36, "#c2bc73"),
    (0.58, "#c69c6e"),
    (0.80, "#a68e7d"),
    (1.00, "#eeeae1"),
)


def _interpolate(positions, stops):
    knots = np.array([x for x, _ in stops], dtype="float64")
    colors = np.array(
        [[int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)] for _, color in stops],
        dtype="float64",
    )
    return np.stack(
        [np.interp(positions, knots, colors[:, i]) for i in range(3)], axis=-1
    )


def elevation_rgba(values, limits, water_level=0.0):
    """RGBA with a strictly water_level-anchored color-family transition.

    A value below water_level is always blue, even outside percentile display
    bounds. Values at/above water_level are always land-colored. All values
    are in the original raster units; NaNs become transparent.
    """
    lo, hi = [float(v) for v in limits]
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("Elevation limits must be increasing finite values")
    if (
        isinstance(water_level, bool)
        or not np.isscalar(water_level)
        or not np.isfinite(water_level)
    ):
        raise ValueError(
            "water_level must be a finite number in raster elevation units"
        )
    water_level = float(water_level)
    z = np.asarray(values, dtype="float64")
    finite = np.isfinite(z)
    wet = finite & (z < water_level)
    dry = finite & ~wet
    rgb = np.zeros((*z.shape, 3), dtype="float64")
    if np.any(wet):
        # With a wholly submerged display range, use all available blue
        # contrast; at the transition shallows approach pale blue.
        end = min(hi, water_level)
        span = max(end - lo, np.finfo("float64").eps)
        positions = np.clip((z[wet] - lo) / span, 0.0, 1.0)
        rgb[wet] = _interpolate(positions, _WATER)
    if np.any(dry):
        start = max(lo, water_level)
        span = max(hi - start, np.finfo("float64").eps)
        positions = np.clip((z[dry] - start) / span, 0.0, 1.0)
        rgb[dry] = _interpolate(positions, _LAND)
    return np.concatenate([rgb, finite[..., None].astype("float64")], axis=-1)
