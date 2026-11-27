"""Original-data local graph. Source-wide statistics and reference inputs absent."""

from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from pyproj import CRS


@dataclass(frozen=True)
class Rule:
    max_gap_m: float = 30.0
    noise_floor_m: float = 1.0
    plane_iterations: int = 2


DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


@lru_cache(maxsize=16)
def _crs_units(value):
    crs = CRS.from_user_input(value)
    if crs.is_compound:
        crs = crs.sub_crs_list[0]
    return crs.is_geographic, None if crs.is_geographic else crs.axis_info[
        0
    ].unit_conversion_factor


def metric(profile, rows, other_rows):
    t = profile["transform"]
    if t.b or t.d or t.a <= 0 or t.e >= 0:
        raise ValueError("North-up native grid required")
    geographic, factor = _crs_units(str(profile["crs"]))
    if geographic:
        lat = t.f + ((rows + other_rows) / 2 + 0.5) * t.e
        return abs(t.a) * 111320 * np.cos(np.deg2rad(lat)), abs(t.e) * 111132
    return np.full(np.shape(rows), abs(t.a) * factor), abs(t.e) * factor


def build(z, profile, rule=Rule()):
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or not np.all(np.isfinite(z) | np.isnan(z)):
        raise ValueError("Invalid elevation array")
    if not (
        np.isfinite(rule.max_gap_m)
        and rule.max_gap_m > 0
        and np.isfinite(rule.noise_floor_m)
        and rule.noise_floor_m > 0
        and isinstance(rule.plane_iterations, int)
        and rule.plane_iterations > 0
    ):
        raise ValueError("Invalid graph rule")
    if z.shape != (profile["height"], profile["width"]):
        raise ValueError("Profile shape mismatch")
    valid = np.isfinite(z)
    yy, xx = np.where(valid)
    n = len(yy)
    v = z[valid]
    dx0, dy0 = metric(profile, yy, yy)
    minstep = min(float(np.min(dx0)) if n else dy0, dy0)
    max_offset = int(np.ceil(rule.max_gap_m / minstep))
    ids = np.full(z.shape, n, np.int32)
    ids[valid] = np.arange(n, dtype=np.int32)
    links = np.full((n, 8), n, np.int32)
    sx = np.zeros((n, 8))
    sy = np.zeros((n, 8))
    for j, (oy, ox) in enumerate(DIRECTIONS):
        for step in range(1, max_offset + 1):
            ry = yy + step * oy
            cx = xx + step * ox
            ok = (
                (links[:, j] == n)
                & (ry >= 0)
                & (ry < z.shape[0])
                & (cx >= 0)
                & (cx < z.shape[1])
            )
            ii = np.flatnonzero(ok)
            target = ids[ry[ii], cx[ii]]
            keep = target < n
            ii = ii[keep]
            target = target[keep]
            dx, dy = metric(profile, yy[ii], ry[ii])
            inside = np.hypot(step * ox * dx, step * oy * dy) <= rule.max_gap_m
            ii = ii[inside]
            target = target[inside]
            dx = dx[inside]
            links[ii, j] = target
            sx[ii, j] = step * ox * dx
            sy[ii, j] = -step * oy * dy
    exists = links < n
    dz = np.r_[v, 0.0][links] - v[:, None]
    fit_weights = exists.astype(float)
    gx = np.zeros(n)
    gy = np.zeros(n)
    for _ in range(rule.plane_iterations):
        a = np.sum(fit_weights * sx * sx, axis=1)
        b = np.sum(fit_weights * sx * sy, axis=1)
        c = np.sum(fit_weights * sy * sy, axis=1)
        u = np.sum(np.where(exists, fit_weights * sx * dz, 0), axis=1)
        w = np.sum(np.where(exists, fit_weights * sy * dz, 0), axis=1)
        ridge = 1e-9 * (a + c + 1)
        a += ridge
        c += ridge
        det = a * c - b * b
        gx = np.divide(u * c - w * b, det, out=np.zeros(n), where=det > 0)
        gy = np.divide(w * a - u * b, det, out=np.zeros(n), where=det > 0)
        residual = np.where(exists, abs(dz - gx[:, None] * sx - gy[:, None] * sy), 0.0)
        # Fixed, bounded original-neighbor statistics. No domain percentile.
        count = exists.sum(axis=1)
        sorted_res = np.sort(np.where(exists, residual, np.inf), axis=1)
        med = (
            sorted_res[np.arange(n), np.maximum((count - 1) // 2, 0)]
            if n
            else np.zeros(0)
        )
        sigma = np.where(
            count > 0, np.maximum(rule.noise_floor_m, 1.4826 * med), rule.noise_floor_m
        )
        fit_weights = exists / (1.0 + (residual / sigma[:, None]) ** 2)
    peer_gx = np.r_[gx, 0.0][links]
    peer_gy = np.r_[gy, 0.0][links]
    peer_sigma = np.r_[sigma, rule.noise_floor_m][links]
    discrepancy = 0.5 * (
        abs(dz - gx[:, None] * sx - gy[:, None] * sy)
        + abs(dz - peer_gx * sx - peer_gy * sy)
    )
    scale = 0.5 * (sigma[:, None] + peer_sigma)
    length = np.hypot(sx, sy)
    dx0, dy0 = metric(profile, yy, yy)
    cell = np.sqrt(dx0 * dy0)
    spatial = np.divide(
        cell[:, None], length, out=np.zeros_like(length), where=length > 0
    )
    return {
        "links": links,
        "rows": yy,
        "cols": xx,
        "shape": z.shape,
        "valid": valid,
        "max_gap_m": rule.max_gap_m,
        "lengths": length,
        "raw_cost": (discrepancy / scale) ** 2,
        "spatial": spatial,
        "edge_count": int(exists.sum()),
        "max_edge_m": float(length.max()) if n else 0.0,
    }


def grid_edge_limit(profile):
    """At least immediate eight-cell adjacency, including at 1 arc-second."""
    rows = np.array([0, profile["height"] - 1])
    dx, dy = metric(profile, rows, rows)
    return max(30.0, float(np.hypot(dx.max(), dy)) * (1 + 1e-9))
