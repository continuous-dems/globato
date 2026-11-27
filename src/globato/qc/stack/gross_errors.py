"""First-pass gross-error screening from original measured source cells.

A stable questioned source requires a stable peer from an independently known
survey group. Unstable sources may use other files under the existing policy.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy import ndimage as ndi

SCREEN_MODE = "C"  # Experimental; native science digest includes this file


@dataclass(frozen=True)
class Policy:
    gross_difference_m: float = 500.0
    radius_m: float = 260.0
    reference_std_m: float = 10.0
    min_reference_neighbors: int = 12
    peer_stable_fraction: float = 0.80
    suspect_stable_fraction: float = 0.50
    max_slope_m_per_m: float = 2.0
    extra_slope_allowance_m: float = 100.0
    allow_same_survey: bool = (
        True  # EXPERIMENTAL only; same survey IS NOT independent corroboration
    )


def native_pixel_m(inventory):
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    g = inventory["grid"]
    a = Affine(*g["affine"])
    crs = CRS.from_string(g["crs"])
    if abs(a.b) > 1e-10 or abs(a.d) > 1e-10:
        raise ValueError("Rotated native grid not supported by gross-screen experiment")
    if crs.is_geographic:
        # Geographic X is the shorter physical dimension at mid latitude.
        lat = a.f + 0.5 * g["shape"][0] * a.e
        return (
            max(0.1, abs(a.e) * 111132.0),
            max(0.1, abs(a.a) * 111320.0 * math.cos(math.radians(lat))),
        )
    unit = crs.linear_units_factor[1]
    return (max(0.1, abs(a.e) * unit), max(0.1, abs(a.a) * unit))


def stable_mask(z, valid, p=Policy()):
    """Qualifying reference: local density and SD in 9x9 *measured* cells."""
    v = valid.astype("float64")
    count = ndi.uniform_filter(v, size=9, mode="constant") * 81
    mean = (
        ndi.uniform_filter(
            np.where(valid, z, 0).astype("float64"), size=9, mode="constant"
        )
        * 81
        / np.maximum(count, 1)
    )
    sq = (
        ndi.uniform_filter(
            np.where(valid, z * z, 0).astype("float64"), size=9, mode="constant"
        )
        * 81
        / np.maximum(count, 1)
    )
    return (
        valid
        & (count >= p.min_reference_neighbors)
        & (np.maximum(0.0, sq - mean * mean) <= p.reference_std_m**2)
    )


def screen_block(pool, window, parent, policy=Policy()):
    """Window-local cached peer tests; distance maps built once per peer."""
    rows = list(pool.active_records(window))
    if len(rows) < 2:
        return {}
    layers = []
    for r in rows:
        z, valid, _, prior = pool.source(r, window, parent)
        layers.append((r["source_id"], r["group"], z, valid & ~prior))
    trusted = {}
    suspects = []
    for i, (sid, group, z, valid) in enumerate(layers):
        count = int(valid.sum())
        if not count:
            continue
        stable = stable_mask(z, valid, policy)
        fraction = float(stable.sum()) / count
        if fraction < policy.suspect_stable_fraction or SCREEN_MODE == "C":
            suspects.append(i)
        if (
            stable.sum() >= policy.min_reference_neighbors
            and fraction >= policy.peer_stable_fraction
        ):
            trusted[i] = stable
    if not suspects or not trusted:
        return {}
    pixel_m = native_pixel_m(pool.inv)
    distances = {
        i: ndi.distance_transform_edt(~stable, sampling=pixel_m, return_indices=True)
        for i, stable in trusted.items()
    }
    flags = {}
    for i in suspects:
        sid, group, z, valid = layers[i]
        fraction = float(stable_mask(z, valid, policy).sum()) / max(int(valid.sum()), 1)
        result = np.zeros(z.shape, bool)
        zlo, zhi = float(np.nanmin(z[valid])), float(np.nanmax(z[valid]))
        for j, (other_id, peer_group, peer_z, peer_valid) in enumerate(layers):
            if j == i or j not in distances:
                continue
            if not policy.allow_same_survey and peer_group == group:
                continue
            # The stable-source exception may only use an independently
            # identified survey group. Unknown groups NEVER establish independence.
            if fraction >= policy.suspect_stable_fraction and (
                not group or not peer_group or peer_group == group
            ):
                continue
            plo, phi = (
                float(np.nanmin(peer_z[peer_valid])),
                float(np.nanmax(peer_z[peer_valid])),
            )
            if max(abs(zlo - phi), abs(zhi - plo)) < policy.gross_difference_m:
                continue
            distance, index = distances[j]
            nearby = valid & (distance <= policy.radius_m)
            if not nearby.any():
                continue
            peer = peer_z[tuple(index)]
            cutoff = np.maximum(
                policy.gross_difference_m,
                policy.extra_slope_allowance_m + policy.max_slope_m_per_m * distance,
            )
            result |= nearby & (np.abs(z - peer) >= cutoff)
        if result.any():
            flags[sid] = result
    return flags
