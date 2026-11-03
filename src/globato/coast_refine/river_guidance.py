"""Lightweight Coast Refine river interpolation guidance.

Science borrowed deliberately from the separate GLOS river workflow:
  * river/science/wse_profiles.py: PAVA longitudinal non-increasing guidance
  * river/science/surface_interpolation.py: cubic smoothstep lateral taper

This is NOT the river-bathymetry solver and cannot infer channel depth from
banks. It only constructs a continuous bank-to-bank proxy for *unobserved*
river cells, for simple elongated reaches with adequate observations on both
banks. Meanders/branches/insufficient support retain legacy behavior.
"""

from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi

from .grid_metrics import cell_spacing_m


def pava_non_decreasing(values):
    """PAVA from GLOS river/science/wse_profiles.py (pure numerical kernel)."""
    y = np.asarray(values, dtype=float)
    if y.size <= 1:
        return y.copy()
    starts, ends, weights, means = [], [], [], []
    for i, v in enumerate(y):
        starts.append(i)
        ends.append(i + 1)
        weights.append(1.0)
        means.append(float(v))
        while len(means) >= 2 and means[-2] > means[-1]:
            w = weights[-2] + weights[-1]
            m = (means[-2] * weights[-2] + means[-1] * weights[-1]) / w
            ends[-2] = ends[-1]
            weights[-2] = w
            means[-2] = m
            starts.pop()
            ends.pop()
            weights.pop()
            means.pop()
    out = np.empty_like(y)
    for first, last, mean in zip(starts, ends, means):
        out[first:last] = mean
    return out


def bank_smooth_surface(
    raw_z, observed, land, river, *, transform, crs, max_bank_gap_m=20.0
):
    """Return (provisional surface, acceptance mask, diagnostics).

    A simple, elongated reach is fitted from observed banks. A robust PAVA
    centerline prior prevents bank noise from creating downstream dams; any
    measured channel bathymetry anchors nearby inferred stations while always
    remaining immutable. Cubic smoothstep connects the guide to both banks.

    This is a terrain-continuity model, NOT a depth prediction.
    """
    z = np.asarray(raw_z, dtype=float)
    measured = np.asarray(observed, bool) & np.isfinite(z)
    land = np.asarray(land, bool)
    water = np.asarray(river, bool)
    if z.shape != measured.shape or z.shape != land.shape or z.shape != water.shape:
        raise ValueError("river_bank grid shape mismatch")
    dy, dx = cell_spacing_m(transform, crs, z.shape)
    out = np.full(z.shape, np.nan, dtype="float64")
    accepted = np.zeros(z.shape, dtype=bool)
    stats = {
        "method": "bank_guided_pava_smoothstep",
        "components": 0,
        "accepted_components": 0,
        "skipped_components": 0,
        "modeled_cells": 0,
        "observed_cells_preserved": int(np.count_nonzero(water & measured)),
        "low_gradient_components": 0,
        "reason_counts": {},
    }
    labels, n = ndi.label(water, structure=ndi.generate_binary_structure(2, 1))
    bank_support = measured & land & ~water
    for k in range(1, n + 1):
        comp = labels == k
        if int(comp.sum()) < 50:
            continue
        stats["components"] += 1

        def skip(reason):
            stats["skipped_components"] += 1
            stats["reason_counts"][reason] = stats["reason_counts"].get(reason, 0) + 1

        rc = np.argwhere(comp)
        xy = rc[:, ::-1] * [dx, dy]
        origin = xy.mean(axis=0)
        centered = xy - origin
        eig, vec = np.linalg.eigh(centered.T @ centered / max(1, len(xy)))
        if eig[0] <= 0 or eig[1] / eig[0] < 6.0:
            skip("complex_or_branched_reach")
            continue
        axis = vec[:, 1]
        normal = np.array([-axis[1], axis[0]])
        s = centered @ axis
        lateral = centered @ normal
        length = float(np.ptp(s))
        if length < 40 or length / max(2 * np.sqrt(max(eig[0], 0)), 1) < 4:
            skip("short_reach")
            continue
        # Measured channel bathymetry is a hard observation, not a reason to
        # abandon guidance for every other unmeasured cell in the same reach.
        # Its influence is limited to nearby stations and never replaces data.
        # Sample an adjacent ring no farther than max_bank_gap_m from the channel.
        d_to_river = ndi.distance_transform_edt(~comp, sampling=(dy, dx))
        bank_ring = bank_support & (d_to_river > 0) & (d_to_river <= max_bank_gap_m)
        if int(bank_ring.sum()) < 12:
            skip("no_bank_observations")
            continue
        rb = np.argwhere(bank_ring)
        bxy = rb[:, ::-1] * [dx, dy] - origin
        bs = bxy @ axis
        bl = bxy @ normal
        # Restrict to station range of this river component. Other components
        # cannot donate observed cells through the river water mask.
        within = (bs >= s.min()) & (bs <= s.max())
        bs, bl, bv = bs[within], bl[within], z[rb[within, 0], rb[within, 1]]
        if len(bs) < 12:
            skip("insufficient_bank_support")
            continue
        step = max(8.0, 2 * min(dx, dy))
        bins = max(5, int(np.ceil(length / step)))
        centers = s.min() + (np.arange(bins) + 0.5) * length / bins
        bi = np.clip(np.floor((bs - s.min()) / length * bins).astype(int), 0, bins - 1)
        # Determine center of the river locally so each bank is classified
        # relative to the river's shape, not the arbitrary raster origin.
        ri = np.clip(np.floor((s - s.min()) / length * bins).astype(int), 0, bins - 1)
        mid = np.full(bins, np.nan)
        left_edge = np.full(bins, np.nan)
        right_edge = np.full(bins, np.nan)
        for b in range(bins):
            local = lateral[ri == b]
            if local.size:
                mid[b] = float(np.median(local))
                left_edge[b] = float(local.min())
                right_edge[b] = float(local.max())
        good_mid = np.isfinite(mid)
        if good_mid.sum() < bins * 0.8:
            skip("noncontinuous_reach")
            continue
        mid = np.interp(np.arange(bins), np.where(good_mid)[0], mid[good_mid])
        # A PCA axis is not reliable when the channel folds back along itself.
        if np.any(np.diff(mid) > max(50.0, length * 0.3)):
            skip("complex_reach")
            continue
        bm = np.interp(bs, centers, mid, left=mid[0], right=mid[-1])
        side = bl < bm
        bank_values = []
        for side_choice in (True, False):
            v = np.full(bins, np.nan)
            count = np.zeros(bins, int)
            for b in range(bins):
                vals = bv[(bi == b) & (side == side_choice)]
                if vals.size:
                    count[b] = vals.size
                    v[b] = float(np.median(vals))
            valid = np.isfinite(v)
            if valid.sum() < max(3, int(bins * 0.2)) or np.max(
                np.diff(np.r_[-1, np.where(valid)[0], bins])
            ) > max(12, int(bins * 0.5)):
                break
            v = np.interp(np.arange(bins), np.where(valid)[0], v[valid])
            v = ndi.gaussian_filter1d(v, sigma=max(0.75, 15 / step), mode="nearest")
            bank_values.append(v)
        if len(bank_values) != 2:
            skip("both_banks_not_supported")
            continue
        left, right = bank_values
        # Choose the downstream direction from the broad two-bank slope.
        # When it is ambiguous, do not invent a flow direction.
        trend = 0.5 * (left + right)
        ambiguous = abs(float(trend[0] - trend[-1])) < 0.05
        if ambiguous:
            # No defensible downslope direction. Keep a smoothly bank-derived
            # low-gradient proxy, not a made-up 0 m datum or a forced flow slope.
            stats["low_gradient_components"] += 1
            left[:] = float(np.median(left))
            right[:] = float(np.median(right))
        # PAVA from original river WSE model removes reverse-slope dams in the
        # *guidance* only. Hard observed bank/bathy cells are untouched.
        decreasing_in_s = bool(trend[0] > trend[-1])

        def enforce(v):
            if decreasing_in_s:
                return -pava_non_decreasing(-v)
            return pava_non_decreasing(v)

        left, right = enforce(left), enforce(right)
        # A single longitudinal *channel* guide prevents spurious thalweg
        # dams caused by independently varying banks or channel width.
        # Bank elevations do not reveal depth: absent bathymetry this proxy
        # is the lower of the two smoothly fitted bank profiles.
        center = enforce(np.minimum(left, right))
        bed_rc = np.argwhere(comp & measured)
        if len(bed_rc):
            bed_xy = bed_rc[:, ::-1] * [dx, dy] - origin
            bed_s = bed_xy @ axis
            bed_bin = np.clip(
                np.floor((bed_s - s.min()) / length * bins).astype(int), 0, bins - 1
            )
            bed_z = z[bed_rc[:, 0], bed_rc[:, 1]]
            bed_stations = np.full(bins, np.nan)
            for b in np.unique(bed_bin):
                bed_stations[b] = float(np.median(bed_z[bed_bin == b]))
            valid_stations = np.where(np.isfinite(bed_stations))[0]
            # Only anchor the provisional profile within a few channel widths
            # of a real bed measurement. Beyond that, rely on bank guidance.
            nearest = np.min(
                np.abs(np.arange(bins)[:, None] - valid_stations[None, :]), axis=1
            )
            spacing_station = length / bins
            influence = np.clip(
                1.0 - nearest * spacing_station / max(30.0, 3 * 2 * np.sqrt(eig[0])),
                0,
                1,
            )
            bed_interp = np.interp(
                np.arange(bins), valid_stations, bed_stations[valid_stations]
            )
            center = enforce(
                (1 - influence) * center + influence * np.minimum(center, bed_interp)
            )
            stats["observed_channel_anchor_cells"] = stats.get(
                "observed_channel_anchor_cells", 0
            ) + len(bed_rc)
        m = ~measured[rc[:, 0], rc[:, 1]]
        if not np.any(m):
            continue
        rcm = rc[m]
        sm = s[m]
        lm = lateral[m]
        ind = np.clip(np.floor((sm - s.min()) / length * bins).astype(int), 0, bins - 1)
        le = np.interp(
            np.arange(bins),
            np.where(np.isfinite(left_edge))[0],
            left_edge[np.isfinite(left_edge)],
        )
        re = np.interp(
            np.arange(bins),
            np.where(np.isfinite(right_edge))[0],
            right_edge[np.isfinite(right_edge)],
        )
        q = np.clip((lm - le[ind]) / np.maximum(re[ind] - le[ind], min(dx, dy)), 0, 1)
        # Reuse the original GLOS cubic smoothstep as a lateral *bank taper*.
        # It equals zero at the center channel and one at each adjacent bank.
        bank_fraction = np.clip(2.0 * np.abs(q - 0.5), 0, 1)
        taper = bank_fraction * bank_fraction * (3 - 2 * bank_fraction)
        selected_bank = np.where(q <= 0.5, left[ind], right[ind])
        estimate = center[ind] + taper * (selected_bank - center[ind])
        out[rcm[:, 0], rcm[:, 1]] = estimate
        accepted[rcm[:, 0], rcm[:, 1]] = True
        stats["accepted_components"] += 1
        stats["modeled_cells"] += int(len(rcm))
    return out, accepted, stats
