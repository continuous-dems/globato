"""Independent-survey evidence and coverage decisions, array-only.

Survey layers are verified acquisition groups, never individual track files.
Evidence sufficiency is fixed across aggressiveness and restoration levels.
"""

import warnings
import numpy as np
from .geometry import shift


POLICY = dict(
    raw_tolerance_m=3.5,
    maximum_neighbor_step_m=3.5,
    minimum_spatial_neighbors=2,
    minimum_assessed_cells=64,
    minimum_assessed_blocks=3,
    minimum_assessed_fraction=0.25,
    poor_bad_fraction=0.50,
    reference_good_fraction=0.98,
    reference_bad_fraction=0.02,
    replacement_agreement_m=3.5,
    minimum_redundant_fraction=0.95,
)


def consensus_votes(values, raw):
    """Fixed-threshold survey votes without an all-pairs survey loop.
    An outsider pair within raw metres exists iff adjacent sorted outsiders
    are within that distance. Sorting is per native cell, not a spatial search.
    """
    v = np.asarray(values, "float32")
    g, n = v.shape
    keep = np.zeros(v.shape, bool)
    category = np.ones(n, "uint8")
    if g < 3 or n == 0:
        return keep, category
    finite = np.isfinite(v)
    count = finite.sum(axis=0)
    eligible = count >= 3
    category[eligible] = 4
    if not eligible.any():
        return keep, category
    cols = np.flatnonzero(eligible)
    v = v[:, cols]
    finite = finite[:, cols]
    count = count[cols]
    med = np.nanmedian(v, axis=0)
    near = finite & (abs(v - med) <= raw / 2)
    outsiders = finite & ~near
    core = near.sum(axis=0)
    span = np.max(np.where(finite, v, -np.inf), axis=0) - np.min(
        np.where(finite, v, np.inf), axis=0
    )
    unanimous = (core == count) & (span <= raw / 2)
    clear = (core > count / 2) & (core < count)
    gap = np.min(np.where(near, v, np.inf), axis=0) - np.max(
        np.where(outsiders, v, -np.inf), axis=0
    )
    opposite = np.min(np.where(outsiders, v, np.inf), axis=0) - np.max(
        np.where(near, v, -np.inf), axis=0
    )
    clear &= (gap >= 2 * raw) | (opposite >= 2 * raw)
    ordered = np.sort(np.where(outsiders, v, np.inf), axis=0)
    valid_pairs = np.isfinite(ordered[1:]) & np.isfinite(ordered[:-1])
    gaps = np.subtract(
        ordered[1:],
        ordered[:-1],
        out=np.full(ordered[1:].shape, np.inf, "float32"),
        where=valid_pairs,
    )
    clear &= ~np.any(gaps <= raw, axis=0)
    core_span = np.max(np.where(near, v, -np.inf), axis=0) - np.min(
        np.where(near, v, np.inf), axis=0
    )
    clear &= core_span <= raw
    keep[:, cols] = (near & clear) | (finite & unanimous)
    category[cols[unanimous]] = 2
    category[cols[clear & ~unanimous]] = 3
    return keep, category


def evidence(values, unknown, policy=POLICY):
    values = np.asarray(values, "float32")
    g, h, w = values.shape
    good = np.zeros(values.shape, bool)
    bad = np.zeros_like(good)
    center = np.full((h, w), np.nan, "float32")
    finite = np.isfinite(values)
    eligible = (finite.sum(axis=0) >= 3) & ~unknown
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(values, axis=0)
    steps = np.zeros((h, w), "float32")
    neighbors = np.zeros((h, w), "uint8")
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        other = shift(median, dr, dc)
        v = np.isfinite(median) & np.isfinite(other)
        steps = np.maximum(steps, np.where(v, abs(other - median), 0))
        neighbors += v
    eligible &= (steps <= policy["maximum_neighbor_step_m"]) & (
        neighbors >= policy["minimum_spatial_neighbors"]
    )
    # A smooth majority can hide a steep feature sampled by the questioned
    # survey. Require spatially stable support in each observed survey too;
    # isolated/jagged cases remain for the geometry route, not this broad-offset
    # evidence route. No interpolation supplies missing survey neighbors.
    source_step = np.zeros(values.shape, "float32")
    source_neighbors = np.zeros(values.shape, "uint8")
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        other = np.roll(values, (dr, dc), axis=(1, 2)).copy()
        if dr > 0:
            other[:, :dr, :] = np.nan
        if dr < 0:
            other[:, dr:, :] = np.nan
        if dc > 0:
            other[:, :, :dc] = np.nan
        if dc < 0:
            other[:, :, dc:] = np.nan
        v = finite & np.isfinite(other)
        source_step = np.maximum(source_step, np.where(v, abs(values - other), 0))
        source_neighbors += v
    stable = (source_step <= policy["maximum_neighbor_step_m"]) & (
        source_neighbors >= policy["minimum_spatial_neighbors"]
    )
    eligible &= np.all(~finite | stable, axis=0)
    rr, cc = np.nonzero(eligible)
    keep, category = consensus_votes(values[:, rr, cc], policy["raw_tolerance_m"])
    supported = (category == 2) | (category == 3)
    good[:, rr, cc] = keep & supported
    bad[:, rr, cc] = finite[:, rr, cc] & (category == 3) & ~keep
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        c = np.nanmedian(np.where(keep, values[:, rr, cc], np.nan), axis=0)
    center[rr, cc] = c
    return dict(
        good=good, bad=bad, center=center, assessed=good | bad, stable_overlap=eligible
    )


def quality(observed, good, bad, blocks, policy=POLICY):
    assessed = good + bad
    fraction = assessed / max(observed, 1)
    bf = bad / max(assessed, 1)
    gf = good / max(assessed, 1)
    sufficient = (
        assessed >= policy["minimum_assessed_cells"]
        and blocks >= policy["minimum_assessed_blocks"]
        and fraction >= policy["minimum_assessed_fraction"]
    )
    return dict(
        observed_cells=int(observed),
        assessed_cells=int(assessed),
        confirmed_good_cells=int(good),
        confirmed_bad_cells=int(bad),
        assessed_blocks=int(blocks),
        assessed_fraction=fraction,
        confirmed_bad_fraction=bf,
        poor=bool(sufficient and bf >= policy["poor_bad_fraction"]),
        reference=bool(
            sufficient
            and gf >= policy["reference_good_fraction"]
            and bf <= policy["reference_bad_fraction"]
        ),
    )


def qualified_votes(values, raw, unknown=None):
    """Survey votes with unknown identities excluded from corroboration."""
    keep, category = consensus_votes(values, raw)
    if unknown is not None and values.shape[0] >= 3:
        blocked = (np.isfinite(values).sum(axis=0) >= 3) & np.asarray(unknown, bool)
        keep[:, blocked] = False
        category[blocked] = 5
    return keep, category


def redundant(own, own_valid, peer, peer_valid, policy=POLICY):
    return (
        own_valid
        & peer_valid
        & np.isfinite(own)
        & np.isfinite(peer)
        & (abs(own - peer) <= policy["replacement_agreement_m"])
    )


EXCLUSION = dict(
    minimum_source_cells=64,
    minimum_blocks=3,
    poor_rejected_fraction=0.30,
    reference_rejected_fraction=0.05,
)


def rejection_quality(observed, rejected, blocks):
    fraction = rejected / max(observed, 1)
    sufficient = (
        observed >= EXCLUSION["minimum_source_cells"]
        and blocks >= EXCLUSION["minimum_blocks"]
    )
    return dict(
        observed_source_cells=int(observed),
        aggressive_rejected_source_cells=int(rejected),
        aggressive_rejected_fraction=float(fraction),
        occupied_blocks=int(blocks),
        rejection_flag=bool(
            sufficient and fraction >= EXCLUSION["poor_rejected_fraction"]
        ),
        low_rejection_reference=bool(
            sufficient and fraction <= EXCLUSION["reference_rejected_fraction"]
        ),
    )
