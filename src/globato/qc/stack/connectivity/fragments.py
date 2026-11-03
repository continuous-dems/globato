"""One simultaneous, bounded closure of sparse accepted fragments. No reference data."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from .graph import metric


def complete(
    g,
    first,
    gross,
    primary,
    protected,
    profile,
    radius_m=400.0,
    connection_tolerance=2.0,
):
    valid = g["valid"]
    n = int(valid.sum())
    out = np.zeros(g["shape"], bool)
    if n == 0:
        return out, {"added": 0, "components": 0}
    for array in [first, gross]:
        if (
            array.shape != g["shape"]
            or not np.array_equal(np.isfinite(array), valid)
            or not np.isin(array[valid], [0, 1]).all()
        ):
            raise ValueError("Fragment evidence mismatch")
    if np.any((gross[valid] == 1) & (first[valid] != 1)):
        raise ValueError("Gross evidence is not first-pass rejection")
    f = first[valid] == 1
    anchors = gross[valid] == 1
    removed = f | primary[valid]
    keep = ~removed
    ids = np.flatnonzero(keep)
    if not len(ids):
        return out, {"added": 0, "components": 0}
    src = np.repeat(np.arange(n), 8)
    dst = g["links"].ravel()
    exists = dst < n
    src = src[exists]
    dst = dst[exists]
    length = g["lengths"].ravel()[exists]
    cost = g["raw_cost"].ravel()[exists]

    def components(mask):
        ii = np.flatnonzero(mask)
        local = np.full(n, -1, dtype=np.int32)
        local[ii] = np.arange(len(ii))
        ok = mask[src] & mask[dst]
        matrix = coo_matrix(
            (np.ones(int(ok.sum()), np.uint8), (local[src[ok]], local[dst[ok]])),
            shape=(len(ii), len(ii)),
        ).tocsr()
        count, labels = connected_components(matrix, directed=False)
        all_labels = np.full(n, -1, dtype=np.int32)
        all_labels[ii] = labels
        return count, all_labels

    total, original_labels = components(np.ones(n, bool))
    count, labels = components(keep)
    yy, xx = g["rows"], g["cols"]
    dx, dy = metric(profile, yy, yy)

    def diameters(which, k, mask):
        lowx = np.full(k, np.inf)
        highx = np.full(k, -np.inf)
        lowy = lowx.copy()
        highy = highx.copy()
        np.minimum.at(lowx, which[mask], xx[mask])
        np.maximum.at(highx, which[mask], xx[mask])
        np.minimum.at(lowy, which[mask], yy[mask])
        np.maximum.at(highy, which[mask], yy[mask])
        middle = (lowy + highy) * 0.5
        local_dx, _ = metric(profile, middle, middle)
        return np.hypot((highx - lowx) * local_dx, (highy - lowy) * dy)

    original_diameter = diameters(original_labels, total, np.ones(n, bool))
    diameter = diameters(labels, count, keep)
    blocked = np.zeros(count, bool)
    np.logical_or.at(blocked, labels[keep], protected[valid][keep])
    halo = radius_m + 2 * g["max_gap_m"]
    context = (
        ((yy + 0.5) * dy >= halo)
        & ((g["shape"][0] - yy - 0.5) * dy >= halo)
        & ((xx + 0.5) * dx >= halo)
        & ((g["shape"][1] - xx - 0.5) * dx >= halo)
    )
    np.logical_or.at(blocked, labels[keep], ~context[keep])
    compatible = cost <= connection_tolerance**2
    boundary = keep[src] & removed[dst] & compatible
    original_reject_boundary = np.zeros(count, bool)
    use = boundary & f[dst]
    np.logical_or.at(original_reject_boundary, labels[src[use]], True)
    distances = np.full(n, np.inf)
    if anchors.any():
        use = removed[src] & removed[dst] & compatible
        matrix = coo_matrix((length[use], (src[use], dst[use])), shape=(n, n)).tocsr()
        distances = dijkstra(
            matrix,
            directed=False,
            indices=np.flatnonzero(anchors),
            min_only=True,
            limit=radius_m,
        )
    nearest = np.full(count, np.inf)
    np.minimum.at(
        nearest, labels[src[boundary]], distances[dst[boundary]] + length[boundary]
    )
    orphan = np.zeros(count, bool)
    np.logical_or.at(
        orphan, labels[keep], original_diameter[original_labels[keep]] <= g["max_gap_m"]
    )
    accepted = (
        (diameter <= g["max_gap_m"])
        & ~blocked
        & original_reject_boundary
        & ((nearest <= radius_m) | orphan)
    )
    selected = ids[accepted[labels[ids]]]
    out[yy[selected], xx[selected]] = True
    return out, {
        "added": int(len(selected)),
        "components": int(accepted.sum()),
        "gross_anchored_components": int((accepted & (nearest <= radius_m)).sum()),
        "isolated_original_components": int((accepted & orphan).sum()),
        "max_span_m": g["max_gap_m"],
        "maximum_anchor_path_m": radius_m,
        "iterations": 1,
    }
