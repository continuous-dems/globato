"""Fixed original-label path masses. Parameters are applied without retracing paths."""

import numpy as np
from .graph import metric
from .native_paths import measure as native_measure


def measure(
    g, first, profile, radius_m=400.0, connection_tolerance=2.0, eligible_mask=None
):
    valid = g["valid"]
    a = np.asarray(first, float)
    if (
        a.shape != g["shape"]
        or not np.array_equal(np.isfinite(a), valid)
        or not np.isin(a[valid], [0, 1]).all()
    ):
        raise ValueError("Original first-pass evidence mismatch")
    seed = a[valid]
    n = len(seed)
    yy, xx = g["rows"], g["cols"]
    dx, dy = metric(profile, yy, yy)
    rejected = seed.copy()
    accepted = 1 - seed
    support = np.zeros(n)
    squared = np.zeros(n)
    steps = int(np.ceil(radius_m / min(float(dx.min()) if n else dy, dy))) + 1
    halo = radius_m + 2 * g["max_gap_m"]
    context = ((yy + 0.5) * dy >= halo) & ((g["shape"][0] - yy - 0.5) * dy >= halo)
    context &= ((xx + 0.5) * dx >= halo) & ((g["shape"][1] - xx - 0.5) * dx >= halo)
    eligible = (seed == 0) & context
    if eligible_mask is not None:
        eligible &= eligible_mask[valid]
    centers = np.flatnonzero(eligible).astype(np.int32)
    m = len(centers)
    if native_measure(
        g,
        seed,
        centers,
        steps,
        radius_m,
        connection_tolerance,
        rejected,
        accepted,
        support,
        squared,
    ):
        contacts = np.divide(
            support * support, squared, out=np.zeros(n), where=squared > 0
        )
        return {
            "rejected": rejected,
            "accepted": accepted,
            "contacts": contacts,
            "context": context,
        }
    ext = np.r_[seed, 0.0]
    for direction in range(8):
        # A fixed edge has the same penalty on every path that visits it.
        # Preserve the original multiplication/addition order while computing
        # this edge-local expression once instead of at every path step.
        edge_loss = (
            g["raw_cost"][:, direction] / connection_tolerance**2
            - np.log(np.maximum(g["spatial"][:, direction], 1e-30))
        ) * g["lengths"][:, direction]
        node = centers.copy()
        distance = np.zeros(m)
        loss = np.zeros(m)
        active = np.ones(m, bool)
        for _ in range(steps):
            ii = np.flatnonzero(active)
            if not len(ii):
                break
            old = node[ii]
            nxt = g["links"][old, direction]
            length = g["lengths"][old, direction]
            distance[ii] += length
            loss[ii] += edge_loss[old]
            ok = (nxt < n) & (distance[ii] <= radius_m)
            active[ii[~ok]] = False
            ii = ii[ok]
            nxt = nxt[ok]
            if not len(ii):
                continue
            node[ii] = nxt
            w = np.exp(-np.minimum(loss[ii] / distance[ii], 50)) * np.exp(
                -distance[ii] / radius_m
            )
            s = ext[nxt]
            target = centers[ii]
            rejected[target] += w * s
            accepted[target] += w * (1 - s)
            support[target] += w * s
            squared[target] += w * w * s
    contacts = np.divide(support * support, squared, out=np.zeros(n), where=squared > 0)
    halo = radius_m + 2 * g["max_gap_m"]
    context = ((yy + 0.5) * dy >= halo) & ((g["shape"][0] - yy - 0.5) * dy >= halo)
    context &= ((xx + 0.5) * dx >= halo) & ((g["shape"][1] - xx - 0.5) * dx >= halo)
    return {
        "rejected": rejected,
        "accepted": accepted,
        "contacts": contacts,
        "context": context,
    }


def decide(
    g,
    first,
    features,
    protected,
    rejection_ratio=8.0,
    score_threshold=0.70,
    minimum_contacts=12.0,
):
    if rejection_ratio <= 0 or not 0 < score_threshold < 1 or minimum_contacts < 1:
        raise ValueError("Invalid decision parameters")
    r = rejection_ratio * features["rejected"]
    a = features["accepted"]
    q = np.divide(r, r + a, out=np.zeros_like(r), where=r + a > 0)
    proposal = np.zeros(g["shape"], bool)
    proposal[g["valid"]] = (
        (first[g["valid"]] == 0)
        & (q >= score_threshold)
        & (features["contacts"] >= minimum_contacts)
        & features["context"]
    )
    return proposal & ~protected
