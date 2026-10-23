"""Cheap interpolated DEM geometry evidence, without reference surfaces or source survey labels.

Only measured sources are rejected later. This module never alters DEM depths.
"""

import numpy as np
from scipy import ndimage as ndi

# Public levels 1–3 use these cutoffs. A fixed broader set supplies protection
# evidence at every level; it is never exposed as a fourth filtering setting.
ELIGIBILITY_LEVEL = 4
LEVELS = {
    1: dict(local9=5.0, local25=8.0, flank=10.0, flank_similarity=16.0),
    2: dict(local9=3.5, local25=5.5, flank=7.0, flank_similarity=19.0),
    3: dict(local9=2.625, local25=4.125, flank=5.25, flank_similarity=19.0),
    4: dict(local9=1.75, local25=2.75, flank=3.5, flank_similarity=19.0),
}


def shift(a, dy, dx):
    """Never count rolled opposite-edge neighbors."""
    out = np.roll(a, (dy, dx), (0, 1)).copy()
    if dy > 0:
        out[:dy, :] = np.nan
    if dy < 0:
        out[dy:, :] = np.nan
    if dx > 0:
        out[:, :dx] = np.nan
    if dx < 0:
        out[:, dx:] = np.nan
    return out


def evidence(z, valid=None):
    z = np.asarray(z, dtype="float32")
    valid = (
        np.isfinite(z)
        if valid is None
        else (np.asarray(valid, dtype=bool) & np.isfinite(z))
    )
    if not valid.any():
        raise ValueError("No valid DEM cells")
    # Nearest valid value is only an edge-padding aid; never publish pad as data.
    if not valid.all():
        locations = ndi.distance_transform_edt(
            ~valid, return_distances=False, return_indices=True
        )
        filled = z[tuple(locations)]
    else:
        filled = z
    m3 = ndi.median_filter(filled, size=3, mode="nearest")
    m9 = ndi.median_filter(filled, size=9, mode="nearest")
    m25 = ndi.median_filter(filled, size=25, mode="nearest")
    residual = filled - m25
    coherent_hits = {i: np.zeros(z.shape, dtype="uint8") for i in LEVELS}
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for distance in (4, 6, 9, 12):
            lf = shift(m3, dr * distance, dc * distance)
            rt = shift(m3, -dr * distance, -dc * distance)
            offset = np.abs(filled - 0.5 * (lf + rt))
            flank_agreement = np.abs(lf - rt)
            for level, t in LEVELS.items():
                hit = (offset >= t["flank"]) & (
                    flank_agreement <= t["flank_similarity"]
                )
                coherent_hits[level] += hit.astype("uint8")
    masks = {}
    for level, t in LEVELS.items():
        local = (np.abs(filled - m9) >= t["local9"]) & (
            np.abs(residual) >= t["local25"]
        )
        line = (coherent_hits[level] >= 2) & (np.abs(residual) >= t["local25"] * 0.75)
        mask = (local | line) & valid
        masks[level] = mask
    if not all(
        np.all(masks[a] <= masks[b]) for a, b in zip(sorted(masks), sorted(masks)[1:])
    ):
        raise RuntimeError("Aggressiveness masks must be nested")
    return dict(
        masks=masks,
        reference=m25.astype("float32"),
        residual=residual.astype("float32"),
    )
