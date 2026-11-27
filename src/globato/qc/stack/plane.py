"""Measured-source terrain explanations using surviving ORIGINAL source cells only.

A supported explanation is a veto, never an instruction to remove a cell.
The fixed tests do not change with aggressiveness. No reference surface, DEM smoothing,
measurement-density veto, or survey-count requirement enters these tests.
"""

import numpy as np
from scipy import ndimage as ndi

TOLERANCE_M = 1.75  # Fixed evidence tolerance, independent of filtering level.
PLANE_RADIUS = 4
MIN_PLANE_CELLS = 12


def _sum(a, wy, wx):
    return ndi.correlate1d(
        ndi.correlate1d(a, wx, axis=1, mode="constant", cval=0),
        wy,
        axis=0,
        mode="constant",
        cval=0,
    )


def plane_protection(z, valid, candidates, tolerance=TOLERANCE_M):
    """Centered least-squares plane outside flagged cells, checked for support.

    This is not an iterative robust regression: excluding the suspect cells,
    requiring four-quadrant support and bounding fit RMS guard its applicability.
    """
    z = np.asarray(z, "float64")
    valid = np.asarray(valid, bool) & np.isfinite(z)
    use = valid & ~np.asarray(candidates, bool)
    w = use.astype("float64")
    v = np.where(use, z, 0.0)
    x = np.arange(-PLANE_RADIUS, PLANE_RADIUS + 1, dtype="float64")
    one = np.ones_like(x)
    n = _sum(w, one, one)
    den = np.maximum(n, 1.0)
    sx = _sum(w, one, x)
    sy = _sum(w, x, one)
    xx = _sum(w, one, x * x) - sx * sx / den
    yy = _sum(w, x * x, one) - sy * sy / den
    xy = _sum(w, x, x) - sx * sy / den
    sz = _sum(v, one, one)
    xz = _sum(v, one, x) - sx * sz / den
    yz = _sum(v, x, one) - sy * sz / den
    determinant = xx * yy - xy * xy
    rank = determinant > 1e-8 * np.maximum(xx * yy, 1.0)
    ax = np.divide(xz * yy - yz * xy, determinant, out=np.zeros_like(z), where=rank)
    ay = np.divide(yz * xx - xz * xy, determinant, out=np.zeros_like(z), where=rank)
    predicted = (sz - ax * sx - ay * sy) / den
    residual_ss = _sum(v * v, one, one) - sz * sz / den - ax * xz - ay * yz
    rms = np.sqrt(np.maximum(residual_ss, 0.0) / den)
    support = (n >= MIN_PLANE_CELLS) & rank & (rms <= tolerance)
    for ys in (x < 0, x > 0):
        for xs in (x < 0, x > 0):
            support &= _sum(w, ys.astype(float), xs.astype(float)) >= 2
    protect = (
        np.asarray(candidates, bool)
        & valid
        & support
        & (np.abs(z - predicted) <= tolerance)
    )
    return protect, dict(
        predicted=predicted.astype("float32"),
        supported=support,
        rms=rms.astype("float32"),
    )
