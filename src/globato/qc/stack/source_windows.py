"""Bounded native-window QC. Original arrays and interpolation are never edited."""

from pathlib import Path
import numpy as np
import rasterio
from scipy import ndimage as ndi
from .io import log, windows, raster_window, mm
from .sources import MaskWriter
from . import geometry as detector, plane as terrain_protection, surveys as se

CORE = 256


def geometry(dem, out, levels):
    """Global nearest-valid padding, then exact finite-footprint operators."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shape = None
    with rasterio.open(dem) as ds:
        shape = ds.shape
        z = mm(out / "DEM.npy", shape, fill=np.nan)
        for s, _, _ in windows(shape):
            z[s] = (
                ds.read(1, window=raster_window(s), masked=True)
                .astype("float32")
                .filled(np.nan)
            )
    valid = np.isfinite(z)
    if not valid.any():
        raise RuntimeError("Empty parent DEM")
    if not valid.all():
        idx = ndi.distance_transform_edt(
            ~valid, return_distances=False, return_indices=True
        )
        for s, _, _ in windows(shape):
            z[s] = z[tuple(idx[(slice(None), *s)])]
        del idx
    masks = {
        i: mm(out / f"L{i}.npy", shape, "bool")
        for i in sorted(set(levels) | {detector.ELIGIBILITY_LEVEL})
    }
    for n, (s, b, inner) in enumerate(windows(shape, 512, 13), 1):
        ev = detector.evidence(z[b])
        for i, a in masks.items():
            a[s] = ev["masks"][i][inner] & valid[s]
        if n % 32 == 0:
            log(f"Geometry blocks complete: {n}")
    for a in masks.values():
        a.flush()
    del z
    return masks


def survey_protection(values, candidates, unknown):
    """Agreement protects sources at candidates, even when consensus is off."""
    rows, cols = np.nonzero(candidates)
    keep, category = se.qualified_votes(
        values[:, rows, cols], 3.5, unknown=unknown[rows, cols]
    )
    return (rows, cols, keep), category


def source_protection(z, active, candidates, group, pool, votes, categories):
    """A corroborated value or supported local plane vetoes geometry removal."""
    rows, cols, keep = votes
    protected = np.zeros(z.shape, bool)
    clear_outlier = np.zeros(z.shape, bool)
    if group in pool.group_index:
        index = pool.group_index[group]
        protected[rows, cols] = keep[index]
        clear_outlier[rows, cols] = (categories == 3) & ~keep[index]
    if (candidates & active).any():
        plane, _ = terrain_protection.plane_protection(z, active, candidates)
        protected |= plane & ~clear_outlier
    return protected


def disagreement_rejection(z, active, group, pool, evidence):
    """Optional independent route; a geometry protection does not veto it."""
    if evidence is None or group not in pool.group_index:
        return np.zeros(z.shape, bool)
    index = pool.group_index[group]
    return (
        evidence["bad"][index]
        & active
        & np.isfinite(evidence["center"])
        & (abs(z - evidence["center"]) >= 7.0)
    )


def reject(pool, parent, dem, out, level, consensus=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    g = geometry(dem, out / "GEOMETRY", [level])
    m = MaskWriter(out / "MASKS", pool)
    stats = {r["source_id"]: dict(observed=0, rejected=0) for r in pool.records}
    for n, (s, b, inner) in enumerate(windows(pool.shape, CORE, 4), 1):
        union = g[detector.ELIGIBILITY_LEVEL][b]
        candidate = g[level][b]
        vals, unknown = pool.values(b, parent)
        votes, categories = survey_protection(vals, union, unknown)
        disagreement = se.evidence(vals, unknown) if consensus else None
        active_ids = {r["source_id"] for r in pool.active_records(b)}
        for r in pool.records:
            if r["source_id"] not in active_ids:
                # Masks must still hash every cell in every source, including
                # empty ones. Avoid source reads and terrain fits here.
                m.put(r, s, np.zeros(tuple(i.stop - i.start for i in s), bool))
                continue
            z, valid, w, prior = pool.source(r, b, parent)
            active = valid & ~prior
            protected = source_protection(
                z, active, union, r["group"], pool, votes, categories
            )
            survey_outlier = disagreement_rejection(
                z, active, r["group"], pool, disagreement
            )
            new = (candidate & active & ~protected) | survey_outlier
            final = prior | new
            m.put(r, s, final[inner], previous=prior[inner])
            st = stats[r["source_id"]]
            st["observed"] += int(valid[inner].sum())
            st["rejected"] += int((final[inner] & valid[inner]).sum())
        if n % 64 == 0:
            log(f"Source rejection blocks complete: {n}")
    p, d = m.finish(dict(source_counts=stats, consensus=bool(consensus), level=level))
    pool.close()
    # Geometry was needed only for this exact parent; keep masks/receipts, not duplicate float DEM copies.
    del g
    for pth in (out / "GEOMETRY").glob("*.npy"):
        pth.unlink()
    return p, d
