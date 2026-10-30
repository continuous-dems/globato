"""Bounded native-window QC. Original arrays and interpolation are never edited."""

from pathlib import Path
import numpy as np
import rasterio
from scipy import ndimage as ndi
from .io import log, windows, raster_window, mm
from .parallel import process_pool
from .sources import MaskWriter
from . import geometry as detector, plane as terrain_protection, surveys as se

CORE = 256


def geometry(dem, out, levels, settings=None, workers=1):
    """Global nearest-valid padding, then exact finite-footprint operators."""
    if settings is None:
        thresholds = detector.LEVELS
        windows_, flanks = detector.DEFAULT_WINDOWS, detector.DEFAULT_FLANKS
    else:
        thresholds = settings.detection_levels()
        windows_, flanks = settings.median_windows, settings.flank_distances
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
    blocks = list(windows(shape, 512, detector.halo(windows_, flanks)))
    options = dict(levels=thresholds, windows=windows_, flanks=flanks)
    if workers > 1 and len(blocks) > 1:
        # Blocks are independent and each is computed exactly as below, so the
        # masks do not depend on the number of workers.
        z.flush()
        for a in masks.values():
            a.flush()
        np.save(out / "VALID.npy", valid)
        paths = {i: str(out / f"L{i}.npy") for i in masks}
        tasks = [blocks[i::workers] for i in range(workers)]
        with process_pool(workers) as pool:
            done = 0
            for count in pool.map(
                _geometry_blocks,
                [str(out / "DEM.npy")] * len(tasks),
                [str(out / "VALID.npy")] * len(tasks),
                [paths] * len(tasks),
                tasks,
                [options] * len(tasks),
            ):
                done += count
        log(f"Geometry blocks complete: {done} ({workers} workers)")
        (out / "VALID.npy").unlink(missing_ok=True)
        masks = {i: np.load(p, mmap_mode="r") for i, p in paths.items()}
    else:
        for n, (s, b, inner) in enumerate(blocks, 1):
            ev = detector.evidence(z[b], **options)
            for i, a in masks.items():
                a[s] = ev["masks"][i][inner] & valid[s]
            if n % 32 == 0:
                log(f"Geometry blocks complete: {n}")
        for a in masks.values():
            a.flush()
    del z
    return masks


def _geometry_blocks(dem, valid, paths, blocks, options):
    """Worker: evidence masks for a set of blocks, written into shared memmaps."""
    z = np.load(dem, mmap_mode="r")
    valid = np.load(valid, mmap_mode="r")
    masks = {i: np.load(p, mmap_mode="r+") for i, p in paths.items()}
    for s, b, inner in blocks:
        ev = detector.evidence(np.array(z[b]), **options)
        for i, a in masks.items():
            a[s] = ev["masks"][i][inner] & valid[s]
    for a in masks.values():
        a.flush()
    return len(blocks)


def survey_protection(values, candidates, unknown):
    """Agreement protects sources at candidates, even when consensus is off."""
    rows, cols = np.nonzero(candidates)
    keep, category = se.qualified_votes(
        values[:, rows, cols], 3.5, unknown=unknown[rows, cols]
    )
    return (rows, cols, keep), category


def source_protection(
    z, active, candidates, group, pool, votes, categories, tolerance=None
):
    """A corroborated value or supported local plane vetoes geometry removal."""
    rows, cols, keep = votes
    protected = np.zeros(z.shape, bool)
    clear_outlier = np.zeros(z.shape, bool)
    if group in pool.group_index:
        index = pool.group_index[group]
        protected[rows, cols] = keep[index]
        clear_outlier[rows, cols] = (categories == 3) & ~keep[index]
    if (candidates & active).any():
        plane, _ = terrain_protection.plane_protection(
            z,
            active,
            candidates,
            terrain_protection.TOLERANCE_M if tolerance is None else tolerance,
        )
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


def _reject_block(pool, g, parent, level, consensus, tolerance, block):
    """Rejection decisions for one block, for every source (pure; no writes).

    Returns one item per source in pool order: None when the source has no
    data near the block, else (final, prior, observed, rejected) for the core.
    """
    s, b, inner = block
    union = g[detector.ELIGIBILITY_LEVEL][b]
    candidate = g[level][b]
    vals, unknown = pool.values(b, parent)
    votes, categories = survey_protection(vals, union, unknown)
    disagreement = se.evidence(vals, unknown) if consensus else None
    active_ids = {r["source_id"] for r in pool.active_records(b)}
    out = []
    for r in pool.records:
        if r["source_id"] not in active_ids:
            out.append(None)
            continue
        z, valid, w, prior = pool.source(r, b, parent)
        active = valid & ~prior
        protected = source_protection(
            z, active, union, r["group"], pool, votes, categories, tolerance
        )
        survey_outlier = disagreement_rejection(
            z, active, r["group"], pool, disagreement
        )
        new = (candidate & active & ~protected) | survey_outlier
        final = prior | new
        out.append(
            (
                final[inner],
                prior[inner],
                int(valid[inner].sum()),
                int((final[inner] & valid[inner]).sum()),
            )
        )
    return out


_WORKER: dict = {}


def _init_reject_worker(inventory, paths, parent, level, consensus, tolerance):
    import atexit

    from .sources import SourcePool

    pool = SourcePool(inventory)
    atexit.register(pool.close)
    g = {i: np.load(p, mmap_mode="r") for i, p in paths.items()}
    _WORKER.update(pool=pool, g=g, args=(parent, level, consensus, tolerance))


def _reject_worker_block(block):
    parent, level, consensus, tolerance = _WORKER["args"]
    return _reject_block(
        _WORKER["pool"], _WORKER["g"], parent, level, consensus, tolerance, block
    )


def reject(pool, parent, dem, out, level, consensus=False, settings=None, workers=1):
    """One removal pass. With ``workers`` > 1, blocks are evaluated in worker
    processes and written here in block order, so the masks and their hashes
    do not depend on the worker count."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    g = geometry(dem, out / "GEOMETRY", [level], settings, workers)
    tolerance = None if settings is None else settings.plane_tolerance_m
    m = MaskWriter(out / "MASKS", pool)
    stats = {r["source_id"]: dict(observed=0, rejected=0) for r in pool.records}
    blocks = list(windows(pool.shape, CORE, 4))
    executor = None
    if workers > 1 and len(blocks) > 1:
        for a in g.values():
            if hasattr(a, "flush"):
                a.flush()
        paths = {i: str(out / "GEOMETRY" / f"L{i}.npy") for i in g}
        executor = process_pool(
            workers,
            _init_reject_worker,
            (pool.inv, paths, parent, level, consensus, tolerance),
        )
        results = executor.map(_reject_worker_block, blocks, chunksize=4)
    else:
        results = (
            _reject_block(pool, g, parent, level, consensus, tolerance, block)
            for block in blocks
        )
    try:
        for n, ((s, _, _), decisions) in enumerate(zip(blocks, results), 1):
            for r, item in zip(pool.records, decisions):
                if item is None:
                    # Masks must still hash every cell in every source,
                    # including empty ones.
                    m.put(r, s, np.zeros(tuple(i.stop - i.start for i in s), bool))
                    continue
                final, prior, observed, rejected = item
                m.put(r, s, final, previous=prior)
                st = stats[r["source_id"]]
                st["observed"] += observed
                st["rejected"] += rejected
            if n % 64 == 0:
                log(f"Source rejection blocks complete: {n}")
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)
    p, d = m.finish(dict(source_counts=stats, consensus=bool(consensus), level=level))
    pool.close()
    # Geometry was needed only for this exact parent; keep masks/receipts, not duplicate float DEM copies.
    del g
    for pth in (out / "GEOMETRY").glob("*.npy"):
        pth.unlink()
    return p, d
