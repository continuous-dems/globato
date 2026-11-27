"""Connectivity first pass: reuse upstream geometry; add original-source gross-error screen.

This is an opt-in method, NOT used by upstream geometry. Gross rejection
masks are preserved separately for restoration protection and cleanup.
"""

from pathlib import Path
import numpy as np
from . import gross_errors, source_windows
from .sources import MaskWriter
from .io import windows, write


def reject(pool, parent, dem, out, level, consensus, settings, workers):
    out = Path(out)
    geometry_path, geometry_manifest = source_windows.reject(
        pool,
        parent,
        dem,
        out / "GEOMETRY_PASS",
        level,
        consensus,
        settings=settings,
        workers=workers,
    )
    # source_windows.reject closes pool's raster handles; SourcePool can reopen them.
    screened = MaskWriter(out / "MASKS", pool)
    permanent = MaskWriter(out / "GROSS_LOCK/MASKS", pool)
    stats = {r["source_id"]: dict(observed=0, rejected=0) for r in pool.records}
    added = 0
    p = gross_errors.Policy()
    dy, dx = gross_errors.native_pixel_m(pool.inv)
    halo = max(4, int(np.ceil(p.radius_m / min(dy, dx))) + 10)
    for core, bounds, inner in windows(pool.shape, 256, halo):
        extra = gross_errors.screen_block(pool, bounds, parent, p)
        for rec in pool.records:
            sid = rec["source_id"]
            z, valid, _, prev = pool.source(rec, core, geometry_manifest)
            # SourcePool() returns manifest decision on the same grid.
            flags = extra.get(sid)
            local = np.zeros_like(valid) if flags is None else flags[inner] & valid
            combined = prev | local
            screened.put(rec, core, combined, previous=prev)
            permanent.put(rec, core, local)
            added += int((local & ~prev).sum())
            stats[sid]["observed"] += int(valid.sum())
            stats[sid]["rejected"] += int((combined & valid).sum())
    gross_path, _ = permanent.finish(dict(role="permanent_gross_veto"))
    final_path, data = screened.finish(
        dict(
            source_counts=stats,
            consensus=bool(consensus),
            level=level,
            gross_lock_manifest=str(gross_path),
            gross_candidate_source_cells=added,
        )
    )
    # Keep the receipt in the manifest, as the v64 connectivity engine requires.
    data["gross_lock_manifest"] = str(gross_path)
    write(final_path, data)
    return final_path, data
