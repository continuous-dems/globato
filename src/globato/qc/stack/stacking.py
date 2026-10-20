"""Exact block rebuilds from original FusionState; no subtraction or interpolation.

Keep the native source order and native reduction policy. Only changed mask blocks
need original sources again. Other blocks are copied from the preceding state.
"""

import logging
import time
from contextlib import ExitStack
import numpy as np
import rasterio
from rasterio.windows import Window
from .grid import grid_offset
from .sources import SourcePool

logger = logging.getLogger(__name__)


def overlap(record, window):
    dy, dx = record["offset"]
    h, w = record["grid"]["shape"]
    y0, x0 = max(dy, window.row_off), max(dx, window.col_off)
    y1 = min(dy + h, window.row_off + window.height)
    x1 = min(dx + w, window.col_off + window.width)
    if y1 <= y0 or x1 <= x0:
        return None
    source = Window(x0 - dx, y0 - dy, x1 - x0, y1 - y0)
    target = (
        slice(int(y0 - window.row_off), int(y1 - window.row_off)),
        slice(int(x0 - window.col_off), int(x1 - window.col_off)),
    )
    return source, target


def flags(pool, manifest, record, window):
    if manifest is None:
        return np.zeros((int(window.height), int(window.width)), dtype="uint8")
    data = pool.ds(manifest["masks"][record["source_id"]]["mask"]).read(
        1, window=window
    )
    if np.any((data != 0) & (data != 1)):
        raise ValueError("Invalid source rejection flags")
    return data


def rebuild(accumulator, records, manifest, previous=None):
    """Write each global block once; return unchanged status and sparse source index."""
    started = time.monotonic()
    dst = accumulator.dataset
    pool = SourcePool(dict(grid=dict(shape=[dst.height, dst.width]), sources=records))
    occupied = {r["source_id"]: [] for r in records}
    index = previous["source_blocks"] if previous else None
    memberships = {sid: set(blocks) for sid, blocks in index.items()} if index else None
    old_masks = previous["masks"] if previous else None
    changed_sources = {
        r["source_id"]
        for r in records
        if previous
        and (manifest or {})
        .get("masks", {})
        .get(r["source_id"], {})
        .get("mask_bits_sha256")
        != (old_masks or {})
        .get("masks", {})
        .get(r["source_id"], {})
        .get("mask_bits_sha256")
    }
    rebuilt, changed, source_reads = 0, 0, 0
    try:
        with ExitStack() as cleanup:
            old = (
                cleanup.enter_context(rasterio.open(previous["state"]))
                if previous
                else None
            )
            for r in records:
                src = pool.ds(r["path"])
                r["offset"] = list(grid_offset(src, dst))
                if manifest:
                    mask = pool.ds(manifest["masks"][r["source_id"]]["mask"])
                    if grid_offset(mask, src) != (0, 0) or (
                        mask.height,
                        mask.width,
                    ) != (src.height, src.width):
                        raise ValueError(
                            "QC mask does not match the original seven-band grid"
                        )
            windows = list(dst.block_windows(1))
            last_report = time.monotonic()
            for number, (block, window) in enumerate(windows, 1):
                key = f"{block[0]},{block[1]}"
                active = [
                    r
                    for r in records
                    if (memberships is None or key in memberships[r["source_id"]])
                    and overlap(r, window)
                ]
                before = old.read(window=window) if old else None
                dirty = old is None
                for r in active:
                    if r["source_id"] in changed_sources:
                        source, _ = overlap(r, window)
                        if not np.array_equal(
                            flags(pool, manifest, r, source),
                            flags(pool, old_masks, r, source),
                        ):
                            dirty = True
                            break
                if dirty:
                    rebuilt += 1
                    data = np.zeros(
                        (7, int(window.height), int(window.width)), dtype="float64"
                    )
                    for r in active:
                        source, target = overlap(r, window)
                        src = pool.ds(r["path"])
                        if memberships is None:
                            if not np.any(src.read(2, window=source) > 0):
                                continue
                            occupied[r["source_id"]].append(key)
                        incoming = src.read(window=source)
                        source_reads += 1
                        incoming[:, flags(pool, manifest, r, source) == 1] = 0.0
                        accumulator.reduce_state(data[(slice(None), *target)], incoming)
                    if before is None or not np.array_equal(
                        data, before, equal_nan=True
                    ):
                        changed += 1
                else:
                    data = before
                dst.write(data, window=window)
                if time.monotonic() - last_report >= 10 or number == len(windows):
                    logger.info(
                        "QC stack blocks: %d/%d; rebuilt=%d; changed=%d; source reads=%d",
                        number,
                        len(windows),
                        rebuilt,
                        changed,
                        source_reads,
                    )
                    last_report = time.monotonic()
    finally:
        pool.close()
    stats = dict(
        total_blocks=len(windows),
        rebuilt_blocks=rebuilt,
        changed_blocks=changed,
        source_block_reads=source_reads,
        stacking_seconds=time.monotonic() - started,
    )
    logger.info("QC stack complete: %s", stats)
    return dict(source_blocks=index if index is not None else occupied, timings=stats)
