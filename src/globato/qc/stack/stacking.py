"""Exact block rebuilds from original FusionState; no subtraction or interpolation.

Keep the native source order and native reduction policy. Only changed mask blocks
need original sources again. Other blocks are copied from the preceding state.
"""

import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import numpy as np
import rasterio
from rasterio.windows import Window
from .grid import grid_offset
from .sources import SourcePool

logger = logging.getLogger(__name__)


def written_blocks(path):
    """Stored blocks of a tiled GeoTIFF as (block height, width, {(row, col)}).

    A sparse GeoTIFF block that was never written reads as zeros, so a source
    window covering only unwritten blocks holds no measurements. None when GDAL
    cannot tell.
    """
    import warnings

    try:
        from osgeo import gdal
    except ImportError:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        ds = gdal.Open(str(path))
        if ds is None:
            return None
        band = ds.GetRasterBand(1)
        bx, by = band.GetBlockSize()
        nx = (ds.RasterXSize + bx - 1) // bx
        ny = (ds.RasterYSize + by - 1) // by
        stored = {
            (y, x)
            for y in range(ny)
            for x in range(nx)
            if band.GetMetadataItem(f"BLOCK_OFFSET_{x}_{y}", "TIFF")
        }
    return by, bx, stored


def _holds_data(info, window):
    """False only when every block under ``window`` was never written."""
    if info is None:
        return True
    by, bx, stored = info
    rows = range(
        int(window.row_off) // by, (int(window.row_off + window.height) - 1) // by + 1
    )
    cols = range(
        int(window.col_off) // bx, (int(window.col_off + window.width) - 1) // bx + 1
    )
    return any((y, x) in stored for y in rows for x in cols)


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


def _block_state(
    pool,
    old,
    records,
    memberships,
    changed_sources,
    manifest,
    old_masks,
    accumulator,
    block,
    window,
    written=None,
):
    """Rebuilt seven-band state for one global block (pure; no writes)."""
    key = f"{block[0]},{block[1]}"
    active = [
        r
        for r in records
        if (memberships is None or key in memberships[r["source_id"]])
        and overlap(r, window)
    ]
    before = old.read(window=window) if old is not None else None
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
    occupied, reads = [], 0
    if not dirty:
        return before, False, False, occupied, reads
    data = np.zeros((7, int(window.height), int(window.width)), dtype="float64")
    for r in active:
        source, target = overlap(r, window)
        if memberships is None and not _holds_data(
            (written or {}).get(r["source_id"]), source
        ):
            continue  # Never-written blocks read as zeros; don't even open it.
        src = pool.ds(r["path"])
        if memberships is None:
            if not np.any(src.read(2, window=source) > 0):
                continue
            occupied.append(r["source_id"])
        incoming = src.read(window=source)
        reads += 1
        incoming[:, flags(pool, manifest, r, source) == 1] = 0.0
        accumulator.reduce_state(data[(slice(None), *target)], incoming)
    changed = before is None or not np.array_equal(data, before, equal_nan=True)
    return data, True, changed, occupied, reads


def _bounded_map(executor, fn, items, ahead):
    """executor.map in order, with at most ``ahead`` results pending."""
    pending = deque()
    for item in items:
        pending.append(executor.submit(fn, item))
        if len(pending) >= ahead:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def rebuild(accumulator, records, manifest, previous=None, workers=1):
    """Write each global block once; return unchanged status and sparse source index.

    With ``workers`` > 1, blocks are read and reduced in threads (each with its
    own dataset handles) and written here in block order; the result does not
    depend on the worker count.
    """
    started = time.monotonic()
    dst = accumulator.dataset
    sparse = getattr(accumulator, "sparse_state", False) and not accumulator.resume
    inventory = dict(grid=dict(shape=[dst.height, dst.width]), sources=records)
    pool = SourcePool(inventory)
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
    local = threading.local()
    opened = []
    lock = threading.Lock()

    def thread_handles():
        if not hasattr(local, "pool"):
            local.pool = SourcePool(inventory)
            local.old = rasterio.open(previous["state"]) if previous else None
            with lock:
                opened.append((local.pool, local.old))
        return local.pool, local.old

    def compute(item):
        block, window = item
        p, o = thread_handles()
        return _block_state(
            p,
            o,
            records,
            memberships,
            changed_sources,
            manifest,
            old_masks,
            accumulator,
            block,
            window,
            written,
        )

    executor = None
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
            # Only the first build scans sources for data; later builds use the
            # block index it returns.
            written = (
                {r["source_id"]: written_blocks(r["path"]) for r in records}
                if memberships is None
                else {}
            )
            if workers > 1 and len(windows) > 1:
                executor = ThreadPoolExecutor(max_workers=workers)
                results = _bounded_map(executor, compute, windows, 4 * workers)
            else:
                results = (
                    _block_state(
                        pool,
                        old,
                        records,
                        memberships,
                        changed_sources,
                        manifest,
                        old_masks,
                        accumulator,
                        block,
                        window,
                        written,
                    )
                    for block, window in windows
                )
            last_report = time.monotonic()
            for number, ((block, window), state) in enumerate(zip(windows, results), 1):
                data, dirty, block_changed, block_occupied, reads = state
                key = f"{block[0]},{block[1]}"
                if dirty:
                    rebuilt += 1
                    changed += int(block_changed)
                    source_reads += reads
                    for sid in block_occupied:
                        occupied[sid].append(key)
                # A sparse state reads unwritten blocks as zero; skip the write.
                if not (sparse and not np.any(data)):
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
        if executor is not None:
            executor.shutdown(cancel_futures=True)
        for p, o in opened:
            p.close()
            if o is not None:
                o.close()
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
