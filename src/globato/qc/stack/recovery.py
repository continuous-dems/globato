"""Restore original source contributions, then optionally exclude redundant data.

Only retained measurements fit the terrain. Removed values are checked after
fitting, never used as support. No interpolated post-removal surface is needed.
"""

from pathlib import Path
import numpy as np
import rasterio
from . import restoration, settings, surveys, survey_coverage
from .grid import metres
from .io import log, windows, write
from .sources import MaskWriter

CORE = 256


def retained_context(pool, bounds, parent, independent_poor, poor, references):
    """One layer per known survey, plus at most one unknown terrain layer."""
    terrain, _ = pool.values(bounds, parent, local_context=True)
    anchors = terrain[: len(pool.groups)]
    original, unknown = pool.values(bounds)
    evidence = surveys.evidence(original, unknown)
    covered = np.zeros(terrain.shape[1:], bool)
    for index, group in enumerate(pool.groups):
        if group in references:
            covered |= np.isfinite(anchors[index]) & ~evidence["bad"][index]
    trusted = []
    for index, group in enumerate(pool.groups):
        if group in independent_poor:
            continue
        values = anchors[index].copy()
        if group in poor:
            values[covered] = np.nan
        trusted.append(values)
    if pool.by_group[None]:
        trusted.append(terrain[-1])
    context = np.stack(trusted) if trusted else np.empty((0, *covered.shape), "float32")
    return context, original, unknown, evidence


def predict_levels(context, removed, inner, spacing, maximum, scales=(1.0, 1.0)):
    """Higher strengths extend recovery while preserving earlier predictions."""
    predictions = {0: np.full(removed.shape, np.nan, "float32")}
    for level in range(1, maximum + 1):
        prediction = np.full(removed.shape, np.nan, "float32")
        if removed[inner].any() and len(context):
            prediction = restoration.tile_model(
                context, removed, spacing, settings.restoration_policy(level, *scales)
            )["prediction"]
        earlier = np.isfinite(predictions[level - 1])
        prediction[earlier] = predictions[level - 1][earlier]
        predictions[level] = prediction
    return predictions


def clear_survey_outliers(group, pool, evidence, votes, categories, rows, cols):
    """Recovery cannot reinstate an independently supported survey outlier."""
    clear = np.zeros(evidence["center"].shape, bool)
    if group in pool.group_index:
        index = pool.group_index[group]
        clear = evidence["bad"][index].copy()
        clear[rows, cols] |= (categories == 3) & ~votes[index]
    return clear


def _restore_block(pool, parent, block, sets, spacing, evaluated, scales):
    """Restoration decisions for one block, for every source and level.

    Returns one item per source in pool order: None when the source has no
    data near the block, otherwise {level: (final, previous, restored_cells)}.
    """
    core, bounds, inner = block
    independent_poor, poor, references = sets
    context, original, unknown, evidence = retained_context(
        pool, bounds, parent, independent_poor, poor, references
    )
    removed = np.zeros(context.shape[1:], bool)
    active_ids = {r["source_id"] for r in pool.active_records(bounds)}
    for source in pool.active_records(bounds):
        removed |= pool.rejection_mask(source, bounds, parent)
    predictions = predict_levels(
        context, removed, inner, spacing, max(evaluated), scales
    )
    rows, cols = np.nonzero(np.isfinite(predictions[max(evaluated)]))
    votes, categories = surveys.qualified_votes(
        original[:, rows, cols], 3.5, unknown=unknown[rows, cols]
    )
    out = []
    for source in pool.records:
        if source["source_id"] not in active_ids:
            out.append(None)
            continue
        z, valid, _, rejected = pool.source(source, bounds, parent)
        clear = clear_survey_outliers(
            source["group"], pool, evidence, votes, categories, rows, cols
        )
        decisions = {}
        for level in evaluated:
            recovered = (
                restoration.accept_source(
                    z,
                    valid,
                    rejected,
                    predictions[level],
                    clear_outlier=clear,
                    policy=settings.restoration_policy(level, *scales),
                )
                if level
                else np.zeros(z.shape, bool)
            )
            decisions[level] = (
                (rejected & ~recovered)[inner],
                rejected[inner],
                int(recovered[inner].sum()),
            )
        out.append(decisions)
    return out


_WORKER: dict = {}


def _init_worker(inventory, parent, sets, spacing, evaluated, scales):
    import atexit

    from .sources import SourcePool

    pool = SourcePool(inventory)
    # Close datasets before interpreter shutdown, not during it.
    atexit.register(pool.close)
    _WORKER.update(pool=pool, args=(parent, sets, spacing, evaluated, scales))


def _worker_block(block):
    parent, sets, spacing, evaluated, scales = _WORKER["args"]
    return _restore_block(
        _WORKER["pool"], parent, block, sets, spacing, evaluated, scales
    )


def restore(
    pool,
    parent,
    baseline,
    out,
    levels,
    quality,
    exclusion="off",
    scales=(1.0, 1.0),
    workers=1,
):
    """Bounded recovery stage followed by a separate optional exclusion stage.

    Blocks are independent; with ``workers`` > 1 they are evaluated in worker
    processes and written here in the original block order, so the masks and
    their hashes do not depend on the worker count.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    requested = sorted(set(levels))
    # Near-complete eligibility must be fixed across all recovery strengths.
    evaluated = list(range(4)) if exclusion == "near-complete" else requested
    writers = {level: MaskWriter(out / f"R{level}/MASKS", pool) for level in evaluated}
    totals = {level: 0 for level in evaluated}
    flags, independent_poor, poor, references = survey_coverage.quality_flags(
        pool, parent, quality
    )
    sets = (independent_poor, poor, references)
    with rasterio.open(baseline) as before:
        dx, dy = metres(before)
    spacing = (dy, dx)
    halo = restoration.context_halo(spacing, settings.restoration_policy(0, *scales))
    blocks = list(windows(pool.shape, CORE, halo))
    executor = None
    if workers > 1 and len(blocks) > 1:
        from .parallel import process_pool

        executor = process_pool(
            workers, _init_worker, (pool.inv, parent, sets, spacing, evaluated, scales)
        )
        results = executor.map(_worker_block, blocks, chunksize=4)
    else:
        results = (
            _restore_block(pool, parent, b, sets, spacing, evaluated, scales)
            for b in blocks
        )
    try:
        for number, ((core, _, _), decisions) in enumerate(zip(blocks, results), 1):
            for source, per_level in zip(pool.records, decisions):
                if per_level is None:
                    zero = np.zeros(tuple(i.stop - i.start for i in core), bool)
                    for writer in writers.values():
                        writer.put(source, core, zero)
                    continue
                for level, writer in writers.items():
                    final, previous, restored = per_level[level]
                    writer.put(source, core, final, previous=previous)
                    totals[level] += restored
            if number % 64 == 0:
                log(f"Restoration blocks complete: {number}")
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)
    branches = {
        level: writer.finish(
            dict(
                restored_source_cells=totals[level],
                terrain_restoration=level,
                source_counts=parent["source_counts"],
            )
        )
        for level, writer in writers.items()
    }
    pool.close()
    results = {level: branches[level] for level in requested}
    if exclusion != "off":
        results = survey_coverage.exclude(
            pool, parent, branches, out, requested, poor, references, exclusion
        )
    write(
        out / "DECISIONS.json",
        dict(
            restored={level: totals[level] for level in requested},
            survey_quality=flags,
            independent_poor=sorted(independent_poor),
            poor=sorted(poor),
            reference=sorted(references),
            exclusion=exclusion,
        ),
    )
    return results
