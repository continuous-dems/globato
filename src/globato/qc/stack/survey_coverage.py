"""Measured-survey quality and optional coverage-aware exclusion.

This stage reads cached source windows. It never detects terrain, fits flanks,
interpolates a DEM, or removes unique accepted coverage. Near-complete eligibility
is assessed at every restoration level so changing recovery cannot change the
survey-eligibility gate.
"""

from pathlib import Path
import numpy as np
from . import surveys as se
from .io import read, write, log, windows
from .sources import MaskWriter

CORE = 256


def _quality_block(pool, block):
    """Integer evidence counts for one block, per survey group."""
    s, b, inner = block
    values, unknown = pool.values(b)
    ev = se.evidence(values, unknown)
    counts = []
    for gi, _ in enumerate(pool.groups):
        v = np.isfinite(values[gi][inner])
        a = ev["good"][gi][inner]
        bad = ev["bad"][gi][inner]
        blocks = occupied = 0
        for ss, _, _ in windows(v.shape, 32):
            blocks += int((a[ss] | bad[ss]).any())
            occupied += int(v[ss].any())
        counts.append((int(v.sum()), int(a.sum()), int(bad.sum()), blocks, occupied))
    return counts


_WORKER: dict = {}


def _init_quality_worker(inventory):
    import atexit

    from .sources import SourcePool

    pool = SourcePool(inventory)
    atexit.register(pool.close)
    _WORKER["pool"] = pool


def _quality_worker_block(block):
    return _quality_block(_WORKER["pool"], block)


def original_quality(pool, out, workers=1):
    """Per-survey evidence totals; sums of integers, so any block order is exact."""
    if Path(out).exists():
        return read(out)
    acc = {
        g: dict(observed=0, good=0, bad=0, blocks=0, occupied=0) for g in pool.groups
    }
    blocks = list(windows(pool.shape, CORE, 1))
    executor = None
    if workers > 1 and len(blocks) > 1:
        from .parallel import process_pool

        executor = process_pool(workers, _init_quality_worker, (pool.inv,))
        results = executor.map(_quality_worker_block, blocks, chunksize=8)
    else:
        results = (_quality_block(pool, block) for block in blocks)
    try:
        for n, counts in enumerate(results, 1):
            for g, (observed, good, bad, blocks_, occupied) in zip(pool.groups, counts):
                q = acc[g]
                q["observed"] += observed
                q["good"] += good
                q["bad"] += bad
                q["blocks"] += blocks_
                q["occupied"] += occupied
            if n % 128 == 0:
                log(f"Original survey evidence blocks complete: {n}")
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)
    result = {
        g: dict(
            se.quality(a["observed"], a["good"], a["bad"], a["blocks"]),
            occupied_blocks=a["occupied"],
        )
        for g, a in acc.items()
    }
    write(out, result)
    pool.close()
    return result


def quality_flags(pool, parent, quality):
    burden = {g: dict(observed=0, rejected=0) for g in pool.groups}
    for r in pool.records:
        if r["group"] in burden:
            q = parent["source_counts"][r["source_id"]]
            b = burden[r["group"]]
            b["observed"] += q["observed"]
            b["rejected"] += q["rejected"]
    q = {
        g: se.rejection_quality(
            b["observed"], b["rejected"], quality[g]["occupied_blocks"]
        )
        for g, b in burden.items()
    }
    independent = {g for g in pool.groups if quality[g]["poor"]}
    poor = independent | {g for g in pool.groups if q[g]["rejection_flag"]}
    references = {g for g in pool.groups if q[g]["low_rejection_reference"]} - poor
    return q, independent, poor, references


def replacement_support(z, valid, group, pool, anchors, evidence, references):
    """Per-source agreement or confirmed error with an accepted reference."""
    replacement = np.zeros(z.shape, bool)
    for index, peer in enumerate(pool.groups):
        if peer in references:
            replacement |= se.redundant(
                z,
                valid,
                anchors[index],
                ~evidence["bad"][index] & np.isfinite(anchors[index]),
            ) | (
                valid
                & evidence["bad"][pool.group_index[group]]
                & evidence["good"][index]
                & np.isfinite(anchors[index])
            )
    return replacement


def eligible_surveys(pool, parent, branches, poor, references):
    """Count accepted/replaced survey area once per cell, not once per file."""
    totals = {level: {group: [0, 0] for group in poor} for level in range(4)}
    for core, bounds, inner in windows(pool.shape, CORE, 1):
        shape = tuple(axis.stop - axis.start for axis in core)
        accepted = {
            level: {g: np.zeros(shape, bool) for g in poor} for level in range(4)
        }
        replaced = {
            level: {g: np.zeros(shape, bool) for g in poor} for level in range(4)
        }
        anchors, _ = pool.values(bounds, parent)
        original, unknown = pool.values(bounds)
        evidence = se.evidence(original, unknown)
        for source in pool.records:
            group = source["group"]
            if group not in poor:
                continue
            z, valid, _, _ = pool.source(source, bounds)
            replacement = replacement_support(
                z, valid, group, pool, anchors, evidence, references
            )
            for level in range(4):
                rejected = pool.source(source, bounds, branches[level][1])[3]
                remaining = valid & ~rejected
                accepted[level][group] |= remaining[inner]
                replaced[level][group] |= (remaining & replacement)[inner]
        for level in range(4):
            for group in poor:
                totals[level][group][0] += int(accepted[level][group].sum())
                totals[level][group][1] += int(replaced[level][group].sum())
    pool.close()
    return {
        group
        for group in poor
        if all(
            totals[level][group][0]
            and totals[level][group][1] / totals[level][group][0]
            >= se.POLICY["minimum_redundant_fraction"]
            for level in range(4)
        )
    }


def exclude(pool, parent, branches, out, levels, poor, references, mode):
    """Apply the optional exclusion after terrain restoration is complete."""
    eligible = (
        eligible_surveys(pool, parent, branches, poor, references)
        if mode == "near-complete"
        else poor
    )
    results = {}
    for level in levels:
        restored = branches[level][1]
        writer = MaskWriter(Path(out) / f"R{level}/EXCLUDED_MASKS", pool)
        extra = restored_then_excluded = 0
        for core, bounds, inner in windows(pool.shape, CORE, 1):
            anchors, _ = pool.values(bounds, parent)
            original, unknown = pool.values(bounds)
            evidence = se.evidence(original, unknown)
            for source in pool.records:
                z, valid, _, rejected = pool.source(source, bounds, restored)
                add = np.zeros(z.shape, bool)
                if source["group"] in eligible:
                    add = (
                        replacement_support(
                            z,
                            valid,
                            source["group"],
                            pool,
                            anchors,
                            evidence,
                            references,
                        )
                        & ~rejected
                    )
                original_rejected = pool.source(source, bounds, parent)[3]
                extra += int(add[inner].sum())
                restored_then_excluded += int((add & original_rejected)[inner].sum())
                writer.put(
                    source, core, (rejected | add)[inner], previous=rejected[inner]
                )
        results[level] = writer.finish(
            dict(
                restored_source_cells=restored["restored_source_cells"]
                - restored_then_excluded,
                gross_restored_source_cells=restored["restored_source_cells"],
                restored_then_excluded_source_cells=restored_then_excluded,
                terrain_restoration=level,
                coverage_excluded_source_cells=extra,
                source_counts=parent["source_counts"],
            )
        )
    pool.close()
    return results
