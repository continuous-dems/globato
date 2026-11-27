"""Experimental connectivity method in the PR #289 global-hook architecture.

Reuses upstream source inventory, WorkingSurface callback, geometry second pass,
restoration and accepted side-stack publishing. Not asserted science-equivalent
until full MBDB field validation against the frozen v64 masks.
"""

from pathlib import Path
import time
from . import (
    source_windows,
    connectivity_first_pass,
    recovery,
    survey_coverage,
    cleanup,
)
from .io import read, write, digest, sha
from .sources import SourcePool, zero, MaskWriter


def _validate(pool, path):
    doc = read(path)
    if set(doc["masks"]) != {r["source_id"] for r in pool.records}:
        raise RuntimeError("Connectivity manifest source inventory changed")
    for rec in doc["masks"].values():
        if sha(rec["mask"]) != rec["mask_sha256"]:
            raise RuntimeError("Connectivity source rejection mask changed")
    return doc


def _reimpose_gross(pool, source, lock, out):
    """Gross evidence is a permanent rejection; restoration must not clear it."""
    from .io import windows

    writer = MaskWriter(Path(out) / "MASKS", pool)
    stats = {r["source_id"]: dict(observed=0, rejected=0) for r in pool.records}
    for core, _, _ in windows(pool.shape, 256, 0):
        for rec in pool.records:
            _, valid, _, previous = pool.source(rec, core, source)
            locked = pool.rejection_mask(rec, core, lock)
            new = previous | locked
            writer.put(rec, core, new, previous=previous)
            d = stats[rec["source_id"]]
            d["observed"] += int(valid.sum())
            d["rejected"] += int((new & valid).sum())
    path, data = writer.finish(
        dict(source_counts=stats, role="gross_locked_restoration")
    )
    return path, data


def process(inventory, settings, cache, build, workers=1):
    if settings.max_passes != 2 or settings.stop_when_stable:
        raise ValueError("Connectivity requires exactly two forced passes")
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    pool = SourcePool(inventory)
    started = time.perf_counter()
    try:
        start = cache / "zero/MANIFEST.json"
        if start.exists():
            original = _validate(pool, start)
        else:
            start, original = zero(pool, start.parent)
        baseline = str(build(None, cache / "baseline"))
        t0 = time.perf_counter()
        pass1_dir = cache / "connectivity" / "pass1" / original["signature"]
        pass1 = pass1_dir / "MASKS/MANIFEST.json"
        if pass1.exists():
            first = _validate(pool, pass1)
        else:
            pass1, first = connectivity_first_pass.reject(
                pool,
                original,
                baseline,
                pass1_dir,
                settings.aggressiveness,
                settings.consensus,
                settings,
                workers,
            )
        gross = Path(first["gross_lock_manifest"])
        gross_doc = _validate(pool, gross)
        first_seconds = time.perf_counter() - t0
        context1 = str(build(pass1, cache / "dems" / first["signature"]))
        t1 = time.perf_counter()
        pass2_dir = cache / "connectivity" / "pass2" / first["signature"]
        pass2 = pass2_dir / "MASKS/MANIFEST.json"
        if pass2.exists():
            second = _validate(pool, pass2)
        else:
            pass2, second = source_windows.reject(
                pool,
                first,
                context1,
                pass2_dir,
                settings.aggressiveness,
                settings.consensus,
                settings=settings,
                workers=workers,
            )
        second_seconds = time.perf_counter() - t1
        t2 = time.perf_counter()
        needs_recovery = (
            settings.terrain_restoration > 0 and second["rejected_source_cells"] > 0
        )
        if needs_recovery or settings.survey_exclusion != "off":
            quality = survey_coverage.original_quality(
                pool, cache / "survey_quality.json", workers=workers
            )
            dest = (
                cache
                / "connectivity"
                / "restoration"
                / second["signature"]
                / (f"R{settings.terrain_restoration}_{settings.survey_exclusion}")
            )
            final_recovery = dest / "COMPLETE.json"
            if final_recovery.exists():
                restored_path = Path(read(final_recovery)["manifest"])
                restored = _validate(pool, restored_path)
            else:
                branches = recovery.restore(
                    pool,
                    second,
                    baseline,
                    dest,
                    [settings.terrain_restoration],
                    quality,
                    settings.survey_exclusion,
                    scales=(
                        settings.restoration_length_scale,
                        settings.restoration_fit_scale,
                    ),
                    workers=workers,
                )
                restored_path, restored = branches[settings.terrain_restoration]
                write(final_recovery, dict(manifest=str(restored_path)))
        else:
            restored_path, restored = pass2, second
        # Reimpose any permanent first-pass gross flags before cleanup.
        lock_dir = cache / "connectivity" / "gross_restored" / restored["signature"]
        locked_path = lock_dir / "MASKS/MANIFEST.json"
        if locked_path.exists():
            locked = _validate(pool, locked_path)
        else:
            locked_path, locked = _reimpose_gross(pool, restored, gross_doc, lock_dir)
        restored_seconds = time.perf_counter() - t2
        t3 = time.perf_counter()
        final_dir = cache / "connectivity" / "cleanup" / locked["signature"]
        receipt = final_dir / "DONE.json"
        if receipt.exists():
            clean = read(receipt)
        else:
            clean = cleanup.run(
                pool,
                str(pass1),
                str(gross),
                str(locked_path),
                final_dir,
                detection_passes=2,
            )
        result_path = Path(clean["manifest"])
        final = _validate(pool, result_path)
        cleanup_seconds = time.perf_counter() - t3
        restored_cells = max(
            0,
            int(restored.get("restored_source_cells", 0))
            - int(locked.get("newly_rejected_source_cells", 0)),
        )
        result = dict(
            manifest=str(result_path),
            signature=final["signature"],
            passes_completed=2,
            settings=settings.to_dict(),
            history=[
                dict(pass_number=1, manifest=str(pass1), dem=context1),
                dict(pass_number=2, manifest=str(pass2), dem=None),
            ],
            rejected_source_cells=final["rejected_source_cells"],
            restored_source_cells=restored_cells,
            coverage_excluded_source_cells=final.get(
                "coverage_excluded_source_cells", 0
            ),
            cleanup=clean,
            timing_seconds=dict(
                initial_context=t0 - started,
                pass_1=first_seconds,
                pass_2=second_seconds,
                restoration=restored_seconds,
                cleanup=cleanup_seconds,
                total=time.perf_counter() - started,
            ),
        )
        write(cache / ("result_" + digest(settings.to_dict()) + ".json"), result)
        return result
    finally:
        pool.close()
