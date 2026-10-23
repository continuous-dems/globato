"""Source-mask loop, independent of Fetchez and native interpolation owners.

The caller supplies ``build(manifest_path, output_dir) -> DEM path``. Only
aggressively filtered parents feed another pass; restoration runs once at end.
"""

from pathlib import Path
import numpy as np
import rasterio
from . import source_windows as removal, recovery, survey_coverage
from .io import read, write, sha, digest, fingerprint, grid, log
from .sources import SourcePool, zero


def changes(before, after):
    if Path(before).resolve() == Path(after).resolve():
        return dict(maximum_absolute_change_m=0.0, coverage_changed_cells=0)
    largest = 0.0
    coverage = 0
    with rasterio.open(before) as a, rasterio.open(after) as b:
        if grid(a) != grid(b):
            raise ValueError("QC interpolation changed the native grid")
        for _, w in a.block_windows(1):
            x = a.read(1, window=w, masked=True).astype("float64").filled(np.nan)
            y = b.read(1, window=w, masked=True).astype("float64").filled(np.nan)
            good = np.isfinite(x) & np.isfinite(y)
            coverage += int(np.count_nonzero(np.isfinite(x) != np.isfinite(y)))
            if good.any():
                largest = max(largest, float(np.max(abs(x[good] - y[good]))))
    return dict(maximum_absolute_change_m=largest, coverage_changed_cells=coverage)


def process(inventory, settings, cache, build, workers=1):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    pool = SourcePool(inventory)
    try:
        return _process(inventory, settings, cache, build, pool, workers)
    finally:
        pool.close()


def _process(inventory, settings, cache, build, pool, workers=1):
    memo = {}
    verified = {}

    def validate(path):
        d = read(path)
        if set(d["masks"]) != {r["source_id"] for r in inventory["sources"]}:
            raise RuntimeError("QC source inventory changed")
        for item in d["masks"].values():
            p = item["mask"]
            fp = fingerprint(p)
            if verified.get(p) != fp:
                if sha(p) != item["mask_sha256"]:
                    raise RuntimeError("Cached QC mask changed")
                verified[p] = fp
        return d

    def dem(path, manifest):
        key = manifest["signature"]
        if key not in memo:
            memo[key] = str(build(path, cache / "dems" / key))
        return memo[key]

    zero_path = cache / "zero/MANIFEST.json"
    if zero_path.exists():
        parent = validate(zero_path)
    else:
        zero_path, parent = zero(pool, zero_path.parent)
    baseline = dem(zero_path, parent)
    parent_dem = baseline
    stable = 0
    history = []
    for number in range(1, settings.max_passes + 1):
        log(f"Stack QC: evaluating removal pass {number}/{settings.max_passes}")
        folder = (
            cache
            / "parents"
            / f"level{settings.aggressiveness}_consensus{int(settings.consensus)}"
            / parent["signature"]
        )
        path = folder / "MASKS/MANIFEST.json"
        if path.exists():
            current = validate(path)
        else:
            path, current = removal.reject(
                pool,
                parent,
                parent_dem,
                folder,
                settings.aggressiveness,
                settings.consensus,
                settings=settings,
                workers=workers,
            )
        # A context DEM is useful only if another QC pass can consume it.
        # The final mask goes directly to the normal global build interpolation.
        terminal = number == settings.max_passes
        current_dem = None if terminal else dem(path, current)
        delta = None if terminal else changes(parent_dem, current_dem)
        observed = sum(q["observed"] for q in current["source_counts"].values())
        fraction = current["newly_rejected_source_cells"] / max(observed, 1)
        nearly_stable = (
            (
                fraction <= settings.stable_changed_fraction
                and not delta["coverage_changed_cells"]
                and delta["maximum_absolute_change_m"] <= settings.stable_dem_change_m
            )
            if delta is not None
            else None
        )
        stable = stable + 1 if nearly_stable else 0
        history.append(
            dict(
                pass_number=number,
                manifest=str(path),
                dem=current_dem,
                context_interpolated=not terminal,
                new_rejection_fraction=fraction,
                changes=delta,
                stable=nearly_stable,
            )
        )
        log(
            f"Stack QC pass {number}/{settings.max_passes}: {current['newly_rejected_source_cells']:,} new rejected source cells"
        )
        parent = current
        if current_dem is not None:
            parent_dem = current_dem
        if settings.stop_when_stable and stable >= settings.stable_consecutive_passes:
            break
    needs_recovery = (
        settings.terrain_restoration > 0 and parent["rejected_source_cells"] > 0
    )
    if not needs_recovery and settings.survey_exclusion == "off":
        final_path, final = path, dict(parent, restored_source_cells=0)
    else:
        log(
            "Stack QC: assessing original survey evidence for terrain recovery/coverage"
        )
        quality = survey_coverage.original_quality(
            pool, cache / "survey_quality.json", workers=workers
        )
        restored = (
            cache
            / "restoration"
            / parent["signature"]
            / f"level{settings.terrain_restoration}_{settings.survey_exclusion}"
        )
        done = restored / "COMPLETE.json"
        if done.exists():
            final_path = Path(read(done)["manifest"])
            final = validate(final_path)
        else:
            branches = recovery.restore(
                pool,
                parent,
                baseline,
                restored,
                [settings.terrain_restoration],
                quality,
                settings.survey_exclusion,
                scales=(
                    settings.restoration_length_scale,
                    settings.restoration_fit_scale,
                ),
                workers=workers,
            )
            final_path, final = branches[settings.terrain_restoration]
            write(done, dict(manifest=str(final_path)))
    result = dict(
        manifest=str(final_path),
        signature=final["signature"],
        passes_completed=len(history),
        settings=settings.to_dict(),
        history=history,
        rejected_source_cells=final["rejected_source_cells"],
        restored_source_cells=final["restored_source_cells"],
        coverage_excluded_source_cells=final.get("coverage_excluded_source_cells", 0),
    )
    write(cache / ("result_" + digest(settings.to_dict()) + ".json"), result)
    return result
