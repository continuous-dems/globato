"""Read-only overlap and elevation disagreement among source FusionStates.

Reports evidence availability and z disagreements. Does not alter Globato
MultiStack weighting, source masking, or scientific source priority.
"""

from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
import math
import numpy as np
import rasterio


def measure_overlaps(paths):
    paths = [str(Path(p).resolve(strict=True)) for p in paths]
    if len(set(paths)) != len(paths):
        raise ValueError(
            "Duplicate original source FusionState: would double-weight authority evidence"
        )
    results = []
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(p)) for p in paths]
        if not sources:
            return results
        first = sources[0]
        for src in sources:
            if (
                src.crs != first.crs
                or src.transform != first.transform
                or src.shape != first.shape
            ):
                raise ValueError(
                    "Contributors in one group must share the same evidence grid alignment"
                )
            names = tuple(src.descriptions)
            if (
                src.tags().get("GLOBATO_DATATYPE") != "FUSION_STATE"
                or names.count("z_weighted_sum") != 1
                or names.count("weight_sum") != 1
                or names.count("count") != 1
            ):
                raise ValueError(
                    "Unsupported source-specific additive FusionState schema"
                )
        from itertools import combinations

        for i, j in combinations(range(len(sources)), 2):
            a, b = sources[i], sources[j]
            ni = {v: k + 1 for k, v in enumerate(a.descriptions)}
            nj = {v: k + 1 for k, v in enumerate(b.descriptions)}
            count = 0
            sq = 0.0
            sumabs = 0.0
            maxabs = 0.0
            gt1m = 0
            for _, win in a.block_windows(1):
                ca = a.read(ni["count"], window=win)
                cb = b.read(nj["count"], window=win)
                wa = a.read(ni["weight_sum"], window=win)
                wb = b.read(nj["weight_sum"], window=win)
                za = a.read(ni["z_weighted_sum"], window=win)
                zb = b.read(nj["z_weighted_sum"], window=win)
                valid = (
                    (ca > 0)
                    & (cb > 0)
                    & (wa > 0)
                    & (wb > 0)
                    & np.isfinite(wa)
                    & np.isfinite(wb)
                    & np.isfinite(za)
                    & np.isfinite(zb)
                )
                if not np.any(valid):
                    continue
                diff = za[valid] / wa[valid] - zb[valid] / wb[valid]
                n = int(diff.size)
                count += n
                sq += float(np.dot(diff, diff))
                sumabs += float(np.abs(diff).sum())
                maxabs = max(maxabs, float(np.abs(diff).max()))
                gt1m += int((np.abs(diff) > 1).sum())
            results.append(
                dict(
                    source_a=paths[i],
                    source_b=paths[j],
                    overlap_cells=count,
                    rmse_z_units=(math.sqrt(sq / count) if count else None),
                    mean_abs_diff_z_units=(sumabs / count if count else None),
                    max_abs_diff_z_units=(maxabs if count else None),
                    cells_abs_diff_gt_1_z_unit=gt1m,
                    note="Observed cell-mean disagreement in source vertical units; "
                    "NOT calibrated uncertainty or a priority rule",
                )
            )
    return results
