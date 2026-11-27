"""Fixed-input source-mask regression against the released v15.0.1 rules.

Expected signatures were generated with the frozen release before refactoring.
The fixture crosses the 256-row processing boundary and includes overlapping
files, unknown identities, unique coverage, ridges/valleys and competing groups.
"""

from pathlib import Path
import json
import numpy as np
import pytest
import rasterio
from affine import Affine
from globato.qc.stack import source_windows, recovery
from globato.qc.stack.sources import SourcePool, MaskWriter, zero
from globato.qc.stack.io import grid, sha, fingerprint


def put(path, data):
    if data.ndim == 2:
        data = data[None]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=len(data),
        dtype="float32",
        nodata=np.nan,
        crs="EPSG:32611",
        transform=Affine(3, 0, 350000, 0, -3, 3750000),
    ) as ds:
        ds.write(data.astype("float32"))


def fixture(case, out):
    folder = out / case
    folder.mkdir(exist_ok=True)
    rows, cols = np.indices((270, 104))
    z = (
        -200 + (-1 if case == "ridge" else 1) * 0.65 * abs(cols - 52) + 0.04 * rows
    ).astype("float32")
    z[30, 20] += 55
    z[210:215, 80:84] -= 25
    put(folder / "baseline.tif", z)
    records = []
    groups = [None] if case == "unknown" else ["a", "b", "c", "bad", "a", None]
    for index, group in enumerate(groups):
        values = z.copy()
        if group == "bad":
            values[50:220, :45] += 25
        if case == "competing" and group in ("c", "bad"):
            values += 20
        if group in ("a", "b", "c"):
            values[:, -12:] = np.nan
        if index == 4:
            values += 0.2
        values[:3, :4] = np.nan
        bands = np.ones((7, *z.shape), "float32")
        bands[0] = values
        bands[3] = values**2
        bands[:, ~np.isfinite(values)] = 0
        path = folder / f"source{index}.tif"
        put(path, bands)
        with rasterio.open(path) as ds:
            info = grid(ds)
        records.append(
            dict(
                source_id=f"s{index}",
                group=group,
                path=str(path),
                source_sha256=sha(path),
                fingerprint=fingerprint(path),
                grid=info,
                offset=[0, 0],
            )
        )
    with rasterio.open(folder / "baseline.tif") as ds:
        info = grid(ds)
    pool = SourcePool(dict(sources=records, grid=info))
    writer = MaskWriter(folder / "parent", pool)
    counts = {}
    for rec in records:
        removed = (abs(cols - 52) <= 2) & (rows >= 20) & (rows < 260)
        if rec["group"] == "bad":
            removed |= (cols < 45) & (rows >= 50) & (rows < 220)
        valid = pool.source(rec, (slice(0, 270), slice(0, 104)))[1]
        removed &= valid
        # Exact writer block layout is the production layout.
        for y in range(0, 270, 256):
            core = (slice(y, min(y + 256, 270)), slice(0, 104))
            writer.put(rec, core, removed[core])
        counts[rec["source_id"]] = dict(
            observed=int(valid.sum()), rejected=int(removed.sum())
        )
    _, parent = writer.finish(dict(source_counts=counts))
    quality = {g: dict(poor=g == "bad", occupied_blocks=12) for g in pool.groups}
    return pool, folder / "baseline.tif", parent, quality


@pytest.mark.parametrize("case", ["valley", "ridge", "unknown", "competing"])
def test_refactor_preserves_released_source_masks(tmp_path, case):
    golden = json.loads(
        (Path(__file__).parent / "data/stack_qc_v15_0_1_masks.json").read_text()
    )
    expected = {item["case"]: item for item in golden["cases"]}
    pool, baseline, parent, quality = fixture(case, tmp_path)

    def check(label, manifest):
        actual = dict(
            case=label,
            signature=manifest["signature"],
            rejected=manifest["rejected_source_cells"],
            restored=manifest.get("restored_source_cells"),
            excluded=manifest.get("coverage_excluded_source_cells"),
        )
        assert actual == expected[label]

    try:
        for mode in ("off", "near-complete", "redundant"):
            branches = recovery.restore(
                pool,
                parent,
                baseline,
                tmp_path / f"all_{mode}",
                [0, 1, 2, 3],
                quality,
                mode,
            )
            for level in range(4):
                check(f"{case}/{mode}/R{level}", branches[level][1])
            single = recovery.restore(
                pool, parent, baseline, tmp_path / f"single_{mode}", [2], quality, mode
            )
            check(f"{case}/{mode}/single_R2", single[2][1])
        if case == "valley":
            _, zero_parent = zero(pool, tmp_path / "zero")
            for name, previous in [("zero", zero_parent), ("removed", parent)]:
                for level in (1, 2, 3):
                    for consensus in (False, True):
                        folder = tmp_path / f"{name}_L{level}_C{consensus}"
                        _, manifest = source_windows.reject(
                            pool, previous, baseline, folder, level, consensus
                        )
                        check(f"reject/{name}/L{level}/C{consensus}", manifest)
    finally:
        pool.close()
