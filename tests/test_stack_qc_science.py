"""Array science and real source-mask regressions, independent of fetching."""

import numpy as np
import pytest
import rasterio
from affine import Affine
from globato.qc.stack import (
    geometry,
    restoration as tr,
    surveys,
    source_windows as sw,
    recovery,
    survey_coverage,
)
from globato.qc.stack.sources import SourcePool, MaskWriter, zero as zero_masks
from globato.qc.stack.settings import Settings, restoration_policy
from globato.qc.stack.io import sha, fingerprint, grid, read
from globato.qc.stack.native import masked_stream
from globato.qc.stack.grid import grid_offset


def put(path, z, tf):
    if z.ndim == 2:
        z = z[None]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=z.shape[2],
        height=z.shape[1],
        count=len(z),
        dtype="float32",
        nodata=np.nan,
        crs="EPSG:32611",
        transform=tf,
    ) as ds:
        ds.write(z.astype("float32"))


def pool(tmp_path, z, groups=1):
    tf = Affine(3, 0, 350000, 0, -3, 3750000)
    base = tmp_path / "base.tif"
    put(base, z, tf)
    records = []
    for i in range(groups):
        b = np.ones((7, *z.shape), "float32")
        b[0] = z
        b[3] = z * z
        p = tmp_path / f"s{i}.tif"
        put(p, b, tf)
        with rasterio.open(p) as ds:
            g = grid(ds)
        records.append(
            dict(
                source_id=f"s{i}",
                group=f"g{i}",
                path=str(p),
                grid=g,
                offset=[0, 0],
                source_sha256=sha(p),
                fingerprint=fingerprint(p),
            )
        )
    with rasterio.open(base) as ds:
        g = grid(ds)
    return SourcePool(dict(sources=records, grid=g)), base


def test_tiled_geometry_matches_whole_surface_at_seams_and_nodata(tmp_path):
    y, x = np.indices((535, 529))
    z = (-200.0 + 0.03 * x + 0.02 * y).astype("float32")
    z[510:516, 510:516] += 60.0
    z[:40, :50] = np.nan
    p, b = pool(tmp_path, z)
    expected = geometry.evidence(z)["masks"]
    got = sw.geometry(b, tmp_path / "geometry", [1, 2, 3])
    for k, a in got.items():
        np.testing.assert_array_equal(a, expected[k])
    p.close()


def test_actual_native_source_masks_are_nested_at_three_public_strengths(tmp_path):
    z = np.full((96, 96), -100.0, "float32")
    z[30, 30] += 100.0
    z[50, 50] += 4.0
    p, b = pool(tmp_path, z)
    zp, zero = zero_masks(p, tmp_path / "zero")
    masks = []
    for k in (1, 2, 3):
        _, d = sw.reject(p, zero, b, tmp_path / f"level{k}", k)
        with rasterio.open(d["masks"]["s0"]["mask"]) as ds:
            masks.append(ds.read(1).astype(bool))
    assert all(not (a & ~b).any() for a, b in zip(masks, masks[1:]))
    assert masks[0][30, 30]
    p.close()


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_retained_flanks_recover_ridge_and_valley_without_deleted_values(sign):
    y, x = np.indices((160, 160))
    z = (-200.0 + sign * 0.6 * abs(x - 80) + 0.04 * y).astype("float32")
    removed = (abs(x - 80) <= 2) & (abs(y - 80) <= 50)
    prediction = tr.model(z[None], removed, (3.0, 3.0), policy=restoration_policy(2))[
        "prediction"
    ]
    poisoned = z.copy()
    poisoned[removed] = 1e7
    other = tr.model(
        poisoned[None],
        removed,
        (3.0, 3.0),
        policy=restoration_policy(2),
    )["prediction"]
    np.testing.assert_allclose(prediction, other, equal_nan=True)
    good = tr.accept_source(
        z, np.ones(z.shape, bool), removed, prediction, policy=restoration_policy(2)
    )
    assert good.sum() > 0.7 * removed.sum()
    assert not tr.accept_source(
        z + 30,
        np.ones(z.shape, bool),
        removed,
        prediction,
        policy=restoration_policy(2),
    ).any()


def test_public_restoration_masks_are_nested(tmp_path):
    y, x = np.indices((160, 160))
    z = (-200.0 + 0.6 * abs(x - 80) + 0.04 * y).astype("float32")
    removed = (abs(x - 80) <= 2) & (abs(y - 80) <= 50)
    p, b = pool(tmp_path, z)
    writer = MaskWriter(tmp_path / "removed", p)
    writer.put(p.records[0], (slice(0, 160), slice(0, 160)), removed)
    _, parent = writer.finish(
        dict(source_counts={"s0": dict(observed=z.size, rejected=int(removed.sum()))})
    )
    branches = recovery.restore(
        p,
        parent,
        b,
        tmp_path / "restored",
        [1, 2, 3],
        {"g0": dict(poor=False, occupied_blocks=25)},
    )
    masks = []
    for _, d in branches.values():
        with rasterio.open(d["masks"]["s0"]["mask"]) as ds:
            masks.append(ds.read(1).astype(bool))
        assert d["restored_source_cells"] > 0
    assert all(not (b & ~a).any() for a, b in zip(masks, masks[1:]))
    p.close()


def test_unknown_identity_can_restore_local_terrain_without_independent_votes(tmp_path):
    y, x = np.indices((160, 160))
    z = (-200.0 + 0.6 * abs(x - 80) + 0.04 * y).astype("float32")
    removed = (abs(x - 80) <= 2) & (abs(y - 80) <= 50)
    original_pool, baseline = pool(tmp_path, z)
    original_pool.records[0]["group"] = None
    p = SourcePool(original_pool.inv)
    writer = MaskWriter(tmp_path / "removed", p)
    writer.put(p.records[0], (slice(0, 160), slice(0, 160)), removed)
    _, parent = writer.finish(
        dict(source_counts={"s0": dict(observed=z.size, rejected=int(removed.sum()))})
    )
    s = (slice(0, 160), slice(0, 160))
    independent, unknown = p.values(s)
    assert independent.shape[0] == 0 and unknown.all()
    local, _ = p.values(s, parent, local_context=True)
    assert local.shape[0] == 1 and not np.isfinite(local[0][removed]).any()
    restored = recovery.restore(p, parent, baseline, tmp_path / "restored", [2], {})[2][
        1
    ]
    assert restored["restored_source_cells"] == int(removed.sum())
    assert p.groups == []
    p.close()


def test_survey_votes_require_independence_and_reject_competing_groups():
    values = np.array([[-100.0], [-100.2], [-99.9], [-120.0]], "float32")
    keep, cat = surveys.qualified_votes(values, 3.5)
    assert cat[0] == 3 and keep[:3].all() and not keep[3].any()
    keep, cat = surveys.qualified_votes(values, 3.5, unknown=np.array([True]))
    assert not keep.any() and cat[0] == 5
    competing = np.array([[-100.0], [-100.2], [-120.0], [-120.2]], "float32")
    keep, cat = surveys.qualified_votes(competing, 3.5)
    assert not keep.any()
    keep, cat = surveys.qualified_votes(values[:2], 3.5)
    assert not keep.any()


def test_survey_evidence_does_not_explain_steep_terrain_as_broad_bad_survey():
    y, x = np.indices((20, 20))
    z = (-100.0 + 10 * x).astype("float32")
    v = np.stack([z, z, z - 30.0])
    result = surveys.evidence(v, np.zeros(z.shape, bool))
    assert not result["bad"].any()


def test_all_seven_original_bands_are_zeroed_only_at_rejected_cells(tmp_path):
    z = np.full((64, 64), -100.0, "float32")
    p, b = pool(tmp_path, z)
    rec = p.records[0]
    before = sha(rec["path"])
    mask = np.zeros(z.shape, bool)
    mask[20, 20] = True
    writer = MaskWriter(tmp_path / "masked", p)
    writer.put(rec, (slice(0, 64), slice(0, 64)), mask)
    _, d = writer.finish()
    stream = masked_stream(rec["path"], d["masks"]["s0"]["mask"])
    next(stream)
    data = np.zeros((7, *z.shape), "float32")
    for item in stream:
        win = item[0]
        r, c, h, w = map(int, (win.row_off, win.col_off, win.height, win.width))
        data[:, r : r + h, c : c + w] = item[2]
    assert np.all(data[:, 20, 20] == 0) and np.all(data[1:, 10, 10] != 0)
    stream.close()
    assert sha(rec["path"]) == before
    p.close()


def test_fractional_grid_offset_is_refused(tmp_path):
    z = np.full((32, 32), -100.0, "float32")
    p, b = pool(tmp_path, z)
    other = tmp_path / "shifted.tif"
    put(other, z, Affine(3, 0, 350001, 0, -3, 3750000))
    with rasterio.open(b) as a, rasterio.open(other) as c:
        with pytest.raises(ValueError, match="noninteger"):
            grid_offset(c, a)
    p.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"aggressiveness": 4},
        {"aggressiveness": True},
        {"max_passes": 0},
        {"max_passes": 11},
        {"terrain_restoration": 4},
        {"stable_dem_change_m": float("nan")},
    ],
)
def test_invalid_settings_are_refused(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_final_requested_pass_hands_mask_to_build_without_context_interpolation(
    tmp_path, monkeypatch
):
    """Force distinct masks: exact-state reuse cannot hide an extra final build."""
    from globato.qc.stack import processing

    z = np.full((64, 64), -100.0, "float32")
    p, baseline = pool(tmp_path, z)
    inventory = p.inv
    p.close()
    builds = []

    def build(manifest, output):
        builds.append(read(manifest)["rejected_source_cells"])
        return baseline

    def reject(
        pool,
        parent,
        parent_dem,
        folder,
        detector_level,
        consensus,
        settings=None,
        workers=1,
    ):
        with rasterio.open(parent["masks"]["s0"]["mask"]) as ds:
            previous = ds.read(1).astype(bool)
        mask = previous.copy()
        mask[20, 20 + int(previous.sum())] = True
        writer = MaskWriter(folder / "MASKS", pool)
        writer.put(pool.records[0], (slice(0, 64), slice(0, 64)), mask, previous)
        return writer.finish(
            dict(source_counts={"s0": dict(observed=z.size, rejected=int(mask.sum()))})
        )

    monkeypatch.setattr(sw, "reject", reject)

    def unnecessary(*args, **kwargs):
        raise AssertionError("Disabled recovery/exclusion must not scan surveys")

    monkeypatch.setattr(survey_coverage, "original_quality", unnecessary)
    monkeypatch.setattr(recovery, "restore", unnecessary)
    result = processing.process(
        inventory,
        Settings(max_passes=5, stop_when_stable=False, terrain_restoration=0),
        tmp_path / "loop",
        build,
    )
    assert builds == [0, 1, 2, 3, 4]
    assert result["rejected_source_cells"] == 5
    assert result["history"][-1]["dem"] is None
    assert result["history"][-1]["changes"] is None
    assert [h["context_interpolated"] for h in result["history"]] == [
        True,
        True,
        True,
        True,
        False,
    ]


def test_survey_exclusion_preserves_unique_coverage_and_requires_replacement(tmp_path):
    z = np.full((128, 128), -100.0, "float32")
    p, baseline = pool(tmp_path, z, groups=4)
    # Three good overlapping surveys end before the suspect survey's unique part.
    for rec in p.records[:3]:
        with rasterio.open(rec["path"], "r+") as ds:
            b = ds.read()
            b[:, :, 96:] = 0
            ds.write(b)
    for rec in p.records:
        rec["source_sha256"] = sha(rec["path"])
        rec["fingerprint"] = fingerprint(rec["path"])
    writer = MaskWriter(tmp_path / "parent", p)
    counts = {}
    for rec in p.records:
        mask = np.zeros(z.shape, bool)
        if rec["group"] == "g3":
            mask[:, :48] = True
        writer.put(rec, (slice(0, 128), slice(0, 128)), mask)
        counts[rec["source_id"]] = dict(
            observed=128 * (128 if rec["group"] == "g3" else 96),
            rejected=int(mask.sum()),
        )
    _, parent = writer.finish(dict(source_counts=counts))
    quality = {g: dict(poor=g == "g3", occupied_blocks=16) for g in p.groups}
    for mode in ("near-complete", "redundant"):
        branches = recovery.restore(
            p, parent, baseline, tmp_path / mode, [0], quality, mode
        )
        _, result = branches[0]
        with rasterio.open(result["masks"]["s3"]["mask"]) as ds:
            mask = ds.read(1).astype(bool)
        assert not mask[:, 96:].any()  # Unique observations survive both policies.
        if mode == "redundant":
            assert mask[:, :96].all()
        else:
            assert not mask[:, 48:96].any()  # Too much unique coverage for 95% gate.
    p.close()


def test_parallel_restoration_matches_serial(tmp_path):
    """Worker processes compute blocks; writing stays in block order."""
    y, x = np.indices((600, 600))
    z = (-200.0 + 0.6 * abs(x - 300) + 0.04 * y).astype("float32")
    removed = (abs(x - 300) <= 2) & (abs(y - 300) <= 250)
    p, b = pool(tmp_path, z)
    writer = MaskWriter(tmp_path / "removed", p)
    writer.put(p.records[0], (slice(0, 600), slice(0, 600)), removed)
    _, parent = writer.finish(
        dict(source_counts={"s0": dict(observed=z.size, rejected=int(removed.sum()))})
    )
    quality = {"g0": dict(poor=False, occupied_blocks=9)}
    serial = recovery.restore(p, parent, b, tmp_path / "serial", [2], quality)
    parallel = recovery.restore(
        p, parent, b, tmp_path / "parallel", [2], quality, workers=3
    )
    assert serial[2][1]["restored_source_cells"] > 0
    assert serial[2][1]["signature"] == parallel[2][1]["signature"]
    assert (
        serial[2][1]["restored_source_cells"] == parallel[2][1]["restored_source_cells"]
    )
    p.close()


def test_parallel_rejection_pass_matches_serial(tmp_path):
    """Worker processes evaluate blocks; masks are written in block order."""
    from globato.qc.stack import source_windows

    rng = np.random.default_rng(11)
    y, x = np.indices((600, 600))
    z = (-200.0 + 0.05 * x + 0.03 * y).astype("float32")
    spikes = rng.random(z.shape) < 0.002
    z[spikes] += 40
    p, b = pool(tmp_path, z, groups=2)
    _, parent = zero_masks(p, tmp_path / "zero")
    _, serial = source_windows.reject(p, parent, b, tmp_path / "serial", 2)
    _, parallel = source_windows.reject(
        p, parent, b, tmp_path / "parallel", 2, workers=3
    )
    assert serial["rejected_source_cells"] > 0
    assert serial["signature"] == parallel["signature"]
    assert serial["source_counts"] == parallel["source_counts"]
    p.close()


def test_parallel_survey_quality_matches_serial(tmp_path):
    from globato.qc.stack import survey_coverage

    rng = np.random.default_rng(5)
    y, x = np.indices((600, 600))
    z = (-200.0 + 0.05 * x + 0.03 * y + rng.normal(0, 0.5, (600, 600))).astype(
        "float32"
    )
    p, _ = pool(tmp_path, z, groups=3)
    serial = survey_coverage.original_quality(p, tmp_path / "serial.json")
    parallel = survey_coverage.original_quality(
        p, tmp_path / "parallel.json", workers=3
    )
    assert serial == parallel
    p.close()


def test_pool_workers_stop_when_the_build_is_killed(tmp_path):
    """A killed build must not leave orphaned Stack QC workers behind."""
    import os
    import signal
    import subprocess
    import sys
    import time

    script = tmp_path / "parent.py"
    script.write_text(
        "import os, sys, time\n"
        "from globato.qc.stack.parallel import process_pool\n"
        "def worker_pid(_):\n"
        "    return os.getpid()\n"
        "if __name__ == '__main__':\n"
        "    with process_pool(2) as pool:\n"
        "        pids = set(pool.map(worker_pid, range(8)))\n"
        "        print(' '.join(map(str, pids)), flush=True)\n"
        "        time.sleep(60)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, str(script)], stdout=subprocess.PIPE, text=True
    )
    pids = [int(v) for v in proc.stdout.readline().split()]
    assert pids
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    deadline = time.monotonic() + 20
    alive = pids
    while alive and time.monotonic() < deadline:
        time.sleep(0.2)
        alive = []
        for pid in pids:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            # A zombie still answers kill(0); check its state.
            try:
                state = open(f"/proc/{pid}/stat").read().split()[2]
            except FileNotFoundError:
                continue
            if state != "Z":
                alive.append(pid)
    assert not alive, f"orphaned workers: {alive}"
