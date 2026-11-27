"""One native collection barrier and one source-resolved QC loop across modules."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from fetchez.core import run_fetchez
from fetchez.spatial import Region

from globato.hooks.filters.stack_qc import StackQC, StackQCGrid
from globato.hooks.sinks.side_stack import SideStackGenerate
from globato.hooks.tools.stack_combine import StackCombine
from globato.qc.stack import io, source_windows
from globato.qc.stack.context import WorkingSurface
from test_stack_qc_hook import FixtureMB

CRS = "EPSG:32611"
INTERPOLATION = dict(
    steps=1,
    resolutions="3/9",
    weights=[0, 0],
    algos="raster_fill/raster_fill",
    barrier=None,
    inland_decay_dist=0,
    keep_steps=False,
)


class JointFixture(FixtureMB):
    def __init__(self, path, name, survey_id, **kwargs):
        self.name = name
        self.survey_id = survey_id
        super().__init__(path, **kwargs)

    def run(self):
        super().run()
        self.results[0]["survey_id"] = self.survey_id
        return self


def build_joint(
    root,
    method="geometry",
    *,
    fine=False,
    reverse=False,
    ref_weight=10,
    tag="cold",
    viz=False,
    drop_targets=False,
    spikes=True,
    prepare_only=False,
):
    region = Region(350000, 350192, 3749808, 3750000, srs=CRS)
    yy, xx = np.indices((64, 64))
    defs = [
        ("target_a", "A", -100.0, 1.0),
        ("target_b", "B", -100.25, 0.5),
        ("reference", "R1", -99.0, ref_weight),
        ("reference", "R2", -100.5, 5.0),
    ]
    modules = []
    for index, (name, group, z, weight) in enumerate(defs):
        path = root / f"{group}.xyz"
        if not path.exists():
            values = np.full((64, 64), z)
            if index == 0 and spikes:
                values[31, 31] += 100
            if index == 1 and spikes:
                values[35, 39] += 70
            np.savetxt(
                path,
                np.column_stack(
                    (
                        (350000 + xx * 3 + 1.5).ravel(),
                        (3750000 - yy * 3 - 1.5).ravel(),
                        values.ravel(),
                    )
                ),
            )
        mod = JointFixture(
            str(path),
            name,
            group,
            src_region=region,
            outdir=str(root / "sources"),
            weight=weight,
        )
        mod.run()
        modules.append(mod)
    if drop_targets:
        modules = [m for m in modules if m.name == "reference"]
    if reverse:
        modules.reverse()
    res = "9" if fine else "3"
    qc_res = "3" if fine else None
    config = dict(
        modules=["target_a", "target_b"],
        add_ref_modules=["reference"],
        res=res,
        qc_res=qc_res,
        crs=CRS,
    )
    qc = StackQC(
        **config,
        method=method,
        max_passes=5 if method == "geometry" else 2,
        stop_when_stable=False,
        terrain_restoration=2,
        consensus=True,
        stack_strategy="weighted_mean",
        stack_weight_threshold="0.1",
        interpolation=dict(INTERPOLATION, resolutions=f"{res}/{int(res) * 3}"),
        cache_dir=str(root / "qc"),
        workers=1,
        viz=viz,
    )
    output = root / tag
    output.mkdir(exist_ok=True)
    hooks = [
        StackQCGrid(**config),
        SideStackGenerate(res=res, crs=CRS),
        qc,
        StackCombine(
            res=res,
            crs=CRS,
            strategy="weighted_mean",
            weight_threshold="0.1",
            source_artifact="qc-side-stack",
            missing="error",
            output=str(output / "stack.tif"),
        ),
    ]
    if prepare_only:
        hooks = hooks[:2]
    entries = run_fetchez(modules, threads=2, global_hooks=hooks, ignore_failures=False)
    return entries, modules


@pytest.mark.parametrize("method", ["geometry", "connectivity"])
@pytest.mark.parametrize("fine", [False, True])
def test_joint_context_weights_origins_and_fixed_references(
    tmp_path, monkeypatch, method, fine
):
    monkeypatch.chdir(tmp_path)
    calls = []
    original = source_windows.reject

    def observe(pool, parent, dem, *args, **kwargs):
        assert {r["module"] for r in pool.records} == {"target_a", "target_b"}
        assert len(pool.reference_records) == 2
        assert set(parent["masks"]) == {r["source_id"] for r in pool.records}
        calls.append(parent["signature"])
        return original(pool, parent, dem, *args, **kwargs)

    monkeypatch.setattr(source_windows, "reject", observe)
    baseline = []
    original_interp = WorkingSurface._interpolate

    def interpolate(self, path, work):
        if self.previous is None:
            with rasterio.open(path) as ds:
                baseline.append(float(ds.read(1)[8, 8]))
        return original_interp(self, path, work)

    monkeypatch.setattr(WorkingSurface, "_interpolate", interpolate)
    entries, modules = build_joint(tmp_path, method, fine=fine)
    assert len(baseline) == 1  # One context, not one baseline per target module.
    expected = (-100 - 100.25 * 0.5 - 99 * 10 - 100.5 * 5) / 16.5
    assert baseline[0] == pytest.approx(expected, abs=1e-5)
    assert 1 <= len(calls) <= (5 if method == "geometry" else 2)
    roots = list((tmp_path / "qc").glob("*/SOURCES.json"))
    assert len(roots) == 1
    inv = io.read(roots[0])
    assert len(inv["sources"]) == 2 and len(inv["reference_sources"]) == 2
    result = io.read(next(roots[0].parent.glob("result_*.json")))
    assert result["passes_completed"] == (5 if method == "geometry" else 2)
    assert result["rejected_source_cells"] >= 1
    origins = io.read(next(roots[0].parent.glob("contributions_*.json")))["sources"]
    assert {r["role"] for r in origins} == {"filter", "reference"}
    assert len({r["module_id"] for r in origins}) == 4
    (assembly_profile,) = (tmp_path / "cold").glob("*.profile.json")
    order = io.read(assembly_profile)["inputs"]["sources"]
    identities = {str(Path(r["output"]).resolve()): r["source_id"] for r in origins}
    assert [identities[path] for path in order] == sorted(identities.values())
    for owner, entry in entries:
        assert owner in modules
        assert owner.name == next(
            r["module"]
            for r in origins
            if r["output"] == entry["artifacts"]["qc-side-stack"]
        )
        with rasterio.open(entry["artifacts"]["qc-side-stack"]) as ds:
            assert ds.read(2).sum() > 0
            if owner.name == "reference":
                with rasterio.open(entry["artifacts"]["stack-qc-grid"]) as original:
                    np.testing.assert_array_equal(
                        ds.read().sum(axis=(1, 2)), original.read().sum(axis=(1, 2))
                    )
        if owner.name == "reference":
            assert "stack_qc_mask" not in entry["artifacts"]
    masks = io.read(result["manifest"])["masks"]
    assert set(masks) == {r["source_id"] for r in inv["sources"]}
    # Original reference input statistics are never rewritten by the pass loop.
    for ref in inv["reference_sources"]:
        assert io.sha(ref["path"]) == ref["source_sha256"]


def test_warm_joint_run_reuses_filter_and_order_is_independent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    build_joint(tmp_path)
    (receipt,) = list((tmp_path / "qc").glob("*/profile_*.json"))
    key = receipt.parent.name
    original_result = next(receipt.parent.glob("result_*.json")).read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("warm run refiltered")

    monkeypatch.setattr(source_windows, "reject", forbidden)
    build_joint(tmp_path, reverse=True, tag="warm")
    assert io.read(receipt)["cache"]["filter_hit"]
    assert next(receipt.parent.glob("result_*.json")).read_bytes() == original_result
    assert len(list((tmp_path / "qc").glob("*/profile_*.json"))) == 1
    assert receipt.parent.name == key
    a = next((tmp_path / "cold").glob("stack_*.tif"))
    b = next((tmp_path / "warm").glob("stack_*.tif"))
    with rasterio.open(a) as aa, rasterio.open(b) as bb:
        np.testing.assert_array_equal(aa.read(), bb.read())


def test_recipe_style_selectors_distinguish_two_invocations(tmp_path):
    from globato.hooks.filters.stack_qc import _selected

    a = JointFixture(str(tmp_path / "a.xyz"), "reference", "one")
    b = JointFixture(str(tmp_path / "b.xyz"), "reference", "two")
    selectors = [dict(module="reference", args=dict(survey_id="one"))]
    assert _selected(a, selectors) and not _selected(b, selectors)


def test_overlap_and_different_regions_fail_before_filtering(tmp_path):
    from types import SimpleNamespace

    region = Region(0, 1, 0, 1, srs="EPSG:4326")
    mod = SimpleNamespace(name="a", region=region)
    qc = StackQC(modules=["a"], add_ref_modules=["a"])
    with pytest.raises(ValueError, match="both"):
        qc.run([(mod, {})])
    other = SimpleNamespace(name="b", region=Region(1, 2, 0, 1, srs="EPSG:4326"))
    qc = StackQC(modules=["a", "b"])
    with pytest.raises(ValueError, match="same region"):
        qc.run([(mod, {}), (other, {})])


def test_reference_weight_change_uses_current_stream_and_invalidates_result(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    build_joint(tmp_path)
    (old_result,) = list((tmp_path / "qc").glob("*/result_*.json"))
    build_joint(tmp_path, ref_weight=2.0, tag="different_weight")
    results = list((tmp_path / "qc").glob("*/result_*.json"))
    assert len(results) == 2
    new_result = next(p for p in results if p != old_result)
    inv = io.read(new_result.parent / "SOURCES.json")
    ref = next(r for r in inv["reference_sources"] if r["group"] == "R1")
    assert ref["weight"] == 2
    with rasterio.open(ref["path"]) as ds:
        count = ds.read(2)
        weight = ds.read(3)
        np.testing.assert_array_equal(weight[count > 0] / count[count > 0], 2.0)
    result = io.read(new_result)
    assert {r["source_id"] for r in inv["sources"]} == set(
        io.read(result["manifest"])["masks"]
    )


def test_upstream_same_size_source_edit_is_not_hidden_by_joint_grid_cache(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    build_joint(tmp_path)
    path = tmp_path / "R1.xyz"
    original = path.read_text()
    changed = original.replace("-9.900000000000000000e+01", "-9.800000000000000000e+01")
    assert len(changed) == len(original) and changed != original
    path.write_text(changed)
    build_joint(tmp_path, tag="changed_source")
    assert len(list((tmp_path / "qc").glob("*/result_*.json"))) == 2


def test_refs_do_not_create_extra_independent_votes_for_same_acquisition(tmp_path):
    from globato.qc.stack.sources import SourcePool

    records = [dict(source_id="a", group="shared"), dict(source_id="b", group="other")]
    refs = [dict(source_id="ref", group="shared")]
    pool = SourcePool(
        dict(grid=dict(shape=[2, 2]), sources=records, reference_sources=refs)
    )
    try:
        assert pool.groups == ["other", "shared"]
        assert pool.records == records
        assert len(pool.by_group["shared"]) == 2
        assert not pool.rejection_mask(
            refs[0], (slice(0, 2), slice(0, 2)), dict(masks={})
        ).any()
    finally:
        pool.close()


def test_reference_only_coverage_survives_when_targets_have_no_entries(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    entries, modules = build_joint(tmp_path, drop_targets=True, fine=True)
    assert len(entries) == 2
    for mod, entry in entries:
        assert mod.name == "reference"
        assert "stack_qc_mask" not in entry["artifacts"]
        with rasterio.open(entry["artifacts"]["qc-side-stack"]) as ds:
            assert ds.read(2).sum() > 0
    (inventory,) = (tmp_path / "qc").glob("*/SOURCES.json")
    assert io.read(inventory)["sources"] == []


def test_numeric_weight_selector_matches_normalized_module_weight(tmp_path):
    from globato.hooks.filters.stack_qc import _selected

    mod = JointFixture(str(tmp_path / "a.xyz"), "reference", "one", weight=100)
    assert _selected(mod, [dict(module="reference", args=dict(weight=100))])


def test_same_file_different_processing_keeps_distinct_contributions(
    tmp_path, monkeypatch
):
    from fetchez.hooks import FetchHook

    class Shift(FetchHook):
        name = "test_shift"
        meta_stage = "stream"

        def __init__(self, shift):
            super().__init__()
            self.shift = shift

        def run(self, entries):
            for _, entry in entries:
                upstream = entry["stream"]

                def shifted(stream, amount):
                    for chunk in stream:
                        chunk = chunk.copy()
                        chunk["z"] += amount
                        yield chunk

                entry["stream"] = shifted(upstream, self.shift)
            return entries

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "same.xyz"
    y, x = np.indices((32, 32))
    np.savetxt(
        path,
        np.column_stack(
            (
                (350000 + x * 3 + 1.5).ravel(),
                (3750000 - y * 3 - 1.5).ravel(),
                np.full(x.size, -100.0),
            )
        ),
    )
    region = Region(350000, 350096, 3749904, 3750000, srs=CRS)
    mods = []
    for offset in [0, 1]:
        mod = JointFixture(
            str(path),
            "same",
            "acquisition",
            src_region=region,
            outdir=str(tmp_path / "sources"),
            hook=[Shift(offset)],
        )
        mod.run()
        mods.append(mod)
    assert (
        mods[0].module_id() == mods[1].module_id()
    )  # Fetchez excludes execution hooks from module identity.
    cfg = dict(modules=["same", "unused"], res="3", crs=CRS)
    qc = StackQC(
        **cfg,
        method="geometry",
        max_passes=1,
        stop_when_stable=False,
        terrain_restoration=0,
        interpolation=INTERPOLATION,
        stack_strategy="weighted_mean",
        stack_weight_threshold="0.1",
        cache_dir=str(tmp_path / "qc"),
        workers=1,
    )
    run_fetchez(
        mods, threads=2, global_hooks=[StackQCGrid(**cfg), qc], ignore_failures=False
    )
    (manifest,) = list((tmp_path / "qc").glob("*/SOURCES.json"))
    records = io.read(manifest)["sources"]
    assert len(records) == 2
    assert len({r["source_id"] for r in records}) == 2
    assert len({r["path"] for r in records}) == 2
    assert {r["group"] for r in records} == {"acquisition"}
    means = []
    for rec in records:
        with rasterio.open(rec["path"]) as ds:
            z = ds.read(1)
            weights = ds.read(3)
            means.append(float((z[weights > 0] / weights[weights > 0]).mean()))
    assert sorted(means) == [-100.0, -99.0]


def test_joint_scope_survives_spatially_absent_target(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from globato.qc.stack import preparation

    mod = SimpleNamespace(name="a", region=Region(0, 1, 0, 1, srs="EPSG:4326"))
    called = []
    monkeypatch.setattr(
        preparation, "prepare", lambda *args, **kwargs: called.append(kwargs)
    )
    StackQC(modules=["a", "b"]).run([(mod, {})])
    assert len(called) == 1 and called[0]["joint"]
