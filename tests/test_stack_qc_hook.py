"""stack_qc as a standard collection hook in the ordinary fetchez pipeline.

Side-stacks are made by the normal stream stage, stack_qc replaces the
module's side-stacks with filtered ones, and stack_combine assembles the stack.
No external data service is used.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from fetchez.core import run_fetchez
from fetchez.hooks.focus import FocusSink
from fetchez.modules import FetchModule
from fetchez.spatial import Region

from globato.hooks.filters.stack_qc import StackQC
from globato.hooks.filters.stack_qc_review import StackQCReview
from globato.hooks.rasters.binary_cudem import BinaryCudemStepDown
from globato.hooks.sinks.side_stack import SideStackCheck, SideStackGenerate
from globato.hooks.tools.stack_combine import StackCombine
from globato.qc.stack import geometry, io
from globato.qc.stack.settings import Settings

INTERPOLATION = dict(
    steps=1,
    resolutions="3/9",
    weights=[0.5],
    algos="raster_fill/raster_fill",
    barrier=None,
    inland_decay_dist=0,
)


class FixtureMB(FetchModule):
    name = "mbdb"

    def __init__(self, path, **kw):
        self.path = path
        kw.setdefault("use_cache", False)
        super().__init__(**kw)

    def run(self):
        self.results = [
            dict(
                url=Path(self.path).as_uri(),
                dst_fn=self.path,
                data_type="xyz",
                src_srs="EPSG:32611",
                survey_id="synthetic-acquisition",
            )
        ]
        return self

    def fetch_entry(self, entry, **kw):
        assert Path(entry["dst_fn"]).is_file()
        return 0


def points(tmp_path):
    y, x = np.indices((64, 64))
    xx = 350000.0 + x * 3 + 1.5
    yy = 3750000.0 - y * 3 - 1.5
    z = -100.0 + x * 0.01 + y * 0.01
    z[32, 32] += 100.0  # one spike
    path = tmp_path / "points.xyz"
    if not path.exists():
        np.savetxt(path, np.column_stack((xx.ravel(), yy.ravel(), z.ravel())))
    return path


def build(tmp_path, *, qc_args=None, weight=1.0, final_weights=(0.5,), tag="a"):
    region = Region(350000.0, 350192.0, 3749808.0, 3750000.0, srs="EPSG:32611")
    args = dict(
        res="3",
        crs="EPSG:32611",
        stack_strategy="mean",
        stack_weight_threshold="1",
        max_passes=1,
        stop_when_stable=False,
        terrain_restoration=0,
        interpolation=INTERPOLATION,
        cache_dir=str(tmp_path / "qc"),
        threads=1,
        workers=1,
    )
    qc = StackQC(**dict(args, **(qc_args or {})))
    mod = FixtureMB(
        str(points(tmp_path)),
        src_region=region,
        outdir=str(tmp_path / "source_cache"),
        weight=weight,
        hook=[qc],
    )
    mod.run()
    out = tmp_path / tag
    out.mkdir(exist_ok=True)
    hooks = [
        SideStackCheck(res="3", crs="EPSG:32611"),
        SideStackGenerate(res="3", crs="EPSG:32611"),
        StackCombine(
            res="3",
            crs="EPSG:32611",
            strategy="mean",
            output=str(out / "stack.tif"),
        ),
        FocusSink(target="multi_stack"),
        BinaryCudemStepDown(
            **dict(INTERPOLATION, weights=list(final_weights)),
            output=str(out / "dem.tif"),
        ),
    ]
    return run_fetchez([mod], threads=1, global_hooks=hooks, ignore_failures=False)


def results(tmp_path):
    return sorted((tmp_path / "qc").glob("*/result_*.json"))


def test_hook_filters_spike_and_feeds_stack_combine(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = points(tmp_path).read_bytes()
    out = build(tmp_path)
    dem = Path(out[0][1]["dst_fn"])
    with rasterio.open(dem) as ds:
        assert ds.count == 1
        assert float(ds.read(1, masked=True).max()) < -90.0  # spike removed
    (report,) = results(tmp_path)
    report = io.read(report)
    assert report["rejected_source_cells"] >= 1
    assert points(tmp_path).read_bytes() == original
    # The entry's side-stack is the filtered one; the original is recorded.
    accepted = list((tmp_path / "qc").glob("*/accepted_*/*.tif"))
    assert len(accepted) == 1
    assert out[0][1]["artifacts"]["side-stack"] == str(accepted[0])
    original_stack = out[0][1]["artifacts"]["stack_qc_original_side_stack"]
    assert Path(original_stack).is_file() and original_stack != str(accepted[0])


def test_cached_result_is_reused_and_unrelated_changes_do_not_refilter(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    build(tmp_path, tag="a")
    (first,) = results(tmp_path)
    stamp = first.stat().st_mtime_ns
    # Same filter, different final DEM tiers: no new filter run.
    build(tmp_path, final_weights=(0.25,), tag="b")
    assert results(tmp_path) == [first]
    assert first.stat().st_mtime_ns == stamp
    assert len(list((tmp_path / "qc").glob("*/baseline/COMPLETE.json"))) == 1


@pytest.mark.parametrize(
    "change",
    [
        dict(qc_args=dict(threshold_scale=2.0)),
        dict(qc_args=dict(median_windows="3/5/9")),
        dict(weight=0.5),
    ],
)
def test_filter_settings_or_module_weight_refilter(tmp_path, monkeypatch, change):
    monkeypatch.chdir(tmp_path)
    build(tmp_path, tag="a")
    build(tmp_path, tag="b", **change)
    assert len(list((tmp_path / "qc").glob("*/baseline/COMPLETE.json"))) == 2


def test_more_passes_reuse_earlier_passes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    build(tmp_path, tag="a")
    build(tmp_path, tag="b", qc_args=dict(max_passes=2))
    # Same key (passes are not part of it), a second result file, one baseline.
    assert len(results(tmp_path)) == 2
    assert len(list((tmp_path / "qc").glob("*/baseline/COMPLETE.json"))) == 1


def test_hook_refuses_a_multi_stack_pipeline(tmp_path):
    region = Region(350000.0, 350192.0, 3749808.0, 3750000.0, srs="EPSG:32611")
    qc = StackQC(res="3", crs="EPSG:32611", interpolation=INTERPOLATION)
    mod = FixtureMB(str(points(tmp_path)), src_region=region, outdir=str(tmp_path))
    entry = dict(artifacts=dict(multi_stack="stack.tif"))
    with pytest.raises(ValueError, match="stack_combine"):
        qc.run([(mod, entry)])


def test_global_hook_filters_only_named_modules(tmp_path):
    qc = StackQC(
        res="3", crs="EPSG:32611", interpolation=INTERPOLATION, modules="other"
    )
    region = Region(350000.0, 350192.0, 3749808.0, 3750000.0, srs="EPSG:32611")
    mod = FixtureMB(str(points(tmp_path)), src_region=region, outdir=str(tmp_path))
    entries = [(mod, dict(artifacts=dict(multi_stack="stack.tif")))]
    assert qc.run(entries) is entries  # mbdb is not named, so nothing runs


def test_qc_res_scales_the_working_ladder_and_must_nest():
    qc = StackQC(res="1s", qc_res="1/9s")
    fine = 1 / 3600 / 9
    assert qc.qc_increment == pytest.approx(fine)
    assert qc.interpolation_args()["resolutions"][0] == pytest.approx(fine)
    assert qc.interpolation_args()["resolutions"][1] == pytest.approx(3 / 3600 / 9)
    with pytest.raises(ValueError, match="nest"):
        StackQC(res="1s", qc_res="0.4s")
    assert StackQC(res="1s").qc_increment is None


def test_tuning_arguments_reach_settings():
    qc = StackQC(
        threshold_scale=2,
        median_windows="3/3/5",
        flank_distances="2/3/4",
        plane_tolerance=3.5,
        restoration_length_scale=9,
        restoration_fit_scale=2,
    )
    s = qc.settings
    assert s.median_windows == (3, 3, 5) and s.flank_distances == (2, 3, 4)
    assert s.detection_levels()[2]["local9"] == geometry.LEVELS[2]["local9"] * 2
    assert s.plane_tolerance_m == 3.5
    with pytest.raises(ValueError):
        StackQC(median_windows="3/4/5")


def test_default_settings_keep_the_original_constants():
    s = Settings()
    assert s.detection_levels() is geometry.LEVELS
    assert geometry.halo(s.median_windows, s.flank_distances) == 13


def test_parallel_geometry_matches_serial(tmp_path):
    from globato.qc.stack import source_windows

    rng = np.random.default_rng(3)
    z = np.cumsum(rng.normal(0, 1, (700, 650)), axis=0).astype("float32")
    z[rng.random(z.shape) < 0.01] += 50
    z[:20, :20] = np.nan
    path = tmp_path / "dem.tif"
    profile = dict(
        driver="GTiff",
        width=z.shape[1],
        height=z.shape[0],
        count=1,
        dtype="float32",
        nodata=np.nan,
        crs="EPSG:32611",
        transform=rasterio.transform.from_origin(0, 0, 3, 3),
    )
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(z, 1)
    serial = source_windows.geometry(path, tmp_path / "s", [2], workers=1)
    parallel = source_windows.geometry(path, tmp_path / "p", [2], workers=3)
    for level in serial:
        np.testing.assert_array_equal(
            np.asarray(serial[level]), np.asarray(parallel[level])
        )


def test_review_writes_page_after_final_dem(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.chdir(tmp_path)
    out = build(tmp_path)
    StackQCReview(dem=str(tmp_path / "a" / "dem.tif")).run(out)
    pages = list((tmp_path / "a" / "stack_qc_review").glob("*/review.html"))
    assert len(pages) == 1
    assert not list((tmp_path / "a" / "stack_qc_review").glob("*/review_ERROR.txt"))


def test_review_without_results_does_nothing(tmp_path):
    pytest.importorskip("matplotlib")
    entries = [(None, dict(dst_fn=str(tmp_path / "x.tif")))]
    assert StackQCReview().run(entries) is entries


def test_finer_qc_grid_filters_then_delivers_on_the_dem_grid(tmp_path, monkeypatch):
    """qc_res finer than res: the hook grids the sources again on the QC grid."""
    monkeypatch.chdir(tmp_path)
    region = Region(350000.0, 350189.0, 3749811.0, 3750000.0, srs="EPSG:32611")
    ladder = dict(INTERPOLATION, resolutions="9/27")
    qc = StackQC(
        res="9",
        qc_res="3",
        crs="EPSG:32611",
        stack_strategy="mean",
        stack_weight_threshold="1",
        max_passes=1,
        stop_when_stable=False,
        terrain_restoration=0,
        interpolation=ladder,
        cache_dir=str(tmp_path / "qc"),
        threads=1,
        workers=1,
    )
    assert qc.interpolation_args()["resolutions"] == [3.0, 9.0]
    mod = FixtureMB(
        str(points(tmp_path)),
        src_region=region,
        outdir=str(tmp_path / "source_cache"),
        hook=[qc],
    )
    mod.run()
    hooks = [
        SideStackCheck(res="9", crs="EPSG:32611"),
        SideStackGenerate(res="9", crs="EPSG:32611"),
        StackCombine(
            res="9",
            crs="EPSG:32611",
            strategy="mean",
            output=str(tmp_path / "stack.tif"),
        ),
        FocusSink(target="multi_stack"),
        BinaryCudemStepDown(**ladder, output=str(tmp_path / "dem.tif")),
    ]
    out = run_fetchez([mod], threads=1, global_hooks=hooks, ignore_failures=False)
    (report,) = results(tmp_path)
    report = io.read(report)
    assert report["rejected_source_cells"] >= 1
    sources = io.read(Path(report["manifest"]).parents[0] / "MANIFEST.json")["masks"]
    (item,) = sources.values()
    with rasterio.open(item["mask"]) as mask:
        assert mask.res == (3.0, 3.0)  # filtered on the QC grid
    with rasterio.open(out[0][1]["artifacts"]["side-stack"]) as accepted:
        assert accepted.res == (9.0, 9.0)  # delivered on the DEM grid
        assert accepted.tags()["GLOBATO_STACK_QC_SIGNATURE"] == report["signature"]
    with rasterio.open(out[0][1]["dst_fn"]) as dem:
        assert dem.res == (9.0, 9.0)
        assert float(dem.read(1, masked=True).max()) < -90.0


def test_regridding_restores_the_source_crs_from_before_the_stream_stage():
    """stream_reproject rewrites src_srs; the QC grid must start from the original."""
    import json

    from globato.qc.stack.preparation import _fresh

    identity = dict(
        processing=[
            dict(name="stream-reproject", config={}),
            dict(
                name="__fusion_context__",
                config=dict(source_crs="EPSG:4326+vdatum:msl"),
            ),
        ]
    )
    collected = dict(
        url="file:///x.fbt",
        src_srs="EPSG:4326+3855",  # after stream_reproject
        data_type="fusion-state",
        stream=iter(()),
        artifacts={"side-stack": "x.tif"},
        _sidestack_identity=json.dumps(identity),
    )
    fresh = _fresh(collected)
    assert fresh["src_srs"] == "EPSG:4326+vdatum:msl"
    assert "stream" not in fresh and "artifacts" not in fresh
    assert "data_type" not in fresh and fresh["_stack_qc_restored_srs"] is True
    assert (
        _fresh(dict(url="file:///y.fbt", src_srs="A"))["_stack_qc_restored_srs"]
        is False
    )
