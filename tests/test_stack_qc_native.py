"""Real Fetchez lifecycle and native Globato owners; no external data service."""

from pathlib import Path
import numpy as np
import pytest
import rasterio
from fetchez.modules import FetchModule
from fetchez.spatial import Region
from globato.qc.stack.pipeline import StackQCPlan
from fetchez.hooks.focus import FocusSink
from globato.hooks.filters.stack_qc import StackQC
from globato.hooks.sinks.multi_stack import MultiStackHook
from globato.hooks.rasters.binary_cudem import BinaryCudemStepDown
from globato.qc.stack.io import read


def run_plan(modules, *, threads=1, global_hooks=None, ignore_failures=False):
    # These fixtures have already discovered sources in setup(). Keep deliberate
    # empty selections/claim metadata; exercise the real adapter after discovery.
    from contextlib import ExitStack
    from unittest.mock import patch

    with ExitStack() as patches:
        for module in modules:
            patches.enter_context(patch.object(module, "run", lambda: None))
        return StackQCPlan(
            config={},
            modules=modules,
            threads=threads,
            global_hooks=global_hooks or [],
            ignore_failures=ignore_failures,
        ).execute()


run_fetchez = run_plan


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
        metadata = Path(self.path).with_suffix(".inf")
        metadata.write_text("ancillary metadata, not elevation samples")
        self.results.append(
            dict(url=metadata.as_uri(), dst_fn=str(metadata), data_type="mb_inf")
        )
        return self

    def fetch_entry(self, entry, **kw):
        assert Path(entry["dst_fn"]).is_file()
        return 0


def setup(tmp_path, passes=1, restore=0):
    y, x = np.indices((64, 64))
    xx = 350000.0 + x * 3 + 1.5
    yy = 3750000.0 - y * 3 - 1.5
    z = -100.0 + x * 0.01 + y * 0.01
    z[32, 32] += 100.0
    file = tmp_path / "points.xyz"
    if not file.exists():
        np.savetxt(file, np.column_stack((xx.ravel(), yy.ravel(), z.ravel())))
    region = Region(350000.0, 350192.0, 3749808.0, 3750000.0, srs="EPSG:32611")
    qc = StackQC(
        aggressiveness=2,
        terrain_restoration=restore,
        max_passes=passes,
        stop_when_stable=False,
    )
    # Tests use a stable scratch directory. This is not a public recipe argument.
    qc.cache_dir = str(tmp_path / "qc")
    mod = FixtureMB(
        str(file), src_region=region, outdir=str(tmp_path / "source_cache"), hook=[qc]
    )
    mod.run()
    stack = MultiStackHook(
        res="3",
        crs="EPSG:32611",
        output=str(tmp_path / "normal_stack.tif"),
        strategy="mean",
    )
    interp = BinaryCudemStepDown(
        steps=1,
        resolutions="3/9",
        weights=".5",
        algos="raster_fill/raster_fill",
        barrier=None,
        inland_decay_dist=0,
        output=str(tmp_path / "normal_dem.tif"),
    )
    return mod, qc, [stack, FocusSink(target="multi_stack"), interp]


def test_normal_module_hook_builds_native_dem_and_does_not_mutate_sources(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    mod, qc, hooks = setup(tmp_path)
    original = Path(mod.path).read_bytes()
    result = run_fetchez([mod], threads=1, global_hooks=hooks, ignore_failures=False)
    assert len(result) == 1 and Path(result[0][1]["dst_fn"]).is_file()
    with rasterio.open(result[0][1]["dst_fn"]) as ds:
        assert ds.count == 1 and ds.crs.to_epsg() == 32611
        assert float(ds.read(1, masked=True).max()) < -90.0
    reports = list((tmp_path / "qc").glob("*/result_*.json"))
    assert len(reports) == 1
    report = read(reports[0])
    assert report["passes_completed"] == 1 and report["rejected_source_cells"] >= 1
    assert Path(mod.path).read_bytes() == original
    masks = read(report["manifest"])["masks"]
    assert len(masks) == 1
    assert all(Path(v["mask"]).is_file() for v in masks.values())
    assert report["history"][-1]["context_interpolated"] is False
    assert len(list((tmp_path / "qc").glob("*/dems/*/dem.tif"))) == 0


def test_public_controls_are_singular_and_strict():
    qc = StackQC()
    assert qc.stage == "collection"
    assert qc.settings.aggressiveness == 2 and qc.settings.terrain_restoration == 2
    assert qc.settings.max_passes == 4 and qc.settings.stop_when_stable
    assert StackQC(aggressiveness=1).settings.aggressiveness == 1
    assert StackQC(aggressiveness=3).settings.aggressiveness == 3
    for invalid in (0, 4, True, "2", 2.0):
        with pytest.raises(ValueError, match="aggressiveness"):
            StackQC(aggressiveness=invalid)
    with pytest.raises(TypeError, match="renamed level to aggressiveness"):
        StackQC(level=2)
    with pytest.raises(
        TypeError, match="renamed restoration_level to terrain_restoration"
    ):
        StackQC(restoration_level=2)
    assert "level" not in StackQC._cli_arg_help
    assert "aggressiveness" in StackQC._cli_arg_help
    assert "restoration_level" not in StackQC._cli_arg_help
    assert "terrain_restoration" in StackQC._cli_arg_help
    assert "level" not in qc.settings.to_dict()
    assert "restoration_level" not in qc.settings.to_dict()
    with pytest.raises(TypeError):
        StackQC(levels=2)
    with pytest.raises(TypeError):
        StackQC(filter_levels=[2])
    with pytest.raises(ValueError):
        StackQC(max_passes=11)
    for obsolete in (
        dict(consensus=True),
        dict(survey_exclusion="redundant"),
        dict(stable_changed_fraction=0.1),
        dict(stable_dem_change_m=0.1),
        dict(stable_consecutive_passes=3),
        dict(cache_dir="/tmp/qc"),
        dict(stage="post"),
    ):
        with pytest.raises(TypeError, match="accepts only"):
            StackQC(**obsolete)
    assert set(StackQC._cli_arg_help) == {
        "aggressiveness",
        "terrain_restoration",
        "max_passes",
        "stop_when_stable",
        "review",
        "qc_res",
    }
    assert qc.settings.consensus is False
    assert qc.settings.survey_exclusion == "off"
    assert qc.settings.stable_changed_fraction == 0.00001
    assert qc.settings.stable_dem_change_m == 0.05
    assert qc.settings.stable_consecutive_passes == 2


def test_forced_passes_reuse_native_context_and_preserve_main_pipeline(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    mod, qc, hooks = setup(tmp_path, passes=5, restore=2)
    run_fetchez([mod], threads=1, global_hooks=hooks, ignore_failures=False)
    report = read(next((tmp_path / "qc").glob("*/result_*.json")))
    assert report["passes_completed"] == 5
    assert [x["pass_number"] for x in report["history"]] == [1, 2, 3, 4, 5]
    assert len({x["manifest"] for x in report["history"]}) < 5
    assert len(list((tmp_path / "qc").glob("*/dems/*/dem.tif"))) < 5
    # A fresh ordinary build can reuse the exact prefix without rerunning raw
    # transforms or native context interpolation. Its final DEM remains native.
    completed = {
        str(p): p.stat().st_mtime_ns for p in (tmp_path / "qc").rglob("dem.tif")
    }
    mod2, qc2, hooks2 = setup(tmp_path, passes=5, restore=2)
    result = run_fetchez([mod2], threads=1, global_hooks=hooks2, ignore_failures=False)
    assert Path(result[0][1]["dst_fn"]).is_file()
    assert len(list((tmp_path / "qc").glob("*/SOURCES.json"))) == 1
    assert all(Path(p).stat().st_mtime_ns == stamp for p, stamp in completed.items())


def test_normal_globato_cli_loads_singular_yaml_hook(tmp_path):
    import subprocess
    import sys
    import yaml

    mod, qc, hooks = setup(tmp_path)
    recipe = dict(
        project={"name": "native_yaml_qc"},
        region=mod.region.to_list(),
        region_srs="EPSG:32611",
        execution={"threads": 1},
        modules=[
            dict(
                module="local_fs",
                args=dict(path=mod.path, data_type="xyz"),
                hooks=[
                    dict(name="set_srs", args={"srs": "EPSG:32611"}),
                    dict(
                        name="set_entry",
                        args={"key": "survey_id", "value": "yaml-acquisition"},
                    ),
                    dict(
                        name="stack_qc",
                        args=dict(
                            aggressiveness=2,
                            terrain_restoration=0,
                            max_passes=1,
                            stop_when_stable=False,
                        ),
                    ),
                ],
            )
        ],
        global_hooks=[
            dict(
                name="multi_stack",
                args=dict(res="3", crs="EPSG:32611", strategy="mean"),
            ),
            dict(name="focus_sink", args={"target": "multi_stack"}),
            dict(
                name="ms_binary_cudem",
                args=dict(
                    steps=1,
                    resolutions="3/9",
                    weights=".5",
                    algos="raster_fill/raster_fill",
                    barrier=None,
                    inland_decay_dist=0,
                    output=str(tmp_path / "yaml_dem.tif"),
                ),
            ),
        ],
    )
    path = tmp_path / "normal.yaml"
    path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    command = [
        str(Path(sys.executable).with_name("globato")),
        "run",
        str(path),
        "--outdir",
        str(tmp_path / "build"),
        "--shared-cache",
        str(tmp_path / "shared"),
        "--fail-fast",
    ]
    done = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, timeout=120
    )
    (tmp_path / "cli.log").write_text(done.stdout + done.stderr)
    assert done.returncode == 0, done.stdout + done.stderr
    with rasterio.open(tmp_path / "yaml_dem.tif") as ds:
        assert ds.count == 1 and float(ds.read(1, masked=True).max()) < -90.0
    report = read(next(tmp_path.rglob("result_*.json")))
    assert report["settings"]["aggressiveness"] == 2


def test_exact_stack_grid_and_cached_points_are_processed_only_once(
    tmp_path, monkeypatch
):
    from globato.hooks.filters.spatial_crop import SpatialCrop
    from globato.hooks.sinks.side_stack import SideStackCheck, SideStackGenerate
    from fetchez.hooks import FetchHook

    class MeasuredOnly(SpatialCrop):
        def __init__(self):
            super().__init__()
            self.called = 0

        def run(self, entries):
            for _, entry in entries:
                if entry.get("stream") is not None:
                    assert self.is_point_stream(entry), (
                        "Cached FusionState must bypass point filters"
                    )
                    self.called += 1
            return super().run(entries)

    class CountManifest(FetchHook):
        name = "manifest-once"
        meta_stage = "manifest"

        def run(self, entries):
            for _, entry in entries:
                entry["prepared_count"] = entry.get("prepared_count", 0) + 1
            return entries

    class InputLabelOnHit(SideStackCheck):
        def run(self, entries):
            entries = super().run(entries)
            for _, entry in entries:
                if entry.get("_sidestack_hit"):
                    entry["src_srs"] = "EPSG:4326"  # Simulate the original input label.
            return entries

    monkeypatch.chdir(tmp_path)
    for iteration in range(2):
        mod, qc, hooks = setup(tmp_path)
        point_filter = MeasuredOnly()
        # Reproduce a preset's inconsistent source-vs-stack increment. The
        # Stack QC source owner must use the actual 3-m native stack grid.
        hooks[:0] = [
            CountManifest(),
            InputLabelOnHit(res="9", crs="EPSG:32611"),
            point_filter,
            SideStackGenerate(res="9", crs="EPSG:32611"),
        ]
        result = run_fetchez(
            [mod], threads=1, global_hooks=hooks, ignore_failures=False
        )
        assert Path(result[0][1]["dst_fn"]).is_file()
        assert point_filter.called == 0  # Prep owns a deep copy of the global filter.
        inv = read(next((tmp_path / "qc").glob("*/SOURCES.json")))
        assert inv["sources"][0]["entry"]["prepared_count"] == 1
        assert (
            rasterio.crs.CRS.from_user_input(
                inv["sources"][0]["entry"]["src_srs"]
            ).to_epsg()
            == 32611
        )
        with rasterio.open(inv["sources"][0]["path"]) as ds:
            assert ds.transform.a == 3.0 and ds.transform.e == -3.0
    assert len(list((tmp_path / "qc").glob("*/SOURCES.json"))) == 1


def test_qc_and_ordinary_modules_keep_ordinary_transform_and_contributions(
    tmp_path, monkeypatch
):
    from fetchez.modules.local_fs import LocalFS
    from fetchez.hooks import FetchHook

    monkeypatch.chdir(tmp_path)
    measured, qc, hooks = setup(tmp_path)
    file = tmp_path / "ordinary.xyz"
    data = np.loadtxt(measured.path)
    data[:, 2] = -50.0
    np.savetxt(file, data)

    class CountTransform(FetchHook):
        name = "fixture_ordinary_transform"
        meta_stage = "stream"

        def __init__(self):
            super().__init__()
            self.calls = 0

        def run(self, entries):
            self.calls += 1
            return entries

    transform = CountTransform()
    ordinary = LocalFS(
        path=str(file),
        data_type="xyz",
        src_region=measured.region,
        outdir=str(tmp_path / "ordinary_cache"),
        hook=[transform],
    )
    ordinary.run()
    for entry in ordinary.results:
        entry["src_srs"] = "EPSG:32611"
    run_plan([measured, ordinary], global_hooks=hooks)
    assert transform.calls == 1
    with rasterio.open(hooks[0].output) as ds:
        count = ds.read(2)
        assert count[32, 32] == 1 and count[5, 5] == 2
        assert ds.read(1)[32, 32] == -50.0
    with rasterio.open(tmp_path / "normal_dem.tif") as ds:
        assert ds.count == 1 and ds.read(1, masked=True).max() == -50.0
