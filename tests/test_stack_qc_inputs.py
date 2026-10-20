"""Native reader compatibility, shared selection and lossless source caches."""

from pathlib import Path
import json
import numpy as np
import rasterio
from rasterio.transform import from_origin
from test_stack_qc_native import run_plan as run_fetchez
from fetchez.hooks import FetchHook
from fetchez.hooks.spatial_claim import SpatialClaimHook
from fetchez.spatial import Region
from globato.hooks.filters.claim_grid import ClaimGridFilter
from globato.hooks.filters.stack_qc import StackQC
from globato.hooks.sinks.side_stack import (
    SideStackGenerate,
    hook_identity,
)
from test_stack_qc_native import setup


def test_empty_qc_module_does_not_abort_another_modules_normal_build(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    empty_dir, good_dir = tmp_path / "empty", tmp_path / "good"
    empty_dir.mkdir()
    good_dir.mkdir()
    empty, _, _ = setup(empty_dir)
    empty.results = []
    good, _, hooks = setup(good_dir)
    result = run_fetchez(
        [empty, good], threads=1, global_hooks=hooks, ignore_failures=False
    )
    assert Path(result[0][1]["dst_fn"]).is_file()
    assert not (empty_dir / "qc").exists()


def test_module_side_stack_runs_once_after_point_filters(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module, _, hooks = setup(tmp_path)
    module.add_hook(SideStackGenerate(cache_dir=str(tmp_path / "explicit_side_cache")))

    class Offset(FetchHook):
        name = "fixture_offset"
        meta_stage = "stream"

        def run(self, entries):
            for _, entry in entries:
                if not self.is_point_stream(entry):
                    continue
                stream = entry["stream"]

                def shifted(stream=stream):
                    for chunk in stream:
                        chunk["z"] -= 10
                        yield chunk

                entry["stream"] = shifted()
            return entries

    module.add_hook(Offset())
    result = run_fetchez([module], threads=1, global_hooks=hooks, ignore_failures=False)
    assert Path(result[0][1]["dst_fn"]).is_file()
    files = list((tmp_path / "explicit_side_cache").glob("*sidestack.tif"))
    # The ancillary metadata entry has no side stack.
    assert len(files) == 1
    with rasterio.open(files[0]) as ds:
        b = ds.read()
        valid = b[1] > 0
        z = b[0][valid] / b[2][valid]
        assert np.median(z) < -109


def test_raster_reader_reprojection_and_unknown_identity_work_in_normal_build(
    tmp_path, monkeypatch
):
    from fetchez.modules.local_fs import LocalFS
    from fetchez.hooks.stream_init import DataStream
    from globato.hooks.transforms.reproject import StreamReproject

    monkeypatch.chdir(tmp_path)
    _, _, hooks = setup(tmp_path)
    region = Region(350000, 350192, 3749808, 3750000, srs="EPSG:32611")
    path = tmp_path / "elevation.tif"
    z = np.full((64, 64), -100, dtype="float32")
    z[32, 32] = 0
    z[0, 0] = -9999
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=1,
        dtype="float32",
        nodata=-9999,
        crs="EPSG:32611",
        transform=from_origin(350000, 3750000, 3, 3),
    ) as ds:
        ds.write(z, 1)
    original = path.read_bytes()
    qc = StackQC(max_passes=1, terrain_restoration=2)
    qc.cache_dir = str(tmp_path / "raster_qc")
    module = LocalFS(
        path=str(path),
        data_type="raster",
        src_region=region,
        outdir=str(tmp_path / "source_cache"),
        hook=[DataStream(), StreamReproject(dst_srs="EPSG:32611"), qc],
    )
    module.run()
    result = run_fetchez([module], threads=1, global_hooks=hooks, ignore_failures=False)
    with rasterio.open(result[0][1]["dst_fn"]) as ds:
        assert ds.read(1, masked=True).max() < -90
    inventory = json.loads(
        next((tmp_path / "raster_qc").glob("*/SOURCES.json")).read_text()
    )
    assert inventory["sources"][0]["group"] is None
    assert path.read_bytes() == original


def test_raster_with_no_valid_observations_does_not_abort_other_sources(
    tmp_path, monkeypatch
):
    from fetchez.modules.local_fs import LocalFS
    from fetchez.hooks.stream_init import DataStream

    monkeypatch.chdir(tmp_path)
    good, _, hooks = setup(tmp_path)
    path = tmp_path / "empty.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=1,
        dtype="float32",
        nodata=-9999,
        crs="EPSG:32611",
        transform=from_origin(350000, 3750000, 3, 3),
    ) as ds:
        ds.write(np.full((64, 64), -9999, dtype="float32"), 1)
    empty = LocalFS(
        path=str(path),
        data_type="raster",
        src_region=good.region,
        outdir=str(tmp_path / "empty_cache"),
        hook=[DataStream(), StackQC()],
    )
    empty.run()
    result = run_fetchez(
        [empty, good], threads=1, global_hooks=hooks, ignore_failures=False
    )
    assert Path(result[0][1]["dst_fn"]).is_file()
    assert not list((tmp_path / "empty_qc").glob("*/result_*.json"))


def test_spatial_claim_sees_all_modules_before_qc_and_invalidates_changed_coverage(
    tmp_path, monkeypatch
):
    from shapely.geometry import box
    from shapely import from_wkt

    monkeypatch.chdir(tmp_path)
    high_dir, low_dir = tmp_path / "high", tmp_path / "low"
    high_dir.mkdir()
    low_dir.mkdir()
    high, _, _ = setup(high_dir)
    low, _, hooks = setup(low_dir)
    high.results = [high.results[0]]
    low.results = [low.results[0]]
    whole = box(350000, 3749808, 350192, 3750000)
    for mod, priority, geometry in [
        (high, 2, box(350000, 3749808, 350096, 3750000)),
        (low, 1, whole),
    ]:
        mod.results[0].update(
            claim_priority=priority, claim_required=True, claim_geometry=geometry.wkt
        )

    class Claims(SpatialClaimHook):
        def run(self, entries):
            assert len(entries) == 2
            self.calls += 1
            return super().run(entries)

    claims = Claims()
    claims.calls = 0
    clipping = ClaimGridFilter(res="3", geometry_srs="EPSG:32611")
    result = run_fetchez(
        [high, low],
        threads=1,
        global_hooks=[claims, clipping, *hooks],
        ignore_failures=False,
    )
    assert claims.calls == 1 and Path(result[0][1]["dst_fn"]).is_file()
    assert from_wkt(low.results[0]["accepted_geometry"]).area == whole.area / 2
    inventory = json.loads(next((low_dir / "qc").glob("*/SOURCES.json")).read_text())
    with rasterio.open(inventory["sources"][0]["path"]) as ds:
        # Exactly one half remains; lower-priority data cannot leak through cache.
        assert ds.read(2)[:, :32].sum() == 0
        assert ds.read(2)[:, 32:].sum() == 2048
    one = hook_identity(clipping, {"accepted_geometry": whole.wkt})
    two = hook_identity(
        clipping, {"accepted_geometry": box(350096, 3749808, 350192, 3750000).wkt}
    )
    assert one != two
    previous = inventory["sources"][0]["path"]
    high2, _, _ = setup(high_dir)
    low2, _, hooks2 = setup(low_dir)
    high2.results = [high2.results[0]]
    low2.results = [low2.results[0]]
    for mod, priority, geometry in [
        (high2, 2, box(350000, 3749808, 350048, 3750000)),
        (low2, 1, whole),
    ]:
        mod.results[0].update(
            claim_priority=priority, claim_required=True, claim_geometry=geometry.wkt
        )
    run_fetchez(
        [high2, low2],
        threads=1,
        global_hooks=[claims, clipping, *hooks2],
        ignore_failures=False,
    )
    assert claims.calls == 2
    inventories = [
        json.loads(p.read_text()) for p in (low_dir / "qc").glob("*/SOURCES.json")
    ]
    updated = next(d for d in inventories if d["sources"][0]["path"] != previous)
    with rasterio.open(updated["sources"][0]["path"]) as ds:
        assert ds.read(2)[:, :16].sum() == 0
        assert ds.read(2)[:, 16:].sum() == 3072


def test_native_help_explains_public_controls():
    from click.testing import CliRunner
    from globato.cli import cli

    result = CliRunner().invoke(
        cli, ["build", "mbdb_filter_test", "stack_qc", "--help"]
    )
    # The local bundle is not required just to inspect a registered hook.
    if result.exit_code:
        result = CliRunner().invoke(cli, ["build", "stack_qc", "--help"])
    assert result.exit_code == 0, result.output
    assert (
        "Filtering strength" in result.output and "Terrain restoration" in result.output
    )
