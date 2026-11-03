"""Regression coverage for the interoperable catalog and incremental refresh."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from globato.viz.raster_viz.review import Review
from globato.viz.raster_viz.manifest import sync_catalog
from globato.viz.raster_viz.catalog import ProductCatalog

ROOT = Path(__file__).resolve().parents[3]


def source(path, bias=0, count=1):
    array = np.arange(20 * 24, dtype="float32").reshape(20, 24) + bias
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=20,
        width=24,
        count=count,
        dtype="float32",
        crs="EPSG:32610",
        transform=from_origin(500000, 4500000, 5, 5),
        nodata=-9999,
    ) as dst:
        for band in range(1, count + 1):
            dst.write(array + band, band)
    return path


def run_catalog(*args, cwd):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    return subprocess.run(
        [sys.executable, "-m", "globato.viz.raster_viz.catalog_cli", *map(str, args)],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=45,
    )


def test_catalog_command_explicit_and_styles(tmp_path):
    a = source(tmp_path / "a.tif")
    b = source(tmp_path / "b.tif", bias=10)
    rgb = source(tmp_path / "rgb.tif", count=3)
    out = tmp_path / "catalog/products.json"
    p = run_catalog(
        "--out",
        out,
        "--raster",
        f"before={a}",
        "--raster",
        f"after={b}",
        "--rgb",
        f"s2={rgb}",
        "--compare",
        "before,after",
        "--label",
        "before=Original",
        "--stage",
        "after=Filtered",
        cwd=tmp_path,
    )
    assert p.returncode == 0, p.stderr
    obj = json.loads(out.read_text())
    assert [r["style"] for r in obj["products"]] == ["elevation", "elevation", "rgb"]
    assert [r["group"] for r in obj["products"]] == [
        "elevation",
        "elevation",
        "imagery",
    ]
    assert obj["defaults"]["default"] == ["before", "after"]
    assert obj["products"][0]["label"] == "Original"
    assert obj["products"][1]["stage"] == "Filtered"
    assert (out.parent / obj["products"][0]["path"]).resolve() == a


def test_catalog_command_scan_bounded_and_reject_duplicates(tmp_path):
    source(tmp_path / "a.tif")
    source(tmp_path / "b.tif")
    out = tmp_path / "products.json"
    p = run_catalog("--scan", tmp_path, "--out", out, "--max-files", "1", cwd=tmp_path)
    assert p.returncode != 0 and not out.exists()
    p = run_catalog("--scan", tmp_path, "--out", out, cwd=tmp_path)
    assert p.returncode == 0, p.stderr
    d = json.loads(out.read_text())
    assert len(d["products"]) == 2
    assert len({p["key"] for p in d["products"]}) == 2
    p = run_catalog("--raster", "a=a.tif", "--raster", "a=b.tif", cwd=tmp_path)
    assert p.returncode != 0


def test_catalog_limits_change_and_removed_source(tmp_path):
    a = source(tmp_path / "a.tif")
    b = source(tmp_path / "b.tif", bias=100)
    m = tmp_path / "products.json"
    c = ProductCatalog(m)
    c.add_raster("a", a, limits=[0, 500]).add_raster("b", b, limits=[0, 500]).compare(
        "a", "b"
    )
    c.publish(running=True)
    rv = Review(tmp_path / "viz", render_3d=False)
    assert set(sync_catalog(rv, m)["added"]) == {"a", "b"}
    c2 = ProductCatalog(m)
    c2.add_raster("a", a, limits=[-50, 600]).add_raster(
        "b", b, limits=[-50, 600]
    ).compare("a", "b")
    c2.publish(running=True)
    result = sync_catalog(rv, m)
    assert not result["failed"], result
    state = rv._load()["areas"]["default"]
    assert state["products"]["a"]["limits"] == [-50, 600]
    assert state["products"]["b"]["limits"] == [-50, 600]
    assert sync_catalog(rv, m)["unchanged"] == ["a", "b"]
    b.unlink()
    result = sync_catalog(rv, m)
    assert result["pending"] and "b" not in rv._load()["areas"]["default"]["products"]


def test_rgb_texture_update_does_not_invalidate_elevation_scene(tmp_path):
    dem = source(tmp_path / "dem.tif")
    rgb = source(tmp_path / "rgb.tif", count=3)
    cat = ProductCatalog(tmp_path / "products.json")
    cat.add_raster("dem", dem).add_raster("rgb", rgb, style="rgb").compare("dem", "dem")
    cat.publish()
    viewer = Review(tmp_path / "viz", render_3d=True, mesh_side=25)
    result = sync_catalog(viewer, cat.path)
    assert not result["failed"], result
    first = viewer._load()["areas"]["default"]["products"]["dem"]["terrain_scene"]["id"]
    with rasterio.open(rgb, "r+") as ds:
        arr = ds.read(1)
        ds.write(arr + 100, 1)
    result = sync_catalog(viewer, cat.path)
    assert not result["failed"], result
    second = viewer._load()["areas"]["default"]["products"]["dem"]["terrain_scene"][
        "id"
    ]
    assert first == second, "RGB background must not force elevation mesh regeneration"
    assert viewer._load()["areas"]["default"]["products"]["rgb"]["style"] == "rgb"


def test_installed_entrypoint_contract_returns_success(tmp_path):
    """Setuptools entrypoints call sys.exit(main()), so main must return 0."""
    from globato.viz.raster_viz.cli import main

    source(tmp_path / "dem.tif")
    assert main([str(tmp_path), "--out", str(tmp_path / "viz"), "--no-open"]) == 0


def test_directory_discovers_late_rgb_for_3d(tmp_path):
    from globato.viz.raster_viz.discovery import discover_into

    source(tmp_path / "dem.tif")
    viewer = Review(tmp_path / "viz", render_3d=True, mesh_side=25)
    options = dict(
        max_files=10,
        exclude_processing=True,
        skip_existing=True,
        extensions={".tif"},
        recursive=False,
    )
    first = discover_into(viewer, tmp_path, **options)
    assert len(first["added"]) == 1
    record = next(iter(viewer._load()["areas"]["default"]["products"].values()))
    before = record["terrain_scene"]["id"]
    source(tmp_path / "rgb.tif", count=3)
    # The automatic classifier relies on recognizable RGB naming.
    result = discover_into(viewer, tmp_path, **options)
    assert result["added"] == ["rgb.tif"], (
        "late RGB must not re-render an unchanged DEM"
    )
    record = next(
        v
        for v in viewer._load()["areas"]["default"]["products"].values()
        if v["style"] == "elevation"
    )
    assert record["terrain_scene"]["id"] == before, (
        "late imagery should not require an elevation mesh rebuild"
    )
    assert any(
        p["style"] == "rgb"
        for p in viewer._load()["areas"]["default"]["products"].values()
    )
