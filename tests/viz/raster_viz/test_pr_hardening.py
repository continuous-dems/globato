"""Regression coverage for stale data, conservative classification and group framing."""

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from globato.viz.raster_viz.review import Review
from globato.viz.raster_viz.discovery import discover_into, raster_kind
from globato.viz.raster_viz.manifest import sync_catalog, read_catalog
from globato.viz.raster_viz.catalog import ProductCatalog


def dem(path, west=500000, value=100):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=16,
        height=12,
        count=1,
        dtype="float32",
        crs="EPSG:32610",
        nodata=-9999,
        transform=from_origin(west, 4500000, 5, 5),
    ) as ds:
        ds.write(np.full((12, 16), value, dtype="float32"), 1)
    return path


def state(rv):
    return json.loads(rv.path.read_text())


def test_directory_removed_files_are_pruned_without_deleting_manual_products(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    first = dem(root / "first.tif")
    second = dem(root / "second.tif")
    manual = dem(tmp_path / "manual.tif")
    rv = Review(tmp_path / "viz")
    rv.add_raster("manual", manual)
    assert len(discover_into(rv, root)["added"]) == 2
    first.unlink()
    result = discover_into(rv, root)
    assert str(first) in result["removed"]
    records = state(rv)["areas"]["default"]["products"]
    assert len(records) == 2
    assert "manual" in records
    assert not any(
        p.get("source", {}).get("path") == str(first) for p in records.values()
    )
    assert any(p.get("source", {}).get("path") == str(second) for p in records.values())
    assert all(k in records for k in state(rv)["areas"]["default"].get("defaults", []))


def test_directory_different_scan_roots_do_not_prune_each_other(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    one = dem(a / "one.tif")
    two = dem(b / "two.tif")
    rv = Review(tmp_path / "viz")
    discover_into(rv, a)
    discover_into(rv, b)
    one.unlink()
    discover_into(rv, b)
    paths = {
        v["source"]["path"] for v in state(rv)["areas"]["default"]["products"].values()
    }
    assert str(two) in paths
    # The other root has not been scanned again, so its previous preview persists.
    assert str(one) in paths
    discover_into(rv, a)
    paths = {
        v["source"]["path"] for v in state(rv)["areas"]["default"]["products"].values()
    }
    assert str(one) not in paths and str(two) in paths


def test_diff_filename_not_3d_terrain(tmp_path):
    change = dem(tmp_path / "bathy_diff.tif")
    style, why = raster_kind(change)
    assert style == "difference" and not why["terrain_eligible"]
    rv = Review(tmp_path / "viz", render_3d=True)
    root = tmp_path / "root"
    root.mkdir()
    dem(root / "final_delta.tif")
    discover_into(rv, root)
    records = state(rv)["areas"]["default"]["products"]
    assert len(records) == 1
    product = next(iter(records.values()))
    assert product["style"] == "difference"
    assert "terrain_scene" not in product
    assert product["group"] == "discovered_difference"
    assert raster_kind(dem(tmp_path / "changeover.tif"))[0] == "elevation"


def test_invalid_catalog_refresh_does_not_show_previous_valid_product(tmp_path):
    source = dem(tmp_path / "src.tif")
    c = ProductCatalog(tmp_path / "publish/products.json")
    c.add_raster("src", source).compare("src", "src").publish()
    rv = Review(tmp_path / "viz")
    assert sync_catalog(rv, c.path)["added"] == ["src"]
    source.write_bytes(b"not a geotiff")
    result = sync_catalog(rv, c.path)
    assert result["failed"] and result["failed"][0]["key"] == "src"
    assert "src" not in state(rv)["areas"]["default"]["products"]


def test_camera_bounds_union_and_revised_extent(tmp_path):
    left = dem(tmp_path / "left.tif", west=500000)
    right = dem(tmp_path / "right.tif", west=502000)
    c = ProductCatalog(tmp_path / "publish/products.json")
    c.add_raster("left", left).add_raster("right", right).compare(
        "left", "right"
    ).publish()
    rv = Review(tmp_path / "viz")
    sync_catalog(rv, c.path)
    group = state(rv)["areas"]["default"]["groups"]["elevation"]
    with rasterio.open(left) as ds:
        bl = transform_bounds(ds.crs, "EPSG:3857", *ds.bounds)
    with rasterio.open(right) as ds:
        br = transform_bounds(ds.crs, "EPSG:3857", *ds.bounds)
    actual = group["camera_bounds_3857"]
    assert actual[0] <= min(bl[0], br[0]) + 1e-5
    assert actual[2] >= max(bl[2], br[2]) - 1e-5
    # Re-using the output location for a new AOI must not pin an old group camera.
    dem(left, west=503000)
    dem(right, west=504000)
    sync_catalog(rv, c.path)
    updated = state(rv)["areas"]["default"]["groups"]["elevation"]["camera_bounds_3857"]
    assert updated[0] > actual[0]
    assert updated[2] > actual[2]


def test_explicit_difference_3d_requires_opt_in(tmp_path):
    source = dem(tmp_path / "diff.tif")
    c = ProductCatalog(tmp_path / "publish/products.json")
    c.add_raster(
        "difference", source, style="difference", group="change", views=["2d", "3d"]
    ).compare("difference", "difference")
    data, _ = read_catalog(c.publish())
    assert data["products"][0]["views"] == ["2d", "3d"]
    rv = Review(tmp_path / "viz", render_3d=True, mesh_side=20)
    assert sync_catalog(rv, c.path)["added"] == ["difference"]
    assert state(rv)["areas"]["default"]["products"]["difference"]["terrain_scene"]
    with pytest.raises(ValueError, match="3d"):
        c.add_raster("rgb", source, style="rgb", views=["2d", "3d"])
    malformed = {
        "schema_version": 1,
        "products": [
            {"key": "x", "path": "x.tif", "style": "scalar", "views": ["2d", "3d"]}
        ],
    }
    cat_path = tmp_path / "malformed.json"
    cat_path.write_text(json.dumps(malformed))
    with pytest.raises(ValueError, match="3d"):
        read_catalog(cat_path)


def test_rerun_without_3d_clears_previous_mode(tmp_path):
    from globato.viz.raster_viz.cli import main

    dem(tmp_path / "terrain.tif")
    out = tmp_path / "viz"
    assert (
        main(
            [str(tmp_path), "--out", str(out), "--3d", "--no-open", "--mesh-side", "20"]
        )
        == 0
    )
    assert state(Review(out))["enable_3d"] is True
    assert main([str(tmp_path), "--out", str(out), "--no-open"]) == 0
    updated = state(Review(out))
    assert updated["enable_3d"] is False
    row = next(iter(updated["areas"]["default"]["products"].values()))
    assert row["render_settings"]["make_3d"] is False
    assert "terrain_scene" not in row


def test_html_embedded_catalog_metadata_is_not_executable(tmp_path):
    source = dem(tmp_path / "terrain.tif")
    rv = Review(tmp_path / "viz", title="Check </script><script>bad()</script>")
    rv.add_raster(
        "terrain", source, metadata={"note": "</script><script>bad()</script>"}
    )
    html = (rv.output / "viz.html").read_text()
    assert "<script>bad()</script>" not in html
    assert "\\u003c/script>" in html


def test_viewer_javascript_syntax_when_node_available():
    import shutil
    import subprocess
    import re

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not available")
    html = (
        Path(__file__).resolve().parents[3] / "src/globato/viz/raster_viz/viewer.html"
    ).read_text()
    js = re.search(r"<script>\s*(.*?)\s*</script>", html, flags=re.S).group(1)
    check = subprocess.run(
        [node, "--check", "-"], input=js, text=True, capture_output=True, timeout=15
    )
    assert check.returncode == 0, check.stderr


def test_discovery_refreshes_style_after_filename_metadata_revision(
    tmp_path, monkeypatch
):
    """A cached older inference must not silently survive a new discovery rule."""
    from globato.viz.raster_viz import discovery
    import unittest.mock

    root = tmp_path / "input"
    root.mkdir()
    dem(root / "change_diff.tif")
    rv = Review(tmp_path / "viz")

    def before(path, band=1):
        return "elevation", {"type_inference": "old_rule"}

    with unittest.mock.patch.object(discovery, "raster_kind", before):
        assert discovery.discover_into(rv, root)["added"]
    assert (
        next(iter(state(rv)["areas"]["default"]["products"].values()))["style"]
        == "elevation"
    )
    result = discovery.discover_into(rv, root)
    assert result["added"] and not result["unchanged"]
    assert (
        next(iter(state(rv)["areas"]["default"]["products"].values()))["style"]
        == "difference"
    )


def test_long_running_event_history_is_bounded(tmp_path):
    from globato.viz.raster_viz.review import MAX_EVENTS

    rv = Review(tmp_path / "viz")
    for i in range(MAX_EVENTS + 15):
        rv.event(f"failure_{i}", status="skipped")
    events = state(rv)["events"]
    assert len(events) == MAX_EVENTS
    assert events[0]["text"] == "failure_15"
    assert events[-1]["text"] == f"failure_{MAX_EVENTS + 14}"
