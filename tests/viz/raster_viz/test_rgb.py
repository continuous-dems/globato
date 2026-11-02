"""RGB underlay and 3D color-only integration tests (all synthetic, no network)."""

import json
import subprocess
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
import pytest
from PIL import Image
from rasterio.transform import from_origin
from globato.viz.raster_viz.catalog import ProductCatalog
from globato.viz.raster_viz.manifest import read_catalog

ROOT = Path(__file__).resolve().parents[3]


def synthetic(root):
    raster = root / "dem.tif"
    rgb = root / "RGB_10m.tif"
    transform = from_origin(500000, 4500000, 5, 5)
    h, w = 32, 40
    surface = np.arange(h * w, dtype="float32").reshape(h, w) * 0.01 - 40
    surface[:5, :5] = -9999
    with rasterio.open(
        raster,
        "w",
        driver="GTiff",
        width=w,
        height=h,
        count=1,
        dtype="float32",
        crs="EPSG:32610",
        transform=transform,
        nodata=-9999.0,
    ) as ds:
        ds.write(surface, 1)
    with rasterio.open(
        rgb,
        "w",
        driver="GTiff",
        width=w,
        height=h,
        count=3,
        dtype="uint8",
        crs="EPSG:32610",
        transform=transform,
        nodata=0,
    ) as ds:
        for b, value in enumerate((180, 90, 20), start=1):
            block = np.full((h, w), value, dtype="uint8")
            block[10:15, 10:15] = 0
            ds.write(block, b)
    return raster, rgb


def cli(*args, cwd):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "globato.viz.raster_viz", *map(str, args), "--no-open"],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_rgb_catalog_2d_and_3d(tmp_path):
    dem, rgb = synthetic(tmp_path)
    cat = ProductCatalog(tmp_path / "products.json", title="RGB test")
    cat.add_raster("dem", dem, label="DEM")
    cat.add_raster("s2", rgb, label="Sentinel RGB", style="rgb")
    cat.compare("dem", "dem")
    cat.publish()
    data, groups = read_catalog(cat.path)
    assert groups[("default", "imagery")]["style"] == "rgb"
    check = cli(
        "--manifest",
        cat.path,
        "--out",
        tmp_path / "viz",
        "--3d",
        "--mesh-side",
        "40",
        cwd=tmp_path,
    )
    assert check.returncode == 0, check.stderr + check.stdout
    state = json.loads((tmp_path / "viz/review.json").read_text())["areas"]["default"][
        "products"
    ]
    assert state["s2"]["style"] == "rgb" and state["s2"]["legend"] is None
    assert "rgb_data_script" in state["s2"]
    texture_js = (tmp_path / "viz" / state["s2"]["rgb_data_script"]).read_text()
    assert "data:image/png;base64," in texture_js

    png = Image.open(tmp_path / "viz" / state["s2"]["image"]).convert("RGBA")
    rgba = np.asarray(png)
    assert rgba.shape[2] == 4 and rgba[..., 0].max() >= 170 and rgba[..., 1].max() >= 80
    assert (rgba[..., 3] == 0).any(), "RGB NoData must be transparent"
    sc = state["dem"]["terrain_scene"]
    assert sc["triangles"] > 0
    text = (tmp_path / "viz" / sc["script"]).read_text()
    json_data = json.loads(text.split("]=", 1)[1].removesuffix(";\n"))
    # Standalone 3D RGB now uses the georeferenced preview as a texture
    # on an independent plane; no per-vertex RGB colors are required.
    assert not json_data.get("rgb_colors")
    html = (tmp_path / "viz/viz.html").read_text()
    assert "RGB background" in html
    assert "function drawBackground3d(p,mvp)" in html
    assert "backgroundFloor(p)" in html
    assert "RGB transparency" in html
    with rasterio.open(dem) as ds:
        assert (ds.read(1) == -9999).sum() == 25


def test_rgb_discovery_and_no_3d_requirement(tmp_path):
    dem, rgb = synthetic(tmp_path)
    run = cli("--out", tmp_path / "viz", cwd=tmp_path)
    assert run.returncode == 0, run.stderr + run.stdout
    products = json.loads((tmp_path / "viz/review.json").read_text())["areas"][
        "default"
    ]["products"]
    assert {v["style"] for v in products.values()} == {"elevation", "rgb"}
    assert not any(v.get("terrain_scene") for v in products.values())


def test_invalid_rgb_rejected(tmp_path):
    dem, rgb = synthetic(tmp_path)
    with pytest.raises(ValueError, match="RGB"):
        ProductCatalog(tmp_path / "products.json").add_raster(
            "rgb", rgb, style="rgb", limits=[0, 255]
        )
    cat = ProductCatalog(tmp_path / "products.json")
    cat.add_raster("bad", dem, style="rgb").publish()
    cli("--manifest", cat.path, "--out", tmp_path / "viz", cwd=tmp_path)
    state = json.loads((tmp_path / "viz/review.json").read_text())
    assert (
        state["catalog_result"]["failed"]
        and "3 bands" in state["catalog_result"]["failed"][0]["reason"]
    )
