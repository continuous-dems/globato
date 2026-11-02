"""Display-only water-level color threshold: publisher, CLI, 2D, 3D and caching.

The test values are synthetic and are not authoritative Great Lakes water levels.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from globato.viz.raster_viz.palette import elevation_rgba
from globato.viz.raster_viz.legends import elevation_ramp
from globato.viz.raster_viz.review import Review
from globato.viz.raster_viz.manifest import read_catalog, sync_catalog
from globato.viz.raster_viz.catalog import ProductCatalog


def _raster(path):
    z = np.array(
        [
            [100.0, 150.0, 175.0, 176.0],
            [176.01, 190.0, 225.0, 300.0],
            [110.0, 176.0, 185.0, 240.0],
            [160.0, 165.0, 180.0, 210.0],
        ],
        dtype="float32",
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:32616",
        transform=from_origin(500000, 5000400, 25, 25),
    ) as ds:
        ds.write(z, 1)
    return path


def _pixel_color(array):
    return tuple(np.round(np.asarray(array)[:3] * 255).astype(int))


def test_default_zero_and_great_lakes_threshold_are_in_source_elevation_units():
    z = np.array([-10.0, 0.0, 0.1, 175.0, 175.99, 176.0, 180.0, np.nan])
    limits = [-20.0, 220.0]
    default = elevation_rgba(z, limits)
    lake = elevation_rgba(z, limits, water_level=176)
    # Default below-zero is blue and zero is land.
    assert default[0, 2] > default[0, 0]
    assert default[1, 1] > default[1, 2]
    # With lake threshold, 175 m still blue; exactly 176 m is land.
    assert lake[4, 2] > lake[4, 0]
    assert lake[5, 1] > lake[5, 2]
    assert not np.array_equal(default[3, :3], lake[3, :3])
    assert lake[-1, 3] == 0
    for invalid in (np.nan, np.inf, -np.inf, True):
        with pytest.raises(ValueError):
            elevation_rgba(z, limits, water_level=invalid)


def test_scale_has_transition_at_exact_level_and_2d_3d_use_identical_palette(tmp_path):
    bounds = [100.0, 250.0]
    # Palette assets differ for different levels at unchanged limits.
    zero = elevation_ramp(tmp_path, bounds, 0)
    lake = elevation_ramp(tmp_path, bounds, 176)
    assert zero != lake
    assert (tmp_path / lake).exists()
    # In a fully non-negative display range, the zero palette is all land,
    # whereas the 176 palette has both blue and land segments.
    img_zero = np.array(Image.open(tmp_path / zero))
    img_lake = np.array(Image.open(tmp_path / lake))
    assert not np.array_equal(img_zero, img_lake)
    for value in [130, 175.99, 176.0, 210.0]:
        rgba = elevation_rgba(np.array([value]), bounds, water_level=176)
        from globato.viz.raster_viz.review import color_image
        from rasterio.transform import Affine

        # Single-pixel rendering avoids hillshade; 3D uses the same public palette function.
        image = color_image(
            np.array([[value]]), Affine.translation(0, 0), bounds, water_level=176
        )
        # A single pixel gets only the fixed flat-terrain hillshade factor.
        flat_shade = 0.4 + 0.6 * np.sin(np.deg2rad(45))
        assert (
            np.max(
                np.abs(
                    image[0, 0, :3].astype("int")
                    - (rgba[0, :3] * 255 * flat_shade).astype("int")
                )
            )
            <= 2
        )


def test_catalog_group_inherits_explicit_level_and_rejects_conflicts(tmp_path):
    dem = _raster(tmp_path / "dem.tif")
    c = ProductCatalog(tmp_path / "products.json")
    c.add_raster("before", dem, water_level=176.0)
    c.add_raster("after", dem)  # Same group inherits 176.0, not default 0.
    c.compare("before", "after")
    c.publish()
    _, groups = read_catalog(c.path)
    assert groups[("default", "elevation")]["water_level"] == 176
    result = sync_catalog(Review(tmp_path / "viz", mesh_side=10), c.path)
    assert result["failed"] == [], result
    items = json.loads((tmp_path / "viz/review.json").read_text())["areas"]["default"][
        "products"
    ]
    assert items["before"]["water_level"] == items["after"]["water_level"] == 176
    assert items["before"]["legend"]["ramp"] == items["after"]["legend"]["ramp"]
    assert items["before"]["limits"] == items["after"]["limits"]
    assert items["before"]["legend"]["title"] == "Elevation"
    # Same group cannot mix thresholds; separate group can.
    raw = json.loads(c.path.read_text())
    raw["products"][1]["water_level"] = 175
    c.path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Conflicting water_level"):
        read_catalog(c.path)
    raw["products"][1]["group"] = "other"
    c.path.write_text(json.dumps(raw))
    read_catalog(c.path)


def test_threshold_is_rejected_for_difference_and_other_styles(tmp_path):
    dem = _raster(tmp_path / "dem.tif")
    with pytest.raises(ValueError, match="water_level"):
        ProductCatalog(tmp_path / "products.json").add_raster(
            "delta", dem, style="difference", water_level=176
        )
    with pytest.raises(ValueError, match="water_level"):
        ProductCatalog(tmp_path / "products.json").add_raster(
            "rgb", dem, style="rgb", water_level=176
        )
    for level in (True, "176", float("inf")):
        with pytest.raises(ValueError, match="water_level"):
            ProductCatalog(tmp_path / "products.json").add_raster(
                "x", dem, water_level=level
            )


def test_color_threshold_refresh_recolors_2d_and_3d_with_source_unchanged(tmp_path):
    dem = _raster(tmp_path / "dem.tif")
    c = ProductCatalog(tmp_path / "products.json")
    c.add_raster("dem", dem, water_level=0.0, limits=[100.0, 300.0])
    c.compare("dem", "dem")
    c.publish()
    rv = Review(tmp_path / "viz", render_3d=True, mesh_side=8)
    r1 = sync_catalog(rv, c.path)
    assert r1["added"] == ["dem"], r1
    before = json.loads((tmp_path / "viz/review.json").read_text())["areas"]["default"][
        "products"
    ]["dem"]
    before_png = (tmp_path / "viz" / before["image"]).read_bytes()
    before_scene = before["terrain_scene"]["id"]
    assert before["terrain_3d_status"] == "ready"
    assert sync_catalog(rv, c.path)["unchanged"] == ["dem"]
    source_bytes = dem.read_bytes()
    c = ProductCatalog(tmp_path / "products.json")
    c.add_raster("dem", dem, water_level=176.0, limits=[100.0, 300.0])
    c.compare("dem", "dem")
    c.publish()
    r2 = sync_catalog(rv, c.path)
    assert r2["added"] == ["dem"], r2
    after = json.loads((tmp_path / "viz/review.json").read_text())["areas"]["default"][
        "products"
    ]["dem"]
    assert after["water_level"] == 176
    assert before_png != (tmp_path / "viz" / after["image"]).read_bytes()
    assert before_scene != after["terrain_scene"]["id"]
    assert before["legend"]["ramp"] != after["legend"]["ramp"]
    assert dem.read_bytes() == source_bytes
    script = (tmp_path / "viz" / after["terrain_scene"]["script"]).read_text()
    scene = json.loads(script.partition("]=")[2].removesuffix(";\n"))
    assert scene["water_level"] == 176
    # 3D vertex palette is the same threshold-aware elevation_rgba function.
    colors = np.frombuffer(base64.b64decode(scene["colors"]), dtype="uint8").reshape(
        -1, 3
    )
    assert len(colors) == scene["vertex_count"]
    assert np.any(colors[:, 2] > colors[:, 0])
    assert np.any(colors[:, 1] > colors[:, 2])


def test_cli_directory_and_catalog_threshold(tmp_path):
    dem = _raster(tmp_path / "dem.tif")
    root = Path(__file__).resolve().parents[3]
    cp = subprocess.run(
        [
            sys.executable,
            "-m",
            "globato.viz.raster_viz.catalog_cli",
            "--out",
            str(tmp_path / "products.json"),
            "--raster",
            f"dem={dem}",
            "--water-level",
            "176.4",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0, cp.stderr
    raw = json.loads((tmp_path / "products.json").read_text())
    assert raw["products"][0]["water_level"] == 176.4
    cp = subprocess.run(
        [
            sys.executable,
            "-m",
            "globato.viz.raster_viz.cli",
            str(tmp_path),
            "--out",
            str(tmp_path / "directory_viz"),
            "--no-open",
            "--water-level",
            "176.4",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0, cp.stderr
    rendered = json.loads((tmp_path / "directory_viz/review.json").read_text())
    assert (
        next(iter(rendered["areas"]["default"]["products"].values()))["water_level"]
        == 176.4
    )


def test_rgb_cannot_be_default_dem_comparison(tmp_path):
    dem = _raster(tmp_path / "dem.tif")
    cat = ProductCatalog(tmp_path / "products.json")
    cat.add_raster("dem", dem)
    cat.add_raster("rgb", dem, style="rgb")
    with pytest.raises(ValueError, match="Comparison"):
        cat.compare("dem", "rgb")
    cat.compare("dem", "dem").publish()
    data = json.loads(cat.path.read_text())
    data["defaults"]["default"] = ["rgb", "dem"]
    cat.path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Default comparison"):
        read_catalog(cat.path)


def test_cli_fallback_for_manifest_and_reuse_output(tmp_path):
    dem = _raster(tmp_path / "dem.tif")
    cat = ProductCatalog(tmp_path / "products.json")
    cat.add_raster("dem", dem)
    cat.compare("dem", "dem").publish()
    root = Path(__file__).resolve().parents[3]
    out = tmp_path / "viz"

    def render(level):
        cp = subprocess.run(
            [
                sys.executable,
                "-m",
                "globato.viz.raster_viz.cli",
                "--manifest",
                str(cat.path),
                "--out",
                str(out),
                "--no-open",
                "--water-level",
                str(level),
            ],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert cp.returncode == 0, cp.stderr
        return json.loads((out / "review.json").read_text())["areas"]["default"][
            "products"
        ]["dem"]

    first = render(0)
    second = render(175.5)
    assert first["water_level"] == 0
    assert second["water_level"] == 175.5
    assert first["legend"]["ramp"] != second["legend"]["ramp"]
    assert first["limits"] == second["limits"]
