"""Vector style/visibility contract and interactive browser behavior."""

import io
import json
import shutil
from pathlib import Path

import pytest
import rasterio
import numpy as np
from PIL import Image
from rasterio.transform import from_origin

from globato.viz.raster_viz.manifest import read_catalog, sync_catalog
from globato.viz.raster_viz.review import Review
from globato.viz.raster_viz.catalog import ProductCatalog

ROOT = Path(__file__).resolve().parents[3]


def raster(path):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=3,
        height=3,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(0, 100, 10, 10),
    ) as ds:
        ds.write(np.ones((3, 3), dtype="float32"), 1)
    return path


def vectors(path):
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[0, 0], [50, 50]],
                        },
                        "properties": {},
                    }
                ],
            }
        )
    )
    return path


def test_catalog_display_settings_round_trip(tmp_path):
    cat = ProductCatalog(tmp_path / "products.json")
    cat.add_raster("a", raster(tmp_path / "a.tif"))
    cat.compare("a", "a")
    displays = [
        {"visible": False, "color": "#2563eb", "width": 1.8, "dash": [7, 4]},
        {"visible": True, "color": "#dd7314", "width": 2.2, "dash": []},
        {"visible": False, "color": "#7c3aed", "width": 1.8, "dash": [2, 4]},
    ]
    for i, d in enumerate(displays):
        cat.add_vector(
            "v" + str(i),
            vectors(tmp_path / f"v{i}.geojson"),
            label=f"Boundary {i}",
            display=d,
        )
    data, _ = read_catalog(cat.publish())
    assert [x["display"] for x in data["products"][1:]] == displays
    review = Review(tmp_path / "viz", open_browser=False)
    result = sync_catalog(review, cat.path)
    assert not result["failed"], result
    state = review._load()
    assert [
        state["areas"]["default"]["vectors"]["v" + str(i)]["display"] for i in range(3)
    ] == displays
    # Style-only catalog edits must invalidate only the changed vector preview.
    data["products"][1]["display"]["color"] = "#aa0011"
    cat.path.write_text(json.dumps(data))
    changed = sync_catalog(review, cat.path)
    assert "v0" in changed["added"] and "v1" in changed["unchanged"]


@pytest.mark.parametrize(
    "display",
    [
        {"visible": "yes"},
        {"color": "red"},
        {"color": "#12FG00"},
        {"width": 0},
        {"width": float("nan")},
        {"dash": [2, -1]},
        {"dash": ["2", 4]},
        {"dash": [1, 2, 3, 4, 5]},
        {"extra": 1},
        [],
    ],
)
def test_invalid_vector_display_rejected(tmp_path, display):
    data = {
        "schema_version": 1,
        "products": [
            {"key": "v", "type": "vector", "path": "v.geojson", "display": display}
        ],
    }
    f = tmp_path / "products.json"
    f.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        read_catalog(f)


def test_raster_display_rejected(tmp_path):
    f = tmp_path / "products.json"
    f.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "products": [
                    {"key": "x", "path": "x.tif", "display": {"visible": False}}
                ],
            }
        )
    )
    with pytest.raises(ValueError):
        read_catalog(f)


def test_vector_control_browser():
    pw = pytest.importorskip("playwright.sync_api")
    chromium = shutil.which("chromium") or shutil.which("chromium-browser")
    if not chromium:
        pytest.skip("Chromium not installed")
    raster_product = {
        "label": "Elevation",
        "style": "elevation",
        "image": "images/dem.png",
        "bounds": [0, 0, 100, 100],
        "preview_shape": [2, 2],
        "limits": [0, 20],
        "group": "elevation",
    }
    sample_geometry = [{"type": "LineString", "coordinates": [[10, 10], [90, 90]]}]
    displays = [
        {"visible": False, "color": "#2563eb", "dash": [7, 4]},
        {"visible": True, "color": "#dd7314"},
        {"visible": False, "color": "#7c3aed", "dash": [2, 4]},
    ]
    v = {
        key: {"label": label, "display": d, "geometry": sample_geometry}
        for key, label, d in zip(
            ["original", "retention", "topology"],
            ["Original water", "Source exclusion", "Interpolation topology"],
            displays,
        )
    }
    data = {
        "title": "Coast Mask",
        "updated_epoch": 100,
        "enable_3d": True,
        "events": [],
        "areas": {
            "malibu": {
                "products": {"dem": raster_product},
                "groups": {},
                "vectors": v,
                "defaults": ["dem", "dem"],
            },
            "raster_only": {
                "products": {"dem": raster_product},
                "groups": {},
                "vectors": {},
            },
            "other": {
                "products": {"dem": raster_product},
                "groups": {},
                "vectors": {"new": {"label": "Other", "geometry": sample_geometry}},
            },
        },
    }
    html = (
        (ROOT / "src/globato/viz/raster_viz/viewer.html")
        .read_text()
        .replace("__REVIEW_DATA__", json.dumps(data))
    )
    b = io.BytesIO()
    Image.new("RGBA", (2, 2), (155, 115, 68, 255)).save(b, format="PNG")
    try:
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=chromium,
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
                timeout=10000,
            )
            try:
                page = browser.new_page()
                errors = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.route(
                    "**/images/*.png*",
                    lambda route: route.fulfill(
                        status=200, body=b.getvalue(), content_type="image/png"
                    ),
                )
                page.set_content(html, wait_until="load", timeout=10000)
                box = page.locator("#vector-controls")
                assert box.is_visible()
                assert (
                    page.locator("#vector-summary").inner_text() == "Boundaries (1/3)"
                )
                inputs = page.locator("#vector-menu input")
                assert [inputs.nth(i).is_checked() for i in range(3)] == [
                    False,
                    True,
                    False,
                ]
                # Separate swatches and lines: inspector should show styling per layer.
                assert (
                    page.locator(".vector-swatch")
                    .nth(0)
                    .get_attribute("style")
                    .find("rgb(37, 99, 235)")
                    >= 0
                )
                assert (
                    page.locator(".vector-swatch")
                    .nth(1)
                    .get_attribute("style")
                    .find("rgb(221, 115, 20)")
                    >= 0
                )
                page.locator("#vector-summary").click()
                inputs.nth(0).check()
                inputs.nth(1).uncheck()
                assert (
                    page.locator("#vector-summary").inner_text() == "Boundaries (1/3)"
                )
                page.locator("#area").select_option("raster_only")
                assert not box.is_visible()
                page.locator("#area").select_option("other")
                assert box.is_visible()
                assert (
                    page.locator("#vector-summary").inner_text() == "Boundaries (1/1)"
                )
                page.locator("#area").select_option("malibu")
                assert [inputs.nth(i).is_checked() for i in range(3)] == [
                    True,
                    False,
                    False,
                ]
                page.locator("#viewmode").select_option("3d")
                assert not box.is_visible()
                page.locator("#viewmode").select_option("2d")
                assert box.is_visible()
                assert not errors, errors
            finally:
                browser.close()
    except (OSError, pw.Error) as exc:
        pytest.skip(f"Browser unavailable: {exc}")
