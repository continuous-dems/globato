"""Optional browser-level regression for contextual RGB and opacity controls.

Runs only if Playwright and a Chromium executable are available. Browser may be
installed outside the CI environment; the dependency-free Node test is primary.
"""

import io
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]


def test_rgb_opacity_visibility_in_browser():
    playwright = pytest.importorskip("playwright.sync_api")
    chromium = shutil.which("chromium") or shutil.which("chromium-browser")
    if not chromium:
        pytest.skip("Chromium not installed")
    img = io.BytesIO()
    Image.new("RGBA", (2, 2), (220, 90, 20, 255)).save(img, format="PNG")
    raster = {
        "label": "DEM",
        "style": "elevation",
        "image": "images/dem.png",
        "bounds": [0, 0, 100, 100],
        "preview_shape": [2, 2],
        "limits": [0, 20],
        "group": "elevation",
    }
    rgb = {
        "label": "RGB",
        "style": "rgb",
        "image": "images/rgb.png",
        "bounds": [0, 0, 100, 100],
        "preview_shape": [2, 2],
        "group": "imagery",
    }
    missing_rgb = {**rgb, "image": "images/missing.png"}
    data = {
        "title": "Opacity regression",
        "updated_epoch": 100,
        "enable_3d": True,
        "events": [],
        "areas": {
            "dem_only": {"products": {"dem": raster}, "groups": {}, "vectors": {}},
            "with_rgb": {
                "products": {"dem": raster, "rgb": rgb},
                "groups": {},
                "vectors": {},
            },
            "missing_rgb": {
                "products": {"dem": raster, "rgb": missing_rgb},
                "groups": {},
                "vectors": {},
            },
        },
    }
    html = (
        (ROOT / "src/globato/viz/raster_viz" / "viewer.html")
        .read_text()
        .replace("__REVIEW_DATA__", json.dumps(data))
    )
    # Chromium's sandbox may block local file:// navigation; keep this test offline
    # by serving the two synthetic images through browser request interception.
    html = html.replace("<title>", '<base href="https://viewer.invalid/"><title>', 1)
    try:
        with playwright.sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=chromium,
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
                timeout=10000,
            )
            try:
                page = browser.new_page()
                errors = []
                page.on("pageerror", lambda exc: errors.append(str(exc)))

                def image_response(route):
                    if "missing.png" in route.request.url:
                        route.fulfill(status=404, body="missing")
                    else:
                        route.fulfill(
                            status=200, body=img.getvalue(), content_type="image/png"
                        )

                page.route("**/images/*.png*", image_response)
                page.set_content(html, wait_until="load", timeout=10000)
                opacity = page.locator(".rgb-opacity-control").first
                controls = page.locator(".rgb-controls").first
                assert not controls.is_visible() and not opacity.is_visible()
                page.locator("#area").select_option("with_rgb")
                page.wait_for_function("panes[0].bgReady", timeout=5000)
                assert controls.is_visible() and opacity.is_visible()
                assert page.locator(".rgb-transparency-control").first.is_visible()
                assert page.locator(".pane").first.locator(
                    "> label:first-child > select > option"
                ).all_text_contents() == ["DEM"]
                page.locator(".rgb-transparency").first.fill("100")
                assert (
                    page.locator(".rgb-transparency-value").first.inner_text() == "100%"
                )
                page.locator(".rgb-transparency").first.fill("0")
                page.locator(".rgb-select").first.select_option("")
                assert not page.locator(".rgb-transparency-control").first.is_visible()
                page.locator(".rgb-select").first.select_option("rgb")
                page.wait_for_function("panes[0].bgReady", timeout=5000)
                page.locator("#viewmode").select_option("3d")
                assert (
                    opacity.is_visible()
                    and page.locator(".rgb-transparency-control").first.is_visible()
                )
                page.locator("#viewmode").select_option("2d")
                page.locator("#area").select_option("missing_rgb")
                page.wait_for_function("!panes[0].bgReady", timeout=5000)
                assert (
                    controls.is_visible()
                    and not page.locator(".rgb-transparency-control").first.is_visible()
                )
                page.locator("#area").select_option("dem_only")
                assert not controls.is_visible() and not opacity.is_visible()
                assert not errors, errors
            finally:
                browser.close()
    except (OSError, playwright.Error) as exc:
        pytest.skip(f"Browser unavailable in this environment: {exc}")
