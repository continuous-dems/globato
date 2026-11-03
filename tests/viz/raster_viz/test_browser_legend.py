"""Browser regression: captions are scientific labels, not renderer diagnostics."""

import json
import shutil
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_legend_headings_in_browser():
    playwright = pytest.importorskip("playwright.sync_api")
    executable = shutil.which("chromium") or shutil.which("chromium-browser")
    if not executable:
        pytest.skip("Chromium not available")
    base = {
        "label": "Elevation",
        "style": "elevation",
        "image": "images/missing.png",
        "bounds": [0, 0, 100, 100],
        "preview_shape": [2, 2],
        "preview_valid_cells": 4,
        "limits": [-10, 100],
        "legend": {
            "title": "Elevation",
            "ramp": "legends/terrain.png",
            "palette": "terrain",
            "units": "m",
            "datum": "NAVD88",
        },
    }
    delta = {
        **base,
        "style": "difference",
        "legend": {
            "title": "Elevation difference",
            "ramp": "legends/RdBu_r.png",
            "units": "m",
            "datum": "",
        },
        "metadata": {"vertical_datum": "NAVD88"},
    }
    data = {
        "title": "Legend display",
        "updated_epoch": 100,
        "enable_3d": False,
        "events": [],
        "legend_ramps": {},
        "areas": {
            "default": {
                "products": {"before": base, "delta": delta},
                "defaults": ["before", "delta"],
                "groups": {},
                "vectors": {},
            }
        },
    }
    html = (
        (ROOT / "src/globato/viz/raster_viz/viewer.html")
        .read_text()
        .replace("__REVIEW_DATA__", json.dumps(data))
    )
    try:
        with playwright.sync_playwright() as pw:
            browser = pw.chromium.launch(
                executable_path=executable,
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
                timeout=10000,
            )
            try:
                page = browser.new_page()
                errors = []
                page.on("pageerror", lambda exc: errors.append(str(exc)))
                page.set_content(html, wait_until="load", timeout=10000)
                assert page.locator(".legend-title").all_text_contents() == [
                    "Elevation (m) · NAVD88",
                    "Elevation difference (m)",
                ]
                assert (
                    "Blue below 0 m"
                    in page.locator(".legend-explainer").first.inner_text()
                )
                assert not errors, errors
            finally:
                browser.close()
    except (OSError, playwright.Error) as exc:
        pytest.skip(f"Browser execution unavailable: {exc}")
