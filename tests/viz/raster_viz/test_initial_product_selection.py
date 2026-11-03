"""Keep an available raster selected when saved/default keys become stale."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_refill_ignores_unavailable_or_rgb_selection_keys():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is unavailable")
    html = (ROOT / "src/globato/viz/raster_viz/viewer.html").read_text()
    refill = html.split("function refill(", 1)[1].split("let activeMode", 1)[0]
    script = r"""
const assert = require('node:assert/strict');
let D, saved, panes, camera3d = {};
const $ = () => ({value: 'default'});
const prods = () => D.areas.default.products;
const updateVectorControls = () => {};
const selOptions = (select, rows) => {select.value = rows[0]?.[0] ?? ''};
const fit2d = pane => {pane.view = null};
const validCamera = () => false;
const productFrame = () => ({});
const setBackground = () => {};
const load = () => {};
const draw2d = () => {};
__REFILL__
function check(products, defaults, savedPanes, expected) {
  D = {areas: {default: {products, defaults}}}; saved = {panes: savedPanes};
  panes = [0,1].map(() => ({select: {}, bgSelect: {}, opacity: {},
    rgbTransparency: {}, rgbTransparencyLabel: {}, opacityLabel: {}}));
  refill(true);
  assert.deepEqual(panes.map(p => p.select.value), expected);
}
const products = {before: {label:'Before',style:'elevation'},
  after: {label:'After',style:'elevation'},rgb: {label:'RGB',style:'rgb'}};
check(products, ['before','after'], undefined, ['before','after']);
check(products, ['before','after'], [{key:''},{key:''}], ['before','after']);
check(products, ['before','after'], [{key:'removed'},{key:'removed'}], ['before','after']);
check(products, ['before','after'], [{},{key:'rgb'}], ['before','after']);
check(products, ['gone','rgb'], undefined, ['before','after']);
check(products, undefined, [{key:'after'},{key:'before'}], ['after','before']);
check({before:products.before}, ['before','after'], [{key:''},{key:'after'}], ['before','before']);
check({}, undefined, [{key:''},{key:''}], ['','']);
""".replace("__REFILL__", "function refill(" + refill)
    subprocess.run([node, "-e", script], check=True, capture_output=True, text=True)


def test_real_browser_restores_defaults_after_empty_saved_selection(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    from PIL import Image

    Image.new("RGB", (4, 4), (80, 150, 200)).save(tmp_path / "preview.png")
    product = dict(
        label="Before",
        style="elevation",
        image="preview.png",
        bounds=[0, 0, 100, 100],
        preview_shape=[1, 1],
        preview_valid_cells=1,
        limits=[-10, 10],
        group="elevation",
    )
    data = dict(
        title="Startup selection",
        updated_epoch=100,
        enable_3d=False,
        events=[],
        areas={
            "default": dict(
                products={"before": product, "after": dict(product, label="After")},
                defaults=["before", "after"],
                groups={},
                vectors={},
            )
        },
    )
    page_file = tmp_path / "viz.html"
    page_file.write_text(
        (ROOT / "src/globato/viz/raster_viz/viewer.html")
        .read_text()
        .replace("__REVIEW_DATA__", json.dumps(data))
    )
    with playwright.sync_playwright() as pw:
        executable = shutil.which("chromium") or shutil.which("chromium-browser")
        executable = executable or pw.chromium.executable_path
        if not Path(executable).is_file():
            pytest.skip("Chromium is unavailable")
        browser = pw.chromium.launch(
            executable_path=executable,
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        try:
            page = browser.new_page()
            page.add_init_script("""localStorage.setItem(
              'universal_viz_v13:'+location.pathname.replace(/[^/]*$/, ''),
              JSON.stringify({panes:[{key:''},{key:''}]}));""")
            page.goto(page_file.as_uri(), wait_until="load")
            page.wait_for_function("panes.every(p => p.ready)", timeout=10000)
            assert page.locator(".pane > label select").count() == 2
            assert page.evaluate("panes.map(p=>p.select.value)") == ["before", "after"]
            assert all(
                "Waiting for raster products" not in text
                for text in page.locator(".caption").all_text_contents()
            )
        finally:
            browser.close()
