"""Regression: 3D RGB imagery covers full bounds independent of a sparse DEM.

Runs entirely offline, including Chromium; uses synthetic rasters and no producer
hooks. The 3D image-plane and mesh remain different geometry buffers.
"""

from __future__ import annotations
import json
from pathlib import Path
from urllib.parse import urlsplit
import shutil
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from globato.viz.raster_viz.review import Review
from globato.viz.raster_viz.catalog import ProductCatalog
from globato.viz.raster_viz.manifest import sync_catalog


def test_sparse_dem_full_rgb_webgl(tmp_path):
    pw = pytest.importorskip("playwright.sync_api")
    chromium = shutil.which("chromium") or shutil.which("chromium-browser")
    if not chromium:
        pytest.skip("Chromium unavailable")
    dem, rgb = tmp_path / "sparse_dem.tif", tmp_path / "sentinel_rgb.tif"
    transform = from_origin(500000, 5000100, 5, 5)
    z = np.full((40, 40), -9999, dtype="float32")
    z[15:25, 15:25] = 175
    with rasterio.open(
        dem,
        "w",
        driver="GTiff",
        width=40,
        height=40,
        count=1,
        dtype="float32",
        crs="EPSG:32616",
        transform=transform,
        nodata=-9999.0,
    ) as d:
        d.write(z, 1)
    with rasterio.open(
        rgb,
        "w",
        driver="GTiff",
        width=40,
        height=40,
        count=3,
        dtype="uint8",
        crs="EPSG:32616",
        transform=transform,
        nodata=0,
    ) as d:
        for i, v in enumerate((210, 160, 80), 1):
            d.write(np.full((40, 40), v, dtype="uint8"), i)
    cat = ProductCatalog(tmp_path / "products.json", title="Independent RGB 3D test")
    cat.add_raster("dem", dem, label="Sparse DEM")
    cat.add_raster("rgb", rgb, label="Sentinel RGB", style="rgb")
    cat.compare("dem", "dem")
    cat.publish()
    output = tmp_path / "viz"
    viewer = Review(output, render_3d=True, mesh_side=80)
    result = sync_catalog(viewer, cat.path)
    assert not result["failed"], result
    data = json.loads((output / "review.json").read_text())
    prod = data["areas"]["default"]["products"]
    assert prod["dem"]["terrain_scene"]
    assert prod["rgb"]["style"] == "rgb"
    assert prod["dem"]["preview_valid_cells"] < prod["rgb"]["preview_valid_cells"]
    errors = []
    with pw.sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=chromium,
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--enable-webgl",
                "--enable-unsafe-swiftshader",
                "--ignore-gpu-blocklist",
                "--disable-gpu-sandbox",
                "--use-gl=angle",
                "--use-angle=swiftshader",
            ],
        )
        try:
            page = browser.new_page(viewport={"width": 1450, "height": 900})
            page.on("pageerror", lambda error: errors.append(str(error)))

            def serve(route):
                path = urlsplit(route.request.url).path.lstrip("/")
                file = (output / path).resolve()
                if not file.is_relative_to(output.resolve()) or not file.is_file():
                    return route.fulfill(status=404, body="not found")
                ext = file.suffix.lower()
                mime = {
                    ".html": "text/html",
                    ".js": "text/javascript",
                    ".png": "image/png",
                    ".json": "application/json",
                }.get(ext, "application/octet-stream")
                return route.fulfill(
                    status=200, body=file.read_bytes(), content_type=mime
                )

            page.route("https://viewer.invalid/**", serve)
            html = (
                (output / "viz.html")
                .read_text()
                .replace("<title>", '<base href="https://viewer.invalid/"><title>', 1)
            )
            page.set_content(html, wait_until="load", timeout=20000)
            # Some containerized Chromium builds have WebGL2 disabled by policy.
            if not page.evaluate(
                "!!document.createElement('canvas').getContext('webgl2')"
            ):
                assert page.locator(".pane").first.locator(
                    "> label:first-child > select > option"
                ).all_text_contents() == ["Sparse DEM"]
                pytest.skip(
                    "WebGL2 unavailable in sandbox Chromium; run GPU browser test locally"
                )
            page.locator("#viewmode").select_option("3d")
            try:
                page.wait_for_function(
                    "panes[0].scene && panes[0].bgReady && panes[0].bgGpu && panes[0].bgGpu.planeKey",
                    timeout=12000,
                )
            except Exception as exc:
                state = page.evaluate(
                    """() => ({scene:!!panes[0].scene,bgReady:panes[0].bgReady,gpu:!!panes[0].gpu,gl:!!panes[0].gl,bgGpu:!!panes[0].bgGpu,placeholder:panes[0].placeholder.textContent,caption:panes[0].caption.textContent,errors:[]})"""
                )
                pytest.fail(f"3D browser state: {state} page errors: {errors} ({exc})")
            props = page.evaluate("""() => {
              const p=panes[0],b=prods().rgb.bounds,a=prods().dem.bounds;
              return {mode:document.querySelector('#viewmode').value,
                productLabels:[...p.select.options].map(o=>o.textContent),
                imageryOptions:[...p.bgSelect.options].map(o=>o.textContent),
                bg:!!p.bgGpu,scene:!!p.scene,
                bgBounds:b,demBounds:a,planeKey:p.bgGpu.planeKey,
                rgbOpacityVisible:!p.rgbTransparencyControl.hidden,
                demOpacityVisible:!p.opacityControl.hidden,
                terrainCount:p.gpu.count};
            }""")
            assert props["mode"] == "3d" and props["bg"] and props["scene"]
            assert props["productLabels"] == ["Sparse DEM"]
            assert props["imageryOptions"] == ["None", "Sentinel RGB"]
            assert props["rgbOpacityVisible"] and props["demOpacityVisible"]
            assert props["terrainCount"] > 0
            assert len(props["bgBounds"]) == 4 and props["planeKey"]
            assert not errors, errors
            page.screenshot(path=str(tmp_path / "3d_full_rgb.png"))
            # The independent texture remains present when DEM completely hidden.
            page.locator(".rgb-opacity").first.fill("0")
            assert page.evaluate("Number(panes[0].opacity.value)") == 0
            assert page.evaluate("!!panes[0].bgGpu.planeKey")
            page.locator(".rgb-transparency").first.fill("100")
            assert page.evaluate("Number(panes[0].rgbTransparency.value)") == 100
            page.locator(".rgb-transparency").first.fill("0")
            page.locator(".rgb-opacity").first.fill("100")
            assert not errors, errors
        finally:
            browser.close()


def test_3d_rgb_plane_geometry_independent_of_dem(tmp_path):
    """Execute actual renderer logic with a fake GL, no GPU or browser required."""
    import subprocess
    import shutil

    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    html = (
        Path(__file__).resolve().parents[3] / "src/globato/viz/raster_viz/viewer.html"
    ).read_text()
    begin = html.index("function backgroundFloor(p){")
    end = html.index("function renderModeForPane(p){", begin)
    code = html[begin:end]
    script = (
        r"""
'use strict';
let uploaded=null,drawCount=0,opacity=null;
const products={rgb:{style:'rgb',bounds:[0,0,400,300]},dem:{style:'elevation',bounds:[100,100,120,120],terrain_scene:{data_elevation_range:[170,180],vertical_unit_to_m:1}}};
function prods(){return products;}
function productFrame(p){return {mercator_to_local:[1,0,0,1],center_3857:[0,0],origin_xy_m:[0,0]};}
function mercatorToLocal(frame,x,y){return [x,y]};
function backgroundShaderProgram(gl){return {};}
function $(key){if(key==='#vertical')return {value:'1'};throw Error(key)}
const gl=new Proxy({
    getParameter:()=>4096,
    getAttribLocation:(p,k)=>k==='position'?0:1,
    bufferData:(target,data)=>{if(data instanceof Float32Array&&data.length===18)uploaded=data;},
    uniform1f:(location,val)=>{if(location==='opacity')opacity=val;},
    getUniformLocation:(p,name)=>name,
    drawArrays:(mode,start,count)=>{drawCount=count;},
    createVertexArray:()=>({}),createBuffer:()=>({}),createTexture:()=>({}),
}, {get:(target,key)=>key in target?target[key]:typeof key==='string'&&key===key.toUpperCase()?1:(()=>{})});
const p={gl,scene:{id:'terrain',data_elevation_range:[170,180]},bgReady:true,
    bgImage:{naturalWidth:10,naturalHeight:10},rgbTextureDirty:true,
    bgSelect:{value:'rgb'},rgbTransparency:{value:25}};
"""
        + code
        + r"""
const assert=(v,msg)=>{if(!v)throw Error(msg)};
assert(backgroundFloor(p)<170,'imagery plane must be below DEM');
drawBackground3d(p,new Array(16).fill(0));
assert(drawCount===6,'2 triangle background plane');
assert(uploaded&&uploaded.length===18,'full georeferenced rectangle');
const xy=[];for(let i=0;i<uploaded.length;i+=3)xy.push(uploaded[i],uploaded[i+1]);
assert(Math.min(...xy)===0&&Math.max(...xy)===400,'full RGB x extent not clipped to DEM');
assert(opacity===.75,'RGB transparency is independent alpha');
assert(!('rgb_colors' in p.scene),'background is not a terrain vertex color');
console.log('PASS: 3D RGB full-footprint plane and alpha');
"""
    )
    result = subprocess.run(
        [node, "-e", script], text=True, capture_output=True, timeout=10
    )
    assert result.returncode == 0, result.stderr


def test_sparse_3d_autofit_includes_imagery_footprint():
    html = (
        Path(__file__).resolve().parents[3] / "src/globato/viz/raster_viz/viewer.html"
    ).read_text()
    start = html.index("function fitValid3d(){")
    end = html.index("function sync3d(p)", start)
    snippet = html[start:end]
    assert "prods()[p.bgSelect.value]" in snippet
    assert (
        "merc.push([bg.bounds[0],bg.bounds[1]],[bg.bounds[2],bg.bounds[3]])" in snippet
    )
