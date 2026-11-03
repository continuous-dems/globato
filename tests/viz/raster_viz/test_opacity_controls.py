"""Independent RGB transparency and terrain opacity controls."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
VIEWER = ROOT / "src/globato/viz/raster_viz" / "viewer.html"


def test_rgb_controls_without_browser():
    """Run the actual viewer control function under Node, without a DOM package."""
    node = shutil.which("node")
    if not node:
        pytest.skip(
            "Node is not installed; optional browser regression covers the same behavior"
        )
    html = VIEWER.read_text()
    assert 'class="rgb-controls" hidden' in html
    assert 'class="rgb-opacity-control" hidden' in html
    assert "drawBackground3d(p,M)" in html, "3D background must be an independent plane"
    begin = html.index("function demAlpha(p){")
    end = html.index("// file:// compatible", begin)
    function = html[begin:end]
    script = (
        r"""
'use strict';
let products = {dem:{style:'elevation'}}, mode='2d';
function prods(){return products;}
function $(selector){if(selector==='#viewmode')return {value:mode};throw Error(selector);}
"""
        + function
        + r"""
const p={rgbControls:{hidden:false},opacityControl:{hidden:false},rgbTransparencyControl:{hidden:false},bgSelect:{value:''},select:{value:'dem'},opacity:{value:'70'},bgReady:false};
const assert=(condition,name)=>{if(!condition)throw Error(name)};
updateBackgroundControls(p);
assert(demAlpha(p)===1,'no RGB must render DEM fully opaque');
assert(p.rgbControls.hidden && p.opacityControl.hidden && p.rgbTransparencyControl.hidden,'no RGB means no DEM opacity control');
products.rgb={style:'rgb'};
updateBackgroundControls(p);
assert(!p.rgbControls.hidden && p.rgbTransparencyControl.hidden && p.opacityControl.hidden,'background None');
p.bgSelect.value='rgb';updateBackgroundControls(p);
assert(demAlpha(p)===1,'unloaded RGB must not dim DEM');
assert(p.rgbTransparencyControl.hidden && p.opacityControl.hidden,'background not loaded');
p.bgReady=true;updateBackgroundControls(p);
assert(demAlpha(p)===.7,'RGB loaded restores selected DEM alpha');
assert(!p.opacityControl.hidden && !p.rgbTransparencyControl.hidden,'both loaded 2D controls');
mode='3d';updateBackgroundControls(p);
assert(!p.opacityControl.hidden && !p.rgbTransparencyControl.hidden,'both loaded 3D controls');
p.bgReady=false;updateBackgroundControls(p);
assert(demAlpha(p)===1,'RGB failure must not dim DEM');
assert(p.rgbTransparencyControl.hidden && p.opacityControl.hidden,'missing RGB preview hides DEM opacity');
products={dem:{style:'elevation'}};updateBackgroundControls(p);
assert(demAlpha(p)===1,'other AOI without RGB resets DEM draw alpha');
assert(p.rgbControls.hidden && p.opacityControl.hidden && p.rgbTransparencyControl.hidden,'area without RGB');
console.log('PASS: RGB opacity visibility states');
"""
    )
    result = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
