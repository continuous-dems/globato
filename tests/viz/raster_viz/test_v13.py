import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from globato.viz.raster_viz.catalog import ProductCatalog
from globato.viz.raster_viz.manifest import read_catalog

ROOT = Path(__file__).resolve().parents[3]


def tif(path, offset=0, mask=False):
    a = np.arange(2400, dtype="float32").reshape(40, 60) / 10 + offset
    if mask:
        a[:10, :] = np.nan
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=40,
        width=60,
        count=1,
        dtype="float32",
        crs="EPSG:32610",
        transform=from_origin(500000, 4500000, 5, 5),
        nodata=-9999.0,
    ) as ds:
        ds.write(np.where(np.isfinite(a), a, -9999).astype("float32"), 1)
    return path


def run_cli(*args, cwd):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "globato.viz.raster_viz", *map(str, args), "--no-open"],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=100,
    )


def test_writer_independent_and_atomic(tmp_path):
    before = tif(tmp_path / "before.tif")
    after = tif(tmp_path / "after.tif", 10)
    cat = ProductCatalog(
        tmp_path / "catalog/products.json", title="Independent workflow"
    )
    cat.add_raster(
        "raw", before, label="Original", stage="source", metadata={"value_units": "m"}
    )
    cat.add_raster("post", after, label="Updated", stage="filtered")
    cat.compare("raw", "post")
    catalog = cat.publish()
    assert not list(catalog.parent.glob("*.tmp"))
    data, groups = read_catalog(catalog)
    assert data["defaults"]["default"] == ["raw", "post"]
    assert groups[("default", "elevation")]["limits"] is None
    assert (catalog.parent / data["products"][0]["path"]).resolve() == before
    with pytest.raises(FileNotFoundError):
        cat.add_raster("missing", tmp_path / "bad.tif")
    with pytest.raises(ValueError):
        cat.add_raster("raw", after)


def test_directory_mode_generates_linked_html(tmp_path):
    tif(tmp_path / "left.tif")
    tif(tmp_path / "right.tif", -20)
    result = run_cli("--out", tmp_path / "viz", cwd=tmp_path)
    assert result.returncode == 0, result.stderr + result.stdout
    html = (tmp_path / "viz/viz.html").read_text()
    assert "color-legend" in html and "Legend" not in result.stderr
    assert '<select id="preset"' not in html and '<select id="sparsemode"' not in html
    assert '<button id="reset">Reset view</button>' in html
    assert "renderModeForPane" in html
    review = json.loads((tmp_path / "viz/review.json").read_text())
    assert len(review["areas"]["default"]["products"]) == 2


def test_manifest_shared_scales_3d(tmp_path):
    a = tif(tmp_path / "a.tif", 0, mask=True)
    b = tif(tmp_path / "b.tif", 100)
    cat = ProductCatalog(tmp_path / "catalog/products.json")
    cat.add_raster("a", a, label="Initial").add_raster(
        "b", b, label="Modified"
    ).compare("a", "b")
    cat.publish()
    result = run_cli(
        "--manifest",
        cat.path,
        "--out",
        tmp_path / "viz",
        "--3d",
        "--mesh-side",
        "60",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    state = json.loads((tmp_path / "viz/review.json").read_text())
    one, two = [state["areas"]["default"]["products"][k] for k in ("a", "b")]
    assert one["limits"] == two["limits"]
    assert one["limits"][1] > 100
    assert (tmp_path / "viz/viz.html").is_file()
    assert (tmp_path / "viz" / one["legend"]["ramp"]).is_file()
    assert one.get("terrain_scene"), one.get("terrain_3d_error")
    assert two.get("terrain_scene"), two.get("terrain_3d_error")


def test_no_producer_hooks_in_release():
    viewer = ROOT / "src/globato/viz/raster_viz"
    assert not (viewer / "hooks.py").exists()
    assert not (viewer / "modifiers.py").exists()
    import ast

    for file in (ROOT / "src/globato/viz/raster_viz").glob("*.py"):
        src = ast.parse(file.read_text())
        for node in ast.walk(src):
            if isinstance(node, ast.Import):
                assert not any(
                    alias.name.startswith(
                        (
                            "fetchez",
                            "globato.qc",
                            "globato.coast_refine",
                            "globato.hooks",
                        )
                    )
                    for alias in node.names
                ), file
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(
                    ("fetchez", "globato.qc", "globato.coast_refine", "globato.hooks")
                ), file


def test_difference_centered_and_categorical_hidden_from_legends(tmp_path):
    before = tif(tmp_path / "a.tif", -100)
    after = tif(tmp_path / "b.tif", 50)
    cat = ProductCatalog(tmp_path / "catalog/products.json")
    cat.add_raster("difference_a", before, style="difference", group="difference")
    cat.add_raster("difference_b", after, style="difference", group="difference")
    cat.add_raster("classes", after, style="categorical", group="classes")
    cat.compare("difference_a", "difference_b")
    cat.publish()
    r = run_cli("--manifest", cat.path, "--out", tmp_path / "viz", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    state = json.loads((tmp_path / "viz/review.json").read_text())["areas"]["default"][
        "products"
    ]
    a, b = state["difference_a"]["limits"], state["difference_b"]["limits"]
    assert a == b and abs(a[0] + a[1]) < 1e-6
    assert state["classes"]["style"] == "categorical"


def test_multiple_areas_and_metadata_work_without_special_imports(tmp_path):
    t = tif(tmp_path / "source.tif")
    c = ProductCatalog(tmp_path / "publish/products.json", title="Independent producer")
    c.add_raster(
        "source",
        t,
        area="a",
        group="heights",
        metadata={"value_units": "m", "vertical_datum": "local"},
    )
    c.add_raster("source", t, area="b", group="heights")
    c.compare("source", "source", area="a")
    result = c.publish()
    data, _ = read_catalog(result)
    assert len(data["products"]) == 2
    assert data["products"][0]["metadata"]["vertical_datum"] == "local"
    assert data["defaults"]["a"] == ["source", "source"]


def test_v12_opaque_fields_remain_compatible(tmp_path):
    tif(tmp_path / "a.tif")
    catalog = tmp_path / "products.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "products": [
                    {
                        "key": "old",
                        "path": "a.tif",
                        "method": "arbitrary_method",
                        "layer": "measured_elevation",
                    }
                ],
            }
        )
    )
    data, _ = read_catalog(catalog)
    assert data["products"][0]["layer"] == "measured_elevation"
    r = run_cli("--manifest", catalog, "--out", tmp_path / "viz", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    published = json.loads((tmp_path / "viz/review.json").read_text())
    assert (
        published["areas"]["default"]["products"]["old"]["metadata"]["method"]
        == "arbitrary_method"
    )


def vector_geojson(path):
    """A tiny real vector overlapping the fixture raster, independent of any producer."""
    from pyproj import Transformer

    to_lonlat = Transformer.from_crs(
        "EPSG:32610", "EPSG:4326", always_xy=True
    ).transform
    xys = [
        (500020, 4499980),
        (500140, 4499980),
        (500140, 4499880),
        (500020, 4499880),
        (500020, 4499980),
    ]
    coordinates = [list(to_lonlat(*xy)) for xy in xys]
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"label": "Example boundary"},
                        "geometry": {"type": "Polygon", "coordinates": [coordinates]},
                    }
                ],
            }
        )
    )
    return path


def test_vector_discovery_and_visible_toggle(tmp_path):
    pytest.importorskip("shapely")
    tif(tmp_path / "left.tif")
    vector_geojson(tmp_path / "boundary.geojson")
    result = run_cli("--out", tmp_path / "viz", cwd=tmp_path)
    assert result.returncode == 0, result.stderr + result.stdout
    state = json.loads((tmp_path / "viz/review.json").read_text())
    vectors = state["areas"]["default"]["vectors"]
    assert len(vectors) == 1
    v = next(iter(vectors.values()))
    assert v["geometry_count"] == 1
    assert v["geometry_crs"] == "EPSG:3857"
    assert v["view"] == "2d overlay"
    html = (tmp_path / "viz/viz.html").read_text()
    assert 'id="vector-controls"' in html
    assert '<input type="checkbox" id="auto" checked>' in html
    assert '<button id="refresh"' in html
    assert "vectorVisibilityByArea" in html and "saved.auto!==false" in html
    assert "vectorIsVisible" in html and "updateVectorControls" in html
    assert "if($('#auto').checked" in html


def test_generic_manifest_vector_writer(tmp_path):
    pytest.importorskip("shapely")
    raster = tif(tmp_path / "terrain.tif")
    vector = vector_geojson(tmp_path / "boundary.geojson")
    cat = ProductCatalog(tmp_path / "catalog/products.json")
    cat.add_raster("terrain", raster)
    cat.add_vector(
        "edge", vector, label="Example shoreline", metadata={"source": "user-provided"}
    )
    cat.compare("terrain", "terrain")
    data, _ = read_catalog(cat.publish())
    assert data["products"][1]["type"] == "vector"
    res = run_cli("--manifest", cat.path, "--out", tmp_path / "viz", cwd=tmp_path)
    assert res.returncode == 0, res.stderr + res.stdout
    review = json.loads((tmp_path / "viz/review.json").read_text())
    overlay = review["areas"]["default"]["vectors"]["edge"]
    assert overlay["label"] == "Example shoreline"
    assert overlay["geometry_count"] == 1
    assert overlay["metadata"]["source"] == "user-provided"
