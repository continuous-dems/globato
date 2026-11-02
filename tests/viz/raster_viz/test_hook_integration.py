"""Installed CLI and producer contracts shared by unrelated processing hooks."""

import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from globato.viz.raster_viz.catalog import ProductCatalog
from globato.viz.raster_viz.cli import main
from globato.viz.raster_viz.manifest import sync_catalog
from globato.viz.raster_viz.review import Review
from globato.viz.raster_viz.validate import check_catalog


def raster(path):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=12,
        height=10,
        count=2,
        dtype="float32",
        crs="EPSG:32610",
        transform=from_origin(500000, 4500000, 5, 5),
        nodata=-9999,
    ) as dst:
        dst.write(np.arange(120, dtype="float32").reshape(10, 12), 1)
        dst.write(np.ones((10, 12), dtype="float32"), 2)
    return path


def layers(path):
    import pyogrio
    import geopandas as gpd
    from shapely.geometry import box

    for i, name in enumerate(("first", "selected")):
        gpd.GeoDataFrame(
            {"name": [name]},
            geometry=[box(500000 + i * 50, 4499950, 500020 + i * 50, 4499970)],
            crs="EPSG:32610",
        ).to_file(path, layer=name, engine="pyogrio", driver="GPKG")
    assert len(pyogrio.list_layers(path)) == 2
    return path


def test_installed_commands_and_html_resource():
    from globato.viz.raster_viz import review

    commands = {
        e.name: e.value for e in importlib.metadata.distribution("globato").entry_points
    }
    assert commands["raster-viz"] == "globato.viz.raster_viz.cli:main"
    assert commands["raster-viz-catalog"] == "globato.viz.raster_viz.catalog_cli:main"
    assert commands["raster-viz-check"] == "globato.viz.raster_viz.validate:main"
    assert Path(review.__file__).with_name("viewer.html").is_file()


def test_publisher_does_not_import_renderer(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from globato.viz.raster_viz import ProductCatalog; "
            "import sys; assert 'globato.viz.raster_viz.review' not in sys.modules",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_invalid_publish_preserves_previous_catalog(tmp_path):
    c = ProductCatalog(tmp_path / "products.json")
    c.add_raster("a", raster(tmp_path / "a.tif"))
    c.publish()
    previous = c.path.read_bytes()
    c.add_raster("b", tmp_path / "a.tif", style="difference", group="elevation")
    with pytest.raises(ValueError, match="one rendering style"):
        c.publish()
    assert c.path.read_bytes() == previous


@pytest.mark.parametrize("legacy", [False, True])
def test_vector_layer_selection_survives_publication_and_render(tmp_path, legacy):
    c = ProductCatalog(tmp_path / "products.json")
    gpkg = layers(tmp_path / "boundaries.gpkg")
    opts = {"metadata": {"layer": "selected"}} if legacy else {"layer": "selected"}
    c.add_vector("outline", gpkg, **opts)
    c.publish()
    assert check_catalog(c.path)["ok"]
    rv = Review(tmp_path / "viz")
    result = sync_catalog(rv, c.path)
    assert not result["failed"]
    record = rv._load()["areas"]["default"]["vectors"]["outline"]
    assert record["layer"] == "selected"
    from pyproj import Transformer

    selected_x = Transformer.from_crs(32610, 3857, always_xy=True).transform(
        500050, 4499950
    )[0]
    coordinates = record["geometry"][0]["coordinates"][0]
    assert min(x for x, y in coordinates) == pytest.approx(selected_x)


def test_vector_preflight_reads_actual_layer(tmp_path):
    c = ProductCatalog(tmp_path / "products.json")
    c.add_vector("outline", layers(tmp_path / "boundaries.gpkg"), layer="missing")
    c.publish()
    report = check_catalog(c.path)
    assert not report["ok"]
    assert "cannot read vector layer" in " ".join(report["errors"])
    (tmp_path / "boundaries.gpkg").write_text("corrupt")
    assert not check_catalog(c.path)["ok"]


def test_paused_catalog_does_not_read_rasters(tmp_path, monkeypatch):
    c = ProductCatalog(tmp_path / "products.json")
    c.add_raster("a", raster(tmp_path / "a.tif"))
    c.publish(running=True, rendering_enabled=False)
    rv = Review(tmp_path / "viz")

    def forbidden(*args, **kwargs):
        pytest.fail("Paused catalog accessed a scientific source")

    monkeypatch.setattr(rasterio, "open", forbidden)
    report = sync_catalog(rv, c.path)
    assert report["rendering_paused"] and not report["added"]


@pytest.mark.parametrize("watch", [False, True])
def test_completed_missing_product_fails_cli(tmp_path, watch):
    c = ProductCatalog(tmp_path / "products.json")
    c.add_raster("a", raster(tmp_path / "a.tif"))
    c.publish()
    (tmp_path / "a.tif").unlink()
    args = [str(c.path), "--out", str(tmp_path / "viz"), "--no-open"]
    if watch:
        args.append("--watch")
    assert main(args) == 2


def test_pending_running_catalog_can_be_rendered_once(tmp_path):
    c = ProductCatalog(tmp_path / "products.json")
    c.add_raster("a", raster(tmp_path / "a.tif"))
    c.publish(running=True)
    (tmp_path / "a.tif").unlink()
    assert main([str(c.path), "--out", str(tmp_path / "viz"), "--no-open"]) == 0


def test_raw_fusion_state_cannot_be_rendered_as_elevation(tmp_path):
    path = raster(tmp_path / "state.tif")
    with rasterio.open(path, "r+") as dst:
        dst.update_tags(GLOBATO_DATATYPE="FUSION_STATE")
    c = ProductCatalog(tmp_path / "products.json")
    c.add_raster("raw", path)
    c.publish()
    assert main([str(c.path), "--out", str(tmp_path / "viz"), "--no-open"]) == 2
    data = json.loads((tmp_path / "viz/review.json").read_text())
    assert not data["areas"]["default"]["products"]


def test_cli_catalog_supports_band_layer_and_visibility(tmp_path):
    from globato.viz.raster_viz.catalog_cli import main as publish

    source = raster(tmp_path / "state.tif")
    gpkg = layers(tmp_path / "boundaries.gpkg")
    catalog = tmp_path / "catalog/products.json"
    assert (
        publish(
            [
                "--out",
                str(catalog),
                "--scalar",
                f"count={source}",
                "--band",
                "count=2",
                "--vector",
                f"boundary={gpkg}",
                "--layer",
                "boundary=selected",
                "--hidden",
                "boundary",
            ]
        )
        == 0
    )
    assert check_catalog(catalog)["ok"]
    data = json.loads(catalog.read_text())
    assert data["products"][0]["band"] == 2
    assert data["products"][1]["display"] == {"visible": False}
    assert data["products"][1]["vector_layer"] == "selected"


def test_relative_catalog_survives_moving_whole_run(tmp_path, monkeypatch):
    import shutil

    run = tmp_path / "run"
    run.mkdir()
    c = ProductCatalog(run / "products.json")
    c.add_raster("before", raster(run / "before.tif"))
    c.add_raster("after", raster(run / "after.tif"))
    c.compare("before", "after").publish()
    moved = tmp_path / "moved"
    shutil.move(run, moved)
    monkeypatch.chdir(tmp_path)
    assert check_catalog(moved / "products.json")["ok"]
    assert (
        main(
            [str(moved / "products.json"), "--out", str(tmp_path / "viz"), "--no-open"]
        )
        == 0
    )


def test_legacy_vector_layer_label_is_not_a_dataset_selector(tmp_path):
    vector = tmp_path / "topology.geojson"
    vector.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [-118.2, 34.1],
                                    [-118.19, 34.1],
                                    [-118.19, 34.09],
                                    [-118.2, 34.1],
                                ]
                            ],
                        },
                    }
                ],
            }
        )
    )
    catalog = tmp_path / "products.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "products": [
                    {
                        "type": "vector",
                        "key": "topology",
                        "path": vector.name,
                        "layer": "coastal_topology",
                        "metadata": {"role": "prepared_coastline"},
                    }
                ],
            }
        )
    )
    assert check_catalog(catalog)["ok"]
    result = sync_catalog(Review(tmp_path / "viz"), catalog)
    assert result["added"] == ["topology"] and not result["failed"]


def test_catalog_default_output_is_next_to_catalog(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    c = ProductCatalog(run / "products.json")
    c.add_raster("a", raster(run / "a.tif")).publish()
    monkeypatch.chdir(tmp_path)
    assert main([str(c.path), "--no-open"]) == 0
    assert (run / "viz/viz.html").is_file()
    assert not (tmp_path / "viz").exists()
