"""Test the neutral catalog interface used by Coast Mask and Stack QC.

No dependency on those projects, their scientific engines, or experiment runners.
"""

import json

import numpy as np
import rasterio
from rasterio.transform import from_origin

from globato.viz.raster_viz.manifest import read_catalog
from globato.viz.raster_viz.validate import check_catalog
from globato.viz.raster_viz.catalog import ProductCatalog


def tif(path, count=1, crs="EPSG:5498"):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=count,
        dtype="float32",
        crs=crs,
        transform=from_origin(-118.2, 34.2, 0.001, 0.001),
        nodata=-9999,
    ) as ds:
        for b in range(1, count + 1):
            ds.write(np.ones((8, 8), "float32") * b, b)
    return path


def test_standard_coast_mask_pair(tmp_path):
    before, after = tif(tmp_path / "before.tif"), tif(tmp_path / "after.tif")
    c = ProductCatalog(tmp_path / "products.json", title="Coast Mask")
    c.add_raster("before", before, stage="before", group="participant_elevation")
    c.add_raster("after", after, stage="after", group="participant_elevation")
    c.compare("before", "after")
    path = c.publish()
    data, _ = read_catalog(path)
    assert data["defaults"] == {"default": ["before", "after"]}
    assert check_catalog(path)["ok"]


def test_standard_stack_qc_four_products_without_fabricated_datum(tmp_path):
    c = ProductCatalog(tmp_path / "products.json", title="Stack QC")
    for key, group in [
        ("before_measured", "qc_measured_elevation"),
        ("after_measured", "qc_measured_elevation"),
        ("before_interpolated", "qc_working_elevation"),
        ("after_interpolated", "final_combined_elevation"),
    ]:
        c.add_raster(
            key,
            tif(tmp_path / f"{key}.tif"),
            group=group,
            metadata={"scope": "QC context"},
        )
    c.compare("before_measured", "after_measured")
    data, groups = read_catalog(c.publish())
    assert len(groups) == 3
    assert data["defaults"]["default"] == ["before_measured", "after_measured"]
    assert all("vertical_datum" not in p.get("metadata", {}) for p in data["products"])
    assert check_catalog(tmp_path / "products.json")["ok"]


def test_default_group_matches_style(tmp_path):
    p = tif(tmp_path / "a.tif")
    c = ProductCatalog(tmp_path / "products.json")
    c.add_raster("elevation", p)
    c.add_raster("difference", p, style="difference")
    data, groups = read_catalog(c.publish())
    assert data["products"][1]["group"] == "difference"
    assert len(groups) == 2


def test_checker_reports_missing_and_bad_bands(tmp_path):
    p = tif(tmp_path / "a.tif")
    c = ProductCatalog(tmp_path / "products.json")
    c.add_raster("good", p)
    c.publish()
    data = json.loads((tmp_path / "products.json").read_text())
    data["products"][0]["band"] = 2
    (tmp_path / "products.json").write_text(json.dumps(data))
    assert not check_catalog(tmp_path / "products.json")["ok"]
    data["products"][0]["band"] = 1
    data["products"][0]["path"] = "missing.tif"
    (tmp_path / "products.json").write_text(json.dumps(data))
    assert not check_catalog(tmp_path / "products.json")["ok"]
    data["running"] = True
    (tmp_path / "products.json").write_text(json.dumps(data))
    report = check_catalog(tmp_path / "products.json", allow_pending=True)
    assert report["ok"] and report["warnings"]


def test_checker_rejects_raw_fusion_state(tmp_path):
    p = tif(tmp_path / "state.tif")
    with rasterio.open(p, "r+") as ds:
        ds.update_tags(GLOBATO_DATATYPE="FUSION_STATE")
    c = ProductCatalog(tmp_path / "products.json")
    c.add_raster("raw", p)
    c.publish()
    assert "FusionState" in "\n".join(
        check_catalog(tmp_path / "products.json")["errors"]
    )
