"""Finalization: horizontal-only geometry and native branch identity."""

from types import SimpleNamespace
import numpy as np
import geopandas as gpd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
from pyproj import CRS
from fetchez.entry import dataset_id
from globato.coast_refine.geometry_crs import (
    horizontal_frame,
    operation_record,
    science_horizontal_view,
)
from globato.hooks.sinks.coast_mask_side_stack import native_source_identity


def test_compound_geometry_is_explicitly_horizontal():
    frame = gpd.GeoDataFrame(geometry=[box(-124.21, 41.90, -124.20, 41.92)], crs=5498)
    result = horizontal_frame(frame)
    assert result.crs == CRS(4269)
    assert result.geometry.iloc[0].wkb == frame.geometry.iloc[0].wkb
    audit = operation_record(5498, 4326, (-124.205, 41.91))
    assert audit["source_horizontal_crs"] == "EPSG:4269"
    assert audit["vertical_operation"] is False and audit["pipeline"]


def test_science_view_preserves_data_mask_and_original(tmp_path):
    source = tmp_path / "source.tif"
    dest = tmp_path / "view.tif"
    data = np.arange(48, dtype="float32").reshape(6, 8)
    valid = np.ones((6, 8), dtype="uint8") * 255
    valid[2, 3] = 0
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=8,
        height=6,
        count=1,
        crs=5498,
        transform=from_origin(-124.21, 41.92, 0.00003, 0.00003),
        dtype="float32",
    ) as ds:
        ds.write(data, 1)
        ds.write_mask(valid)
    original = source.read_bytes()
    science_horizontal_view(source, dest)
    with rasterio.open(source) as a, rasterio.open(dest) as b:
        assert a.crs == CRS(5498) and b.crs == CRS(4269)
        assert a.read().tobytes() == b.read().tobytes()
        assert np.array_equal(a.dataset_mask(), b.dataset_mask())
        assert a.transform == b.transform
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    "selector",
    [
        "class",
        "weight",
        "entry_weight",
        "module_uncertainty",
        "entry_uncertainty",
        "partition",
        "claim",
    ],
)
def test_native_identity_distinguishes_processing_but_is_not_survey_id(
    tmp_path, selector
):
    path = tmp_path / "survey.las"
    path.write_bytes(b"fixture asset")
    a = {
        "url": path.as_uri(),
        "dst_fn": str(path),
        "data_type": "lidar",
        "stack_partition": "ground",
    }
    b = dict(a)
    hook = SimpleNamespace(name="drop-class", target_classes=[2], invert=True)
    mod = SimpleNamespace(name="dav", weight=100, uncertainty=0, hooks=[hook])
    first = native_source_identity(mod, a)
    assert first == native_source_identity(mod, dict(a))
    if selector == "class":
        hook.target_classes = [29]
    if selector == "weight":
        mod.weight = 3
    if selector == "entry_weight":
        b["weight"] = 3
    if selector == "module_uncertainty":
        mod.uncertainty = 0.5
    if selector == "entry_uncertainty":
        b["uncertainty"] = 0.5
    if selector == "partition":
        b["stack_partition"] = "bottom"
    if selector == "claim":
        b["claim_geometry"] = box(0, 0, 1, 1)
    second = native_source_identity(mod, b)
    assert first["native_dataset_id"] != second["native_dataset_id"]
    assert second["native_dataset_id"] == dataset_id(
        dict(b, checksum=second["identity_checksum"]), **second["selectors"]
    )
    assert second["identity_owner"] == "fetchez.entry.dataset_id"


def test_archive_members_with_shared_provider_checksum_do_not_alias(tmp_path):
    first = tmp_path / "a.las"
    second = tmp_path / "b.las"
    first.write_bytes(b"first member")
    second.write_bytes(b"other member")
    module = SimpleNamespace(name="dav", weight=100, uncertainty=0, hooks=[])
    a = dict(
        url="file://archive",
        dst_fn=str(first),
        checksum="archive-checksum",
        data_type="lidar",
    )
    b = dict(a, dst_fn=str(second))
    assert (
        native_source_identity(module, a)["native_dataset_id"]
        != native_source_identity(module, b)["native_dataset_id"]
    )
