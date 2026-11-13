"""Shared proposal support and class-specific vetoes precede connectivity."""

import numpy as np
import pytest

from globato.coast_refine._science.coastline_classify import connected_classification


def test_common_screen_is_invariant_under_class_and_sign_inversion():
    rng = np.random.default_rng(43)
    for _ in range(25):
        z = rng.normal(size=(23, 31))
        z[rng.random(z.shape) < 0.04] = 0
        water = rng.random(z.shape) < 0.5
        valid = rng.random(z.shape) > 0.08
        land_gate = rng.random(z.shape) > 0.1
        water_gate = rng.random(z.shape) > 0.1
        result, land, added_water = connected_classification(
            z, valid, water, land_gate, water_proposals=water_gate
        )
        inverse, inverse_land, inverse_water = connected_classification(
            -z, valid, ~water, water_gate, water_proposals=land_gate
        )
        np.testing.assert_array_equal(inverse, ~result)
        np.testing.assert_array_equal(inverse_land, added_water)
        np.testing.assert_array_equal(inverse_water, land)


@pytest.mark.parametrize("sign", [1, -1])
def test_thin_opposite_sign_spur_cannot_extend_either_class(sign):
    z = np.full((11, 15), -float(sign))
    z[:, :3] = sign
    z[5, 3:12] = sign
    water = np.ones(z.shape, bool)
    water[:, :3] = False
    if sign < 0:
        water = ~water
    result, land, added_water = connected_classification(
        z, np.ones_like(water), water, np.ones_like(water)
    )
    changed = land if sign > 0 else added_water
    assert not changed[5, 6:].any()
    assert result[5, 11] == water[5, 11]


@pytest.mark.parametrize("sign", [1, -1])
def test_veto_cannot_bridge_to_supported_proposals_beyond_it(sign):
    z = np.full((9, 15), float(sign))
    water = np.ones(z.shape, bool)
    water[:, :3] = False
    gate = np.ones_like(water)
    gate[:, 6] = False
    if sign > 0:
        result, land, _ = connected_classification(z, np.ones_like(gate), water, gate)
        assert land[:, 3:6].all()
        assert result[:, 6:].all()
    else:
        result, _, added = connected_classification(
            z, np.ones_like(gate), ~water, np.ones_like(gate), water_proposals=gate
        )
        assert added[:, 3:6].all()
        assert not result[:, 6:].any()


def test_flat_negative_terrain_is_valid_water_support():
    z = np.full((9, 15), -0.2)
    water = np.zeros(z.shape, bool)
    water[:, :3] = True
    result, _, added = connected_classification(z, np.ones_like(water), water, water)
    assert result.all() and added[:, 3:].all()


def test_withheld_review_cells_cannot_support_disconnected_land(tmp_path):
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import box
    from globato.coast_refine._science.coastline_classify import classify_full
    from globato.coast_refine._science.grid_utils import mask_on_grid

    transform = from_origin(500000, 4000000, 1, 1)
    source = tmp_path / "raw.tif"
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=15,
        height=9,
        count=1,
        dtype="float32",
        nodata=-9999,
        crs=32611,
        transform=transform,
    ) as ds:
        ds.write(np.ones((9, 15), dtype="float32"), 1)
    baseline = tmp_path / "baseline.gpkg"
    gpd.GeoDataFrame(
        geometry=[box(500003, 3999991, 500015, 4000000)], crs=32611
    ).to_file(baseline, layer="mask")
    restoration = tmp_path / "restoration.gpkg"
    gpd.GeoDataFrame(
        geometry=[box(500014, 3999991, 500015, 4000000)], crs=32611
    ).to_file(restoration, layer="mask")
    review = tmp_path / "review.gpkg"
    gpd.GeoDataFrame(
        geometry=[box(500006, 3999991, 500007, 4000000)], crs=32611
    ).to_file(review, layer="unsupported_positive_review")
    result = classify_full(
        source, baseline, restoration, review, tmp_path / "classification"
    )
    water = mask_on_grid(
        tmp_path / "classification/refined_water.gpkg", (9, 15), transform, 32611
    )
    assert not water[:, :6].any()
    assert water[:, 6:].all()
    assert result["screen_audit"]["passed"]
