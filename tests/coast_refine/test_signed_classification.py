"""Signed observations change only connected classes; uncertainty keeps baseline."""

import numpy as np
import pytest
from globato.coast_refine._science.coastline_classify import connected_classification


def classify(z, water, proposals=None, valid=None, **kwargs):
    z = np.asarray(z, dtype="float64")
    water = np.asarray(water, dtype=bool)
    if proposals is None:
        proposals = np.ones_like(water)
    if valid is None:
        valid = np.ones_like(water)
    return connected_classification(z, valid, water, proposals, **kwargs)


def test_sloping_submerged_terrain_extends_existing_water():
    water = np.array([[False, False, False, True, True]])
    result, land, added = classify([[-0.1, -0.4, -0.8, -1.2, -2.0]], water)
    assert result.all() and added.sum() == 3 and not land.any()


def test_positive_observations_extend_existing_land():
    water = np.array([[False, False, True, True, True]])
    result, land, added = classify([[2, 1, 0.5, 0.1, -1]], water)
    np.testing.assert_array_equal(result, [[False, False, False, False, True]])
    assert land.sum() == 2 and not added.any()


@pytest.mark.parametrize("level", [-0.01, 0, 0.01, np.nan, np.inf, -np.inf])
def test_near_zero_and_nonfinite_cells_retain_starting_class(level):
    water = np.array([[False, True, False, True]])
    result, land, added = classify([[level] * 4], water)
    np.testing.assert_array_equal(result, water)
    assert not land.any() and not added.any()


def test_missing_and_near_zero_observations_block_both_connections():
    water = np.array([[False, True, True, True, False, False]])
    z = [[1, 0, 1, -1, 0, -1]]
    result, land, added = classify(z, water)
    np.testing.assert_array_equal(result, water)
    assert not land.any() and not added.any()
    result, _, _ = classify(
        [[1, 1, 1, -1, -1, -1]],
        water,
        valid=np.array([[True, False, True, True, False, True]]),
    )
    np.testing.assert_array_equal(result, water)


def test_diagonal_contact_is_not_a_supported_connection():
    water = np.array([[True, False], [False, False]])
    result, _, added = classify([[-1, 1], [1, -1]], water)
    assert not added.any()
    np.testing.assert_array_equal(result, water)


def test_height_alone_does_not_override_rejected_land_proposals():
    water = np.array([[False, True, True]])
    result, land, _ = classify([[2, 1, 0.5]], water, proposals=np.zeros_like(water))
    np.testing.assert_array_equal(result, water)
    assert not land.any()


def test_no_original_class_seed_cannot_invent_that_class():
    for water, value in [(np.ones((2, 3), bool), 1.0), (np.zeros((2, 3), bool), -1.0)]:
        result, land, added = classify(np.full((2, 3), value), water)
        np.testing.assert_array_equal(result, water)
        assert not land.any() and not added.any()


def test_shifted_reference_and_complement_partition():
    water = np.array([[False, True, True, True, False, False]])
    result, land, added = classify([[12, 11, 10, 9, 8, 10]], water, shore_level_m=10)
    np.testing.assert_array_equal(result, [[False, False, True, True, True, False]])
    assert land.sum() == 1 and added.sum() == 1
    assert not np.any(result & ~result)
    assert np.all(result | ~result)


@pytest.mark.parametrize("mapped_edge", [36, 60])
def test_continuous_topobathy_preparation_corrects_either_side(tmp_path, mapped_edge):
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import box
    from globato.coast_refine._io import digest
    from globato.coast_refine.prepare import prepare_from_baseline
    from globato.coast_refine._science.grid_utils import mask_on_grid

    size = 96
    cell = 1 / 32400
    span = size * cell
    source = tmp_path / "continuous.tif"
    z = np.tile((47.5 - np.arange(size, dtype="float32")) * 0.25, (size, 1))
    transform = from_origin(0, span, cell, cell)
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        count=1,
        width=size,
        height=size,
        dtype="float32",
        crs=4326,
        nodata=-9999,
        transform=transform,
    ) as dst:
        dst.write(z, 1)
    aoi = tmp_path / "aoi.gpkg"
    baseline = tmp_path / "baseline.gpkg"
    gpd.GeoDataFrame(geometry=[box(0, 0, span, span)], crs=4326).to_file(aoi)
    gpd.GeoDataFrame(
        geometry=[box(mapped_edge * cell, -span, 2 * span, 2 * span)], crs=4326
    ).to_file(baseline)
    before = digest(source)
    artifact = prepare_from_baseline(
        source,
        aoi,
        tmp_path / "prepared",
        tmp_path / "cache",
        mask=baseline,
        polygon_class="ocean",
    )
    water = mask_on_grid(artifact.products["refined_water"], z.shape, transform, 4326)
    land = mask_on_grid(artifact.products["refined_land"], z.shape, transform, 4326)
    assert water[:, 48:].all()
    assert not water[:, :48].any()
    np.testing.assert_array_equal(land, ~water)
    assert digest(source) == before
