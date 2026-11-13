"""Published water subtypes use complete cells without changing water or land."""

import hashlib
import json
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.features import shapes, rasterize
from rasterio.transform import from_origin
import shapely
from shapely.geometry import box, shape, Polygon

from globato.coast_refine.topology import _topology_frame, publish_topology
from globato.coast_refine._io import write_polygon
from globato.coast_refine.application import _polygonize_flag, _polygonize_labels


def union_flag(flag, affine):
    return shapely.union_all(
        [shape(g) for g, c in shapes(flag.astype("uint8"), transform=affine) if c == 1]
    )


def setup_grid(width=16, height=16):
    affine = from_origin(-118, 34.01, 0.0001, 0.0001)
    return dict(
        crs="EPSG:4326", width=width, height=height, transform=list(affine)[:6]
    ), affine


def assert_whole_cells(frame, grid):
    inv = ~rasterio.Affine(*grid["transform"])
    for geometry in frame.loc[
        frame["class"].isin(["river", "lake", "water", "wetland", "estuary", "ocean"])
    ].geometry:
        coordinates = shapely.get_coordinates(geometry)
        x, y = inv * (coordinates[:, 0], coordinates[:, 1])
        np.testing.assert_allclose(x, np.rint(x), atol=1e-7)
        np.testing.assert_allclose(y, np.rint(y), atol=1e-7)


def test_diagonal_lake_has_no_subcell_wedges_and_keeps_exact_water(tmp_path):
    grid, affine = setup_grid()
    rows, cols = np.indices((16, 16))
    mask = (cols > rows // 2) & (cols < 14)
    water = union_flag(mask, affine)
    extent = box(*rasterio.transform.array_bounds(16, 16, affine))
    source = gpd.GeoDataFrame(
        {"class": ["lake"]},
        geometry=[
            Polygon(
                [(-118.0005, 34.0105), (-117.99855, 34.0105), (-117.99855, 34.0075)]
            )
        ],
        crs=4326,
    )
    frame = _topology_frame(
        tmp_path, source, extent.difference(water), water, shapely.Polygon(), grid
    )
    assert_whole_cells(frame, grid)
    published = shapely.union_all(frame.loc[frame["class"] != "land"].geometry)
    assert published.equals(water)
    # The host's existing cell-center rasterizer defines subtype membership.
    expected = (
        rasterize(
            [(source.geometry[0], 1)], out_shape=mask.shape, transform=affine
        ).astype(bool)
        & mask
    )
    lake = shapely.union_all(frame.loc[frame["class"] == "lake"].geometry)
    actual = rasterize([(lake, 1)], out_shape=mask.shape, transform=affine).astype(bool)
    np.testing.assert_array_equal(actual, expected)


def test_narrow_river_and_island_survive_without_area_filter(tmp_path):
    grid, affine = setup_grid()
    mask = np.zeros((16, 16), bool)
    mask[2:14, 7] = True
    mask[5:11, 4:11] = True
    mask[7, 7] = False
    water = union_flag(mask, affine)
    extent = box(*rasterio.transform.array_bounds(16, 16, affine))
    source = gpd.GeoDataFrame({"class": ["river"]}, geometry=[extent], crs=4326)
    frame = _topology_frame(
        tmp_path, source, extent.difference(water), water, shapely.Polygon(), grid
    )
    assert_whole_cells(frame, grid)
    river = shapely.union_all(frame.loc[frame["class"] == "river"].geometry)
    np.testing.assert_array_equal(
        rasterize([(river, 1)], out_shape=mask.shape, transform=affine).astype(bool),
        mask,
    )
    assert not river.contains(shapely.Point(*(affine * (7.5, 7.5))))


def test_existing_priority_and_marine_domain_are_preserved(tmp_path):
    grid, affine = setup_grid()
    extent = box(*rasterio.transform.array_bounds(16, 16, affine))
    marine = union_flag(np.indices((16, 16))[1] > 11, affine)
    source = gpd.GeoDataFrame(
        {"class": ["lake", "river"]}, geometry=[extent, extent], crs=4326
    )
    frame = _topology_frame(tmp_path, source, shapely.Polygon(), extent, marine, grid)
    assert "lake" not in set(frame["class"])
    assert frame.loc[frame["class"] == "ocean"].geometry.iloc[0].equals(marine)
    assert (
        frame.loc[frame["class"] == "river"]
        .geometry.iloc[0]
        .equals(extent.difference(marine))
    )


@pytest.mark.parametrize("water_present", [False, True])
def test_uniform_domains_and_absent_subtype_are_valid(tmp_path, water_present):
    grid, affine = setup_grid()
    extent = box(*rasterio.transform.array_bounds(16, 16, affine))
    water = extent if water_present else shapely.Polygon()
    land = extent.difference(water)
    source = gpd.GeoDataFrame({"class": []}, geometry=[], crs=4326)
    frame = _topology_frame(tmp_path, source, land, water, shapely.Polygon(), grid)
    assert set(frame["class"]) == ({"water"} if water_present else {"land"})
    assert frame.geometry.iloc[0].equals(extent)


def test_stripe_boundary_does_not_split_geometry_or_change_cells(tmp_path):
    grid, affine = setup_grid(16, 520)
    extent = box(*rasterio.transform.array_bounds(520, 16, affine))
    source = gpd.GeoDataFrame(
        {"class": ["lake"]},
        geometry=[Polygon([(-118, 34.01), (-117.9984, 34.01), (-117.9984, 33.958)])],
        crs=4326,
    )
    frame = _topology_frame(
        tmp_path, source, shapely.Polygon(), extent, shapely.Polygon(), grid
    )
    assert_whole_cells(frame, grid)
    expected = rasterize(
        [(source.geometry[0], 1)], out_shape=(520, 16), transform=affine
    )
    lake = frame.loc[frame["class"] == "lake"].geometry.iloc[0]
    np.testing.assert_array_equal(
        rasterize([(lake, 1)], out_shape=(520, 16), transform=affine), expected
    )


def test_publication_never_rewrites_scientific_masks_or_input_vectors(tmp_path):
    grid, affine = setup_grid()
    extent = box(*rasterio.transform.array_bounds(16, 16, affine))
    water = union_flag(np.indices((16, 16))[1] > 6, affine)
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "application_grid.json").write_text(json.dumps(grid))
    for name, g in [
        ("refined_water", water),
        ("refined_land", extent.difference(water)),
        ("marine_water", shapely.Polygon()),
        ("original_water", water),
    ]:
        write_polygon(root / (name + ".gpkg"), g)
    source = gpd.GeoDataFrame({"class": ["lake"]}, geometry=[extent], crs=4326)
    source.to_file(root / "original_topology.gpkg", layer="topology")
    protected = [p for p in root.glob("*.gpkg")]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    baseline = SimpleNamespace(
        paths={
            "original_topology": root / "original_topology.gpkg",
            "water_mask": root / "original_water.gpkg",
        }
    )
    publish_topology(root, baseline, extent)
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    assert_whole_cells(gpd.read_file(root / "refined_topology.gpkg"), grid)


def test_binary_polygonizer_compatibility_and_multiclass_single_pass(tmp_path):
    grid, affine = setup_grid()
    data = np.zeros((16, 16), "uint8")
    data[2:5, 2:5] = 1
    data[6:9, 6:9] = 2
    path = tmp_path / "labels.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=1,
        dtype="uint8",
        crs=4326,
        transform=affine,
    ) as dst:
        dst.write(data, 1)
    values = _polygonize_labels(path, 1)
    assert set(values) == {1, 2}
    assert _polygonize_flag(path, 1).equals(values[1])
    for code, geometry in values.items():
        assert geometry.equals(union_flag(data == code, affine))
