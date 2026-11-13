"""Regression tests for Fiona -> Pyogrio Coast Mask I/O migration.

Place in tests/coast_refine/test_pyogrio_migration.py and run with:
    pytest -q tests/coast_refine/test_pyogrio_migration.py

These tests deliberately use small temporary GeoPackages. They exercise the
public-on-disk layer contracts rather than implementation-specific I/O calls.
"""

from __future__ import annotations

import json

import geopandas as gpd
import pandas as pd
import pyogrio
import pytest
import shapely
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box

from globato.coast_refine._io import write_polygon
from globato.coast_refine._science.coast_refine_v6_6_1.coast_refine_core import (
    validate_dropin,
    write_dropin_mask,
)
from globato.coast_refine._science.coast_refine_v6_6_1.input_resolver import (
    _choose_layer,
    normalize_existing_mask,
)
from globato.coast_refine._science.coast_refine_v6_6_1.restore import _schema_summary
from globato.coast_refine._science.raster_science import vector
from globato.coast_refine.baseline import _write_topology
from globato.coast_refine.prepare import _write_lines


def _layer_names(path):
    return pyogrio.list_layers(path)[:, 0].tolist()


def _assert_layer(path, name, geometry_type, count, crs="EPSG:4326"):
    assert path.is_file()
    assert name in _layer_names(path)
    info = pyogrio.read_info(path, layer=name)
    assert info["features"] == count
    assert info["geometry_type"] == geometry_type
    assert CRS.from_user_input(info["crs"]) == CRS.from_user_input(crs)
    return info


def test_write_polygon_empty_and_nonempty(tmp_path):
    empty = tmp_path / "empty.gpkg"
    write_polygon(empty, MultiPolygon([]))
    info = _assert_layer(empty, "mask", "MultiPolygon", 0)
    assert "role" in info["fields"]

    polygon = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)],
        holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]],
    )
    full = tmp_path / "nonempty.gpkg"
    write_polygon(full, polygon)
    _assert_layer(full, "mask", "MultiPolygon", 1)
    result = pyogrio.read_dataframe(full, layer="mask")
    assert shapely.equals(result.geometry.iloc[0], MultiPolygon([polygon]))
    assert result["role"].iloc[0] == full.stem


def test_write_lines_preserves_segments_and_publishes_empty_layer(tmp_path):
    path = tmp_path / "coast.gpkg"
    lines = MultiLineString(
        [
            [(0, 0), (1, 0)],
            [(2, 0), (2, 2)],
        ]
    )
    _write_lines(path, lines, role="coastline")
    _assert_layer(path, "mask", "MultiLineString", 1)
    frame = pyogrio.read_dataframe(path, layer="mask")
    assert frame["role"].iloc[0] == "coastline"
    assert shapely.equals(frame.geometry.iloc[0], lines)

    # Rewriting the same path must replace the previous layer/contents.
    _write_lines(path, MultiLineString([]))
    _assert_layer(path, "mask", "MultiLineString", 0)


def test_write_topology_empty_layer_has_declared_schema(tmp_path):
    path = tmp_path / "topology.gpkg"
    frame = gpd.GeoDataFrame(
        {"class": pd.Series(dtype="str")},
        geometry=gpd.GeoSeries([], crs="EPSG:4326"),
        crs="EPSG:4326",
    )
    _write_topology(frame, path)
    info = _assert_layer(path, "topology", "MultiPolygon", 0)
    assert "class" in info["fields"]
    assert pyogrio.read_dataframe(path, layer="topology").empty


def test_vector_empty_and_nonempty_geometry_and_types(tmp_path):
    path = tmp_path / "vectors.gpkg"
    vector(
        path,
        "empty",
        MultiPolygon([]),
        properties={"name": "none", "count": 3, "score": 1.25},
    )
    info = _assert_layer(path, "empty", "MultiPolygon", 0)
    assert {"name", "count", "score"} <= set(info["fields"])

    polygon = Polygon(
        [(0, 0), (5, 0), (5, 5), (0, 5)],
        holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]],
    )
    vector(path, "full", polygon, properties={"name": "land", "count": 7, "score": 2.5})
    _assert_layer(path, "full", "MultiPolygon", 1)
    assert set(_layer_names(path)) == {"empty", "full"}
    saved = pyogrio.read_dataframe(path, layer="full")
    assert shapely.equals(saved.geometry.iloc[0], polygon)
    assert saved["count"].iloc[0] == 7
    assert saved["score"].iloc[0] == pytest.approx(2.5)


def test_normalize_existing_mask_recognizes_canonical_schema(tmp_path):
    path = tmp_path / "canonical.gpkg"
    frame = gpd.GeoDataFrame(
        {"class": ["ocean"], "source": ["fixture"]},
        geometry=[MultiPolygon([box(0, 0, 3, 3)])],
        crs="EPSG:4326",
    )
    pyogrio.write_dataframe(
        frame, path, layer="mask", driver="GPKG", geometry_type="MultiPolygon"
    )

    normalized, metadata = normalize_existing_mask(path, tmp_path)
    assert normalized == path.resolve()
    assert metadata["normalized"] is False
    assert _choose_layer(path, None, preferred=("mask",)) == "mask"


def test_normalize_noncanonical_mask_preserves_polygon(tmp_path):
    source = tmp_path / "input.gpkg"
    polygon = box(0, 0, 2, 2)
    frame = gpd.GeoDataFrame({"class": ["water"]}, geometry=[polygon], crs=4326)
    pyogrio.write_dataframe(frame, source, layer="raw", driver="GPKG")

    result, metadata = normalize_existing_mask(source, tmp_path, mask_layer="raw")
    assert metadata["normalized"] is True
    info = _assert_layer(result, "mask", "MultiPolygon", 1)
    assert {"class", "source"} <= set(info["fields"])
    output = pyogrio.read_dataframe(result, layer="mask")
    assert shapely.equals(output.geometry.iloc[0], polygon)


def test_dropin_mask_keeps_features_attributes_holes_and_input(tmp_path):
    source = tmp_path / "baseline.gpkg"
    original = Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8)],
        holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]],
    )
    another = box(10, 0, 12, 2)
    before = gpd.GeoDataFrame(
        {"class": ["ocean", "ocean"], "source": ["a", "b"], "score": [1, 2]},
        geometry=[MultiPolygon([original]), MultiPolygon([another])],
        crs=4326,
    )
    pyogrio.write_dataframe(
        before, source, layer="mask", driver="GPKG", geometry_type="MultiPolygon"
    )
    source_bytes = source.read_bytes()
    output = tmp_path / "refined.gpkg"
    addition = box(3, 3, 4, 4)

    result = write_dropin_mask(source, output, addition)
    assert result["drop_in_compatible"] is True
    _assert_layer(output, "mask", "MultiPolygon", 2)
    after = pyogrio.read_dataframe(output, layer="mask")
    assert list(after["class"]) == ["ocean", "ocean"]
    assert list(after["source"]) == ["a", "b"]
    assert list(after["score"]) == [1, 2]
    assert shapely.equals(after.geometry.iloc[0], original.difference(addition))
    assert shapely.equals(after.geometry.iloc[1], another)
    assert source.read_bytes() == source_bytes


def test_dropin_mask_rejects_total_feature_erasure(tmp_path):
    source = tmp_path / "baseline.gpkg"
    pyogrio.write_dataframe(
        gpd.GeoDataFrame(
            {"class": ["ocean"]}, geometry=[MultiPolygon([box(0, 0, 1, 1)])], crs=4326
        ),
        source,
        layer="mask",
        driver="GPKG",
        geometry_type="MultiPolygon",
    )
    output = tmp_path / "refined.gpkg"
    with pytest.raises(ValueError, match="erase an entire baseline feature"):
        write_dropin_mask(source, output, box(-1, -1, 2, 2))
    assert not output.exists()


def test_dropin_validation_detects_schema_mismatch(tmp_path):
    a, b = tmp_path / "a.gpkg", tmp_path / "b.gpkg"
    geom = [MultiPolygon([box(0, 0, 1, 1)])]
    pyogrio.write_dataframe(
        gpd.GeoDataFrame({"class": ["ocean"]}, geometry=geom, crs=4326),
        a,
        layer="mask",
        driver="GPKG",
        geometry_type="MultiPolygon",
    )
    pyogrio.write_dataframe(
        gpd.GeoDataFrame({"other": ["ocean"]}, geometry=geom, crs=4326),
        b,
        layer="mask",
        driver="GPKG",
        geometry_type="MultiPolygon",
    )
    with pytest.raises(RuntimeError, match="drop-in schema/CRS/layer"):
        validate_dropin(a, b)


def test_schema_summary_is_serializable_and_correct(tmp_path):
    path = tmp_path / "mask.gpkg"
    pyogrio.write_dataframe(
        gpd.GeoDataFrame(
            {"class": ["water"]}, geometry=[MultiPolygon([box(0, 0, 1, 1)])], crs=4326
        ),
        path,
        layer="mask",
        driver="GPKG",
        geometry_type="MultiPolygon",
    )
    summary = _schema_summary(path)
    json.dumps(summary)  # Used in the worker's input_schema.json receipt.
    assert summary["layers"] == ["mask"]
    assert summary["feature_count"] == 1
    assert summary["schema"]["geometry"] == "MultiPolygon"
    assert "class" in summary["schema"]["properties"]


def test_write_lines_nested_geometry_collection(tmp_path):
    import pyogrio
    from shapely.geometry import (
        GeometryCollection,
        MultiLineString,
        Point,
    )

    from globato.coast_refine.prepare import _write_lines

    geometry = GeometryCollection(
        [
            Point(0, 0),
            LineString([(0, 0), (1, 1)]),
            MultiLineString(
                [
                    [(2, 2), (3, 3)],
                    [(4, 4), (5, 5)],
                ]
            ),
        ]
    )

    path = tmp_path / "coastline.gpkg"
    _write_lines(path, geometry)

    info = pyogrio.read_info(path, layer="mask")
    frame = pyogrio.read_dataframe(path, layer="mask")

    assert info["geometry_type"] == "MultiLineString"
    assert info["features"] == 1

    saved = frame.geometry.iloc[0]
    assert len(saved.geoms) == 3


def test_write_empty_topology(tmp_path):
    import geopandas as gpd
    import pandas as pd
    import pyogrio

    frame = gpd.GeoDataFrame(
        {
            "class": pd.Series(dtype="str"),
            "geometry": gpd.GeoSeries([], crs="EPSG:4326"),
        },
        crs="EPSG:4326",
    )

    path = tmp_path / "empty_topology.gpkg"
    _write_topology(frame, path)

    info = pyogrio.read_info(path, layer="topology")

    assert info["features"] == 0
    assert info["geometry_type"] == "MultiPolygon"
    assert "class" in info["fields"]
    assert pyogrio.read_dataframe(path, layer="topology").empty
