from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import shapely
import yaml
from fetchez.recipe import Recipe
from fetchez.spatial import Region
from pyproj import Transformer

from globato.hooks.filters.tnm_coverage import TNMCoverage
from globato.hooks.filters.tnm_coverage_filter import TNMCoverageFilter


def _module(bounds=(0, 5, 0, 1)):
    return SimpleNamespace(
        wgs_region=SimpleNamespace(w=bounds[0], e=bounds[1], s=bounds[2], n=bounds[3])
    )


def _entry(project, bounds, geometry, year=None):
    return {
        "bounds": bounds,
        "tnm_project": project,
        "tnm_source_coverage": [
            {
                "geometry": shapely.to_wkt(geometry, rounding_precision=-1),
                "year": year,
                "fid": 1,
                "project": project,
            }
        ],
        "metadata": {"name": "tnm", "source": "USGS"},
    }


def _coverage(entry):
    return shapely.from_wkt(entry["_tnm_coverage_wkt"])


def test_different_year_projects_resolve_newer_first():
    old = _entry("old", (0, 2, 0, 1), shapely.box(0, 0, 2, 1), 2018)
    new = _entry("new", (1, 3, 0, 1), shapely.box(1, 0, 3, 1), 2024)

    selected = TNMCoverage(product="1m", start=True).run(
        [(_module(), old), (_module(), new)]
    )

    assert len(selected) == 2
    assert _coverage(old).equals(shapely.box(0, 0, 1, 1))
    assert _coverage(new).equals(shapely.box(1, 0, 3, 1))
    assert shapely.from_wkt(old["_tnm_excluded_wkt"]).equals(shapely.box(1, 0, 2, 1))
    assert _coverage(old).intersection(_coverage(new)).area == 0


def test_same_year_projects_are_not_ranked():
    left = _entry("left", (0, 2, 0, 1), shapely.box(0, 0, 2, 1), 2024)
    right = _entry("right", (1, 3, 0, 1), shapely.box(1, 0, 3, 1), 2024)

    TNMCoverage(product="1m", start=True).run([(_module(), left), (_module(), right)])

    assert _coverage(left).intersection(_coverage(right)).area == 1
    assert "_tnm_excluded_wkt" not in left
    assert "_tnm_excluded_wkt" not in right


def test_products_use_source_footprints_for_strict_fallback():
    one_meter = _entry("one-meter", (0, 2, 0, 1), shapely.box(0, 0, 2, 1), 2024)
    one_ninth = _entry("one-ninth", (0, 4, 0, 1), shapely.box(0, 0, 4, 1))
    one_third = {"bounds": (0, 5, 0, 1), "metadata": {}}

    TNMCoverage(product="1m", start=True).run([(_module(), one_meter)])
    TNMCoverage(product="1_9as").run([(_module(), one_ninth)])
    TNMCoverage(product="1_3as").run([(_module(), one_third)])

    assert _coverage(one_meter).equals(shapely.box(0, 0, 2, 1))
    assert _coverage(one_ninth).equals(shapely.box(2, 0, 4, 1))
    assert _coverage(one_third).equals(shapely.box(4, 0, 5, 1))
    assert _coverage(one_meter).intersection(_coverage(one_ninth)).area == 0
    assert _coverage(one_ninth).intersection(_coverage(one_third)).area == 0


def test_lower_tier_does_not_fill_an_interior_nodata_hole():
    footprint = shapely.box(0, 0, 2, 1)
    one_meter = _entry("one-meter", (0, 2, 0, 1), footprint, 2024)
    one_ninth = _entry("one-ninth", (0, 3, 0, 1), shapely.box(0, 0, 3, 1))

    TNMCoverage(product="1m", start=True).run([(_module(), one_meter)])
    TNMCoverage(product="1_9as").run([(_module(), one_ninth)])

    assert _coverage(one_ninth).intersection(footprint).area == 0


def test_missing_provider_source_coverage_fails_closed():
    entry = {"bounds": (0, 1, 0, 1), "tnm_project": "unknown"}

    with pytest.raises(RuntimeError, match="no provider source coverage"):
        TNMCoverage(product="1m", start=True).run([(_module(), entry)])


def test_tier_metadata_uses_only_common_nested_fields():
    entry = _entry("Project A", (0, 1, 0, 1), shapely.box(0, 0, 1, 1), 2024)
    entry["tnm_project"] = "provider-project-a"

    TNMCoverage(product="1m", start=True).run([(_module(), entry)])

    assert entry["tnm_resolution_tier"] == "1m"
    assert entry["metadata"] == {
        "name": "tnm",
        "source": "USGS",
        "category": "elevation",
        "dataset": "TNM 1 m: Project A",
        "resolution": "1 m",
    }
    assert not any(key.startswith("tnm_") for key in entry["metadata"])


def test_stream_filter_excludes_shared_higher_tier_boundary():
    dtype = [("x", "f8"), ("y", "f8")]
    stream = iter([np.array([(1.0, 0.5), (1.5, 0.5)], dtype=dtype)])
    entry = {
        "stream": stream,
        "_tnm_coverage_wkt": shapely.to_wkt(shapely.box(1, 0, 2, 1)),
        "_tnm_excluded_wkt": shapely.to_wkt(shapely.box(0, 0, 1, 1)),
    }
    hook = TNMCoverageFilter(res=0.01)
    hook.is_point_stream = lambda value: True

    selected = hook.run([(SimpleNamespace(region=Region(0, 300000, 0, 100000)), entry)])
    points = list(selected[0][1]["stream"])[0]

    assert points["x"].tolist() == [1.5]


@pytest.mark.parametrize(
    "target_srs", ["EPSG:3857", "EPSG:3857+5703", "EPSG:4269+5703"]
)
def test_stream_filter_transforms_coverage_to_the_active_horizontal_crs(target_srs):
    horizontal = target_srs.split("+")[0]
    transformer = Transformer.from_crs("EPSG:4326", horizontal, always_xy=True)
    inside = transformer.transform(0.5, 0.5)
    outside = transformer.transform(2.0, 0.5)
    dtype = [("x", "f8"), ("y", "f8")]
    stream = iter([np.array([inside, outside], dtype=dtype)])
    entry = {
        "stream": stream,
        "src_srs": target_srs,
        "_tnm_coverage_wkt": shapely.to_wkt(shapely.box(0, 0, 1, 1)),
    }
    hook = TNMCoverageFilter(res=0.01)
    hook.is_point_stream = lambda value: True

    selected = hook.run([(SimpleNamespace(region=Region(0, 300000, 0, 100000)), entry)])
    points = list(selected[0][1]["stream"])[0]

    assert points["x"].tolist() == [inside[0]]
    assert points["y"].tolist() == [inside[1]]


def test_stream_filter_transforms_higher_product_exclusions():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    coordinates = [transformer.transform(x, 0.5) for x in (0.5, 1.0, 1.5, 2.5)]
    dtype = [("x", "f8"), ("y", "f8")]
    entry = {
        "stream": iter([np.array(coordinates, dtype=dtype)]),
        "src_srs": "EPSG:3857",
        "_tnm_coverage_wkt": shapely.to_wkt(shapely.box(1, 0, 2, 1)),
        "_tnm_excluded_wkt": shapely.to_wkt(shapely.box(0, 0, 1, 1)),
    }
    hook = TNMCoverageFilter(res=0.01)
    hook.is_point_stream = lambda value: True

    selected = hook.run([(SimpleNamespace(region=Region(0, 300000, 0, 100000)), entry)])
    points = list(selected[0][1]["stream"])[0]

    assert points["x"].tolist() == [coordinates[2][0]]
    assert points["y"].tolist() == [coordinates[2][1]]


def test_stream_geometry_preserves_multipart_coverage_and_holes():
    polygon = shapely.box(0, 0, 2, 2).difference(shapely.box(0.5, 0.5, 1.5, 1.5))
    geometry = shapely.MultiPolygon([polygon, shapely.box(3, 0, 4, 1)])

    projected = TNMCoverageFilter._stream_geometry(geometry, "EPSG:3857")

    assert projected.geom_type == "MultiPolygon"
    assert len(projected.geoms) == 2
    assert len(projected.geoms[0].interiors) == 1
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    for coordinates, expected in [
        ((0.25, 0.25), True),
        ((1, 1), False),
        ((3.5, 0.5), True),
    ]:
        assert (
            projected.contains(shapely.Point(transformer.transform(*coordinates)))
            == expected
        )


def test_glob_tnm_expands_in_precedence_order_with_cudem_weights():
    modules = Recipe({})._expand_modules([{"bundle": "glob-tnm"}])

    assert [module["args"]["datasets"] for module in modules] == [
        "s1m",
        "1m",
        "5m",
        "1_9as",
        "1_3as",
        "1_as",
        "2_as",
    ]
    assert [module["args"]["weight"] for module in modules] == [
        10.0,
        5.0,
        4.0,
        3.0,
        2.0,
        1.0,
        0.5,
    ]
    assert all(module["args"]["use_cache"] is False for module in modules)
    assert modules[1]["args"]["source_coverage"] is True
    assert modules[3]["args"]["source_coverage"] is True
    assert all(
        "source_coverage" not in module["args"]
        for index, module in enumerate(modules)
        if index not in (0, 1, 2, 3)
    )


def test_glob_tnm_products_select_any_subset_in_canonical_order():
    modules = Recipe({})._expand_modules(
        [
            {
                "bundle": "glob-tnm",
                "args": {"products": "2_as,5m,1_as"},
            }
        ]
    )

    assert [module["args"]["datasets"] for module in modules] == [
        "5m",
        "1_as",
        "2_as",
    ]
    assert modules[0]["hooks"][0] == {
        "name": "tnm-coverage",
        "args": {"product": "5m", "start": True},
    }
    assert all("start" not in module["hooks"][0]["args"] for module in modules[1:])


def test_tnm_weights_route_to_the_intended_cudem_classes():
    preset_path = (
        Path(__file__).parents[1]
        / "src"
        / "globato"
        / "hooks"
        / "presets"
        / "mr_globato_cudem.yaml"
    )
    preset = yaml.safe_load(preset_path.read_text())
    cudem = preset["hooks"][0]["args"][0]["args"]

    assert cudem["resolutions"] == ".1111111s/.3333333s/1s/3s/9s/15s"
    assert cudem["weights"] == [3.0, 2.0, 1.0, 0.5, 0.25]

    modules = Recipe({})._expand_modules([{"bundle": "glob-tnm"}])
    weights = {
        module["args"]["datasets"]: module["args"]["weight"] for module in modules
    }
    assert weights == {
        "s1m": 10.0,
        "1m": 5.0,
        "5m": 4.0,
        "1_9as": 3.0,
        "1_3as": 2.0,
        "1_as": 1.0,
        "2_as": 0.5,
    }


def test_skipped_products_still_use_cumulative_coverage():
    fine = {
        "bounds": (0, 2, 0, 1),
        "metadata": {},
        "tnm_source_coverage": [{"geometry": shapely.to_wkt(shapely.box(0, 0, 2, 1))}],
    }
    medium = {"bounds": (0, 4, 0, 1), "metadata": {}}
    coarse = {"bounds": (0, 5, 0, 1), "metadata": {}}

    TNMCoverage(product="5m", start=True).run([(_module(), fine)])
    TNMCoverage(product="1_as").run([(_module(), medium)])
    TNMCoverage(product="2_as").run([(_module(), coarse)])

    assert _coverage(fine).equals(shapely.box(0, 0, 2, 1))
    assert _coverage(medium).equals(shapely.box(2, 0, 4, 1))
    assert _coverage(coarse).equals(shapely.box(4, 0, 5, 1))


def test_unknown_product_is_rejected():
    with pytest.raises(ValueError, match="Unsupported TNM product"):
        TNMCoverage(product="1as")


def test_projected_raster_footprint_leaves_catalog_corner_for_fallback():
    fine = {
        "bounds": (0, 2, 0, 2),
        "metadata": {},
        "tnm_source_coverage": [{"geometry": "POLYGON ((0 0, 2 0, 0 2, 0 0))"}],
    }
    coarse = {"bounds": (0, 2, 0, 2), "metadata": {}}
    module = _module((0, 2, 0, 2))
    TNMCoverage(product="s1m", start=True).run([(module, fine)])
    TNMCoverage(product="1_as").run([(module, coarse)])
    assert _coverage(coarse).covers(shapely.Point(1.5, 1.5))
    assert _coverage(coarse).intersection(_coverage(fine)).area == 0


@pytest.mark.parametrize("high_points", [True, False])
def test_output_cell_exclusion_prevents_same_class_mixing(tmp_path, high_points):
    from globato.hooks.sinks.multi_stack import MultiStackAccumulator
    import rasterio

    dtype = [("x", "f8"), ("y", "f8"), ("z", "f8"), ("w", "f8")]
    higher = shapely.box(0, 0, 0.6, 1)
    lower = shapely.box(0.6, 0, 3, 1)
    points = np.array([(0.8, 0.5, 30, 3), (2.5, 0.5, 30, 3)], dtype=dtype)
    entry = {
        "stream": iter([points[:1], points[1:]]),
        "_tnm_coverage_wkt": shapely.to_wkt(lower),
        "_tnm_excluded_wkt": shapely.to_wkt(higher),
    }
    region = Region(0, 3, 0, 1)
    hook = TNMCoverageFilter(res=1)
    hook.is_point_stream = lambda value: True
    hook.run([(SimpleNamespace(region=region), entry)])
    selected = np.concatenate(list(entry["stream"]))
    assert selected["x"].tolist() == [2.5]

    if high_points:
        old_stack = MultiStackAccumulator(
            region,
            1,
            1,
            str(tmp_path / "old_stack.tif"),
            mode="mixed",
            weight_threshold="0.5/1/2/3",
        )
        old_stack.update(np.array([(0.4, 0.5, 10, 10)], dtype=dtype))
        old_stack.update(points)
        old_stack.finalize()
        with rasterio.open(tmp_path / "old_stack.tif") as src:
            assert src.read(3)[0, 0] == 6.5

    stack = MultiStackAccumulator(
        region,
        1,
        1,
        str(tmp_path / "native_stack.tif"),
        mode="mixed",
        weight_threshold="0.5/1/2/3",
    )
    if high_points:
        stack.update(np.array([(0.4, 0.5, 10, 10)], dtype=dtype))
    stack.update(selected)
    stack.finalize()
    with rasterio.open(tmp_path / "native_stack.tif") as src:
        weights = src.read(3)
    assert weights[0, 0] == (10 if high_points else -9999)
    assert weights[0, 2] == 3


def test_stream_filter_requires_output_grid():
    entry = {"stream": iter([]), "_tnm_coverage_wkt": "POLYGON ((0 0, 1 0, 1 1, 0 0))"}
    hook = TNMCoverageFilter()
    hook.is_point_stream = lambda value: True
    with pytest.raises(RuntimeError, match="output resolution"):
        hook.run([(SimpleNamespace(region=Region(0, 1, 0, 1)), entry)])


def test_bundle_filter_uses_build_resolution():
    from globato.utils import globatize_modules

    modules = globatize_modules(
        [{"bundle": "glob-tnm", "args": {"products": "s1m/1_9as"}}],
        crs="EPSG:4269+5703",
        res="0.1111111111s",
    )
    for module in modules:
        filters = [h for h in module["hooks"] if h["name"] == "tnm-coverage-filter"]
        assert len(filters) == 1
        assert filters[0]["args"]["res"] == pytest.approx(0.1111111111 / 3600)
