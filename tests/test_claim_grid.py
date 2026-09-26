from types import SimpleNamespace

import numpy as np
import pytest
import shapely
from fetchez.spatial import Region

from globato.hooks.filters.claim_grid import ClaimGridFilter


def _hook(res=1):
    hook = ClaimGridFilter(res=res)
    hook.is_point_stream = lambda value: True
    return hook


def test_excluded_claim_blocks_entire_output_cell():
    dtype = [("x", "f8"), ("y", "f8")]
    higher = shapely.box(0, 0, 0.6, 1)
    lower = shapely.box(0.6, 0, 3, 1)
    points = np.array([(0.8, 0.5), (2.5, 0.5)], dtype=dtype)
    entry = {
        "stream": iter([points]),
        "stream_type": "point-stream",
        "src_srs": "EPSG:4326",
        "claim_required": True,
        "accepted_geometry": shapely.to_wkt(lower),
        "excluded_geometry": shapely.to_wkt(higher),
    }

    _hook().run([(SimpleNamespace(region=Region(0, 3, 0, 1)), entry)])
    selected = np.concatenate(list(entry["stream"]))

    # x=.8 is not inside the higher footprint, but shares the 0..1 grid cell
    # with it and must therefore be excluded to prevent tier mixing.
    assert selected["x"].tolist() == [2.5]


def test_equal_claim_without_exclusion_is_plain_spatial_crop():
    dtype = [("x", "f8"), ("y", "f8")]
    points = np.array([(0.5, 0.5), (1.5, 0.5)], dtype=dtype)
    entry = {
        "stream": iter([points]),
        "stream_type": "point-stream",
        "src_srs": "EPSG:4326",
        "claim_required": True,
        "accepted_geometry": shapely.to_wkt(shapely.box(0, 0, 1, 1)),
    }

    _hook().run([(SimpleNamespace(region=Region(0, 2, 0, 1)), entry)])
    selected = np.concatenate(list(entry["stream"]))
    assert selected["x"].tolist() == [0.5]


def test_marked_entry_without_manifest_claim_fails_closed():
    entry = {
        "stream": iter([]),
        "stream_type": "point-stream",
        "claim_required": True,
    }
    with pytest.raises(RuntimeError, match="accepted_geometry"):
        _hook().run([(SimpleNamespace(region=Region(0, 1, 0, 1)), entry)])


def test_unmarked_point_stream_is_ignored():
    dtype = [("x", "f8"), ("y", "f8")]
    points = np.array([(0.5, 0.5)], dtype=dtype)
    entry = {
        "stream": iter([points]),
        "stream_type": "point-stream",
    }
    _hook().run([(SimpleNamespace(region=Region(0, 1, 0, 1)), entry)])
    assert np.concatenate(list(entry["stream"]))["x"].tolist() == [0.5]


def test_output_resolution_is_required_for_claimed_stream():
    entry = {
        "stream": iter([]),
        "stream_type": "point-stream",
        "claim_required": True,
        "accepted_geometry": shapely.to_wkt(shapely.box(0, 0, 1, 1)),
    }
    with pytest.raises(RuntimeError, match="output resolution"):
        _hook(res=None).run([(SimpleNamespace(region=Region(0, 1, 0, 1)), entry)])


def test_output_cell_exclusion_prevents_multistack_mixing(tmp_path, monkeypatch):
    from fetchez.spatial import Region
    from globato.hooks.sinks.multi_stack import MultiStackAccumulator
    import inspect
    import rasterio

    dtype = [("x", "f8"), ("y", "f8"), ("z", "f8"), ("w", "f8")]
    higher = shapely.box(0, 0, 0.6, 1)
    lower = shapely.box(0.6, 0, 3, 1)
    lower_points = np.array([(0.8, 0.5, 30, 3), (2.5, 0.5, 30, 3)], dtype=dtype)
    entry = {
        "stream": iter([lower_points[:1], lower_points[1:]]),
        "accepted_geometry": shapely.to_wkt(lower),
        "excluded_geometry": shapely.to_wkt(higher),
        "claim_required": True,
    }
    region = Region(0, 3, 0, 1)
    hook = ClaimGridFilter(res=1)
    hook.is_point_stream = lambda value: True
    hook.run([(SimpleNamespace(region=region), entry)])
    selected = np.concatenate(list(entry["stream"]))
    assert selected["x"].tolist() == [2.5]

    # MultiStackAccumulator may create auxiliary scratch rasters relative to
    # the process working directory. Keep every generated artifact inside
    # pytest's disposable tmp_path so focused tests can never dirty the repo.
    monkeypatch.chdir(tmp_path)
    stack = MultiStackAccumulator(
        region,
        1,
        1,
        str(tmp_path / "claim_stack.tif"),
    )

    # The regression tests claim filtering, not the sink's optional mode or
    # weight thresholds. Its two sources occupy different cells after
    # filtering; default aggregation preserves their individual weights.
    # The update API may optionally require a dataset identity.
    def _update(points, dataset_id):
        if "dataset_id" in inspect.signature(stack.update).parameters:
            stack.update(points, dataset_id)
        else:
            stack.update(points)

    _update(np.array([(0.4, 0.5, 10, 10)], dtype=dtype), "higher-tier")
    _update(selected, "lower-tier")
    stack.finalize()
    with rasterio.open(tmp_path / "claim_stack.tif") as src:
        weights = src.read(3)
    assert weights[0, 0] == 10
    assert weights[0, 2] == 3


@pytest.mark.parametrize(
    "target_srs", ["EPSG:3857", "EPSG:3857+5703", "EPSG:4269+5703"]
)
def test_claim_grid_transforms_geometry_to_active_horizontal_crs(target_srs):
    from fetchez.spatial import Region
    from pyproj import Transformer

    horizontal = target_srs.split("+")[0]
    transformer = Transformer.from_crs("EPSG:4326", horizontal, always_xy=True)
    inside = transformer.transform(0.5, 0.5)
    outside = transformer.transform(2.0, 0.5)
    dtype = [("x", "f8"), ("y", "f8")]
    entry = {
        "stream": iter([np.array([inside, outside], dtype=dtype)]),
        "src_srs": target_srs,
        "accepted_geometry": shapely.to_wkt(shapely.box(0, 0, 1, 1)),
        "claim_required": True,
    }
    hook = ClaimGridFilter(res=0.01)
    hook.is_point_stream = lambda value: True
    hook.run([(SimpleNamespace(region=Region(0, 300000, 0, 100000)), entry)])
    points = list(entry["stream"])[0]
    assert points["x"].tolist() == [inside[0]]
    assert points["y"].tolist() == [inside[1]]


def test_claim_grid_preserves_multipart_geometry_and_holes():
    from pyproj import Transformer

    polygon = shapely.box(0, 0, 2, 2).difference(shapely.box(0.5, 0.5, 1.5, 1.5))
    geometry = shapely.MultiPolygon([polygon, shapely.box(3, 0, 4, 1)])
    hook = ClaimGridFilter(res=1)
    projected = hook._stream_geometry(geometry, "EPSG:3857")

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


def test_boundary_only_touch_preserves_lower_tier_cell():
    dtype = [("x", "f8"), ("y", "f8")]
    # Upper footprint ends precisely where the next cell starts.
    higher = shapely.box(0, 0, 1, 1)
    lower = shapely.box(1, 0, 3, 1)
    entry = {
        "stream": iter([np.array([(1.25, 0.5), (2.25, 0.5)], dtype=dtype)]),
        "stream_type": "point-stream",
        "src_srs": "EPSG:4326",
        "claim_required": True,
        "accepted_geometry": shapely.to_wkt(lower),
        "excluded_geometry": shapely.to_wkt(higher),
    }
    _hook().run([(SimpleNamespace(region=Region(0, 3, 0, 1)), entry)])
    assert np.concatenate(list(entry["stream"]))["x"].tolist() == [1.25, 2.25]


def test_positive_cell_overlap_distinguishes_area_from_touch():
    cells = shapely.box([0, 1], [0, 0], [1, 2], [1, 1])
    # The higher-priority footprint ends exactly at the second cell's edge.
    # A footprint extending to x=1.1 would overlap BOTH cells with positive area.
    higher = shapely.box(0, 0, 1, 1)
    blocked = ClaimGridFilter._positive_cell_overlap(higher, cells)
    assert blocked.tolist() == [True, False]


def test_positive_cell_overlap_falls_back_to_explicit_intersection(monkeypatch):
    higher = shapely.box(0, 0, 1.1, 1)
    cells = shapely.box([0], [0], [1], [1])
    monkeypatch.setattr(
        shapely,
        "relate_pattern",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    blocked = ClaimGridFilter._positive_cell_overlap(higher, cells)
    assert blocked.tolist() == [True]


def test_small_positive_area_overlap_still_blocks_whole_cell():
    dtype = [("x", "f8"), ("y", "f8")]
    higher = shapely.box(0, 0, 1.01, 1)
    entry = {
        "stream": iter([np.array([(1.5, 0.5), (2.5, 0.5)], dtype=dtype)]),
        "stream_type": "point-stream",
        "src_srs": "EPSG:4326",
        "claim_required": True,
        "accepted_geometry": shapely.to_wkt(shapely.box(1.01, 0, 3, 1)),
        "excluded_geometry": shapely.to_wkt(higher),
    }
    _hook().run([(SimpleNamespace(region=Region(0, 3, 0, 1)), entry)])
    assert np.concatenate(list(entry["stream"]))["x"].tolist() == [2.5]
