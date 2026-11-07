import numpy as np
import pytest

from fetchez.spatial import Region

from globato.hooks.transforms.reproject import StreamReproject


POINT_DTYPE = [("x", "f8"), ("y", "f8"), ("z", "f8")]
SRC = "EPSG:4326+vdatum:mllw"
DST = "EPSG:4326+3855"


class FakeModule:
    name = "charts"

    def __init__(self, region):
        self.region = region


class FakeGrid:
    """Stands in for RasterQuery: a constant shift everywhere."""

    def query(self, x, y):
        return np.full(len(x), 1.5)


def make_chunks(xs, ys, size=2):
    pts = np.zeros(len(xs), dtype=POINT_DTYPE)
    pts["x"], pts["y"], pts["z"] = xs, ys, -10.0
    return (pts[i : i + size].copy() for i in range(0, len(pts), size))


@pytest.fixture
def hook(monkeypatch):
    h = StreamReproject(dst_srs=DST)
    monkeypatch.setattr(h, "is_point_stream", lambda entry: True)
    h.builds = []

    def fake_get_pipeline(src_srs, region=None):
        h.builds.append(src_srs)
        return (None, FakeGrid())

    monkeypatch.setattr(h, "_get_pipeline", fake_get_pipeline)
    return h


def run_hook(hook, region, xs, ys):
    entry = {"stream": make_chunks(xs, ys), "src_srs": SRC}
    [(_, out)] = hook.run([(FakeModule(region), entry)])
    assert out["src_srs"] == DST
    return np.concatenate(list(out["stream"]))


def test_no_grid_when_no_points_in_region(hook):
    """A wide-footprint file with no points in the region never builds a grid."""

    region = Region(-116.8, -116.45, 32.95, 33.3, srs="EPSG:4326")
    out = run_hook(hook, region, [-120.5, -121.0, -119.2], [33.5, 34.0, 32.0])

    assert hook.builds == []
    np.testing.assert_array_equal(out["z"], -10.0)


def test_empty_stream_builds_no_grid(hook):
    region = Region(-116.8, -116.45, 32.95, 33.3, srs="EPSG:4326")
    entry = {"stream": iter([]), "src_srs": SRC}
    [(_, out)] = hook.run([(FakeModule(region), entry)])

    assert list(out["stream"]) == []
    assert hook.builds == []


def test_grid_built_once_points_reach_region(hook):
    """Out-of-region chunks before the first in-region point stay unshifted."""

    region = Region(-117.3, -117.05, 32.6, 32.8, srs="EPSG:4326")
    xs = [-120.5, -121.0, -117.2, -117.1, -125.0, -126.0]
    ys = [33.5, 34.0, 32.7, 32.65, 40.0, 41.0]
    out = run_hook(hook, region, xs, ys)

    assert hook.builds == [SRC]
    # First chunk is outside the region and passes through with no shift; once
    # the grid exists, every later chunk is queried as before.
    np.testing.assert_array_equal(out["z"][:2], -10.0)
    np.testing.assert_array_equal(out["z"][2:], -8.5)


def test_grid_built_when_region_unknown(hook):
    out = run_hook(hook, None, [-120.5, -121.0], [33.5, 34.0])

    assert hook.builds == [SRC]
    np.testing.assert_array_equal(out["z"], -8.5)


def test_longitude_wrap_matches_raster_query():
    bounds = (-117.3, -117.05, 32.6, 32.8)
    chunk = np.zeros(1, dtype=POINT_DTYPE)
    chunk["x"], chunk["y"] = 360 - 117.2, 32.7

    assert StreamReproject._any_in_grid(chunk, bounds)
