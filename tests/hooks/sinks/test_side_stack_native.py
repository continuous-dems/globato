"""Native additive side stacks and exact immutable cache publication."""

from types import SimpleNamespace
import numpy as np
import pytest
import rasterio
from fetchez.spatial import Region
from globato.hooks.sinks.side_stack import (
    SIDESTACK_CACHE_VERSION,
    SideStackGenerate,
)
from globato.hooks.transforms.point_pixels import Point2PixelStream


def native_points(z):
    data = np.zeros(
        len(z), dtype=[("x", "f8"), ("y", "f8"), ("z", "f8"), ("w", "f8"), ("u", "f8")]
    )
    data["x"], data["y"], data["z"] = 350001.5, 3749998.5, z
    data["w"], data["u"] = 2, 0.5
    return data


def test_side_stack_adds_overlapping_chunks_and_reuses_exact_state(tmp_path):
    region = Region(350000, 350012, 3749988, 3750000, srs="EPSG:32611")
    module = SimpleNamespace(
        region=region, weight=1, uncertainty=0, hooks=[], _outdir=str(tmp_path)
    )
    entry = dict(
        url="file://overlap",
        src_srs="EPSG:32611",
        data_type="xyz",
        stream_type="point-stream",
        stream=iter([native_points([-10]), native_points([-20, -30])]),
    )
    writer = SideStackGenerate(res="3", crs="EPSG:32611")
    writer.run([(module, entry)])
    emitted = list(entry["stream"])
    with rasterio.open(entry["artifacts"]["side-stack"]) as ds:
        cached = ds.read()
        assert ds.tags()["GLOBATO_POINT_COUNT"] == "3"
    summed = np.zeros_like(cached)
    for window, _, data, *_ in emitted[1:]:
        summed[
            :,
            int(window.row_off) : int(window.row_off + window.height),
            int(window.col_off) : int(window.col_off + window.width),
        ] += data
    np.testing.assert_array_equal(cached, summed)
    valid = cached[1] > 0
    assert cached[1][valid].item() == 3
    assert (cached[0][valid] / cached[2][valid]).item() == -20
    # The schema still has seven additive bands; legacy overwrite caches have
    # a different cache key and cannot be reused as additive caches.
    with rasterio.open(entry["artifacts"]["side-stack"]) as ds:
        assert ds.tags()["GLOBATO_SIDESTACK_VERSION"] == str(SIDESTACK_CACHE_VERSION)


def test_existing_fusion_stream_uses_the_same_side_stack_writer(tmp_path):
    region = Region(350000, 350012, 3749988, 3750000, srs="EPSG:32611")
    module = SimpleNamespace(
        region=region, weight=1, uncertainty=0, hooks=[], _outdir=str(tmp_path)
    )
    entry = dict(
        url="file://fusion",
        src_srs="EPSG:32611",
        data_type="xyz",
        stream_type="point-stream",
        stream=iter([native_points([-10])]),
    )
    Point2PixelStream(x_inc="3", y_inc="3").run([(module, entry)])
    SideStackGenerate(res="3", crs="EPSG:32611").run([(module, entry)])
    list(entry["stream"])
    with rasterio.open(entry["artifacts"]["side-stack"]) as ds:
        assert ds.count == 7 and ds.read(2).sum() == 1


def test_side_stack_refuses_untransformed_crs(tmp_path):
    region = Region(350000, 350012, 3749988, 3750000, srs="EPSG:32611")
    module = SimpleNamespace(
        region=region, weight=1, uncertainty=0, hooks=[], _outdir=str(tmp_path)
    )
    entry = dict(
        url="file://wrong_crs",
        src_srs="EPSG:4326",
        data_type="xyz",
        stream_type="point-stream",
        stream=iter([native_points([-10])]),
    )
    SideStackGenerate(res="3", crs="EPSG:32611").run([(module, entry)])
    with pytest.raises(ValueError, match="stream_reproject"):
        list(entry["stream"])


def test_publishing_same_source_identity_does_not_replace_a_valid_cache(tmp_path):
    import os

    region = Region(350000, 350012, 3749988, 3750000, srs="EPSG:32611")
    module = SimpleNamespace(
        region=region, weight=1, uncertainty=0, hooks=[], _outdir=str(tmp_path)
    )

    def entry():
        return dict(
            url="file://immutable",
            src_srs="EPSG:32611",
            data_type="xyz",
            stream_type="point-stream",
            stream=iter([native_points([-10, -20])]),
        )

    first = entry()
    SideStackGenerate(res="3", crs="EPSG:32611").run([(module, first)])
    list(first["stream"])
    cache = first["artifacts"]["side-stack"]
    os.utime(cache, ns=(1_000_000_000, 1_000_000_000))
    before = os.stat(cache)
    second = entry()
    SideStackGenerate(res="3", crs="EPSG:32611").run([(module, second)])
    list(second["stream"])
    after = os.stat(cache)
    assert (before.st_ino, before.st_mtime_ns, before.st_size) == (
        after.st_ino,
        after.st_mtime_ns,
        after.st_size,
    )
    assert second["artifacts"]["side-stack"] == cache


def test_first_download_identity_is_reusable_on_the_next_lookup(tmp_path):
    from globato.hooks.sinks.side_stack import SideStackCheck

    region = Region(350000, 350012, 3749988, 3750000, srs="EPSG:32611")
    module = SimpleNamespace(
        region=region, weight=1, uncertainty=0, hooks=[], _outdir=str(tmp_path)
    )
    source = tmp_path / "download.xyz"

    def entry():
        return dict(
            url="https://example.invalid/survey.xyz",
            dst_fn=str(source),
            src_srs="EPSG:32611",
            data_type="xyz",
        )

    first = entry()
    SideStackCheck(res="3", crs="EPSG:32611").run([(module, first)])
    assert not first["_sidestack_hit"]
    # Fetchez materializes the source AFTER manifest-stage cache lookup.
    source.write_text("downloaded bytes")
    first.update(stream_type="point-stream", stream=iter([native_points([-10])]))
    SideStackGenerate(res="3", crs="EPSG:32611").run([(module, first)])
    list(first["stream"])
    second = entry()
    SideStackCheck(res="3", crs="EPSG:32611").run([(module, second)])
    assert second["_sidestack_hit"]
    assert first["artifacts"]["side-stack"] == second["_sidestack_path"]
