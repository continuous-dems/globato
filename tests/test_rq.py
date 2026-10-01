import pytest
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from fetchez.spatial import Region
from fetchez.modules.local_fs import LocalFS

from globato.hooks.filters import rq as rq_module
from globato.hooks.filters.rq import ReferenceQuality


def make_entry(**overrides):
    entry = {
        "url": "https://example.com/data.tif",
        "dst_fn": "/tmp/data.tif",
        "data_type": "raster",
    }
    entry.update(overrides)
    return entry


@pytest.fixture
def module():
    return LocalFS(src_region=Region(0, 2, 0, 2))


def write_raster(path, data, *, transform=None, nodata=None):
    data = np.asarray(data, dtype=np.float32)
    transform = transform or from_origin(0, data.shape[0], 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def test_reference_identity_changes_when_source_artifact_changes(tmp_path, module):
    source = tmp_path / "reference.tif"
    write_raster(source, np.ones((2, 2), dtype=np.float32))

    hook = ReferenceQuality(reference="gmrt")  # .setup(module, make_entry())
    hook.wgs_region = Region(0, 2, 0, 2, srs="EPSG:4326")
    first = hook._reference_identity([source])

    time.sleep(0.001)
    write_raster(source, np.full((2, 2), 2, dtype=np.float32))

    second = hook._reference_identity([source])

    assert first != second


def test_reference_identity_is_stable_for_same_inputs(tmp_path):
    source = tmp_path / "reference.tif"
    write_raster(source, np.ones((2, 2), dtype=np.float32))

    hook = ReferenceQuality(reference="gmrt")
    hook.wgs_region = Region(0, 2, 0, 2, srs="EPSG:4326")

    assert hook._reference_identity([source]) == hook._reference_identity([source])


def test_grid_builder_uses_rq_cache_and_floor_dimensions(tmp_path, monkeypatch):
    source = tmp_path / "reference.tif"
    write_raster(source, np.ones((10, 10), dtype=np.float32))

    hook = ReferenceQuality(reference=str(source), res=3)
    hook.wgs_region = Region(0, 10.1, 0, 6.1, srs="EPSG:4326")

    calls = []

    class FakeGridEngine:
        @staticmethod
        def load_and_interpolate(files, region, nx, ny):
            calls.append((list(files), nx, ny))
            return np.ones((ny, nx), dtype=np.float32)

    class FakeGridWriter:
        @staticmethod
        def write(path, data, region):
            write_raster(
                path,
                data,
                transform=from_origin(
                    region.xmin,
                    region.ymax,
                    (region.xmax - region.xmin) / data.shape[1],
                    (region.ymax - region.ymin) / data.shape[0],
                ),
            )

    monkeypatch.setattr(rq_module, "GridEngine", FakeGridEngine)
    monkeypatch.setattr(rq_module, "GridWriter", FakeGridWriter)
    monkeypatch.setattr(rq_module, "HAS_GRID_ENGINE", True)

    first = hook._build_grid([source], hook.wgs_region, tmp_path)

    assert first is not None
    assert Path(first).parent == tmp_path / "rq"
    assert calls == [([source], 3, 2)]

    second = hook._build_grid([source], hook.wgs_region, tmp_path)

    assert second == first
    assert len(calls) == 1


def test_grid_builder_uses_new_artifact_when_source_changes(tmp_path, monkeypatch):
    source = tmp_path / "reference.tif"
    write_raster(source, np.ones((4, 4), dtype=np.float32))

    hook = ReferenceQuality(reference=str(source), res=1)
    hook.wgs_region = Region(0, 4, 0, 4, srs="EPSG:4326")

    calls = []

    class FakeGridEngine:
        @staticmethod
        def load_and_interpolate(files, region, nx, ny):
            calls.append(1)
            return np.ones((ny, nx), dtype=np.float32)

    class FakeGridWriter:
        @staticmethod
        def write(path, data, region):
            write_raster(path, data)

    monkeypatch.setattr(rq_module, "GridEngine", FakeGridEngine)
    monkeypatch.setattr(rq_module, "GridWriter", FakeGridWriter)
    monkeypatch.setattr(rq_module, "HAS_GRID_ENGINE", True)

    first = hook._build_grid([source], hook.wgs_region, tmp_path)

    time.sleep(0.001)
    write_raster(source, np.full((4, 4), 2, dtype=np.float32))

    second = hook._build_grid([source], hook.wgs_region, tmp_path)

    assert first != second
    assert len(calls) == 2
    assert Path(first).parent == Path(second).parent == tmp_path / "rq"


def test_grid_builder_overwrite_rebuilds_same_artifact(tmp_path, monkeypatch):
    source = tmp_path / "reference.tif"
    write_raster(source, np.ones((4, 4), dtype=np.float32))

    hook = ReferenceQuality(reference=str(source), res=1, overwrite=True)
    hook.wgs_region = Region(0, 4, 0, 4, srs="EPSG:4326")

    calls = []

    class FakeGridEngine:
        @staticmethod
        def load_and_interpolate(files, region, nx, ny):
            calls.append((nx, ny))
            return np.ones((ny, nx), dtype=np.float32)

    class FakeGridWriter:
        @staticmethod
        def write(path, data, region):
            write_raster(path, data)

    monkeypatch.setattr(rq_module, "GridEngine", FakeGridEngine)
    monkeypatch.setattr(rq_module, "GridWriter", FakeGridWriter)
    monkeypatch.setattr(rq_module, "HAS_GRID_ENGINE", True)

    first = hook._build_grid([source], hook.wgs_region, tmp_path)
    second = hook._build_grid([source], hook.wgs_region, tmp_path)

    assert first == second
    assert calls == [(4, 4), (4, 4)]


def test_reference_is_invalid_for_missing_empty_or_unreadable_file(tmp_path):
    missing = tmp_path / "missing.tif"
    empty = tmp_path / "empty.tif"
    empty.touch()
    corrupt = tmp_path / "corrupt.tif"
    corrupt.write_bytes(b"not a raster")

    assert not ReferenceQuality._reference_is_valid(missing)
    assert not ReferenceQuality._reference_is_valid(empty)
    assert not ReferenceQuality._reference_is_valid(corrupt)


def test_filter_chunk_does_not_clip_out_of_bounds_points(tmp_path):
    source = tmp_path / "reference.tif"
    data = np.full((2, 2), 10, dtype=np.float32)
    write_raster(source, data, transform=from_origin(0, 2, 1, 1))

    hook = ReferenceQuality(reference=str(source), threshold=1, mode="diff")
    hook.src = rasterio.open(source)
    hook.inv_transform = ~hook.src.transform
    hook.ref_data = hook.src.read(1).astype(np.float64)

    chunk = {
        "x": np.array([0.5, -0.5, 2.5], dtype=float),
        "y": np.array([1.5, 1.5, 1.5], dtype=float),
        "z": np.array([10.0, 100.0, 100.0], dtype=float),
    }

    try:
        result = hook.filter_chunk(chunk)
    finally:
        hook.src.close()

    # The interior point matches the reference; both outside points are not
    # evaluated and therefore must not be rejected merely because they are
    # farther from the reference surface.
    np.testing.assert_array_equal(result, np.array([False, False, False]))
    assert hook.out_of_bounds_points == 2
    assert hook.dropped_points == 0


def test_filter_chunk_counts_nodata_reference_separately(tmp_path):
    source = tmp_path / "reference.tif"
    data = np.array([[10, -9999], [10, 10]], dtype=np.float32)
    write_raster(
        source,
        data,
        transform=from_origin(0, 2, 1, 1),
        nodata=-9999,
    )

    hook = ReferenceQuality(reference=str(source), threshold=1, mode="diff")
    hook.src = rasterio.open(source)
    hook.inv_transform = ~hook.src.transform
    raw = hook.src.read(1).astype(np.float64)
    hook.ref_data = np.where(np.isclose(raw, hook.src.nodata), np.nan, raw)

    chunk = np.rec.fromarrays(
        [
            np.array([0.5, 1.5]),
            np.array([1.5, 1.5]),
            np.array([10.0, 100.0]),
        ],
        names=["x", "y", "z"],
    )

    assert isinstance(chunk, np.recarray)

    try:
        result = hook.filter_chunk(chunk)
    finally:
        hook.src.close()

    np.testing.assert_array_equal(result, np.array([False, False]))
    assert hook.invalid_reference_points == 1
    assert hook.dropped_points == 0
