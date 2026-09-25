import os
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from globato.hooks.metadata.stack_provenance import (
    DiskStackTierState,
    MemoryStackTierState,
    StackProvenance,
)
from globato.hooks.transforms.point_pixels import FUSION_BAND_MAP, FUSION_BANDS


WIDTH = 4
HEIGHT = 4
TRANSFORM = from_origin(0, 4, 1, 1)
CRS = "EPSG:4326"
TIERS = "0.25/0.5/1"


def _fusion_array(count, weight_sum):
    data = np.zeros((len(FUSION_BANDS), HEIGHT, WIDTH), dtype=np.float64)
    data[FUSION_BAND_MAP["count"] - 1] = count
    data[FUSION_BAND_MAP["weight_sum"] - 1] = weight_sum
    return data


def _write_fusion_state(path, count, weight_sum, *, strategy="mixed"):
    profile = {
        "driver": "GTiff",
        "dtype": "float64",
        "count": len(FUSION_BANDS),
        "width": WIDTH,
        "height": HEIGHT,
        "transform": TRANSFORM,
        "crs": CRS,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(_fusion_array(count, weight_sum))
        for name, index in FUSION_BAND_MAP.items():
            dst.set_band_description(index, name)
        dst.update_tags(
            GLOBATO_DATATYPE="FUSION_STATE",
            GLOBATO_STACK_STRATEGY=strategy,
            GLOBATO_WEIGHT_TIERS="[0.25, 0.5, 1.0]",
        )
    return str(path)


def _source_chunk(weights):
    """Create one full-grid raster-stream FusionState chunk for a source."""
    count = (weights > 0).astype(np.float64)
    data = _fusion_array(count, weights.astype(np.float64))
    window = Window(0, 0, WIDTH, HEIGHT)
    return window, window, data, None, TRANSFORM


def _drain(entry):
    list(entry["stream"])


def _mask_data(hook, dataset_id):
    path = hook._final_masks.masks[dataset_id]
    assert os.path.exists(path)
    with rasterio.open(path) as src:
        return src.read(1)


def _run_mixed_case(tmp_path, storage):
    tmp_path.mkdir(parents=True, exist_ok=True)

    # Final reducer result:
    #   [0,0] B(1.0) supersedes A(0.5)
    #   [0,1] A(0.5) and B(0.5) merge at the same winning tier
    #   [1,0] only A(0.25)
    #   [1,1] only B(0.25)
    final_count = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    final_wsum = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    final_count[0, 0], final_wsum[0, 0] = 1, 1.0
    final_count[0, 1], final_wsum[0, 1] = 2, 1.0
    final_count[1, 0], final_wsum[1, 0] = 1, 0.25
    final_count[1, 1], final_wsum[1, 1] = 1, 0.25

    fusion = _write_fusion_state(
        tmp_path / f"fusion_{storage}.tif",
        final_count,
        final_wsum,
    )

    a = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    a[0, 0] = 0.5
    a[0, 1] = 0.5
    a[1, 0] = 0.25

    b = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    b[0, 0] = 1.0
    b[0, 1] = 0.5
    b[1, 1] = 0.25

    hook = StackProvenance(
        output=str(tmp_path / f"stack_sources_{storage}.vrt"),
        output_dir=str(tmp_path / f"masks_{storage}"),
        state_dir=str(tmp_path / f"state_{storage}"),
        storage=storage,
        compress_state=False,
    )

    mod_a = SimpleNamespace(
        name="a",
        title="Source A",
        meta_category="test",
        meta_agency="TEST",
        weight=1.0,
    )
    mod_b = SimpleNamespace(
        name="b",
        title="Source B",
        meta_category="test",
        meta_agency="TEST",
        weight=1.0,
    )

    entry_a = {
        "checksum": "source-a",
        "dst_fn": "source_a.tif",
        "data_type": "test",
        "artifacts": {"fusion-state": fusion},
        "stream": iter([_source_chunk(a)]),
    }
    entry_b = {
        "checksum": "source-b",
        "dst_fn": "source_b.tif",
        "data_type": "test",
        "artifacts": {"fusion-state": fusion},
        "stream": iter([_source_chunk(b)]),
    }

    hook.run([(mod_a, entry_a)])
    _drain(entry_a)
    hook.run([(mod_b, entry_b)])
    _drain(entry_b)
    hook.teardown()

    return hook, entry_a, entry_b


@pytest.mark.parametrize("storage", ["disk", "memory"])
def test_mixed_provenance_tracks_surviving_sources_not_parsed_coverage(
    tmp_path, storage
):
    hook, _, _ = _run_mixed_case(tmp_path, storage)

    expected_a = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    expected_a[0, 1] = 1  # equal-tier merge survives
    expected_a[1, 0] = 1  # sole contributor survives

    expected_b = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    expected_b[0, 0] = 1  # higher tier wins
    expected_b[0, 1] = 1  # equal-tier merge survives
    expected_b[1, 1] = 1  # sole contributor survives

    # A did provide [0,0], but it must NOT appear in stack provenance because B
    # superseded it there. This is the semantic distinction from source_masks.
    np.testing.assert_array_equal(_mask_data(hook, "source-a"), expected_a)
    np.testing.assert_array_equal(_mask_data(hook, "source-b"), expected_b)


def test_disk_and_memory_backends_materialize_identical_masks(tmp_path):
    disk, _, _ = _run_mixed_case(tmp_path / "disk_run", "disk")
    memory, _, _ = _run_mixed_case(tmp_path / "memory_run", "memory")

    for dataset_id in ("source-a", "source-b"):
        np.testing.assert_array_equal(
            _mask_data(disk, dataset_id),
            _mask_data(memory, dataset_id),
        )


def test_storage_backend_artifact_contract(tmp_path):
    disk, disk_a, _ = _run_mixed_case(tmp_path / "disk", "disk")
    memory, memory_a, _ = _run_mixed_case(tmp_path / "memory", "memory")

    state_path = disk_a["artifacts"]["stack-provenance-state"]
    assert os.path.exists(state_path)
    assert "stack-provenance-state" not in memory_a["artifacts"]
    assert isinstance(disk._tier_state, DiskStackTierState)
    assert isinstance(memory._tier_state, MemoryStackTierState)


def test_disk_backend_resume_preserves_highest_tier_seen(tmp_path):
    count = np.ones((HEIGHT, WIDTH), dtype=np.float64)
    wsum = np.full((HEIGHT, WIDTH), 0.25, dtype=np.float64)
    fusion = _write_fusion_state(tmp_path / "fusion.tif", count, wsum)
    state_dir = tmp_path / "states"

    first = DiskStackTierState(fusion, str(state_dir), resume=True, compress=False)
    path = first.register("source-a")
    first.update(path, Window(0, 0, 1, 1), np.array([[2]], dtype=np.uint8))

    resumed = DiskStackTierState(fusion, str(state_dir), resume=True, compress=False)
    same_path = resumed.register("source-a")
    assert same_path == path
    resumed.update(same_path, Window(0, 0, 1, 1), np.array([[4]], dtype=np.uint8))
    resumed.update(same_path, Window(0, 0, 1, 1), np.array([[3]], dtype=np.uint8))

    with rasterio.open(same_path) as src:
        assert src.read(1, window=Window(0, 0, 1, 1)).item() == 4


def test_mean_strategy_keeps_every_observed_source_cell(tmp_path):
    final_count = np.ones((HEIGHT, WIDTH), dtype=np.float64)
    final_wsum = np.ones((HEIGHT, WIDTH), dtype=np.float64)
    fusion = _write_fusion_state(
        tmp_path / "fusion_mean.tif",
        final_count,
        final_wsum,
        strategy="mean",
    )

    weights = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    weights[0, 0] = 0.1
    weights[2, 3] = 2.0

    hook = StackProvenance(
        output=str(tmp_path / "mean.vrt"),
        output_dir=str(tmp_path / "mean_masks"),
        state_dir=str(tmp_path / "mean_state"),
        storage="memory",
    )
    mod = SimpleNamespace(name="mean-source", title="Mean Source", weight=1.0)
    entry = {
        "checksum": "mean-source",
        "dst_fn": "mean_source.tif",
        "artifacts": {"fusion-state": fusion},
        "stream": iter([_source_chunk(weights)]),
    }

    hook.run([(mod, entry)])
    _drain(entry)
    hook.teardown()

    expected = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    expected[0, 0] = 1
    expected[2, 3] = 1
    np.testing.assert_array_equal(_mask_data(hook, "mean-source"), expected)


def test_supercede_is_deliberately_not_reconstructed(tmp_path):
    count = np.ones((HEIGHT, WIDTH), dtype=np.float64)
    wsum = np.ones((HEIGHT, WIDTH), dtype=np.float64)
    fusion = _write_fusion_state(
        tmp_path / "fusion_supercede.tif",
        count,
        wsum,
        strategy="supercede",
    )

    hook = StackProvenance(
        output=str(tmp_path / "supercede.vrt"),
        storage="memory",
    )
    mod = SimpleNamespace(name="source", title="Source", weight=1.0)
    entry = {
        "checksum": "source",
        "dst_fn": "source.tif",
        "artifacts": {"fusion-state": fusion},
        "stream": iter([_source_chunk(np.ones((HEIGHT, WIDTH)))]),
    }

    hook.run([(mod, entry)])
    _drain(entry)
    hook.teardown()

    assert len(hook._tier_state) == 0
    assert not os.path.exists(hook.output)


def test_invalid_storage_is_rejected():
    with pytest.raises(ValueError, match="storage must be 'disk' or 'memory'"):
        StackProvenance(storage="ramdisk")


def test_memory_backend_reports_predictable_dense_state_size(tmp_path):
    count = np.ones((HEIGHT, WIDTH), dtype=np.float64)
    wsum = np.ones((HEIGHT, WIDTH), dtype=np.float64)
    fusion = _write_fusion_state(tmp_path / "fusion.tif", count, wsum)

    state = MemoryStackTierState(fusion)
    state.register("a")
    state.register("b")

    assert state.bytes_used == WIDTH * HEIGHT * 2  # uint8: one byte/cell/source
