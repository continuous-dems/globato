"""Collection-stage side-stack assembly regressions."""

from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from fetchez.spatial import Region
from globato.hooks.tools.stack_combine import StackCombine
from globato.hooks.transforms.point_pixels import FUSION_BANDS, FUSION_STATE_VERSION


def _make_side_stack(path, z_sum, *, version=FUSION_STATE_VERSION, transform=None):
    data = np.zeros((len(FUSION_BANDS), 2, 2), dtype="float64")
    values = {
        "z_weighted_sum": z_sum,
        "count": 1,
        "weight_sum": 1,
        "z2_weighted_sum": z_sum * z_sum,
        "weighted_uncertainty_sq": 0,
        "x_weighted_sum": 0,
        "y_weighted_sum": 0,
    }
    for i, name in enumerate(FUSION_BANDS):
        data[i] = values[name]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=len(FUSION_BANDS),
        dtype="float64",
        crs="EPSG:4326",
        transform=transform or from_origin(0, 2, 1, 1),
    ) as dst:
        dst.write(data)
        for i, name in enumerate(FUSION_BANDS, 1):
            dst.set_band_description(i, name)
        dst.update_tags(
            GLOBATO_DATATYPE="FUSION_STATE",
            GLOBATO_FUSION_VERSION=str(version),
            GLOBATO_SIDESTACK_HASH=str(path),
        )


def _entries(*paths):
    mod = SimpleNamespace(region=Region(0, 2, 0, 2))
    return [
        (
            mod,
            {
                "artifacts": {"side-stack": str(path)},
                "url": str(path),
                "dst_fn": str(path),
                "data_type": "fusion-state",
            },
        )
        for path in paths
    ]


def test_combine_finalizes_and_registers_artifacts(tmp_path):
    a, b = tmp_path / "a.tif", tmp_path / "b.tif"
    _make_side_stack(a, 10)
    _make_side_stack(b, 30)
    entries = _entries(a, b, a)  # duplicate entry must not double count
    hook = StackCombine(res="1", output=str(tmp_path / "combined.tif"))
    result = hook.run(entries)
    assert result is entries
    artifacts = result[0][1]["artifacts"]
    with rasterio.open(artifacts["fusion-state"]) as src:
        assert src.tags()["GLOBATO_DATATYPE"] == "FUSION_STATE"
        count = src.read(FUSION_BANDS.index("count") + 1)
        np.testing.assert_array_equal(count, 2)
    with rasterio.open(artifacts["multi_stack"]) as src:
        assert src.tags()["GLOBATO_DATATYPE"] == "MULTI_STACK"
        np.testing.assert_allclose(src.read(1), 20)
    assert artifacts["stack_combine"] == artifacts["multi_stack"]
    assert all(
        e["artifacts"]["fusion-state"] == artifacts["fusion-state"] for _, e in entries
    )
    hook.teardown()  # safe if invoked by framework after run


def test_combine_rebuilds_from_current_source_collection(tmp_path):
    a, b = tmp_path / "a.tif", tmp_path / "b.tif"
    _make_side_stack(a, 10)
    _make_side_stack(b, 30)
    args = {"res": "1", "output": str(tmp_path / "out.tif")}
    _first = StackCombine(**args).run(_entries(a, b))
    second = StackCombine(**args).run(_entries(a))
    with rasterio.open(second[0][1]["artifacts"]["multi_stack"]) as src:
        np.testing.assert_allclose(src.read(1), 10)
    with rasterio.open(second[0][1]["artifacts"]["fusion-state"]) as src:
        np.testing.assert_array_equal(src.read(FUSION_BANDS.index("count") + 1), 1)


def test_incompatible_source_does_not_replace_previous_outputs(tmp_path):
    a, b = tmp_path / "a.tif", tmp_path / "bad.tif"
    _make_side_stack(a, 10)
    _make_side_stack(b, 20, version=FUSION_STATE_VERSION + 1)
    args = {"res": "1", "output": str(tmp_path / "out.tif")}
    first = StackCombine(**args).run(_entries(a))
    output = first[0][1]["artifacts"]["multi_stack"]
    with pytest.raises(ValueError, match="FusionState version"):
        StackCombine(**args).run(_entries(a, b))
    with rasterio.open(output) as src:
        np.testing.assert_allclose(src.read(1), 10)


def test_no_sidestacks_passes_entries_through(tmp_path):
    entries = [(SimpleNamespace(region=Region(0, 2, 0, 2)), {"artifacts": {}})]
    assert StackCombine(output=str(tmp_path / "out.tif")).run(entries) is entries


@pytest.mark.parametrize("strategy", ["mixed", "supercede"])
def test_priority_strategies_use_existing_reducer(tmp_path, strategy):
    a, b = tmp_path / "a.tif", tmp_path / "b.tif"
    _make_side_stack(a, 10)
    _make_side_stack(b, 30)
    # Equal source weights: mixed combines; supercede keeps first.
    result = StackCombine(
        res="1",
        output=str(tmp_path / "out.tif"),
        strategy=strategy,
    ).run(_entries(a, b))
    with rasterio.open(result[0][1]["artifacts"]["multi_stack"]) as src:
        np.testing.assert_allclose(src.read(1), 20 if strategy == "mixed" else 10)


def test_grid_mismatch_is_rejected(tmp_path):
    a = tmp_path / "wrong-grid.tif"
    _make_side_stack(a, 10, transform=from_origin(0.5, 2, 1, 1))
    with pytest.raises(ValueError, match="geotransform"):
        StackCombine(res="1", output=str(tmp_path / "out.tif")).run(_entries(a))
