import json

import numpy as np
import pytest
from rasterio.windows import Window

from globato.hooks.sinks.multi_stack import MultiStackAccumulator, StackUpdate
from globato.hooks.transforms.point_pixels import FUSION_BAND_MAP, FUSION_BANDS


def _state(shape=(1, 1), *, z=0.0, count=0.0, weight=0.0):
    """Build a minimal additive FusionState for direct reducer tests."""
    count_arr = np.full(shape, count, dtype=np.float64)
    weight_sum = np.full(shape, count * weight, dtype=np.float64)
    z_weighted_sum = np.full(shape, count * weight * z, dtype=np.float64)

    state = {name: np.zeros(shape, dtype=np.float64) for name in FUSION_BANDS}
    state["count"] = count_arr
    state["weight_sum"] = weight_sum
    state["z_weighted_sum"] = z_weighted_sum
    state["z2_weighted_sum"] = np.full(shape, count * weight * (z**2), dtype=np.float64)
    state["x_weighted_sum"] = np.zeros(shape, dtype=np.float64)
    state["y_weighted_sum"] = np.zeros(shape, dtype=np.float64)
    state["weighted_uncertainty_sq"] = np.zeros(shape, dtype=np.float64)
    return state


def _accumulator(tmp_path, strategy="mixed", weight_threshold="0.25/0.5/1"):
    return MultiStackAccumulator(
        region=[0, 2, 0, 2],
        x_inc=1,
        y_inc=1,
        output_fn=str(tmp_path / "stack.tif"),
        strategy=strategy,
        weight_threshold=weight_threshold,
        crs="EPSG:4326",
        compress_state=False,
        state_fn=str(tmp_path / "state.tif"),
        resume=False,
    )


def _read_pixel(acc):
    data = acc.dataset.read(window=Window(0, 0, 1, 1))[:, 0, 0]
    return {name: data[index - 1] for name, index in FUSION_BAND_MAP.items()}


def test_mixed_stackupdate_distinguishes_replace_merge_and_reject(tmp_path):
    acc = _accumulator(tmp_path, strategy="mixed")
    try:
        # First observation enters an empty cell: accepted and replacing emptiness.
        first = acc.update_state(_state(z=10, count=1, weight=0.25), Window(0, 0, 1, 1))
        assert isinstance(first, StackUpdate)
        assert first.accepted.item() is True
        assert first.replaced.item() is True

        # Equal tier merges into the cell rather than replacing the prior source.
        equal = acc.update_state(_state(z=20, count=1, weight=0.25), Window(0, 0, 1, 1))
        assert equal.accepted.item() is True
        assert equal.replaced.item() is False

        pixel = _read_pixel(acc)
        assert pixel["count"] == pytest.approx(2.0)
        assert pixel["weight_sum"] == pytest.approx(0.5)
        assert pixel["z_weighted_sum"] == pytest.approx(7.5)

        # A lower tier is ignored entirely.
        lower = acc.update_state(_state(z=999, count=1, weight=0.1), Window(0, 0, 1, 1))
        assert lower.accepted.item() is False
        assert lower.replaced.item() is False
        assert _read_pixel(acc)["z_weighted_sum"] == pytest.approx(7.5)

        # A higher tier replaces all lower-tier accumulated state.
        higher = acc.update_state(
            _state(z=100, count=1, weight=1.0), Window(0, 0, 1, 1)
        )
        assert higher.accepted.item() is True
        assert higher.replaced.item() is True

        pixel = _read_pixel(acc)
        assert pixel["count"] == pytest.approx(1.0)
        assert pixel["weight_sum"] == pytest.approx(1.0)
        assert pixel["z_weighted_sum"] == pytest.approx(100.0)
    finally:
        acc.close()


def test_mean_accepts_every_valid_input_without_replacement(tmp_path):
    acc = _accumulator(tmp_path, strategy="mean")
    try:
        first = acc.update_state(_state(z=10, count=1, weight=1), Window(0, 0, 1, 1))
        second = acc.update_state(_state(z=20, count=1, weight=1), Window(0, 0, 1, 1))

        assert first.accepted.item() is True
        assert first.replaced.item() is False
        assert second.accepted.item() is True
        assert second.replaced.item() is False

        pixel = _read_pixel(acc)
        assert pixel["count"] == pytest.approx(2.0)
        assert pixel["weight_sum"] == pytest.approx(2.0)
        assert pixel["z_weighted_sum"] == pytest.approx(30.0)
    finally:
        acc.close()


# def test_supercede_equal_weight_replaces_and_is_order_sensitive(tmp_path):
#     acc = _accumulator(tmp_path, strategy="supercede")
#     try:
#         acc.update_state(_state(z=10, count=1, weight=1), Window(0, 0, 1, 1))
#         result = acc.update_state(_state(z=20, count=1, weight=1), Window(0, 0, 1, 1))

#         assert result.accepted.item() is True
#         assert result.replaced.item() is True
#         assert _read_pixel(acc)["z_weighted_sum"] == pytest.approx(20.0)
#     finally:
#         acc.close()


def test_empty_update_returns_false_stackupdate_without_touching_state(tmp_path):
    acc = _accumulator(tmp_path, strategy="mixed")
    try:
        result = acc.update_state(_state(count=0, weight=0), Window(0, 0, 1, 1))
        assert not np.any(result.accepted)
        assert not np.any(result.replaced)
        assert _read_pixel(acc)["count"] == 0
    finally:
        acc.close()


def test_fusion_state_persists_reducer_contract_metadata(tmp_path):
    acc = _accumulator(
        tmp_path,
        strategy="mixed",
        weight_threshold="1/0.25/0.5",
    )
    try:
        tags = acc.dataset.tags()
        assert tags["GLOBATO_STACK_STRATEGY"] == "mixed"
        assert json.loads(tags["GLOBATO_WEIGHT_TIERS"]) == [0.25, 0.5, 1.0]
    finally:
        acc.close()
