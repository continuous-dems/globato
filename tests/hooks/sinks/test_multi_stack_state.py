import numpy as np
import rasterio

from globato.hooks.sinks.multi_stack import MultiStackAccumulator
from globato.hooks.transforms.point_pixels import FUSION_BANDS


POINT_DTYPE = [
    ("x", "f8"),
    ("y", "f8"),
    ("z", "f8"),
    ("w", "f8"),
    ("u", "f8"),
]


def make_points(rows):
    return np.array(rows, dtype=POINT_DTYPE)


def read_state(accumulator):
    accumulator.dataset.flush() if hasattr(accumulator.dataset, "flush") else None

    with rasterio.open(accumulator.state_fn) as src:
        data = src.read()

    return {band: data[i] for i, band in enumerate(FUSION_BANDS)}


def assert_state_equal(a, b):
    for band in FUSION_BANDS:
        np.testing.assert_allclose(
            a[band],
            b[band],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"FusionState band differs: {band}",
        )


def read_live_state(acc):
    data = acc.dataset.read()
    return {band: data[i] for i, band in enumerate(FUSION_BANDS)}


def make_accumulator(tmp_path, name, **kwargs):
    return MultiStackAccumulator(
        region=[0.0, 2.0, 0.0, 1.0],
        x_inc=1.0,
        y_inc=1.0,
        output_fn=str(tmp_path / f"{name}.tif"),
        state_fn=str(tmp_path / f"{name}.fusion.tif"),
        strategy=kwargs.pop("strategy", "mean"),
        resume=kwargs.pop("resume", False),
        **kwargs,
    )


def test_update_points_equals_update_state(tmp_path):
    """Point ingestion and direct FusionState ingestion must be equivalent."""

    points = make_points(
        [
            (0.20, 0.20, 10.0, 0.25, 2.0),
            (0.30, 0.30, 30.0, 0.75, 4.0),
        ]
    )

    direct = make_accumulator(tmp_path, "direct")
    state_acc = make_accumulator(tmp_path, "state")

    direct.update(points)

    state, window_tuple, _ = state_acc.pixel_binner.accumulate(points)

    from rasterio.windows import Window

    state_acc.update_state(
        state,
        Window(*window_tuple),
    )

    direct_state = read_state(direct)
    state_state = read_state(state_acc)

    assert_state_equal(direct_state, state_state)

    direct.close()
    state_acc.close()


def test_resume_matches_single_run(tmp_path):
    """Persistent state resumed across processes must equal one continuous run."""

    points_a = make_points(
        [
            (0.20, 0.20, 10.0, 0.25, 1.0),
            (0.30, 0.30, 20.0, 0.25, 1.0),
        ]
    )

    points_b = make_points(
        [
            (0.40, 0.40, 30.0, 0.50, 2.0),
            (1.20, 0.20, -50.0, 0.75, 3.0),
        ]
    )

    continuous = make_accumulator(tmp_path, "continuous")
    continuous.update(points_a)
    continuous.update(points_b)
    continuous.close()
    expected = read_state(continuous)

    state_fn = tmp_path / "resumed.fusion.tif"

    first = MultiStackAccumulator(
        region=[0.0, 2.0, 0.0, 1.0],
        x_inc=1.0,
        y_inc=1.0,
        output_fn=str(tmp_path / "resumed.tif"),
        state_fn=str(state_fn),
        strategy="mean",
        resume=False,
    )
    first.update(points_a)
    first.close()

    second = MultiStackAccumulator(
        region=[0.0, 2.0, 0.0, 1.0],
        x_inc=1.0,
        y_inc=1.0,
        output_fn=str(tmp_path / "resumed.tif"),
        state_fn=str(state_fn),
        strategy="mean",
        resume=True,
    )
    second.update(points_b)
    second.close()

    actual = read_state(second)

    assert_state_equal(expected, actual)

    second.close()


def one_cell_points(z, weight):
    return make_points(
        [
            (0.25, 0.25, z, weight, 0.0),
        ]
    )


def test_mixed_higher_tier_replaces(tmp_path):
    acc = MultiStackAccumulator(
        region=[0.0, 1.0, 0.0, 1.0],
        x_inc=1.0,
        y_inc=1.0,
        output_fn=str(tmp_path / "out.tif"),
        state_fn=str(tmp_path / "state.tif"),
        strategy="mixed",
        weight_threshold="0.5",
        resume=False,
    )

    acc.update(one_cell_points(10.0, 0.25))
    acc.update(one_cell_points(50.0, 0.75))

    state = read_live_state(acc)

    # Higher tier replaces, so count remains one and Z is 50.
    assert state["count"][0, 0] == 1.0
    assert state["z_weighted_sum"][0, 0] == 50.0 * 0.75

    acc.close()


def test_mixed_equal_tier_merges(tmp_path):
    acc = MultiStackAccumulator(
        region=[0.0, 1.0, 0.0, 1.0],
        x_inc=1.0,
        y_inc=1.0,
        output_fn=str(tmp_path / "out.tif"),
        state_fn=str(tmp_path / "state.tif"),
        strategy="mixed",
        weight_threshold="0.5",
        resume=False,
    )

    acc.update(one_cell_points(10.0, 0.75))
    acc.update(one_cell_points(50.0, 0.80))

    count = acc.dataset.read(2)

    assert count[0, 0] == 2.0

    acc.close()


def test_mixed_lower_tier_is_ignored(tmp_path):
    acc = MultiStackAccumulator(
        region=[0.0, 1.0, 0.0, 1.0],
        x_inc=1.0,
        y_inc=1.0,
        output_fn=str(tmp_path / "out.tif"),
        state_fn=str(tmp_path / "state.tif"),
        strategy="mixed",
        weight_threshold="0.5",
        resume=False,
    )

    acc.update(one_cell_points(50.0, 0.75))
    before = read_state(acc)

    acc.update(one_cell_points(10.0, 0.25))
    after = read_state(acc)

    assert_state_equal(before, after)

    acc.close()
