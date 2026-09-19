import numpy as np
import pytest

from fetchez.spatial import Region

from globato.hooks.transforms.point_pixels import (
    FUSION_BANDS,
    PointPixels,
    finalize_fusion_state,
    merge_fusion_states,
)


POINT_DTYPE = [
    ("x", "f8"),
    ("y", "f8"),
    ("z", "f8"),
    ("w", "f8"),
    ("u", "f8"),
]


def make_points(rows):
    return np.array(rows, dtype=POINT_DTYPE)


@pytest.fixture
def points():
    return make_points(
        [
            # Pixel 0
            (0.20, 0.20, 10.0, 0.10, 1.0),
            (0.30, 0.30, 20.0, 0.30, 2.0),
            (0.40, 0.40, 40.0, 0.60, 3.0),
            # Pixel 1
            (1.20, 0.20, -50.0, 0.25, 4.0),
            (1.30, 0.30, -30.0, 0.75, 2.0),
        ]
    )


@pytest.fixture
def pixels():
    return PointPixels(
        src_region=Region(0.0, 2.0, 0.0, 1.0),
        x_size=2,
        y_size=1,
    )


def assert_state_equal(a, b):
    for band in FUSION_BANDS:
        np.testing.assert_allclose(
            a[band],
            b[band],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"FusionState band differs: {band}",
        )


def test_accumulate_is_associative(points, pixels):
    """Chunking must not change FusionState."""

    whole, whole_window, _ = pixels.accumulate(points)

    a, a_window, _ = pixels.accumulate(points[:3])
    b, b_window, _ = pixels.accumulate(points[3:])

    # This fixture intentionally splits on pixel boundaries, so local
    # states have different windows. Embed both into the whole window
    # before comparing.
    merged = {band: np.zeros_like(whole[band]) for band in FUSION_BANDS}

    for state, window in ((a, a_window), (b, b_window)):
        col, row, width, height = window
        wcol, wrow, _, _ = whole_window

        x0 = col - wcol
        y0 = row - wrow

        for band in FUSION_BANDS:
            merged[band][y0 : y0 + height, x0 : x0 + width] += state[band]

    assert_state_equal(whole, merged)


def test_merge_fusion_states_is_additive():
    """Merging compatible same-shaped states must be pure addition."""

    state_a = {band: np.array([[float(i + 1)]]) for i, band in enumerate(FUSION_BANDS)}

    state_b = {
        band: np.array([[float((i + 1) * 10)]]) for i, band in enumerate(FUSION_BANDS)
    }

    merged = merge_fusion_states(state_a, state_b)

    for band in FUSION_BANDS:
        np.testing.assert_allclose(
            merged[band],
            state_a[band] + state_b[band],
        )


def test_count_matches_fusion_state(points, pixels):
    """The optimized count path must agree with FusionState count."""

    state, state_window, _ = pixels.accumulate(points)
    count, count_window, _ = pixels.count(points)

    assert count_window == state_window
    np.testing.assert_array_equal(
        count,
        state["count"],
    )


def test_coverage_matches_count(points, pixels):
    """Coverage is exactly count > 0."""

    count, count_window, _ = pixels.count(points)
    coverage, coverage_window, _ = pixels.coverage(points)

    assert coverage_window == count_window

    np.testing.assert_array_equal(
        coverage,
        count > 0,
    )


def test_known_weighted_statistics():
    """Finalization must reproduce hand-computable weighted statistics."""

    points = make_points(
        [
            (0.20, 0.20, 10.0, 0.25, 2.0),
            (0.30, 0.30, 30.0, 0.75, 4.0),
        ]
    )

    pixels = PointPixels(
        src_region=Region(0.0, 1.0, 0.0, 1.0),
        x_size=1,
        y_size=1,
    )

    state, _, _ = pixels.accumulate(points)
    final = finalize_fusion_state(state)

    # Weighted mean:
    # (10 * .25 + 30 * .75) / 1 = 25
    assert final.z[0, 0] == pytest.approx(25.0)

    # Total weight = 1, count = 2, mean weight = .5
    assert final.weight_sum[0, 0] == pytest.approx(1.0)
    assert final.count[0, 0] == pytest.approx(2.0)
    assert final.mean_weight[0, 0] == pytest.approx(0.5)

    # Weighted second moment:
    # (10^2*.25 + 30^2*.75) / 1 = 700
    #
    # variance = 700 - 25^2 = 75
    assert final.stddev[0, 0] == pytest.approx(np.sqrt(75.0))

    # Propagated uncertainty:
    # sqrt((.25*2)^2 + (.75*4)^2) / 1
    expected_u = np.sqrt(0.5**2 + 3.0**2)

    assert final.uncertainty[0, 0] == pytest.approx(expected_u)


def test_source_weight_scales_weighted_statistics():
    """source_weight must scale effective point weights consistently."""

    points = make_points(
        [
            (0.20, 0.20, 10.0, 0.5, 1.0),
            (0.30, 0.30, 20.0, 0.5, 1.0),
        ]
    )

    pixels = PointPixels(
        src_region=Region(0.0, 1.0, 0.0, 1.0),
        x_size=1,
        y_size=1,
    )

    base, _, _ = pixels.accumulate(points)
    scaled, _, _ = pixels.accumulate(
        points,
        source_weight=2.0,
    )

    # Counts are unaffected by source weight.
    np.testing.assert_allclose(
        scaled["count"],
        base["count"],
    )

    # First-order weighted sums scale linearly.
    for band in (
        "weight_sum",
        "z_weighted_sum",
        "z2_weighted_sum",
        "x_weighted_sum",
        "y_weighted_sum",
    ):
        np.testing.assert_allclose(
            scaled[band],
            base[band] * 2.0,
        )

    # Uncertainty numerator contains w^2, so it scales quadratically.
    np.testing.assert_allclose(
        scaled["weighted_uncertainty_sq"],
        base["weighted_uncertainty_sq"] * 4.0,
    )


def test_nonfinite_xyz_are_excluded():
    """Invalid spatial/elevation observations must not enter FusionState."""

    points = make_points(
        [
            (0.20, 0.20, 10.0, 1.0, 0.0),
            (np.nan, 0.20, 20.0, 1.0, 0.0),
            (0.20, np.inf, 30.0, 1.0, 0.0),
            (0.20, 0.20, np.nan, 1.0, 0.0),
        ]
    )

    pixels = PointPixels(
        src_region=Region(0.0, 1.0, 0.0, 1.0),
        x_size=1,
        y_size=1,
    )

    state, _, _ = pixels.accumulate(points)

    assert state["count"][0, 0] == pytest.approx(1.0)
    assert state["z_weighted_sum"][0, 0] == pytest.approx(10.0)


def test_invalid_weight_and_uncertainty_use_defaults():
    """Non-finite w/u follow the documented fallback behavior."""

    points = make_points(
        [
            (0.20, 0.20, 10.0, np.nan, np.nan),
        ]
    )

    pixels = PointPixels(
        src_region=Region(0.0, 1.0, 0.0, 1.0),
        x_size=1,
        y_size=1,
    )

    state, _, _ = pixels.accumulate(points)

    assert state["weight_sum"][0, 0] == pytest.approx(1.0)
    assert state["weighted_uncertainty_sq"][0, 0] == pytest.approx(0.0)


# def test_state_is_persisted_on_close(tmp_path):
#     acc = make_accumulator(tmp_path, "state")

#     acc.update(one_cell_points(10.0, 0.75))

#     state_fn = acc.state_fn
#     live_count = acc.dataset.read(
#         FUSION_BAND_MAP["count"]
#     ).copy()

#     acc.close()

#     with rasterio.open(state_fn) as src:
#         disk_count = src.read(
#             FUSION_BAND_MAP["count"]
#         )

#     np.testing.assert_array_equal(
#         disk_count,
#         live_count,
#     )
