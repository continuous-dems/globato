# tests/test_binary_cudem

import numpy as np

from globato.hooks.rasters.binary_cudem import BinaryCudemStepDown


def _land_mask(z, w=None, threshold=1.0):
    """Run the observed-land policy on a tiny synthetic MultiStack."""
    if w is None:
        w = np.ones_like(z, dtype=float)

    hook = BinaryCudemStepDown()

    valid = np.isfinite(z)
    core = w >= threshold

    return hook._observed_land_mask(
        z=z,
        valid_mask=valid,
        core_mask=core,
    )


def test_observed_positive_data_is_land():
    """Positive, sufficiently weighted observations must disable the cap."""
    z = np.array(
        [
            [np.nan, np.nan, np.nan],
            [np.nan, 2.0, np.nan],
            [np.nan, np.nan, np.nan],
        ]
    )

    result = _land_mask(z)

    assert result[1, 1]


def test_low_weight_positive_data_does_not_override_cap():
    """Positive elevation alone is insufficient if it is below the tier weight."""
    z = np.array(
        [
            [np.nan, np.nan, np.nan],
            [np.nan, 2.0, np.nan],
            [np.nan, np.nan, np.nan],
        ]
    )

    w = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    result = _land_mask(z, w=w, threshold=1.0)

    assert not result[1, 1]


def test_small_gap_between_positive_lidar_cells_is_closed():
    """A tiny nodata gap in coherent positive lidar should relax the cap."""
    z = np.array(
        [
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, 1.0, 1.0, 1.0, np.nan],
            [np.nan, 1.0, np.nan, 1.0, np.nan],
            [np.nan, 1.0, 1.0, 1.0, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
    )

    result = _land_mask(z)

    # The central interpolation hole is surrounded by observed positive
    # elevation and should therefore inherit the no-cap policy.
    assert result[2, 2]


def test_separate_offshore_features_are_not_bridged():
    """Nearby positive features must not create an invented land connection."""
    z = np.full((7, 9), np.nan)

    # Two small offshore rocks/islands.
    z[2:5, 1:3] = 2.0
    z[2:5, 6:8] = 2.0

    result = _land_mask(z)

    # Original observations survive.
    assert np.all(result[2:5, 1:3])
    assert np.all(result[2:5, 6:8])

    # The ocean between them should remain constrained.
    assert not np.any(result[:, 4])


def test_isolated_offshore_rock_does_not_gain_large_halo():
    """An isolated observation should remain local rather than expanding outward."""
    z = np.full((9, 9), np.nan)
    z[4, 4] = 3.0

    result = _land_mask(z)

    assert result[4, 4]

    # Closing should not behave like dilation around an isolated point.
    assert result.sum() == 1


def test_negative_observations_do_not_disable_water_cap():
    """Reliable bathymetric observations remain water evidence."""
    z = np.array(
        [
            [np.nan, np.nan, np.nan],
            [np.nan, -1.5, np.nan],
            [np.nan, np.nan, np.nan],
        ]
    )

    result = _land_mask(z)

    assert not result[1, 1]


def test_zero_elevation_is_neutral():
    """Exactly zero elevation should not override the topology prior."""
    z = np.array(
        [
            [np.nan, np.nan, np.nan],
            [np.nan, 0.0, np.nan],
            [np.nan, np.nan, np.nan],
        ]
    )

    result = _land_mask(z)

    assert not result[1, 1]


def test_topological_cap_never_changes_observed_data():
    hook = BinaryCudemStepDown()

    z = np.array(
        [
            [2.5, 1.2],
            [1.5, -3.0],
        ],
        dtype=float,
    )

    # First row is real source data; second row represents interpolation.
    observed = np.array(
        [
            [True, True],
            [False, False],
        ]
    )

    cap = np.full(z.shape, -0.01)

    result = hook._apply_topological_cap(
        z.copy(),
        cap,
        observed,
        ndv=-9999,
    )

    # Positive observed lidar survives even though OSM says ocean.
    assert result[0, 0] == 2.5
    assert result[0, 1] == 1.2

    # Interpolated positive elevation gets constrained.
    assert result[1, 0] == -0.01

    # Existing interpolated bathymetry below the cap remains unchanged.
    assert result[1, 1] == -3.0


def test_larger_gap_is_not_forced_closed():
    """The policy should not reconstruct shoreline across unsupported gaps."""
    z = np.full((7, 7), np.nan)

    z[2:5, 1] = 1.0
    z[2:5, 5] = 1.0

    result = _land_mask(z)

    # These two areas are too far apart for the small closing operation
    # and should not become one artificial land feature.
    assert not result[3, 3]
