"""Wave rejection preserves beach slopes and agrees with independent fits."""

import numpy as np


from globato.coast_refine._science.coast_refine_v6_6_1.coast_refine_core import (
    _broad_wave_surface,
)


def _wave(shape=(250, 250)):
    y, x = np.indices(shape, dtype="float64")
    # Slowly varying mean water level plus intersecting short wave fronts;
    # positive, nonflat and with enough 15 m relief to pass the old screen.
    return (2.0 + 0.00015 * x + 0.19 * np.sin(x / 3.4) + 0.18 * np.cos(y / 4.7)).astype(
        "float32"
    )


def test_coherent_sloping_beach_does_not_become_wave_surface():
    y, x = np.indices((250, 250))
    beach = (0.2 + 0.012 * x + 0.005 * y + 0.005 * np.sin(x / 5.0)).astype("float32")
    wave = _broad_wave_surface(
        beach,
        np.ones(beach.shape, bool),
        dx_m=3,
        dy_m=3,
        seed_window_m=15,
        min_seed_area_m2=400,
        max_seed_relief_m=0.02,
    )
    assert not wave.any(), "stable terrain profile must be retained"


def test_wave_return_classified_even_without_flat_hydroflat_seeds():
    z = _wave()
    wave = _broad_wave_surface(
        z,
        np.ones(z.shape, bool),
        dx_m=3,
        dy_m=3,
        seed_window_m=15,
        min_seed_area_m2=400,
        max_seed_relief_m=0.02,
    )
    assert wave.sum() > 0.45 * z.size
    assert wave[
        :, 0
    ].any()  # observed source-edge cells now have masked profile support


def test_masked_plane_matches_independent_least_squares_at_block_seams():
    from globato.coast_refine._science.coast_refine_v6_6_1.coast_refine_core import (
        _incoherent_profile_windows,
    )

    rng = np.random.default_rng(42)
    y, x = np.indices((730, 650), dtype="float64")
    z = 0.01 * x + 0.004 * y + 0.19 * np.sin(x / 3.4) + 0.18 * np.cos(y / 4.7)
    valid = rng.random(z.shape) > 0.25
    actual = _incoherent_profile_windows(
        z,
        valid,
        half_x=10,
        half_y=10,
        max_seed_relief_m=0.02,
        max_explained_fraction=0.5,
        min_cells=30,
    )
    for r, c in [(511, 255), (512, 256), (513, 257), (0, 200), (200, 0), (700, 620)]:
        r0, r1 = max(0, r - 10), min(z.shape[0], r + 11)
        c0, c1 = max(0, c - 10), min(z.shape[1], c + 11)
        v = valid[r0:r1, c0:c1]
        yy, xx = np.indices(v.shape)
        values = z[r0:r1, c0:c1][v]
        design = np.column_stack([np.ones(v.sum()), xx[v], yy[v]])
        fit = design @ np.linalg.lstsq(design, values, rcond=None)[0]
        variance = np.var(values)
        explained = variance - np.mean((values - fit) ** 2)
        expected = bool(
            valid[r, c]
            and len(values) >= 30
            and variance > 0.02**2
            and explained < 0.5 * variance
        )
        assert bool(actual[r, c]) == expected
