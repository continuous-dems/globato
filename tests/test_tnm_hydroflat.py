import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from globato.hooks.rasters.tnm_hydroflat import TNMHydroflat

NDV = -9999.0


def _write_raster(path, data):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype="float32",
        nodata=NDV,
        crs="EPSG:32610",
        transform=from_origin(0, data.shape[0], 1, 1),
    ) as dst:
        dst.write(data.astype("float32"), 1)


def _landmask_on_right(hook, monkeypatch, land_left, height, width):
    hook.hydroflat_seed_mode = "water"
    monkeypatch.setattr(hook, "_resolve_landmask", lambda src: "land.gpkg")
    monkeypatch.setattr(
        hook,
        "_read_land_geometries",
        lambda path, crs, **kwargs: [box(land_left, 0, width, height)],
    )


def test_defaults_are_the_validated_klamath_profile():
    hook = TNMHydroflat()

    assert hook.stage == "file"
    assert hook.landmask == "osm"
    assert hook.hydroflat_seed_mode == "raster"
    assert hook.land_class_field == "class"
    assert hook.land_class_value == "land"
    assert hook.water_seed_inset_m == 0
    assert hook.min_seed_area_m2 == 25
    assert hook.min_component_area_m2 == 1000
    assert hook.flat_min_elevation == -5
    assert hook.flat_max_elevation == 5
    assert hook.max_flat_values == 64
    assert hook.flat_tolerance == "auto"
    assert hook.auto_exact_min_fraction == 0.75
    assert hook.auto_fuzzy_min_fraction == 0.25
    assert hook.auto_fuzzy_tolerance_max == 0.35
    assert hook.auto_fuzzy_quantile == 0.999
    assert hook.auto_fuzzy_buffer_m == 2
    assert hook.fuzzy_water_barrier is False
    assert hook.fuzzy_require_exact_contact is True
    assert hook.connectivity == 8
    assert hook.clip_flats_to_water is False
    assert hook.seam_cleanup_m == 2
    assert hook.irregular_water_max_elevation == 4.5


def test_raster_seed_mode_removes_exact_flats_without_resolving_landmask(
    tmp_path,
    monkeypatch,
):
    src_path = tmp_path / "source.tif"
    dst_path = tmp_path / "filtered.tif"
    data = np.full((20, 30), 20.0, dtype="float32")
    data[:, :12] = 1.05
    _write_raster(src_path, data)

    hook = TNMHydroflat(
        min_seed_area_m2=5,
        min_component_area_m2=20,
        flat_tolerance=0.0,
        seam_cleanup_m=0,
    )

    def unexpected_landmask(_src):
        raise AssertionError("exact raster mode must not resolve a landmask")

    monkeypatch.setattr(hook, "_resolve_landmask", unexpected_landmask)

    assert hook.process_raster(src_path, dst_path, {})
    with rasterio.open(dst_path) as result:
        filtered = result.read(1)
        tags = result.tags()

    assert np.all(filtered[:, :12] == NDV)
    assert np.all(filtered[:, 12:] == data[:, 12:])
    assert tags["TNM_HYDROFLAT_MODE"] == "exact"
    assert tags["TNM_HYDROFLAT_SEED_MODE"] == "raster"
    assert tags["TNM_HYDROFLAT_LANDMASK_USED"] == "false"


def test_raster_seed_mode_removes_fuzzy_flats_without_resolving_landmask(
    tmp_path,
    monkeypatch,
):
    src_path = tmp_path / "source.tif"
    dst_path = tmp_path / "filtered.tif"
    data = np.full((30, 50), 20.0, dtype="float32")
    data[:, :8] = 1.05
    data[:, 8:40] = np.linspace(
        0.90,
        1.18,
        30 * 32,
        dtype="float32",
    ).reshape(30, 32)
    _write_raster(src_path, data)

    hook = TNMHydroflat(
        min_seed_area_m2=5,
        min_component_area_m2=20,
        seam_cleanup_m=0,
    )

    def unexpected_landmask(_src):
        raise AssertionError("fuzzy raster mode must not resolve a landmask")

    monkeypatch.setattr(hook, "_resolve_landmask", unexpected_landmask)

    assert hook.process_raster(src_path, dst_path, {})
    with rasterio.open(dst_path) as result:
        filtered = result.read(1)
        tags = result.tags()

    assert np.all(filtered[:, :40] == NDV)
    assert np.all(filtered[:, 40:] == data[:, 40:])
    assert tags["TNM_HYDROFLAT_MODE"] == "fuzzy"
    assert tags["TNM_HYDROFLAT_SEED_MODE"] == "raster"
    assert tags["TNM_HYDROFLAT_LANDMASK_USED"] == "false"


def test_raster_seed_mode_resolves_landmask_only_for_fallback(
    tmp_path,
    monkeypatch,
):
    src_path = tmp_path / "source.tif"
    dst_path = tmp_path / "filtered.tif"
    data = np.empty((20, 20), dtype="float32")
    data[:, :15] = np.arange(300, dtype="float32").reshape(20, 15) / 100.0
    data[:, 15:] = 10.0
    _write_raster(src_path, data)

    hook = TNMHydroflat(
        min_seed_area_m2=5,
        min_component_area_m2=10,
        irregular_water_max_elevation=2.0,
    )
    calls = []

    def resolve_landmask(_src):
        calls.append(True)
        return "land.gpkg"

    monkeypatch.setattr(hook, "_resolve_landmask", resolve_landmask)
    monkeypatch.setattr(
        hook,
        "_read_land_geometries",
        lambda path, crs, **kwargs: [box(15, 0, 20, 20)],
    )

    assert hook.process_raster(src_path, dst_path, {})
    with rasterio.open(dst_path) as result:
        filtered = result.read(1)
        tags = result.tags()

    expected_remove = np.zeros(data.shape, dtype=bool)
    expected_remove[:, :15] = data[:, :15] <= 2.0
    assert calls == [True]
    assert np.all(filtered[expected_remove] == NDV)
    assert np.all(filtered[:, 15:] == data[:, 15:])
    assert tags["TNM_HYDROFLAT_MODE"] == "fallback"
    assert tags["TNM_HYDROFLAT_SEED_MODE"] == "raster"
    assert tags["TNM_HYDROFLAT_LANDMASK_USED"] == "true"


def test_osm_resolution_transforms_projected_bounds_to_geographic(
    tmp_path,
    monkeypatch,
):
    src_path = tmp_path / "projected.tif"
    data = np.ones((10, 10), dtype="float32")
    with rasterio.open(
        src_path,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="float32",
        nodata=NDV,
        crs="EPSG:26910",
        transform=from_origin(380000, 4680000, 1, 1),
    ) as dst:
        dst.write(data, 1)

    hook = TNMHydroflat(local_tmp=str(tmp_path))
    captured = {}

    from fetchez.registry import ModuleRegistry

    import globato.utils

    monkeypatch.setattr(
        ModuleRegistry,
        "get_class",
        lambda name: object(),
    )

    def fake_resolve_barrier(barrier, *, region, **kwargs):
        captured["region"] = region
        return str(tmp_path / "landmask.geojson")

    monkeypatch.setattr(globato.utils, "resolve_barrier", fake_resolve_barrier)

    with rasterio.open(src_path) as src:
        assert hook._resolve_landmask(src).endswith("landmask.geojson")

    region = captured["region"]
    values = (region.xmin, region.xmax, region.ymin, region.ymax)
    assert np.all(np.isfinite(values))
    assert -180 <= region.xmin < region.xmax <= 180
    assert -90 <= region.ymin < region.ymax <= 90


def test_three_hydroflat_values_and_intervening_slivers_are_removed(
    tmp_path,
    monkeypatch,
):
    src_path = tmp_path / "source.tif"
    dst_path = tmp_path / "filtered.tif"
    data = np.full((20, 50), 20.0, dtype="float32")
    data[:, 0:8] = -0.8
    data[:, 9:17] = 0.5
    data[:, 18:26] = 1.0
    # Non-flat one-cell seams between separately encoded water plateaus.
    data[:, 8] = np.arange(20, dtype="float32") / 100.0
    data[:, 17] = np.arange(20, dtype="float32") / 100.0 + 0.2
    # Variable water farther than the two-metre collar must survive.
    data[:, 28:40] = np.arange(240, dtype="float32").reshape(20, 12) / 10.0
    _write_raster(src_path, data)

    hook = TNMHydroflat(
        min_seed_area_m2=5,
        min_component_area_m2=20,
        flat_tolerance=0.0,
    )
    _landmask_on_right(hook, monkeypatch, land_left=40, height=20, width=50)

    assert hook.process_raster(src_path, dst_path, {})
    with rasterio.open(dst_path) as result:
        filtered = result.read(1)
        tags = result.tags()

    assert np.all(filtered[:, 0:28] == NDV)
    assert np.all(filtered[:, 28:] == data[:, 28:])
    assert tags["TNM_HYDROFLAT_METHOD"] == "exact_hydroflat"
    assert int(tags["TNM_HYDROFLAT_COMPONENTS"]) == 3
    assert int(tags["TNM_HYDROFLAT_SEAM_CELLS"]) > 0
    assert sorted(
        float(value) for value in tags["TNM_HYDROFLAT_VALUES"].split(",")
    ) == pytest.approx([-0.8, 0.5, 1.0])


def test_auto_mode_preserves_exact_priority_for_klamath_style_tile(
    tmp_path,
    monkeypatch,
):
    src_path = tmp_path / "source.tif"
    dst_path = tmp_path / "filtered.tif"
    data = np.full((20, 50), 20.0, dtype="float32")
    data[:, 0:8] = -0.9
    data[:, 9:17] = 0.5
    data[:, 18:26] = 1.0
    data[:, 8] = np.arange(20, dtype="float32") / 100.0
    data[:, 17] = np.arange(20, dtype="float32") / 100.0 + 0.2
    # A small amount of variable water is not enough to demote the exact
    # interpretation or activate a broad fuzzy tolerance.
    data[:, 28:30] = np.linspace(
        0.85,
        1.15,
        40,
        dtype="float32",
    ).reshape(20, 2)
    _write_raster(src_path, data)

    hook = TNMHydroflat(
        flat_tolerance="auto",
        min_seed_area_m2=5,
        min_component_area_m2=20,
    )
    _landmask_on_right(hook, monkeypatch, land_left=40, height=20, width=50)

    assert hook.process_raster(src_path, dst_path, {})
    with rasterio.open(dst_path) as result:
        filtered = result.read(1)
        tags = result.tags()

    assert tags["TNM_HYDROFLAT_MODE"] == "exact"
    assert tags["TNM_HYDROFLAT_METHOD"] == "exact_hydroflat"
    assert int(tags["TNM_HYDROFLAT_FUZZY_CELLS"]) == 0
    assert tags["TNM_HYDROFLAT_FUZZY_TOLERANCE"] == "none"
    assert np.all(filtered[:, 28:30] == data[:, 28:30])


def test_auto_mode_removes_fuzzy_water_without_propagating_inland(
    tmp_path,
    monkeypatch,
):
    src_path = tmp_path / "source.tif"
    dst_path = tmp_path / "filtered.tif"
    data = np.full((30, 50), 20.0, dtype="float32")
    # Priority 1: a strongly supported exact water surface.
    data[:, 0:8] = 1.05
    # Priority 2: a spatially separate noisy hydroflat with unique values.
    data[:, 12:30] = np.linspace(
        0.90,
        1.18,
        540,
        dtype="float32",
    ).reshape(30, 18)
    # Similar low terrain lies on the land side. Only the configured two-metre
    # collar may be removed; the remaining inland cells must survive.
    data[:, 40:50] = 1.10
    _write_raster(src_path, data)

    hook = TNMHydroflat(
        flat_tolerance="auto",
        min_seed_area_m2=5,
        min_component_area_m2=20,
        fuzzy_water_barrier=True,
        fuzzy_require_exact_contact=False,
        seam_cleanup_m=0,
    )
    _landmask_on_right(hook, monkeypatch, land_left=40, height=30, width=50)

    assert hook.process_raster(src_path, dst_path, {})
    with rasterio.open(dst_path) as result:
        filtered = result.read(1)
        tags = result.tags()

    assert tags["TNM_HYDROFLAT_MODE"] == "fuzzy"
    assert tags["TNM_HYDROFLAT_METHOD"] == "exact_fuzzy_hydroflat"
    assert int(tags["TNM_HYDROFLAT_EXACT_CELLS"]) == 30 * 8
    assert int(tags["TNM_HYDROFLAT_FUZZY_CELLS"]) > 0
    assert 0 < float(tags["TNM_HYDROFLAT_FUZZY_TOLERANCE"]) <= 0.2
    assert np.all(filtered[:, 0:8] == NDV)
    assert np.count_nonzero(filtered[:, 12:30] != NDV) <= 3
    assert np.all(filtered[:, 42:50] == data[:, 42:50])


def test_barrier_free_fuzzy_mode_removes_complete_seeded_component(
    tmp_path,
    monkeypatch,
):
    src_path = tmp_path / "source.tif"
    dst_path = tmp_path / "filtered.tif"
    data = np.full((30, 50), 20.0, dtype="float32")
    # Priority-one exact evidence lies beneath vector water.
    data[:, 0:8] = 1.05
    # One continuous noisy hydroflat crosses six metres into vector land.
    data[:, 8:46] = np.linspace(
        0.90,
        1.18,
        30 * 38,
        dtype="float32",
    ).reshape(30, 38)
    # Similar low terrain is disconnected from the accepted hydroflat and
    # must survive even though the coastline barrier is disabled.
    data[:, 48:50] = 1.10
    _write_raster(src_path, data)

    hook = TNMHydroflat(
        flat_tolerance="auto",
        min_seed_area_m2=5,
        min_component_area_m2=20,
        fuzzy_water_barrier=False,
        fuzzy_require_exact_contact=True,
        seam_cleanup_m=0,
    )
    _landmask_on_right(hook, monkeypatch, land_left=40, height=30, width=50)

    assert hook.process_raster(src_path, dst_path, {})
    with rasterio.open(dst_path) as result:
        filtered = result.read(1)
        tags = result.tags()

    assert tags["TNM_HYDROFLAT_MODE"] == "fuzzy"
    assert tags["TNM_HYDROFLAT_FUZZY_WATER_BARRIER"] == "false"
    assert tags["TNM_HYDROFLAT_FUZZY_REQUIRE_EXACT_CONTACT"] == "true"
    assert np.all(filtered[:, 0:46] == NDV)
    assert np.all(filtered[:, 46:48] == data[:, 46:48])
    assert np.all(filtered[:, 48:50] == data[:, 48:50])


def test_barrier_free_fuzzy_mode_rejects_component_without_exact_contact(
    tmp_path,
    monkeypatch,
):
    src_path = tmp_path / "source.tif"
    dst_path = tmp_path / "filtered.tif"
    data = np.full((30, 50), 20.0, dtype="float32")
    data[:, 0:8] = 1.05
    # This noisy water component has seed support but is separated from the
    # exact hydroflat by four cells of unrelated elevation.
    data[:, 12:30] = np.linspace(
        0.90,
        1.18,
        30 * 18,
        dtype="float32",
    ).reshape(30, 18)
    _write_raster(src_path, data)

    hook = TNMHydroflat(
        flat_tolerance="auto",
        min_seed_area_m2=5,
        min_component_area_m2=20,
        fuzzy_water_barrier=False,
        fuzzy_require_exact_contact=True,
        seam_cleanup_m=0,
    )
    _landmask_on_right(hook, monkeypatch, land_left=40, height=30, width=50)

    assert hook.process_raster(src_path, dst_path, {})
    with rasterio.open(dst_path) as result:
        filtered = result.read(1)
        tags = result.tags()

    assert tags["TNM_HYDROFLAT_MODE"] == "exact"
    assert int(tags["TNM_HYDROFLAT_FUZZY_CELLS"]) == 0
    assert np.all(filtered[:, 0:8] == NDV)
    assert np.all(filtered[:, 12:30] == data[:, 12:30])


def test_confirmed_hydroflat_is_not_hard_clipped_to_landmask(
    tmp_path,
    monkeypatch,
):
    src_path = tmp_path / "source.tif"
    dst_path = tmp_path / "filtered.tif"
    data = np.full((20, 20), 10.0, dtype="float32")
    data[:, 0:17] = -0.9
    _write_raster(src_path, data)

    hook = TNMHydroflat(
        min_seed_area_m2=5,
        min_component_area_m2=10,
        flat_tolerance=0.0,
        seam_cleanup_m=0,
    )
    _landmask_on_right(hook, monkeypatch, land_left=15, height=20, width=20)

    assert hook.process_raster(src_path, dst_path, {})
    with rasterio.open(dst_path) as result:
        filtered = result.read(1)

    # The confirmed component crosses the imperfect vector shoreline by two
    # cells, and the full connected synthetic plateau is still removed.
    assert np.all(filtered[:, :17] == NDV)
    assert np.all(filtered[:, 17:] == data[:, 17:])


def test_optional_hard_clip_keeps_landmask_side_of_flat(
    tmp_path,
    monkeypatch,
):
    src_path = tmp_path / "source.tif"
    dst_path = tmp_path / "filtered.tif"
    data = np.full((20, 20), 10.0, dtype="float32")
    data[:, 0:17] = -0.9
    _write_raster(src_path, data)

    hook = TNMHydroflat(
        min_seed_area_m2=5,
        min_component_area_m2=10,
        flat_tolerance=0.0,
        clip_flats_to_water=True,
        seam_cleanup_m=0,
    )
    _landmask_on_right(hook, monkeypatch, land_left=15, height=20, width=20)

    assert hook.process_raster(src_path, dst_path, {})
    with rasterio.open(dst_path) as result:
        filtered = result.read(1)

    assert np.all(filtered[:, :15] == NDV)
    assert np.all(filtered[:, 15:] == data[:, 15:])


def test_irregular_water_uses_water_side_elevation_fallback(
    tmp_path,
    monkeypatch,
):
    src_path = tmp_path / "source.tif"
    dst_path = tmp_path / "filtered.tif"
    data = np.empty((20, 20), dtype="float32")
    data[:, :15] = np.arange(300, dtype="float32").reshape(20, 15) / 100.0
    data[:, 15:] = 1.0
    _write_raster(src_path, data)

    hook = TNMHydroflat(
        min_seed_area_m2=5,
        min_component_area_m2=10,
        irregular_water_max_elevation=2.0,
    )
    _landmask_on_right(hook, monkeypatch, land_left=15, height=20, width=20)

    assert hook.process_raster(src_path, dst_path, {})
    with rasterio.open(dst_path) as result:
        filtered = result.read(1)
        tags = result.tags()

    expected_remove = np.zeros(data.shape, dtype=bool)
    expected_remove[:, :15] = data[:, :15] <= 2.0
    assert np.all(filtered[expected_remove] == NDV)
    assert np.all(filtered[:, 15:] == data[:, 15:])
    assert tags["TNM_HYDROFLAT_METHOD"] == "fallback"
    assert int(tags["TNM_HYDROFLAT_FALLBACK_CELLS"]) > 0


def test_fallback_is_skipped_when_any_hydroflat_is_accepted(
    tmp_path,
    monkeypatch,
):
    src_path = tmp_path / "source.tif"
    dst_path = tmp_path / "filtered.tif"
    data = np.full((20, 20), 10.0, dtype="float32")
    data[:, :5] = -0.9
    data[:, 7:15] = np.arange(160, dtype="float32").reshape(20, 8) / 100.0
    data[:, 15:] = 1.0
    _write_raster(src_path, data)

    hook = TNMHydroflat(
        min_seed_area_m2=5,
        min_component_area_m2=10,
        flat_tolerance=0.0,
        seam_cleanup_m=0,
        irregular_water_max_elevation=2.0,
    )
    _landmask_on_right(hook, monkeypatch, land_left=15, height=20, width=20)

    assert hook.process_raster(src_path, dst_path, {})
    with rasterio.open(dst_path) as result:
        filtered = result.read(1)
        tags = result.tags()

    assert np.all(filtered[:, :5] == NDV)
    assert np.all(filtered[:, 5:] == data[:, 5:])
    assert int(tags["TNM_HYDROFLAT_FALLBACK_CELLS"]) == 0


def test_fallback_refuses_a_landmask_without_both_land_and_water(
    tmp_path,
    monkeypatch,
):
    src_path = tmp_path / "source.tif"
    dst_path = tmp_path / "filtered.tif"
    data = np.arange(400, dtype="float32").reshape(20, 20) / 100.0
    _write_raster(src_path, data)

    hook = TNMHydroflat(
        min_seed_area_m2=5,
        min_component_area_m2=10,
        irregular_water_max_elevation=5.0,
    )
    monkeypatch.setattr(hook, "_resolve_landmask", lambda src: "land.gpkg")
    monkeypatch.setattr(
        hook,
        "_read_land_geometries",
        lambda path, crs, **kwargs: [box(30, 0, 40, 20)],
    )

    assert not hook.process_raster(src_path, dst_path, {})
    assert not dst_path.exists()


def test_output_names_are_collision_safe_for_equal_basenames(tmp_path):
    first = tmp_path / "a" / "tile.tif"
    second = tmp_path / "b" / "tile.tif"
    hook = TNMHydroflat()

    first_output = hook._unique_output_path(first)
    second_output = hook._unique_output_path(second)

    assert first_output != second_output
    assert first_output.endswith("_hydroflat_clean.tif")
    assert second_output.endswith("_hydroflat_clean.tif")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"connectivity": 6}, "connectivity"),
        ({"hydroflat_seed_mode": "coast"}, "hydroflat_seed_mode"),
        (
            {"flat_min_elevation": 2, "flat_max_elevation": 1},
            "flat_min_elevation",
        ),
        ({"flat_tolerance": "wide"}, "flat_tolerance"),
        ({"auto_exact_min_fraction": 1.1}, "auto_exact_min_fraction"),
        ({"auto_fuzzy_min_fraction": -0.1}, "auto_fuzzy_min_fraction"),
        ({"auto_fuzzy_quantile": 0}, "auto_fuzzy_quantile"),
        ({"auto_fuzzy_tolerance_max": np.inf}, "finite"),
        ({"fuzzy_water_barrier": "maybe"}, "fuzzy_water_barrier"),
        (
            {"fuzzy_require_exact_contact": "maybe"},
            "fuzzy_require_exact_contact",
        ),
        (
            {
                "fuzzy_water_barrier": False,
                "fuzzy_require_exact_contact": False,
            },
            "fuzzy_require_exact_contact",
        ),
        ({"clip_flats_to_water": "maybe"}, "clip_flats_to_water"),
        ({"irregular_water_max_elevation": np.inf}, "finite"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TNMHydroflat(**kwargs)
