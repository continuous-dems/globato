"""Independent QC resolution: exact additive statistics and native delivery."""

from types import SimpleNamespace
import copy
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from globato.qc.stack.resampling import (
    accepted_stream,
    alignment,
    context_hooks,
    output_grid,
)
from globato.hooks.filters.stack_qc import StackQC
from globato.hooks.sinks.multi_stack import MultiStackHook
from globato.hooks.rasters.binary_cudem import BinaryCudemStepDown
from globato.hooks.transforms.point_pixels import FUSION_BANDS
from fetchez.spatial import Region


def source(tmp_path, shape=(19, 23), origin=(350000, 3750000)):
    y, x = np.indices(shape)
    count = ((x + y) % 3 + 1).astype("float64")
    weight = count * (1 + (x % 4) * 0.25)
    z = -100 + y * 0.02 + x * 0.03
    data = np.stack(
        [
            z * weight,
            count,
            weight,
            z * z * weight,
            weight * 0.25,
            weight * (origin[0] + x + 0.5),
            weight * (origin[1] - y - 0.5),
        ]
    )
    data[:, :2, :3] = 0
    flags = np.zeros(shape, "uint8")
    flags[4:8, 5:9] = 1
    profile = dict(
        driver="GTiff",
        width=shape[1],
        height=shape[0],
        count=7,
        dtype="float64",
        nodata=None,
        crs="EPSG:32611",
        transform=from_origin(*origin, 1, 1),
    )
    a = tmp_path / "source.tif"
    b = tmp_path / "mask.tif"
    with rasterio.open(a, "w", **profile) as ds:
        ds.write(data)
        ds.descriptions = FUSION_BANDS
    with rasterio.open(b, "w", **{**profile, "dtype": "uint8", "count": 1}) as ds:
        ds.write(flags, 1)
    return a, b, data, flags


def collect(stream):
    profile = next(stream)
    data = np.zeros((7, profile["height"], profile["width"]))
    for window, _, block, *_ in stream:
        rows = slice(int(window.row_off), int(window.row_off + window.height))
        cols = slice(int(window.col_off), int(window.col_off + window.width))
        data[:, rows, cols] += block
    return profile, data


@pytest.mark.parametrize("offset", [(0, 0), (2, 5), (-2, -5)])
def test_every_band_rejected_before_aggregation_with_partial_edges(tmp_path, offset):
    dy, dx = offset
    path, mask, data, flags = source(tmp_path, origin=(350000 + dx, 3750000 - dy))
    target = dict(
        driver="GTiff",
        width=9,
        height=8,
        count=7,
        dtype="float64",
        nodata=None,
        crs=rasterio.crs.CRS.from_epsg(32611),
        transform=from_origin(350000, 3750000, 3, 3),
    )
    profile, actual = collect(accepted_stream(path, mask, target))
    expected = np.zeros_like(actual)
    for y, x in np.ndindex(flags.shape):
        row, col = (y + dy) // 3, (x + dx) // 3
        if 0 <= row < 8 and 0 <= col < 9 and flags[y, x] == 0:
            expected[:, row, col] += data[:, y, x]
    np.testing.assert_array_equal(actual, expected)
    assert profile == target


def test_stream_crosses_target_block_boundary_and_bounds_fine_reads(
    tmp_path, monkeypatch
):
    path, mask, data, flags = source(tmp_path, shape=(6, 401))
    target = dict(
        driver="GTiff",
        width=134,
        height=2,
        count=7,
        dtype="float64",
        nodata=None,
        crs=rasterio.crs.CRS.from_epsg(32611),
        transform=from_origin(350000, 3750000, 3, 3),
    )
    _, actual = collect(accepted_stream(path, mask, target))
    expected = np.zeros_like(actual)
    for y, x in np.ndindex(flags.shape):
        if flags[y, x] == 0:
            expected[:, y // 3, x // 3] += data[:, y, x]
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=1e-10)
    np.testing.assert_array_equal(actual[1:3], expected[1:3])


@pytest.mark.parametrize(
    "value",
    [0, -1, True, "invalid", "1x", "1/0s", "0/9s", "1_9", "1/9", "nan", float("inf")],
)
def test_resolution_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="qc_res"):
        StackQC(qc_res=value)


def test_context_ladder_scales_proportionally_without_changing_final_hooks():
    stack = MultiStackHook(res="1s", crs="EPSG:4269")
    interp = BinaryCudemStepDown(resolutions="1s/3s/9s/15s", weights="2/1/.5")
    original = copy.deepcopy(interp.__dict__)
    fine, post = context_hooks(stack, [interp], ".111111111111s")
    assert fine is not stack and fine.res == pytest.approx(1 / 32400)
    assert post[0].resolutions[0] == fine.res
    np.testing.assert_allclose(
        post[0].resolutions, np.array(interp.resolutions) / 9, rtol=1e-11
    )
    assert interp.__dict__ == original and stack.res == "1s"
    assert context_hooks(stack, [interp], None)[0] is stack
    assert context_hooks(stack, [interp], "1s")[0] is stack
    for invalid in ("2s", ".4s"):
        with pytest.raises(ValueError, match="integer"):
            context_hooks(stack, [interp], invalid)


def test_incompatible_source_grids_and_invalid_masks_fail(tmp_path):
    path, mask, _, _ = source(tmp_path)
    target = dict(
        width=8,
        height=8,
        crs=rasterio.crs.CRS.from_epsg(32611),
        transform=from_origin(350000, 3750000, 3, 3),
    )
    with rasterio.open(path) as src:
        with pytest.raises(ValueError, match="aligned"):
            alignment(
                src, {**target, "transform": from_origin(350000.5, 3750000, 3, 3)}
            )
        with pytest.raises(ValueError, match="CRS"):
            alignment(src, {**target, "crs": rasterio.crs.CRS.from_epsg(4326)})
    with rasterio.open(mask, "r+") as ds:
        flags = ds.read(1)
        flags[0, 0] = 2
        ds.write(flags, 1)
    with pytest.raises(ValueError, match="rejection flags"):
        list(accepted_stream(path, mask, target))


def test_nine_to_one_matches_existing_experiment_with_float_roundoff_only(tmp_path):
    # Independent padded reshape reduction used by the supplied experiment.
    path, mask, data, flags = source(tmp_path, shape=(20, 23))
    target = dict(
        driver="GTiff",
        width=3,
        height=3,
        count=7,
        dtype="float64",
        nodata=None,
        crs=rasterio.crs.CRS.from_epsg(32611),
        transform=from_origin(350000, 3750000, 9, 9),
    )
    _, actual = collect(accepted_stream(path, mask, target))
    padded = np.zeros((7, 27, 27))
    data[:, flags == 1] = 0
    padded[:, :20, :23] = data
    expected = padded.reshape(7, 3, 9, 3, 9).sum(axis=(2, 4))
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=1e-10)
    np.testing.assert_array_equal(actual[1], expected[1])


def test_numeric_resolution_ending_in_zero_and_real_nad83_nesting():
    from globato.qc.stack.resampling import resolution

    assert resolution(10) == resolution("10") == 10
    stack = MultiStackHook(res="1s", crs="EPSG:4269+5703")
    region = Region(-118.55, -118.50, 33.77, 33.81, srs="EPSG:4269")
    fine = copy.copy(stack)
    fine.res = resolution(".111111111111s")
    target = output_grid(stack, region)
    source_grid = output_grid(fine, region)
    ds = SimpleNamespace(transform=source_grid["transform"], crs=source_grid["crs"])
    assert alignment(ds, target) == ((9, 9), (0, 0))


def test_sparse_occupied_index_skips_empty_fine_reads(tmp_path, monkeypatch):
    from globato.qc.stack import resampling

    path, mask, data, flags = source(tmp_path, shape=(270, 401))
    data[:, 256:, :] = 0
    data[:, :, 256:] = 0
    with rasterio.open(path, "r+") as ds:
        ds.write(data)
    target = dict(
        driver="GTiff",
        width=134,
        height=90,
        count=7,
        dtype="float64",
        nodata=None,
        crs=rasterio.crs.CRS.from_epsg(32611),
        transform=from_origin(350000, 3750000, 3, 3),
    )
    _, expected = collect(accepted_stream(path, mask, target))
    calls = []
    original = resampling.rasterio.open

    class Reader:
        def __init__(self, ds):
            self.ds = ds

        def __getattr__(self, key):
            return getattr(self.ds, key)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.ds.close()

        def read(self, *args, **kw):
            win = kw["window"]
            calls.append(win)
            assert win.height <= 256 and win.width <= 256
            return self.ds.read(*args, **kw)

    monkeypatch.setattr(
        resampling.rasterio, "open", lambda *a, **kw: Reader(original(*a, **kw))
    )
    _, actual = collect(accepted_stream(path, mask, target, active_blocks=["0,0"]))
    np.testing.assert_array_equal(actual, expected)
    assert calls and all(w.row_off < 256 and w.col_off < 256 for w in calls)


def test_fractional_arcseconds_are_easy_to_specify():
    from globato.qc.stack.resampling import resolution

    assert resolution("1/9s") == 1 / 32400
    assert resolution("1/3s") == 1 / 10800
    assert resolution(" 1 / 9s ") == 1 / 32400
    assert resolution("1/9s", crs="EPSG:4269+5703") == 1 / 32400
    assert resolution(".111111111111s") == pytest.approx(resolution("1/9s"))
    assert StackQC(qc_res="1/9s").qc_res == "1/9s"


def test_arcseconds_are_rejected_for_projected_grids():
    stack = MultiStackHook(res=3, crs="EPSG:32611")
    interp = BinaryCudemStepDown(resolutions="3/9", weights=".5")
    with pytest.raises(ValueError, match="projected grid"):
        context_hooks(stack, [interp], "1/9s")
    fine, post = context_hooks(stack, [interp], 1)
    assert fine.res == 1
    assert post[0].resolutions == [1, 3]
    assert stack.res == 3


@pytest.mark.parametrize("crs", ["EPSG:4269", "EPSG:32611"])
@pytest.mark.parametrize("value", ["3m", "3t"])
def test_qc_res_rejects_ambiguous_metres_suffixes(crs, value):
    from globato.qc.stack.resampling import resolution

    with pytest.raises(ValueError, match="m/t suffixes"):
        resolution(value, crs=crs)


@pytest.mark.parametrize("crs", ["EPSG:4269", "EPSG:4269+5703"])
@pytest.mark.parametrize("value", [3, "3", 1 / 32400])
def test_geographic_qc_res_rejects_bare_numbers_even_when_ratio_is_valid(crs, value):
    stack = MultiStackHook(res=3, crs=crs)
    interp = BinaryCudemStepDown(resolutions="3/9", weights=".5")
    with pytest.raises(
        ValueError, match="geographic grids require an arcsecond suffix"
    ):
        context_hooks(stack, [interp], value)


@pytest.mark.parametrize("value", ["1s", "1/2s", "1/3s", "1/4s", "1/9s", "0.2s"])
def test_one_arcsecond_dem_accepts_any_integer_subdivision(value):
    stack = MultiStackHook(res="1s", crs="EPSG:4269")
    interp = BinaryCudemStepDown(resolutions="1s/3s", weights=".5")
    fine, _ = context_hooks(stack, [interp], value)
    from globato.qc.stack.resampling import resolution

    ratio = (1 / 3600) / resolution(fine.res)
    assert ratio >= 1 and ratio == pytest.approx(round(ratio))


@pytest.mark.parametrize(
    "value, ratio", [("0.4s", "2.5"), ("2s", "0.5"), ("1e-309s", "inf")]
)
def test_nesting_error_explains_the_actual_dem_grid(value, ratio):
    stack = MultiStackHook(res="1s", crs="EPSG:4269")
    interp = BinaryCudemStepDown(resolutions="1s/3s", weights=".5")
    with pytest.raises(ValueError) as error:
        context_hooks(stack, [interp], value)
    text = str(error.value)
    assert "final DEM increment (1s)" in text
    assert f"DEM increment / qc_res = {ratio}" in text
    assert "positive integer" in text and "never coarser" in text


def test_projected_nesting_uses_native_linear_units():
    stack = MultiStackHook(res=30, crs="EPSG:32611")
    interp = BinaryCudemStepDown(resolutions="30/90", weights=".5")
    fine, _ = context_hooks(stack, [interp], 3)
    assert fine.res == 3
    with pytest.raises(ValueError, match="30 metre"):
        context_hooks(stack, [interp], 4)


@pytest.mark.parametrize("value", [10, 30, "10", "30"])
def test_native_owners_preserve_numeric_increments_ending_in_zero(value):
    from globato.hooks.transforms.point_pixels import Point2PixelStream
    from globato.hooks.sinks.side_stack import _resolve_grid

    region = Region(350000, 350300, 3749700, 3750000, srs="EPSG:32611")
    expected = float(value)
    points = Point2PixelStream(x_inc=value, y_inc=value)
    assert points.x_inc == points.y_inc == expected
    grid = _resolve_grid(region, value, "EPSG:32611")
    assert grid["resolution"] == [expected, expected]
    interp = BinaryCudemStepDown(resolutions=f"{value}/{expected * 3}")
    assert interp.resolutions == [expected, expected * 3]


def test_geographic_half_arcsecond_grid_has_identical_outer_dem_bounds():
    from globato.qc.stack.resampling import nested_region, resolution

    final = MultiStackHook(res="1s", crs="EPSG:4269+5703")
    fine = copy.copy(final)
    fine.res = resolution("1/2s")
    region = Region(
        -118.55, -118.55 + 6.4 / 3600, 33.81 - 6.4 / 3600, 33.81, srs="EPSG:4269"
    )
    target = output_grid(final, region)
    native, grid = nested_region(region, fine, target)
    assert (target["width"], target["height"]) == (6, 6)
    assert (grid["width"], grid["height"]) == (12, 12)
    assert rasterio.transform.array_bounds(
        grid["height"], grid["width"], grid["transform"]
    ) == rasterio.transform.array_bounds(
        target["height"], target["width"], target["transform"]
    )
    assert region.xmax != native.xmax  # original recipe AOI is not mutated


def test_half_arcsecond_ratio_rejects_shifted_cell_edges():
    target = dict(
        crs=rasterio.crs.CRS.from_epsg(4269),
        transform=from_origin(-118.55, 33.81, 1 / 3600, 1 / 3600),
    )
    shifted = SimpleNamespace(
        crs=target["crs"],
        transform=from_origin(-118.55 + 0.25 / 3600, 33.81, 0.5 / 3600, 0.5 / 3600),
    )
    with pytest.raises(ValueError, match="origin is not aligned"):
        alignment(shifted, target)


def test_accepted_decimal_spelling_uses_exact_final_grid_subdivision():
    stack = MultiStackHook(res="1s", crs="EPSG:4269")
    interp = BinaryCudemStepDown(resolutions="1s/3s/9s")
    fraction, _ = context_hooks(stack, [interp], "1/9s")
    decimal, post = context_hooks(stack, [interp], ".111111111111s")
    assert decimal.res == fraction.res == (1 / 3600) / 9
    assert post[0].resolutions[0] == decimal.res


@pytest.mark.parametrize("requested", ["1s", "1/9s"])
def test_explicit_same_or_finer_qc_checks_aoi_crs_before_discovery(requested):
    from globato.qc.stack.resampling import validate_request

    stack = MultiStackHook(res="1s", crs="EPSG:4269")
    interp = BinaryCudemStepDown(resolutions="1s/3s", weights=".5")
    region = Region(-118.55, -118.50, 33.77, 33.81, srs="EPSG:4326")
    with pytest.raises(ValueError, match="AOI CRS must match"):
        validate_request(requested, region, [stack, interp])
