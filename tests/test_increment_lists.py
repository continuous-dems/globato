"""Tests for slash-delimited increment lists and x/y increment pairs."""

import pytest
from fetchez.spatial import Region

from globato.hooks.rasters.binary_cudem import BinaryCudemStepDown
from globato.hooks.rasters.cudem import CudemStepDown
from globato.hooks.sinks.simple_stack import SimpleStack

ARCSEC = 1 / 3600


@pytest.mark.parametrize(
    "resolutions, expected",
    [
        ("(1/3)s/1s/3s", [1 / 3, 1, 3]),
        ("(1/9)s / (1/3)s / 1s", [1 / 9, 1 / 3, 1]),
        (["(1/3)s", "1s", "3s"], [1 / 3, 1, 3]),
        ("1s", [1]),
    ],
)
def test_cudem_resolution_list(resolutions, expected):
    hook = CudemStepDown(
        resolutions=resolutions, weights="1/0.5/0.25", blend_dist="5/5/5"
    )
    assert hook.resolutions[: len(expected)] == pytest.approx(
        [v * ARCSEC for v in expected]
    )
    assert hook.weights == [1.0, 0.5, 0.25]


@pytest.mark.parametrize(
    "resolutions, expected",
    [
        ("(1/9)s/(1/3)s/1s", [1 / 9, 1 / 3, 1]),
        (["1s", "3s"], [1, 3]),
    ],
)
def test_binary_cudem_resolution_list(resolutions, expected):
    hook = BinaryCudemStepDown(resolutions=resolutions)
    assert hook.resolutions == pytest.approx([v * ARCSEC for v in expected])


def test_binary_cudem_no_resolutions():
    assert BinaryCudemStepDown().resolutions == []


@pytest.mark.parametrize("resolutions", ["(1/9s/1s", "1s/abc", "1s//3s"])
def test_invalid_resolution_list(resolutions):
    with pytest.raises(ValueError):
        CudemStepDown(resolutions=resolutions)
    with pytest.raises(ValueError):
        BinaryCudemStepDown(resolutions=resolutions)


@pytest.mark.parametrize(
    "res, expected",
    [
        ("1s", (ARCSEC, ARCSEC)),
        ("1s/2s", (ARCSEC, 2 * ARCSEC)),
        ("(1/9)s/(1/3)s", (ARCSEC / 9, ARCSEC / 3)),
        ("0.001/0.002", (0.001, 0.002)),
        (0.001, (0.001, 0.001)),
    ],
)
def test_simple_stack_res(tmp_path, res, expected):
    hook = SimpleStack(res=res)
    acc = hook._create_accumulator(
        str(tmp_path / "stack.tif"), Region(0, 0.01, 0, 0.01, srs="EPSG:4326")
    )
    assert (acc.x_inc, acc.y_inc) == pytest.approx(expected)


@pytest.mark.parametrize("res", ["abc", "1s/2s/3s"])
def test_simple_stack_invalid_res(tmp_path, res):
    hook = SimpleStack(res=res)
    with pytest.raises(ValueError):
        hook._create_accumulator(
            str(tmp_path / "stack.tif"), Region(0, 0.01, 0, 0.01, srs="EPSG:4326")
        )
