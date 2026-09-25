import numpy as np
import pytest

from fetchez.spatial import Region
from globato.hooks.transforms.point_pixels import PointPixels


@pytest.mark.parametrize(
    "region",
    [
        # Standard CRM-style quarter-degree region with a 10% buffer.
        # The buffered extent is exactly 0.3 degrees in both dimensions,
        # or 1080 cells at 1 arc-second.
        Region(-117.25, -117.0, 28.0, 28.25).buffer(10),
        # Exercise different coordinate values so this is not accidentally
        # dependent on one particular floating-point representation.
        Region(-123.75, -123.50, 36.25, 36.50).buffer(10),
        Region(-80.50, -80.25, 24.50, 24.75).buffer(10),
    ],
)
def test_point_pixels_preserves_buffered_one_arcsecond_grid(region):
    """PointPixels must use the same grid represented by the requested increment.

    Buffered geographic regions commonly contain coordinates that cannot be
    represented exactly as binary floating-point values. Grid dimensions must
    therefore retain the intended cell count rather than losing a row/column
    through truncation.
    """
    inc = 1.0 / 3600.0

    xcount, ycount, grid_gt = region.geo_transform(
        x_inc=inc,
        y_inc=inc,
        node="grid",
    )

    # 0.25-degree tile buffered by 10% becomes 0.30 degrees:
    #
    #     0.30 * 3600 = 1080 cells
    #
    assert xcount == 1080
    assert ycount == 1080

    pixels = PointPixels(
        src_region=region,
        x_size=xcount,
        y_size=ycount,
    )
    pixels.init_gt()

    assert np.allclose(
        pixels.dst_gt,
        grid_gt,
        rtol=0.0,
        atol=1e-14,
    )

    assert pixels.dst_gt[1] == pytest.approx(inc, abs=1e-14)
    assert pixels.dst_gt[5] == pytest.approx(-inc, abs=1e-14)


def test_point_pixels_bins_buffered_region_without_registration_drift():
    """Points near opposite edges must remain registered to the 1s output grid."""
    region = Region(-117.25, -117.0, 28.0, 28.25).buffer(10)
    inc = 1.0 / 3600.0

    xcount, ycount, gt = region.geo_transform(
        x_inc=inc,
        y_inc=inc,
        node="grid",
    )

    assert (xcount, ycount) == (1080, 1080)

    pixels = PointPixels(
        src_region=region,
        x_size=xcount,
        y_size=ycount,
    )

    # Test cells well apart so any progressive registration drift is exposed.
    expected_cells = [
        (0, 0),
        (100, 100),
        (540, 540),
        (1000, 1000),
        (1079, 1079),
    ]

    xs = []
    ys = []
    zs = []

    for col, row in expected_cells:
        # Geographic center of the requested output cell.
        x = gt[0] + (col + 0.5) * gt[1]
        y = gt[3] + (row + 0.5) * gt[5]

        xs.append(x)
        ys.append(y)
        zs.append(float(row * xcount + col))

    points = np.rec.fromarrays(
        [xs, ys, zs],
        names=["x", "y", "z"],
    )

    prepared = pixels._prepare(points)

    assert prepared is not None

    actual_cells = list(zip(prepared.pixel_x, prepared.pixel_y))
    assert actual_cells == expected_cells
