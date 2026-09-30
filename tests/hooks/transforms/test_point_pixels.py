import numpy as np
import pytest
from rasterio import Affine

from globato.hooks.transforms.point_pixels import PointPixels


def _points(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    return np.rec.fromarrays(
        [
            xs,
            ys,
            np.ones(xs.size, dtype=np.float64),
        ],
        names=["x", "y", "z"],
    )


def test_point_pixels_preserves_explicit_geotransform():
    """An explicit grid transform must be authoritative."""

    gt = (-121.05, 1.0 / 3600.0, 0.0, 35.5, 0.0, -(1.0 / 3600.0))

    reducer = PointPixels(
        src_region=[-121.05, -121.025, 35.475, 35.5],
        x_size=90,
        y_size=90,
        dst_gt=gt,
    )

    # The supplied transform must survive untouched rather than being
    # reconstructed from region bounds and grid dimensions.
    assert reducer.dst_gt == gt


def test_point_pixels_accepts_affine_as_authoritative_transform():
    """Rasterio Affine and GDAL tuple inputs should normalize identically."""

    gt = (-121.05, 1.0 / 3600.0, 0.0, 35.5, 0.0, -(1.0 / 3600.0))
    affine = Affine.from_gdal(*gt)

    from_tuple = PointPixels(
        src_region=[-121.05, -121.025, 35.475, 35.5],
        x_size=90,
        y_size=90,
        dst_gt=gt,
    )
    from_affine = PointPixels(
        src_region=[-121.05, -121.025, 35.475, 35.5],
        x_size=90,
        y_size=90,
        dst_gt=affine,
    )

    assert from_affine.dst_gt == pytest.approx(from_tuple.dst_gt)


def test_points_on_pixel_boundaries_have_stable_indices():
    """Exact boundaries and adjacent floats must fall in predictable cells."""

    gt = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)

    reducer = PointPixels(
        src_region=[0.0, 4.0, 0.0, 4.0],
        x_size=4,
        y_size=4,
        dst_gt=gt,
    )

    # Interior corner at x=2, y=2.
    #
    # With floor-based binning:
    #   exact x boundary belongs to the cell on its east/right side
    #   exact y boundary belongs to the cell on its south/below side
    #
    # np.nextafter gives us the immediately representable float on either side.

    transform = Affine.from_gdal(*gt)
    x_edge, y_edge = transform * (2, 2)

    x_west = np.nextafter(x_edge, -np.inf)
    x_east = np.nextafter(x_edge, np.inf)

    y_south = np.nextafter(y_edge, -np.inf)
    y_north = np.nextafter(y_edge, np.inf)

    cases = [
        # x, y, expected col, expected row
        (x_edge, y_edge, 2, 2),
        (x_west, y_edge, 1, 2),
        (x_east, y_edge, 2, 2),
        (x_edge, y_north, 2, 1),
        (x_edge, y_south, 2, 2),
    ]

    for x, y, expected_col, expected_row in cases:
        prepared = reducer._prepare(_points([x], [y]))

        assert prepared is not None
        assert prepared.pixel_x.item() == expected_col
        assert prepared.pixel_y.item() == expected_row


def test_points_on_geographic_pixel_boundaries_are_stable():
    """Exercise the 1-arc-second geographic case that exposed the regression."""

    x0 = -121.05
    y0 = 35.5
    res = 1.0 / 3600.0

    gt = (x0, res, 0.0, y0, 0.0, -res)

    reducer = PointPixels(
        src_region=[-121.05, -121.025, 35.475, 35.5],
        x_size=90,
        y_size=90,
        dst_gt=gt,
    )

    # Pick an interior grid intersection well away from the outer boundary.
    col = 27
    row = 43

    transform = Affine.from_gdal(*gt)
    x_edge, y_edge = transform * (col, row)

    prepared = reducer._prepare(_points([x_edge], [y_edge]))

    assert prepared is not None
    assert prepared.pixel_x.item() == col
    assert prepared.pixel_y.item() == row
