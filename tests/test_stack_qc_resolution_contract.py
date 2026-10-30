"""Document existing resolution semantics without changing filtering science."""

from types import SimpleNamespace

import pytest
from affine import Affine
from rasterio.crs import CRS

from globato.qc.stack.grid import grid_offset, metres
from globato.qc.stack import geometry, plane


def test_different_source_and_output_grids_require_aggregation():
    origin = (float(-118.55), float(33.81))

    def native(seconds):
        step = seconds / 3600
        return SimpleNamespace(
            transform=Affine(step, 0, origin[0], 0, -step, origin[1]),
            crs=CRS.from_epsg(4269),
            width=90,
            height=90,
        )

    fine = native(1 / 9)
    coarse = native(1)
    assert grid_offset(fine, fine) == (0, 0)
    with pytest.raises(ValueError, match="resolution/orientation"):
        grid_offset(fine, coarse)
    dx_f, dy_f = metres(fine)
    dx_c, dy_c = metres(coarse)
    assert 8.99 < dx_c / dx_f < 9.01
    assert 8.99 < dy_c / dy_f < 9.01


def test_qc_geometry_plane_neighborhoods_are_in_cells_not_metres():
    # Windows are cell counts; the defaults keep the original fixed windows.
    assert plane.PLANE_RADIUS == 4
    assert geometry.DEFAULT_WINDOWS == (3, 9, 25)
    assert geometry.DEFAULT_FLANKS == (4, 6, 9, 12)
