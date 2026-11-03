"""One strict cell-centre selection rule for native rasters and FusionState.

Transform centres into the geometry CRS, rather than moving polygon vertices
onto a source grid. Boundary ties are excluded consistently in both paths.
"""

import numpy as np
import shapely
from pyproj import CRS, Transformer


def horizontal_crs(value):
    crs = CRS.from_user_input(value)
    return crs.sub_crs_list[0] if crs.is_compound else crs


def centre_selection(
    native, geometry, window, *, geometry_crs="EPSG:4326", transformer=None
):
    shape = (int(window.height), int(window.width))
    if geometry is None or not geometry.is_valid:
        raise ValueError("Missing or invalid source-retention geometry")
    if geometry.is_empty:
        return np.zeros(shape, dtype=bool)
    if geometry.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError("Source-retention geometry must be polygonal")
    rows, cols = np.indices(shape, dtype=np.float64)
    affine = native.window_transform(window)
    x, y = affine * (cols + 0.5, rows + 0.5)
    operation = transformer or Transformer.from_crs(
        horizontal_crs(native.crs), horizontal_crs(geometry_crs), always_xy=True
    )
    x, y = operation.transform(x, y)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(
            "Nonfinite source cell centres after horizontal transformation"
        )
    return shapely.contains_xy(geometry, x, y)
