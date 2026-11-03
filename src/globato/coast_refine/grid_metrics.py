"""Physical grid spacing shared by coastal and river interpolation guidance."""

from __future__ import annotations
from pyproj import CRS


def cell_spacing_m(transform, crs, shape):
    """Approximate row/col spacing in metres at the centre of a small AOI.

    Fail closed for large geographic tiles where metre spacing varies
    substantially: this experiment is deliberately scoped to small AOIs.
    """
    obj = CRS.from_user_input(crs)
    if obj.is_compound:
        obj = obj.sub_crs_list[0]
    x, y = transform * (shape[1] / 2, shape[0] / 2)
    if obj.is_geographic:
        geod = obj.get_geod()
        dx = abs(geod.inv(x, y, x + transform.a, y)[2])
        dy = abs(geod.inv(x, y, x, y + transform.e)[2])
        x2, y2 = transform * (shape[1] / 2, 0)
        x3, y3 = transform * (shape[1] / 2, shape[0])
        dx2 = abs(geod.inv(x2, y2, x2 + transform.a, y2)[2])
        dx3 = abs(geod.inv(x3, y3, x3 + transform.a, y3)[2])
        if max(dx2, dx3) / min(dx2, dx3) > 1.025:
            raise ValueError(
                "coastal_transition requires an AOI with <2.5% latitude spacing variation"
            )
    else:
        factor = obj.axis_info[0].unit_conversion_factor or 1.0
        dx, dy = abs(transform.a * factor), abs(transform.e * factor)
    if dx <= 0 or dy <= 0 or transform.b or transform.d:
        raise ValueError("coastal_transition requires north-up metric-compatible grid")
    return float(dy), float(dx)
