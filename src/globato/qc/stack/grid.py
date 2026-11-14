"""Exact indexed native-grid operations. No warp/resample repair path."""

from __future__ import annotations
import numpy as np
from affine import Affine
from pyproj import CRS, Transformer


def grid_offset(source, native):
    if source.crs is None or native.crs is None or source.crs != native.crs:
        raise ValueError("Native grid CRS mismatch")
    a, b = source.transform, native.transform
    if (a.a, a.b, a.d, a.e) != (b.a, b.b, b.d, b.e):
        raise ValueError("Native grid affine basis mismatch (resolution/orientation)")
    col, row = (~b) * (a.c, a.f)
    # Only floating serialization roundoff is tolerated, never a fractional cell.
    if abs(col - round(col)) > 1e-7 or abs(row - round(row)) > 1e-7:
        raise ValueError("Native grid origin mismatch: noninteger cell offset")
    col, row = int(round(col)), int(round(row))
    expected = b * Affine.translation(col, row)
    tol = max(abs(b.a), abs(b.b), abs(b.d), abs(b.e)) * 1e-7
    if any(abs(x - y) > tol for x, y in zip(tuple(a)[:6], tuple(expected)[:6])):
        raise ValueError("Native grid origin does not reconstruct from indexed window")
    return row, col


def metres(ds):
    t = ds.transform
    crs = CRS.from_user_input(ds.crs)
    # Rectangular source windows require orthogonal native axes.
    u = np.array([t.a, t.d])
    v = np.array([t.b, t.e])
    if abs(float(u @ v)) > 1e-10 * np.linalg.norm(u) * np.linalg.norm(v):
        raise ValueError("Stack QC requires an orthogonal native grid")
    xy = t * (ds.width / 2, ds.height / 2)
    geo = crs.geodetic_crs
    tr = Transformer.from_crs(crs, geo, always_xy=True)
    p = tr.transform(*xy)
    col = tr.transform(xy[0] + t.a, xy[1] + t.d)
    row = tr.transform(xy[0] + t.b, xy[1] + t.e)
    geod = crs.get_geod()
    dx = abs(geod.inv(*p, *col)[2])
    dy = abs(geod.inv(*p, *row)[2])
    if not all(np.isfinite([dx, dy])) or min(dx, dy) <= 0:
        raise ValueError("Invalid native ground spacing")
    return dx, dy
