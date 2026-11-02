"""Generic georeferenced DEM preview mesh preparation (no CUDEM or PyVista dependency).

Mesh geometry is in a local azimuthal equidistant metric CRS shared by a
comparison group. The source raster and original elevations are never altered.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.enums import Resampling


def check_bounds(bounds):
    if len(bounds) != 4 or not np.isfinite(bounds).all():
        raise ValueError("Camera bounds must be four finite EPSG:3857 coordinates")
    west, south, east, north = map(float, bounds)
    if east <= west or north <= south:
        raise ValueError("Camera bounds must have nonzero positive area")
    return west, south, east, north


def metric_crs(camera_bounds):
    west, south, east, north = check_bounds(camera_bounds)
    to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform((west + east) / 2, (south + north) / 2)
    if abs(lat) > 85:
        raise ValueError(
            "3D local projection does not support latitudes above 85 degrees"
        )
    local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat:.12g} +lon_0={lon:.12g} +datum=WGS84 +units=m +no_defs"
    )
    return local


def camera_span(camera_bounds, local_crs):
    west, south, east, north = check_bounds(camera_bounds)
    to_local = Transformer.from_crs("EPSG:3857", local_crs, always_xy=True)
    x, y = to_local.transform([west, west, east, east], [south, north, south, north])
    return max(float(np.ptp(x)), float(np.ptp(y)), 1.0)


def navigation_frame(camera_bounds):
    """Locally linear georeference for offline 2D (3857) / 3D (AEQD) navigation.

    A common scene CRS and center are derived from the *comparison group*, not
    the provider or raster label. The finite-difference Jacobian maps displayed
    Web Mercator meters to projected 3D meters near that center. It correctly
    handles latitude-dependent Mercator scale and local grid convergence.
    The approximation is intended for preview navigation, not for measurement.
    """
    west, south, east, north = check_bounds(camera_bounds)
    cx, cy = (west + east) / 2, (south + north) / 2
    crs = metric_crs(camera_bounds)
    to_local = Transformer.from_crs("EPSG:3857", crs, always_xy=True)
    # A small finite-difference baseline measures local scale/rotation without
    # requiring a JavaScript projection library or network access in file://.
    delta = max(1.0, min(100.0, min(east - west, north - south) * 0.01))
    x0, y0 = to_local.transform(cx, cy)
    xp, yp = to_local.transform(cx + delta, cy)
    xm, ym = to_local.transform(cx, cy + delta)
    a, c = (xp - x0) / delta, (yp - y0) / delta
    b, d = (xm - x0) / delta, (ym - y0) / delta
    det = a * d - b * c
    if not np.isfinite([x0, y0, a, b, c, d, det]).all() or abs(det) < 1e-12:
        raise ValueError("Unable to derive a non-singular local navigation frame")
    return {
        "center_3857": [cx, cy],
        "origin_xy_m": [x0, y0],
        "mercator_to_local": [a, b, c, d],
        "method": "local 2D Jacobian of EPSG:3857 to group AEQD; navigation only",
    }


def prepare_terrain(
    path,
    camera_bounds,
    *,
    band=1,
    max_side=500,
    vertical_exaggeration=1.0,
    vertical_unit_to_m=1.0,
    elevation_reference=0.0,
    nodata_policy="strict",
):
    """Return a PyVista-independent indexed triangular surface.

    Triangles are emitted ONLY when all four corners of a source-grid cell are
    valid, so holes remain open. Unmeshed valid preview samples are returned as
    points: sparse measured grids remain inspectable without inventing surfaces.
    The preview uses masked-average source-grid downsampling, not interpolation.
    """
    if int(max_side) < 2:
        raise ValueError("3D mesh max_side must be at least 2")
    for name, value in [
        ("vertical_exaggeration", vertical_exaggeration),
        ("vertical_unit_to_m", vertical_unit_to_m),
    ]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(name + " must be a positive finite value")
    if not np.isfinite(elevation_reference):
        raise ValueError("elevation_reference must be finite")
    if nodata_policy not in {"strict", "overview"}:
        raise ValueError("nodata_policy must be strict or overview")
    local = metric_crs(camera_bounds)
    with rasterio.open(Path(path)) as src:
        if src.width < 2 or src.height < 2:
            raise ValueError("A terrain mesh requires at least 2 pixels per axis")
        if src.crs is None:
            raise ValueError("Raster CRS required for 3D geometry")
        if not (1 <= band <= src.count):
            raise ValueError("Requested band is unavailable")
        f = max(1.0, src.width / max_side, src.height / max_side)
        width = max(2, min(src.width, math.ceil(src.width / f)))
        height = max(2, min(src.height, math.ceil(src.height / f)))
        if width < 2 or height < 2:
            raise ValueError("A terrain mesh requires at least 2 pixels per axis")
        samples = (
            src.read(
                band,
                out_shape=(height, width),
                resampling=Resampling.average,
                masked=True,
            )
            .astype("float64")
            .filled(np.nan)
        )
        # Retain these genuine (possibly bin-averaged) observations for point
        # rendering even if strict surface coverage rejects their triangles.
        values = samples.copy()
        # Masked averaging alone hides thin source gaps. In strict mode mark
        # every preview cell touched by any invalid source pixel. Scan by GDAL
        # blocks, never allocate a full-resolution validity array.
        coverage = np.isfinite(values)
        if nodata_policy == "strict" and (height, width) != (src.height, src.width):
            for _, window in src.block_windows(band):
                a = src.read(band, window=window, masked=True)
                bad = np.ma.getmaskarray(a) | ~np.isfinite(a.data)
                if not bad.any():
                    continue
                # In sparse source grids, whole GDAL blocks can be NoData.
                # Reject their intersected output cells in one vectorized
                # operation, avoiding millions of per-pixel index updates.
                if bad.all():
                    r0 = int(window.row_off) * height // src.height
                    r1 = (
                        int(window.row_off + window.height) * height - 1
                    ) // src.height
                    c0 = int(window.col_off) * width // src.width
                    c1 = (int(window.col_off + window.width) * width - 1) // src.width
                    coverage[r0 : r1 + 1, c0 : c1 + 1] = False
                    continue
                rr, cc = np.nonzero(bad)
                rr = rr.astype("int64") + int(window.row_off)
                cc = cc.astype("int64") + int(window.col_off)
                r0, r1 = (
                    rr * height // src.height,
                    ((rr + 1) * height - 1) // src.height,
                )
                c0, c1 = cc * width // src.width, ((cc + 1) * width - 1) // src.width
                for rows, cols in [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]:
                    coverage[rows, cols] = False
            values[~coverage] = np.nan
        t = src.transform * rasterio.Affine.scale(
            src.width / width, src.height / height
        )
        rows, cols = np.indices((height, width))
        x_src, y_src = t * (cols + 0.5, rows + 0.5)
        local_transformer = Transformer.from_crs(src.crs, local, always_xy=True)
        x, y = local_transformer.transform(x_src, y_src)
        # Ground footprint of one display-grid sample, independent of elevation
        # validity. Only used to size optional 3D point sprites in the browser.
        # A sprite is a *preview bin*, never new measured or interpolated data.
        step_x = math.hypot(float(x[0, 1] - x[0, 0]), float(y[0, 1] - y[0, 0]))
        step_y = math.hypot(float(x[1, 0] - x[0, 0]), float(y[1, 0] - y[0, 0]))
        preview_cell_size_m = math.sqrt(step_x * step_y)
        if not np.isfinite(preview_cell_size_m) or preview_cell_size_m <= 0:
            raise ValueError("Invalid projected 3D preview cell footprint")
        finite_xy = np.isfinite(x) & np.isfinite(y)
        valid = np.isfinite(values) & finite_xy
        sampled = np.isfinite(samples) & finite_xy
        if not sampled.any():
            raise ValueError("No valid elevation samples for 3D preview")
        # No fabricated triangles: 4 *strictly* valid corners are required.
        cell_ok = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
        rr, cc = np.nonzero(cell_ok)
        q = rr * width + cc
        faces = np.empty((len(q) * 2, 4), dtype=np.int64)
        faces[0::2] = np.column_stack((np.full(len(q), 3), q, q + width, q + 1))
        faces[1::2] = np.column_stack(
            (np.full(len(q), 3), q + 1, q + width, q + width + 1)
        )
        # Any observed preview sample not part of a triangle is visible as a
        # point, including isolated samples and strict-mode boundary cells.
        used_by_surface = np.zeros(height * width, dtype=bool)
        if len(faces):
            used_by_surface[faces[:, 1:].ravel()] = True
        point_ids = np.flatnonzero(sampled.ravel() & ~used_by_surface)
        # Compress away blank parts of large AOIs before shipping to WebGL.
        keep = used_by_surface | sampled.ravel()
        original_ids = np.flatnonzero(keep)
        remap = np.full(height * width, -1, dtype=np.int64)
        remap[original_ids] = np.arange(len(original_ids))
        if len(faces):
            faces[:, 1:] = remap[faces[:, 1:]]
        point_indices = remap[point_ids].astype("uint32")
        # Explicit all-sample indexing lets the generic viewer display every
        # populated preview bin as a point/cell *instead* of a surface.
        all_point_indices = remap[np.flatnonzero(sampled.ravel())].astype("uint32")
        # A 3D preview is not a vertical-datum conversion. Source vertical units
        # must be supplied explicitly if different from meters.
        z = (
            (samples.ravel()[original_ids] - elevation_reference)
            * vertical_unit_to_m
            * vertical_exaggeration
        )
        points = np.column_stack(
            (
                np.asarray(x).ravel()[original_ids],
                np.asarray(y).ravel()[original_ids],
                z,
            )
        ).astype("float64")
        extent = [
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
        ]
        return dict(
            points=points,
            faces=faces.ravel(),
            point_indices=point_indices,
            all_point_indices=all_point_indices,
            all_point_count=int(all_point_indices.size),
            elevations=samples.ravel()[original_ids],
            preview_shape=[height, width],
            preview_cell_size_m=float(preview_cell_size_m),
            valid_cells=int(valid.sum()),
            valid_samples=int(sampled.sum()),
            point_count=int(point_indices.size),
            triangles=int(faces.shape[0]),
            data_bounds_local_xy=extent,
            render_mode=(
                "hybrid"
                if len(faces) and len(point_indices)
                else "surface"
                if len(faces)
                else "points"
            ),
            source_crs=str(src.crs),
            metric_crs=local.to_string(),
            resampling="masked source-grid average (points represent populated preview bins)",
            nodata_policy=nodata_policy,
            missing="exclude triangles with invalid corners; strict source-gap coverage"
            if nodata_policy == "strict"
            else "overview validity only; thin source gaps may disappear",
        )
