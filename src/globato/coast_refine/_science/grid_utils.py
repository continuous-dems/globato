"""Raster-grid distances and mask sampling used by the scientific classifier."""

import geopandas as gpd
import numpy as np
from pyproj import CRS
from rasterio.features import rasterize
from scipy import ndimage


def mask_on_grid(path, shape, transform, crs, layer="mask"):
    g = gpd.read_file(path, layer=layer)
    if g.crs is None:
        raise ValueError(f"Missing vector CRS: {path}")
    g = g.to_crs(crs)
    geoms = [(x, 1) for x in g.geometry if x is not None and not x.is_empty]
    return (
        rasterize(
            geoms, out_shape=shape, transform=transform, fill=0, dtype="uint8"
        ).astype(bool)
        if geoms
        else np.zeros(shape, bool)
    )


def pixel_metres(transform, crs, bounds):
    """Axis-specific metric sampling; geographic spacing is midpoint-geodesic."""
    crs = CRS.from_user_input(crs)
    if transform.b != 0 or transform.d != 0:
        raise ValueError("Classification requires a north-up raster")
    if crs.is_projected:
        return (
            abs(transform.a) * crs.axis_info[0].unit_conversion_factor,
            abs(transform.e) * crs.axis_info[1].unit_conversion_factor,
        )
    if crs.is_geographic:
        # Source-neutral ellipsoidal measurement at the raster midpoint.
        # Fixed 111320*cos(lat)/111132 factors silently distort high-latitude
        # and non-WGS84 geographic rasters. This remains a local
        # approximation for very large geographic raster extents.
        lat = (bounds.bottom + bounds.top) / 2
        lon = (bounds.left + bounds.right) / 2
        geod = crs.get_geod()
        _, _, dx = geod.inv(
            lon - float(abs(transform.a)) / 2,
            lat,
            lon + float(abs(transform.a)) / 2,
            lat,
        )
        _, _, dy = geod.inv(
            lon,
            lat - float(abs(transform.e)) / 2,
            lon,
            lat + float(abs(transform.e)) / 2,
        )
        if not (np.isfinite(dx) and np.isfinite(dy) and dx > 0 and dy > 0):
            raise ValueError("Cannot determine geographic pixel size in metres")
        return dx, dy
    raise ValueError(f"Unsupported CRS: {crs}")


def coast_distances(water, dx, dy, *, need_water=True):
    # Preserve infinite distance when there is no target class. scipy otherwise
    # treats the array's exterior as an implicit target, creating a false coast.
    land_distance = (
        ndimage.distance_transform_edt(~water, sampling=(dy, dx))
        if water.any()
        else np.full(water.shape, np.inf)
    )
    water_distance = (
        (
            ndimage.distance_transform_edt(water, sampling=(dy, dx))
            if (~water).any()
            else np.full(water.shape, np.inf)
        )
        if need_water
        else None
    )
    return land_distance, water_distance


def metric_halo(geographic_geometry, metres=2000.0):
    """Metre-based halo even on geographic source rasters and at high latitudes."""
    center = geographic_geometry.centroid
    metric = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center.y:.12f} "
        f"+lon_0={center.x:.12f} +datum=WGS84 +units=m +no_defs"
    )
    return (
        gpd.GeoSeries([geographic_geometry], crs=4326)
        .to_crs(metric)
        .buffer(metres)
        .to_crs(4326)
        .iloc[0]
    )
