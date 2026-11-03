"""Bounded, display-only RGB rendering.

RGB never enters an elevation stack or alters a terrain mesh. Inputs must be
georeferenced 3+ band rasters ordered red/green/blue (such as Sentinel-2 RGB).
"""

from __future__ import annotations
import math
import numpy as np
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


def stretch(src):
    if src.count < 3 or src.crs is None:
        raise ValueError("RGB requires >=3 bands and a georeferenced raster")
    if all(np.dtype(src.dtypes[i]) == np.dtype("uint8") for i in range(3)):
        return [(0.0, 255.0)] * 3
    size = max(1.0, src.width / 384, src.height / 384)
    w, h = max(1, math.ceil(src.width / size)), max(1, math.ceil(src.height / size))
    data = src.read(
        [1, 2, 3], masked=True, out_shape=(3, h, w), resampling=Resampling.average
    )
    out = []
    for b in range(3):
        values = data[b].compressed().astype("float64")
        values = values[np.isfinite(values)]
        if not values.size:
            out.append((0.0, 1.0))
        else:
            lo, hi = np.percentile(values, [2, 98])
            if hi <= lo:
                lo, hi = float(values.min()), float(values.max())
            if hi <= lo:
                hi = lo + 1.0
            out.append((float(lo), float(hi)))
    return out


def colors(data, ranges):
    """Convert a 3-band masked Rasterio array to RGBA; do not fill missing RGB."""
    masked = np.ma.getmaskarray(data)
    invalid = np.any(masked, axis=0) | ~np.all(np.isfinite(data.data), axis=0)
    height, width = data.shape[1:]
    rgba = np.zeros((height, width, 4), dtype="uint8")
    for b, (lo, hi) in enumerate(ranges):
        rgba[..., b] = np.clip(
            (data.data[b].astype("float64") - lo) * (255.0 / (hi - lo)), 0, 255
        ).astype("uint8")
    rgba[..., 3] = (~invalid).astype("uint8") * 255
    return rgba


def preview(src, *, transform, width, height):
    ranges = stretch(src)
    with WarpedVRT(
        src,
        crs="EPSG:3857",
        transform=transform,
        width=width,
        height=height,
        resampling=Resampling.bilinear,
    ) as vrt:
        data = vrt.read([1, 2, 3], masked=True)
    return colors(data, ranges)
