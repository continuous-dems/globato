#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.streams.readers.rio
~~~~~~~~~~~~~

Rasterio data parsing

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import numpy as np

import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds
from rasterio.errors import WindowError

from fetchez.utils import int_or, float_or

from .base import BaseGlobatoReader

logger = logging.getLogger(__name__)
logging.getLogger("rasterio").setLevel(logging.ERROR)


class RasterioReader(BaseGlobatoReader):
    """Streaming Raster Parser using Rasterio."""

    name = "rasterio-point-reader"
    meta_category = "point-stream"
    meta_dtype = ["rio", "rasterio", "rio-raster", "raster"]
    meta_desc = "Read raster data through rasterio into a point stream"
    meta_extensions = ["tif", "tiff", "vrt", "dt0", "dt1", "dt2", "img"]

    def __init__(
        self,
        path,
        band_no=1,
        chunk_size=None,
        region=None,
        path_prefix="",
        path_suffix="",
        w_band=None,
        u_band=None,
        x_band=None,
        y_band=None,
        auto_weight=False,
        min_weight=None,
        uncertainty_scale=1.0,
        **kwargs,
    ):
        super().__init__(path, **kwargs)

        # Build the GDAL subdataset path if prefixes/suffixes are provided
        if path_prefix or path_suffix:
            self.src_fn = f"{path_prefix}{path}{path_suffix}"
        else:
            self.src_fn = path

        self.band_no = band_no
        self.chunk_size = chunk_size
        self.region = region
        self.u_band = int_or(u_band)
        self.w_band = int_or(w_band)
        self.x_band = int_or(x_band)
        self.y_band = int_or(y_band)
        self.auto_weight = auto_weight
        self.min_weight = float_or(min_weight)
        self.uncertainty_scale = float_or(uncertainty_scale, 1.0)
        self.kwargs = kwargs

    def get_srs(self):
        """Get SRS as WKT."""

        try:
            with rasterio.Env(CPL_MIN_LOG_LEVEL=rasterio.logging.ERROR):
                with rasterio.open(self.src_fn) as src:
                    return src.crs.to_wkt() if src.crs else "EPSG:4326"
        except Exception:
            return "EPSG:4326"

    def _yield_raw_chunks(self):
        yield from self._process_rio_dataset()
        return

    def _process_rio_dataset(self, src=None):
        """Yield chunks using Rasterio Windows. Accepts an optional open dataset."""

        if src is not None:
            yield from self._read_chunks_from_src(src)
        else:
            try:
                with rasterio.Env(CPL_MIN_LOG_LEVEL=rasterio.logging.ERROR):
                    with rasterio.open(self.src_fn) as new_src:
                        yield from self._read_chunks_from_src(new_src)
            except Exception as e:
                logger.exception(f"Rasterio read failed: {e}")
                raise

    def _read_chunks_from_src(self, src):
        """The core windowing and extraction logic, isolated from file-opening."""

        if self.region:
            west, east, south, north = self.region

            # Dynamically grab the SRS from the Region object, fallback to WGS84
            region_srs = getattr(self.region, "srs", None) or "EPSG:4326"

            if src.crs and src.crs.to_string() != region_srs:
                try:
                    west, south, east, north = transform_bounds(
                        region_srs, src.crs, west, south, east, north
                    )
                except Exception as err:
                    logger.error(f"Failed to transform bounds for {self.src_fn}: {err}")
                    raise

            req_window = from_bounds(west, south, east, north, transform=src.transform)

            try:
                master_window = req_window.intersection(
                    Window(0, 0, src.width, src.height)
                )
            except WindowError:
                logger.debug(
                    f"Raster {self.src_fn} is entirely outside the requested region of {self.region}."
                )
                return

            if master_window.width <= 0 or master_window.height <= 0:
                logger.debug(
                    f"Raster {self.src_fn} is entirely outside the requested region of {self.region}."
                )
                return
        else:
            master_window = Window(0, 0, src.width, src.height)

        block_h, block_w = src.block_shapes[0]
        h_chunk = self.chunk_size or block_h
        w_chunk = self.chunk_size or block_w

        y_start = int(master_window.row_off)
        y_end = int(master_window.row_off + master_window.height)
        x_start = int(master_window.col_off)
        x_end = int(master_window.col_off + master_window.width)

        for y in range(y_start, y_end, h_chunk):
            rows = min(h_chunk, y_end - y)

            for x in range(x_start, x_end, w_chunk):
                cols = min(w_chunk, x_end - x)

                window = Window(x, y, cols, rows)
                z = src.read(self.band_no, window=window)
                u = (
                    src.read(self.u_band, window=window)
                    if self.u_band
                    else np.zeros_like(z)
                )
                if self.w_band:
                    w = src.read(self.w_band, window=window)
                else:
                    # Use the instance's calculated weight if available, otherwise default to 1.0
                    fallback_weight = getattr(self, "weight", 1.0)

                    if self.auto_weight and self.u_band is not None:
                        scale = max(self.uncertainty_scale, 1e-12)

                        # penalty = 1.0 / np.sqrt(1.0 + u / uncertainty_scale)
                        w = fallback_weight / (1.0 + (u / scale))

                        if self.min_weight is not None:
                            w = np.maximum(w, self.min_weight)
                        else:
                            w = np.full_like(z, fallback_weight)
                    else:
                        w = np.full_like(z, fallback_weight)

                x_arr = src.read(self.x_band, window=window) if self.x_band else None
                y_arr = src.read(self.y_band, window=window) if self.y_band else None

                if not np.issubdtype(z.dtype, np.floating):
                    z = z.astype(np.float32)

                mask = ~np.isnan(z)
                if src.nodata is not None:
                    if np.issubdtype(z.dtype, np.floating):
                        mask &= ~np.isclose(z, src.nodata, rtol=1e-5, equal_nan=True)
                    else:
                        mask &= z != src.nodata

                if not np.any(mask):
                    continue

                z_valid = z[mask]
                w_valid = w[mask]
                u_valid = u[mask]

                if x_arr is not None and y_arr is not None:
                    xs = x_arr[mask]
                    ys = y_arr[mask]
                else:
                    local_rows, local_cols = np.where(mask)

                    global_rows = local_rows + window.row_off
                    global_cols = local_cols + window.col_off

                    xs, ys = rasterio.transform.xy(
                        src.transform,
                        global_rows,
                        global_cols,
                        offset="center",
                    )
                count = len(z_valid)
                chunk = np.zeros(
                    count,
                    dtype=[
                        ("x", "f8"),
                        ("y", "f8"),
                        ("z", "f4"),
                        ("w", "f4"),
                        ("u", "f4"),
                    ],
                )

                chunk["x"] = xs
                chunk["y"] = ys
                chunk["z"] = z_valid
                chunk["u"] = u_valid
                chunk["w"] = w_valid
                yield chunk
