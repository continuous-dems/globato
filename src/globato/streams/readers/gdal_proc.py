#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.streams.readers.gdal_proc
~~~~~~~~~~~~~

GDAL data parsing

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
from typing import Any
import numpy as np

try:
    from osgeo import gdal

    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False

from fetchez.utils import int_or
from .base import BaseGlobatoReader

logger = logging.getLogger(__name__)


class GDALReader(BaseGlobatoReader):
    """Streaming GDAL Raster Parser.

    Reads a raster in chunks and yields structured numpy arrays.
    """

    name = "gdal-point-reader"
    meta_category = "point-stream"
    meta_dtype = ["gdal-raster", "gdal"]
    meta_desc = "Read raster data through gdal into a point stream"
    # meta_extensions = ["tif", "tiff", "vrt", "dt0", "dt1", "dt2"]
    meta_extnesions: list[Any] = []

    def __init__(
        self,
        path,
        region=None,
        band_no=1,
        mask_band=None,
        weight_band=None,
        unc_band=None,
        chunk_size=4096,
        node="pixel",
        **kwargs,
    ):

        if not HAS_GDAL:
            raise ImportError("GDAL is required for this processor.")

        super().__init__(path, **kwargs)
        self.src_fn = path
        self.req_region_bounds = region
        self.band_no = int_or(band_no, 1)
        self.mask_band = int_or(mask_band)
        self.weight_band = int_or(weight_band)
        self.unc_band = int_or(unc_band)
        self.chunk_size = int_or(chunk_size, 4096)
        self.node = node.lower()

    def get_read_window(self, ds):
        """Calculate the source window to read."""

        if not self.req_region_bounds:
            return 0, 0, ds.RasterXSize, ds.RasterYSize

        file_srs = self.get_srs()
        if not file_srs:
            return 0, 0, ds.RasterXSize, ds.RasterYSize

        from ..spatial import Region

        roi = Region(*self.req_region_bounds, epsg=4326)
        roi.warp(file_srs)

        gt = ds.GetGeoTransform()
        return roi.srcwin(gt, ds.RasterXSize, ds.RasterYSize)

    def get_gt(self):
        try:
            ds = gdal.Open(self.src_fn, gdal.GA_ReadOnly)
            if not ds:
                raise IOError(f"Could not open {self.src_fn}")

            gt = ds.GetGeoTransform()
            ds = None
            return gt
        except Exception:
            return None

    def get_srs(self):
        try:
            ds = gdal.Open(self.src_fn, gdal.GA_ReadOnly)
            if not ds:
                raise IOError(f"Could not open {self.src_fn}")

            src_srs = ds.GetProjection()
            ds = None
            return src_srs
        except Exception:
            return "EPSG:4326"

    def _yield_raw_chunks(self):
        """Yield numpy recarrays (x,y,z,w,u) from raster chunks."""

        ds = gdal.Open(self.src_fn, gdal.GA_ReadOnly)
        if not ds:
            raise IOError(f"Could not open {self.src_fn}")

        try:
            gt = ds.GetGeoTransform()
            band = ds.GetRasterBand(self.band_no)
            ndv = band.GetNoDataValue()

            x_size = ds.RasterXSize
            y_size = ds.RasterYSize

            for y in range(0, y_size, self.chunk_size):
                rows = min(self.chunk_size, y_size - y)

                for x in range(0, x_size, self.chunk_size):
                    cols = min(self.chunk_size, x_size - x)

                    # 1. Read Elevation
                    z_data = band.ReadAsArray(x, y, cols, rows).astype(np.float64)

                    if ndv is not None:
                        z_data[z_data == ndv] = np.nan

                    if self.mask_band:
                        m_data = ds.GetRasterBand(self.mask_band).ReadAsArray(
                            x, y, cols, rows
                        )
                        z_data[m_data == 0] = np.nan

                    if np.all(np.isnan(z_data)):
                        continue

                    # 2. Coordinates
                    # Pixel center vs corner logic
                    x_offset = 0.5 if self.node == "pixel" else 0.0
                    y_offset = 0.5 if self.node == "pixel" else 0.0

                    curr_x = np.arange(x, x + cols) + x_offset
                    curr_y = np.arange(y, y + rows) + y_offset

                    grid_x = gt[0] + curr_x * gt[1] + (y + y_offset) * gt[2]
                    grid_y = gt[3] + (x + x_offset) * gt[4] + curr_y * gt[5]

                    X, Y = np.meshgrid(grid_x, grid_y)

                    # Flatten
                    z_flat = z_data.flatten()
                    x_flat = X.flatten()
                    y_flat = Y.flatten()

                    valid = ~np.isnan(z_flat)
                    if not np.any(valid):
                        continue

                    x_flat = x_flat[valid]
                    y_flat = y_flat[valid]
                    z_flat = z_flat[valid]

                    # 3. Weights & Uncertainty
                    if self.weight_band:
                        w_data = (
                            ds.GetRasterBand(self.weight_band)
                            .ReadAsArray(x, y, cols, rows)
                            .astype(np.float32)
                        )
                        w_flat = w_data.flatten()[valid]
                    else:
                        w_flat = np.ones_like(z_flat, dtype=np.float32)

                    if self.unc_band:
                        u_data = (
                            ds.GetRasterBand(self.unc_band)
                            .ReadAsArray(x, y, cols, rows)
                            .astype(np.float32)
                        )
                        u_flat = u_data.flatten()[valid]
                    else:
                        u_flat = np.zeros_like(z_flat, dtype=np.float32)

                    # 4. Create Structured Array (RecArray)
                    # This is the "Standard Chunk" for the pipeline
                    chunk = np.rec.fromarrays(
                        [x_flat, y_flat, z_flat, w_flat, u_flat],
                        names=["x", "y", "z", "w", "u"],
                    )
                    yield chunk

        except Exception:
            logger.error(f"could not chunk gdal file {self.src_fn}.")
        finally:
            ds = None
