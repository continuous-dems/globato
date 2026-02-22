#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.formats.rio
~~~~~~~~~~~~~

Rasterio data parsing

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
import numpy as np
import rasterio
from rasterio.windows import Window
from fetchez.hooks import FetchHook
from fetchez.utils import float_or

logger = logging.getLogger(__name__)


class RasterioReader:
    """Streaming Raster Parser using Rasterio."""

    def __init__(self, src_fn, band_no=1, chunk_size=4096):
        self.src_fn = src_fn
        self.band_no = band_no
        self.chunk_size = chunk_size


    def get_srs(self):
        """Get SRS as WKT."""

        try:
            with rasterio.Env(CPL_MIN_LOG_LEVEL=rasterio.logging.ERROR):
                with rasterio.open(self.src_fn) as src:
                    return src.crs.to_wkt() if src.crs else 'EPSG:4326'
        except Exception:
            return 'EPSG:4326'


    def yield_chunks(self):
        """Yield chunks using Rasterio Windows."""

        try:
            with rasterio.Env(CPL_MIN_LOG_LEVEL=rasterio.logging.ERROR):
                with rasterio.open(self.src_fn) as src:
                    ndv = float_or(src.nodata, -9999)
                    height, width = src.shape
                    # transform = src.transform

                    for y in range(0, height, self.chunk_size):
                        rows = min(self.chunk_size, height - y)

                        for x in range(0, width, self.chunk_size):
                            cols = min(self.chunk_size, width - x)
                            window = Window(x, y, cols, rows)

                            z_data = src.read(self.band_no, window=window)
                            if not np.issubdtype(z_data.dtype, np.floating):
                                z_data = z_data.astype(np.float32)

                            if ndv is not None:
                                z_data[z_data == ndv] = np.nan

                            if np.all(np.isnan(z_data)): continue

                            win_transform = src.window_transform(window)
                            xs, _ = rasterio.transform.xy(
                                win_transform,
                                [0] * cols,
                                range(cols),
                                offset='center'
                            )
                            _, ys = rasterio.transform.xy(
                                win_transform,
                                range(rows),
                                [0] * rows,
                                offset='center'
                            )
                            X, Y = np.meshgrid(xs, ys)
                            # xs, ys = rasterio.transform.xy(
                            #     win_transform,
                            #     range(rows),
                            #     range(cols),
                            #     offset='center' # PixelIsPoint / Center
                            # )

                            # X, Y = np.meshgrid(xs, ys)

                            flat_z = z_data.flatten()
                            flat_x = X.flatten()
                            flat_y = Y.flatten()

                            valid = ~np.isnan(flat_z)
                            if not np.any(valid): continue

                            flat_w = np.ones_like(flat_z, dtype=np.float32)
                            flat_u = np.zeros_like(flat_z, dtype=np.float32)

                            out_chunk = np.rec.fromarrays(
                                [flat_x[valid], flat_y[valid], flat_z[valid], flat_w[valid], flat_u[valid]],
                                names=['x', 'y', 'z', 'w', 'u']
                            )

                            yield out_chunk

        except Exception as e:
            logger.error(f'Rasterio read failed: {e}')


class RasterioStream(FetchHook):
    name = "rasterio_stream"
    stage = "file"
    category = "format-stream"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = kwargs

    def run(self, entries):
        for mod, entry in entries:
            src = entry.get('dst_fn')
            if not src or not os.path.exists(src): continue
            try:
                reader = RasterioReader(src, **self.params)
                entry['stream'] = reader.yield_chunks()
                entry['stream_type'] = 'xyz_recarray'
            except Exception as e:
                logger.warning(f"RasterioStream failed for {src}: {e}")
        return entries
