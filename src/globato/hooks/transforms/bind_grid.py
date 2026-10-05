#!/usr/bin/env python

"""
globato.hooks.transforms.bind_grid
~~~~~~~~~~~~~

Bind grid values to a point-stream field

:copyright: (c) 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import os

import numpy as np
import rasterio
from fetchez.hooks import FetchHook
from rasterio.windows import Window

from globato.hooks.filters.reference import RasterSampling
from globato.utils import add_field_to_recarray

logger = logging.getLogger(__name__)


class BindGrid(FetchHook, RasterSampling):
    """Dynamically samples an external raster and binds the values
    to a specific column in the point stream.
    """

    name = "bind-grid"
    meta_stage = "stream"
    meta_category = "stream-transform"
    meta_consumes = "point-stream"
    meta_produces = "point-stream"
    meta_desc = (
        "Samples an external raster and binds the values to a point-stream column."
    )

    # Map target columns to their required Globato numpy dtypes
    DTYPE_MAP = {
        "w": np.float32,
        "u": np.float32,
        "classification": np.uint8,
        "confidence": np.int16,
    }

    def __init__(
        self,
        column="classification",
        grid_path=None,
        path_replace=None,
        band=1,
        **kwargs,
    ):
        """
        Args:
            column (str): Target array column (e.g., 'classification', 'w', 'confidence').
            grid_path (str): Explicit path to the raster to sample.
            path_replace (str): Format "old,new". Dynamically derives the raster path
                                from the current entry's dst_fn.
            band (int): Which band of the raster to sample.
        """

        super().__init__(**kwargs)
        self.column = column.lower()
        self.grid_path = grid_path
        self.path_replace = path_replace
        self.band = int(band)

    def _process(self, stream, target_raster):
        if not target_raster or not os.path.exists(target_raster):
            logger.warning(f"[{self.name}] Raster not found: {target_raster}")
            for chunk in stream:
                yield chunk
            return

        target_dtype = self.DTYPE_MAP.get(self.column, np.float32)

        # for chunk in stream:
        #     vals = self.sample_raster(target_raster, chunk, default_val=0)
        #     print(vals)
        #     yield chunk
        try:
            with rasterio.Env(CPL_MIN_LOG_LEVEL=rasterio.logging.ERROR):
                with rasterio.open(target_raster) as src:
                    for chunk in stream:
                        if self.column not in chunk.dtype.names:
                            chunk = add_field_to_recarray(
                                chunk, self.column, target_dtype, 0
                            )

                        rows, cols = rasterio.transform.rowcol(
                            src.transform, chunk["x"], chunk["y"]
                        )
                        valid = (
                            (rows >= 0)
                            & (rows < src.height)
                            & (cols >= 0)
                            & (cols < src.width)
                        )
                        if not np.any(valid):
                            yield chunk
                            continue

                        r_min, r_max = np.min(rows[valid]), np.max(rows[valid])
                        c_min, c_max = np.min(cols[valid]), np.max(cols[valid])

                        window = Window(
                            c_min, r_min, c_max - c_min + 1, r_max - r_min + 1
                        )

                        data = src.read(self.band, window=window)

                        l_rows = rows[valid] - r_min
                        l_cols = cols[valid] - c_min

                        chunk[self.column][valid] = data[l_rows, l_cols]

                        yield chunk

        except Exception as e:
            logger.error(f"[{self.name}] Failed to sample {target_raster}: {e}")
            for chunk in stream:
                yield chunk

    def run(self, entries):
        for mod, entry in entries:
            if not self.is_point_stream(entry):
                continue

            # Determine which raster to sample
            target_raster = self.grid_path
            if self.path_replace and entry.get("dst_fn"):
                old_str, new_str = self.path_replace.split(",")
                target_raster = entry["dst_fn"].replace(
                    old_str.strip(), new_str.strip()
                )

            entry["stream"] = self._process(entry["stream"], target_raster)

        return entries
