#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.hooks.rasters.crop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Crops a raster to the absolute boundaries of its valid data (removes NoData moats).
"""

import os
import logging
import numpy as np
import rasterio
from rasterio.windows import Window, intersection
from .base import RasterGlobalHook, copy_metadata

logger = logging.getLogger(__name__)


class RasterCrop(RasterGlobalHook):
    """Crops a raster to remove NoData margins.

    Usage: --hook raster_crop
    """

    name = "raster-crop"
    meta_desc = "Crops a raster to remove NoData margins."
    meta_aliases = ["raster_crop"]
    default_suffix = "_cropped"

    def process_raster(self, src_path, dst_path, entry):
        """Find the tightest data window and rewrite the raster."""

        with rasterio.open(src_path) as src:
            logger.debug(
                f"[{self.name}] Scanning {os.path.basename(src_path)} for valid data bounds..."
            )

            min_row, min_col = src.height, src.width
            max_row, max_col = 0, 0

            for _, window in src.block_windows(1):
                data = src.read(window=window)

                mask = (
                    (data != src.nodata) if src.nodata is not None else ~np.isnan(data)
                )
                if mask.ndim == 3:
                    mask = np.any(mask, axis=0)  # Flatten bands

                if not np.any(mask):
                    continue

                valid_rows = np.any(mask, axis=1)
                valid_cols = np.any(mask, axis=0)
                r_min, r_max = np.where(valid_rows)[0][[0, -1]]
                c_min, c_max = np.where(valid_cols)[0][[0, -1]]

                min_row = min(min_row, window.row_off + r_min)
                max_row = max(max_row, window.row_off + r_max)
                min_col = min(min_col, window.col_off + c_min)
                max_col = max(max_col, window.col_off + c_max)

            if min_row > max_row or min_col > max_col:
                logger.warning(f"[{self.name}] Raster is entirely NoData. Cannot crop.")
                return False

            crop_window = Window.from_slices(
                (min_row, max_row + 1), (min_col, max_col + 1)
            )

            kwargs = src.profile.copy()
            kwargs = self.modify_profile(kwargs)
            kwargs.update(
                {
                    "height": crop_window.height,
                    "width": crop_window.width,
                    "transform": src.window_transform(crop_window),
                }
            )

            logger.debug(
                f"[{self.name}] Cropping from {src.width}x{src.height} to {crop_window.width}x{crop_window.height}..."
            )

            with rasterio.open(dst_path, "w", **kwargs) as dst:
                copy_metadata(src, dst)
                for _, write_window in src.block_windows(1):
                    try:
                        overlap = intersection(write_window, crop_window)
                    except rasterio.errors.WindowError:
                        continue

                    if overlap.width == 0 or overlap.height == 0:
                        continue

                    data = src.read(window=overlap)

                    dst_window = Window(
                        overlap.col_off - crop_window.col_off,
                        overlap.row_off - crop_window.row_off,
                        overlap.width,
                        overlap.height,
                    )
                    dst.write(data, window=dst_window)

        return True
