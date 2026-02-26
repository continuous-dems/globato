#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.rasters.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Base classes for Raster operations.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import fiona
from transformez.spatial import TransRegion
from fetchez.hooks import FetchHook

logger = logging.getLogger(__name__)


class RasterHook(FetchHook):
    """Base class for hooks that operate on raster files.

    Features:
    - Auto-chunking with buffers.
    - Barrier support (e.g. Coastline splitting).
    - **Auto-Stack Filtering**: If input is a Multi-Stack (3+ bands),
      automatically masks data based on Weight and Count.
    - Set the stage to either 'post' or 'file'
    """

    stage = "post"
    category = "raster-op"
    default_suffix = "_processed"

    def __init__(
            self,
            output=None,
            suffix=None,
            barrier=None,
            buffer=0,
            min_weight=0.0,
            strip_bands=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.output = output
        self.suffix = suffix or self.default_suffix
        self.barrier = barrier
        self.buffer = int(buffer)
        self.min_weight = float(min_weight)
        self.region = None
        self.strip_bands=strip_bands

    def _get_region_from_entries(self, entries):
        regions = [getattr(mod, "region", None) for mod, _ in entries]
        valid_regions = [r for r in regions if r]

        if not valid_regions:
            return entries

        # Union of all requested regions
        w = min(r[0] for r in valid_regions)
        e = max(r[1] for r in valid_regions)
        s = min(r[2] for r in valid_regions)
        n = max(r[3] for r in valid_regions)
        target_region = [w, e, s, n]

        return TransRegion(*target_region)

    def run(self, entries):
        if not getattr(self, 'region', None):
            self.region = self._get_region_from_entries(entries)

        if self.barrier and self.barrier.lower() == "coastline":
            osm_path = "auto_coastline.geojson"
            if not os.path.exists(osm_path):
                logger.info("Auto-generating OSM Coastline barrier...")
                from ..hooks.osm_landmask import OSMLandmask
                osm_hook = OSMLandmask(filename=osm_path)
                osm_hook.run(entries)
            self.barrier = osm_path

        new_entries = []
        for mod, entry in entries:
            src_fn = entry.get("dst_fn")

            if not src_fn or not os.path.exists(src_fn) or not src_fn.lower().endswith(".tif"):
                new_entries.append((mod, entry))
                continue

            if self.output:
                dst_fn = self.output
            else:
                base, ext = os.path.splitext(src_fn)
                dst_fn = f"{base}{self.suffix}{ext}"

            logger.info(f"Running {self.name} on {os.path.basename(src_fn)}")

            try:
                success = self.process_raster(src_fn, dst_fn, entry)
                if success:
                    entry["src_fn"] = src_fn
                    entry["dst_fn"] = dst_fn
                    entry.setdefault("artifacts", {})[self.name] = dst_fn
            except Exception as e:
                logger.error(f"RasterHook {self.name} failed on {src_fn}: {e}")

            new_entries.append((mod, entry))

        return new_entries

    def process_raster(self, src_path, dst_path, entry):
        barrier_geoms = self._get_barrier_geometries()

        with rasterio.open(src_path) as src:
            profile = src.profile.copy()
            is_stack = (src.count >= 3)

            if self.strip_bands:
                profile["count"] = 1

            with rasterio.open(dst_path, 'w', **profile) as dst:
                for window, buff_win in self.yield_buffered_windows(src, buffer_size=self.buffer):

                    data = src.read(1, window=buff_win)
                    ndv = src.nodata

                    chunk_transform = rasterio.windows.transform(buff_win, src.transform)

                    if is_stack:
                        try:
                            count_arr = src.read(2, window=buff_win)
                            weight_arr = src.read(3, window=buff_win)
                            if np.any(count_arr):
                                invalid_mask = (count_arr == 0) | (weight_arr < self.min_weight)
                                data[invalid_mask] = ndv
                        except Exception:
                            pass

                    # Barrier Logic (Coastline Split)
                    if barrier_geoms:
                        win_transform = rasterio.windows.transform(buff_win, src.transform)
                        barrier_mask = rasterize(
                            barrier_geoms, out_shape=data.shape,
                            transform=win_transform, fill=0, default_value=1, dtype='uint8'
                        ).astype(bool)

                        # Split and process independently
                        data_a = np.where(barrier_mask, data, ndv)
                        data_b = np.where(~barrier_mask, data, ndv)

                        # Process chunks
                        res_a = self.process_chunk(data_a, ndv, entry, transform=chunk_transform, window=buff_win)
                        res_b = self.process_chunk(data_b, ndv, entry, transform=chunk_transform, window=buff_win)
                        #res_a[:] = ndv
                        #res_b[res_b > 0] = ndv

                        # Stitch
                        result = np.where(barrier_mask, res_a, res_b)
                    else:
                        # Standard Processing
                        result = self.process_chunk(
                            data, ndv, entry,
                            transform=chunk_transform,
                            window=buff_win,
                        )

                    # Crop buffer
                    y_off = window.row_off - buff_win.row_off
                    x_off = window.col_off - buff_win.col_off

                    final_chunk = result[y_off : y_off + window.height,
                                         x_off : x_off + window.width]

                    dst.write(final_chunk, 1, window=window)

        return True

    def process_chunk(self, data, ndv, entry, transform=None, window=None):
        """Must be implemented by subclasses. Returns processed numpy array."""

        raise NotImplementedError

    def yield_buffered_windows(self, src, buffer_size=0):
        for block_index, window in src.block_windows(1):
            if buffer_size == 0:
                yield window, window
                continue

            row_start = max(0, window.row_off - buffer_size)
            col_start = max(0, window.col_off - buffer_size)
            row_stop = min(src.height, window.row_off + window.height + buffer_size)
            col_stop = min(src.width, window.col_off + window.width + buffer_size)

            buffered_window = Window.from_slices((row_start, row_stop), (col_start, col_stop))
            yield window, buffered_window

    def _get_barrier_geometries(self):
        if not self.barrier or not os.path.exists(self.barrier):
            return None
        try:
            with fiona.open(self.barrier, "r") as vec:
                return [feature["geometry"] for feature in vec]
        except Exception:
            return None

    def get_outliers(self, in_array, percentile=75, k=1.5):
        if np.all(np.isnan(in_array)):
            return np.nan, np.nan

        p_max = np.nanpercentile(in_array, percentile)
        p_min = np.nanpercentile(in_array, 100 - percentile)
        iqr = (p_max - p_min) * k
        return p_max + iqr, p_min - iqr
