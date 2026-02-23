#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.sinks.simple_stack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the grid engine utility for combining data into a grid.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
import numpy as np
import threading
import rasterio
from rasterio.windows import Window

from fetchez.hooks import FetchHook
from fetchez import utils

logger = logging.getLogger(__name__)

class PointAccumulator:
    """A lightweight streaming gridder.

    Accumulates Weighted Mean (Sum_Z / Sum_W) into a temporary 2-band GeoTIFF.
    """

    def __init__(
            self,
            filename,
            region,
            x_inc,
            y_inc,
            crs="EPSG:4326",
            verbose=False,
    ):
        self.filename = filename
        self.region = region
        self.x_inc = float(x_inc)
        self.y_inc = float(y_inc)
        self.crs = crs
        self.verbose = verbose
        self.lock = threading.Lock()

        self.nx = int(self.region.width / self.x_inc)
        self.ny = int(self.region.height / self.y_inc)

        self.transform = rasterio.transform.from_origin(
            self.region.xmin, self.region.ymax, self.x_inc, self.y_inc
        )

        self.acc_fn = f"{os.path.splitext(filename)[0]}_acc.tif"
        self._init_accumulator()


    def _init_accumulator(self):
        """Create the temporary zero-filled accumulation raster."""

        if not os.path.exists(os.path.dirname(os.path.abspath(self.acc_fn))):
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.acc_fn)))
            except:
                pass

        profile = {
            "driver": "GTiff",
            "dtype": 'float32',
            "count": 2,
            "width": self.nx,
            "height": self.ny,
            "crs": self.crs,
            "transform": self.transform,
            "tiled": True,
            "compress": "lzw",
            "nodata": 0,
        }

        with rasterio.open(self.acc_fn, "w", **profile) as dst:
            dst.set_band_description(1, "Weighted_Sum_Z")
            dst.set_band_description(2, "Sum_Weights")


    def add_points(self, points):
        """Bin points and update the accumulator grid."""

        if points is None or len(points) == 0: return

        # Vectorized Indexing
        cols = np.floor((points["x"] - self.region.xmin) / self.x_inc).astype(int)
        rows = np.floor((self.region.ymax - points["y"]) / self.y_inc).astype(int)

        mask = (cols >= 0) & (cols < self.nx) & (rows >= 0) & (rows < self.ny)
        if not np.any(mask): return

        valid_cols = cols[mask]
        valid_rows = rows[mask]
        valid_z = points["z"][mask]

        if "w" in points.dtype.names:
            valid_w = points["w"][mask]
        else:
            valid_w = np.ones_like(valid_z)

        flat_idx = valid_rows * self.nx + valid_cols

        # Aggregate duplicates in memory before I/O
        unique_flat, inverse = np.unique(flat_idx, return_inverse=True)
        pixel_sum_z = np.bincount(inverse, weights=(valid_z * valid_w))
        pixel_sum_w = np.bincount(inverse, weights=valid_w)

        u_rows = unique_flat // self.nx
        u_cols = unique_flat % self.nx

        # Determine IO Window
        r_min, r_max = u_rows.min(), u_rows.max()
        c_min, c_max = u_cols.min(), u_cols.max()

        win_w = c_max - c_min + 1
        win_h = r_max - r_min + 1
        window = Window(c_min, r_min, win_w, win_h)

        with self.lock:
            with rasterio.open(self.acc_fn, "r+") as dst:
                current_sum = dst.read(1, window=window)
                current_w = dst.read(2, window=window)

                rel_r = u_rows - r_min
                rel_c = u_cols - c_min

                # Update Window
                current_sum[rel_r, rel_c] += pixel_sum_z
                current_w[rel_r, rel_c] += pixel_sum_w

                dst.write(current_sum, 1, window=window)
                dst.write(current_w, 2, window=window)


    def finalize(self, ndv=-9999):
        """Divide Sums by Weights to produce final Z grid."""

        if self.verbose:
            logger.info(f"Finalizing grid: {self.filename}")

        if not os.path.exists(self.acc_fn):
            return None

        with rasterio.open(self.acc_fn) as src:
            profile = src.profile.copy()
            profile.update(count=1, nodata=ndv, dtype="float32")

            with rasterio.open(self.filename, "w", **profile) as dst:
                for _, window in src.block_windows(1):
                    sums = src.read(1, window=window)
                    weights = src.read(2, window=window)

                    out_z = np.full(sums.shape, ndv, dtype="float32")

                    valid = weights > 0
                    out_z[valid] = sums[valid] / weights[valid]

                    dst.write(out_z, 1, window=window)

        if os.path.exists(self.acc_fn):
            try:
                os.remove(self.acc_fn)
            except:
                pass

        return self.filename


class SimpleStack(FetchHook):
    """Simple Weighted Mean Stacker.

    Args:
        res (str): Resolution (e.g. '1s', '10m').
        output (str): Output filename. Supports templates like '{base}_stack.tif'.
        mode (str): Aggregation mode (currently only 'mean').
        plug (bool): If True, consumes the stream, writes the file, and replaces
                     the entry with the raster result.
    """

    name = "simple_stack"
    stage = "file"
    category = "stream-sink"

    def __init__(
            self,
            res="1s",
            output="output.tif",
            mode="mean",
            plug=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.res = res
        self.output = output
        self.plug = utils.str2bool(plug)
        self._accumulator = None

        self._global_mode = '{' not in self.output and '}' not in self.output

    def _create_accumulator(self, filename, region):
        """Factory method to create an accumulator."""

        if isinstance(self.res, str) and self.res.endswith("s"):
            inc = float(self.res[:-1]) / 3600.0
            x_inc, y_inc = inc, inc
        elif '/' in str(self.res):
            x_inc, y_inc = map(float, self.res.split("/"))
        else:
            inc = float(self.res)
            x_inc, y_inc = inc, inc

        return PointAccumulator(
            filename=filename,
            region=region,
            x_inc=x_inc,
            y_inc=y_inc,
            verbose=True
        )

    def _init_accumulator(self, region):
        """Initialize the single global accumulator."""

        if self._accumulator:
            return

        logger.info(f"Initializing Global Stack: {self.output}")
        self._accumulator = self._create_accumulator(self.output, region)

    def run(self, entries):
        new_entries = []
        processed_count = 0

        if self._global_mode and not self._accumulator:
            region = next((mod.region for mod, _ in entries if getattr(mod, "region", None)), None)
            if region:
                self._init_accumulator(region)
            else:
                # If no region found, we can't global stack. Return entries as-is.
                return entries

        for mod, entry in entries:
            stream = entry.get("stream")
            if not stream:
                new_entries.append((mod, entry))
                continue

            processed_count += 1

            if not self._global_mode:
                # --- Per-File Mode ---
                # Filename template
                src_base = os.path.splitext(os.path.basename(entry['dst_fn']))[0]
                out_fn = self.output.format(base=src_base, name=mod.name)

                if not os.path.isabs(out_fn):
                    out_fn = os.path.join(os.path.dirname(entry["dst_fn"]), out_fn)

                # Use module region for per-file stacking
                acc = self._create_accumulator(out_fn, mod.region)

                if self.plug:
                    # Consume -> Finalize -> Replace Entry
                    self._consume_stream(stream, acc)
                    acc.finalize()

                    entry["stream"] = None
                    entry["dst_fn"] = out_fn
                    entry["data_type"] = "raster"
                    new_entries.append((mod, entry))
                else:
                    entry["stream"] = self._intercept_per_file(stream, acc)
                    entry["dst_fn"] = out_fn
                    new_entries.append((mod, entry))

                entry.setdefault("artifacts", {})[self.name] = out_fn

            else:
                # --- Global Mode ---
                if not self._accumulator:
                    new_entries.append((mod, entry))
                    continue

                if self.plug:
                    self._consume_stream(stream, self._accumulator)
                else:
                    entry["stream"] = self._intercept(stream, entry)
                    new_entries.append((mod, entry))

        if self._global_mode and self.plug and processed_count > 0:
            if self._accumulator:
                self._accumulator.finalize()


                mod, _ = entries[0]
                stack_entry = {
                    "url": f"file://{self.output}",
                    "dst_fn": self.output,
                    "data_type": "raster",
                    "status": 0,
                }
                new_entries.append((mod, stack_entry))
                entry.setdefault("artifacts", {})[self.name] = self.output

                self._accumulator = None

        return new_entries

    def _consume_stream(self, stream, accumulator):
        """Drains the stream into the accumulator."""

        for chunk in stream:
            accumulator.add_points(chunk)

    def _intercept(self, stream, entry):
        """Pass-through generator for global accumulator."""

        for chunk in stream:
            if self._accumulator:
                self._accumulator.add_points(chunk)
            yield chunk

    def _intercept_per_file(self, stream, accumulator):
        """Pass-through generator for per-file accumulator."""

        for chunk in stream:
            accumulator.add_points(chunk)
            yield chunk
        accumulator.finalize()

    def teardown(self):
        """Called by Core after execution. Only used for Global Non-Plug mode."""

        if self._accumulator:
            self._accumulator.finalize()
            self._accumulator = None
