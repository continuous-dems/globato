#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.sinks.multi_stack
~~~~~~~~~~~~~~~~~~~~~~~

Multi-band Statistical Gridder (The "Heavy" Stacker).
Generates Z, Count, Weight, Uncertainty, etc.
Maintains a '.sums.tif' for continuous updates and provenance tracking.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import sys
import json
import logging
import threading
import numpy as np

import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin
from rasterio.crs import CRS
from rasterio.enums import ColorInterp

from transformez.spatial import TransRegion as Region
from fetchez.hooks import FetchHook
from fetchez.utils import float_or, parse_fmod
from fetchez import utils

from ..transforms.point_pixels import PointPixels

logger = logging.getLogger(__name__)


# MULTI_STACK ACCUMULATOR
class MultiStackAccumulator:
    """Multi-band statistical grid accumulator"""

    BAND_MAP = {
        "z": 1, "count": 2, "weights": 3, "uncertainty": 4,
        "src_uncertainty": 5, "x": 6, "y": 7
    }

    def __init__(
            self,
            region,
            x_inc,
            y_inc,
            output_fn,
            mode="mean",
            weight_threshold="1",
            crs="EPSG:4326",
            verbose=False,
    ):
        self.region = Region.from_list(region)
        self.x_inc = float(x_inc)
        self.y_inc = float(y_inc)
        self.output_fn = output_fn
        self.mode = mode.lower()
        self.crs = crs
        self.verbose = verbose
        self.lock = threading.Lock()

        base, ext = os.path.splitext(self.output_fn)
        self.sums_fn = f"{base}.sums{ext}"

        self.wts = np.sort([float(x) for x in str(weight_threshold).split('/')])

        self.xcount, self.ycount, self.dst_gt = self.region.geo_transform(
            x_inc=self.x_inc, y_inc=self.y_inc, node="grid"
        )

        self.transform = rasterio.transform.from_origin(
            self.dst_gt[0], self.dst_gt[3], self.dst_gt[1], abs(self.dst_gt[5])
        )

        self._init_raster()

        self.pixel_binner = PointPixels(
            src_region=self.region,
            x_size=self.xcount,
            y_size=self.ycount
        )


    def _init_raster(self):
        """Create the zero-filled accumulation file or load existing."""

        if os.path.exists(self.sums_fn):
            logger.info(
                f"Found existing sums file: {os.path.basename(self.sums_fn)}. Operating in UPDATE mode."
            )
            return

        if not os.path.exists(os.path.dirname(os.path.abspath(self.sums_fn))):
            os.makedirs(os.path.dirname(os.path.abspath(self.sums_fn)))

        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "nodata": -9999,
            "width": self.xcount,
            "height": self.ycount,
            "count": 7,
            "crs": CRS.from_string(self.crs) if self.crs else None,
            "transform": self.transform,
            "tiled": True,
            "compress": "lzw",
            "predictor": 2,
            "bigtiff": "YES",
        }

        with rasterio.open(self.sums_fn, "w", **profile) as dst:
            for key, idx in self.BAND_MAP.items():
                dst.set_band_description(idx, key)

            dst.update_tags(GLOBATO_PROVENANCE="[]")

    def is_registered(self, dataset_id):
        """Check the GeoTIFF header to see if dataset is already stacked."""

        if not os.path.exists(self.sums_fn):
            return False

        with self.lock:
            with rasterio.open(self.sums_fn, "r") as src:
                reg_str = src.tags().get("GLOBATO_PROVENANCE", "[]")
                registry = json.loads(reg_str)
                return dataset_id in registry

    def mark_registered(self, dataset_id):
        """Add dataset to the GeoTIFF header registry."""

        with self.lock:
            with rasterio.open(self.sums_fn, "r+") as dst:
                reg_str = dst.tags().get("GLOBATO_PROVENANCE", "[]")
                registry = json.loads(reg_str)
                if dataset_id not in registry:
                    registry.append(dataset_id)
                    dst.update_tags(GLOBATO_PROVENANCE=json.dumps(registry))
                    logger.debug(
                        f"Registered {os.path.basename(dataset_id)} in stack provenance."
                    )

    def update(self, points):
        """Process a chunk of points: Bin in memory -> Update Disk."""

        if points is None or len(points) == 0:
            return

        arrays, sub_win, _ = self.pixel_binner(points, mode="sums")
        if arrays['z'] is None:
            return

        col_off, row_off, width, height = sub_win
        window = Window(col_off, row_off, width, height)

        with self.lock:
            with rasterio.open(self.sums_fn, 'r+') as dst:
                current_data = dst.read(window=window)

                def get_band(name):
                    return current_data[self.BAND_MAP[name]-1]

                valid_new = arrays["count"] > 0

                current_data[current_data == -9999] = 0
                current_data[np.isnan(current_data)] = 0

                if self.mode in ["mean", "weighted_mean"]:
                    get_band("z")[valid_new] += arrays["z"][valid_new]
                    get_band("weights")[valid_new] += arrays["weightd"][valid_new]
                    get_band("count")[valid_new] += arrays["count"][valid_new]
                    get_band("uncertainty")[valid_new] += np.square(arrays["uncertainty"][valid_new])

                    if "src_uncertainty" in arrays and arrays["src_uncertainty"] is not None:
                         get_band("src_uncertainty")[valid_new] += arrays["src_uncertainty"][valid_new]

                    get_band('x')[valid_new] += arrays["x"][valid_new]
                    get_band('y')[valid_new] += arrays["y"][valid_new]

                elif self.mode in ["supercede", "mixed"]:
                    cur_cnt = get_band("count")
                    cur_wt = get_band("weights")

                    with np.errstate(divide="ignore", invalid="ignore"):
                        arr_w_avg = np.where(valid_new, arrays["weight"] / arrays["count"], 0)
                        cur_w_avg = np.where(cur_cnt > 0, cur_wt / cur_cnt, 0)

                    if self.mode == "supercede":
                        sup_mask = valid_new & (arr_w_avg > cur_w_avg)
                        avg_mask = np.zeros_like(sup_mask, dtype=bool)
                    else:
                        arr_tier = np.digitize(arr_w_avg, self.wts)
                        cur_tier = np.digitize(cur_w_avg, self.wts)
                        cur_tier[cur_cnt == 0] = -1

                        sup_mask = valid_new & (arr_tier > cur_tier)
                        avg_mask = valid_new & (arr_tier == cur_tier)

                    if np.any(sup_mask):
                        get_band("z")[sup_mask] = arrays["z"][sup_mask]
                        get_band("weights")[sup_mask] = arrays["weight"][sup_mask]
                        get_band("count")[sup_mask] = arrays["count"][sup_mask]
                        get_band("uncertainty")[sup_mask] = np.square(arrays["uncertainty"][sup_mask])

                        if "src_uncertainty" in arrays and arrays["src_uncertainty"] is not None:
                            get_band("src_uncertainty")[sup_mask] = arrays["src_uncertainty"][sup_mask]

                        get_band("x")[sup_mask] = arrays["x"][sup_mask]
                        get_band("y")[sup_mask] = arrays["y"][sup_mask]

                    if np.any(avg_mask):
                        get_band("z")[avg_mask] += arrays["z"][avg_mask]
                        get_band("weights")[avg_mask] += arrays["weight"][avg_mask]
                        get_band("count")[avg_mask] += arrays["count"][avg_mask]
                        get_band("uncertainty")[avg_mask] += np.square(arrays["uncertainty"][avg_mask])

                        if "src_uncertainty" in arrays and arrays["src_uncertainty"] is not None:
                            get_band("src_uncertainty")[avg_mask] += arrays["src_uncertainty"][avg_mask]

                        get_band("x")[avg_mask] += arrays["x"][avg_mask]
                        get_band("y")[avg_mask] += arrays["y"][avg_mask]

                elif self.mode == "min":
                    cur_z = get_band("z")
                    cur_z[~valid_new & (cur_z == 0)] = 999999
                    update_mask = valid_new & (arrays["z"] < cur_z)
                    get_band("z")[update_mask] = arrays["z"][update_mask]
                    get_band("count")[update_mask] = 1

                elif self.mode == "max":
                    cur_z = get_band("z")
                    cur_z[~valid_new & (cur_z == 0)] = -999999
                    update_mask = valid_new & (arrays["z"] > cur_z)
                    get_band("z")[update_mask] = arrays["z"][update_mask]
                    get_band("count")[update_mask] = 1

                dst.write(current_data, window=window)

    def finalize(self, ndv=-9999):
        """Convert accumulated sums from .sums.tif into the final output .tif."""

        if self.verbose:
            logger.info(
                f"Finalizing Averages: {os.path.basename(self.sums_fn)} -> {os.path.basename(self.output_fn)}"
            )

        with rasterio.open(self.sums_fn, "r") as src:
            profile = src.profile.copy()

            with rasterio.open(self.output_fn, "w", **profile) as dst:
                dst.colorinterp = [ColorInterp.undefined] * dst.count

                for _, window in src.block_windows(1):
                    data = src.read(window=window)

                    z = data[self.BAND_MAP["z"]-1]
                    cnt = data[self.BAND_MAP["count"]-1]
                    w = data[self.BAND_MAP["weights"]-1]
                    unc = data[self.BAND_MAP["uncertainty"]-1]
                    src_u = data[self.BAND_MAP["src_uncertainty"]-1]
                    x = data[self.BAND_MAP["x"]-1]
                    y = data[self.BAND_MAP["y"]-1]

                    valid = cnt > 0
                    data[:, ~valid] = ndv

                    if self.mode in ["mean", "weighted_mean", "mixed", "supercede"]:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            z[valid] = z[valid] / w[valid]
                            x[valid] = x[valid] / w[valid]
                            y[valid] = y[valid] / w[valid]
                            src_u[valid] = src_u[valid] / w[valid]
                            unc[valid] = np.sqrt(unc[valid]) / cnt[valid]

                    dst.write(data, window=window)

                # Copy the provenance registry over to the final file!
                dst.update_tags(**src.tags())

        # Generate Statistics on the Finalized TIF
        with rasterio.open(self.output_fn, 'r+') as dst:
            stats_dict = {}
            for idx in range(1, dst.count + 1):
                stats_dict[idx] = dst.statistics(idx, approx=False, clear_cache=True)

            for idx, stats in stats_dict.items():
                desc = [k for k, v in self.BAND_MAP.items() if v == idx][0]
                dst.update_tags(bidx=idx,
                    STATISTICS_MINIMUM=str(stats.min),
                    STATISTICS_MAXIMUM=str(stats.max),
                    STATISTICS_MEAN=str(stats.mean),
                    STATISTICS_STDDEV=str(stats.std),
                    DESCRIPTION=desc
                )

        return self.output_fn


# MULTI_STACK HOOK
class MultiStackHook(FetchHook):
    """Multi_Stack Gridding Hook.

    accumulates streaming data into a multi-band statistical grid.
    Maintains a continuous .sums.tif to prevent duplication.
    """

    name = "multi_stack"
    stage = "file"
    category = "stream-sink"

    def __init__(
            self,
            res="1s",
            output="multi_stack_output.tif",
            mode="mean",
            weight_threshold="1",
            crs=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.res = res
        self.output = output
        self.mode = mode.lower()
        self.weight_threshold = weight_threshold
        self.crs = crs
        self._accumulator = None

    def _init_accumulator(self, region):
        if self._accumulator:
            return

        if isinstance(self.res, str) and self.res.endswith("s"):
            inc = float(self.res[:-1]) / 3600.0
            x_inc, y_inc = inc, inc
        elif '/' in str(self.res):
            x_inc, y_inc = map(float, self.res.split("/"))
        else:
            inc = float(self.res)
            x_inc, y_inc = inc, inc

        logger.info(
            f"Initializing Multi_Stack: {self.output} @ {x_inc},{y_inc} ({self.mode})"
        )
        self._accumulator = MultiStackAccumulator(
            region=region,
            x_inc=x_inc,
            y_inc=y_inc,
            output_fn=self.output,
            mode=self.mode,
            weight_threshold=self.weight_threshold,
            crs=self.crs,
            verbose=True
        )

    def run(self, entries):
        if not self._accumulator:
            region = next((mod.region for mod, _ in entries if getattr(mod, "region", None)), None)
            if region:
                self._init_accumulator(region)
            else:
                return entries

        for mod, entry in entries:
            dataset_id = entry.get("checksum")

            if not dataset_id:
                url = entry.get("url", "")
                dst_fn = entry.get("dst_fn")

                if url and not url.startswith("file://"):
                    dataset_id = url

                elif dst_fn and os.path.exists(dst_fn):
                    size = os.path.getsize(dst_fn)
                    dataset_id = f"{os.path.basename(dst_fn)}|{size}B"

                else:
                    dataset_id = os.path.basename(dst_fn or url or "unknown_dataset")

            if self._accumulator and self._accumulator.is_registered(dataset_id):
                logger.debug(f"Dataset '{dataset_id}' already inside stack. Skipping.")
                entry.pop("stream", None)
            else:
                stream = entry.get("stream")
                if stream:
                    entry["stream"] = self._intercept(stream, dataset_id)

            entry.setdefault("artifacts", {})[self.name] = self.output

        return entries

    def _intercept(self, stream, dataset_id):
        """Generator wrapper to feed the accumulator and mark registry."""

        count = 0
        for chunk in stream:
            count += len(chunk)
            if self._accumulator:
                self._accumulator.update(chunk)
            yield chunk
        logger_str = f"Passed {count} data points from {dataset_id}"
        logger.info(f"{utils.colorize(logger_str, utils.BOLD):<15}")

        # The stream is exhausted; permanently mark this dataset as completed
        if self._accumulator and dataset_id:
            self._accumulator.mark_registered(dataset_id)

    def teardown(self):
        """Finalize the grid after all streams are exhausted."""

        if self._accumulator:
            logger.info("Streams finished. Finalizing averages...")
            self._accumulator.finalize()
            self._accumulator = None
