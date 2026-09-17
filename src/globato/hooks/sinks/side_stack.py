#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.hooks.sinks.side_stack
~~~~~~~~~~~~~~~~~~~~~~~

Generates a 'side-stack' for every entry; The mainifest hook checks
for an existing side-stack and replaces the entry dst_fn with it if
its found; the stream hook checks for a side-stack and generates one
if it doesn't exist.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import hashlib
import logging
import rasterio
import numpy as np

from fetchez.hooks import FetchHook
from globato.hooks.transforms.point_pixels import Point2PixelStream, PixelsToPoints

logger = logging.getLogger(__name__)


def get_sidestack_hash(region, res, crs, mode):
    """Helper to generate the invalidation hash."""
    region_str = region.format("fn") if region else "global"
    seed = f"{region_str}_{res}_{crs}_{mode}".encode("utf-8")
    return hashlib.md5(seed).hexdigest()[:8]


def get_sidestack_path(entry, cache_dir, hash_str):
    """Helper to resolve the sidestack path."""
    base_dir = cache_dir or os.path.dirname(entry.get("dst_fn", "."))
    os.makedirs(base_dir, exist_ok=True)

    base_name = entry.get("_orig_basename")
    if not base_name:
        base_name = os.path.splitext(os.path.basename(entry.get("dst_fn", "stream")))[0]

    return os.path.join(base_dir, f"{base_name}_{hash_str}_sidestack.tif")


# ==========================================
# HOOK 1: The Manifest Swap
# ==========================================
class SideStackCheck(FetchHook):
    """Manifest hook to intercept cache hits and hijack the entry."""

    name = "side_stack_check"
    meta_stage = "manifest"

    def __init__(
        self, res="1s", crs="EPSG:4326", mode="sums", cache_dir=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.res = res
        self.crs = crs
        self.mode = mode
        self.cache_dir = cache_dir

    def run(self, entries):
        for mod, entry in entries:
            region = getattr(mod, "region", None)
            if not region:
                continue

            if "_orig_basename" not in entry:
                entry["_orig_basename"] = os.path.splitext(
                    os.path.basename(entry.get("dst_fn", "stream"))
                )[0]

            cache_dir = self.cache_dir or getattr(
                mod, "_outdir", getattr(mod, "outdir", None)
            )

            hash_str = get_sidestack_hash(region, self.res, self.crs, self.mode)
            cache_path = get_sidestack_path(entry, cache_dir, hash_str)

            if os.path.exists(cache_path):
                logger.debug(
                    f"Sidestack Hit! Swapping source to {os.path.basename(cache_path)}"
                )

                # Hijack the entry
                entry["dst_fn"] = cache_path
                entry["data_type"] = "multi-stack"  # This triggers your YAML profile!

                # Sanitize Hooks: Strip anything that might break oaur TIF reader
                _safe_hooks = ["sidestack_generate", "multi_stack", "set_srs"]
                entry["srs"] = None
                mod.clear_hooks()

                from fetchez.hooks.stream_init import DataStream

                mod.add_hook(DataStream())
                # mod.hooks = []
                # if hasattr(mod, "hooks"):
                #     mod.hooks = [] # [h for h in mod.hooks if h.get("name") in safe_hooks]

                # if "hooks" in entry:
                #     entry["hooks"] = [] # [h for h in entry["hooks"] if h.get("name") in safe_hooks]

        return entries


# ==========================================
# HOOK 2: The Stream Generator
# ==========================================
class SideStackGenerate(FetchHook):
    """Stream hook to build the cache on a miss."""

    name = "side_stack"
    meta_stage = "stream"

    def __init__(
        self, res="1s", crs="EPSG:4326", mode="sums", cache_dir=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.res = res
        self.crs = crs
        self.mode = mode
        self.cache_dir = cache_dir

    def _save_raster_stream(self, raster_stream, cache_path):
        """Intercepts, finalized for disk, and passes the raw sums along."""
        profile = next(raster_stream)
        profile.update(
            compress="lzw", tiled=True, blockxsize=256, blockysize=256, dtype="float32"
        )
        yield profile

        with rasterio.open(cache_path, "w", **profile) as dst:
            for chunk in raster_stream:
                window, buff_win, data, ndv, transform = chunk

                z, cnt, w, unc, src_u, x, y = data
                valid = cnt > 0

                out_data = np.full_like(data, -9999, dtype=np.float32)

                with np.errstate(divide="ignore", invalid="ignore"):
                    out_data[0][valid] = z[valid] / w[valid]  # Mean Z
                    out_data[1][valid] = cnt[valid]  # Count
                    out_data[2][valid] = w[valid] / cnt[valid]  # Mean W
                    out_data[3][valid] = np.sqrt(unc[valid]) / cnt[valid]  # Unc
                    out_data[4][valid] = src_u[valid] / w[valid]  # Src Unc
                    out_data[5][valid] = x[valid] / w[valid]  # True X!
                    out_data[6][valid] = y[valid] / w[valid]  # True Y!

                out_data[np.isinf(out_data)] = -9999
                out_data[np.isnan(out_data)] = -9999

                dst.write(out_data, window=window)

                yield chunk

    def run(self, entries):
        p2p = Point2PixelStream(
            x_inc=self.res, y_inc=self.res, want_sums=(self.mode == "sums")
        )
        px2pts = PixelsToPoints()

        for mod, entry in entries:
            if entry.get("data_type") == "multi-stack":
                # entry["stream"] = px2pts._raster_to_xyz(entry.get("dst_fn"))
                continue

            region = getattr(mod, "region", None)
            if not region or not self.is_point_stream(entry):
                continue

            cache_dir = self.cache_dir or getattr(
                mod, "_outdir", getattr(mod, "outdir", None)
            )

            hash_str = get_sidestack_hash(region, self.res, self.crs, self.mode)
            cache_path = get_sidestack_path(entry, cache_dir, hash_str)

            logger.debug(
                f"Sidestack Miss. Generating cache at {os.path.basename(cache_path)}"
            )

            raster_stream = p2p._stream_wrapper(
                entry["stream"], entry=entry, region=region
            )
            saved_raster_stream = self._save_raster_stream(raster_stream, cache_path)
            entry["stream"] = px2pts._raster_to_xyz(saved_raster_stream)

        return entries
