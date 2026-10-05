#!/usr/bin/env python

"""
globato.hooks.rasters.rio_warp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Warp/reproject a raster dataset

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import os

import rasterio
from fetchez.hooks import FetchHook
from fetchez.utils import str2inc
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
    transform_bounds,
)

logger = logging.getLogger(__name__)


class RioWarpHook(FetchHook):
    """Reprojects and/or resamples physical raster files to a target CRS and cell-size.
    Safely ignores non-raster data types.
    """

    name = "raster_warp"
    meta_stage = "file"
    meta_category = "raster-file"
    meta_desc = "Reprojects, resamples, and clips raster files."

    def __init__(self, dst_crs=None, res=None, resampling="bilinear", **kwargs):
        super().__init__(**kwargs)
        self.dst_crs = dst_crs
        self.res = res
        self.resampling = getattr(Resampling, resampling.lower(), Resampling.bilinear)

    def run(self, entries):
        for mod, entry in entries:
            src_fn = entry.get("dst_fn")

            if not src_fn or not src_fn.lower().endswith(
                (".tif", ".tiff", ".nc", ".vrt")
            ):
                continue

            tmp_dir = os.path.abspath("tmp")
            os.makedirs(tmp_dir, exist_ok=True)

            basename = os.path.basename(src_fn)
            name, ext = os.path.splitext(basename)
            out_fn = os.path.join(tmp_dir, f"{name}_warped{ext}")

            if os.path.exists(out_fn):
                logger.debug(f"[{self.name}] Warped file already exists: {out_fn}")
                entry["dst_fn"] = out_fn
                continue

            try:
                with rasterio.open(src_fn) as src:
                    target_crs = (
                        self.dst_crs
                        if self.dst_crs
                        else (src.crs.to_string() if src.crs else None)
                    )

                    target_res = None
                    if self.res:
                        target_res = str2inc(self.res)

                    needs_warp = False

                    if self.dst_crs and src.crs and src.crs.to_string() != self.dst_crs:
                        needs_warp = True

                    if target_res:
                        if (
                            abs(src.res[0] - target_res) > 1e-6
                            or abs(src.res[1] - target_res) > 1e-6
                        ):
                            needs_warp = True

                    if not needs_warp or not target_crs:
                        continue

                    w, s, e, n = src.bounds

                    if hasattr(mod, "region") and mod.region and mod.region.valid_p():
                        region_bounds = (
                            mod.region.xmin,
                            mod.region.ymin,
                            mod.region.xmax,
                            mod.region.ymax,
                        )

                        r_w, r_s, r_e, r_n = transform_bounds(
                            mod.region.srs, src.crs, *region_bounds
                        )

                        # Intersect the bounds
                        w = max(w, r_w)
                        s = max(s, r_s)
                        e = min(e, r_e)
                        n = min(n, r_n)

                        # Skip if there is no spatial overlap
                        if w >= e or s >= n:
                            logger.warning(
                                f"[{self.name}] {basename} does not overlap with target region."
                            )
                            continue

                    res_log = target_res if target_res else "Native"
                    logger.info(
                        f"[{self.name}] Warping {os.path.basename(src_fn)} (CRS: {target_crs}, Res: {res_log})..."
                    )

                    transform, width, height = calculate_default_transform(
                        src.crs,
                        target_crs,
                        src.width,
                        src.height,
                        w,
                        s,
                        e,
                        n,
                        resolution=target_res,
                    )

                    kwargs = src.profile.copy()
                    kwargs.update(
                        {
                            "crs": target_crs,
                            "transform": transform,
                            "width": width,
                            "height": height,
                            "compress": "deflate",
                            "tiled": True,
                            "blockxsize": 256,
                            "blockysize": 256,
                        }
                    )

                    # temp_fn = src_fn + ".warp.tif"
                    with rasterio.open(out_fn, "w", **kwargs) as dst:
                        for i in range(1, src.count + 1):
                            reproject(
                                source=rasterio.band(src, i),
                                destination=rasterio.band(dst, i),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=transform,
                                dst_crs=target_crs,
                                resampling=self.resampling,
                            )

                # shutil.move(temp_fn, src_fn)
                entry["dst_fn"] = out_fn

            except Exception as e:
                logger.error(f"[{self.name}] Failed to warp {src_fn}: {e}")

        return entries
