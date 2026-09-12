#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.hooks.rasters.binary_cudem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Binary CUDEM: Morphological Multi-Resolution Step-Down.
Uses weights to route specific datasets to specific resolutions
and interpolator settings, bridging gaps in sparse data without
degrading the high-frequency fidelity of dense coastal data.
"""

import os
import shutil
import logging
import numpy as np

import scipy.ndimage
from scipy.interpolate import griddata

import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize

import pyogrio
import shapely

import fetchez
from fetchez.spatial import Region
from fetchez.utils import (
    remove_glob2,
    str2inc,
    inc2str,
    int_or,
    str_or,
    parse_hook_string,
    parse_arg_to_list,
)
from fetchez.registry import HookRegistry

from .base import RasterGlobalHook

logger = logging.getLogger(__name__)


class BinaryCudemStepDown(RasterGlobalHook):
    """Multi-Resolution Morphological step-down."""

    name = "ms_binary_cudem"
    default_suffix = "_binary_cudem"
    meta_desc = (
        "Interpolate NoData voids using Multi Resolution Morphological stacking."
    )
    meta_tags = ["globato", "interpolation", "multi-stack"]
    meta_requires = "multi-stack"

    def __init__(
        self,
        steps=3,
        weights=None,
        resolutions=None,
        algos=None,
        blend_dists=None,
        decimation_mode="weighted_mean",
        bathy_max_z="-0.01",
        inland_decay_dist=5.0,  # km
        keep_steps=False,
        **kwargs,
    ):
        super().__init__(strip_bands=True, **kwargs)

        self.valid_algos = [
            "interp_gmt",
            "interp_rbf",
            "raster_fill",
            "interp_nn",
            "interp_idw",
            "interp_scipy",
        ]
        self.steps = int_or(steps)
        self.weights = parse_arg_to_list(weights, float)
        self.resolutions = parse_arg_to_list(resolutions, str2inc)
        self.blend_dists = parse_arg_to_list(blend_dists, int)
        self.algos = parse_arg_to_list(algos, str)
        self.decimation_mode = str_or(decimation_mode, "weighted_mean")
        self.keep_steps = keep_steps

        # Parse the spatial cap rules
        self.cap_rules = self._parse_cap_rules(bathy_max_z)
        self.inland_decay_dist = float(inland_decay_dist)

    def _parse_cap_rules(self, cap_input):
        if cap_input is None:
            return {}

        if isinstance(cap_input, (float, int)):
            return {"ocean": float(cap_input), "water": float(cap_input)}

        if isinstance(cap_input, str):
            try:
                val = float(cap_input)

                return {"ocean": val, "water": val}
            except ValueError:
                rules = {}
                for pair in cap_input.split(","):
                    if ":" in pair:
                        k, v = pair.split(":")
                        if v.strip().lower() != "none":
                            rules[k.strip().lower()] = float(v)
                return rules
        return {}

    def _generate_cap(self, z, missing_mask, ndv, barrier_mask=None):
        dilated = scipy.ndimage.binary_dilation(missing_mask)
        boundary_mask = dilated & (~missing_mask) & (~np.isnan(z)) & (z != ndv)

        if not np.any(boundary_mask):
            return None

        y_bnd, x_bnd = np.where(boundary_mask)
        z_bnd = z[y_bnd, x_bnd]
        y_void, x_void = np.where(missing_mask)

        cap_vals = griddata((y_bnd, x_bnd), z_bnd, (y_void, x_void), method="linear")

        nan_caps = np.isnan(cap_vals)
        if np.any(nan_caps):
            nearest_vals = griddata(
                (y_bnd, x_bnd),
                z_bnd,
                (y_void[nan_caps], x_void[nan_caps]),
                method="nearest",
            )
            cap_vals[nan_caps] = nearest_vals

        cap_grid = np.full_like(z, np.nan)
        cap_grid[y_void, x_void] = cap_vals

        return cap_grid

    def _setup_steps(self, src_path):
        target_tiers = max(
            self.steps + 1,
            len(self.resolutions),
            len(self.weights),
            len(self.blend_dists),
            len(self.algos),
        )
        self.steps = target_tiers - 1

        self.weights = sorted(self.weights, reverse=True)
        while len(self.weights) < self.steps:
            if len(self.weights) == 0:
                self.weights.append(1.0)

            next_weight = self.weights[-1] / 2.0
            if next_weight == 0:
                next_weight = 1e-20

            self.weights.append(next_weight)

        if self.weights[-1] > 0:
            self.weights.append(0.0)

        with rasterio.open(src_path) as src:
            base_res = src.profile["transform"][0]

        while len(self.resolutions) < target_tiers:
            if len(self.resolutions) == 0:
                self.resolutions.append(base_res)
            self.resolutions.append(self.resolutions[-1] * 3)

        if len(self.algos) == 0:
            self.algos = ["raster_fill:max_dist=10"] * max(0, self.steps)
            self.algos.append("interp_rbf")

        while len(self.algos) < target_tiers:
            self.algos.append(self.algos[-1])

        while len(self.blend_dists) < target_tiers:
            if len(self.blend_dists) == 0:
                self.blend_dists.append(20)

            self.blend_dists.append(self.blend_dists[-1])

    def _get_interp_hook(self, parsed_algo_hook):
        HookRegistry.load_all()

        algo_name = parsed_algo_hook["name"]
        algo_args = parsed_algo_hook.get("args", {})
        if algo_name in self.valid_algos:
            return HookRegistry.get_class(algo_name)(**algo_args)
        else:
            return HookRegistry.get_class("interp_rbf")()

    def _decimate_raster(self, src_path, dst_path, target_res):
        # local_tmp = os.path.abspath("tmp")
        # os.makedirs(local_tmp, exist_ok=True)

        with rasterio.open(src_path) as src:
            bounds = src.bounds
            region = Region(bounds.left, bounds.right, bounds.bottom, bounds.top)
            src_crs = src.crs.to_string() if src.crs else None
            region.srs = src_crs

        decimated_stack = fetchez.get(
            "file",
            outdir=self.local_tmp,
            region=region,
            region_srs=src_crs,
            path=src_path,
            use_cache=False,
            hooks=[
                "set_datatype:data_type=multi-stack",
                "stream-init",
                {
                    "name": "multi_stack",
                    "args": {
                        "res": target_res,
                        "output": dst_path,
                        "crs": src_crs,
                        "mode": "mixed",
                        "weight_threshold": "/".join([str(x) for x in self.weights]),
                        "overwrite": True,
                    },
                },
                "focus_sink:target=multi_stack",
            ],
        )
        return decimated_stack

    def _create_topological_grids(self, shape, transform, barrier_path):
        if not barrier_path:
            return None, None

        try:
            meta, fids, geometry_wkb, fields = pyogrio.raw.read(barrier_path)
            geoms = shapely.from_wkb(geometry_wkb)

            has_class = "class" in meta.get("fields", [])
            class_data = (
                fields[list(meta["fields"]).index("class")] if has_class else None
            )

            cap_shapes, land_shapes = [], []

            for i, geom in enumerate(geoms):
                if geom is None:
                    continue

                cls_name = str(class_data[i]).lower() if has_class else "land"

                if cls_name in ["land", "reef", "breakwater", "island"]:
                    land_shapes.append((geom, 1))

                cap_val = self.cap_rules.get(cls_name)
                if cap_val is not None:
                    cap_shapes.append((geom, cap_val))

            land_mask = (
                rasterize(
                    land_shapes,
                    out_shape=shape,
                    transform=transform,
                    fill=0,
                    dtype="uint8",
                ).astype(bool)
                if land_shapes
                else None
            )

            default_cap = self.cap_rules.get("ocean") or self.cap_rules.get("water")

            if default_cap is not None or cap_shapes:
                base_fill = default_cap if default_cap is not None else np.nan
                cap_grid = np.full(shape, base_fill, dtype="float32")

                if default_cap is not None and land_mask is not None:
                    cap_grid[land_mask] = np.nan

                if cap_shapes:
                    rasterize(
                        cap_shapes,
                        out_shape=shape,
                        transform=transform,
                        out=cap_grid,  # Updates the array in-place!
                        dtype="float32",
                    )
            else:
                cap_grid = None

            return cap_grid, land_mask
            # if cap_shapes:
            #     cap_grid = rasterize(
            #         cap_shapes,
            #         out_shape=shape,
            #         transform=transform,
            #         fill=np.nan,
            #         dtype="float32",
            #     )
            # elif not has_class:
            #     default_cap = self.cap_rules.get("ocean") or self.cap_rules.get("water")
            #     if default_cap is not None:
            #         cap_grid = np.full(shape, default_cap, dtype="float32")

            #         if land_mask is not None:
            #             cap_grid[land_mask] = np.nan
            #     else:
            #         cap_grid = None
            # else:
            #     cap_grid = None

            # return cap_grid, land_mask

        except Exception as e:
            logger.error(f"[{self.name}] Failed to generate topological grids: {e}")
            return None, None

    def _process_tier(
        self,
        step_stack,
        previous_surface,
        current_weight,
        current_algo,
        current_blend_dist,
        is_coarsest,
        barrier_path,
        d2c_path=None,
    ):
        with rasterio.open(step_stack, "r+") as src:
            data = src.read()
            ndv = src.nodata if src.nodata is not None else -9999
            z = data[0].astype("float64")
            w = data[2].astype("float64")

            cap_grid, barrier_mask = self._create_topological_grids(
                z.shape, src.transform, barrier_path
            )

            # --- D2C Cap Inland Decay ---
            if cap_grid is not None and d2c_path is not None:
                d2c_grid = np.zeros(z.shape, dtype="float32")
                with rasterio.open(d2c_path) as d2c_src:
                    reproject(
                        source=rasterio.band(d2c_src, 1),
                        destination=d2c_grid,
                        src_transform=d2c_src.transform,
                        src_crs=d2c_src.crs,
                        dst_transform=src.transform,
                        dst_crs=src.crs,
                        resampling=Resampling.bilinear,
                        num_threads=1,
                    )

                # Find water pixels that are inland (negative D2C value)
                inland_water = (d2c_grid < 0) & (~np.isnan(cap_grid))
                dist_inland = np.abs(np.minimum(d2c_grid, 0))

                # Weight is 1.0 at coastline, decaying to 0.0 at self.inland_decay_dist
                decay_weight = np.clip(
                    1.0 - (dist_inland / self.inland_decay_dist), 0.0, 1.0
                )

                with np.errstate(divide="ignore", invalid="ignore"):
                    faded_caps = cap_grid[inland_water] / (
                        decay_weight[inland_water] + 1e-6
                    )

                # If weight approaches zero, remove the cap constraint entirely
                faded_caps[decay_weight[inland_water] < 0.01] = np.nan
                cap_grid[inland_water] = faded_caps

            valid_mask = (z != ndv) & (~np.isnan(z))
            core_mask = w >= current_weight

            if is_coarsest:
                tier_zone = np.ones_like(core_mask, dtype=bool)
                missing_mask = ~valid_mask
            else:
                dist_to_core = scipy.ndimage.distance_transform_edt(~core_mask)
                tier_zone = dist_to_core <= current_blend_dist
                missing_mask = (~valid_mask) & tier_zone

            y_miss, x_miss = np.where(missing_mask)
            if len(y_miss) > 0:
                temp_in = step_stack.replace(".tif", f"_{current_weight}_in.tif")
                temp_out = step_stack.replace(".tif", f"_{current_weight}_out.tif")
                shutil.copy(step_stack, temp_in)

                current_algo_hook = parse_hook_string(current_algo)
                interp_hook = self._get_interp_hook(current_algo_hook)
                success = interp_hook.process_raster(step_stack, temp_out, entry={})

                if success and os.path.exists(temp_out):
                    with rasterio.open(temp_out) as filled_src:
                        filled_z = filled_src.read(1)
                        z[y_miss, x_miss] = filled_z[y_miss, x_miss]

                if os.path.exists(temp_in):
                    os.remove(temp_in)

                if os.path.exists(temp_out):
                    os.remove(temp_out)
            else:
                logger.info("Tier is filled by data; nothing to interpolate")

            if previous_surface:
                bg_aligned = np.full(z.shape, ndv, dtype=z.dtype)
                with rasterio.open(previous_surface) as bg_src:
                    reproject(
                        source=rasterio.band(bg_src, 1),
                        destination=bg_aligned,
                        src_transform=bg_src.transform,
                        src_crs=bg_src.crs,
                        dst_transform=src.transform,
                        dst_crs=src.crs,
                        src_nodata=bg_src.nodata,
                        dst_nodata=ndv,
                        resampling=Resampling.bilinear,
                        num_threads=1,
                    )

                if not np.any(core_mask):
                    z[:] = bg_aligned[:]
                else:
                    if barrier_mask is not None:
                        dist_land = scipy.ndimage.distance_transform_edt(
                            ~(core_mask & barrier_mask)
                        )
                        dist_water = scipy.ndimage.distance_transform_edt(
                            ~(core_mask & ~barrier_mask)
                        )
                        dist = np.where(barrier_mask, dist_land, dist_water)
                    else:
                        dist = scipy.ndimage.distance_transform_edt(~core_mask)

                    if current_blend_dist > 0:
                        weights = np.clip(dist / float(current_blend_dist), 0.0, 1.0)
                        bg_only_mask = dist >= current_blend_dist
                        trans_mask = (dist > 0) & (dist < current_blend_dist)
                    else:
                        weights = np.ones_like(dist)
                        bg_only_mask = dist > 0
                        trans_mask = np.zeros_like(dist, dtype=bool)

                    z[bg_only_mask] = bg_aligned[bg_only_mask]

                    if np.any(trans_mask):
                        valid_bg = (bg_aligned != ndv) & (~np.isnan(bg_aligned))
                        valid_z = (z != ndv) & (~np.isnan(z))
                        blend_active = trans_mask & valid_bg & valid_z
                        z[blend_active] = (
                            (1.0 - weights[blend_active]) * z[blend_active]
                        ) + (weights[blend_active] * bg_aligned[blend_active])

                    remaining_holes = (z == ndv) | np.isnan(z)
                    valid_bg_overall = (bg_aligned != ndv) & (~np.isnan(bg_aligned))
                    fill_from_bg = remaining_holes & valid_bg_overall
                    z[fill_from_bg] = bg_aligned[fill_from_bg]

            # --- Topological Capping (Post-Interpolation) ---
            if cap_grid is not None:
                to_cap = (z != ndv) & (~np.isnan(z)) & (~np.isnan(cap_grid))
                z[to_cap] = np.minimum(z[to_cap], cap_grid[to_cap])

            src.write(z.astype(rasterio.float32), 1)

    def process_raster(self, src_path, dst_path, entry):
        previous_surface = None
        self._setup_steps(src_path)

        barrier_path = self._get_barrier(output_mode="topology")

        d2c_path = None
        if self.inland_decay_dist > 0:
            try:
                with rasterio.open(src_path) as base_src:
                    bounds = base_src.bounds
                fetch_region = [bounds.left, bounds.right, bounds.bottom, bounds.top]

                logger.info(
                    f"[{self.name}] Fetching Dist2Coast for region {fetch_region}..."
                )
                d2c_files = fetchez.get(
                    "dist2coast",
                    region=fetch_region,
                    variant="base",
                    outdir=os.path.join(self.local_tmp, "auto_barriers"),
                    use_cache=True,
                    verbose=False,
                )
                if d2c_files:
                    d2c_path = d2c_files[0]
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to fetch D2C grid: {e}")

        with rasterio.open(src_path) as base_src:
            native_res = base_src.transform[0]

        for i, res_str in enumerate(reversed(self.resolutions)):
            current_weight = self.weights[::-1][i]
            current_algo = self.algos[::-1][i]
            current_blend_dist = self.blend_dists[::-1][i]
            is_coarsest = i == 0

            step_stack = src_path.replace(".tif", f"_step_{inc2str(res_str)}.tif")

            if np.isclose(res_str, native_res, atol=1e-9):
                shutil.copy(src_path, step_stack)
            else:
                self._decimate_raster(src_path, step_stack, target_res=res_str)

            self._process_tier(
                step_stack,
                previous_surface,
                current_weight,
                current_algo,
                current_blend_dist,
                is_coarsest,
                barrier_path=barrier_path,
                d2c_path=d2c_path,
            )

            if previous_surface and not self.keep_steps:
                remove_glob2(f"{previous_surface.split('.')[0]}.*")

            previous_surface = step_stack

        if previous_surface:
            if self.keep_steps:
                shutil.copy(previous_surface, dst_path)
            else:
                shutil.move(previous_surface, dst_path)
                remove_glob2(
                    "temp_stack_step*.tif",
                    "temp_interp_step*.tif",
                    "*.blend.tif",
                    "*_step_*.tif*",
                    f"{previous_surface.split('.')[0]}.*",
                )
            return True

        return False
