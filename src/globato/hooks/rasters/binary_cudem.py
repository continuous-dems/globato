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
import json
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
    float_or,
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
        previous_tier_mode="raster",
        previous_tier_resampling="bilinear",
        bathy_max_z="-0.01",
        inland_decay_dist=5.0,  # km
        keep_steps=True,
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
        self.previous_tier_mode = str_or(previous_tier_mode, "raster").lower()
        self.previous_tier_resampling = str_or(
            previous_tier_resampling, "bilinear"
        ).lower()
        self.keep_steps = keep_steps

        if self.previous_tier_mode not in {"points", "raster"}:
            raise ValueError("previous_tier_mode must be either 'points' or 'raster'")

        self.bathy_max_z = float_or(bathy_max_z)
        # Parse the spatial cap rules
        self.cap_rules = self._parse_cap_rules(bathy_max_z)
        self.inland_decay_dist = float(inland_decay_dist)

    def _stack_weight_tiers(self, src_path):
        with rasterio.open(src_path) as src:
            tags = src.tags()

        raw = tags.get("GLOBATO_WEIGHT_TIERS")
        if not raw:
            return []

        try:
            return sorted(
                [float(v) for v in json.loads(raw)],
                reverse=True,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning(
                "[%s] Invalid GLOBATO_WEIGHT_TIERS metadata: %r",
                self.name,
                raw,
            )
            return []

    @staticmethod
    def _apply_topological_cap(z, cap_grid, observed_mask, ndv):
        to_cap = ~observed_mask & (z != ndv) & np.isfinite(z) & np.isfinite(cap_grid)

        z[to_cap] = np.minimum(z[to_cap], cap_grid[to_cap])
        return z

    def _observed_land_mask(self, z, valid_mask, core_mask):
        observed_land = valid_mask & core_mask & (z > 0)

        structure = scipy.ndimage.generate_binary_structure(2, 2)
        closed_land = scipy.ndimage.binary_closing(
            observed_land,
            structure=structure,
            iterations=1,
        )

        # Never lose actual positive observations.
        return observed_land | closed_land

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
                        # if v.strip().lower() != "none":
                        rules[k.strip().lower()] = float_or(v)
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

        if not self.weights:
            self.weights = self._stack_weight_tiers(src_path)
        else:
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
                else:
                    cap_shapes.append((geom, np.nan))

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

            default_cap = self.cap_rules.get("ocean")
            if default_cap is None:
                default_cap = self.cap_rules.get("water")

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

    def _raster_resampling(self):
        """Return the configured Rasterio resampling method."""
        methods = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
            "cubic_spline": Resampling.cubic_spline,
            "lanczos": Resampling.lanczos,
        }
        try:
            return methods[self.previous_tier_resampling]
        except KeyError as exc:
            valid = ", ".join(sorted(methods))
            raise ValueError(
                f"Unknown previous_tier_resampling "
                f"'{self.previous_tier_resampling}'; choose from {valid}"
            ) from exc

    def _align_background(self, previous_surface, shape, transform, crs, ndv):
        """Resample the previous coarser surface onto the current tier grid."""
        bg_aligned = np.full(shape, ndv, dtype="float64")

        with rasterio.open(previous_surface) as bg_src:
            reproject(
                source=rasterio.band(bg_src, 1),
                destination=bg_aligned,
                src_transform=bg_src.transform,
                src_crs=bg_src.crs,
                dst_transform=transform,
                dst_crs=crs,
                src_nodata=bg_src.nodata,
                dst_nodata=ndv,
                resampling=self._raster_resampling(),
                num_threads=1,
            )

        return bg_aligned

    def _previous_tier_guides(
        self,
        previous_surface,
        shape,
        transform,
        crs,
        ndv,
    ):
        """Map previous-tier cell centers to sparse current-tier constraints.

        The completed previous surface is sampled only at its native cell
        centers.  Those values become guide constraints for a fresh
        interpolation at the current resolution; the coarse raster is never
        promoted into a continuous finer-resolution background.
        """
        guide_z = np.full(shape, ndv, dtype="float64")
        guide_mask = np.zeros(shape, dtype=bool)

        with rasterio.open(previous_surface) as src:
            prev_z = src.read(1).astype("float64")
            prev_ndv = src.nodata

            valid = np.isfinite(prev_z)
            if prev_ndv is not None:
                valid &= prev_z != prev_ndv

            if not np.any(valid):
                return guide_z, guide_mask

            src_rows, src_cols = np.where(valid)
            values = prev_z[src_rows, src_cols]
            xs, ys = rasterio.transform.xy(
                src.transform,
                src_rows,
                src_cols,
                offset="center",
            )
            xs = np.asarray(xs, dtype="float64")
            ys = np.asarray(ys, dtype="float64")

            if src.crs is not None and crs is not None and src.crs != crs:
                xs, ys = rasterio.warp.transform(
                    src.crs,
                    crs,
                    xs.tolist(),
                    ys.tolist(),
                )
                xs = np.asarray(xs, dtype="float64")
                ys = np.asarray(ys, dtype="float64")

        dst_rows, dst_cols = rasterio.transform.rowcol(
            transform,
            xs,
            ys,
        )
        dst_rows = np.asarray(dst_rows, dtype="int64")
        dst_cols = np.asarray(dst_cols, dtype="int64")

        inside = (
            (dst_rows >= 0)
            & (dst_rows < shape[0])
            & (dst_cols >= 0)
            & (dst_cols < shape[1])
        )

        dst_rows = dst_rows[inside]
        dst_cols = dst_cols[inside]
        values = values[inside]

        guide_z[dst_rows, dst_cols] = values
        guide_mask[dst_rows, dst_cols] = True

        return guide_z, guide_mask

    @staticmethod
    def _distance_to_core(core_mask, barrier_mask=None):
        """Return distance to same-topology core observations.

        Land and water are measured independently when topology is available so
        a dense land survey does not create a transition moat through adjacent
        water (or vice versa).
        """
        if barrier_mask is None:
            return scipy.ndimage.distance_transform_edt(~core_mask)

        dist = np.full(core_mask.shape, np.inf, dtype="float64")

        land_core = core_mask & barrier_mask
        if np.any(land_core):
            land_dist = scipy.ndimage.distance_transform_edt(~land_core)
            dist[barrier_mask] = land_dist[barrier_mask]

        water_core = core_mask & ~barrier_mask
        if np.any(water_core):
            water_dist = scipy.ndimage.distance_transform_edt(~water_core)
            dist[~barrier_mask] = water_dist[~barrier_mask]

        return dist

    @staticmethod
    def _support_mask(core_mask, barrier_mask=None, closing_dist=2):
        """Build a conservative footprint of coherent current-tier coverage."""

        if closing_dist <= 0:
            return core_mask.copy()

        def close(mask, dist):
            yy, xx = np.ogrid[
                -dist : dist + 1,
                -dist : dist + 1,
            ]
            structure = (xx * xx + yy * yy) <= dist * dist

            return scipy.ndimage.binary_closing(
                mask,
                structure=structure,
            )

        if barrier_mask is None:
            support = close(core_mask, closing_dist)
        else:
            support = np.zeros_like(core_mask, dtype=bool)

            land_core = core_mask & barrier_mask
            water_core = core_mask & ~barrier_mask

            if np.any(land_core):
                land_support = close(land_core, closing_dist)
                support[barrier_mask] = land_support[barrier_mask]

            if np.any(water_core):
                water_support = close(water_core, closing_dist)
                support[~barrier_mask] = water_support[~barrier_mask]

        support |= core_mask

        return support

    def _compose_tier_surface_raster(
        self,
        z,
        w,
        ndv,
        previous_surface,
        transform,
        crs,
        current_weight,
        current_blend_dist,
        barrier_mask=None,
    ):
        """Compose a tier using a resampled previous-tier background."""
        valid_mask = (z != ndv) & np.isfinite(z)
        core_mask = valid_mask & (w >= current_weight)

        if previous_surface is None:
            work_z = z.copy()
            interp_mask = ~valid_mask
            return work_z, valid_mask, core_mask, interp_mask

        bg_aligned = self._align_background(
            previous_surface,
            z.shape,
            transform,
            crs,
            ndv,
        )
        bg_valid = (bg_aligned != ndv) & np.isfinite(bg_aligned)

        if not np.any(core_mask):
            work_z = np.full(z.shape, ndv, dtype="float64")
            work_z[bg_valid] = bg_aligned[bg_valid]
            interp_mask = (work_z == ndv) | ~np.isfinite(work_z)
            return work_z, valid_mask, core_mask, interp_mask

        support_mask = self._support_mask(
            core_mask,
            barrier_mask=barrier_mask,
            closing_dist=2,
        )

        work_z = np.full(z.shape, ndv, dtype="float64")
        work_z[bg_valid] = bg_aligned[bg_valid]
        work_z[core_mask] = z[core_mask]

        if current_blend_dist > 0:
            dist_to_core = self._distance_to_core(core_mask, barrier_mask)
            blend_mask = (
                support_mask & ~core_mask & (dist_to_core < float(current_blend_dist))
            )
            work_z[blend_mask] = ndv

        interp_mask = (work_z == ndv) | ~np.isfinite(work_z)
        return work_z, valid_mask, core_mask, interp_mask

    def _compose_tier_surface_points(
        self,
        z,
        w,
        ndv,
        previous_surface,
        transform,
        crs,
        current_weight,
        current_blend_dist,
        barrier_mask=None,
    ):
        """Compose a tier from sparse previous-tier guide constraints."""
        valid_mask = (z != ndv) & np.isfinite(z)
        core_mask = valid_mask & (w >= current_weight)

        if previous_surface is None:
            work_z = z.copy()
            interp_mask = ~valid_mask
            return work_z, valid_mask, core_mask, interp_mask

        guide_z, guide_mask = self._previous_tier_guides(
            previous_surface,
            z.shape,
            transform,
            crs,
            ndv,
        )

        # Current-tier observations always supersede coarse guide constraints.
        guide_mask &= ~core_mask

        # Blending is optional in point mode.  Rather than carving a raster moat,
        # it suppresses coarse guide constraints near coherent high-resolution
        # support so the newly admitted observations have more freedom.
        if current_blend_dist > 0 and np.any(core_mask):
            support_mask = self._support_mask(
                core_mask,
                barrier_mask=barrier_mask,
                closing_dist=2,
            )
            dist_to_core = self._distance_to_core(core_mask, barrier_mask)
            suppress_guides = (
                guide_mask & support_mask & (dist_to_core < float(current_blend_dist))
            )
            guide_mask[suppress_guides] = False

        work_z = np.full(z.shape, ndv, dtype="float64")
        work_z[guide_mask] = guide_z[guide_mask]
        work_z[core_mask] = z[core_mask]

        # Every non-constraint cell is freshly resolved at the current tier.
        interp_mask = (work_z == ndv) | ~np.isfinite(work_z)
        return work_z, valid_mask, core_mask, interp_mask

    def _compose_tier_surface(
        self,
        z,
        w,
        ndv,
        previous_surface,
        transform,
        crs,
        current_weight,
        current_blend_dist,
        barrier_mask=None,
    ):
        """Compose one tier using the configured previous-tier strategy."""
        compose = (
            self._compose_tier_surface_points
            if self.previous_tier_mode == "points"
            else self._compose_tier_surface_raster
        )
        return compose(
            z,
            w,
            ndv,
            previous_surface,
            transform,
            crs,
            current_weight,
            current_blend_dist,
            barrier_mask=barrier_mask,
        )

    def _write_interpolation_input(
        self,
        step_stack,
        temp_in,
        work_z,
        core_mask,
        current_weight,
        ndv,
    ):
        """Write a temporary MultiStack whose metadata matches tier."""
        with rasterio.open(step_stack) as src:
            profile = src.profile.copy()
            profile.update(nodata=ndv)
            data = src.read()

        data[0] = work_z.astype(data[0].dtype, copy=False)

        # Keep auxiliary bands consistent when an interpolation hook inspects
        # more than band 1. Core observations retain their source metadata.
        # Synthetic previous-tier constraints (either resampled raster cells or
        # sparse point guides) act at the current tier weight.
        work_valid = (work_z != ndv) & np.isfinite(work_z)
        background_valid = work_valid & ~core_mask

        if data.shape[0] >= 2:
            data[1][~work_valid] = 0
            data[1][background_valid] = 1

        if data.shape[0] >= 3:
            data[2][~work_valid] = 0
            data[2][background_valid] = current_weight

        with rasterio.open(temp_in, "w", **profile) as dst:
            dst.write(data)

    def _interpolate_tier(
        self,
        step_stack,
        work_z,
        interp_mask,
        core_mask,
        current_weight,
        current_algo,
        ndv,
    ):
        """Fill only the unresolved portion of a composed tier surface."""
        if not np.any(interp_mask):
            logger.info("Tier is already resolved; nothing to interpolate")
            return work_z

        temp_in = step_stack.replace(".tif", f"_{current_weight}_in.tif")
        temp_out = step_stack.replace(".tif", f"_{current_weight}_out.tif")

        self._write_interpolation_input(
            step_stack,
            temp_in,
            work_z,
            core_mask,
            current_weight,
            ndv,
        )

        try:
            current_algo_hook = parse_hook_string(current_algo)
            interp_hook = self._get_interp_hook(current_algo_hook)

            if interp_hook.processing_mode == "chunk":
                success = interp_hook._process_file_fallback(
                    temp_in,
                    temp_out,
                    entry={},
                )
            else:
                success = interp_hook.process_raster(
                    temp_in,
                    temp_out,
                    entry={},
                )

            if success and os.path.exists(temp_out):
                with rasterio.open(temp_out) as filled_src:
                    filled_z = filled_src.read(1)

                filled_valid = (filled_z != ndv) & np.isfinite(filled_z)
                accept = interp_mask & filled_valid
                work_z[accept] = filled_z[accept]

            return work_z
        finally:
            if os.path.exists(temp_in):
                os.remove(temp_in)
            if os.path.exists(temp_out):
                os.remove(temp_out)

    def _decay_inland_caps(self, cap_grid, d2c_path, shape, transform, crs):
        """Fade water caps inland using the signed distance-to-coast grid."""
        if cap_grid is None or d2c_path is None:
            return cap_grid

        d2c_grid = np.zeros(shape, dtype="float32")
        with rasterio.open(d2c_path) as d2c_src:
            reproject(
                source=rasterio.band(d2c_src, 1),
                destination=d2c_grid,
                src_transform=d2c_src.transform,
                src_crs=d2c_src.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.bilinear,
                num_threads=1,
            )

        inland_water = (d2c_grid < 0) & np.isfinite(cap_grid)
        if not np.any(inland_water):
            return cap_grid

        dist_inland = np.abs(np.minimum(d2c_grid, 0))
        decay_weight = np.clip(
            1.0 - (dist_inland / self.inland_decay_dist),
            0.0,
            1.0,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            faded_caps = cap_grid[inland_water] / (decay_weight[inland_water] + 1e-6)

        faded_caps[decay_weight[inland_water] < 0.01] = np.nan
        cap_grid[inland_water] = faded_caps
        return cap_grid

    def _process_tier(
        self,
        step_stack,
        previous_surface,
        current_weight,
        current_algo,
        current_blend_dist,
        barrier_path,
        d2c_path=None,
    ):
        """Resolve one resolution/weight tier.

        The ordering is:

        1. classify current-tier hard observations;
        2. compose them with the previous coarser surface;
        3. interpolate only the unresolved transition/gaps;
        4. apply the topology-aware cap to generated values.

        Low-weight observations therefore influence finer tiers through the
        previous coarse surface instead of surviving as point-scale artifacts.
        """
        with rasterio.open(step_stack, "r+") as src:
            data = src.read()
            ndv = src.nodata if src.nodata is not None else -9999
            z_raw = data[0].astype("float64")
            w = data[2].astype("float64")

            cap_grid, barrier_mask = self._create_topological_grids(
                z_raw.shape,
                src.transform,
                barrier_path,
            )
            cap_grid = self._decay_inland_caps(
                cap_grid,
                d2c_path,
                z_raw.shape,
                src.transform,
                src.crs,
            )

            work_z, valid_mask, core_mask, interp_mask = self._compose_tier_surface(
                z_raw,
                w,
                ndv,
                previous_surface,
                src.transform,
                src.crs,
                current_weight,
                current_blend_dist,
                barrier_mask=barrier_mask,
            )

            # Positive current-tier observations can locally correct the OSM
            # water prior.  The small closing only consolidates tiny gaps in
            # coherent observed land; it does not create a standalone coastline.
            observed_land = self._observed_land_mask(
                z_raw,
                valid_mask,
                core_mask,
            )
            if cap_grid is not None:
                cap_grid[observed_land] = np.nan

            z = self._interpolate_tier(
                step_stack,
                work_z,
                interp_mask,
                core_mask,
                current_weight,
                current_algo,
                ndv,
            )

            # Hard observations at this tier are immutable.  Morphologically
            # consolidated positive observations also relax the cap in the tiny
            # interpolation gaps they enclose.
            if cap_grid is not None:
                cap_protected = core_mask | observed_land
                z = self._apply_topological_cap(
                    z,
                    cap_grid,
                    cap_protected,
                    ndv,
                )

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
                    outdir=os.path.join(self.cache_dir, "auto_barriers"),
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
