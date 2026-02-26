#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.rasters.blend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Blends high-weight data into low-weight background using spatial buffering,
interpolation, and slope-gated randomization.

Based on cudem.grits.blend

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import numpy as np
import scipy.ndimage
import scipy.interpolate
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window

from .base import RasterHook

logger = logging.getLogger(__name__)

class MultiStackBlend(RasterHook):
    """Blends multi-stack raster data based on weight thresholds.
    Creates a smooth transition (buffer) between high-weight (Foreground)
    and low-weight (Background) areas.

    Args:
        weight_threshold (float): Value separating Foreground from Background.
        blend_dist (int): Width of the blending buffer in pixels.
        slope_scale (float): 0-1. Normalizes slope to gate random noise in the seam.
        random_scale (float): 0-1. Density of random points to keep in the seam (dithering).
        algo (str): Interpolation method ('linear', 'cubic', 'nearest').
    """

    name = "ms_blend"
    default_suffix = "_blended"
    category = "multi-stack"

    def __init__(self, weight_threshold=1.0, blend_dist=5, slope_scale=0.5,
                 random_scale=0.025, algo='linear', **kwargs):

        # Ensure we have enough buffer to cover the blend distance + interpolation context
        buffer_req = int(blend_dist) * 2
        super().__init__(buffer=buffer_req, **kwargs)

        self.weight_threshold = float(weight_threshold)
        self.blend_dist = int(blend_dist)
        self.slope_scale = float(slope_scale)
        self.random_scale = float(random_scale)
        self.algo = algo

    def _get_slope_norm(self, z_arr, scale_arr):
        """Compute normalized slope (0-1) for gating."""

        if np.all(np.isnan(z_arr)): return None

        # Calculate gradients
        gy, gx = np.gradient(z_arr)
        slope = np.sqrt(gx**2 + gy**2)

        # Normalize relative to the blend area
        vals = slope[scale_arr] # Only look at slope in the blend zone
        vals = vals[np.isfinite(vals)]

        if vals.size == 0: return None

        m = np.nanmax(np.abs(vals))
        if m == 0: return None

        slope_norm = np.abs(slope) / m
        slope_norm[np.isnan(slope_norm)] = 0.0
        return slope_norm

    def process_raster(self, src_path, dst_path, entry):
        with rasterio.open(src_path) as src:
            profile = src.profile.copy()
            is_stack = (src.count >= 3)

            with rasterio.open(dst_path, 'w', **profile) as dst:
                for window, buff_win in self.yield_buffered_windows(src, buffer_size=self.buffer):

                    z = src.read(1, window=buff_win)
                    w = src.read(3, window=buff_win)
                    ndv = src.nodata
                    chunk_transform = rasterio.windows.transform(buff_win, src.transform)

                    valid_mask = (z != ndv) & (~np.isnan(z))
                    if not np.any(valid_mask):
                        return z

                    fg_mask = valid_mask & (w >= self.weight_threshold)

                    bg_mask = valid_mask & (~fg_mask)

                    if not np.any(fg_mask):
                        return z # No foreground to blend from

                    struct = scipy.ndimage.generate_binary_structure(2, 2)
                    fg_closed = scipy.ndimage.binary_closing(fg_mask, structure=struct)
                    blend_mask = scipy.ndimage.binary_dilation(fg_closed, iterations=self.blend_dist)
                    transition_zone = blend_mask & (~fg_closed) & valid_mask
                    if not np.any(transition_zone):
                        return z

                    if self.random_scale > 0:
                        rand_arr = np.random.rand(*z.shape)
                        rand_keep = rand_arr < self.random_scale

                        if self.slope_scale > 0:
                            slope_norm = self._get_slope_norm(z, transition_zone)
                            if slope_norm is not None:
                                flat_areas = slope_norm < self.slope_scale
                                rand_keep[flat_areas] = False

                    anchors_mask = fg_mask | (bg_mask & ~transition_zone)
                    if self.random_scale > 0:
                        anchors_mask = anchors_mask | (transition_zone & rand_keep)

                    rows, cols = np.indices(z.shape)
                    anchor_pts = np.array([rows[anchors_mask], cols[anchors_mask]]).T
                    anchor_vals = z[anchors_mask]
                    target_pts = np.array([rows[transition_zone], cols[transition_zone]]).T
                    if len(anchor_pts) < 4 or len(target_pts) == 0:
                        return z

                    try:
                        interp_vals = scipy.interpolate.griddata(
                            anchor_pts,
                            anchor_vals,
                            target_pts,
                            method=self.algo
                        )
                        dist = scipy.ndimage.distance_transform_cdt(~fg_mask, metric='taxicab')
                        d_vals = dist[transition_zone]
                        if d_vals.size > 0:
                            d_min, d_max = d_vals.min(), d_vals.max()
                            if d_max > d_min:
                                weights = (d_vals - d_min) / (d_max - d_min)
                            else:
                                weights = np.zeros_like(d_vals)
                        else:
                            weights = np.zeros(len(interp_vals))

                        original_vals = z[transition_zone]
                        original_vals[np.isnan(original_vals)] = interp_vals[np.isnan(original_vals)]
                        blended_vals = (1 - weights) * interp_vals + (weights) * original_vals

                        z[transition_zone] = blended_vals

                    except Exception as e:
                        logger.warning(f"Blend failed in chunk: {e}")
                        pass

                    # Crop buffer
                    y_off = window.row_off - buff_win.row_off
                    x_off = window.col_off - buff_win.col_off

                    final_chunk = z[y_off : y_off + window.height,
                                    x_off : x_off + window.width]

                    dst.write(final_chunk, 1, window=window)

        return True


class RasterBlend(RasterHook):
    """Blends raster data based on weight thresholds.
    Creates a smooth transition (buffer) between high-weight (Foreground)
    and low-weight (Background) areas.

    Args:
        weight_threshold (float): Value separating Foreground from Background.
        blend_dist (int): Width of the blending buffer in pixels.
        core_dist (int): Width of the core blending zone in pixels.
        slope_scale (float): 0-1. Normalizes slope to gate random noise in the seam.
        random_scale (float): 0-1. Density of random points to keep in the seam (dithering).
        algo (str): Interpolation method ('linear', 'cubic', 'nearest').
    """

    name = "raster_blend"
    default_suffix = "_blended"

    def __init__(
            self,
            aux_path=None,
            blend_dist=5,
            core_dist=1,
            slope_scale=0.5,
            random_scale=0.025,
            algo="linear",
            **kwargs
    ):

        buffer_req = int(blend_dist) * 2
        super().__init__(buffer=buffer_req, **kwargs)

        self.aux_path = aux_path
        self.core_dist = int(core_dist)
        self.blend_dist = int(blend_dist)
        self.slope_scale = float(slope_scale)
        self.random_scale = float(random_scale)
        self.algo = algo


    def _get_slope_norm(self, z_arr, scale_arr):
        """Compute normalized slope (0-1) for gating."""

        if np.all(np.isnan(z_arr)): return None

        # Calculate gradients
        gy, gx = np.gradient(z_arr)
        slope = np.sqrt(gx**2 + gy**2)

        # Normalize relative to the blend area
        vals = slope[scale_arr] # Only look at slope in the blend zone
        vals = vals[np.isfinite(vals)]

        if vals.size == 0: return None

        m = np.nanmax(np.abs(vals))
        if m == 0: return None

        slope_norm = np.abs(slope) / m
        slope_norm[np.isnan(slope_norm)] = 0.0
        return slope_norm

    def binary_closed_dilation(self, arr, iterations=1, closing_iterations=1):
        closed_and_dilated_arr = arr.copy()
        struct_ = scipy.ndimage.generate_binary_structure(2, 2)
        for i in range(closing_iterations):
            closed_and_dilated_arr = scipy.ndimage.binary_dilation(
                closed_and_dilated_arr, iterations=1, structure=struct_
            )
            closed_and_dilated_arr = scipy.ndimage.binary_erosion(
                closed_and_dilated_arr, iterations=1, border_value=1, structure=struct_
            )

        closed_and_dilated_arr = scipy.ndimage.binary_dilation(
            closed_and_dilated_arr, iterations=iterations, structure=struct_
        )

        return closed_and_dilated_arr

    def binary_reversion(self, arr, iterations, closing_iterations):
        reversion_arr = arr.copy()
        closing_iterations += iterations

        struct_ = scipy.ndimage.generate_binary_structure(2, 2)
        reversion_arr = scipy.ndimage.binary_dilation(
            reversion_arr, iterations=closing_iterations, structure=struct_
        )
        erosion_iterations = max(closing_iterations-iterations, 0)
        if erosion_iterations > 0:
            reversion_arr = scipy.ndimage.binary_erosion(
                reversion_arr, iterations=erosion_iterations, border_value=1, structure=struct_
            )

        return reversion_arr

    def process_raster(self, src_path, dst_path, entry):
        if not self.aux_path or not os.path.exists(self.aux_path):
            logger.error(f"[Blend] Aux path not found: {self.aux_path}")
            return False

        with rasterio.open(src_path) as src:
            profile = src.profile.copy()
            with rasterio.open(self.aux_path) as aux:
                with rasterio.open(dst_path, 'w', **profile) as dst:
                    for window, buff_win in self.yield_buffered_windows(src, buffer_size=self.blend_dist):
                        # we probably don't want windowed for this...maybe buffer is sufficient?
                        src_data = src.read(1, window=window)
                        src_ndv = src.nodata

                        aux_data = aux.read(1, window=window)
                        aux_ndv = aux.nodata

                        src_arr = src_data.astype(np.float32)
                        src_arr[src_data == src_ndv] = np.nan

                        src_mask = np.isfinite(src_arr)

                        aux_arr = aux_data.astype(np.float32)
                        if aux_ndv is not None:
                            aux_arr[aux_data == aux_ndv] = np.nan

                        aux_mask = ~np.isnan(a_arr)

                        if self.core_dist is not None:
                            aux_core_mask = self.binary_closed_dilation(
                                aux_mask, iterations=self.core_dist
                            )
                        else:
                            aux_core_mask = np.zeros_like(aux_mask, dtype=bool)

                        if self.blend_dist is not None:
                            aux_mask = self.binary_closed_dilation(
                                aux_mask, iterations=self.blend_dist
                            )

                        buffer_mask = aux_mask & src_mask
                        aux_core_mask = aux_core_mask & src_mask   # core seam: no randomization
                        slope_norm = None

                        # combined_arr now has a nan buffer between aux and src data
                        aux_arr[~aux_mask] = src_arr[~aux_mask]

                        # initial distance transform
                        dt = scipy.ndimage.distance_transform_cdt(
                            a_mask, metric="taxicab"
                        )

                        random_arr = np.random.rand(*aux_arr.shape)
                        random_mask = random_arr < self.random_scale  # base density of random picks

                        # Never randomize in core seam
                        random_mask[aux_core_mask] = False

                        # if slope gating is enabled, only allow flips in steeper areas
                        # slope-aware random src mask in the OUTER buffer only
                        if self.slope_scale > 0:
                            outer_mask = buffer_mask & (~aux_core_mask)
                            slope_norm = self._get_slope_norm(src_arr, outer_mask)

                            if slope_norm is not None:
                                low_slope = slope_norm < self.slope_scale
                                random_mask[outer_mask][low_slope] = False
                                slope_norm = outer_mask = low_slope = None

                        # Only apply randomization where both src + combined_mask are valid (buffer region)
                        valid_random = random_mask & buffer_mask

                        # Inject source values in the outer buffer; this helps preserve shape.
                        aux_arr[valid_random] = src_arr[valid_random]
                        aux_mask[valid_random] = True

                        # Recompute distance transform after randomization.
                        dt = scipy.ndimage.distance_transform_cdt(
                            aux_mask, metric="taxicab"
                        )

                        # extract and normalize dt in the seam/buffer region
                        dt_vals = dt[buffer_mask].astype(np.float32)
                        if dt_vals.size > 0:
                            dt_min = np.min(dt_vals)
                            dt_max = np.max(dt_vals)
                            if dt_max > dt_min:
                                dt_norm = (dt_vals - dt_min) / (dt_max - dt_min)
                            else:
                                dt_norm = np.zeros_like(dt_vals, dtype=np.float32)
                        else:
                            dt_norm = np.zeros(0, dtype=np.float32)


                        interp_arr = np.full(aux_arr.shape, np.nan)
                        point_indices = np.nonzero(~np.isnan(aux_arr))
                        grid_y, grid_x = np.mgrid[0:data.shape[0], 0:data.shape[1]]

                        # Interpolate
                        interp_arr = scipy.interpolate.griddata(
                            np.transpose(point_indices),
                            point_values,
                            (grid_y, grid_x),
                            method=self.algo
                        )


                        interp_data_in_buffer = interp_arr[buffer_mask]
                        src_data_in_buffer = src_arr[buffer_mask]
                        interp_arr = None

                        buffer_diffs = interp_data_in_buffer - src_data_in_buffer
                        buffer_diffs[np.isnan(buffer_diffs)] = 0

                        # Apply the normalized results to the differences and set and write out the results
                        aux_arr[buffer_mask] = src_arr[buffer_mask] + (buffer_diffs * dt_norm)
                        aux_arr[~src_mask] = np.nan
                        aux_arr[np.isnan(aux_arr)] = ndv

                        dst.write(aux_arr, 1, window=window)
        return True
