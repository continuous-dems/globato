#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.hooks.transforms.point_pixels
~~~~~~~~~~~~~

pointz class to bin point data and a Point Cloud Filtering Engine.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import numpy as np

from fetchez.hooks import FetchHook
from fetchez.utils import int_or, float_or, str2inc
from fetchez.spatial import Region

import rasterio
from rasterio.windows import Window

logger = logging.getLogger(__name__)


# Gridding Helper (From CUDEM)
class PointPixels:
    """Bins point cloud data into a grid coinciding with a desired region.
    Returns aggregated values (Z, Weights, Uncertainty) for each grid cell.

    Incoming data are numpy structured arrays (rec-arrays) of x, y, z, <w, u>.
    """

    def __init__(
        self,
        src_region=None,
        x_size=None,
        y_size=None,
        verbose=True,
        ppm=False,
        **kwargs,
    ):
        self.src_region = src_region
        self.x_size = int_or(x_size, 10)
        self.y_size = int_or(y_size, 10)
        self.verbose = verbose
        self.ppm = ppm
        self.dst_gt = None

    def init_region_from_points(self, points):
        """Initialize the source region based on point extents."""

        if self.src_region is None:
            self.src_region = Region.from_list(
                [
                    np.min(points["x"]),
                    np.max(points["x"]),
                    np.min(points["y"]),
                    np.max(points["y"]),
                ]
            )
        else:
            self.src_region = Region(*self.src_region)

        if not self.src_region.valid_p():
            self.src_region.buffer(2)

            if not self.src_region.valid_p():
                _epsilon = 0.00001
                # todo: expand in each direction by epsilon
                self.src_region.buffer(10)

        self.init_gt()

    def init_gt(self):
        """Initialize the GeoTransform based on region and size."""

        if self.src_region is not None:
            self.dst_gt = self.src_region.geo_transform_from_count(
                x_count=self.x_size, y_count=self.y_size
            )

    def __call__(self, points, weight=1.0, uncertainty=0.0, mode="mean"):
        """Process points into a gridded array."""

        out_arrays = {
            "z": None,
            "count": None,
            "weight": None,
            "uncertainty": None,
            "x": None,
            "y": None,
            "pixel_x": None,
            "pixel_y": None,
        }

        if points is None or len(points) == 0:
            return out_arrays, None, None

        if hasattr(points, "to_records"):
            points = points.to_records(index=False)

        if self.src_region is None:
            self.init_region_from_points(points)
        elif self.dst_gt is None:
            self.init_gt()

        weight = float_or(weight, 1)
        uncertainty = float_or(uncertainty, 0.0)
        mode = mode.lower()

        points_x = np.array(points["x"], dtype=np.float64)
        points_y = np.array(points["y"], dtype=np.float64)
        pixel_z = np.array(points["z"], dtype=np.float64)

        pixel_w = (
            np.array(points["w"], dtype=np.float64)
            if "w" in points.dtype.names
            else np.ones_like(pixel_z)
        )
        pixel_u = (
            np.array(points["u"], dtype=np.float64)
            if "u" in points.dtype.names
            else np.zeros_like(pixel_z)
        )

        pixel_w[np.isnan(pixel_w)] = 1
        pixel_u[np.isnan(pixel_u)] = 0

        pixel_x = np.floor((points_x - self.dst_gt[0]) / self.dst_gt[1]).astype(int)
        pixel_y = np.floor((points_y - self.dst_gt[3]) / self.dst_gt[5]).astype(int)

        valid_mask = (
            (pixel_x >= 0)
            & (pixel_x < self.x_size)
            & (pixel_y >= 0)
            & (pixel_y < self.y_size)
        )

        if not np.any(valid_mask):
            return out_arrays, None, None

        pixel_x, pixel_y = pixel_x[valid_mask], pixel_y[valid_mask]
        pixel_z, pixel_w, pixel_u = (
            pixel_z[valid_mask],
            pixel_w[valid_mask],
            pixel_u[valid_mask],
        )
        points_x, points_y = points_x[valid_mask], points_y[valid_mask]

        # Local Source Window Calculation
        min_px, max_px = int(np.min(pixel_x)), int(np.max(pixel_x))
        min_py, max_py = int(np.min(pixel_y)), int(np.max(pixel_y))

        rows = max_py - min_py + 1
        cols = max_px - min_px + 1
        this_srcwin = (min_px, min_py, cols, rows)
        grid_shape = (rows, cols)
        max_idx = rows * cols

        local_px = pixel_x - min_px
        local_py = pixel_y - min_py

        # ==========================================
        # Mode == "sums" (Using bincount)
        # ==========================================
        if mode == "sums":
            # Flatten 2D coordinates to 1D index
            flat_indices = local_py * cols + local_px
            ww = pixel_w * weight

            # Direct Accumulation
            grid_count = np.bincount(flat_indices, minlength=max_idx).reshape(
                grid_shape
            )
            grid_w = np.bincount(flat_indices, weights=ww, minlength=max_idx).reshape(
                grid_shape
            )
            grid_z = np.bincount(
                flat_indices, weights=pixel_z * ww, minlength=max_idx
            ).reshape(grid_shape)
            grid_x = np.bincount(
                flat_indices, weights=points_x * ww, minlength=max_idx
            ).reshape(grid_shape)
            grid_y = np.bincount(
                flat_indices, weights=points_y * ww, minlength=max_idx
            ).reshape(grid_shape)

            # Uncertainty via mathematical standard deviation: sqrt(E[x^2] - E[x]^2)
            grid_z_sum = np.bincount(
                flat_indices, weights=pixel_z, minlength=max_idx
            ).reshape(grid_shape)
            grid_z_sq = np.bincount(
                flat_indices, weights=(pixel_z**2), minlength=max_idx
            ).reshape(grid_shape)

            with np.errstate(invalid="ignore", divide="ignore"):
                mean_z = grid_z_sum / grid_count
                var_z = (grid_z_sq / grid_count) - (mean_z**2)
                var_z = np.clip(var_z, 0, None)
                std_z = np.sqrt(var_z)

            grid_u_sq = np.bincount(
                flat_indices, weights=(pixel_u**2), minlength=max_idx
            ).reshape(grid_shape)
            grid_u = np.sqrt(grid_u_sq + (uncertainty**2) + (std_z**2))

            # Mask out empty pixels
            empty_mask = grid_count == 0
            grid_z[empty_mask] = np.nan
            grid_x[empty_mask] = np.nan
            grid_y[empty_mask] = np.nan
            grid_u[empty_mask] = 0.0

            out_arrays["z"] = grid_z
            out_arrays["x"] = grid_x
            out_arrays["y"] = grid_y
            out_arrays["count"] = grid_count
            out_arrays["weight"] = grid_w
            out_arrays["uncertainty"] = grid_u

            return out_arrays, this_srcwin, self.dst_gt

        # ==========================================
        # Mode != "sums" (Legacy fallback)
        # ==========================================
        pixel_xy = np.vstack((local_py, local_px)).T

        unq, unq_idx, unq_inv, unq_cnt = np.unique(
            pixel_xy, axis=0, return_inverse=True, return_index=True, return_counts=True
        )

        # Initial values

        zz = pixel_z[unq_idx]
        ww = pixel_w[unq_idx]
        xx = points_x[unq_idx]
        yy = points_y[unq_idx]

        uu = pixel_u[unq_idx]

        # --- Handle Duplicates ---
        cnt_msk = unq_cnt > 1

        if np.any(cnt_msk):
            ## Sort indices to group by pixel
            srt_idx = np.argsort(unq_inv)
            split_indices = np.cumsum(unq_cnt)[:-1]
            grouped_indices = np.split(srt_idx, split_indices)

            # Filter groups with duplicates
            dup_indices = [grouped_indices[i] for i in np.flatnonzero(cnt_msk)]
            # dup_stds = []
            dup_stds = np.zeros(len(dup_indices))

            if mode == "min":
                zz[cnt_msk] = [np.min(pixel_z[idx]) for idx in dup_indices]
                xx[cnt_msk] = [np.min(points_x[idx]) for idx in dup_indices]
                yy[cnt_msk] = [np.min(points_y[idx]) for idx in dup_indices]
                dup_stds = np.zeros(len(dup_indices))

            elif mode == "max":
                zz[cnt_msk] = [np.max(pixel_z[idx]) for idx in dup_indices]
                xx[cnt_msk] = [np.max(points_x[idx]) for idx in dup_indices]
                yy[cnt_msk] = [np.max(points_y[idx]) for idx in dup_indices]
                dup_stds = np.zeros(len(dup_indices))

            elif mode == "mean":
                zz[cnt_msk] = [np.mean(pixel_z[idx]) for idx in dup_indices]
                xx[cnt_msk] = [np.mean(points_x[idx]) for idx in dup_indices]
                yy[cnt_msk] = [np.mean(points_y[idx]) for idx in dup_indices]
                dup_stds = [np.std(pixel_z[idx]) for idx in dup_indices]

            elif mode == "median":
                zz[cnt_msk] = [np.median(pixel_z[idx]) for idx in dup_indices]
                xx[cnt_msk] = [np.mean(points_x[idx]) for idx in dup_indices]
                yy[cnt_msk] = [np.mean(points_y[idx]) for idx in dup_indices]
                dup_stds = [np.std(pixel_z[idx]) for idx in dup_indices]

            elif mode == "std":
                zz[cnt_msk] = [np.std(pixel_z[idx]) for idx in dup_indices]
                xx[cnt_msk] = [np.mean(points_x[idx]) for idx in dup_indices]
                yy[cnt_msk] = [np.mean(points_y[idx]) for idx in dup_indices]
                dup_stds = np.zeros(len(dup_indices))

            elif mode == "var":
                zz[cnt_msk] = [np.var(pixel_z[idx]) for idx in dup_indices]
                xx[cnt_msk] = [np.mean(points_x[idx]) for idx in dup_indices]
                yy[cnt_msk] = [np.mean(points_y[idx]) for idx in dup_indices]
                dup_stds = np.zeros(len(dup_indices))

            # uncertainty
            uu[cnt_msk] = np.sqrt(np.power(uu[cnt_msk], 2) + np.power(dup_stds, 2))

        # --- Fill Output Grids ---
        grid_shape = (this_srcwin[3], this_srcwin[2])  # rows, cols

        # -- Safety First ---
        if grid_shape[0] <= 0 or grid_shape[1] <= 0:
            for key in out_arrays:
                out_arrays[key] = None
            return out_arrays, None, None

        def fill_grid(values, fill_val=np.nan):
            grid = np.full(grid_shape, fill_val)
            grid[unq[:, 0], unq[:, 1]] = values
            return grid

        out_arrays["z"] = fill_grid(zz)
        out_arrays["x"] = fill_grid(xx)
        out_arrays["y"] = fill_grid(yy)
        out_arrays["count"] = fill_grid(unq_cnt, fill_val=0)

        # Uncertainty
        out_arrays["uncertainty"] = fill_grid(
            np.sqrt(uu**2 + (uncertainty) ** 2), fill_val=0.0
        )

        # Weights
        out_arrays["weight"] = np.ones(grid_shape)
        out_arrays["weight"][:] = weight
        out_arrays["weight"][unq[:, 0], unq[:, 1]] *= ww * unq_cnt

        # Helper coords for calling class to map back
        out_arrays["pixel_x"] = local_px
        out_arrays["pixel_y"] = local_py

        return out_arrays, this_srcwin, self.dst_gt


class PixelsToPoints(FetchHook):
    """Converts an in-memory raster_stream back into an xyz_recarray point stream."""

    name = "pixels2points"
    meta_stage = "stream"
    meta_category = "stream-transform"
    meta_aliases = ["pixels_to_points"]
    meta_desc = "Converts a raster-stream back into an xyz-recarray point stream."

    def _raster_to_xyz(self, raster_stream):
        _profile = next(raster_stream)
        for window, buff_win, data, ndv, transform in raster_stream:
            bands, rows, cols = data.shape

            z_raw = data[0].flatten()
            if bands >= 7:
                count = data[1].flatten()
                weight = data[2].flatten()

                valid = (count > 0) & (~np.isnan(z_raw))

                z = z_raw[valid] / weight[valid]
                x = data[5].flatten()[valid] / weight[valid]
                y = data[6].flatten()[valid] / weight[valid]

                arrays = [x, y, z]
                names = ["x", "y", "z"]

                arrays.append(data[2].flatten()[valid])
                names.append("w")
                arrays.append(data[3].flatten()[valid])
                names.append("u")

            else:
                valid = ~np.isnan(z_raw)
                z = z_raw[valid]

                col_indices, row_indices = np.meshgrid(np.arange(cols), np.arange(rows))
                global_cols = col_indices + window.col_off
                global_rows = row_indices + window.row_off
                xs, ys = rasterio.transform.xy(
                    transform, global_rows, global_cols, offset="center"
                )

                x = np.array(xs).flatten()[valid]
                y = np.array(ys).flatten()[valid]

                arrays = [x, y, z]
                names = ["x", "y", "z"]

            try:
                chunk = np.rec.fromarrays(arrays, names=names)
                if chunk.size > 0:
                    yield chunk
            except Exception as e:
                logger.error(f"pixels2points crash: {e}")

    def run(self, entries):
        for mod, entry in entries:
            if self.is_raster_stream(entry):
                stream = entry["stream"]
                entry["stream"] = self._raster_to_xyz(stream)
                entry["stream_type"] = "point-stream"

        return entries


# Base Stream Transformer
class Point2PixelStream(FetchHook):
    """Base class for streaming point filters."""

    name = "points2pixels"
    meta_stage = "stream"
    meta_category = "stream-transform"
    meta_aliases = ["point2pixel", "points_to_pixel"]
    meta_desc = "Converts an xyz-recarray point-stream into a raster-stream."

    def __init__(self, x_inc=None, y_inc=None, want_sums=True, **kwargs):
        super().__init__(**kwargs)
        self.x_inc = float_or(str2inc(x_inc))
        self.y_inc = float_or(str2inc(y_inc))
        self.want_sums = want_sums

    def process_chunk(self, chunk, region=None):
        """Override this. Return filtered chunk (recarray) or None."""

        if region:
            xcount, ycount, _ = region.geo_transform(
                x_inc=self.x_inc, y_inc=self.y_inc, node="pixel"
            )

            point_array = PointPixels(
                src_region=region, x_size=xcount, y_size=ycount, verbose=True
            )
            arrs, srcwin, gt = point_array(
                chunk,
                weight=1,
                uncertainty=0,
                mode="sums" if self.want_sums else "mean",
            )

            return arrs, srcwin, gt

    def _stream_wrapper(self, input_stream, entry=None, region=None):
        count = 0

        if region:
            xcount, ycount, gt = region.geo_transform(
                x_inc=self.x_inc, y_inc=self.y_inc, node="pixel"
            )
            transform = rasterio.transform.from_origin(gt[0], gt[3], gt[1], abs(gt[5]))

            # If we want sums, we output a 7-band stack. Otherwise, just 1 band (Z).
            band_count = 7 if self.want_sums else 1

            profile = {
                "driver": "GTiff",
                "dtype": "float32",
                "nodata": -9999,
                "width": xcount,
                "height": ycount,
                "count": band_count,
                "crs": entry.get("src_srs", "EPSG:4326"),
                "transform": transform,
            }
            yield profile

            for chunk in input_stream:
                count += chunk.size
                arrs, srcwin, chunk_gt = self.process_chunk(chunk, region=region)
                if arrs is None or arrs.get("z") is None:
                    continue

                col_off, row_off, width, height = srcwin
                window = Window(col_off, row_off, width, height)
                chunk_transform = rasterio.transform.from_origin(
                    chunk_gt[0], chunk_gt[3], chunk_gt[1], abs(chunk_gt[5])
                )

                # Stack the dictionary of arrays into a 3D numpy array (Bands, Rows, Cols)
                if self.want_sums:
                    data = np.stack(
                        [
                            arrs["z"],
                            arrs["count"],
                            arrs["weight"],
                            arrs["uncertainty"],
                            arrs.get("src_uncertainty", np.zeros_like(arrs["z"])),
                            arrs["x"],
                            arrs["y"],
                        ]
                    ).astype(np.float64)
                else:
                    data = arrs["z"].astype(np.float64)
                    # Add a band dimension so it's (1, Rows, Cols)
                    data = data[np.newaxis, ...]

                yield window, window, data, -9999, chunk_transform

        logger.info(f"Parsed {count} data records from {entry['dst_fn']}")

    def run(self, entries):
        for mod, entry in entries:
            if self.is_point_stream(entry):
                stream = entry["stream"]
                entry["stream"] = self._stream_wrapper(
                    stream, entry=entry, region=getattr(mod, "region", None)
                )
                entry["stream_type"] = "raster-stream"

        return entries
