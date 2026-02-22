#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.transforms.point_pixels
~~~~~~~~~~~~~

pointz class to bin point data and a Point Cloud Filtering Engine.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import sys
import logging
import numpy as np
import warnings
from fetchez.hooks import FetchHook
from fetchez.utils import int_or, float_or, str2bool

import rasterio
from scipy.spatial import cKDTree

from transformez.spatial import TransRegion as Region

logger = logging.getLogger(__name__)


# Gridding Helper (From CUDEM)
class PointPixels:
    """Bins point cloud data into a grid coinciding with a desired region.
    Returns aggregated values (Z, Weights, Uncertainty) for each grid cell.

    Incoming data are numpy structured arrays (rec-arrays) of x, y, z, <w, u>.
    """

    def __init__(self, src_region=None, x_size=None, y_size=None,
                 verbose=True, ppm=False, **kwargs):
        self.src_region = src_region
        self.x_size = int_or(x_size, 10)
        self.y_size = int_or(y_size, 10)
        self.verbose = verbose
        self.ppm = ppm
        self.dst_gt = None


    def init_region_from_points(self, points):
        """Initialize the source region based on point extents."""

        if self.src_region is None:
            self.src_region = Region.from_list([
                np.min(points['x']), np.max(points['x']),
                np.min(points['y']), np.max(points['y'])
            ])

        if not self.src_region.valid_p():
            self.src_region.buffer(2)

            if not self.src_region.valid_p():
                epsilon = 0.00001
                # todo: expand in each direction by epsilon
                self.src_region.buffer(10)

        self.init_gt()


    def init_gt(self):
        """Initialize the GeoTransform based on region and size."""

        if self.src_region is not None:
            self.dst_gt = self.src_region.geo_transform_from_count(
                x_count=self.x_size, y_count=self.y_size
            )


    def __call__(self, points, weight=1.0, uncertainty=0.0, mode='mean'):
        """Process points into a gridded array.

        Args:
            points (np.recarray): Input data containing 'x', 'y', 'z'.
            weight (float): Global weight multiplier.
            uncertainty (float): Global uncertainty value.
            mode (str): Aggregation mode.
                        Options: 'mean', 'min', 'max', 'median', 'std', 'var', 'sums'.
        """

        # mrl: removed 'mask': None
        out_arrays = {
            'z': None, 'count': None, 'weight': None, 'uncertainty': None,
            'x': None, 'y': None, 'pixel_x': None, 'pixel_y': None
        }

        if points is None or len(points) == 0:
            return out_arrays, None, None

        # If input points are pandas dataframe, tranform it to recarray
        if hasattr(points, 'to_records'):
            points = points.to_records(index=False)

        # Ensure region and geotransform are set
        if self.src_region is None:
            self.init_region_from_points(points)
        elif self.dst_gt is None:
            self.init_gt()

        weight = float_or(weight, 1)
        uncertainty = float_or(uncertainty, 0.0)
        mode = mode.lower()

        points_x = np.array(points['x'])
        points_y = np.array(points['y'])
        pixel_z = np.array(points['z'])

        # This still gives a warning sometimes:
        #   RuntimeWarning: invalid value encountered in divide
        #   pixel_x = np.floor((points_x - self.dst_gt[0]) / self.dst_gt[1]).astype(int)
        #   RuntimeWarning: invalid value encountered in cast
        # TODO: Figure this out and fix.
        pixel_w = np.array(points['w']) if 'w' in points.dtype.names else np.ones_like(pixel_z)
        pixel_u = np.array(points['u']) if 'u' in points.dtype.names else np.zeros_like(pixel_z)

        pixel_w[np.isnan(pixel_w)] = 1
        pixel_u[np.isnan(pixel_u)] = 0

        # Convert to pixel coordinates
        # dst_gt: [origin_x, pixel_width, 0, origin_y, 0, pixel_height]
        #print(self.dst_gt)
        pixel_x = np.floor((points_x - self.dst_gt[0]) / self.dst_gt[1]).astype(int)
        pixel_y = np.floor((points_y - self.dst_gt[3]) / self.dst_gt[5]).astype(int)

        # Filter pixels outside window
        valid_mask = (
            (pixel_x >= 0) & (pixel_x < self.x_size) &
            (pixel_y >= 0) & (pixel_y < self.y_size)
        )

        if not np.any(valid_mask):
            return out_arrays, None, None

        # Apply mask
        pixel_x = pixel_x[valid_mask]
        pixel_y = pixel_y[valid_mask]
        pixel_z = pixel_z[valid_mask]
        pixel_w = pixel_w[valid_mask]
        pixel_u = pixel_u[valid_mask]
        points_x = points_x[valid_mask]
        points_y = points_y[valid_mask]

        if len(pixel_x) == 0:
            return out_arrays, None, None

        # Local Source Window Calculation
        min_px, max_px = int(np.min(pixel_x)), int(np.max(pixel_x))
        min_py, max_py = int(np.min(pixel_y)), int(np.max(pixel_y))

        this_srcwin = (min_px, min_py, max_px - min_px + 1, max_py - min_py + 1)

        # Shift to local coordinates
        local_px = pixel_x - min_px
        local_py = pixel_y - min_py

        # Unique pixel identification (row-major: y, x)
        pixel_xy = np.vstack((local_py, local_px)).T

        unq, unq_idx, unq_inv, unq_cnt = np.unique(
            pixel_xy, axis=0, return_inverse=True,
            return_index=True, return_counts=True
        )

        # Initial values
        if mode == 'sums':
            ww = pixel_w[unq_idx] * weight
            zz = pixel_z[unq_idx] * ww
            xx = points_x[unq_idx] * ww
            yy = points_y[unq_idx] * ww
        else:
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
            #dup_stds = []
            dup_stds = np.zeros(len(dup_indices))

            if mode == 'min':
                zz[cnt_msk] = [np.min(pixel_z[idx]) for idx in dup_indices]
                xx[cnt_msk] = [np.min(points_x[idx]) for idx in dup_indices]
                yy[cnt_msk] = [np.min(points_y[idx]) for idx in dup_indices]
                dup_stds = np.zeros(len(dup_indices))

            elif mode == 'max':
                zz[cnt_msk] = [np.max(pixel_z[idx]) for idx in dup_indices]
                xx[cnt_msk] = [np.max(points_x[idx]) for idx in dup_indices]
                yy[cnt_msk] = [np.max(points_y[idx]) for idx in dup_indices]
                dup_stds = np.zeros(len(dup_indices))

            elif mode == 'mean':
                zz[cnt_msk] = [np.mean(pixel_z[idx]) for idx in dup_indices]
                xx[cnt_msk] = [np.mean(points_x[idx]) for idx in dup_indices]
                yy[cnt_msk] = [np.mean(points_y[idx]) for idx in dup_indices]
                dup_stds = [np.std(pixel_z[idx]) for idx in dup_indices]

            elif mode == 'median':
                zz[cnt_msk] = [np.median(pixel_z[idx]) for idx in dup_indices]
                xx[cnt_msk] = [np.mean(points_x[idx]) for idx in dup_indices]
                yy[cnt_msk] = [np.mean(points_y[idx]) for idx in dup_indices]
                dup_stds = [np.std(pixel_z[idx]) for idx in dup_indices]

            elif mode == 'std':
                zz[cnt_msk] = [np.std(pixel_z[idx]) for idx in dup_indices]
                xx[cnt_msk] = [np.mean(points_x[idx]) for idx in dup_indices]
                yy[cnt_msk] = [np.mean(points_y[idx]) for idx in dup_indices]
                dup_stds = np.zeros(len(dup_indices))

            elif mode == 'var':
                zz[cnt_msk] = [np.var(pixel_z[idx]) for idx in dup_indices]
                xx[cnt_msk] = [np.mean(points_x[idx]) for idx in dup_indices]
                yy[cnt_msk] = [np.mean(points_y[idx]) for idx in dup_indices]
                dup_stds = np.zeros(len(dup_indices))

            elif mode == 'sums':
                zz[cnt_msk] = [np.sum(pixel_z[idx] * pixel_w[idx] * weight) for idx in dup_indices]
                xx[cnt_msk] = [np.sum(points_x[idx] * pixel_w[idx] * weight) for idx in dup_indices]
                yy[cnt_msk] = [np.sum(points_y[idx] * pixel_w[idx] * weight) for idx in dup_indices]
                ww[cnt_msk] = [np.sum(pixel_w[idx] * weight) for idx in dup_indices]
                dup_stds = [np.std(pixel_z[idx]) for idx in dup_indices]

            # uncertainty
            uu[cnt_msk] = np.sqrt(np.power(uu[cnt_msk], 2) + np.power(dup_stds, 2))

        # --- Fill Output Grids ---
        grid_shape = (this_srcwin[3], this_srcwin[2]) # rows, cols

        def fill_grid(values, fill_val=np.nan):
            grid = np.full(grid_shape, fill_val)
            grid[unq[:, 0], unq[:, 1]] = values
            return grid

        out_arrays['z'] = fill_grid(zz)
        out_arrays['x'] = fill_grid(xx)
        out_arrays['y'] = fill_grid(yy)
        out_arrays['count'] = fill_grid(unq_cnt, fill_val=0)

        # Uncertainty
        out_arrays['uncertainty'] = fill_grid(
            np.sqrt(uu**2 + (uncertainty)**2), fill_val=0.0
        )

        # Weights
        out_arrays['weight'] = np.ones(grid_shape)
        if mode == 'sums':
            out_arrays['weight'][unq[:, 0], unq[:, 1]] = ww
        else:
            out_arrays['weight'][:] = weight
            out_arrays['weight'][unq[:, 0], unq[:, 1]] *= (ww * unq_cnt)

        # Helper coords for calling class to map back
        out_arrays['pixel_x'] = local_px
        out_arrays['pixel_y'] = local_py

        return out_arrays, this_srcwin, self.dst_gt


# Base Stream Transformer
class Point2PixelStream(FetchHook):
    """Base class for streaming point filters."""

    name = "point2pixel"
    stage = "file"
    category = "pipeline"

    def __init__(self, x_inc=None, y_inc=None, want_sums=True, **kwargs):
        super().__init__(**kwargs)
        self.x_inc = float_or(x_inc)
        self.y_inc = float_or(y_inc)
        self.want_sums = want_sums


    def process_chunk(self, chunk, region=None):
        """Override this. Return filtered chunk (recarray) or None."""

        if region:
            #print(self.x_inc, self.y_inc)
            xcount, ycount, _ = region.geo_transform(
                x_inc=self.x_inc, y_inc=self.y_inc, node='grid'
            )
            #print(xcount, ycount)

            point_array = PointPixels(
                src_region=region,
                x_size=xcount,
                y_size=ycount,
                verbose=True
            )
            arrs, srcwin, gt =  point_array(
                chunk,
                weight=1,
                uncertainty=0,
                mode='sums' if self.want_sums else 'mean'
            )

            return arrs, srcwin, gt


    def _stream_wrapper(self, input_stream, entry=None, region=None):
        count = 0

        if region:
            for chunk in input_stream:
                count += chunk.size
                arrs, srcwin, gt = self.process_chunk(chunk, region=region)
                yield arrs, srcwin, gt
        logger.info(f'Parsed {count} data records from {entry["dst_fn"]}')


    def run(self, entries):
        for mod, entry in entries:
            # Check for existing stream
            stream = entry.get('stream')
            stream_type = entry.get('stream_type')
            if stream and stream_type == 'xyz_recarray':
                entry['stream'] = self._stream_wrapper(stream, entry=entry, region=mod.region)
                entry['stream_type'] = 'point_pixels_arrays'
        return entries
