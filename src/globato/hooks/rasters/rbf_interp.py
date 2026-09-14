#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.hooks.rasters.rbf_interp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interpolates gaps in a stacked DEM using SciPy's RBFInterpolator.
Defaults to 'thin_plate_spline', mathematically mimicking GMT's surface.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import numpy as np
from scipy.interpolate import RBFInterpolator

from .base import RasterStreamHook

logger = logging.getLogger(__name__)


class RBFInterp(RasterStreamHook):
    """Fills voids using Radial Basis Functions.

    Perfect for smooth interpolation of sparse point datasets (like gravity bathy).
    """

    name = "interp_rbf"
    default_suffix = "_rbf"
    meta_desc = "Intpolate NoData voids using RBF."
    meta_consumes = "raster-stream"
    meta_produces = "raster-stream"
    meta_tags = ["globato", "interpolation", "multi-stack"]
    processing_mode = "chunk"

    def __init__(
        self,
        kernel="thin_plate_spline",
        smoothing=20.0,
        neighbors=100,
        epsilon=None,
        degree=6,
        **kwargs,
    ):
        """
        Args:
            kernel (str): 'thin_plate_spline', 'cubic', 'gaussian', 'linear', etc.
            smoothing (float): > 0 allows the surface to pass slightly near points instead of exactly through.
            neighbors (int): Restricts the RBF to the N nearest points. Helps in preventing memory crashes.
        """

        super().__init__(**kwargs)
        self.kernel = kernel.lower()
        self.smoothing = float(smoothing)
        self.neighbors = int(neighbors) if neighbors else None
        self.epsilon = float(epsilon) if epsilon else None
        self.degree = int(degree)

        # We need a large buffer for smooth continuous RBF boundaries
        if getattr(self, "buffer", 0) == 0:
            self.buffer = 40

    def process_chunk(self, data, ndv, entry, transform=None, window=None):
        is_3d = data.ndim == 3
        work_data = data[0] if is_3d else data

        valid_mask = (work_data != ndv) & ~np.isnan(work_data)
        if np.all(valid_mask) or not np.any(valid_mask):
            return data

        # y_valid, x_valid = np.where(valid_mask)
        y_valid_idx, x_valid_idx = np.where(valid_mask)

        x_valid, y_valid = self._extract_subpixel_coords(
            data if is_3d else None,
            y_valid_idx,
            x_valid_idx,
            transform,
            apply_jitter=False,
        )
        points = np.column_stack((x_valid, y_valid))
        values = work_data[valid_mask]

        missing_mask = ~valid_mask
        y_missing_idx, x_missing_idx = np.where(missing_mask)
        xq, yq = transform * (x_missing_idx + 0.5, y_missing_idx + 0.5)
        query_points = np.column_stack((xq, yq))
        # query_points = np.column_stack((x_missing, y_missing))

        if len(query_points) == 0:
            return data

        try:
            kwargs = {
                "kernel": self.kernel,
                "smoothing": self.smoothing,
                "degree": self.degree,
            }
            if self.neighbors and len(points) > self.neighbors:
                kwargs["neighbors"] = self.neighbors
            if self.epsilon:
                kwargs["epsilon"] = self.epsilon

            rbf = RBFInterpolator(points, values, **kwargs)
            interpolated_values = rbf(query_points)

            z_filled = work_data.copy()
            z_filled[missing_mask] = interpolated_values

            if is_3d:
                result = data.copy()
                result[0] = z_filled
            else:
                result = z_filled

            return result.astype(data.dtype)

        except Exception as e:
            logger.error(f"[{self.name}] RBF interpolation failed on chunk: {e}")
            return data
