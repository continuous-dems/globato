#!/usr/bin/env python

"""
globato.hooks.filters.reference
~~~~~~~~~~~~~

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import os

import numpy as np
import rasterio
from fetchez.hooks import FetchHook
from fetchez.utils import float_or, str2bool
from scipy.ndimage import map_coordinates

from globato.utils import add_field_to_recarray

logger = logging.getLogger(__name__)


# Raster Sampling
class RasterSampling:
    """Sampling rasters at point locations using Rasterio."""

    def sample_raster(self, raster_fn, points, default_val=np.nan):
        if not rasterio:
            logger.error("Rasterio required for raster sampling.")
            return np.full(len(points), default_val)

        if not os.path.exists(raster_fn):
            return np.full(len(points), default_val)

        try:
            with rasterio.open(raster_fn) as src:
                coords = list(zip(points["x"], points["y"]))

                sampled = np.array([val[0] for val in src.sample(coords)])

                if src.nodata is not None:
                    sampled[sampled == src.nodata] = np.nan

                return sampled
        except Exception as e:
            logger.error(f"Sampling error {raster_fn}: {e}")
            return np.full(len(points), default_val)


class RasterMask(FetchHook, RasterSampling):
    """Filter using a raster mask (Non-zero = Keep)."""

    name = "raster_mask"
    meta_stage = "stream"
    meta_category = "stream-filter"
    meta_desc = "Filter a point stream with a raster mask."

    def __init__(self, mask_fn=None, invert=False, set_class=7, **kwargs):
        super().__init__(**kwargs)
        self.mask_fn = mask_fn
        self.invert = str2bool(invert)
        self.set_class = set_class

    def run(self, entries):
        for mod, entry in entries:
            stream = entry.get("stream")
            if not stream:
                continue

            if not self.mask_fn:
                continue

            entry["stream"] = self._process_stream(stream)
        return entries

    def _process_stream(self, stream):
        for chunk in stream:
            if "classification" not in chunk.dtype.names:
                chunk = add_field_to_recarray(chunk, "classification", np.uint8, 0)

            vals = self.sample_raster(self.mask_fn, chunk, default_val=0)

            is_inside = (vals != 0) & (~np.isnan(vals))
            # Remove Outside -> ~is_inside
            mask = is_inside if self.invert else ~is_inside
            if np.any(mask):
                chunk["classification"][mask] = self.set_class

            logger.info(
                f"Reclassified {np.count_nonzero(mask)} points using {self.name}"
            )
            yield chunk


class DiffZ(FetchHook, RasterSampling):
    """Filter based on diff from reference raster."""

    name = "diffz"
    meta_stage = "stream"
    meta_desc = "filter points based on a reference raster residuals"
    meta_category = "stream-filter"

    def __init__(
        self, raster=None, min_diff=None, max_diff=None, invert=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.raster = raster
        self.min_diff = float_or(min_diff)
        self.max_diff = float_or(max_diff)
        self.invert = str2bool(invert)

    def run(self, entries):
        for mod, entry in entries:
            stream = entry.get("stream")
            if not stream:
                continue

            if not self.raster:
                continue

            entry["stream"] = self._process_stream(stream)
        return entries

    def filter_chunk(self, chunk):
        if "classification" not in chunk.dtype.names:
            chunk = add_field_to_recarray(chunk, "classification", np.uint8, 0)

        ref_z = self.sample_raster(self.raster, chunk)
        diff = chunk["z"] - ref_z

        keep = np.ones(len(diff), dtype=bool)
        keep &= ~np.isnan(diff)

        if self.min_diff is not None:
            keep &= diff >= self.min_diff
        if self.max_diff is not None:
            keep &= diff <= self.max_diff

        mask = ~keep if not self.invert else keep
        if np.any(mask):
            chunk["classification"][mask] = self.set_class

        logger.debug(f"Reclassified {np.count_nonzero(mask)} points using {self.name}")
        return mask

    # def _process_stream(self, stream):
    #     for chunk in stream:
    #         self.filter_chunk(chunk)
    #         if "classification" not in chunk.dtype.names:
    #             chunk = add_field_to_recarray(chunk, "classification", np.uint8, 0)

    #         ref_z = self.sample_raster(self.raster, chunk)
    #         diff = chunk["z"] - ref_z

    #         keep = np.ones(len(diff), dtype=bool)
    #         keep &= ~np.isnan(diff)

    #         if self.min_diff is not None:
    #             keep &= diff >= self.min_diff
    #         if self.max_diff is not None:
    #             keep &= diff <= self.max_diff

    #         mask = ~keep if not self.invert else keep
    #         if np.any(mask):
    #             chunk["classification"][mask] = self.set_class

    #         # logger.info(f"Reclassified {np.count_nonzero(mask)} points using {self.name}")
    #         yield chunk


class Diff_Z(FetchHook, RasterSampling):
    """Filter based on diff from reference raster."""

    name = "diff-z"
    meta_stage = "stream"
    meta_desc = "Output the z difference based on a reference raster residuals"
    meta_category = "stream-filter"

    def __init__(
        self, raster=None, min_diff=None, max_diff=None, invert=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.raster = raster
        self.min_diff = float_or(min_diff)
        self.max_diff = float_or(max_diff)
        self.invert = str2bool(invert)

    def run(self, entries):
        for mod, entry in entries:
            stream = entry.get("stream")
            if not stream:
                continue

            if not self.raster:
                continue

            entry["stream"] = self._process_stream(stream)
        return entries

    def _process_stream(self, stream):
        for chunk in stream:
            ref_z = self.sample_raster(self.raster, chunk)
            chunk["z"] = chunk["z"] - ref_z
            yield chunk

    def filter_chunk(self, chunk):
        mask = np.isnan(chunk["z"])
        return mask


class DiffZHook(FetchHook):
    """Calculates the residual (Z-diff) between the point stream and a reference DEM
    using Bilinear Interpolation to account for sub-pixel terrain slope.
    """

    name = "z-residual"
    meta_stage = "stream"
    meta_category = "stream-transform"
    meta_desc = (
        "Calculates exact residuals by mapping points onto a sloped bilinear DEM facet."
    )

    def __init__(self, raster=None, **kwargs):
        super().__init__(**kwargs)
        self.raster = raster

    def _process_stream(self, stream):
        # Open the reference DEM once
        with rasterio.open(self.raster) as src:
            data = src.read(1)
            ndv = src.nodata

            # Convert NoData to NaN so the interpolator handles it gracefully
            if ndv is not None:
                data = np.where(data == ndv, np.nan, data)

            # Get the inverse affine transform to map geographic coordinates to array indices
            inv_transform = ~src.transform

            for chunk in stream:
                if chunk is None or len(chunk) == 0:
                    continue

                x_vals = chunk["x"]
                y_vals = chunk["y"]

                # Convert geographic coordinates to fractional array indices
                cols, rows = inv_transform * (x_vals, y_vals)

                # Bilinear Interpolation (order=1) maps the point onto the sloped plane!
                sampled_z = map_coordinates(
                    data,
                    [rows, cols],
                    order=1,
                    mode="constant",
                    cval=np.nan,
                    prefilter=False,
                )

                # Ensure the chunk has a residual column
                if "residual" not in chunk.dtype.names:
                    from numpy.lib.recfunctions import append_fields

                    chunk = append_fields(
                        chunk,
                        "residual",
                        np.full(len(chunk), np.nan, dtype="f4"),
                        usemask=False,
                    )

                # Only calculate residuals where the DEM has valid data
                valid_mask = ~np.isnan(sampled_z)
                chunk["residual"][valid_mask] = (
                    chunk["z"][valid_mask] - sampled_z[valid_mask]
                )
                chunk["z"][valid_mask] = chunk["z"][valid_mask] - sampled_z[valid_mask]
                # chunk["z"][valid_mask] = sampled_z[valid_mask]
                # Yield only the points that successfully intersected the DEM
                yield chunk[valid_mask]

    def run(self, entries):
        for mod, entry in entries:
            if self.is_point_stream(entry):
                stream = entry.get("stream")
                entry["stream"] = self._process_stream(stream)
        return entries
