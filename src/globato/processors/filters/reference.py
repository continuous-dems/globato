#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.filters.references
~~~~~~~~~~~~~

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import numpy as np
import rasterio
from fetchez.hooks import FetchHook
from fetchez import utils

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
                coords = list(zip(points['x'], points['y']))

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
    stage = " file"
    category = "stream-filter"

    def __init__(self, mask_fn=None, invert=False, set_class=7, **kwargs):
        super().__init__(**kwargs)
        self.mask_fn = mask_fn
        self.invert = str2bool(invert)
        self.set_class=set_class

    def run(self, entries):
        for mod, entry in entries:
            stream = entry.get("stream")
            if not stream: continue

            if not self.mask_fn: continue

            entry["stream"] = self._process_stream(stream)
        return entries

    def _process_stream(self, stream):
        for chunk in stream:
            if "classification" not in chunk.dtype.names:
                chunk = utils.add_field_to_recarray(chunk, "classification", np.uint8, 0)

            vals = self.sample_raster(self.mask_fn, chunk, default_val=0)

            is_inside = (vals != 0) & (~np.isnan(vals))
            # Remove Outside -> ~is_inside
            mask = is_inside if self.invert else ~is_inside
            if np.any(mask):
                chunk["classification"][mask] = self.set_class

            yield chunk

class DiffZ(FetchHook, RasterSampling):
    """Filter based on diff from reference raster."""

    def __init__(self, raster=None, min_diff=None, max_diff=None, invert=False, **kwargs):
        super().__init__(**kwargs)
        self.raster = raster
        self.min_diff = float_or(min_diff)
        self.max_diff = float_or(max_diff)
        self.invert = str2bool(invert)

    def run(self, entries):
        for mod, entry in entries:
            stream = entry.get("stream")
            if not stream: continue

            if not self.raster: continue

            entry["stream"] = self._process_stream(stream)
        return entries

    def _process_stream(self, stream):
        for chunk in stream:
            if "classification" not in chunk.dtype.names:
                chunk = utils.add_field_to_recarray(chunk, "classification", np.uint8, 0)

            ref_z = self.sample_raster(self.raster, chunk)
            diff = chunk['z'] - ref_z

            keep = np.ones(len(diff), dtype=bool)
            keep &= ~np.isnan(diff)

            if self.min_diff is not None: keep &= (diff >= self.min_diff)
            if self.max_diff is not None: keep &= (diff <= self.max_diff)

            mask = ~keep if not self.invert else keep
            if np.any(mask):
                chunk["classification"][mask] = self.set_class

            yield chunk
