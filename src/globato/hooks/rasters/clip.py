#!/usr/bin/env python

"""
globato.hooks.rasters.clip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Clip a raster to a vector

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging

import numpy as np

from .base import RasterStreamHook

logger = logging.getLogger(__name__)


class RasterClipHook(RasterStreamHook):
    """Clips a raster array using vector geometries."""

    name = "raster_clip"
    default_suffix = "_clipped"
    meta_desc = "Clip a raster using vector geometries."
    # meta_stage = "file"

    def __init__(self, invert=False, **kwargs):
        super().__init__(**kwargs)

        self.invert = str(invert).lower() == "true"

        logger.debug(f"Clipping {self.name} with {self.barrier}")

    def process_chunk(self, data, ndv, entry, transform=None, window=None):
        """Process individual windows/chunks passed by RasterHook."""

        out_shape = data.shape[-2:] if data.ndim >= 2 else data.shape
        geom_mask = self._create_barrier_mask(out_shape, transform)

        if geom_mask is None:
            return data

        if data.ndim == 3:
            geom_mask = np.broadcast_to(geom_mask, data.shape)

        if self.invert:
            return np.where(~geom_mask, data, ndv)
        else:
            return np.where(geom_mask, data, ndv)
