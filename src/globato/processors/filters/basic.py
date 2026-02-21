#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.filters.basic
~~~~~~~~~~~~~

Basic Filters for point clouds.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import numpy as np
from fetchez.hooks import FetchHook
from fetchez import utils
from .base import GlobatoFilter

logger = logging.getLogger(__name__)


class RangeZ(GlobatoFilter):
    """Classify points outside a Z range as Noise (or specified class).
    Does NOT remove points; only reclassifies them.

    Usage: --hook range_z:min_z=-50:max_z=0:set_class=7
    """

    name = "range_z"
    desc = "Classify points by Z range"

    def __init__(self, min_z=None, max_z=None, **kwargs):
        super().__init__(**kwargs)
        self.min_z = utils.float_or(min_z)
        self.max_z = utils.float_or(max_z)

    def filter_chunk(self, chunk):
        mask = np.zeros(len(chunk), dtype=bool)
        if self.min_z is not None:
            mask |= (chunk["z"] < self.min_z)

        if self.max_z is not None:
            mask |= (chunk["z"] > self.max_z)

        logger.info(f'filtered {np.count_nonzero(mask)} points')
        return mask


class SpatialCrop(FetchHook):
    """Crops the stream to the module's target region.

    Usage:
      --hook spatial_crop                 (Hard Crop: Deletes points)
      --hook spatial_crop:soft=True       (Soft Crop: Classifies as 7)
    """

    name = "spatial_crop"
    desc = "Crop stream to the target bounding box"
    stage = "file"
    category = "stream-filter"

    def __init__(self, soft=False, set_class=7, **kwargs):
        super().__init__(**kwargs)
        self.soft = utils.str2bool(soft)
        self.set_class = int(set_class)

    def run(self, entries):
        for mod, entry in entries:
            stream = entry.get('stream')
            if not stream: continue

            region = getattr(mod, 'region', None)
            if not region:
                continue

            entry['stream'] = self._crop_stream(stream, region)

        return entries

    def _crop_stream(self, stream, region):
        w, e, s, n = region

        dropped_total = 0
        kept_total = 0

        for chunk in stream:
            if chunk is None or len(chunk) == 0:
                continue

            mask = (chunk['x'] >= w) & (chunk['x'] <= e) & \
                   (chunk['y'] >= s) & (chunk['y'] <= n)

            inside_count = np.count_nonzero(mask)
            outside_count = len(chunk) - inside_count

            kept_total += inside_count
            dropped_total += outside_count

            if self.soft:
                # Classify outside points as noise
                if outside_count > 0:
                    if 'classification' not in chunk.dtype.names:
                        chunk = utils.add_field_to_recarray(chunk, 'classification', np.uint8, 0)

                    chunk['classification'][~mask] = self.set_class
                yield chunk

            else:
                # Delete outside points entirely
                if inside_count > 0:
                    yield chunk[mask]

        if dropped_total > 0:
            action = "Classified" if self.soft else "Dropped"
            logger.info(f"[SpatialCrop] {action} {dropped_total} points outside region (Kept {kept_total}).")


class SpatialCrop_(GlobatoFilter):
    """Crops the stream to the module's target region.
    Dramatically improves performance by removing out-of-bounds points early.

    Usage:
      --hook spatial_crop                 (Hard Crop: Deletes points)
      --hook spatial_crop:soft=True       (Soft Crop: Classifies as 7)
    """

    name = "spatial_crop"
    desc = "Crop stream to the target bounding box"

    def __init__(self, soft=False, **kwargs):
        super().__init__(**kwargs)
        self.soft = utils.str2bool(soft)

    def setup(self, mod, entry):
        self.target_region = getattr(mod, 'region', None)
        if not self.target_region:
            return False # Skip filter if no region
        return True

    def filter_chunk(self, chunk):
        w, e, s, n = self.target_region

        mask = (chunk['x'] >= w) & (chunk['x'] <= e) & \
               (chunk['y'] >= s) & (chunk['y'] <= n)

        if self.soft:
            return ~mask
        else:
            return chunk[mask]
