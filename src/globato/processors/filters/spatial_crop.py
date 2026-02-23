#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.filters.spatial_crop
~~~~~~~~~~~~~

crop stream data by region

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import numpy as np
from fetchez.hooks import FetchHook
from fetchez.utils import str2bool
from globato.utils import add_field_to_recarray
# from globato.processors.filter.base import GlobatoFilter

logger = logging.getLogger(__name__)


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
        self.soft = str2bool(soft)
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
                        chunk = add_field_to_recarray(chunk, 'classification', np.uint8, 0)

                    chunk['classification'][~mask] = self.set_class
                yield chunk

            else:
                # Delete outside points entirely
                if inside_count > 0:
                    yield chunk[mask]

        if dropped_total > 0:
            action = "Classified" if self.soft else "Dropped"
            logger.info(f"[SpatialCrop] {action} {dropped_total} points outside region (Kept {kept_total}).")


# class SpatialCrop_(GlobatoFilter):
#     """Crops the stream to the module's target region.
#     Dramatically improves performance by removing out-of-bounds points early.

#     Usage:
#       --hook spatial_crop                 (Hard Crop: Deletes points)
#       --hook spatial_crop:soft=True       (Soft Crop: Classifies as 7)
#     """

#     name = "spatial_crop"
#     desc = "Crop stream to the target bounding box"

#     def __init__(self, soft=False, **kwargs):
#         super().__init__(**kwargs)
#         self.soft = utils.str2bool(soft)

#     def setup(self, mod, entry):
#         self.target_region = getattr(mod, 'region', None)
#         if not self.target_region:
#             return False # Skip filter if no region
#         return True

#     def filter_chunk(self, chunk):
#         w, e, s, n = self.target_region

#         mask = (chunk['x'] >= w) & (chunk['x'] <= e) & \
#                (chunk['y'] >= s) & (chunk['y'] <= n)

#         if self.soft:
#             return ~mask
#         else:
#             return chunk[mask]
