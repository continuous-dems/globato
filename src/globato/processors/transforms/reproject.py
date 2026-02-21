#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.transforms.reproject
~~~~~~~~~~~~~

Reproject the data stream. Hook for fetchez.
"""

import logging
import numpy as np

from fetchez.hooks import FetchHook
from transformez.srs import SRSParser
from transformez.utils import RasterQuery

logger = logging.getLogger(__name__)

class StreamReproject(FetchHook):
    name = "stream_reproject"
    stage = "file"
    desc = "Reproject the stream to the desired SRS using Transformez."
    category = "streams"

    def __init__(self, dst_srs, src_srs=None, vert_grid=None, **kwargs):
        super().__init__(**kwargs)
        self.dst_srs = dst_srs
        self.forced_src_srs = src_srs
        self.vert_grid = vert_grid
        self._cache = {}

    def _get_pipeline(self, entry_src_srs, region=None):
        if not SRSParser: return None

        actual_src = self.forced_src_srs or entry_src_srs or 'EPSG:4326'
        if not actual_src: return None

        if actual_src in self._cache:
            return self._cache[actual_src]

        parser = SRSParser(actual_src, self.dst_srs, region=region, vert_grid=self.vert_grid)
        t_in, t_out, grid_fn = parser.get_components()

        grid_query = RasterQuery(grid_fn) if grid_fn else None

        self._cache[actual_src] = (t_in, t_out, grid_query)
        return self._cache[actual_src]

    def run(self, entries):
        for mod, entry in entries:
            stream = entry.get('stream')
            stream_type = entry.get('stream_type')
            if not stream or stream_type != 'xyz_recarray':
                continue

            src_srs = entry.get('src_srs', 'EPSG:4326')
            pipeline = self._get_pipeline(src_srs, region=mod.region)

            if pipeline:
                entry['stream'] = self._apply_transform(stream, pipeline)
                entry['src_srs'] = self.dst_srs

        return entries

    def _apply_transform(self, stream, pipeline):
        t_to_hub, t_from_hub, grid_query = pipeline

        for chunk in stream:
            h_x, h_y = t_to_hub.transform(chunk['x'], chunk['y'])

            if grid_query and chunk['z'] is not None:
                shifts = grid_query.query(h_x, h_y)
                chunk['z'] += shifts

            d_x, d_y = t_from_hub.transform(h_x, h_y)
            chunk['x'] = d_x
            chunk['y'] = d_y

            yield chunk
