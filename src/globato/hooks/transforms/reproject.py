#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.hooks.transforms.reproject
~~~~~~~~~~~~~

Reproject the data stream. Hook for fetchez.

:copyright: (c) 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
from fetchez.hooks import FetchHook

from transformez.srs import SRSParser
from transformez.utils import RasterQuery

logger = logging.getLogger(__name__)


class StreamReproject(FetchHook):
    name = "stream-reproject"
    meta_stage = "stream"
    meta_desc = "Reproject the stream to the desired SRS using Transformez."
    meta_category = "point-stream"
    meta_requires = "point-stream"
    meta_aliases = ["stream_reproject"]

    def __init__(
        self, dst_srs=None, src_srs=None, vert_grid=None, cache_dir=".", **kwargs
    ):
        super().__init__(**kwargs)
        self.dst_srs = dst_srs
        self.forced_src_srs = src_srs
        self.vert_grid = vert_grid
        self.cache_dir = cache_dir
        self._cache = {}

    def _get_pipeline(self, entry_src_srs, region=None):
        if not SRSParser:
            return None

        actual_src = self.forced_src_srs or entry_src_srs or "EPSG:4326"
        if not actual_src:
            return None

        if actual_src in self._cache:
            return self._cache[actual_src]

        # Use Transformez to generate the shift grid,
        # aligned to the source CRS
        parser = SRSParser(
            actual_src,
            self.dst_srs,
            region=region,
            vert_grid=self.vert_grid,
            cache_dir=self.cache_dir,
            verbose=True,
        )
        horz_transformer, grid_fn = parser.get_components()
        grid_query = RasterQuery(grid_fn) if grid_fn else None

        self._cache[actual_src] = (horz_transformer, grid_query)
        return self._cache[actual_src]

    def run(self, entries):
        for mod, entry in entries:
            if not self.dst_srs:
                continue

            if self.is_point_stream(entry):
                src_srs = entry.get("src_srs", "EPSG:4326")

                safe_region = None
                mod_region = getattr(mod, "region", None)
                if mod_region:
                    try:
                        safe_region = mod_region.copy()  # .buffer(pct=5)
                    except Exception:
                        safe_region = list(mod_region)

                pipeline = self._get_pipeline(src_srs, region=safe_region)
                stream = entry.get("stream")

                if pipeline:
                    entry["stream"] = self._apply_transform(stream, pipeline)
                    entry["src_srs"] = self.dst_srs

        return entries

    def _apply_transform(self, stream, pipeline):
        horz_transformer, grid_query = pipeline

        logger.debug(f"[{self.name}] Applying transformation: {pipeline}")
        for chunk in stream:
            # Vertical Shift (using native, unprojected x/y coordinates)
            if grid_query and chunk["z"] is not None:
                shifts = grid_query.query(chunk["x"], chunk["y"])
                chunk["z"] += shifts

            # Horizontal Shift
            if horz_transformer:
                chunk["x"], chunk["y"] = horz_transformer.transform(
                    chunk["x"], chunk["y"]
                )

            yield chunk
