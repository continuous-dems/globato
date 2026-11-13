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

import numpy as np
from fetchez.hooks import FetchHook
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError

from transformez.reference.parser import parse_reference
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
        self._horz_cache = {}

    def _get_pipeline(self, entry_src_srs, region=None):
        actual_src = self.forced_src_srs or entry_src_srs or "EPSG:4326"
        logger.debug(
            "[stream-reproject] SRSParser(src=%r, dst=%r)",
            actual_src,
            self.dst_srs,
        )
        if not actual_src:
            return None

        if actual_src in self._cache:
            return self._cache[actual_src]

        _effective_logger_level = logger.getEffectiveLevel()

        # Use Transformez to generate the shift grid,
        # aligned to the source CRS
        parser = SRSParser(
            actual_src,
            self.dst_srs,
            region=region,
            vert_grid=self.vert_grid,
            cache_dir=self.cache_dir,
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
                logger.debug(
                    "[stream-reproject] module=%s file=%s "
                    "entry_src_srs=%r forced_src_srs=%r dst_srs=%r",
                    mod.name,
                    entry.get("dst_fn"),
                    src_srs,
                    self.forced_src_srs,
                    self.dst_srs,
                )
                safe_region = None
                mod_region = getattr(mod, "region", None)
                if mod_region:
                    try:
                        safe_region = mod_region.copy()  # .buffer(pct=5)
                    except Exception:
                        safe_region = list(mod_region)

                actual_src = self.forced_src_srs or src_srs or "EPSG:4326"
                stream = entry.get("stream")
                if stream is not None:
                    entry["stream"] = self._apply_transform(
                        stream, actual_src, safe_region
                    )
                entry["src_srs"] = self.dst_srs

        return entries

    def _get_horizontal(self, src_srs):
        """Horizontal-only transformer, built without any vertical grid."""

        if src_srs in self._horz_cache:
            return self._horz_cache[src_srs]

        source = parse_reference(src_srs)
        target = parse_reference(self.dst_srs)
        transformer = None
        if (
            source.horizontal
            and target.horizontal
            and source.horizontal != target.horizontal
        ):
            transformer = Transformer.from_crs(
                source.horizontal, target.horizontal, always_xy=True
            )
        self._horz_cache[src_srs] = transformer
        return transformer

    @staticmethod
    def _grid_bounds(src_srs, region):
        """Lon/lat bounds of the shift grid Transformez would build for region.

        Transformez builds the grid in EPSG:4326 over the region (taken in the
        region's own SRS, or the source's horizontal CRS when it has none).
        Returns None when the bounds cannot be worked out, in which case the
        grid is built at the first chunk, as before.
        """

        if region is None:
            return None
        try:
            w, e, s, n = (float(v) for v in list(region)[:4])
        except (TypeError, ValueError):
            return None

        region_srs = getattr(region, "srs", None)
        try:
            if region_srs:
                region_crs = CRS.from_user_input(region_srs)
            else:
                region_crs = parse_reference(src_srs).horizontal
        except (CRSError, ValueError):
            return None
        if region_crs is None:
            region_crs = CRS.from_epsg(4326)

        if region_crs != CRS.from_epsg(4326):
            to_wgs84 = Transformer.from_crs(region_crs, 4326, always_xy=True)
            w, s, e, n = to_wgs84.transform_bounds(w, s, e, n)
        return w, e, s, n

    @staticmethod
    def _any_in_grid(chunk, bounds):
        """True when any point of the chunk falls inside the shift grid bounds.

        Matches RasterQuery's lookup: native x/y, with the same longitude wrap.
        """

        x = np.asarray(chunk["x"], dtype="f8")
        y = np.asarray(chunk["y"], dtype="f8")
        if x.size == 0:
            return False
        w, e, s, n = bounds
        if w < 0:
            x = np.where(x > 180, x - 360, x)
        else:
            x = np.where(x < 0, x + 360, x)
        return bool(np.any((x >= w) & (x <= e) & (y >= s) & (y <= n)))

    def _apply_transform(self, stream, src_srs, region):
        """Transform the stream, building the vertical shift grid lazily.

        The shift grid (which may need VDatum, xGEOID, etc.) is only built once
        a chunk has a point inside the grid's area. Until then, chunks get the
        horizontal transform alone, which is what they would get anyway: points
        outside the grid get a zero vertical shift from RasterQuery. So a file
        with no points in the region (e.g. a wide-footprint chart over an inland
        tile with no VDatum coverage) passes through without building a grid,
        while a real lack of coverage under in-region points still raises.
        """

        bounds = self._grid_bounds(src_srs, region)
        pipeline = None

        for chunk in stream:
            if pipeline is None and (
                bounds is None or self._any_in_grid(chunk, bounds)
            ):
                pipeline = self._get_pipeline(src_srs, region=region)
                logger.debug(f"[{self.name}] Applying transformation: {pipeline}")

            if pipeline is not None:
                horz_transformer, grid_query = pipeline
                # Vertical Shift (using native, unprojected x/y coordinates)
                if grid_query and chunk["z"] is not None:
                    shifts = grid_query.query(chunk["x"], chunk["y"])
                    chunk["z"] += shifts
            else:
                horz_transformer = self._get_horizontal(src_srs)

            # Horizontal Shift
            if horz_transformer:
                chunk["x"], chunk["y"] = horz_transformer.transform(
                    chunk["x"], chunk["y"]
                )

            yield chunk
