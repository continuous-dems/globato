#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.hooks.filters.tnm_coverage_filter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Crop TNM streams to coverage selected at the manifest stage.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import numpy as np
import shapely
from fetchez.hooks import FetchHook
from fetchez.utils import str2inc
from pyproj import CRS, Transformer
from shapely.ops import transform as shapely_transform


class TNMCoverageFilter(FetchHook):
    """Crop TNM streams to accepted coverage and higher-tier exclusions."""

    name = "tnm-coverage-filter"
    meta_desc = "Crop TNM streams to manifest-selected coverage."
    meta_stage = "stream"
    meta_category = "stream-filter"
    meta_aliases = ["tnm_coverage_filter"]

    def __init__(self, res=None, **kwargs):
        super().__init__(**kwargs)
        self.res = str2inc(res)

    @staticmethod
    def _crop_stream(stream, geometry, excluded=None, grid=None):
        for chunk in stream:
            if len(chunk) == 0:
                yield chunk
                continue
            inside = shapely.intersects_xy(geometry, chunk["x"], chunk["y"])
            if excluded is not None and not excluded.is_empty:
                inside &= ~shapely.intersects_xy(excluded, chunk["x"], chunk["y"])
                if grid is not None and np.any(inside):
                    # Use the same cell indexing as PointPixels. Disjoint source
                    # footprints can still contribute points to one output cell.
                    indices = np.flatnonzero(inside)
                    cols = np.floor((chunk["x"][indices] - grid[0]) / grid[1])
                    rows = np.floor((chunk["y"][indices] - grid[3]) / grid[5])
                    cells, inverse = np.unique(
                        np.column_stack((cols, rows)), axis=0, return_inverse=True
                    )
                    left = grid[0] + cells[:, 0] * grid[1]
                    top = grid[3] + cells[:, 1] * grid[5]
                    boxes = shapely.box(left, top + grid[5], left + grid[1], top)
                    blocked = shapely.intersects(excluded, boxes)
                    inside[indices] &= ~blocked[inverse]
            yield chunk[inside]

    @staticmethod
    def _stream_geometry(geometry, target_srs):
        target = CRS.from_user_input(target_srs or "EPSG:4326")
        if target.is_compound:
            target = next(
                (
                    component
                    for component in target.sub_crs_list
                    if component.is_geographic or component.is_projected
                ),
                target,
            )
        source = CRS.from_epsg(4326)
        if target == source:
            return geometry
        transformer = Transformer.from_crs(source, target, always_xy=True)
        return shapely_transform(transformer.transform, geometry)

    def run(self, entries):
        for mod, entry in entries:
            if not self.is_point_stream(entry):
                continue

            value = entry.get("_tnm_coverage_wkt")
            if not value:
                raise RuntimeError(
                    "TNM coverage is missing. The manifest-stage "
                    "tnm-coverage hook must run first."
                )
            try:
                geometry = shapely.from_wkt(value)
                excluded = (
                    shapely.from_wkt(entry["_tnm_excluded_wkt"])
                    if entry.get("_tnm_excluded_wkt")
                    else None
                )
                target_srs = entry.get("src_srs")
                geometry = self._stream_geometry(geometry, target_srs)
                if excluded is not None:
                    excluded = self._stream_geometry(excluded, target_srs)
                    shapely.prepare(excluded)
            except Exception as exc:
                raise RuntimeError(
                    "TNM coverage geometry or stream CRS is invalid"
                ) from exc
            if geometry is None or geometry.is_empty:
                raise RuntimeError("TNM coverage contains no usable geometry")

            if self.res is None or self.res <= 0:
                raise RuntimeError("tnm-coverage-filter requires the output resolution")
            region = getattr(mod, "region", None)
            if region is None:
                raise RuntimeError("tnm-coverage-filter requires the output region")
            xcount, ycount, _ = region.geo_transform(
                x_inc=self.res, y_inc=self.res, node="pixel"
            )
            grid = region.geo_transform_from_count(x_count=xcount, y_count=ycount)
            entry["stream"] = self._crop_stream(
                entry["stream"], geometry, excluded, grid
            )
        return entries
