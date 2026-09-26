#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Apply manifest-resolved spatial claims to point streams on an output grid."""

from __future__ import annotations

import numpy as np
import shapely
from fetchez.hooks import FetchHook
from fetchez.utils import str2inc
from pyproj import CRS, Transformer
from shapely.ops import transform as shapely_transform


class ClaimGridFilter(FetchHook):
    """Crop point streams to accepted claims and block excluded grid cells.

    Point-in-polygon cropping alone is insufficient for hierarchical DEM sources:
    disjoint source footprints can still place points in the same output cell.
    When excluded higher-priority coverage overlaps a cell with positive area,
    this filter drops lower-priority candidate points from the entire cell.
    Boundary-only contact does not suppress otherwise valid adjacent cells.
    """

    name = "claim-grid-filter"
    meta_aliases = ["claim_grid_filter"]
    meta_stage = "stream"
    meta_category = "stream-filter"
    meta_desc = "Enforce manifest spatial claims on point streams at DEM-cell scale."

    def __init__(
        self,
        res=None,
        accepted_key="accepted_geometry",
        excluded_key="excluded_geometry",
        required_key="claim_required",
        geometry_srs="EPSG:4326",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.res = str2inc(res)
        self.accepted_key = accepted_key
        self.excluded_key = excluded_key
        self.required_key = required_key
        self.geometry_srs = geometry_srs

    @staticmethod
    def _truthy(value):
        if isinstance(value, str):
            return value.lower() in {"true", "1", "yes", "y"}
        return bool(value)

    @staticmethod
    def _horizontal_crs(value):
        target = CRS.from_user_input(value or "EPSG:4326")
        if target.is_compound:
            target = next(
                (
                    component
                    for component in target.sub_crs_list
                    if component.is_geographic or component.is_projected
                ),
                target,
            )
        return target

    def _stream_geometry(self, geometry, target_srs):
        source = self._horizontal_crs(self.geometry_srs)
        target = self._horizontal_crs(target_srs)
        if target == source:
            return geometry
        transformer = Transformer.from_crs(source, target, always_xy=True)
        return shapely_transform(transformer.transform, geometry)

    @staticmethod
    def _positive_cell_overlap(excluded, boxes):
        """Return True only where excluded coverage overlaps a cell with area.

        A full geometric intersection for every cell is expensive and was the
        observed hotspot in the broad v53 study-area run.  For polygonal cells,
        a DE-9IM first-character of ``2`` means the polygon interiors overlap
        with two-dimensional (positive-area) intersection.  This preserves the
        scientific rule that boundary-only contact must not suppress a lower-tier
        cell.  Fall back to an explicit area test only if the predicate itself
        errors on a particular geometry.
        """

        try:
            return np.asarray(
                shapely.relate_pattern(excluded, boxes, "2********"), dtype=bool
            )
        except Exception:
            return shapely.area(shapely.intersection(excluded, boxes)) > 0

    @staticmethod
    def _crop_stream(stream, accepted, excluded=None, grid=None):
        for chunk in stream:
            if len(chunk) == 0:
                yield chunk
                continue

            inside = shapely.intersects_xy(accepted, chunk["x"], chunk["y"])
            if excluded is not None and not excluded.is_empty:
                inside &= ~shapely.intersects_xy(excluded, chunk["x"], chunk["y"])

                if grid is not None and np.any(inside):
                    indices = np.flatnonzero(inside)
                    cols = np.floor((chunk["x"][indices] - grid[0]) / grid[1])
                    rows = np.floor((chunk["y"][indices] - grid[3]) / grid[5])
                    cells, inverse = np.unique(
                        np.column_stack((cols, rows)), axis=0, return_inverse=True
                    )
                    left = grid[0] + cells[:, 0] * grid[1]
                    top = grid[3] + cells[:, 1] * grid[5]
                    boxes = shapely.box(left, top + grid[5], left + grid[1], top)
                    blocked = ClaimGridFilter._positive_cell_overlap(excluded, boxes)
                    inside[indices] &= ~blocked[inverse]

            yield chunk[inside]

    def run(self, entries):
        for mod, entry in entries:
            if not self.is_point_stream(entry):
                continue

            accepted_value = entry.get(self.accepted_key)
            required = self._truthy(entry.get(self.required_key))
            if not accepted_value:
                if required:
                    raise RuntimeError(
                        f"claim-grid-filter requires '{self.accepted_key}' for a marked entry"
                    )
                continue

            try:
                accepted = shapely.from_wkt(accepted_value)
                excluded = (
                    shapely.from_wkt(entry[self.excluded_key])
                    if entry.get(self.excluded_key)
                    else None
                )
                target_srs = entry.get("src_srs")
                accepted = self._stream_geometry(accepted, target_srs)
                shapely.prepare(accepted)
                if excluded is not None:
                    excluded = self._stream_geometry(excluded, target_srs)
                    shapely.prepare(excluded)
            except Exception as exc:
                raise RuntimeError("Claim geometry or stream CRS is invalid") from exc

            if accepted is None or accepted.is_empty:
                raise RuntimeError("Accepted claim contains no usable geometry")
            if self.res is None or self.res <= 0:
                raise RuntimeError("claim-grid-filter requires the output resolution")

            region = getattr(mod, "region", None)
            if region is None:
                raise RuntimeError("claim-grid-filter requires the output region")

            xcount, ycount, _ = region.geo_transform(
                x_inc=self.res, y_inc=self.res, node="pixel"
            )
            grid = region.geo_transform_from_count(x_count=xcount, y_count=ycount)
            entry["stream"] = self._crop_stream(
                entry["stream"], accepted, excluded, grid
            )

        return entries
