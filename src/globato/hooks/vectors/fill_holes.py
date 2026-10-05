#!/usr/bin/env python

"""
globato.hooks.vectors.vector_fill_holes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Removes interior rings (holes/ponds) from polygon vectors.
Perfect for solidifying landmasks and delineating continuous ocean boundaries.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import os

import shapely
from fetchez.hooks import FetchHook
from pyogrio.raw import read, write
from shapely.geometry import MultiPolygon, Polygon

logger = logging.getLogger(__name__)


class VectorFillHoles(FetchHook):
    """Removes interior holes from vector polygons.

    Args:
        min_area (float): Minimum area a hole must be to be kept.
                          Defaults to 0.0 (removes ALL holes, creating solid landmasses).
                          Note: Area is calculated in the native CRS units (e.g., square degrees or meters).
        suffix (str): Suffix for the output file.
    """

    name = "vector_fill_holes"
    meta_stage = "collection"  # post
    meta_category = "vector-op"
    meta_desc = "Removes interior holes from vector polygons."

    def __init__(self, min_area=0.0, suffix="_filled", **kwargs):
        super().__init__(**kwargs)
        self.min_area = float(min_area)
        self.suffix = suffix

    def _remove_holes(self, geom):
        """Recursively removes small interior rings from Polygons/MultiPolygons."""
        if geom.geom_type == "Polygon":
            if self.min_area <= 0.0:
                return Polygon(geom.exterior)

            new_interiors = []
            for ring in geom.interiors:
                ring_poly = Polygon(ring)
                if ring_poly.area > self.min_area:
                    new_interiors.append(ring)

            return Polygon(geom.exterior, new_interiors)

        elif geom.geom_type == "MultiPolygon":
            new_parts = []
            for part in geom.geoms:
                new_parts.append(self._remove_holes(part))
            return MultiPolygon(new_parts)

        return geom

    def run(self, entries):
        new_entries = []

        for mod, entry in entries:
            src_fn = entry.get("dst_fn")

            if not src_fn or not src_fn.lower().endswith((".gpkg", ".shp", ".geojson")):
                new_entries.append((mod, entry))
                continue

            base, ext = os.path.splitext(src_fn)
            dst_fn = f"{base}{self.suffix}{ext}"

            logger.info(
                f"[{self.name}] Filling inland holes in {os.path.basename(src_fn)}..."
            )

            try:
                meta, geometry_wkb, field_data = read(src_fn)
                geoms = shapely.from_wkb(geometry_wkb)

                filled_geoms = [self._remove_holes(g) for g in geoms if g]
                new_geometry_wkb = shapely.to_wkb(filled_geoms)

                write(
                    dst_fn,
                    new_geometry_wkb,
                    field_data,
                    fields=meta["fields"],
                    geometry_type=meta["geometry_type"],
                    crs=meta["crs"],
                )

                entry["src_fn"] = src_fn
                entry["dst_fn"] = dst_fn
                entry.setdefault("artifacts", {})[self.name] = dst_fn

            except Exception as e:
                logger.error(f"[{self.name}] Failed to fill holes: {e}")

            new_entries.append((mod, entry))

        return new_entries
