#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.hooks.osm_landmask
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fetches OSM Coastline data and polygonizes it into a landmask.
Hybrid: Uses OGR to read, Shapely to process, and Fiona to write.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
import math
import fiona
import shapely.wkb
from osgeo import ogr
from shapely.geometry import box, LineString, Point, mapping
from shapely.ops import linemerge, unary_union
from fetchez.hooks import FetchHook
from fetchez.core import Fetch, urlencode
from fetchez import utils

try:
    from fetchez.modules.gmrt import gmrt_fetch_point
except ImportError:
    gmrt_fetch_point = None

logger = logging.getLogger(__name__)

OSM_API = "https://lz4.overpass-api.de/api/interpreter"


class OSMLandmask(FetchHook):
    """Generates a Land/Water mask vector from OpenStreetMap.
    """
    name = "osm_landmask"
    stage = "pre"
    category = "generator"

    def __init__(self, filename="landmask.geojson", **kwargs):
        super().__init__(**kwargs)
        self.filename = filename

    def run(self, entries):
        regions = [getattr(mod, "region", None) for mod, _ in entries]
        valid_regions = [r for r in regions if r]

        if not valid_regions: return entries

        w = min(r[0] for r in valid_regions)
        e = max(r[1] for r in valid_regions)
        s = min(r[2] for r in valid_regions)
        n = max(r[3] for r in valid_regions)
        target_region = [w, e, s, n]

        out_path = os.path.join(os.getcwd(), self.filename)

        if os.path.exists(out_path):
            logger.info(f"[OSM] Using existing landmask: {out_path}")
            return entries

        logger.info(f"[OSM] Fetching coastline for {target_region}...")
        osm_xml = self._fetch_osm(target_region)

        if not osm_xml or os.path.getsize(osm_xml) < 100:
            self._handle_fallback(out_path, target_region)
            return entries

        logger.info("[OSM] Polygonizing and classifying coastline...")
        try:
            self._polygonize(osm_xml, out_path, target_region)
            logger.info(f"[OSM] Generated landmask: {out_path}")
        except Exception as e:
            logger.error(f"[OSM] Polygonization failed: {e}")
            if not os.path.exists(out_path):
                self._handle_fallback(out_path, target_region)

        if os.path.exists(osm_xml):
            os.remove(osm_xml)

        return entries

    def _fetch_osm(self, region):
        w, e, s, n = region
        bbox = f"{s},{w},{n},{e}"

        query = f"""
        [timeout:120][out:xml][bbox:{bbox}];
        (
          way["natural"="coastline"];
          relation["natural"="coastline"];
        );
        (._;>;);
        out meta;
        """
        params = urlencode({"data": query})
        url = f"{OSM_API}?{params}"
        dest = f"temp_osm_{w}_{s}.xml"
        f = Fetch(url)
        if f.fetch_file(dest, verbose=False) == 0:
            return dest

        return None

    def _is_land_by_topology(self, poly, lines_geom, buffer_size):
        """Determine if polygon is Land using the OSM Left-Hand Rule."""

        check_poly = poly.buffer(buffer_size * 2.0)

        if not check_poly.intersects(lines_geom):
            return None # Indeterminate

        if lines_geom.geom_type == "MultiLineString":
            geoms = list(lines_geom.geoms)
        else:
            geoms = [lines_geom]

        votes = []
        for line in geoms:
            if not check_poly.intersects(line):
                continue

            coords = list(line.coords)
            step = max(1, int(len(coords) / 5))

            for i in range(0, len(coords) - 1, step):
                p1 = coords[i]
                p2 = coords[i+1]
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]

                # Normal Vector pointing LEFT (-dy, dx)
                nx, ny = -dy, dx
                mag = math.sqrt(nx*nx + ny*ny)
                if mag == 0:
                    continue

                scale = buffer_size * 4.0
                test_pt = Point(p1[0] + dx*0.5 + (nx/mag)*scale,
                                p1[1] + dy*0.5 + (ny/mag)*scale)

                if poly.contains(test_pt):
                    votes.append(True)
                else:
                    votes.append(False)

        if not votes:
            return None

        return (sum(votes) / len(votes)) > 0.5

    def _is_land_by_gmrt(self, poly):
        """Fallback: Check GMRT elevation."""

        if not gmrt_fetch_point: return False
        try:
            pt = poly.centroid
            val = float(gmrt_fetch_point(latitude=pt.y, longitude=pt.x))
            return val >= 0
        except: return False

    def _handle_fallback(self, dst_file, region):
        """If OSM fails, guess whole tile based on center point."""

        w, e, s, n = region
        cx, cy = (w + e)/2, (s + n)/2
        is_land = False
        if gmrt_fetch_point:
            try: is_land = float(gmrt_fetch_point(latitude=cy, longitude=cx)) >= 0
            except: pass

        poly = box(w, s, e, n) if is_land else None
        self._write_geojson(dst_file, [poly] if poly else [])

    def _write_geojson(self, dst_file, polygons):
        schema = {"geometry": "Polygon", "properties": {"class": "str"}}
        with fiona.open(dst_file, "w", driver="GeoJSON", crs="EPSG:4326", schema=schema) as dst:
            for poly in polygons:
                dst.write({
                    "geometry": mapping(poly),
                    "properties": {"class": "land"}
                })

    def _polygonize(self, osm_file, dst_file, region):
        """The Hybrid Engine: OGR Reader -> Shapely Processor -> Fiona Writer"""

        ds = ogr.Open(osm_file)
        if not ds:
            self._handle_fallback(dst_file, region)
            return

        layer = ds.GetLayer(1) # Layer 1 is usually 'lines' in OSM driver
        if not layer:
            self._handle_fallback(dst_file, region)
            return

        lines = []
        for feat in layer:
            geom_ref = feat.GetGeometryRef()
            if geom_ref:
                shapely_geom = shapely.wkb.loads(bytes(geom_ref.ExportToWkb()))
                lines.append(shapely_geom)

        if not lines:
            self._handle_fallback(dst_file, region)
            return

        merged = linemerge(lines)
        coastline_geom = unary_union(merged)

        w, e, s, n = region
        region_box = box(w, s, e, n)

        cut_width = 1e-6
        cutters = coastline_geom.buffer(cut_width)

        try:
            split_geom = region_box.difference(cutters)
        except Exception:
            self._handle_fallback(dst_file, region)
            return

        land_polys = []
        polys = [split_geom] if split_geom.geom_type == "Polygon" else list(split_geom.geoms)

        for poly in polys:
            if poly.is_empty: continue

            is_land = self._is_land_by_topology(poly, coastline_geom, cut_width)

            if is_land is None:
                is_land = self._is_land_by_gmrt(poly)

            if is_land:
                land_polys.append(poly)

        self._write_geojson(dst_file, land_polys)
