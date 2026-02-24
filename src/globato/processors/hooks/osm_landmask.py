#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.hooks.osm_landmask
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fetches OSM Coastline data and polygonizes it into a landmask.
Includes logic to classify Land vs Water using OSM winding rules
and GMRT elevation fallback.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
import math
import numpy as np
from fetchez.hooks import FetchHook
from fetchez.core import Fetch, urlencode
from fetchez import utils

try:
    from fetchez.modules.gmrt import gmrt_fetch_point
except ImportError:
    gmrt_fetch_point = None

try:
    from osgeo import ogr
    HAS_OGR = True
except ImportError:
    HAS_OGR = False

logger = logging.getLogger(__name__)

OSM_API = "https://lz4.overpass-api.de/api/interpreter"


class OSMLandmask(FetchHook):
    """Generates a Land/Water mask vector from OpenStreetMap.

    Logic:
    1. Fetches 'natural=coastline' ways from Overpass API.
    2. Differences the Region Box with the Coastlines.
    3. Uses Ray Casting (Land is on Left of OSM line) to classify resulting polygons.
    4. Fallback: If no coastlines, queries GMRT at center point to determine Land/Water.

    Usage:
        --hook osm_landmask:filename=landmask.gpkg
    """

    name = "osm_landmask"
    stage = "pre"
    category = "generator"

    def __init__(self, filename="landmask.gpkg", **kwargs):
        super().__init__(**kwargs)
        self.filename = filename

    def run(self, entries):
        regions = [getattr(mod, "region", None) for mod, _ in entries]
        valid_regions = [r for r in regions if r]

        if not valid_regions:
            return entries

        # Union of all requested regions
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
        if not osm_xml:
            return entries

        logger.info("[OSM] Polygonizing and classifying coastline...")
        try:
            self._polygonize(osm_xml, out_path, target_region)
            logger.info(f"[OSM] Generated landmask: {out_path}")
        except Exception as e:
            logger.error(f"[OSM] Polygonization failed: {e}")

        if os.path.exists(osm_xml):
            os.remove(osm_xml)

        return entries

    def _fetch_osm(self, region):
        """Fetch raw OSM XML via Overpass."""

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

        params = urlencode({'data': query})
        url = f"{OSM_API}?{params}"
        dest = f"temp_osm_{w}_{s}.xml"

        f = Fetch(url)
        if f.fetch_file(dest, verbose=False) == 0:
            return dest
        return None

    def _determine_side(self, poly_geom, lines_geom):
        """Determine if a polygon is Land or Water.
        OSM Rule: Land is always on the LEFT side of the coastline way.
        """

        if not lines_geom or lines_geom.IsEmpty():
            return False

        if not poly_geom.Intersects(lines_geom.Buffer(1e-7)):
            return False

        if lines_geom.GetGeometryType() == ogr.wkbMultiLineString:
            geoms = [lines_geom.GetGeometryRef(i) for i in range(lines_geom.GetGeometryCount())]
        else:
            geoms = [lines_geom]

        votes = []

        for line in geoms:
            if not line.Intersects(poly_geom):
                continue

            pts = line.GetPointCount()
            step = max(1, int(pts / 5))

            for i in range(0, pts - 1, step):
                p1 = line.GetPoint(i)
                p2 = line.GetPoint(i+1)

                # Vector P1 -> P2
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]

                # Midpoint
                mx = p1[0] + dx * 0.5
                my = p1[1] + dy * 0.5

                # Normal Vector pointing LEFT (-dy, dx)
                # This points to the "Land" side according to OSM rules
                nx = -dy
                ny = dx

                # Scale normal to be very small
                mag = math.sqrt(nx*nx + ny*ny)
                if mag == 0: continue

                scale = 1e-5
                test_x = mx + (nx / mag) * scale
                test_y = my + (ny / mag) * scale

                # Check if this "Left" point is inside the polygon
                test_pt = ogr.Geometry(ogr.wkbPoint)
                test_pt.AddPoint(test_x, test_y)

                if poly_geom.Contains(test_pt):
                    votes.append(True) # Left is Inside -> Polygon is Land
                else:
                    votes.append(False) # Left is Outside -> Polygon is Water

        if not votes:
            return False

        return (sum(votes) / len(votes)) > 0.5

    def _verify_gmrt(self, region):
        """Fallback: Check GMRT elevation at the center of the region."""

        if not gmrt_fetch_point:
            logger.warning("[OSM] GMRT module missing. Assuming Water.")
            return False

        w, e, s, n = region
        cx = (w + e) / 2
        cy = (s + n) / 2

        try:
            val_str = gmrt_fetch_point(latitude=cy, longitude=cx)
            if val_str:
                val = float(val_str)
                is_land = val >= 0
                logger.info(f"[OSM] GMRT Fallback at ({cx:.4f}, {cy:.4f}): Z={val} -> {'Land' if is_land else 'Water'}")
                return is_land
        except Exception as e:
            logger.warning(f"[OSM] GMRT check failed: {e}")

        return False

    def _create_empty_gpkg(self, dst_file):
        """Create a valid but empty GPKG (implies full water)."""

        drv = ogr.GetDriverByName("GPKG")
        ds = drv.CreateDataSource(dst_file)
        lyr = ds.CreateLayer("landmask", geom_type=ogr.wkbPolygon)
        ds = None

    def _polygonize(self, osm_file, dst_file, region):
        """Convert OSM lines to classified Land Polygons."""

        ds = ogr.Open(osm_file)
        if not ds: raise IOError("Cannot open OSM XML")
        layer = ds.GetLayer(1) # 'lines'

        # Collect all coastline lines
        multi_line = ogr.Geometry(ogr.wkbMultiLineString)
        has_lines = False
        for feat in layer:
            geom = feat.GetGeometryRef()
            if geom:
                multi_line.AddGeometry(geom)
                has_lines = True

        # No Coastline Found (All Land or All Water)
        if not has_lines:
            is_land = self._verify_gmrt(region)
            if is_land:
                self._write_box(dst_file, region) # Full Land
            else:
                self._create_empty_gpkg(dst_file) # Full Water
            return

        w, e, s, n = region
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(w, n); ring.AddPoint(e, n); ring.AddPoint(e, s); ring.AddPoint(w, s); ring.AddPoint(w, n)
        box = ogr.Geometry(ogr.wkbPolygon)
        box.AddGeometry(ring)

        # Difference: Region - Coastlines
        cutters = multi_line.Buffer(1e-9)
        try:
            polys = box.Difference(cutters)
        except Exception:
            logger.warning("[OSM] Topological error. Falling back to GMRT check.")
            is_land = self._verify_gmrt(region)
            if is_land: self._write_box(dst_file, region)
            else: self._create_empty_gpkg(dst_file)
            return

        drv = ogr.GetDriverByName("GPKG")
        dst_ds = drv.CreateDataSource(dst_file)
        dst_lyr = dst_ds.CreateLayer("landmask", geom_type=ogr.wkbPolygon)

        # Classify and Save Land Polygons
        if polys.GetGeometryType() == ogr.wkbPolygon:
            # Single polygon result
            if self._determine_side(polys, multi_line):
                feat = ogr.Feature(dst_lyr.GetLayerDefn())
                feat.SetGeometry(polys)
                dst_lyr.CreateFeature(feat)
        else:
            # MultiPolygon result
            for i in range(polys.GetGeometryCount()):
                poly = polys.GetGeometryRef(i)
                # ONLY write polygons determined to be Land
                if self._determine_side(poly, multi_line):
                    feat = ogr.Feature(dst_lyr.GetLayerDefn())
                    feat.SetGeometry(poly)
                    dst_lyr.CreateFeature(feat)

        dst_ds = None

    def _write_box(self, dst_file, region):
        """Writes the full bbox as a single polygon (Full Land)."""

        w, e, s, n = region
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(w, n); ring.AddPoint(e, n); ring.AddPoint(e, s); ring.AddPoint(w, s); ring.AddPoint(w, n)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

        drv = ogr.GetDriverByName("GPKG")
        ds = drv.CreateDataSource(dst_file)
        lyr = ds.CreateLayer("landmask", geom_type=ogr.wkbPolygon)
        feat = ogr.Feature(lyr.GetLayerDefn())
        feat.SetGeometry(poly)
        lyr.CreateFeature(feat)
        ds = None
