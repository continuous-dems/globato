#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.hooks.osm_landmask
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fetches OSM Coastline data and polygonizes it into a landmask.
Ported from CUDEM/OSM.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
import numpy as np
from fetchez.hooks import FetchHook
from fetchez import utils
from fetchez.core import Fetch, urlencode

try:
    from osgeo import ogr

    HAS_OGR = True
except ImportError as e:
    HAS_OGR = False

logger = logging.getLogger(__name__)

OSM_API = "https://lz4.overpass-api.de/api/interpreter"

# TODO: This should be a module instead of a hook (like transformez)
class OSMLandmask(FetchHook):
    """Generates a Land/Water mask vector from OpenStreetMap.

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
        if not osm_xml:
            return entries

        logger.info("[OSM] Polygonizing coastline...")
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

    def _polygonize(self, osm_file, dst_file, region):
        """Polygonizer.

        1. Reads OSM Lines.
        2. Creates Region Box.
        3. Differences Region - Lines -> Polygons.
        4. Classifies Polygons (Ray casting).
        """

        ds = ogr.Open(osm_file)
        if not ds: raise IOError("Cannot open OSM XML")
        layer = ds.GetLayer(1) # 'lines'

        multi_line = ogr.Geometry(ogr.wkbMultiLineString)
        for feat in layer:
            geom = feat.GetGeometryRef()
            if geom: multi_line.AddGeometry(geom)

        if multi_line.IsEmpty():
            # todo: use gmrt to check the center of the region to see if we are neg or pos.
            self._write_box(dst_file, region, is_land=True)
            return

        # Create Region Polygon
        w, e, s, n = region
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(w, n)
        ring.AddPoint(e, n)
        ring.AddPoint(e, s)
        ring.AddPoint(w, s)
        ring.AddPoint(w, n)
        box = ogr.Geometry(ogr.wkbPolygon)
        box.AddGeometry(ring)

        cutters = multi_line.Buffer(0.000001)

        try:
            polys = box.Difference(cutters)
        except Exception as e:
            logger.warning("[OSM] Topological error in difference. {e}")
            return

        drv = ogr.GetDriverByName("GPKG")
        dst_ds = drv.CreateDataSource(dst_file)
        dst_lyr = dst_ds.CreateLayer("landmask", geom_type=ogr.wkbPolygon)

        # ... todo: move _determine_polygon_side from cudem ...

        for i in range(polys.GetGeometryCount()):
            poly = polys.GetGeometryRef(i)
            feat = ogr.Feature(dst_lyr.GetLayerDefn())
            feat.SetGeometry(poly)
            dst_lyr.CreateFeature(feat)

        dst_ds = None

    def _write_box(self, dst_file, region, is_land=True):
        """Writes the full bbox as a single polygon."""

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
