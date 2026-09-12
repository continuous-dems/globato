#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.modules.osm_landmask
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Super-Module that generates a high-quality Coastline Mask.

Extract topological features from OSM and pieces together a coastline
binary (landmask) or topological vector.
"""

import os
import logging
import json
import math
import hashlib

import numpy as np
from pyogrio.raw import write
import shapely
from shapely.geometry import box, LineString, Point, Polygon
from shapely.ops import linemerge, unary_union, polygonize

from fetchez.modules import FetchModule
from fetchez.core import Fetch, urlencode, CUDEM_USER_AGENT
from fetchez.cli import cli_opts
from fetchez.utils import str2bool

try:
    from fetchez.modules.gmrt import gmrt_fetch_point
except ImportError:
    gmrt_fetch_point = None

logger = logging.getLogger(__name__)

OSM_API = "https://lz4.overpass-api.de/api/interpreter"

HEADERS = {
    "User-Agent": CUDEM_USER_AGENT,
    "Content-Type": "application/x-www-form-urlencoded",
}


@cli_opts(
    help_text="Generates a multi-class topological vector from OpenStreetMap.",
    include_water="Macro flag to includex both lakes and rivers.",
    include_rivers="If True, carves out rivers and estuaries (waterway=riverbank).",
    include_lakes="If True, carves out inland lakes and ponds (natural=water).",
    include_reefs="If True, returns reefs (natural=reef) as land.",
    include_wetlands="If True, carves out tidal flats, salt marshes, and estuaries.",
    include_breakwaters="If True, returns man-made breakwaters, piers, and groynes as land.",
    output_mode="binary (landmask) or topology (all classes included in output, includes all options).",
    min_area_sqm="Minimum area in square meters for a waterbody to be carved out.",
)
class OSMLandmaskModule(FetchModule):
    """Fetches OSM Coastline data and polygonizes it into a classified topological mask."""

    name = "osm_landmask"
    meta_desc = "OpenStreetMap Coastline and Waterbody Generator"
    meta_agency = "OSM"
    meta_category = "Reference"
    meta_tags = ["osm", "coastline", "water", "polygons", "globato"]

    def __init__(
        self,
        include_water=False,
        include_rivers=False,
        include_lakes=False,
        include_reefs=False,
        include_wetlands=False,
        include_breakwaters=False,
        include_estuaries=True,
        min_area_sqm=0,
        output_mode="binary",
        **kwargs,
    ):
        super().__init__(name="osm_landmask", **kwargs)

        self.include_water = str2bool(str(include_water))
        self.include_rivers = (
            str2bool(str(include_rivers))
            if include_rivers is not None
            else self.include_water
        )
        self.include_lakes = (
            str2bool(str(include_lakes))
            if include_lakes is not None
            else self.include_water
        )

        self.include_reefs = str2bool(str(include_reefs))
        self.include_wetlands = str2bool(str(include_wetlands))
        self.include_breakwaters = str2bool(str(include_breakwaters))
        self.include_estuaries = str2bool(str(include_estuaries))

        self.min_area_sqm = float(min_area_sqm)
        self.output_mode = str(output_mode).lower()
        self.headers = HEADERS

        if self.output_mode == "topology":
            self.include_rivers = True
            self.include_lakes = True
            self.include_breakwaters = True
            self.include_estuaries = True
            self.include_reefs = False
            self.include_wetlands = True

    def _generate_cache_key(self):
        region_str = self.wgs_region.format("fn") if self.wgs_region else "global"
        state = (
            f"{region_str}_{self.include_water}_{self.include_rivers}_"
            f"{self.include_lakes}_{self.include_reefs}_{self.include_wetlands}_"
            f"{self.include_breakwaters}_{self.include_estuaries}_"
            f"{self.min_area_sqm}_{self.output_mode}"
        )
        return hashlib.sha256(state.encode("utf-8")).hexdigest()

    def _get_area_sqm(self, poly):
        lat = poly.centroid.y
        deg_to_m_y = 111320.0
        deg_to_m_x = 111320.0 * math.cos(math.radians(lat))
        return poly.area * deg_to_m_x * deg_to_m_y

    def _get_filename_suffix(self):
        parts = []
        if self.include_rivers:
            parts.append("r")
        if self.include_lakes:
            parts.append("l")
        if self.include_wetlands:
            parts.append("w")
        if self.include_reefs:
            parts.append("rf")
        if self.include_breakwaters:
            parts.append("b")
        if self.include_estuaries:
            parts.append("e")
        parts.append(f"_{self.output_mode}")

        suffix = "".join(parts)
        if self.min_area_sqm > 0:
            suffix += f"_m{int(self.min_area_sqm)}"
        return f"_{suffix}" if suffix else ""

    def run(self):
        if not self.wgs_region:
            logger.error(f"[{self.name}] Requires a bounding box region to run.")
            return

        w, e, s, n = self.wgs_region
        suffix = self._get_filename_suffix()
        out_name = f"osm_landmask_{w}_{s}_{e}_{n}{suffix}.geojson"
        out_path = os.path.join(self._outdir, out_name)

        if os.path.exists(out_path):
            logger.info(f"[OSM] Using existing landmask: {out_path}")
            self.add_entry_to_results(f"file://{out_name}", out_path, "osm_landmask")
            return self

        logger.info(f"[OSM] Fetching coastline for {self.wgs_region}...")
        osm_xml = self._fetch_osm(self.wgs_region)

        if not osm_xml or os.path.getsize(osm_xml) < 100:
            self._handle_fallback(out_path, self.wgs_region)
        else:
            logger.info("[OSM] Polygonizing and classifying coastline...")
            try:
                self._polygonize(osm_xml, out_path, self.wgs_region)
                logger.info(f"[OSM] Generated multi-class topological mask: {out_path}")
            except Exception as e:
                logger.error(f"[OSM] Polygonization failed: {e}")
                if not os.path.exists(out_path):
                    self._handle_fallback(out_path, self.wgs_region)

        if osm_xml and os.path.exists(osm_xml):
            os.remove(osm_xml)

        if os.path.exists(out_path):
            self.add_entry_to_results(
                f"file://{os.path.basename(out_path)}", out_path, "osm_landmask"
            )

        return self

    def _fetch_osm(self, region):
        w, e, s, n = region
        bbox = f"{s},{w},{n},{e}"
        query = f"""
        [timeout:120][out:json][bbox:{bbox}];
        (
          way["natural"="coastline"];
          relation["natural"="coastline"];
        """

        if self.include_rivers:
            query += """
            way["waterway"="riverbank"];
            relation["waterway"="riverbank"];
            way["natural"="water"]["water"~"river"];
            relation["natural"="water"]["water"~"river"];
            """
        if self.include_estuaries:
            query += """
            way["natural"="water"]["water"~"estuary|bay"];
            relation["natural"="water"]["water"~"estuary|bay"];
            way["natural"="water"]["tidal"="yes"];
            relation["natural"="water"]["tidal"="yes"];
            way["natural"="bay"];
            relation["natural"="bay"];
            way["estuary"="yes"];
            relation["estuary"="yes"];
            """
        if self.include_lakes:
            query += """
            way["natural"="water"]["water"!~"river|estuary|bay"]["tidal"!="yes"];
            relation["natural"="water"]["water"!~"river|estuary|bay"]["tidal"!="yes"];
            """
        if self.include_reefs:
            query += """
            way["natural"="reef"];
            relation["natural"="reef"];
            """
        if self.include_wetlands:
            query += """
            way["natural"="wetland"];
            relation["natural"="wetland"];
            way["waterway"="tidal_channel"];
            relation["waterway"="tidal_channel"];
            way["natural"="mud"];
            relation["natural"="mud"];
            """
        if self.include_breakwaters:
            query += """
            way["man_made"~"breakwater|pier|groyne|jetty"];
            relation["man_made"~"breakwater|pier|groyne|jetty"];
            """

        query += """
        );
        (._;>;);
        out geom;
        """

        params = urlencode({"data": query})
        url = f"{OSM_API}?{params}"

        suffix = self._get_filename_suffix()
        dest = os.path.join(self._outdir, f"temp_osm_{w}_{s}_{e}_{w}{suffix}.json")
        f = Fetch(url, headers=HEADERS)
        if f.fetch_file(dest, method="GET", verbose=False) == 0:
            return dest
        return None

    def _is_land_by_topology(self, poly, lines_geom, buffer_size, threshold=0.5):
        check_poly = poly.buffer(buffer_size * 2.0)
        if not check_poly.intersects(lines_geom):
            return None

        local_lines = lines_geom.intersection(check_poly)
        if local_lines.geom_type in ["MultiLineString", "GeometryCollection"]:
            geoms = [geom for geom in local_lines.geoms if isinstance(geom, LineString)]
        elif local_lines.geom_type == "LineString":
            geoms = [local_lines]
        else:
            return None

        votes = []
        for line in geoms:
            coords = list(line.coords)
            step = max(1, int(len(coords) / 5))

            for i in range(0, len(coords) - 1, step):
                p1, p2 = coords[i], coords[i + 1]
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]

                nx, ny = -dy, dx
                mag = math.sqrt(nx * nx + ny * ny)
                if mag == 0:
                    continue

                scale = buffer_size * 4.0
                test_pt = Point(
                    p1[0] + dx * 0.5 + (nx / mag) * scale,
                    p1[1] + dy * 0.5 + (ny / mag) * scale,
                )
                votes.append(poly.contains(test_pt))

        if not votes:
            return None
        return (sum(votes) / len(votes)) >= threshold

    def _is_land_by_gmrt(self, poly):
        if not gmrt_fetch_point:
            return False
        try:
            pt = poly.centroid
            val = float(gmrt_fetch_point(latitude=pt.y, longitude=pt.x))
            return val >= 0
        except Exception:
            return False

    def _handle_fallback(self, dst_file, region):
        w, e, s, n = region
        cx, cy = (w + e) / 2, (s + n) / 2
        is_land = False
        if gmrt_fetch_point:
            try:
                is_land = float(gmrt_fetch_point(latitude=cy, longitude=cx)) >= 0
            except Exception:
                pass

        poly = box(w, s, e, n)
        features = []

        if is_land:
            features.append((poly, "land"))
        elif self.output_mode == "topology":
            features.append((poly, "ocean"))

        self._write_geojson_pyogrio(dst_file, features)

    def _write_geojson_pyogrio(self, dst_file, features):
        if not features:
            # Write a valid but empty GeoJSON so downstream tools don't crash
            with open(dst_file, "w") as f:
                f.write('{"type": "FeatureCollection", "features": []}')
                return

        polygons = [f[0] for f in features]
        classes = [f[1] for f in features]

        geometry_wkb = shapely.to_wkb(polygons)
        field_data = [np.array(classes, dtype=object)]

        write(
            dst_file,
            geometry_wkb,
            field_data,
            fields=["class"],
            geometry_type="Polygon",
            crs="EPSG:4326",
            driver="GeoJSON",
        )

    def _polygonize(self, osm_file, dst_file, region):
        coast_lines = []

        # Segmented Water Topology
        water_polys, water_lines = [], []
        river_polys, river_lines = [], []
        estuary_polys, estuary_lines = [], []
        lake_polys, lake_lines = [], []
        wetland_polys, wetland_lines = [], []

        island_polys, island_lines = [], []
        reef_polys, breakwater_polys = [], []

        try:
            with open(osm_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            coast_relation_member_ids = set()
            water_relation_member_ids = set()
            relations, ways = [], []

            for element in data.get("elements", []):
                if element.get("type") == "relation":
                    relations.append(element)
                    tags = element.get("tags", {})
                    is_coast = tags.get("natural") == "coastline"
                    is_river = (
                        tags.get("waterway") == "riverbank"
                        or tags.get("water") == "river"
                    )
                    is_lake = tags.get("natural") == "water" and not is_river
                    is_estuary = (
                        tags.get("water") in ["estuary", "bay"]
                        or tags.get("natural") == "bay"
                        or tags.get("estuary") == "yes"
                        or tags.get("tidal") == "yes"
                    )
                    is_wetland = tags.get("waterway") == "tidal_channel" or tags.get(
                        "natural"
                    ) in ["mud", "wetland"]
                    is_water = (
                        (self.include_rivers and is_river)
                        or (self.include_lakes and is_lake)
                        or (self.include_wetlands and is_wetland)
                        or (self.include_estuaries and is_estuary)
                    )

                    for member in element.get("members", []):
                        if member.get("type") == "way":
                            if is_coast:
                                coast_relation_member_ids.add(member.get("ref"))
                            if is_water:
                                water_relation_member_ids.add(member.get("ref"))
                elif element.get("type") == "way":
                    ways.append(element)

            for rel in relations:
                tags = rel.get("tags", {})
                is_coast = tags.get("natural") == "coastline"
                is_breakwater = self.include_breakwaters and tags.get("man_made") in [
                    "breakwater",
                    "pier",
                    "groyne",
                    "jetty",
                ]

                is_estuary = (
                    tags.get("water") in ["estuary", "bay"]
                    or tags.get("natural") == "bay"
                    or tags.get("estuary") == "yes"
                    or tags.get("tidal") == "yes"
                )

                is_river = (
                    tags.get("waterway") == "riverbank" or tags.get("water") == "river"
                ) and not is_estuary
                is_lake = (
                    tags.get("natural") == "water" and not is_river and not is_estuary
                )
                is_wetland = tags.get("waterway") == "tidal_channel" or tags.get(
                    "natural"
                ) in ["mud", "wetland"]

                is_water = False
                if self.include_rivers and is_river:
                    is_water = True
                if self.include_lakes and is_lake:
                    is_water = True
                if self.include_wetlands and is_wetland:
                    is_water = True
                if self.include_estuaries and is_estuary:
                    is_water = True

                if not is_coast and not is_water and not is_breakwater:
                    continue

                target_polys = (
                    estuary_polys
                    if is_estuary
                    else river_polys
                    if is_river
                    else lake_polys
                    if is_lake
                    else wetland_polys
                    if is_wetland
                    else water_polys
                )
                target_lines = (
                    estuary_lines
                    if is_estuary
                    else river_lines
                    if is_river
                    else lake_lines
                    if is_lake
                    else wetland_lines
                    if is_wetland
                    else water_lines
                )

                for member in rel.get("members", []):
                    if member.get("type") == "way" and "geometry" in member:
                        coords = [(pt["lon"], pt["lat"]) for pt in member["geometry"]]
                        if len(coords) < 2:
                            continue

                        line = LineString(coords)
                        role = member.get("role", "")

                        if is_coast:
                            coast_lines.append(line)
                        elif is_water:
                            if role == "inner":
                                if line.is_closed and len(coords) >= 4:
                                    island_polys.append(Polygon(coords))
                                else:
                                    island_lines.append(line)
                            else:
                                if line.is_closed and len(coords) >= 4:
                                    target_polys.append(Polygon(coords))
                                else:
                                    target_lines.append(line)

            for way in ways:
                way_id = way.get("id")
                tags = way.get("tags", {})
                is_coast = tags.get("natural") == "coastline"
                is_reef = self.include_reefs and tags.get("natural") == "reef"
                is_breakwater = self.include_breakwaters and tags.get("man_made") in [
                    "breakwater",
                    "pier",
                    "groyne",
                    "jetty",
                ]
                is_estuary = (
                    tags.get("water") in ["estuary", "bay"]
                    or tags.get("natural") == "bay"
                    or tags.get("estuary") == "yes"
                    or tags.get("tidal") == "yes"
                )

                is_river = (
                    tags.get("waterway") == "riverbank" or tags.get("water") == "river"
                ) and not is_estuary
                is_lake = (
                    tags.get("natural") == "water" and not is_river and not is_estuary
                )
                is_wetland = tags.get("waterway") == "tidal_channel" or tags.get(
                    "natural"
                ) in ["mud", "wetland"]

                is_water = False
                if self.include_water and (
                    tags.get("natural") == "water"
                    or tags.get("waterway") == "riverbank"
                ):
                    is_water = True
                if self.include_rivers and is_river:
                    is_water = True
                if self.include_lakes and is_lake:
                    is_water = True
                if self.include_wetlands and is_wetland:
                    is_water = True
                if self.include_estuaries and is_estuary:
                    is_water = True

                if not is_coast and not is_water and not is_reef and not is_breakwater:
                    continue

                process_as_coast = is_coast and way_id not in coast_relation_member_ids
                process_as_water = is_water and way_id not in water_relation_member_ids

                if not any(
                    [process_as_coast, process_as_water, is_reef, is_breakwater]
                ):
                    continue

                if "geometry" in way:
                    coords = [(pt["lon"], pt["lat"]) for pt in way["geometry"]]
                    if len(coords) < 2:
                        continue
                    line = LineString(coords)

                    if process_as_coast:
                        coast_lines.append(line)
                    elif process_as_water:
                        target_polys = (
                            estuary_polys
                            if is_estuary
                            else river_polys
                            if is_river
                            else lake_polys
                            if is_lake
                            else wetland_polys
                            if is_wetland
                            else water_polys
                        )
                        target_lines = (
                            estuary_lines
                            if is_estuary
                            else river_lines
                            if is_river
                            else lake_lines
                            if is_lake
                            else wetland_lines
                            if is_wetland
                            else water_lines
                        )

                        if line.is_closed and len(coords) >= 4:
                            target_polys.append(Polygon(coords))
                        else:
                            target_lines.append(line)
                    elif is_reef:
                        if line.is_closed and len(coords) >= 4:
                            reef_polys.append(Polygon(coords))
                    elif is_breakwater:
                        if line.is_closed and len(coords) >= 4:
                            breakwater_polys.append(Polygon(coords))

        except Exception as e:
            logger.error(f"Failed to parse OSM JSON natively: {e}")
            self._handle_fallback(dst_file, region)
            return

        # Stitch fragmented unclosed boundaries for all categories
        for lines, polys in [
            (water_lines, water_polys),
            (river_lines, river_polys),
            (lake_lines, lake_polys),
            (wetland_lines, wetland_polys),
            (island_lines, island_polys),
            (estuary_lines, estuary_polys),
        ]:
            if lines:
                polys.extend(list(polygonize(linemerge(lines))))

        west, east, south, north = region
        region_box = box(west, south, east, north)
        land_polys, ocean_polys = [], []

        if not coast_lines:
            is_land = self._is_land_by_gmrt(region_box)
            if is_land:
                land_polys.append(region_box)
            else:
                ocean_polys.append(region_box)
        else:
            merged_coast = linemerge(coast_lines)
            coastline_geom = unary_union(merged_coast)
            cut_width = 1e-6
            cutters = coastline_geom.buffer(cut_width)

            try:
                split_geom = region_box.difference(cutters)
            except Exception:
                self._handle_fallback(dst_file, region)
                return

            polys = (
                [split_geom]
                if split_geom.geom_type == "Polygon"
                else list(split_geom.geoms)
            )

            for poly in polys:
                if poly.is_empty:
                    continue
                is_land = self._is_land_by_topology(
                    poly, coastline_geom, cut_width, threshold=0.5
                )
                if is_land is None:
                    is_land = self._is_land_by_gmrt(poly)

                if is_land:
                    land_polys.append(poly)
                else:
                    ocean_polys.append(poly)

        # --- Island / Water Subtraction ---
        def subtract_geom(polys, sub_geom):
            out = []
            for p in polys:
                diff = p.difference(sub_geom)
                if diff.is_empty:
                    continue
                if diff.geom_type == "Polygon":
                    out.append(diff)
                elif diff.geom_type == "MultiPolygon":
                    out.extend(list(diff.geoms))
            return out

        if self.min_area_sqm > 0:
            water_polys = [
                p for p in water_polys if self._get_area_sqm(p) >= self.min_area_sqm
            ]
            river_polys = [
                p for p in river_polys if self._get_area_sqm(p) >= self.min_area_sqm
            ]
            lake_polys = [
                p for p in lake_polys if self._get_area_sqm(p) >= self.min_area_sqm
            ]
            wetland_polys = [
                p for p in wetland_polys if self._get_area_sqm(p) >= self.min_area_sqm
            ]
            island_polys = [
                p for p in island_polys if self._get_area_sqm(p) >= self.min_area_sqm
            ]
            estuary_polys = [
                p for p in estuary_polys if self._get_area_sqm(p) >= self.min_area_sqm
            ]

        if island_polys:
            unified_islands = unary_union([p for p in island_polys if p.is_valid])
            water_polys = subtract_geom(water_polys, unified_islands)
            river_polys = subtract_geom(river_polys, unified_islands)
            lake_polys = subtract_geom(lake_polys, unified_islands)
            wetland_polys = subtract_geom(wetland_polys, unified_islands)
            estuary_polys = subtract_geom(estuary_polys, unified_islands)

        all_water = (
            water_polys + river_polys + lake_polys + wetland_polys + estuary_polys
        )
        if all_water:
            unified_water = unary_union(
                [p.buffer(0) for p in all_water if p.buffer(0).is_valid]
            )
            land_polys = subtract_geom(land_polys, unified_water)

        marine_structures = reef_polys + breakwater_polys
        if marine_structures:
            logger.info(
                f"[OSM] Carving {len(marine_structures)} marine structures out of the ocean..."
            )
            unified_marine = unary_union(
                [p.buffer(0) for p in marine_structures if p.buffer(0).is_valid]
            )
            ocean_polys = subtract_geom(ocean_polys, unified_marine)

        # Enforce hole removal on land
        if self.min_area_sqm > 0:
            cleaned_land = []
            for poly in land_polys:
                valid_interiors = [
                    ring
                    for ring in poly.interiors
                    if self._get_area_sqm(Polygon(ring)) >= self.min_area_sqm
                ]
                cleaned_land.append(Polygon(poly.exterior, valid_interiors))
            land_polys = cleaned_land

        # Package it all up for Pyogrio!
        features = []
        for p in land_polys:
            features.append((p, "land"))
        for p in reef_polys:
            features.append((p, "reef"))
        for p in breakwater_polys:
            features.append((p, "breakwater"))

        if self.output_mode == "topology":
            for p in ocean_polys:
                features.append((p, "ocean"))
            for p in river_polys:
                features.append((p, "river"))
            for p in lake_polys:
                features.append((p, "lake"))
            for p in wetland_polys:
                features.append((p, "wetland"))
            for p in estuary_polys:
                features.append((p, "estuary"))
            for p in water_polys:
                features.append((p, "water"))

        self._write_geojson_pyogrio(dst_file, features)
