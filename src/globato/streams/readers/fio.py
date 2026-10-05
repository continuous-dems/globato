#!/usr/bin/env python

"""
globato.streams.readers.fio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fiona/Shapely based Vector Reader (Shapefile, GeoPackage, S-57).

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging

import numpy as np
from fetchez.utils import float_or

from .base import BaseGlobatoReader

try:
    import fiona
    from shapely.geometry import shape

    HAS_FIONA = True
except ImportError:
    HAS_FIONA = False

logger = logging.getLogger(__name__)
logging.getLogger("fiona").setLevel(logging.ERROR)


class FionaReader(BaseGlobatoReader):
    """Streaming Fiona Vector Parser for extracting 3D points/vertices."""

    name = "fiona-point-reader"
    meta_category = "point-stream"
    meta_dtype = "fio-vector"
    meta_desc = "Read vector data through fiona into a point stream"
    meta_extensions = ["shp", "000", "json", "geojson", "kml", "gdb"]

    KNOWN_LAYERS = [
        "SOUNDG",
        "SurveyPoint_HD",
        "SurveyPoint",
        "Mass_Point",
        "Spot_Elevation",
    ]
    KNOWN_Z_FIELDS = [
        "VALSOU",
        "Elevation",
        "elev",
        "z",
        "depth",
        "height",
        "value",
        "Z_use",
        "Z_depth",
    ]

    def __init__(
        self,
        path,
        layer=None,
        z_field=None,
        weight_field=None,
        unc_field=None,
        z_scale=1.0,
        elevation_value=None,
        chunk_size=50000,
        **kwargs,
    ):
        super().__init__(path, **kwargs)
        self.src_fn = path
        # this is very specific to EHydro GDB. We need to generalize this.
        if self.src_fn.lower().endswith(".zip"):
            import zipfile

            internal_target = ""
            try:
                # Peek inside the zip to find the exact .gdb folder!
                with zipfile.ZipFile(self.src_fn, "r") as z:
                    for name in z.namelist():
                        # Look for the Geodatabase folder
                        if name.lower().endswith(".gdb/") or name.lower().endswith(
                            ".gdb"
                        ):
                            internal_target = f"!{name.strip('/')}"
                            break
                        # (Optional fallback) If it's a zipped shapefile instead
                        elif name.lower().endswith(".shp"):
                            internal_target = f"!{name}"

            except Exception as e:
                logger.debug(f"Could not inspect zip file: {e}")

            self.src_fn = f"zip://{self.src_fn}{internal_target}"
        self.target_layer = layer
        self.z_field = z_field
        self.weight_field = weight_field
        self.unc_field = unc_field
        self.z_scale = float_or(z_scale, 1.0)
        self.elevation_value = float_or(elevation_value)
        self.chunk_size = chunk_size

    def get_srs(self):
        """Extracts the Coordinate Reference System (SRS) from the vector file."""

        if not HAS_FIONA:
            return None

        try:
            layer_name = self._resolve_layer()
            with fiona.open(self.src_fn, "r", layer=layer_name) as src:
                if src.crs_wkt:
                    return src.crs_wkt
                elif src.crs and "init" in src.crs:
                    return src.crs["init"].upper()
            return None
        except Exception as e:
            logger.debug(f"Could not extract SRS from {self.src_fn}: {e}")
            return None

    def _resolve_layer(self):
        layers = fiona.listlayers(self.src_fn)
        if self.target_layer and self.target_layer in layers:
            return self.target_layer

        for name in self.KNOWN_LAYERS:
            if name in layers:
                logger.debug(f"Auto-detected vector layer: {name}")
                return name
        return layers[0]

    def _resolve_z_field(self, properties):
        if self.z_field:
            return self.z_field

        for f in self.KNOWN_Z_FIELDS:
            if f in properties:
                return f
        return None

    def _extract_vertices(self, geom):
        """Recursively flattens Shapely geometries into a list of (x,y,[z]) tuples."""

        if geom.is_empty:
            return []
        g_type = geom.geom_type

        if g_type == "Point":
            return [geom.coords[0]]
        elif g_type in ("LineString", "LinearRing"):
            return list(geom.coords)
        elif g_type == "Polygon":
            return list(geom.exterior.coords) + [
                c for r in geom.interiors for c in r.coords
            ]
        elif g_type.startswith("Multi") or g_type == "GeometryCollection":
            pts = []
            for part in geom.geoms:
                pts.extend(self._extract_vertices(part))

            return pts
        return []

    def _yield_raw_chunks(self):
        if not HAS_FIONA:
            logger.error(
                "Fiona/Shapely required for vector processing with fiona; otherwise use the pyogrio reader."
            )
            return

        try:
            layer_name = self._resolve_layer()

            with fiona.open(self.src_fn, "r", layer=layer_name) as src:
                z_attr_name = self._resolve_z_field(src.schema["properties"])

                cx, cy, cz, cw, cu = [], [], [], [], []
                count = 0

                for feat in src:
                    if not feat.geometry:
                        continue
                    geom = shape(feat.geometry)
                    props = feat.properties

                    w_val = (
                        float_or(props.get(self.weight_field), 1.0)
                        if self.weight_field
                        else 1.0
                    )
                    u_val = (
                        float_or(props.get(self.unc_field), 0.0)
                        if self.unc_field
                        else 0.0
                    )
                    z_attr = (
                        props.get(z_attr_name) if z_attr_name else self.elevation_value
                    )

                    for pt in self._extract_vertices(geom):
                        z_val = float_or(pt[2]) if len(pt) > 2 else float_or(z_attr)
                        if z_val is None:
                            continue

                        cx.append(pt[0])
                        cy.append(pt[1])
                        cz.append(z_val * self.z_scale)
                        cw.append(w_val)
                        cu.append(u_val)
                        count += 1

                        if count >= self.chunk_size:
                            yield self._pack(cx, cy, cz, cw, cu)
                            cx, cy, cz, cw, cu = [], [], [], [], []
                            count = 0

                if count > 0:
                    yield self._pack(cx, cy, cz, cw, cu)

        except Exception as e:
            logger.error(f"Fiona processing failed for {self.src_fn}: {e}")

    def _pack(self, x, y, z, w, u):
        dt = [("x", "f8"), ("y", "f8"), ("z", "f4"), ("w", "f4"), ("u", "f4")]
        data = [np.array(x), np.array(y), np.array(z), np.array(w), np.array(u)]
        return np.rec.fromarrays(data, dtype=dt)
