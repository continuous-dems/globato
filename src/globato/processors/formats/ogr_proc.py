#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.formats.ogr_proc
~~~~~~~~~~~~~

OGR data parsing from cudem

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
import numpy as np

try:
    from osgeo import ogr, osr
    HAS_OGR = True
except ImportError:
    HAS_OGR = False

from fetchez.hooks import FetchHook
from fetchez.utils import int_or, float_or

logger = logging.getLogger(__name__)


class OGRReader:
    """Providing an OGR 3D point dataset parser.

    Useful for data such as S-57, ENC, E-Hydro, Shapefiles, etc.
    """

    _known_layer_names = ["SOUNDG", "SurveyPoint_HD", "SurveyPoint", "Mass_Point", "Spot_Elevation"]
    _known_elev_fields = ["Elevation", "elev", "z", "height", "depth", "val", "value",
                          "topography", "surveyPointElev", "Z_depth", "Z_height"]

    def __init__(
            self,
            src_fn: str,
            ogr_layer=None,
            elev_field=None,
            weight_field=None,
            uncertainty_field=None,
            z_scale=None,
            elevation_value=None,
            **kwargs,
    ):

        self.src_fn = src_fn

        self.ogr_layer = ogr_layer
        self.elev_field = elev_field
        self.weight_field = weight_field
        self.uncertainty_field = uncertainty_field

        self.z_scale = float_or(z_scale)
        self.elevation_value = float_or(elevation_value)

    def _get_layer(self, ds):
        """Internal helper to resolve the OGR Layer to process."""

        layer = None

        ## By Index/Name from Config
        if self.ogr_layer is not None:
            layer = ds.GetLayer(self.ogr_layer)

        ## Auto-detect Known Names
        if layer is None:
            for lname in self._known_layer_names:
                layer = ds.GetLayerByName(lname)
                if layer:
                    break

        ## Default to first layer
        if layer is None:
            layer = ds.GetLayer(0)

        return layer

    def _resolve_fields(self, layer_defn):
        """Internal helper to auto-detect field names if not provided."""

        field_count = layer_defn.GetFieldCount()
        field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(field_count)]

        ## Elevation
        if self.elev_field is None:
            for f in self._known_elev_fields:
                if f in field_names:
                    self.elev_field = f
                    break

        # Weight (No auto-detect)
        # Uncertainty (No auto-detect)

    def yield_chunks(self):
        """Yield points from the OGR datasource."""

        if self.src_fn is None:
            return

        try:
            ds_ogr = ogr.Open(self.src_fn)
            if ds_ogr is None:
                logger.warning(f"Could not open OGR file: {self.fn}")
                return

            layer = self._get_layer(ds_ogr)
            if layer is None:
                return

            # Auto-detect fields
            self._resolve_fields(layer.GetLayerDefn())

            # --- Spatial Filter ---
            # if self.region is not None:
            #     ## Determine the Check Region (User ROI)
            #     ## Use trans_region if available (often handles pre-calc transforms), else user region
            #     check_region = self.transform['trans_region'] if self.transform['trans_region'] else self.region

            #     if check_region:
            #         ## Get Layer Native SRS
            #         layer_srs = layer.GetSpatialRef()

            #         ## Create Filter Geometry
            #         filter_geom = ogr.CreateGeometryFromWkt(check_region.export_as_wkt())

            #         ## Reproject Filter to Layer SRS if needed
            #         if layer_srs:
            #             ## Determine SRS of the Check Region
            #             filter_srs_str = check_region.src_srs if check_region.src_srs else 'epsg:4326'

            #             filter_srs = srsfun.osr_srs(filter_srs_str)

            #             if filter_srs and not layer_srs.IsSame(filter_srs):
            #                 try:
            #                     ## Create Transform: Region SRS -> Layer SRS
            #                     transform = osr.CoordinateTransformation(filter_srs, layer_srs)
            #                     filter_geom.Transform(transform)
            #                 except Exception as e:
            #                     ## If transform fails, warn but proceed (might filter incorrectly or empty)
            #                     if self.verbose:
            #                         logger.error(f"Failed to warp spatial filter to layer SRS: {e}")

            #         ## Apply Filter
            #         layer.SetSpatialFilter(filter_geom)

            # --- Buffer Setup for Chunking ---
            chunk_x = []
            chunk_y = []
            chunk_z = []
            chunk_w = []
            chunk_u = []
            chunk_size = 0
            max_chunk = 100000

            def flush_chunk():
                if chunk_size == 0:
                    return None

                dataset = np.column_stack((chunk_x, chunk_y, chunk_z, chunk_w, chunk_u))
                rec_arr = np.rec.fromrecords(dataset, names='x, y, z, w, u')

                ## Apply Z Scale
                if self.z_scale is not None:
                    rec_arr['z'] *= self.z_scale

                return rec_arr

            ## --- Feature Iteration ---
            for feature in layer:
                geom = feature.GetGeometryRef()
                if geom is None:
                    continue

                ## Get Weight/Uncertainty (Per Feature)
                w_val = 1.0
                if self.weight_field:
                    w_val = float_or(feature.GetField(self.weight_field), 1.0)

                u_val = 0.0
                if self.uncertainty_field:
                    u_val = float_or(feature.GetField(self.uncertainty_field), 0.0)

                pts = []

                if geom.GetGeometryCount() == 0:
                    pts = geom.GetPoints()
                else:
                    for i in range(geom.GetGeometryCount()):
                        sub_geom = geom.GetGeometryRef(i)
                        pts.extend(sub_geom.GetPoints())

                if not pts:
                    continue

                is_3d = geom.Is3D()
                for pt in pts:
                    x, y = pt[0], pt[1]
                    z = None

                    if is_3d and len(pt) > 2:
                        z = pt[2]

                    if z is None and self.elev_field:
                        z = float_or(feature.GetField(self.elev_field))

                    if z is None and self.elevation_value is not None:
                        z = self.elevation_value

                    if z is None:
                        continue

                    chunk_x.append(x)
                    chunk_y.append(y)
                    chunk_z.append(z)
                    chunk_w.append(w_val)
                    chunk_u.append(u_val)
                    chunk_size += 1

                    if chunk_size >= max_chunk:
                        yield flush_chunk()
                        chunk_x, chunk_y, chunk_z, chunk_w, chunk_u = [], [], [], [], []
                        chunk_size = 0

            if chunk_size > 0:
                yield flush_chunk()

            ds_ogr = None

        except Exception as e:
            logger.error(f"OGR processing failed for {self.src_fn}: {e}")
            return None


class OGRStream(FetchHook):
    """Convert Vector Data (S-57, Shapefile, GDB) to XYZ points.

    Auto-detects bathymetry layers (SOUNDG) and Z-fields (VALSOU).

    Usage:
      --hook ogr_to_xyz (Auto S-57)
      --hook ogr_to_xyz:layer=SurveyPoint,z_field=depth (eHydro)
    """

    name = "ogr_stream"
    stage = "file"
    category = "format-stream"

    def __init__(self, layer=None, z_field=None,
                 weight_field=None, unc_field=None,
                 z_scale=1, keep_raw=True, **kwargs):
        super().__init__(**kwargs)
        self.keep_raw = str(keep_raw).lower() == "true"
        self.params = {
            "layer": layer,
            "z_field": z_field,
            "weight_field": weight_field,
            "unc_field": unc_field,
            "z_scale": z_scale
        }

    def run(self, entries):
        new_entries = []

        for mod, entry in entries:
            src = entry.get("dst_fn")

            # Basic validation
            if not src or not os.path.exists(src):
                new_entries.append((mod, entry))
                continue

            try:
                reader = OGRReader(src, **self.params)
                entry["stream"] = reader.yield_chunks()
                entry["stream_type"] = 'xyz_recarray'
            except Exception as e:
                logger.warning(f"OGRStream failed for {src}: {e}")

            new_entries.append((mod, entry))

        return new_entries
