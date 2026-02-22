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

    _known_layer_names = ['SOUNDG', 'SurveyPoint_HD', 'SurveyPoint', 'Mass_Point', 'Spot_Elevation']
    _known_elev_fields = ['Elevation', 'elev', 'z', 'height', 'depth', 'val', 'value',
                          'topography', 'surveyPointElev', 'Z_depth', 'Z_height']

    def __init__(self,
                 src_fn: str,
                 ogr_layer=None,
                 elev_field=None,
                 weight_field=None,
                 uncertainty_field=None,
                 z_scale=None,
                 elevation_value=None,
                 **kwargs):

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
                if layer: break

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

        if self.src_fn is None: return

        try:
            ds_ogr = ogr.Open(self.src_fn)
            if ds_ogr is None:
                logger.warning(f"Could not open OGR file: {self.fn}")
                return

            layer = self._get_layer(ds_ogr)
            if layer is None: return

            ## Auto-detect fields
            self._resolve_fields(layer.GetLayerDefn())

            ## --- Spatial Filter Logic ---
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

            ## --- Buffer Setup for Chunking ---
            chunk_x = []
            chunk_y = []
            chunk_z = []
            chunk_w = []
            chunk_u = []
            chunk_size = 0
            max_chunk = 100000


            def flush_chunk():
                if chunk_size == 0: return None

                dataset = np.column_stack((chunk_x, chunk_y, chunk_z, chunk_w, chunk_u))
                rec_arr = np.rec.fromrecords(dataset, names='x, y, z, w, u')

                ## Apply Z Scale
                if self.z_scale is not None:
                    rec_arr['z'] *= self.z_scale

                return rec_arr


            ## --- Feature Iteration ---
            for feature in layer:
                geom = feature.GetGeometryRef()
                if geom is None: continue

                ## Get Weight/Uncertainty (Per Feature)
                w_val = 1.0
                if self.weight_field:
                    w_val = float_or(feature.GetField(self.weight_field), 1.0)

                u_val = 0.0
                if self.uncertainty_field:
                    u_val = float_or(feature.GetField(self.uncertainty_field), 0.0)

                ## Get Coordinates
                pts = []

                ## Simple Geometry
                if geom.GetGeometryCount() == 0:
                    pts = geom.GetPoints()
                else:
                    ## Flatten Multi-Geometries
                    for i in range(geom.GetGeometryCount()):
                        sub_geom = geom.GetGeometryRef(i)
                        pts.extend(sub_geom.GetPoints())

                if not pts: continue

                ## Extract Z
                is_3d = geom.Is3D()

                for pt in pts:
                    x, y = pt[0], pt[1]
                    z = None

                    ## 3D Geometry Z
                    if is_3d and len(pt) > 2:
                        z = pt[2]

                    ## Attribute Field
                    if z is None and self.elev_field:
                        z = float_or(feature.GetField(self.elev_field))

                    ## Constant Value
                    if z is None and self.elevation_value is not None:
                        z = self.elevation_value

                    if z is None: continue

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

            ## Final flush
            if chunk_size > 0:
                yield flush_chunk()

            ds_ogr = None

        except Exception as e:
            logger.error(f"OGR processing failed for {self.src_fn}: {e}")
            return None

# class OGRReader:
#     """
#     Streaming OGR Vector Parser.
#     Reads vector data (S-57, Shapefile, GDB) and extracts 3D points.
#     """

#     # Known layer names for Bathymetry (Priority Order)
#     KNOWN_LAYERS = ['SOUNDG', 'SurveyPoint_HD', 'SurveyPoint', 'Mass_Point', 'Spot_Elevation']

#     # Known elevation fields
#     KNOWN_Z_FIELDS = ['VALSOU', 'Elevation', 'elev', 'z', 'depth', 'height', 'value']

#     def __init__(self, src_fn, layer=None,
#                  z_field=None, weight_field=None, unc_field=None,
#                  z_scale=1, chunk_size=50_000):

#         if not HAS_OGR:
#             raise ImportError("GDAL/OGR is required for this processor.")

#         self.src_fn = src_fn
#         self.layer_name = layer
#         self.z_field = z_field
#         self.weight_field = weight_field
#         self.unc_field = unc_field
#         self.z_scale = float_or(z_scale, 1)
#         self.chunk_size = chunk_size

#     def _get_layer(self, ds):
#         """Resolve the layer to process."""
#         lyr = None

#         # 1. User Specified
#         if self.layer_name:
#             lyr = ds.GetLayerByName(self.layer_name)
#             if not lyr and str(self.layer_name).isdigit():
#                 lyr = ds.GetLayer(int(self.layer_name))

#         # 2. Auto-Detect Known Bathymetry Layers
#         if lyr is None:
#             for name in self.KNOWN_LAYERS:
#                 lyr = ds.GetLayerByName(name)
#                 if lyr:
#                     logger.info(f"Auto-detected layer: {name}")
#                     break

#         # 3. Default to First Layer
#         if lyr is None:
#             lyr = ds.GetLayer(0)

#         return lyr

#     def _get_z_field(self, layer_defn):
#         """Auto-detect Z field if not provided."""
#         if self.z_field:
#             return self.z_field

#         field_count = layer_defn.GetFieldCount()
#         field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(field_count)]

#         for f in self.KNOWN_Z_FIELDS:
#             if f in field_names:
#                 logger.info(f"Auto-detected Z field: {f}")
#                 return f
#         return None

#     def yield_chunks(self):
#         """Yield chunks of (x, y, z, w, u)."""

#         ds = ogr.Open(self.src_fn)
#         if not ds:
#             raise IOError(f"Could not open {self.src_fn}")

#         layer = self._get_layer(ds)
#         if not layer:
#             raise IOError(f"No valid layer found in {self.src_fn}")

#         z_field_name = self._get_z_field(layer.GetLayerDefn())

#         # Buffers
#         bx, by, bz, bw, bu = [], [], [], [], []
#         count = 0

#         for feature in layer:
#             geom = feature.GetGeometryRef()
#             if not geom: continue

#             # Attributes
#             # Get Z from attribute if requested/detected
#             z_attr = None
#             if z_field_name:
#                 # S-57 VALSOU is often a float
#                 try:
#                     z_attr = feature.GetFieldAsDouble(z_field_name)
#                 except: pass

#             w_val = 1.0
#             if self.weight_field:
#                 w_val = feature.GetFieldAsDouble(self.weight_field)

#             u_val = 0.0
#             if self.unc_field:
#                 u_val = feature.GetFieldAsDouble(self.unc_field)

#             # Flatten Geometry to Points
#             # This handles MultiPoint, Point, Point25D
#             # Note: For Lines/Polygons, this extracts vertices (vertices as soundings)

#             # Fast path for single points
#             if geom.GetGeometryType() in [ogr.wkbPoint, ogr.wkbPoint25D]:
#                 pt = geom.GetPoint() # (x, y) or (x, y, z)
#                 val_z = pt[2] if len(pt) > 2 else z_attr

#                 if val_z is not None:
#                     bx.append(pt[0])
#                     by.append(pt[1])
#                     bz.append(val_z)
#                     bw.append(w_val)
#                     bu.append(u_val)
#                     count += 1
#             else:
#                 pts = geom.GetPoints() or []
#                 if not pts and geom.GetGeometryCount() > 0:
#                     for i in range(geom.GetGeometryCount()):
#                         sub = geom.GetGeometryRef(i)
#                         if sub:
#                             sub_pts = sub.GetPoints()
#                             if sub_pts:
#                                 pts.extend(sub_pts)

#                 if not sub_pts: continue

#                 if pts:
#                     for pt in pts:
#                         val_z = pt[2] if len(pt) > 2 else z_attr
#                         if val_z is not None:
#                             bx.append(pt[0])
#                             by.append(pt[1])
#                             bz.append(val_z)
#                             bw.append(w_val)
#                             bu.append(u_val)
#                             count += 1

#             # Flush Chunk
#             if count >= self.chunk_size:
#                 yield self._pack_chunk(bx, by, bz, bw, bu)
#                 bx, by, bz, bw, bu = [], [], [], [], []
#                 count = 0

#         # Final Flush
#         if count > 0:
#             yield self._pack_chunk(bx, by, bz, bw, bu)

#         ds = None

#     def _pack_chunk(self, x, y, z, w, u):
#         x = np.array(x, dtype=float)
#         y = np.array(y, dtype=float)
#         z = np.array(z, dtype=float) * self.z_scale
#         w = np.array(w, dtype=float)
#         u = np.array(u, dtype=float)
#         return x, y, z, w, u


#     def process(self, dst_fn):
#         """Write to XYZ."""
#         try:
#             with open(dst_fn, 'w') as f:
#                 for x, y, z, w, u in self.yield_chunks():

#                     cols = [x, y, z]
#                     fmt = ['%.8f', '%.8f', '%.6f']

#                     # Only write W/U if they are non-default (heuristic)
#                     # or just always write them if specific field requested?
#                     # Let's write them if they were requested or detected
#                     if self.weight_field:
#                         cols.append(w)
#                         fmt.append('%.4f')

#                     if self.unc_field:
#                         cols.append(u)
#                         fmt.append('%.4f')

#                     data = np.column_stack(cols)
#                     np.savetxt(f, data, fmt=fmt, delimiter=' ')
#             return dst_fn
#         except Exception as e:
#             logger.error(f"OGR processing failed: {e}")
#             if os.path.exists(dst_fn):
#                 os.remove(dst_fn)
#             return None


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
        self.keep_raw = str(keep_raw).lower() == 'true'
        self.params = {
            'layer': layer,
            'z_field': z_field,
            'weight_field': weight_field,
            'unc_field': unc_field,
            'z_scale': z_scale
        }

    def run(self, entries):
        new_entries = []

        for mod, entry in entries:
            src = entry.get('dst_fn')

            # Basic validation
            if not src or not os.path.exists(src):
                new_entries.append((mod, entry))
                continue

            # S-57 requires the .000 extension to be recognized sometimes,
            # or directory based GDBs. OGR Open is robust though.

            try:
                reader = OGRReader(src, **self.params)
                entry['stream'] = reader.yield_chunks()
                entry['stream_type'] = 'xyz_recarray'
            except Exception as e:
                logger.warning(f"OGRStream failed for {src}: {e}")

            new_entries.append((mod, entry))

        return new_entries


# class OGRToXYZ(FetchHook):
#     """
#     Convert Vector Data (S-57, Shapefile, GDB) to XYZ points.

#     Auto-detects bathymetry layers (SOUNDG) and Z-fields (VALSOU).

#     Usage:
#       --hook ogr_to_xyz (Auto S-57)
#       --hook ogr_to_xyz:layer=SurveyPoint,z_field=depth (eHydro)
#     """

#     name = "ogr_to_xyz"
#     stage = "file"

#     def __init__(self, layer=None, z_field=None,
#                  weight_field=None, unc_field=None,
#                  z_scale=1, keep_raw=True, **kwargs):
#         super().__init__(**kwargs)
#         self.keep_raw = str(keep_raw).lower() == 'true'
#         self.params = {
#             'layer': layer,
#             'z_field': z_field,
#             'weight_field': weight_field,
#             'unc_field': unc_field,
#             'z_scale': z_scale
#         }

#     def run(self, entries):
#         new_entries = []

#         for mod, entry in entries:
#             src = entry.get('dst_fn')

#             # Basic validation
#             if not src or not os.path.exists(src):
#                 new_entries.append((mod, entry))
#                 continue

#             # S-57 requires the .000 extension to be recognized sometimes,
#             # or directory based GDBs. OGR Open is robust though.

#             dst = f"{src}.xyz"

#             try:
#                 reader = OGRReader(src, **self.params)
#                 result = reader.process(dst)

#                 if result and os.path.exists(result) and os.path.getsize(result) > 0:
#                     entry['dst_fn'] = result
#                     entry['raw_fn'] = src
#                     entry['data_type'] = 'xyz'

#                     if not self.keep_raw:
#                         # Be careful deleting directories (GDB)
#                         if os.path.isfile(src):
#                             os.remove(src)
#                 else:
#                     if result and os.path.exists(result):
#                         os.remove(result)

#             except Exception as e:
#                 # pass on non-vector files
#                 pass

#             new_entries.append((mod, entry))

#         return new_entries
