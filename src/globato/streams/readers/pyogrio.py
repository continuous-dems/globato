#!/usr/bin/env python

"""
globato.streams.readers.pyogrio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pyogrio/shapely vector reader with native 3D Breakline densification.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import os
import warnings

import numpy as np
import pandas as pd
import pyogrio
import shapely
from fetchez.utils import float_or, str2inc

from .base import BaseGlobatoReader

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore", message="Measured \\(M\\) geometry types are not supported.*"
)


class PyogrioReader(BaseGlobatoReader):
    """Vectorized Pyogrio Vector Parser for point extraction and Breakline generation."""

    name = "pyogrio-point-reader"
    meta_category = "point-stream"
    meta_dtype = "pyogrio-vector"
    meta_desc = "Read vector data through pyogrio into a point stream, with optional breakline densification."
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
        # --- Breakline Args ---
        as_breakline=False,
        step=1.0,
        z_raster=None,
        weight=1.0,
        **kwargs,
    ):
        super().__init__(path, **kwargs)
        self.src_fn = path

        if self.src_fn.lower().endswith(".zip"):
            import zipfile

            internal_target = ""
            try:
                with zipfile.ZipFile(self.src_fn, "r") as z:
                    for name in z.namelist():
                        if name.lower().endswith(".gdb/") or name.lower().endswith(
                            ".gdb"
                        ):
                            internal_target = f"!{name.strip('/')}"
                            break
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
        # Breakline Properties
        self.as_breakline = as_breakline
        self.step = step
        self.z_raster = z_raster
        self.weight = float_or(weight, 1.0)
        self.resolved_z_raster = None

        # Pre-fetch any necessary reference DEMs for breakline sampling
        if self.as_breakline and self.z_raster:
            self.step = str2inc(self.step)
            self.resolved_z_raster = self._resolve_z_raster(self.region)

    def get_srs(self):
        try:
            layer_name = self._resolve_layer()
            info = pyogrio.read_info(self.src_fn, layer=layer_name)
            crs_val = info.get("crs")
            if crs_val:
                return str(crs_val)
            return None
        except Exception as e:
            logger.debug(f"Could not extract SRS from {self.src_fn}: {e}")
            return None

    def _resolve_layer(self):
        layers = pyogrio.list_layers(self.src_fn)
        layer_names = [lyr[0] for lyr in layers]

        if self.target_layer and self.target_layer in layer_names:
            return self.target_layer

        for name in self.KNOWN_LAYERS:
            if name in layer_names:
                logger.debug(f"Auto-detected vector layer: {name}")
                return name
        return layer_names[0]

    def _resolve_z_field(self, columns):
        if self.z_field:
            return self.z_field
        for f in self.KNOWN_Z_FIELDS:
            if f in columns:
                return f
        return None

    def _resolve_z_raster(self, region):
        """Resolves the z_raster argument, fetching via Fetchez if necessary."""
        if os.path.exists(self.z_raster) and os.path.isfile(self.z_raster):
            return self.z_raster

        import fetchez

        outdir = os.path.abspath("auto_barriers")
        os.makedirs(outdir, exist_ok=True)

        logger.info(
            f"[{self.name}] Fetching reference raster '{self.z_raster}' via Fetchez..."
        )
        try:
            fetch_region = region.copy()
            fetch_region.buffer(pct=5)

            files = fetchez.get(
                self.z_raster,
                region=fetch_region.to_list(),
                region_srs=region.srs,
                outdir=outdir,
                use_cache=True,
            )

            if not files:
                return None

            if len(files) == 1:
                return files[0]

            try:
                from osgeo import gdal

                vrt_path = os.path.join(
                    outdir, f"breakline_ref_{fetch_region.format('fn')}.vrt"
                )
                if not os.path.exists(vrt_path):
                    vrt_options = gdal.BuildVRTOptions(resampleAlg="bilinear")
                    gdal.BuildVRT(vrt_path, files, options=vrt_options)
                return vrt_path
            except ImportError:
                return files[0]
        except Exception as e:
            logger.error(
                f"[{self.name}] Failed to fetch reference raster '{self.z_raster}': {e}"
            )
            return None

    def _densify_line(self, line):
        length = line.length
        if length == 0 or np.isnan(length):
            return line
        num_segments = int(np.ceil(length / self.step))
        if num_segments <= 1:
            return line
        return line.__class__(
            [line.interpolate(d) for d in np.linspace(0, length, num_segments + 1)]
        )

    def _yield_raw_chunks(self):
        try:
            layer_name = self._resolve_layer()
            skip = 0

            sampler = None
            if self.as_breakline and self.resolved_z_raster:
                import rasterio

                sampler = rasterio.open(self.resolved_z_raster)

            while True:
                chunk_gdf = pyogrio.read_dataframe(
                    self.src_fn,
                    layer=layer_name,
                    max_features=self.chunk_size,
                    skip_features=skip,
                )

                if chunk_gdf.empty:
                    break

                if self.as_breakline:
                    yield from self._yield_breakline_chunks(chunk_gdf, sampler)
                else:
                    yield from self._yield_standard_chunks(chunk_gdf)

                if len(chunk_gdf) < self.chunk_size:
                    break

                skip += self.chunk_size

            if sampler:
                sampler.close()

        except Exception as e:
            logger.exception(f"Pyogrio processing failed for {self.src_fn}: {e}")

    def _yield_breakline_chunks(self, chunk_gdf, sampler):
        import scipy.interpolate

        z_attr_name = self._resolve_z_field(chunk_gdf.columns)

        for i, geom in enumerate(chunk_gdf.geometry):
            if geom is None:
                continue

            lines = []

            # --- Extract both Exteriors and Interiors ---
            if geom.geom_type == "Polygon":
                lines.append(geom.exterior)
                lines.extend(geom.interiors)
            elif geom.geom_type == "MultiPolygon":
                for p in geom.geoms:
                    lines.append(p.exterior)
                    lines.extend(p.interiors)
            elif geom.geom_type in ["LineString", "LinearRing"]:
                lines.append(geom)
            elif geom.geom_type == "MultiLineString":
                lines.extend(list(geom.geoms))

            for line in lines:
                if line.length == 0:
                    continue

                densified_line = self._densify_line(line)
                coords = np.array(densified_line.coords)
                x, y = coords[:, 0], coords[:, 1]
                z = np.zeros_like(x)
                valid_mask = np.ones_like(x, dtype=bool)

                # --- Constant Value ---
                if self.elevation_value is not None:
                    z[:] = self.elevation_value

                # --- Vector Attribute Field ---
                elif z_attr_name:
                    try:
                        z[:] = float(chunk_gdf.iloc[i][z_attr_name])
                    except (ValueError, TypeError):
                        valid_mask[:] = False

                # --- Dynamic Raster Sampling & Interpolation ---
                elif sampler:
                    pts = [(xi, yi) for xi, yi in zip(x, y)]
                    z_samples = np.array([val[0] for val in sampler.sample(pts)])
                    nodata = sampler.nodata if sampler.nodata is not None else -9999
                    z_valid = (z_samples != nodata) & ~np.isnan(z_samples)

                    if not np.any(z_valid):
                        valid_mask[:] = False
                    else:
                        is_closed = densified_line.is_closed or (
                            x[0] == x[-1] and y[0] == y[-1]
                        )
                        if is_closed:
                            z[:] = np.median(z_samples[z_valid])
                        else:
                            dx, dy = np.diff(x), np.diff(y)
                            dists = np.concatenate(
                                ([0], np.cumsum(np.sqrt(dx**2 + dy**2)))
                            )
                            if np.count_nonzero(z_valid) > 1:
                                interp_func = scipy.interpolate.interp1d(
                                    dists[z_valid],
                                    z_samples[z_valid],
                                    kind="linear",
                                    bounds_error=False,
                                    fill_value=(
                                        z_samples[z_valid][0],
                                        z_samples[z_valid][-1],
                                    ),
                                )
                                z[:] = interp_func(dists)
                            else:
                                z[:] = z_samples[z_valid][0]
                else:
                    z[:] = 0.0

                x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]

                if len(x) > 0:
                    w = np.full(len(x), self.weight, dtype="f4")
                    u = np.zeros(len(x), dtype="f4")
                    z = z * self.z_scale
                    yield self._pack(x, y, z, w, u)

    def _yield_standard_chunks(self, chunk_gdf):
        coords = shapely.get_coordinates(chunk_gdf.geometry, include_z=True)
        x = coords[:, 0]
        y = coords[:, 1]
        counts = shapely.get_num_coordinates(chunk_gdf.geometry)

        z_attr_name = self._resolve_z_field(chunk_gdf.columns)
        if z_attr_name:
            z_series = pd.to_numeric(chunk_gdf[z_attr_name], errors="coerce")
            z_raw = z_series.fillna(self.elevation_value or 0.0).values
            z = np.repeat(z_raw, counts)
        elif coords.shape[1] == 3:
            z = coords[:, 2]
        else:
            z = np.full(len(x), self.elevation_value or 0.0)

        z = z * self.z_scale

        if self.weight_field and self.weight_field in chunk_gdf.columns:
            w_raw = chunk_gdf[self.weight_field].fillna(1.0).values
            w = np.repeat(w_raw, counts)
            w *= self.weight
        else:
            w = np.ones(len(x))
            w *= self.weight

        if self.unc_field and self.unc_field in chunk_gdf.columns:
            u_raw = chunk_gdf[self.unc_field].fillna(0.0).values
            u = np.repeat(u_raw, counts)
        else:
            u = np.zeros(len(x))

        yield self._pack(x, y, z, w, u)

    def _pack(self, x, y, z, w, u):
        dt = [("x", "f8"), ("y", "f8"), ("z", "f4"), ("w", "f4"), ("u", "f4")]
        data = [np.array(x), np.array(y), np.array(z), np.array(w), np.array(u)]
        return np.rec.fromarrays(data, dtype=dt)
