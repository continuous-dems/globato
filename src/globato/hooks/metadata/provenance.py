#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.hooks.metadata.provenance
~~~~~~~~~~~~~~~~~~~~~~~

Generate bitmap data mask

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
import threading
import rasterio
from rasterio.windows import Window

from fetchez.hooks import FetchHook
from fetchez.utils import str2inc, inc2str
from ..transforms.point_pixels import PointPixels

logger = logging.getLogger(__name__)


class ProvenanceHook(FetchHook):
    """Generates a 'Provenance' mask raster.
    Each module is assigned a Bit ID. Pixel value = Bitmask of contributing modules.

    Usage:
      fetchez ... --hook provenance:res=1s,output=mask.tif
    """

    name = "provenance"
    meta_stage = "stream"
    meta_category = "metadata"
    meta_desc = "Generate a dataset mask raster using Bit IDs."

    def __init__(self, res="1s", output="provenance.tif", **kwargs):
        super().__init__(**kwargs)
        self.res = str2inc(res)
        self.output = output
        self._initialized = False
        self.lock = threading.Lock()

        # { 'srtm': 1, 'multibeam': 2, ... }
        self.module_bits = {}
        self.next_bit = 0

    def _init_raster(self, region):
        """Create the zero-filled UInt32 raster."""

        if self._initialized:
            return

        x_inc, y_inc = self.res, self.res
        self.xcount, self.ycount, self.dst_gt = region.geo_transform(
            x_inc=x_inc, y_inc=y_inc, node="grid"
        )
        self.transform = rasterio.transform.from_origin(
            region.xmin, region.ymax, x_inc, y_inc
        )

        self.pixel_binner = PointPixels(
            src_region=region, x_size=self.xcount, y_size=self.ycount
        )

        crs_val = getattr(region, "srs", "EPSG:4326") or "EPSG:4326"

        profile = {
            "driver": "GTiff",
            "dtype": "uint32",  # Supports up to 32 modules
            "count": 1,
            "width": self.xcount,
            "height": self.ycount,
            "crs": crs_val,
            "transform": self.transform,
            "compress": "lzw",
            "nodata": 0,
        }

        with rasterio.open(self.output, "w", **profile) as dst:
            dst.set_band_description(1, "Module_Bitmask")

        self._initialized = True
        logger.debug(f"Initialized Provenance Mask: {self.output}")

    def _get_module_bit(self, mod_name):
        """Assign a unique bit (power of 2) to this module."""

        with self.lock:
            if mod_name not in self.module_bits:
                if self.next_bit > 31:
                    logger.warning("Provenance: >32 modules! Mask overflow.")
                    return 0

                bit_val = 1 << self.next_bit
                self.module_bits[mod_name] = bit_val
                self.next_bit += 1

                logger.info(
                    f"Provenance Map: {mod_name} -> Bit {self.next_bit} (Val {bit_val})"
                )

            return self.module_bits[mod_name]

    def run(self, entries):
        if not self._initialized and entries:
            region = next(
                (mod.region for mod, _ in entries if getattr(mod, "region", None)), None
            )
            if region:
                self._init_raster(region)

        for mod, entry in entries:
            if self.is_point_stream(entry):
                stream = entry.get("stream")
                bit_val = self._get_module_bit(mod.name)
                entry["stream"] = self._intercept(stream, bit_val)

                entry.setdefault("artifacts", {})[self.name] = os.path.abspath(
                    self.output
                )

        return entries

    def _intercept(self, stream, bit_val):
        """Pass-through stream to update mask."""

        for chunk in stream:
            self._update_mask(chunk, bit_val)
            yield chunk

    def _update_mask(self, points, bit_val):
        """Bin points and OR the bitmask into the raster."""

        if not self._initialized or len(points) == 0:
            return

        has_data, sub_win, _ = self.pixel_binner.coverage(points)

        if has_data is None:
            return
        # arrays, sub_win, _ = self.pixel_binner(points, mode="count")

        # if arrays["count"] is None:
        #     return

        # has_data = arrays["count"] > 0

        col_off, row_off, w, h = sub_win
        window = Window(col_off, row_off, w, h)

        with self.lock:
            with rasterio.open(self.output, "r+") as dst:
                mask_data = dst.read(1, window=window)

                # Bitwise OR to add this module's presence
                mask_data[has_data] |= bit_val

                dst.write(mask_data, 1, window=window)

    def teardown(self):
        """Write the legend to metadata on exit."""

        if self._initialized:
            with rasterio.open(self.output, "r+") as dst:
                for name, idx in self.module_bits.items():
                    tags = {
                        f"MOD_{name}": str(bit)
                        for name, bit in self.module_bits.items()
                    }
                dst.update_tags(bidx=1, **tags)
            logger.debug("Finalized Provenance Mask.")


class SourceMasks(FetchHook):
    """Generates detailed, per-file source masks.
    Creates a directory of single-band GeoTIFFs (one per file) and builds a
    multi-band VRT for easy debugging in GIS software.

    Usage:
      --hook source_masks:res=1s,output_dir=debug_masks
    """

    name = "source-masks"
    meta_stage = "stream"
    meta_category = "metadata"
    meta_desc = "Generate source masks for all dataset and build a combined VRT/GPKG."
    meta_aliases = ["source_masks"]

    def __init__(
        self,
        res="1s",
        output_dir=None,
        output="source_masks.vrt",
        vector_output=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.res = str2inc(res)
        self.output = output
        self.vector_output = vector_output

        base_name = os.path.splitext(self.output)[0]
        self.output_dir = output_dir or f"{base_name}_temp_masks"

        self._initialized = False
        self.tifs = []
        self.lock = threading.Lock()

    def _init_grid(self, region):
        if self._initialized:
            return

        x_inc, y_inc = self.res, self.res
        self.xcount, self.ycount, self.dst_gt = region.geo_transform(
            x_inc=x_inc, y_inc=y_inc, node="grid"
        )
        self.transform = rasterio.transform.from_origin(
            region.xmin, region.ymax, x_inc, y_inc
        )

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        crs_val = getattr(region, "srs", "EPSG:4326") or "EPSG:4326"
        self.profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 1,
            "width": self.xcount,
            "height": self.ycount,
            "crs": crs_val,
            "transform": self.transform,
            "compress": "lzw",
            "nodata": 0,
        }

        self._initialized = True
        logger.debug(f"Initialized Detailed Source Masks in ./{self.output_dir}")

    def _write_qml_style(self, qml_path, unique_groups):
        """Generates a dynamic Categorized QGIS styling file."""

        palette = [
            "228,26,28,150",  # Red
            "55,126,184,150",  # Blue
            "77,175,74,150",  # Green
            "152,78,163,150",  # Purple
            "255,127,0,150",  # Orange
            "255,255,51,150",  # Yellow
            "166,86,40,150",  # Brown
        ]

        categories_xml = ""
        symbols_xml = ""

        for i, group in enumerate(unique_groups):
            # Loop back to start if we have more groups than colors
            color = palette[i % len(palette)]

            # The legend entry
            categories_xml += (
                f'    <category value="{group}" symbol="{i}" label="{group}"/>\n'
            )

            # The symbology definition
            symbols_xml += f"""
      <symbol type="fill" name="{i}" alpha="1">
        <layer pass="0" class="SimpleFill" locked="0">
          <prop k="color" v="{color}"/>
          <prop k="outline_color" v="0,0,0,255"/>
          <prop k="outline_width" v="0.3"/>
          <prop k="style" v="solid"/>
        </layer>
      </symbol>"""

        # The master QML template
        qml_content = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.10.0" styleCategories="Symbology|Labeling">

  <previewExpression>"GROUP_ID"</previewExpression>
  <!-- Categorized Renderer based on the GROUP_ID field -->
  <renderer-v2 type="categorizedSymbol" attr="GROUP_ID">
    <categories>
{categories_xml}
    </categories>
    <symbols>
{symbols_xml}
    </symbols>
  </renderer-v2>

  <!-- Labeling: Automatically label using the GROUP_ID field -->
  <labeling type="simple">
    <settings>
      <text-style fontFamily="sans-serif" fontSize="9" textColor="0,0,0,255">
        <text-buffer bufferSize="1" bufferColor="255,255,255,255" bufferDraw="1"/>
      </text-style>
      <fieldName>"GROUP_ID"</fieldName>
      <placement dist="0" quadOffset="4" placement="0"/>
    </settings>
  </labeling>
</qgis>
"""
        try:
            with open(qml_path, "w") as f:
                f.write(qml_content)
            logger.debug(f"Generated dynamic Categorized QML style file: {qml_path}")
        except Exception as e:
            logger.error(f"Failed to write QML file: {e}")

    def run(self, entries):
        if not self._initialized and entries:
            region = next(
                (mod.region for mod, _ in entries if getattr(mod, "region", None)), None
            )
            if region:
                self._init_grid(region)

        for mod, entry in entries:
            if self.is_point_stream(entry) and self._initialized:
                stream = entry.get("stream")
                src_name = os.path.basename(entry.get("dst_fn", f"unknown_{id(entry)}"))
                base = os.path.splitext(src_name)[0]
                res_str = inc2str(self.res)
                tif_path = os.path.join(self.output_dir, f"{base}_{res_str}_mask.tif")

                meta_tags = {
                    "MODULE": getattr(mod, "name", "Unknown"),
                    "DATASET": getattr(mod, "title", getattr(mod, "name", "Unknown")),
                    "CATEGORY": getattr(mod, "meta_category", "Unknown"),
                    "AGENCY": getattr(mod, "meta_agency", "Unknown"),
                    "DATATYPE": entry.get("data_type", "Unknown"),
                    "RESOLUTION": getattr(mod, "meta_resolution", "Varies"),
                    "URL": entry.get("url", "Unknown"),
                    "WEIGHT": str(getattr(mod, "weight", 1.0)),
                }

                if "metadata" in entry and isinstance(entry["metadata"], dict):
                    for k, v in entry["metadata"].items():
                        meta_tags[str(k).upper()] = str(v)

                clean_tags = {
                    k: str(v)
                    for k, v in meta_tags.items()
                    if v not in ["Unknown", "None", "", None]
                }

                with rasterio.open(tif_path, "w", **self.profile) as dst:
                    dst.set_band_description(1, base)

                    if clean_tags:
                        dst.update_tags(**clean_tags)

                with self.lock:
                    self.tifs.append(tif_path)

                entry["stream"] = self._intercept(stream, tif_path, mod.region)
                entry.setdefault("artifacts", {})[self.name] = tif_path

        return entries

    def _intercept(self, stream, tif_path, region):
        """Pass-through generator that writes to the specific TIF."""

        pixel_binner = PointPixels(
            src_region=region, x_size=self.xcount, y_size=self.ycount
        )
        with rasterio.open(tif_path, "r+") as dst:
            for chunk in stream:
                has_data, sub_win, _ = pixel_binner.coverage(chunk)

                if has_data is not None:
                    col_off, row_off, w, h = sub_win
                    window = Window(col_off, row_off, w, h)

                    current_mask = dst.read(1, window=window)
                    current_mask |= has_data
                    dst.write(current_mask, 1, window=window)

                yield chunk

    def teardown(self):
        """Build the VRT linking all the individual masks together."""

        if not self._initialized or not self.tifs:
            return

        # vrt_path = os.path.join(self.output_dir, self.vrt_name)
        logger.info(f"Building master VRT mask: {self.output}")

        try:
            with rasterio.open(self.tifs[0]) as src:
                width = src.width
                height = src.height
                transform = src.transform
                crs = src.crs.to_wkt() if src.crs else ""
                dtype = src.dtypes[0]
                _stats = src.stats()

            dtype_map = {
                "uint8": "Byte",
                "uint16": "UInt16",
                "int16": "Int16",
                "uint32": "UInt32",
                "int32": "Int32",
                "float32": "Float32",
                "float64": "Float64",
            }
            gdal_dtype = dtype_map.get(dtype, "Float32")

            gt = f"{transform.c}, {transform.a}, {transform.b}, {transform.f}, {transform.d}, {transform.e}"

            xml_lines = [
                f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">',
                f"  <SRS>{crs}</SRS>",
                f"  <GeoTransform>{gt}</GeoTransform>",
            ]

            for i, tif in enumerate(self.tifs, start=1):
                # remove the mask if no valid data
                if not os.path.exists(tif):
                    continue

                with rasterio.open(tif) as src:
                    tif_stats = src.stats()
                    tif_tags = src.tags()

                if tif_stats[0].max == 0.0:
                    os.remove(tif)
                    continue

                rel_path = os.path.relpath(tif, os.path.dirname(self.output))
                name = os.path.basename(tif).replace("_mask.tif", "")

                xml_lines.extend(
                    [
                        f'  <VRTRasterBand dataType="{gdal_dtype}" band="{i}">',
                        f"    <Description>{name}</Description>",
                    ]
                )

                if tif_tags:
                    import xml.sax.saxutils as saxutils

                    xml_lines.append("    <Metadata>")
                    for k, v in tif_tags.items():
                        safe_v = saxutils.escape(
                            str(v)
                        )  # Prevents XML breakage from URLs with '&'
                        xml_lines.append(f'      <MDI key="{k}">{safe_v}</MDI>')
                    xml_lines.append("    </Metadata>")

                xml_lines.extend(
                    [
                        "    <SimpleSource>",
                        f'      <SourceFilename relativeToVRT="1">{rel_path}</SourceFilename>',
                        "      <SourceBand>1</SourceBand>",
                        f'      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>',
                        f'      <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>',
                        "    </SimpleSource>",
                        "  </VRTRasterBand>",
                    ]
                )
                # xml_lines.extend(
                #     [
                #         f'  <VRTRasterBand dataType="{gdal_dtype}" band="{i}">',
                #         f"    <Description>{name}</Description>",
                #         "    <SimpleSource>",
                #         f'      <SourceFilename relativeToVRT="1">{rel_path}</SourceFilename>',
                #         "      <SourceBand>1</SourceBand>",
                #         f'      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>',
                #         f'      <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>',
                #         "    </SimpleSource>",
                #         "  </VRTRasterBand>",
                #     ]
                # )

            xml_lines.append("</VRTDataset>")

            with open(self.output, "w") as f:
                f.write("\n".join(xml_lines))

            logger.debug(f"VRT {self.output} built successfully.")

            # --- Generate Vector Spatial Metadata, if requested ---
            if self.vector_output:
                logger.info(f"Polygonizing VRT to {self.vector_output}...")
                try:
                    import geopandas as gpd
                    from rasterio.features import shapes
                    from shapely.geometry import shape

                    records = []
                    with rasterio.open(self.output) as src:
                        for i in range(1, src.count + 1):
                            band = src.read(i)
                            band_name = src.descriptions[i - 1] or f"Band_{i}"
                            tags = src.tags(i)
                            # Polygonize where pixel > 0
                            geom_generator = shapes(
                                band, mask=(band > 0), transform=src.transform
                            )
                            for geom, value in geom_generator:
                                record = {
                                    "Filename": band_name,
                                    "geometry": shape(geom),
                                }
                                record.update(tags)
                                records.append(record)

                    if records:
                        from urllib.parse import urlparse

                        gdf = gpd.GeoDataFrame(records, crs=src.crs)
                        agg_funcs = {}

                        for col in gdf.columns:
                            if col in ["MODULE", "DATASET", "WEIGHT", "geometry"]:
                                continue
                            elif "DATE" in col.upper() or "YEAR" in col.upper():
                                # Create a range: min - max
                                agg_funcs[col] = lambda x: (
                                    f"{x.dropna().min()} - {x.dropna().max()}"
                                    if x.dropna().min() != x.dropna().max()
                                    else str(x.dropna().min())
                                )
                            elif "URL" in col.upper():
                                # Extract unique base domains instead of keeping massive URLs
                                agg_funcs[col] = lambda x: ", ".join(
                                    set(
                                        urlparse(str(u)).netloc
                                        if str(u).startswith("http")
                                        else str(u)
                                        for u in x.dropna()
                                        if str(u) != "Unknown"
                                    )
                                )
                            else:
                                # For all other fields, join unique values into a comma-separated list
                                agg_funcs[col] = lambda x: ", ".join(
                                    map(str, set(x.dropna()))
                                )

                        # Apply the custom aggregation during the dissolve
                        gdf = gdf.dissolve(
                            by=["MODULE", "DATASET", "WEIGHT"], aggfunc=agg_funcs
                        ).reset_index()

                        gdf["GROUP_ID"] = (
                            gdf["MODULE"]
                            + " | "
                            + gdf["DATASET"]
                            + " (Wt: "
                            + gdf["WEIGHT"].astype(str)
                            + ")"
                        )
                        if "Filename" in gdf.columns:
                            gdf = gdf.drop(columns=["Filename"])

                        if "AREA_OR_POINT" in gdf.columns:
                            gdf = gdf.drop(columns=["AREA_OR_POINT"])

                        if "DATA_TYPE" in gdf.columns and "DATATYPE" in gdf.columns:
                            gdf = gdf.drop(columns=["DATA_TYPE"])

                        if "NAME" in gdf.columns:
                            gdf = gdf.drop(columns=["NAME"])

                        cols = gdf.columns.tolist()
                        cols.insert(0, cols.pop(cols.index("GROUP_ID")))

                        if "geometry" in cols:
                            cols.append(cols.pop(cols.index("geometry")))

                        gdf = gdf[cols]

                        gdf.to_file(self.vector_output, driver="GPKG", engine="pyogrio")
                        logger.info(
                            f"Spatial metadata vector saved to {self.vector_output}"
                        )

                        unique_groups = gdf["GROUP_ID"].unique().tolist()
                        qml_path = os.path.splitext(self.vector_output)[0] + ".qml"
                        self._write_qml_style(qml_path, unique_groups)
                    else:
                        logger.debug("No valid geometries found to polygonize.")

                except Exception as e:
                    logger.exception(f"Failed to generate spatial metadata vector: {e}")

        except Exception as e:
            logger.error(f"Failed to build VRT with rasterio: {e}")

    # def teardown_(self):
    #     """Build the VRT linking all the individual masks together."""

    #     if not self._initialized or not self.tifs:
    #         return

    #     vrt_path = os.path.join(self.output_dir, self.vrt_name)
    #     logger.info(f"Building master VRT mask: {vrt_path}")

    #     try:
    #         # Use gdal Python bindings if available
    #         from osgeo import gdal

    #         vrt_options = gdal.BuildVRTOptions(separate=True)
    #         vrt_ds = gdal.BuildVRT(vrt_path, self.tifs, options=vrt_options)

    #         if vrt_ds:
    #             # Name the bands after the files so they show up beautifully in QGIS
    #             for i, tif in enumerate(self.tifs):
    #                 band = vrt_ds.GetRasterBand(i + 1)
    #                 name = os.path.basename(tif).replace("_mask.tif", "")
    #                 band.SetDescription(name)

    #             vrt_ds.FlushCache()
    #             vrt_ds = None

    #     except ImportError:
    #         import subprocess

    #         logger.debug(
    #             "osgeo Python bindings not found. Falling back to gdalbuildvrt CLI."
    #         )

    #         cmd = ["gdalbuildvrt", "-separate", vrt_path] + self.tifs

    #         try:
    #             subprocess.run(
    #                 cmd,
    #                 check=True,
    #                 stdout=subprocess.DEVNULL,
    #                 stderr=subprocess.DEVNULL,
    #             )
    #         except FileNotFoundError:
    #             logger.error(
    #                 "gdalbuildvrt command not found on system. Skipping VRT generation."
    #             )
    #         except subprocess.CalledProcessError as e:
    #             logger.error(f"Failed to build VRT via CLI: {e}")

    # def teardown_gdal(self):
    #     """Build the VRT linking all the individual masks together."""

    #     if self._initialized and self.tifs:
    #         vrt_path = os.path.join(self.output_dir, self.vrt_name)
    #         logger.info(f"Building master VRT mask: {vrt_path}")

    #         vrt_options = gdal.BuildVRTOptions(separate=True)
    #         vrt_ds = gdal.BuildVRT(vrt_path, self.tifs, options=vrt_options)

    #         if vrt_ds:
    #             for i, tif in enumerate(self.tifs):
    #                 band = vrt_ds.GetRasterBand(i + 1)
    #                 name = os.path.basename(tif).replace("_mask.tif", "")
    #                 band.SetDescription(name)

    #             vrt_ds.FlushCache()
    #             vrt_ds = None
