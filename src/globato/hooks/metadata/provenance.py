#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.hooks.metadata.provenance
~~~~~~~~~~~~~~~~~~~~~~~

Generate bitmap data mask

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

from __future__ import annotations

import os
import logging
import threading
import hashlib
import tempfile
from urllib.parse import urlparse

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.features import shapes
from rasterio.windows import Window

from fetchez.hooks import FetchHook
from fetchez.utils import str2inc, str2bool, int_or
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
            src_region=region,
            x_size=self.xcount,
            y_size=self.ycount,
            dst_gt=self.dst_gt,
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


def source_id(entry):
    """Return the same stable source identity used by persistent stack state."""

    checksum = entry.get("checksum")
    if checksum:
        return str(checksum)

    url = entry.get("url", "")
    dst_fn = entry.get("dst_fn")

    if url and not url.startswith("file://"):
        return url

    if dst_fn and os.path.exists(dst_fn):
        size = os.path.getsize(dst_fn)
        return f"{os.path.basename(dst_fn)}|{size}B"

    return os.path.basename(dst_fn or url or "unknown_dataset")


def source_token(dataset_id, length=12):
    """Filesystem-safe stable token without encoding an entire URL/path."""
    return hashlib.sha256(dataset_id.encode("utf-8")).hexdigest()[:length]


class MaskSet:
    """Persistent collection of single-source coverage masks.

    The individual GeoTIFFs are the persistent state. The VRT and vector
    products are derived indexes that may be rebuilt cheaply.
    """

    TYPE_TAG = "GLOBATO_DATATYPE"
    SOURCE_TAG = "GLOBATO_SOURCE_ID"

    def __init__(
        self,
        region,
        res,
        output,
        output_dir=None,
        vector_output=None,
        *,
        mask_type="SOURCE_MASK",
        resume=True,
        vector_max_size=2048,
        vector_simplify=0.0,
        qgis_style=True,
        qgis_style_field="SOURCE_ID",
        group_by=None,
        dst_gt=None,
        xcount=None,
        ycount=None,
    ):
        self.region = region
        self.res = float(res)
        self.output = output
        self.vector_output = vector_output
        self.mask_type = mask_type
        self.resume = bool(resume)
        self.vector_max_size = int(vector_max_size)
        self.vector_simplify = max(0.0, float(vector_simplify))
        self.qgis_style = bool(qgis_style)
        self.qgis_style_field = str(qgis_style_field)
        self.group_by = self._parse_group_by(group_by)
        self.dst_gt = dst_gt
        self.xcount = int_or(xcount)
        self.ycount = int_or(ycount)

        base = os.path.splitext(output)[0]
        self.output_dir = output_dir or f"{base}_temp_masks"

        os.makedirs(self.output_dir, exist_ok=True)

        if self.dst_gt is None:
            if not self.xcount or not self.ycount:
                self.xcount, self.ycount, self.dst_gt = region.geo_transform(
                    x_inc=self.res,
                    y_inc=self.res,
                    node="grid",
                )
            else:
                self.dst_gt = region.geo_transform_from_count(
                    x_count=self.xcount,
                    y_count=self.ycount,
                )
        else:
            if not self.xcount or not self.ycount:
                raise ValueError(
                    "MaskSet requires xcount/ycount when dst_gt is supplied"
                )

        self.transform = rasterio.Affine.from_gdal(*self.dst_gt)

        crs = getattr(region, "srs", "EPSG:4326") or "EPSG:4326"

        self.profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 1,
            "width": self.xcount,
            "height": self.ycount,
            "crs": crs,
            "transform": self.transform,
            "compress": "lzw",
            "nodata": 0,
            "tiled": True,
        }

        # dataset_id -> tif path
        self.masks = {}

    def _path(self, dataset_id):
        token = source_token(dataset_id)
        return os.path.join(self.output_dir, f"{token}_mask.tif")

    def _validate(self, path, dataset_id):
        with rasterio.open(path) as src:
            if src.width != self.xcount or src.height != self.ycount:
                return False

            if src.transform != self.transform:
                return False

            if src.count != 1 or src.dtypes[0] != "uint8":
                return False

            tags = src.tags()
            if tags.get(self.TYPE_TAG) != self.mask_type:
                return False

            if tags.get(self.SOURCE_TAG) != dataset_id:
                return False

        return True

    def register(self, dataset_id, *, description=None, tags=None):
        """Create or resume a source mask and return its path."""

        path = self._path(dataset_id)

        reuse = (
            self.resume and os.path.exists(path) and self._validate(path, dataset_id)
        )

        if not reuse:
            with rasterio.open(path, "w", **self.profile) as dst:
                if description:
                    dst.set_band_description(1, description)

                metadata = {}
                if tags:
                    metadata.update(
                        {
                            str(k): str(v)
                            for k, v in tags.items()
                            if v not in (None, "", "None", "Unknown")
                        }
                    )
                metadata[self.TYPE_TAG] = self.mask_type
                metadata[self.SOURCE_TAG] = dataset_id

                dst.update_tags(**metadata)

        self.masks[dataset_id] = path
        return path

    @staticmethod
    def update(path, window, mask):
        if mask is None or not np.any(mask):
            return

        with rasterio.open(path, "r+") as dst:
            data = dst.read(1, window=window)
            data[mask] = 1
            dst.write(data, 1, window=window)

    def clear_others(self, current_id, window, mask):
        """Clear pixels replaced by current_id from previously accepted masks."""

        if mask is None or not np.any(mask):
            return

        for dataset_id, path in self.masks.items():
            if dataset_id == current_id or not os.path.exists(path):
                continue

            with rasterio.open(path, "r+") as dst:
                data = dst.read(1, window=window)
                changed = (data != 0) & mask

                if np.any(changed):
                    data[changed] = 0
                    dst.write(data, 1, window=window)

    def _valid_masks(self):
        valid = []

        for dataset_id, path in sorted(self.masks.items()):
            if not os.path.exists(path):
                continue

            with rasterio.open(path) as src:
                stats = src.stats()

            if stats[0].max == 0:
                # Empty accepted masks are common and harmless.
                continue

            valid.append((dataset_id, path))

        return valid

    def build_vrt(self):
        valid = self._valid_masks()

        if not valid:
            return

        with rasterio.open(valid[0][1]) as src:
            width = src.width
            height = src.height
            transform = src.transform
            crs = src.crs.to_wkt() if src.crs else ""

        gt = (
            f"{transform.c}, {transform.a}, {transform.b}, "
            f"{transform.f}, {transform.d}, {transform.e}"
        )

        xml = [
            f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">',
            f"  <SRS>{crs}</SRS>",
            f"  <GeoTransform>{gt}</GeoTransform>",
        ]

        import xml.sax.saxutils as saxutils

        for band, (_, path) in enumerate(valid, start=1):
            with rasterio.open(path) as src:
                tags = src.tags()
                description = src.descriptions[0] or os.path.basename(path)

            rel_path = os.path.relpath(path, os.path.dirname(self.output) or ".")

            xml.extend(
                [
                    f'  <VRTRasterBand dataType="Byte" band="{band}">',
                    f"    <Description>{saxutils.escape(description)}</Description>",
                    "    <Metadata>",
                ]
            )

            for key, value in tags.items():
                xml.append(
                    f'      <MDI key="{saxutils.escape(key)}">'
                    f"{saxutils.escape(str(value))}</MDI>"
                )

            xml.extend(
                [
                    "    </Metadata>",
                    "    <SimpleSource>",
                    f'      <SourceFilename relativeToVRT="1">'
                    f"{saxutils.escape(rel_path)}</SourceFilename>",
                    "      <SourceBand>1</SourceBand>",
                    f'      <SrcRect xOff="0" yOff="0" '
                    f'xSize="{width}" ySize="{height}"/>',
                    f'      <DstRect xOff="0" yOff="0" '
                    f'xSize="{width}" ySize="{height}"/>',
                    "    </SimpleSource>",
                    "  </VRTRasterBand>",
                ]
            )

        xml.append("</VRTDataset>")

        output_dir = os.path.dirname(os.path.abspath(self.output))
        os.makedirs(output_dir, exist_ok=True)

        fd, tmp = tempfile.mkstemp(
            suffix=".vrt.tmp",
            dir=output_dir,
        )
        os.close(fd)

        try:
            with open(tmp, "w") as dst:
                dst.write("\n".join(xml))
            os.replace(tmp, self.output)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def _footprint(self, path):
        from shapely.geometry import shape
        from shapely.ops import unary_union

        with rasterio.open(path) as src:
            scale = min(
                1.0,
                self.vector_max_size / max(src.width, src.height),
            )

            width = max(1, int(round(src.width * scale)))
            height = max(1, int(round(src.height * scale)))

            dst_transform = src.transform * src.transform.scale(
                (src.width / width), (src.height / height)
            )

            data = np.zeros((height, width), dtype=src.dtypes[0])
            reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=src.crs,
                resampling=Resampling.max,
            )
            # transform = src.transform * Affine.scale(
            #     src.width / width,
            #     src.height / height,
            # )

        occupied = data > 0
        if not np.any(occupied):
            return None

        geometries = [
            shape(geom)
            for geom, value in shapes(
                occupied.astype("uint8"),
                mask=occupied,
                transform=dst_transform,
            )
            if value
        ]

        if not geometries:
            return None

        geom = unary_union(geometries)

        # Preserve the overview-cell geometry by default and only
        # simplify when explicitly requested.
        if self.vector_simplify > 0:
            tolerance = (
                max(abs(dst_transform.a), abs(dst_transform.e)) * self.vector_simplify
            )
            geom = geom.simplify(tolerance, preserve_topology=True)

        return geom

    @staticmethod
    def _parse_group_by(group_by):
        """Normalize a grouping specification to a list of field names."""
        if group_by in (None, "", False):
            return []

        if isinstance(group_by, str):
            # Recipe arguments conventionally use slash-separated values.
            values = [value.strip() for value in group_by.split("/") if value.strip()]
        else:
            values = [str(value).strip() for value in group_by if str(value).strip()]

        return [value.upper() for value in values]

    @staticmethod
    def _aggregate_values(series, field):
        """Aggregate metadata conservatively for a dissolved vector group."""
        values = [
            value
            for value in series.dropna().tolist()
            if str(value) not in ("", "None", "Unknown")
        ]
        if not values:
            return None

        unique = sorted({str(value) for value in values})
        if len(unique) == 1:
            return unique[0]

        upper = field.upper()

        if "DATE" in upper or "YEAR" in upper:
            ordered = sorted(unique)
            return f"{ordered[0]} - {ordered[-1]}"

        if "URL" in upper:
            domains = sorted(
                {
                    urlparse(value).netloc
                    if urlparse(value).scheme and urlparse(value).netloc
                    else value
                    for value in unique
                }
            )
            return ", ".join(domains)

        return ", ".join(unique)

    def _group_vector_records(self, gdf):
        """Optionally dissolve vector records while preserving source identity semantics."""
        # Always provide one stable visualization key.
        if not self.group_by:
            gdf["GROUP_ID"] = gdf["SOURCE_ID"].astype(str)
            gdf["SOURCE_COUNT"] = 1
            return gdf

        work = gdf.copy()
        group_fields = []

        for field in self.group_by:
            if field not in work.columns:
                logger.warning(
                    "Vector group_by field %r is unavailable; falling back to SOURCE_ID",
                    field,
                )
                work[field] = None

            # Missing grouping metadata must remain source-specific rather than all
            # dissolving together into a single NULL group.
            fallback = work["SOURCE_ID"].astype(str)
            values = work[field].astype("object")
            missing = values.isna() | values.astype(str).isin(["", "None", "Unknown"])
            work[field] = values.where(~missing, fallback)
            group_fields.append(field)

        work["GROUP_ID"] = work[group_fields].astype(str).agg(" | ".join, axis=1)

        rows = []
        for group_id, frame in work.groupby("GROUP_ID", sort=True, dropna=False):
            geometry = frame.geometry.union_all()
            record = {
                "GROUP_ID": str(group_id),
                "SOURCE_COUNT": int(len(frame)),
                "geometry": geometry,
            }

            # Preserve explicit grouping fields as first-class metadata.
            for field in group_fields:
                record[field] = self._aggregate_values(frame[field], field)

            # Aggregate all remaining metadata. SOURCE_ID is intentionally not
            # expanded into a potentially enormous comma-separated list.
            for field in frame.columns:
                if field in {"geometry", "GROUP_ID", "SOURCE_ID", *group_fields}:
                    continue
                record[field] = self._aggregate_values(frame[field], field)

            rows.append(record)

        import geopandas as gpd

        return gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)

    def _write_qml_style(self, gdf):
        """Write a sibling QGIS categorized style for the vector output."""
        if not self.vector_output or not self.qgis_style:
            return

        field = self.qgis_style_field
        if self.group_by and field == "SOURCE_ID":
            field = "GROUP_ID"

        if field not in gdf.columns:
            logger.warning(
                "Cannot build QGIS style for %s: field %r is missing",
                self.vector_output,
                field,
            )
            return

        import xml.sax.saxutils as saxutils

        palette = [
            "228,26,28,150",
            "55,126,184,150",
            "77,175,74,150",
            "152,78,163,150",
            "255,127,0,150",
            "255,255,51,150",
            "166,86,40,150",
            "247,129,191,150",
            "153,153,153,150",
            "102,194,165,150",
            "252,141,98,150",
            "141,160,203,150",
        ]

        values = sorted(
            {str(value) for value in gdf[field].dropna().tolist() if str(value)}
        )

        categories = []
        symbols = []

        for index, value in enumerate(values):
            escaped = saxutils.escape(value, {'"': "&quot;"})
            color = palette[index % len(palette)]
            categories.append(
                f'    <category value="{escaped}" symbol="{index}" '
                f'label="{escaped}" render="true"/>'
            )
            symbols.append(
                f"""      <symbol type="fill" name="{index}" alpha="1">
        <layer pass="0" class="SimpleFill" locked="0">
          <prop k="color" v="{color}"/>
          <prop k="outline_color" v="40,40,40,190"/>
          <prop k="outline_width" v="0.2"/>
          <prop k="style" v="solid"/>
        </layer>
      </symbol>"""
            )

        escaped_field = saxutils.escape(field, {'"': "&quot;"})
        qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0" styleCategories="Symbology">
  <previewExpression>"{escaped_field}"</previewExpression>
  <renderer-v2 type="categorizedSymbol" attr="{escaped_field}">
    <categories>
{chr(10).join(categories)}
    </categories>
    <symbols>
{chr(10).join(symbols)}
    </symbols>
  </renderer-v2>
</qgis>
"""

        qml_path = os.path.splitext(self.vector_output)[0] + ".qml"
        try:
            with open(qml_path, "w") as dst:
                dst.write(qml)
            logger.info(
                "QGIS categorized style saved to %s (field=%s)",
                qml_path,
                field,
            )
        except Exception as exc:
            logger.warning("Failed to write QGIS style %s: %s", qml_path, exc)

    def build_vector(self):
        if not self.vector_output:
            return

        import geopandas as gpd

        records = []

        logger.info(f"building vector output: {self.vector_output}")
        for dataset_id, path in self._valid_masks():
            geom = self._footprint(path)
            if geom is None or geom.is_empty:
                continue

            with rasterio.open(path) as src:
                tags = src.tags()

            record = {
                "SOURCE_ID": dataset_id,
                "geometry": geom,
            }
            record.update(tags)
            records.append(record)

        if not records:
            return

        crs = self.profile["crs"]
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
        gdf = self._group_vector_records(gdf)

        # Keep the canonical identifiers first for convenient GIS inspection.
        ordered = [
            name
            for name in ("GROUP_ID", "SOURCE_ID", "SOURCE_COUNT")
            if name in gdf.columns
        ]
        ordered += [
            name for name in gdf.columns if name not in ordered and name != "geometry"
        ]
        if "geometry" in gdf.columns:
            ordered.append("geometry")
        gdf = gdf[ordered]
        gdf = gdf.sort_values("GROUP_ID", kind="stable").reset_index(drop=True)

        gdf.to_file(
            self.vector_output,
            driver="GPKG",
            engine="pyogrio",
        )

        self._write_qml_style(gdf)

    def finalize(self):
        self.build_vrt()
        self.build_vector()


class SourceMasks(FetchHook):
    """Record all valid source coverage observed before stack reduction."""

    name = "source-masks"
    meta_stage = "stream"
    meta_category = "metadata"
    meta_desc = "Record observed source coverage."
    meta_aliases = ["source_masks"]

    def __init__(
        self,
        res="1s",
        output_dir=None,
        output="source_masks.vrt",
        vector_output=None,
        resume=True,
        vector_max_size=2048,
        vector_simplify=0.0,
        qgis_style=True,
        qgis_style_field="SOURCE_ID",
        group_by=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.res = str2inc(res)
        self.output = output
        self.output_dir = output_dir
        self.vector_output = vector_output
        self.resume = str2bool(resume)
        self.vector_max_size = int(vector_max_size)
        self.vector_simplify = max(0.0, float(vector_simplify))
        self.qgis_style = str2bool(qgis_style)
        self.qgis_style_field = str(qgis_style_field)
        self.group_by = group_by

        self._masks = None

    def _init_masks(self, region):
        if self._masks:
            return

        self._masks = MaskSet(
            region,
            self.res,
            self.output,
            output_dir=self.output_dir,
            vector_output=self.vector_output,
            mask_type="SOURCE_MASK",
            resume=self.resume,
            vector_max_size=self.vector_max_size,
            vector_simplify=self.vector_simplify,
            qgis_style=self.qgis_style,
            qgis_style_field=self.qgis_style_field,
            group_by=self.group_by,
        )

    def run(self, entries):
        if not self._masks and entries:
            region = next(
                (mod.region for mod, _ in entries if getattr(mod, "region", None)),
                None,
            )

            if region:
                self._init_masks(region)

        if not self._masks:
            return entries

        for mod, entry in entries:
            if not self.is_point_stream(entry):
                continue

            dataset_id = source_id(entry)
            src_name = os.path.basename(entry.get("dst_fn", dataset_id))
            base = os.path.splitext(src_name)[0]

            tags = {
                "MODULE": getattr(mod, "name", None),
                "DATASET": getattr(
                    mod,
                    "title",
                    getattr(mod, "name", None),
                ),
                "CATEGORY": getattr(mod, "meta_category", None),
                "AGENCY": getattr(mod, "meta_agency", None),
                "DATATYPE": entry.get("data_type"),
                "RESOLUTION": getattr(mod, "meta_resolution", None),
                "URL": entry.get("url"),
                "WEIGHT": getattr(mod, "weight", 1.0),
            }

            if isinstance(entry.get("metadata"), dict):
                tags.update({str(k).upper(): v for k, v in entry["metadata"].items()})

            path = self._masks.register(
                dataset_id,
                description=base,
                tags=tags,
            )

            stream = entry.get("stream")
            entry["stream"] = self._intercept(
                stream,
                path,
                mod.region,
            )

            entry.setdefault("artifacts", {})[self.name] = path

        return entries

    def _intercept(self, stream, path, region):
        pixel_binner = PointPixels(
            src_region=region,
            x_size=self._masks.xcount,
            y_size=self._masks.ycount,
            dst_gt=self._masks.dst_gt,
        )

        for chunk in stream:
            has_data, sub_win, _ = pixel_binner.coverage(chunk)

            if has_data is not None:
                col, row, width, height = sub_win
                window = Window(col, row, width, height)

                MaskSet.update(path, window, has_data)

            yield chunk

    def teardown(self):
        if self._masks:
            self._masks.finalize()
