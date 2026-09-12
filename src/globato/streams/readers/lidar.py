#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.streams.readers.lidar
~~~~~~~~~~~~~

This readers lidar to a point stream.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging

import numpy as np
import laspy as lp
from rasterio.warp import transform_bounds
from pyproj import CRS

from fetchez.utils import str_or, int_or, str2bool

from .base import BaseGlobatoReader

logger = logging.getLogger(__name__)


def _horizontal_crs(crs):
    """Return the horizontal component of a CRS."""
    crs = CRS.from_user_input(crs)

    if crs.is_compound:
        for component in crs.sub_crs_list:
            if component.is_projected or component.is_geographic:
                return component

    if len(crs.axis_info) == 3:
        return crs.to_2d()

    return crs


class LASReader(BaseGlobatoReader):
    """Process LAS/LAZ and COPC lidar files using laspy."""

    name = "lidar-point-reader"
    meta_category = "point-stream"
    meta_dtype = "lidar"
    meta_desc = "Read lidar data through laspy into a point stream"
    meta_extensions = ["las", "laz"]

    def __init__(
        self,
        path: str,
        classes="2/29/40",
        chunk_size=1000000,
        in_memory=False,
        **kwargs,
    ):
        super().__init__(path, **kwargs)
        self.src_fn = path
        self.chunk_size = int_or(chunk_size, 1000000)
        self.in_memory = str2bool(in_memory)

        try:
            if isinstance(str_or(classes), str):
                self.classes = [int(x) for x in str(classes).split("/")]
            elif isinstance(classes, (list, tuple)):
                self.classes = [int(x) for x in classes]
            else:
                self.classes = []
        except Exception:
            self.classes = []

    def get_srs(self):
        """Attempt to parse EPSG/WKT from LAS Header using laspy."""

        try:
            with lp.open(self.src_fn) as lasf:
                try:
                    crs = lasf.header.parse_crs()
                    if crs is not None:
                        return crs.to_wkt()
                except Exception:
                    pass

                # Manual VLR check
                for vlr in lasf.header.vlrs:
                    if vlr.record_id == 2112:
                        try:
                            srs = vlr.string
                            if isinstance(srs, bytes):
                                return srs.decode("utf-8").strip("\0")
                            return srs
                        except Exception:
                            pass
        except Exception:
            pass

        return None

    def _yield_raw_chunks(self):
        """Yield points from local file using standard laspy."""

        try:
            with lp.open(self.src_fn) as lasf:
                is_copc = hasattr(lasf, "query")

                if self.src_fn.lower().endswith(".copc.laz") and not is_copc:
                    logger.debug(
                        f"File {self.src_fn} is named .copc.laz but lacks COPC "
                        "structural VLRs. Falling back to standard chunked LAZ reading."
                    )

                # Native LAS bounds.
                las_w = lasf.header.x_min
                las_e = lasf.header.x_max
                las_s = lasf.header.y_min
                las_n = lasf.header.y_max

                native_clip_bounds = None

                if self.region is not None:
                    req_w, req_e, req_s, req_n = self.region

                    region_srs = getattr(self.region, "srs", None) or "EPSG:4326"

                    try:
                        region_crs = _horizontal_crs(region_srs)

                        las_srs = self.get_srs()
                        if las_srs is None:
                            raise ValueError("LAS/LAZ file does not define a CRS.")

                        las_crs = _horizontal_crs(las_srs)

                        if las_crs == region_crs:
                            region_las_w = las_w
                            region_las_s = las_s
                            region_las_e = las_e
                            region_las_n = las_n
                        else:
                            (
                                region_las_w,
                                region_las_s,
                                region_las_e,
                                region_las_n,
                            ) = transform_bounds(
                                las_crs,
                                region_crs,
                                las_w,
                                las_s,
                                las_e,
                                las_n,
                                densify_pts=21,
                            )

                        # Reject the entire file if its extent does not intersect the
                        # requested region.
                        if (
                            region_las_w > req_e
                            or region_las_e < req_w
                            or region_las_s > req_n
                            or region_las_n < req_s
                        ):
                            logger.debug(
                                f"Skipping {self.src_fn}: Bounding box falls "
                                "outside requested region."
                            )
                            return

                        clip_w = max(req_w, region_las_w)
                        clip_e = min(req_e, region_las_e)
                        clip_s = max(req_s, region_las_s)
                        clip_n = min(req_n, region_las_n)

                        contains_file = (
                            req_w <= region_las_w
                            and req_e >= region_las_e
                            and req_s <= region_las_s
                            and req_n >= region_las_n
                        )

                        if contains_file:
                            native_clip_bounds = (las_w, las_e, las_s, las_n)

                        elif region_crs == las_crs:
                            native_clip_bounds = (
                                clip_w,
                                clip_e,
                                clip_s,
                                clip_n,
                            )

                        else:
                            native_w, native_s, native_e, native_n = transform_bounds(
                                region_crs,
                                las_crs,
                                clip_w,
                                clip_s,
                                clip_e,
                                clip_n,
                                densify_pts=21,
                            )

                            native_clip_bounds = (
                                native_w,
                                native_e,
                                native_s,
                                native_n,
                            )

                    except Exception as exc:
                        logger.debug(
                            f"Could not determine spatial intersection for "
                            f"{self.src_fn}: {exc}"
                        )

                        # If CRS fails, don't discard the file.
                        native_clip_bounds = None

                # In-memory mode.
                if self.in_memory and not is_copc:
                    logger.debug(f"Reading {self.src_fn} using in-memory mode...")
                    chunk_iter = [lasf.read()]

                # COPC can perform the clipping itself, but its query coordinates
                # must be in the native LAS CRS.
                elif is_copc and native_clip_bounds is not None:
                    native_w, native_e, native_s, native_n = native_clip_bounds

                    mins = np.array([native_w, native_s])
                    maxs = np.array([native_e, native_n])

                    logger.debug(
                        f"Using COPC spatial query for {self.src_fn}: "
                        f"{native_clip_bounds}"
                    )

                    try:
                        chunk_iter = lasf.query(mins, maxs)
                    except Exception as exc:
                        logger.debug(
                            f"COPC query failed; falling back to standard chunking: {exc}"
                        )
                        chunk_iter = lasf.chunk_iterator(self.chunk_size)

                else:
                    chunk_iter = lasf.chunk_iterator(self.chunk_size)

                full_dtype = [
                    ("x", "f8"),
                    ("y", "f8"),
                    ("z", "f4"),
                    ("w", "f4"),
                    ("u", "f4"),
                    ("classification", "u1"),
                    ("confidence", "i2"),
                ]

                for chunk in chunk_iter:
                    if len(chunk) == 0:
                        continue

                    if self.classes:
                        class_mask = np.isin(
                            chunk.classification,
                            self.classes,
                        )
                        if not np.any(class_mask):
                            continue
                        chunk = chunk[class_mask]

                    # Standard LAS/LAZ files can't query spatially at read time,
                    # so clip each chunk using native coordinates.
                    if native_clip_bounds is not None and not is_copc:
                        native_w, native_e, native_s, native_n = native_clip_bounds

                        x_vals = chunk.x
                        y_vals = chunk.y

                        spatial_mask = (
                            (x_vals >= native_w)
                            & (x_vals <= native_e)
                            & (y_vals >= native_s)
                            & (y_vals <= native_n)
                        )

                        if not np.any(spatial_mask):
                            continue

                        chunk = chunk[spatial_mask]

                    count = len(chunk)
                    if count == 0:
                        continue

                    points = np.empty(count, dtype=full_dtype)
                    points["x"] = chunk.x
                    points["y"] = chunk.y
                    points["z"] = chunk.z
                    points["classification"] = chunk.classification
                    points["w"] = 1.0
                    points["u"] = 0.0
                    points["confidence"] = 1

                    if self.in_memory and count > self.chunk_size:
                        for i in range(0, count, self.chunk_size):
                            yield points[i : i + self.chunk_size]
                    else:
                        yield points

        except Exception as exc:
            logger.error(f"LAS/Z processing failed for {self.src_fn}: {exc}")
            return None
