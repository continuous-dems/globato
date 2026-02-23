#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.formats.lidar
~~~~~~~~~~~~~

This hook converts lidar to a point stream.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
import numpy as np
import laspy as lp
import subprocess
import json

from fetchez.utils import str_or
from fetchez.hooks import FetchHook

logger = logging.getLogger(__name__)


class LASReader:
    """Process LAS/LAZ lidar files using laspy."""

    def __init__(
            self,
            src_fn: str,
            classes="2/29/40",
            **kwargs,
    ):

        self.src_fn = src_fn
        try:
            if isinstance(str_or(classes), str):
                self.classes = [int(x) for x in str(classes).split('/')]
            elif isinstance(classes, (list, tuple)):
                self.classes = [int(x) for x in classes]
            else:
                self.classes = []
        except Exception as e:
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
                    # Record ID 2112 is "OGC Coordinate System WKT"
                    if vlr.record_id == 2112:
                        try:
                            srs = vlr.string
                            if isinstance(srs, bytes):
                                return srs.decode("utf-8").strip("\0")
                            return srs
                        except:
                            pass
        except Exception:
            pass

        return None


    def yield_chunks(self):
        """Yield points from local file using standard laspy."""

        try:
            with lp.open(self.src_fn) as lasf:
                for chunk in lasf.chunk_iterator(2_000_000):
                    if self.classes:
                        mask = np.isin(chunk.classification, self.classes)
                        points_x = chunk.x[mask]
                        points_y = chunk.y[mask]
                        points_z = chunk.z[mask]
                    else:
                        points_x = chunk.x
                        points_y = chunk.y
                        points_z = chunk.z

                    if len(points_x) == 0: continue

                    w = np.ones_like(points_z)
                    u = np.zeros_like(points_z)

                    dataset = np.column_stack((points_x, points_y, points_z, w, u))
                    points = np.rec.fromrecords(dataset, names="x, y, z, w, u")
                    yield points
        except Exception as e:
            logger.error(f"LAS/Z processing failed for {self.src_fn}: {e}")
            return None


class LASStream(FetchHook):
    """Process raw las/laz files into XYZ format.
    Updates the entry so downstream tools see the .xyz file, not the .laz.
    """

    name = "las_stream"
    stage = "file"
    desc = "stream las data through laspy"
    category = "format-stream"

    def __init__(self, classes="2/7/29/40", **kwargs):
        super().__init__(**kwargs)
        self.classes = classes


    def run(self, entries):
        new_entries = []
        for mod, entry in entries:
            src = entry["dst_fn"]

            if not os.path.exists(src):
                new_entries.append((mod, entry))
                continue

            # Simple check for LAS/LAZ extension
            if not src.lower().endswith((".las", ".laz")):
                new_entries.append((mod, entry))
                continue

            try:
                reader = XYZReader(src, **self.params)
                # Get EPSG for metadata (unused right now but good to have)
                #src_epsg = reader.get_epsg()

                entry["stream"] = reader.yield_chunks()
                entry["stream_type"] = "xyz_recarray"
                entry["las_classes"] = self.classes

            except Exception as e:
                logger.error(f"LAS extraction failed for {src}: {e}")

            new_entries.append((mod, entry))

        return new_entries
