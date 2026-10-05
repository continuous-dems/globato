#!/usr/bin/env python

"""
globato.streams.readers.datalist
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A meta-reader for legacy CUDEM .datalist files.
Delegates reading to the StreamFactory while enforcing spatial indices
and hierarchical dataset weights.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import json
import logging
import os
import shlex

import numpy as np

from globato.utils import add_field_to_recarray

from .base import BaseGlobatoReader

try:
    import h3

    HAS_H3 = True
except ImportError:
    HAS_H3 = False

logger = logging.getLogger(__name__)


class DatalistReader(BaseGlobatoReader):
    name = "datalist-point-reader"
    meta_category = "point-stream"
    meta_dtype = "datalist"
    meta_desc = "Read datalist data into a point stream"
    meta_extensions = ["datalist"]

    def __init__(self, path, region=None, **kwargs):
        super().__init__(path, **kwargs)
        self.src_fn = os.path.abspath(path)
        self.region = region
        self.kwargs = kwargs

    def _get_entries(self):
        """Returns a list of dicts: {'path': str, 'data_type': str, 'weight': float, 'unc': float}"""

        entries = []
        h3_fn = f"{self.src_fn}.h3.json"

        if HAS_H3 and os.path.exists(h3_fn):
            try:
                with open(h3_fn, "r") as f:
                    h3_index = json.load(f)

                matched_indices = set()
                if self.region is None:
                    matched_indices = set(range(len(h3_index.get("entries", []))))
                else:
                    res = h3_index.get("resolution", 6)
                    geo_json_poly = {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [self.region.xmin, self.region.ymin],
                                [self.region.xmin, self.region.ymax],
                                [self.region.xmax, self.region.ymax],
                                [self.region.xmax, self.region.ymin],
                                [self.region.xmin, self.region.ymin],
                            ]
                        ],
                    }
                    query_cells = h3.polygon_to_cells(geo_json_poly, res)
                    idx_map = h3_index.get("index", {})
                    for cell in query_cells:
                        if cell in idx_map:
                            matched_indices.update(idx_map[cell])

                base_dir = os.path.dirname(self.src_fn)
                for i in sorted(list(matched_indices)):
                    meta = h3_index["entries"][i]
                    path = meta.get("path")
                    if not os.path.isabs(path):
                        path = os.path.abspath(os.path.join(base_dir, path))
                    if os.path.exists(path):
                        entries.append(
                            {
                                "path": path,
                                "data_type": meta.get(
                                    "data_type", os.path.splitext(path)[-1]
                                ),
                                "weight": float(meta.get("weight", 1.0)),
                                "unc": float(meta.get("uncertainty", 0.0)),
                            }
                        )
                return entries
            except Exception as e:
                logger.warning(f"Failed to use H3 index {h3_fn}: {e}")

        base_dir = os.path.dirname(self.src_fn)
        with open(self.src_fn, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    parts = shlex.split(line, posix=False)
                    raw_path = parts[0].strip("\"'")
                    path = (
                        raw_path
                        if os.path.isabs(raw_path)
                        else os.path.abspath(os.path.join(base_dir, raw_path))
                    )

                    data_type = (
                        parts[1] if len(parts) > 1 else os.path.spitext(path)[-1]
                    )
                    weight = float(parts[2]) if len(parts) > 2 else 1.0
                    unc = float(parts[3]) if len(parts) > 3 else 0.0

                    if os.path.exists(path):
                        entries.append(
                            {
                                "path": path,
                                "data_type": data_type,
                                "weight": weight,
                                "unc": unc,
                            }
                        )
                except Exception as e:
                    logger.warning(f"Error parsing datalist line '{line}': {e}")

        return entries

    def get_srs(self):
        """Datalists don't have a single strict SRS, defer to individual files."""

        return

    def _yield_raw_chunks(self):
        # from globato.hooks.formats.stream_factory import StreamFactory
        # from fetchez.hooks.stream_init import DataStream
        from fetchez.registry import ProfileRegistry, ReaderRegistry

        ReaderRegistry.load_all()
        ProfileRegistry.load_all()

        entries = self._get_entries()
        logger.info(
            f"Datalist {os.path.basename(self.src_fn)} yielded {len(entries)} intersecting files."
        )

        for entry in entries:
            # reader = StreamFactory.get_reader(
            #    entry["path"], region=self.region, **self.kwargs
            # )
            reader = ReaderRegistry.get_reader(
                entry["path"], entry["data_type"], region=self.region, **self.kwargs
            )
            if not reader:
                continue

            for chunk in reader.yield_chunks():
                chunk = add_field_to_recarray(chunk, "w", float, 1.0)
                chunk = add_field_to_recarray(chunk, "u", float, 0.0)

                if entry["weight"] != 1.0:
                    chunk["w"] *= entry["weight"]

                if entry["unc"] != 0.0:
                    chunk["u"] = np.sqrt(chunk["u"] ** 2 + entry["unc"] ** 2)

                yield chunk
