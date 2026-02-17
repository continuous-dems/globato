#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.metadata.globato_inf
~~~~~~~~~~~~~

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import json
import numpy as np
from fetchez.hooks import FetchHook


class GlobatoInfo(FetchHook):
    """Generates a legacy-compatible .inf metadata file by inspecting the data stream.
    Calculates Min/Max, Point Count, and Bounding Box in-flight.
    """

    name = "stream_inf"
    desc = "Generate .inf metadata (minmax, count, wkt)."
    stage = "file"
    category = "streams"

    def run(self, entries):
        for mod, entry in entries:
            stream = entry.get('stream')
            if not stream:
                continue

            entry['stream'] = self._inf_stream_wrapper(stream, entry)

        return entries

    def _inf_stream_wrapper(self, stream, entry):
        """Iterate over the stream, calculate stats, yield data transparently,
        and write the .inf file at the end.
        """

        # --- Initialize INF ---
        total_pts = 0
        minmax = [np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf]

        try:
            for chunk in stream:
                if chunk is None or len(chunk) == 0:
                    continue

                chunk_len = len(chunk)
                total_pts += chunk_len

                c_min_x, c_max_x = np.min(chunk['x']), np.max(chunk['x'])
                c_min_y, c_max_y = np.min(chunk['y']), np.max(chunk['y'])
                c_min_z, c_max_z = np.min(chunk['z']), np.max(chunk['z'])

                minmax[0] = min(minmax[0], c_min_x) # W
                minmax[1] = max(minmax[1], c_max_x) # E
                minmax[2] = min(minmax[2], c_min_y) # S
                minmax[3] = max(minmax[3], c_max_y) # N
                minmax[4] = min(minmax[4], c_min_z) # Z-min
                minmax[5] = max(minmax[5], c_max_z) # Z-max

                yield chunk

            # --- Write INF ---
            self._write_inf(entry, total_pts, minmax)

        except Exception as e:
            raise e

    def _write_inf(self, entry, numpts, minmax):
        """Write the collected inf to disk."""

        dst_fn = entry.get('dst_fn')
        if not dst_fn:
            return

        inf_fn = dst_fn + ".inf"

        w, e, s, n = minmax[0], minmax[1], minmax[2], minmax[3]
        wkt = (f"POLYGON (({w} {n}, {e} {n}, {e} {s}, {w} {s}, {w} {n}))")

        meta = {
            "name": os.path.basename(dst_fn),
            "numpts": int(numpts),
            "minmax": [float(x) for x in minmax], # JSON requires python floats
            "wkt": wkt,
            "src_srs": entry.get("src_srs", "Unknown"),
            "format": "globato_stream"
        }

        try:
            with open(inf_fn, 'w') as f:
                json.dump(meta, f, indent=4)
        except Exception:
            pass
