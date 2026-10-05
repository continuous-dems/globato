#!/usr/bin/env python

"""
globato.hooks.metadata.globato_inf
~~~~~~~~~~~~~

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import json
import os

import numpy as np
from fetchez.hooks import FetchHook


def generate_stream_inf(stream, out_path=None, **kwargs):
    """Iterates over a point stream, calculates stats, yields data,
    and optionally writes a legacy .inf file.
    """

    total_pts = 0
    minmax = [np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf]

    for chunk in stream:
        if chunk is None or len(chunk) == 0:
            continue

        chunk_len = len(chunk)
        total_pts += chunk_len

        c_min_x, c_max_x = np.min(chunk["x"]), np.max(chunk["x"])
        c_min_y, c_max_y = np.min(chunk["y"]), np.max(chunk["y"])
        c_min_z, c_max_z = np.min(chunk["z"]), np.max(chunk["z"])

        minmax[0] = min(minmax[0], c_min_x)  # W
        minmax[1] = max(minmax[1], c_max_x)  # E
        minmax[2] = min(minmax[2], c_min_y)  # S
        minmax[3] = max(minmax[3], c_max_y)  # N
        minmax[4] = min(minmax[4], c_min_z)  # Z-min
        minmax[5] = max(minmax[5], c_max_z)  # Z-max

        yield chunk

    w, e, s, n = minmax[0], minmax[1], minmax[2], minmax[3]
    wkt = f"POLYGON (({w} {n}, {e} {n}, {e} {s}, {w} {s}, {w} {n}))"

    meta = {
        "numpts": int(total_pts),
        "minmax": [float(x) for x in minmax],  # JSON requires python floats
        "wkt": wkt,
    }

    for key in kwargs:
        meta[key] = kwargs[key]

    if out_path:
        try:
            with open(out_path, "w") as f:
                json.dump(meta, f, indent=4)
        except Exception:
            pass

    return meta


class GlobatoInfo(FetchHook):
    """Fetchez Hook Adapter.

    Wraps the core `generate_stream_inf` generator so it can be used
    seamlessly inside a YAML recipe or Fetchez pipeline.
    """

    name = "stream-inf"
    meta_desc = "Generate .inf metadata (minmax, count, wkt)."
    meta_stage = "stream"
    meta_category = "metadata"
    meta_aliases = ["stream_inf"]

    def run(self, entries):
        for mod, entry in entries:
            stream = entry.get("stream")
            dst_fn = entry.get("dst_fn")

            if stream:
                inf_out = dst_fn + ".inf" if dst_fn else None

                entry["stream"] = generate_stream_inf(
                    stream,
                    out_path=inf_out,
                    name=os.path.basename(dst_fn),
                    src_srs=entry.get("src_srs", "Unknown"),
                    format="globato_stream",
                )

        return entries
