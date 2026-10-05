#!/usr/bin/env python

"""
globato.hooks.sinks.xyz_writer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Writes the point stream to an ASCII XYZ file inline.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import os
import sys

import numpy as np
from fetchez.hooks import FetchHook

logger = logging.getLogger(__name__)


class XYZWrite(FetchHook):
    """Saves stream to an ASCII XYZ file and passes the stream onward.

    Args:
        output_path (str): Static path or template (e.g., '{base}_out.xyz').
        columns (list): Fields to export. Defaults to ["x", "y", "z"].
        fmt (str): Float formatting. Defaults to '%.6f'.
    """

    name = "xyz-write"
    meta_stage = "stream"
    meta_category = "stream-sink"
    meta_desc = "Write a point stream to an xyz file."
    meta_aliases = ["xyz_write", "write_xyz"]

    def __init__(self, output_path=None, columns=None, fmt="%.6f", **kwargs):
        super().__init__(**kwargs)
        self.output_path = output_path
        self.fmt = fmt
        self.columns = columns or ["x", "y", "z"]

        if (
            self.output_path
            and "{" not in self.output_path
            and os.path.exists(self.output_path)
        ):
            open(self.output_path, "w").close()

    def _tap_stream(self, stream, out_fn):
        """Lazy generator that writes chunks to disk while passing them downstream."""

        logger.debug(f"Tapping stream to XYZ: {out_fn or 'STDOUT'}")

        out_port = open(out_fn, "a") if out_fn else sys.stdout
        total_pts = 0

        try:
            for chunk in stream:
                if chunk is not None and len(chunk) > 0:
                    total_pts += len(chunk)

                    cols_to_stack = [
                        chunk[c] for c in self.columns if c in chunk.dtype.names
                    ]

                    if cols_to_stack:
                        data = np.column_stack(cols_to_stack)
                        np.savetxt(out_port, data, fmt=self.fmt, delimiter=" ")

                yield chunk
        finally:
            if out_fn and out_port != sys.stdout:
                out_port.close()

        logger.debug(f"Finished writing {total_pts} points to {out_fn or 'STDOUT'}")

    def run(self, entries):
        for mod, entry in entries:
            # Catch raster streams and auto-convert them if this hook is placed too early
            if self.is_raster_stream(entry):
                from globato.hooks.transforms.point_pixels import PixelsToPoints

                p2p = PixelsToPoints()
                p2p.run([(mod, entry)])

            if not self.has_stream(entry):
                continue

            stream = entry["stream"]
            src_fn = entry.get("dst_fn", "unknown")

            out_fn = None
            if self.output_path:
                base = os.path.splitext(os.path.basename(src_fn))[0]
                out_fn = self.output_path.format(base=base, name=mod.name)

                if not os.path.isabs(out_fn):
                    if "{" in self.output_path and src_fn != "unknown":
                        out_dir = os.path.dirname(src_fn)
                    else:
                        out_dir = os.getcwd()

                    out_fn = os.path.join(out_dir, out_fn)

            entry["stream"] = self._tap_stream(stream, out_fn)

            if out_fn:
                entry.setdefault("artifacts", {})[self.name] = out_fn

        return entries
