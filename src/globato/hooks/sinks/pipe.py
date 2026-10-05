#!/usr/bin/env python

"""
globato.hooks.sinks.pipe
~~~~~~~~~~~~~~~~~~~~~~~

pipe the stream to xyz (stdout)

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import sys

import numpy as np
from fetchez.hooks import FetchHook

logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger(__name__)


class XYZPrinter(FetchHook):
    """Sink Hook: Prints the XYZ stream to stdout.

    Useful for piping to other tools like GMT, MB-System, or text files.
    If the input stream is 'xyz_recarray', it prints the points,
    if the input stream is 'point_pixels_arrays' it prints the y/x/z pixel locations.

    Usage:
      --hook pipe_xyz
      --hook pipe_xyz:fmt=%.4f:delimiter=,
    """

    name = "stream-dump-xyz"
    meta_stage = "stream"
    meta_desc = "Send the point-stream xyz data to stdout"
    meta_category = "point-stream"
    meta_requires = "point-stream"
    meta_aliases = ["stream_pipe_xyz"]

    def __init__(self, fmt="%.6f", delimiter=" ", **kwargs):
        super().__init__(**kwargs)
        self.fmt = fmt
        self.delimiter = delimiter

    def run(self, entries):
        for mod, entry in entries:
            if not self.has_stream(entry):
                continue

            stream = entry["stream"]
            try:
                if self.is_point_stream(entry):
                    for chunk in stream:
                        columns = [chunk["x"], chunk["y"], chunk["z"]]

                        if "w" in chunk.dtype.names:
                            columns.append(chunk["w"])
                        if "u" in chunk.dtype.names:
                            columns.append(chunk["u"])
                        if "classification" in chunk.dtype.names:
                            columns.append(chunk["classification"])
                        if "confidence" in chunk.dtype.names:
                            columns.append(chunk["confidence"])

                        data = np.column_stack(columns)
                        np.savetxt(
                            sys.stdout, data, fmt=self.fmt, delimiter=self.delimiter
                        )

                elif self.is_raster_stream(entry):
                    stream = entry.get("stream")

                    # Pop the profile off the top of the generator
                    _profile = next(stream)
                    # print(profile)
                    for window, buff_win, data, ndv, transform in stream:
                        # Grab the Z band (first band)
                        z_data = data[0] if data.ndim == 3 else data

                        # Find valid pixels
                        if ndv is not None:
                            valid_mask = z_data != ndv
                        else:
                            valid_mask = ~np.isnan(z_data)

                        # Generate pixel coordinates
                        rows, cols = np.where(valid_mask)

                        # Convert to geographic coordinates using the chunk's transform
                        import rasterio

                        xs, ys = rasterio.transform.xy(transform, rows, cols)

                        z_vals = z_data[valid_mask]

                        columns = [np.array(xs), np.array(ys), z_vals]
                        data_out = np.column_stack(columns)

                        # Print to stdout
                        self.fmt = ["%.6f", "%.6f", "%.6f"]
                        np.savetxt(
                            sys.stdout, data_out, fmt=self.fmt, delimiter=self.delimiter
                        )
                # elif stream_type == "point_pixels_arrays":
                #     for arrs, srcwin, gt in stream:

                #         x_vals = arrs["pixel_x"].astype(int)
                #         y_vals = arrs["pixel_y"].astype(int)
                #         z_vals = arrs["z"][y_vals, x_vals]

                #         columns = [y_vals, x_vals, z_vals]

                #         data = np.column_stack(columns)
                #         self.fmt = ["%d", "%d", "%.6f"]
                #         np.savetxt(sys.stdout, data, fmt=self.fmt, delimiter=self.delimiter)

            except OSError:
                try:
                    sys.stdout.close()
                except Exception:
                    pass
                # return entries
            except BrokenPipeError:
                try:
                    sys.stdout.close()
                except Exception:
                    pass
                # return entries
            except Exception:
                # logger.error(f"Error piping XYZ: {e}\n")
                try:
                    sys.stdout.close()
                except Exception:
                    pass

            del entry["stream"]
            del entry["stream_type"]

        return entries
