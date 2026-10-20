#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.hooks.sinks.raster_writer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Writes the stream to a raster

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import shutil
import logging
import rasterio
from fetchez.hooks import FetchHook
from fetchez.utils import str2bool
from globato.hooks.rasters.base import carry_metadata

logger = logging.getLogger(__name__)


class RasterWrite(FetchHook):
    """Universal sink that writes a TIFF. Can act as a terminal sink or an inline tap."""

    name = "raster-write"
    meta_stage = "collection"  # set to 'stream' to dump from stream
    meta_category = "raster-stream"
    meta_desc = "Write a raster-stream to raster"
    meta_requires = "raster-stream"
    meta_aliases = ["raster-stream-write-raster", "raster_write"]

    def __init__(
        self,
        output_path=None,
        suffix="_final",
        artifact_id=None,
        inline=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_path = output_path
        self.suffix = suffix
        self.artifact_id = artifact_id or self.name
        self.inline = str2bool(inline)

    def _write_stream(self, stream, dst_fn, source_path=None):
        """Generator that intercepts the stream, writes to disk, and yields onward."""

        profile = next(stream)
        yield profile

        with rasterio.open(dst_fn, "w", **profile) as dst:
            for window, buff_win, data, ndv, transform in stream:
                y_off = window.row_off - buff_win.row_off
                x_off = window.col_off - buff_win.col_off

                if data.ndim == 3:
                    final_chunk = data[
                        :, y_off : y_off + window.height, x_off : x_off + window.width
                    ]
                    dst.write(final_chunk, window=window)
                else:
                    final_chunk = data[
                        y_off : y_off + window.height, x_off : x_off + window.width
                    ]
                    dst.write(final_chunk, 1, window=window)

                yield window, buff_win, data, ndv, transform

        if source_path:
            carry_metadata(source_path, dst_fn)

    def run(self, entries):
        for mod, entry in entries:
            stream = entry.get("stream") or entry.get("raster_stream")
            base = os.path.splitext(entry.get("dst_fn", "out"))[0]
            dst_fn = self.output_path or f"{base}{self.suffix}.tif"

            if stream is not None and (
                self.is_raster_stream(entry) or entry.get("raster_stream") is not None
            ):
                # It's an active stream. Write it chunk by chunk!
                writer_generator = self._write_stream(
                    stream, dst_fn, source_path=entry.get("dst_fn")
                )

                if self.inline:
                    entry["stream"] = writer_generator
                    entry["stream_type"] = "raster-stream"
                    entry.pop("raster_stream", None)
                else:
                    logger.info(f"Writing raster stream to disk: {dst_fn}")
                    for _ in writer_generator:
                        pass
                    entry.pop("raster_stream", None)
                    entry.pop("stream", None)
                    entry["dst_fn"] = dst_fn

            else:
                # The stream was drained by a global hook!
                src_fn = entry.get("dst_fn")
                if dst_fn and src_fn and src_fn != dst_fn:
                    logger.info(f"Promoting final artifact to: {dst_fn}")
                    shutil.copy2(src_fn, dst_fn)
                    entry["dst_fn"] = dst_fn

            entry.setdefault("artifacts", {})[self.artifact_id] = entry.get("dst_fn")

        return entries
