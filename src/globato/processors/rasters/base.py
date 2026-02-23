#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.rasters.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Base classes for Raster operations.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
import rasterio
from rasterio.windows import Window
from fetchez.hooks import FetchHook

logger = logging.getLogger(__name__)

class RasterHook(FetchHook):
    """Base class for hooks that operate on raster files.
    Abstracts pipeline iteration and output file generation.
    """

    stage = "post"
    category = "raster-op"
    # Subclasses should define their own default suffix
    default_suffix = "_processed"

    def __init__(self, output=None, suffix=None, **kwargs):
        super().__init__(**kwargs)
        self.output = output
        self.suffix = suffix or self.default_suffix

    def run(self, entries):
        new_entries = []
        for mod, entry in entries:
            src_fn = entry.get("dst_fn")

            if not src_fn or not os.path.exists(src_fn) or not src_fn.lower().endswith(".tif"):
                new_entries.append((mod, entry))
                continue

            if self.output:
                dst_fn = self.output
            else:
                base, ext = os.path.splitext(src_fn)
                dst_fn = f"{base}{self.suffix}{ext}"

            logger.info(f"Running {self.name} on {os.path.basename(src_fn)}")

            try:
                success = self.process_raster(src_fn, dst_fn, entry, entries)
                if success:
                    entry["src_fn"] = src_fn
                    entry["dst_fn"] = dst_fn
                    entry.setdefault("artifacts", {})[self.name] = dst_fn
            except Exception as e:
                logger.error(f"RasterHook {self.name} failed on {src_fn}: {e}")

            new_entries.append((mod, entry))

        return new_entries

    def process_raster(self, src_path, dst_path, entry, entries):
        """Must be implemented by subclasses. Return True if successful."""

        raise NotImplementedError

    def yield_buffered_windows(self, src, buffer_size=0):
        """Helper to yield (target_window, buffered_window) for chunked processing."""

        for block_index, window in src.block_windows(1):
            if buffer_size == 0:
                yield window, window
                continue

            row_start = max(0, window.row_off - buffer_size)
            col_start = max(0, window.col_off - buffer_size)
            row_stop = min(src.height, window.row_off + window.height + buffer_size)
            col_stop = min(src.width, window.col_off + window.width + buffer_size)

            buffered_window = Window.from_slices((row_start, row_stop), (col_start, col_stop))
            yield window, buffered_window
