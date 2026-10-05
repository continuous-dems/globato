import logging

import rasterio

from globato.hooks.rasters.base import RasterStreamHook

logger = logging.getLogger(__name__)


class RasterStreamInit(RasterStreamHook):
    """Initiates a streaming generator directly from a GeoTIFF."""

    name = "raster_stream"
    meta_desc = "Initiates a raster-stream."
    meta_stage = "manifest"  # post
    meta_category = "raster-stream"

    def _yield_chunks(self, src_path):
        with rasterio.open(src_path) as src:
            yield src.profile.copy()
            for window, buff_win in self.yield_buffered_windows(
                src, buffer_size=self.buffer
            ):
                data = src.read(window=buff_win)
                transform = rasterio.windows.transform(buff_win, src.transform)
                yield window, buff_win, data, src.nodata, transform

    def run(self, entries):
        for mod, entry in entries:
            entry.pop("stream", None)

            src_fn = entry.get("dst_fn")
            if src_fn and src_fn.endswith(".tif"):
                entry["raster_stream"] = self._yield_chunks(src_fn)
                entry["stream_type"] = "raster"
        return entries
