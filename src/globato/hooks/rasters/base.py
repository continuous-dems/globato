#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.hooks.rasters.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unified architecture for Raster processing hooks.
Handles Streaming (Local/Chunked) and Global (Whole-File) operations.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
import shutil
import numpy as np
import rasterio
from rasterio.windows import Window
from pyogrio.raw import read
import shapely

from fetchez.spatial import parse_region
from fetchez.hooks import FetchHook
from fetchez.utils import float_or, parse_arg_to_list  # , inc2str

logger = logging.getLogger(__name__)


def _missing_metadata(src, dst):
    """What `src` carries that `dst` lacks: dataset tags, band tags, descriptions and units.

    Metadata describes a particular raster, so it is only carried when `dst` is that
    raster, modified, i.e. when it has the same bands. A product with a different band
    structure (the DEM stripped out of a 7-band stack, an RGB hillshade of a DEM) starts
    clean, and does not inherit labels such as GLOBATO_DATATYPE=MULTI_STACK.

    Only missing items are reported, so anything a hook set on its own output wins.
    """

    if src.count != dst.count:
        return {}, {}

    dst_tags = dst.tags()
    tags = {k: v for k, v in src.tags().items() if k not in dst_tags}

    bands = {}
    for bidx in range(1, src.count + 1):
        dst_band_tags = dst.tags(bidx)
        item = {
            # Band statistics describe the pixels they were computed from.
            "tags": {
                k: v
                for k, v in src.tags(bidx).items()
                if k not in dst_band_tags and not k.startswith("STATISTICS_")
            },
            "description": src.descriptions[bidx - 1]
            if not dst.descriptions[bidx - 1]
            else None,
            "units": src.units[bidx - 1] if not dst.units[bidx - 1] else None,
        }
        if item["tags"] or item["description"] or item["units"]:
            bands[bidx] = item

    return tags, bands


def _apply_metadata(dst, tags, bands):
    if tags:
        dst.update_tags(**tags)
    for bidx, item in bands.items():
        if item["tags"]:
            dst.update_tags(bidx, **item["tags"])
        if item["description"]:
            dst.set_band_description(bidx, item["description"])
        if item["units"]:
            dst.set_band_unit(bidx, item["units"])


def copy_metadata(src, dst):
    """Carry tags, band descriptions and units from the open dataset `src` to the writable `dst`."""

    _apply_metadata(dst, *_missing_metadata(src, dst))


def carry_metadata(src_path, dst_path):
    """As copy_metadata(), for files on disk. `dst_path` is only reopened if something is missing,
    so an output that already has its metadata (e.g. a COG) is left byte-for-byte alone.
    """

    if os.path.abspath(src_path) == os.path.abspath(dst_path):
        return

    try:
        with rasterio.open(src_path) as src:
            with rasterio.open(dst_path) as dst:
                tags, bands = _missing_metadata(src, dst)
            if tags or bands:
                with rasterio.open(dst_path, "r+") as dst:
                    _apply_metadata(dst, tags, bands)
    except Exception as e:
        logger.debug(f"Could not carry metadata from {src_path} to {dst_path}: {e}")


class RasterHook(FetchHook):
    """Unified base class for all raster hooks.

    Child classes must set `processing_mode`:
    - "chunk": The hook implements `process_chunk(data, ndv, ...)` and operates on numpy arrays.
    - "global": The hook implements `process_raster(src_path, dst_path, ...)` and operates on full files.
    """

    meta_stage = "collection"
    default_suffix = "_processed"
    meta_desc = "Process a raster."
    processing_mode = "chunk"

    def __init__(
        self,
        suffix=None,
        barrier=None,
        region=None,
        output=None,
        upper=None,
        lower=None,
        strip_bands=False,
        buffer=0,
        chunk_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output = output
        self.suffix = suffix or self.default_suffix
        self.barrier = barrier
        self.barrier_geoms = None
        self.region = parse_region(region)
        self.upper = float_or(upper)
        self.lower = float_or(lower)
        self.strip_bands = strip_bands

        self.buffer = int(buffer)

        # If the user doesn't specify a chunk size, global hooks default to 'full' (the entire file)
        # while chunk hooks default to Rasterio's native block size.
        if chunk_size is None and self.processing_mode == "global":
            self.chunk_size = "full"
        else:
            self.chunk_size = chunk_size

        self.local_tmp = os.path.abspath("tmp")

    @property
    def cache_dir(self):
        """Directory for reusable fetched dependencies."""

        mod = getattr(self, "current_mod", None)
        mod_outdir = getattr(mod, "outdir", getattr(mod, "_outdir", None))
        return mod_outdir if mod_outdir else self.local_tmp

    # --- Utilities ---
    def modify_profile(self, profile):
        """Override this to change dtype, count, or nodata for the output raster."""

        profile.update(
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="deflate",
            predictor=3,
            bigtiff="YES",
        )
        return profile

    def _strip_to_single_band(self, raster_path):
        """Removes auxiliary bands from a GeoTIFF, retaining only Band 1."""

        if not self.strip_bands:
            return

        with rasterio.open(raster_path) as src:
            if src.count == 1:
                return

            profile = src.profile.copy()
            profile.update(count=1)

            temp_path = raster_path + ".strip.tif"
            with rasterio.open(temp_path, "w", **profile) as dst:
                dst.write(src.read(1), 1)

        shutil.move(temp_path, raster_path)
        logger.debug(
            f"[{self.name}] Stripped auxiliary bands, retaining only Elevation (Band 1)."
        )

    def _clamp_raster(self, raster_path):
        """Clamp raster values to enforce lower/upper bounds."""

        if self.upper is None and self.lower is None:
            return

        with rasterio.open(raster_path, "r+") as src:
            data = src.read(1)
            nodata = src.nodata if src.nodata is not None else -9999

            is_float = data.dtype.kind == "f"
            if is_float:
                valid_mask = (data != nodata) & (~np.isnan(data))
            else:
                valid_mask = data != nodata

            clamped = False
            if self.upper is not None:
                mask = (data > self.upper) & valid_mask
                if np.any(mask):
                    data[mask] = self.upper
                    clamped = True

            if self.lower is not None:
                mask = (data < self.lower) & valid_mask
                if np.any(mask):
                    data[mask] = self.lower
                    clamped = True

            if clamped:
                logger.info(
                    f"[{self.name}] Clamped values to bounds (Lower: {self.lower}, Upper: {self.upper})"
                )
                src.write(data, 1)

    def _get_barrier(
        self,
        include_rivers=True,
        include_lakes=False,
        include_reefs=False,
        include_wetlands=True,
        include_breakwaters=True,
        include_estuaries=True,
        output_mode="binary",
    ):
        if not self.barrier:
            return None

        mod = getattr(self, "current_mod", None)
        region = getattr(mod, "region", None) if mod else None

        cache_dir = self.cache_dir
        from globato.utils import resolve_barrier

        barrier_path = resolve_barrier(
            self.barrier,
            region=region,
            outdir=os.path.join(cache_dir, "auto_barriers"),
            output_type="vector",
            include_rivers=include_rivers,
            include_lakes=include_lakes,
            include_reefs=include_reefs,
            include_wetlands=include_wetlands,
            include_breakwaters=include_breakwaters,
            include_estuaries=include_estuaries,
            target_crs=region.srs,
            output_mode=output_mode,
        )

        return barrier_path

    def _get_barrier_geometries(
        self,
        include_rivers=True,
        include_lakes=False,
        include_reefs=False,
        include_wetlands=True,
        include_breakwaters=True,
        include_estuaries=True,
        output_mode="binary",
    ):
        barrier_path = self._get_barrier(
            include_rivers=include_rivers,
            include_lakes=include_lakes,
            include_reefs=include_reefs,
            include_wetlands=include_wetlands,
            include_breakwaters=include_breakwaters,
            include_estuaries=include_estuaries,
            output_mode=output_mode,
        )
        # if not self.barrier:
        #     return None

        # mod = getattr(self, "current_mod", None)
        # region = getattr(mod, "region", None) if mod else None

        # mod_outdir = getattr(mod, "outdir", getattr(mod, "_outdir", None))
        # cache_dir = mod_outdir if mod_outdir else os.getcwd()

        # from globato.utils import resolve_barrier

        # barrier_path = resolve_barrier(
        #     self.barrier,
        #     region=region,
        #     outdir=os.path.join(cache_dir, "auto_barriers"),
        #     output_type="vector",
        #     include_rivers=include_rivers,
        #     include_lakes=include_lakes,
        #     include_reefs=include_reefs,
        #     include_wetlands=include_wetlands,
        #     include_breakwaters=include_breakwaters,
        #     target_crs=region.srs,
        #     output_mode="binary",
        # )

        if not barrier_path:
            return None

        try:
            meta, fids, geometry_wkb, fields = read(barrier_path)
            geoms = shapely.from_wkb(geometry_wkb)
            return list(geoms)

        except Exception as e:
            logger.error(f"Could not parse geometries from {barrier_path}: {e}")
            return None

    def _create_barrier_mask(
        self,
        shape,
        transform,
        include_rivers=True,
        include_lakes=False,
        include_reefs=False,
        include_wetlands=True,
        include_breakwaters=True,
        include_estuaries=True,
        output_mode="binary",
    ):
        """Generates a boolean numpy mask from the barrier.
        Automatically fetches or generates the geometries on-demand.
        Returns True inside the polygons, False outside.
        """

        if not self.barrier:
            return None

        # _get_barrier_geometries fetches the data if not provided
        if not self.barrier_geoms:
            self.barrier_geoms = self._get_barrier_geometries(
                include_rivers=include_rivers,
                include_lakes=include_lakes,
                include_wetlands=include_wetlands,
                include_estuaries=include_estuaries,
                include_reefs=include_reefs,
                include_breakwaters=include_breakwaters,
                output_mode=output_mode,
            )

        # If fetching failed or returned nothing, abort
        if not self.barrier_geoms:
            return None

        from rasterio.features import rasterize

        mask = rasterize(
            self.barrier_geoms,
            out_shape=shape,
            transform=transform,
            fill=0,
            default_value=1,
            dtype="uint8",
        ).astype(bool)

        return mask

    def get_outliers(self, in_array, percentile=75, k=1.5):
        if np.all(np.isnan(in_array)):
            return np.nan, np.nan
        p_max = np.nanpercentile(in_array, percentile)
        p_min = np.nanpercentile(in_array, 100 - percentile)
        iqr = (p_max - p_min) * k
        return p_max + iqr, p_min - iqr

    def yield_buffered_windows(self, src, buffer_size=0, chunk_size=None):
        if str(chunk_size).lower() == "full" or chunk_size == -1:
            windows = [((0, 0), Window(0, 0, src.width, src.height))]
        elif chunk_size:
            windows = []
            c_size = int(chunk_size)
            for row_off in range(0, src.height, c_size):
                for col_off in range(0, src.width, c_size):
                    width = min(c_size, src.width - col_off)
                    height = min(c_size, src.height - row_off)
                    windows.append(
                        ((row_off, col_off), Window(col_off, row_off, width, height))
                    )
        else:
            windows = list(src.block_windows(1))

        for block_index, window in windows:
            if buffer_size == 0:
                yield window, window
                continue

            row_start = max(0, window.row_off - buffer_size)
            col_start = max(0, window.col_off - buffer_size)
            row_stop = min(src.height, window.row_off + window.height + buffer_size)
            col_stop = min(src.width, window.col_off + window.width + buffer_size)

            buffered_window = Window.from_slices(
                (row_start, row_stop), (col_start, col_stop)
            )
            yield window, buffered_window

    def _extract_subpixel_coords(
        self, data_stack, rows, cols, transform, apply_jitter=True
    ):
        """Extracts X/Y coordinates from a MultiStack or falls back to cell-centers."""

        # transform = src.transform
        x_vals, y_vals = transform * (cols + 0.5, rows + 0.5)

        #    is_multi_stack = src.tags().get("GLOBATO_DATATYPE") == "MULTI_STACK"

        # if is_multi_stack and data_stack is not None and data_stack.shape[0] >= 7:
        if data_stack is not None and data_stack.ndim == 3 and data_stack.shape[0] >= 7:
            x_band = data_stack[5][rows, cols]
            y_band = data_stack[6][rows, cols]

            nan_xy = np.isnan(x_band) | np.isnan(y_band)
            if np.any(nan_xy):
                x_band[nan_xy] = x_vals[nan_xy]
                y_band[nan_xy] = y_vals[nan_xy]

            x_vals = x_band
            y_vals = y_band
            logger.debug(f"[{self.name}] Using x/y bands from input")

        if apply_jitter:
            rng = np.random.default_rng(seed=42)
            x_vals = x_vals + rng.uniform(-1e-10, 1e-10, size=len(x_vals))
            y_vals = y_vals + rng.uniform(-1e-10, 1e-10, size=len(y_vals))

        return x_vals, y_vals

    def _promote_to_multistack(self, src_path, dst_path):
        """Auto-promotes a 1-band DEM to a 7-band multi-stack for advanced hooks."""
        with rasterio.open(src_path) as src:
            profile = src.profile.copy()
            profile.update(count=7, dtype="float32")

            z = src.read(1)
            nodata = src.nodata if src.nodata is not None else -9999
            valid = (z != nodata) & ~np.isnan(z)

            count_arr = np.zeros_like(z, dtype="float32")
            count_arr[valid] = 1.0

            weight_arr = np.zeros_like(z, dtype="float32")
            weight_arr[valid] = 1.0

            unc_arr = np.zeros_like(z, dtype="float32")

            rows, cols = np.indices(z.shape)
            xs, ys = rasterio.transform.xy(src.transform, rows, cols, offset="center")
            x_arr = np.array(xs, dtype="float32")
            y_arr = np.array(ys, dtype="float32")

            x_arr[~valid] = nodata
            y_arr[~valid] = nodata

            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.write(z.astype("float32"), 1)
                dst.write(count_arr, 2)
                dst.write(weight_arr, 3)
                dst.write(unc_arr, 4)
                dst.write(unc_arr, 5)
                dst.write(x_arr, 6)
                dst.write(y_arr, 7)

    # --- Implementation Functions ---
    def process_chunk(self, data, ndv, entry, transform=None, window=None):
        raise NotImplementedError("Chunk-mode hooks must implement process_chunk()")

    def process_raster(self, src_path, dst_path, entry):
        raise NotImplementedError("Global-mode hooks must implement process_raster()")

    def _dst_fn(self, src_fn):
        """Where to write the result for `src_fn`. Defaults to a suffixed file in tmp."""

        return self.output or os.path.join(
            self.local_tmp,
            f"{os.path.splitext(os.path.basename(src_fn))[0]}{self.suffix}.tif",
        )

    # --- Routing and Processing ---
    def run(self, entries):
        logger.info(
            f"[{self.name}] Running in '{self.processing_mode}' mode on {len(entries)} entries"
        )
        new_entries = []

        self.local_tmp = os.path.abspath("tmp")
        os.makedirs(self.local_tmp, exist_ok=True)

        for mod, entry in entries:
            self.current_mod = mod
            stream = entry.get("stream")
            src_fn = entry.get("dst_fn")

            # Stream Data -> Chunk Hook
            if stream and self.processing_mode == "chunk":
                entry["stream"] = self._stream_wrapper(stream, entry)
                entry["stream_type"] = "raster-stream"
                new_entries.append((mod, entry))
                continue

            # Stream Data -> Global Hook (Requires Draining!)
            if stream and self.processing_mode == "global":
                logger.debug(
                    f"[{self.name}] Global hook detected active stream. Draining to disk..."
                )
                from globato.hooks.sinks.raster_writer import RasterWrite

                base_name = os.path.basename(src_fn) if src_fn else "streamed_raster"
                drain_fn = os.path.join(
                    self.local_tmp,
                    f"{os.path.splitext(base_name)[0]}_drained_{self.name}.tif",
                )

                entry["dst_fn"] = drain_fn
                drainer = RasterWrite(suffix="", inline=False)
                drainer.run([(mod, entry)])
                src_fn = entry.get("dst_fn")

            # Ensure we have a valid file at this point
            if not src_fn or not os.path.exists(src_fn):
                new_entries.append((mod, entry))
                continue

            dst_fn = self._dst_fn(src_fn)
            logger.debug(f"[{self.name}] Processing file: {os.path.basename(src_fn)}")

            if getattr(self, "meta_requires", None) == "multi-stack":
                with rasterio.open(src_fn) as chk_src:
                    if chk_src.count < 7:
                        logger.warning(
                            f"⚠️ [{self.name}] Requires a 7-band multi-stack but received a {chk_src.count}-band raster."
                        )
                        logger.warning(
                            f"⚠️ Auto-promoting {os.path.basename(src_fn)} to a stack. (Note: Weights will be uniform)."
                        )
                        multi_fn = os.path.join(
                            self.local_tmp, f"multi_{os.path.basename(src_fn)}"
                        )
                        self._promote_to_multistack(src_fn, multi_fn)
                        src_fn = multi_fn

            try:
                # File Data -> Chunk Hook
                if self.processing_mode == "chunk":
                    success = self._process_file_fallback(src_fn, dst_fn, entry)

                # File Data -> Global Hook
                else:
                    success = self.process_raster(src_fn, dst_fn, entry)

                if success:
                    self._clamp_raster(dst_fn)
                    self._strip_to_single_band(dst_fn)
                    # A hook rewrites the raster from its profile alone, which drops
                    # tags and band descriptions (e.g. those set by raster_metadata).
                    carry_metadata(src_fn, dst_fn)
                    entry["src_fn"] = str(src_fn)
                    entry["dst_fn"] = str(dst_fn)
                    entry.setdefault("artifacts", {})[self.name] = dst_fn

            except Exception as e:
                logger.error(f"[{self.name}] Failed on {src_fn}: {e}")
                raise

            new_entries.append((mod, entry))

        return new_entries

    # --- Generaters & Fallbacks ---
    def _stream_wrapper(self, input_stream, entry):
        """Pass-through generator for in-memory chunk processing."""
        profile = next(input_stream)
        profile = self.modify_profile(profile)
        yield profile

        self.barrier_geoms = self._get_barrier_geometries(profile.get("transform"))
        for window, buff_win, data, ndv, transform in input_stream:
            processed_data = self.process_chunk(data, ndv, entry, transform, buff_win)
            yield window, buff_win, processed_data, ndv, transform

    def _process_file_fallback(self, src_path, dst_path, entry):
        """Applies chunked processing to a file block-by-block."""
        with rasterio.open(src_path) as src:
            self.barrier_geoms = self._get_barrier_geometries(src.transform)
            profile = src.profile.copy()
            profile = self.modify_profile(profile)

            with rasterio.open(dst_path, "w", **profile) as dst:
                copy_metadata(src, dst)
                for window, buff_win in self.yield_buffered_windows(
                    src, self.buffer, self.chunk_size
                ):
                    data = src.read(window=buff_win)
                    chunk_transform = rasterio.windows.transform(
                        buff_win, src.transform
                    )

                    result = self.process_chunk(
                        data,
                        src.nodata,
                        entry,
                        transform=chunk_transform,
                        window=buff_win,
                    )

                    y_off = window.row_off - buff_win.row_off
                    x_off = window.col_off - buff_win.col_off

                    if result.ndim == 3:
                        final_chunk = result[
                            :,
                            y_off : y_off + window.height,
                            x_off : x_off + window.width,
                        ]
                        dst.write(final_chunk, window=window)
                    else:
                        final_chunk = result[
                            y_off : y_off + window.height, x_off : x_off + window.width
                        ]
                        dst.write(final_chunk, 1, window=window)

        return True


class RasterGlobalHook(RasterHook):
    processing_mode = "global"


class RasterStreamHook(RasterHook):
    processing_mode = "chunk"


def is_cog(path):
    """True if GDAL reports `path` as having a Cloud-Optimized GeoTIFF layout."""

    with rasterio.open(path) as src:
        return src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"


def _cog_predictor(src):
    # PREDICTOR=3 is floating point only (e.g. it fails on a uint8 hillshade).
    return 3 if np.dtype(src.dtypes[0]).kind == "f" else 2


def _copy_as_cog(src_path, dst_path, predictor, **options):
    """COG-driver copy of `src_path` to `dst_path`, which may be the same file.

    Written next to the destination and moved into place, so a failed copy never
    leaves a partial COG behind.
    """

    from rasterio.shutil import copy

    tmp_path = f"{dst_path}.cog_tmp"
    try:
        with rasterio.Env(GDAL_TIFF_OVR_BLOCKSIZE=256):
            copy(
                src_path,
                tmp_path,
                driver="COG",
                compress="deflate",
                predictor=predictor,
                blocksize=256,  # the COG driver ignores blockxsize/blockysize
                bigtiff="YES",
                **options,
            )
        os.replace(tmp_path, dst_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def update_cog_metadata(path, tags=None, band_descriptions=None):
    """Add tags and band descriptions to a COG by rewriting it, keeping it a valid COG.

    GDAL refuses to edit a COG in place, because that moves the header to the end of the
    file and breaks the layout. The metadata goes onto a small VRT of the file instead,
    which is then copied back as a COG. The existing overviews are reused, not rebuilt.
    """

    from rasterio.shutil import copy

    vrt_path = f"{path}.meta.vrt"
    try:
        copy(path, vrt_path, driver="VRT")
        with rasterio.open(vrt_path, "r+") as vrt:
            if tags:
                vrt.update_tags(**tags)
            for bidx, name in enumerate(band_descriptions or [], start=1):
                if bidx <= vrt.count:
                    vrt.set_band_description(bidx, name)
            predictor = _cog_predictor(vrt)
        _copy_as_cog(vrt_path, path, predictor, overviews="AUTO")
    finally:
        if os.path.exists(vrt_path):
            os.remove(vrt_path)


def write_cog(src_path, dst_path, overviews=(2, 4, 8, 16, 32), resampling="average"):
    """Rewrite `src_path` as a Cloud-Optimized GeoTIFF at `dst_path`.

    `dst_path` may be `src_path` itself. Overviews are built into `src_path` first.
    """

    from rasterio.enums import Resampling

    resampling_enum = getattr(Resampling, resampling.lower(), Resampling.average)
    with rasterio.open(src_path, "r+") as src:
        src.build_overviews(list(overviews), resampling_enum)
        src.update_tags(ns="rio_overview", resampling=resampling.lower())
        predictor = _cog_predictor(src)

    _copy_as_cog(src_path, dst_path, predictor, copy_src_overviews=True)


class RasterCOG(RasterHook):
    """Converts a standard GeoTIFF into a strict Cloud-Optimized GeoTIFF (COG).
    Builds overviews (2, 4, 8, 16, 32) and aligns the byte structure for HTTP streaming.

    With no `output`, the raster is converted in place: the COG replaces the input
    file, so later hooks (and copy_artifact) pick it up under the same name.
    """

    name = "format-cog"
    default_suffix = "_cog"
    meta_desc = "Transforms a stadard GeoTiff to a Cloud-Optimized GeoTiff (COG)."
    meta_aliases = ["format_cog"]
    processing_mode = "global"

    def __init__(self, overviews=[2, 4, 8, 16, 32], resampling="average", **kwargs):
        super().__init__(**kwargs)
        self.overviews = parse_arg_to_list(overviews, int)
        self.resampling = resampling

    def _dst_fn(self, src_fn):
        # In place unless told otherwise. A copy left in tmp would be removed by
        # cleanup_tmp, and the file that gets delivered would not be the COG.
        return self.output or src_fn

    def process_raster(self, src_path, dst_path, entry):
        logger.info(
            f"[{self.name}] Building {self.overviews} overviews and aligning COG..."
        )
        write_cog(src_path, dst_path, self.overviews, self.resampling)
        return True
