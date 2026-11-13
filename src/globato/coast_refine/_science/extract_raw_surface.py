#!/usr/bin/env python3
"""Integer crop of direct MultiStack band 1; no value change or resampling."""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window


def blocks(width, height):
    for row in range(0, height, 1024):
        for col in range(0, width, 1024):
            yield Window(col, row, min(1024, width - col), min(1024, height - row))


def verify_self_contained_raster(path):
    """Reject unsealed GDAL sidecars on generated, internally masked TIFFs.

    Hashing the TIFF alone cannot detect a later external validity mask or
    PAM georeference. Original inputs may have sidecars; our outputs may not.
    """
    path = Path(path).resolve(strict=True)
    with rasterio.open(path) as ds:
        companions = [name for name in ds.files if Path(name).resolve() != path]
    if companions:
        raise ValueError("Cached raster has unsealed GDAL sidecars: " + str(companions))


def copy_band_one(source, out, window=None, *, band=1):
    """Copy an effective stack band and validity without resampling (default Z)."""
    source, out = Path(source), Path(out)
    with rasterio.open(source) as src:
        win = window if window is not None else Window(0, 0, src.width, src.height)
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            width=int(win.width),
            height=int(win.height),
            transform=src.window_transform(win),
            compress="deflate",
            BIGTIFF="IF_SAFER",
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(out, "w", **profile) as dst:
                for block in blocks(int(win.width), int(win.height)):
                    source_block = Window(
                        win.col_off + block.col_off,
                        win.row_off + block.row_off,
                        block.width,
                        block.height,
                    )
                    dst.write(src.read(band, window=source_block), 1, window=block)
                    dst.write_mask(
                        src.read_masks(band, window=source_block), window=block
                    )
                dst.scales = (src.scales[band - 1],)
                dst.offsets = (src.offsets[band - 1],)
                dst.update_tags(**src.tags())
                dst.set_band_description(1, src.descriptions[band - 1] or "Elevation")
                dst.update_tags(1, **src.tags(band))
                if src.units[band - 1]:
                    dst.set_band_unit(1, src.units[band - 1])
                tags = src.tags()
                is_stack = tags.get("GLOBATO_DATATYPE") == "MULTI_STACK"
                dst.update_tags(
                    SOURCE_ELEVATION_FILE=str(source.resolve()),
                    COPY_OPERATION=f"native_band_{band}_integer_window",
                    COPY_RESAMPLING="false",
                )
                if is_stack:
                    dst.update_tags(
                        SOURCE_RAW_STACK=str(source.resolve()),
                        SURFACE_TYPE=f"direct_multi_stack_band_{band}_integer_crop",
                        POST_STACK_INTERPOLATION="false",
                        POST_STACK_RESAMPLING="false",
                    )
        with rasterio.open(out) as dst:
            for block in blocks(dst.width, dst.height):
                source_block = Window(
                    win.col_off + block.col_off,
                    win.row_off + block.row_off,
                    block.width,
                    block.height,
                )
                if not np.array_equal(
                    src.read(band, window=source_block),
                    dst.read(1, window=block),
                    equal_nan=True,
                ):
                    raise ValueError("Value-preservation check failed")
                if not np.array_equal(
                    src.read_masks(band, window=source_block),
                    dst.read_masks(1, window=block),
                ):
                    raise ValueError("Validity-mask preservation check failed")
