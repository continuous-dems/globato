"""Review products built from the cleaned module-only source stack."""

from pathlib import Path

import numpy as np
import rasterio
from matplotlib.colors import LightSource
from rasterio.enums import Resampling
from rasterio.fill import fillnodata
from scipy.ndimage import distance_transform_edt
from globato.qc.stack.grid import metres

from globato.qc.stack import io
from globato.qc.stack.stacking import rebuild


_MEASURED_NODATA = -9999.0


def _rebuild_stack(run, root, manifest, previous, stack, region, filename):
    """Rebuild a module-only native MultiStack with the current rejection mask."""
    output = Path(run) / "review" / filename
    # Rebuild on each review run. A previous run in this folder may have used
    # different QC settings or different source observations.
    output.parent.mkdir(parents=True, exist_ok=True)
    state = output.with_suffix(".fusion.tif")
    temporary = output.with_name(output.stem + ".partial.tif")
    temporary.unlink(missing_ok=True)
    state.unlink(missing_ok=True)
    acc = stack.create_accumulator(region, output=str(temporary), state=str(state))
    try:
        rebuild(acc, io.read(root / "SOURCES.json")["sources"], manifest, previous)
        acc.finalize(statistics=False)
        temporary.replace(output)
    finally:
        acc.close()
        state.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    return output


def final_measured_stack(run, root, manifest, stack, region):
    """Final cleaned seven-band MultiStack for this module only."""
    # A review accumulator starts blank. It must not reuse an unrelated
    # previous FusionState; rebuild every cell from the module's own sources.
    return _rebuild_stack(
        run, root, manifest, None, stack, region, "module_final_measured_multistack.tif"
    )


def last_pass_stack(run, root, manifest, previous, stack, region):
    """Rebuild native sufficient statistics for display, retaining NoData gaps."""
    return _rebuild_stack(
        run, root, manifest, previous, stack, region, "last_pass_measured.tif"
    )


def write_measured_surface(multistack, output):
    """Single-band measured surface. Read/write by blocks to bound memory."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".partial.tif")
    temporary.unlink(missing_ok=True)
    try:
        with rasterio.open(multistack) as src:
            if src.count != 7 or src.tags().get("GLOBATO_DATATYPE") != "MULTI_STACK":
                raise ValueError("Expected a native seven-band MultiStack")
            profile = src.profile.copy()
            profile.update(
                count=1, dtype="float32", nodata=_MEASURED_NODATA, compress="deflate"
            )
            with rasterio.open(temporary, "w", **profile) as dst:
                for _, window in dst.block_windows(1):
                    z = src.read(1, window=window).astype("float32")
                    valid = src.read(2, window=window) > 0
                    z[~valid] = _MEASURED_NODATA
                    dst.write(z, 1, window=window)
                dst.update_tags(
                    GLOBATO_DATATYPE="STACK_QC_MEASURED_SURFACE",
                    STACK_QC_SOURCE=str(multistack),
                )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _valid_surface(z, nodata):
    valid = np.isfinite(z)
    if nodata is not None:
        valid &= z != nodata
    return valid


def write_hillshade_tif(surface_path, output):
    """Write a hillshade by blocks; only a one-cell halo is needed for slopes."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".partial.tif")
    temporary.unlink(missing_ok=True)
    light = LightSource(azdeg=315, altdeg=45)
    try:
        with rasterio.open(surface_path) as src:
            dx, dy = metres(src)
            profile = src.profile.copy()
            profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
            with rasterio.open(temporary, "w", **profile) as dst:
                for _, block in dst.block_windows(1):
                    # Clip the one-pixel halo at the dataset edge. numpy.gradient
                    # then uses the same one-sided edge rule as the full grid.
                    x0 = max(0, int(block.col_off) - 1)
                    y0 = max(0, int(block.row_off) - 1)
                    x1 = min(src.width, int(block.col_off + block.width) + 1)
                    y1 = min(src.height, int(block.row_off + block.height) + 1)
                    halo = rasterio.windows.Window(x0, y0, x1 - x0, y1 - y0)
                    z = src.read(1, window=halo).astype("float64")
                    valid = _valid_surface(z, src.nodata)
                    if not valid.any():
                        dst.write(
                            np.zeros(
                                (int(block.height), int(block.width)), dtype="uint8"
                            ),
                            1,
                            window=block,
                        )
                        continue
                    # Fill only to calculate local slopes; the output retains
                    # NoData. Using nearby elevations avoids artificial cliffs
                    # at the edges of gaps in the measured surface.
                    if not valid.all():
                        nearest = distance_transform_edt(
                            ~valid, return_distances=False, return_indices=True
                        )
                        z[~valid] = z[tuple(nearest[:, ~valid])]
                    # Fixed light direction (no per-block contrast stretch):
                    # equal slopes must have equal brightness across blocks.
                    gy, gx = np.gradient(z, -dy, dx)
                    direction = light.direction
                    shade = np.clip(
                        (-gx * direction[0] - gy * direction[1] + direction[2])
                        / np.sqrt(1.0 + gx * gx + gy * gy),
                        0,
                        1,
                    )
                    ys = slice(
                        int(block.row_off - y0), int(block.row_off - y0 + block.height)
                    )
                    xs = slice(
                        int(block.col_off - x0), int(block.col_off - x0 + block.width)
                    )
                    out_valid = valid[ys, xs]
                    pixels = np.where(
                        out_valid, np.clip(np.round(shade[ys, xs] * 254) + 1, 1, 255), 0
                    ).astype("uint8")
                    dst.write(pixels, 1, window=block)
                dst.update_tags(
                    GLOBATO_DATATYPE="STACK_QC_HILLSHADE",
                    STACK_QC_SOURCE=str(surface_path),
                )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def module_interpolated_preview(measured, output, *, max_pixels=2_000_000):
    """Short-range interpolation for display only; never used by Globato science.

    Work on a bounded preview grid for predictable review cost. Preserve
    measured values, interpolate nearby gaps with GDAL's fillnodata, and leave
    large unsupported regions as NoData. This is NOT the production algorithm.
    """
    output = Path(output)
    # Never reuse a preview from a previous run with a different final mask.
    with rasterio.open(measured) as ds:
        scale = min(1.0, (max_pixels / (ds.width * ds.height)) ** 0.5)
        width = max(1, int(ds.width * scale))
        height = max(1, int(ds.height * scale))
        data = ds.read(
            1, out_shape=(height, width), masked=True, resampling=Resampling.average
        ).astype("float32")
        valid = ~np.ma.getmaskarray(data) & np.isfinite(data.filled(np.nan))
        if not valid.any():
            raise ValueError(
                "No cleaned measurements are available for the module preview"
            )
        original = data.filled(0).astype("float32")
        # GDAL fill is restricted to at most 32 preview pixels. No artificial
        # surface is extended across large gaps.
        filled = fillnodata(
            original.copy(),
            mask=valid.astype("uint8"),
            max_search_distance=32,
            smoothing_iterations=1,
        )
        nearby = distance_transform_edt(~valid) <= 32
        filled[~nearby] = _MEASURED_NODATA
        filled[valid] = original[valid]
        transform = ds.transform * rasterio.Affine.scale(
            ds.width / width, ds.height / height
        )
        profile = ds.profile.copy()
    profile.update(
        count=1,
        width=width,
        height=height,
        transform=transform,
        dtype="float32",
        nodata=_MEASURED_NODATA,
        compress="deflate",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".partial.tif")
    temporary.unlink(missing_ok=True)
    try:
        with rasterio.open(temporary, "w", **profile) as dst:
            dst.write(filled, 1)
            dst.update_tags(
                GLOBATO_DATATYPE="STACK_QC_DISPLAY_PREVIEW",
                DESCRIPTION="Module-only short-range review fill; not production Globato interpolation",
                STACK_QC_SOURCE=str(measured),
            )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
