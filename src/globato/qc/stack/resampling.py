"""Optional fine-QC grid and bounded aggregation of accepted source statistics.

No interpolation here: sum all seven additive bands for each original source
onto aligned, integer-nested output cells before native stacking across sources.
"""

import copy
import math
import re

import numpy as np
import rasterio
from pyproj import CRS
from rasterio.windows import Window
from fetchez.utils import str2inc
from globato.hooks.sinks.side_stack import _resolve_grid
from .native import masked_stream


def resolution(value, *, crs=None):
    """Parse native CRS increments or explicit arcseconds such as ``1/9s``.

    GMT-style suffixes retain the native increment convention. They are for
    geographic grids; projected grids use a number in their CRS's linear units.
    There is no latitude-dependent approximation from metres to degrees.
    """
    error = "qc_res must be positive: use 1/9s for arcseconds or a number in CRS units"
    if isinstance(value, bool) or "_" in str(value):
        raise ValueError(error)
    text = str(value).strip()
    fraction = re.fullmatch(r"([+]?[0-9]+)\s*/\s*([0-9]+)s", text)
    if fraction:
        denominator = int(fraction[2])
        if denominator == 0:
            raise ValueError(error)
        text = str(int(fraction[1]) / denominator) + "s"
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        if not re.fullmatch(
            r"[+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?[csmt]",
            text,
        ):
            raise ValueError(error) from None
        if crs is not None and text[-1] in "mt":
            raise ValueError(
                "qc_res accepts arcseconds such as 1/9s or a number in CRS units; "
                "m/t suffixes are ambiguous, so use a number for projected metres"
            )
        if crs is not None and not rasterio.crs.CRS.from_user_input(crs).is_geographic:
            raise ValueError(
                "qc_res with a unit suffix requires a geographic grid; "
                "for a projected grid use a number in that CRS's linear units"
            )
        parsed = str2inc(text)
    else:
        if crs is not None and rasterio.crs.CRS.from_user_input(crs).is_geographic:
            raise ValueError(
                f"qc_res={value!r}: geographic grids require an arcsecond suffix. "
                "Use '1s' or '1/9s', or omit qc_res to follow the DEM grid. "
                "Bare numbers are only accepted for projected grids."
            )
    if parsed is None or not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(error)
    return float(parsed)


def context_hooks(stack, post, requested):
    """Scale a copy of the context resolution ladder; leave final hooks unchanged."""
    if requested is None:
        return stack, post
    fine, final = resolution(requested, crs=stack.crs), resolution(stack.res)
    ratio = final / fine
    if (
        not math.isfinite(ratio)
        or ratio < 1 - 1e-8
        or not math.isclose(ratio, round(ratio), rel_tol=1e-8)
    ):
        grid_crs = rasterio.crs.CRS.from_user_input(stack.crs)
        final_label = (
            f"{final * 3600:.12g}s"
            if grid_crs.is_geographic
            else f"{final:.12g} {grid_crs.linear_units}"
        )
        raise ValueError(
            f"qc_res={requested!r} does not nest in the final DEM increment "
            f"({final_label}): DEM increment / qc_res = {ratio:.12g}. "
            "Choose qc_res = DEM increment / N, where N is a positive integer "
            "(1, 2, 3, ...). QC may equal the DEM grid or be finer, never coarser."
        )
    factor = round(ratio)
    fine = final / factor  # Canonical nesting, including decimal serialization noise.
    if factor == 1:
        return stack, post
    native_stack = copy.copy(stack)
    native_stack.res = fine
    native_post = copy.deepcopy(post)
    interpolators = [h for h in native_post if h.name == "ms_binary_cudem"]
    if len(interpolators) != 1:
        raise ValueError("qc_res requires one native context interpolator")
    hook = interpolators[0]
    if hook.resolutions:
        if not math.isclose(hook.resolutions[0], final, rel_tol=1e-8):
            raise ValueError(
                "The first ms_binary_cudem resolution must match multi_stack.res when using qc_res"
            )
        hook.resolutions = [value / factor for value in hook.resolutions]
        hook.resolutions[0] = fine
    # Empty resolutions are already inferred by the native owner from its input.
    return native_stack, native_post


def validate_request(requested, region, hooks):
    """Reject invalid requests before discovery; no sources or caches are touched."""
    stacks = [h for h in hooks if h.name.replace("-", "_") == "multi_stack"]
    interpolators = [h for h in hooks if h.name.replace("-", "_") == "ms_binary_cudem"]
    if len(stacks) != 1 or len(interpolators) != 1:
        raise ValueError(
            "Stack QC needs one multi_stack and one ms_binary_cudem in global_hooks"
        )
    fine, _ = context_hooks(stacks[0], interpolators, requested)
    target = output_grid(stacks[0], region)
    if fine is not stacks[0]:
        nested_region(region, fine, target)


def output_grid(stack, region):
    if not region.srs or not CRS.from_user_input(region.srs).to_2d().equals(
        CRS.from_user_input(stack.crs).to_2d(), ignore_axis_order=True
    ):
        raise ValueError(
            "Recipe AOI CRS must match the horizontal stack CRS for a nested QC grid. "
            "Set region_srs correctly and transform the AOI coordinates; changing an EPSG label is not reprojection."
        )
    geom = _resolve_grid(region, resolution(stack.res), stack.crs)
    return dict(
        driver="GTiff",
        dtype="float64",
        nodata=None,
        count=7,
        width=geom["width"],
        height=geom["height"],
        transform=rasterio.Affine(*geom["transform"]),
        crs=rasterio.crs.CRS.from_wkt(geom["crs"]),
    )


def nested_region(region, fine_stack, target):
    """Use the resolved final DEM bounds for QC; never round two AOIs independently."""
    left, bottom, right, top = rasterio.transform.array_bounds(
        target["height"], target["width"], target["transform"]
    )
    native = region.copy()
    native.xmin, native.xmax, native.ymin, native.ymax = left, right, bottom, top
    grid = output_grid(fine_stack, native)
    factor = round(target["transform"].a / resolution(fine_stack.res))
    if (
        grid["width"] != target["width"] * factor
        or grid["height"] != target["height"] * factor
        or grid["transform"].c != target["transform"].c
        or grid["transform"].f != target["transform"].f
    ):
        raise ValueError(
            "QC grid cannot nest exactly inside the resolved final DEM bounds. "
            "Check the AOI and choose qc_res = final DEM increment / positive integer."
        )
    return native, grid


def alignment(source, target):
    """Validate exact nested cells, tolerating only coordinate serialization noise."""
    a, b = source.transform, target["transform"]
    if (
        source.crs != target["crs"]
        or a.b
        or a.d
        or b.b
        or b.d
        or a.a <= 0
        or a.e >= 0
        or b.a <= 0
        or b.e >= 0
    ):
        raise ValueError("qc_res requires matching CRS and north-up grids")
    ratios = (b.a / a.a, b.e / a.e)
    if any(v < 1 - 1e-8 or not math.isclose(v, round(v), rel_tol=1e-8) for v in ratios):
        raise ValueError("QC and DEM cells must have integer-nested resolutions")
    offset = ((a.f - b.f) / a.e, (a.c - b.c) / a.a)
    if any(abs(v - round(v)) > 1e-5 for v in offset):
        raise ValueError("QC source origin is not aligned to the DEM grid")
    return tuple(round(v) for v in ratios), tuple(round(v) for v in offset)


def accepted_stream(
    path, mask_path, target, *, active_blocks=None, source_offset=(0, 0)
):
    """Bounded source-by-source aggregation; no averaging of finalized elevations."""
    active = (
        None
        if active_blocks is None
        else {tuple(map(int, block.split(","))) for block in active_blocks}
    )
    with rasterio.open(path) as src, rasterio.open(mask_path) as mask:
        (fx, fy), (oy, ox) = alignment(src, target)
        if (mask.shape, mask.crs, mask.transform) != (
            src.shape,
            src.crs,
            src.transform,
        ):
            raise ValueError("QC mask does not match the original source grid")
        if (
            fx == fy == 1
            and src.shape == (target["height"], target["width"])
            and src.transform == target["transform"]
        ):
            yield from masked_stream(path, mask_path)
            return
        if src.count != 7:
            raise ValueError("Expected seven additive source bands")
        yield target.copy()
        # Target blocks are bounded, and fine reads never exceed 256x256 even
        # for very large resolution ratios. Each coarse source cell is complete
        # before it reaches the native mixed/supercede/weighted reducer.
        for row in range(0, target["height"], 128):
            height = min(128, target["height"] - row)
            for col in range(0, target["width"], 128):
                width = min(128, target["width"] - col)
                top, bottom = (
                    max(0, row * fy - oy),
                    min(src.height, (row + height) * fy - oy),
                )
                left, right = (
                    max(0, col * fx - ox),
                    min(src.width, (col + width) * fx - ox),
                )
                if top >= bottom or left >= right:
                    continue
                out = np.zeros((7, height, width), dtype="float64")
                for sy in range(top, bottom, 256):
                    h = min(256, bottom - sy)
                    rows = (np.arange(sy, sy + h) + oy) // fy - row
                    for sx in range(left, right, 256):
                        w = min(256, right - sx)
                        columns = (np.arange(sx, sx + w) + ox) // fx - col
                        indices = (rows[:, None] * width + columns[None, :]).ravel()
                        if active is not None and not any(
                            (by, bx) in active
                            for by in range(
                                (sy + source_offset[0]) // 256,
                                (sy + source_offset[0] + h - 1) // 256 + 1,
                            )
                            for bx in range(
                                (sx + source_offset[1]) // 256,
                                (sx + source_offset[1] + w - 1) // 256 + 1,
                            )
                        ):
                            continue
                        win = Window(sx, sy, w, h)
                        flags = mask.read(1, window=win)
                        if np.any((flags != 0) & (flags != 1)):
                            raise ValueError("Invalid source rejection flags")
                        for band in range(7):
                            values = src.read(band + 1, window=win).astype("float64")
                            if not np.isfinite(values).all():
                                raise ValueError(
                                    "Non-finite additive source statistics"
                                )
                            values[flags == 1] = 0
                            out[band] += np.bincount(
                                indices,
                                weights=values.ravel(),
                                minlength=height * width,
                            ).reshape(height, width)
                if np.any(out[1] < 0) or np.any(out[2] < 0):
                    raise ValueError("Invalid aggregated count or weight")
                if np.any(out[1] > 0):
                    win = Window(col, row, width, height)
                    yield (
                        win,
                        win,
                        out,
                        None,
                        rasterio.windows.transform(win, target["transform"]),
                    )
