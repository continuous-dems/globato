"""Opt-in source comparisons using native Globato Perspecto; no scientific writes."""

from pathlib import Path
import tempfile
import logging
from urllib.parse import quote
import numpy as np
import rasterio
from ._io import digest
import html
import json
import math
import re
from PIL import Image, ImageDraw
from ._io import json_write

log = logging.getLogger(__name__)

SETTINGS = {
    "palette": "coastal_relief",
    "vertical_exaggeration": 2,
    "blend": "multiply",
    "alpha": True,
}


def preserve_failed_preparation(work, output_dir, request, error):
    """Retain unaccepted scratch artifacts before their temporary owner exits.

    Never turn diagnostic I/O failure into a different scientific outcome.
    Signed files retain their original bytes and paths; this receipt records
    the path mapping into the preserved copy for subsequent review.
    """
    import shutil

    try:
        root = Path(output_dir) / "failed_preparations"
        root.mkdir(parents=True, exist_ok=True)
        target = Path(tempfile.mkdtemp(prefix="failed_", dir=root))
        shutil.copytree(work, target, dirs_exist_ok=True)
        json_write(
            target / "failure.json",
            {
                "status": "FAILED_NOT_ACCEPTED",
                "error": str(error),
                "original_scratch_root": str(Path(work).resolve()),
                "preserved_root": str(target.resolve()),
                "request": request,
                "note": "Unaccepted diagnostics only; preserve signed artifact bytes and resolve original scratch paths below this root.",
            },
        )
        log.error("Failed Coast Mask preparation preserved for review: %s", target)
        return target
    except Exception:
        log.exception("Could not preserve failed preparation; original error retained")
        return None


def native_renderer():
    """Check the real render owner without importing the unrelated CLI."""
    from globato.hooks.viz.geohillshade import GeoHillshade

    if not callable(getattr(GeoHillshade, "_process_file_fallback", None)):
        raise RuntimeError("Installed Globato GeoHillshade lacks the required file API")
    return GeoHillshade


def render_perspecto(source, output, *, z_limits=None, max_dimension=None):
    """Render native Perspecto; max_dimension limits only a display grid."""
    GeoHillshade = native_renderer()
    source = Path(source).resolve(strict=True)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source) as src:
        if src.count != 1 or src.crs is None or not src.width or not src.height:
            raise ValueError(
                "Perspecto requires one georeferenced elevation band: " + str(source)
            )
    display_grid = None
    options = {}
    if z_limits is not None:
        with rasterio.open(source) as src:
            scale = (
                111120.0 if src.crs.is_geographic else src.crs.linear_units_factor[1]
            )
        options = dict(z_min=z_limits[0], z_max=z_limits[1], scale=scale)
    hook = GeoHillshade(
        cmap=SETTINGS["palette"],
        vert_exag=SETTINGS["vertical_exaggeration"],
        blend_mode=SETTINGS["blend"],
        alpha=SETTINGS["alpha"],
        **options,
    )
    try:
        # Native renderer reads NoData, but not independent GDAL mask bands.
        # A temporary display input makes all invalid cells NaN on the same grid.
        # Source rasters are read-only; only the temporary display is normalized.
        with tempfile.TemporaryDirectory(
            prefix="perspecto_", dir=output.parent
        ) as work:
            display = Path(work) / "display.tif"
            with rasterio.open(source) as src:
                profile = src.profile.copy()
                profile.update(
                    nodata=np.nan,
                    dtype="float64" if src.dtypes[0] == "float64" else "float32",
                )
                height, width = src.height, src.width
                if max_dimension is not None:
                    ratio = min(1.0, max_dimension / max(width, height))
                    height, width = (
                        max(1, round(height * ratio)),
                        max(1, round(width * ratio)),
                    )
                    profile.update(
                        height=height,
                        width=width,
                        transform=src.transform
                        * src.transform.scale(src.width / width, src.height / height),
                    )
                display_grid = {
                    "source_shape": [src.height, src.width],
                    "render_shape": [height, width],
                    "resampling": "nearest_preview_only"
                    if (height, width) != src.shape
                    else "none",
                }
                with rasterio.open(display, "w", **profile) as dst:
                    if (height, width) != src.shape:
                        # PNGs are at most 1024 pixels per side. Sample this
                        # display input first instead of shading a huge native
                        # raster merely to reduce it after rendering.
                        values = src.read(
                            1,
                            out_shape=(height, width),
                            resampling=rasterio.enums.Resampling.nearest,
                        ).astype(profile["dtype"])
                        valid = (
                            src.read_masks(
                                1,
                                out_shape=(height, width),
                                resampling=rasterio.enums.Resampling.nearest,
                            )
                            > 0
                        )
                        values[~valid | ~np.isfinite(values)] = np.nan
                        dst.write(values, 1)
                    else:
                        for _, window in src.block_windows(1):
                            values = src.read(1, window=window).astype(profile["dtype"])
                            values[
                                (src.read_masks(1, window=window) == 0)
                                | ~np.isfinite(values)
                            ] = np.nan
                            dst.write(values, 1, window=window)
            ok = hook._process_file_fallback(
                str(display),
                str(output),
                {"src_fn": str(source), "dst_fn": str(output)},
            )
        if not ok or not output.is_file():
            raise RuntimeError(
                "Native Globato Perspecto rendering failed: " + str(output)
            )
        with rasterio.open(output) as rendered:
            if (
                rendered.count != 4
                or rendered.crs is None
                or not rendered.width
                or not rendered.height
            ):
                raise ValueError(
                    "Invalid georeferenced RGBA Perspecto output: " + str(output)
                )
        return {
            "path": str(output),
            "sha256": digest(output),
            "settings": {**SETTINGS, **options},
            "renderer": "globato.hooks.viz.geohillshade.GeoHillshade",
            "display_grid": display_grid,
        }
    except Exception:
        output.unlink(missing_ok=True)
        raise


def pair_statistics(original, masked):
    """Read native crops without interpolation; verify retained bits and validity."""
    low, high = np.inf, -np.inf
    before = after = 0
    with rasterio.open(original) as a, rasterio.open(masked) as b:
        if (a.crs, a.transform, a.shape, a.dtypes, a.count) != (
            b.crs,
            b.transform,
            b.shape,
            b.dtypes,
            b.count,
        ) or a.count != 1:
            raise ValueError(
                "Original and masked crops must have identical single-band native grids"
            )
        for _, window in a.block_windows(1):
            x = a.read(1, window=window)
            y = b.read(1, window=window)
            av = (a.read_masks(1, window=window) > 0) & np.isfinite(x)
            bv = (b.read_masks(1, window=window) > 0) & np.isfinite(y)
            if np.any(bv & ~av) or x[bv].tobytes() != y[bv].tobytes():
                raise ValueError(
                    "Masked crop restored original invalid data or changed retained elevation bits"
                )
            if av.any():
                low = min(low, float(x[av].min()))
                high = max(high, float(x[av].max()))
            before += int(av.sum())
            after += int(bv.sum())
    if not before:
        low, high = 0.0, 1.0
    if low == high:
        low, high = low - 0.5, high + 0.5
    return {
        "original_valid_cells": before,
        "retained_cells": after,
        "removed_cells": before - after,
        "shared_color_range_m": [low, high],
        "retained_bits_verified": True,
    }


def comparison(before, after, target, title, stats):
    def preview(path):
        with rasterio.open(path) as src:
            scale = min(1.0, 1024 / src.width, 1024 / src.height)
            image = Image.fromarray(
                np.moveaxis(
                    src.read(
                        out_shape=(
                            4,
                            max(1, round(src.height * scale)),
                            max(1, round(src.width * scale)),
                        )
                    ),
                    0,
                    -1,
                )
            )
        background = Image.new("RGBA", image.size, "#e5e5e5")
        background.alpha_composite(image)
        return background.convert("RGB")

    a, b = preview(before), preview(after)
    width = max(400, a.width)
    height = a.height + 110
    output = Image.new("RGB", (2 * width + 24, height), "white")
    draw = ImageDraw.Draw(output)
    draw.text((12, 10), title, fill="black")
    draw.text(
        (12, 34),
        f"ORIGINAL AOI: {stats['original_valid_cells']:,} valid cells",
        fill="black",
    )
    draw.text(
        (width + 12, 34),
        f"MASKED AOI: {stats['retained_cells']:,} retained cells",
        fill="black",
    )
    output.paste(a, (12, 56))
    output.paste(b, (width + 12, 56))
    draw.text(
        (12, height - 38),
        "Same Perspecto palette, elevation range, lighting and vertical exaggeration. Grey = NoData.",
        fill="black",
    )
    draw.text(
        (12, height - 21),
        "Hillshade previews may be reduced to 1024 pixels/side. Scientific TIFFs and native checks retain full resolution.",
        fill="black",
    )
    output.save(target)


def _crop_window(source, receipt):
    """Bound a display crop using the source's sealed coordinate operation."""
    import shapely
    from pyproj import Transformer, enums
    from rasterio.windows import from_bounds, Window

    scope = shapely.from_wkb(receipt["processing_inputs"]["scope_wkb"])
    step = (
        max(scope.bounds[2] - scope.bounds[0], scope.bounds[3] - scope.bounds[1]) / 64
    )
    xy = shapely.get_coordinates(scope.segmentize(step))
    operation = receipt["processing_inputs"]["coordinate_operation"]["pipeline"]
    x, y = Transformer.from_pipeline(operation).transform(
        xy[:, 0], xy[:, 1], direction=enums.TransformDirection.INVERSE
    )
    with rasterio.open(source) as src:
        fractional = from_bounds(min(x), min(y), max(x), max(y), src.transform)
        col, row = math.floor(fractional.col_off), math.floor(fractional.row_off)
        window = Window(
            col,
            row,
            math.ceil(fractional.col_off + fractional.width) - col,
            math.ceil(fractional.row_off + fractional.height) - row,
        )
        return window.intersection(Window(0, 0, src.width, src.height))


def render_mask_comparison(artifact, target):
    """Display independent source-use and interpolation products together."""
    import geopandas as gpd
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    coverage = gpd.read_file(artifact.products["coverage"])
    bounds = coverage.total_bounds
    figure = Figure(figsize=(18, 5), layout="constrained")
    FigureCanvasAgg(figure)
    for axis, key, title in zip(
        figure.subplots(1, 3),
        ("original_water", "retention_water", "refined_water"),
        ("Starting water", "Source-use water exclusion", "Interpolation water domain"),
        strict=True,
    ):
        axis.set_facecolor("#d9d3c7")
        water = gpd.read_file(artifact.products[key]).to_crs(coverage.crs)
        if not water.empty:
            water.plot(ax=axis, color="#4285b4", edgecolor="none")
        axis.set_xlim(bounds[0], bounds[2])
        axis.set_ylim(bounds[1], bounds[3])
        axis.set_title(title)
        axis.ticklabel_format(useOffset=False)
    figure.savefig(target, dpi=120)
    figure.clear()


def write_review_index(root):
    """One stable entry page for all module-attached reviews in this run."""
    root = Path(root)
    pages = sorted(root.glob("coast_refine_*/sources/review.html"))
    links = [
        '<!doctype html><meta charset="utf-8"><title>Coast Refine review</title>',
        "<style>body{font:16px system-ui;margin:2rem;max-width:1000px}li{margin:1rem 0}</style>",
        "<h1>Coast Refine before/after review</h1>",
        "<p>Compare the starting water, source-retention mask, independent interpolation polygon, and original/masked source elevations.</p><ul>",
    ]
    for page in pages:
        data = json.loads((page.parent / "review.json").read_text())
        label = data["selection"] + " — " + str(len(data["sources"])) + " sources"
        links.append(
            '<li><a href="'
            + quote(page.relative_to(root).as_posix())
            + '">'
            + html.escape(label)
            + "</a></li>"
        )
    (root / "review.html").write_text("".join(links) + "</ul>")
    return root / "review.html"


def render_source_review(artifact, receipts, output):
    """Opt-in PNG gallery from verified source products; crops/shades are temporary.

    Publication already checked every native source pixel. This function checks
    only its display crops and never creates a second engineering-validation pass,
    a full review ZIP, or permanent elevation/hillshade TIFF copies.
    """
    from ._science.extract_raw_surface import copy_band_one
    import shutil
    import time

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        saved = output / "review.json"
        if (
            saved.is_file()
            and (output / "review.html").is_file()
            and json.loads(saved.read_text()).get("processing_ids")
            == sorted(r["processing_id"] for r in receipts)
        ):
            return output
        raise FileExistsError(
            "Review already exists with different source processing: " + str(output)
        )
    started = time.monotonic()
    records = []
    with tempfile.TemporaryDirectory(
        prefix="review_pending_", dir=output.parent
    ) as work:
        work = Path(work)
        published = work / "published"
        published.mkdir()
        for index, receipt in enumerate(receipts, 1):
            name = receipt["source_identity"]["native_name"]
            log.info("Coast Refine review [%s/%s]: %s", index, len(receipts), name)
            label = (
                re.sub(r"[^a-z0-9_.-]+", "_", Path(name).stem.lower())[:100]
                + "_"
                + receipt["processing_id"][:10]
            )
            window = _crop_window(receipt["original"], receipt)
            with tempfile.TemporaryDirectory(prefix="source_", dir=work) as source_work:
                source_work = Path(source_work)
                original = source_work / "original.tif"
                masked = source_work / "masked.tif"
                copy_band_one(receipt["original"], original, window)
                copy_band_one(receipt["masked"], masked, window)
                stats = pair_statistics(original, masked)
                for role, path in [("original", original), ("masked", masked)]:
                    log.info("Perspecto %s: %s", role, name)
                    rendered = render_perspecto(
                        path,
                        source_work / (role + "_shade.tif"),
                        z_limits=stats["shared_color_range_m"],
                        max_dimension=1024,
                    )
                    stats[role + "_display_grid"] = rendered["display_grid"]
                png = published / (label + "__before_after.png")
                comparison(
                    source_work / "original_shade.tif",
                    source_work / "masked_shade.tif",
                    png,
                    name
                    + " | "
                    + ("remove: " + receipt["processing_inputs"]["remove"]),
                    stats,
                )
            records.append(
                {
                    "source": name,
                    "comparison_png": png.name,
                    "statistics": stats,
                    "original": receipt["original"],
                    "masked": receipt["masked"],
                    "processing_id": receipt["processing_id"],
                }
            )
        # Small reference vectors belong with the review, not duplicate TIFF sets.
        for key in (
            "original_water",
            "original_land",
            "original_coastline",
            "refined_topology",
            "refined_water",
            "refined_land",
            "added_land",
            "added_water",
            "retention_water",
            "retention_land",
            "retention_marine_water",
            "retention_classified_ocean",
            "retention_classified_inland_water",
        ):
            if key in artifact.products:
                shutil.copy2(artifact.products[key], published / (key + ".gpkg"))
        render_mask_comparison(artifact, published / "mask_before_after.png")
        links = [
            '<!doctype html><meta charset="utf-8"><title>Coast Refine review</title>',
            "<style>body{font:16px system-ui;margin:24px;max-width:1500px}img{max-width:100%}</style>",
            "<h1>Coast Refine source review</h1><p>Hillshade previews use at most 1024 pixels per side. Scientific source TIFFs are unchanged. Grey is NoData.</p>",
            '<h2>Source retention and interpolation are separate</h2><p>Below-zero starting land is not removed by a signed barrier water addition. Source-use polygons are evaluated at native source cell centres. The interpolation polygon constrains generated terrain.</p><img src="mask_before_after.png" alt="Starting water, source-use mask and interpolation domain">',
            "<h2>Native source comparisons</h2>",
        ]
        for vector in sorted(published.glob("*.gpkg")):
            links.append(
                '<a href="'
                + html.escape(vector.name)
                + '">'
                + html.escape(vector.stem)
                + "</a> · "
            )
        for record in records:
            links += [
                "<h2>" + html.escape(record["source"]) + "</h2>",
                '<img src="'
                + html.escape(record["comparison_png"])
                + '" alt="Original and masked elevation">',
            ]
        (published / "review.html").write_text("".join(links))
        json_write(
            published / "review.json",
            {
                "preparation_identity": artifact.preparation_id,
                "selection": ("remove: " + receipts[0]["processing_inputs"]["remove"]),
                "processing_ids": sorted(r["processing_id"] for r in receipts),
                "sources": records,
                "temporary_raster_copies_retained": False,
                "review_zip_created": False,
                "preview_max_dimension": 1024,
            },
        )
        published.rename(output)
    log.info(
        "Coast Refine review complete (%.1fs): %s",
        time.monotonic() - started,
        output / "review.html",
    )
    return output
