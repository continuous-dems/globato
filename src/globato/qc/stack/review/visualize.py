"""Optional review figures, deliberately separate from QC scientific code."""

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
from matplotlib.ticker import MaxNLocator
from rasterio.enums import Resampling
from rasterio.warp import reproject
import numpy as np
import rasterio

from globato.qc.stack.grid import metres, grid_offset

# A full CUDEM tile is about 66 million cells. Pictures never require that
# many pixels; the original GeoTIFFs and QC masks remain full resolution.
logger = logging.getLogger(__name__)

MAX_REVIEW_PIXELS = 2_000_000


def _thumbnail_size(src, max_pixels):
    if max_pixels is None or src.width * src.height <= max_pixels:
        return src.height, src.width
    scale = (max_pixels / (src.width * src.height)) ** 0.5
    return max(1, int(src.height * scale)), max(1, int(src.width * scale))


def surface(path, *, max_pixels=None):
    """Display only; downsample very large surfaces before allocating arrays."""
    with rasterio.open(path) as ds:
        height, width = _thumbnail_size(ds, max_pixels)
        z = (
            ds.read(
                1, out_shape=(height, width), masked=True, resampling=Resampling.average
            )
            .astype("float64")
            .filled(np.nan)
        )
        dx, dy = metres(ds)
        return (
            z,
            (dx * ds.width / width, dy * ds.height / height),
            ds.bounds,
            ds.crs.is_geographic,
        )


def removed(manifest, dem, restored_from=None, *, max_pixels=None):
    """Review-only source-mask footprints, read in bounded source windows.

    Preserve isolated rejected/restored cells when reducing to thumbnails.
    Each source block is projected to the display grid independently; this
    never changes the actual QC masks or their full-resolution count.
    """
    from rasterio.windows import Window
    from affine import Affine

    with rasterio.open(dem) as grid:
        height, width = _thumbnail_size(grid, max_pixels)
        reduced = (height, width) != (grid.height, grid.width)
        result = np.zeros((height, width), bool)
        target_transform = grid.transform * Affine.scale(
            grid.width / width, grid.height / height
        )
        for sid, entry in manifest["masks"].items():
            with rasterio.open(entry["mask"]) as src:
                dy, dx = grid_offset(src, grid)
                before_context = (
                    rasterio.open(restored_from["masks"][sid]["mask"])
                    if restored_from is not None
                    else None
                )
                try:
                    if before_context is not None and (
                        grid_offset(before_context, src) != (0, 0)
                        or before_context.shape != src.shape
                    ):
                        raise ValueError(
                            "Restoration review masks must share the same source grid"
                        )
                    # The source can be larger than the global DEM or sparse,
                    # so clip each 512-cell read to the overlap. Never allocate
                    # a complete quarter-degree source-mask array here.
                    row0 = max(0, -dy)
                    col0 = max(0, -dx)
                    row1 = min(src.height, grid.height - dy)
                    col1 = min(src.width, grid.width - dx)
                    if row1 <= row0 or col1 <= col0:
                        continue
                    for sy in range(row0, row1, 512):
                        ey = min(sy + 512, row1)
                        for sx in range(col0, col1, 512):
                            ex = min(sx + 512, col1)
                            window = Window(sx, sy, ex - sx, ey - sy)
                            flags = src.read(1, window=window) == 1
                            if before_context is not None:
                                flags = (
                                    before_context.read(1, window=window) == 1
                                ) & ~flags
                            if not flags.any():
                                continue
                            if not reduced:
                                result[sy + dy : ey + dy, sx + dx : ex + dx] |= flags
                                continue
                            # Coordinates are from the *global* grid. A one-cell
                            # border absorbs warp-kernel rounding at block seams.
                            row_start = max(
                                0, int(np.floor((sy + dy) * height / grid.height)) - 1
                            )
                            row_stop = min(
                                height,
                                int(np.ceil((ey + dy) * height / grid.height)) + 1,
                            )
                            col_start = max(
                                0, int(np.floor((sx + dx) * width / grid.width)) - 1
                            )
                            col_stop = min(
                                width, int(np.ceil((ex + dx) * width / grid.width)) + 1
                            )
                            if row_stop <= row_start or col_stop <= col_start:
                                continue
                            small = np.zeros(
                                (row_stop - row_start, col_stop - col_start),
                                dtype="uint8",
                            )
                            reproject(
                                source=flags.astype("uint8"),
                                destination=small,
                                src_transform=src.window_transform(window),
                                src_crs=src.crs,
                                dst_transform=target_transform
                                * Affine.translation(col_start, row_start),
                                dst_crs=grid.crs,
                                src_nodata=0,
                                dst_nodata=0,
                                resampling=Resampling.max,
                            )
                            result[row_start:row_stop, col_start:col_stop] |= small != 0
                finally:
                    if before_context is not None:
                        before_context.close()
    return result


def expanded_cells(mask, radius=3):
    """Enlarge point symbols for a picture; never change scientific masks."""
    mask = np.asarray(mask, dtype=bool)
    output = np.zeros_like(mask)
    rows, cols = mask.shape
    for y in range(-radius, radius + 1):
        for x in range(-radius, radius + 1):
            src_y0, src_y1 = max(0, -y), min(rows, rows - y)
            src_x0, src_x1 = max(0, -x), min(cols, cols - x)
            if src_y1 > src_y0 and src_x1 > src_x0:
                output[src_y0 + y : src_y1 + y, src_x0 + x : src_x1 + x] |= mask[
                    src_y0:src_y1, src_x0:src_x1
                ]
    return output


def footprint_colors(rejected, restored=None):
    """Red rejected cells, with enlarged black restored symbols drawn last."""
    result = np.zeros((*rejected.shape, 4), dtype="float32")
    result[rejected] = (1.0, 0.05, 0.1, 0.75)
    if restored is not None and restored.any():
        result[expanded_cells(restored, 4)] = (1, 1, 1, 0.93)
        result[expanded_cells(restored, 3)] = (0, 0, 0, 1)
    return result


def measured_cells(path, *, max_pixels=None):
    """Read native *measured* coverage, never the interpolated DEM footprint."""
    with rasterio.open(path) as ds:
        if ds.count != 7 or ds.tags().get("GLOBATO_DATATYPE") != "MULTI_STACK":
            raise ValueError(
                "The final measured-cell view requires a native seven-band MultiStack"
            )
        height, width = _thumbnail_size(ds, max_pixels)
        if (height, width) == (ds.height, ds.width):
            z = ds.read(1, masked=True).astype("float64").filled(np.nan)
            counts = ds.read(2, masked=True)
            coverage = np.asarray(counts.filled(0) > 0) & np.isfinite(z)
        else:
            # A source MultiStack's band 1 may hold zero even at unmeasured
            # locations. Average only contributions with count > 0, in native
            # blocks, without loading seven full-resolution bands into RAM.
            total = np.zeros((height * width,), dtype="float64")
            weights = np.zeros_like(total)
            for _, window in ds.block_windows(1):
                values = ds.read(1, window=window)
                counts = ds.read(2, window=window)
                valid = (counts > 0) & np.isfinite(values)
                if not valid.any():
                    continue
                rows, cols = np.nonzero(valid)
                yy = ((rows + int(window.row_off)) * height) // ds.height
                xx = ((cols + int(window.col_off)) * width) // ds.width
                indexes = yy * width + xx
                w = counts[valid].astype("float64")
                np.add.at(weights, indexes, w)
                np.add.at(total, indexes, values[valid] * w)
            coverage = weights.reshape(height, width) > 0
            z = np.full((height, width), np.nan, dtype="float64")
            z[coverage] = (
                total.reshape(height, width)[coverage]
                / weights.reshape(height, width)[coverage]
            )
        return z, coverage, ds.bounds, ds.crs.is_geographic


def render(
    dem,
    output,
    baseline=None,
    manifest=None,
    title="DEM",
    restored_from=None,
    restored_cells=None,
):
    z, (dx, dy), bounds, geographic = surface(dem, max_pixels=MAX_REVIEW_PIXELS)
    if not np.isfinite(z).any():
        raise ValueError(f"Cannot render an empty DEM: {dem}")
    reference = surface(baseline, max_pixels=MAX_REVIEW_PIXELS)[0] if baseline else z
    vmin, vmax = np.nanpercentile(reference, [1, 99])
    vmax = max(vmax, vmin + 1)
    norm = Normalize(vmin, vmax)
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    fig, axes = plt.subplots(
        1,
        2 if manifest else 1,
        figsize=(12 if manifest else 7, min(12, max(6, 6 * z.shape[0] / z.shape[1]))),
        squeeze=False,
        sharey=True,
        layout="constrained",
    )
    filled = np.where(np.isfinite(z), z, np.nanmedian(z))
    light = LightSource(azdeg=315, altdeg=45)
    rgb = light.shade(
        filled,
        cmap=plt.get_cmap("gist_earth"),
        norm=norm,
        dx=dx,
        dy=dy,
        vert_exag=1,
        blend_mode="soft",
    )
    rgb[~np.isfinite(z), :3] = 1
    for ax in axes.flat:
        ax.imshow(rgb, extent=extent, interpolation="nearest")
        ax.set_xlabel("Longitude" if geographic else "Easting (m)")
        ax.set_ylabel("Latitude" if geographic else "Northing (m)")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.ticklabel_format(style="plain", useOffset=False)
    axes[0, 0].set_title(title, fontsize=12)
    if manifest:
        original = surface(baseline or dem, max_pixels=MAX_REVIEW_PIXELS)[0]
        original_rgb = light.shade(
            np.where(np.isfinite(original), original, np.nanmedian(original)),
            cmap=plt.get_cmap("gist_earth"),
            norm=norm,
            dx=dx,
            dy=dy,
            blend_mode="soft",
        )
        original_rgb[~np.isfinite(original), :3] = 1
        axes[0, 1].set_ylabel("")
        axes[0, 1].images[0].set_data(original_rgb)
        restored = (
            removed(manifest, dem, restored_from, max_pixels=MAX_REVIEW_PIXELS)
            if restored_from is not None
            else restored_cells
        )
        axes[0, 1].imshow(
            footprint_colors(
                removed(manifest, dem, max_pixels=MAX_REVIEW_PIXELS), restored
            ),
            extent=extent,
            interpolation="nearest",
        )
        axes[0, 1].set_title(
            "Rejected (red), finally restored (black)"
            if restored is not None
            else "Rejected locations on original terrain",
            fontsize=12,
        )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="gist_earth"),
        ax=list(axes.flat),
        label="Elevation (m)",
        shrink=0.75,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info("Review PNG ready: %s", output)


def render_measurements(stack, baseline, output, *, restored=None):
    """Show only restored sites, or all final measured locations and elevations."""
    z, measured, bounds, geographic = measured_cells(
        stack, max_pixels=MAX_REVIEW_PIXELS
    )
    if restored is not None and restored.shape != measured.shape:
        raise ValueError(
            "Restoration footprint must match the final measured-cell grid"
        )
    with rasterio.open(stack) as source:
        thumbnail = (source.height, source.width) != z.shape
    location_label = "display locations" if thumbnail else "grid cells"
    selected = measured if restored is None else (measured & restored)
    terrain, (dx, dy), original_bounds, _ = surface(
        baseline, max_pixels=MAX_REVIEW_PIXELS
    )
    if terrain.shape != z.shape or tuple(bounds) != tuple(original_bounds):
        raise ValueError("Original terrain and final measured grid do not match")
    valid = np.isfinite(terrain)
    vmin, vmax = np.nanpercentile(terrain[valid], [1, 99])
    norm = Normalize(vmin, max(vmin + 1, vmax))
    base = LightSource(azdeg=315, altdeg=45).shade(
        np.where(valid, terrain, np.nanmedian(terrain)),
        cmap=plt.get_cmap("Greys"),
        norm=norm,
        dx=dx,
        dy=dy,
        blend_mode="soft",
    )
    base[..., :3] = 0.84 + 0.16 * base[..., :3]
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    fig, ax = plt.subplots(
        figsize=(8, min(12, max(6, 8 * z.shape[0] / z.shape[1]))),
        layout="constrained",
    )
    ax.imshow(base, extent=extent, interpolation="nearest")
    cmap = plt.get_cmap("gist_earth").copy()
    cmap.set_bad((0, 0, 0, 0))
    if restored is None:
        ax.imshow(
            np.ma.masked_where(~selected, z),
            cmap=cmap,
            norm=norm,
            extent=extent,
            interpolation="nearest",
        )
        title = f"All final measured cells ({np.count_nonzero(selected):,} {location_label})"
    else:
        rows, cols = np.nonzero(selected)
        if len(rows) > 50_000:
            # Millions of Matplotlib scatter artists are expensive. At this
            # scale one display pixel already represents several native cells.
            ax.imshow(
                np.ma.masked_where(~selected, z),
                cmap=cmap,
                norm=norm,
                extent=extent,
                interpolation="nearest",
                zorder=4,
            )
        elif len(rows):
            with rasterio.open(stack) as ds:
                display_transform = ds.transform * rasterio.Affine.scale(
                    ds.width / z.shape[1], ds.height / z.shape[0]
                )
                xs, ys = rasterio.transform.xy(display_transform, rows, cols)
            ax.scatter(
                xs,
                ys,
                c=z[selected],
                cmap=cmap,
                norm=norm,
                s=26,
                marker="s",
                linewidths=0.5,
                edgecolors="black",
                zorder=4,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No restored measured cells in this run",
                transform=ax.transAxes,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="#999999", alpha=0.95),
            )
        title = f"Restored sites only ({np.count_nonzero(selected):,} {location_label})"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Longitude" if geographic else "Easting (m)")
    ax.set_ylabel("Latitude" if geographic else "Northing (m)")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.ticklabel_format(style="plain", useOffset=False)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        label="Final measured elevation (m)",
        shrink=0.75,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info("Review PNG ready: %s", output)


def completed(run, *, cache, result, stack, region, final):
    """Publish ordered review stages and module-only review products."""
    run = Path(run)
    from .final_stages import (
        final_measured_stack,
        last_pass_stack,
        module_interpolated_preview,
        write_hillshade_tif,
        write_measured_surface,
    )
    from .review_page import publish
    from .perspecto import colored_hillshade, reference_limits

    result = dict(result)
    settings = result["settings"]
    count = int(settings["max_passes"])
    complete = Path(final).is_file()
    stages = [dict(label="Original interpolated DEM", status="Pending")]
    stages += [
        dict(label=f"After pass {i}", status="Pending") for i in range(1, count + 1)
    ]
    stages += [
        dict(label="Measured terrain after restoration", status="Pending"),
        dict(label="Only the restored measured cells", status="Pending"),
        dict(label="All final measured cells", status="Pending"),
        dict(label="Module-only interpolated review DEM", status="Pending"),
        dict(label="Final combined Globato DEM", status="Pending"),
    ]
    details = (
        f"W/E/S/N: {'/'.join(str(v) for v in region.to_list())}. "
        f"Filtering aggressiveness {settings.get('aggressiveness', 2)}; "
        f"terrain restoration {settings.get('terrain_restoration', 2)}; "
        f"up to {count} passes."
    )
    if "qc_res" in result:
        details += f" QC grid: {result['qc_res']} coordinate units; final DEM grid: {result['dem_resolution']}."
    root = Path(cache)
    baseline_receipt = json.loads((root / "baseline/COMPLETE.json").read_text())
    baseline = baseline_receipt["dem"]
    colors = reference_limits(baseline)

    def picture(
        name, dem, title, manifest=None, restored_from=None, restored_cells=None
    ):
        dest = run / "review" / name
        render(
            dem,
            dest,
            baseline,
            manifest,
            title,
            restored_from,
            restored_cells=restored_cells,
        )
        return str(dest.relative_to(run))

    first_color = colored_hillshade(
        baseline,
        run / "review" / "00_initial_colored_hillshade.tif",
        color_limits=colors,
    )
    stages[0].update(
        image=picture("00_original.png", baseline, "Original interpolated DEM"),
        status="Initial module interpolation before cleaning",
        downloads=[
            ("Initial colored hillshade (GeoTIFF)", str(first_color.relative_to(run)))
        ],
    )
    if not (root / "zero/MANIFEST.json").exists():
        publish(run, stages, complete=False, details=details)
        return sorted((run / "review").glob("*.png"))
    parent = json.loads((root / "zero/MANIFEST.json").read_text())
    previous = baseline_receipt
    previous_dem = baseline
    last_manifest = None
    final_manifest = (
        json.loads(Path(result["manifest"]).read_text())
        if complete and result
        else None
    )
    actual = result["passes_completed"] if result else count
    pre_restoration = (
        json.loads(Path(result["history"][actual - 1]["manifest"]).read_text())
        if final_manifest is not None and actual
        else None
    )
    final_restored = (
        removed(final_manifest, baseline, pre_restoration, max_pixels=MAX_REVIEW_PIXELS)
        if pre_restoration is not None
        else None
    )
    for i in range(1, count + 1):
        if result and i > actual:
            stages[i]["status"] = "Not run — early stopping"
            continue
        path = Path(result["history"][i - 1]["manifest"]) if result else None
        if path is None or not path.exists():
            break
        current = json.loads(path.read_text())
        last_manifest = current
        context = root / "dems" / current["signature"] / "COMPLETE.json"
        dem = None
        status = "Removal footprint ready; interpolated context pending"
        if context.exists():
            previous = json.loads(context.read_text())
            dem = previous["dem"]
            status = "Filtered interpolated DEM before restoration"
        elif (
            current["signature"] == parent["signature"]
            or current["rejected_source_cells"] == 0
        ):
            dem = previous_dem
            status = "Unchanged mask state — existing interpolated DEM reused"
        elif complete and i == actual:
            dem = last_pass_stack(run, root, current, previous, stack, region)
            status = "Retained measured cells; gaps open — no extra interpolation"
        if dem is not None:
            stages[i]["image"] = picture(
                f"{i:02}_pass_{i:02}.png",
                dem,
                "Retained measured cells after last pass"
                if complete and i == actual and not context.exists()
                else "Filtered interpolated DEM before restoration",
            )
            previous_dem = dem
        stages[i].update(
            status=f"{status}. New rejections: {current['newly_rejected_source_cells']:,}; cumulative: {current['rejected_source_cells']:,}.",
            overlay=picture(
                f"{i:02}_pass_{i:02}_footprint.png",
                baseline,
                "Removal footprint on original terrain",
                current,
                restored_cells=removed(
                    final_manifest, baseline, current, max_pixels=MAX_REVIEW_PIXELS
                )
                if final_manifest is not None
                else None,
            ),
        )
        parent = current
    final = Path(final)
    if complete and result and final.exists() and final_manifest is not None:
        measured = final_measured_stack(run, root, final_manifest, stack, region)
        measured_surface = write_measured_surface(
            measured, run / "review" / "module_final_measured.tif"
        )
        write_hillshade_tif(
            measured_surface, run / "review" / "module_final_measured_hillshade.tif"
        )
        stages[-5].update(
            downloads=[
                ("Cleaned measurements (GeoTIFF)", "review/module_final_measured.tif"),
                (
                    "Measured terrain hillshade (GeoTIFF)",
                    "review/module_final_measured_hillshade.tif",
                ),
            ],
            image=picture(
                "90_restored_terrain.png",
                measured,
                "Measured terrain after restoration",
            ),
            overlay=picture(
                "91_restoration_footprint.png",
                measured,
                "Measured terrain after restoration",
                final_manifest,
                last_manifest,
            ),
            status=(
                f"Module-only measured terrain after restoration, including "
                f"{result['restored_source_cells']:,} restored source-cell contributions; "
                f"gaps remain open. GeoTIFFs: review/module_final_measured.tif and "
                f"review/module_final_measured_hillshade.tif"
            ),
        )
        restored = final_restored
        _, measured_coverage, _, _ = measured_cells(
            measured, max_pixels=MAX_REVIEW_PIXELS
        )
        with rasterio.open(measured) as native_grid:
            thumbnail = (
                native_grid.height,
                native_grid.width,
            ) != measured_coverage.shape
        restored_label = "restored display locations" if thumbnail else "restored sites"
        measured_label = "displayed locations" if thumbnail else "final grid cells"
        for index, filename, selected in (
            (-4, "92_restored_cells_only.png", restored),
            (-3, "93_all_final_measured_cells.png", None),
        ):
            destination = run / "review" / filename
            render_measurements(measured, baseline, destination, restored=selected)
            stages[index].update(
                image=str(destination.relative_to(run)),
                status=(
                    f"{np.count_nonzero(restored & measured_coverage):,} {restored_label} with final measurements; "
                    f"{result['restored_source_cells']:,} source-cell contributions restored."
                    if selected is not None
                    else f"{np.count_nonzero(measured_coverage):,} {measured_label} with measurements, before interpolation."
                ),
            )
        preview_dem = module_interpolated_preview(
            measured_surface, run / "review" / "module_cleaned_interpolated.tif"
        )
        write_hillshade_tif(
            preview_dem, run / "review" / "module_cleaned_interpolated_hillshade.tif"
        )
        preview_color = colored_hillshade(
            preview_dem,
            run / "review" / "94_module_only_colored_hillshade.tif",
            color_limits=colors,
        )
        stages[-2].update(
            downloads=[
                (
                    "Module-only colored hillshade (GeoTIFF)",
                    str(preview_color.relative_to(run)),
                )
            ]
            + [
                (
                    "Module-only preview (GeoTIFF)",
                    "review/module_cleaned_interpolated.tif",
                ),
                (
                    "Module-only preview hillshade (GeoTIFF)",
                    "review/module_cleaned_interpolated_hillshade.tif",
                ),
            ],
            image=picture(
                "94_module_only_interpolated.png",
                preview_dem,
                "Module-only interpolated review DEM",
            ),
            status=(
                "Bounded short-range review interpolation built from this module's final cleaned measured cells; not the production Globato algorithm. "
                "GeoTIFFs: review/module_cleaned_interpolated.tif and "
                "review/module_cleaned_interpolated_hillshade.tif"
            ),
        )
        final_color = colored_hillshade(
            final,
            run / "review" / "99_globato_colored_hillshade.tif",
            color_limits=colors,
        )
        stages[-1].update(
            downloads=[
                (
                    "Globato final colored hillshade (GeoTIFF)",
                    str(final_color.relative_to(run)),
                )
            ],
            image=picture("99_final_dem.png", final, "Final combined Globato DEM"),
            status="Combined Globato interpolation after all contributing modules are stacked",
        )
    publish(run, stages, complete=complete, details=details)
    return sorted((run / "review").glob("*.png"))
