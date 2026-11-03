"""Evidence validity and native interpolation-grid coverage; no execution policy."""

from pathlib import Path


def _require_valid_evidence(path):
    """Stop before classification if the native stream produced no evidence."""
    import numpy as np
    import rasterio
    from ._science.extract_raw_surface import blocks

    with rasterio.open(path) as src:
        for window in blocks(src.width, src.height):
            if np.any(
                (src.read_masks(1, window=window) > 0)
                & np.isfinite(src.read(1, window=window))
            ):
                return
    raise ValueError(
        "Native unfilled evidence has no finite valid cells; check source "
        "coverage, stream CRS and transform/crop ordering before refinement"
    )


def pad_evidence_for_native_tiers(evidence, tier_resolutions):
    """Extend only NoData fringe to cover *actual* native decimator grids.

    Binary CUDEM decimates via Region(source bounds).geo_transform(node='grid').
    The grid node rounds dimensions independently at coarse resolutions. The
    southern/easternmost coarse pixels can therefore reach outside the finest
    stack. Plan that exact envelope with the same Region owner, then add blank
    whole finest-grid cells. No observations are resampled, invented or moved.
    The original unfilled evidence is preserved byte-for-byte under its name.
    """
    import math
    import numpy as np
    import rasterio
    from fetchez.spatial import Region
    from ._science.extract_raw_surface import blocks

    evidence = Path(evidence)
    if not tier_resolutions:
        return evidence, None
    with rasterio.open(evidence) as src:
        if (
            src.crs is None
            or src.transform.b
            or src.transform.d
            or src.transform.a <= 0
            or src.transform.e >= 0
        ):
            raise ValueError(
                "Tier preparation requires a north-up georeferenced unfilled evidence raster"
            )
        bounds, dx, dy = src.bounds, src.transform.a, -src.transform.e
        region = Region(bounds.left, bounds.right, bounds.bottom, bounds.top)
        region.srs = src.crs.to_string()
        extent_east, extent_south = bounds.right, bounds.bottom
        footprints = []
        for resolution in tier_resolutions:
            if not np.isfinite(resolution) or resolution <= 0:
                raise ValueError("Invalid native interpolation tier resolution")
            cols, rows, gt = region.geo_transform(
                x_inc=resolution, y_inc=resolution, node="grid"
            )
            if cols <= 0 or rows <= 0:
                raise ValueError("Interpolation tier has an empty working grid")
            east, south = gt[0] + cols * resolution, gt[3] - rows * resolution
            extent_east, extent_south = max(extent_east, east), min(extent_south, south)
            footprints.append(
                {
                    "resolution": resolution,
                    "shape": [rows, cols],
                    "bounds": [gt[0], south, east, gt[3]],
                }
            )
        # Tiny numeric cancellation must not introduce an extra cell.
        right_extra = max(0, math.ceil((extent_east - bounds.right) / dx - 1e-8))
        bottom_extra = max(0, math.ceil((bounds.bottom - extent_south) / dy - 1e-8))
        plan = {
            "original_shape": [src.height, src.width],
            "planned_shape": [src.height + bottom_extra, src.width + right_extra],
            "additional_blank_rows": bottom_extra,
            "additional_blank_cols": right_extra,
            "native_tiers": footprints,
            "crs": src.crs.to_string(),
        }
        if not right_extra and not bottom_extra:
            return evidence, plan
        if src.nodata is None:
            raise ValueError(
                "Cannot represent blank interpolation support without source NoData"
            )
        dest = evidence.with_name(evidence.stem + "_tier_support.tif")
        profile = src.profile.copy()
        profile.update(
            width=src.width + right_extra,
            height=src.height + bottom_extra,
            compress="deflate",
            BIGTIFF="IF_SAFER",
        )
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(dest, "w", **profile) as dst:
                # Mask all fringe cells before copying the unmodified original.
                for win in blocks(dst.width, dst.height):
                    dst.write(
                        np.full(
                            (int(win.height), int(win.width)),
                            src.nodata,
                            dtype=src.dtypes[0],
                        ),
                        1,
                        window=win,
                    )
                    dst.write_mask(
                        np.zeros((int(win.height), int(win.width)), dtype="uint8"),
                        window=win,
                    )
                for win in blocks(src.width, src.height):
                    dst.write(src.read(1, window=win), 1, window=win)
                    dst.write_mask(src.read_masks(1, window=win), window=win)
                dst.update_tags(**src.tags())
                dst.update_tags(
                    COAST_REFINE_SUPPORT_GRID="native_decimation_tiers",
                    COAST_REFINE_ADDED_BLANK_ROWS=str(bottom_extra),
                    COAST_REFINE_ADDED_BLANK_COLS=str(right_extra),
                )
                dst.scales = src.scales
                dst.offsets = src.offsets
                if src.units[0]:
                    dst.set_band_unit(1, src.units[0])
        with rasterio.open(dest) as padded:
            for win in blocks(src.width, src.height):
                if not np.array_equal(
                    src.read(1, window=win), padded.read(1, window=win), equal_nan=True
                ):
                    raise ValueError("Padding changed an original elevation pixel")
                if not np.array_equal(
                    src.read_masks(1, window=win), padded.read_masks(1, window=win)
                ):
                    raise ValueError("Padding changed an original validity bit")
            for tier in footprints:
                left, south, east, top = tier["bounds"]
                eps = 1e-8 * min(dx, dy)
                if (
                    left < padded.bounds.left - eps
                    or south < padded.bounds.bottom - eps
                    or east > padded.bounds.right + eps
                    or top > padded.bounds.top + eps
                ):
                    raise ValueError(
                        "Planned evidence does not cover native interpolation tier"
                    )
        return dest, plan


def publish_preparation_evidence(original, working, artifact_root):
    """Keep true unfilled source evidence separate from tier-support padding.

    Called only after successful science preparation. Return hashes for the
    manifest; scientific consumers continue using `working` before publishing.
    """
    from ._io import digest

    original, working, artifact_root = (
        Path(original),
        Path(working),
        Path(artifact_root),
    )
    root = artifact_root.resolve(strict=True)
    if not original.is_file() or not working.is_file():
        raise ValueError("Missing unfilled or padded preparation evidence")
    if original.resolve() == working.resolve():
        original.rename(root / "unmasked_evidence.tif")
        return {"unmasked_evidence_sha256": digest(root / "unmasked_evidence.tif")}
    working.rename(root / "tier_support_evidence.tif")
    original.rename(root / "unmasked_evidence.tif")
    return {
        "unmasked_evidence_sha256": digest(root / "unmasked_evidence.tif"),
        "tier_support_evidence_sha256": digest(root / "tier_support_evidence.tif"),
    }
