"""Native TIFF masking and bit-preservation checks; no host discovery or streaming."""

from pathlib import Path
import json
import os
import tempfile
from filelock import FileLock
import numpy as np
import rasterio
import shapely
from ._science.input_contracts import check_encoding
from ._io import digest, identity, json_write, raster_files
from .selection import horizontal_crs


def source_operation(native):
    """Seal the selected operation so review/reuse cannot select another datum route."""
    import pyproj

    transformer = pyproj.Transformer.from_crs(
        horizontal_crs(native.crs), "EPSG:4326", always_xy=True
    )
    transformer.transform(
        (native.bounds.left + native.bounds.right) / 2,
        (native.bounds.bottom + native.bounds.top) / 2,
    )
    try:
        transformer = transformer.get_last_used_operation()
    except RuntimeError:
        # Identity transforms have no last-used operation; their definition is
        # already the complete, reproducible noop pipeline.
        if transformer.definition.startswith("unavailable"):
            raise
    return {
        "pipeline": transformer.definition,
        "description": transformer.description,
        "accuracy_m": transformer.accuracy,
        "proj_version": pyproj.proj_version_str,
    }


def _native_classification(native, geographic, window, transformer=None):
    """OSM keep decision at original native cell centers; no vector reprojection.

    GDAL rasterization of projected sub-cell slivers is deliberately avoided.
    Only the approved geographic geometry decides which source observations
    survive. Work is bounded by the native raster's individual blocks.
    """
    from .selection import centre_selection

    return centre_selection(native, geographic, window, transformer=transformer)


def _source_valid(src, window, values):
    """Preserve GDAL mask / NoData AND reject NaN/inf, never guess elevations."""
    return (src.read_masks(1, window=window) > 0) & np.isfinite(values)


def validate_product(source, output, geographic, operation=None, *, progress=None):
    """Verify native grid, exact retained elevation bits, and NaN exclusions."""
    from ._science.extract_raw_surface import blocks, verify_self_contained_raster

    # Cache reuse enters here without the writer's geometry preparation.
    # Shapely 2.0 does not prepare contains_xy inputs automatically.
    shapely.prepare(geographic)
    verify_self_contained_raster(output)
    counts = {"original_valid_cells": 0, "included_cells": 0}
    from pyproj import Transformer

    with rasterio.open(source) as src, rasterio.open(output) as dst:
        transformer = Transformer.from_pipeline(
            (operation or source_operation(src))["pipeline"]
        )
        if (
            src.crs != dst.crs
            or src.transform != dst.transform
            or src.shape != dst.shape
            or src.dtypes != dst.dtypes
            or src.count != 1
            or dst.count != 1
            or dst.nodata is None
            or not np.isnan(dst.nodata)
            or src.scales != dst.scales
            or src.offsets != dst.offsets
            or src.units != dst.units
            or src.descriptions != dst.descriptions
        ):
            raise ValueError(
                "Masking changed native grid, band encoding or NaN NoData contract"
            )
        # Source TIFFs may use enormous native tiles or strips. Reuse the
        # existing bounded raster iterator, independent of their storage layout.
        total = ((src.height + 1023) // 1024) * ((src.width + 1023) // 1024)
        for index, window in enumerate(blocks(src.width, src.height), 1):
            z = src.read(1, window=window)
            result = dst.read(1, window=window)
            original_valid = _source_valid(src, window, z)
            expected = original_valid & _native_classification(
                src, geographic, window, transformer
            )
            actual = (dst.read_masks(1, window=window) > 0) & np.isfinite(result)
            if not np.array_equal(actual, expected) or not np.all(
                np.isnan(result[~expected])
            ):
                raise ValueError(
                    "Published validity differs from the requested clipping geometry"
                )
            if z[expected].tobytes() != result[expected].tobytes():
                raise ValueError("Masking changed retained elevation bits")
            counts["original_valid_cells"] += int(original_valid.sum())
            counts["included_cells"] += int(expected.sum())
            if progress is not None:
                progress(index, total)
    return counts


def write_masked_tiff(source, target, geographic, operation=None):
    """Portable file-stage TIFF producer; does not depend on Globato patches.

    Output is single-band float32/float64 with NaN as NoData, a native grid,
    and original scale/offset/unit. Masked data are physically NaN, not merely
    tagged invalid by a GDAL mask that an older raster reader could ignore.
    """
    if Path(target).exists() or Path(target).is_symlink():
        raise FileExistsError("Mask output must be a new file: " + str(target))
    from pyproj import Transformer

    with rasterio.open(source) as src:
        check_encoding(src)
        operation = operation or source_operation(src)
        transformer = Transformer.from_pipeline(operation["pipeline"])
        profile = {
            "driver": "GTiff",
            "width": src.width,
            "height": src.height,
            "count": 1,
            "dtype": src.dtypes[0],
            "crs": src.crs,
            "transform": src.transform,
            "nodata": np.nan,
            "compress": "deflate",
            "predictor": 3,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "BIGTIFF": "IF_SAFER",
        }
        shapely.prepare(geographic)
        with rasterio.open(target, "w", **profile) as dst:
            dst.scales = src.scales
            dst.offsets = src.offsets
            if src.units[0]:
                dst.set_band_unit(1, src.units[0])
            dst.update_tags(**src.tags())
            dst.update_tags(
                1,
                **{
                    k: v
                    for k, v in src.tags(1).items()
                    if not k.startswith("STATISTICS_")
                },
            )
            if src.descriptions[0]:
                dst.set_band_description(1, src.descriptions[0])
            for _, window in dst.block_windows(1):
                z = src.read(1, window=window)
                valid = _source_valid(src, window, z)
                keep = valid & _native_classification(
                    src, geographic, window, transformer
                )
                masked = z.copy()
                masked[~keep] = np.nan
                dst.write(masked, 1, window=window)
        # Read back through GDAL and verify the actual published semantics.
    return validate_product(source, target, geographic, operation)


def publish_masked_source(
    source,
    keep,
    output_dir,
    *,
    provenance,
    mask_manifest,
    preparation_id,
    source_files=None,
):
    """Write or reuse one verified product; no module or hook objects enter here.

    The adapter supplies checked geometry and JSON provenance. This owner seals
    the coordinate operation, verifies original bytes, and publishes TIFF before
    receipt under one lock. An interrupted TIFF-only write can be regenerated.
    """
    source = Path(source).expanduser().resolve(strict=True)
    root = Path(output_dir).expanduser().resolve()
    files = raster_files(source) if source_files is None else source_files
    with rasterio.open(source) as native:
        operation = source_operation(native)
    processing = {
        **provenance,
        "coordinate_operation": operation,
        "source_sha256": next(
            item["sha256"] for item in files if item["name"] == source.name
        ),
        "source_files": files,
        "masking_sha256": digest(Path(__file__)),
        "keep_wkb": keep.normalize().wkb_hex,
        "writer": "native_nan_tiff_v1",
    }
    processing_id = identity(processing)
    root.mkdir(parents=True, exist_ok=True)
    target = root / (processing_id + ".tif")
    receipt_path = root / (processing_id + ".json")
    expected = {
        "status": "VALIDATED",
        "processing_id": processing_id,
        "mask_manifest": str(mask_manifest),
        "processing_inputs": processing,
        "source_identity": provenance["source_identity"],
        "source_weight": provenance["source_weight"],
        "module_weight": provenance["module_weight"],
        "original": str(source),
        "masked": str(target),
        "preparation_identity": preparation_id,
        "application_geometry_selection": provenance["application_geometry_selection"],
    }
    with FileLock(str(target) + ".lock"):
        # Never follow or silently unlink a redirected cache product. Both
        # published paths must be real files owned by this processing ID.
        if target.is_symlink() or receipt_path.is_symlink():
            raise ValueError("Derived raster cache contains a symlink; refusing reuse")
        if target.is_file() and not receipt_path.exists():
            target.unlink()
        if target.exists() or receipt_path.exists():
            if not (target.is_file() and receipt_path.is_file()):
                raise ValueError(
                    "Incomplete derived raster; use a fresh processing run"
                )
            receipt = json.loads(receipt_path.read_text())
            if any(receipt.get(key) != value for key, value in expected.items()):
                raise ValueError("Derived raster cache receipt provenance mismatch")
            if digest(target) != receipt.get("output_sha256"):
                raise ValueError("Derived raster cache identity/hash mismatch")
            counts = validate_product(source, target, keep, operation)
            if any(receipt.get(key) != value for key, value in counts.items()):
                raise ValueError(
                    "Derived raster cache receipt cell counts differ from validated raster"
                )
            outcome = "EMPTY_AFTER_MASK" if counts["included_cells"] == 0 else "MASKED"
            if receipt.get("outcome") != outcome:
                raise ValueError(
                    "Derived raster cache receipt outcome differs from validated raster"
                )
            if raster_files(source) != files:
                raise ValueError(
                    "Original source changed during derived raster cache validation"
                )
        else:
            with tempfile.TemporaryDirectory(
                prefix="_pending_clip_", dir=root
            ) as pending:
                pending = Path(pending)
                stats = write_masked_tiff(
                    source, pending / "masked.tif", keep, operation
                )
                if raster_files(source) != files:
                    raise ValueError("Original source changed during clipping")
                receipt = {
                    **expected,
                    "output_sha256": digest(pending / "masked.tif"),
                    "outcome": "EMPTY_AFTER_MASK"
                    if stats["included_cells"] == 0
                    else "MASKED",
                    **stats,
                }
                json_write(pending / "receipt.json", receipt)
                os.replace(pending / "masked.tif", target)
                os.replace(pending / "receipt.json", receipt_path)
    return receipt
