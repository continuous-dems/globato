"""Grid-native per-entry FusionState masking and comparison (v61.3).

This acts on unfilled, *unfused per-entry* additive FusionState grids produced by
Globato's own PointPixels/MultiStack machinery. No point returns are invented.
The comparison path is opt-in and NEVER replaces a production side-stack.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import rasterio
from pyproj import Transformer
from .selection import centre_selection, horizontal_crs


def _schema(expected):
    if expected is not None:
        return tuple(expected)
    from globato.hooks.transforms.point_pixels import FUSION_BANDS

    return tuple(FUSION_BANDS)


def inspect_fusion_state(path, *, expected_bands=None, require_source_role=True):
    """Fail closed on finalized DEM, missing CRS and unrecognized FusionState."""
    bands = _schema(expected_bands)
    if "count" not in bands or len(bands) < 2:
        raise ValueError("Expected FusionState schema must contain 'count'")
    with rasterio.open(path) as src:
        tags = src.tags()
        if tags.get("GLOBATO_DATATYPE") != "FUSION_STATE":
            raise ValueError("Input must be an additive FUSION_STATE, never a DEM")
        if require_source_role and not (
            tags.get("GLOBATO_SIDESTACK_IDENTITY")
            or tags.get("COAST_MASK_GRID_ROLE") == "source_unfilled"
        ):
            raise ValueError("Fused/global FusionState is not a per-entry source grid")
        if (
            src.count != len(bands)
            or tuple(src.descriptions) != bands
            or src.crs is None
            or src.dtypes[0] != "float64"
        ):
            raise ValueError("Per-entry FusionState band schema, CRS or dtype mismatch")
        if not src.transform.is_rectilinear or abs(src.transform.determinant) < 1e-30:
            raise ValueError("Invalid FusionState geotransform")
        for _, w in src.block_windows(1):
            data = src.read(window=w)
            count = data[bands.index("count")]
            if not np.isfinite(count).all() or (count < 0).any():
                raise ValueError("FusionState has invalid counts")
            if not np.isfinite(data).all():
                raise ValueError("FusionState contains nonfinite sufficient statistics")
            for key in ("weight_sum", "z2_weighted_sum", "weighted_uncertainty_sq"):
                if key in bands and (data[bands.index(key)] < 0).any():
                    raise ValueError("FusionState has negative " + key)
            if np.any(data[:, count == 0] != 0):
                raise ValueError("Empty FusionState cells contain nonzero statistics")
        return {
            "shape": [src.height, src.width],
            "crs": src.crs.to_string(),
            "transform": list(src.transform)[:6],
            "bands": list(bands),
            "datatype": tags["GLOBATO_DATATYPE"],
        }


def mask_fusion_state(
    source,
    output,
    keep_geometry,
    geometry_crs,
    *,
    expected_bands=None,
    source_id="",
    group="",
    validated_binding=None,
):
    """Keep grid-cell centres inside keep_geometry; zero ALL additive bands outside.

    `keep_geometry` is the authoritative vector retention polygon, not a
    source extent and not a rasterized *final* stack. Existing source stays
    untouched; derived mask cannot validate as the original side-stack cache.
    """
    if validated_binding is None:
        meta = inspect_fusion_state(source, expected_bands=expected_bands)
    else:
        # Verified by the collection-stage source binding, not by arbitrary
        # user-supplied raster metadata. Recheck file identity before masking.
        source_path = Path(source).resolve(strict=True)
        if validated_binding.get("role") != "source_unfilled" or (
            Path(validated_binding.get("source_path", "")).resolve() != source_path
        ):
            raise ValueError("Validated source-grid binding does not match input")
        from .source_grid_binding import _sha256

        if _sha256(source_path) != validated_binding.get("source_sha256"):
            raise ValueError("Source grid changed after its provenance validation")
        meta = inspect_fusion_state(
            source, expected_bands=expected_bands, require_source_role=False
        )
        if meta != validated_binding.get("grid"):
            raise ValueError(
                "Source grid alignment/schema differs from validated binding"
            )
    if keep_geometry is None or not keep_geometry.is_valid:
        raise ValueError("Missing/invalid keep polygon")
    if not keep_geometry.is_empty and keep_geometry.geom_type not in (
        "Polygon",
        "MultiPolygon",
    ):
        raise ValueError("Keep geometry must be polygonal")
    source = Path(source).resolve(strict=True)
    output = Path(output).resolve()
    if source == output:
        raise ValueError("Refusing to overwrite original FusionState")
    output.parent.mkdir(parents=True, exist_ok=True)
    bands = meta["bands"]
    fd, tmp = tempfile.mkstemp(prefix="masked_state_", suffix=".tif", dir=output.parent)
    os.close(fd)
    tmp = Path(tmp)
    counts = {
        "before_cells": 0,
        "removed_cells": 0,
        "after_cells": 0,
        "before_observations": 0.0,
        "removed_observations": 0.0,
        "after_observations": 0.0,
    }
    try:
        with rasterio.open(source) as src:
            transformer = Transformer.from_crs(
                horizontal_crs(src.crs), horizontal_crs(geometry_crs), always_xy=True
            )
            profile = src.profile.copy()
            with rasterio.open(tmp, "w", **profile) as dst:
                for i, name in enumerate(bands, 1):
                    dst.set_band_description(i, name)
                # A derived mask is NOT the original unmasked side-stack cache.
                tags = {
                    k: v
                    for k, v in src.tags().items()
                    if k not in ("GLOBATO_SIDESTACK_HASH", "GLOBATO_SIDESTACK_IDENTITY")
                }
                dst.update_tags(**tags)
                dst.update_tags(
                    COAST_MASK_GRID_ROLE="masked_fusion_state",
                    COAST_MASK_SOURCE_ID=str(source_id),
                    COAST_MASK_GROUP=str(group),
                )
                ci = bands.index("count")
                for _, win in src.block_windows(1):
                    data = src.read(window=win)
                    count = data[ci]
                    valid = count > 0
                    keep = centre_selection(
                        src,
                        keep_geometry,
                        win,
                        geometry_crs=geometry_crs,
                        transformer=transformer,
                    )
                    removed = valid & ~keep
                    counts["before_cells"] += int(valid.sum())
                    counts["removed_cells"] += int(removed.sum())
                    counts["after_cells"] += int((valid & keep).sum())
                    counts["before_observations"] += float(count[valid].sum())
                    counts["removed_observations"] += float(count[removed].sum())
                    counts["after_observations"] += float(count[valid & keep].sum())
                    data[:, ~keep] = 0.0
                    dst.write(data, window=win)
        inspect_fusion_state(tmp, expected_bands=bands, require_source_role=False)
        os.replace(tmp, output)
    finally:
        tmp.unlink(missing_ok=True)
    return {
        "schema": "coast_mask.grid_fusion.v61.3",
        "source_state": str(source),
        "masked_state": str(output),
        "source_id": str(source_id),
        "group": str(group),
        "counts": counts,
        "selection": "cell_center",
        "state_schema": meta,
        "source_binding": {
            "identity_mode": validated_binding.get("identity_mode"),
            "source_sha256": validated_binding.get("source_sha256"),
            "sidestack_hash": validated_binding.get("sidestack_hash"),
        }
        if validated_binding
        else None,
        "outcome": "EMPTY_AFTER_MASK" if counts["after_cells"] == 0 else "MASKED",
        "note": "Zeroed all additive bands outside strict source-retention cell centres; no original point replay",
    }


def compare_fusion_states(candidate, reference, *, expected_bands=None, tolerance=1e-8):
    """Compare cell support, elevations' sufficient stats and every state band.

    Does not equate matching cell counts with equal elevation statistics.
    """
    a = inspect_fusion_state(
        candidate, expected_bands=expected_bands, require_source_role=False
    )
    b = inspect_fusion_state(
        reference, expected_bands=expected_bands, require_source_role=False
    )
    if any(a[k] != b[k] for k in ("shape", "crs", "transform", "bands")):
        raise ValueError("Reference and candidate grids are not aligned")
    ci = a["bands"].index("count")
    result = {
        "schema": "coast_mask.grid_comparison.v61.3",
        "candidate": str(candidate),
        "reference": str(reference),
        "candidate_cells": 0,
        "reference_cells": 0,
        "support_only_candidate": 0,
        "support_only_reference": 0,
        "shared_cells": 0,
        "band_differences": {},
        "max_abs_difference": {},
        "tolerance": tolerance,
    }
    stats = {name: 0 for name in a["bands"]}
    maxima = {name: 0.0 for name in a["bands"]}
    with rasterio.open(candidate) as cand, rasterio.open(reference) as ref:
        for _, win in cand.block_windows(1):
            aa = cand.read(window=win)
            bb = ref.read(window=win)
            va = aa[ci] > 0
            vb = bb[ci] > 0
            shared = va & vb
            result["candidate_cells"] += int(va.sum())
            result["reference_cells"] += int(vb.sum())
            result["support_only_candidate"] += int((va & ~vb).sum())
            result["support_only_reference"] += int((vb & ~va).sum())
            result["shared_cells"] += int(shared.sum())
            for n, name in enumerate(a["bands"]):
                if shared.any():
                    delta = np.abs(aa[n][shared] - bb[n][shared])
                    diff = ~np.isfinite(delta) | (delta > tolerance)
                    stats[name] += int(diff.sum())
                    if np.isfinite(delta).any():
                        maxima[name] = max(
                            maxima[name], float(np.max(delta[np.isfinite(delta)]))
                        )
    result["band_differences"] = stats
    result["max_abs_difference"] = maxima
    result["exact_support"] = not (
        result["support_only_candidate"] or result["support_only_reference"]
    )
    result["matched_within_tolerance"] = result["exact_support"] and not any(
        stats.values()
    )
    return result


def write_comparison_report(path, report):
    """Atomic JSON report for downstream QA; never silently declare parity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
