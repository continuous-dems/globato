"""Opt-in neutral, current-collection raster catalog; no viewer dependency."""

from pathlib import Path
import os
import tempfile
import numpy as np
import rasterio
from globato.hooks.transforms.point_pixels import FUSION_BANDS, finalize_fusion_state
from ._io import digest, identity, json_write


def _publish_z(source, destination):
    fd, pending = tempfile.mkstemp(
        prefix="." + destination.stem + "_", suffix=".tif", dir=destination.parent
    )
    os.close(fd)
    try:
        _write_z(source, Path(pending))
        os.replace(pending, destination)
    finally:
        Path(pending).unlink(missing_ok=True)


def _write_z(source, destination):
    with rasterio.open(source) as src:
        profile = src.profile.copy()
        profile.update(
            count=1, dtype="float64", nodata=np.nan, compress="deflate", predictor=3
        )
        with rasterio.open(destination, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                state = dict(zip(FUSION_BANDS, src.read(window=window)))
                result = finalize_fusion_state(state)
                dst.write(np.where(result.valid, result.z, np.nan), 1, window=window)
            dst.set_band_description(1, "accepted source cell mean; no interpolation")


def publish_grid_products(
    group, entries, *, res, crs, strategy, weight_threshold, artifact
):
    """Publish one matched participant comparison, using native stack reduction.

    The source set and reduction contract are identical before and after masking.
    Ordinary, unmasked modules are context in the accepted collection, not members
    of this comparison. Entry dictionaries are copied: display reduction must never
    replace the production side-stack or reapply source weights.
    """
    from globato.hooks.tools.stack_combine import StackCombine
    import shutil

    participants = []
    root = None
    for module, entry in entries:
        request = (
            entry.get("coast_refine_request")
            or entry.get("coast_refine_consumer")
            or {}
        )
        if (request.get("group") or request.get("authority")) != group:
            continue
        binding = entry.get("_coast_mask_source_grid_binding")
        accepted = (entry.get("artifacts") or {}).get("side-stack")
        if not binding or not accepted:
            raise ValueError(
                "Completed source states required for Coast Mask visualization"
            )
        if root is None:
            root = Path(request["output_dir"]) / "products" / identity({"group": group})
        participants.append((module, entry, binding, accepted))
    if root is None:
        raise ValueError(
            "No completed Coast Mask entries for requested visualization group"
        )
    root.mkdir(parents=True, exist_ok=True)
    catalog = root / "products.json"
    json_write(
        catalog,
        {
            "schema_version": 1,
            "title": "Coast Mask " + group,
            "running": True,
            "products": [],
        },
    )
    products = []
    for phase, label in (
        ("before", "Before masking"),
        ("after", "Accepted measurements"),
    ):
        copied = []
        for module, entry, binding, accepted in participants:
            view = dict(entry)
            view["artifacts"] = dict(entry.get("artifacts") or {})
            view["artifacts"]["side-stack"] = (
                binding["source_path"] if phase == "before" else accepted
            )
            copied.append((module, view))
        # Native StackCombine owns all additive/priority reduction. Temporary native
        # outputs are visualization inputs, never production artifacts or a cache.
        with tempfile.TemporaryDirectory(prefix=".display-", dir=root) as temporary:
            stack = Path(temporary) / "combined.tif"
            reducer = StackCombine(
                res=res,
                crs=crs,
                strategy=strategy,
                weight_threshold=weight_threshold,
                output=str(stack),
                missing="error",
            )
            try:
                reducer.run(copied)
                state = copied[0][1]["artifacts"]["fusion-state"]
                target = root / (phase + ".tif")
                _publish_z(state, target)
            finally:
                reducer.teardown()
        products.append(
            {
                "key": phase,
                "path": target.name,
                "label": label,
                "area": "default",
                "stage": phase,
                "type": "raster",
                "group": "participant_elevation",
                "style": "elevation",
                "band": 1,
                "metadata": {
                    "interpolated": False,
                    "role": "before_measured"
                    if phase == "before"
                    else "after_measured",
                    "participant_count": len(participants),
                    "scope": "Coast Mask group participants only",
                    "reduction_owner": "globato.hooks.tools.stack_combine.StackCombine",
                },
            }
        )
    # Copy closed, verified geometry products for a portable catalog. The signed
    # preparation remains authoritative; these are byte-identical display copies.
    # Three distinct scientific geometries, but only the masking boundary is
    # displayed initially. Other layers are optional diagnostics, not additive
    # polygons to interpret as one composite boundary.
    vector_views = (
        (
            "original_water",
            "Original water boundary",
            {"visible": False, "color": "#2563eb", "width": 1.8, "dash": [7, 4]},
        ),
        (
            "retention_water",
            "Source-exclusion boundary",
            {"visible": True, "color": "#dd7314", "width": 2.2, "dash": []},
        ),
        (
            "refined_topology",
            "Interpolation topology (no extension)",
            {"visible": False, "color": "#7c3aed", "width": 1.8, "dash": [2, 4]},
        ),
    )
    for key, label, display in vector_views:
        target = root / (key + ".gpkg")
        shutil.copyfile(artifact.products[key], target)
        products.append(
            {
                "key": key,
                "path": target.name,
                "label": label,
                "area": "default",
                "stage": "geometry",
                "type": "vector",
                "display": display,
                "metadata": {
                    "interpolated": False,
                    "role": key,
                    "layer": artifact.layers[key],
                    "sha256": digest(target),
                },
            }
        )
    json_write(
        catalog,
        {
            "schema_version": 1,
            "title": "Coast Mask " + group,
            "running": False,
            "status": "Coast Mask participant comparison complete",
            "producer": {"hook": "coast_mask", "group": str(group)},
            "products": products,
            "defaults": {"default": ["before", "after"]},
        },
    )
    return catalog
