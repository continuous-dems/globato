"""Coast Refine's small, science-aware artifact *declarations* only.

No visualization engine, gridding, filtering or raster export lives here.
Publication uses Raster Viz 0.13.10's neutral schema-v1 contract; rendering
remains a separate CLI operation, not a dependency of Coast Mask.
"""

from __future__ import annotations

import logging
import json
import os
from pathlib import Path

from ._io import identity, json_write

log = logging.getLogger(__name__)


def _relative_product_paths(products, catalog):
    """Portable references for the Raster Viz v13.10 catalog, no viewer import.

    Paths are interpreted as completed file locations, never rediscovered by
    filename. Scientific product paths and underlying files are not modified.
    """
    root = Path(catalog).resolve().parent
    result = []
    for source in products:
        item = dict(source)
        path = Path(item["path"])
        item["path"] = os.path.relpath(path.resolve(), root)
        if item.get("type") != "vector":
            item.setdefault("type", "raster")
            item.setdefault("style", "elevation")
            item.setdefault("band", 1)
        result.append(item)
    return result


def _publish_catalog(
    catalog,
    *,
    title,
    products,
    defaults=None,
    producer=None,
    running=False,
    status=None,
):
    """Atomic schema-v1 publication consumed directly by Raster Viz v13.10."""
    document = {
        "schema_version": 1,
        "title": title,
        "running": bool(running),
        "products": _relative_product_paths(products, catalog),
        "defaults": defaults or {},
    }
    if producer is not None:
        document["producer"] = producer
    if status is not None:
        document["status"] = status
    Path(catalog).parent.mkdir(parents=True, exist_ok=True)
    json_write(catalog, document)
    return Path(catalog)


def publish_source_group(group, selected, artifact, *, deferred):
    """Publish a tile/group-level overview without per-source TIFF proliferation.

    All raster paths are references to existing Coast Refine files. This is
    deliberately NOT advertised as a before/after full-tile stack: the
    authority's unmasked native evidence and the later complete MultiStack
    differ in source scope. Detailed per-source outputs are still available
    through their existing mask receipts when specifically requested.
    """
    first = selected[0][1]["coast_refine_request"]
    key = identity(
        {
            "group": group,
            "preparation": artifact.preparation_id,
            "viz_schema": "tile_overview_v53",
        }
    )[:20]
    root = Path(first["output_dir"]).resolve() / "viz_products" / key
    catalog = root / "products.json"
    area = f"Coast Refine / {group} / source overview"
    metadata = dict(
        scope="authority-group native evidence before masking; NOT all-source DEM",
        source_count=len(selected),
        preparation_id=artifact.preparation_id,
        role="original_authority_evidence",
    )
    records = []
    evidence = artifact.root / "unmasked_evidence.tif"
    if evidence.is_file():
        records.append(
            dict(
                key="original_authority_evidence",
                path=str(evidence),
                area=area,
                label="Original authority elevation evidence (unfilled)",
                stage="Before source refinement",
                layer="source_evidence",
                method="coast_refine",
                group="authority_elevation",
                metadata=metadata,
            )
        )
    topology = artifact.products.get("refined_topology")
    if topology and Path(topology).is_file():
        records.append(
            dict(
                key="sealed_coastline",
                path=str(topology),
                area=area,
                label="Sealed refined coastal topology",
                type="vector",
                display={"visible": True, "color": "#dd7314", "width": 2.2, "dash": []},
                stage="After source refinement",
                layer="coastal_topology",
                group="barrier",
                method="coast_refine",
                metadata=dict(
                    scope="refinement result",
                    role="prepared_coastline",
                    source_count=len(selected),
                ),
            )
        )
    # The source overview is complete even when a global DEM is still pending.
    # No hidden renderer or legacy Globato product publisher is invoked.
    _publish_catalog(
        catalog,
        title=f"Coast Refine / {group}",
        products=records,
        producer=dict(
            hook="coast_refine",
            group=group,
            preparation_id=artifact.preparation_id,
            source_records_default=False,
            source_count=len(selected),
        ),
        running=bool(deferred),
        status=(
            "Source overview complete; global interpolation pending"
            if deferred
            else "Source overview complete"
        ),
    )
    return catalog


def publish_global_result(entry):
    """Called after final DEM and deferred barrier audit are fully written."""
    path = entry.get("coast_refine_viz_catalog")
    if not path:
        return None
    artifacts = entry.get("artifacts") or {}
    audit = artifacts.get("coast_refine_barrier_audit")
    if not audit or not Path(audit).is_file():
        raise ValueError("Deferred barrier audit is not complete")
    catalog = Path(path)
    area = "Global interpolation / " + Path(entry["dst_fn"]).name
    metadata = dict(
        scope="final composite Globato DEM, NOT a source-refinement after",
        barrier_audit=str(audit),
        role="final_composite_dem",
    )
    products = [
        dict(
            key="global_dem_" + identity(str(Path(entry["dst_fn"]).resolve()))[:12],
            path=str(entry["dst_fn"]),
            area=area,
            label="Global interpolated DEM",
            group="final_dem",
            stage="Interpolation complete",
            layer="interpolated_dem",
            method="ms_binary_cudem",
            metadata=metadata,
        )
    ]
    # Reuse the already-produced, finalized, UNFILLED multi-source stack.
    # This is tile-level context, not an individual source raster and not a
    # second interpolation. Do not manufacture large before/after products.
    completed = json.loads(Path(audit).read_text())
    stack = completed.get("source_stack_path")
    if stack and Path(stack).is_file():
        products.append(
            dict(
                key="tile_unfilled_source_stack",
                path=str(stack),
                area=area,
                label="Combined accepted observations (unfilled)",
                group="final_dem",
                stage="Before interpolation",
                layer="observed_elevation",
                method="stack_combine",
                metadata={
                    **metadata,
                    "role": "combined_unfilled_observations",
                    "scope": "all accepted source observations; no DEM filling",
                },
            )
        )
    barrier = artifacts.get("coast_refine_barrier")
    if barrier and Path(barrier).is_file():
        products.append(
            dict(
                key="resolved_interpolation_barrier",
                path=str(barrier),
                area=area,
                label="Resolved interpolation barrier",
                type="vector",
                display={"visible": True, "color": "#dd7314", "width": 2.2, "dash": []},
                group="barrier",
                stage="Interpolation complete",
                layer="barrier",
                metadata={**metadata, "role": "resolved_barrier"},
            )
        )
    # Overlay the actually added ocean-facing area, not just the complete
    # resolved topology. These are ordinary source vector products read by
    # standalone Raster Viz, never additional interpolation input.
    extension_file = completed.get("extension_receipt_path")
    if extension_file and Path(extension_file).is_file():
        from ._io import digest

        if digest(extension_file) != completed.get("extension_receipt_sha256"):
            raise ValueError(
                "Interpolation extension receipt does not match completed audit"
            )
        info = json.loads(Path(extension_file).read_text())
        root = Path(extension_file).parent
        for filename, key, label, role in (
            (
                "extension_area.gpkg",
                "added_interpolation_allowance",
                "Additional interpolation allowance only",
                "interpolation_extension_area",
            ),
            (
                "interpolation_coastline.gpkg",
                "inferred_interpolation_limit",
                "Smoothed offshore interpolation limit",
                "inferred_interpolation_limit",
            ),
        ):
            candidate = root / filename
            if not candidate.is_file() or digest(candidate) != info["products"].get(
                filename
            ):
                raise ValueError(
                    "Verified extension vector missing or changed: " + filename
                )
            products.append(
                dict(
                    key=key,
                    path=str(candidate),
                    area=area,
                    label=label,
                    type="vector",
                    display={
                        "visible": False,
                        "color": "#059669",
                        "width": 1.9,
                        "dash": [6, 3],
                    },
                    group="barrier",
                    stage="Interpolation complete",
                    layer="inferred_boundary",
                    method="coast_refine",
                    metadata={
                        **metadata,
                        "role": role,
                        "requested_extension_m": info["inputs"]["policy"][
                            "barrier_extension_m"
                        ],
                        "global_fallback": bool(
                            (info.get("topology_safety") or {}).get("global_fallback")
                        ),
                        "extension_status": info.get("extension_status"),
                        "rejection_reason": info.get("rejection_reason"),
                    },
                )
            )
    manifest = completed.get("manifest_path")
    if manifest and Path(manifest).is_file():
        from .artifact import MaskArtifact

        authority = MaskArtifact(manifest)
        coastline = authority.products.get("refined_coastline")
        if coastline and Path(coastline).is_file():
            products.append(
                dict(
                    key="sealed_refined_coastline",
                    path=str(coastline),
                    area=area,
                    label="Original sealed Coast Refine coastline",
                    type="vector",
                    display={
                        "visible": False,
                        "color": "#2563eb",
                        "width": 1.8,
                        "dash": [7, 4],
                    },
                    group="barrier",
                    stage="Preparation",
                    layer="prepared_coastline",
                    method="coast_refine",
                    metadata={**metadata, "role": "sealed_refined_coastline"},
                )
            )
    defaults = (
        {area: ["tile_unfilled_source_stack", products[0]["key"]]}
        if stack and Path(stack).is_file()
        else {}
    )
    # Preserve the earlier source overview in its own area. The global DEM
    # and accepted observations form a separate comparison, never a misleading
    # masking before/after pair. Catalog paths remain relative to the catalog.
    existing = json.loads(catalog.read_text())
    previous = existing.get("products", [])
    incoming = _relative_product_paths(products, catalog)
    identities = {(p.get("area", "default"), p["key"]) for p in incoming}
    retained = [
        p for p in previous if (p.get("area", "default"), p["key"]) not in identities
    ]
    existing["products"] = retained + incoming
    existing.setdefault("defaults", {}).update(defaults)
    existing["running"] = False
    existing["status"] = "Global interpolation and source overview complete"
    json_write(catalog, existing)
    return catalog
