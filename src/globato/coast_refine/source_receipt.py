"""Publish original entry -> accepted source state lineage, without physics.

This is a read-only bridge for downstream frameworks. Missing survey, datum or
quality stays unknown; it never invents a per-pixel source from a fused DEM.
"""

from pathlib import Path
import json
from collections.abc import Mapping
from ._io import digest, json_write, serializable, raster_files


def publish_accepted_source(module, entry, state, masking_receipt, artifact):
    request = (
        entry.get("coast_refine_request") or entry.get("coast_refine_consumer") or {}
    )
    binding = entry.get("_coast_mask_source_grid_binding") or {}
    snapshot = request.get("source") or {}
    if snapshot:
        if float(getattr(module, "weight", 1)) != snapshot["module_weight"]:
            raise ValueError("Module weight changed after Coast Mask registration")
        original = Path(snapshot["source"])
        inventory = (
            [{"name": original.name, "sha256": digest(original)}]
            if request.get("point_source")
            else raster_files(original)
        )
        if inventory != snapshot["files"]:
            raise ValueError(
                "Original source changed before accepted-state publication"
            )
    registered = request.get("registered_source_metadata") or {}
    # Registered class/survey metadata is more specific than runtime metadata,
    # which an ordinary reader may replace. Explicit source ID is independent
    # of module and contribution tokens.
    live = entry.get("metadata")
    metadata = {**(dict(live) if isinstance(live, Mapping) else {}), **registered}
    source_entry = snapshot.get("entry") or {}
    source_id = source_entry.get("source_id") or entry.get("source_id")
    explicit_id = source_id is not None
    if source_id is None:
        source_id = source_entry.get("url") or entry.get("url")
    survey_id = (
        source_entry.get("survey_id")
        or metadata.get("survey_id")
        or entry.get("survey_id")
    )
    record = {
        "schema": "coast_mask.accepted_source.v1",
        "status": "completed",
        "module": module.name,
        "source_id": source_id,
        "survey_id": survey_id,
        "source_identifier_kind": "provider_id"
        if explicit_id
        else "canonical_source_url"
        if source_id
        else "unverified",
        "contribution_id": serializable(
            entry.get("contribution_id")
            or {
                "native_dataset": entry.get("_sidestack_processing_identity", {}).get(
                    "native_dataset_id"
                )
            }
        ),
        "native_source_identity": entry.get("_sidestack_processing_identity"),
        "source_snapshot": snapshot or None,
        "original_grid_binding": binding,
        "registered_source_metadata": registered,
        "class_branch_claims": request.get("class_branch_claims", []),
        "processing_identity": entry.get("_sidestack_processing_identity"),
        "module_weight": getattr(module, "weight", None),
        "entry_weight": entry.get("weight"),
        "accepted_state": str(Path(state).resolve()),
        "accepted_state_sha256": digest(state),
        "mask_manifest": str(artifact.path),
        "mask_manifest_sha256": artifact.manifest_hash,
        "group": request.get("group") or request.get("authority"),
        "remove": request.get("remove"),
        "counts": masking_receipt["counts"],
        "selection": "unmasked_evidence"
        if request.get("global_refinement") and request.get("remove") is None
        else "source_retention_native_centres",
        "accepted_grid_crs": binding.get("grid", {}).get("crs"),
        "coordinate_operation": serializable(entry.get("stream_transform_identity")),
        "validation_excluded": metadata.get("validation_excluded"),
        "guidance_excluded": metadata.get("guidance_excluded"),
        "lineage_status": "original_asset_bound"
        if snapshot and source_id
        else "source_identity_unverified",
        "survey_identity_verified": False,
        "independent_validation": False,
        "physical_guidance_generated": False,
    }
    path = Path(state).with_suffix(".source.json")
    json_write(path, serializable(record))
    entry.setdefault("artifacts", {})["coast_mask_source_receipt"] = str(path)
    return record


def verify_accepted_source(path):
    record = json.loads(Path(path).read_text())
    if (
        record.get("schema") != "coast_mask.accepted_source.v1"
        or record.get("status") != "completed"
    ):
        raise ValueError("Unsupported/incomplete accepted-source receipt")
    for key, checksum in [
        ("accepted_state", "accepted_state_sha256"),
        ("mask_manifest", "mask_manifest_sha256"),
    ]:
        if digest(record[key]) != record[checksum]:
            raise ValueError("Accepted-source receipt asset changed: " + key)
    return record
