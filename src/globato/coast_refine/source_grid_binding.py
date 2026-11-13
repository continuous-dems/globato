"""Bind an upstream per-entry side-stack to its original collection entry.

Consumes native completion receipts, or existing canonical embedded bindings.
No cached unstamped raster is trusted. This never alters the source cache.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import rasterio
from pyproj import CRS
from .grid_native_fusion import inspect_fusion_state


def _sha256(path):
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for part in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            sha.update(part)
    return sha.hexdigest()


# Only the scientific, registered source policy belongs in the end-to-end
# request seal.  Runtime entry weight/uncertainty can be populated by stream
# readers after registration; validate those separately against the declared
# module/source values.  The upstream namespace also seals the full hook and
# class configuration when this source branch is registered.
_STABLE_POLICY = (
    "source",
    "point_source",
    "group",
    "authority",
    "remove",
    "mask",
    "polygon_class",
    "barrier",
    "grid_native",
    "source_hook",
    "outlet_lonlat",
    "river_guidance",
    "barrier_extension_m",
    "barrier_extension_min_z",
    "barrier_extension_method",
    "barrier_slope_window_m",
)


def _source_request_signature(entry):
    request = (
        entry.get("coast_refine_request") or entry.get("coast_refine_consumer") or {}
    )
    keys = _STABLE_POLICY
    if request.get("global_refinement"):
        keys = (
            "source",
            "point_source",
            "group",
            "authority",
            "remove",
            "grid_native",
            "source_hook",
            "global_refinement",
            "evidence",
        )
    return {
        "source_file": str(entry.get("dst_fn") or ""),
        "policy": {key: request[key] for key in keys if key in request},
    }


def _source_request_digest(entry):
    """Seal source file and immutable Coast Mask policy, not mutable entry data."""
    value = _source_request_signature(entry)
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _numeric_equivalent(left, right):
    if left is None or right is None:
        return left is None and right is None
    try:
        return float(left) == float(right)
    except (ValueError, TypeError):
        return left == right


def _entry_claim_valid(initial, declared, current):
    """Permit only ordinary filling of a missing entry claim from its module.

    Explicit claims cannot change after side-stack registration. If the entry
    had no claim, the only valid resolved value is the declared module claim.
    This preserves normal per-module weighting without accepting arbitrary
    post-registration weight or uncertainty changes.
    """
    if _numeric_equivalent(current, initial):
        return True
    return (
        initial is None
        and declared is not None
        and _numeric_equivalent(current, declared)
    )


def bind_source_grid(entry, *, expected_crs, expected_bands=None):
    """Validate a completed native source grid and its unchanged masking policy."""
    artifact = (entry.get("artifacts") or {}).get("side-stack")
    if not artifact or not Path(artifact).is_file():
        raise ValueError("Source-specific side-stack is missing or not materialized")
    path = Path(artifact).resolve(strict=True)
    request = entry.get("coast_refine_request") or entry.get("coast_refine_consumer")
    if not request:
        raise ValueError("Grid lacks a registered Coast Mask source/consumer")
    if entry.get("dst_fn") and Path(entry["dst_fn"]).resolve() == path:
        raise ValueError("Source grid may not be the original input file")
    registered_path = entry.get("_sidestack_path")
    if registered_path and Path(registered_path).resolve() != path:
        raise ValueError("Source grid differs from registered side-stack cache path")
    file_sha256 = _sha256(path)
    receipt = entry.get("_sidestack_receipt")
    fresh = entry.get("_sidestack_registered") is True
    canonical = bool(
        entry.get("_sidestack_hash")
        and entry.get("_sidestack_identity")
        and registered_path
    )
    if fresh:
        if not isinstance(receipt, dict):
            raise ValueError("Native side-stack has no completed sink receipt")
        context = receipt.get("context") or {}
        namespace = receipt.get("namespace")
        checks = {
            "schema": receipt.get("schema") == "globato.side_stack.receipt.v1",
            "fresh_namespace": receipt.get("created_in_fresh_namespace") is True,
            "sink": receipt.get("sink") == "globato.side_stack",
            "namespace": bool(namespace)
            and namespace == entry.get("_sidestack_namespace")
            and path.parent.name == namespace,
            "source_file": str(receipt.get("source_file"))
            == str(entry.get("dst_fn") or ""),
            "source_request": context.get("request_sha256")
            == _source_request_digest(entry),
            "artifact_path": Path(str(receipt.get("path", ""))).resolve() == path,
            "file_sha256": receipt.get("sha256") == file_sha256,
            "processing_identity": receipt.get("processing_identity")
            == entry.get("_sidestack_processing_identity"),
            "not_cache_hit": entry.get("_sidestack_hit") is not True,
        }
        issues = [name for name, passed in checks.items() if not passed]
        claims = receipt.get("registered_claims")
        if not isinstance(claims, dict):
            issues.append("registered_claims")
        else:
            for field in ("weight", "uncertainty"):
                if not _entry_claim_valid(
                    claims.get("entry_" + field),
                    claims.get("module_" + field),
                    entry.get(field),
                ):
                    issues.append("entry_" + field)
        if issues:
            raise ValueError(
                "Invalid or stale native source-grid sink receipt (failed: "
                + ", ".join(issues)
                + ")"
            )
    elif not canonical:
        raise ValueError(
            "Side-stack not registered with canonical source-entry binding"
        )
    elif (entry.get("stream_type"), entry.get("data_type")) != (
        "raster-stream",
        "fusion-state",
    ):
        raise ValueError("Canonical side-stack entry must declare FusionState metadata")
    meta = inspect_fusion_state(
        path, expected_bands=expected_bands, require_source_role=False
    )
    if CRS.from_user_input(meta["crs"]) != CRS.from_user_input(expected_crs):
        raise ValueError("Source grid CRS differs from authority evidence CRS")
    with rasterio.open(path) as ds:
        tags = ds.tags()
    role = tags.get("COAST_MASK_GRID_ROLE")
    if role and role != "source_unfilled":
        raise ValueError("Source side-stack is already derived/masked: " + role)
    file_hash = tags.get("GLOBATO_SIDESTACK_HASH")
    file_identity = tags.get("GLOBATO_SIDESTACK_IDENTITY")
    entry_hash = entry.get("_sidestack_hash")
    entry_identity = entry.get("_sidestack_identity")
    if entry_identity and not isinstance(entry_identity, str):
        entry_identity = json.dumps(
            entry_identity, sort_keys=True, separators=(",", ":")
        )
    if not file_hash or file_hash != entry_hash:
        raise ValueError("Source grid/entry identity hash differs or is missing")
    if file_identity and file_identity != entry_identity:
        raise ValueError("Source grid/entry identity manifest differs")
    if (
        entry_identity
        and hashlib.sha256(entry_identity.encode()).hexdigest() != file_hash
    ):
        raise ValueError("Source entry identity does not match side-stack hash")
    return {
        "schema": "coast_mask.source_grid_binding.v13",
        "role": "source_unfilled",
        "entry_stream_type": entry.get("stream_type"),
        "entry_data_type": entry.get("data_type"),
        "binding_evidence": "file_schema_and_native_sink_receipt"
        if fresh
        else "canonical_embedded_identity",
        "source_path": str(path),
        "source_sha256": file_sha256,
        "identity_mode": "native_fresh_receipt" if fresh else "embedded",
        "fresh_sink_receipt": fresh,
        "source_file": str(entry.get("dst_fn") or ""),
        "source_request": request.get("source") or {},
        "sidestack_hash": file_hash,
        "sidestack_token_length": len(file_hash),
        "sidestack_identity": file_identity or entry_identity,
        "contribution_id": str(entry.get("contribution_id") or ""),
        "grid": meta,
    }
