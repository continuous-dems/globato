"""Pure Phase 1 source-group selection, with no interpolation policy.

Existing unscoped coast_refine barrier/point_raster_mask consumers accept one
binding. A multi-group recipe MUST explicitly select which group, if any,
exports that legacy unscoped binding. All consumers still name an authority.
"""

from __future__ import annotations
import hashlib


def group_token(group: str) -> str:
    if not isinstance(group, str) or not group.strip():
        raise ValueError("Source group must be a nonempty string")
    # Full SHA256: source labels are never interpolated into filesystem paths.
    return hashlib.sha256(group.encode("utf-8")).hexdigest()


def select_barrier_group(groups, *, binding_group=None, require_barrier=False):
    """Return the one group allowed to use unscoped legacy barrier exports.

    groups: group -> list of request dictionaries. Other groups may still be
    prepared and masked independently, but are never implicitly unioned.
    """
    eligible = [
        group
        for group, requests in groups.items()
        if requests and requests[0].get("barrier") == "coast_refine"
    ]
    if binding_group is not None:
        if binding_group not in groups:
            raise ValueError(
                "Unknown explicit coast_refine binding_group: %r" % binding_group
            )
        if binding_group not in eligible:
            raise ValueError(
                "Explicit binding_group has no coast_refine barrier: %r" % binding_group
            )
        return binding_group
    if len(eligible) > 1:
        raise ValueError(
            "Multiple Coast Mask authority groups require an explicit "
            "coast_refine_finalize binding_group; never select or union implicitly: %s"
            % ", ".join(sorted(eligible))
        )
    selected = eligible[0] if eligible else None
    if require_barrier and selected is None:
        raise ValueError(
            "Exactly one explicit Coast Refine barrier authority group is required"
        )
    return selected


def validate_consumer_groups(groups, entries):
    """Preflight explicit consumer target before modifying/publishing anything."""
    for entry in entries:
        if "coast_refine_consumer" not in entry:
            continue
        consumer = entry["coast_refine_consumer"]
        if not isinstance(consumer, dict):
            raise ValueError(
                "Coast Mask consumer must explicitly reference one registered authority group"
            )
        group = consumer.get("authority") or consumer.get("group")
        if not isinstance(group, str) or not group.strip() or group not in groups:
            raise ValueError(
                "Coast Mask consumer must explicitly reference one "
                "registered authority group: %r" % group
            )


def publication_record(group, artifact, *, binding_group=None):
    connected = artifact.record.get("connected_water_selection", {})
    return dict(
        schema="coast_mask_group_publication.v1",
        status="completed",
        group=group,
        group_token=group_token(group),
        manifest=str(artifact.path),
        manifest_sha256=artifact.manifest_hash,
        topology=str(artifact.products["refined_topology"]),
        refined_water=str(artifact.root / "refined_water.gpkg"),
        connected_water=(
            str(artifact.root / "connected_water.gpkg")
            if connected.get("status") == "selected"
            else None
        ),
        connected_receipt=str(artifact.root / "connected_water.json"),
        connection_status=connected.get("status", "unresolved"),
        connection_reason=connected.get("reason"),
        legacy_binding=(group == binding_group),
    )
