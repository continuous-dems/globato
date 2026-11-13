"""Optional post-stack zero estimate through the existing extension component."""

from pathlib import Path
import json
import shutil

import numpy as np
from fetchez.hooks import FetchHook

from ._io import digest, identity, json_write
from .extension import build, settings, validate
from .global_refinement import resolve_group_publication


def publish_zero_estimate(entry, group, stack, *, max_distance_m=40, min_weight):
    """Publish the existing smooth estimator from a complete native stack.

    Used by the normal interpolator and the optional vector-only collection hook.
    Scientific fitting and topology repair remain owned by extension/shore_profile.
    """
    policy = settings(max_distance_m, 0.25, "slope_smooth", 30)
    if float(max_distance_m) <= 0:
        raise ValueError("zero_max_distance_m must be positive")
    min_weight = float(min_weight)
    if not np.isfinite(min_weight) or min_weight <= 0:
        raise ValueError("Zero-estimate support weight must be finite and positive")
    index, record, publication, artifact = resolve_group_publication(entry, group)
    artifact.barrier_for(stack)  # Validate the full working footprint first.
    receipt = build(
        artifact,
        policy["barrier_extension_m"],
        minimum_z=policy["barrier_extension_min_z"],
        method=policy["barrier_extension_method"],
        slope_window_m=policy["barrier_slope_window_m"],
        support_stack=stack,
        support_min_weight=min_weight,
        use_stack_for_land=True,
    )
    validate(artifact, receipt)
    estimate = json.loads(receipt.read_text())
    # Export closed, sealed products into the existing run-owned product
    # directory. No new cache: the estimator's existing cache owns reuse.
    catalog_root = index.parent / "products" / identity({"group": group})
    target = catalog_root / "estimated_zero" / estimate["identity"]
    target.mkdir(parents=True, exist_ok=True)
    for name, checksum in estimate["products"].items():
        path = target / name
        shutil.copyfile(receipt.parent / name, path)
        if digest(path) != checksum:
            raise ValueError("Zero-estimate export changed during copying")
    published_receipt = target / "extension.json"
    shutil.copyfile(receipt, published_receipt)
    topology = validate(artifact, published_receipt)
    coastline = target / "interpolation_coastline.gpkg"
    publication["estimated_zero"] = dict(
        receipt=str(published_receipt),
        receipt_sha256=digest(published_receipt),
        topology=str(topology),
        topology_sha256=digest(topology),
        coastline=str(coastline),
        coastline_sha256=digest(coastline),
        support_stack=stack,
        support_stack_sha256=estimate["inputs"]["support_stack_sha256"],
        status=estimate["extension_status"],
        interpretation="Estimated native-stack zero with topology safeguards and exact base fallback; not a surveyed or tidal coastline",
    )
    json_write(index, record)
    catalog = catalog_root / "products.json"
    if catalog.exists():
        data = json.loads(catalog.read_text())
        if data.get("running"):
            raise ValueError("Coast Mask product catalog is not complete")
        relative = str(coastline.relative_to(catalog_root))
        product = dict(
            key="estimated_zero",
            path=relative,
            type="vector",
            area="default",
            label="Estimated 0 m boundary (" + policy["barrier_extension_method"] + ")",
            stage="boundary_estimate",
            display=dict(visible=False, color="#059669", width=2.2, dash=[8, 3]),
            metadata=dict(
                role="estimated_zero",
                layer="mask",
                estimated=True,
                source_masks_changed=False,
                scope="complete accepted unfilled MultiStack",
                status=estimate["extension_status"],
                maximum_distance_m=policy["barrier_extension_m"],
                min_weight=min_weight,
                sha256=digest(coastline),
            ),
        )
        data["products"] = [
            p for p in data["products"] if p["key"] != "estimated_zero"
        ] + [product]
        json_write(catalog, data)
    return publication["estimated_zero"]


class CoastMaskBoundary(FetchHook):
    """Optional vector-only adapter for runs without ms_binary_cudem."""

    name = "coast_mask_boundary"
    meta_stage = "collection"

    def __init__(self, group, max_distance_m=40, min_weight=2, **kwargs):
        if set(kwargs) - {"stage"}:
            raise TypeError(
                "Use group, max_distance_m and min_weight for boundary estimation"
            )
        super().__init__(**kwargs)
        if not isinstance(group, str) or not group.strip():
            raise ValueError("A named Coast Mask group is required")
        self.group = group.strip()
        self.max_distance_m = settings(max_distance_m)["barrier_extension_m"]
        self.min_weight = float(min_weight)
        if (
            self.max_distance_m <= 0
            or not np.isfinite(self.min_weight)
            or self.min_weight <= 0
        ):
            raise ValueError("Boundary estimation needs positive distance and weight")

    def run(self, entries):
        if not entries:
            raise ValueError("coast_mask_boundary requires a completed collection")
        stacks = set()
        indexes = set()
        for _, entry in entries:
            artifacts = entry.get("artifacts") or {}
            stack = artifacts.get("multi_stack")
            index = artifacts.get("coast_mask_groups_index")
            if not stack or not index:
                raise ValueError(
                    "Run coast_mask_refine and stack_combine before coast_mask_boundary"
                )
            stacks.add(str(Path(stack).resolve(strict=True)))
            indexes.add(str(Path(index).resolve(strict=True)))
        if len(stacks) != 1 or len(indexes) != 1:
            raise ValueError("Expected one complete native MultiStack and group index")
        publish_zero_estimate(
            entries[0][1],
            self.group,
            stacks.pop(),
            max_distance_m=self.max_distance_m,
            min_weight=self.min_weight,
        )
        return entries
