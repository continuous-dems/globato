"""Global coordination through the existing source-grid/finalizer lifecycle."""

from pathlib import Path
import json
from fetchez.hooks import FetchHook
from fetchez.utils import str2bool
from .integration import (
    CoastRefineFinalize,
    _register_sources,
    _consumer_registered_class_metadata,
    _source_policy,
)
from ._io import json_write, digest, serializable
from .artifact import MODES, MaskArtifact


class CoastMaskSource(FetchHook):
    """Declare evidence participation separately from optional source masking."""

    name = "coast_mask_source"
    meta_stage = "file"

    def __init__(self, group, evidence=False, remove=None, **kwargs):
        if set(kwargs) - {"stage"}:
            raise TypeError("Shared geometry/grid settings belong on coast_mask_refine")
        super().__init__(**kwargs)
        if self.stage != "file":
            raise ValueError("coast_mask_source must register before streaming")
        if not isinstance(group, str) or not group.strip():
            raise ValueError("A named Coast Mask group is required")
        if remove is not None and remove not in MODES:
            raise ValueError("Unknown source removal policy")
        self.group = group.strip()
        self.evidence = str2bool(evidence)
        self.remove = remove
        if not self.evidence and remove is None:
            raise ValueError(
                "Source must contribute evidence or request masking; otherwise omit this declaration"
            )

    def run(self, entries):
        settings = dict(
            group=self.group,
            grid_native=True,
            remove=self.remove,
            global_refinement=True,
            evidence=self.evidence,
            source_hook="coast_mask",
        )
        # Registration snapshots and native encoding checks are shared. Consumer
        # encodings need no terrain-authority decoding contract.
        selected = _register_sources(entries, settings, None)
        for module, entry in selected:
            request = entry["coast_refine_request"]
            metadata, claims = _consumer_registered_class_metadata(module, entry)
            request.update(
                registered_source_metadata=serializable(metadata),
                class_branch_claims=claims,
            )
            if not self.evidence:
                request["authority"] = request.pop("group")
                entry["coast_refine_consumer"] = entry.pop("coast_refine_request")
        return selected


class CoastMaskRefine(CoastRefineFinalize):
    """One global owner of geometry policy, preparation and presentation."""

    name = "coast_mask_refine"
    meta_input_paths = ("mask",)
    meta_output_paths = ("output_dir",)
    meta_named_inputs = {"mask": ("osm",)}

    def __init__(
        self,
        mask="osm",
        polygon_class=None,
        output_dir=None,
        viz=False,
        outlet_lonlat=None,
        **kwargs,
    ):
        if "binding_group" in kwargs or "require_barrier" in kwargs:
            raise TypeError(
                "Select coast_mask_group on ms_binary_cudem, not on source refinement"
            )
        unknown = set(kwargs) - {
            "res",
            "crs",
            "resolutions",
            "audit_overlaps",
            "stage",
        }
        if unknown:
            raise TypeError(
                "Unsupported global refinement settings: " + ", ".join(sorted(unknown))
            )
        self.policy = _source_policy(
            "shared", mask, polygon_class, "water", outlet_lonlat
        )
        self.output_dir = output_dir
        self.viz = str2bool(viz)
        super().__init__(grid_native=True, defer_barrier=True, **kwargs)

    def run(self, entries):
        from globato.hooks.sinks.side_stack import unique_source_contributions

        entries = unique_source_contributions(entries)
        registered = []
        for module, entry in entries:
            request = entry.get("coast_refine_request") or entry.get(
                "coast_refine_consumer"
            )
            if not request:
                continue
            if not request.get("global_refinement"):
                raise ValueError(
                    "Use coast_mask_source declarations with coast_mask_refine; legacy recipes keep coast_mask_finalize"
                )
            registered.append((module, entry))
            request.update(
                {k: v for k, v in self.policy.items() if k not in ("group", "remove")}
            )
            request.update(
                barrier="coast_refine",
                review=False,
                viz=self.viz,
                output_dir=str(
                    Path(self.output_dir or Path.cwd() / "coast_mask").resolve()
                ),
            )
        if not registered:
            raise ValueError("No coast_mask_source declarations were registered")
        result = super().run(entries)
        # A file-oriented collection index retains all actual contributions,
        # including those which did not participate in coastline refinement.
        rows = []
        original_hashes = {}
        for module, entry in result:
            state = (entry.get("artifacts") or {}).get("side-stack")
            if not state or not Path(state).is_file():
                raise ValueError(
                    "Collection source lacks a materialized native side-stack"
                )
            request = (
                entry.get("coast_refine_request")
                or entry.get("coast_refine_consumer")
                or {}
            )
            source_path = entry.get("dst_fn")
            asset_hash = None
            if source_path and Path(source_path).is_file():
                key = str(Path(source_path).resolve())
                if key not in original_hashes:
                    original_hashes[key] = digest(source_path)
                asset_hash = original_hashes[key]
            registered_hash = entry.get("_sidestack_processing_identity", {}).get(
                "asset_sha256"
            )
            if registered_hash and asset_hash != registered_hash:
                raise ValueError("Original source changed during gridding")
            rows.append(
                dict(
                    module=module.name,
                    original_file=entry.get("dst_fn"),
                    original_file_sha256=asset_hash,
                    source_id=entry.get("source_id"),
                    source_url=entry.get("url"),
                    module_weight=getattr(module, "weight", None),
                    entry_weight=entry.get("weight"),
                    contribution_id=serializable(
                        entry.get("contribution_id")
                        or {
                            "native_dataset": entry["_sidestack_processing_identity"][
                                "native_dataset_id"
                            ]
                        }
                    ),
                    native_source_identity=entry.get("_sidestack_processing_identity"),
                    source_aliases=serializable(entry.get("_sidestack_aliases", [])),
                    group=request.get("group") or request.get("authority"),
                    evidence=request.get("evidence", False),
                    remove=request.get("remove"),
                    accepted_state=str(Path(state).resolve()),
                    sha256=digest(state),
                    source_receipt=(entry.get("artifacts") or {}).get(
                        "coast_mask_source_receipt"
                    ),
                    lineage_assurance="registered_source"
                    if request
                    else "unmasked_native_entry_only",
                    survey_identity_inferred=False,
                )
            )
        index = (
            Path(
                registered[0][1].get(
                    "coast_refine_request",
                    registered[0][1].get("coast_refine_consumer"),
                )["output_dir"]
            )
            / "accepted_collection.json"
        )
        json_write(
            index,
            dict(
                schema="coast_mask.accepted_collection.v1",
                status="completed",
                sources=rows,
            ),
        )
        for _, entry in result:
            entry.setdefault("artifacts", {})["coast_mask_accepted_collection"] = str(
                index
            )
        return result


def resolve_group_publication(entry, group):
    """Resolve the existing completed group contract for any native consumer."""
    if not isinstance(group, str) or not group.strip():
        raise ValueError("coast_mask_group must name a published group")
    exports = entry.setdefault("artifacts", {})
    index = exports.get("coast_mask_groups_index")
    if not index:
        raise ValueError(
            "No completed Coast Mask group index; run coast_mask_refine first"
        )
    record = json.loads(Path(index).read_text())
    if (
        record.get("schema") != "coast_mask_group_index.v1"
        or record.get("status") != "completed"
        or group.strip() not in record.get("groups", {})
    ):
        raise ValueError("Unknown/incomplete Coast Mask interpolation group: " + group)
    publication = record["groups"][group.strip()]
    if publication.get("group") != group.strip():
        raise ValueError("Coast Mask publication group mismatch")
    artifact = MaskArtifact(publication["manifest"])
    if publication.get("manifest_sha256") != artifact.manifest_hash:
        raise ValueError("Published Coast Mask manifest changed")
    return Path(index).resolve(strict=True), record, publication, artifact


def select_group_barrier(entry, group, source_raster, *, boundary="refined"):
    """Resolve one explicitly selected vector against the current full stack."""
    if boundary not in ("refined", "estimated_zero"):
        raise ValueError("coast_mask_boundary must be refined or estimated_zero")
    _, _, publication, artifact = resolve_group_publication(entry, group)
    exports = entry.setdefault("artifacts", {})
    topology = artifact.barrier_for(source_raster)
    for stale in (
        "coast_refine_barrier_extension",
        "coast_refine_barrier_options",
        "coast_refine_connected_water",
        "coast_refine_connected_receipt",
    ):
        exports.pop(stale, None)
    if publication.get("connection_status") == "selected":
        exports["coast_refine_connected_water"] = publication["connected_water"]
        exports["coast_refine_connected_receipt"] = publication["connected_receipt"]
    exports.update(
        coast_refine_manifest=str(artifact.path),
        coast_refine_barrier=str(topology),
        coast_refine_river_guidance="off",
    )
    if boundary == "estimated_zero":
        estimate = publication.get("estimated_zero")
        if not estimate:
            raise ValueError(
                "No estimated zero boundary; run coast_mask_boundary after stack_combine"
            )
        receipt = Path(estimate["receipt"])
        if digest(receipt) != estimate["receipt_sha256"]:
            raise ValueError("Published zero-estimate receipt changed")
        from .extension import validate

        resolved = validate(artifact, receipt)
        estimate_record = json.loads(receipt.read_text())
        if digest(source_raster) != estimate_record["inputs"].get(
            "support_stack_sha256"
        ):
            raise ValueError(
                "Estimated boundary does not describe the current complete stack"
            )
        if (
            str(resolved) != estimate["topology"]
            or digest(resolved) != estimate["topology_sha256"]
        ):
            raise ValueError("Published zero-estimate topology changed")
        exports["coast_refine_barrier_extension"] = str(receipt)
        exports["coast_refine_barrier"] = str(resolved)
        topology = resolved
    return str(topology)
