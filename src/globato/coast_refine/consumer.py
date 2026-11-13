"""Complete native point masks after their selected coastline authority exists.

The existing capture hook saves prepared samples. Collection uses native point
masking and side-stack generation, without replaying source readers/transforms.
"""

import json
import logging
from pathlib import Path
import tempfile
from time import perf_counter
from types import SimpleNamespace

from filelock import FileLock
from fetchez.entry import set_contribution_id
from fetchez.utils import atomic_output
from globato.hooks.filters.point_raster_mask import PointRasterMask
from globato.hooks.sinks.side_stack import SideStackGenerate
from globato.streams.readers import gtpc

from ._io import digest, identity, json_write
from .artifact import _covers_without_roundoff
from .capture import inventory
from .provenance import source_snapshot

log = logging.getLogger(__name__)


def register_consumer(module, entry, *, authority, remove, res):
    if entry.get("coast_refine_request") or entry.get("coast_refine_consumer"):
        raise ValueError("Source already has a Coast Refine producer/consumer binding")
    if entry.get("_sidestack_hit") or entry.get("coast_refine_mask_receipt"):
        raise ValueError(
            "Bind authority masks before replacing or accumulating the source"
        )
    path = Path(entry["dst_fn"]).expanduser().resolve(strict=True)
    root = (
        Path(getattr(module, "_outdir", None) or Path.cwd()) / "coast_refine_consumers"
    )
    entry["coast_refine_consumer"] = {
        "point_source": True,
        "authority": authority,
        "remove": remove,
        "res": res,
        "output_dir": str(root.resolve()),
        "source": source_snapshot(module, entry, path, point_source=True),
    }
    entry["coast_refine_deferred"] = True


def finish_consumers(entries, prepared, authorities, *, res, crs):
    for module, entry in entries:
        request = entry.get("coast_refine_consumer")
        if request is None:
            continue
        group = request["authority"]
        if group not in authorities or group not in prepared:
            raise ValueError(f"Required Coast Refine authority is unavailable: {group}")
        root = Path(request["output_dir"])
        locks = root / "locks"
        locks.mkdir(parents=True, exist_ok=True)
        key = identity(
            {
                "source": request["source"],
                "authority": group,
                "remove": request["remove"],
                "res": request["res"],
            }
        )
        with FileLock(str(locks / (key + ".lock"))):
            _finish_one(module, entry, request, prepared[group], res=res, crs=crs)


def _finish_one(module, entry, request, artifact, *, res, crs):
    from .integration import check_source, horizontal_crs

    started = perf_counter()
    check_source(module, entry, request)
    if not request.get("points") or digest(request["points"]) != request.get(
        "points_sha256"
    ):
        raise ValueError(
            "Required prepared-point capture is absent, incomplete or changed"
        )
    saved = inventory(request["points"])
    stream_srs = request["stream_srs"]
    if horizontal_crs(stream_srs) != horizontal_crs(crs):
        raise ValueError("Deferred point mask needs the prepared output horizontal CRS")
    if horizontal_crs(module.region.srs) != horizontal_crs(stream_srs):
        raise ValueError("Deferred point mask region and point horizontal CRS differ")
    scope = artifact.scope_for(module)
    if not artifact._coverage_covers(scope) or not _covers_without_roundoff(
        artifact.aoi, scope
    ):
        raise ValueError(
            "Coast Refine authority does not cover the consumer processing region"
        )
    product = "refined_land" if request["remove"] == "water" else "refined_water"
    policy = {
        "authority": request["authority"],
        "preparation": artifact.preparation_id,
        "manifest_sha256": artifact.manifest_hash,
        "points_sha256": request["points_sha256"],
        "source": request["source"],
        "remove": request["remove"],
        "mask_res": request["res"],
        "stream_srs": stream_srs,
        "grid_res": res,
        "grid_crs": crs,
        "adapter_sha256": digest(__file__),
    }
    processing_id = identity(policy)
    set_contribution_id(entry, "coast_refine_mask", processing_id)
    root = Path(request["output_dir"])
    receipt_path = root / "receipts" / (processing_id + ".json")
    receipt = None
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt.get("processing_id") != processing_id
            or receipt.get("policy") != policy
        ):
            raise ValueError(
                "Masked source receipt differs from this processing policy"
            )
        if digest(receipt["side_stack"]) != receipt.get("side_stack_sha256"):
            raise ValueError("Completed masked source cache changed")
    # Namespace the generic barrier cache by the immutable authority preparation.
    context = SimpleNamespace(
        name=module.name,
        region=module.region,
        outdir=str(root / "mask_cache" / artifact.preparation_id),
    )
    local = dict(
        entry,
        src_srs=stream_srs,
        stream_type="point-stream",
        artifacts=dict(entry.get("artifacts", {})),
    )
    local.pop("_sidestack_path", None)
    local.pop("_sidestack_hit", None)
    local["artifacts"].pop("side-stack", None)
    mask = PointRasterMask(
        barrier=str(artifact.products[product]),
        res=request["res"],
        invert=False,
        soft=False,
    )
    if not mask.setup(context, local):
        raise ValueError("Required Coast Refine point mask could not be resolved")
    counts = {"input_points": 0, "retained_points": 0}
    read_chunks = getattr(gtpc, "iter_gtpc_chunks", None)
    if not callable(read_chunks):
        raise ValueError(
            "Prepared-point Coast Refine requires a runtime with iter_gtpc_chunks; "
            "use the public grid-native coast_mask workflow on this Globato version"
        )

    def filtered():
        for chunk in read_chunks(request["points"]):
            counts["input_points"] += len(chunk)
            accepted = mask.filter_chunk(chunk)
            counts["retained_points"] += len(accepted)
            yield accepted

    local["stream"] = filtered()
    if receipt is not None:
        local["_sidestack_path"] = receipt["side_stack"]
        cache_dir = str(Path(receipt["side_stack"]).parent)
    else:
        # An interrupted attempt can leave a complete state without its receipt.
        # A fresh native writer attempt preserves that orphan and safely resumes.
        states = root / "states"
        states.mkdir(parents=True, exist_ok=True)
        cache_dir = tempfile.mkdtemp(prefix=processing_id[:16] + "_", dir=states)
    sink = SideStackGenerate(res=res, crs=crs, cache_dir=cache_dir)
    sink.run([(module, local)])
    if local.get("stream_type") == "raster-stream":
        # Core does not automatically drain streams created by collection hooks.
        for _ in local["stream"]:
            pass
        if counts["input_points"] != saved["points"]:
            raise ValueError("Deferred capture was not completely consumed")
        state = local["artifacts"]["side-stack"]
        receipt = {
            "processing_id": processing_id,
            "policy": policy,
            "counts": counts,
            "side_stack": state,
            "side_stack_sha256": digest(state),
        }
        with atomic_output(receipt_path) as pending:
            json_write(pending, receipt)
    else:
        local["stream"].close()
        if receipt is None:
            raise ValueError("Native side-stack cache has no completed mask receipt")
    entry.setdefault("artifacts", {})["side-stack"] = local["artifacts"]["side-stack"]
    entry["artifacts"]["coast_refine_mask_receipt"] = str(receipt_path)
    entry["coast_refine_mask_receipt"] = str(receipt_path)
    entry.pop("stream", None)
    log.info(
        "Coast Refine consumer %s: %s; retained %s/%s; %.2fs",
        Path(entry["dst_fn"]).name,
        request["remove"],
        receipt["counts"]["retained_points"],
        receipt["counts"]["input_points"],
        perf_counter() - started,
    )
