"""Native integration: prepare source stacks, run QC, return original streams.

This is the only source-preparation adapter. Scientific array modules do not
import Fetchez. Shared source selection has already run in Globato StackQCPlan.
"""

import copy
import inspect
import json
import logging
from pathlib import Path
import rasterio
from filelock import FileLock
from fetchez.core import run_fetchez
from .pipeline import PreparedSources
from fetchez.utils import merge_hooks
from fetchez.hooks.focus import FocusSink
from globato.hooks.rasters.base import RasterHook
from globato.hooks.sinks.raster_writer import RasterWrite
from globato.hooks.metadata.provenance import source_id
from globato.hooks.sinks.side_stack import (
    SideStackCheck,
    SideStackGenerate,
    SIDESTACK_DATATYPE,
    SIDESTACK_HASH_TAG,
    hook_identity,
)
from . import io, processing
from .grid import grid_offset
from .native import NativeBuilder, masked_stream

logger = logging.getLogger(__name__)

# Native consumers run after QC. Final output hooks run only in the real build.
SOURCE_CONSUMERS = {"provenance", "source_masks", "multi_stack", "stack_provenance"}
FINAL_OUTPUTS = {
    "raster_metadata",
    "viz_geoshade",
    "format_cog",
    "copy_artifact",
    "cleanup_tmp",
    "transfer_log",
    "audit",
}


def name(hook):
    return hook.name.replace("-", "_")


def source_pipeline(qc, module, global_hooks, stack):
    """Keep native source transforms and put side-stack persistence last."""
    if any(h is not qc and h.stage == "collection" for h in module.hooks):
        raise ValueError(
            "Stack QC needs one source preparation owner per module. Move "
            "unrelated final collection hooks to global_hooks."
        )
    if any(name(h) in SOURCE_CONSUMERS for h in module.hooks):
        raise ValueError(
            "Stack QC source consumers must be in global_hooks so they run after QC"
        )
    hooks = [
        copy.deepcopy(h)
        for h in global_hooks
        if name(h) not in SOURCE_CONSUMERS and h.stage not in {"collection", "manifest"}
    ]
    local_checks = [
        copy.deepcopy(h) for h in module.hooks if name(h) == "side_stack_check"
    ]
    checks = [copy.deepcopy(h) for h in global_hooks if name(h) == "side_stack_check"]
    checks = merge_hooks(local_checks, checks)
    if not checks:
        checks = [SideStackCheck(res=stack.res, crs=stack.crs)]
    # The default is automatic; if users supply an explicit writer it must be
    # unambiguous. Never silently select one of several incompatible caches.
    local_writers = [h for h in module.hooks if name(h) == "side_stack"]
    global_writers = [h for h in hooks if name(h) == "side_stack"]
    if len(local_writers) + len(global_writers) > 1:
        raise ValueError(
            "Stack QC accepts at most one explicit side_stack writer; "
            "omit side_stack entirely to use the automatic default"
        )
    writers = local_writers or global_writers
    writer = copy.deepcopy(writers[0]) if writers else SideStackGenerate()
    directories = {
        Path(h.cache_dir).resolve() for h in [*checks, writer] if h.cache_dir
    }
    if len(directories) > 1:
        raise ValueError("side_stack and side_stack_check must use the same cache_dir")
    directory = str(next(iter(directories))) if directories else None
    for hook in [*checks, writer]:
        hook.res, hook.crs = stack.res, stack.crs
        hook.cache_dir = directory
    return checks, [h for h in hooks if name(h) != "side_stack"] + [writer]


def fetch_sources(qc, module, hooks, checks, threads):
    """Check native caches, then fetch/transform only the misses."""
    clone = copy.copy(module)
    clone.results = [copy.deepcopy(e) for e in module.results]
    clone.internal_hooks = [
        copy.deepcopy(h)
        for h in module.internal_hooks
        if h is not qc and h.stage != "manifest" and name(h) != "side_stack"
    ]
    clone.external_hooks = [
        copy.deepcopy(h)
        for h in module.external_hooks
        if h is not qc and h.stage != "manifest" and name(h) != "side_stack"
    ]
    for entry in clone.results:
        processing_receipt = [
            hook_identity(h, entry) for h in hooks if name(h) != "side_stack"
        ]
        entry["hooks"] = list(entry.get("hooks", [])) + [
            dict(name="stack_qc_source_pipeline", args=processing_receipt)
        ]
    entries = [(clone, e) for e in clone.results]
    for hook in checks:
        entries = hook.run(entries) or []
    hits = [(m, e) for m, e in entries if e.get("_sidestack_hit")]
    clone.results = [e for _, e in entries if not e.get("_sidestack_hit")]
    pending = (
        run_fetchez([clone], threads=threads, global_hooks=hooks, ignore_failures=False)
        if clone.results
        else []
    )
    logger.info(
        "Stack QC source cache: %d hits; %d source/metadata misses",
        len(hits),
        len(clone.results),
    )
    return hits + list(pending or [])


def collect_sources(module, prepared, root):
    records = []
    ancillary = []
    source_entries = {}
    seen = set()
    for owner, entry in prepared:
        raw = entry.get("_sidestack_path") or entry.get("artifacts", {}).get(
            "side-stack"
        )
        if not raw or not Path(raw).is_file():
            if entry.get("stream") is None:
                # Keep ancillary metadata for normal provenance, without
                # treating it as elevation support or reparsing it.
                ancillary.append(entry.copy())
                continue
            raise ValueError("A measured source stream has no original side-stack")
        path = Path(raw).resolve(strict=True)
        with rasterio.open(path) as ds:
            tags = ds.tags()
            if ds.count != 7 or tags.get("GLOBATO_DATATYPE") != SIDESTACK_DATATYPE:
                raise ValueError(
                    "Stack QC requires original native seven-band source grids"
                )
            geom = io.grid(ds)
            if tags.get("GLOBATO_POINT_COUNT") == "0":
                ancillary.append(entry.copy())
                continue
        # A cache hit bypasses reprojection; describe the processed raster,
        # rather than the input CRS left by source manifest hooks.
        entry["src_srs"] = geom["crs"]
        sid = source_id(entry) + "|sidestack=" + tags[SIDESTACK_HASH_TAG]
        if sid in seen:
            raise ValueError("Duplicate measured source identity")
        seen.add(sid)
        fp = io.fingerprint(path)
        receipt = (
            root
            / "hashes"
            / (io.digest(dict(path=str(path), fingerprint=fp)) + ".json")
        )
        if receipt.exists():
            source_sha = io.read(receipt)["sha256"]
        else:
            source_sha = io.sha(path)
            io.write(receipt, dict(sha256=source_sha))
        saved = {}
        for key, value in entry.items():
            if key in {"stream", "raster_stream"}:
                continue
            try:
                json.dumps(value, allow_nan=False)
                saved[key] = value
            except (TypeError, ValueError):
                pass
        metadata = entry.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        group = entry.get("survey_id") or metadata.get("survey_id")
        records.append(
            dict(
                source_id=sid,
                path=str(path),
                source_sha256=source_sha,
                fingerprint=fp,
                grid=geom,
                group=str(group).strip()
                if group is not None and str(group).strip()
                else None,
                weight=module.weight,
                module=module.name,
                cache_dir=module.outdir or module._outdir,
                entry=saved,
            )
        )
        source_entries[sid] = entry
    # Fetch completion order is not a scientific input. Stable ordering is
    # essential for reusing exact source states across ordinary DEM builds.
    records.sort(key=lambda r: r["source_id"])
    unknown = sum(r["group"] is None for r in records)
    logger.info(
        "Stack QC: %d sources, %d independent survey groups, %d unknown identities",
        len(records),
        len({r["group"] for r in records if r["group"] is not None}),
        unknown,
    )
    return records, ancillary, source_entries


def context_key(qc, module, records, stack, post):
    def config(h):
        d = hook_identity(h)
        cfg = d["config"]
        # Output locations and transient scratch paths cannot change science.
        for key in (
            "output",
            "output_path",
            "local_tmp",
            "cache_dir",
            "state",
            "resume",
            "keep_steps",
            "current_mod",
        ):
            cfg.pop(key, None)
        return d

    science = Path(__file__).parent
    return io.digest(
        dict(
            sources=[[r["source_id"], r["source_sha256"], r["group"]] for r in records],
            region=module.region.to_list(),
            native=[config(h) for h in [stack, *post]],
            owners={
                str(p): io.sha(p)
                for p in {Path(inspect.getfile(h.__class__)) for h in [stack, *post]}
            },
            science={p.name: io.sha(p) for p in science.glob("*.py")},
            adapter=io.sha(inspect.getfile(qc.__class__)),
        )
    )


def native_context_hooks(global_hooks, interpolator):
    """Replay native raster operations; leave publication/notifications to the real build.

    Context starts from an already finalized multi_stack artifact. Native raster
    initialization, processing, checkpoints and artifact focusing are supported.
    Generic collection hooks are not part of temporary scientific interpolation.
    """
    context = []
    for hook in global_hooks:
        if (
            hook.stage == "collection"
            and name(hook) not in FINAL_OUTPUTS
            and isinstance(hook, (RasterHook, FocusSink, RasterWrite))
        ):
            context.append(hook)
        if hook is interpolator:
            break
    if not context or context[-1] is not interpolator:
        raise ValueError("Native interpolation must be a collection raster hook")
    return context


def run_qc(qc, module, records, root, stack, post):
    """Build initial context, align original grids, then run the removal/recovery loop."""
    key = context_key(qc, module, records, stack, post)
    cache = root / key
    cache.mkdir(exist_ok=True)
    with FileLock(str(cache / ".lock")):
        builder = NativeBuilder(stack, post, records, module.region)
        logger.info("Stack QC: obtaining initial native context surface")
        baseline = builder(None, cache / "baseline")
        builder.baseline = baseline
        with rasterio.open(baseline) as ds:
            inventory = dict(sources=records, grid=io.grid(ds))
            for rec in records:
                with rasterio.open(rec["path"]) as src:
                    rec["offset"] = list(grid_offset(src, ds))
                rec["active_blocks"] = builder.previous["source_blocks"][
                    rec["source_id"]
                ]
        io.write(cache / "SOURCES.json", inventory)
        result = processing.process(inventory, qc.settings, cache, builder)
    return result, cache


def accepted_sources(records, source_entries, manifest, target=None):
    """Return original native streams with the final 0/1 rejection masks applied."""
    from .resampling import accepted_stream

    final = io.read(manifest)["masks"]
    entries = []
    for rec in records:
        entry = source_entries[rec["source_id"]].copy()
        entry.update(
            stream=(
                masked_stream(rec["path"], final[rec["source_id"]]["mask"])
                if target is None
                else accepted_stream(
                    rec["path"],
                    final[rec["source_id"]]["mask"],
                    target,
                    active_blocks=rec.get("active_blocks"),
                    source_offset=rec.get("offset", (0, 0)),
                )
            ),
            stream_type="raster-stream",
            data_type="fusion-state",
            _stack_qc_source_id=rec["source_id"],
        )
        entry["artifacts"] = dict(
            entry.get("artifacts", {}),
            stack_qc_mask=final[rec["source_id"]]["mask"],
        )
        entries.append(entry)
    return entries


def prepare(qc, module, global_hooks, *, threads):
    """One readable orchestration path for every compatible elevation module."""
    if not module.results:
        logger.info(
            "Stack QC: %s has no selected sources; continuing normally", module.name
        )
        return PreparedSources(module, [])
    stacks = [h for h in global_hooks if name(h) == "multi_stack"]
    interpolators = [h for h in global_hooks if name(h) == "ms_binary_cudem"]
    if len(stacks) != 1 or len(interpolators) != 1:
        raise ValueError(
            "Stack QC needs one multi_stack and one ms_binary_cudem in global_hooks"
        )
    stack = stacks[0]
    from .resampling import context_hooks, output_grid, alignment, nested_region

    post = native_context_hooks(global_hooks, interpolators[0])
    qc_stack, post = context_hooks(stack, post, qc.qc_res)
    target = output_grid(stack, module.region) if qc_stack is not stack else None
    source_module = module
    expected_qc_grid = None
    if target is not None:
        # The final native grid owns extent rounding. Preparation uses a copy;
        # discovery and final consumers retain the original recipe module.
        source_module = copy.copy(module)
        source_module.region, expected_qc_grid = nested_region(
            module.region, qc_stack, target
        )
    # 1. Resolve native source preparation, including automatic side-stacks.
    checks, hooks = source_pipeline(qc, source_module, global_hooks, qc_stack)
    logger.info(
        "Stack QC: preparing native sources (%d selected entries)", len(module.results)
    )
    # 2. Reuse prepared sources where possible; fetch only cache misses.
    prepared = fetch_sources(qc, source_module, hooks, checks, threads)
    root = Path(qc.cache_dir or Path(module._outdir) / "stack_qc").resolve()
    root.mkdir(parents=True, exist_ok=True)
    # 3. Inventory immutable source contributions and native grid metadata.
    records, ancillary, source_entries = collect_sources(source_module, prepared, root)
    if not records:
        logger.info(
            "Stack QC: %s has no elevation observations in this region; continuing normally",
            module.name,
        )
        return PreparedSources(module, ancillary)
    # Validate nested geometry before any expensive scientific interpolation.
    if target is not None:
        for rec in records:
            with rasterio.open(rec["path"]) as src:
                alignment(src, target)
                if (src.width, src.height, src.transform, src.crs) != (
                    expected_qc_grid["width"],
                    expected_qc_grid["height"],
                    expected_qc_grid["transform"],
                    expected_qc_grid["crs"],
                ):
                    raise ValueError(
                        "Source side-stack does not match the QC grid nested in the final DEM. "
                        "Check its CRS, bounds, origin and resolution; no automatic resampling is allowed."
                    )
    # 4. Rebuild native context as needed, apply rejection and restoration.
    result, cache = run_qc(qc, source_module, records, root, qc_stack, post)
    if target is not None:
        result = dict(result, qc_res=qc_stack.res, dem_resolution=stack.res)
        logger.info(
            "Stack QC grids: QC=%s; final DEM=%s (accepted source statistics aggregated)",
            qc_stack.res,
            stack.res,
        )
    # 5. Publish accepted original measurements for normal Globato stacking.
    # Optional pictures are scheduled after the real final DEM is produced.
    if qc.review:
        # A normal global collection hook runs *after* the real DEM build.
        # Scientific preparation and final interpolation never depend on PNGs.
        from .review.publisher import ReviewPublisher

        existing = [h for h in global_hooks if isinstance(h, ReviewPublisher)]
        review = ReviewPublisher(
            cache=cache,
            result=result,
            stack=qc_stack,
            region=source_module.region,
            interpolator=interpolators[0],
            module_name=module.name,
        )
        if existing:
            for hook in existing:
                if hook.output is None:
                    hook.output = (
                        hook.final.parent / "stack_qc_reviews" / hook.cache.name[:12]
                    )
            review.output = review.final.parent / "stack_qc_reviews" / cache.name[:12]
        global_hooks.append(review)
    entries = list(ancillary) + accepted_sources(
        records, source_entries, result["manifest"], target
    )
    logger.info(
        "Stack QC complete: aggressiveness=%d, passes=%d, restored=%d; %s",
        qc.settings.aggressiveness,
        result["passes_completed"],
        result["restored_source_cells"],
        cache,
    )
    return PreparedSources(module, entries)
