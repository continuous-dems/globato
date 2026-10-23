"""Stack QC orchestration for one module's collected sources.

Runs at the collection stage, after the normal stream stage has written each
source's side-stack on the DEM grid and before ``stack_combine`` assembles
the MultiStack:

1. Use the module's side-stacks as the QC grids, or, with a finer ``qc_res``,
   grid each source again on the QC grid with the module's own file and
   stream hooks (cached by the ordinary side-stack cache).
2. Run the removal/restoration passes against in-process working DEMs.
3. Write each source's accepted measurements as a side-stack on the DEM grid
   and point the entry's ``side-stack`` artifact at it, so ``stack_combine``
   stacks the filtered data. The original side-stacks are left untouched.

The QC cache key covers only what can change the decisions: the module's
sources and their processing, the module weight, the QC grid and region, the
filter settings, the working-DEM settings and the QC code. Changes elsewhere
in a recipe (another dataset's weight, a new dataset, final DEM tiers) reuse
the cached decisions.
"""

import copy
import inspect
import json
import logging
import time
from pathlib import Path

import rasterio
from filelock import FileLock

from fetchez.core import run_fetchez
from globato.hooks.metadata.provenance import source_id
from globato.hooks.sinks.side_stack import (
    SideStackCheck,
    SideStackGenerate,
    SIDESTACK_DATATYPE,
    SIDESTACK_HASH_TAG,
)
from . import io, processing
from .context import WorkingSurface, cache_root_for
from .grid import grid_offset
from .native import masked_stream

logger = logging.getLogger(__name__)

# Bump when the orchestration changes in a way that changes decisions.
ORCHESTRATION_VERSION = "stack_qc_hook_v1"
INDEX = "_stack_qc_index"


def _name(hook):
    return str(getattr(hook, "name", "")).replace("-", "_")


def _source_hooks(module, qc):
    """The module's own per-file transforms, without QC, manifests or caches."""

    def keep(h):
        return (
            h is not qc
            and _name(h) not in {"stack_qc", "side_stack", "side_stack_check"}
            and h.stage not in {"manifest", "collection"}
        )

    return (
        [copy.deepcopy(h) for h in module.internal_hooks if keep(h)],
        [copy.deepcopy(h) for h in module.external_hooks if keep(h)],
    )


RUNTIME_KEYS = {"stream", "raster_stream", "stream_type", "artifacts"}


def _manifest_srs(entry):
    """The entry's source CRS before its stream stage, if it was recorded.

    Stream transforms such as stream_reproject overwrite ``src_srs``; the
    manifest-stage side-stack lookup recorded the original in its identity.
    """
    raw = entry.get("_sidestack_identity")
    if not raw:
        return None
    for hook in json.loads(raw).get("processing", []):
        if hook.get("name") == "__fusion_context__":
            return hook.get("config", {}).get("source_crs")
    return None


def _fresh(entry):
    """A copy of a collected entry as it was before its stream stage."""
    keep = {
        k: v
        for k, v in entry.items()
        if k not in RUNTIME_KEYS and not k.startswith("_sidestack")
    }
    if keep.get("data_type") == "fusion-state":
        keep.pop("data_type")
    srs = _manifest_srs(entry)
    if srs is not None:
        keep["src_srs"] = srs
    keep["_stack_qc_restored_srs"] = srs is not None
    return copy.deepcopy(keep)


def _replay_manifest_hooks(module, qc, entries):
    """Re-apply the module's own manifest hooks (e.g. set-srs) to fresh copies."""
    for hook in module.hooks:
        if hook is qc or hook.stage != "manifest":
            continue
        if _name(hook) in {"side_stack_check", "stack_qc"}:
            continue
        entries = copy.deepcopy(hook).run(entries) or []
    return entries


def grid_sources(qc, module, entries, region, res, threads, *, collected=True):
    """Seven-band QC-grid side-stacks for every selected entry (cached).

    ``collected`` entries have been through their stream stage; manifest-stage
    entries have not, so they need no source-CRS restoration.
    """
    clone = copy.copy(module)
    clone.region = region
    clone.internal_hooks, clone.external_hooks = _source_hooks(module, qc)
    fresh = [_fresh(e) for _, e in entries]
    if not collected:
        for e in fresh:
            e["_stack_qc_restored_srs"] = True
    if not all(e["_stack_qc_restored_srs"] for e in fresh):
        # No recorded manifest-time CRS: rebuild it from the module's hooks.
        fresh = [
            e
            for _, e in _replay_manifest_hooks(module, qc, [(clone, e) for e in fresh])
        ]
    for e in fresh:
        e.pop("_stack_qc_restored_srs", None)
    clone.results = fresh
    check = SideStackCheck(res=res, crs=qc.crs, cache_dir=qc.side_stack_cache)
    generate = SideStackGenerate(res=res, crs=qc.crs, cache_dir=qc.side_stack_cache)
    checked = check.run([(clone, e) for e in clone.results]) or []
    hits = [(m, e) for m, e in checked if e.get("_sidestack_hit")]
    clone.results = [e for _, e in checked if not e.get("_sidestack_hit")]
    made = (
        run_fetchez(
            [clone], threads=threads, global_hooks=[generate], ignore_failures=False
        )
        if clone.results
        else []
    )
    logger.info(
        "Stack QC %s: QC-grid side-stacks: %d cached, %d made",
        module.name,
        len(hits),
        len(clone.results),
    )
    return hits + list(made or [])


def collect_sources(module, prepared, root):
    """Inventory measured QC-grid sources; entries with no data come back separately."""
    records, empty = [], []
    seen = set()
    for _, entry in prepared:
        raw = entry.get("_sidestack_path") or entry.get("artifacts", {}).get(
            "side-stack"
        )
        if not raw or not Path(raw).is_file():
            empty.append(entry[INDEX])
            continue
        path = Path(raw).resolve(strict=True)
        with rasterio.open(path) as ds:
            tags = ds.tags()
            if ds.count != 7 or tags.get("GLOBATO_DATATYPE") != SIDESTACK_DATATYPE:
                raise ValueError("Stack QC requires seven-band source side-stacks")
            geom = io.grid(ds)
            if tags.get("GLOBATO_POINT_COUNT") == "0":
                empty.append(entry[INDEX])
                continue
        sid = source_id(entry) + "|sidestack=" + tags[SIDESTACK_HASH_TAG]
        if sid in seen:
            raise ValueError(f"Duplicate measured source identity: {sid}")
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
                index=entry[INDEX],
                src_srs=geom["crs"],
            )
        )
    # Fetch completion order is not a scientific input.
    records.sort(key=lambda r: r["source_id"])
    logger.info(
        "Stack QC %s: %d sources with data, %d survey groups, %d without survey id, %d empty",
        module.name,
        len(records),
        len({r["group"] for r in records if r["group"] is not None}),
        sum(r["group"] is None for r in records),
        len(empty),
    )
    return records, empty


def _code_hashes():
    here = Path(__file__).parent
    from globato.hooks.filters import stack_qc as hook_module
    from globato.hooks.rasters import binary_cudem
    from globato.hooks.sinks import multi_stack

    files = {p.name: io.sha(p) for p in sorted(here.glob("*.py"))}
    for mod in (hook_module, binary_cudem, multi_stack):
        path = Path(inspect.getfile(mod))
        files[path.name] = io.sha(path)
    return files


def cache_key(qc, module, records, region, working):
    """Everything that can change the decisions, and nothing else."""
    return io.digest(
        dict(
            version=ORCHESTRATION_VERSION,
            sources=[[r["source_id"], r["source_sha256"], r["group"]] for r in records],
            module_weight=module.weight,
            region=region.to_list(),
            science=qc.settings.science(),
            working=working,
            code=_code_hashes(),
        )
    )


def result_path(cache, settings):
    return Path(cache) / ("result_" + io.digest(settings.to_dict()) + ".json")


def run_filter(qc, module, records, region, cache, stack):
    """Build the working DEMs and run the passes, or reuse the cached result."""
    done = result_path(cache, qc.settings)
    with FileLock(str(cache / ".lock")):
        if done.exists():
            result = io.read(done)
            final = io.read(result["manifest"])
            for item in final["masks"].values():
                if io.sha(item["mask"]) != item["mask_sha256"]:
                    raise RuntimeError(f"Cached QC mask changed: {item['mask']}")
            inventory = io.read(cache / "SOURCES.json")
            saved = {r["source_id"]: r for r in inventory["sources"]}
            for rec in records:
                rec["offset"] = saved[rec["source_id"]]["offset"]
                rec["active_blocks"] = saved[rec["source_id"]]["active_blocks"]
            logger.info(
                "Stack QC %s: reusing the cached filter result in %s",
                module.name,
                cache,
            )
            return result, True
        builder = WorkingSurface(
            stack,
            qc.interpolator_hooks,
            records,
            region,
            cache_root_for(module),
            workers=qc.workers,
        )
        logger.info("Stack QC %s: building the initial working DEM", module.name)
        baseline = builder(None, cache / "baseline")
        with rasterio.open(baseline) as ds:
            for rec in records:
                with rasterio.open(rec["path"]) as src:
                    rec["offset"] = list(grid_offset(src, ds))
                rec["active_blocks"] = builder.previous["source_blocks"][
                    rec["source_id"]
                ]
            inventory = dict(sources=records, grid=io.grid(ds))
        io.write(cache / "SOURCES.json", inventory)
        result = processing.process(
            inventory, qc.settings, cache, builder, workers=qc.workers
        )
    return result, False


def _accepted_stream(rec, masks, target):
    """Accepted measurements, aggregated from the QC grid to the DEM grid."""
    from .resampling import accepted_stream as aggregate

    mask = masks[rec["source_id"]]["mask"]
    if target is None:
        return masked_stream(rec["path"], mask)
    return aggregate(
        rec["path"],
        mask,
        target,
        active_blocks=rec.get("active_blocks"),
        source_offset=rec.get("offset", (0, 0)),
    )


def write_accepted(rec, masks, target, signature, path):
    """Write one source's accepted measurements as a DEM-grid side-stack."""
    from globato.hooks.sinks.side_stack import SIDESTACK_CACHE_VERSION
    from globato.hooks.transforms.point_pixels import (
        FUSION_BAND_MAP,
        FUSION_BANDS,
        FUSION_STATE_VERSION,
    )

    path = Path(path)
    if path.is_file():
        with rasterio.open(path) as ds:
            if ds.tags().get("GLOBATO_STACK_QC_SIGNATURE") == signature:
                return path
    stream = _accepted_stream(rec, masks, target)
    profile = next(stream)
    creation = {
        k: profile[k]
        for k in ("driver", "width", "height", "crs", "transform")
        if k in profile
    }
    creation.update(
        driver="GTiff",
        count=len(FUSION_BANDS),
        dtype="float64",
        nodata=None,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        bigtiff="YES",
        compress="lzw",
        predictor=3,
        sparse_ok=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".partial.tif")
    temporary.unlink(missing_ok=True)
    points = 0
    try:
        with rasterio.open(temporary, "w", **creation) as dst:
            for name, index in FUSION_BAND_MAP.items():
                dst.set_band_description(index, name)
            for window, _, data, *_ in stream:
                dst.write(data, window=window)
                points += int(data[FUSION_BAND_MAP["count"] - 1].sum())
            original = rec["source_id"].rsplit("|sidestack=", 1)[-1]
            dst.update_tags(
                GLOBATO_DATATYPE=SIDESTACK_DATATYPE,
                GLOBATO_FUSION_VERSION=str(FUSION_STATE_VERSION),
                GLOBATO_SIDESTACK_VERSION=str(SIDESTACK_CACHE_VERSION),
                GLOBATO_POINT_COUNT=str(points),
                GLOBATO_STACK_QC_SIGNATURE=signature,
                GLOBATO_STACK_QC_SOURCE=rec["source_id"],
                **{SIDESTACK_HASH_TAG: io.digest([original, signature])},
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def prepare(qc, module, entries, *, threads):
    """Filter one module's selected entries in place and return them.

    At the manifest stage (``multi_stack`` recipes) each filtered entry gets
    its accepted measurements as a FusionState stream on the DEM grid, which
    the normal stream stage hands to ``multi_stack``. At the collection stage
    (``side_stack`` + ``stack_combine`` recipes) the entry's ``side-stack``
    artifact is pointed at the accepted side-stack instead.
    """
    from .resampling import nested_region, output_grid

    manifest = qc.stage == "manifest"
    if not entries:
        logger.info("Stack QC %s: no selected sources; nothing to filter", module.name)
        return entries
    if not manifest and any(
        "multi_stack" in (e.get("artifacts") or {}) for _, e in entries
    ):
        raise ValueError(
            "stack_qc at the collection stage filters side-stacks before "
            "stack_combine, but this recipe already stacked the sources with "
            "multi_stack. Use stage: manifest with multi_stack."
        )
    region = module.region
    if region is None:
        raise ValueError(f"Stack QC {module.name}: module has no region")
    dem_stack = qc.stack_hook(qc.res)
    target = output_grid(dem_stack, region)
    for i, (_, entry) in enumerate(entries):
        entry[INDEX] = i
    started = time.monotonic()
    if qc.qc_increment is not None:
        qc_stack = qc.stack_hook(qc.qc_increment)
        qc_region, _ = nested_region(region, qc_stack, target)
        prepared = grid_sources(
            qc,
            module,
            entries,
            qc_region,
            qc_stack.res,
            threads,
            collected=not manifest,
        )
    elif manifest:
        # Nothing is gridded yet: grid on the DEM grid itself.
        qc_stack, qc_region = dem_stack, region
        prepared = grid_sources(
            qc, module, entries, qc_region, qc_stack.res, threads, collected=False
        )
    else:
        # The module's own DEM-grid side-stacks are the QC grids.
        qc_stack, qc_region, prepared = dem_stack, region, entries
    root = Path(qc.cache_dir or Path(module._outdir) / "stack_qc").resolve()
    root.mkdir(parents=True, exist_ok=True)
    records, _ = collect_sources(module, prepared, root)
    for _, entry in entries:
        entry.pop(INDEX, None)
    if not records:
        logger.info("Stack QC %s: no measurements in this region", module.name)
        return entries
    working = qc.working_identity(qc_stack)
    cache = root / cache_key(qc, module, records, qc_region, working)
    cache.mkdir(exist_ok=True)
    io.write(
        cache / "HOOK.json",
        dict(
            module=module.name,
            module_weight=module.weight,
            region=qc_region.to_list(),
            region_srs=qc_region.srs,
            qc_res=qc_stack.res,
            dem_res=dem_stack.res,
            crs=qc.crs,
            settings=qc.settings.to_dict(),
            working=working,
        ),
    )
    gridded = time.monotonic()
    result, reused = run_filter(qc, module, records, qc_region, cache, qc_stack)
    filtered = time.monotonic()
    masks = io.read(result["manifest"])["masks"]
    signature = result["signature"]
    tag = io.digest(
        dict(
            settings=qc.settings.to_dict(),
            dem=list(target["transform"])[:6],
            shape=[target["height"], target["width"]],
        )
    )[:16]
    receipt = cache / f"review_{tag}.json"
    io.write(
        receipt,
        dict(
            cache=str(cache),
            result=result,
            module=module.name,
            region=qc_region.to_list(),
            region_srs=qc_region.srs,
            stack=dict(
                res=qc_stack.res,
                crs=qc_stack.crs,
                strategy=qc_stack.strategy,
                weight_threshold=qc_stack.weight_threshold,
            ),
            dem_res=dem_stack.res,
            reused=reused,
        ),
    )
    stream_target = target if qc_stack is not dem_stack else None
    by_index = {r["index"]: r for r in records}
    written = 0
    with FileLock(str(cache / ".accepted.lock")):
        for i, (_, entry) in enumerate(entries):
            rec = by_index.get(i)
            artifacts = entry.setdefault("artifacts", {})
            if rec is None:
                if not manifest and artifacts.get("side-stack"):
                    # No measurements on the QC grid, so none on the DEM grid.
                    artifacts["stack_qc_original_side_stack"] = artifacts.pop(
                        "side-stack"
                    )
                continue
            accepted = write_accepted(
                rec,
                masks,
                stream_target,
                signature,
                cache / f"accepted_{tag}" / (io.digest(rec["source_id"])[:24] + ".tif"),
            )
            written += 1
            artifacts["stack_qc"] = str(receipt)
            artifacts["stack_qc_mask"] = masks[rec["source_id"]]["mask"]
            entry["_stack_qc_signature"] = signature
            if manifest:
                _attach_accepted(entry, accepted, rec, signature)
            else:
                if artifacts.get("side-stack"):
                    artifacts["stack_qc_original_side_stack"] = artifacts["side-stack"]
                artifacts["side-stack"] = str(accepted)
    logger.info(
        "Stack QC %s: %s; %d passes; %s source cells rejected, %s restored; "
        "%d accepted sources (QC grid %s, DEM grid %s); "
        "QC grids %.1f min, filter %.1f min, accepted side-stacks %.1f min",
        module.name,
        "cached result reused" if reused else "filtered",
        result["passes_completed"],
        f"{result['rejected_source_cells']:,}",
        f"{result['restored_source_cells']:,}",
        written,
        qc_stack.res,
        dem_stack.res,
        (gridded - started) / 60,
        (filtered - gridded) / 60,
        (time.monotonic() - filtered) / 60,
    )
    return entries


def _attach_accepted(entry, accepted, rec, signature):
    """Give a manifest-stage entry its accepted measurements as a FusionState stream.

    The stream stage then skips reading the source (an entry that already has a
    stream is not read again; point transforms skip raster streams) and
    multi_stack stacks it. The contribution id keeps the filtered contribution
    distinct from an unfiltered one in multi_stack's registered datasets. Any
    side-stack lookup made for the unfiltered source is dropped, and the QC
    result joins the entry's processing record, so a side-stack cache can never
    mix filtered and unfiltered data.
    """
    from fetchez.entry import set_contribution_id
    from globato.hooks.sinks.side_stack import read_fusion_state_stream

    for key in [k for k in entry if k.startswith("_sidestack")]:
        entry.pop(key)
    entry["artifacts"].pop("side-stack", None)
    entry["artifacts"]["stack_qc_accepted"] = str(accepted)
    entry["stream"] = read_fusion_state_stream(str(accepted))
    entry["stream_type"] = "raster-stream"
    entry["data_type"] = "fusion-state"
    entry["src_srs"] = rec["src_srs"]
    set_contribution_id(entry, "stack_qc", signature[:16])
    entry["hooks"] = list(entry.get("hooks", []) or []) + [
        dict(name="stack_qc_result", signature=signature)
    ]
