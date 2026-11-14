"""Prepare a collected Stack QC context before native final assembly.

Inventory per-owner source grids, evaluate targets with optional fixed reference
sources using the existing WorkingSurface/engines, then publish per-source
accepted statistics on the final DEM grid. Joint grids bind exact contents and
source roles/weights/invocation IDs in the decision key. Legacy explicit
single-module recipes retain their original source selection and reduction.

References contribute evidence but never receive rejection masks. Fine-grid
reference statistics use the same native additive aggregation with no mask.
Original source grids and entry ownership are retained for provenance.
"""

import inspect
import logging
import time
from pathlib import Path

import rasterio
from filelock import FileLock

from pyproj import CRS

from fetchez.entry import dataset_id

from globato.hooks.sinks.side_stack import SIDESTACK_DATATYPE, SIDESTACK_HASH_TAG
from . import io, processing
from .context import WorkingSurface, cache_root_for
from .grid import grid_offset
from .native import masked_stream

logger = logging.getLogger(__name__)

# Bump when the orchestration changes in a way that changes decisions.
ORCHESTRATION_VERSION = "stack_qc_hook_v1"
INDEX = "_stack_qc_index"


def _on_grid(ds, expected):
    """Whether a side-stack is on the expected QC grid (horizontal CRS only)."""
    if (ds.width, ds.height) != (expected["width"], expected["height"]):
        return False
    if any(
        abs(x - y) > 1e-9 * max(1.0, abs(y))
        for x, y in zip(tuple(ds.transform)[:6], tuple(expected["transform"])[:6])
    ):
        return False
    return (
        CRS.from_user_input(ds.crs.to_wkt())
        .to_2d()
        .equals(
            CRS.from_user_input(expected["crs"].to_wkt()).to_2d(),
            ignore_axis_order=True,
        )
    )


def collect_sources(
    module, entries, root, artifact, expected, *, content_identity=False
):
    """Inventory measured QC-grid sources; entries with no data come back separately."""
    records, empty = [], []
    seen = set()
    for owner, entry in entries:
        raw = (entry.get("artifacts") or {}).get(artifact)
        if not raw or not Path(raw).is_file():
            empty.append(entry[INDEX])
            continue
        path = Path(raw).resolve(strict=True)
        with rasterio.open(path) as ds:
            tags = ds.tags()
            if ds.count != 7 or tags.get("GLOBATO_DATATYPE") != SIDESTACK_DATATYPE:
                raise ValueError("Stack QC requires seven-band source side-stacks")
            if not _on_grid(ds, expected):
                raise ValueError(
                    f"Stack QC {module.name}: side-stack {path.name} "
                    f"({ds.width}x{ds.height}, {abs(ds.transform.a)!r} per cell) "
                    f"is not on the QC grid ({expected['width']}x"
                    f"{expected['height']}, {abs(expected['transform'].a)!r} per "
                    "cell). With qc_res, use stack_qc_grid with the same res, "
                    "qc_res and crs as stack_qc."
                )
            geom = io.grid(ds)
            if tags.get("GLOBATO_POINT_COUNT") == "0":
                empty.append(entry[INDEX])
                continue
        sid = dataset_id(entry) + "|sidestack=" + tags[SIDESTACK_HASH_TAG]
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
        if content_identity:
            sid += "|qc_content=" + source_sha
        if sid in seen:
            raise ValueError(f"Duplicate measured source identity: {sid}")
        seen.add(sid)
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
                weight=owner.weight,
                module=owner.name,
                module_id=owner.module_id()
                if callable(getattr(owner, "module_id", None))
                else owner.name,
                **(
                    dict(
                        instance_id=getattr(owner, "instance_id", None),
                        module_datatype=getattr(owner, "datatype", None),
                        module_survey_id=getattr(owner, "survey_id", None),
                    )
                    if content_identity
                    else {}
                ),
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

    files = {
        str(p.relative_to(here)): io.sha(p)
        for p in sorted(here.rglob("*.py"))
        if "review" not in p.relative_to(here).parts
    }
    for mod in (hook_module, binary_cudem, multi_stack):
        path = Path(inspect.getfile(mod))
        files[path.name] = io.sha(path)
    return files


def cache_key(qc, module, records, region, working, *, references=(), joint=False):
    """Everything that can change the decisions, and nothing else."""
    return io.digest(
        dict(
            version=ORCHESTRATION_VERSION,
            sources=[[r["source_id"], r["source_sha256"], r["group"]] for r in records],
            module_weight=module.weight
            if not joint
            else sorted(
                [r["source_id"], r["module_id"], r["weight"]]
                for r in [*records, *references]
            ),
            **(
                dict(
                    scope="joint_v1",
                    references=[
                        [
                            r["source_id"],
                            r["source_sha256"],
                            r["group"],
                            r["module_id"],
                            r["weight"],
                        ]
                        for r in references
                    ],
                )
                if joint
                else {}
            ),
            region=region.to_list(),
            science=qc.settings.science(),
            working=working,
            code=_code_hashes(),
        )
    )


def result_path(cache, settings):
    return Path(cache) / ("result_" + io.digest(settings.to_dict()) + ".json")


def run_filter(
    qc, module, records, region, cache, stack, *, profile=None, references=()
):
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
            saved = {
                r["source_id"]: r
                for r in [
                    *inventory["sources"],
                    *inventory.get("reference_sources", []),
                ]
            }
            for rec in [*records, *references]:
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
            sorted([*records, *references], key=lambda r: r["source_id"]),
            region,
            cache_root_for(module),
            workers=qc.workers,
        )
        logger.info("Stack QC %s: building the initial working DEM", module.name)
        baseline = builder(None, cache / "baseline")
        with rasterio.open(baseline) as ds:
            for rec in [*records, *references]:
                with rasterio.open(rec["path"]) as src:
                    rec["offset"] = list(grid_offset(src, ds))
                rec["active_blocks"] = builder.previous["source_blocks"][
                    rec["source_id"]
                ]
            inventory = dict(sources=records, grid=io.grid(ds))
            if references:
                inventory["reference_sources"] = list(references)
        io.write(cache / "SOURCES.json", inventory)
        result = processing.process(
            inventory, qc.settings, cache, builder, workers=qc.workers
        )
        if profile is not None:
            profile.extend(builder.profile_events)
    return result, False


def _accepted_stream(rec, masks, target):
    """Accepted measurements, aggregated from the QC grid to the DEM grid."""
    from .resampling import accepted_stream as aggregate

    mask = masks[rec["source_id"]]["mask"]
    if target is None:
        return masked_stream(
            rec["path"],
            mask,
            active_blocks=rec.get("active_blocks"),
            source_offset=rec.get("offset", (0, 0)),
        )
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


def prepare(qc, module, entries, *, references=(), joint=False):
    """Filter collected targets in place, with optional fixed references.

    Each entry with measurements gets its accepted measurements as a DEM-grid
    side-stack in ``qc.output_artifact``; entries without any get none.
    """
    from .resampling import nested_region, output_grid

    participants = [*entries, *references]
    label = "joint" if joint else module.name
    if not entries and not references:
        logger.info("Stack QC %s: no selected sources; nothing to filter", module.name)
        return entries
    if any("multi_stack" in (e.get("artifacts") or {}) for _, e in participants):
        raise ValueError(
            "stack_qc filters side-stacks before stack_combine, but this recipe "
            "already stacked the sources with multi_stack. Stack with "
            "side_stack + stack_combine."
        )
    region = module.region
    if region is None:
        raise ValueError(f"Stack QC {module.name}: module has no region")
    dem_stack = qc.stack_hook(qc.res)
    target = output_grid(dem_stack, region)
    for i, (_, entry) in enumerate(entries):
        entry[INDEX] = i
    for i, (_, entry) in enumerate(references):
        entry[INDEX] = i
    started = time.perf_counter()
    if qc.qc_increment is not None:
        qc_stack = qc.stack_hook(qc.qc_increment)
        qc_region, qc_grid = nested_region(region, qc_stack, target)
    else:
        qc_stack, qc_region, qc_grid = dem_stack, region, target
    root = Path(qc.cache_dir or Path(module._outdir) / "stack_qc").resolve()
    root.mkdir(parents=True, exist_ok=True)
    artifact = (
        qc.grid_artifact
        if qc.qc_increment is not None
        or all(
            qc.grid_artifact in (entry.get("artifacts") or {})
            for _, entry in participants
        )
        else qc.source_artifact
    )
    if joint and not all(
        qc.grid_artifact in (entry.get("artifacts") or {}) for _, entry in participants
    ):
        raise ValueError(
            "Joint stack_qc requires matching stack_qc_grid for every target/reference, even when qc_res is omitted"
        )
    if not any(artifact in (e.get("artifacts") or {}) for _, e in participants):
        raise ValueError(
            f"Stack QC {module.name}: no entry has a {artifact!r} artifact. "
            + (
                "With qc_res, add stack_qc_grid (same modules, res, qc_res and "
                "crs) before side_stack."
                if qc.qc_increment is not None
                else "Side-stack the module with side_stack first."
            )
        )
    records, _ = collect_sources(
        module, entries, root, artifact, qc_grid, content_identity=joint
    )
    ref_records, _ = collect_sources(
        module, references, root, artifact, qc_grid, content_identity=joint
    )
    for rec in ref_records:
        rec["role"] = "reference"
    if {r["source_id"] for r in records} & {r["source_id"] for r in ref_records}:
        raise ValueError("The same source contribution cannot be target and reference")
    # Every non-empty participant must have passed the configured grid producer.
    for _, entry in participants:
        if entry.get("status") not in {"empty", "skipped"} and artifact not in (
            entry.get("artifacts") or {}
        ):
            raise ValueError(
                "stack_qc: every target/reference needs a source grid; use matching stack_qc_grid modules/add_ref_modules"
            )
    for _, entry in participants:
        entry.pop(INDEX, None)
    if not records and not ref_records:
        logger.info("Stack QC %s: no measurements in this region", module.name)
        return entries
    working = qc.working_identity(qc_stack)
    cache = root / cache_key(
        qc, module, records, qc_region, working, references=ref_records, joint=joint
    )
    cache.mkdir(exist_ok=True)
    io.write(
        cache / "HOOK.json",
        dict(
            module=label,
            module_weight=module.weight,
            filtered_modules=qc.modules,
            add_ref_modules=qc.add_ref_modules,
            scope="joint" if joint else "single_module",
            region=qc_region.to_list(),
            region_srs=qc_region.srs,
            qc_res=qc_stack.res,
            dem_res=dem_stack.res,
            crs=qc.crs,
            settings=qc.settings.to_dict(),
            working=working,
        ),
    )
    gridded = time.perf_counter()
    working_events = []
    result, reused = run_filter(
        qc,
        module,
        records,
        qc_region,
        cache,
        qc_stack,
        profile=working_events,
        references=ref_records,
    )
    filtered = time.perf_counter()
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
    profile_path = cache / f"profile_{tag}.json"
    io.write(
        receipt,
        dict(
            cache=str(cache),
            result=result,
            module=label,
            scope="joint" if joint else "single_module",
            contributions=str(cache / f"contributions_{tag}.json") if joint else None,
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
            viz=qc.viz,
            product_spec="stack_qc_viz/1" if qc.viz else None,
            profile=str(profile_path),
        ),
    )
    stream_target = target if qc_stack is not dem_stack else None
    reference_outputs = {}
    if references:
        with FileLock(str(cache / ".references.lock")):
            for rec in ref_records:
                owner, entry = references[rec["index"]]
                if stream_target is None:
                    published = rec["path"]
                else:
                    published = str(
                        write_accepted(
                            rec,
                            {rec["source_id"]: dict(mask=None)},
                            stream_target,
                            io.digest(
                                [
                                    rec["source_sha256"],
                                    list(target["transform"])[:6],
                                    target["height"],
                                    target["width"],
                                ]
                            ),
                            cache
                            / f"references_{tag}"
                            / (io.digest(rec["source_id"])[:24] + ".tif"),
                        )
                    )
                entry.setdefault("artifacts", {})[qc.output_artifact] = published
                reference_outputs[rec["source_id"]] = published
    by_index = {r["index"]: r for r in records}
    written = 0
    with FileLock(str(cache / ".accepted.lock")):
        for i, (_, entry) in enumerate(entries):
            rec = by_index.get(i)
            if rec is None:
                # No measurements on the QC grid, so none on the DEM grid.
                continue
            artifacts = entry.setdefault("artifacts", {})
            accepted = write_accepted(
                rec,
                masks,
                stream_target,
                signature,
                cache / f"accepted_{tag}" / (io.digest(rec["source_id"])[:24] + ".tif"),
            )
            written += 1
            artifacts["stack_qc"] = str(receipt)
            artifacts["stack_qc_profile"] = str(profile_path)
            artifacts["stack_qc_mask"] = masks[rec["source_id"]]["mask"]
            entry["_stack_qc_signature"] = signature
            artifacts[qc.output_artifact] = str(accepted)
    from .profile import SCHEMA

    finished = time.perf_counter()
    io.write(
        profile_path,
        dict(
            schema=SCHEMA,
            clock="perf_counter",
            module=label,
            settings=qc.settings.to_dict(),
            working=working,
            scope="joint" if joint else "single_module",
            filtered_modules=sorted({r["module"] for r in records}),
            requested_reference_modules=qc.add_ref_modules,
            reference_modules=sorted({r["module"] for r in ref_records}),
            reference_inputs=[
                dict(
                    source_id=r["source_id"],
                    sha256=r["source_sha256"],
                    module=r["module"],
                    module_id=r["module_id"],
                    weight=r["weight"],
                    qc_grid=references[r["index"]][1].get("_stack_qc_grid_profile"),
                )
                for r in ref_records
            ],
            region=qc_region.to_list(),
            output_grid=dict(
                shape=[target["height"], target["width"]],
                transform=list(target["transform"])[:6],
                crs=qc.crs,
            ),
            cache=dict(key=cache.name, filter_hit=reused),
            phases_seconds=dict(
                inventory_and_identity=gridded - started,
                filter_or_cache_validation=filtered - gridded,
                accepted_publication=finished - filtered,
            ),
            total_seconds=finished - started,
            # Engine totals include their passes and working-surface calls.
            # On a hit these are historical, never current invocation costs.
            engine=dict(
                timing_seconds=result.get("timing_seconds", {}),
                origin="cached_execution" if reused else "current_execution",
            ),
            working_surfaces=working_events,
            inputs=[
                dict(
                    source_id=r["source_id"],
                    module=r["module"],
                    module_id=r["module_id"],
                    sha256=r["source_sha256"],
                    fingerprint=r["fingerprint"],
                    grid=r["grid"],
                    weight=r["weight"],
                    qc_grid=entries[r["index"]][1].get("_stack_qc_grid_profile"),
                )
                for r in records
            ],
            outputs=dict(
                manifest=result["manifest"],
                signature=signature,
                accepted=[
                    entries[r["index"]][1]["artifacts"][qc.output_artifact]
                    for r in records
                ],
            ),
            timing_semantics=(
                "Only phases_seconds partition total_seconds. Engine and working "
                "surface times are nested diagnostics; QC-grid producer times "
                "precede prepare and may overlap other stream-stage consumers."
            ),
        ),
    )
    if joint:
        contributions = []
        for role, records_, entries_, outputs in (
            (
                "filter",
                records,
                entries,
                {
                    r["source_id"]: entries[r["index"]][1]["artifacts"][
                        qc.output_artifact
                    ]
                    for r in records
                },
            ),
            ("reference", ref_records, references, reference_outputs),
        ):
            for rec in records_:
                contributions.append(
                    dict(
                        source_id=rec["source_id"],
                        module=rec["module"],
                        module_id=rec["module_id"],
                        instance_id=rec.get("instance_id"),
                        module_datatype=rec.get("module_datatype"),
                        module_survey_id=rec.get("module_survey_id"),
                        role=role,
                        group=rec["group"],
                        weight=rec["weight"],
                        source_sha256=rec["source_sha256"],
                        source_grid=rec["path"],
                        output=outputs[rec["source_id"]],
                        mask=masks[rec["source_id"]]["mask"]
                        if role == "filter"
                        else None,
                    )
                )
        contribution_path = cache / f"contributions_{tag}.json"
        io.write(
            contribution_path,
            dict(schema="stack_qc_contributions/1", sources=contributions),
        )
        for _, entry in participants:
            entry.setdefault("artifacts", {})["stack_qc_contributions"] = str(
                contribution_path
            )
    logger.info(
        "Stack QC %s: %s; %d passes; %s source cells rejected, %s restored; "
        "%d accepted sources (QC grid %s, DEM grid %s); "
        "inventory %.1f min, filter %.1f min, accepted side-stacks %.1f min",
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
        (time.perf_counter() - filtered) / 60,
    )
    return entries
