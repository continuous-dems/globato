#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pyogrio
import geopandas as gpd
import shapely
from shapely.geometry import MultiPolygon
from ..._io import digest as sha256


from .coastal_plan import make_plan
from .input_resolver import (
    normalize_existing_mask,
    resolve_area,
)
from .coast_refine_core import (
    extract_positive_evidence,
    classify_connectivity,
    screen_hydroflat_transition_fringe,
    local_terrain_support_screen,
    screen_nearshore_transition_artifacts,
    clean_accepted_by_source_resolution,
    annotate_osm_structure_support,
    write_dropin_mask,
)


def _write_review_gpkg(
    path,
    *,
    mask,
    plan,
    evidence,
    terrain_candidates,
    flat_review,
    nearshore_review,
    accepted,
    detached,
    terrain_union,
    osm_evidence=None,
    osm_supported=None,
    hydroflat_seeds=None,
    hydroflat_rejected=None,
    hydroflat_osm_protected=None,
):
    path = Path(path)
    if path.exists():
        path.unlink()
    layers = []
    baseline = gpd.read_file(mask, layer="mask")
    layers.append(("original_mask", baseline))
    for layer in ["coastline", "landward_context", "seaward_search", "fetch_boxes"]:
        layers.append((layer, gpd.read_file(plan, layer=layer)))
    if not evidence.empty:
        ctx = evidence[evidence["kind"] == "land_context_positive"].copy()
        sea = evidence[evidence["kind"] == "seaward_positive"].copy()
        if not ctx.empty:
            layers.append(("tnm_land_context", ctx))
        if not sea.empty:
            layers.append(("tnm_seaward_positive", sea))
    if hydroflat_seeds is not None and not hydroflat_seeds.empty:
        layers.append(("hydroflat_water_seeds", hydroflat_seeds))
    if hydroflat_rejected is not None and not hydroflat_rejected.empty:
        layers.append(("rejected_hydroflat_transition", hydroflat_rejected))
    if hydroflat_osm_protected is not None and not hydroflat_osm_protected.empty:
        layers.append(("hydroflat_osm_protected", hydroflat_osm_protected))
    if not terrain_candidates.empty:
        layers.append(("local_terrain_support", terrain_candidates))
    if not flat_review.empty:
        layers.append(("unsupported_positive_review", flat_review))
    if nearshore_review is not None and not nearshore_review.empty:
        layers.append(("nearshore_transition_review", nearshore_review))
    if osm_evidence is not None and not osm_evidence.empty:
        layers.append(("optional_osm_coastal_structures", osm_evidence))
    if not accepted.empty:
        layers.append(("accepted_land_additions", accepted))
    if osm_supported is not None and not osm_supported.empty:
        layers.append(("accepted_osm_structure_supported", osm_supported))
    if not detached.empty:
        layers.append(("detached_review", detached))
    if terrain_union is not None and not terrain_union.is_empty:
        layers.append(
            (
                "terrain_accepted_land_union",
                gpd.GeoDataFrame(
                    {"status": ["terrain_accepted"]},
                    geometry=[terrain_union],
                    crs=evidence.crs,
                ),
            )
        )
    for name, g in layers:
        if g is None or g.empty:
            continue
        g.to_file(path, layer=name, driver="GPKG", engine="pyogrio", index=False)
    available = set(pyogrio.list_layers(path)[:, 0])
    for name, g in layers:
        if g is None or g.empty:
            continue
        if name not in available:
            raise RuntimeError(f"Review layer missing after write: {name}")
        info = pyogrio.read_info(path, layer=name)
        if info["features"] != len(g):
            raise RuntimeError(f"Review feature count changed after write: {name}")


def _schema_summary(mask, layer="mask"):
    """Summarize a vector layer without loading feature geometries."""
    info = pyogrio.read_info(mask, layer=layer)
    properties = {
        name: ogr_type
        for name, ogr_type in zip(info["fields"], info["ogr_types"], strict=True)
    }
    return {
        "layers": pyogrio.list_layers(mask)[:, 0].tolist(),
        "layer": layer,
        "driver": info["driver"],
        "schema": {"geometry": info["geometry_type"], "properties": properties},
        "crs_wkt": info["crs"],  # Pyogrio returns a CRS string, not necessarily WKT.
        "feature_count": info["features"],
    }


def _fmt_bbox(b):
    return "/".join(f"{float(v):.8f}".rstrip("0").rstrip(".") for v in b)


def _geometry_sha256(path, layer):
    """Stable geometry-only fingerprint, independent of GeoPackage byte metadata."""
    g = gpd.read_file(path, layer=layer)
    if g.empty:
        return hashlib.sha256(b"EMPTY").hexdigest()
    geom = shapely.make_valid(shapely.union_all(g.geometry.values))
    geom = shapely.normalize(geom)
    payload = shapely.to_wkb(geom, hex=False, byte_order=1, output_dimension=2)
    return hashlib.sha256(payload).hexdigest()


def run(a):
    local_relief_window_m = 15.0
    min_local_relief_m = float(getattr(a, "min_local_relief_m", 0.10))
    max_flat_bridge_m = 5.0
    cleanup_min_close_m = 1.0
    cleanup_close_fraction = 0.35
    cleanup_max_close_m = 3.0
    cleanup_min_hole_m2 = 9.0
    cleanup_hole_cells = 2.0
    residual_sliver_max_area_m2 = 0.0
    output = Path(a.output).expanduser().resolve()
    if output.suffix.lower() != ".gpkg":
        raise ValueError("--output must be a .gpkg GeoPackage path")
    work = (
        Path(a.workdir).expanduser().resolve()
        if a.workdir
        else output.parent / (output.stem + "_coast_refine_work")
    )
    cache = Path(a.cache).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print("COAST REFINE scientific core — raster coastal validity mask", flush=True)
    print("=" * 72, flush=True)
    print(f"Output: {output}", flush=True)
    print(f"Work:   {work}", flush=True)
    print(f"Cache:  {cache}", flush=True)
    print("INPUT 1/4: resolving requested Coast Refine ROI...", flush=True)

    roi_gdf, roi_meta = resolve_area(a.area, workdir=work)
    roi_path = roi_meta.get("roi_gpkg") if roi_meta else None
    if roi_meta:
        print(f"  ROI mode:   {roi_meta.get('mode')}", flush=True)
        print(f"  ROI source: {roi_meta.get('source')}", flush=True)
        if roi_meta.get("layer"):
            print(f"  ROI layer:  {roi_meta.get('layer')}", flush=True)
        print(
            f"  exact ROI bbox (WGS84): {_fmt_bbox((roi_meta['bounds_wgs84'][0], roi_meta['bounds_wgs84'][2], roi_meta['bounds_wgs84'][1], roi_meta['bounds_wgs84'][3]))}",
            flush=True,
        )
    else:
        print(
            "  no explicit ROI supplied; coastal plan will be derived from the input mask",
            flush=True,
        )

    mask_arg = getattr(a, "mask", None)
    if not mask_arg:
        raise ValueError(
            "Coast Refine v6.6.1 requires --mask. Generate a baseline separately (for example with "
            "OSM Mask Helper v23.2.1), then pass its water-mask GPKG/GeoJSON here. "
            "The optional run_osm_then_refine.sh wrapper can chain the two tools without merging them."
        )
    print("INPUT 2/4: using supplied coastline/water mask...", flush=True)
    print(
        f"  mask source: {Path(mask_arg).expanduser().resolve(strict=False)}",
        flush=True,
    )
    mask, mask_meta = normalize_existing_mask(
        mask_arg, work, mask_layer=getattr(a, "mask_layer", None)
    )
    print(f"  canonical mask: {mask}", flush=True)
    print(f"  mask mode: {mask_meta.get('mode')}", flush=True)
    mask = Path(mask).resolve()
    if output == mask:
        raise ValueError("Output must be a new path; the input mask is never modified")

    plan = work / "coastal_plan.gpkg"

    input_schema = _schema_summary(mask)
    print(
        f"  mask schema: layer={input_schema.get('layer')} driver={input_schema.get('driver')} features={input_schema.get('feature_count')}",
        flush=True,
    )
    (work / "input_schema.json").write_text(json.dumps(input_schema, indent=2) + "\n")
    osm_evidence = gpd.GeoDataFrame(
        columns=["class", "group", "geometry"], geometry="geometry", crs="EPSG:4326"
    )
    osm_evidence_meta = None

    (work / "input_resolution.json").write_text(
        json.dumps(
            {
                "mask": mask_meta,
                "roi": roi_meta,
                "optional_osm_evidence": osm_evidence_meta,
            },
            indent=2,
        )
        + "\n"
    )

    mask_geometry_hash = _geometry_sha256(mask, "mask")
    roi_geometry_hash = _geometry_sha256(roi_path, "roi") if roi_path else None

    print(
        "INPUT 3/4: building/loading narrow coastal ribbon and fetch-box plan...",
        flush=True,
    )
    plan_meta = make_plan(
        mask,
        plan,
        tiles=roi_path,
        mask_layer="mask",
        tile_layer="roi",
        inland_m=a.inland_m,
        seaward_m=250.0,
        segment_m=5000.0,
        halo_m=100.0,
        full_search=True,
    )
    plan_meta["mask_geometry_sha256"] = mask_geometry_hash
    plan_meta["roi_geometry_sha256"] = roi_geometry_hash
    plan.with_suffix(".json").write_text(json.dumps(plan_meta, indent=2) + "\n")

    boxes = plan_meta["boxes"]
    if not boxes:
        raise RuntimeError("No coastal fetch boxes")
    print(f"  coastal fetch boxes: {len(boxes)}", flush=True)
    print(f"  plan: {plan}", flush=True)

    evidence_raster = a.evidence
    raster = Path(evidence_raster).expanduser().resolve(strict=True)
    print(f"INPUT 4/4: using supplied effective elevation band 1: {raster}")
    roi_ll = gpd.read_file(plan, layer="roi").to_crs("EPSG:4326")
    w, s, e, n = roi_ll.total_bounds
    evidence, raster_report = extract_positive_evidence(
        raster_path=raster,
        planning_gpkg=plan,
        bbox_wgs84=(w, e, s, n),
        accepted_geometry_wkt=None,
        threshold=a.threshold,
        min_seaward_cells=4,
        box_id="common_raw",
        product="effective_raster_terrain",
        source_title="Supplied effective terrain raster band 1; source attribution retained in RAW identity",
    )
    if raster_report.get("status") != "OK":
        raise RuntimeError(f"Common raster evidence failed: {raster_report}")
    (work / "common_raster_evidence.json").write_text(
        json.dumps(raster_report, indent=2) + "\n"
    )
    (work / "cache_reuse_receipt.json").write_text(
        json.dumps(
            {
                "mode": "common_raw_raster",
                "raster": str(raster),
                "discovery_calls": 0,
                "native_source_attribution": False,
            },
            indent=2,
        )
        + "\n"
    )
    # Record positive candidates as well as final acceptances for adaptive
    # edge checks; rejected/detached candidates must not disappear from them.
    common_evidence_path = work / "common_raw_evidence.gpkg"
    if common_evidence_path.exists():
        common_evidence_path.unlink()
    if not evidence.empty:
        evidence.to_file(common_evidence_path, layer="evidence", driver="GPKG")
    metric_crs = plan_meta["metric_crs"]

    print(
        "FINALIZE 2/9: suppressing broad positive hydroflattened-water / smooth-TIN transition fringes..."
    )
    t0 = time.perf_counter()
    (
        screened_evidence,
        hydroflat_seeds,
        hydroflat_rejected,
        hydroflat_osm_protected,
        hydroflat_stats,
    ) = screen_hydroflat_transition_fringe(
        evidence,
        metric_crs=metric_crs,
        osm_evidence=osm_evidence,
        min_valid_cells=4,
    )
    print(
        f"  hydroflat screen complete in {time.perf_counter() - t0:.2f}s: "
        f"rows_with_seed={hydroflat_stats.get('rows_with_seed', 0)}, "
        f"seed_cells={hydroflat_stats.get('seed_cells', 0)}, "
        f"rejected_cells={hydroflat_stats.get('rejected_cells', 0)}, "
        f"retained_cells={hydroflat_stats.get('retained_cells', 0)}, "
        f"osm_protected_cells={hydroflat_stats.get('osm_protected_cells', 0)}"
    )

    print(
        "FINALIZE 3/9: deriving local terrain support from hydroflat-screened TNM evidence..."
    )
    t0 = time.perf_counter()
    terrain_candidates, flat_review = local_terrain_support_screen(
        screened_evidence,
        local_window_m=local_relief_window_m,
        min_local_relief_m=min_local_relief_m,
        max_flat_bridge_m=max_flat_bridge_m,
        min_valid_cells=4,
    )
    print(
        f"  local terrain screen complete in {time.perf_counter() - t0:.2f}s: "
        f"supported={len(terrain_candidates)}, unsupported_review={len(flat_review)}"
    )

    # Connectivity may propagate only through locally terrain-supported subgeometry,
    # never through the rejected hydroflat transition fringe or the remainder of a
    # broad flat positive/hydroflattened water component.
    context = screened_evidence[
        screened_evidence["kind"] == "land_context_positive"
    ].copy()
    print("FINALIZE 4/9: screening ambiguous nearshore hydroflat survivors...")
    t0 = time.perf_counter()
    terrain_candidates_before_nearshore = int(len(terrain_candidates))
    terrain_candidates, nearshore_review = screen_nearshore_transition_artifacts(
        terrain_candidates, mask, metric_crs=metric_crs
    )
    print(
        f"  nearshore screen complete in {time.perf_counter() - t0:.2f}s: kept={len(terrain_candidates)}, review={len(nearshore_review)}"
    )

    print(
        "FINALIZE 5/9: classifying land-connected locally supported terrain with STRtree..."
    )
    t0 = time.perf_counter()
    filtered = gpd.GeoDataFrame(
        __import__("pandas").concat([context, terrain_candidates], ignore_index=True),
        geometry="geometry",
        crs=screened_evidence.crs,
    )
    accepted, detached, _raw_union = classify_connectivity(filtered, metric_crs)
    accepted = clean_accepted_by_source_resolution(
        accepted,
        metric_crs=metric_crs,
        min_close_m=cleanup_min_close_m,
        close_fraction=cleanup_close_fraction,
        max_close_m=cleanup_max_close_m,
        min_hole_area_m2=cleanup_min_hole_m2,
        hole_cells=cleanup_hole_cells,
    )
    accepted, osm_supported = annotate_osm_structure_support(
        accepted, osm_evidence, metric_crs=metric_crs
    )
    protected_rows = [
        g
        for frame in (hydroflat_rejected, flat_review, nearshore_review, detached)
        for g in frame.geometry
        if g is not None and not g.is_empty
    ]
    protected_water = (
        shapely.union_all(protected_rows) if protected_rows else MultiPolygon([])
    )
    if not accepted.empty and not protected_water.is_empty:
        accepted.geometry = shapely.difference(
            accepted.geometry.values, protected_water
        )
        accepted = accepted[~accepted.geometry.is_empty].copy()
        osm_supported = osm_supported.copy()
        if not osm_supported.empty:
            osm_supported.geometry = shapely.difference(
                osm_supported.geometry.values, protected_water
            )
            osm_supported = osm_supported[~osm_supported.geometry.is_empty].copy()
    accepted_union = (
        shapely.union_all(accepted.geometry.values)
        if not accepted.empty
        else MultiPolygon([])
    )
    print(
        f"  connectivity + source-resolution cleanup complete in {time.perf_counter() - t0:.2f}s: "
        f"accepted={len(accepted)}, detached_review={len(detached)}"
    )
    # Never allow additions outside the requested seaward search zone or outside
    # the baseline ocean geometry.
    print(
        "FINALIZE 6/9: clipping accepted additions to the exact seaward search/ocean domain..."
    )
    baseline = gpd.read_file(mask, layer="mask")
    baseline = baseline.to_crs("EPSG:4326")
    ocean = shapely.union_all(baseline.geometry.values)
    seaward = shapely.union_all(
        gpd.read_file(plan, layer="seaward_search").to_crs("EPSG:4326").geometry.values
    )
    if accepted_union is not None and not accepted_union.is_empty:
        accepted_union = shapely.make_valid(
            accepted_union.intersection(seaward).intersection(ocean)
        )
    else:
        accepted_union = MultiPolygon([])
    terrain_union = accepted_union

    if output.exists():
        raise FileExistsError(
            f"Output exists: {output} (use a new scientific workspace)"
        )
    print(f"FINALIZE 7/9: writing terrain-refined drop-in mask: {output}")
    t0 = time.perf_counter()
    dropin = write_dropin_mask(mask, output, terrain_union, layer="mask")
    print(f"  terrain-refined mask written in {time.perf_counter() - t0:.2f}s")

    print("FINALIZE 8/9: preserving residual water; area alone cannot prove land...")
    t0 = time.perf_counter()
    # Area-only sliver removal was retired (its maximum area was always zero).
    # Preserve the historical receipt counters, not the dead mutation/review path.
    original_parts = len(shapely.get_parts(shapely.union_all(baseline.geometry)))
    refined_parts = len(
        shapely.get_parts(
            shapely.union_all(gpd.read_file(output, layer="mask").geometry)
        )
    )
    area_cleanup_stats = {
        "original_parts": original_parts,
        "refined_parts_before": refined_parts,
        "preexisting_small_parts_restored": 0,
        "small_created_parts_protected": 0,
        "small_created_area_protected_m2": 0.0,
        "small_created_parts_removed": 0,
        "small_created_area_removed_m2": 0.0,
        "refined_parts_after": refined_parts,
        "max_area_m2": 0.0,
    }
    sliver_area = float(area_cleanup_stats.get("small_created_area_removed_m2", 0.0))
    print(
        f"  area cleanup complete in {time.perf_counter() - t0:.2f}s: "
        f"parts_before={area_cleanup_stats['refined_parts_before']}, "
        f"removed={area_cleanup_stats['small_created_parts_removed']}, parts_after={area_cleanup_stats['refined_parts_after']}, "
        f"area={sliver_area:.3f} m2"
    )
    review = work / "coast_refine_review.gpkg"
    print(f"FINALIZE 9/9: writing review GeoPackage: {review}")
    t0 = time.perf_counter()
    _write_review_gpkg(
        review,
        mask=mask,
        plan=plan,
        evidence=evidence,
        terrain_candidates=terrain_candidates,
        flat_review=flat_review,
        nearshore_review=nearshore_review,
        accepted=accepted,
        detached=detached,
        terrain_union=terrain_union,
        osm_evidence=osm_evidence,
        osm_supported=osm_supported,
        hydroflat_seeds=hydroflat_seeds,
        hydroflat_rejected=hydroflat_rejected,
        hydroflat_osm_protected=hydroflat_osm_protected,
    )
    print(f"  review GeoPackage written in {time.perf_counter() - t0:.2f}s")

    # Area in the local metric CRS for a physically meaningful receipt.
    if terrain_union.is_empty:
        terrain_accepted_area = 0.0
    else:
        terrain_accepted_area = float(
            gpd.GeoSeries([terrain_union], crs="EPSG:4326")
            .to_crs(metric_crs)
            .area.iloc[0]
        )
    accepted_area = terrain_accepted_area  # no area-only sliver removals
    receipt = {
        "status": "COMPLETE",
        "review_only_water_preserved_through_cleanup": True,
        "evidence_mode": "common_raw_raster",
        "evidence_raster": str(Path(evidence_raster).resolve()),
        "individual_source_attribution_available": False,
        "input_mask": str(mask),
        "input_mask_sha256": sha256(mask),
        "output_mask": str(output),
        "output_mask_sha256": sha256(output),
        "drop_in_validation": dropin,
        "roi_input": roi_meta,
        "baseline_mask_input": mask_meta,
        "optional_osm_evidence": osm_evidence_meta,
        "tiles": roi_path,
        "coastal_plan": str(plan),
        "review_gpkg": str(review),
        "cache": str(cache),
        "parameters": {
            "threshold_m": float(a.threshold),
            "inland_context_m": float(a.inland_m),
            "seaward_search_m": float(250.0),
            "box_segment_m": float(5000.0),
            "box_halo_m": float(100.0),
            "min_seaward_cells": int(4),
            "local_relief_window_m": local_relief_window_m,
            "min_local_relief_m": min_local_relief_m,
            "max_flat_bridge_m": max_flat_bridge_m,
            "cleanup_min_close_m": cleanup_min_close_m,
            "cleanup_close_fraction": cleanup_close_fraction,
            "cleanup_max_close_m": cleanup_max_close_m,
            "cleanup_min_hole_m2": cleanup_min_hole_m2,
            "cleanup_hole_cells": cleanup_hole_cells,
            "residual_sliver_max_area_m2": residual_sliver_max_area_m2,
            "nearshore_transition_distance_m": 100.0,
            "nearshore_transition_max_area_m2": 50000.0,
        },
        "evidence_features": int(len(evidence)),
        "hydroflat_screen": hydroflat_stats,
        "hydroflat_screened_evidence_features": int(len(screened_evidence)),
        "hydroflat_seed_features": int(len(hydroflat_seeds)),
        "hydroflat_rejected_features": int(len(hydroflat_rejected)),
        "hydroflat_osm_protected_features": int(len(hydroflat_osm_protected)),
        "terrain_relief_candidate_features_before_nearshore": terrain_candidates_before_nearshore,
        "terrain_relief_candidate_features": int(len(terrain_candidates)),
        "unsupported_positive_review_features": int(len(flat_review)),
        "nearshore_transition_review_features": int(len(nearshore_review)),
        "accepted_features": int(len(accepted)),
        "accepted_osm_structure_supported_features": int(len(osm_supported)),
        "detached_review_features": int(len(detached)),
        "terrain_accepted_land_area_m2": terrain_accepted_area,
        "residual_water_slivers_removed": int(
            area_cleanup_stats.get("small_created_parts_removed", 0)
        ),
        "residual_water_sliver_area_m2": sliver_area,
        "residual_water_area_cleanup": area_cleanup_stats,
        "accepted_land_area_m2": accepted_area,
        "science_contract": [
            "A user-supplied water/ocean polygon or multipolygon is the required starting source-validity domain; Coast Refine does not own coastline acquisition.",
            "OSM Mask Helper is an independent upstream tool; its final mask may be supplied through --mask but Coast Refine works equally with any suitable single-polygon/vector water mask.",
            "Feature-level OSM evidence is optional. Only solid terrain-like classes (breakwater, groyne, dyke, embankment) may protect their immediate positive-TNM footprint; piers, quays, docks, floating infrastructure, seawalls, and training walls do not promote land. OSM never creates TNM-positive cells or bypasses inland connectivity.",
            "The v6.6 nearshore screen is review-only and applies only to unprotected hydroflat-survivor additions with no local terrain core that remain within 100 m of the original shoreline; locally terrain-supported additions and allowed OSM-protected solid structures are not rejected by this screen.",
            "A broad nearly constant positive water surface (15 m local relief <=0.02 m over >=400 m2) may seed hydroflat-fringe suppression; rejection propagates only through positive cells whose four-neighbor steps stay <=10% grade and <=0.5 m absolute step, and is blocked by nonplanar terrain more than 15 m from a seed (15 m opposite-sample midpoint residual >0.02 m). This added curvature cue is experimental.",
            "Hydroflat-fringe suppression targets positive hydroflattened water plateaus with smooth interpolated landward ramps; the rule is evidence-driven but contains no regional identifier or coordinate test.",
            "Source-edge/nodata/claim-edge metrics remain diagnostic only; no source-edge rejection threshold is applied.",
            "The supplied effective raster provides refinement evidence; no source module is inferred.",
            "Superseded older/coarser TNM coverage cannot corroborate a newer/higher-priority claim.",
            f"Only gridded terrain above {a.threshold:g} m in its source vertical datum is eligible to add land; no tidal or datum conversion is inferred.",
            "Automatic additions are derived from locally terrain-supported raster cells; component-wide relief cannot make an entire flat harbor eligible.",
            f"Automatic additions must connect to above-threshold terrain in the {a.inland_m:g} m landward context.",
            "Only new TNM additions receive source-resolution-aware micro-topology cleanup; the baseline water-mask geometry is never globally smoothed.",
            "Area-only water-part deletion is disabled in the active owner; residual water and withheld/detached evidence are preserved. Outer topology repair uses observed source-cell inconsistencies.",
            "Search covers all starting water inside the ROI; detached positive terrain is review-only.",
            "Below-threshold TNM never retracts baseline land in the standalone restoration owner; the outer classifier may add separately marked experimental land-to-water cells to its final comparison vectors.",
            "Final GPKG preserves the canonical baseline mask layer/schema/CRS; non-canonical GeoJSON/GPKG inputs are normalized first.",
        ],
    }
    (work / "run_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(
        description="Refine resolved water polygons from original elevation evidence"
    )
    for key in ("mask", "area", "evidence", "output", "cache", "workdir"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--mask-layer", default="mask")
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--inland-m", type=float, default=100.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
