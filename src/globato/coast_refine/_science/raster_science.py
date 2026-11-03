#!/usr/bin/env python3
"""Direct original-raster science using existing refinement and classification owners."""

from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import geopandas as gpd
import fiona
import numpy as np
import rasterio
import shapely
from shapely.geometry import box

from .coast_refine_v6_6_1.coastal_plan import _metric_crs
from .coastline_classify import (
    classify_full,
    ocean_components,
    polygon_water,
    stitch_water,
)
from .extract_raw_surface import copy_band_one
from .input_contracts import (
    select_baseline,
    raster_signature,
    sha256,
    signature_differences,
    check_encoding,
)


def record(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def read_water(path):
    frame = gpd.read_file(path, layer="mask").to_crs(4326)
    return polygon_water(shapely.union_all(frame.geometry))


def vector(path, layer, geometry, properties=None):
    # One row can contain several clipped pieces; preserve every land hole.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    geometry = polygon_water(geometry)
    if geometry.is_empty:
        # Fiona/GDAL round-trips a one-row empty polygon as a null geometry.
        # Publish zero features instead, retaining a real polygon layer/CRS.
        fields = {
            k: "int"
            if isinstance(v, int)
            else "float"
            if isinstance(v, float)
            else "str"
            for k, v in (properties or {}).items()
        }
        with fiona.open(
            path,
            "w",
            driver="GPKG",
            layer=layer,
            crs="EPSG:4326",
            schema={"geometry": "MultiPolygon", "properties": fields},
        ):
            pass
        return
    frame = gpd.GeoDataFrame(
        {k: [v] for k, v in (properties or {}).items()}, geometry=[geometry], crs=4326
    )
    frame.to_file(path, layer=layer, driver="GPKG", engine="fiona")


def regional_water_components(water, context, seeds, baseline, *, ocean_anchor=None):
    """Do not certify newly isolated starting marine water as inland water.

    Refinement can close a modeled route across a bridge, structure or uncertain
    shoreline. Enclosure alone does not prove the remaining water is a lake.
    Keep its geometry intact and withhold TNM until its connection is reviewed.
    Existing disconnected inland water and explicitly seeded water keep their
    classifications. Positive-area ancestry is required; point/edge contacts
    cannot establish either a water route or marine ancestry.
    """
    starting, starting_receipt = ocean_components(
        baseline, context, seeds, ocean_anchor=ocean_anchor
    )
    groups, receipt = ocean_components(water, context, seeds, ocean_anchor=ocean_anchor)
    parts = sorted(
        shapely.get_parts(polygon_water(water.intersection(context))),
        key=lambda p: (-p.area, p.bounds),
    )
    # One overlay avoids comparing the large regional ocean polygon repeatedly
    # against thousands of small enclosed components.
    overlap = polygon_water(groups["inland"].intersection(starting["marine"]))
    changed = set()
    if not overlap.is_empty:
        tree = shapely.STRtree(parts)
        for part in shapely.get_parts(overlap):
            changed.update(
                int(i)
                for i in tree.query(part.representative_point(), predicate="within")
            )
    disconnected = polygon_water(shapely.union_all([parts[i] for i in sorted(changed)]))
    if changed:
        groups["inland"] = polygon_water(
            shapely.union_all(
                [
                    parts[i]
                    for i, row in enumerate(receipt["components"])
                    if row["status"] == "inland" and i not in changed
                ]
            )
        )
        groups["unresolved"] = polygon_water(
            shapely.union_all([groups["unresolved"], disconnected])
        )
    for i, row in enumerate(receipt["components"]):
        row["newly_disconnected_baseline_marine"] = i in changed
        if i in changed:
            row.update(
                status="unresolved",
                unresolved_reason="refinement_closed_starting_marine_connection",
            )
    receipt.update(
        newly_disconnected_baseline_marine_components=len(changed),
        newly_disconnected_marine_policy="unresolved; preserve water geometry; exclude TNM; no artificial connections",
        starting_marine_selection_mode=starting_receipt.get("marine_selection_mode"),
        starting_marine_selection_rule=starting_receipt.get("marine_selection_rule"),
    )
    return groups, receipt, disconnected


def read_polygon_layer(path, layer="mask"):
    path = Path(path)
    if not path.is_file():
        return shapely.MultiPolygon([])
    if path.suffix.lower() == ".gpkg" and layer not in fiona.listlayers(path):
        return shapely.MultiPolygon([])
    frame = gpd.read_file(path, layer=layer if path.suffix.lower() == ".gpkg" else None)
    if frame.empty:
        return shapely.MultiPolygon([])
    if frame.crs is None:
        raise ValueError(f"Missing CRS: {path}")
    return polygon_water(shapely.union_all(frame.to_crs(4326).geometry))


def collect_land_to_water_cells(tile_dirs, processed_extent):
    """Collect exact raw-grid land-to-water cell footprints before tile clipping.

    `classification/refined_water.gpkg` stores these cells in the
    land_to_water layer.  Keeping the full cells until after all
    tile interiors are stitched removes seam fragments without rasterizing the
    rest of the shoreline.  The final union is clipped once to the union of all
    processed tile interiors, so a cell may cross an internal seam but cannot
    leak into unprocessed exterior context.
    """
    geoms = []
    sources = []
    for tile, td in tile_dirs:
        path = Path(td) / "classification/refined_water.gpkg"
        geom = read_polygon_layer(path, "land_to_water")
        if not geom.is_empty:
            geoms.append(geom)
            sources.append(tile)
    combined = (
        polygon_water(shapely.union_all(geoms)).intersection(processed_extent)
        if geoms
        else shapely.MultiPolygon([])
    )
    return polygon_water(combined), sources


def repair_stitched_water_topology(water, land_to_water_cells, root):
    """Restore exact full-cell additions lost only by vector/tile clipping.

    This does not smooth, buffer or snap the complete coastline.  It only unions
    the already-classified land-to-water raw-grid cells after regional stitching.
    Explicit reviewed add_land edits still run later and therefore override this
    repair when necessary.
    """
    additions = polygon_water(land_to_water_cells)
    repair = polygon_water(additions.difference(water))
    repaired = polygon_water(shapely.union_all([water, additions]))
    parts = []
    if not repair.is_empty:
        metric = _metric_crs(gpd.GeoDataFrame(geometry=[repaired], crs=4326))
        repair_m = gpd.GeoSeries([repair], crs=4326).to_crs(metric).iloc[0]
        parts = [p for p in shapely.get_parts(repair_m) if not p.is_empty]
    vector(
        root / "review/post_stitch_cell_topology_repair.gpkg",
        "mask",
        repair,
        {"role": "classified_land_to_water_cells_restored_after_stitch"},
    )
    record(
        root / "review/post_stitch_cell_topology_repair.json",
        {
            "status": "PASS",
            "method": "union exact land_to_water raw-grid cell footprints after tile stitching; no global smoothing/buffering",
            "repair_components": len(parts),
            "repair_area_m2": float(sum(p.area for p in parts)),
            "min_component_area_m2": float(min((p.area for p in parts), default=0.0)),
            "max_component_area_m2": float(max((p.area for p in parts), default=0.0)),
            "classified_cell_geometry_fully_present_after_repair": bool(
                additions.difference(repaired).is_empty
            ),
            "explicit_reviewed_add_land_runs_after_this_repair": True,
        },
    )
    return repaired


def _grid_specs(tile_dirs):
    """Read only native RAW grid geometry for vector/raster consistency checks."""
    specs = []
    for tile, td in tile_dirs:
        path = Path(td) / "raw_elevation.tif"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing RAW grid required for topology consistency: {path}"
            )
        with rasterio.open(path) as ds:
            specs.append(
                {
                    "tile": tile,
                    "path": str(path),
                    "crs": ds.crs,
                    "transform": ds.transform,
                    "width": ds.width,
                    "height": ds.height,
                    "bounds": ds.bounds,
                }
            )
    return specs


def _contains_native_grid_center(geometry_wgs84, specs):
    """True when any native TNM pixel center falls inside/on a polygon.

    This intentionally tests grid geometry, not elevation value. The cleanup is
    about vector/raster ownership consistency: a residual polygon with no native
    cell center can change class only when another module rasterizes the vector
    on a different grid.
    """
    for spec in specs:
        if geometry_wgs84.is_empty:
            return False
        if spec["crs"] is None:
            raise ValueError(f"RAW grid has no CRS: {spec['path']}")
        if spec["crs"].to_epsg() == 4326:
            geom = geometry_wgs84
        else:
            geom = gpd.GeoSeries([geometry_wgs84], crs=4326).to_crs(spec["crs"]).iloc[0]
        raster_bounds = box(*spec["bounds"])
        if not geom.intersects(raster_bounds):
            continue
        minx, miny, maxx, maxy = geom.bounds
        inv = ~spec["transform"]
        # Transform all four corners so rotated grids are bounded safely.
        corners = [inv * (x, y) for x in (minx, maxx) for y in (miny, maxy)]
        cols = [c for c, _ in corners]
        rows = [r for _, r in corners]
        c0 = max(0, math.floor(min(cols)) - 1)
        c1 = min(spec["width"], math.ceil(max(cols)) + 2)
        r0 = max(0, math.floor(min(rows)) - 1)
        r1 = min(spec["height"], math.ceil(max(rows)) + 2)
        if c1 <= c0 or r1 <= r0:
            continue
        # A split mainland ancestor can span millions of cells. Bound scratch
        # memory and stop on the first center; avoid allocating GEOS points.
        tr = spec["transform"]
        for row in range(r0, r1, 256):
            for col in range(c0, c1, 256):
                rr, cc = np.meshgrid(
                    np.arange(row, min(row + 256, r1)),
                    np.arange(col, min(col + 256, c1)),
                    indexing="ij",
                )
                xs = tr.c + (cc + 0.5) * tr.a + (rr + 0.5) * tr.b
                ys = tr.f + (cc + 0.5) * tr.d + (rr + 0.5) * tr.e
                if np.any(shapely.intersects_xy(geom, xs, ys)):
                    return True
    return False


def repair_orphan_land_fragments(
    water, baseline_water, land_to_water_cells, processed_extent, tile_dirs, root
):
    """Remove only correction-created land remnants invisible to native grids.

    Square land-to-water cells can sever sub-cell corners/strips from a larger
    baseline OSM land component. Those fragments remain geometrically land even
    though no native TNM pixel center can ever classify them as land. Downstream
    rasterization on another grid can therefore resurrect them as false land.

    A fragment is absorbed into water only when all of these are true:
      * it lies in baseline land (not newly restored TNM terrain);
      * its baseline land ancestor was split into multiple pieces by refinement;
      * it touches an already-classified land-to-water cell footprint; and
      * no native RAW grid cell center from any processed tile falls in it.

    There is deliberately no area threshold, simplify, buffer or global close.
    Pre-existing islands remain; any fragment with a native grid center remains;
    reviewed add_land corrections still run afterward and can override this.
    """
    additions = polygon_water(land_to_water_cells)
    if additions.is_empty:
        removed = retained = shapely.MultiPolygon([])
        vector(
            root / "review/post_stitch_orphan_land_removed.gpkg",
            "mask",
            removed,
            {"role": "no_grid_center_orphan_land_removed"},
        )
        vector(
            root / "review/post_stitch_orphan_land_retained.gpkg",
            "mask",
            retained,
            {"role": "orphan_land_retained_native_grid_center"},
        )
        record(
            root / "review/post_stitch_orphan_land_cleanup.json",
            {
                "status": "PASS",
                "candidate_fragments": 0,
                "removed_fragments": 0,
                "retained_with_native_grid_center": 0,
                "removed_area_m2": 0.0,
                "rule": "baseline ancestor split + touches classified water cells + no native TNM grid center",
                "global_smoothing_or_area_threshold": False,
            },
        )
        return water

    baseline_land = polygon_water(processed_extent.difference(baseline_water))
    current_land = polygon_water(processed_extent.difference(water))
    baseline_parts = [p for p in shapely.get_parts(baseline_land) if not p.is_empty]
    current_parts = [p for p in shapely.get_parts(current_land) if not p.is_empty]
    tree = shapely.STRtree(current_parts) if current_parts else None
    candidate_ids = set()
    split_ancestors = 0
    if tree is not None:
        for ancestor in baseline_parts:
            ids = [
                int(i)
                for i in tree.query(ancestor, predicate="intersects")
                if current_parts[int(i)].intersection(ancestor).area > 1e-14
            ]
            if len(ids) <= 1:
                continue
            split_ancestors += 1
            for i in ids:
                part = current_parts[i]
                overlap = part.intersection(ancestor).area
                if part.area <= 0 or overlap / part.area < 0.999:
                    continue  # not an inherited baseline-land remnant
                if not part.intersects(additions):
                    continue
                candidate_ids.add(i)

    specs = _grid_specs(tile_dirs)
    removed_parts = []
    retained_parts = []
    for i in sorted(candidate_ids):
        part = current_parts[i]
        if _contains_native_grid_center(part, specs):
            retained_parts.append(part)
        else:
            removed_parts.append(part)

    removed = (
        polygon_water(shapely.union_all(removed_parts))
        if removed_parts
        else shapely.MultiPolygon([])
    )
    retained = (
        polygon_water(shapely.union_all(retained_parts))
        if retained_parts
        else shapely.MultiPolygon([])
    )
    repaired = polygon_water(shapely.union_all([water, removed]))
    metric = _metric_crs(gpd.GeoDataFrame(geometry=[processed_extent], crs=4326))
    removed_m = gpd.GeoSeries([removed], crs=4326).to_crs(metric).iloc[0]
    removed_metric_parts = [p for p in shapely.get_parts(removed_m) if not p.is_empty]
    vector(
        root / "review/post_stitch_orphan_land_removed.gpkg",
        "mask",
        removed,
        {"role": "no_grid_center_orphan_land_removed"},
    )
    vector(
        root / "review/post_stitch_orphan_land_retained.gpkg",
        "mask",
        retained,
        {"role": "orphan_land_retained_native_grid_center"},
    )
    record(
        root / "review/post_stitch_orphan_land_cleanup.json",
        {
            "status": "PASS",
            "baseline_land_components_examined": len(baseline_parts),
            "baseline_components_split_by_refinement": split_ancestors,
            "candidate_fragments": len(candidate_ids),
            "removed_fragments": len(removed_parts),
            "retained_with_native_grid_center": len(retained_parts),
            "removed_area_m2": float(sum(p.area for p in removed_metric_parts)),
            "min_removed_area_m2": float(
                min((p.area for p in removed_metric_parts), default=0.0)
            ),
            "max_removed_area_m2": float(
                max((p.area for p in removed_metric_parts), default=0.0)
            ),
            "rule": "baseline ancestor split + touches classified water cells + no native TNM grid center",
            "native_grid_sources": [s["tile"] for s in specs],
            "global_smoothing_or_area_threshold": False,
            "preexisting_baseline_islands_preserved": True,
            "tnm_restored_land_preserved": True,
            "explicit_reviewed_add_land_runs_after_this_cleanup": True,
        },
    )
    return repaired


def deck_domains(mapped_decks, starting_water):
    """Keep complete mapped evidence for review; certify only overwater pieces.

    A mapped pier may also include its genuine onshore foundation. Do not
    discard the geometry silently, nor assert missing OSM water is ocean.
    """
    return polygon_water(mapped_decks), polygon_water(
        mapped_decks.intersection(starting_water)
    )


def prepare_overwater_structures(a, root, context):
    """Resolve supplied structures or acquire bulk OSM locally; no API fallback."""
    masks = root / "masks"
    masks.mkdir(exist_ok=True)
    supplied = getattr(a, "overwater_structures", None)
    supplied_layer = getattr(a, "overwater_structures_layer", "mask")
    if supplied:
        source = Path(supplied).expanduser().resolve(strict=True)
        # An explicitly supplied missing/invalid layer is an input error, never
        # evidence that there are no docks. Empty polygon layers are legitimate.
        frame = gpd.read_file(
            source, layer=supplied_layer if source.suffix.lower() == ".gpkg" else None
        )
        if (
            frame.crs is None
            or frame.geometry.isna().any()
            or not frame.geometry.is_valid.all()
        ):
            raise ValueError(
                "Supplied overwater structures need CRS and valid polygon geometry"
            )
        if not frame.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
            raise ValueError("Supplied overwater structures must be polygons")
        structure = polygon_water(shapely.union_all(frame.to_crs(4326).geometry))
        vector(
            masks / "overwater_structure_exclusion.gpkg",
            "mask",
            structure,
            {"class": "overwater_structure_exclusion", "source": "supplied"},
        )
        record(
            masks / "OVERWATER_STRUCTURE_RECEIPT.json",
            {
                "status": "PASS",
                "source": str(source),
                "source_sha256": sha256(source),
                "layer": supplied_layer,
                "mode": "supplied",
                "policy": "exact supplied polygon geometry",
            },
        )
    else:
        raise ValueError(
            "Science requires explicit structure polygons; use an empty layer when none are supplied"
        )
    vector(
        root / "overwater_structure_exclusion_context.gpkg",
        "mask",
        structure.intersection(context),
        {"class": "overwater_structure_exclusion"},
    )
    return polygon_water(structure.intersection(context))


def parse_region(value):
    w, e, s, n = map(float, value.split("/"))
    if not (w < e and s < n):
        raise ValueError(
            "Context bounds must be W/E/S/N with positive width and height"
        )
    return box(w, s, e, n)


def has_coast(water, roi):
    metric = _metric_crs(gpd.GeoDataFrame(geometry=[roi], crs=4326))
    r = gpd.GeoSeries([roi], crs=4326).to_crs(metric).iloc[0]
    w = gpd.GeoSeries([water.intersection(roi)], crs=4326).to_crs(metric).iloc[0]
    w = polygon_water(w)
    return not w.is_empty and not w.boundary.difference(r.boundary.buffer(2)).is_empty


def homogeneous_water_precheck(
    roi, baseline_water, context, structures, mapped_water=None, *, inland_m=100.0
):
    """Prove a source footprint is all land or all water using existing OSM evidence.

    The metric guard protects nearby coastlines and inland-water boundaries;
    partial coverage, mapped decks, and contradictory mapped water are uncertain.
    Elevations and NoData are never treated as evidence of land/water class.
    This bypasses only coastline SCIENCE, not the module's native gridding.
    """
    from .grid_utils import metric_halo

    footprint = metric_halo(roi, metres=max(100.0, float(inland_m)) + 20.0)
    if not context.covers(footprint):
        return None, "OSM context does not cover guarded source footprint"
    if not structures.disjoint(footprint):
        return None, "Mapped overwater structure intersects guarded footprint"
    if mapped_water is not None and not mapped_water.disjoint(footprint):
        if not baseline_water.covers(mapped_water.intersection(footprint)):
            return None, "Mapped water not fully present in baseline"
    if baseline_water.disjoint(footprint):
        # A water-only map does not affirmatively establish land. In particular,
        # erroneous offshore TNM strips can sit outside incomplete OSM water
        # polygons. Fail closed: do not publish a LAND_ONLY bypass unless an
        # independent, positively mapped land domain becomes available.
        return None, "No mapped water: land NOT proven; do not fast-pass offshore TNM"
    if baseline_water.covers(footprint):
        return "WATER_ONLY", "Starting OSM water covers guarded source footprint"
    return None, "Mapped land-water boundary in guarded source footprint"


def source_check(path, roi, *, fingerprint=True):
    with rasterio.open(path) as ds:
        check_encoding(ds)
        wanted = gpd.GeoSeries([roi], crs=4326).to_crs(ds.crs).iloc[0]
        # Native Region.geo_transform rounds output dimensions to whole cells.
        # A buffered AOI can therefore fall just beyond the evidence raster on
        # TWO axes at once (e.g., 0.4 cell east and 0.4 cell south). A Shapely
        # geometry buffer uses radial distance at corners: sqrt(.4**2+.4**2)
        # exceeds the original 0.55-cell tolerance despite each axis being
        # within its own half-cell rounding allowance. Check rectangular,
        # axis-specific cell-center coverage instead. Do not pad elevations,
        # relax the per-axis limit, or resample the original evidence.
        tx = abs(ds.transform.a) * 0.55
        ty = abs(ds.transform.e) * 0.55
        available = box(
            ds.bounds.left - tx,
            ds.bounds.bottom - ty,
            ds.bounds.right + tx,
            ds.bounds.top + ty,
        )
        if not available.covers(wanted):
            raise ValueError(f"RAW does not cover requested tile: {path}")
    return raster_signature(path) if fingerprint else None


def scientific_parameters(a):
    """One recorded contract for the existing science; elevations are metres."""
    parameters = {
        name: float(getattr(a, name, default))
        for name, default in [
            ("shore_level_m", 0.0),
            ("threshold_m", 0.01),
            ("inland_m", 100.0),
        ]
    }
    if not all(math.isfinite(value) for value in parameters.values()):
        raise ValueError("Scientific parameters must be finite")
    if parameters["threshold_m"] < 0 or parameters["inland_m"] <= 0:
        raise ValueError(
            "threshold-m must be nonnegative and inland-m must be positive"
        )
    return parameters


def copy_verified_raw(source, target, expected):
    """Verify the saved RAW against the preflight identity, not just a live source."""
    copy_band_one(source, target)
    changed = signature_differences(raster_signature(target), expected)
    if changed:
        Path(target).unlink()
        raise ValueError("RAW copy differs from verified input: " + str(changed))


def has_valid_terrain(path):
    """Bounded-memory check; missing elevations never imply land or water."""
    from .extract_raw_surface import blocks

    with rasterio.open(path) as ds:
        for window in blocks(ds.width, ds.height):
            valid = ds.read_masks(1, window=window) > 0
            if valid.any() and np.any(valid & np.isfinite(ds.read(1, window=window))):
                return True
    return False


def empty_terrain_classification(raw, baseline, out):
    """Retain known OSM, avoid full-grid science arrays, keep the actual empty TIFF."""
    from .extract_raw_surface import blocks

    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(baseline, out / "refined_water.gpkg")
    with rasterio.open(raw) as ds:
        profile = ds.profile.copy()
        for name in ("land", "water", "withheld"):
            gpd.GeoDataFrame(
                {"decision": [name]}, geometry=[shapely.MultiPolygon([])], crs=ds.crs
            ).to_file(out / "decision_support.gpkg", layer=name, driver="GPKG")
        profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
        with rasterio.open(out / "water_additions.tif", "w", **profile) as dst:
            for window in blocks(ds.width, ds.height):
                dst.write(
                    np.zeros((int(window.height), int(window.width)), "uint8"),
                    1,
                    window=window,
                )
    result = {
        "status": "NO_VALID_TERRAIN",
        "raw_valid_cells": 0,
        "restored_cells": 0,
        "water_added_cells": 0,
        "terrain_refinement": False,
        "decision_support": "decision_support.gpkg",
        "science_note": "Starting OSM retained; no-data is not evidence of land or water.",
    }
    record(out / "classification.json", result)
    return result


def homogeneous_terrain_classification(raw, baseline, out, kind):
    """Publish baseline-only evidence, with no invented terrain decisions.

    Use the same required artifact contract as normal classification. A valid
    source can be entirely water, so record real valid cells instead of
    misrepresenting such a tile as lacking terrain observations.
    """
    from .extract_raw_surface import blocks

    if kind not in ("LAND_ONLY", "WATER_ONLY"):
        raise ValueError("Invalid homogeneous source classification")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(baseline, out / "refined_water.gpkg")
    nvalid = 0
    with rasterio.open(raw) as ds:
        profile = ds.profile.copy()
        for name in ("land", "water", "withheld"):
            gpd.GeoDataFrame(
                {"decision": [name]}, geometry=[shapely.MultiPolygon([])], crs=ds.crs
            ).to_file(out / "decision_support.gpkg", layer=name, driver="GPKG")
        profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
        with rasterio.open(out / "water_additions.tif", "w", **profile) as dst:
            for window in blocks(ds.width, ds.height):
                mask = ds.read_masks(1, window=window) > 0
                z = ds.read(1, window=window)
                nvalid += int((mask & np.isfinite(z)).sum())
                dst.write(
                    np.zeros((int(window.height), int(window.width)), "uint8"),
                    1,
                    window=window,
                )
    result = {
        "status": "PASS_HOMOGENEOUS_BYPASS",
        "homogeneous_bypass": kind,
        "raw_valid_cells": nvalid,
        "restored_cells": 0,
        "water_added_cells": 0,
        "terrain_refinement": False,
        "decision_support": "decision_support.gpkg",
        "science_note": "Unchanged starting OSM water and empty scientific decisions; no new elevation-based coastline evidence.",
    }
    record(out / "classification.json", result)
    return result


def run(a, root):
    science = scientific_parameters(a)
    # A single original raster and its AOI constitute one science job. There
    # is no regional tile catalogue, selection, source map, or runner in the
    # installed hook. Keep the historical internal tile name (area) to avoid
    # changing scientific artifacts and their downstream identities.
    area = gpd.read_file(a.area)
    if area.crs is None or area.empty:
        raise ValueError("Original-raster AOI must have geometry and CRS")
    roi = shapely.union_all(area.to_crs(4326).geometry)
    if (
        roi.is_empty
        or not roi.is_valid
        or roi.geom_type not in ("Polygon", "MultiPolygon")
    ):
        raise ValueError("Original-raster AOI must be a valid polygon")
    source = Path(a.evidence).expanduser().resolve(strict=True)
    signature = source_check(source, roi)
    context_request = (
        parse_region(a.context_region) if a.context_region else box(*roi.bounds)
    )
    if not context_request.covers(roi):
        raise ValueError("Connectivity context must cover the requested AOI")
    if not a.water_mask or not a.context_region:
        raise ValueError("Science requires resolved water polygons and known context")
    select_baseline(
        root,
        water_mask=a.water_mask,
        mask_layer=a.mask_layer,
        allow_empty=bool(a.ocean_anchor),
    )
    context = context_request
    baseline = root / "masks/baseline_water_mask.gpkg"
    base_geometry = read_water(baseline).intersection(context)
    anchor_path = a.ocean_anchor
    ocean_anchor = None
    if anchor_path:
        ocean_anchor = read_polygon_layer(anchor_path).intersection(context)
        # OSM marine connectivity requires ocean evidence beyond the
        # refinement region. A local class=ocean polygon is itself explicit
        # evidence, even when entirely within the AOI. Both then enter the
        # same connectivity classifier; the supplied input is never buffered.
        anchor_scope = getattr(a, "anchor_scope", "external")
        if anchor_scope not in ("external", "declared"):
            raise ValueError("Unknown ocean-anchor evidence scope")
        if anchor_scope == "external":
            ocean_anchor = polygon_water(ocean_anchor.difference(roi))
        else:
            ocean_anchor = polygon_water(ocean_anchor)
        vector(
            root / "resolved_ocean_anchor.gpkg",
            "mask",
            ocean_anchor,
            {"source": "resolved_ocean_anchor", "scope": anchor_scope},
        )
        science["ocean_anchor_geometry_sha256"] = hashlib.sha256(
            shapely.normalize(ocean_anchor).wkb
        ).hexdigest()
    seeds = []  # Production uses OSM connectivity; no regional seed overrides.
    # Validate automatic exterior-based marine inference (or an explicit user
    # override) before expensive RAW acquisition.
    _, marine_inference = ocean_components(
        base_geometry, context, seeds, ocean_anchor=ocean_anchor
    )
    vector(
        root / "connectivity_context.gpkg",
        "context",
        context,
        {"role": "known_baseline_coverage"},
    )
    # Input validation precedes structure resolution and terrain science.
    structures = prepare_overwater_structures(a, root, context)
    vector(
        root / "review/mapped_structure_footprints.gpkg",
        "mask",
        structures,
        {"role": "mapped_structure_before_water_only_resolution"},
    )
    # OSM polygons can include a real onshore landing. Only their baseline-water
    # portions are known over-water decks; neither convert nor erase that land.
    # Source-use exclusion retains only the positively mapped over-water
    # portion of pier/dock footprints. The onshore landing remains land;
    # locations OSM omitted from water are explicitly withheld for review.
    mapped_structures, structures = deck_domains(structures, base_geometry)
    vector(
        root / "overwater_structure_exclusion_context.gpkg",
        "mask",
        structures,
        {
            "class": "overwater_structure_exclusion",
            "policy": "mapped_structure_intersect_baseline_water",
        },
    )
    vector(
        root / "review/mapped_structure_outside_osm_water.gpkg",
        "mask",
        polygon_water(mapped_structures.difference(base_geometry)),
        {"role": "structure_deck_outside_starting_osm_water_for_review"},
    )
    mapped_for_precheck = (
        read_polygon_layer(a.mapped_water_areas).intersection(context)
        if getattr(a, "mapped_water_areas", None)
        else None
    )
    record(
        root / "run_config.json",
        {
            "source": "raster",
            "evidence": str(source),
            "baseline": str(baseline),
            "context_bounds": list(context.bounds),
            "marine_selection_mode": marine_inference.get("marine_selection_mode"),
            "marine_selection_rule": marine_inference.get("marine_selection_rule"),
            "science_parameters": science,
            "ocean_anchor": str(anchor_path) if anchor_path else None,
            "ocean_anchor_scope": getattr(a, "anchor_scope", "external"),
            "terrain_contract": "single float band, metres in declared source datum, identity scale/offset, north-up",
            "external_context_is_unrefined": True,
        },
    )
    # Preserve the established workspace path, without a tile catalogue.
    td = root / "tiles" / "area"
    td.mkdir(parents=True)
    roi_path = td / "tile.geojson"
    gpd.GeoDataFrame({"NAME": ["area"]}, geometry=[roi], crs=4326).to_file(
        roi_path, driver="GeoJSON"
    )
    raw = td / "raw_elevation.tif"
    copy_verified_raw(source, raw, signature)
    record(
        td / "raw_identity.json",
        {
            "mode": "supplied_unmasked_raw",
            "source": str(source),
            "source_sha256": sha256(source),
            "effective_signature": signature,
            "provenance_note": "Original elevation raster in metres, without source-specific assumptions.",
        },
    )
    full = td / "full_refined_water.gpkg"
    review = None
    # Actual source validity is consulted before spatial bypass decisions:
    # an empty TNM raster cannot prove water, nor should it trigger a
    # terrain classifier. For nonempty sources, the OSM domain remains the
    # authority for exclusion; positive heights never certify dry land.
    valid_source = has_valid_terrain(raw)
    homogeneous, why = (
        homogeneous_water_precheck(
            roi,
            base_geometry,
            context,
            structures,
            mapped_for_precheck,
            inland_m=science["inland_m"],
        )
        if valid_source
        else (None, "No valid source pixels")
    )
    if homogeneous is not None:
        # Preserve exact baseline and downstream vector/decision receipt
        # contracts, but bypass expensive raster terrain refinement and
        # all full-grid shoreline classification arrays.
        print(
            f"[coast_refine] homogeneous_bypass {homogeneous} reason={why}", flush=True
        )
        shutil.copy2(baseline, full)
        classification = homogeneous_terrain_classification(
            raw, baseline, td / "classification", homogeneous
        )
        record(
            td / "full_performance.json",
            {
                "status": "HOMOGENEOUS_BYPASS",
                "class": homogeneous,
                "reason": why,
                "refinement_skipped": True,
                "elevation_science_changes": 0,
                "baseline_preserved": True,
            },
        )
        refined = polygon_water(base_geometry.intersection(roi))
    elif not valid_source:
        print(
            "== zero valid terrain cells; starting OSM geometry retained ==", flush=True
        )
        shutil.copy2(baseline, full)
        classification = empty_terrain_classification(
            raw, baseline, td / "classification"
        )
        record(
            td / "full_performance.json",
            {"status": "NO_VALID_TERRAIN", "refinement_skipped": True},
        )
        refined = polygon_water(base_geometry.intersection(roi))
    else:
        if has_coast(base_geometry, roi):
            work = td / "full_work"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "globato.coast_refine._science.coast_refine_v6_6_1.restore",
                    "--mask",
                    str(baseline),
                    "--area",
                    str(roi_path),
                    "--output",
                    str(full),
                    "--cache",
                    str(a.cache),
                    "--workdir",
                    str(work),
                    "--evidence",
                    str(raw),
                    "--inland-m",
                    str(science["inland_m"]),
                    "--threshold",
                    str(science["shore_level_m"] + science["threshold_m"]),
                ],
                check=True,
            )
            review = work / "coast_refine_review.gpkg"
        else:
            shutil.copy2(baseline, full)
            record(
                td / "full_performance.json",
                {"status": "NO_INTERIOR_COASTLINE", "refinement_skipped": True},
            )
        classification = classify_full(
            raw,
            baseline,
            full,
            review,
            td / "classification",
            shore_level_m=science["shore_level_m"],
            threshold_m=science["threshold_m"],
        )
        refined = read_water(td / "classification/refined_water.gpkg")
    processed_extent = polygon_water(roi)
    water = stitch_water(base_geometry, [(roi, refined)])
    tile_dirs = [("area", td)]
    land_to_water_cells, addition_sources = collect_land_to_water_cells(
        tile_dirs, processed_extent
    )
    water = repair_stitched_water_topology(water, land_to_water_cells, root)
    water = repair_orphan_land_fragments(
        water, base_geometry, land_to_water_cells, processed_extent, tile_dirs, root
    )
    record(
        root / "review/post_stitch_cell_topology_sources.json",
        {"tiles_with_land_to_water_cells": addition_sources},
    )
    if getattr(a, "mapped_water_areas", None):
        mapped_water = read_polygon_layer(a.mapped_water_areas).intersection(
            base_geometry
        )
        vector(
            root / "review/mapped_water_preserved.gpkg",
            "mask",
            mapped_water,
            {"role": "mapped_water_area_not_dry_land_from_positive_sea_level_test"},
        )
        water = polygon_water(shapely.union_all([water, mapped_water]))
    # Structures are temporarily water only for hydrologic connectivity. They are
    # then subtracted from source-use masks, so a dock cannot become a land barrier
    # and its deck elevation cannot become either topography or bathymetry.
    connectivity_water = polygon_water(shapely.union_all([water, structures]))
    groups, connectivity, disconnected = regional_water_components(
        connectivity_water, context, seeds, base_geometry, ocean_anchor=ocean_anchor
    )
    # A mapped pier is a deck above water, not a seafloor void. The marine
    # bathymetry domain includes the water beneath; keep the deck footprint in
    # a separate mask for screening aerial/overwater-structure returns.
    bathymetry_domain = polygon_water(groups["marine"])
    refined_land = polygon_water(context.difference(connectivity_water))
    vector(
        root / "newly_disconnected_marine_review.gpkg",
        "mask",
        disconnected,
        {
            "class": "unresolved_starting_marine_water",
            "requires_connectivity_review": 1,
        },
    )
    vector(
        root / "refined_water_context.gpkg",
        "mask",
        connectivity_water,
        {"class": "refined_all_hydrologic_water"},
    )
    vector(
        root / "refined_land_context.gpkg",
        "mask",
        refined_land,
        {"class": "refined_land_within_known_context"},
    )
    vector(
        root / "marine_connectivity_water_context.gpkg",
        "mask",
        groups["marine"],
        {"class": "marine_connectivity_water"},
    )
    vector(
        root / "marine_water_context.gpkg",
        "mask",
        bathymetry_domain,
        {"class": "marine_bathymetry_domain", "marine_id": 1},
    )
    vector(
        root / "bathymetry_domain_context.gpkg",
        "mask",
        bathymetry_domain,
        {"class": "marine_bathymetry_domain"},
    )
    # Only the observed land/water interface, restricted to processed tiles.
    # Context/tile envelope edges are not manufactured coastlines.
    coast = connectivity_water.boundary.difference(context.boundary).intersection(
        processed_extent
    )
    coast_parts = [
        g
        for g in shapely.get_parts(coast)
        if g.geom_type in ("LineString", "MultiLineString")
    ]
    coast = (
        shapely.union_all(coast_parts) if coast_parts else shapely.MultiLineString([])
    )
    gpd.GeoDataFrame(
        {"role": ["refined_hydrologic_coastline"]}, geometry=[coast], crs=4326
    ).to_file(
        root / "refined_coastline.gpkg",
        layer="coastline",
        driver="GPKG",
        engine="fiona",
    )
    for status, geometry in groups.items():
        if status != "marine":
            vector(
                root / f"{status}_water_context.gpkg",
                "mask",
                geometry,
                {"class": status, "marine_id": 0},
            )
    connectivity.update(
        overwater_structure_policy="marine water continues beneath mapped decks; screen deck returns by source kind; onshore parts withheld for review",
        overwater_structure_area_present=not structures.is_empty,
        bathymetry_domain_includes_water_under_piers=True,
    )
    record(root / "connectivity_review.json", connectivity)
    empty_source = classification.get("raw_valid_cells") == 0
    # Only vector handoff is required here. Native TIFF publication belongs
    # to masking.py after the single target-grid classification is approved.
    final = root / "final/masks"
    final.mkdir(parents=True)
    for source_name, product in {
        "marine_water_context": "marine_water",
        "bathymetry_domain_context": "bathymetry_domain",
        "marine_connectivity_water_context": "marine_connectivity_water",
        "refined_water_context": "refined_water",
        "refined_land_context": "refined_land",
        "refined_coastline": "refined_coastline",
        "inland_water_context": "inland_water",
        "unresolved_water_context": "unresolved_water",
        "overwater_structure_exclusion_context": "overwater_structure_exclusion",
        "connectivity_context": "known_context",
    }.items():
        shutil.copy2(root / (source_name + ".gpkg"), final / (product + ".gpkg"))
    relevant_unresolved = groups["unresolved"].intersection(roi)
    status = (
        "COMPLETE_WITH_UNRESOLVED_WATER"
        if not relevant_unresolved.is_empty
        else "COMPLETE"
    )
    if empty_source:
        status = "COMPLETE_WITH_NO_VALID_TERRAIN"
    record(
        root / "status.json",
        {
            "status": status,
            "processed_tiles": ["area"],
            "terrain_evidence_tiles": [] if empty_source else ["area"],
            "no_source_coverage_tiles": [],
            "raster_product_tiles": ["area"],
            "no_valid_terrain_tiles": ["area"] if empty_source else [],
            "no_source_coverage_policy": "starting OSM geometry retained; no terrain evidence or TIFF invented",
            "marine_components_in_context": connectivity["marine_components"],
            "newly_disconnected_baseline_marine_components": connectivity[
                "newly_disconnected_baseline_marine_components"
            ],
            "science_status": "EXPERIMENTAL_PENDING_SCIENCE_REVIEW",
            "remove": getattr(a, "remove", None),
            "final_folder": str(root / "final"),
            "barrier_semantics": "marine_water/bathymetry_domain represent water beneath piers; separate deck-return exclusion is source-specific",
        },
    )
