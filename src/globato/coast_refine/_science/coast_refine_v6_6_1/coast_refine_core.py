#!/usr/bin/env python3
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import pyogrio
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from rasterio.features import rasterize, shapes
from rasterio.windows import from_bounds
from scipy import ndimage as ndi
import shapely
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import transform as shp_transform

FOUR = ndi.generate_binary_structure(2, 1)


# Optional OSM evidence may protect only solid terrain-like coastal structures.
# Piers/quays/docks/floating structures deliberately do NOT promote land: where
# water exists underneath or alongside the structure, that surface remains in the
# bathymetry domain.
OSM_STRUCTURE_CLASSES = {
    "breakwater",
    "groyne",
    "dyke",
    "embankment",
}


def load_optional_osm_evidence(path, layer="catalog", target_crs="EPSG:4326"):
    """Load optional feature-level OSM evidence without making it a requirement.

    The canonical Coast Refine input remains a single water polygon/multipolygon.
    If an OSM Mask Helper catalog is supplied, only solid terrain-like coastal
    structures (breakwater/groyne/dyke/embankment) are retained as optional
    support/protection evidence. Piers, quays, docks and floating structures are
    deliberately excluded. OSM never
    creates TNM-positive cells and never bypasses inland connectivity; when OSM
    is absent, the standalone TNM-only science remains fully operational.
    """
    if not path:
        return gpd.GeoDataFrame(
            columns=["class", "group", "geometry"], geometry="geometry", crs=target_crs
        )
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    layers = (
        pyogrio.list_layers(path)[:, 0].tolist()
        if path.suffix.lower() == ".gpkg"
        else []
    )
    if path.suffix.lower() == ".gpkg":
        if layer not in layers:
            raise ValueError(
                f"OSM evidence layer {layer!r} not found in {path}; available={layers}"
            )
        g = gpd.read_file(path, layer=layer)
    else:
        g = gpd.read_file(path)
    if g.empty:
        return gpd.GeoDataFrame(
            columns=["class", "group", "geometry"], geometry="geometry", crs=target_crs
        )
    if g.crs is None:
        raise ValueError(f"Optional OSM evidence has no CRS: {path}")
    cls = (
        g["class"].astype(str).str.lower().str.strip() if "class" in g.columns else None
    )
    if cls is None:
        raise ValueError(
            f"Optional OSM evidence {path}:{layer} lacks a 'class' field; "
            "Coast Refine will not promote generic coastal_structure records without an explicit solid-structure class"
        )
    # Deliberately conservative allow-list.  A generic coastal_structure group is
    # not enough because that group also contains piers/quays and other structures
    # with water underneath that belong in the bathymetry domain.
    keep = cls.isin(OSM_STRUCTURE_CLASSES)
    out = g.loc[keep].copy()
    if out.empty:
        return gpd.GeoDataFrame(
            columns=list(g.columns), geometry="geometry", crs=target_crs
        )
    out["geometry"] = out.geometry.map(shapely.make_valid)
    out = out[~out.geometry.is_empty].copy()
    return out.to_crs(target_crs)


def annotate_osm_structure_support(
    accepted,
    osm_evidence,
    *,
    metric_crs,
    min_buffer_m=2.0,
    pixel_buffer_fraction=1.5,
    max_buffer_m=20.0,
):
    """Annotate accepted TNM candidates with optional mapped-structure support.

    This post-classification annotation is deliberately geometry-neutral: output
    geometry is unchanged.  Earlier v6.5 hydroflat screening may optionally use
    the same structure evidence as a protection mask, but this function only
    records which final accepted candidates overlap/approach mapped structures.
    """
    if accepted is None:
        return accepted, gpd.GeoDataFrame(
            columns=["geometry"], geometry="geometry", crs="EPSG:4326"
        )
    out = accepted.copy()
    out["osm_structure_supported"] = False
    out["osm_structure_count"] = 0
    out["osm_structure_classes"] = ""
    if out.empty or osm_evidence is None or osm_evidence.empty:
        return out, out.iloc[0:0].copy()

    am = out.to_crs(metric_crs).copy()
    om = osm_evidence.to_crs(metric_crs).copy()
    geoms = list(om.geometry.values)
    if not geoms:
        return out, out.iloc[0:0].copy()
    tree = shapely.STRtree(geoms)
    flags = []
    counts = []
    classes = []
    for _, row in am.iterrows():
        pix = float(row.get("pixel_m", 1.0) or 1.0)
        if not math.isfinite(pix) or pix <= 0:
            pix = 1.0
        tol = min(
            float(max_buffer_m),
            max(float(min_buffer_m), pix * float(pixel_buffer_fraction)),
        )
        q = row.geometry.buffer(tol)
        idxs = [
            int(i)
            for i in np.asarray(
                tree.query(q, predicate="intersects"), dtype=int
            ).tolist()
        ]
        flags.append(bool(idxs))
        counts.append(len(idxs))
        vals = []
        for i in idxs:
            if "class" in om.columns:
                v = str(om.iloc[i].get("class", "") or "").strip()
                if v:
                    vals.append(v)
        classes.append(",".join(sorted(set(vals))))
    out["osm_structure_supported"] = flags
    out["osm_structure_count"] = counts
    out["osm_structure_classes"] = classes
    supported = out[out["osm_structure_supported"]].copy()
    return out, supported


def _polygonal(g):
    if g is None or g.is_empty:
        return MultiPolygon([])
    if isinstance(g, Polygon):
        return MultiPolygon([g])
    if isinstance(g, MultiPolygon):
        return g
    parts = []
    if hasattr(g, "geoms"):
        for p in g.geoms:
            if isinstance(p, Polygon):
                parts.append(p)
            elif isinstance(p, MultiPolygon):
                parts.extend(list(p.geoms))
    return MultiPolygon(parts)


def _window_for_bbox(src, bbox_wgs84):
    w, e, s, n = map(float, bbox_wgs84)
    tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    pts = [tr.transform(x, y) for x, y in ((w, s), (w, n), (e, s), (e, n))]
    xs, ys = zip(*pts, strict=False)
    win = from_bounds(min(xs), min(ys), max(xs), max(ys), src.transform)
    col0 = max(0, int(math.floor(win.col_off)) - 1)
    row0 = max(0, int(math.floor(win.row_off)) - 1)
    col1 = min(src.width, int(math.ceil(win.col_off + win.width)) + 1)
    row1 = min(src.height, int(math.ceil(win.row_off + win.height)) + 1)
    if col1 <= col0 or row1 <= row0:
        return None
    return rasterio.windows.Window(col0, row0, col1 - col0, row1 - row0)


@lru_cache(maxsize=32)
def _xy_transformer(source_wkt, target):
    """Reuse CRS operations; measure each window at its own center."""
    return Transformer.from_crs(source_wkt, target, always_xy=True)


def _pixel_metres(transform, crs, h, w):
    x, y = transform * (w / 2.0, h / 2.0)
    source_wkt = CRS.from_user_input(crs).to_wkt()
    ll = _xy_transformer(source_wkt, "EPSG:4326")
    lon, lat = ll.transform(x, y)
    zone = int(math.floor((lon + 180) / 6) + 1)
    metric = f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"
    tm = _xy_transformer(source_wkt, metric)
    a = tm.transform(x, y)
    b = tm.transform(*(transform * (w / 2.0 + 1, h / 2.0)))
    c = tm.transform(*(transform * (w / 2.0, h / 2.0 + 1)))
    dx = math.hypot(b[0] - a[0], b[1] - a[1])
    dy = math.hypot(c[0] - a[0], c[1] - a[1])
    return float(dx), float(dy)


def _fraction(mask, valid, size=3):
    n = ndi.uniform_filter(mask.astype("float32"), size=size, mode="constant", cval=0)
    d = ndi.uniform_filter(valid.astype("float32"), size=size, mode="constant", cval=0)
    return np.divide(n, d, out=np.zeros_like(n), where=d > 0)


def _read_union(path, layer, target_crs):
    """Read polygonal context without passing invalid projected rings to GEOS.

    Context polygons are valid when the coastal plan is written, but coordinate
    reprojection can introduce tiny self-crossings at near-coincident coastal
    vertices. Validate in the *raster CRS*, after reprojection, before union.
    Valid polygons are left unchanged. A context that loses all area is an
    error, never a reason to silently skip the shoreline refinement.
    """
    gdf = gpd.read_file(path, layer=layer)
    if gdf.empty:
        return MultiPolygon([])
    if gdf.crs is None:
        raise ValueError(f"{path}:{layer} has no CRS")
    if str(gdf.crs) != str(target_crs):
        gdf = gdf.to_crs(target_crs)
    repaired = 0
    polygons = []
    for geom in gdf.geometry.values:
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            geom = shapely.make_valid(geom)
            repaired += 1
        polygon = _polygonal(geom)
        if not polygon.is_empty:
            polygons.append(polygon)
    if not polygons:
        raise ValueError(
            f"{path}:{layer} contains no polygonal context after validation"
        )
    result = shapely.union_all(polygons)
    if result.is_empty or not result.is_valid:
        raise ValueError(f"{path}:{layer} did not produce valid polygonal context")
    if repaired:
        print(
            f"[coast_refine] repaired {repaired} projected context feature(s) "
            f"before union: {layer}",
            flush=True,
        )
    return result


def _poly_records(mask, transform, crs, kind, common, component_metric_masks=None):
    recs = []
    if not mask.any():
        return recs
    labels, n = ndi.label(mask, structure=FOUR)
    if not n:
        return recs
    counts = np.bincount(labels[mask], minlength=n + 1)
    metric_counts = {}
    for name, mm in (component_metric_masks or {}).items():
        arr = np.asarray(mm, dtype=bool) & mask
        metric_counts[name] = np.bincount(labels[arr], minlength=n + 1)
    for geo, lab in shapes(labels, mask=mask, transform=transform, connectivity=4):
        lab = int(lab)
        if lab <= 0:
            continue
        rec = dict(common)
        cells = int(counts[lab])
        rec.update(kind=kind, cells=cells, geometry=shape(geo))
        if cells > 0:
            for name, vals in metric_counts.items():
                rec[name] = float(vals[lab]) / float(cells)
        recs.append(rec)
    return recs


def extract_positive_evidence(
    *,
    raster_path,
    planning_gpkg,
    bbox_wgs84,
    accepted_geometry_wkt,
    accepted_geometry_srs="EPSG:4326",
    threshold=0.01,
    min_seaward_cells=4,
    box_id="box",
    product=None,
    source_url=None,
    source_title=None,
    publication_date=None,
    project=None,
    out_crs="EPSG:4326",
    elevation_band=1,
):
    """Extract effective positive TNM evidence only in the precomputed coastal ribbon.

    Returns a GeoDataFrame with two kinds of geometry:
      land_context_positive -- >threshold cells in the 100 m landward context
      seaward_positive      -- coherent >threshold components in the 250 m seaward search

    Cross-source connectivity is intentionally decided later, after all effective
    hierarchy pieces have been merged.
    """
    raster_path = Path(raster_path)
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")
        if not (1 <= int(elevation_band) <= src.count):
            raise ValueError(f"Invalid elevation band {elevation_band}: {raster_path}")
        win = _window_for_bbox(src, bbox_wgs84)
        if win is None:
            return gpd.GeoDataFrame(
                columns=["kind", "geometry"], geometry="geometry", crs=out_crs
            ), {"status": "NO_OVERLAP"}
        arr = src.read(int(elevation_band), window=win, masked=True)
        z = np.asarray(
            arr.filled(np.nan),
            dtype="float64" if arr.dtype == np.float64 else "float32",
        )
        valid = (~np.ma.getmaskarray(arr)) & np.isfinite(z)
        del arr
        raster_valid = valid.copy()
        t = src.window_transform(win)
        h, w = z.shape
        dx, dy = _pixel_metres(t, src.crs, h, w)

        claim_mask = np.ones(z.shape, dtype=bool)
        if accepted_geometry_wkt:
            claim = shapely.from_wkt(str(accepted_geometry_wkt))
            if claim.is_empty:
                return gpd.GeoDataFrame(
                    columns=["kind", "geometry"], geometry="geometry", crs=out_crs
                ), {"status": "EMPTY_ACCEPTED_CLAIM"}
            if str(src.crs) != str(accepted_geometry_srs):
                tr = Transformer.from_crs(
                    accepted_geometry_srs, src.crs, always_xy=True
                )
                claim = shp_transform(tr.transform, claim)
            claim_mask = rasterize(
                [(claim, 1)],
                out_shape=z.shape,
                transform=t,
                fill=0,
                dtype="uint8",
                all_touched=False,
            ).astype(bool)
            valid &= claim_mask

        land_geom = _read_union(planning_gpkg, "landward_context", src.crs)
        sea_geom = _read_union(planning_gpkg, "seaward_search", src.crs)
        land_zone = rasterize(
            [(land_geom, 1)],
            out_shape=z.shape,
            transform=t,
            fill=0,
            dtype="uint8",
            all_touched=False,
        ).astype(bool)
        sea_zone = rasterize(
            [(sea_geom, 1)],
            out_shape=z.shape,
            transform=t,
            fill=0,
            dtype="uint8",
            all_touched=False,
        ).astype(bool)

        positive = valid & (z > float(threshold))
        # Light coherence only. The baseline coastline has already reduced the
        # problem to a narrow ribbon; this rejects isolated single-pixel returns
        # without trying to infer geomorphology.
        coherent = positive & (_fraction(positive, valid, size=3) >= 0.5)
        land_positive = positive & land_zone
        sea_positive = coherent & sea_zone

        # Native-resolution minimum component screen for seaward additions.
        labs, nlab = ndi.label(sea_positive, structure=FOUR)
        if nlab:
            counts = np.bincount(labs[sea_positive], minlength=nlab + 1)
            sea_positive &= counts[labs] >= int(min_seaward_cells)
            del counts
        del labs

        # Diagnostic-only source-edge metrics.  These do not change acceptance.
        # They record whether a positive seaward component is dominated by the
        # source raster's invalid/nodata edge, the effective TNM claim boundary,
        # or the physical outer edge of the raster dataset.  This supports
        # evidence-driven investigation of TNM edge artifacts without introducing
        # an unvalidated rejection rule.
        nodata_adj = positive & ndi.binary_dilation(~raster_valid, structure=FOUR)
        claim_adj = positive & ndi.binary_dilation(~claim_mask, structure=FOUR)
        dataset_edge = np.zeros(z.shape, dtype=bool)
        if int(win.row_off) <= 0:
            dataset_edge[0, :] = True
        if int(win.col_off) <= 0:
            dataset_edge[:, 0] = True
        if int(win.row_off + win.height) >= int(src.height):
            dataset_edge[-1, :] = True
        if int(win.col_off + win.width) >= int(src.width):
            dataset_edge[:, -1] = True
        dataset_edge &= positive

        common = {
            "box_id": str(box_id),
            "product": None if product is None else str(product),
            "source_url": None if source_url is None else str(source_url),
            "source_title": None if source_title is None else str(source_title),
            "publication": None if publication_date is None else str(publication_date),
            "project": None if project is None else str(project),
            "pixel_m": float(max(dx, dy)),
            "threshold_m": float(threshold),
            "raster": str(raster_path),
            "edge_metrics_version": 1,
        }
        recs = []
        recs.extend(
            _poly_records(land_positive, t, src.crs, "land_context_positive", common)
        )
        recs.extend(
            _poly_records(
                sea_positive,
                t,
                src.crs,
                "seaward_positive",
                common,
                component_metric_masks={
                    "nodata_edge_fraction": nodata_adj,
                    "claim_edge_fraction": claim_adj,
                    "raster_edge_fraction": dataset_edge,
                },
            )
        )
        if recs:
            gdf = gpd.GeoDataFrame(recs, geometry="geometry", crs=src.crs).to_crs(
                out_crs
            )
        else:
            gdf = gpd.GeoDataFrame(
                columns=list(common) + ["kind", "cells", "geometry"],
                geometry="geometry",
                crs=out_crs,
            )

        report = {
            "status": "OK",
            "raster": str(raster_path),
            "raster_crs": str(src.crs),
            "window": [
                int(win.col_off),
                int(win.row_off),
                int(win.width),
                int(win.height),
            ],
            "pixel_m": [dx, dy],
            "threshold_m": float(threshold),
            "accepted_claim_cells": int(claim_mask.sum()),
            "valid_claim_cells": int(valid.sum()),
            "land_context_positive_cells": int(land_positive.sum()),
            "seaward_positive_cells": int(sea_positive.sum()),
            "feature_count": int(len(gdf)),
        }
        return gdf, report


def classify_connectivity(
    evidence: gpd.GeoDataFrame, metric_crs, min_bridge_m=1.0, max_bridge_m=20.0
):
    """Classify seaward components reachable from positive land context.

    This preserves the v5.3 scientific rule exactly but replaces the repeated
    O(N^2)-style full scan + growing union loop with a spatial-index graph walk.

    A seaward component j is reachable when its own tolerance-buffer intersects
    any land-context geometry or any already-reachable seaward component.  That
    is equivalent to v5.3's ``geom_j.distance(reachable_union) <= tol_j`` rule.
    """
    if evidence.empty:
        empty = gpd.GeoDataFrame(
            columns=evidence.columns, geometry="geometry", crs=evidence.crs
        )
        return empty, empty, MultiPolygon([])

    em = evidence.to_crs(metric_crs)
    context = em[em["kind"] == "land_context_positive"].copy()
    sea = em[em["kind"] == "seaward_positive"].copy()
    if sea.empty or context.empty:
        return (
            sea.iloc[0:0].to_crs(evidence.crs),  # type: ignore[arg-type]
            sea.to_crs(evidence.crs),  # type: ignore[arg-type]
            MultiPolygon([]),
        )

    from collections import deque
    from shapely.strtree import STRtree

    sea_geoms = list(sea.geometry.values)
    buffered = []
    for _, row in sea.iterrows():
        tol = max(
            float(min_bridge_m),
            min(float(max_bridge_m), float(row.get("pixel_m", 1.0)) * 1.5),
        )
        buffered.append(row.geometry.buffer(tol))

    # The tree is built on candidate-specific buffers. Querying with a reachable
    # geometry therefore identifies exactly those candidates whose own tolerance
    # reaches that geometry. This preserves the asymmetric per-candidate tolerance
    # used by the previous implementation.
    tree = STRtree(buffered)
    accepted_pos = set()
    queue: deque[int] = deque()

    def add_hits(reachable_geom):
        # predicate='intersects' applies the exact geometry predicate after the
        # bounding-box query and is supported by the Shapely 2 runtime used here.
        for pos in tree.query(reachable_geom, predicate="intersects"):
            j = int(pos)
            if j not in accepted_pos:
                accepted_pos.add(j)
                queue.append(j)

    for geom in context.geometry.values:
        if geom is not None and not geom.is_empty:
            add_hits(geom)

    while queue:
        pos = queue.popleft()
        geom = sea_geoms[pos]
        if geom is not None and not geom.is_empty:
            add_hits(geom)

    accepted_positions = sorted(accepted_pos)
    detached_positions = [i for i in range(len(sea)) if i not in accepted_pos]
    accepted = (
        sea.iloc[accepted_positions].copy()
        if accepted_positions
        else sea.iloc[0:0].copy()
    )
    detached = (
        sea.iloc[detached_positions].copy()
        if detached_positions
        else sea.iloc[0:0].copy()
    )
    accepted_out = accepted.to_crs(evidence.crs)  # type: ignore[arg-type]
    detached_out = detached.to_crs(evidence.crs)  # type: ignore[arg-type]
    accepted_union = (
        shapely.union_all(accepted_out.geometry.values)
        if not accepted_out.empty
        else MultiPolygon([])
    )
    return accepted_out, detached_out, accepted_union


# Reviewed hydroflattened-water artifact screen (v6.5).
# These are scientific defaults, intentionally not exposed as production knobs.
HYDROFLAT_SEED_WINDOW_M = 15.0
HYDROFLAT_SEED_MAX_RELIEF_M = 0.02
HYDROFLAT_MIN_SEED_AREA_M2 = 400.0
HYDROFLAT_MAX_NEIGHBOR_GRADE = 0.10
HYDROFLAT_MAX_NEIGHBOR_STEP_M = 0.50
HYDROFLAT_OSM_MIN_BUFFER_M = 2.0
HYDROFLAT_OSM_PIXEL_BUFFER_FRACTION = 1.5
HYDROFLAT_OSM_MAX_BUFFER_M = 20.0


def _max_four_neighbor_step_bad(z, positive, *, dx_m, dy_m, max_grade, max_step_m):
    """Return cells participating in an excessive 4-neighbor elevation step.

    A transition is considered smooth only when every positive-to-positive
    4-neighbor step stays below both the physical grade limit and an absolute
    step cap.  Marking both sides of a sharp edge prevents propagation from a
    flat-water seed across a real raised narrow structure such as a jetty.
    """
    bad = np.zeros(positive.shape, dtype=bool)
    xlim = min(float(max_step_m), float(max_grade) * float(dx_m))
    ylim = min(float(max_step_m), float(max_grade) * float(dy_m))

    if z.shape[1] > 1:
        both = positive[:, 1:] & positive[:, :-1]
        diff = np.abs(z[:, 1:] - z[:, :-1])
        hit = both & (diff > xlim)
        bad[:, 1:] |= hit
        bad[:, :-1] |= hit
    if z.shape[0] > 1:
        both = positive[1:, :] & positive[:-1, :]
        diff = np.abs(z[1:, :] - z[:-1, :])
        hit = both & (diff > ylim)
        bad[1:, :] |= hit
        bad[:-1, :] |= hit
    return bad


def _nonplanar_terrain(z, positive, *, dx_m, dy_m, window_m, tolerance_m):
    """Detect curvature at the existing physical seed-window scale.

    Compare a cell with the midpoint of two opposite samples. A constant or
    planar ramp has zero residual regardless of its grade. Only fully positive
    three-point profiles count; nodata and a candidate boundary provide no
    curvature evidence. This is a terrain cue, not proof of a beach.
    """
    curved = np.zeros(z.shape, dtype=bool)
    for axis, spacing in ((0, dy_m), (1, dx_m)):
        half = max(1, int(math.ceil(window_m / (2.0 * spacing))))
        if z.shape[axis] <= 2 * half:
            continue
        mid = [slice(None), slice(None)]
        low = mid.copy()
        high = mid.copy()
        mid[axis] = slice(half, -half)
        low[axis] = slice(None, -2 * half)
        high[axis] = slice(2 * half, None)
        mid, low, high = tuple(mid), tuple(low), tuple(high)
        complete = positive[mid] & positive[low] & positive[high]
        residual = np.abs(z[mid] - (z[low] + z[high]) * 0.5)
        curved[mid] |= complete & (residual > float(tolerance_m))
    return curved


def _incoherent_profile_windows(
    z,
    positive,
    *,
    half_x,
    half_y,
    max_seed_relief_m,
    max_explained_fraction,
    min_cells=None,
    partial_explained_fraction=None,
    partial_eligible=None,
):
    """Bounded-memory local plane-fit residual screen on positive candidate cells.

    The 512-row slabs overlap by the entire window half-height, so no seam is
    introduced by the performance optimization. With min_cells, incomplete
    windows use a masked least-squares plane over observed positive cells.
    Missing values never supply elevations. Rank-deficient or inadequately
    supported windows remain unknown rather than becoming wave evidence.
    """
    nx, ny = 2 * half_x + 1, 2 * half_y + 1
    result = np.zeros(positive.shape, dtype=bool)
    if min_cells is None and (z.shape[0] < ny or z.shape[1] < nx):
        return result
    x = np.arange(-half_x, half_x + 1, dtype="float64")
    y = np.arange(-half_y, half_y + 1, dtype="float64")
    vx, vy = float(np.mean(x * x)), float(np.mean(y * y))
    for start in range(0, positive.shape[0], 512):
        stop = min(positive.shape[0], start + 512)
        top = max(0, start - half_y)
        bottom = min(positive.shape[0], stop + half_y)
        valid = positive[top:bottom]
        complete = ndi.minimum_filter(
            valid.astype("uint8"), size=(ny, nx), mode="constant", cval=0
        ).astype(bool)
        if min_cells is None and not complete.any():
            continue
        zz = np.asarray(np.where(valid, z[top:bottom], 0.0), dtype="float64")
        # Exact centred least-squares plane on a complete rectangular window.
        # Explained variance = slope_x**2*variance(x) + slope_y**2*variance(y).
        sx = ndi.correlate1d(zz, x / (nx * vx), axis=1, mode="constant", cval=0)
        sx = ndi.uniform_filter1d(sx, size=ny, axis=0, mode="constant", cval=0)
        sy = ndi.correlate1d(zz, y / (ny * vy), axis=0, mode="constant", cval=0)
        sy = ndi.uniform_filter1d(sy, size=nx, axis=1, mode="constant", cval=0)
        mean = ndi.uniform_filter(zz, size=(ny, nx), mode="constant", cval=0)
        variance = np.maximum(
            0.0,
            ndi.uniform_filter(zz * zz, size=(ny, nx), mode="constant", cval=0)
            - mean * mean,
        )
        explained = sx * sx * vx + sy * sy * vy
        # Flat water is handled by the existing plateau-seed owner. This test
        # deliberately targets nonflat waves that pass local terrain relief.
        suspect = (
            complete
            & (variance > float(max_seed_relief_m) ** 2)
            & (explained < float(max_explained_fraction) * variance)
        )
        if min_cells is not None:
            # Centered coordinate moments avoid large absolute-coordinate
            # cancellation. Box filters preserve bounded slab memory.
            vf = valid.astype("float64")
            count = ndi.uniform_filter(vf, size=(ny, nx), mode="constant", cval=0) * (
                nx * ny
            )
            partial = valid & ~complete & (count >= int(min_cells) - 1e-8)
            if partial_eligible is not None:
                partial &= partial_eligible[top:bottom]
            if partial.any():
                # Only fit column blocks containing an incomplete supported
                # window. Full halos make this identical at block boundaries;
                # extra moment arrays remain bounded to ~256 columns.
                for col in range(0, valid.shape[1], 256):
                    end = min(valid.shape[1], col + 256)
                    if not partial[:, col:end].any():
                        continue
                    left, right = (
                        max(0, col - half_x),
                        min(valid.shape[1], end + half_x),
                    )
                    center = slice(col - left, end - left)
                    vv, zt = vf[:, left:right], zz[:, left:right]
                    denom = np.maximum(count[:, left:right], 1.0)

                    def moment(a, xweights=None, yweights=None):
                        if xweights is None:
                            a = (
                                ndi.uniform_filter1d(
                                    a, size=nx, axis=1, mode="constant", cval=0
                                )
                                * nx
                            )
                        else:
                            a = ndi.correlate1d(
                                a, xweights, axis=1, mode="constant", cval=0
                            )
                        if yweights is None:
                            a = (
                                ndi.uniform_filter1d(
                                    a, size=ny, axis=0, mode="constant", cval=0
                                )
                                * ny
                            )
                        else:
                            a = ndi.correlate1d(
                                a, yweights, axis=0, mode="constant", cval=0
                            )
                        return a / denom  # noqa: B023 - Called synchronously before this loop advances.

                    mx, my, mz = moment(vv, x), moment(vv, yweights=y), moment(zt)
                    xx = np.maximum(0.0, moment(vv, x * x) - mx * mx)
                    yy = np.maximum(0.0, moment(vv, yweights=y * y) - my * my)
                    xy = moment(vv, x, y) - mx * my
                    xz = moment(zt, x) - mx * mz
                    yz = moment(zt, yweights=y) - my * mz
                    var = np.maximum(0.0, moment(zt * zt) - mz * mz)
                    det = xx * yy - xy * xy
                    evaluated = partial[:, left:right] & (
                        det > 1e-10 * np.maximum(xx * yy, 1.0)
                    )
                    fitted = np.zeros_like(var)
                    np.divide(
                        xz * xz * yy + yz * yz * xx - 2 * xz * yz * xy,
                        det,
                        out=fitted,
                        where=evaluated,
                    )
                    fraction = (
                        max_explained_fraction
                        if partial_explained_fraction is None
                        else partial_explained_fraction
                    )
                    decision = (
                        evaluated
                        & (var > float(max_seed_relief_m) ** 2)
                        & (fitted < float(fraction) * var)
                    )
                    suspect[:, col:end] |= decision[:, center]
        suspect &= valid
        result[start:stop] = suspect[start - top : stop - top]
    return result


def _coherent_long_profile(
    z, positive, *, dx_m, dy_m, window_m, min_relief_m, explained_fraction=0.85
):
    """Protect monotone beach profiles from false broad-wave classifications.

    A curved beach fails a local plane fit despite having a persistent slope;
    waves instead reverse sign repeatedly. Compare signed change against
    total change along each raster axis over the *same* physical window as
    the broad wave screen (4x the hydroflat seed window). Require actual
    positive adjacent samples over at least half the window and more than
    the existing hydroflat flatness tolerance of observed relief. Require
    at least four observed neighbour pairs; a couple of coarse-grid cells
    cannot establish a stable profile. Neither
    missing data nor profile edges are filled or treated as elevations.

    Work in 512-row slabs with a full filter halo, preserving edge decisions
    without allocating multiple full-raster float64 moment arrays.
    """
    supported = np.zeros(positive.shape, dtype=bool)
    height = z.shape[0]
    for axis, cell_m in ((0, dy_m), (1, dx_m)):
        half = max(1, int(math.ceil(window_m / (2.0 * cell_m))))
        if z.shape[axis] < 2:
            continue
        size = 2 * half + 1
        for start in range(0, height, 512):
            stop = min(height, start + 512)
            top = max(0, start - (half + 1 if axis == 0 else 0))
            bottom = min(height, stop + (half + 1 if axis == 0 else 0))
            values = np.asarray(z[top:bottom], dtype="float64")
            present = positive[top:bottom]
            neighbours = np.take(
                present, range(1, present.shape[axis]), axis=axis
            ) & np.take(present, range(present.shape[axis] - 1), axis=axis)
            delta = np.where(neighbours, np.diff(values, axis=axis), 0.0)

            def total(a):
                return (
                    ndi.uniform_filter1d(
                        a,
                        size=size,
                        axis=axis,  # noqa: B023 - Called synchronously before this loop advances.
                        mode="constant",
                        cval=0,
                    )
                    * size
                )  # noqa: B023 - Called synchronously before this loop advances.

            signed = total(delta)
            variation = total(np.abs(delta))
            count = total(neighbours.astype("float64"))
            sl = [slice(None), slice(None)]
            sl[axis] = slice(1, None)
            candidate = np.zeros_like(present, dtype=bool)
            candidate[tuple(sl)] = (
                (count >= max(4, half) - 1e-8)
                & (variation > float(min_relief_m))
                & (np.abs(signed) >= float(explained_fraction) * variation)
            )
            supported[start:stop] |= candidate[start - top : stop - top]
    return supported & positive


# Experimental entry-height overshoot limit on an observed
# source-datum zero profile. It is separate from the unchanged terrain relief,
# flat-seed and bridging tests; no elevation samples are smoothed or rewritten.
_WAVE_MAX_REVERSE_EXCURSION_M = 0.40


def _wave_context_support(
    src,
    candidate_geometry,
    window,
    *,
    dx_m,
    dy_m,
    threshold_m,
    seed_window_m,
    relief_tolerance_m,
    _trace=None,
):
    """Corroborate a bounded observed profile across source-datum zero.

    Assess the complete positive proposal run from its landward entry.
    Bound overshoot above that entry height; an intermediate terrain hollow
    must not reset the height reference and admit only a disconnected tail.
    Positive proposal and negative observations must each supply their own
    grade on one complete profile. This only exempts the broad-wave screen:
    flat seeds, local relief, class and inland connectivity still apply.
    """
    hx = max(4, int(math.ceil(5 * seed_window_m / dx_m)))
    hy = max(4, int(math.ceil(5 * seed_window_m / dy_m)))
    c0, r0 = max(0, int(window.col_off) - hx), max(0, int(window.row_off) - hy)
    c1 = min(src.width, int(window.col_off + window.width) + hx)
    r1 = min(src.height, int(window.row_off + window.height) + hy)
    expanded = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
    arr = src.read(1, window=expanded, masked=True)
    z = np.asarray(arr.filled(np.nan), dtype="float64")
    observed = ~np.ma.getmaskarray(arr) & np.isfinite(z)
    inside = rasterize(
        [(candidate_geometry, 1)],
        out_shape=z.shape,
        transform=src.window_transform(expanded),
        dtype="uint8",
    ).astype(bool)
    positive = observed & inside & (z > threshold_m)
    negative = observed & (z < threshold_m - 0.02)
    support = np.zeros(z.shape, bool)
    top, left = int(window.row_off) - r0, int(window.col_off) - c0
    crop = (slice(top, top + int(window.height)), slice(left, left + int(window.width)))
    if not negative.any():
        return support[crop]
    # Experimental: add both native-cell diagonals. These are discrete
    # observed sample paths, not rotated/resampled or gap-filled elevations.
    # Every existing grade, ownership, zero-transition and oscillation guard
    # applies independently on each complete path; masks are never dilated.
    corridor = observed.copy()
    corridor[1:] &= observed[:-1]
    corridor[:-1] &= observed[1:]
    corridor[:, 1:] &= observed[:, :-1]
    corridor[:, :-1] &= observed[:, 1:]
    corridor[[0, -1], :] = False
    corridor[:, [0, -1]] = False
    for axis, spacing in (
        (0, dy_m),
        (1, dx_m),
        (2, math.hypot(dx_m, dy_m)),
        (3, math.hypot(dx_m, dy_m)),
    ):
        if axis >= 2:
            height, width = z.shape
            offset = np.arange(1 - height, width)[:, None]
            step = np.arange(min(height, width))[None, :]
            rows = np.maximum(0, -offset) + step
            columns = np.maximum(0, offset) + step
            present = (rows < height) & (columns < width)
            if axis == 3:
                columns = width - 1 - columns
            row_index = np.clip(rows, 0, height - 1)
            col_index = np.clip(columns, 0, width - 1)
        for reverse in (False, True):

            def orient(a):
                if axis >= 2:
                    a = np.where(
                        present,
                        a[row_index, col_index],
                        False if a.dtype == bool else np.nan,
                    )
                else:
                    a = a.T if axis == 0 else a
                return a[:, ::-1] if reverse else a

            zz, vv, pp, nn = map(orient, (z, observed, positive, negative))
            if axis >= 2:
                # A diagonal cannot jump between the corners of NoData cells.
                # Require observed rook neighbours along the full path too.
                vv = orient(corridor)
            width = zz.shape[1]
            indices = np.broadcast_to(np.arange(width), zz.shape)
            anchor = np.minimum.accumulate(
                np.where(nn, indices, width)[:, ::-1], axis=1
            )[:, ::-1]
            begin = np.maximum.accumulate(np.where(~pp, indices + 1, 0), axis=1)
            # A sliding four-cell prefix can admit only a tail of one proposal.
            start = begin.copy()
            # Preserve the original broad interval wherever observations exist.
            # A gap farther offshore may end corroboration only after the
            # minimum continuous negative-water interval is already observed.
            minimum_end = anchor + max(4, int(math.ceil(seed_window_m / spacing)))
            # A diagonal must inspect at least the original axial reach in
            # each grid component, rather than shortening oscillation context.
            context_spacing = min(dx_m, dy_m) if axis >= 2 else spacing
            broad_end = np.maximum(
                minimum_end, start + int(math.ceil(4 * seed_window_m / context_spacing))
            )
            next_gap = np.minimum.accumulate(
                np.where(~vv, indices, width)[:, ::-1], axis=1
            )[:, ::-1]
            first_gap = np.take_along_axis(
                next_gap, np.minimum(anchor, width - 1), axis=1
            )
            end = (
                broad_end
                if axis >= 2
                else np.maximum(minimum_end, np.minimum(broad_end, first_gap - 1))
            )
            # New directions require the whole broad interval. A diagonal
            # cannot escape an observed return by ending at a raster edge.
            eligible = (
                pp & (end < width) & ((anchor - indices) * spacing <= 4 * seed_window_m)
            )
            start, end = np.clip(start, 0, width - 1), np.clip(end, 0, width - 1)

            def interval(a, *, edges=False):
                prefix = np.pad(
                    np.cumsum(a.astype("float64"), axis=1), ((0, 0), (1, 0))
                )
                return np.take_along_axis(prefix, end + 1, axis=1) - np.take_along_axis(
                    prefix, start + int(edges), axis=1
                )

            eligible &= interval(vv) >= end - start + 1 - 1e-8
            eligible &= (
                interval(vv & ((zz <= threshold_m) | pp)) >= end - start + 1 - 1e-8
            )
            conditions = (
                {"bounded_observed_owned_path": eligible.copy()}
                if _trace is not None
                else {}
            )
            for label, domain, minimum in (
                ("full", vv, 8),
                ("positive", pp, 4),
                ("negative", nn, 4),
            ):
                pair = np.zeros(zz.shape, bool)
                delta = np.zeros(zz.shape, "float64")
                pair[:, 1:] = domain[:, 1:] & domain[:, :-1]
                delta[:, 1:] = np.where(pair[:, 1:], zz[:, 1:] - zz[:, :-1], 0.0)
                enough = interval(pair, edges=True) >= minimum - 1e-8
                falling = -interval(delta, edges=True) > relief_tolerance_m
                eligible &= enough & falling
                if _trace is not None:
                    conditions[label + "_grade"] = eligible.copy()
                    conditions[label + "_enough_pairs"] = pp & enough
                    conditions[label + "_falling"] = pp & falling
            # Negative corroboration is uninterrupted, excluding multiple
            # sign crossings and borrowing separate wave troughs across land.
            prefix = np.pad(np.cumsum(nn.astype("int32"), axis=1), ((0, 0), (1, 0)))
            negative_count = np.take_along_axis(
                prefix, end + 1, axis=1
            ) - np.take_along_axis(prefix, np.minimum(anchor, width), axis=1)
            eligible &= negative_count == end - anchor + 1
            if _trace is not None:
                conditions["continuous_negative_side"] = eligible.copy()
            # Evaluate the full proposal against one landward entry height.
            # Several small steps cannot hide a peak above that entry limit. Work only on eligible profiles to bound I/O
            # and temporary memory; missing samples are never filled.
            # Inspect an additional broad window landward for oscillations.
            # It can veto a release but cannot supply the proposal's grade.
            # Missing context retains the original wave decision.
            earlier = start - int(math.ceil(4 * seed_window_m / context_spacing))
            extended = np.clip(earlier, 0, width - 1)
            prefix_valid = np.pad(
                np.cumsum(vv.astype("int32"), axis=1), ((0, 0), (1, 0))
            )
            observed_context = np.take_along_axis(
                prefix_valid, end + 1, axis=1
            ) - np.take_along_axis(prefix_valid, extended, axis=1)
            eligible &= (earlier >= 0) & (observed_context == end - extended + 1)
            if _trace is not None:
                conditions["complete_oscillation_context"] = eligible.copy()
            # A previous negative lobe would make this a repeated water
            # crossing, not a single corroborated land-to-water interface.
            previous_negative = np.take_along_axis(
                prefix, np.minimum(anchor, width), axis=1
            ) - np.take_along_axis(prefix, extended, axis=1)
            eligible &= previous_negative == 0
            if _trace is not None:
                conditions["single_zero_transition"] = eligible.copy()
            rr, cc = np.nonzero(eligible)
            accepted = np.zeros(zz.shape, bool)
            if rr.size:
                # Only the positive proposal's path is screened for entry-height
                # excursions. Submerged bed relief is valid water evidence,
                # not evidence that the positive bench is a wave return.
                first, last = start[rr, cc], anchor[rr, cc] - 1
                entry_height = zz[rr, first]
                rise = np.zeros(rr.size)
                for offset in range(int(np.max(last - first)) + 1):
                    at = np.minimum(first + offset, last)
                    value = zz[rr, at]
                    rise = np.maximum(rise, value - entry_height)
                accepted[rr, cc] = rise <= _WAVE_MAX_REVERSE_EXCURSION_M + 1e-12

            def restore(a):
                a = a[:, ::-1] if reverse else a
                if axis >= 2:
                    result = np.zeros(z.shape, dtype=bool)
                    result[row_index[present], col_index[present]] = a[present]
                    return result
                return a.T if axis == 0 else a

            support |= restore(accepted)
            if _trace is not None:
                conditions["supported"] = accepted
                _trace.append(
                    {
                        "axis": axis,
                        "reverse": reverse,
                        "masks": {
                            key: restore(value)[crop]
                            for key, value in conditions.items()
                        },
                    }
                )
    return support[crop]


def _broad_wave_surface(
    z,
    positive,
    *,
    dx_m,
    dy_m,
    seed_window_m,
    min_seed_area_m2,
    max_seed_relief_m,
    max_neighbor_grade=0.1,
    max_neighbor_step_m=0.5,
    additional_coherent=None,
):
    """Find broad positive wave returns without stable terrain-profile support.

    A wave surface has substantial local relief but no persistent directional
    elevation trend. First identify physically broad incoherent regions at 4x
    the existing hydroflat scale, then extend only through adjacent short-scale
    incoherent cells. A coherent beach grade stops propagation. This procedure
    does not assume a shoreline elevation, provider, wave height or offshore
    distance; unknown/narrow candidates retain the previous review route.
    """
    min_cells = max(
        1, int(math.ceil(float(min_seed_area_m2) / (float(dx_m) * float(dy_m))))
    )
    if np.count_nonzero(positive) < min_cells:
        return np.zeros(positive.shape, dtype=bool)
    smooth = ~_max_four_neighbor_step_bad(
        z,
        positive,
        dx_m=dx_m,
        dy_m=dy_m,
        max_grade=max_neighbor_grade,
        max_step_m=max_neighbor_step_m,
    )
    # 4x the existing hydroflat window: enough width to distinguish an
    # oscillation from a sustained terrain grade without a new configuration.
    broad = _incoherent_profile_windows(
        z,
        positive,
        half_x=max(1, int(math.ceil(2.0 * seed_window_m / dx_m))),
        half_y=max(1, int(math.ceil(2.0 * seed_window_m / dy_m))),
        max_seed_relief_m=max_seed_relief_m,
        max_explained_fraction=0.5,
        min_cells=min_cells,
        # An incomplete broad window must retain the existing short-profile
        # coherence standard; a partly fitted wave slope is not terrain proof.
        partial_explained_fraction=0.85,
        partial_eligible=smooth,
    )
    coherent = _coherent_long_profile(
        z,
        positive,
        dx_m=dx_m,
        dy_m=dy_m,
        window_m=4.0 * seed_window_m,
        min_relief_m=max_seed_relief_m,
    )
    if additional_coherent is not None:
        coherent |= additional_coherent & positive
    labs, n = ndi.label(broad & ~coherent, structure=FOUR)
    if not n:
        return np.zeros(positive.shape, dtype=bool)
    counts = np.bincount(labs[broad], minlength=n + 1)
    good = np.flatnonzero(counts >= min_cells)
    good = good[good != 0]
    if not good.size:
        return np.zeros(positive.shape, dtype=bool)
    seeds = np.isin(labs, good)
    # Let the existing 15 m terrain scale decide the wave surface at its
    # boundary: a strong directional terrain grade explains >=85% of local
    # variance and blocks spread. Full positive windows are required here too.
    short = _incoherent_profile_windows(
        z,
        positive,
        half_x=max(1, int(math.ceil(seed_window_m / (2.0 * dx_m)))),
        half_y=max(1, int(math.ceil(seed_window_m / (2.0 * dy_m)))),
        max_seed_relief_m=max_seed_relief_m,
        max_explained_fraction=0.85,
        min_cells=4,
        partial_eligible=smooth,
    )
    return ndi.binary_propagation(
        seeds,
        mask=(short | seeds) & ~coherent,
        structure=FOUR,
    )


def _land_connected_raised_plateau(
    z, positive, inland, *, dx_m, dy_m, max_grade, max_step_m, max_relief_m
):
    """Veto wave rejection only on a sharp, land-attached level ridge.

    A raised, flat-topped jetty has a strong step down on each flank yet its
    top remains at the *same observed elevation* as terrain inland. A wave
    crest offshore must not gain that exemption simply by having sharp steps.
    Edges must be supported by valid positive source cells on both sides;
    the nearest raised flank height must agree within the existing hydroflat
    relief tolerance; and a connected plateau must actually touch inland
    positive evidence. No new synthetic elevations, thresholds, or OSM labels.
    """
    if not inland.any() or not positive.any():
        return np.zeros_like(positive)
    valid_positive = positive | inland
    high = np.zeros(positive.shape, dtype=bool)
    for axis, spacing in ((0, dy_m), (1, dx_m)):
        if z.shape[axis] < 2:
            continue
        sl0 = [slice(None), slice(None)]
        sl1 = sl0.copy()
        sl0[axis] = slice(None, -1)
        sl1[axis] = slice(1, None)
        left, right = tuple(sl0), tuple(sl1)
        a, b = z[left], z[right]
        step = (
            valid_positive[left]
            & valid_positive[right]
            & (np.abs(a - b) > min(max_step_m, max_grade * spacing))
        )
        high[left] |= step & (a > b)
        high[right] |= step & (b > a)
    if not high.any():
        return np.zeros_like(positive)
    nearest = ndi.distance_transform_edt(
        ~high, sampling=(dy_m, dx_m), return_distances=False, return_indices=True
    )
    # Follow only source-observed cells at an elevated flank's matching height;
    # a broad low water plateau across the sharp edge cannot qualify.
    plateau = valid_positive & (np.abs(z - z[tuple(nearest)]) <= max_relief_m)
    labels, n = ndi.label(plateau, structure=FOUR)
    if not n:
        return np.zeros_like(positive)
    connected = np.unique(labels[inland & plateau])
    connected = connected[connected != 0]
    return positive & np.isin(labels, connected)


def screen_hydroflat_transition_fringe(
    evidence: gpd.GeoDataFrame,
    *,
    metric_crs,
    osm_evidence=None,
    seed_window_m=HYDROFLAT_SEED_WINDOW_M,
    max_seed_relief_m=HYDROFLAT_SEED_MAX_RELIEF_M,
    min_seed_area_m2=HYDROFLAT_MIN_SEED_AREA_M2,
    max_neighbor_grade=HYDROFLAT_MAX_NEIGHBOR_GRADE,
    max_neighbor_step_m=HYDROFLAT_MAX_NEIGHBOR_STEP_M,
    osm_min_buffer_m=HYDROFLAT_OSM_MIN_BUFFER_M,
    osm_pixel_buffer_fraction=HYDROFLAT_OSM_PIXEL_BUFFER_FRACTION,
    osm_max_buffer_m=HYDROFLAT_OSM_MAX_BUFFER_M,
    min_valid_cells=4,
):
    """Remove a demonstrated hydroflattened-water/TIN transition failure mode.

    A demonstrated hydroflat/TIN-fringe regression is not a
    raster edge or nodata artifact: the effective 1 m TNM raster is valid and
    positive throughout the bad wedge.  It contains a broad nearly constant
    +0.4 m water surface connected landward through a very smooth interpolated
    ramp.  That ramp satisfies the existing >+0.01 m and local-relief tests.

    This screen is intentionally narrow:

    1. Find a *broad* positive component whose 15 m local elevation range is no
       more than 0.02 m over at least 400 m2.  Such a component is a water-surface
       seed; an ordinary slope alone cannot trigger the screen.
    2. Starting only from those seeds, propagate through positive cells whose
       four-neighbor elevation steps remain no steeper than 10% and no larger
       than 0.5 m.  This identifies the smooth TIN/interpolation fringe tied to
       the flat water surface.
    3. Beyond one seed-window distance from a seed, nonplanar terrain blocks
       propagation. At the same window scale, a center elevation differing
       from opposite-sample interpolation by more than the existing flatness
       tolerance is a curvature cue. The immediate plateau/ramp join stays
       screened. This is an experimental terrain cue, not independent truth.
    4. Independently screen broad oscillatory positive water surfaces whose
       short-scale relief is not supported by a coherent longer-scale terrain
       grade. Such wave returns otherwise pass the existing terrain-core route.
    5. If optional OSM Mask Helper coastal-structure evidence is supplied, a
       small source-resolution-aware buffer around mapped structures blocks this
       propagation.  OSM never creates accepted terrain: protected cells still
       must pass the normal positive-elevation, local-terrain, and inland-
       connectivity tests later.

    A single dissolved water polygon remains fully supported.  Missing optional
    OSM evidence does not prevent the TNM-only screen from operating.

    Returns ``(screened_evidence, flat_seeds, rejected_fringe,
    osm_protected, stats)``.  Land-context evidence is passed through unchanged.
    """
    if evidence is None or evidence.empty:
        empty = (
            evidence.copy()
            if evidence is not None
            else gpd.GeoDataFrame(
                columns=["kind", "geometry"], geometry="geometry", crs="EPSG:4326"
            )
        )
        return (
            empty,
            empty.copy(),
            empty.copy(),
            empty.copy(),
            {
                "candidate_cells": 0,
                "seed_cells": 0,
                "rejected_cells": 0,
                "retained_cells": 0,
                "osm_protected_cells": 0,
                "rows_screened": 0,
                "rows_with_seed": 0,
            },
        )

    context = evidence[evidence["kind"] == "land_context_positive"].copy()
    sea = evidence[evidence["kind"] == "seaward_positive"].copy()
    if sea.empty:
        empty = sea.copy()
        return (
            evidence.copy(),
            empty,
            empty.copy(),
            empty.copy(),
            {
                "candidate_cells": 0,
                "seed_cells": 0,
                "rejected_cells": 0,
                "retained_cells": 0,
                "osm_protected_cells": 0,
                "rows_screened": 0,
                "rows_with_seed": 0,
            },
        )

    kept_rows = []
    seed_rows: list = []
    rejected_rows: list = []
    protected_rows = []
    stats = {
        "candidate_cells": 0,
        "seed_cells": 0,
        "rejected_cells": 0,
        "retained_cells": 0,
        "osm_protected_cells": 0,
        "rows_screened": 0,
        "rows_with_seed": 0,
        "nonplanar_barrier_cells": 0,
        "legacy_rejected_cells": 0,
        "wave_surface_rejected_cells": 0,
        "seed_window_m": float(seed_window_m),
        "max_seed_relief_m": float(max_seed_relief_m),
        "min_seed_area_m2": float(min_seed_area_m2),
        "max_neighbor_grade": float(max_neighbor_grade),
        "max_neighbor_step_m": float(max_neighbor_step_m),
    }

    # PHASE 1 — Gather optional mapped-structure protection (never creates terrain).
    # Optional OSM evidence is only a protection mask for this specific screen.
    # Query in a metric CRS so buffers are physical and independent of source CRS.
    osm_metric = None
    osm_tree = None
    osm_geoms = []
    if osm_evidence is not None and not osm_evidence.empty:
        osm_metric = osm_evidence.to_crs(metric_crs).copy()
        osm_geoms = [
            g for g in osm_metric.geometry.values if g is not None and not g.is_empty
        ]
        if osm_geoms:
            osm_tree = shapely.STRtree(osm_geoms)

    # PHASE 2 — Group positive coastal candidates by original native raster.
    groups: dict = {}
    for idx, row in sea.iterrows():
        rp = row.get("raster")
        groups.setdefault(str(rp) if rp else "", []).append((idx, row))

    def frame(rows):
        if not rows:
            return gpd.GeoDataFrame(
                columns=list(sea.columns) + ["hydroflat_status"],
                geometry="geometry",
                crs=sea.crs,
            )
        # ``iterrows`` yields Series; polygonization yields dicts. Normalize
        # before constructing a GeoDataFrame (Pandas 2.3+ cannot mix both).
        records = [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in rows]
        return gpd.GeoDataFrame(records, geometry="geometry", crs=sea.crs)

    def append_components(
        cells, *, transform, crs, kind, common, status, destination, review_kind=None
    ):
        """Publish one polygonized screening class and its optional review copy.

        A single path preserves the same CRS, fields and status for hydroflat
        seeds, retained terrain, rejected waves and protected structures.
        """
        records = _poly_records(cells, transform, crs, kind, common)
        if not records:
            return
        for record in records:
            record["hydroflat_status"] = status
        rows = gpd.GeoDataFrame(records, geometry="geometry", crs=crs).to_crs(sea.crs)
        destination.extend(rows.to_dict("records"))
        if review_kind is not None:
            review = rows.copy()
            review["kind"] = review_kind
            protected_rows.extend(review.to_dict("records"))

    for raster_name, items in groups.items():
        p = Path(raster_name) if raster_name else None
        if p is None or not p.is_file():
            # Fail closed with respect to this *new* filter: do not remove terrain
            # merely because the source raster cannot be reopened.  The existing
            # local-terrain screen will independently handle its missing raster.
            for _, row in items:
                rr = row.copy()
                rr["hydroflat_status"] = "MISSING_RASTER_UNCHANGED"
                kept_rows.append(rr)
            continue
        try:
            src = rasterio.open(p)
        except Exception:
            for _, row in items:
                rr = row.copy()
                rr["hydroflat_status"] = "OPEN_FAILED_UNCHANGED"
                kept_rows.append(rr)
            continue
        try:
            for _, row in items:
                stats["rows_screened"] += 1
                geom = row.geometry
                if geom is None or geom.is_empty:
                    rr = row.copy()
                    rr["hydroflat_status"] = "EMPTY_GEOMETRY_UNCHANGED"
                    kept_rows.append(rr)
                    continue

                gsrc = gpd.GeoSeries([geom], crs=sea.crs).to_crs(src.crs).iloc[0]
                pix = float(row.get("pixel_m", 1.0) or 1.0)
                if not math.isfinite(pix) or pix <= 0:
                    pix = 1.0
                half_cells = max(1, int(math.ceil((float(seed_window_m) / 2.0) / pix)))
                ksize = half_cells * 2 + 1
                pad_cells = half_cells + 2

                minx, miny, maxx, maxy = gsrc.bounds
                win = from_bounds(minx, miny, maxx, maxy, src.transform)
                col0 = max(0, int(math.floor(win.col_off)) - pad_cells)
                row0 = max(0, int(math.floor(win.row_off)) - pad_cells)
                col1 = min(
                    src.width, int(math.ceil(win.col_off + win.width)) + pad_cells
                )
                row1 = min(
                    src.height, int(math.ceil(win.row_off + win.height)) + pad_cells
                )
                if col1 <= col0 or row1 <= row0:
                    rr = row.copy()
                    rr["hydroflat_status"] = "NO_OVERLAP_UNCHANGED"
                    kept_rows.append(rr)
                    continue

                rw = rasterio.windows.Window(col0, row0, col1 - col0, row1 - row0)
                arr = src.read(1, window=rw, masked=True)
                z = np.asarray(
                    arr.filled(np.nan),  # type: ignore[attr-defined]
                    dtype="float64" if arr.dtype == np.float64 else "float32",
                )
                valid = (~np.ma.getmaskarray(arr)) & np.isfinite(z)
                rt = src.window_transform(rw)
                inside = rasterize(
                    [(gsrc, 1)],
                    out_shape=z.shape,
                    transform=rt,
                    fill=0,
                    dtype="uint8",
                    all_touched=False,
                ).astype(bool)
                threshold = float(row.get("threshold_m", 0.01))
                positive = valid & inside & (z > threshold)
                npos = int(positive.sum())
                stats["candidate_cells"] += npos
                if npos < int(min_valid_cells):
                    rr = row.copy()
                    rr["hydroflat_status"] = "TOO_FEW_CELLS_UNCHANGED"
                    kept_rows.append(rr)
                    stats["retained_cells"] += npos
                    continue

                dx, dy = _pixel_metres(rt, src.crs, z.shape[0], z.shape[1])
                pixel_area = max(1e-12, float(dx) * float(dy))

                # Optional structure protection.  Query only nearby OSM geometry,
                # buffer it in metres, and then rasterize in the source raster CRS.
                protected = np.zeros(z.shape, dtype=bool)
                if osm_tree is not None:
                    gm = gpd.GeoSeries([gsrc], crs=src.crs).to_crs(metric_crs).iloc[0]
                    tol = min(
                        float(osm_max_buffer_m),
                        max(
                            float(osm_min_buffer_m),
                            pix * float(osm_pixel_buffer_fraction),
                        ),
                    )
                    ids = np.asarray(
                        osm_tree.query(
                            gm.buffer(float(osm_max_buffer_m)), predicate="intersects"
                        ),
                        dtype=int,
                    )
                    if ids.size:
                        og = shapely.union_all([osm_geoms[int(i)] for i in ids]).buffer(
                            tol
                        )
                        ogs = (
                            gpd.GeoSeries([og], crs=metric_crs).to_crs(src.crs).iloc[0]
                        )
                        protected = rasterize(
                            [(ogs, 1)],
                            out_shape=z.shape,
                            transform=rt,
                            fill=0,
                            dtype="uint8",
                            all_touched=True,
                        ).astype(bool)
                        protected &= positive

                # Curved water waves have enough small-scale relief to pass the
                # normal terrain-core test. This separate broad/coherence screen
                # requires no water-surface elevation or source-specific label.
                context_support = _wave_context_support(
                    src,
                    gsrc,
                    rw,
                    dx_m=dx,
                    dy_m=dy,
                    threshold_m=threshold,
                    seed_window_m=seed_window_m,
                    relief_tolerance_m=max_seed_relief_m,
                )
                wave = _broad_wave_surface(
                    z,
                    positive & ~protected,
                    dx_m=dx,
                    dy_m=dy,
                    additional_coherent=context_support,
                    seed_window_m=seed_window_m,
                    min_seed_area_m2=min_seed_area_m2,
                    max_seed_relief_m=max_seed_relief_m,
                    max_neighbor_grade=max_neighbor_grade,
                    max_neighbor_step_m=max_neighbor_step_m,
                )
                # A sharp positive plateau must also connect to *observed*
                # inland elevation at matching height; sharp offshore waves
                # alone are not an exemption from the oscillatory-water test.
                if wave.any() and not context.empty:
                    inland_geoms = [
                        r.geometry
                        for _, r in context.iterrows()
                        if str(r.get("raster", "")) == raster_name
                        and r.geometry is not None
                        and not r.geometry.is_empty
                    ]
                    if inland_geoms:
                        inland_geom = gpd.GeoSeries(
                            inland_geoms, crs=context.crs
                        ).to_crs(src.crs)
                        inland = rasterize(
                            [(g, 1) for g in inland_geom],
                            out_shape=z.shape,
                            transform=rt,
                            fill=0,
                            dtype="uint8",
                        ).astype(bool)
                        inland &= valid & (z > threshold)
                        wave &= ~_land_connected_raised_plateau(
                            z,
                            positive & ~protected,
                            inland,
                            dx_m=dx,
                            dy_m=dy,
                            max_grade=max_neighbor_grade,
                            max_step_m=max_neighbor_step_m,
                            max_relief_m=max_seed_relief_m,
                        )
                wave_count = int(wave.sum())
                stats["wave_surface_rejected_cells"] += wave_count

                # PHASE 3 — Identify physically broad, nearly level hydroflat seeds.
                # Local flatness is computed only among positive candidate cells.
                hi = np.where(positive, z, -np.inf)
                lo = np.where(positive, z, np.inf)
                zmax = ndi.maximum_filter(hi, size=ksize, mode="constant", cval=-np.inf)
                zmin = ndi.minimum_filter(lo, size=ksize, mode="constant", cval=np.inf)
                count = ndi.uniform_filter(
                    positive.astype("float32"), size=ksize, mode="constant", cval=0.0
                ) * float(ksize * ksize)
                relief = zmax - zmin
                seed_candidate = (
                    positive
                    & (count >= min(4, int(min_valid_cells)))
                    & np.isfinite(relief)
                    & (relief <= float(max_seed_relief_m))
                    & ~protected
                )

                # Require a broad physical seed, not a handful of flat cells.
                labs, nlab = ndi.label(seed_candidate, structure=FOUR)
                seeds = np.zeros(z.shape, dtype=bool)
                if nlab:
                    counts = np.bincount(labs[seed_candidate], minlength=nlab + 1)
                    min_seed_cells = max(
                        1, int(math.ceil(float(min_seed_area_m2) / pixel_area))
                    )
                    keep_ids = np.flatnonzero(counts >= min_seed_cells)
                    keep_ids = keep_ids[keep_ids != 0]
                    if keep_ids.size:
                        seeds = np.isin(labs, keep_ids)

                if not seeds.any():
                    stats["retained_cells"] += npos - wave_count
                    stats["rejected_cells"] += wave_count
                    if not protected.any() and not wave_count:
                        rr = row.copy()
                        rr["hydroflat_status"] = "NO_BROAD_FLAT_SEED"
                        kept_rows.append(rr)
                        continue

                    # Optional mapped coastal structures may preserve only their
                    # immediate positive-TNM footprint even when there is no broad
                    # hydroflat seed.  This lets a genuinely flat-topped solid coastal structure
                    # reach the normal inland-connectivity test while the remainder
                    # of the candidate still undergoes the standard local-relief
                    # screen.  OSM alone cannot create candidate cells because
                    # ``protected`` is explicitly intersected with ``positive``.
                    stats["osm_protected_cells"] += int(protected.sum())
                    common = dict(row)
                    common.pop("geometry", None)
                    common.pop("cells", None)
                    common.update(
                        {
                            "hydroflat_seed_window_m": float(seed_window_m),
                            "hydroflat_seed_max_relief_m": float(max_seed_relief_m),
                            "hydroflat_min_seed_area_m2": float(min_seed_area_m2),
                            "hydroflat_max_neighbor_grade": float(max_neighbor_grade),
                            "hydroflat_max_neighbor_step_m": float(max_neighbor_step_m),
                            "hydroflat_candidate_cells": npos,
                            "hydroflat_seed_cells": 0,
                            "hydroflat_rejected_cells": wave_count,
                            "hydroflat_wave_surface_cells": wave_count,
                            "hydroflat_retained_cells": npos - wave_count,
                            "hydroflat_osm_protected_cells": int(protected.sum()),
                        }
                    )
                    unprotected = positive & ~protected & ~wave
                    append_components(
                        unprotected,
                        transform=rt,
                        crs=src.crs,
                        kind="seaward_positive",
                        common=common,
                        status="NO_BROAD_FLAT_SEED",
                        destination=kept_rows,
                    )
                    append_components(
                        protected,
                        transform=rt,
                        crs=src.crs,
                        kind="seaward_positive",
                        common=common,
                        status="OSM_STRUCTURE_PROTECTED",
                        destination=kept_rows,
                        review_kind="hydroflat_osm_protected",
                    )
                    if wave_count:
                        append_components(
                            wave,
                            transform=rt,
                            crs=src.crs,
                            kind="rejected_wave_surface",
                            common=common,
                            status="REJECTED_BROAD_INCOHERENT_RELIEF",
                            destination=rejected_rows,
                        )
                    continue

                stats["rows_with_seed"] += 1
                stats["seed_cells"] += int(seeds.sum())
                stats["osm_protected_cells"] += int(protected.sum())

                # PHASE 4 — Propagate from confirmed seeds; terrain curvature and
                # protected structures block excessive inland propagation.
                bad_step = _max_four_neighbor_step_bad(
                    z,
                    positive,
                    dx_m=dx,
                    dy_m=dy,
                    max_grade=max_neighbor_grade,
                    max_step_m=max_neighbor_step_m,
                )
                smooth = positive & ~bad_step & ~protected
                # A low neighbor grade alone cannot distinguish an artificial
                # planar ramp from a curved beach profile. Keep the flat seed and
                # its immediate window-scale transition, but stop propagation at
                # nonplanar terrain farther away. Reuse the existing 15 m window
                # and 0.02 m tolerance; no new user threshold is introduced.
                seed_distance = ndi.distance_transform_edt(~seeds, sampling=(dy, dx))
                curved = _nonplanar_terrain(
                    z,
                    positive,
                    dx_m=dx,
                    dy_m=dy,
                    window_m=seed_window_m,
                    tolerance_m=max_seed_relief_m,
                )
                barrier = curved & (seed_distance > float(seed_window_m)) & ~protected
                legacy = ndi.binary_propagation(seeds, structure=FOUR, mask=smooth)
                legacy &= positive & ~protected
                rejected = ndi.binary_propagation(
                    seeds, structure=FOUR, mask=smooth & ~barrier
                )
                rejected &= positive & ~protected
                rejected |= wave
                retained = positive & ~rejected
                stats["nonplanar_barrier_cells"] += int(barrier.sum())
                stats["legacy_rejected_cells"] += int(legacy.sum())

                stats["rejected_cells"] += int(rejected.sum())
                stats["retained_cells"] += int(retained.sum())

                common = dict(row)
                common.pop("geometry", None)
                common.pop("cells", None)
                common.update(
                    {
                        "hydroflat_seed_window_m": float(seed_window_m),
                        "hydroflat_seed_max_relief_m": float(max_seed_relief_m),
                        "hydroflat_min_seed_area_m2": float(min_seed_area_m2),
                        "hydroflat_max_neighbor_grade": float(max_neighbor_grade),
                        "hydroflat_max_neighbor_step_m": float(max_neighbor_step_m),
                        "hydroflat_candidate_cells": npos,
                        "hydroflat_seed_cells": int(seeds.sum()),
                        "hydroflat_nonplanar_barrier_cells": int(barrier.sum()),
                        "hydroflat_legacy_rejected_cells": int(legacy.sum()),
                        "hydroflat_rejected_cells": int(rejected.sum()),
                        "hydroflat_wave_surface_cells": wave_count,
                        "hydroflat_retained_cells": int(retained.sum()),
                        "hydroflat_osm_protected_cells": int(protected.sum()),
                    }
                )

                # PHASE 5 — Publish existing retained/rejected/seed classifications.
                append_components(
                    retained,
                    transform=rt,
                    crs=src.crs,
                    kind="seaward_positive",
                    common=common,
                    status="SCREENED_RETAINED",
                    destination=kept_rows,
                )
                append_components(
                    seeds,
                    transform=rt,
                    crs=src.crs,
                    kind="hydroflat_water_seed",
                    common=common,
                    status="BROAD_FLAT_POSITIVE_SEED",
                    destination=seed_rows,
                )
                append_components(
                    rejected,
                    transform=rt,
                    crs=src.crs,
                    kind="rejected_hydroflat_transition",
                    common=common,
                    status="REJECTED_SMOOTH_FROM_FLAT_SEED",
                    destination=rejected_rows,
                )
                if protected.any():
                    append_components(
                        protected,
                        transform=rt,
                        crs=src.crs,
                        kind="hydroflat_osm_protected",
                        common=common,
                        status="OPTIONAL_OSM_STRUCTURE_PROTECTION",
                        destination=protected_rows,
                    )
        finally:
            src.close()

    kept = frame(kept_rows)
    # Land context is never modified by this seaward artifact screen.
    screened = gpd.GeoDataFrame(
        pd.concat([context, kept], ignore_index=True)
        if not kept.empty
        else context.copy(),
        geometry="geometry",
        crs=evidence.crs,
    )
    return (
        screened,
        frame(seed_rows),
        frame(rejected_rows),
        frame(protected_rows),
        stats,
    )


def local_terrain_support_screen(
    evidence: gpd.GeoDataFrame,
    *,
    local_window_m=15.0,
    min_local_relief_m=0.10,
    max_flat_bridge_m=5.0,
    min_valid_cells=4,
):
    """Derive locally terrain-supported subgeometry from cached positive evidence.

    v5.5 screened whole connected positive components.  That can still admit a broad
    hydroflattened harbor when only a small part of the component contains relief from
    docks, quay walls, or adjacent terrain.  This function works at source-raster cell
    scale instead:

    * the +0.01 m (row threshold) positive mask remains the candidate delineation;
    * a cell is a terrain core only when positive cells in a physical local window show
      at least ``min_local_relief_m`` of elevation range;
    * terrain support may expand only ``max_flat_bridge_m`` through positive cells, so a
      short flat shoreline margin can survive but a broad flat harbor cannot be flooded;
    * unsupported positive cells remain review evidence.

    Hydroflat survivors are evaluated for local relief before ambiguity review.
    Flat survivors keep the existing HYDROFLAT_SURVIVOR route so narrow raised
    structures still reach that review. Only explicit OSM_STRUCTURE_PROTECTED
    geometry bypasses local relief. Detached survivors still fail connectivity.

    No TNM discovery or download is performed.  Raster paths are taken from the cached
    evidence records.
    """
    if evidence.empty:
        empty = evidence.copy()
        return empty, empty

    sea = evidence[evidence["kind"] == "seaward_positive"].copy()
    if sea.empty:
        return sea, sea.copy()

    support_rows = []
    review_rows = []
    groups: dict = {}
    for idx, row in sea.iterrows():
        rp = row.get("raster")
        groups.setdefault(str(rp) if rp else "", []).append((idx, row))

    for raster_name, items in groups.items():
        p = Path(raster_name) if raster_name else None
        if p is None or not p.is_file():
            for _, row in items:
                rr = row.copy()
                rr["terrain_local_status"] = "MISSING_RASTER"
                review_rows.append(rr)
            continue
        try:
            src = rasterio.open(p)
        except Exception:
            for _, row in items:
                rr = row.copy()
                rr["terrain_local_status"] = "OPEN_FAILED"
                review_rows.append(rr)
            continue
        try:
            for _, row in items:
                geom = row.geometry
                if geom is None or geom.is_empty:
                    rr = row.copy()
                    rr["terrain_local_status"] = "EMPTY_GEOMETRY"
                    review_rows.append(rr)
                    continue

                # Explicit mapped solid structures retain their established
                # route. Other survivors need an actual terrain-relief measurement.
                if str(row.get("hydroflat_status", "")) == "OSM_STRUCTURE_PROTECTED":
                    rr = row.copy()
                    rr["terrain_local_status"] = "HYDROFLAT_SURVIVOR"
                    rr["local_window_m"] = float(local_window_m)
                    rr["min_local_relief_m"] = float(min_local_relief_m)
                    rr["max_flat_bridge_m"] = float(max_flat_bridge_m)
                    rr["terrain_core_cells"] = 0
                    rr["terrain_support_cells"] = int(row.get("cells", 0) or 0)
                    rr["unsupported_positive_cells"] = 0
                    support_rows.append(rr)
                    continue

                gsrc = gpd.GeoSeries([geom], crs=sea.crs).to_crs(src.crs).iloc[0]
                pix = float(row.get("pixel_m", 1.0) or 1.0)
                if not math.isfinite(pix) or pix <= 0:
                    pix = 1.0
                half_cells = max(1, int(math.ceil((float(local_window_m) / 2.0) / pix)))
                ksize = half_cells * 2 + 1
                flat_steps = max(
                    0, int(math.floor(float(max_flat_bridge_m) / pix + 1e-12))
                )
                pad_cells = half_cells + flat_steps + 2

                minx, miny, maxx, maxy = gsrc.bounds
                win = from_bounds(minx, miny, maxx, maxy, src.transform)
                col0 = max(0, int(math.floor(win.col_off)) - pad_cells)
                row0 = max(0, int(math.floor(win.row_off)) - pad_cells)
                col1 = min(
                    src.width, int(math.ceil(win.col_off + win.width)) + pad_cells
                )
                row1 = min(
                    src.height, int(math.ceil(win.row_off + win.height)) + pad_cells
                )
                if col1 <= col0 or row1 <= row0:
                    rr = row.copy()
                    rr["terrain_local_status"] = "NO_OVERLAP"
                    review_rows.append(rr)
                    continue

                rw = rasterio.windows.Window(col0, row0, col1 - col0, row1 - row0)
                arr = src.read(1, window=rw, masked=True)
                z = np.asarray(
                    arr.filled(np.nan),  # type: ignore[attr-defined]
                    dtype="float64" if arr.dtype == np.float64 else "float32",
                )
                valid = (~np.ma.getmaskarray(arr)) & np.isfinite(z)
                rt = src.window_transform(rw)
                inside = rasterize(
                    [(gsrc, 1)],
                    out_shape=z.shape,
                    transform=rt,
                    fill=0,
                    dtype="uint8",
                    all_touched=False,
                ).astype(bool)
                threshold = float(row.get("threshold_m", 0.01))
                positive = valid & inside & (z > threshold)
                npos = int(positive.sum())
                if npos < int(min_valid_cells):
                    rr = row.copy()
                    rr["terrain_local_status"] = "TOO_FEW_CELLS"
                    review_rows.append(rr)
                    continue

                # Restrict the local relief calculation to this candidate's own positive
                # cells. Nearby terrestrial structures therefore cannot make a broad flat
                # water surface terrain-like merely because they fall in the same window.
                hi = np.where(positive, z, -np.inf)
                lo = np.where(positive, z, np.inf)
                zmax = ndi.maximum_filter(hi, size=ksize, mode="constant", cval=-np.inf)
                zmin = ndi.minimum_filter(lo, size=ksize, mode="constant", cval=np.inf)
                count = ndi.uniform_filter(
                    positive.astype("float32"), size=ksize, mode="constant", cval=0.0
                ) * float(ksize * ksize)
                relief = zmax - zmin
                core = (
                    positive
                    & (count >= min(4, int(min_valid_cells)))
                    & np.isfinite(relief)
                    & (relief >= float(min_local_relief_m))
                )

                support = core.copy()
                for _step in range(flat_steps):
                    support |= ndi.binary_dilation(support, structure=FOUR) & positive
                unsupported = positive & ~support

                common = dict(row)
                common.pop("geometry", None)
                common.update(
                    {
                        "terrain_local_status": "OK"
                        if support.any()
                        else "NO_LOCAL_TERRAIN",
                        "local_window_m": float(local_window_m),
                        "min_local_relief_m": float(min_local_relief_m),
                        "max_flat_bridge_m": float(max_flat_bridge_m),
                        "terrain_core_cells": int(core.sum()),
                        "terrain_support_cells": int(support.sum()),
                        "unsupported_positive_cells": int(unsupported.sum()),
                    }
                )

                def rows_from_mask(mask, status):
                    out_rows = []
                    if not mask.any():
                        return out_rows
                    labs, nlab = ndi.label(mask, structure=FOUR)
                    if not nlab:
                        return out_rows
                    counts = np.bincount(labs[mask], minlength=nlab + 1)
                    for geo, lab in shapes(
                        labs.astype("int32"),
                        mask=labs > 0,
                        transform=rt,
                        connectivity=4,
                    ):  # noqa: B023 - Called synchronously before this loop advances.
                        lab = int(lab)
                        if lab <= 0:
                            continue
                        rr = dict(common)  # noqa: B023 - Called synchronously before this loop advances.
                        rr["terrain_local_status"] = status
                        rr["cells"] = int(counts[lab])
                        rr["geometry"] = shape(geo)
                        out_rows.append(rr)
                    return out_rows

                sup = rows_from_mask(support, "SUPPORTED")
                is_survivor = (
                    str(row.get("hydroflat_status", "")) == "SCREENED_RETAINED"
                )
                rej = rows_from_mask(
                    unsupported,
                    "HYDROFLAT_SURVIVOR" if is_survivor else "LOCAL_FLAT_REVIEW",
                )
                if is_survivor and rej:
                    # These cells have no measured terrain core. The existing
                    # nearshore ambiguity screen decides their disposition.
                    for rr in rej:
                        rr["terrain_core_cells"] = 0
                    sup.extend(rej)
                    rej = []
                if sup:
                    sg = gpd.GeoDataFrame(sup, geometry="geometry", crs=src.crs).to_crs(
                        sea.crs  # type: ignore[arg-type]
                    )
                    support_rows.extend(sg.to_dict("records"))
                if rej:
                    rg = gpd.GeoDataFrame(rej, geometry="geometry", crs=src.crs).to_crs(
                        sea.crs  # type: ignore[arg-type]
                    )
                    review_rows.extend(rg.to_dict("records"))
                elif not sup:
                    rr = row.copy()
                    rr["terrain_local_status"] = "NO_LOCAL_TERRAIN"
                    review_rows.append(rr)
        finally:
            src.close()

    cols = list(sea.columns)

    def frame(rows):
        if not rows:
            return gpd.GeoDataFrame(
                columns=cols + ["terrain_local_status"],
                geometry="geometry",
                crs=sea.crs,
            )
        # v6.5 can legitimately mix pass-through rows copied from ``iterrows``
        # (pandas Series) with polygonized rows represented as dictionaries.
        # Normalize both forms to plain records so GeoPandas/Pandas version
        # differences cannot break finalization.
        records = [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in rows]
        return gpd.GeoDataFrame(records, geometry="geometry", crs=sea.crs)

    return frame(support_rows), frame(review_rows)


def screen_nearshore_transition_artifacts(
    candidates,
    baseline_mask,
    *,
    metric_crs,
    nearshore_distance_m=100.0,
    max_area_m2=50000.0,
):
    """Move ambiguous hydroflat survivors near the original shoreline to review.

    This screen is deliberately narrower than a generic shape/flatness filter.
    It only acts on *unprotected hydroflat survivors* that reached this stage
    without any local terrain core.  Candidates that passed the normal local
    terrain-support test are retained, as are positive-TNM cells protected by an
    allowed solid OSM structure (breakwater/groyne/dyke/embankment).

    Distances and areas are evaluated in ``metric_crs``.  This is important: the
    evidence geometry is normally EPSG:4326 while the coastline metric work is UTM.
    The baseline mask itself is not changed; reviewed geometry is simply withheld
    from automatic TNM-land promotion.
    """
    if candidates.empty:
        return candidates, candidates.copy()

    base = gpd.read_file(baseline_mask, layer="mask")
    if base.crs is None:
        raise ValueError(f"Baseline mask has no CRS: {baseline_mask}")
    base_m = base.to_crs(metric_crs)
    shoreline_m = shapely.union_all(base_m.geometry.values).boundary

    cand_m = candidates.to_crs(metric_crs)
    kept_records = []
    review_records = []

    for pos, (_, row_m) in enumerate(cand_m.iterrows()):
        row_src = candidates.iloc[pos]
        geom_m = row_m.geometry
        if geom_m is None or geom_m.is_empty:
            continue

        distance_m = float(geom_m.distance(shoreline_m))
        area_m2 = float(geom_m.area)
        hydro_status = str(row_src.get("hydroflat_status", "") or "")
        terrain_status = str(row_src.get("terrain_local_status", "") or "")
        terrain_core_cells = int(row_src.get("terrain_core_cells", 0) or 0)

        # OSM support is carried into this stage by the hydroflat screen as a
        # status, before final post-connectivity annotation occurs.
        osm_protected = hydro_status == "OSM_STRUCTURE_PROTECTED"
        ambiguous_hydroflat_survivor = (
            terrain_status == "HYDROFLAT_SURVIVOR"
            and terrain_core_cells == 0
            and not osm_protected
        )
        suspicious = (
            ambiguous_hydroflat_survivor
            and distance_m <= float(nearshore_distance_m)
            and area_m2 <= float(max_area_m2)
        )

        rr = row_src.to_dict() if hasattr(row_src, "to_dict") else dict(row_src)
        rr["nearshore_distance_m"] = distance_m
        rr["nearshore_area_m2"] = area_m2
        rr["nearshore_osm_protected"] = bool(osm_protected)
        rr["nearshore_status"] = (
            "AMBIGUOUS_HYDROFLAT_SURVIVOR_REVIEW"
            if suspicious
            else "ACCEPTED_NEARSHORE_SCREEN"
        )
        if suspicious:
            review_records.append(rr)
        else:
            kept_records.append(rr)

    extra = [
        "nearshore_status",
        "nearshore_distance_m",
        "nearshore_area_m2",
        "nearshore_osm_protected",
    ]
    cols = list(candidates.columns) + [c for c in extra if c not in candidates.columns]

    def frame(records):
        if not records:
            return gpd.GeoDataFrame(
                columns=cols, geometry="geometry", crs=candidates.crs
            )
        return gpd.GeoDataFrame(records, geometry="geometry", crs=candidates.crs)

    return frame(kept_records), frame(review_records)


def clean_accepted_by_source_resolution(
    accepted: gpd.GeoDataFrame,
    *,
    metric_crs,
    min_close_m=0.5,
    close_fraction=0.35,
    max_close_m=3.0,
    min_hole_area_m2=4.0,
    hole_cells=2.0,
):
    """Clean only accepted TNM additions using each feature's native pixel size.

    Fixed 1 m / 4 m2 cleanup is too small for 1/3 arc-second (~10 m) fallback data.
    This keeps cleanup proportional to the raster that created each geometry while
    capping the closing radius so the OSM shoreline is not broadly smoothed.
    """
    if accepted is None or accepted.empty:
        return accepted.copy() if accepted is not None else accepted
    am = accepted.to_crs(metric_crs).copy()
    cleaned = []
    for _idx, row in am.iterrows():
        pix = float(row.get("pixel_m", 1.0) or 1.0)
        if not math.isfinite(pix) or pix <= 0:
            pix = 1.0
        close_m = min(
            float(max_close_m), max(float(min_close_m), pix * float(close_fraction))
        )
        hole_m2 = max(float(min_hole_area_m2), float(hole_cells) * pix * pix)
        g = clean_addition_geometry(
            row.geometry,
            src_crs=metric_crs,
            metric_crs=metric_crs,
            close_m=close_m,
            max_hole_area_m2=hole_m2,
        )
        rr = row.copy()
        rr.geometry = g  # type: ignore[attr-defined]
        rr["cleanup_close_m"] = close_m
        rr["cleanup_hole_m2"] = hole_m2
        cleaned.append(rr)
    out = gpd.GeoDataFrame(cleaned, geometry="geometry", crs=metric_crs)
    out = out[~out.geometry.is_empty].copy()
    return out.to_crs(accepted.crs)  # type: ignore[arg-type]


def write_dropin_mask(input_gpkg, output_gpkg, additions_geom, layer="mask"):
    """Subtract land additions without changing baseline attributes or feature count."""
    input_gpkg = Path(input_gpkg).resolve()
    output_gpkg = Path(output_gpkg).resolve()
    if input_gpkg == output_gpkg:
        raise ValueError("Output GPKG must differ from input GPKG")
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_gpkg.with_name(output_gpkg.stem + ".tmp.gpkg")
    if tmp.exists():
        tmp.unlink()

    frame = pyogrio.read_dataframe(input_gpkg, layer=layer)
    if frame.empty:
        raise ValueError(f"No features in {input_gpkg}:{layer}")
    if frame.crs is None:
        raise ValueError(f"Missing CRS in {input_gpkg}:{layer}")

    add = additions_geom
    if add is None or add.is_empty:
        add_src = MultiPolygon([])
    else:
        add_src = gpd.GeoSeries([add], crs="EPSG:4326").to_crs(frame.crs).iloc[0]

    revised = []
    for geom in frame.geometry:
        result = _polygonal(shapely.make_valid(geom.difference(add_src)))
        if result.is_empty:
            raise ValueError(
                "Refinement would erase an entire baseline feature; refusing"
            )
        revised.append(result)

    frame = frame.copy()
    frame.geometry = gpd.GeoSeries(revised, index=frame.index, crs=frame.crs)
    # Write the complete replacement before publishing it. Keep a .gpkg
    # suffix so GDAL recognizes the temporary file as a GeoPackage.
    pyogrio.write_dataframe(
        frame,
        tmp,
        driver="GPKG",
        layer=layer,
        geometry_type="MultiPolygon",
        promote_to_multi=True,
    )
    tmp.replace(output_gpkg)
    return validate_dropin(input_gpkg, output_gpkg, layer=layer)


def validate_dropin(input_gpkg, output_gpkg, layer="mask"):
    """Check layer names, field types, geometry type, CRS and feature count."""
    from pyproj import CRS

    input_layers = pyogrio.list_layers(input_gpkg)[:, 0].tolist()
    output_layers = pyogrio.list_layers(output_gpkg)[:, 0].tolist()
    a = pyogrio.read_info(input_gpkg, layer=layer)
    b = pyogrio.read_info(output_gpkg, layer=layer)

    input_schema = {
        "geometry": a["geometry_type"],
        "properties": dict(zip(a["fields"].tolist(), a["ogr_types"], strict=True)),
    }
    output_schema = {
        "geometry": b["geometry_type"],
        "properties": dict(zip(b["fields"].tolist(), b["ogr_types"], strict=True)),
    }
    same_crs = bool(
        a["crs"]
        and b["crs"]
        and CRS.from_user_input(a["crs"]) == CRS.from_user_input(b["crs"])
    )
    compatible = bool(
        input_schema == output_schema
        and same_crs
        and input_layers == output_layers
        and a["features"] == b["features"]
    )
    result = {
        "input": str(Path(input_gpkg).resolve()),
        "output": str(Path(output_gpkg).resolve()),
        "layer": layer,
        "input_layers": input_layers,
        "output_layers": output_layers,
        "input_schema": input_schema,
        "output_schema": output_schema,
        "input_crs_wkt": a["crs"],  # Retained key; Pyogrio may return EPSG strings.
        "output_crs_wkt": b["crs"],
        "input_count": a["features"],
        "output_count": b["features"],
        "drop_in_compatible": compatible,
    }
    if not compatible:
        raise RuntimeError("Output GPKG failed drop-in schema/CRS/layer validation")
    return result


def clean_addition_geometry(
    additions,
    *,
    src_crs="EPSG:4326",
    metric_crs=None,
    close_m=1.0,
    max_hole_area_m2=4.0,
):
    """Remove only micro raster/polygonization artifacts from *new* additions.

    The baseline coastline is untouched.  A small metric closing bridges sub-cell
    gaps, then only interior holes below max_hole_area_m2 are filled.
    """
    if additions is None or additions.is_empty:
        return MultiPolygon([])
    if metric_crs is None:
        from .coastal_plan import _metric_crs

        metric_crs = _metric_crs(gpd.GeoDataFrame(geometry=[additions], crs=src_crs))
    gm = (
        gpd.GeoSeries([shapely.make_valid(additions)], crs=src_crs)
        .to_crs(metric_crs)
        .iloc[0]
    )
    if float(close_m) > 0:
        gm = shapely.make_valid(gm.buffer(float(close_m)).buffer(-float(close_m)))

    def no_small_holes(poly):
        keep = []
        for ring in poly.interiors:
            hp = Polygon(ring)
            if hp.area >= float(max_hole_area_m2):
                keep.append(ring.coords)
        return Polygon(poly.exterior.coords, keep)

    parts = []
    for p in shapely.get_parts(shapely.make_valid(gm)):
        if isinstance(p, Polygon) and not p.is_empty:
            parts.append(no_small_holes(p))
    if not parts:
        return MultiPolygon([])
    cleaned = shapely.union_all(parts)
    return gpd.GeoSeries([cleaned], crs=metric_crs).to_crs(src_crs).iloc[0]
