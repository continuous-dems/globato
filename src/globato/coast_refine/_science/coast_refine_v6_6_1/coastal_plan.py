#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import shapely
from shapely.geometry import box
from .input_resolver import _polygonal


def _read(path, layer=None):
    g = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if g.empty:
        raise ValueError(f"No features: {path}")
    if g.crs is None:
        raise ValueError(f"Missing CRS: {path}")
    return g


def _metric_crs(gdf):
    # Reuse an existing metre-based projected CRS when possible. This is both
    # more precise and avoids unnecessary datum/UTM guessing for projected ROIs.
    crs0 = gdf.crs
    if crs0 is not None and getattr(crs0, "is_projected", False):
        try:
            units = [ax.unit_name.lower() for ax in crs0.axis_info if ax.unit_name]
            if units and all(u in {"metre", "meter"} for u in units):
                return crs0
        except Exception:
            pass
    metric = gdf.estimate_utm_crs()
    if metric is None:
        raise ValueError(
            "Cannot select a local metric CRS; supply a metre-based projected ROI"
        )
    return metric


def make_plan(
    mask,
    output,
    *,
    tiles=None,
    mask_layer="mask",
    tile_layer=None,
    inland_m=100.0,
    seaward_m=250.0,
    segment_m=5000.0,
    halo_m=100.0,
    edge_exclusion_m=2.0,
    full_search=False,
):
    mask = Path(mask).resolve()
    output = Path(output).resolve()
    mg = _read(mask, mask_layer)
    mask_crs = mg.crs
    ocean = shapely.union_all(mg.geometry.values)
    if ocean.is_empty:
        raise ValueError("Baseline ocean mask is empty")

    if tiles:
        tg = _read(tiles, tile_layer).to_crs(mask_crs)
        roi = shapely.union_all(tg.geometry.values)
    else:
        # Fallback for a baseline mask already clipped to the intended project extent.
        # A tile/ROI vector is preferred because it identifies non-coast outer edges explicitly.
        roi = ocean.envelope
        tg = gpd.GeoDataFrame(
            {"tile_id": ["mask_extent"]}, geometry=[roi], crs=mask_crs
        )

    roi_g = gpd.GeoDataFrame({"name": ["roi"]}, geometry=[roi], crs=mask_crs)
    metric = _metric_crs(roi_g)
    ocean_m = (
        gpd.GeoSeries([ocean], crs=mask_crs)
        .to_crs(metric)
        .iloc[0]
        .intersection(gpd.GeoSeries([roi], crs=mask_crs).to_crs(metric).iloc[0])
    )
    roi_m = gpd.GeoSeries([roi], crs=mask_crs).to_crs(metric).iloc[0]

    # Clipping can leave zero-area line contacts beside valid water polygons.
    # A GeometryCollection has no boundary; retain the polygonal water domain
    # through the existing input-normalization owner before deriving the coast.
    ocean_m = _polygonal(ocean_m)
    if ocean_m.is_empty:
        raise ValueError(
            "Baseline ocean mask has no polygonal overlap with the requested ROI"
        )

    # Remove only boundary portions coincident with the outer project ROI. What remains
    # is the actual land/water interface, including island/estuary boundaries.
    coast_m = ocean_m.boundary.difference(
        roi_m.boundary.buffer(float(edge_exclusion_m))
    )
    coast_m = coast_m.intersection(
        roi_m.buffer(-0.01) if not roi_m.buffer(-0.01).is_empty else roi_m
    )
    if coast_m.is_empty:
        raise ValueError(
            "No interior coastline found. Supply the production tile/ROI vector with --tiles."
        )

    land_domain_m = roi_m.difference(ocean_m)
    landward_m = coast_m.buffer(
        float(inland_m), cap_style=2, join_style=2
    ).intersection(land_domain_m)
    seaward_m_geom = (
        ocean_m
        if full_search
        else coast_m.buffer(float(seaward_m), cap_style=2, join_style=2).intersection(
            ocean_m
        )
    )
    ribbon_m = shapely.union_all([landward_m, seaward_m_geom])
    if ribbon_m.is_empty:
        raise ValueError("Coastal ribbon is empty")

    minx, miny, maxx, maxy = ribbon_m.bounds
    s = float(segment_m)
    x0 = math.floor(minx / s) * s
    y0 = math.floor(miny / s) * s
    records = []
    seq = 1
    y = y0
    while y < maxy:
        x = x0
        while x < maxx:
            cell = box(x, y, x + s, y + s)
            piece = ribbon_m.intersection(cell)
            if not piece.is_empty and piece.area > 0.1:
                fetch = piece.envelope.buffer(float(halo_m), cap_style=3, join_style=2)
                # A rectangular bounding box is what Globato/Fetchez needs.
                fetch = box(*fetch.bounds)
                records.append(
                    {
                        "box_id": f"coast_{seq:04d}",
                        "ribbon_area_m2": float(piece.area),
                        "fetch_area_m2": float(fetch.area),
                        "geometry": fetch,
                    }
                )
                seq += 1
            x += s
        y += s
    if not records:
        raise ValueError("No fetch boxes were generated")

    boxes_m = gpd.GeoDataFrame(records, geometry="geometry", crs=metric)
    boxes_wgs84 = boxes_m.to_crs(mask_crs)
    boxes_ll = boxes_m.to_crs("EPSG:4326")
    boxes_wgs84["west"] = boxes_ll.bounds.minx.values
    boxes_wgs84["east"] = boxes_ll.bounds.maxx.values
    boxes_wgs84["south"] = boxes_ll.bounds.miny.values
    boxes_wgs84["north"] = boxes_ll.bounds.maxy.values

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    layers = [
        (
            "roi",
            gpd.GeoDataFrame({"name": ["roi"]}, geometry=[roi_m], crs=metric).to_crs(
                mask_crs
            ),
        ),
        (
            "coastline",
            gpd.GeoDataFrame(
                {"name": ["baseline_coastline"]}, geometry=[coast_m], crs=metric
            ).to_crs(mask_crs),
        ),
        (
            "landward_context",
            gpd.GeoDataFrame(
                {"buffer_m": [float(inland_m)]}, geometry=[landward_m], crs=metric
            ).to_crs(mask_crs),
        ),
        (
            "seaward_search",
            gpd.GeoDataFrame(
                {"buffer_m": [float(seaward_m)]}, geometry=[seaward_m_geom], crs=metric
            ).to_crs(mask_crs),
        ),
        (
            "analysis_ribbon",
            gpd.GeoDataFrame(
                {"name": ["analysis_ribbon"]}, geometry=[ribbon_m], crs=metric
            ).to_crs(mask_crs),
        ),
        ("fetch_boxes", boxes_wgs84),
    ]
    for layer, g in layers:
        # Reprojection can turn numerical edge contacts into invalid polygon
        # rings. Use the same normalization owner for serialized water/search
        # domains so the next intersection reads valid polygonal geometry.
        if layer != "coastline":
            g = g.copy()
            g.geometry = g.geometry.map(_polygonal)
        g.to_file(output, layer=layer, driver="GPKG")

    manifest = {
        "mask": str(mask),
        "tiles": None if tiles is None else str(Path(tiles).resolve()),
        "mask_layer": mask_layer,
        "metric_crs": str(metric),
        "inland_m": float(inland_m),
        "seaward_m": float(seaward_m),
        "full_search": bool(full_search),
        "segment_m": float(segment_m),
        "halo_m": float(halo_m),
        "box_count": int(len(boxes_wgs84)),
        "boxes": boxes_wgs84[["box_id", "west", "east", "south", "north"]].to_dict(
            "records"
        ),
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
