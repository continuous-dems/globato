"""Phase 1: select a connected water component without subtype dependence.

This is *selection*, not a shoreline-refinement algorithm. It operates on
existing refined water and never changes measured elevations, source masks, or
the legacy classed-topology publication. A non-unique selection fails closed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import shapely
from shapely.geometry import Point, Polygon, MultiPolygon


def _parts(geometry, label):
    if geometry is None or geometry.is_empty:
        return []
    if not geometry.is_valid:
        raise ValueError(f"{label} is invalid: {shapely.is_valid_reason(geometry)}")
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        raise ValueError(f"{label} must be polygon geometry")
    # Unify edge-sharing polygons; point-only contacts remain disconnected.
    unified = shapely.union_all(list(shapely.get_parts(geometry)))
    return sorted(list(shapely.get_parts(unified)), key=lambda p: p.wkb_hex)


def select_connected_water(water, *, marine=None, outlet=None):
    """Return (selected polygon or None, structured selection decision).

    No class labels, area thresholds, buffers or provider-specific rules.
    `marine` is optional polygonal evidence in the SAME CRS as `water`.
    `outlet` is an optional Point in that CRS, which takes precedence.
    If neither marine nor outlet distinguishes components, a *single* water
    component is sufficient. Multiple unresolved components are not guessed.
    """
    parts = _parts(water, "Refined water")
    report = {
        "status": "unresolved",
        "method": None,
        "component_count": len(parts),
        "reason": None,
        "excluded_component_count": 0,
    }
    if not parts:
        report["reason"] = "no_refined_water"
        return None, report
    hits = []
    if outlet is not None:
        if not isinstance(outlet, Point) or outlet.is_empty:
            raise ValueError("Outlet must be a nonempty point in geometry CRS")
        hits = [p for p in parts if p.covers(outlet)]
        report["method"] = "outlet"
        if not hits:
            report["reason"] = "outlet_outside_refined_water"
        elif len(hits) > 1:
            # In particular, a point-only contact cannot establish hydraulic
            # connectivity between the two touching polygons.
            report["reason"] = "outlet_touches_multiple_components"
    elif marine is not None and not marine.is_empty:
        marine_parts = _parts(marine, "Marine evidence")
        union = shapely.union_all(marine_parts)
        hits = [p for p in parts if p.intersection(union).area > 0]
        report["method"] = "marine_intersection"
        report["reason"] = (
            "marine_evidence_matches_multiple_components"
            if len(hits) > 1
            else ("marine_evidence_outside_refined_water" if not hits else None)
        )
    else:
        hits = parts if len(parts) == 1 else []
        report["method"] = "single_component"
        if not hits:
            report["reason"] = "ambiguous_without_marine_evidence_or_outlet"
    if len(hits) != 1:
        return None, report
    selected = hits[0]
    report.update(
        status="selected", reason=None, excluded_component_count=len(parts) - 1
    )
    return selected, report


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish_connected_water(root, *, outlet_lonlat=None):
    """Publish the separately selected guidance component and auditable receipt.

    Reads already-published refined geometry. Optional lon/lat is transformed
    into the native application CRS. No OSM class names are used. This product
    is NOT a replacement for the existing refined mask/topology publication.
    """
    root = Path(root)
    grid = json.loads((root / "application_grid.json").read_text())
    crs = grid["crs"]
    from ._io import read_polygon

    water_file = root / "refined_water.gpkg"
    marine_file = root / "marine_water.gpkg"
    water = read_polygon(water_file, crs=crs, require_native=True)
    marine = read_polygon(marine_file, crs=crs, require_native=True)
    outlet = None
    if outlet_lonlat is not None:
        if len(outlet_lonlat) != 2:
            raise ValueError("outlet_lonlat requires (longitude, latitude)")
        lon, lat = map(float, outlet_lonlat)
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("Invalid outlet longitude/latitude")
        outlet = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs).iloc[0]
    selected, decision = select_connected_water(water, marine=marine, outlet=outlet)
    target = root / "connected_water.gpkg"
    if target.exists():
        target.unlink()
    receipt = {
        "schema": "coast_refine.connected_water.v1",
        "purpose": "guidance_only; does not modify legacy masks or elevation data",
        "crs": str(crs),
        "refined_water_sha256": _sha(water_file),
        "marine_water_sha256": _sha(marine_file),
        "outlet_lonlat": list(map(float, outlet_lonlat))
        if outlet_lonlat is not None
        else None,
        **decision,
    }
    if selected is not None:
        gpd.GeoDataFrame(
            {"water_system": ["ocean_connected"]}, geometry=[selected], crs=crs
        ).to_file(target, layer="water", driver="GPKG")
        published = gpd.read_file(target, layer="water")
        if len(published) != 1 or not published.geometry.iloc[0].equals(selected):
            target.unlink(missing_ok=True)
            raise ValueError(
                "Published connected water differs from selected refined geometry"
            )
        receipt["file"] = target.name
        receipt["geometry_sha256"] = _sha(target)
        receipt["area_native_units_squared"] = float(selected.area)
    else:
        receipt["file"] = None
        receipt["geometry_sha256"] = None
    (root / "connected_water.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt
