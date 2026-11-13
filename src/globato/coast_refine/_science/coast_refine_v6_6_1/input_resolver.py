#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import geopandas as gpd
import pyogrio
import shapely
from shapely.geometry import MultiPolygon, Polygon

WATER_CLASSES = {"ocean", "water", "river", "lake", "wetland", "estuary"}


def _polygonal(geom):
    if geom is None or geom.is_empty:
        return MultiPolygon([])
    geom = shapely.make_valid(geom)
    parts = []
    for g in shapely.get_parts(geom):
        if isinstance(g, Polygon) and not g.is_empty:
            parts.append(g)
        elif isinstance(g, MultiPolygon):
            parts.extend([p for p in g.geoms if not p.is_empty])
    if not parts:
        return MultiPolygon([])
    u = shapely.union_all(parts)
    if isinstance(u, Polygon):
        return MultiPolygon([u])
    if isinstance(u, MultiPolygon):
        return u
    p2 = [g for g in shapely.get_parts(u) if isinstance(g, Polygon) and not g.is_empty]
    return MultiPolygon(p2)


def _choose_layer(
    path: Path,
    requested: Optional[str],
    preferred: Sequence[str] = ("mask", "tiles", "roi"),
):
    if path.suffix.lower() != ".gpkg":
        return None
    layers = pyogrio.list_layers(path)[:, 0].tolist()
    if requested:
        if requested not in layers:
            raise ValueError(
                f"Layer {requested!r} not found in {path}; available={layers}"
            )
        return requested
    for name in preferred:
        if name in layers:
            return name
    if len(layers) == 1:
        return layers[0]
    raise ValueError(
        f"{path} has multiple layers {layers}. Supply --mask-layer or --tile-layer/--region-layer."
    )


def read_vector(path, layer=None, *, preferred=("tiles", "roi", "mask")):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    lyr = _choose_layer(path, layer, preferred=preferred)
    g = gpd.read_file(path, layer=lyr) if lyr else gpd.read_file(path)
    if g.empty:
        raise ValueError(f"No features in {path}" + (f":{lyr}" if lyr else ""))
    if g.crs is None:
        raise ValueError(f"Missing CRS in {path}" + (f":{lyr}" if lyr else ""))
    return g, lyr


def resolve_area(path, *, workdir):
    """Resolve the worker's one polygon AOI; no tile catalogue or bbox aliases."""
    path = Path(path).expanduser().resolve(strict=True)
    frame, layer = read_vector(path, preferred=("tiles", "roi"))
    frame = frame.to_crs("EPSG:4326")
    geometry = _polygonal(shapely.union_all(frame.geometry.values))
    if geometry.is_empty:
        raise ValueError("Refinement AOI must contain polygons")
    roi = gpd.GeoDataFrame({"name": ["requested_roi"]}, geometry=[geometry], crs=4326)
    out = Path(workdir) / "requested_roi.gpkg"
    roi.to_file(out, layer="roi", driver="GPKG")
    metadata = {
        "mode": "vector",
        "source": str(path),
        "layer": layer,
        "roi_gpkg": str(out),
        "bounds_wgs84": [float(v) for v in roi.total_bounds],
    }
    out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return roi, metadata


def _canonical_mask_from_gdf(
    g, output, *, source_label="input_vector", clip_roi=None, allow_empty=False
):
    if g.crs is None:
        raise ValueError("Mask input has no CRS")
    gg = g.copy()
    if "class" in gg.columns:
        cls = gg["class"].astype(str).str.lower().str.strip()
        keep = cls.isin(WATER_CLASSES)
        if bool(keep.any()):
            gg = gg.loc[keep].copy()
    if gg.empty:
        raise ValueError("No water/polygon features remain in input mask")
    gg = gg.to_crs("EPSG:4326")
    geom = _polygonal(shapely.union_all(gg.geometry.values))
    if clip_roi is not None:
        roi = (
            clip_roi.to_crs("EPSG:4326")
            if clip_roi.crs.to_string() != "EPSG:4326"
            else clip_roi
        )
        rgeom = _polygonal(shapely.union_all(roi.geometry.values))
        geom = _polygonal(geom.intersection(rgeom))
    if geom.is_empty and not allow_empty:
        raise ValueError("Canonical water mask is empty after normalization/clipping")

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    # Keep the original one-feature contract, including an empty MultiPolygon
    # when allow_empty=True. The source field is capped at 32 characters.
    frame = gpd.GeoDataFrame(
        {"class": ["ocean"], "source": [str(source_label)[:32]]},
        geometry=[geom],
        crs="EPSG:4326",
    )
    pyogrio.write_dataframe(
        frame,
        output,
        driver="GPKG",
        layer="mask",
        geometry_type="MultiPolygon",
        promote_to_multi=True,
    )
    return output


def normalize_existing_mask(mask_path, workdir, *, mask_layer=None, allow_empty=False):
    """Accept GPKG or GeoJSON. Reuse a canonical GPKG unchanged when possible."""
    path = Path(mask_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    layer = (
        _choose_layer(path, mask_layer, preferred=("mask",))
        if path.suffix.lower() == ".gpkg"
        else None
    )

    if path.suffix.lower() == ".gpkg" and layer == "mask":
        try:
            info = pyogrio.read_info(path, layer="mask")
            fields = set(info["fields"].tolist())
            if (
                info["features"] == 1
                and info["geometry_type"] == "MultiPolygon"
                and {"class", "source"} <= fields
                and info["crs"]
            ):
                return path, {
                    "mode": "existing_gpkg",
                    "source": str(path),
                    "layer": "mask",
                    "normalized": False,
                }
        except (OSError, ValueError):
            pass

    g, used_layer = read_vector(
        path, layer if path.suffix.lower() == ".gpkg" else None, preferred=("mask",)
    )
    out = Path(workdir).resolve() / "normalized_input_mask.gpkg"
    _canonical_mask_from_gdf(
        g, out, source_label="input_vector", allow_empty=allow_empty
    )
    return out, {
        "mode": "normalized_vector",
        "source": str(path),
        "layer": used_layer,
        "normalized": True,
        "normalized_path": str(out),
    }
