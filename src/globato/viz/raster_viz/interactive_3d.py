"""Offline browser 3D assets for universal raster review (no VTK runtime required).

A scene is an ordinary JavaScript file, not fetched JSON: browsers can open the
self-contained review from file:// without cross-origin fetch permissions.
Sources remain unchanged. Meshes are downsampled display representations only.
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from pathlib import Path

import numpy as np
from matplotlib import colormaps

from .terrain import prepare_terrain, navigation_frame
from .legends import COLORMAPS
from .palette import elevation_rgba


def _encoded(array, dtype):
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode(
        "ascii"
    )


def publish_scene(
    path,
    dest,
    scene_id,
    bounds,
    limits,
    *,
    band=1,
    mesh_side=350,
    vertical_unit_to_m=1.0,
    default_exaggeration=1.0,
    style="elevation",
    nodata_policy="strict",
    water_level=0.0,
):
    """Build a metric shared-CRS triangulated scene; publish JS atomically.

    Coordinates are relative to the group's AEQD origin, and elevation is
    relative to the fixed zero datum (meters). Colors come from shared limits.
    No triangles cross invalid cells; no source raster or nodata is changed.
    """
    mesh = prepare_terrain(
        path,
        bounds,
        band=band,
        max_side=mesh_side,
        vertical_exaggeration=1.0,
        vertical_unit_to_m=vertical_unit_to_m,
        elevation_reference=0.0,
        nodata_policy=nodata_policy,
    )
    z = mesh["elevations"]
    low, high = map(float, limits)
    fraction = np.clip((np.nan_to_num(z, nan=low) - low) / max(high - low, 1e-9), 0, 1)
    rgba = (
        elevation_rgba(z, limits, water_level=water_level)
        if style == "elevation"
        else colormaps[COLORMAPS[style]](fraction)
    )
    color = np.clip(rgba[:, :3] * 255, 0, 255).astype("uint8")
    points = np.asarray(mesh["points"], dtype="float32")
    faces = np.asarray(mesh["faces"], dtype="uint32").reshape(-1, 4)[:, 1:].ravel()
    point_indices = np.asarray(mesh["point_indices"], dtype="uint32")
    all_point_indices = np.asarray(mesh["all_point_indices"], dtype="uint32")
    if points.shape[0] > 1_000_000 or faces.size > 6_000_000:
        raise ValueError("Interactive terrain exceeds bounded mesh safety limit")
    scene = {
        "id": str(scene_id),
        "points": _encoded(points, "<f4"),
        "indices": _encoded(faces, "<u4"),
        "point_indices": _encoded(point_indices, "<u4"),
        "all_point_indices": _encoded(all_point_indices, "<u4"),
        "colors": _encoded(color, "u1"),
        "vertex_count": int(len(points)),
        "index_count": int(len(faces)),
        "triangles": int(mesh["triangles"]),
        "valid_cells": int(mesh["valid_cells"]),
        "point_count": int(mesh["point_count"]),
        "all_point_count": int(mesh["all_point_count"]),
        "valid_samples": int(mesh["valid_samples"]),
        "render_mode": mesh["render_mode"],
        "preview_cell_size_m": mesh["preview_cell_size_m"],
        "data_bounds_local_xy": mesh["data_bounds_local_xy"],
        "data_elevation_range": [float(np.min(z)), float(np.max(z))],
        "preview_shape": mesh["preview_shape"],
        "source_crs": mesh["source_crs"],
        "metric_crs": mesh["metric_crs"],
        "source_resampling": mesh["resampling"],
        "vertical_unit_to_m": float(vertical_unit_to_m),
        "default_exaggeration": float(default_exaggeration),
        "water_level": float(water_level),
        "navigation_frame": navigation_frame(bounds),
        "source_nodata": mesh["missing"],
    }
    scene["nodata_policy"] = nodata_policy
    output = Path(dest)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        # Serialization into JavaScript avoids file:// fetch/CORS issues.
        temporary.write_text(
            "window.__reviewScenes=window.__reviewScenes||{};"
            "window.__reviewScenes["
            + json.dumps(str(scene_id))
            + "]="
            + json.dumps(scene, separators=(",", ":"))
            + ";\n",
            encoding="utf8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        key: scene[key]
        for key in (
            "id",
            "triangles",
            "valid_cells",
            "point_count",
            "valid_samples",
            "all_point_count",
            "render_mode",
            "preview_cell_size_m",
            "data_bounds_local_xy",
            "data_elevation_range",
            "preview_shape",
            "source_crs",
            "metric_crs",
            "source_resampling",
            "vertical_unit_to_m",
            "default_exaggeration",
            "source_nodata",
            "navigation_frame",
            "nodata_policy",
        )
    }
