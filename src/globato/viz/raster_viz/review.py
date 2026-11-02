"""Versioned product registry, bounded previews, and atomic offline publication."""

from __future__ import annotations

import hashlib
import base64
import webbrowser
import json
import math
import os
import logging
import time
import uuid
from pathlib import Path

import numpy as np
import rasterio
from filelock import FileLock
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, reproject, transform as warp_xy

from .legends import COLORMAPS, create_ramps, elevation_ramp, value_label, raster_units
from .palette import elevation_rgba

logger = logging.getLogger(__name__)
MAX_EVENTS = 200  # Bound long-running watch output; latest diagnostics remain visible.


def atomic(path, body):
    path = Path(path)
    temp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(body, encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def safe_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)


def sparse_elevation_cells(src, band):
    """Collect valid source cells only for <=1% coverage, bounded to 100k cells.

    Dense rasters stop scanning as soon as the budget is exceeded. Sparse
    rasters are read by source blocks so isolated values cannot disappear in
    preview downsampling. This changes display symbols only, never source data.
    """
    budget = min(100_000, src.width * src.height // 100)
    if not budget:
        return None
    chunks, count = [], 0
    for _, window in src.block_windows(int(band)):
        a = src.read(int(band), window=window, masked=True)
        valid = ~np.ma.getmaskarray(a) & np.isfinite(a.data)
        count += int(np.count_nonzero(valid))
        if count > budget:
            return None
        if valid.any():
            rows, cols = np.nonzero(valid)
            x, y = src.transform * (
                cols + window.col_off + 0.5,
                rows + window.row_off + 0.5,
            )
            chunks.append((x, y, a.data[valid].astype("float64")))
    if not count:
        return None
    x, y, values = (np.concatenate(parts) for parts in zip(*chunks))
    x, y = warp_xy(src.crs, "EPSG:3857", x, y)
    if not np.isfinite([x, y]).all():
        raise ValueError("Sparse source-cell coordinates cannot be projected")
    return np.asarray(x), np.asarray(y), values


def color_image(
    z,
    transform,
    bounds,
    style="elevation",
    spacing=None,
    vertical_unit_to_m=1.0,
    water_level=0.0,
):
    from matplotlib import colormaps
    from scipy.ndimage import distance_transform_edt

    # Accept integer and float arrays consistently without mutating callers.
    z = np.asarray(z, dtype="float64")
    valid = np.isfinite(z)
    normalized = np.clip((z - bounds[0]) / (bounds[1] - bounds[0]), 0, 1)
    cmap = COLORMAPS.get(style, "terrain")
    rgba = (
        elevation_rgba(z, bounds, water_level=water_level)
        if style == "elevation"
        else colormaps[cmap](np.nan_to_num(normalized))
    )
    if style == "categorical":
        # Discrete classes, not a shaded elevation model. Preserve 0/1 as
        # distinguishable categories; other integers get a stable palette.
        rgba = colormaps["tab20"](np.mod(np.nan_to_num(z).astype("int64"), 20) / 20.0)
        rgba[..., :3] = np.where((z == 0)[..., None], [0.91, 0.93, 0.95], rgba[..., :3])
        rgba[..., :3] = np.where((z == 1)[..., None], [0.89, 0.28, 0.13], rgba[..., :3])
    if style == "elevation" and valid.any():
        filled = z.copy()
        if not valid.all():
            iy, ix = distance_transform_edt(
                ~valid, return_distances=False, return_indices=True
            )
            filled[~valid] = z[iy[~valid], ix[~valid]]
        # Elevation values may be in feet; metric XY spacing requires metric Z
        # for physically correct hillshade slopes (display-only conversion).
        filled *= vertical_unit_to_m
        # Web Mercator scale correction at the preview's center latitude.
        center_y = transform.f + transform.e * z.shape[0] / 2
        scale = 1 / math.cosh(center_y / 6378137)
        dx = max(abs(transform.a) * scale, 1e-9) if spacing is None else spacing[1]
        dy = max(abs(transform.e) * scale, 1e-9) if spacing is None else spacing[0]
        gy = np.gradient(filled, dy, axis=0) if z.shape[0] > 1 else np.zeros_like(z)
        gx = np.gradient(filled, dx, axis=1) if z.shape[1] > 1 else np.zeros_like(z)
        # Array rows run south; north component of the upward normal is +gy.
        az, alt = np.deg2rad([315, 45])
        shade = (
            -gx * np.sin(az) * np.cos(alt) + gy * np.cos(az) * np.cos(alt) + np.sin(alt)
        ) / np.sqrt(1 + gx * gx + gy * gy)
        rgba[..., :3] *= (0.4 + 0.6 * np.clip(shade, 0, 1))[..., None]
    rgba[..., 3] = valid
    return np.clip(rgba * 255, 0, 255).astype(np.uint8)


def native_spacing(crs, transform, shape):
    """Physical pixel spacing for a bounded north-up source preview."""
    from pyproj import CRS

    c = CRS.from_user_input(crs)
    x, y = transform * (shape[1] / 2, shape[0] / 2)
    if c.is_geographic:
        geod = c.get_geod()
        dx = abs(geod.inv(x, y, x + transform.a, y)[2])
        dy = abs(geod.inv(x, y, x, y + transform.e)[2])
    else:
        factor = c.axis_info[0].unit_conversion_factor
        dx, dy = abs(transform.a) * factor, abs(transform.e) * factor
    return max(dy, 1e-9), max(dx, 1e-9)


def project_shaded_image(
    z,
    transform,
    crs,
    dst_transform,
    dst_shape,
    limits,
    style,
    vertical_unit_to_m=1.0,
    water_level=0.0,
):
    """Shade BEFORE geographic reprojection, avoiding periodic derivative stripes.

    RGB is alpha-normalized and bilinearly warped. A separate nearest mask keeps
    original nodata holes transparent. No elevation raster is changed or filled.
    """
    rgba = color_image(
        z,
        transform,
        limits,
        style,
        native_spacing(crs, transform, z.shape),
        vertical_unit_to_m,
        water_level,
    )
    valid = np.isfinite(z).astype("float32")
    params = dict(
        src_transform=transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:3857",
        init_dest_nodata=True,
    )
    weight = np.zeros(dst_shape, "float32")
    reproject(valid, weight, resampling=Resampling.bilinear, **params)
    mask = np.zeros(dst_shape, "float32")
    reproject(valid, mask, resampling=Resampling.nearest, **params)
    result = np.zeros((*dst_shape, 4), "uint8")
    for band in range(3):
        values = np.zeros(dst_shape, "float32")
        reproject(
            rgba[..., band].astype("float32") * valid,
            values,
            resampling=Resampling.bilinear,
            **params,
        )
        result[..., band] = np.clip(
            np.divide(values, weight, out=np.zeros_like(values), where=weight > 0),
            0,
            255,
        ).astype("uint8")
    result[..., 3] = ((mask > 0.5) & (weight > 0)).astype("uint8") * 255
    return result


class Review:
    """One output directory, multiple areas/products, and an offline viewer.

    add_raster captures a preview immediately, preserving before images even if
    a later producer stage overwrites its input. Original raster files are not copied.
    Registry writes are serialized across workers; unrelated products are retained.
    """

    def __init__(
        self,
        output,
        *,
        title="Raster review",
        max_side=1600,
        render_3d=False,
        mesh_side=500,
        vertical_exaggeration=1.0,
        vertical_unit_to_m=1.0,
        open_browser=False,
        mesh_nodata_policy="strict",
        water_level=0.0,
    ):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output / "review.json"
        self.lock = FileLock(str(self.output / ".review.lock"))
        self.title = title
        self.max_side = int(max_side)
        if self.max_side < 2:
            raise ValueError("max_side must be at least 2")
        self.render_3d = bool(render_3d)
        self.open_browser = bool(open_browser)
        self.mesh_side = int(mesh_side)
        self.vertical_exaggeration = float(vertical_exaggeration)
        self.vertical_unit_to_m = float(vertical_unit_to_m)
        if isinstance(water_level, bool) or not np.isfinite(water_level):
            raise ValueError("water_level must be a finite number")
        self.water_level = float(water_level)
        if mesh_nodata_policy not in {"strict", "overview"}:
            raise ValueError("mesh_nodata_policy must be strict or overview")
        self.mesh_nodata_policy = mesh_nodata_policy
        if self.mesh_side < 2:
            raise ValueError("mesh_side >= 2 required")
        if any(
            not np.isfinite(v) or v <= 0
            for v in (self.vertical_exaggeration, self.vertical_unit_to_m)
        ):
            raise ValueError(
                "3D vertical_exaggeration and vertical_unit_to_m must be positive"
            )
        self.legend_ramps = create_ramps(self.output)
        should_open = False
        with self.lock:
            if not self.path.exists():
                self._save(
                    {
                        "schema_version": 1,
                        "title": title,
                        "running": True,
                        "areas": {},
                        "events": [],
                        "metrics": {},
                        "enable_3d": self.render_3d,
                        "legend_ramps": self.legend_ramps,
                        "session_water_level": self.water_level,
                    }
                )
            else:
                data = self._load()
                changed = data.get("legend_ramps") != self.legend_ramps
                # A fresh viewer run may use a different display water level
                # with the same --out folder. Existing previews are rebuilt by
                # discovery/catalog fingerprints; reset the group configuration
                # so the new run does not reject its own products as conflicting.
                if data.get("session_water_level", 0.0) != self.water_level:
                    for area_data in data.get("areas", {}).values():
                        for group_data in area_data.get("groups", {}).values():
                            if group_data.get("style") == "elevation":
                                group_data["water_level"] = self.water_level
                    changed = True
                if data.get("session_water_level") != self.water_level:
                    data["session_water_level"] = self.water_level
                    changed = True
                if self.render_3d and not data.get("enable_3d", False):
                    data["enable_3d"] = True
                    changed = True
                if changed:
                    data["legend_ramps"] = self.legend_ramps
                    self._save(data)
            # Many producer workers can share one review; open at most one window.
            marker = self.output / ".browser_open_attempted"
            if self.open_browser and not marker.exists():
                marker.touch()
                should_open = True
        if should_open:
            try:
                webbrowser.open((self.output / "viz.html").as_uri())
            except Exception as exc:
                logger.warning("Could not launch HTML review browser: %s", exc)

    def _load(self):
        data = json.loads(self.path.read_text())
        if data["schema_version"] != 1:
            raise ValueError("Unsupported review schema")
        return data

    def render_settings(self, make_3d=None):
        """Semantic preview cache key; source timestamp alone is insufficient."""
        return dict(
            renderer_version=11,
            water_level=self.water_level,
            max_side=self.max_side,
            make_3d=self.render_3d if make_3d is None else bool(make_3d),
            mesh_side=self.mesh_side,
            vertical_unit_to_m=self.vertical_unit_to_m,
            vertical_exaggeration=self.vertical_exaggeration,
            mesh_nodata_policy=self.mesh_nodata_policy,
        )

    def _save(self, data):
        data["updated_epoch"] = time.time()
        atomic(self.path, safe_json(data) + "\n")
        template = Path(__file__).with_name("viewer.html").read_text()
        embedded = safe_json(data).replace("<", "\\u003c")
        rendered = template.replace("__REVIEW_DATA__", embedded)
        # A single offline HTML entry point.
        atomic(self.output / "viz.html", rendered)

    def event(self, text, *, status="info", **metadata):
        with self.lock:
            data = self._load()
            data["events"].append(
                dict(text=str(text), status=status, epoch=time.time(), **metadata)
            )
            data["events"] = data["events"][-MAX_EVENTS:]
            self._save(data)

    def start(self):
        """Mark a review live so existing local HTML refreshes as products arrive."""
        with self.lock:
            data = self._load()
            data["running"] = True
            self._save(data)

    def finish(self):
        with self.lock:
            data = self._load()
            data["running"] = False
            self._save(data)

    def metrics(self, name, value):
        with self.lock:
            data = self._load()
            data["metrics"][name] = value
            self._save(data)

    def defaults(self, left, right, *, area="default"):
        with self.lock:
            data = self._load()
            a = data["areas"][area]
            if left not in a["products"] or right not in a["products"]:
                raise ValueError("Default products must already be registered")
            a["defaults"] = [left, right]
            self._save(data)

    def table(self, name, columns, rows):
        with self.lock:
            data = self._load()
            data.setdefault("tables", {})[name] = {"columns": columns, "rows": rows}
            self._save(data)

    def add_raster(
        self,
        key,
        path,
        *,
        label=None,
        area="default",
        group="elevation",
        stage=None,
        band=1,
        style="elevation",
        limits=None,
        metadata=None,
        views=None,
        water_level=None,
    ):
        start = time.perf_counter()
        if water_level is None:
            water_level = self.water_level
        if isinstance(water_level, bool) or not np.isfinite(water_level):
            raise ValueError("water_level must be finite")
        water_level = float(water_level)
        if style != "elevation" and water_level != self.water_level:
            raise ValueError("water_level applies only to elevation style")
        path = Path(path).resolve()
        # A change/difference raster is not an absolute terrain surface.
        # 3D for differences remains possible only when explicitly requested.
        views = (
            (("2d", "3d") if self.render_3d and style == "elevation" else ("2d",))
            if views is None
            else views
        )
        if isinstance(views, str):
            views = tuple(s.strip().lower() for s in views.split(","))
        if "2d" not in views or set(views) - {"2d", "3d"}:
            raise ValueError("views must include 2d and may include 3d")
        if style == "rgb" and group == "elevation":
            group = "imagery"
        make_3d = "3d" in views
        if style not in {"elevation", "difference"} and make_3d:
            raise ValueError(
                'Non-elevation products are not terrain; choose views=("2d",)'
            )
        if make_3d and not self.render_3d:
            with self.lock:
                data = self._load()
                if not data.get("enable_3d"):
                    data["enable_3d"] = True
                    self._save(data)
        if style not in (
            "elevation",
            "difference",
            "categorical",
            "grayscale",
            "scalar",
            "rgb",
        ):
            raise ValueError("Unknown raster visualization style")
        with rasterio.open(path) as src:
            if src.tags().get("GLOBATO_DATATYPE") == "FUSION_STATE":
                raise ValueError(
                    "Native FusionState is not a display elevation; finalize it first"
                )
            if style == "rgb" and src.count < 3:
                raise ValueError(
                    "RGB display requires 3 bands ordered red, green, blue"
                )
            if not src.crs:
                raise ValueError(
                    "A raster needs a CRS for linked geographic navigation"
                )
            source_cells = (
                sparse_elevation_cells(src, band) if style == "elevation" else None
            )
            if src.transform.b or src.transform.d:
                # WarpedVRT below handles rotation; bounds are the full envelope.
                pass
            t, w, h = calculate_default_transform(
                src.crs, "EPSG:3857", src.width, src.height, *src.bounds
            )
            factor = max(1.0, w / self.max_side, h / self.max_side)
            pw, ph = max(1, math.ceil(w / factor)), max(1, math.ceil(h / factor))
            pt = t * rasterio.Affine.scale(w / pw, h / ph)
            if style == "rgb":
                from .rgb import preview as rgb_preview

                rgb = rgb_preview(src, transform=pt, width=pw, height=ph)
                z = np.where(rgb[..., 3] > 0, 1.0, np.nan)

                def renderer(_limits):
                    return rgb
            else:
                with WarpedVRT(
                    src,
                    crs="EPSG:3857",
                    transform=pt,
                    width=pw,
                    height=ph,
                    resampling=Resampling.nearest,
                    dtype="float64",
                    nodata=float("nan"),
                ) as vrt:
                    z = vrt.read(int(band), masked=True).astype(float).filled(np.nan)
                renderer = None
            if (
                style in {"elevation", "difference"}
                and source_cells is None
                and not src.transform.b
                and not src.transform.d
            ):
                # Nearest-reprojecting elevations before taking derivatives repeats
                # some rows/columns and drops others. A plane then gains a false grid.
                # Read a bounded source-grid preview, shade it, then project its color.
                factor_native = max(
                    1.0, src.width / self.max_side, src.height / self.max_side
                )
                nw, nh = (
                    max(1, math.ceil(src.width / factor_native)),
                    max(1, math.ceil(src.height / factor_native)),
                )
                nz = (
                    src.read(
                        int(band),
                        out_shape=(nh, nw),
                        masked=True,
                        resampling=Resampling.average,
                    )
                    .astype(float)
                    .filled(np.nan)
                )
                nt = src.transform * rasterio.Affine.scale(
                    src.width / nw, src.height / nh
                )
                source_crs = src.crs

                def renderer(limits):
                    return project_shaded_image(
                        nz,
                        nt,
                        source_crs,
                        pt,
                        z.shape,
                        limits,
                        style,
                        self.vertical_unit_to_m,
                        water_level,
                    )

            info = {
                "path": str(path),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "crs": str(src.crs),
                "shape": [src.height, src.width],
                "band": int(band),
                "transform": list(src.transform)[:6],
                "preview_rendering": (
                    "valid_source_cells_as_visible_dots"
                    if source_cells is not None
                    else "rgb_georeferenced_3band"
                    if style == "rgb"
                    else "shade_on_source_grid_then_project_RGBA"
                    if renderer is not None
                    else "warped_grid_rotation_fallback"
                ),
            }
            value_units, vertical_datum = raster_units(
                src, metadata, self.vertical_unit_to_m, style, band=int(band)
            )
        return self._add_preview(
            key,
            z,
            pt,
            label=label,
            area=area,
            group=group,
            stage=stage,
            style=style,
            limits=limits,
            source=info,
            metadata=metadata,
            started=start,
            renderer=renderer,
            value_units=value_units,
            vertical_datum=vertical_datum,
            three_d_path=path if make_3d else None,
            three_d_band=band,
            water_level=water_level,
            source_cells=source_cells,
        )

    def _add_preview(
        self,
        key,
        z,
        transform,
        *,
        label=None,
        area="default",
        group="elevation",
        stage=None,
        style="elevation",
        limits=None,
        source=None,
        metadata=None,
        started=None,
        renderer=None,
        three_d_path=None,
        three_d_band=1,
        value_units="",
        vertical_datum="",
        water_level=0.0,
        source_cells=None,
    ):
        from PIL import Image

        started = time.perf_counter() if started is None else started
        valid = source_cells[2] if source_cells is not None else z[np.isfinite(z)]
        bounds = list(array_bounds(*z.shape, transform))
        # Reserve limits and 3D camera geometry under a short lock, but render
        # images outside the lock. This keeps parallel workers responsive.
        with self.lock:
            data = self._load()
            area_data = data["areas"].setdefault(area, {"products": {}, "groups": {}})
            group_data = area_data["groups"].setdefault(
                group, {"style": style, "limits": None}
            )
            if group_data["style"] != style:
                raise ValueError("A comparison group must use one rendering style")
            if style == "elevation":
                existing_level = group_data.setdefault("water_level", water_level)
                if existing_level != water_level:
                    raise ValueError(
                        "Comparison-group water_level conflict; use matching water_level or a separate group"
                    )
            if limits is not None:
                requested = [float(v) for v in limits]
                if (
                    len(requested) != 2
                    or not np.isfinite(requested).all()
                    or requested[0] >= requested[1]
                ):
                    raise ValueError(
                        "limits must contain two increasing finite numbers"
                    )
                if (
                    group_data["limits"] is not None
                    and group_data["limits"] != requested
                ):
                    raise ValueError(
                        "Group limits already fixed; use a different group"
                    )
                group_data["limits"] = requested
            if group_data["limits"] is None and valid.size:
                lo, hi = (
                    [float(np.min(valid)), float(np.max(valid))]
                    if style == "categorical"
                    else np.percentile(valid, [1, 99])
                )
                if style == "elevation":
                    # Percentile clipping must not hide a sparsely represented
                    # opposite sign; both below- and above-zero cells keep
                    # their physically meaningful color family.
                    if float(np.min(valid)) < water_level and lo >= water_level:
                        lo = float(np.min(valid))
                    if float(np.max(valid)) >= water_level and hi < water_level:
                        hi = float(np.max(valid))
                if style == "difference":
                    hi = max(abs(lo), abs(hi), 1e-9)
                    lo = -hi
                if hi <= lo:
                    lo, hi = lo - 0.5, hi + 0.5
                group_data["limits"] = [float(lo), float(hi)]
            color_limits = list(group_data["limits"] or [0.0, 1.0])
            # Same AOI projection/camera for every product in this group.
            group_data.setdefault("camera_bounds_3857", bounds)
            camera_bounds = list(group_data["camera_bounds_3857"])
            self._save(data)

        image = (
            renderer(color_limits)
            if renderer
            else color_image(
                z,
                transform,
                color_limits,
                style,
                vertical_unit_to_m=self.vertical_unit_to_m,
                water_level=water_level,
            )
        )
        if source_cells is not None:
            from PIL import ImageDraw

            # Symbols are larger than source cell footprints for visibility.
            # Paint from the original cells, not the resampled elevation grid.
            canvas = Image.new("RGBA", (z.shape[1], z.shape[0]))
            draw = ImageDraw.Draw(canvas)
            px, py = ~transform * (source_cells[0], source_cells[1])
            colors = np.clip(
                elevation_rgba(valid, color_limits, water_level=water_level) * 255,
                0,
                255,
            ).astype("uint8")
            radius = max(2, round(max(z.shape) / 320))
            for x, y, color in zip(px, py, colors):
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill=tuple(map(int, color)),
                    outline=(255, 255, 255, 255),
                    width=1,
                )
            image = np.asarray(canvas)
        name = hashlib.sha256((area + "\0" + key).encode()).hexdigest()[:24]
        folder = self.output / "images"
        folder.mkdir(exist_ok=True)
        png = folder / (name + ".png")
        temp = folder / (name + "." + uuid.uuid4().hex + ".tmp")
        try:
            Image.fromarray(image).save(temp, format="PNG")
            os.replace(temp, png)
        finally:
            temp.unlink(missing_ok=True)
        # Publish 2D immediately; the HTML can be opened while 3D is preparing.
        record = {
            "label": label or key,
            "stage": stage,
            "group": group,
            "style": style,
            "image": str(png.relative_to(self.output)),
            "bounds": bounds,
            "display_crs": "EPSG:3857",
            "preview_shape": list(z.shape),
            "preview_valid_cells": int(valid.size),
            "limits": color_limits,
            "water_level": water_level if style == "elevation" else None,
            "source": source,
            "metadata": metadata or {},
            "legend": (
                {
                    "palette": COLORMAPS[style],
                    "ramp": (
                        elevation_ramp(
                            self.output, color_limits, water_level=water_level
                        )
                        if style == "elevation"
                        else self.legend_ramps[style]
                    ),
                    "title": value_label(style),
                    "units": value_units,
                    "datum": vertical_datum,
                }
                if style in COLORMAPS and valid.size
                else None
            ),
            "review_seconds": time.perf_counter() - started,
            "render_settings": self.render_settings(three_d_path is not None),
        }
        if source_cells is not None:
            record["preview_valid_cells"] = int(np.count_nonzero(image[..., 3]))
            record["sparse_display"] = {
                "valid_source_cells": int(valid.size),
                "symbol": "circle",
                "radius_preview_pixels": radius,
                "note": "Display symbols only; source values, cell footprints and NoData are unchanged",
            }
        if style == "rgb" and self.render_3d:
            # Browsers often refuse WebGL texture uploads sourced directly from
            # file:// images. A companion local JS asset carries a data: URI,
            # just as 3D scene assets use JS instead of fetch(). This remains
            # entirely standalone/offline and avoids bloating viz.html.
            rgb_key = name
            rgb_js = folder / (name + ".rgb.js")
            uri = "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode(
                "ascii"
            )
            atomic(
                rgb_js,
                "window.__reviewRgbData=window.__reviewRgbData||{};"
                "window.__reviewRgbData["
                + json.dumps(rgb_key)
                + "]="
                + json.dumps(uri)
                + ";\n",
            )
            record["rgb_data_script"] = str(rgb_js.relative_to(self.output))
            record["rgb_data_key"] = rgb_key
        if three_d_path is not None:
            record["terrain_3d_status"] = "preparing"
        with self.lock:
            data = self._load()
            data["areas"][area]["products"][key] = record
            self._save(data)
        if three_d_path is not None:
            scene_error = None
            scene_info = None
            from .interactive_3d import publish_scene

            # RGB is now independently rendered from its 2D georeferenced
            # preview, not sampled onto each terrain mesh. This removes costly
            # per-terrain RGB resampling and decouples imagery from NoData.
            version = hashlib.sha256(
                safe_json(
                    {
                        "source": source,
                        "settings": self.render_settings(True),
                        "limits": color_limits,
                        "water_level": water_level,
                        "camera": camera_bounds,
                        "palette": COLORMAPS.get(style),
                    }
                ).encode()
            ).hexdigest()[:12]
            scene_id = name + "_" + version
            scene_file = self.output / "scenes" / (scene_id + ".js")
            try:
                scene_info = publish_scene(
                    three_d_path,
                    scene_file,
                    scene_id,
                    camera_bounds,
                    color_limits,
                    band=three_d_band,
                    mesh_side=self.mesh_side,
                    vertical_unit_to_m=self.vertical_unit_to_m,
                    default_exaggeration=self.vertical_exaggeration,
                    style=style,
                    nodata_policy=self.mesh_nodata_policy,
                    water_level=water_level,
                )
                scene_info["script"] = str(scene_file.relative_to(self.output))
            except Exception as exc:
                scene_error = str(exc)
                logger.warning("Interactive 3D scene skipped for %s: %s", key, exc)
            with self.lock:
                data = self._load()
                # Other workers may have created newer products meanwhile.
                rec = data["areas"][area]["products"][key]
                rec["terrain_3d_status"] = "ready" if scene_info else "error"
                if scene_info:
                    rec["terrain_scene"] = scene_info
                if scene_error:
                    rec["terrain_3d_error"] = scene_error
                    data["events"].append(
                        dict(
                            text=f"{label or key}: interactive 3D skipped: {scene_error}",
                            status="skipped",
                            epoch=time.time(),
                        )
                    )
                    data["events"] = data["events"][-MAX_EVENTS:]
                rec["review_seconds"] = time.perf_counter() - started
                record = dict(rec)
                self._save(data)
        return record

    def add_vector(
        self,
        key,
        path,
        *,
        label=None,
        area="default",
        stage=None,
        metadata=None,
        display=None,
        layer=None,
        max_features=150,
        max_vertices=12000,
    ):
        """Register an optional vector overlay without depending on producer names.

        Safely simplify and bound geometry for offline 2D preview; original
        features are never changed. Interactive 3D vector draping is not claimed.
        """
        from pyproj import Transformer
        from shapely.geometry import mapping
        from shapely.ops import transform as geometry_transform
        from .vectors import read_features

        path = Path(path).resolve()
        if path.suffix.lower() not in {".shp", ".geojson", ".gpkg"}:
            raise ValueError("Supported vector types: .shp, .geojson, .gpkg")
        if layer is None and path.suffix.lower() == ".gpkg":
            layer = (metadata or {}).get("layer")
        crs, geometries, backend = read_features(path, max_features, layer=layer)
        if not crs:
            raise ValueError("Vector needs a CRS")
        to_map = Transformer.from_crs(crs, "EPSG:3857", always_xy=True).transform
        features = []
        vertices = 0

        def positions(coords):
            if not isinstance(coords, (tuple, list)) or not coords:
                return 0
            if isinstance(coords[0], (int, float)):
                return 1
            return sum(positions(c) for c in coords)

        def count_geometry(g):
            if g["type"] == "GeometryCollection":
                return sum(count_geometry(x) for x in g.get("geometries", []))
            return positions(g.get("coordinates", []))

        for geometry in geometries:
            geom = geometry_transform(to_map, geometry)
            if geom.is_empty:
                continue
            mapped = mapping(geom)
            vertices += count_geometry(mapped)
            if vertices > max_vertices:
                raise ValueError(
                    f"Vector exceeds {max_vertices} vertices; select/simplify first"
                )
            features.append(mapped)
        stat = path.stat()
        record = {
            "label": label or key,
            "stage": stage,
            "metadata": metadata or {},
            "source": {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            },
            "geometry_crs": "EPSG:3857",
            "geometry_count": len(features),
            "reader_backend": backend,
            "render_settings": {"vector_renderer_version": 5},
            "geometry": features,
            "layer": layer,
            "view": "2d overlay",
            "display": dict(display or {}),
        }
        with self.lock:
            data = self._load()
            a = data["areas"].setdefault(area, {"products": {}, "groups": {}})
            a.setdefault("vectors", {})[key] = record
            self._save(data)
        return record
