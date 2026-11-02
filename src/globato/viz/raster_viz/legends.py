"""Consistent Matplotlib color ramps and reliable raster value labels.

The display label uses explicit catalog metadata or information embedded in the
GeoTIFF. A horizontal CRS alone never establishes vertical units or datum.
"""

from __future__ import annotations

from pathlib import Path

import hashlib
import json

import numpy as np
from matplotlib import colormaps
from PIL import Image

from .palette import ELEVATION_PALETTE, elevation_rgba

COLORMAPS = {
    "elevation": ELEVATION_PALETTE,
    "difference": "RdBu_r",
    "scalar": "viridis",
    "grayscale": "gray",
}


def create_ramps(output: Path) -> dict[str, str]:
    """Write bounded, deterministic assets. Category rasters have no ramp."""
    output = Path(output)
    folder = output / "legends"
    folder.mkdir(parents=True, exist_ok=True)
    assets = {}
    for style, name in COLORMAPS.items():
        dest = folder / (name + ".png")
        if not dest.is_file() or not dest.stat().st_size:
            values = (np.arange(1024, dtype="float64") + 0.5) / 1024
            rgba = (
                elevation_rgba(np.linspace(-100, 100, len(values)), [-100, 100])
                if style == "elevation"
                else colormaps[name](values)
            )
            rgb = (rgba[:, :3] * 255).clip(0, 255).astype("uint8")
            Image.fromarray(np.tile(rgb[None, :, :], (18, 1, 1)), "RGB").save(dest)
        assets[style] = str(dest.relative_to(output))
    return assets


def elevation_ramp(output: Path, limits, water_level=0.0) -> str:
    """Publish a scale-accurate shoreline ramp for these exact display limits."""
    output = Path(output)
    bounds = [float(x) for x in limits]
    token = hashlib.sha256(
        json.dumps([ELEVATION_PALETTE, bounds, float(water_level)]).encode()
    ).hexdigest()[:20]
    dest = output / "legends" / ("elevation_" + token + ".png")
    if not dest.is_file() or not dest.stat().st_size:
        dest.parent.mkdir(parents=True, exist_ok=True)
        values = np.linspace(bounds[0], bounds[1], 1024)
        rgb = (
            np.rint(
                elevation_rgba(values, bounds, water_level=water_level)[:, :3] * 255
            )
            .clip(0, 255)
            .astype("uint8")
        )
        Image.fromarray(np.tile(rgb[None, :, :], (18, 1, 1)), "RGB").save(dest)
    return str(dest.relative_to(output))


def value_label(style):
    return {
        "elevation": "Elevation",
        "difference": "Elevation difference",
        "scalar": "Value",
        "grayscale": "Grayscale value",
    }.get(style, "Value")


def _first(tags, names):
    """Match common GDAL tag spellings without depending on their case."""
    lowered = {str(k).lower(): v for k, v in tags.items()}
    for name in names:
        value = lowered.get(name.lower())
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _unit_label(unit):
    """Normalize common units without converting or altering source elevations."""
    if not unit:
        return ""
    raw = str(unit).strip()
    common = {
        "meter": "m",
        "metre": "m",
        "meters": "m",
        "metres": "m",
        "m": "m",
        "foot": "ft",
        "feet": "ft",
        "international foot": "ft",
        "ft": "ft",
        "us survey foot": "ftUS",
        "us survey feet": "ftUS",
        "u.s. survey foot": "ftUS",
        "foot_us": "ftUS",
        "ftus": "ftUS",
    }
    return common.get(raw.casefold(), raw)


def _vertical_crs_info(crs):
    """Only a genuine vertical component authorizes datum/units inference."""
    if crs is None:
        return "", ""
    try:
        from pyproj import CRS

        parsed = CRS.from_user_input(crs)
        # pyproj considers a compound CRS with a vertical axis is_vertical=True,
        # so inspect the compound members *first* to avoid including NAD83.
        if parsed.is_compound:
            vertical = next((c for c in parsed.sub_crs_list if c.is_vertical), None)
        elif parsed.is_vertical:
            vertical = parsed
        else:
            vertical = None
        if vertical is None:
            return "", ""
        axis = next(
            (a for a in vertical.axis_info if a.direction.lower() in ("up", "down")),
            None,
        )
        unit = _unit_label(axis.unit_name) if axis else ""
        datum = vertical.name
        # EPSG commonly names the vertical CRS "NAVD88 height". Present the
        # recognizable datum name, while keeping the source CRS unchanged.
        if datum.endswith(" height"):
            datum = datum[: -len(" height")]
        return unit, datum
    except (ValueError, TypeError, AttributeError):
        return "", ""


def raster_units(src, metadata, vertical_unit_to_m, style, band=1):
    """Return (value units, vertical datum) for a *display legend*.

    Priority: explicit catalog metadata, TIFF band/dataset tags, vertical CRS.
    A vertical CRS does not describe arbitrary scalar or grayscale values.
    Difference rasters carry units but must not claim an absolute height datum.
    ``vertical_unit_to_m`` is an optional caller-supplied *display conversion*;
    it does not transform vertical datums or imply an unknown source datum.
    """
    meta = metadata or {}
    tagged = {**src.tags(), **src.tags(band)}
    unit_names = ("value_units", "vertical_units", "z_units", "units", "unittype")
    datum_names = ("vertical_datum", "vertical_reference", "vdatum")
    unit = _first(meta, unit_names) or _first(tagged, unit_names)
    datum = _first(meta, datum_names + ("datum",)) or _first(tagged, datum_names)
    if not unit:
        # GDAL/Rasterio exposes band unit strings through DatasetReader.units,
        # independently of ordinary TIFF tags.
        units = src.units or ()
        if band <= len(units):
            unit = units[band - 1] or ""
    if style in ("elevation", "difference"):
        crs_unit, crs_datum = _vertical_crs_info(src.crs)
        unit = unit or crs_unit
        datum = datum or crs_datum
        if not unit and abs(vertical_unit_to_m - 0.3048) < 1e-8:
            unit = "ft"
    else:
        # Elevation-specific metadata is not a valid legend for an arbitrary
        # scalar, grayscale, or RGB raster. Require explicit value units.
        unit = _first(meta, ("value_units", "units")) or _first(
            tagged, ("value_units", "units", "unittype")
        )
        datum = ""
    if style != "elevation":
        datum = ""
    return _unit_label(unit), datum
