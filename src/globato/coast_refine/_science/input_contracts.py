#!/usr/bin/env python3
"""Generic input identity and baseline-freezing utilities for Coast Refine."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from .._io import digest as sha256
import shutil

from pyproj import CRS
import rasterio
from rasterio.windows import Window


def check_encoding(src):
    """One contract for preflight, evidence and native TIFF publication."""
    if src.crs is None or src.count != 1 or src.dtypes[0] not in ("float32", "float64"):
        raise ValueError(
            "Coast Refine requires a single floating elevation band with CRS"
        )
    if (
        src.transform.b
        or src.transform.d
        or src.transform.a <= 0
        or src.transform.e >= 0
    ):
        raise ValueError("Coast Refine requires a north-up native raster")
    if src.scales != (1.0,) or src.offsets != (0.0,):
        raise ValueError("Packed elevations require source-owned decoding")
    if src.units[0] and src.units[0].strip().lower() not in (
        "m",
        "meter",
        "meters",
        "metre",
        "metres",
    ):
        raise ValueError("Coast Refine requires metre elevations")


def raster_signature(path):
    """Fixed row strips make pixel fingerprints independent of TIFF layout."""
    with rasterio.open(path) as ds:
        if ds.count != 1:
            raise ValueError("RAW must contain exactly one band")
        gcps, gcp_crs = ds.gcps
        signature = {
            "width": ds.width,
            "height": ds.height,
            "count": ds.count,
            "dtypes": list(ds.dtypes),
            "transform": list(ds.transform),
            "crs": None if ds.crs is None else ds.crs.to_wkt(),
            "nodata": [repr(x) for x in ds.nodatavals],
            "scales": list(ds.scales),
            "offsets": list(ds.offsets),
            "gcps": [[g.row, g.col, g.x, g.y, g.z] for g in gcps],
            "gcp_crs": None if gcp_crs is None else gcp_crs.to_wkt(),
        }
        values, validity = hashlib.sha256(), hashlib.sha256()
        for row in range(0, ds.height, 256):
            window = Window(0, row, ds.width, min(256, ds.height - row))
            values.update(ds.read(1, window=window).tobytes())
            validity.update((ds.read_masks(1, window=window) > 0).tobytes())
        signature["band_1_values_sha256"] = values.hexdigest()
        signature["band_1_validity_sha256"] = validity.hexdigest()
    return signature


def signature_differences(actual, expected):
    changed = []
    for key in expected:
        if key in ("crs", "gcp_crs") and actual[key] and expected[key]:
            same = CRS.from_user_input(actual[key]).equals(
                CRS.from_user_input(expected[key]), ignore_axis_order=True
            )
        else:
            same = actual[key] == expected[key]
        if not same:
            changed.append(key)
    return changed


def select_baseline(
    destination, *, water_mask, mask_layer=None, source_kind=None, allow_empty=False
):
    """Normalize and freeze an explicitly selected water polygon.

    The universal production path never consults a bundled geographic baseline.
    Callers either supply their own mask or first build one from the current
    tile catalog and then pass that generated mask here.
    """
    if water_mask is None:
        raise ValueError("select_baseline requires an explicit water-mask path")
    destination = Path(destination).expanduser().resolve()
    target = destination / "masks/baseline_water_mask.gpkg"
    if target.exists():
        raise ValueError(
            "Baseline destination exists; refusing to overwrite: " + str(target)
        )

    if mask_layer and not water_mask:
        raise ValueError("--mask-layer requires --water-mask")

    from .coast_refine_v6_6_1.input_resolver import normalize_existing_mask, read_vector

    source = Path(water_mask).expanduser().resolve(strict=True)
    frame, layer = read_vector(source, mask_layer, preferred=("mask",))
    if not frame.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError(
            "Supply water-area polygons; a shoreline line does not define its water side"
        )
    normalized, metadata = normalize_existing_mask(
        source,
        destination / "baseline_input",
        mask_layer=mask_layer,
        allow_empty=allow_empty,
    )
    kind = source_kind or "supplied_water_mask"

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(normalized, target)
    if sha256(normalized) != sha256(target):
        raise ValueError("Selected baseline copy failed")

    receipt = {
        "status": "PASS",
        "source_kind": kind,
        "source": str(source),
        "source_sha256": sha256(source),
        "selected_layer": metadata["layer"],
        "normalization": metadata,
        "baseline": str(target),
        "baseline_sha256": sha256(target),
        "input_modified": False,
        "water_domain": "inside supplied polygons; outside is starting land",
        "universal_contract": "no bundled ROI-specific baseline may be selected implicitly",
    }
    dependencies = [source]
    if source.suffix.lower() == ".shp":
        dependencies += [
            source.with_suffix(suffix)
            for suffix in (".shx", ".dbf", ".prj", ".cpg")
            if source.with_suffix(suffix).is_file()
        ]
    receipt["source_files_sha256"] = {str(path): sha256(path) for path in dependencies}
    review = destination / "review"
    review.mkdir(parents=True, exist_ok=True)
    (review / "baseline_input.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
