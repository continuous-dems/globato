"""Source and coverage invariants, independent of framework execution."""

from pathlib import Path
import math
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import box
from shapely.ops import unary_union
from ._io import digest, raster_files, serializable


def preparation_scope_with_tier_support(declared_aoi, evidence_bounds, evidence_crs):
    """Cover the declared AOI and the raster tiers in a common 2D CRS.

    The polygon supplied by the user is not modified. This working AOI is
    separate from the water/land input mask. Native source selection and
    delivery cropping continue to use the original recipe region.

    EPSG:5498 is a *compound* NAD83 + NAVD88 CRS; only its horizontal part
    applies to polygon extents. Never use vertical transforms for 2D support.
    """
    source_crs = CRS.from_user_input(evidence_crs)
    horizontal = source_crs.sub_crs_list[0] if source_crs.is_compound else source_crs
    if not horizontal.is_geographic and not horizontal.is_projected:
        raise ValueError("Tier support needs a horizontal raster CRS")
    support = (
        gpd.GeoSeries([box(*evidence_bounds)], crs=horizontal).to_crs(4326).iloc[0]
    )
    expanded = unary_union([declared_aoi, support])
    if not expanded.covers(declared_aoi) or not expanded.covers(support):
        # GEOS unions of nearly coincident projected edges sometimes leave an
        # ULP-scale gap. Only in that case, guard the *temporary preparation
        # scope* by ~0.01 mm; do not buffer the supplied local water polygon,
        # the native raster footprint, the mask, or any final DEM grid.
        # A real uncovered area larger than numerical roundoff still fails.
        magnitude = max(1.0, *(abs(n) for n in (*declared_aoi.bounds, *support.bounds)))
        numerical_guard = max(1e-10, 64 * math.ulp(magnitude))
        expanded = unary_union(
            [
                declared_aoi.buffer(numerical_guard, join_style=2),
                support.buffer(numerical_guard, join_style=2),
            ]
        )
    if (
        expanded.is_empty
        or not expanded.is_valid
        or expanded.geom_type not in ("Polygon", "MultiPolygon")
    ):
        raise ValueError(
            "Prepared support and declared AOI do not form a valid polygon"
        )
    if not expanded.covers(declared_aoi) or not expanded.covers(support):
        raise ValueError("Preparation scope lost original AOI or tier support")
    return expanded


def source_snapshot(module, entry, path, *, point_source=False):
    """Scientific identity: source-owned weights, claims, CRS and raster bytes."""
    archive = entry.get("source_archive")
    if archive is not None:
        if digest(Path(archive["path"]).resolve(strict=True)) != archive["sha256"]:
            raise ValueError("Native source archive changed since early extraction")
    original = entry.get("raster_original")
    if original is not None:
        for name, expected in original["files"].items():
            if digest(Path(name).resolve(strict=True)) != expected:
                raise ValueError("Original raster changed since TIFF normalization")
    fields = (
        "url",
        "source_id",
        "contribution_id",
        "stack_partition",
        "weight",
        "src_srs",
        "source_archive",
        "raster_original",
        "claim_required",
        "claim_geometry",
        "accepted_geometry",
        "excluded_geometry",
        "claim_priority",
    )
    return {
        "module": module.name,
        "module_weight": float(getattr(module, "weight", 1)),
        "source": str(path),
        "files": (
            [{"name": path.name, "sha256": digest(path)}]
            if point_source
            else raster_files(path)
        ),
        "entry": {key: serializable(entry.get(key)) for key in fields},
    }
