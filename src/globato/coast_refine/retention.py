"""Source-use geography, independent of the signed interpolation classification.

Below-zero observations cannot enlarge a source's exclusion polygon. The
starting water domain can shrink only through the existing screened land
restoration; it cannot grow through the barrier's signed water additions.
"""

from pathlib import Path

from ._io import read_polygon, write_polygon

CONTRACT = "independent_source_retention_and_interpolation_v1"
KEYS = (
    "water",
    "land",
    "marine_water",
    "inland_water",
    "unresolved_water",
    "classified_ocean",
    "classified_inland_water",
)


def validate_source_water(water, original):
    """One containment rule for preparation and sealed-artifact loading."""
    from .artifact import _covers_without_roundoff

    # GEOS can leave ULP-wide overlay residuals. Validate only the residual,
    # using the existing guard; never buffer or snap a published mask.
    outside = water.difference(original)
    if not outside.is_empty and not _covers_without_roundoff(original, outside):
        raise ValueError("Source retention exclusion exceeds starting water domain")


def publish_retention(pending, baseline, *, terrain):
    """Reuse scientific connectivity; keep original native-source geography.

    The interpolation products retain their target-cell lattice. Source-use
    polygons stay geographic so the existing TIFF writer decides at each
    source's original cell centre, without imposing an interpolation grid.
    No separate source discovery, evidence stack or terrain algorithm runs.
    """
    from .application import declared_water_subtypes
    from ._science.raster_science import regional_water_components
    from ._science.coastline_classify import polygon_water

    root = Path(pending)
    context = read_polygon(root / "coverage.gpkg")
    original = read_polygon(root / "original_water.gpkg")
    refined = read_polygon(root / "science_vectors/refined_water.gpkg")
    water = refined.intersection(original).difference(terrain)
    if not water.is_valid:
        raise ValueError("Source-retention water intersection is invalid")
    validate_source_water(water, original)
    anchor = read_polygon(baseline.paths["ocean_anchor"])
    groups, connectivity, _ = regional_water_components(
        water, context, None, original, ocean_anchor=anchor
    )
    ocean, inland = declared_water_subtypes(baseline)
    products = {
        "water": water,
        "land": context.difference(water),
        "marine_water": groups["marine"],
        "inland_water": groups["inland"],
        "unresolved_water": groups["unresolved"],
        "classified_ocean": polygon_water(ocean.intersection(water)),
        "classified_inland_water": polygon_water(inland.intersection(water)),
    }
    for name, geometry in products.items():
        write_polygon(root / ("retention_" + name + ".gpkg"), geometry)
    return {
        "contract": CONTRACT,
        "source_mask_coordinates": "EPSG:4326; evaluated at original source cell centres",
        "source_water_rule": "starting water intersect screened refined water, minus retained wetland terrain",
        "negative_elevation_alone_adds_source_exclusion": False,
        "barrier_rule": "existing screened signed classification; target-cell topology",
        "barrier_does_not_clip_original_observations": True,
        "source_connectivity": connectivity,
        "supporting_source_fusion": "not implemented; one authoritative module group",
    }
