"""Classed publication and change audit. No fetching, raster streaming or new science."""

from pathlib import Path
import json
import tempfile
import numpy as np
import rasterio
from rasterio.features import rasterize
from .application import _polygonize_labels
import geopandas as gpd
import shapely
from ._io import json_write, read_polygon, write_polygon
from .baseline import LAND, INLAND, CLASSES
from ._science.coastline_classify import polygon_water
from .connected_water import publish_connected_water


def publish_topology(root, baseline, aoi):
    """Partition the AOI using scientific classes, retaining native class detail.

    Existing water subtype labels are sampled on the native application grid.
    New marine water is ocean, new nonmarine water is water; original island,
    reef and breakwater labels survive only where final science says land.
    There is no structure-specific override or minimum-area deletion.
    """
    root = Path(root)
    original = gpd.read_file(baseline.paths["original_topology"])
    grid_path = root / "application_grid.json"
    grid = json.loads(grid_path.read_text())
    crs = grid["crs"]
    source = original.to_crs(crs)
    final_water = read_polygon(
        root / "refined_water.gpkg", crs=crs, require_native=True
    )
    marine = read_polygon(
        root / "marine_water.gpkg", crs=crs, require_native=True
    ).intersection(final_water)
    land = read_polygon(root / "refined_land.gpkg", crs=crs, require_native=True)
    extent = shapely.union_all([land, final_water])
    frame = _topology_frame(root, source, land, final_water, marine, grid)
    verify_partition(frame, extent, crs=crs)
    target = root / "refined_topology.gpkg"
    frame.to_file(target, layer="topology", driver="GPKG")
    verify_partition(gpd.read_file(target), extent, crs=crs)
    # Audit in geographic coordinates independently of publication geometry.
    original = original.to_crs(4326)
    final_water = polygon_water(
        read_polygon(root / "refined_water.gpkg").intersection(aoi)
    )
    land = aoi.difference(final_water)
    baseline_water = polygon_water(
        read_polygon(baseline.paths["water_mask"]).intersection(aoi)
    )
    added_land = baseline_water.difference(final_water)
    added_water = final_water.difference(baseline_water)
    write_polygon(root / "added_land.gpkg", added_land)
    write_polygon(root / "added_water.gpkg", added_water)

    def area_m2(geom):
        return float(gpd.GeoSeries([geom], crs=4326).to_crs(6933).area.iloc[0])

    by_class = {}
    if "class" in original:
        for name, rows in original.groupby("class"):
            geom = polygon_water(shapely.union_all(rows.geometry).intersection(aoi))
            by_class[name] = {
                "added_land_m2": area_m2(added_land.intersection(geom)),
                "added_water_m2": area_m2(added_water.intersection(geom)),
            }
    audit = {
        "added_land_m2": area_m2(added_land),
        "added_water_m2": area_m2(added_water),
        "by_original_class": by_class,
        "policy": "Changes reported without automatic rollback or structure-specific rules",
        "area_method": "EPSG:6933 equal-area; input polygons in EPSG:4326",
    }
    if "unclassified" in baseline.paths:
        gap = polygon_water(
            read_polygon(baseline.paths["unclassified"]).intersection(aoi)
        )
        write_polygon(root / "original_unclassified.gpkg", gap)
        audit["original_unclassified"] = {
            "area_m2": area_m2(gap),
            "refined_land_m2": area_m2(gap.intersection(land)),
            "refined_water_m2": area_m2(gap.intersection(final_water)),
            "policy": "Uncovered native area is not original OSM land or water evidence",
        }
    json_write(root / "change_audit.json", audit)
    # Companion for future unified guidance; never substitutes the legacy
    # refined_water/refined_topology used by source masking and barriers.
    publish_connected_water(root)
    return audit


def _water_subtypes(root, source, water, marine, grid):
    """Assign whole native water cells using the existing subtype priority.

    This changes class publication only. Refined water, marine connectivity,
    native source retention and original input vectors are never rewritten.
    """
    domain = polygon_water(water.difference(marine))
    if domain.is_empty:
        return []
    codes = {name: index for index, name in enumerate(INLAND, 1)}
    # GDAL's last feature wins; reverse the existing first-match priority.
    burn = []
    for name in reversed(INLAND):
        geometry = shapely.union_all(source.loc[source["class"] == name].geometry)
        if not geometry.is_empty:
            burn.append((geometry, codes[name]))
    if not burn:
        return [("water", domain)]
    transform = rasterio.Affine(*grid["transform"])
    profile = dict(
        driver="GTiff",
        width=grid["width"],
        height=grid["height"],
        count=1,
        dtype="uint8",
        crs=grid["crs"],
        transform=transform,
    )
    with tempfile.TemporaryDirectory(
        prefix="topology_cells_", dir=Path(root).parent
    ) as temporary:
        path = Path(temporary) / "subtypes.tif"
        with rasterio.open(path, "w", **profile) as dst:
            for start in range(0, dst.height, 256):
                rows = min(256, dst.height - start)
                window = rasterio.windows.Window(0, start, dst.width, rows)
                affine = rasterio.windows.transform(window, transform)
                present = rasterize(
                    [(domain, 1)],
                    out_shape=(rows, dst.width),
                    transform=affine,
                    fill=0,
                    dtype="uint8",
                ).astype(bool)
                labels = (
                    rasterize(
                        burn,
                        out_shape=(rows, dst.width),
                        transform=affine,
                        fill=0,
                        dtype="uint8",
                    )
                    if present.any()
                    else np.zeros((rows, dst.width), dtype="uint8")
                )
                labels[~present] = 0
                labels[present & (labels == 0)] = codes["water"]
                dst.write(labels, 1, window=window)
        polygons = _polygonize_labels(path, 1)
    return [(name, polygons[code]) for name, code in codes.items() if code in polygons]


def _topology_frame(root, source, land, water, marine, grid):
    """Keep land-detail geometry; publish water subtypes on the sealed grid."""
    parts = []
    remaining_land = land
    for name in LAND[1:]:
        old = shapely.union_all(source.loc[source["class"] == name].geometry)
        geometry = polygon_water(old.intersection(remaining_land))
        if not geometry.is_empty:
            parts.append((name, geometry))
            remaining_land = remaining_land.difference(geometry)
    parts.extend(_water_subtypes(root, source, water, marine, grid))
    for name, geometry in (("land", remaining_land), ("ocean", marine)):
        geometry = polygon_water(geometry)
        if not geometry.is_empty:
            parts.append((name, geometry))
    frame = gpd.GeoDataFrame(
        {"class": [name for name, _ in parts]},
        geometry=[geometry for _, geometry in parts],
        crs=grid["crs"],
    )
    verify_partition(frame, shapely.union_all([land, water]), crs=grid["crs"])
    return frame


def verify_partition(frame, aoi, *, crs="EPSG:4326"):
    """Check a partition in the declared domain CRS, without a datum round trip."""
    if (
        frame.crs is None
        or frame.empty
        or frame.geometry.isna().any()
        or not frame.geom_type.isin(["Polygon", "MultiPolygon"]).all()
        or "class" not in frame
        or frame["class"].isna().any()
        or not set(frame["class"]) <= set(CLASSES)
    ):
        raise ValueError("Invalid refined topology")
    for name, geom in zip(frame["class"], frame.geometry, strict=False):
        if not geom.is_valid:
            raise ValueError(
                f"Invalid refined topology class {name!r}: {shapely.is_valid_reason(geom)}"
            )
    if frame.crs != gpd.GeoSeries([], crs=crs).crs:
        raise ValueError(
            f"Refined topology CRS disagrees with publication domain {crs}"
        )
    geoms = list(frame.geometry)
    # Numerical tolerance only; never buffer or delete published slivers.
    tolerance = max(aoi.area * 1e-12, 1e-20)
    union = shapely.union_all(geoms)
    if union.symmetric_difference(aoi).area > tolerance:
        raise ValueError("Refined topology has gaps or exceeds AOI")
    if sum(geom.area for geom in geoms) - union.area > tolerance:
        raise ValueError("Refined topology classes overlap")
