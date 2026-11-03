"""Target-cell representation for the interpolation topology.

The existing scientific vectors are preserved separately. Wetland habitat is
retained terrain, and newly disconnected coastal water polygons smaller than
one target cell are removed under the explicit user policy. The remaining
classification is sampled at target centres and polygonized on cell edges.
No global smoothing, closing, or deletion of mapped small islands is applied.
"""

from pathlib import Path
import logging
import tempfile
import numpy as np
import rasterio
import shapely
from shapely.geometry import shape
from rasterio.features import shapes, rasterize
from pyproj import Transformer
from .selection import horizontal_crs
from ._io import write_polygon, json_write

log = logging.getLogger(__name__)
_DOMAINS = ("refined_water", "marine_water", "inland_water")


def _polygonize_labels(tif, band):
    """Polygonize nonzero class codes once, using GDAL's existing cell edges."""
    grouped = {}
    with rasterio.open(tif) as src:
        for geometry, code in shapes(
            rasterio.band(src, band), transform=src.transform, connectivity=4
        ):
            if code != 0:
                grouped.setdefault(int(code), []).append(shape(geometry))
    output = {}
    for code, polygons in grouped.items():
        value = shapely.union_all(polygons)
        if not value.is_valid or value.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(
                "Application-grid polygonization produced invalid topology"
            )
        output[code] = value
    return output


def _polygonize_flag(tif, band):
    """Stream GDAL's edge-aligned polygonization, retaining foreground only."""
    return _polygonize_labels(tif, band).get(1, shapely.Polygon())


def _assert_grid_equivalence(tif, products):
    """Verify published vector/raster mask parity for *every* application cell."""
    from ._io import read_polygon

    with rasterio.open(tif) as src:
        if src.crs is None:
            raise ValueError("Application mask has no CRS")
        for index, key in enumerate(_DOMAINS, 1):
            # The published vector must reproduce every application-grid cell,
            # not just a random sample of important boundary pixels.
            projected = read_polygon(products[key], crs=src.crs, require_native=True)
            recaptured = (
                np.zeros((src.height, src.width), dtype="uint8")
                if projected.is_empty
                else rasterize(
                    [(projected, 1)],
                    out_shape=(src.height, src.width),
                    transform=src.transform,
                    fill=0,
                    all_touched=False,
                    dtype="uint8",
                )
            )
            if not np.array_equal(recaptured, src.read(index)):
                raise ValueError(
                    f"Cleaned {key} changed an application-grid cell classification"
                )
            products[key + "_cells"] = int(recaptured.sum())
        # The science distinguishes marine, inland and unresolved water.
        # Publishing overlapping or out-of-water marine/inland cells would
        # silently misclassify terrain and corrupt the unresolved count.
        for _, window in src.block_windows(1):
            water = src.read(1, window=window).astype(bool)
            marine = src.read(2, window=window).astype(bool)
            inland = src.read(3, window=window).astype(bool)
            if np.any(marine & inland):
                raise ValueError("Marine and inland water overlap on target-grid cells")
            if np.any((marine | inland) & ~water):
                raise ValueError(
                    "Marine/inland water includes cells outside refined water"
                )
        # The fourth vector must be the true complement of the *published*
        # refined-water layer on this grid, not merely a valid GEOS polygon.
        # An overlap/gap introduced by vector serialization must fail closed.
        land = read_polygon(products["refined_land"], crs=src.crs, require_native=True)
        land_cells = (
            np.zeros((src.height, src.width), dtype="uint8")
            if land.is_empty
            else rasterize(
                [(land, 1)],
                out_shape=(src.height, src.width),
                transform=src.transform,
                fill=0,
                all_touched=False,
                dtype="uint8",
            )
        )
        water_cells = src.read(1)
        if not np.array_equal(land_cells, 1 - water_cells):
            raise ValueError(
                "Published clean land/water polygons do not complement at grid-cell centers"
            )


def _in_grid_crs(geom, crs, *, report_repair=False):
    """Reproject just once for GDAL rasterization, never repair raw science."""
    from rasterio.crs import CRS

    if CRS.from_user_input(crs) == CRS.from_epsg(4326):
        return (geom, False) if report_repair else geom
    from shapely.ops import transform

    transformer = Transformer.from_crs("EPSG:4326", horizontal_crs(crs), always_xy=True)
    value = transform(transformer.transform, geom)
    repaired_projection = not value.is_valid
    if repaired_projection:
        # This repair affects only the derived rasterization candidate.
        # Raw Coast Refine science remains untouched, and every published
        # application cell is checked against original geography below.
        repaired = shapely.make_valid(value)

        from ._io import polygonal_geometry

        value = polygonal_geometry(repaired)
        if value.is_empty:
            raise ValueError("Derived projected geometry has no polygonal area")
        if not value.is_valid or value.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError("Derived projected geometry still invalid after repair")
        log.warning(
            "Repaired projected application geometry only; original scientific mask unchanged"
        )
    return (value, repaired_projection) if report_repair else value


def _exact_classification_on_grid(geom, crs, transform, height, width):
    """Rare safety path for invalid projected geometry: original source truth.

    Bound memory by row stripes. Do NOT trust a make_valid result to preserve
    small channels or source-cell decisions, even when it is topologically valid.
    """
    data = np.zeros((height, width), dtype="uint8")
    transformer = Transformer.from_crs(horizontal_crs(crs), "EPSG:4326", always_xy=True)
    shapely.prepare(geom)
    for start in range(0, height, 128):
        stop = min(height, start + 128)
        rr, cc = np.indices((stop - start, width), dtype="float64")
        rr += start
        xx = transform.c + (cc + 0.5) * transform.a + (rr + 0.5) * transform.b
        yy = transform.f + (cc + 0.5) * transform.d + (rr + 0.5) * transform.e
        lon, lat = transformer.transform(xx, yy)
        data[start:stop] = shapely.contains_xy(geom, lon, lat)
    return data


def _classify_on_grid(geom, crs, transform, height, width):
    """Fast GDAL center rasterization, with exact GEOS boundary-cell decisions.

    GDAL fills the large interiors. Re-evaluate every cell touched by the
    *actual polygon boundary* against the original WGS84 scientific geometry.
    This catches subcell islands, holes and edge-tie discrepancies while
    avoiding GEOS point-in-polygon calls for tens of millions of interior cells.
    """
    if geom.is_empty:
        return np.zeros((height, width), dtype="uint8")
    try:
        projected, repaired = _in_grid_crs(geom, crs, report_repair=True)
    except ValueError as exc:
        log.warning(
            "Projected application polygon unusable (%s); using geographic exact grid classification",
            exc,
        )
        return _exact_classification_on_grid(geom, crs, transform, height, width)
    if repaired:
        log.warning(
            "Classifying original geographic polygons at ALL application-grid centers after projection topology change"
        )
        return _exact_classification_on_grid(geom, crs, transform, height, width)
    data = rasterize(
        [(projected, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )
    edge = rasterize(
        [(projected.boundary, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    rr, cc = np.nonzero(edge)
    if rr.size:
        xx = transform.c + (cc + 0.5) * transform.a + (rr + 0.5) * transform.b
        yy = transform.f + (cc + 0.5) * transform.d + (rr + 0.5) * transform.e
        lon, lat = Transformer.from_crs(
            horizontal_crs(crs), "EPSG:4326", always_xy=True
        ).transform(xx, yy)
        shapely.prepare(geom)
        data[rr, cc] = shapely.contains_xy(geom, lon, lat).astype("uint8")
    return data


def prepare_application_vectors(
    evidence,
    raw,
    pending,
    *,
    aoi=None,
    terrain=None,
    fragments=None,
    native_bounds=None,
):
    """Build clean, source-grid-aligned polygons from the raw scientific classes.

    The native evidence/target stack sets resolution and origin. Explicit
    terrain and subcell coastal-fragment policies run before centre decisions.
    Polygonization produces complete cells; automatic preparation promotes
    these products to the interpolation classification. Independent geographic
    retention products supply native source masking.
    """
    from ._io import read_polygon

    evidence = Path(evidence)
    pending = Path(pending)
    with rasterio.open(evidence) as ds:
        if (
            ds.crs is None
            or not ds.transform.is_rectilinear
            or ds.transform.a <= 0
            or ds.transform.e >= 0
        ):
            raise ValueError(
                "Application cleanup requires a georeferenced north-up evidence grid"
            )
        if ds.width * ds.height > 150_000_000:
            raise ValueError(
                "Application grid exceeds 150 million cells: use a smaller prepared AOI"
            )
        profile = {
            "driver": "GTiff",
            "width": ds.width,
            "height": ds.height,
            "count": len(_DOMAINS),
            "dtype": "uint8",
            "crs": ds.crs,
            "transform": ds.transform,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": "deflate",
        }
    # Reuse the evidence lattice, expanding by whole cells to cover the AOI.
    # MultiStack may truncate a fractional final column; never clip a cell edge
    # to an arbitrary requested longitude in the authoritative classification.
    projected_aoi_bounds = None
    if aoi is not None:
        import math
        import geopandas as gpd
        from affine import Affine
        from shapely.geometry import box
        from .artifact import _covers_without_roundoff

        bounds = (
            gpd.GeoSeries(
                [
                    aoi.segmentize(
                        max(
                            aoi.bounds[2] - aoi.bounds[0], aoi.bounds[3] - aoi.bounds[1]
                        )
                        / 64
                    )
                ],
                crs=4326,
            )
            .to_crs(horizontal_crs(profile["crs"]))
            .total_bounds
        )
        projected_aoi_bounds = [float(value) for value in bounds]
        if native_bounds is not None:
            # The native envelope anchors the lattice; it cannot replace the
            # actual AOI envelope. Horizontal projection/chord construction
            # and the existing GEOS preparation-scope guard can put the AOI
            # just beyond that anchor. Enclose both on whole original cells,
            # rather than publishing an artifact its strict coverage check
            # must reject. No evidence pixel is moved or resampled.
            bounds = (
                min(bounds[0], native_bounds[0]),
                min(bounds[1], native_bounds[1]),
                max(bounds[2], native_bounds[2]),
                max(bounds[3], native_bounds[3]),
            )
        t = profile["transform"]

        def pixel_bound(value):
            nearest = round(value)
            return nearest if abs(value - nearest) <= 1e-7 else value

        left = math.floor(pixel_bound((bounds[0] - t.c) / t.a))
        top = math.floor(pixel_bound((bounds[3] - t.f) / t.e))
        right = math.ceil(pixel_bound((bounds[2] - t.c) / t.a))
        bottom = math.ceil(pixel_bound((bounds[1] - t.f) / t.e))
        planned = box(
            t.c + left * t.a,
            t.f + bottom * t.e,
            t.c + right * t.a,
            t.f + top * t.e,
        )
        if not _covers_without_roundoff(planned, box(*bounds)):
            # Integer snapping is only a numerical convenience. It must not
            # discard an overhang that the actual coverage validator rejects.
            left -= int(planned.bounds[0] > bounds[0])
            bottom += int(planned.bounds[1] > bounds[1])
            right += int(planned.bounds[2] < bounds[2])
            top -= int(planned.bounds[3] < bounds[3])
        # Affine 2.x uses * for transform composition; @ requires 3.x.
        width, height = right - left, bottom - top
        # The AOI can be much larger than the original raster. Limit the
        # ACTUAL publication grid, not just the initial evidence footprint.
        if width <= 0 or height <= 0 or width * height > 150_000_000:
            raise ValueError(
                "Expanded application grid is empty or exceeds 150 million cells; use a smaller prepared AOI"
            )
        profile.update(
            transform=t * Affine.translation(left, top), width=width, height=height
        )
    raw_geom = {key: read_polygon(raw[key]) for key in _DOMAINS}
    removed = shapely.Polygon()
    if fragments is not None and not fragments.is_empty:
        projected = _in_grid_crs(fragments, profile["crs"])
        cell_area = abs(profile["transform"].a * profile["transform"].e)
        removed = shapely.union_all(
            [g for g in shapely.get_parts(projected) if g.area < cell_area]
        )
        if not removed.is_empty:
            import geopandas as gpd

            removed = (
                gpd.GeoSeries([removed], crs=horizontal_crs(profile["crs"]))
                .to_crs(4326)
                .iloc[0]
            )
    excluded = shapely.union_all(
        [removed, terrain if terrain is not None else shapely.Polygon()]
    )
    for key in _DOMAINS:
        raw_geom[key] = raw_geom[key].difference(excluded)
    for geom in raw_geom.values():
        shapely.prepare(geom)
    with tempfile.TemporaryDirectory(
        prefix="coast_app_grid_", dir=pending.parent
    ) as tmp:
        tif = Path(tmp) / "classification.tif"
        with rasterio.open(tif, "w", **profile) as dst:
            for band, key in enumerate(_DOMAINS, 1):
                result = _classify_on_grid(
                    raw_geom[key],
                    profile["crs"],
                    profile["transform"],
                    dst.height,
                    dst.width,
                )
                dst.write(result, band)
                del result
        vector = {key: _polygonize_flag(tif, i) for i, key in enumerate(_DOMAINS, 1)}
        # Land and water are mutually exclusive at the application grid scale.
        # Use the same domain raster for their complementary classification.
        from shapely.geometry import box

        with rasterio.open(tif) as ds:
            extent = box(*ds.bounds)
            native_crs = ds.crs.to_string()
            resolution = [abs(ds.transform.a), abs(ds.transform.e)]
            grid = {
                "crs": native_crs,
                "width": ds.width,
                "height": ds.height,
                "transform": list(ds.transform)[:6],
                "cell_resolution": resolution,
                "policy": "cell_center_exact_no_morphology",
                "extent_owner": "native_anchor_union_projected_preparation_aoi"
                if native_bounds is not None
                else "horizontal_aoi_on_evidence_lattice",
                "geometry_crs_policy": "horizontal_only_no_vertical_operation",
                "native_anchor_bounds": None
                if native_bounds is None
                else [float(value) for value in native_bounds],
                "projected_aoi_bounds": projected_aoi_bounds,
                "evidence": str(evidence),
                "fragment_policy": "Remove newly disconnected coastal water polygons smaller than one target cell before polygonization",
                "wetland_policy": "OSM wetland habitat is retained terrain; original OSM topology is preserved separately",
            }
        # Retain each water class independently; avoid silently reconciling
        # conflicting scientific marine and inland classifications.
        vector["refined_land"] = extent.difference(vector["refined_water"])
        names = {
            key: pending / (key + ".gpkg")
            for key in ("refined_land", "refined_water", "marine_water", "inland_water")
        }
        for key, path in names.items():
            write_polygon(path, vector[key], crs=native_crs)
        # Verify the *published* vector representation, not just in-memory geometry.
        checked = dict(names)
        _assert_grid_equivalence(tif, checked)
        grid["represented_cells"] = {key: checked[key + "_cells"] for key in _DOMAINS}
        grid["land_cells"] = (
            grid["width"] * grid["height"] - checked["refined_water_cells"]
        )
        from ._io import digest

        write_polygon(pending / "removed_subcell_fragments.gpkg", removed)
        grid["removed_subcell_fragment_count"] = sum(
            1 for part in shapely.get_parts(removed) if not part.is_empty
        )
        grid["evidence_sha256"] = digest(evidence)
        json_write(pending / "application_grid.json", grid)
        return {"grid": grid, "products": names}


def declared_water_subtypes(baseline):
    """Read the same explicit ocean/inland vocabulary for either product role."""
    import geopandas as gpd
    from ._io import read_polygon

    source = gpd.read_file(baseline.paths["original_topology"]).to_crs(4326)
    inland_names = ("river", "lake", "inland_water", "wetland", "estuary")
    inland = shapely.union_all(source.loc[source["class"].isin(inland_names)].geometry)
    ocean = shapely.union_all(
        source.loc[source["class"] == "ocean"].geometry
    ).difference(read_polygon(baseline.paths["mapped_water"]))
    return ocean, inland


def publish_class_selections(pending, baseline, grid):
    """Keep declared feature identity separate from hydrologic connectivity.

    Existing scientific water/connectivity and barrier products are unchanged.
    Subtype declarations are sampled on the very same target lattice and clipped
    to final water. Generic water is never silently declared ocean or inland.
    Newly refined water beyond declared subtype geometry remains available in
    the all-water selection. Connected rivers retain their inland declaration.
    """
    from ._io import read_polygon

    pending = Path(pending)
    crs = grid["crs"]
    transform = rasterio.Affine(*grid["transform"])
    height, width = grid["height"], grid["width"]
    profile = dict(
        driver="GTiff",
        width=width,
        height=height,
        count=2,
        dtype="uint8",
        crs=crs,
        transform=transform,
    )
    # A row stripe bounds extra memory even on large application grids. All
    # rasterization uses the existing center/boundary classification machinery.
    water = read_polygon(pending / "refined_water.gpkg", crs=crs, require_native=True)
    ocean, inland = declared_water_subtypes(baseline)
    with tempfile.TemporaryDirectory(
        prefix="coast_selection_", dir=pending.parent
    ) as temporary:
        path = Path(temporary) / "selections.tif"
        with rasterio.open(path, "w", **profile) as dst:
            for start in range(0, height, 256):
                rows = min(256, height - start)
                window = rasterio.windows.Window(0, start, width, rows)
                stripe_transform = rasterio.windows.transform(window, transform)
                present = (
                    np.zeros((rows, width), dtype="uint8")
                    if water.is_empty
                    else rasterize(
                        [(water, 1)],
                        out_shape=(rows, width),
                        transform=stripe_transform,
                        fill=0,
                        dtype="uint8",
                    )
                )
                for band, geometry in enumerate((ocean, inland), 1):
                    selected = _classify_on_grid(
                        geometry, crs, stripe_transform, rows, width
                    )
                    dst.write(selected & present, band, window=window)
        for band, key in enumerate(("classified_ocean", "classified_inland_water"), 1):
            write_polygon(
                pending / (key + ".gpkg"), _polygonize_flag(path, band), crs=crs
            )
    return {
        "version": 1,
        "ocean": "Refined water within explicitly declared ocean geometry",
        "inland_water": "Refined water within declared river/lake/inland-water/estuary geometry",
        "ocean_connected": "Existing scientific marine-connectivity water, including connected rivers",
        "water": "All refined water, including water without established subtype",
        "land": "Existing refined land",
        "subtype_boundary_policy": "Declared subtype sampled at target centers; no new subtype inferred outside its supplied geometry",
    }
