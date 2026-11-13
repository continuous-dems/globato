"""Translate host topology into explicit science inputs; never acquire OSM twice."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from .geometry_crs import horizontal_frame
from .selection import horizontal_crs
import shutil
import json
import tempfile
from filelock import FileLock
import geopandas as gpd
import pyogrio
import shapely
from shapely.geometry import box
from pyproj import CRS
from ._io import digest, identity, json_write, vector_files, write_polygon
from ._science.coastline_classify import polygon_water

LAND = ("land", "island", "breakwater", "reef")
INLAND = ("river", "lake", "water", "wetland", "estuary")
CLASSES = LAND + INLAND + ("ocean", "inland_water", "ocean_connected")
LOCAL_CLASSES = ("ocean", "ocean_connected", "inland_water", "water", "land")


def host_versions():
    result = {}
    for name in ("fetchez", "globato"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def polygon_area(path):
    path = Path(path).expanduser().resolve(strict=True)
    frame = gpd.read_file(path)
    if frame.crs is None or frame.empty:
        raise ValueError("Area needs nonempty polygons and CRS")
    if (
        frame.geometry.isna().any()
        or not frame.geometry.is_valid.all()
        or not frame.geom_type.isin(["Polygon", "MultiPolygon"]).all()
    ):
        raise ValueError("Area needs valid polygon geometry")
    return shapely.union_all(horizontal_frame(frame).to_crs(4326).geometry)


def _local_geometry_in_context(source, context, polygon_class):
    """Read only intersecting local features and clip to the host's WGS84 ROI.

    The source file is never modified. The bbox is a read optimization in the
    *source* CRS; the exact intersection is computed in WGS84 afterward.
    Uniform YAML classes override attributes; osm mode preserves and validates
    each feature class attribute. No geometry buffer is applied.
    """

    info = pyogrio.read_info(source)

    if not info["crs"]:
        raise ValueError("Local polygon file must declare its CRS")

    if info["features"] == 0:
        raise ValueError("Local vector must contain at least one polygon")

    source_crs = CRS.from_user_input(info["crs"])

    # Curved projected ROI edges may reach beyond their four transformed
    # corners. Densify this read filter so valid intersecting features are
    # not omitted before the precise WGS84 intersection below.
    from pyproj import Transformer

    query = Transformer.from_crs(
        4326, horizontal_crs(source_crs), always_xy=True
    ).transform_bounds(*context.bounds, densify_pts=64)
    selected = gpd.read_file(source, bbox=tuple(query))
    if selected.crs is None:
        raise ValueError("Local polygon file must declare its CRS")
    attributes = polygon_class == "osm"
    if attributes and (
        "class" not in selected
        or selected["class"].isna().any()
        or not set(selected["class"]) <= set(CLASSES)
    ):
        raise ValueError(
            "Classified topology requires a non-null class attribute using supported OSM topology labels"
        )
    if not selected.empty and (
        selected.geometry.isna().any()
        or not selected.geometry.is_valid.all()
        or not selected.geom_type.isin(["Polygon", "MultiPolygon"]).all()
    ):
        raise ValueError("Intersecting local features must be valid polygons")
    selected = horizontal_frame(selected).to_crs(4326)
    selected.geometry = selected.geometry.map(
        lambda geometry: polygon_water(geometry.intersection(context))
    )
    selected = selected.loc[~selected.geometry.is_empty].copy()
    if not attributes:
        selected = selected[["geometry"]].copy()
        selected["class"] = polygon_class
    return selected


def _write_topology(frame, path):
    """Write classed topology, including an empty topology layer."""
    pyogrio.write_dataframe(
        frame,
        path,
        driver="GPKG",
        layer="topology",
        geometry_type="MultiPolygon",
    )


class Baseline:
    """Immutable, hash-checked baseline and explicit inputs for existing science."""

    def __init__(self, root):
        self.root = Path(root)
        self.receipt = json.loads((self.root / "baseline.json").read_text())
        self.context = shapely.from_wkb(self.receipt["context_wkb"])
        self.paths = {}
        for name, record in self.receipt["products"].items():
            path = (self.root / record["path"]).resolve(strict=True)
            if (
                not path.is_relative_to(self.root.resolve())
                or digest(path) != record["sha256"]
            ):
                raise ValueError(f"Baseline hash/path mismatch: {name}")
            self.paths[name] = path


def load_baseline(mask, context, cache, *, polygon_class=None):
    """Normalize native OSM topology or local geometry plus YAML class once.

    Context is a declared WGS84 rectangle, never inferred from polygon extent.
    Native land classes win overlapping water; inland classes win ocean. Raw
    OSM features remain unchanged in normalized original_topology. Local files
    are preserved separately byte-for-byte; their working features are clipped
    to the exact host-provided context, not buffered or extrapolated.
    Native topology can leave a coastline cutter gap. Preserve and audit that
    unclassified area rather than requiring the host to provide a partition.
    The existing science takes explicit water; its complement is non-water,
    not independent OSM land evidence. A one-domain local file declares the
    opposite binary domain in the remaining context.
    """
    if (
        context.is_empty
        or not context.is_valid
        or not context.equals(box(*context.bounds))
    ):
        raise ValueError("Baseline context must be a valid WGS84 rectangle")
    is_osm = str(mask).lower() == "osm"
    if is_osm:
        if polygon_class is not None:
            raise ValueError("OSM owns its class labels; omit polygon_class")
    elif polygon_class not in (*LOCAL_CLASSES, "osm"):
        raise ValueError(
            "Local geometry requires polygon_class: ocean, ocean_connected, inland_water, water, land, or osm (class attributes)"
        )
    source = None if is_osm else Path(mask).expanduser().resolve(strict=True)
    if is_osm:
        from globato.modules import osm_landmask

        osm_owner_sha256 = digest(Path(osm_landmask.__file__))
    else:
        osm_owner_sha256 = None
    request = {
        "mask": "osm" if is_osm else str(source),
        "polygon_class": polygon_class,
        "source_files": None if is_osm else vector_files(source),
        "context_wkb": context.normalize().wkb_hex,
        "local_clip_policy": "exact_intersection_no_buffer_v1" if not is_osm else None,
        "adapter_sha256": digest(Path(__file__)),
        "osm_owner_sha256": osm_owner_sha256,
        "host_versions": host_versions() if is_osm else {},
    }
    key = identity(request)
    parent = Path(cache).expanduser().resolve() / "coast_refine" / "baselines"
    parent.mkdir(parents=True, exist_ok=True)
    out = parent / key
    with FileLock(str(out) + ".lock"):
        if out.exists():
            baseline = Baseline(out)
            if baseline.receipt["request"] != request:
                raise ValueError("Baseline request identity mismatch")
            return baseline
        with tempfile.TemporaryDirectory(
            prefix=key + "_pending_", dir=parent
        ) as pending:
            pending = Path(pending)
            if is_osm:
                from fetchez.spatial import Region
                from fetchez.registry import ModuleRegistry
                from globato.utils import resolve_barrier

                # Native acquisition cache is independent of adapter releases.
                # Keep the native module's own on-disk products; no new OSM
                # downloader, endpoint retry or fallback is introduced here.
                native_key = identity(
                    {"context": request["context_wkb"], "osm_owner": osm_owner_sha256}
                )
                native_cache = parent.parent / "native_osm" / native_key
                native_cache.parent.mkdir(parents=True, exist_ok=True)
                source_origin = "native_resolver"
                with FileLock(str(native_cache) + ".lock"):
                    # Older releases already sealed the original fetched vector.
                    # Reuse a compatible mixed topology instead of querying again
                    # merely because integration/adapter code changed.
                    for saved in sorted(
                        parent.glob("*/baseline.json"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    ):
                        meta = json.loads(saved.read_text())
                        prior = meta["request"]
                        same_owner = prior.get("osm_owner_sha256") == osm_owner_sha256
                        if (
                            prior.get("mask") == "osm"
                            and prior.get("context_wkb") == request["context_wkb"]
                            and same_owner
                            and meta.get("uniform_class") is None
                        ):
                            cached = Baseline(saved.parent)
                            original = cached.root / (
                                "source_" + cached.receipt["source_files"][0]["name"]
                            )
                            name = original.name
                            while name.startswith("source_"):
                                name = name[len("source_") :]
                            (pending / "host").mkdir()
                            source = pending / "host" / name
                            shutil.copy2(original, source)
                            source_origin = "verified_previous_baseline"
                            break
                    if source is None:
                        ModuleRegistry.load_all()
                        w, s, e, n = context.bounds
                        source = resolve_barrier(
                            "osm",
                            region=Region.from_list([w, e, s, n]),
                            outdir=str(native_cache),
                            output_mode="topology",
                            output_type="vector",
                            target_crs="EPSG:4326",
                        )
                    if not source:
                        raise ValueError("Globato did not return a topology vector")
                    source = Path(source).resolve(strict=True)
            raw_hashes = vector_files(source)
            frame = _local_geometry_in_context(
                source, context, "osm" if is_osm else polygon_class
            )
            for item in raw_hashes:
                shutil.copy2(
                    source.parent / item["name"], pending / ("source_" + item["name"])
                )
            if vector_files(source) != raw_hashes:
                raise ValueError(
                    "Original topology changed while the baseline was being read"
                )
            # The original local file is sealed under source_* unchanged;
            # Normalized topology is the exact WGS84 context intersection.
            # Uniform YAML classes or explicit osm attribute mode own labels.
            _write_topology(frame, pending / "original_topology.gpkg")
            geo = horizontal_frame(frame).to_crs(4326)
            empty = shapely.Polygon()
            unclassified = empty
            if (
                "class" not in geo
                or geo["class"].isna().any()
                or not set(geo["class"]) <= set(CLASSES)
            ):
                raise ValueError("OSM topology needs supported land/water classes")
            input_domain = (
                None
                if is_osm or polygon_class == "osm"
                else ("land" if polygon_class == "land" else "water")
            )
            classes = {
                name: polygon_water(
                    shapely.union_all(
                        geo.loc[geo["class"] == name].geometry
                    ).intersection(context)
                )
                for name in CLASSES
            }
            coverage = shapely.union_all(list(classes.values()))
            unclassified = polygon_water(context.difference(coverage))
            if input_domain is not None:
                # File meaning defines a complete binary mask inside known context.
                # OSM retains native gaps instead; no missing subtype is invented.
                opposite = "land" if input_domain == "water" else "water"
                classes[opposite] = polygon_water(classes[opposite].union(unclassified))
                unclassified = empty
            land = shapely.union_all([classes[name] for name in LAND])
            water = shapely.union_all(
                [
                    classes[name]
                    for name in INLAND + ("inland_water", "ocean", "ocean_connected")
                ]
            ).difference(land)
            # Both sources use the same established class protections.
            inland = shapely.union_all(
                [classes[name] for name in INLAND + ("inland_water",)]
            ).difference(land)
            ocean = (
                shapely.union_all([classes["ocean"], classes["ocean_connected"]])
                .difference(land)
                .difference(inland)
            )
            for name, geom in {
                "water_mask": water,
                "mapped_water": inland,
                "ocean_anchor": ocean,
                "structures": empty,
                "context": context,
                "unclassified": unclassified,
            }.items():
                write_polygon(pending / (name + ".gpkg"), geom)
            uniform = (
                "land"
                if water.is_empty and land.buffer(1e-12).covers(context)
                else ("ocean" if ocean.covers(context) else None)
            )
            receipt = {
                "request": request,
                "source_files": raw_hashes,
                "context_wkb": context.normalize().wkb_hex,
                "local_clip_policy": "exact_intersection_no_buffer_v1"
                if not is_osm
                else None,
                "uniform_class": uniform,
                "input_domain": input_domain,
                "polygon_class": polygon_class,
                "source_origin": source_origin if is_osm else "supplied_local_vector",
                "native_source_cache": str(native_cache) if is_osm else None,
                "unclassified_area_m2": float(
                    gpd.GeoSeries([unclassified], crs=4326).to_crs(6933).area.iloc[0]
                ),
                "coverage_policy": "Native gaps preserved as unclassified; explicit water is unchanged; science uses its complement as non-water, not positive land evidence",
                "class_counts": {
                    name: int((geo["class"] == name).sum()) for name in CLASSES
                }
                if "class" in geo
                else {},
                "class_areas_m2": {
                    name: float(
                        gpd.GeoSeries([geom], crs=4326).to_crs(6933).area.iloc[0]
                    )
                    for name, geom in classes.items()
                },
                "policy": (
                    "YAML polygon_class; geometry-only local file; opposite binary complement in declared context; shared science"
                    if not is_osm and polygon_class != "osm"
                    else "Globato OSM as-is; land overrides water overlaps; inland overrides ocean; no new structure rules"
                ),
                "structures_policy": "Native topology has no deck class; explicit empty input prevents a second OSM acquisition.",
                "products": {
                    p.stem: {"path": p.name, "sha256": digest(p)}
                    for p in sorted(pending.iterdir())
                    if p.is_file()
                },
            }
            json_write(pending / "baseline.json", receipt)
            if (pending / "host").exists():
                shutil.rmtree(pending / "host")
            Baseline(pending)
            pending.rename(out)
    return Baseline(out)
