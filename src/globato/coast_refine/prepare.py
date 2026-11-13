"""Generic artifact preparation around the existing science; no provider discovery."""

from pathlib import Path
import json
import os
import shutil
import subprocess
import rasterio
import sys
import tempfile
import numpy as np
import pyogrio
import pyogrio.raw
import geopandas as gpd
import shapely
from shapely.geometry import MultiLineString

from ._io import (
    digest,
    identity,
    json_write,
    read_polygon,
    write_polygon,
    raster_files,
    vector_files,
)
from .artifact import MaskArtifact


def _baseline_context(mask, aoi, context_region=None):
    """Use host-expanded ROI directly for local input; preserve OSM margin.

    The processing ROI has already been expanded by Globato's -X modifier.
    Local clipping must not buffer it a second time. The exterior OSM context
    is an existing acquisition/science requirement, not a local-vector rule.
    """
    from shapely.geometry import box

    if context_region:
        w, e, s, n = map(float, context_region.split("/"))
        if not (w < e and s < n):
            raise ValueError("Context must be W/E/S/N with positive dimensions")
        context = box(w, s, e, n)
    else:
        w, s, e, n = aoi.bounds
        dx, dy = (
            ((e - w) * 0.05, (n - s) * 0.05)
            if str(mask).lower() == "osm"
            else (0.0, 0.0)
        )
        context = box(w - dx, s - dy, e + dx, n + dy)
        w, s, e, n = context.bounds
        context_region = f"{w}/{e}/{s}/{n}"
    if not context.covers(aoi):
        raise ValueError("Baseline context does not cover AOI")
    return context, context_region


def linear_parts(geom):
    """Yield nonzero-length lines from nested Shapely geometries.

    Ignore point-only intersections and empty geometries.
    """
    if geom is None or geom.is_empty:
        return

    if geom.geom_type == "LineString":
        if geom.length > 0:
            yield geom

    elif geom.geom_type in ("MultiLineString", "GeometryCollection"):
        for child in geom.geoms:
            yield from linear_parts(child)


def _write_lines(path, geom, role="original_coastline", crs="EPSG:4326"):
    """Write all coastline segments, including a valid empty layer."""
    lines = list(linear_parts(geom))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)

    geometry = (
        np.array(
            [shapely.to_wkb(MultiLineString(lines))],
            dtype=object,
        )
        if lines
        else np.empty(0, dtype=object)
    )
    roles = np.array([role] if lines else [], dtype=object)

    pyogrio.raw.write(
        path,
        geometry=geometry,
        field_data=[roles],
        fields=["role"],
        layer="mask",
        driver="GPKG",
        geometry_type="MultiLineString",
        crs=crs,
    )


def _publish_target_cell_coastline(pending, lattice):
    """Publish the coastline on the same native cell lattice as land/water.

    The original science coastline remains geographic under science_vectors/;
    this application product must not round-trip target-cell edges through
    EPSG:4326, even when the evidence raster is NAD83 or projected.
    """
    crs = lattice["crs"]
    water = read_polygon(
        Path(pending) / "refined_water.gpkg", crs=crs, require_native=True
    )
    land = read_polygon(
        Path(pending) / "refined_land.gpkg", crs=crs, require_native=True
    )
    coast = (
        shapely.Polygon()
        if water.is_empty or land.is_empty
        else water.boundary.intersection(land.boundary)
    )
    target = Path(pending) / "refined_coastline.gpkg"
    _write_lines(target, coast, role="target_cell_coastline", crs=crs)
    # Seal the exact native coastline, not merely its CRS metadata. An
    # incomplete/empty/rounded-off line may satisfy a coarse extent test.
    saved = gpd.read_file(target, layer="mask")
    if saved.crs is None or saved.crs != gpd.GeoSeries([], crs=crs).crs:
        raise ValueError(
            "Refined coastline publication changed native application grid CRS"
        )
    actual = shapely.union_all(saved.geometry)
    if not actual.equals(coast):
        raise ValueError(
            "Refined coastline GeoPackage changed native land/water boundary"
        )


def _prepare_science(
    evidence,
    area,
    output_dir,
    cache,
    *,
    evidence_files,
    water_mask,
    context_region,
    ocean_anchor=None,
    structures=None,
    mapped_water=None,
    anchor_scope="external",
):
    """Internal intermediate: run science once; publish no application grid.

    Only :func:`prepare_from_baseline` is the production preparation entrypoint.
    The final AOI/wetland/fragment policy belongs to that owner exclusively.
    """
    evidence = Path(evidence).expanduser().resolve(strict=True)
    area = Path(area).expanduser().resolve(strict=True)
    out = Path(output_dir).expanduser().resolve()
    frame = gpd.read_file(area)
    if frame.crs is None or frame.empty:
        raise ValueError("AOI is empty or missing CRS")
    geom = shapely.union_all(frame.to_crs(4326).geometry)
    if (
        geom.is_empty
        or not geom.is_valid
        or geom.geom_type not in ("Polygon", "MultiPolygon")
    ):
        raise ValueError("AOI must be valid polygon geometry")
    from . import _science

    science_root = Path(_science.__file__).parent
    # The public owner already froze the TIFF and validity sidecars before
    # resolving the baseline. Reuse those exact hashes rather than hashing a
    # potentially multi-GB evidence TIFF again before the science worker.
    primary = [
        item["sha256"] for item in evidence_files if item["name"] == evidence.name
    ]
    if len(primary) != 1:
        raise ValueError(
            "Evidence TIFF is missing or ambiguous in its native raster file inventory"
        )
    request = {
        "preparation_code_sha256": digest(Path(__file__)),
        "application_code_sha256": digest(Path(__file__).with_name("application.py")),
        "evidence_sha256": primary[0],
        "aoi_wkb": geom.normalize().wkb_hex,
        "science": {
            str(p.relative_to(science_root)): digest(p)
            for p in sorted(science_root.rglob("*.py"))
        },
        "context_region": context_region,
        "anchor_scope": anchor_scope,
        "evidence_files": evidence_files,
        "inputs": {},
    }
    if anchor_scope not in ("external", "declared"):
        raise ValueError("Internal ocean-anchor scope must be external or declared")
    options = {
        "water_mask": water_mask,
        "ocean_anchor": ocean_anchor,
        "structures": structures,
        "mapped_water": mapped_water,
    }
    for name, path in options.items():
        if path:
            request["inputs"][name] = vector_files(
                Path(path).expanduser().resolve(strict=True)
            )
    if not water_mask or not context_region or not structures:
        raise ValueError(
            "Science requires resolved water, context and explicit structure polygons"
        )
    env = os.environ.copy()
    request_id = identity(request)
    # Only the public owner may reuse a sealed artifact. This helper receives a
    # fresh private staging path and must never publish an independently sealed
    # "raw science" artifact that the caller immediately replaces.
    if out.exists():
        raise ValueError(f"Science staging directory already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=out.name + "_preparation_", dir=out.parent))
    science = work / "science"
    from .geometry_crs import science_horizontal_view, operation_record

    science_evidence = science_horizontal_view(
        evidence, work / "science_evidence_2d.tif"
    )
    with rasterio.open(evidence) as native:
        centre = (
            (native.bounds.left + native.bounds.right) / 2,
            (native.bounds.bottom + native.bounds.top) / 2,
        )
        geometry_operation = operation_record(native.crs, 4326, centre)
    # Keep the scientific subprocess for bounded worker lifetime, but pass
    # the actual source/AOI directly; no fabricated tile input catalogue.
    cmd = [
        sys.executable,
        "-m",
        "globato.coast_refine._science.run_one_raster",
        "--area",
        str(area),
        "--evidence",
        str(science_evidence),
        "--output",
        str(science),
        "--cache",
        str(Path(cache).expanduser().resolve()),
    ]
    flags = {
        "water_mask": "--water-mask",
        "ocean_anchor": "--ocean-anchor",
        "structures": "--overwater-structures",
        "mapped_water": "--mapped-water-areas",
    }
    for name, value in options.items():
        if value:
            cmd.extend([flags[name], str(Path(value).expanduser().resolve())])
    if water_mask:
        cmd.extend(["--mask-layer", "mask"])
    if context_region:
        cmd.append("--context-region=" + context_region)
    cmd.append("--anchor-scope=" + anchor_scope)
    science_log = work / "science.log"
    try:
        with science_log.open("w") as log:
            subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=True
            )
    except subprocess.CalledProcessError as exc:
        # Worker stdout is redirected to a scratch file, otherwise the recipe
        # reports only an opaque exit code. Keep the file for full diagnosis
        # and surface a bounded, immediately useful traceback to the caller.
        with science_log.open("rb") as log:
            log.seek(max(0, science_log.stat().st_size - 16384))
            tail = log.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Coast Refine scientific worker failed (exit {exc.returncode}). "
            f"Full log: {science_log}\nLast worker output:\n{tail}"
        ) from exc
    state = json.loads((science / "status.json").read_text())
    if not str(state.get("status", "")).startswith("COMPLETE"):
        raise ValueError("Science did not complete successfully")
    from ._science.input_contracts import raster_signature, signature_differences

    raw = science / "tiles/area/raw_elevation.tif"
    if signature_differences(raster_signature(raw), raster_signature(science_evidence)):
        raise ValueError(
            "Scientific workspace RAW changed from original evidence; artifact not published"
        )
    if state.get("no_valid_terrain_tiles") or state.get("no_source_coverage_tiles"):
        raise ValueError(
            "No terrain-backed refinement available; baseline alone is not an approved shared artifact"
        )
    # One post-worker inventory validates both the TIFF and all external
    # validity files; the original digest check was a duplicate full-file read.
    if raster_files(evidence) != request["evidence_files"]:
        raise ValueError(
            "Evidence or external validity files changed during preparation"
        )
    masks = science / "final/masks"
    pending = out
    pending.mkdir()
    names = {
        "refined_water": "refined_water",
        "refined_land": "refined_land",
        "refined_coastline": "refined_coastline",
        "marine_water": "marine_connectivity_water",
        "inland_water": "inland_water",
        "unresolved_water": "unresolved_water",
        "structures": "overwater_structure_exclusion",
        "coverage": "known_context",
    }
    for name, original in names.items():
        if name == "coverage":
            write_polygon(
                pending / (name + ".gpkg"),
                read_polygon(masks / (original + ".gpkg"), layer="context"),
            )
        else:
            shutil.copy2(masks / (original + ".gpkg"), pending / (name + ".gpkg"))
    shutil.copy2(
        science / "newly_disconnected_marine_review.gpkg",
        pending / "newly_disconnected_water.gpkg",
    )
    shutil.copy2(
        science / "masks/baseline_water_mask.gpkg", pending / "original_water.gpkg"
    )
    write_polygon(pending / "aoi.gpkg", geom)
    coverage = read_polygon(pending / "coverage.gpkg")
    baseline = read_polygon(pending / "original_water.gpkg")
    write_polygon(pending / "original_land.gpkg", coverage.difference(baseline))
    _write_lines(
        pending / "original_coastline.gpkg", baseline.boundary.intersection(geom)
    )
    preparation = {
        "request_identity": request_id,
        "baseline_sha256": digest(pending / "original_water.gpkg"),
        "coverage_sha256": digest(pending / "coverage.gpkg"),
        "science_status": state["status"],
    }
    record = {
        "schema_version": 1,
        "status": "VALIDATED",
        "request_identity": request_id,
        "request_inputs": request,
        "preparation_inputs": preparation,
        "preparation_identity": identity(preparation),
        "products": {},
        "science_status": state,
        "science_policy": "Signed near-zero class additions with shared 3x3 sign coherence and original-class connectivity; class vetoes precede propagation; positive-water artifact screens retained",
    }
    # Keep the native science receipt with the permanent artifact, rather
    # than requiring its multi-GB temporary preparation workspace for review.
    shutil.copy2(work / "science.log", pending / "science.log")
    json_write(
        pending / "geometry_operation.json",
        dict(
            schema="coast_mask.geometry_operation.v1",
            horizontal_to_geographic=geometry_operation,
            original_evidence_sha256=digest(evidence),
            science_view_policy="same_pixels_same_mask_horizontal_crs_only_no_vertical_conversion",
        ),
    )
    record["preparation_inputs"]["geometry_operation_sha256"] = digest(
        pending / "geometry_operation.json"
    )
    record["preparation_identity"] = identity(record["preparation_inputs"])

    # The caller adds the single authoritative target-cell vectors, seals the
    # final manifest and validates the *complete* artifact before publication.
    # Raw geographic vectors are still preserved in science_vectors/ there.
    # On failure above, leave the scratch workspace intact for diagnostics.
    # Once publication has succeeded it is no longer needed; otherwise the
    # nine-tile workflow leaves redundant evidence rasters on disk per tile.
    try:
        shutil.rmtree(work)
    except OSError as exc:
        import logging

        logging.getLogger(__name__).warning(
            "Unable to clean preparation scratch %s: %s", work, exc
        )
    return record


def _publish_unresolved_native(pending, lattice):
    """Publish the water subtype complement on the sealed *native* cell lattice.

    Refined water, marine and inland are polygonized from the same target grid
    by application.prepare_application_vectors. Never reproject them through
    EPSG:4326 to do boolean operations: those round trips create thin slivers
    for users of the vector products on different grids.

    The original geographic scientific unresolved layer remains untouched in
    science_vectors/. This is an output-publication operation, not new science.
    """
    crs = lattice["crs"]
    classes = {
        name: read_polygon(
            Path(pending) / (name + ".gpkg"), crs=crs, require_native=True
        )
        for name in ("refined_water", "marine_water", "inland_water")
    }
    water, marine, inland = (
        classes[name] for name in ("refined_water", "marine_water", "inland_water")
    )
    if marine.intersects(inland) and marine.intersection(inland).area:
        raise ValueError(
            "Marine and inland water overlap outside authoritative grid cells"
        )
    subclasses = shapely.union_all([marine, inland])
    if subclasses.difference(water).area:
        raise ValueError("Native marine/inland geometry exceeds refined water")
    unresolved = water.difference(subclasses)
    if not unresolved.is_valid:
        raise ValueError("Native unresolved water complement is invalid")
    target = Path(pending) / "unresolved_water.gpkg"
    write_polygon(target, unresolved, crs=crs)
    published = gpd.read_file(target, layer="mask")
    if published.crs is None or published.crs != gpd.GeoSeries([], crs=crs).crs:
        raise ValueError("Unresolved-water publication lost native CRS")
    roundtrip = shapely.union_all(published.geometry)
    if not roundtrip.equals(unresolved):
        raise ValueError(
            "Unresolved-water GeoPackage changed native complement geometry"
        )
    return unresolved


def prepare_from_baseline(
    evidence,
    area,
    output_dir,
    cache,
    *,
    mask="osm",
    context_region=None,
    provenance=None,
    polygon_class=None,
    native_bounds=None,
):
    """Normalize OSM or geometry-only local input, then run the same science."""
    from filelock import FileLock
    from shapely.geometry import box
    from .baseline import load_baseline, polygon_area
    from .topology import publish_topology
    import rasterio
    from .artifact import check_encoding

    with rasterio.open(evidence) as source:
        check_encoding(source)
    geom = polygon_area(area)
    context, context_region = _baseline_context(
        "osm" if polygon_class == "osm" else mask, geom, context_region
    )
    baseline = load_baseline(mask, context, cache, polygon_class=polygon_class)
    request = {
        "native_application_bounds": native_bounds,
        "baseline_sha256": digest(baseline.root / "baseline.json"),
        "evidence_files": raster_files(evidence),
        "aoi_wkb": geom.normalize().wkb_hex,
        "code": {
            str(p.relative_to(Path(__file__).parent)): digest(p)
            for p in sorted(Path(__file__).parent.rglob("*.py"))
        },
        "provenance": provenance or {},
    }
    request_id = identity(request)
    out = Path(output_dir).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out) + ".lock"):
        if out.exists():
            artifact = MaskArtifact(out)
            if artifact.record.get("baseline_request_identity") != request_id:
                raise ValueError(
                    "Existing preparation has different baseline/evidence/provenance"
                )
            return artifact
        work = Path(tempfile.mkdtemp(prefix=out.name + "_baseline_", dir=out.parent))
        pending = work / "artifact"
        if baseline.receipt["uniform_class"]:
            pending.mkdir()
            water = read_polygon(baseline.paths["water_mask"])
            marine = read_polygon(baseline.paths["ocean_anchor"])
            empty = shapely.Polygon()
            for name, shape in {
                "coverage": context,
                "aoi": geom,
                "original_water": water,
                "original_land": context.difference(water),
                "refined_water": water,
                "refined_land": context.difference(water),
                "marine_water": marine,
                "inland_water": water.difference(marine),
                "unresolved_water": empty,
                "structures": empty,
            }.items():
                write_polygon(pending / (name + ".gpkg"), shape)
            for name in ("original_coastline", "refined_coastline"):
                _write_lines(pending / (name + ".gpkg"), empty)
            record = {
                "schema_version": 1,
                "status": "VALIDATED",
                "preparation_inputs": dict(request),
                "preparation_identity": identity(request),
                "products": {},
                "science_status": {"status": "NOT_RUN_UNIFORM_BASELINE"},
            }
        else:
            record = _prepare_science(
                evidence,
                area,
                pending,
                cache,
                evidence_files=request["evidence_files"],
                context_region=context_region,
                **{
                    name: baseline.paths[name]
                    for name in (
                        "water_mask",
                        "ocean_anchor",
                        "structures",
                        "mapped_water",
                    )
                },
                # Both adapters provide an explicit ocean declaration.
                # Use the same anchor policy for a finite polygon
                # wholly inside the AOI and for exterior ocean.
                anchor_scope="declared",
            )
        if (
            baseline.receipt["uniform_class"]
            and raster_files(evidence) != request["evidence_files"]
        ):
            # Nonuniform preparations already verify the frozen evidence in
            # _prepare_science. The uniform shortcut skips that worker, so it
            # needs its own post-baseline acquisition integrity boundary.
            raise ValueError(
                "Evidence or external validity files changed during uniform preparation"
            )
        shutil.copytree(baseline.root, pending / "baseline")
        shutil.copy2(
            baseline.paths["original_topology"], pending / "original_topology.gpkg"
        )
        # Keep the original science. Source-use polygons and the interpolation
        # classification have independent roles and coordinate contracts.
        original = pending / "science_vectors"
        original.mkdir()
        if not (pending / "newly_disconnected_water.gpkg").exists():
            write_polygon(pending / "newly_disconnected_water.gpkg", shapely.Polygon())
        for key in (
            "refined_water",
            "refined_land",
            "marine_water",
            "inland_water",
            "unresolved_water",
            "refined_coastline",
        ):
            shutil.copy2(pending / (key + ".gpkg"), original / (key + ".gpkg"))
        native = gpd.read_file(baseline.paths["original_topology"]).to_crs(4326)
        marsh = (
            shapely.union_all(native.loc[native["class"] == "wetland"].geometry)
            if baseline.receipt["input_domain"] is None and "class" in native
            else shapely.Polygon()
        )
        from .retention import publish_retention

        record["preparation_inputs"]["product_roles"] = publish_retention(
            pending, baseline, terrain=marsh
        )
        operation_path = pending / "geometry_operation.json"
        if not operation_path.exists():
            from .geometry_crs import operation_record

            with rasterio.open(evidence) as src:
                op = operation_record(
                    src.crs,
                    4326,
                    (
                        (src.bounds.left + src.bounds.right) / 2,
                        (src.bounds.bottom + src.bounds.top) / 2,
                    ),
                )
            json_write(
                operation_path,
                dict(
                    schema="coast_mask.geometry_operation.v1",
                    horizontal_to_geographic=op,
                    science_view_policy="uniform_baseline_no_science_worker",
                ),
            )
        record["preparation_inputs"]["geometry_operation_sha256"] = digest(
            operation_path
        )
        from .application import prepare_application_vectors

        grid = prepare_application_vectors(
            evidence,
            {
                key: pending / (key + ".gpkg")
                for key in ("refined_water", "marine_water", "inland_water")
            },
            pending,
            aoi=geom,
            native_bounds=native_bounds,
            terrain=marsh,
            fragments=read_polygon(pending / "newly_disconnected_water.gpkg"),
        )
        # Derived water subtypes must share the exact native lattice edges.
        # A geographic complement makes boundary slivers on other grids.
        _publish_unresolved_native(pending, grid["grid"])
        lattice = grid["grid"]
        from .application import publish_class_selections

        record["preparation_inputs"]["selection_contract"] = publish_class_selections(
            pending, baseline, lattice
        )
        _publish_target_cell_coastline(pending, lattice)
        # The rectangular grid extent is the class publication domain; the
        # requested AOI remains a separate source clipping constraint.
        water = read_polygon(pending / "refined_water.gpkg")
        land = read_polygon(pending / "refined_land.gpkg")
        extent = shapely.union_all([water, land])
        native_extent = box(
            *rasterio.transform.array_bounds(
                lattice["height"],
                lattice["width"],
                rasterio.Affine(*lattice["transform"]),
            )
        )
        write_polygon(pending / "coverage.gpkg", native_extent, crs=lattice["crs"])
        record["application_grid"] = grid["grid"]
        record["preparation_inputs"]["application_grid_sha256"] = digest(
            pending / "application_grid.json"
        )
        record["application_policy"] = (
            "Source retention at original native centres; independent target-cell interpolation classification"
        )
        record["classification_policy"] = (
            "OSM wetland terrain retained; newly disconnected coastal water smaller than one cell removed; original OSM and raw science retained"
        )
        record["unresolved_target_cells"] = (
            grid["grid"]["represented_cells"]["refined_water"]
            - grid["grid"]["represented_cells"]["marine_water"]
            - grid["grid"]["represented_cells"]["inland_water"]
        )
        record["classification_status"] = (
            "COMPLETE_WITH_UNRESOLVED_WATER"
            if record["unresolved_target_cells"]
            else "COMPLETE"
        )
        record["preparation_inputs"]["science_vectors"] = {
            p.name: digest(p) for p in original.glob("*.gpkg")
        }
        record["target_cell_authoritative"] = True
        record["preparation_inputs"]["target_cell_authoritative"] = True
        publish_topology(pending, baseline, extent)
        if baseline.receipt.get("input_domain") is not None:
            record["polygon_class"] = baseline.receipt["polygon_class"]
            record["preparation_inputs"]["polygon_class"] = record["polygon_class"]
            record["input_domain"] = baseline.receipt["input_domain"]
            record["preparation_inputs"]["input_domain"] = record["input_domain"]
        record["outcome"] = (
            "BASELINE_ONLY" if baseline.receipt["uniform_class"] else "REFINED"
        )
        record["baseline_request_identity"] = request_id
        record["baseline_request_inputs"] = request
        record["preparation_inputs"]["baseline_request_identity"] = request_id
        record["preparation_inputs"]["baseline_receipt_sha256"] = digest(
            pending / "baseline/baseline.json"
        )
        record["preparation_inputs"]["change_audit_sha256"] = digest(
            pending / "change_audit.json"
        )
        record["preparation_identity"] = identity(record["preparation_inputs"])
        record["osm_policy"] = baseline.receipt["policy"]
        # Seal every GeoPackage once, after the target-cell coastline and
        # topology have replaced the intermediate geographic classes. Refuse
        # ambiguous multi-layer products rather than selecting a guessed layer.
        products = {}
        for path in sorted(pending.glob("*.gpkg")):
            layers = pyogrio.list_layers(path)[:, 0].tolist()
            if len(layers) != 1:
                raise ValueError(
                    f"Expected exactly one vector layer in {path}: {layers}"
                )
            products[path.stem] = {
                "path": path.name,
                "layer": layers[0],
                "sha256": digest(path),
            }
        record["products"] = products
        json_write(pending / "manifest.json", record)
        artifact = MaskArtifact(pending)
        artifact.check_source(evidence, geom)
        pending.rename(out)
        # Publication has already passed MaskArtifact validation. Scratch
        # cleanup can encounter files removed during directory traversal on
        # Python 3.12/shared filesystems; it must not invalidate that product.
        try:
            shutil.rmtree(work)
        except OSError as exc:
            import logging

            logging.getLogger(__name__).warning(
                "Unable to clean preparation scratch %s: %s", work, exc
            )
        return MaskArtifact(out)
