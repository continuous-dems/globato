"""Seal independent native-source retention and target-cell interpolation roles."""

from pathlib import Path
import json
import geopandas as gpd
import pyogrio
import rasterio
from shapely.geometry import box
from ._io import digest, identity, read_polygon, polygonal_geometry
from ._science.input_contracts import check_encoding
from .selection import horizontal_crs
from .geometry_crs import horizontal_frame

MODES = ("ocean", "ocean_connected", "inland_water", "water", "land")

# GEOS polygon intersection may generate vertices a few ULPs outside an
# intersected edge, so ``outer.covers(outer.intersection(source))`` is not
# reliable at exact coincident CUDEM tile borders.  This guard covers only
# sub-micrometre numerical noise in lon/lat degrees; missing mask coverage
# larger than this remains an error.  Never buffer the published mask.
_COVERAGE_NUMERICAL_TOLERANCE_DEG = 1e-12


def _covers_without_roundoff(outer, inner):
    if outer.covers(inner):
        return True
    return outer.buffer(_COVERAGE_NUMERICAL_TOLERANCE_DEG).covers(inner)


class MaskArtifact:
    def __init__(self, manifest, expected_identity=None):
        path = Path(manifest).expanduser().resolve(strict=True)
        self.path = path / "manifest.json" if path.is_dir() else path
        self.root = self.path.parent
        self.record = json.loads(self.path.read_text())
        # Only canonical, cell-aligned coastline artifacts are supported.
        # Historic dev19 'selection' and dev13-15 alias products must be rebuilt.
        if self.record.get("artifact_kind", "coastline") != "coastline":
            raise ValueError("Obsolete artifact kind; regenerate with the current hook")
        self.kind = "coastline"
        if (
            self.record.get("schema_version") != 1
            or self.record.get("status") != "VALIDATED"
        ):
            raise ValueError("Shared mask is not a validated supported classification")
        self.input_domain = self.record.get("input_domain")
        if self.input_domain != self.record["preparation_inputs"].get(
            "input_domain"
        ) or self.input_domain not in (None, "land", "water"):
            raise ValueError("Local input meaning differs from sealed preparation")
        self.preparation_id = self.record["preparation_identity"]
        if identity(self.record["preparation_inputs"]) != self.preparation_id:
            raise ValueError("Mask preparation identity is inconsistent")
        if expected_identity and expected_identity != self.preparation_id:
            raise ValueError("Shared mask does not match expected preparation identity")
        self.products = {}
        self.layers = {}
        for key, product in self.record["products"].items():
            target = (self.root / product["path"]).resolve(strict=True)
            if (
                not target.is_relative_to(self.root)
                or digest(target) != product["sha256"]
            ):
                raise ValueError(f"Mask product hash/path mismatch: {key}")
            # Current preparations and the active independent adapter both
            # seal explicit layer names; do not guess for obsolete artifacts.
            layer = product.get("layer")
            if (
                not isinstance(layer, str)
                or layer not in pyogrio.list_layers(target)[:, 0].tolist()
            ):
                raise ValueError(
                    f"Mask product has missing/wrong GeoPackage layer: {key}"
                )
            self.products[key] = target
            self.layers[key] = layer
        needed = {
            "coverage",
            "aoi",
            "original_water",
            "original_land",
            "original_coastline",
            "refined_water",
            "refined_land",
            "refined_coastline",
            "marine_water",
            "inland_water",
            "unresolved_water",
            "structures",
            "refined_topology",
        }
        from .retention import CONTRACT, KEYS, validate_source_water

        if (
            self.record["preparation_inputs"].get("product_roles", {}).get("contract")
            != CONTRACT
        ):
            raise ValueError(
                "Artifact couples source masking to the interpolation barrier; regenerate with the current hook"
            )
        needed.update("retention_" + key for key in KEYS)
        if not needed <= self.products.keys():
            raise ValueError("Incomplete shared mask classification")
        self.coverage = read_polygon(
            self.products["coverage"], layer=self.layers["coverage"]
        )
        self.aoi = read_polygon(self.products["aoi"], layer=self.layers["aoi"])
        class_keys = (
            (
                "refined_water",
                "refined_land",
                "marine_water",
                "inland_water",
                "unresolved_water",
                "structures",
            )
            + tuple(
                k
                for k in ("classified_ocean", "classified_inland_water")
                if k in self.products
            )
            + tuple("retention_" + key for key in KEYS)
        )
        self._classes = {
            key: read_polygon(
                self.products[key],
                layer=self.layers[key],
                require_native=key.startswith("retention_"),
            )
            for key in class_keys
        }
        original_water = read_polygon(
            self.products["original_water"], layer=self.layers["original_water"]
        )
        validate_source_water(self._classes["retention_water"], original_water)
        if not self.record.get("target_cell_authoritative"):
            raise ValueError(
                "Obsolete non-cell-aligned artifact; regenerate with the current hook"
            )
        if any(name.endswith("_clean_1_9as") for name in self.products):
            raise ValueError(
                "Obsolete duplicate application-grid aliases; regenerate current artifact"
            )
        grid_path = self.root / "application_grid.json"
        if not grid_path.is_file() or digest(grid_path) != self.record[
            "preparation_inputs"
        ].get("application_grid_sha256"):
            raise ValueError("Application-grid identity mismatch")
        grid = json.loads(grid_path.read_text())
        self._native_coverage = box(
            *rasterio.transform.array_bounds(
                grid["height"], grid["width"], rasterio.Affine(*grid["transform"])
            )
        )
        self._coverage_crs = grid["crs"]
        if (
            self.coverage.is_empty
            or self.aoi.is_empty
            or not self._coverage_covers(self.aoi)
        ):
            raise ValueError("Validated classification does not cover its declared AOI")
        if "baseline_receipt_sha256" in self.record["preparation_inputs"]:
            from .baseline import Baseline

            if (
                digest(self.root / "baseline/baseline.json")
                != self.record["preparation_inputs"]["baseline_receipt_sha256"]
            ):
                raise ValueError("Baseline receipt hash mismatch")
            Baseline(self.root / "baseline")
            if (
                digest(self.root / "change_audit.json")
                != self.record["preparation_inputs"]["change_audit_sha256"]
            ):
                raise ValueError("Change audit hash mismatch")
        self._validate_native_vector_crs(grid)
        from .topology import verify_partition

        verify_partition(
            gpd.read_file(
                self.products["refined_topology"], layer=self.layers["refined_topology"]
            ),
            self._native_coverage,
            crs=grid["crs"],
        )
        for name in ("frozen_sources", "unmasked_evidence"):
            key = name + "_sha256"
            if key in self.record["preparation_inputs"]:
                suffix = ".json" if name == "frozen_sources" else ".tif"
                if (
                    digest(self.root / (name + suffix))
                    != self.record["preparation_inputs"][key]
                ):
                    raise ValueError(f"{name} hash mismatch")
        support_hash = self.record["preparation_inputs"].get(
            "tier_support_evidence_sha256"
        )
        if support_hash is not None:
            support_path = self.root / "tier_support_evidence.tif"
            if digest(support_path) != support_hash:
                raise ValueError("Padded native tier-support evidence hash mismatch")
            with (
                rasterio.open(support_path) as supported,
                rasterio.open(self.root / "unmasked_evidence.tif") as original,
            ):
                if (
                    supported.crs != original.crs
                    or supported.transform != original.transform
                ):
                    raise ValueError("Tier-support evidence moved original grid")
                if (
                    supported.height < original.height
                    or supported.width < original.width
                ):
                    raise ValueError("Tier-support evidence shrank the original grid")
        for name, expected in (
            self.record["preparation_inputs"].get("science_vectors", {}).items()
        ):
            if (
                Path(name).name != name
                or digest(self.root / "science_vectors" / name) != expected
            ):
                raise ValueError("Original scientific vector hash mismatch")
        operation_hash = self.record["preparation_inputs"].get(
            "geometry_operation_sha256"
        )
        if (
            operation_hash
            and digest(self.root / "geometry_operation.json") != operation_hash
        ):
            raise ValueError("Geometry coordinate-operation audit changed")
        self.manifest_hash = digest(self.path)

    def _validate_native_vector_crs(self, grid):
        """Validate native vector CRS and coastline geometry metadata.

        Inspect GeoPackage layer metadata without loading geometries.
        Original geographic OSM/science vectors are intentionally
        excluded from this raster-aligned native-grid contract.
        """
        expected = rasterio.crs.CRS.from_user_input(grid["crs"])

        names = (
            "coverage",
            "refined_land",
            "refined_water",
            "marine_water",
            "inland_water",
            "unresolved_water",
            "refined_coastline",
            "refined_topology",
            *[
                k
                for k in ("classified_ocean", "classified_inland_water")
                if k in self.products
            ],
        )

        for name in names:
            if name not in self.products:
                raise ValueError(f"Missing authoritative native vector: {name}")

            info = pyogrio.read_info(
                self.products[name],
                layer=self.layers[name],
            )

            crs = info["crs"]
            if not crs or rasterio.crs.CRS.from_user_input(crs) != expected:
                raise ValueError(
                    f"{name} CRS disagrees with native application grid {grid['crs']}"
                )

            if (
                name == "refined_coastline"
                and info["geometry_type"] != "MultiLineString"
            ):
                raise ValueError(
                    "Refined coastline must be a native multilinestring layer"
                )
            if name == "refined_topology" and info["geometry_type"] != "MultiPolygon":
                raise ValueError("Refined topology must be a native multipolygon layer")

    def _coverage_covers(self, geom, crs="EPSG:4326"):
        """Check coverage in its authoritative CRS, preserving curved edges.

        A geographic union of separately reprojected land/water polygons can
        have edge gaps even though they partition the native lattice exactly.
        The sealed grid and validated native partition define target coverage.
        Only the canonical native-grid representation is accepted.
        """
        native = getattr(self, "_native_coverage", None)
        target_crs = self._coverage_crs if native is not None else "EPSG:4326"
        outer = native if native is not None else self.coverage
        if rasterio.crs.CRS.from_user_input(crs) != rasterio.crs.CRS.from_user_input(
            target_crs
        ):
            step = (
                max(geom.bounds[2] - geom.bounds[0], geom.bounds[3] - geom.bounds[1])
                / 64
            )
            geom = (
                gpd.GeoSeries([geom.segmentize(step)], crs=crs)
                .to_crs(horizontal_crs(target_crs))
                .iloc[0]
            )
        return _covers_without_roundoff(outer, geom)

    def barrier_for(self, raster):
        """Validate working-grid coverage before native interpolation consumes it."""
        if not self.record.get("target_cell_authoritative"):
            raise ValueError(
                "barrier: coast_refine requires a current cell-aligned artifact; prepare again"
            )
        with rasterio.open(raster) as src:
            if src.crs is None:
                raise ValueError(
                    "Interpolation working grid has no CRS; configure the native stack CRS"
                )
            footprint = box(*src.bounds)
            crs = src.crs
        if crs != rasterio.crs.CRS.from_user_input(self._coverage_crs):
            raise ValueError(
                "Coast Refine barrier CRS differs from the interpolation working grid"
            )
        if not self._coverage_covers(footprint, crs):
            raise ValueError(
                "Coast Refine barrier does not cover the interpolation working grid, including its buffer"
            )
        return self.products["refined_topology"]

    def keep_geometry(self, remove, aoi=None):
        if remove not in MODES:
            raise ValueError("Unknown Coast Refine removal mode")
        # Removal selects SOURCE-USE geography, never the signed barrier.
        key = {
            "ocean": "retention_classified_ocean",
            "ocean_connected": "retention_marine_water",
            "inland_water": "retention_classified_inland_water",
            "water": "retention_water",
            "land": "retention_water",
        }[remove]
        classes = self._classes
        if key not in classes:
            raise ValueError(
                "Prepared artifact lacks five-class selections; regenerate with the current hook"
            )
        water, structures = classes[key], self._classes["structures"]
        import shapely

        domain = (
            water.difference(structures)
            if remove == "land"
            else shapely.union_all([water, structures])
        )
        exclude = remove != "land"
        scope = self.aoi if aoi is None else aoi
        if (
            scope.is_empty
            or not self._coverage_covers(scope)
            or not _covers_without_roundoff(self.aoi, scope)
        ):
            raise ValueError(
                "Requested processing AOI exceeds prepared AOI or validated classification coverage"
            )
        selected = scope.difference(domain) if exclude else scope.intersection(domain)
        return polygonal_geometry(selected)

    def scope_for(self, module, explicit_aoi=None):
        if explicit_aoi:
            frame = gpd.read_file(explicit_aoi)
            if frame.crs is None or frame.empty:
                raise ValueError("AOI is empty or missing CRS")
            import shapely

            scope = shapely.union_all(horizontal_frame(frame).to_crs(4326).geometry)
        else:
            region = getattr(module, "region", None)
            if region is None:
                scope = self.aoi
            else:
                crs = region.srs or "EPSG:4326"
                scope = (
                    gpd.GeoSeries(
                        [
                            box(
                                region.xmin, region.ymin, region.xmax, region.ymax
                            ).segmentize(
                                max(
                                    region.xmax - region.xmin, region.ymax - region.ymin
                                )
                                / 64
                            )
                        ],
                        crs=horizontal_crs(crs),
                    )
                    .to_crs(4326)
                    .iloc[0]
                )
        if scope.is_empty or not scope.is_valid:
            raise ValueError("Requested processing AOI is invalid")
        return scope

    def check_source(self, path, scope):
        with rasterio.open(path) as src:
            check_encoding(src)
            # A projected raster's edges are not generally straight lines in
            # longitude/latitude. Transforming only four corners can omit real
            # source coverage (and silently skip its required mask checks).
            native_bounds = box(*src.bounds)
            step = max(
                (src.bounds.right - src.bounds.left) / 64,
                (src.bounds.top - src.bounds.bottom) / 64,
            )
            footprint = (
                gpd.GeoSeries(
                    [native_bounds.segmentize(step)], crs=horizontal_crs(src.crs)
                )
                .to_crs(4326)
                .iloc[0]
            )
        required = scope.intersection(footprint)
        if required.is_empty or required.area == 0:
            return None
        if not self._coverage_covers(required) or not _covers_without_roundoff(
            self.aoi, required
        ):
            raise ValueError(
                "Shared mask does not cover AOI intersect source footprint within its prepared AOI"
            )
        return required
