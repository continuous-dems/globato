#!/usr/bin/env python

"""
globato.hooks.rasters.tnm_hydroflat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove synthetic coastal water surfaces from TNM elevation rasters.

The hook handles two common TNM products:

* hydro-flattened tiles containing one or more large, exactly repeated water
  elevations; and
* tiles containing irregular low-elevation lidar returns over water.

Exact and fuzzy hydroflats can be discovered from the complete raster without
using a coastline as a search barrier. The OSM landmask is resolved lazily only
when the irregular-water fallback is needed. A legacy water-seeded mode remains
available for callers that explicitly want vector-guided discovery.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import hashlib
import logging
import math
import os
from collections import Counter

import numpy as np
import rasterio
import scipy.ndimage
from rasterio.features import rasterize

from .base import RasterGlobalHook

logger = logging.getLogger(__name__)


def _copy_metadata(src, dst):
    """Copy dataset and band metadata, excluding stale statistics."""

    dst.update_tags(**src.tags())
    for band_index in range(1, src.count + 1):
        band_tags = {
            key: value
            for key, value in src.tags(band_index).items()
            if not key.startswith("STATISTICS_")
        }
        if band_tags:
            dst.update_tags(band_index, **band_tags)

        description = src.descriptions[band_index - 1]
        if description:
            dst.set_band_description(band_index, description)

        unit = src.units[band_index - 1]
        if unit:
            dst.set_band_unit(band_index, unit)


def _as_bool(value, name):
    """Parse booleans safely when hook arguments arrive as strings."""

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
        raise ValueError(f"{name} must be true or false")
    return bool(value)


class TNMHydroflat(RasterGlobalHook):
    """Remove TNM coastal hydroflats and irregular water returns.

    The defaults are the raster-first profile validated on Klamath, Yurok, and
    South Coast 1 m TNM tiles. The exact-flat path discovers every sufficiently
    supported repeated elevation across the raster. When exact components do
    not explain enough of the candidate surface, a bounded residual fuzzy pass
    can remove noisy hydroflats connected to exact evidence. If no exact
    component qualifies, the hook resolves the landmask and removes low
    irregular returns only from its water side.

    Args:
      landmask : str, optional
        ``"osm"`` for an automatically generated OSM topology landmask, or a
        path to a polygon vector representing land. In raster seed mode this is
        not resolved unless the irregular-water fallback is needed.
      hydroflat_seed_mode : str, optional
        ``"raster"`` searches the complete raster for exact and fuzzy
        hydroflats and is the default. ``"water"`` retains vector-water-seeded
        discovery for compatibility.
      land_class_field : str, optional
        Attribute containing the land class. If absent from the vector, all
        polygons are treated as land. Set to ``None`` explicitly for an
        unclassified land polygon.
      land_class_value : str, optional
        Value in ``land_class_field`` identifying land polygons.
      water_seed_inset_m : float, optional
        Distance water seeds are inset from the coastline. Zero preserves the
        tested Klamath behavior.
      min_seed_area_m2 : float, optional
        Minimum area supporting an exact elevation before it can become a
        candidate hydroflat value.
      min_component_area_m2 : float, optional
        Minimum total area of a connected candidate component.
      flat_min_elevation, flat_max_elevation : float, optional
        Elevation search interval for synthetic water plateaus.
      max_flat_values : int, optional
        Safety limit on the number of qualifying exact elevation candidates.
      flat_tolerance : float, optional
        Absolute tolerance around each encoded flat value. Zero requires an
        exact native-precision match. Set to ``"auto"`` to preserve the exact
        pass and add a raster-derived fuzzy pass only when exact components
        explain too little of the seeded water surface.
      auto_exact_min_fraction : float, optional
        In auto mode, the minimum fraction of valid, in-range seed-water cells
        explained by exact accepted components. Tiles meeting this threshold
        retain the validated exact-only behavior.
      auto_fuzzy_min_fraction : float, optional
        Minimum fraction of residual seed-water cells within
        ``auto_fuzzy_tolerance_max`` of an accepted exact target before the
        fuzzy pass is allowed to run.
      auto_fuzzy_tolerance_max : float, optional
        Maximum tolerance auto mode may derive for a noisy hydroflat surface.
      auto_fuzzy_quantile : float, optional
        Quantile of qualifying residual deviations used to derive the fuzzy
        tolerance. A value below one ignores a very small outlier tail.
      auto_fuzzy_buffer_m : float, optional
        Maximum distance the fuzzy pass may extend beyond vector water in
        ``hydroflat_seed_mode="water"``. Exact components still follow
        ``clip_flats_to_water`` unchanged. Ignored when
        ``fuzzy_water_barrier`` is false or raster seed mode is active.
      fuzzy_water_barrier : bool, optional
        In water seed mode, constrain fuzzy candidates to vector water plus
        ``auto_fuzzy_buffer_m``. Raster seed mode always searches the complete
        raster.
      fuzzy_require_exact_contact : bool, optional
        Require every accepted fuzzy component to touch an accepted exact
        hydroflat component. This is the primary land-side safeguard when the
        coastline barrier is disabled.
      connectivity : int, optional
        Connected-component neighborhood, either 4 or 8.
      clip_flats_to_water : bool, optional
        Hard-clip accepted flat components to the vector water side in water
        seed mode. Ignored in raster seed mode.
      seam_cleanup_m : float, optional
        Maximum dilation distance around accepted flat components. In water
        seed mode the collar remains water-only. Set to zero to disable.
      irregular_water_max_elevation : float or None, optional
        If no exact component is accepted, remove vector-water cells at or
        below this elevation. Set to ``None`` to disable the fallback.
    """

    name = "tnm_hydroflat"
    meta_aliases = ["tnm-hydroflat"]
    meta_category = "raster-op"
    meta_stage = "file"
    meta_desc = (
        "Remove exact and fuzzy TNM hydroflats raster-wide, with lazy OSM "
        "fallback for low irregular water returns."
    )
    default_suffix = "_hydroflat_clean"

    def __init__(
        self,
        landmask="osm",
        hydroflat_seed_mode="raster",
        land_class_field="class",
        land_class_value="land",
        water_seed_inset_m=0.0,
        min_seed_area_m2=25.0,
        min_component_area_m2=1000.0,
        flat_min_elevation=-5.0,
        flat_max_elevation=5.0,
        max_flat_values=64,
        flat_tolerance="auto",
        auto_exact_min_fraction=0.75,
        auto_fuzzy_min_fraction=0.25,
        auto_fuzzy_tolerance_max=0.35,
        auto_fuzzy_quantile=0.999,
        auto_fuzzy_buffer_m=2.0,
        fuzzy_water_barrier=False,
        fuzzy_require_exact_contact=True,
        connectivity=8,
        clip_flats_to_water=False,
        seam_cleanup_m=2.0,
        irregular_water_max_elevation=4.5,
        **kwargs,
    ):
        kwargs.setdefault("stage", self.meta_stage)
        super().__init__(**kwargs)

        self.landmask = landmask
        self.hydroflat_seed_mode = str(hydroflat_seed_mode).strip().lower()
        self.land_class_field = land_class_field
        self.land_class_value = str(land_class_value).lower()
        self.water_seed_inset_m = max(0.0, float(water_seed_inset_m))
        self.min_seed_area_m2 = max(0.0, float(min_seed_area_m2))
        self.min_component_area_m2 = max(
            0.0,
            float(min_component_area_m2),
        )
        self.flat_min_elevation = float(flat_min_elevation)
        self.flat_max_elevation = float(flat_max_elevation)
        self.max_flat_values = max(1, int(max_flat_values))
        if isinstance(flat_tolerance, str):
            normalized_tolerance = flat_tolerance.strip().lower()
            if normalized_tolerance != "auto":
                raise ValueError(
                    "flat_tolerance must be a non-negative number or 'auto'"
                )
            self.flat_tolerance = "auto"
        else:
            self.flat_tolerance = max(0.0, float(flat_tolerance))
        self.auto_exact_min_fraction = float(auto_exact_min_fraction)
        self.auto_fuzzy_min_fraction = float(auto_fuzzy_min_fraction)
        self.auto_fuzzy_tolerance_max = max(
            0.0,
            float(auto_fuzzy_tolerance_max),
        )
        self.auto_fuzzy_quantile = float(auto_fuzzy_quantile)
        self.auto_fuzzy_buffer_m = max(0.0, float(auto_fuzzy_buffer_m))
        self.fuzzy_water_barrier = _as_bool(
            fuzzy_water_barrier,
            "fuzzy_water_barrier",
        )
        self.fuzzy_require_exact_contact = _as_bool(
            fuzzy_require_exact_contact,
            "fuzzy_require_exact_contact",
        )
        if not self.fuzzy_water_barrier and not self.fuzzy_require_exact_contact:
            raise ValueError(
                "fuzzy_require_exact_contact must be true when "
                "fuzzy_water_barrier is false"
            )
        self.connectivity = int(connectivity)
        self.clip_flats_to_water = _as_bool(
            clip_flats_to_water,
            "clip_flats_to_water",
        )
        self.seam_cleanup_m = max(0.0, float(seam_cleanup_m))
        self.irregular_water_max_elevation = (
            None
            if irregular_water_max_elevation is None
            else float(irregular_water_max_elevation)
        )

        if self.flat_min_elevation > self.flat_max_elevation:
            raise ValueError("flat_min_elevation must be <= flat_max_elevation")
        if self.hydroflat_seed_mode not in {"raster", "water"}:
            raise ValueError("hydroflat_seed_mode must be either 'raster' or 'water'")
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity must be either 4 or 8")
        if not 0 <= self.auto_exact_min_fraction <= 1:
            raise ValueError("auto_exact_min_fraction must be between 0 and 1")
        if not 0 <= self.auto_fuzzy_min_fraction <= 1:
            raise ValueError("auto_fuzzy_min_fraction must be between 0 and 1")
        if not 0 < self.auto_fuzzy_quantile <= 1:
            raise ValueError("auto_fuzzy_quantile must be greater than 0 and <= 1")
        if not np.isfinite(self.auto_fuzzy_tolerance_max):
            raise ValueError("auto_fuzzy_tolerance_max must be finite")
        if not np.isfinite(self.auto_fuzzy_buffer_m):
            raise ValueError("auto_fuzzy_buffer_m must be finite")
        if not np.isfinite(self.flat_min_elevation):
            raise ValueError("flat_min_elevation must be finite")
        if not np.isfinite(self.flat_max_elevation):
            raise ValueError("flat_max_elevation must be finite")
        if self.irregular_water_max_elevation is not None and not np.isfinite(
            self.irregular_water_max_elevation
        ):
            raise ValueError("irregular_water_max_elevation must be finite or None")

    @staticmethod
    def _window_slices(window):
        row0 = int(window.row_off)
        col0 = int(window.col_off)
        return (
            slice(row0, row0 + int(window.height)),
            slice(col0, col0 + int(window.width)),
        )

    @staticmethod
    def _pixel_area_m2(src):
        """Approximate one pixel's area at raster center in square metres."""

        import pyproj

        if src.crs is None:
            raise ValueError("source raster has no CRS")

        if src.crs.is_geographic:
            col = src.width / 2.0
            row = src.height / 2.0
            corners = [
                src.transform * (col, row),
                src.transform * (col + 1, row),
                src.transform * (col + 1, row + 1),
                src.transform * (col, row + 1),
            ]
            geod = pyproj.Geod(ellps="WGS84")
            area, _ = geod.polygon_area_perimeter(
                [point[0] for point in corners],
                [point[1] for point in corners],
            )
            return abs(float(area))

        crs = pyproj.CRS.from_user_input(src.crs)
        factor = crs.axis_info[0].unit_conversion_factor if crs.axis_info else 1.0
        pixel_area = abs(
            src.transform.a * src.transform.e - src.transform.b * src.transform.d
        )
        return float(pixel_area) * float(factor) ** 2

    def _resolve_landmask(self, src):
        """Resolve a vector path or generate an OSM topology landmask."""

        if self.landmask and os.path.exists(self.landmask):
            return self.landmask

        if str(self.landmask).lower() not in {
            "osm",
            "coastline",
            "landmask",
            "glob_coast",
        }:
            return None

        from fetchez.registry import ModuleRegistry
        from fetchez.spatial import Region
        from rasterio.warp import transform_bounds

        from globato.utils import resolve_barrier

        # Direct/API use may resolve OSM before the CLI loads entry points.
        if ModuleRegistry.get_class("osm_landmask") is None:
            ModuleRegistry.load_all()

        # OSM generators query in geographic coordinates. Constructing the
        # Region directly from projected metre bounds can leave wgs_region at
        # infinity in direct hook use, which then fails while formatting the
        # cache key. Transform explicitly before resolving the barrier.
        bounds = transform_bounds(
            src.crs,
            "EPSG:4326",
            *src.bounds,
            densify_pts=21,
        )
        region = Region(bounds[0], bounds[2], bounds[1], bounds[3])
        region.srs = "EPSG:4326"

        mod = getattr(self, "current_mod", None)
        cache_dir = None
        if mod is not None:
            cache_dir = getattr(mod, "outdir", None) or getattr(
                mod,
                "_outdir",
                None,
            )
        cache_dir = cache_dir or self.local_tmp

        return resolve_barrier(
            self.landmask,
            region=region,
            outdir=os.path.join(cache_dir, "tnm_hydroflat_landmasks"),
            include_rivers=False,
            include_lakes=False,
            include_reefs=True,
            include_wetlands=False,
            include_breakwaters=True,
            include_estuaries=True,
            output_mode="topology",
            output_type="vector",
            target_crs=src.crs.to_string(),
        )

    def _read_land_geometries(
        self,
        vector_path,
        raster_crs,
        apply_seed_inset=True,
    ):
        """Read, select, reproject, and optionally expand land polygons."""

        import pyogrio
        import pyproj
        import shapely
        from shapely.ops import transform as transform_geometry

        meta, _, geometry_wkb, fields = pyogrio.raw.read(vector_path)
        vector_crs = meta.get("crs")
        if not vector_crs:
            raise ValueError("landmask has no CRS")

        geometries = shapely.from_wkb(geometry_wkb)
        field_names = list(meta.get("fields", []))
        class_values = None
        if self.land_class_field in field_names:
            class_values = fields[field_names.index(self.land_class_field)]
        elif self.land_class_field is not None:
            logger.debug(
                "[%s] Land class field %r was not found; using every polygon.",
                self.name,
                self.land_class_field,
            )

        selected = []
        for index, geometry in enumerate(geometries):
            if geometry is None or geometry.is_empty:
                continue
            if class_values is not None:
                value = str(class_values[index]).lower()
                if value != self.land_class_value:
                    continue
            if not geometry.is_valid:
                geometry = shapely.make_valid(geometry)
            if not geometry.is_empty:
                selected.append(geometry)

        if not selected:
            return []

        vector_to_raster = pyproj.Transformer.from_crs(
            vector_crs,
            raster_crs,
            always_xy=True,
        )
        land = shapely.union_all(selected)
        land = transform_geometry(vector_to_raster.transform, land)

        if apply_seed_inset and self.water_seed_inset_m > 0:
            raster_to_wgs84 = pyproj.Transformer.from_crs(
                raster_crs,
                "EPSG:4326",
                always_xy=True,
            )
            wgs84_to_raster = pyproj.Transformer.from_crs(
                "EPSG:4326",
                raster_crs,
                always_xy=True,
            )
            land_wgs84 = transform_geometry(
                raster_to_wgs84.transform,
                land,
            )
            center = land_wgs84.centroid
            local_crs = pyproj.CRS.from_proj4(
                f"+proj=aeqd +lat_0={center.y} +lon_0={center.x} "
                "+datum=WGS84 +units=m +no_defs"
            )
            to_local = pyproj.Transformer.from_crs(
                "EPSG:4326",
                local_crs,
                always_xy=True,
            )
            from_local = pyproj.Transformer.from_crs(
                local_crs,
                "EPSG:4326",
                always_xy=True,
            )
            land_local = transform_geometry(to_local.transform, land_wgs84)
            land_local = land_local.buffer(self.water_seed_inset_m)
            if land_local.is_empty:
                return []
            land_wgs84 = transform_geometry(
                from_local.transform,
                land_local,
            )
            land = transform_geometry(wgs84_to_raster.transform, land_wgs84)

        if land.is_empty:
            return []
        return [land]

    def _water_masks(self, src, landmask_path):
        """Build inset seed water and uninset coastline water masks."""

        seed_land_geometries = self._read_land_geometries(
            landmask_path,
            src.crs,
            apply_seed_inset=True,
        )
        if not seed_land_geometries:
            raise ValueError("landmask supplied no usable land polygons")

        seed_land = rasterize(
            seed_land_geometries,
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            default_value=1,
            dtype="uint8",
        ).astype(bool)
        seed_water = ~seed_land

        if self.water_seed_inset_m == 0:
            # Both masks are read-only below, so share the array on the common
            # zero-inset path. A 10 km 1 m tile saves roughly 100 MB here.
            coastline_water = seed_water
        else:
            coastline_land_geometries = self._read_land_geometries(
                landmask_path,
                src.crs,
                apply_seed_inset=False,
            )
            if not coastline_land_geometries:
                raise ValueError("landmask supplied no usable land polygons")
            coastline_land = rasterize(
                coastline_land_geometries,
                out_shape=(src.height, src.width),
                transform=src.transform,
                fill=0,
                default_value=1,
                dtype="uint8",
            ).astype(bool)
            coastline_water = ~coastline_land

        return seed_water, coastline_water

    def _flat_targets(self, src, seed_water, min_seed_cells):
        """Find exact elevations strongly supported by the search seeds."""

        counts = Counter()
        nodata = src.nodata

        for _, window in src.block_windows(1):
            slices = self._window_slices(window)
            seeds = seed_water[slices]
            if not np.any(seeds):
                continue

            elevation = src.read(1, window=window)
            valid = seeds & np.isfinite(elevation)
            if nodata is not None:
                valid &= elevation != nodata
            valid &= elevation >= self.flat_min_elevation
            valid &= elevation <= self.flat_max_elevation

            values = elevation[valid]
            if values.size:
                unique, value_counts = np.unique(values, return_counts=True)
                counts.update(
                    {
                        value.item(): int(count)
                        for value, count in zip(unique, value_counts)
                    }
                )

        ranked = [
            (float(value), count)
            for value, count in counts.most_common()
            if count >= min_seed_cells
        ]
        if not ranked:
            return []

        if len(ranked) > self.max_flat_values:
            logger.warning(
                "[%s] Found %d qualifying exact water values, exceeding the "
                "safety limit of %d; using the irregular-water fallback.",
                self.name,
                len(ranked),
                self.max_flat_values,
            )
            return []

        label = (
            "Exact hydroflat candidates"
            if self.hydroflat_seed_mode == "raster"
            else "Water-seeded exact candidates"
        )
        logger.info(
            "[%s] %s: %s",
            self.name,
            label,
            ", ".join(f"{value:.6f} ({count} cells)" for value, count in ranked),
        )
        return [value for value, _ in ranked]

    def _accepted_flat_mask(
        self,
        src,
        targets,
        seed_water,
        coastline_water,
        min_seed_cells,
        min_component_cells,
        structure,
        tolerance=None,
        clip_to_water=None,
    ):
        """Return complete accepted components and their diagnostics."""

        if tolerance is None:
            tolerance = 0.0 if self.flat_tolerance == "auto" else self.flat_tolerance
        if clip_to_water is None:
            clip_to_water = self.clip_flats_to_water

        remove_mask = np.zeros((src.height, src.width), dtype=bool)
        accepted_targets = []
        accepted_components = 0

        for target in targets:
            candidate = np.zeros((src.height, src.width), dtype=bool)
            native_target = np.asarray(target, dtype=src.dtypes[0]).item()

            for _, window in src.block_windows(1):
                slices = self._window_slices(window)
                elevation = src.read(1, window=window)
                valid = np.isfinite(elevation)
                if src.nodata is not None:
                    valid &= elevation != src.nodata
                candidate[slices] = valid & (
                    np.abs(elevation.astype("float64") - float(native_target))
                    <= tolerance
                )

            labels = np.empty(candidate.shape, dtype="int32")
            component_count = scipy.ndimage.label(
                candidate,
                structure=structure,
                output=labels,
            )
            del candidate

            if component_count == 0:
                del labels
                continue

            component_counts = np.bincount(labels.ravel())
            # Count seed overlap blockwise. Boolean indexing the full label
            # raster can allocate hundreds of MB on a 1 m TNM tile.
            seed_counts = np.zeros(component_count + 1, dtype="int64")
            for _, window in src.block_windows(1):
                slices = self._window_slices(window)
                seed_labels = labels[slices][seed_water[slices]]
                if not seed_labels.size:
                    continue
                unique_labels, unique_counts = np.unique(
                    seed_labels,
                    return_counts=True,
                )
                seed_counts[unique_labels] += unique_counts
            accepted_labels = np.flatnonzero(
                (component_counts >= min_component_cells)
                & (seed_counts >= min_seed_cells)
            )
            accepted_labels = accepted_labels[accepted_labels != 0]

            if not accepted_labels.size:
                logger.info(
                    "[%s] Candidate %.6f had no component meeting the "
                    "water-seed and area safeguards.",
                    self.name,
                    target,
                )
                del labels
                continue

            component_remove = np.isin(labels, accepted_labels)
            if clip_to_water:
                component_remove &= coastline_water
            remove_mask |= component_remove
            accepted_targets.append(target)
            accepted_components += int(accepted_labels.size)
            del component_remove
            del labels

        return remove_mask, accepted_targets, accepted_components

    def _seed_search_cell_count(self, src, seed_water):
        """Count valid seed-water cells in the hydroflat search interval."""

        total = 0
        for _, window in src.block_windows(1):
            slices = self._window_slices(window)
            elevation = src.read(1, window=window)
            valid = seed_water[slices] & np.isfinite(elevation)
            if src.nodata is not None:
                valid &= elevation != src.nodata
            valid &= elevation >= self.flat_min_elevation
            valid &= elevation <= self.flat_max_elevation
            total += int(np.count_nonzero(valid))
        return total

    @staticmethod
    def _nearest_target_distance(values, targets):
        """Return each value's distance from its nearest accepted target."""

        distances = np.full(values.shape, np.inf, dtype="float64")
        values = values.astype("float64", copy=False)
        for target in targets:
            np.minimum(
                distances,
                np.abs(values - float(target)),
                out=distances,
            )
        return distances

    def _auto_fuzzy_tolerance(
        self,
        src,
        targets,
        seed_water,
        exact_remove_mask,
    ):
        """Derive a bounded tolerance from residual seeded-water values."""

        if not targets or self.auto_fuzzy_tolerance_max <= 0:
            return None, 0.0

        # A fixed five-millimetre histogram step is fine enough to preserve
        # sub-decimetre TNM noise while avoiding storage of millions of source
        # values. The final bin is always clamped to the configured safety cap.
        bin_width = min(0.005, self.auto_fuzzy_tolerance_max)
        bin_count = max(
            1,
            int(math.ceil(self.auto_fuzzy_tolerance_max / bin_width)),
        )
        histogram = np.zeros(bin_count, dtype="int64")
        residual_seed_cells = 0
        near_target_cells = 0

        for _, window in src.block_windows(1):
            slices = self._window_slices(window)
            elevation = src.read(1, window=window)
            valid = seed_water[slices] & ~exact_remove_mask[slices]
            valid &= np.isfinite(elevation)
            if src.nodata is not None:
                valid &= elevation != src.nodata
            valid &= elevation >= self.flat_min_elevation
            valid &= elevation <= self.flat_max_elevation

            values = elevation[valid]
            if not values.size:
                continue
            residual_seed_cells += int(values.size)
            distances = self._nearest_target_distance(values, targets)
            near = distances <= self.auto_fuzzy_tolerance_max
            if not np.any(near):
                continue

            near_distances = distances[near]
            near_target_cells += int(near_distances.size)
            indices = np.minimum(
                (near_distances / bin_width).astype("int64"),
                bin_count - 1,
            )
            histogram += np.bincount(indices, minlength=bin_count)

        if residual_seed_cells == 0:
            return None, 0.0

        near_fraction = near_target_cells / residual_seed_cells
        if near_target_cells == 0 or near_fraction < self.auto_fuzzy_min_fraction:
            logger.info(
                "[%s] Auto fuzzy pass skipped: %.3f of residual seeded "
                "water was near exact targets (minimum %.3f).",
                self.name,
                near_fraction,
                self.auto_fuzzy_min_fraction,
            )
            return None, near_fraction

        threshold = int(math.ceil(self.auto_fuzzy_quantile * near_target_cells))
        selected_bin = int(
            np.searchsorted(np.cumsum(histogram), threshold, side="left")
        )
        tolerance = min(
            self.auto_fuzzy_tolerance_max,
            (selected_bin + 1) * bin_width,
        )
        return tolerance, near_fraction

    def _accepted_fuzzy_mask(
        self,
        src,
        targets,
        tolerance,
        seed_water,
        coastline_water,
        exact_remove_mask,
        min_seed_cells,
        min_component_cells,
        structure,
        pixel_area_m2,
    ):
        """Accept noisy residual hydroflats without allowing inland growth."""

        if tolerance is None or tolerance <= 0:
            return np.zeros(exact_remove_mask.shape, dtype=bool), 0

        if self.fuzzy_water_barrier:
            allowed = coastline_water
        else:
            allowed = np.ones(coastline_water.shape, dtype=bool)

        if self.fuzzy_water_barrier and self.auto_fuzzy_buffer_m > 0:
            pixel_size_m = math.sqrt(pixel_area_m2)
            buffer_steps = int(math.ceil(self.auto_fuzzy_buffer_m / pixel_size_m))
            if buffer_steps > 0:
                allowed = scipy.ndimage.binary_dilation(
                    coastline_water,
                    structure=structure,
                    iterations=buffer_steps,
                )

        candidate = np.zeros(exact_remove_mask.shape, dtype=bool)
        for _, window in src.block_windows(1):
            slices = self._window_slices(window)
            elevation = src.read(1, window=window)
            valid = allowed[slices] & ~exact_remove_mask[slices]
            valid &= np.isfinite(elevation)
            if src.nodata is not None:
                valid &= elevation != src.nodata
            valid &= elevation >= self.flat_min_elevation
            valid &= elevation <= self.flat_max_elevation

            values = elevation.astype("float64", copy=False)
            distances = self._nearest_target_distance(values, targets)
            candidate[slices] = valid & (distances <= tolerance)

        labels = np.empty(candidate.shape, dtype="int32")
        component_count = scipy.ndimage.label(
            candidate,
            structure=structure,
            output=labels,
        )
        del candidate

        if component_count == 0:
            del labels
            return np.zeros(exact_remove_mask.shape, dtype=bool), 0

        component_counts = np.bincount(labels.ravel())
        seed_counts = np.zeros(component_count + 1, dtype="int64")
        for _, window in src.block_windows(1):
            slices = self._window_slices(window)
            seed_labels = labels[slices][seed_water[slices]]
            if not seed_labels.size:
                continue
            unique_labels, unique_counts = np.unique(
                seed_labels,
                return_counts=True,
            )
            seed_counts[unique_labels] += unique_counts

        exact_contact = np.ones(component_count + 1, dtype=bool)
        if self.fuzzy_require_exact_contact:
            exact_contact[:] = False
            exact_neighborhood = scipy.ndimage.binary_dilation(
                exact_remove_mask,
                structure=structure,
                iterations=1,
            )
            exact_neighborhood &= ~exact_remove_mask
            touching_labels = np.unique(labels[exact_neighborhood])
            touching_labels = touching_labels[touching_labels != 0]
            exact_contact[touching_labels] = True

        accepted_labels = np.flatnonzero(
            (component_counts >= min_component_cells)
            & (seed_counts >= min_seed_cells)
            & exact_contact
        )
        accepted_labels = accepted_labels[accepted_labels != 0]
        if not accepted_labels.size:
            del labels
            return np.zeros(exact_remove_mask.shape, dtype=bool), 0

        fuzzy_remove_mask = np.isin(labels, accepted_labels)
        del labels
        return fuzzy_remove_mask, int(accepted_labels.size)

    def _unique_output_path(self, src_path):
        """Avoid collisions when TNM cache folders contain equal basenames."""

        identity = hashlib.sha256(
            os.path.abspath(src_path).encode("utf-8")
        ).hexdigest()[:16]
        stem = os.path.splitext(os.path.basename(src_path))[0]
        return os.path.join(
            os.path.abspath("tmp"),
            f"{stem}_{identity}{self.suffix}.tif",
        )

    def run(self, entries):
        """Use collision-safe default output paths for TNM collections."""

        if self.output:
            return super().run(entries)

        original_output = self.output
        processed_entries = []
        try:
            for mod, entry in entries:
                src_path = entry.get("dst_fn")
                self.output = self._unique_output_path(src_path) if src_path else None
                processed_entries.extend(super().run([(mod, entry)]))
        finally:
            self.output = original_output

        return processed_entries

    def process_raster(self, src_path, dst_path, entry=None):
        """Clean one TNM raster, returning True only when cells are removed."""

        with rasterio.open(src_path) as src:
            if src.count < 1 or src.crs is None:
                logger.warning(
                    "[%s] Source has no elevation band or CRS; unchanged.",
                    self.name,
                )
                return False

            if src.nodata is None and not np.issubdtype(
                np.dtype(src.dtypes[0]),
                np.floating,
            ):
                logger.warning(
                    "[%s] Integer source has no NoData value; unchanged.",
                    self.name,
                )
                return False

            landmask_used = False
            landmask_path = None
            if self.hydroflat_seed_mode == "raster":
                # The search interval and component safeguards provide the
                # evidence for exact/fuzzy removal. Avoid resolving OSM unless
                # the irregular-water fallback is actually selected.
                seed_water = np.ones((src.height, src.width), dtype=bool)
                coastline_water = seed_water
                logger.info(
                    "[%s] Searching the complete raster for exact and fuzzy "
                    "hydroflats; no coastline barrier is active.",
                    self.name,
                )
            else:
                try:
                    landmask_path = self._resolve_landmask(src)
                    if not landmask_path:
                        raise ValueError("landmask is missing or unresolved")
                    seed_water, coastline_water = self._water_masks(
                        src,
                        landmask_path,
                    )
                    landmask_used = True
                except Exception as exc:
                    logger.warning(
                        "[%s] Could not prepare landmask: %s; unchanged.",
                        self.name,
                        exc,
                    )
                    return False

            if not np.any(seed_water):
                logger.info(
                    "[%s] Landmask has no water-side cells in this raster; unchanged.",
                    self.name,
                )
                return False

            pixel_area_m2 = self._pixel_area_m2(src)
            min_seed_cells = max(
                1,
                int(math.ceil(self.min_seed_area_m2 / pixel_area_m2)),
            )
            min_component_cells = max(
                1,
                int(math.ceil(self.min_component_area_m2 / pixel_area_m2)),
            )
            structure = scipy.ndimage.generate_binary_structure(
                2,
                2 if self.connectivity == 8 else 1,
            )

            targets = self._flat_targets(src, seed_water, min_seed_cells)
            (
                exact_remove_mask,
                accepted_targets,
                exact_components,
            ) = self._accepted_flat_mask(
                src,
                targets,
                seed_water,
                coastline_water,
                min_seed_cells,
                min_component_cells,
                structure,
                tolerance=(
                    0.0 if self.flat_tolerance == "auto" else self.flat_tolerance
                ),
            )

            seed_search_cells = self._seed_search_cell_count(src, seed_water)
            exact_seed_cells = int(np.count_nonzero(exact_remove_mask & seed_water))
            exact_seed_fraction = (
                exact_seed_cells / seed_search_cells if seed_search_cells else 0.0
            )

            fuzzy_remove_mask = np.zeros(exact_remove_mask.shape, dtype=bool)
            fuzzy_components = 0
            fuzzy_tolerance = None
            fuzzy_near_fraction = 0.0
            selected_mode = "exact" if accepted_targets else "fallback"

            if self.flat_tolerance == "auto" and accepted_targets:
                if exact_seed_fraction >= self.auto_exact_min_fraction:
                    logger.info(
                        "[%s] Auto mode kept exact priority: exact "
                        "components explain %.3f of seeded water "
                        "(minimum %.3f).",
                        self.name,
                        exact_seed_fraction,
                        self.auto_exact_min_fraction,
                    )
                else:
                    if not self.fuzzy_water_barrier:
                        logger.info(
                            "[%s] Auto fuzzy mode is using vector water as "
                            "a seed, not as a coastline barrier%s.",
                            self.name,
                            (
                                "; exact-component contact is required"
                                if self.fuzzy_require_exact_contact
                                else ""
                            ),
                        )
                    (
                        fuzzy_tolerance,
                        fuzzy_near_fraction,
                    ) = self._auto_fuzzy_tolerance(
                        src,
                        accepted_targets,
                        seed_water,
                        exact_remove_mask,
                    )
                    if fuzzy_tolerance is not None:
                        (
                            fuzzy_remove_mask,
                            fuzzy_components,
                        ) = self._accepted_fuzzy_mask(
                            src,
                            accepted_targets,
                            fuzzy_tolerance,
                            seed_water,
                            coastline_water,
                            exact_remove_mask,
                            min_seed_cells,
                            min_component_cells,
                            structure,
                            pixel_area_m2,
                        )
                        if fuzzy_components:
                            selected_mode = "fuzzy"
                            logger.info(
                                "[%s] Auto mode selected residual fuzzy "
                                "cleanup with %.3f m tolerance; exact "
                                "components explain %.3f of seeded water.",
                                self.name,
                                fuzzy_tolerance,
                                exact_seed_fraction,
                            )
                        else:
                            fuzzy_tolerance = None
                            logger.info(
                                "[%s] Auto fuzzy candidates did not meet "
                                "the component safeguards; retaining exact "
                                "results only.",
                                self.name,
                            )

            # Exact decisions are authoritative. Fuzzy cleanup is residual-only
            # and can never replace or broaden an exact component.
            fuzzy_remove_mask &= ~exact_remove_mask
            flat_remove_mask = exact_remove_mask | fuzzy_remove_mask
            accepted_components = exact_components + fuzzy_components

            cleanup_region = None
            cleanup_steps = 0
            if accepted_targets and self.seam_cleanup_m > 0:
                pixel_size_m = math.sqrt(pixel_area_m2)
                cleanup_steps = int(math.ceil(self.seam_cleanup_m / pixel_size_m))
                if cleanup_steps > 0:
                    cleanup_region = scipy.ndimage.binary_dilation(
                        flat_remove_mask & coastline_water,
                        structure=structure,
                        iterations=cleanup_steps,
                        mask=coastline_water,
                    )

            fallback_water = None
            if not accepted_targets:
                if self.irregular_water_max_elevation is None:
                    logger.info(
                        "[%s] No flat component passed safeguards and the "
                        "irregular-water fallback is disabled; unchanged.",
                        self.name,
                    )
                    return False

                if not landmask_used:
                    try:
                        landmask_path = self._resolve_landmask(src)
                        if not landmask_path:
                            raise ValueError("landmask is missing or unresolved")
                        _, coastline_water = self._water_masks(
                            src,
                            landmask_path,
                        )
                        landmask_used = True
                    except Exception as exc:
                        logger.warning(
                            "[%s] Could not prepare the fallback landmask: "
                            "%s; unchanged.",
                            self.name,
                            exc,
                        )
                        return False

                # Requiring both classes prevents a failed or non-overlapping
                # landmask from clipping an entire tile by elevation alone.
                if not np.any(coastline_water) or not np.any(~coastline_water):
                    logger.warning(
                        "[%s] Landmask does not contain both land and water; "
                        "the irregular-water fallback is unsafe and was "
                        "skipped.",
                        self.name,
                    )
                    return False

                fallback_water = coastline_water
                logger.info(
                    "[%s] No qualifying hydroflat was accepted; using the "
                    "OSM water-side elevation fallback at or below %.3f m.",
                    self.name,
                    self.irregular_water_max_elevation,
                )

            profile = self.modify_profile(src.profile.copy())
            if not np.issubdtype(np.dtype(src.dtypes[0]), np.floating):
                profile["predictor"] = 2

            exact_removed = int(np.count_nonzero(exact_remove_mask))
            fuzzy_removed = int(np.count_nonzero(fuzzy_remove_mask))
            flat_removed = exact_removed + fuzzy_removed
            cleanup_removed = 0
            fallback_removed = 0
            total_removed = 0

            with rasterio.open(dst_path, "w", **profile) as dst:
                _copy_metadata(src, dst)

                for _, window in src.block_windows(1):
                    slices = self._window_slices(window)
                    data = src.read(window=window)
                    elevation = data[0]
                    valid = np.isfinite(elevation)
                    if src.nodata is not None:
                        valid &= elevation != src.nodata

                    remove = flat_remove_mask[slices].copy()

                    if cleanup_region is not None:
                        cleanup_remove = cleanup_region[slices] & valid & ~remove
                        cleanup_removed += int(np.count_nonzero(cleanup_remove))
                        remove |= cleanup_remove

                    if fallback_water is not None:
                        fallback_remove = (
                            fallback_water[slices]
                            & valid
                            & (elevation <= self.irregular_water_max_elevation)
                            & ~remove
                        )
                        fallback_removed += int(np.count_nonzero(fallback_remove))
                        remove |= fallback_remove

                    total_removed += int(np.count_nonzero(remove))
                    if src.nodata is not None:
                        data[0][remove] = src.nodata
                    else:
                        data[0][remove] = np.nan
                    dst.write(data, window=window)

                if selected_mode == "fuzzy":
                    method = "exact_fuzzy_hydroflat"
                elif accepted_targets:
                    method = "exact_hydroflat"
                else:
                    method = "fallback"
                dst.update_tags(
                    GLOBATO_HOOK=self.name,
                    TNM_HYDROFLAT_METHOD=method,
                    TNM_HYDROFLAT_MODE=selected_mode,
                    TNM_HYDROFLAT_SEED_MODE=self.hydroflat_seed_mode,
                    TNM_HYDROFLAT_LANDMASK_USED=str(landmask_used).lower(),
                    TNM_HYDROFLAT_VALUES=",".join(
                        f"{value:.9g}" for value in accepted_targets
                    ),
                    TNM_HYDROFLAT_COMPONENTS=str(accepted_components),
                    TNM_HYDROFLAT_EXACT_COMPONENTS=str(exact_components),
                    TNM_HYDROFLAT_FUZZY_COMPONENTS=str(fuzzy_components),
                    TNM_HYDROFLAT_FLAT_CELLS=str(flat_removed),
                    TNM_HYDROFLAT_EXACT_CELLS=str(exact_removed),
                    TNM_HYDROFLAT_FUZZY_CELLS=str(fuzzy_removed),
                    TNM_HYDROFLAT_EXACT_SEED_FRACTION=(f"{exact_seed_fraction:.9g}"),
                    TNM_HYDROFLAT_FUZZY_NEAR_FRACTION=(f"{fuzzy_near_fraction:.9g}"),
                    TNM_HYDROFLAT_FUZZY_TOLERANCE=(
                        "none" if fuzzy_tolerance is None else f"{fuzzy_tolerance:.9g}"
                    ),
                    TNM_HYDROFLAT_FUZZY_WATER_BARRIER=str(
                        self.fuzzy_water_barrier
                    ).lower(),
                    TNM_HYDROFLAT_FUZZY_REQUIRE_EXACT_CONTACT=str(
                        self.fuzzy_require_exact_contact
                    ).lower(),
                    TNM_HYDROFLAT_SEAM_CELLS=str(cleanup_removed),
                    TNM_HYDROFLAT_FALLBACK_CELLS=str(fallback_removed),
                )

        if total_removed == 0:
            try:
                os.remove(dst_path)
            except OSError:
                pass
            logger.info(
                "[%s] Water cleanup selected no cells; unchanged.",
                self.name,
            )
            return False

        logger.info(
            "[%s] Removed %d exact-flat cells from %d component(s) [%s], "
            "%d fuzzy cells from %d component(s), %d seam cells within "
            "%.1f m, and %d fallback cells "
            "(%.3f km² total).",
            self.name,
            exact_removed,
            exact_components,
            ", ".join(f"{value:.6f}" for value in accepted_targets) or "none",
            fuzzy_removed,
            fuzzy_components,
            cleanup_removed,
            self.seam_cleanup_m,
            fallback_removed,
            total_removed * pixel_area_m2 / 1_000_000.0,
        )
        return True
