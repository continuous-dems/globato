#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.hooks.filters.water_surface_filter
~~~~~~~~~~~~~

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import os
import numpy as np
import rasterio

from fetchez.utils import str2bool, float_or, str2inc
from .base import GlobatoFilter

logger = logging.getLogger(__name__)


class WaterSurfaceFilter(GlobatoFilter):
    """Remove likely water-surface returns from nominally topographic point data.

    OSM topology defines where points are candidates for removal. Point-cloud
    structure is then used to estimate the local water surface and preserve
    points that stand materially above it, such as rocks, jetties, and small
    islands.

    Roughness is deliberately used only to identify reliable water-level seed
    cells. A rough cell is *not* automatically treated as terrain: its water
    level can be estimated from nearby smooth seed cells.
    """

    name = "water_surface_filter"
    meta_desc = (
        "Remove likely water-surface points using an OSM water prior and "
        "locally estimated water elevation."
    )
    meta_aliases = ["water_filter", "osm_water_refinement"]

    def __init__(
        self,
        barrier="osm_landmask",
        res="1s",
        max_roughness=0.75,
        max_height_above_water=1.25,
        max_height_above_min=None,
        min_seed_points=3,
        seed_neighbors=4,
        max_seed_distance=4.0,
        soft=False,
        skip_entry=None,
        hard_clip_below=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.barrier = barrier
        self.res = str2inc(res)

        # Roughness is used only to decide whether a cell is trustworthy enough
        # to estimate water level. It is NOT a terrain classifier.
        self.max_roughness = float_or(max_roughness, 0.75)

        # Compatibility with the initial experimental option name.
        if max_height_above_min is not None:
            max_height_above_water = max_height_above_min
        self.max_height_above_water = float_or(max_height_above_water, 1.0)

        self.min_seed_points = max(1, int(min_seed_points))
        self.seed_neighbors = max(1, int(seed_neighbors))
        self.max_seed_distance = float_or(max_seed_distance, 4.0)
        self.soft = str2bool(soft)
        self.skip_entry = skip_entry
        self.hard_clip_below = float_or(hard_clip_below)

        # In-memory raster attributes
        self.mask_array = None
        self.mask_transform = None
        self.mask_width = None
        self.mask_height = None

    def setup(self, mod, entry):
        if not self.barrier:
            logger.warning(f"[{self.name}] No barrier specified. Skipping.")
            return False

        if self.skip_entry and str2bool(entry.get(self.skip_entry)):
            logger.warning(
                f"[{self.name}] {self.skip_entry} detected in "
                f"{entry.get('dst_fn', '')}. Skipping."
            )
            return False

        region = getattr(mod, "region", None)
        mod_outdir = getattr(mod, "_outdir", getattr(mod, "outdir", None))
        cache_dir = mod_outdir if mod_outdir else os.getcwd()
        target_crs = entry.get("src_srs", "EPSG:4326")

        self.entry_weight = entry.get("weight")

        if self.entry_weight is None:
            self.entry_weight = entry.get("metadata", {}).get("weight")

        from globato.utils import resolve_barrier

        barrier_path = resolve_barrier(
            self.barrier,
            region=region,
            outdir=os.path.join(cache_dir, "auto_barriers"),
            res=self.res,
            include_rivers=True,
            include_lakes=True,
            include_breakwaters=True,
            include_wetlands=True,
            include_reefs=False,
            output_type="raster",
            target_crs=target_crs,
        )

        if not barrier_path:
            logger.error(f"[{self.name}] Failed to resolve raster barrier.")
            return False

        with rasterio.open(barrier_path) as src:
            self.mask_array = src.read(1)
            self.mask_transform = src.transform
            self.mask_width = src.width
            self.mask_height = src.height

        return True

    @staticmethod
    def _group_medians(values, groups, num_groups):
        """Return a median for each integer group id."""
        medians = np.full(num_groups, np.nan, dtype="float64")
        for group_id in range(num_groups):
            group_values = values[groups == group_id]
            if group_values.size:
                medians[group_id] = np.median(group_values)
        return medians

    def _estimate_cell_water_levels(
        self,
        cell_coords,
        cell_median_z,
        seed_mask,
    ):
        """Estimate water level for every candidate-water cell from seed cells.

        Smooth, sufficiently populated cells are trusted water-level seeds.
        Other cells inherit an inverse-distance weighted level from nearby seeds.
        Cells with no nearby seed remain unresolved and are conservatively kept.
        """
        levels = np.full(len(cell_coords), np.nan, dtype="float64")
        seed_ids = np.flatnonzero(seed_mask)
        if seed_ids.size == 0:
            return levels

        seed_coords = cell_coords[seed_ids].astype("float64")
        seed_levels = cell_median_z[seed_ids]

        # scipy is already a Globato dependency through the raster/interpolation
        # stack; import locally so this point filter stays light at module import.
        from scipy.spatial import cKDTree

        tree = cKDTree(seed_coords)
        k = min(self.seed_neighbors, seed_ids.size)
        distances, neighbors = tree.query(
            cell_coords.astype("float64"),
            k=k,
            distance_upper_bound=self.max_seed_distance,
        )

        if k == 1:
            distances = distances[:, np.newaxis]
            neighbors = neighbors[:, np.newaxis]

        for cell_id in range(len(cell_coords)):
            dist = distances[cell_id]
            nbr = neighbors[cell_id]
            valid = np.isfinite(dist) & (nbr < seed_ids.size)
            if not np.any(valid):
                continue

            dist = dist[valid]
            nbr = nbr[valid]

            # If this is itself a seed cell, prefer its own measured median.
            exact = dist == 0
            if np.any(exact):
                levels[cell_id] = seed_levels[nbr[np.flatnonzero(exact)[0]]]
                continue

            weights = 1.0 / np.maximum(dist, 1e-6)
            levels[cell_id] = np.average(seed_levels[nbr], weights=weights)

        return levels

    def filter_chunk(self, chunk):
        if len(chunk) == 0 or self.mask_array is None:
            return chunk if not self.soft else np.zeros(0, dtype=bool)

        # -------------------------------------------------------------
        # OSM water prior
        # -------------------------------------------------------------
        inv_transform = ~self.mask_transform
        cols, rows = inv_transform * (chunk["x"], chunk["y"])
        cols = np.floor(cols).astype(int)
        rows = np.floor(rows).astype(int)

        candidate_water = np.zeros(len(chunk), dtype=bool)
        valid = (
            (cols >= 0)
            & (cols < self.mask_width)
            & (rows >= 0)
            & (rows < self.mask_height)
        )
        if np.any(valid):
            sampled_vals = self.mask_array[rows[valid], cols[valid]]
            # osm_landmask raster convention: 0 = water, 1 = land.
            candidate_water[valid] = sampled_vals == 0

        if not np.any(candidate_water):
            return self._format_output(chunk, candidate_water)

        if (
            self.hard_clip_below is not None
            and self.entry_weight is not None
            and self.entry_weight < self.hard_clip_below
        ):
            logger.debug(
                "[%s] hard-clipping water for weight %.3f: %d / %d candidate points",
                self.name,
                self.entry_weight,
                np.count_nonzero(candidate_water),
                len(chunk),
            )
            return self._format_output(chunk, candidate_water)

        # -------------------------------------------------------------
        # Candidate-water analysis cells
        # -------------------------------------------------------------
        x_idx = np.floor(chunk["x"] / self.res).astype(np.int64)
        y_idx = np.floor(chunk["y"] / self.res).astype(np.int64)

        candidate_ids = np.flatnonzero(candidate_water)
        candidate_coords = np.column_stack((x_idx[candidate_ids], y_idx[candidate_ids]))
        cell_coords, cell_indices = np.unique(
            candidate_coords,
            axis=0,
            return_inverse=True,
        )
        num_cells = len(cell_coords)

        z_vals = chunk["z"].astype("float64", copy=False)
        candidate_z = z_vals[candidate_ids]

        cell_count = np.bincount(cell_indices, minlength=num_cells)
        cell_min_z = np.full(num_cells, np.inf, dtype="float64")
        cell_max_z = np.full(num_cells, -np.inf, dtype="float64")
        np.minimum.at(cell_min_z, cell_indices, candidate_z)
        np.maximum.at(cell_max_z, cell_indices, candidate_z)

        cell_roughness = cell_max_z - cell_min_z
        cell_median_z = self._group_medians(
            candidate_z,
            cell_indices,
            num_cells,
        )

        # Smooth cells are trusted only as water-level estimators. Rough cells
        # are unresolved here rather than being declared terrain.
        seed_mask = (
            (cell_count >= self.min_seed_points)
            & np.isfinite(cell_median_z)
            & (cell_roughness <= self.max_roughness)
        )

        cell_water_level = self._estimate_cell_water_levels(
            cell_coords,
            cell_median_z,
            seed_mask,
        )
        point_water_level = cell_water_level[cell_indices]
        has_water_level = np.isfinite(point_water_level)

        # -------------------------------------------------------------
        # Water / terrain decision
        # -------------------------------------------------------------
        # OSM-water points are removed unless they stand substantially above
        # the locally estimated water surface. This deliberately tolerates wave
        # crests and chop while rescuing positive-relief terrain/structures.
        height_above_water = candidate_z - point_water_level
        candidate_is_water = has_water_level & (
            height_above_water <= self.max_height_above_water
        )

        water_mask = np.zeros(len(chunk), dtype=bool)
        water_mask[candidate_ids] = candidate_is_water

        logger.debug(
            "[%s] candidate=%d cells=%d seeds=%d resolved=%d removed=%d rescued=%d",
            self.name,
            candidate_ids.size,
            num_cells,
            np.count_nonzero(seed_mask),
            np.count_nonzero(has_water_level),
            np.count_nonzero(water_mask),
            np.count_nonzero(candidate_water & ~water_mask),
        )

        return self._format_output(chunk, water_mask)

    def _format_output(self, chunk, water_mask):
        """Return target mask for classification or destructively remove targets."""
        if self.soft:
            # GlobatoFilter interprets True as the target/classified points.
            return water_mask

        if self.invert:
            water_mask = ~water_mask

        keep_mask = ~water_mask
        logger.debug(
            "[%s] Retained %d / %d points.",
            self.name,
            np.count_nonzero(keep_mask),
            len(chunk),
        )
        return chunk[keep_mask]
