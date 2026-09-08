#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.hooks.filters.tnm_coverage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply TNM chronology and product fallback to provider coverage.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging

import shapely
from fetchez import spatial
from fetchez.hooks import FetchHook
from shapely.geometry.base import BaseGeometry


logger = logging.getLogger(__name__)


class TNMCoverage(FetchHook):
    """Select TNM streams using cumulative source coverage."""

    name = "tnm-coverage"
    meta_desc = "Apply TNM chronology and product fallback."
    meta_stage = "manifest"
    meta_category = "manifest-filter"
    meta_aliases = ["tnm_coverage"]

    products = ("s1m", "1m", "5m", "1_9as", "1_3as", "1_as", "2_as")
    _product_metadata = {
        "s1m": ("TNM Seamless 1 m", "1 m"),
        "1m": ("TNM 1 m", "1 m"),
        "5m": ("TNM Alaska 5 m", "5 m"),
        "1_9as": ("TNM 1/9 arc-second", "1/9 arc-second"),
        "1_3as": ("TNM 1/3 arc-second", "1/3 arc-second"),
        "1_as": ("TNM 1 arc-second", "1 arc-second"),
        "2_as": ("TNM Alaska 2 arc-second", "2 arc-second"),
    }
    _claim: BaseGeometry | None = None

    def __init__(self, product=None, start=False, **kwargs):
        super().__init__(**kwargs)
        if product not in self.products:
            raise ValueError(f"Unsupported TNM product: {product}")
        self.product = product
        self.start = start

    @classmethod
    def reset(cls):
        """Clear cumulative coverage before a new hierarchy."""

        cls._claim = None

    @staticmethod
    def _module_geometry(mod):
        region = getattr(mod, "wgs_region", None)
        if region is None:
            return None
        bounds = (
            getattr(region, "w", None),
            getattr(region, "e", None),
            getattr(region, "s", None),
            getattr(region, "n", None),
        )
        if any(value is None for value in bounds):
            return None
        return spatial.region_to_shapely(bounds)

    @classmethod
    def _entry_geometry(cls, mod, entry):
        bounds = entry.get("bounds")
        if not bounds or len(bounds) != 4 or any(value is None for value in bounds):
            raise RuntimeError("tnm-coverage requires bounds for every TNM entry")
        geometry = spatial.region_to_shapely(bounds)
        roi = cls._module_geometry(mod)
        return geometry if roi is None else geometry.intersection(roi)

    @staticmethod
    def _project(entry):
        project = entry.get("tnm_project")
        if project is None or not str(project).strip():
            raise RuntimeError(
                "tnm-coverage requires a provider project identity for source coverage"
            )
        return str(project)

    @classmethod
    def _dataset_project(cls, entry):
        projects = {
            str(claim["project"]).strip()
            for claim in entry.get("tnm_source_coverage", [])
            if claim.get("project") is not None and str(claim["project"]).strip()
        }
        return " | ".join(sorted(projects)) if projects else cls._project(entry)

    @classmethod
    def _source_rows(cls, entries, require_year=False):
        rows = []
        for mod, entry in entries:
            project = cls._project(entry)
            source = cls._entry_geometry(mod, entry)
            claims = entry.get("tnm_source_coverage")
            if not isinstance(claims, list) or not claims:
                raise RuntimeError(
                    f"tnm-coverage received no provider source coverage for {project}"
                )
            found = False
            for claim in claims:
                if not isinstance(claim, dict) or not claim.get("geometry"):
                    raise RuntimeError(
                        f"tnm-coverage received invalid provider coverage for {project}"
                    )
                try:
                    geometry = shapely.from_wkt(claim["geometry"]).intersection(source)
                except Exception as exc:
                    raise RuntimeError(
                        f"tnm-coverage could not read provider coverage for {project}"
                    ) from exc
                if geometry.is_empty:
                    continue
                year = claim.get("year")
                if require_year:
                    try:
                        year = int(year)
                    except (TypeError, ValueError) as exc:
                        raise RuntimeError(
                            f"tnm-coverage requires a WESM collection year for {project}"
                        ) from exc
                rows.append((year, project, geometry))
                found = True
            if not found:
                raise RuntimeError(
                    f"tnm-coverage received no intersecting provider coverage for {project}"
                )
        return rows

    @staticmethod
    def _add_geometry(values, key, geometry):
        current = values.get(key)
        values[key] = (
            geometry if current is None else shapely.union_all([current, geometry])
        )

    @classmethod
    def _one_meter_masks(cls, rows, excluded=None):
        projects = {project for _, project, _ in rows}
        masks = {project: None for project in projects}
        exclusions = {project: None for project in projects}
        claimed_by_project = {}
        claimed = excluded

        for year in sorted({year for year, _, _ in rows}, reverse=True):
            same_year = [row for row in rows if row[0] == year]
            year_geometries = []
            year_by_project = {}
            for _, project, geometry in same_year:
                allowed = geometry if claimed is None else geometry.difference(claimed)
                if not allowed.is_empty:
                    cls._add_geometry(masks, project, allowed)

                if claimed is not None:
                    own_claim = claimed_by_project.get(project)
                    higher_other = (
                        claimed if own_claim is None else claimed.difference(own_claim)
                    )
                    removed = geometry.intersection(higher_other)
                    if not removed.is_empty:
                        cls._add_geometry(exclusions, project, removed)

                year_geometries.append(geometry)
                cls._add_geometry(year_by_project, project, geometry)

            year_claim = shapely.union_all(year_geometries)
            claimed = (
                year_claim
                if claimed is None
                else shapely.union_all([claimed, year_claim])
            )
            for project, geometry in year_by_project.items():
                cls._add_geometry(claimed_by_project, project, geometry)

        return masks, exclusions, claimed

    @staticmethod
    def _set_mask(entry, geometry, excluded=None):
        entry["_tnm_coverage_wkt"] = shapely.to_wkt(geometry, rounding_precision=-1)
        entry.pop("_tnm_excluded_wkt", None)
        if excluded is not None and not excluded.is_empty:
            entry["_tnm_excluded_wkt"] = shapely.to_wkt(excluded, rounding_precision=-1)

    @classmethod
    def _set_metadata(cls, entry, product):
        dataset, resolution = cls._product_metadata[product]
        if product == "1m":
            dataset = f"{dataset}: {cls._dataset_project(entry)}"

        metadata = entry.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            entry["metadata"] = metadata
        metadata.update(
            {
                "category": "elevation",
                "dataset": dataset,
                "resolution": resolution,
            }
        )
        entry["tnm_resolution_tier"] = product

    @classmethod
    def _cumulative_claim(cls, entries, excluded):
        if not entries:
            return excluded
        coverage = shapely.union_all(
            [shapely.from_wkt(entry["_tnm_coverage_wkt"]) for _, entry in entries]
        )
        return coverage if excluded is None else shapely.union_all([excluded, coverage])

    def _run_1m(self, entries, excluded):
        rows = self._source_rows(entries, require_year=True)
        masks, exclusions, tier_claim = self._one_meter_masks(rows, excluded)
        selected = []
        for mod, entry in entries:
            source = self._entry_geometry(mod, entry)
            project = self._project(entry)
            geometry = masks.get(project)
            if geometry is None:
                continue
            geometry = geometry.intersection(source)
            if geometry.is_empty:
                continue
            removed = exclusions.get(project)
            if removed is not None:
                removed = removed.intersection(source)
            self._set_mask(entry, geometry, removed)
            self._set_metadata(entry, self.product)
            selected.append((mod, entry))
        return selected, tier_claim

    def _run_1_9as(self, entries, excluded):
        rows = self._source_rows(entries)
        coverage = {}
        for _, project, geometry in rows:
            self._add_geometry(coverage, project, geometry)

        selected = []
        for mod, entry in entries:
            source = self._entry_geometry(mod, entry)
            geometry = coverage[self._project(entry)].intersection(source)
            removed = None
            if excluded is not None:
                geometry = geometry.difference(excluded)
                removed = excluded.intersection(source)
            if geometry.is_empty:
                continue
            self._set_mask(entry, geometry, removed)
            self._set_metadata(entry, self.product)
            selected.append((mod, entry))
        return selected, self._cumulative_claim(selected, excluded)

    def _run_product(self, entries, excluded):
        selected = []
        for mod, entry in entries:
            source = self._entry_geometry(mod, entry)
            geometry = source
            if self.product in {"s1m", "5m"}:
                claims = entry.get("tnm_source_coverage")
                if not claims:
                    raise RuntimeError("Projected TNM product requires raster coverage")
                geometry = shapely.union_all(
                    [shapely.from_wkt(claim["geometry"]) for claim in claims]
                ).intersection(source)
            removed = None
            if excluded is not None:
                geometry = geometry.difference(excluded)
                removed = excluded.intersection(source)
            if geometry.is_empty:
                continue
            self._set_mask(entry, geometry, removed)
            self._set_metadata(entry, self.product)
            selected.append((mod, entry))
        return selected, self._cumulative_claim(selected, excluded)

    def run(self, entries):
        if self.start:
            self.reset()

        excluded = self._claim
        if entries:
            if self.product == "1m":
                entries, claim = self._run_1m(entries, excluded)
            elif self.product == "1_9as":
                entries, claim = self._run_1_9as(entries, excluded)
            else:
                entries, claim = self._run_product(entries, excluded)
            self.__class__._claim = claim
        logger.debug(f"[tnm-coverage] {self.product}: selected {len(entries)} entries.")
        return entries
