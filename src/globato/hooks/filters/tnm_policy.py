#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""TNM-specific hierarchy policy built on generic spatial claim primitives.

This hook intentionally contains policy, not remote I/O. It expects authoritative
coverage to have already been attached by the appropriate generic footprint hook
(or, for project 1 m, by the TNM/WESM coverage adapter).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import PurePosixPath
import re
from urllib.parse import unquote, urlsplit
import logging

import shapely

from fetchez import spatial
from fetchez.hooks import FetchHook

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TNMProductPolicy:
    rank: int
    weight: float
    dataset: str
    resolution: str
    coverage: str


# Precedence and stack weight are deliberately distinct. Alaska 5 m belongs in
# the 1/9-arc-second CUDEM weight class but follows 1/9 in source precedence.
TNM_PRODUCT_POLICY = {
    "s1m": TNMProductPolicy(700, 10.0, "TNM Seamless 1 m", "1 m", "geometry"),
    "1m": TNMProductPolicy(600, 5.0, "TNM 1 m", "1 m", "wesm"),
    "1_9as": TNMProductPolicy(
        500, 3.0, "TNM 1/9 arc-second", "1/9 arc-second", "geometry"
    ),
    "5m": TNMProductPolicy(450, 3.0, "TNM Alaska 5 m", "5 m", "geometry"),
    "1_3as": TNMProductPolicy(
        300, 1.0, "TNM 1/3 arc-second", "1/3 arc-second", "bounds"
    ),
    "1_as": TNMProductPolicy(200, 0.5, "TNM 1 arc-second", "1 arc-second", "bounds"),
    "2_as": TNMProductPolicy(
        100, 0.25, "TNM Alaska 2 arc-second", "2 arc-second", "bounds"
    ),
}

# The USGS seamless geographic tiles and the observed NED 1/9-arc-second
# project/quarter-degree archives use different naming conventions. Do not
# identify editions using just a bounding box: distinct named datasets can
# overlap without representing successive editions of the same source.
_SEAMLESS_TILE = re.compile(r"^USGS_(?:13|1|2)_([NS]\d{2,3}[EW]\d{3})(?:_|\.)", re.I)
# Observed TNM archive example: ned19_n33x50_w117x75_ca_orangeco_2011.zip.
# Keep the named series, including its geographic tile and project designation;
# strip only its terminal four/eight-digit source/acquisition-year suffix.
# Publication dates come from the TNM API, never from this filename suffix.
_NED19_ARCHIVE = re.compile(
    r"^(ned19_[ns]\d{2}x\d{2}_[ew]\d{3}x\d{2}_[a-z0-9_]+)_(?:19|20)\d{2}(?:\d{4})?\.zip$",
    re.I,
)


TNM_CANONICAL_ORDER = tuple(
    sorted(
        TNM_PRODUCT_POLICY, key=lambda key: TNM_PRODUCT_POLICY[key].rank, reverse=True
    )
)


class TNMPolicy(FetchHook):
    """Annotate TNM entries with canonical hierarchy, weights and provenance."""

    name = "tnm-policy"
    meta_aliases = ["tnm_policy"]
    meta_stage = "manifest"
    meta_category = "manifest-filter"
    meta_desc = "Apply TNM product precedence, WESM chronology, weights and metadata."

    def __init__(
        self,
        coverage_key="tnm_source_coverage",
        claim_geometry_key="claim_geometry",
        excluded_key="excluded_geometry",
        priority_key="claim_priority",
        required_key="claim_required",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.coverage_key = coverage_key
        self.claim_geometry_key = claim_geometry_key
        self.excluded_key = excluded_key
        self.priority_key = priority_key
        self.required_key = required_key

    @staticmethod
    def _to_wkt(geometry):
        return shapely.to_wkt(geometry, rounding_precision=-1)

    @staticmethod
    def _coerce_geometry(value):
        if value is None:
            return None
        if hasattr(value, "geom_type"):
            return value
        if isinstance(value, (bytes, bytearray)):
            return shapely.from_wkb(value)
        if isinstance(value, str):
            return shapely.from_wkt(value)
        if isinstance(value, dict):
            from shapely.geometry import shape

            return shape(value)
        raise RuntimeError(
            f"TNM hierarchy cannot interpret authoritative geometry type {type(value).__name__}"
        )

    @staticmethod
    def _entry_bounds(entry):
        bounds = entry.get("bounds") or entry.get("bbox")
        if not bounds or len(bounds) != 4 or any(value is None for value in bounds):
            raise RuntimeError("TNM hierarchy requires bounds for every TNM entry")
        geometry = spatial.region_to_shapely(bounds)
        if geometry.is_empty or not geometry.is_valid:
            raise RuntimeError("TNM entry bounds contain no usable geometry")
        return geometry

    @staticmethod
    def _module_geometry(mod):
        region = getattr(mod, "wgs_region", None)
        if region is None:
            return None
        bounds = (
            getattr(region, "w", getattr(region, "xmin", None)),
            getattr(region, "e", getattr(region, "xmax", None)),
            getattr(region, "s", getattr(region, "ymin", None)),
            getattr(region, "n", getattr(region, "ymax", None)),
        )
        if any(value is None for value in bounds):
            return None
        geometry = spatial.region_to_shapely(bounds)
        return None if geometry.is_empty else geometry

    @classmethod
    def _entry_geometry(cls, mod, entry):
        geometry = cls._entry_bounds(entry)
        roi = cls._module_geometry(mod)
        return geometry if roi is None else geometry.intersection(roi)

    @staticmethod
    def _project(entry):
        project = entry.get("tnm_project")
        if project is None or not str(project).strip():
            raise RuntimeError("TNM 1 m hierarchy requires provider project identity")
        return str(project).strip()

    @staticmethod
    def _claim_geometry(claim, project):
        if not isinstance(claim, dict) or not claim.get("geometry"):
            raise RuntimeError(
                f"TNM hierarchy received invalid authoritative coverage for {project}"
            )
        try:
            geometry = shapely.from_wkt(claim["geometry"])
        except Exception as exc:
            raise RuntimeError(
                f"TNM hierarchy could not read authoritative coverage for {project}"
            ) from exc
        if geometry is None or geometry.is_empty or not geometry.is_valid:
            raise RuntimeError(
                f"TNM hierarchy received unusable authoritative coverage for {project}"
            )
        return geometry

    def _authoritative_geometry(self, entry):
        # Fetchez records a manifest hook only *after* it completes. TNM's
        # initial API entry already has bounding-box geometry, so geometry
        # validity alone cannot establish that the required footprint ran.
        expected = {
            "s1m": "remote_raster_footprint",
            "1_9as": "remote_archive_footprint",
            "5m": "remote_raster_footprint",
        }.get(entry.get("tnm_product"))
        history = entry.get("history") or []
        if expected and not any(
            isinstance(record, dict)
            and record.get("hook") == expected
            and record.get("stage") == "manifest"
            for record in history
        ):
            raise RuntimeError(
                f"TNM hierarchy requires completed {expected} for "
                f"{entry.get('url') or entry.get('title') or entry.get('tnm_product')}"
            )
        geometry = self._coerce_geometry(entry.get("geometry"))
        if geometry is None or geometry.is_empty or not geometry.is_valid:
            project = entry.get("tnm_project") or entry.get("title") or "TNM source"
            raise RuntimeError(
                f"TNM hierarchy received no authoritative generic footprint for {project}"
            )
        return geometry

    @staticmethod
    def _add_geometry(values, key, geometry):
        current = values.get(key)
        values[key] = (
            geometry if current is None else shapely.union_all([current, geometry])
        )

    def _one_meter_project_masks(self, entries):
        rows = []
        entry_bounds = {}
        original_by_project = {}
        project_claim_names: dict[str, set[str]] = {}

        for mod, entry in entries:
            project = self._project(entry)
            source = self._entry_geometry(mod, entry)
            entry_bounds[id(entry)] = source
            claims = entry.get(self.coverage_key)
            if not isinstance(claims, list) or not claims:
                raise RuntimeError(
                    f"TNM 1 m hierarchy received no WESM source coverage for {project}"
                )
            found = False
            for claim in claims:
                geometry = self._claim_geometry(claim, project).intersection(source)
                if geometry.is_empty:
                    continue
                try:
                    year = int(claim.get("year"))
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"TNM 1 m hierarchy requires WESM collection year for {project}"
                    ) from exc
                rows.append((year, project, geometry))
                self._add_geometry(original_by_project, project, geometry)
                value = claim.get("project")
                if value is not None and str(value).strip():
                    project_claim_names.setdefault(project, set()).add(
                        str(value).strip()
                    )
                found = True
            if not found:
                raise RuntimeError(
                    f"TNM 1 m hierarchy received no intersecting WESM coverage for {project}"
                )

        masks = {project: None for _, project, _ in rows}
        exclusions = {project: None for _, project, _ in rows}
        claimed_by_project = {}
        claimed = None

        # Equal-year projects see the same pre-year claim and therefore never
        # supersede one another. The entire year is added only after all projects
        # in that year have been evaluated.
        for year in sorted({year for year, _, _ in rows}, reverse=True):
            same_year = [row for row in rows if row[0] == year]
            year_geometries = []
            year_by_project = {}
            for _, project, geometry in same_year:
                allowed = geometry if claimed is None else geometry.difference(claimed)
                if not allowed.is_empty:
                    self._add_geometry(masks, project, allowed)

                if claimed is not None:
                    own_claim = claimed_by_project.get(project)
                    higher_other = (
                        claimed if own_claim is None else claimed.difference(own_claim)
                    )
                    removed = geometry.intersection(higher_other)
                    if not removed.is_empty:
                        self._add_geometry(exclusions, project, removed)

                year_geometries.append(geometry)
                self._add_geometry(year_by_project, project, geometry)

            year_claim = shapely.union_all(year_geometries)
            claimed = (
                year_claim
                if claimed is None
                else shapely.union_all([claimed, year_claim])
            )
            for project, geometry in year_by_project.items():
                self._add_geometry(claimed_by_project, project, geometry)

        selected = []
        for mod, entry in entries:
            project = self._project(entry)
            source = entry_bounds[id(entry)]
            geometry = masks.get(project)
            if geometry is None:
                continue
            geometry = geometry.intersection(source)
            if geometry.is_empty:
                continue

            # Audit the validated original WESM/source intersection, not its
            # chronology-trimmed remainder. spatial-claim then subtracts the
            # pre-existing edition exclusion from accepted_geometry.
            original = original_by_project[project].intersection(source)
            entry[self.claim_geometry_key] = self._to_wkt(original)
            removed = exclusions.get(project)
            if removed is not None:
                removed = removed.intersection(source)
                if not removed.is_empty:
                    entry[self.excluded_key] = self._to_wkt(removed)

            names = sorted(project_claim_names.get(project, []))
            entry["tnm_wesm_dataset"] = " | ".join(names) if names else project
            selected.append((mod, entry))

        return selected

    @staticmethod
    def _seamless_tile_key(entry, product):
        """Recognize edition series, never conflating sources by bounding box."""
        filename = PurePosixPath(unquote(urlsplit(entry.get("url") or "").path)).name
        if product == "1_9as":
            match = _NED19_ARCHIVE.fullmatch(filename)
            return (product, match.group(1).lower()) if match else None
        if product not in {"1_3as", "1_as", "2_as"}:
            return None
        match = _SEAMLESS_TILE.match(filename)
        return (product, match.group(0).rstrip("_.").lower()) if match else None

    @staticmethod
    def _publication_date(entry):
        value = entry.get("tnm_publication_date") or entry.get("date")
        try:
            return date.fromisoformat(str(value)[:10])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "TNM seamless tile/archive edition needs a valid publication date: "
                f"{entry.get('url') or entry.get('title')}"
            ) from exc

    def _select_seamless_editions(self, entries, product):
        """Exclude older duplicate-tile coverage, retaining its unique remainder.

        Only entries sharing a recognizable tile/archive-series identifier compete.
        Distinct tiles and named archive series remain peers. Same-day
        overlapping editions are ambiguous and are rejected.
        """
        groups = {}
        for mod, entry in entries:
            key = self._seamless_tile_key(entry, product)
            if key is not None:
                groups.setdefault(key, []).append((mod, entry))

        dropped = set()
        for key, editions in groups.items():
            if len(editions) < 2:
                continue
            dated = sorted(
                (
                    (self._publication_date(entry), mod, entry)
                    for mod, entry in editions
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            newer_claim = None
            previous_date = None
            same_date_claim = None
            for published, mod, entry in dated:
                original = shapely.from_wkt(entry[self.claim_geometry_key])
                if previous_date != published:
                    same_date_claim = None
                    previous_date = published
                if (
                    same_date_claim is not None
                    and original.intersection(same_date_claim).area > 0
                ):
                    raise RuntimeError(
                        f"Ambiguous same-day TNM seamless tile editions for {key}: "
                        f"{published}"
                    )
                same_date_claim = (
                    original
                    if same_date_claim is None
                    else shapely.union_all([same_date_claim, original])
                )
                accepted = (
                    original
                    if newer_claim is None
                    else original.difference(newer_claim)
                )
                if newer_claim is not None:
                    excluded = original.intersection(newer_claim)
                    if not excluded.is_empty:
                        entry[self.excluded_key] = self._to_wkt(excluded)
                if accepted.is_empty or accepted.area == 0:
                    dropped.add(id(entry))
                # Keep the authoritative *original* footprint as the claim.
                # spatial-claim applies the pre-existing edition exclusion to
                # accepted_geometry; narrowing the claim here would instead
                # put excluded_geometry outside the audited source claim.
                # Use original coverage to prevent older editions filling NoData
                # or partial-claim holes inside the newer edition's footprint.
                newer_claim = (
                    original
                    if newer_claim is None
                    else shapely.union_all([newer_claim, original])
                )
        if dropped:
            logger.info(
                "TNM hierarchy omitted %d fully superseded seamless editions",
                len(dropped),
            )
        return [(mod, entry) for mod, entry in entries if id(entry) not in dropped]

    def _dataset_name(self, entry, product):
        base = TNM_PRODUCT_POLICY[product].dataset
        if product == "1m":
            return f"{base}: {entry.get('tnm_wesm_dataset') or self._project(entry)}"
        return base

    def _set_common_policy(self, mod, entry, product):
        policy = TNM_PRODUCT_POLICY[product]
        entry[self.priority_key] = policy.rank
        entry[self.required_key] = True
        entry["weight"] = policy.weight
        entry["claim_product"] = product
        entry["tnm_resolution_tier"] = product
        metadata = entry.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            entry["metadata"] = metadata
        effective_weight = float(getattr(mod, "weight", 1.0)) * policy.weight
        metadata.update(
            {
                "category": "elevation",
                "agency": "USGS",
                "dataset": self._dataset_name(entry, product),
                "resolution": policy.resolution,
                # SourceMasks metadata must describe the same effective weight
                # used by stream-init (module weight x entry weight).
                "weight": effective_weight,
            }
        )
        # Generic, opt-in provenance grouping. SourceMasks ignores this key for
        # every unrelated source, so glob-tnm does not alter established builds.
        entry["source_mask_group_by"] = "MODULE/DATASET/WEIGHT"

    def run(self, entries):
        if not entries:
            return entries

        tnm_by_product = {key: [] for key in TNM_PRODUCT_POLICY}
        passthrough = []
        for mod, entry in entries:
            product = entry.get("tnm_product")
            if product is None:
                passthrough.append((mod, entry))
                continue
            if product not in TNM_PRODUCT_POLICY:
                raise RuntimeError(f"Unsupported TNM product identity: {product}")
            tnm_by_product[product].append((mod, entry))

        selected = []
        for product in TNM_CANONICAL_ORDER:
            product_entries = tnm_by_product[product]
            if not product_entries:
                continue

            if product == "1m":
                product_entries = self._one_meter_project_masks(product_entries)
            else:
                policy = TNM_PRODUCT_POLICY[product]
                resolved_entries = []
                for mod, entry in product_entries:
                    source = self._entry_geometry(mod, entry)
                    if source.is_empty:
                        continue
                    if policy.coverage == "geometry":
                        geometry = self._authoritative_geometry(entry).intersection(
                            source
                        )
                    else:
                        geometry = source
                    if geometry.is_empty:
                        continue
                    entry[self.claim_geometry_key] = self._to_wkt(geometry)
                    resolved_entries.append((mod, entry))
                product_entries = self._select_seamless_editions(
                    resolved_entries, product
                )

            for mod, entry in product_entries:
                self._set_common_policy(mod, entry, product)
                selected.append((mod, entry))

        # Preserve non-TNM inputs; global claim logic will simply ignore them.
        return passthrough + selected
