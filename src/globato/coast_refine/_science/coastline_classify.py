#!/usr/bin/env python3
"""Generic coastline classification and regional marine-connectivity owners.

This module is the active production path.  It contains no ROI names, fixed
coordinates, bundled reviewed edits, or region-specific defaults.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import shapely
import pyogrio
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import Point, Polygon, shape as geometry_shape

from .grid_utils import mask_on_grid, pixel_metres

FOUR = ndimage.generate_binary_structure(2, 1)


def polygon_water(geometry):
    """Normalize arbitrary valid polygonal water geometry."""
    from .coast_refine_v6_6_1.input_resolver import _polygonal

    return _polygonal(geometry)


def _union_input_water_features(frame, *, label, allow_empty=False):
    """Validate polygon features *before* GEOS overlay or union.

    Projection and GeoPackage roundtripping can create self-intersections,
    even when the source file was valid.  GEOS ``union_all`` throws a
    ``side location conflict`` if even one feature in a multi-feature input is
    invalid.  Calling ``make_valid`` only on the *union result* is too late.

    Preserve the source file. Repair only the in-memory projected polygons,
    before union, and let the caller enforce the original cell-center water
    decisions on the actual written output. This does not infer new land/water
    from area, simplify shoreline geometry, or introduce region-specific rules.
    """
    parts = []
    repaired = 0
    saw_nonempty = False
    for idx, geom in enumerate(frame.geometry):
        if geom is None or geom.is_empty:
            continue
        saw_nonempty = True
        if not shapely.is_valid(geom):
            geom = shapely.make_valid(geom)
            repaired += 1
        try:
            # _polygonal discards only non-area fragments of make_valid,
            # and ensures overlapping parts of a single feature are dissolved.
            part = polygon_water(geom)
        except shapely.errors.GEOSException as exc:
            raise ValueError(
                f"{label}: cannot normalize polygon feature {idx} "
                "without changing scientific classification"
            ) from exc
        if not part.is_empty:
            if not shapely.is_valid(part):
                raise ValueError(f"{label}: invalid normalized polygon feature {idx}")
            parts.append(part)
    if not parts:
        # An explicitly empty water domain is legitimate (an entirely land
        # raster, or a restoration that excludes all starting water).  Never
        # treat a nonempty, malformed source geometry repaired down to lines
        # or points as an empty water mask: that remains a hard failure.
        if allow_empty and not saw_nonempty:
            return polygon_water(shapely.GeometryCollection()), repaired
        raise ValueError(f"{label}: no nonempty polygonal geometry")
    try:
        result = polygon_water(shapely.union_all(parts))
    except shapely.errors.GEOSException as exc:
        raise ValueError(
            f"{label}: GEOS union failed after individual feature repair; "
            "input needs review, no approximate buffering or snapping applied"
        ) from exc
    if not shapely.is_valid(result):
        raise ValueError(f"{label}: union result is not valid")
    return result, repaired


def stitch_water(baseline, owned_outputs):
    """Replace only owned tile interiors; retain exterior baseline context."""
    owned = [roi for roi, _ in owned_outputs]
    for i, a in enumerate(owned):
        for b in owned[i + 1 :]:
            if a.intersection(b).area > 1e-14:
                raise ValueError(
                    "Tile interiors overlap; ambiguous correction ownership"
                )
    exterior = baseline.difference(shapely.union_all(owned))
    patches = [water.intersection(roi) for roi, water in owned_outputs]
    return polygon_water(shapely.union_all([exterior] + patches))


def ocean_components(water, context, seeds=None, *, ocean_anchor=None):
    """Classify water components with conservative marine inference.

    With ocean_anchor, positive-area overlap with that independently supplied
    ocean evidence selects marine components. Empty ocean/empty water are valid
    known all-land or inland-only cases. No corner inference is used.

    Without ocean_anchor, the legacy explicit-mask route uses context topology:
    water is clipped to the known
    connectivity context and split into polygon components. Marine selection is
    then entirely topological:

    * components occupying a context corner are certified marine; this is the
      normal coastal case because context extends beyond the processed tiles;
    * if no context corner contains water, automatic marine inference stops
      rather than guessing from component size, direction, or boundary contact;
    * other components leaving known context are ``unresolved`` rather than
      guessed marine or inland;
    * fully enclosed components are inland water.

    Optional ``seeds`` are explicit user evidence only. No built-in workflow
    supplies a location-specific seed.
    """
    water = polygon_water(water.intersection(context))
    parts = sorted(
        [p for p in shapely.get_parts(water) if isinstance(p, Polygon)],
        key=lambda p: (-p.area, p.bounds),
    )
    if not parts and ocean_anchor is None:
        raise ValueError("Connectivity context contains no water polygons")

    minx, miny, maxx, maxy = context.bounds
    scale = max(maxx - minx, maxy - miny, 1.0)
    contact_tol = scale * 1e-12
    boundary = context.boundary
    contact = [float(p.boundary.intersection(boundary).length) for p in parts]
    exterior = {i for i, length in enumerate(contact) if length > contact_tol}

    selected = set()
    mode = "automatic_context_exterior"
    rule = None
    corner_hits = {i: [] for i in range(len(parts))}

    if ocean_anchor is not None and not seeds:
        mode = "explicit_ocean_anchor"
        rule = "positive-area overlap with supplied ocean anchor; no corner inference"
        selected = {
            i for i, p in enumerate(parts) if p.intersection(ocean_anchor).area > 0
        }
    elif seeds:
        mode = "explicit_user_seed_override"
        rule = "user-supplied points strictly inside water components"
        for x, y in seeds:
            ids = [i for i, p in enumerate(parts) if p.contains(Point(x, y))]
            if len(ids) != 1:
                raise ValueError(
                    f"Marine seed {(x, y)} must be strictly inside one water component"
                )
            selected.add(ids[0])
    else:
        # Probe just inside the four context corners. The offset is relative to
        # the context itself, so this adds no region- or resolution-specific
        # distance parameter and a mere point contact cannot certify marine.
        dx = max(maxx - minx, 0.0)
        dy = max(maxy - miny, 0.0)
        epsx = max(dx * 1e-9, 1e-12)
        epsy = max(dy * 1e-9, 1e-12)
        probes = [
            ("SW", Point(minx + epsx, miny + epsy)),
            ("NW", Point(minx + epsx, maxy - epsy)),
            ("SE", Point(maxx - epsx, miny + epsy)),
            ("NE", Point(maxx - epsx, maxy - epsy)),
        ]
        for i, p in enumerate(parts):
            corner_hits[i] = [name for name, point in probes if p.contains(point)]
        selected = {i for i in exterior if corner_hits[i]}
        if selected:
            rule = "water occupying one or more context corners"
        elif not exterior:
            raise ValueError(
                "Cannot infer marine water automatically because no water component "
                "has positive-length contact with the connectivity-context boundary. "
                "Expand the context or supply an explicit --marine-seed override."
            )
        else:
            raise ValueError(
                "Cannot infer marine water automatically because exterior water exists "
                "but no context corner is water. Boundary-contact length, component area, "
                "and compass direction are not universal marine evidence. Expand the "
                "connectivity context so marine water reaches a corner, or supply an "
                "explicit --marine-seed override."
            )

    if not selected and ocean_anchor is None:
        raise ValueError("Marine classification selected no water component")

    groups = {"marine": [], "inland": [], "unresolved": []}
    rows = []
    for i, p in enumerate(parts):
        edge = i in exterior
        status = "marine" if i in selected else ("unresolved" if edge else "inland")
        groups[status].append(p)
        rows.append(
            {
                "component": i + 1,
                "status": status,
                "context_edge": bool(edge),
                "marine_selected": i in selected,
                # Historical receipt key retained for review-package compatibility.
                "seeded": i in selected,
                "selection_mode": mode,
                "boundary_contact_length": contact[i],
                "corner_hits": corner_hits[i],
                "bounds": list(p.bounds),
                "interior_land_holes": len(p.interiors),
            }
        )
    geometries = {k: polygon_water(shapely.union_all(v)) for k, v in groups.items()}
    return geometries, {
        "components": rows,
        "marine_components": len(selected),
        "seeded_components": len(selected),
        "single_connected_component_verified": len(selected) == 1,
        "marine_selection_mode": mode,
        "marine_selection_rule": rule,
        "exterior_components": len(exterior),
        "unresolved_exterior_components": len(exterior - selected),
        "context_method": (
            "refined tile interiors plus unrefined starting water outside tiles"
        ),
        "point_contacts_connect": False,
        "raw_nodata_controls_connectivity": False,
    }


def write(path, arr, profile, dtype, nodata):
    p = profile.copy()
    p.update(
        count=1, dtype=dtype, nodata=nodata, compress="deflate", BIGTIFF="IF_SAFER"
    )
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(dtype, copy=False), 1)


def _cell_polygons(cells, transform):
    """Polygonize only discrepancy cells; 4-connectivity respects cell edges.

    A diagonal contact does not turn two separate raster cells into one ring.
    Pixel geometry is used solely where the written vector demonstrably
    disagrees with its intended cell-center water classification.
    """
    if not cells.any():
        return shapely.GeometryCollection()
    parts = [
        geometry_shape(g)
        for g, _ in shapes(
            cells.astype("uint8"),
            mask=cells,
            transform=transform,
            connectivity=4,
        )
    ]
    return shapely.union_all(parts)


def write_final_water(
    path,
    restoration_water_path,
    rejected,
    expected_water,
    transform,
    crs,
    *,
    baseline_water_path=None,
):
    """Add signed-elevation land-to-water cells without losing grid membership.

    Retain the true shoreline geometry from the refined vector.  Raster/vector
    overlays and a round trip through the input GPKG CRS can occasionally
    change the classification at isolated raster cell centers.  Repair only
    those *observed* disagreements with exact source-grid cell footprints,
    then re-read and verify the actual written vector.  No raster values or
    shoreline classification decisions are changed; unresolved differences
    remain fatal.
    """
    path = Path(path)
    expected_water = np.asarray(expected_water, dtype=bool)
    if expected_water.shape != rejected.shape:
        raise ValueError("Expected water and rejected rasters have different shapes")
    water = gpd.read_file(restoration_water_path, layer="mask")
    if water.crs is None:
        raise ValueError(f"Missing vector CRS: {restoration_water_path}")
    original_crs = water.crs
    properties = water.iloc[:1].copy()
    if properties.empty:
        raise ValueError(f"Empty restoration vector: {restoration_water_path}")
    water = water.to_crs(crs)
    restored_geometry, restored_input_repairs = _union_input_water_features(
        water, label="restoration water", allow_empty=True
    )
    baseline_input_repairs = 0
    clipped_restoration_area_native = 0.0
    if baseline_water_path is not None:
        baseline = gpd.read_file(baseline_water_path, layer="mask")
        if baseline.crs is None:
            raise ValueError(f"Missing baseline vector CRS: {baseline_water_path}")
        baseline = baseline.to_crs(crs)
        baseline_geometry, baseline_input_repairs = _union_input_water_features(
            baseline, label="baseline water", allow_empty=True
        )
        # Enforce the original owner's restoration-only contract in continuous
        # geometry, not only at sampled raster centers.  This removes stray
        # slivers from geometry repair and cannot promote a land cell to water.
        outside = restored_geometry.difference(baseline_geometry)
        clipped_restoration_area_native = float(outside.area)
        restored_geometry = polygon_water(
            shapely.intersection(restored_geometry, baseline_geometry)
        )
    if restored_input_repairs or baseline_input_repairs:
        print(
            "[coast_refine] normalized invalid projected water polygon features "
            f"before union (restoration={restored_input_repairs}, "
            f"baseline={baseline_input_repairs}); original vectors unchanged; "
            "written raster-cell membership will be checked",
            flush=True,
        )
    polygons = [
        shapely.make_valid(geometry_shape(g))
        for g, _ in shapes(
            rejected.astype("uint8"),
            mask=rejected,
            transform=transform,
            connectivity=8,
        )
    ]
    removal = shapely.union_all(polygons) if polygons else shapely.GeometryCollection()
    final = shapely.make_valid(shapely.union_all([restored_geometry, removal]))
    initial_missing = initial_extra = 0
    correction_passes = 0
    output_projection_repairs = 0
    # Initial comparison, followed by at most two bounded, cell-local repairs.
    # Never accept a changed grid simply because a geometry is GEOS-valid.
    for attempt in range(3):
        output_geometry = gpd.GeoSeries([final], crs=crs).to_crs(original_crs).iloc[0]
        # CRS roundtrip alone can make a valid projected polygon invalid,
        # particularly along narrow cell-edge slivers. Avoid republishing
        # another invalid GPKG for the next raster module to ingest. Any
        # geometry correction is still checked against the intended native
        # raster-cell decisions immediately after writing.
        if not shapely.is_valid(output_geometry):
            output_geometry = polygon_water(shapely.make_valid(output_geometry))
            output_projection_repairs += 1
        properties["geometry"] = gpd.GeoSeries(
            [output_geometry], crs=original_crs
        ).values
        if path.exists():
            path.unlink()
        properties.to_file(path, layer="mask", driver="GPKG")
        actual = mask_on_grid(path, rejected.shape, transform, crs)
        missing = expected_water & ~actual
        extra = actual & ~expected_water
        n_missing, n_extra = int(missing.sum()), int(extra.sum())
        if attempt == 0:
            initial_missing, initial_extra = n_missing, n_extra
        if not (n_missing or n_extra):
            if polygons:
                gpd.GeoDataFrame(
                    {"status": ["land_to_water"]},
                    geometry=[removal],
                    crs=crs,
                ).to_file(path, layer="land_to_water", driver="GPKG")
            return {
                "path": str(path),
                "cell_center_water_membership_verified": True,
                "land_to_water_cells": int(rejected.sum()),
                "initial_missing_water_cells": initial_missing,
                "initial_extra_water_cells": initial_extra,
                "vector_repair_passes": correction_passes,
                "final_membership_disagreement_cells": 0,
                "restoration_subset_vector_clipped": baseline_water_path is not None,
                "restoration_invalid_input_features_repaired": restored_input_repairs,
                "baseline_invalid_input_features_repaired": baseline_input_repairs,
                "output_crs_projection_repair_passes": output_projection_repairs,
                "restoration_outside_baseline_area_in_raster_crs_units2": clipped_restoration_area_native,
            }
        if attempt == 2:
            raise ValueError(
                f"Final coastline/raster water membership differs: {path}; "
                f"missing_water_cells={n_missing}, extra_water_cells={n_extra}, "
                f"initial_missing={initial_missing}, initial_extra={initial_extra}"
            )
        # Subtract false water first; restore missing water afterwards. The
        # two masks are disjoint by construction and use the original grid.
        if n_extra:
            final = shapely.difference(final, _cell_polygons(extra, transform))
        if n_missing:
            final = shapely.union(final, _cell_polygons(missing, transform))
        final = shapely.make_valid(final)
        correction_passes += 1
    raise AssertionError("Unreachable water-membership repair exit")


def connected_classification(
    z,
    valid,
    baseline_water,
    land_proposals,
    *,
    water_proposals=None,
    shore_level_m=0.0,
    threshold_m=0.01,
):
    """Change class only with observed signed elevation and same-class support.

    Seeds belong to the ORIGINAL class and satisfy the same elevation test as
    their candidates. Propagation uses cell edges, never diagonal contacts,
    NoData, or the near-zero band. Both directions use the existing 3x3
    observed-sign coherence test. Class-specific vetoes restrict proposals
    BEFORE propagation, so a withheld cell cannot support cells beyond it.
    Land proposals also pass the positive-water artifact screens; flat or
    sloping submerged terrain does not require positive terrain relief.
    Both masks derive from the returned single water classification.
    """
    z = np.asarray(z)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(z)
    baseline_water = np.asarray(baseline_water, dtype=bool)
    land_proposals = np.asarray(land_proposals, dtype=bool)
    water_proposals = (
        np.ones(z.shape, dtype=bool)
        if water_proposals is None
        else np.asarray(water_proposals, dtype=bool)
    )
    if not (
        z.shape
        == valid.shape
        == baseline_water.shape
        == land_proposals.shape
        == water_proposals.shape
    ):
        raise ValueError("Classification arrays must share one grid")
    if (
        not np.isfinite(shore_level_m)
        or not np.isfinite(threshold_m)
        or threshold_m < 0
    ):
        raise ValueError(
            "Classification reference must be finite and threshold nonnegative"
        )
    positive = valid & (z > shore_level_m + threshold_m)
    negative = valid & (z < shore_level_m - threshold_m)
    from .coast_refine_v6_6_1.coast_refine_core import _fraction

    land_proposals = (
        land_proposals & positive & (_fraction(positive, valid, size=3) >= 0.5)
    )
    water_proposals = (
        water_proposals & negative & (_fraction(negative, valid, size=3) >= 0.5)
    )
    land_seed = ~baseline_water & positive
    land_domain = positive & (~baseline_water | land_proposals)
    water_seed = baseline_water & negative
    water_domain = negative & (baseline_water | water_proposals)
    land_connected = ndimage.binary_propagation(
        land_seed, structure=FOUR, mask=land_domain
    )
    water_connected = ndimage.binary_propagation(
        water_seed, structure=FOUR, mask=water_domain
    )
    added_land = baseline_water & land_proposals & land_connected
    added_water = ~baseline_water & water_proposals & water_connected
    water = (baseline_water & ~added_land) | added_water
    uncertain = ~(positive | negative)
    if np.any(water[uncertain] != baseline_water[uncertain]):
        raise ValueError("Classification changed an unobserved or near-zero cell")
    return water, added_land, added_water


def screen_decision_audit(
    review_path, restored, transform, crs, *, return_conflicts=False
):
    """Identify withheld-review cells mistakenly promoted to restored land.

    A false result remains available to callers that require a strict audit.
    The full classifier can request the exact conflicting source-grid cells to
    *retain as water*, never as evidence for restoration.
    """
    available = set(pyogrio.list_layers(review_path)[:, 0].tolist())
    rows = []
    conflicts = np.zeros_like(restored, dtype=bool) if return_conflicts else None
    for layer in (
        "rejected_hydroflat_transition",
        "unsupported_positive_review",
        "nearshore_transition_review",
        "detached_review",
    ):
        water = (
            mask_on_grid(review_path, restored.shape, transform, crs, layer)
            if layer in available
            else None
        )
        promoted = (water & restored) if water is not None else None
        rows.append(
            {
                "layer": layer,
                "review_cells": int(water.sum()) if water is not None else 0,
                "promoted_review_cells": int(promoted.sum())
                if promoted is not None
                else 0,
            }
        )
        if conflicts is not None and promoted is not None:
            conflicts |= promoted
    result = {
        "review_gpkg": str(review_path),
        "layers": rows,
        "passed": all(row["promoted_review_cells"] == 0 for row in rows),
    }
    return (result, conflicts) if return_conflicts else result


def beach_grade_evidence(
    z,
    valid,
    baseline_water,
    restored_land,
    *,
    dx_m,
    dy_m,
    profile_m=15.0,
    minimum_rise_m=0.15,
    return_unconfirmed=False,
):
    """Conservatively corroborate *already accepted* nearshore land cells.

    Evaluate a 15 m cross-shore profile on each side of the original OSM
    shoreline, using the unmodified source elevations on BOTH sides. A beach
    needs sufficient observed land and water samples, a persistent increase
    toward land, and a profile whose measured relief is predominantly explained
    by cross-shore distance rather than oscillation. A combined profile alone
    can be fooled by a flat positive water surface next to a rising bank. To
    release a narrow ribbon, independently require a measurable shoreward rise
    among original-water observations on the seaward side. This is evidence
    for a beach-like cross-shore grade, NOT proof of dry land or a tidal datum.
    A single positive pixel, source nodata edge, flat water return or
    oscillating water surface cannot pass.

    This is corroboration for a regional *narrow-ribbon exception*, not a new
    standalone source of land. Uncorroborated cells keep the earlier source
    classification, but remain quarantined regionally if part of a thin ribbon.
    Process bounded 512x512 windows with halos in both axes; do not build
    multi-GB whole-width intermediate arrays.
    """
    candidate = valid & baseline_water & restored_land
    result = np.zeros(candidate.shape, dtype=bool)
    unconfirmed = np.zeros(candidate.shape, dtype=bool) if return_unconfirmed else None
    if not candidate.any() or not (~baseline_water & valid).any():
        return (result, unconfirmed) if return_unconfirmed else result
    hx = max(1, int(np.ceil(float(profile_m) / float(dx_m))))
    hy = max(1, int(np.ceil(float(profile_m) / float(dy_m))))
    # Prevent underdetermined, coarse-resolution stencils from being called
    # reliable measured profiles. A low-resolution source stays review-only.
    if min(2 * hx + 1, 2 * hy + 1) < 5:
        return (result, unconfirmed) if return_unconfirmed else result
    size = (2 * hy + 1, 2 * hx + 1)
    window_area = float(size[0] * size[1])
    # Both axes must be bounded. The previous implementation bounded only
    # rows and allocated a dozen full-width float64 intermediates. On wide
    # native 1 m TNM tiles this could require gigabytes per stripe.
    # A 3-profile-width halo guarantees that every signed shoreline distance
    # used inside the core's profile window is resolved before clipping at
    # +/-2*profile_m. There is no resampling or interpolation.
    halo_y = 3 * hy + 2
    halo_x = 3 * hx + 2
    for first in range(0, z.shape[0], 512):
        last = min(z.shape[0], first + 512)
        for first_x in range(0, z.shape[1], 512):
            last_x = min(z.shape[1], first_x + 512)
            if not candidate[first:last, first_x:last_x].any():
                continue
            top = max(0, first - halo_y)
            bottom = min(z.shape[0], last + halo_y)
            left = max(0, first_x - halo_x)
            right = min(z.shape[1], last_x + halo_x)
            sl = (slice(top, bottom), slice(left, right))
            water = baseline_water[sl]
            good = valid[sl] & np.isfinite(z[sl])
            dsea = ndimage.distance_transform_edt(water, sampling=(dy_m, dx_m))
            dland = ndimage.distance_transform_edt(~water, sampling=(dy_m, dx_m))
            signed = np.clip(dsea - dland, -2 * profile_m, 2 * profile_m)
            data = np.where(good, z[sl], 0.0).astype("float64", copy=False)
            wt = good.astype("float64")

            def summed(a):
                return (
                    ndimage.uniform_filter(
                        a.astype("float64", copy=False),
                        size=size,
                        mode="constant",
                        cval=0.0,
                    )
                    * window_area
                )

            n = summed(wt)
            land_n = summed(wt * (~water))
            sea_n = summed(wt * water)
            sx = summed(wt * signed)
            sy = summed(data)
            sxx = summed(wt * signed * signed)
            syy = summed(data * data)
            sxy = summed(data * signed)
            # Regress ORIGINAL-WATER observations independently. Otherwise a
            # steep landward bank can create an excellent global regression
            # even when all seaward 'terrain' is a flat water-surface return.
            # Physical scales are derived from the existing 15 m profile and
            # 0.15 m minimum full-profile rise, without new YAML parameters.
            sea_wt = wt * water
            sea_sx = summed(sea_wt * signed)
            sea_sy = summed(data * water)
            sea_sxx = summed(sea_wt * signed * signed)
            sea_syy = summed(data * data * water)
            sea_sxy = summed(data * signed * water)
            with np.errstate(divide="ignore", invalid="ignore"):
                vx = sxx - sx * sx / np.maximum(n, 1)
                vy = syy - sy * sy / np.maximum(n, 1)
                cov = sxy - sx * sy / np.maximum(n, 1)
                slope = cov / np.maximum(vx, 1e-12)
                explained = (cov * cov) / np.maximum(vx * vy, 1e-12)
                sea_vx = sea_sxx - sea_sx * sea_sx / np.maximum(sea_n, 1)
                sea_vy = sea_syy - sea_sy * sea_sy / np.maximum(sea_n, 1)
                sea_cov = sea_sxy - sea_sx * sea_sy / np.maximum(sea_n, 1)
                sea_slope = sea_cov / np.maximum(sea_vx, 1e-12)
                sea_explained = (sea_cov * sea_cov) / np.maximum(sea_vx * sea_vy, 1e-12)
            good_grade = (
                (land_n >= 5)
                & (sea_n >= 5)
                & (n >= 0.65 * window_area)
                & (vx > 1e-8)
                & (vy > 1e-8)
                & (slope < 0)
                & ((-slope) * 2 * profile_m >= minimum_rise_m)
                & (explained >= 0.80)
            )
            # Support on the seaward side must not be entirely supplied by a
            # planar/flat sea surface. Use the observed source pixels only;
            # no synthetic beach elevations, vertical datum, or image prior.
            seaward_grade = (
                (sea_n >= 5)
                & (sea_vx > 1e-8)
                & (sea_vy > 1e-8)
                & (sea_slope < 0)
                & ((-sea_slope) * profile_m >= minimum_rise_m / 3)
                & (sea_explained >= 0.60)
            )
            cy = slice(first - top, last - top)
            cx = slice(first_x - left, last_x - left)
            base = candidate[first:last, first_x:last_x] & good_grade[cy, cx]
            support = base & seaward_grade[cy, cx]
            result[first:last, first_x:last_x] = support
            if unconfirmed is not None:
                unconfirmed[first:last, first_x:last_x] = base & ~seaward_grade[cy, cx]
    return (result, unconfirmed) if return_unconfirmed else result


def write_decision_support(
    path,
    *,
    land,
    water,
    withheld,
    transform,
    crs,
    beach_grade=None,
    beach_grade_seaward_unconfirmed=None,
):
    """Publish explicit decisions, not the bounding rectangle of observations.

    Unchanged baseline and missing observations cast no scientific vote.
    Withheld geometry remains separate from positive water rejection evidence.
    """
    if beach_grade is None:
        beach_grade = np.zeros_like(land, dtype=bool)
    if beach_grade_seaward_unconfirmed is None:
        beach_grade_seaward_unconfirmed = np.zeros_like(land, dtype=bool)
    if np.any((beach_grade | beach_grade_seaward_unconfirmed) & ~land):
        raise ValueError(
            "Beach-grade support must be a subset of explicit source land decisions"
        )
    if np.any(beach_grade & beach_grade_seaward_unconfirmed):
        raise ValueError("Supported and unconfirmed beach-grade cells must be disjoint")
    for name, cells in (
        ("land", land),
        ("water", water),
        ("withheld", withheld),
        ("beach_grade", beach_grade),
        ("beach_grade_seaward_unconfirmed", beach_grade_seaward_unconfirmed),
    ):
        geom = _cell_polygons(cells, transform)
        gpd.GeoDataFrame(
            {"decision": [name]}, geometry=[polygon_water(geom)], crs=crs
        ).to_file(path, layer=name, driver="GPKG")


def classify_full(
    raw, baseline, restoration, review, outdir, *, shore_level_m=0.0, threshold_m=0.01
):
    """Production full branch without computing an adaptive comparison."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(raw) as ds:
        z = ds.read(1)
        valid = (ds.read_masks(1) > 0) & np.isfinite(z)
        profile, transform, crs = ds.profile.copy(), ds.transform, ds.crs
        dx, dy = pixel_metres(transform, crs, ds.bounds)
    baseline_water = mask_on_grid(baseline, z.shape, transform, crs)
    source_restored_water = mask_on_grid(restoration, z.shape, transform, crs)
    unexpected_restoration_water = source_restored_water & ~baseline_water
    # Restoring land can only subtract from starting OSM water.  GEOS
    # overlay/repair plus rasterization can nevertheless classify some
    # restored-vector cell centers outside starting water.  Those cells have
    # no permissible scientific restoration interpretation.  Constrain the
    # classification to baseline water, and additionally intersect the
    # *vector* with baseline water during publication below.  Record every
    # corrected center rather than silently accepting a changed domain.
    restored_water = source_restored_water & baseline_water
    if unexpected_restoration_water.any():
        print(
            "[restoration-subset] Reclassified "
            f"{int(unexpected_restoration_water.sum())} out-of-baseline water "
            "cell centers as original land; source restoration is unmodified.",
            flush=True,
        )
    proposals = baseline_water & ~restored_water
    audit = initial_audit = None
    reinstated = 0
    if review:
        initial_audit, conflicts = screen_decision_audit(
            review,
            proposals & valid,
            transform,
            crs,
            return_conflicts=True,
        )
        reinstated = int(conflicts.sum())
        proposals &= ~conflicts
        if reinstated:
            print(
                "[withheld-review] Withheld "
                f"{reinstated} conflicting proposal cells before connectivity; "
                "field inspection required.",
                flush=True,
            )
    _, added_land, reject = connected_classification(
        z,
        valid,
        baseline_water,
        proposals,
        shore_level_m=shore_level_m,
        threshold_m=threshold_m,
    )
    restored_water = baseline_water & ~added_land
    if review:
        # No withheld center may become land or provide a bridge to new land.
        if np.any(conflicts & (baseline_water & ~restored_water & valid)):
            raise ValueError("Withheld review reconciliation is incomplete")
    check = write_final_water(
        out / "refined_water.gpkg",
        restoration,
        reject,
        restored_water | reject,
        transform,
        crs,
        baseline_water_path=baseline,
    )
    if review:
        saved_water = mask_on_grid(
            out / "refined_water.gpkg",
            z.shape,
            transform,
            crs,
        )
        audit = screen_decision_audit(
            review,
            baseline_water & ~saved_water & valid,
            transform,
            crs,
        )
        if not audit["passed"]:
            raise ValueError("Final saved coastline promoted withheld evidence")
        audit["reinstated_unique_water_cells"] = reinstated
        audit["requires_field_review"] = bool(reinstated)
        audit["rule"] = (
            "conflicting withheld cell centers excluded before connectivity and remain baseline water"
        )
    write(out / "water_additions.tif", reject, profile, "uint8", 255)
    saved_water = mask_on_grid(out / "refined_water.gpkg", z.shape, transform, crs)
    explicit_water = reject & valid
    withheld = np.zeros_like(valid)
    if review:
        layers = set(pyogrio.list_layers(review)[:, 0].tolist())
        if "rejected_hydroflat_transition" in layers:
            explicit_water |= (
                mask_on_grid(
                    review, z.shape, transform, crs, "rejected_hydroflat_transition"
                )
                & valid
                & baseline_water
            )
        for layer in (
            "unsupported_positive_review",
            "nearshore_transition_review",
            "detached_review",
        ):
            if layer in layers:
                withheld |= (
                    mask_on_grid(review, z.shape, transform, crs, layer)
                    & valid
                    & baseline_water
                )
    restored_land = valid & baseline_water & ~saved_water
    beach_grade, beach_grade_seaward_unconfirmed = beach_grade_evidence(
        z,
        valid,
        baseline_water,
        restored_land,
        dx_m=dx,
        dy_m=dy,
        return_unconfirmed=True,
    )
    write_decision_support(
        out / "decision_support.gpkg",
        land=restored_land,
        water=explicit_water & saved_water,
        withheld=withheld & saved_water,
        beach_grade=beach_grade,
        beach_grade_seaward_unconfirmed=beach_grade_seaward_unconfirmed,
        transform=transform,
        crs=crs,
    )
    receipt = {
        "status": "PASS",
        "raw_valid_cells": int(valid.sum()),
        "shore_level_m": float(shore_level_m),
        "threshold_m": float(threshold_m),
        "classification_rule": "signed_connected_coherent_v2",
        "shared_proposal_screen": "3x3 same-sign fraction >= 0.5 among valid observations; class vetoes before connectivity",
        "class_specific_screen": "positive-water artifact and local-relief vetoes remain land-proposal safeguards; flat submerged terrain allowed",
        "connectivity": "four-connected valid same-sign observations seeded by the original class",
        "near_zero_or_invalid_cells_changed": int(
            np.count_nonzero(
                (saved_water != baseline_water)
                & (
                    ~valid
                    | (
                        (z >= shore_level_m - threshold_m)
                        & (z <= shore_level_m + threshold_m)
                    )
                )
            )
        ),
        "restored_cells": int((baseline_water & ~restored_water & valid).sum()),
        "restoration_outside_baseline_input_cells": int(
            unexpected_restoration_water.sum()
        ),
        "restoration_subset_enforced": True,
        "restoration_subset_field_review_required": bool(
            unexpected_restoration_water.any()
        ),
        "restoration_subset_note": (
            "Restoration-only water is geometrically clipped to original water. "
            "Its raster-cell membership is intersected with original water before "
            "signed-elevation land-to-water classification.  Nonzero input corrections "
            "require field review; they are not new scientific water claims."
        ),
        "water_added_cells": int(reject.sum()),
        "screen_audit": audit,
        "screen_audit_initial": initial_audit,
        "withheld_review_reinstated_water_cells": reinstated,
        "withheld_review_field_inspection_required": bool(reinstated),
        "vector_check": check,
        "decision_support": "decision_support.gpkg",
        "decision_support_policy": "explicit valid-cell decisions; unchanged or missing baseline is unknown",
        "beach_grade_supported_cells": int(beach_grade.sum()),
        "beach_grade_seaward_unconfirmed_cells": int(
            beach_grade_seaward_unconfirmed.sum()
        ),
        "beach_grade_policy": (
            "15 m observed full cross-shore grade with >=0.80 explained variance; "
            "independent seaward-only grade must show >=0.05 m shoreward rise "
            "across 15 m with >=0.60 explained variance. This corroborates "
            "prior land decisions only; unconfirmed pixels cannot release "
            "narrow regional ribbons. No tidal/vertical datum inference."
        ),
        "science_note": (
            "Signed class additions require field validation; marine connectivity is a "
            "separate geometric operation."
        ),
    }
    (out / "classification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
