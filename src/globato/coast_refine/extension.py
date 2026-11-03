"""Optional interpolation-only coastal allowance from existing elevation evidence.

No new observations, source masks, scientific classifications or streams. Small
overlapping raster windows bound distance-transform memory to the coast. Products
are cached separately from the sealed scientific/source-retention preparation.
"""

from pathlib import Path
import json
import math
import warnings
import tempfile
from time import perf_counter
from contextlib import ExitStack

from filelock import FileLock
import geopandas as gpd
import numpy as np
from pyproj import CRS
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, bounds, transform as window_transform
from scipy import ndimage
import shapely
from shapely.affinity import affine_transform
from shapely.ops import linemerge

from ._io import digest, identity, json_write, read_polygon, write_polygon
from .topology import verify_partition
from .shore_profile import SMOOTH_EVIDENCE_SEARCH_M


def settings(distance=0, minimum_z=0.25, method="fixed", slope_window_m=30):
    distance, minimum_z = float(distance), float(minimum_z)
    if not np.isfinite(distance) or distance < 0:
        raise ValueError("barrier_extension_m must be finite and nonnegative")
    if not np.isfinite(minimum_z) or minimum_z < 0:
        raise ValueError("barrier_extension_min_z must be finite and nonnegative")
    method = str(method).lower()
    slope_window_m = float(slope_window_m)
    if method not in ("fixed", "slope", "slope_smooth"):
        raise ValueError(
            "barrier_extension_method must be fixed, slope or slope_smooth"
        )
    if not np.isfinite(slope_window_m) or slope_window_m <= 0:
        raise ValueError("barrier_slope_window_m must be finite and positive")
    return {
        "barrier_extension_m": distance,
        "barrier_extension_min_z": minimum_z,
        "barrier_extension_method": method,
        "barrier_slope_window_m": slope_window_m,
    }


def _stack_signature(path):
    """Metadata identity for windowed reads, without hashing an entire stack."""
    from globato.hooks.sinks.multi_stack import MULTISTACK_BANDS

    path = Path(path).resolve(strict=True)
    with rasterio.open(path) as src:
        if tuple(src.descriptions) != MULTISTACK_BANDS:
            raise ValueError(
                "Slope support requires a finalized, unfilled MultiStack; not raw FusionState or a DEM"
            )
        files = list(src.files)
    signature = []
    for item in sorted(files):
        source = Path(item).resolve(strict=True)
        stat = source.stat()
        signature.append(
            dict(
                path=str(source),
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                ctime_ns=stat.st_ctime_ns,
                inode=stat.st_ino,
            )
        )
    return dict(path=str(path), files=signature)


def _spacing(crs, transform, window):
    """Conservative metre spacings; geographic longitude/latitude are unequal."""
    crs = CRS.from_user_input(crs)
    crs = crs.sub_crs_list[0] if crs.is_compound else crs
    if crs.is_geographic:
        west, south, east, north = bounds(window, transform)
        geod = crs.get_geod()
        x = (west + east) / 2
        ys = np.array([south, (south + north) / 2, north])
        dx = max(abs(geod.inv(x, y, x + transform.a, y)[2]) for y in ys)
        dy = max(abs(geod.inv(x, y, x, y + transform.e)[2]) for y in ys)
    else:
        dx = abs(transform.a) * crs.axis_info[0].unit_conversion_factor
        dy = abs(transform.e) * crs.axis_info[1].unit_conversion_factor
    if not np.isfinite([dx, dy]).all() or min(dx, dy) <= 0:
        raise ValueError("Cannot determine metre spacing for coastal extension")
    return dy, dx


def _screen(z, valid, land, ocean, minimum_z):
    """Require an observed positive edge and a majority in its 3x3 land strip.

    An immediately adjacent observed near-zero/negative water cell already
    supports a sharp or bracketed interface: do not extend that segment.
    NoData never counts as zero, terrain support or a submerged observation.
    """
    cross = ndimage.generate_binary_structure(2, 1)
    accepted = valid & land
    positive = accepted & (z > minimum_z)
    count = ndimage.convolve(
        accepted.astype("u1"), np.ones((3, 3), "u1"), mode="constant"
    )
    above = ndimage.convolve(
        positive.astype("u1"), np.ones((3, 3), "u1"), mode="constant"
    )
    bracketed = ndimage.binary_dilation(
        ocean & valid & (z <= minimum_z), structure=cross
    )
    return (
        positive
        & ndimage.binary_dilation(ocean, structure=cross)
        & (count >= 3)
        & (above * 2 > count)
        & ~bracketed
    )


def _grow(seeds, ocean, distance, spacing):
    if not np.any(seeds):
        return np.zeros_like(seeds)
    dist = ndimage.distance_transform_edt(~seeds, sampling=spacing)
    candidates = ocean & (dist <= distance)
    starts = candidates & ndimage.binary_dilation(seeds)
    return ndimage.binary_propagation(starts, mask=candidates)


def _polygons(mask, transform):
    return [
        shapely.geometry.shape(g)
        for g, value in shapes(mask.astype("u1"), mask=mask, transform=transform)
        if value == 1
    ]


def build(
    artifact,
    distance,
    minimum_z=0.25,
    *,
    block_size=512,
    method="fixed",
    slope_window_m=30,
    support_stack=None,
    support_min_weight=None,
    use_stack_for_land=False,
):
    """Build/reuse a sealed derived barrier; leave the base artifact untouched."""
    policy = settings(distance, minimum_z, method, slope_window_m)
    distance = policy["barrier_extension_m"]
    minimum_z = policy["barrier_extension_min_z"]
    method = policy["barrier_extension_method"]
    slope_window_m = policy["barrier_slope_window_m"]
    if not policy["barrier_extension_m"]:
        return None
    if use_stack_for_land and method not in ("slope", "slope_smooth"):
        raise ValueError(
            "Complete-stack zero estimation requires slope or slope_smooth"
        )
    support = None
    if method in ("slope", "slope_smooth"):
        if support_stack is None or support_min_weight is None:
            raise ValueError(
                "Slope estimation runs after stack_combine and requires its unfilled MultiStack and support weight"
            )
        support = _stack_signature(support_stack)
        support_min_weight = float(support_min_weight)
        if not np.isfinite(support_min_weight) or support_min_weight <= 0:
            raise ValueError("Slope support weight must be positive")
    inputs = {
        "base_manifest_sha256": artifact.manifest_hash,
        "base_preparation": artifact.preparation_id,
        "policy": policy,
        "implementation_sha256": digest(__file__),
        "method": "positive_ocean_edge_windowed_distance_v3",
    }
    if support is not None:
        from . import shore_profile
        import scipy

        inputs.update(
            support_stack=support,
            support_min_weight=support_min_weight,
            slope_code_sha256=digest(shore_profile.__file__),
            scipy_version=scipy.__version__,
        )
    if use_stack_for_land:
        inputs.update(
            land_evidence="complete_accepted_unfilled_stack",
            support_stack_sha256=digest(support_stack),
        )
    key = identity(inputs)
    parent = artifact.root.parent / "interpolation_extensions"
    parent.mkdir(exist_ok=True)
    target = parent / key
    with FileLock(str(target) + ".lock"):
        if target.exists():
            validate(artifact, target / "extension.json")
            return target / "extension.json"
        started = perf_counter()
        grid = json.loads((artifact.root / "application_grid.json").read_text())
        transform = rasterio.Affine(*grid["transform"])

        def native(geom):
            return affine_transform(
                geom,
                [
                    transform.a,
                    transform.b,
                    transform.d,
                    transform.e,
                    transform.c,
                    transform.f,
                ],
            )

        crs = grid["crs"]
        if transform.b or transform.d:
            raise ValueError("Coastal extension requires a north-up grid")
        frame = gpd.read_file(
            artifact.products["refined_topology"],
            layer=artifact.layers["refined_topology"],
        )
        land = read_polygon(
            artifact.products["refined_land"], crs=crs, require_native=True
        )
        # Explicit ocean only: connected rivers/lakes are not offshore allowance.
        ocean = shapely.union_all(frame.loc[frame["class"] == "ocean"].geometry)
        coast = land.boundary.intersection(ocean.boundary)
        parts = []
        edge_records = []
        evidence_records = []
        counts = dict(
            windows=0, sampled_edge_cells=0, extension_cells=0, max_window_cells=0
        )
        profile_counts = dict(
            bracketed=0,
            one_sided=0,
            unsupported=0,
            beyond_limit=0,
            alongshore_interpolated=0,
            alongshore_tapered=0,
        )
        requested_limit_counts = dict(
            supported_above_requested_limit=0,
            supported_at_or_below_requested_limit=0,
        )
        safety = dict(
            tapered_windows=0,
            rejected_windows=0,
            minimum_width_fraction=1.0,
            global_fallback=False,
            reason=None,
            restored_water_cells=0,
            restored_land_passage_cells=0,
            smoothed_profiles=0,
            maximum_width_adjustment_m=0.0,
            topology_attempts=[],
            locally_repaired_windows=0,
            local_tapered_windows=0,
            local_taper_seed_cells=0,
            local_minimum_width_fraction=1.0,
            whole_window_fallback_windows=0,
        )
        if not coast.is_empty:
            # Minimum possible spacing sets an overlap that cannot omit a seed.
            entire = Window(0, 0, grid["width"], grid["height"])
            # One conservative metric makes decisions independent of window
            # layout, including on unequal geographic longitude/latitude cells.
            spacing = _spacing(crs, transform, entire)
            hcrs = CRS.from_user_input(crs)
            hcrs = hcrs.sub_crs_list[0] if hcrs.is_compound else hcrs
            if hcrs.is_geographic:
                _, south, _, north = bounds(entire, transform)
                geod = hcrs.get_geod()
                x = transform.c
                min_pixel = min(
                    abs(geod.inv(x, y, x + transform.a, y)[2]) for y in (south, north)
                )
                min_pixel = min(
                    min_pixel, abs(geod.inv(x, south, x, south + transform.e)[2])
                )
            else:
                min_pixel = min(_spacing(crs, transform, entire))
            halo_distance = (
                distance
                if method == "fixed"
                else (
                    (
                        SMOOTH_EVIDENCE_SEARCH_M
                        if method == "slope_smooth"
                        else 3 * distance
                    )
                    + 2 * slope_window_m
                )
            )
            halo = int(math.ceil(halo_distance / min_pixel)) + 3
            evidence = (
                Path(support_stack)
                if use_stack_for_land
                else artifact.root / "unmasked_evidence.tif"
            )
            with (
                ExitStack() as handles,
            ):
                src = handles.enter_context(rasterio.open(evidence))
                aligned = handles.enter_context(
                    WarpedVRT(
                        src,
                        crs=crs,
                        transform=transform,
                        width=grid["width"],
                        height=grid["height"],
                        resampling=rasterio.enums.Resampling.nearest,
                        nodata=src.nodata,
                    )
                )
                bathy = None
                if support is not None:
                    stack = handles.enter_context(rasterio.open(support_stack))
                    if (
                        stack.crs != src.crs
                        or not np.isclose(stack.transform.a, transform.a, rtol=1e-9)
                        or not np.isclose(stack.transform.e, transform.e, rtol=1e-9)
                    ):
                        raise ValueError(
                            "Slope support stack must share the authority CRS and native cell spacing"
                        )
                    offset = (~stack.transform) * (transform.c, transform.f)
                    if any(abs(v - round(v)) > 1e-6 for v in offset):
                        raise ValueError(
                            "Slope support stack has a fractional-cell offset"
                        )
                    bathy = handles.enter_context(
                        WarpedVRT(
                            stack,
                            crs=crs,
                            transform=transform,
                            width=grid["width"],
                            height=grid["height"],
                            resampling=rasterio.enums.Resampling.nearest,
                            nodata=stack.nodata,
                        )
                    )
                # Only grid alignment/blank fringe is allowed; never resample evidence.
                if (
                    src.crs != aligned.crs
                    or src.transform.a != transform.a
                    or src.transform.e != transform.e
                ):
                    raise ValueError(
                        "Extension evidence must use the native application lattice"
                    )
                offset = (~src.transform) * (transform.c, transform.f)
                if any(abs(v - round(v)) > 1e-6 for v in offset):
                    raise ValueError("Extension evidence has a fractional-cell offset")
                for row in range(0, grid["height"], block_size):
                    for col in range(0, grid["width"], block_size):
                        core = Window(
                            col,
                            row,
                            min(block_size, grid["width"] - col),
                            min(block_size, grid["height"] - row),
                        )
                        c0, r0 = max(0, col - halo), max(0, row - halo)
                        c1, r1 = (
                            min(grid["width"], col + core.width + halo),
                            min(grid["height"], row + core.height + halo),
                        )
                        win = Window(c0, r0, c1 - c0, r1 - r0)
                        extent = shapely.box(*bounds(win, transform))
                        if not coast.intersects(extent):
                            continue
                        wt = window_transform(win, transform)
                        shape = (int(win.height), int(win.width))

                        def burn(geom):
                            clipped = geom.intersection(extent)
                            return (
                                rasterize(
                                    [(clipped, 1)],
                                    out_shape=shape,
                                    transform=wt,
                                    dtype="uint8",
                                ).astype(bool)
                                if not clipped.is_empty
                                else np.zeros(shape, bool)
                            )

                        land_cells, ocean_cells = burn(land), burn(ocean)
                        z = aligned.read(1, window=win, masked=True)
                        valid = ~np.ma.getmaskarray(z) & np.isfinite(z.data)
                        data = None
                        observed = None
                        if support is not None:
                            data = bathy.read([1, 2, 3], window=win, masked=True)
                            observed = (
                                ~np.ma.getmaskarray(data).any(axis=0)
                                & np.isfinite(data.data).all(axis=0)
                                & (data.data[1] > 0)
                                & (data.data[2] >= support_min_weight)
                            )
                        if use_stack_for_land:
                            # The same accepted all-module measurements support both
                            # sides. No authority-only terrain and no filled DEM.
                            z = data[0]
                            valid = observed
                        seeds = _screen(
                            z.data,
                            valid
                            if method == "fixed" or use_stack_for_land
                            else valid & land_cells,
                            land_cells,
                            ocean_cells,
                            minimum_z,
                        )
                        # Known nonpositive ocean evidence cannot become inferred land.
                        available = (
                            ocean_cells & ~(valid & (z.data <= minimum_z))
                            if method == "fixed"
                            else ocean_cells.copy()
                        )
                        radius = None
                        if method == "fixed":
                            extension = _grow(seeds, available, distance, spacing)
                        else:
                            from .shore_profile import (
                                estimate,
                                grow,
                                topology_safe_smooth_growth,
                            )

                            # Separate evidence retrieval from the user's maximum
                            # interpolation allowance. 40/60/80 m runs infer the
                            # SAME raw zero distances and profile classifications.
                            # Only smooth_growth clips to the requested width.
                            evidence_limit = (
                                SMOOTH_EVIDENCE_SEARCH_M
                                if method == "slope_smooth"
                                else distance
                            )
                            radius, support_type, _ = estimate(
                                z.data,
                                valid,
                                land_cells,
                                ocean_cells,
                                seeds,
                                spacing,
                                evidence_limit,
                                slope_window_m,
                                bathy_z=data.data[0],
                                bathy_valid=observed,
                                search_distance_m=(
                                    SMOOTH_EVIDENCE_SEARCH_M
                                    if method == "slope_smooth"
                                    else None
                                ),
                            )
                            available &= ~(observed & (data.data[0] <= 0))
                            if method == "slope_smooth":
                                from .shore_profile import (
                                    interpolate_alongshore,
                                    regularize_alongshore_width,
                                )

                                # Convert this window's exact refined coastline
                                # into metre-scaled pixel coordinates. Alongshore
                                # arclength, not 2D proximity, connects evidence.
                                dy, dx = spacing
                                metric_coast = affine_transform(
                                    coast.intersection(extent),
                                    [
                                        dx / wt.a,
                                        0,
                                        0,
                                        dy / wt.e,
                                        -dx * wt.c / wt.a,
                                        -dy * wt.f / wt.e,
                                    ],
                                )

                                def extract_lines(g):
                                    if g.geom_type in ("LineString", "LinearRing"):
                                        return (
                                            [shapely.LineString(g)]
                                            if not g.is_empty
                                            else []
                                        )
                                    return [
                                        line
                                        for part in shapely.get_parts(g)
                                        for line in extract_lines(part)
                                    ]

                                lines = extract_lines(metric_coast)
                                if len(lines) > 1:
                                    merged = linemerge(shapely.MultiLineString(lines))
                                    lines = [
                                        g
                                        for g in shapely.get_parts(merged)
                                        if g.geom_type == "LineString"
                                        and not g.is_empty
                                    ]
                                radius, support_type, alongshore = (
                                    interpolate_alongshore(
                                        seeds,
                                        radius,
                                        support_type,
                                        lines,
                                        spacing,
                                        smoothing_m=slope_window_m,
                                    )
                                )
                                # Evidence classes and raw estimates remain intact;
                                # only the downstream, interpolation-only width
                                # field is smoothed along each connected coast.
                                growth_radius, regularization = (
                                    regularize_alongshore_width(
                                        seeds,
                                        radius,
                                        support_type,
                                        lines,
                                        spacing,
                                        smoothing_m=slope_window_m,
                                    )
                                )
                                safety["smoothed_profiles"] += regularization[
                                    "smoothed_profiles"
                                ]
                                safety["maximum_width_adjustment_m"] = max(
                                    safety["maximum_width_adjustment_m"],
                                    regularization["max_adjustment_m"],
                                )
                                local_safety = {}
                                extension, fraction = topology_safe_smooth_growth(
                                    seeds,
                                    available,
                                    ocean_cells,
                                    land_cells,
                                    growth_radius,
                                    spacing,
                                    smoothing_m=slope_window_m,
                                    maximum_m=distance,
                                    diagnostics=local_safety,
                                    alongshore_smoothed=True,
                                )
                                safety["restored_land_passage_cells"] += (
                                    local_safety.get("restored_land_passage_cells", 0)
                                )
                                safety["topology_attempts"].append(
                                    {
                                        "window_row": row,
                                        "window_col": col,
                                        "attempts": local_safety.get("attempts", []),
                                    }
                                )
                                safety["restored_water_cells"] += local_safety.get(
                                    "restored_water_cells", 0
                                )
                                safety["locally_repaired_windows"] += int(
                                    local_safety.get("local_corridor_repair", False)
                                )
                                safety["local_tapered_windows"] += int(
                                    local_safety.get("local_taper_applied", False)
                                )
                                safety["local_taper_seed_cells"] += local_safety.get(
                                    "local_taper_seed_cells", 0
                                )
                                safety["local_minimum_width_fraction"] = min(
                                    safety["local_minimum_width_fraction"],
                                    local_safety.get(
                                        "local_minimum_width_fraction", 1.0
                                    ),
                                )
                                safety["whole_window_fallback_windows"] += int(
                                    local_safety.get("whole_window_fallback", False)
                                )
                                if fraction == 0 and np.any(radius[seeds] > 0):
                                    safety["rejected_windows"] += 1
                                elif fraction < 1:
                                    safety["tapered_windows"] += 1
                                safety["minimum_width_fraction"] = min(
                                    safety["minimum_width_fraction"], fraction
                                )
                            else:
                                extension = grow(seeds, available, radius, spacing)
                        sl = np.s_[
                            row - r0 : row - r0 + int(core.height),
                            col - c0 : col - c0 + int(core.width),
                        ]
                        counts["windows"] += 1
                        if radius is not None:
                            for support_name, code in [
                                ("unsupported", 0),
                                ("bracketed", 1),
                                ("one_sided", 2),
                                ("beyond_limit", 3),
                                ("alongshore_interpolated", 4),
                                ("alongshore_tapered", 5),
                            ]:
                                profile_counts[support_name] += int(
                                    np.count_nonzero(
                                        seeds[sl] & (support_type[sl] == code)
                                    )
                                )
                        if method == "slope_smooth":
                            valid_radius = radius[sl][seeds[sl]]
                            requested_limit_counts[
                                "supported_above_requested_limit"
                            ] += int(np.count_nonzero(valid_radius > distance))
                            requested_limit_counts[
                                "supported_at_or_below_requested_limit"
                            ] += int(
                                np.count_nonzero(
                                    (valid_radius > 0) & (valid_radius <= distance)
                                )
                            )
                        if method == "slope_smooth":
                            for k, title in (
                                (1, "bracketed"),
                                (2, "one_sided"),
                                (3, "beyond_evidence_horizon"),
                                (4, "alongshore_interpolated"),
                                (5, "alongshore_tapered"),
                                (0, "unsupported"),
                            ):
                                sample_mask = seeds[sl] & (support_type[sl] == k)
                                for shape_px in _polygons(
                                    sample_mask, rasterio.Affine.translation(col, row)
                                ):
                                    evidence_records.append(
                                        dict(
                                            geometry=native(shape_px),
                                            evidence=title,
                                            method_code=k,
                                            cells=int(np.count_nonzero(sample_mask)),
                                        )
                                    )
                        counts["max_window_cells"] = max(
                            counts["max_window_cells"], int(win.width * win.height)
                        )
                        counts["sampled_edge_cells"] += int(np.count_nonzero(seeds[sl]))
                        counts["extension_cells"] += int(
                            np.count_nonzero(extension[sl])
                        )
                        # Union exact integer cell edges across windows before
                        # applying the global transform once. Per-window map
                        # origins otherwise leave ULP-wide coastal slivers.
                        ct = rasterio.Affine.translation(col, row)
                        parts.extend(_polygons(extension[sl], ct))
                        local_edges = _polygons(seeds[sl], ct)
                        if local_edges:
                            values = z.data[sl][seeds[sl]]
                            edge_records.append(
                                {
                                    "geometry": native(shapely.union_all(local_edges)),
                                    "min_z": float(values.min()),
                                    "median_z": float(np.median(values)),
                                    "max_z": float(values.max()),
                                    "sample_cells": len(values),
                                }
                            )
                            if radius is not None:
                                proposals = radius[sl][seeds[sl]]
                                proposals = proposals[proposals > 0]
                                edge_records[-1].update(
                                    estimated_profiles=int(proposals.size),
                                    zero_distance_m=float(np.median(proposals))
                                    if proposals.size
                                    else None,
                                )
        added = native(shapely.union_all(parts)) if parts else shapely.Polygon()
        if support is not None and _stack_signature(support_stack) != support:
            raise ValueError("Support stack changed during zero-line estimation")
        water_before = shapely.union_all(
            frame.loc[
                ~frame["class"].isin(["land", "island", "reef", "breakwater"])
            ].geometry
        )

        def cell_components(geom):
            # Count actual grid-scale components, not vector-overlay roundoff.
            return sum(
                p.area >= abs(transform.a * transform.e) * 0.5
                for p in shapely.get_parts(geom)
            )

        def topology_conflict(candidate):
            if len(shapely.get_parts(land.union(candidate))) < len(
                shapely.get_parts(land)
            ):
                return "Extension would merge separate land bodies"
            if cell_components(water_before.difference(candidate)) > cell_components(
                water_before
            ):
                return "Extension would split a water body"
            return None

        conflict = topology_conflict(added)
        # A derived interpolation allowance is optional. A topologically invalid
        # extension must NEVER prevent the original sealed Coast Refine barrier
        # from reaching the normal DEM interpolation path. Keep the fixed/slope
        # algorithms unchanged when their geometry passes; reject the entire
        # invalid candidate (rather than slicing out arbitrary grid cells).
        rejection_reason = None
        if conflict is not None:
            rejection_reason = (
                "land_merger"
                if "merge separate land bodies" in conflict
                else "water_split"
                if "split a water body" in conflict
                else "topology_conflict"
            )
            safety.update(global_fallback=True, reason=conflict)
            warnings.warn(
                f"{method} interpolation extension rejected ({rejection_reason}): "
                + conflict
                + "; using original refined coastline",
                RuntimeWarning,
                stacklevel=2,
            )
            added = shapely.Polygon()
            counts["extension_cells"] = 0
        extension_status = (
            "rejected"
            if rejection_reason is not None
            else "accepted"
            if not added.is_empty
            else "empty"
        )
        extended_land = land.union(added)
        rows = []
        for _, record in frame.iterrows():
            geom = record.geometry.difference(added)
            if not geom.is_empty:
                rows.append({"class": record["class"], "geometry": geom})
        if not added.is_empty:
            rows.append({"class": "land", "geometry": added})
        result = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
        verify_partition(result, artifact._native_coverage, crs=crs)
        with tempfile.TemporaryDirectory(prefix="extension_", dir=parent) as temporary:
            pending = Path(temporary)
            topology = pending / "interpolation_topology.gpkg"
            result.to_file(topology, layer="topology", driver="GPKG")
            verify_partition(
                gpd.read_file(topology), artifact._native_coverage, crs=crs
            )
            write_polygon(pending / "extension_area.gpkg", added, crs=crs)
            if edge_records:
                gpd.GeoDataFrame(edge_records, crs=crs).to_file(
                    pending / "eligible_edge.gpkg", layer="mask", driver="GPKG"
                )
            else:
                # No qualifying shoreline cells: publish a valid, explicitly
                # empty mask. The prior `edges` reference was never defined
                # and prevented no-extension / near-zero cases from finishing.
                write_polygon(
                    pending / "eligible_edge.gpkg", shapely.Polygon(), crs=crs
                )
            if method == "slope_smooth":
                if evidence_records:
                    gpd.GeoDataFrame(evidence_records, crs=crs).to_file(
                        pending / "shoreline_evidence.gpkg",
                        layer="evidence",
                        driver="GPKG",
                    )
                else:
                    gpd.GeoDataFrame(
                        {"evidence": [], "method_code": [], "cells": []},
                        geometry=gpd.GeoSeries([], crs=crs),
                        crs=crs,
                    ).to_file(
                        pending / "shoreline_evidence.gpkg",
                        layer="evidence",
                        driver="GPKG",
                    )
            from .prepare import _write_lines

            water = shapely.union_all(
                result.loc[
                    ~result["class"].isin(["land", "island", "reef", "breakwater"])
                ].geometry
            )
            _write_lines(
                pending / "interpolation_coastline.gpkg",
                extended_land.boundary.intersection(water.boundary),
                role="inferred_interpolation_limit",
                crs=crs,
            )
            record = {
                "schema_version": 1,
                "inputs": inputs,
                "identity": key,
                "base_manifest_path": str(artifact.path),
                "crs": crs,
                "products": {p.name: digest(p) for p in pending.glob("*.gpkg")},
                "counts": counts,
                "extension_status": extension_status,
                "rejection_reason": rejection_reason,
                "rejection_detail": conflict,
                "elapsed_seconds": perf_counter() - started,
                "meaning": "Interpolation allowance only; not an observed zero contour or source mask",
                "sampling": (
                    "complete accepted unfilled MultiStack on both sides; positive land edge, >=3 valid land samples; adjacent observed water veto"
                    if use_stack_for_land
                    else "accepted land-side native cells; 3x3 strict positive majority, >=3 valid land samples; adjacent observed water veto"
                ),
                "distance": "cell-centre distance with conservative whole-grid metre spacing; connected explicit ocean only",
                "smoothing": "On-coast arclength interpolation (180 m maximum gap), 40 m end taper, capped supported-width smoothing; v55 local conflict-driven width taper before conservative whole-window fallback; source masks unchanged"
                if method == "slope_smooth"
                else None,
                "alongshore_gap_limit_m": 180.0 if method == "slope_smooth" else None,
                "alongshore_end_taper_m": 40.0 if method == "slope_smooth" else None,
                "topology_safety": safety
                if method == "slope_smooth" or rejection_reason
                else None,
                "source_masks_changed": False,
                "profile_support": profile_counts
                if method in ("slope", "slope_smooth")
                else None,
                "requested_extension_limit_support": (
                    requested_limit_counts if method == "slope_smooth" else None
                ),
                "profile_evidence_search_m": (
                    SMOOTH_EVIDENCE_SEARCH_M if method == "slope_smooth" else None
                ),
                "profile_evidence_policy": (
                    "Unclipped 0-crossing estimates within a fixed 240 m search horizon; "
                    "requested extension distance limits only the subsequent smooth growth"
                    if method == "slope_smooth"
                    else None
                ),
                "slope_support_note": "Accepted cells from the unfilled finalized MultiStack, using the explicit support weight cutoff; weight is recipe selection, not a calibrated accuracy measure"
                if support
                else None,
            }
            json_write(pending / "extension.json", record)
            pending.rename(target)
        return target / "extension.json"


def validate(artifact, receipt):
    """Verify the derived barrier binding/bytes before interpolation uses it."""
    receipt = Path(receipt).resolve(strict=True)
    record = json.loads(receipt.read_text())
    if (
        record.get("schema_version") != 1
        or identity(record["inputs"]) != record.get("identity")
        or record["inputs"].get("base_manifest_sha256") != artifact.manifest_hash
        or record["inputs"].get("base_preparation") != artifact.preparation_id
    ):
        raise ValueError("Interpolation extension does not match its sealed authority")
    support_hash = record["inputs"].get("support_stack_sha256")
    if (
        support_hash
        and digest(record["inputs"]["support_stack"]["path"]) != support_hash
    ):
        raise ValueError("Zero-estimate support stack changed after publication")
    required = {
        "interpolation_topology.gpkg",
        "interpolation_coastline.gpkg",
        "extension_area.gpkg",
        "eligible_edge.gpkg",
    }
    if record["inputs"]["policy"].get(
        "barrier_extension_method"
    ) == "slope_smooth" and "shoreline_evidence.gpkg" in record.get("products", {}):
        required.add("shoreline_evidence.gpkg")
    if set(record.get("products", {})) != required:
        raise ValueError("Interpolation extension products are incomplete")
    for name, checksum in record["products"].items():
        path = receipt.parent / name
        if path.is_symlink() or digest(path) != checksum:
            raise ValueError("Interpolation extension product checksum mismatch")
    topology = receipt.parent / "interpolation_topology.gpkg"
    frame = gpd.read_file(topology, layer="topology")
    if frame.crs != gpd.GeoSeries([], crs=artifact._coverage_crs).crs:
        raise ValueError("Interpolation extension CRS differs from its authority")
    verify_partition(frame, artifact._native_coverage, crs=artifact._coverage_crs)
    return topology
