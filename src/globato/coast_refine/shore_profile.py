"""Experimental local zero-crossing estimates from unfilled grid profiles.

Only barrier geography is proposed. No measured elevation or source mask is
modified. PCHIP uses adjacent observed profile slopes when both signs bracket
the missing interface; limited one-sided estimates require a descending fit.
"""

import warnings

import numpy as np
from scipy import ndimage
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

# Common survey-evidence search horizon; independent of the extension cap.
SMOOTH_EVIDENCE_SEARCH_M = 240.0

# Conservative, experimental alongshore gap limits (metres). These are not
# statements about physical shoreline stability; the receipt identifies them.
ALONGSHORE_GAP_M = 180.0
ALONGSHORE_END_TAPER_M = 40.0


def _land_fit(t, z, step):
    """Robust short-profile slope and residual, without a flat-slope fallback."""
    good = np.isfinite(z) & (t <= 0)
    if good.sum() < 4 or np.ptp(t[good]) < 3 * step:
        return None
    x, y = t[good], z[good]
    for _ in range(2):
        slope, intercept = np.polyfit(x, y, 1)
        residual = y - (slope * x + intercept)
        mad = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        keep = np.abs(residual) <= max(3 * mad, 1e-6)
        if keep.all() or keep.sum() < 4:
            break
        x, y = x[keep], y[keep]
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    rms = float(np.sqrt(np.mean(residual**2)))
    slope_se = rms / np.sqrt(np.sum((x - x.mean()) ** 2))
    if not np.isfinite(slope) or slope + 2 * slope_se >= -1e-8:
        return None
    return float(slope), rms


def profile_zero(t, land_z, water_z, edge_z, maximum_m, fit_window_m):
    """Return (distance, method, slope, fit RMS) or an unsupported result.

    land_z/water_z are observed cross-shore bin medians, with NaNs for gaps.
    PCHIP remains between the positive and negative endpoints in the missing
    interval. It does not extrapolate a cubic beyond the available data.
    """
    step = float(np.min(np.diff(t)))
    left = (t < 0) & (t >= -fit_window_m) & np.isfinite(land_z)
    fit_t = np.r_[t[left], 0.0]
    fit_z = np.r_[land_z[left], edge_z]
    fit = _land_fit(fit_t, fit_z, step)
    negative = np.flatnonzero(
        (t > 0) & (t <= 3 * maximum_m) & np.isfinite(water_z) & (water_z < 0)
    )
    if negative.size:
        first = int(negative[0])
        right = (
            (t >= t[first])
            & (t <= t[first] + fit_window_m)
            & np.isfinite(water_z)
            & (water_z < 0)
        )
        xs = np.r_[fit_t, t[right]]
        zs = np.r_[fit_z, water_z[right]]
        curve = PchipInterpolator(xs, zs, extrapolate=False)
        distance = float(brentq(curve, 0.0, float(t[first]), xtol=0.01))
        return (distance, "bracketed", fit[0] if fit else None, fit[1] if fit else None)
    if fit:
        return (float(edge_z / -fit[0]), "one_sided", fit[0], fit[1])
    return (None, "unsupported", None, None)


def estimate(
    z,
    valid,
    land,
    ocean,
    seeds,
    spacing,
    maximum_m,
    fit_window_m,
    *,
    bathy_z=None,
    bathy_valid=None,
    search_distance_m=None,
):
    """Sample short raster profiles in bounded batches; no point-cloud files.

    Three parallel sample lines provide an alongshore median in each bin.
    The normal follows a smoothed signed distance field, not an axis-aligned
    image gradient. Ocean sampling stops at an island, bank or domain edge.
    Method codes: 0 unsupported, 1 bracketed, 2 one-sided, 3 beyond limit.
    These are support types,
    not calibrated confidence levels.
    """
    dy, dx = spacing
    signed = ndimage.distance_transform_edt(
        land, sampling=spacing
    ) - ndimage.distance_transform_edt(~land, sampling=spacing)
    signed = ndimage.gaussian_filter(signed, 1)
    nr, nc = np.gradient(-signed, dy, dx)
    length = np.hypot(nr, nc)
    nr = np.divide(nr, length, out=np.zeros_like(nr), where=length > 0)
    nc = np.divide(nc, length, out=np.zeros_like(nc), where=length > 0)
    step = max(spacing)
    before = max(3, int(np.ceil(fit_window_m / step)))
    search_distance = (
        3 * maximum_m if search_distance_m is None else float(search_distance_m)
    )
    if not np.isfinite(search_distance) or search_distance <= 0:
        raise ValueError("search_distance_m must be finite and positive")
    after = max(1, int(np.ceil((search_distance + fit_window_m) / step)))
    t = np.arange(-before, after + 1, dtype="float64") * step
    radius = np.zeros(z.shape, "float32")
    method = np.zeros(z.shape, "uint8")
    stats = dict(bracketed=0, one_sided=0, unsupported=0, beyond_limit=0)
    rows, cols = np.where(seeds)
    if bathy_z is None:
        bathy_z = np.zeros_like(z)
        bathy_valid = np.zeros_like(valid)
    for start in range(0, len(rows), 128):
        rr, cc = rows[start : start + 128], cols[start : start + 128]
        ry, cx = nr[rr, cc], nc[rr, cc]
        cross = np.array([-step, 0.0, step])
        r = (
            rr[:, None, None]
            + ry[:, None, None] * t[None, :, None] / dy
            - cx[:, None, None] * cross[None, None, :] / dy
        )
        c = (
            cc[:, None, None]
            + cx[:, None, None] * t[None, :, None] / dx
            + ry[:, None, None] * cross[None, None, :] / dx
        )
        r, c = np.rint(r).astype("int32"), np.rint(c).astype("int32")
        inside = (r >= 0) & (r < z.shape[0]) & (c >= 0) & (c < z.shape[1])
        r = np.clip(r, 0, z.shape[0] - 1)
        c = np.clip(c, 0, z.shape[1] - 1)
        lvalid = inside & valid[r, c] & land[r, c] & (z[r, c] > 0)
        wvalid = inside & bathy_valid[r, c] & ocean[r, c] & (bathy_z[r, c] < 0)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="All-NaN slice encountered", category=RuntimeWarning
            )
            lz = np.nanmedian(np.where(lvalid, z[r, c], np.nan), axis=2)
            wz = np.nanmedian(np.where(wvalid, bathy_z[r, c], np.nan), axis=2)
        for index, (row, col) in enumerate(zip(rr, cc)):
            # Do not borrow profiles through an inland NoData break.
            left_valid = np.isfinite(lz[index, :before])
            left_valid = np.logical_and.accumulate(left_valid[::-1])[::-1]
            lz[index, :before][~left_valid] = np.nan
            # Geometry barriers stop a normal from sampling a different coast.
            domain = (
                inside[index, before + 1 :, 1]
                & ocean[r[index, before + 1 :, 1], c[index, before + 1 :, 1]]
            )
            path = np.logical_and.accumulate(domain)
            wz[index, before + 1 :][~path] = np.nan
            distance, kind, _, _ = profile_zero(
                t,
                lz[index],
                wz[index],
                float(z[row, col]),
                search_distance / 3,
                fit_window_m,
            )
            if distance is None or not np.isfinite(distance) or distance <= 0:
                stats["unsupported"] += 1
            elif distance > maximum_m:
                # Never pull an estimated crossing artificially inside the cap.
                stats["beyond_limit"] += 1
                method[row, col] = 3
            else:
                radius[row, col] = distance
                method[row, col] = 1 if kind == "bracketed" else 2
                stats[kind] += 1
    return radius, method, stats


def interpolate_alongshore(
    seeds,
    radius,
    support_type,
    coast_lines,
    spacing,
    *,
    maximum_gap_m=ALONGSHORE_GAP_M,
    end_taper_m=ALONGSHORE_END_TAPER_M,
    smoothing_m=30.0,
):
    """Infer unmeasured seed widths along the *same mapped coastline*.

    ``coast_lines`` are shapely lines in local (x,y) **metre** coordinates,
    whose origin is the top-left corner of the window; seed centres are
    located at ((col+.5)*dx, (row+.5)*dy). This is not a planar-distance
    Gaussian: each original shoreline is independently parameterized by
    distance along its line. Opposite sides of an inlet can be close in space
    but cannot share estimates across separate lines or distant arclengths.

    A seed without a local estimate is missing evidence, NOT a zero-width
    measurement. Short bracketed gaps are interpolated. At the ends of a
    supported reach, taper over a bounded distance. Longer gaps have zero
    allowance in their interiors. Existing accepted profile observations stay
    in ``radius`` and are never overwritten. A derived field is returned.
    """
    import shapely
    from shapely.strtree import STRtree

    dy, dx = spacing
    output = radius.copy()
    kind = support_type.copy()
    rows, cols = np.where(seeds)
    if not len(rows) or not coast_lines:
        return output, kind, dict(interpolated=0, tapered=0, unassigned=int(len(rows)))
    points = shapely.points((cols + 0.5) * dx, (rows + 0.5) * dy)
    tree = STRtree(coast_lines)
    assigned = np.asarray(tree.nearest(points), dtype=int)
    d = shapely.distance(points, np.asarray(coast_lines, dtype=object)[assigned])
    # A shoreline cell should lie within a few native cells of its own line;
    # otherwise do not infer from an unrelated nearby geometry.
    assigned[d > 2.5 * np.hypot(dx, dy)] = -1
    derived = dict(
        interpolated=0, tapered=0, unassigned=int(np.count_nonzero(assigned < 0))
    )
    for line_index, line in enumerate(coast_lines):
        indices = np.flatnonzero(assigned == line_index)
        if not len(indices):
            continue
        along = shapely.line_locate_point(line, points[indices])
        order = np.argsort(along, kind="stable")
        indices, along = indices[order], np.asarray(along)[order]
        original = radius[rows[indices], cols[indices]].astype(float)
        original_kind = support_type[rows[indices], cols[indices]]
        good = (original > 0) & np.isin(original_kind, [1, 2])
        if not np.any(good):
            continue
        x, y = along[good], original[good]
        # A linear interpolation has no spline overshoot. Repeated projections
        # to the same line coordinate are collapsed by median.
        unique, inverses = np.unique(x, return_inverse=True)
        if len(unique) != len(x):
            y = np.array([np.median(y[inverses == j]) for j in range(len(unique))])
            x = unique
        if not len(x):
            continue
        missing = ~good
        if not missing.any():
            continue
        loc = np.searchsorted(x, along[missing], side="left")
        left = loc - 1
        right = loc
        missing_idx = indices[missing]
        pos = along[missing]
        filled = np.zeros(len(pos), dtype=float)
        labels = np.zeros(len(pos), dtype="uint8")
        both = (left >= 0) & (right < len(x))
        if both.any():
            g = np.flatnonzero(both)
            span = x[right[g]] - x[left[g]]
            ok = span <= maximum_gap_m
            g = g[ok]
            if len(g):
                filled[g] = y[left[g]] + (
                    (y[right[g]] - y[left[g]])
                    * (pos[g] - x[left[g]])
                    / np.maximum(x[right[g]] - x[left[g]], 1e-9)
                )
                labels[g] = 4
            # If the two anchors are too far apart to justify interpolation,
            # taper from either supported end. Never jump abruptly to zero.
            large = np.flatnonzero(both)
            large = large[(x[right[large]] - x[left[large]]) > maximum_gap_m]
            for j in large:
                dl = pos[j] - x[left[j]]
                dr = x[right[j]] - pos[j]
                if 0 <= dl < end_taper_m:
                    filled[j] = y[left[j]] * (1 - dl / end_taper_m)
                    labels[j] = 5
                elif 0 <= dr < end_taper_m:
                    filled[j] = y[right[j]] * (1 - dr / end_taper_m)
                    labels[j] = 5
        # Bounded extrapolation at the outer ends of a supported segment.
        before = (left < 0) & (right < len(x))
        after = (right >= len(x)) & (left >= 0)
        for condition, dist, endpoint in (
            (before, x[0] - pos, y[0]),
            (after, pos - x[-1], y[-1]),
        ):
            use = condition & (dist >= 0) & (dist < end_taper_m)
            filled[use] = endpoint * (1 - dist[use] / end_taper_m)
            labels[use] = 5
        use = labels > 0
        output[rows[missing_idx[use]], cols[missing_idx[use]]] = filled[use]
        kind[rows[missing_idx[use]], cols[missing_idx[use]]] = labels[use]
        derived["interpolated"] += int(np.count_nonzero(labels == 4))
        derived["tapered"] += int(np.count_nonzero(labels == 5))
    return output, kind, derived


def regularize_alongshore_width(
    seeds,
    estimated_radius,
    support_type,
    coast_lines,
    spacing,
    *,
    smoothing_m=30.0,
):
    """Regularize *growth widths*, not original zero-crossing observations.

    Use only the assigned mapped coastline and arclength for neighboring
    estimates. Bracketed measurements receive greatest influence; one-sided
    extrapolations and gap-fills receive less. Never turn an unsupported zero
    into positive extension, cross unrelated coastline geometries, or exceed
    the original range of positive local evidence.

    This derivative is distinct from the raw estimated_radius and the evidence
    classes, which remain available unchanged for audit and visualization.
    """
    import shapely
    from shapely.strtree import STRtree

    if not coast_lines or not np.any(seeds & (estimated_radius > 0)):
        return estimated_radius.copy(), dict(smoothed_profiles=0, max_adjustment_m=0.0)
    dy, dx = map(float, spacing)
    rows, cols = np.where(seeds)
    points = shapely.points((cols + 0.5) * dx, (rows + 0.5) * dy)
    lines = np.asarray(coast_lines, dtype=object)
    tree = STRtree(coast_lines)
    assigned = np.asarray(tree.nearest(points), dtype=int)
    offsets = shapely.distance(points, lines[assigned])
    assigned[offsets > 2.5 * np.hypot(dx, dy)] = -1
    result = estimated_radius.copy()
    sigma = max(smoothing_m / 2.0, min(dx, dy))
    changed = 0
    max_adjustment = 0.0
    confidence = {1: 1.0, 2: 0.6, 4: 0.4, 5: 0.2}
    for k, line in enumerate(coast_lines):
        subset = np.flatnonzero(assigned == k)
        if not len(subset):
            continue
        along = np.asarray(shapely.line_locate_point(line, points[subset]), dtype=float)
        order = np.argsort(along, kind="stable")
        subset, along = subset[order], along[order]
        y = estimated_radius[rows[subset], cols[subset]].astype(float)
        kind = support_type[rows[subset], cols[subset]]
        eligible = (y > 0) & np.isin(kind, [1, 2, 4, 5])
        if not eligible.any():
            continue
        # An unsupported stretch is a domain break, NOT a zero-width anchor.
        # The pre-existing gap policy chooses which sites are eligible first.
        support_ids = np.flatnonzero(eligible)
        for loc in support_ids:
            start = np.searchsorted(along, along[loc] - 3 * sigma)
            end = np.searchsorted(along, along[loc] + 3 * sigma, side="right")
            neigh = np.arange(start, end)
            neigh = neigh[eligible[neigh]]
            if not len(neigh):
                continue
            weight = np.array([confidence.get(int(kind[j]), 0.0) for j in neigh])
            weight *= np.exp(-0.5 * ((along[neigh] - along[loc]) / sigma) ** 2)
            # Prevent filling discontinuous gaps longer than the allowed
            # alongshore inference horizon with smoothed values.
            gap = np.abs(along[neigh] - along[loc])
            weight[gap > ALONGSHORE_GAP_M] = 0
            total = weight.sum()
            if total <= 0:
                continue
            value = float(np.dot(weight, y[neigh]) / total)
            value = float(np.clip(value, y[neigh].min(), y[neigh].max()))
            old = float(y[loc])
            delta = abs(value - old)
            if delta > 1e-6:
                changed += 1
                max_adjustment = max(max_adjustment, delta)
                result[rows[subset[loc]], cols[subset[loc]]] = value
    return result, dict(
        smoothed_profiles=changed, max_adjustment_m=round(max_adjustment, 4)
    )


def grow(seeds, ocean, radius, spacing):
    if not seeds.any():
        return np.zeros_like(seeds)
    distance, nearest = ndimage.distance_transform_edt(
        ~seeds, sampling=spacing, return_indices=True
    )
    limit = radius[nearest[0], nearest[1]]
    candidates = ocean & (limit > 0) & (distance <= limit)
    starts = candidates & ndimage.binary_dilation(seeds)
    return ndimage.binary_propagation(starts, mask=candidates)


def smooth_growth(
    seeds,
    ocean,
    radius,
    spacing,
    *,
    smoothing_m=30,
    maximum_m=None,
    alongshore_smoothed=False,
):
    """Create a coherent, coast-following interpolation-only allowance.

    The ZERO-distance estimates are first regularized on the ORIGINAL coastal
    seed lattice. Unsupported seeds participate with zero width so solutions
    taper rather than leap across gaps. We then smooth the distance-to-boundary
    *field*, rather than dilating each independently estimated seed radius.
    No unsupported seed is promoted to an observation, and the result never
    crosses a measured/forbidden ocean cell in ``ocean``.

    One continuous component must have at least a short supported alongshore
    run; isolated cell protrusions are discarded, never enlarged. Narrow-water
    centre ridges are kept open to avoid joining opposite shoreline advances.
    The base Coast Refine land and its elevation values are not inputs here.
    """
    if not seeds.any() or not np.any(radius[seeds] > 0):
        return np.zeros_like(seeds, dtype=bool)
    if not np.isfinite(smoothing_m) or smoothing_m <= 0:
        raise ValueError("smoothing_m must be finite and positive")
    dy, dx = map(float, spacing)
    if alongshore_smoothed:
        # Production v52: arclength estimates already occupy each seed.
        # Never average with the opposing bank in a 2D raster neighborhood.
        smooth_radius = np.where(seeds, radius, 0).astype("float32")
    else:
        # Retain the legacy direct helper behavior for regression and callers
        # which provide raw cell-wise estimates without arclength processing.
        # 30-m alongshore window -> ~10-m Gaussian sigma, with exact physical axes.
        sigma = (max(0.7, smoothing_m / (3 * dy)), max(0.7, smoothing_m / (3 * dx)))
        seed_weight = seeds.astype("float32")
        numerator = ndimage.gaussian_filter(
            np.where(seeds, radius, 0).astype("float32"), sigma, mode="constant"
        )
        denominator = ndimage.gaussian_filter(seed_weight, sigma, mode="constant")
        smooth_radius = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-6,
        )
        # A supported shoreline segment must dominate its nearby coast. No
        # cross-gap interpolation from an isolated positive sample into zeros.
        support_fraction = np.divide(
            ndimage.gaussian_filter(
                (seeds & (radius > 0)).astype("float32"), sigma, mode="constant"
            ),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 1e-6,
        )
        # Continuous taper: a hard 50% support cutoff creates exactly the sharp
        # offshore finger ends we are trying to eliminate. With no local support,
        # the taper tends smoothly to zero; do not promote missing profiles.
        taper = np.clip((support_fraction - 0.20) / 0.60, 0, 1)
        smooth_radius *= taper
        smooth_radius[~seeds] = 0
    if maximum_m is not None:
        smooth_radius = np.minimum(smooth_radius, maximum_m)
    # Reject rather than publish a new abrupt alongshore width discontinuity.
    # Test neighboring 8-connected original coast cells in physical metres.
    max_jump = max(4 * max(dx, dy), smoothing_m * 0.5)
    for sy, sx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        if sx >= 0:
            a = np.s_[: None if sy == 0 else -sy, : None if sx == 0 else -sx]
            b = np.s_[sy:, sx:]
        else:
            a = np.s_[:-sy, -sx:]
            b = np.s_[sy:, :sx]
        pairs = seeds[a] & seeds[b]
        if (
            pairs.any()
            and np.max(np.abs(smooth_radius[a][pairs] - smooth_radius[b][pairs]))
            > max_jump
        ):
            raise ValueError(
                "Smoothed coast extension has an abrupt alongshore width jump"
            )
    dseed, closest = ndimage.distance_transform_edt(
        ~seeds, sampling=(dy, dx), return_indices=True
    )
    allowance = smooth_radius[closest[0], closest[1]]
    # Interpolate the allowance field over a few cells so the Voronoi borders
    # between individual source seeds do not create angular teeth.
    contour_sigma = (
        max(0.7, min(1.5, smoothing_m / (8 * dy))),
        max(0.7, min(1.5, smoothing_m / (8 * dx))),
    )
    allowance = ndimage.gaussian_filter(allowance, contour_sigma, mode="nearest")
    # Contour anti-aliasing must not promote a zero-support opposite shore.
    allowance[smooth_radius[closest[0], closest[1]] <= 0] = 0
    mask = ocean & (allowance > 0) & (dseed <= allowance)
    # Smooth the *land-plus-extension* silhouette. Avoid erosion of the base
    # land; the union is used only to calculate a coherent proposed outline.
    core = ~ocean | mask
    outline = (
        ndimage.gaussian_filter(core.astype("float32"), contour_sigma, mode="nearest")
        >= 0.50
    )
    mask = ocean & outline & (dseed <= allowance + min(dx, dy))
    if maximum_m is not None:
        mask &= dseed <= maximum_m
    # Keep ocean pathways on the medial ridge of narrow channels. This
    # prevents opposite coasts from closing the last connected water cells.
    water_depth = ndimage.distance_transform_edt(ocean, sampling=(dy, dx))
    crest = (water_depth > 0) & (
        water_depth
        >= ndimage.maximum_filter(water_depth, size=3, mode="nearest") - 1e-8
    )
    if maximum_m is not None:
        crest &= water_depth < maximum_m * 1.1
    # Only protect ridges that would actually be captured by the extension.
    mask[crest] = False
    # Four-connected shoreline continuity; no detached islands or small
    # protrusions which cannot be supported across a coastal segment.
    starts = mask & ndimage.binary_dilation(
        seeds, structure=ndimage.generate_binary_structure(2, 1)
    )
    mask = (
        ndimage.binary_propagation(
            starts, structure=ndimage.generate_binary_structure(2, 1), mask=mask
        )
        if starts.any()
        else np.zeros_like(mask)
    )
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    minimum_cells = max(3, int(round(smoothing_m / max(dx, dy))))
    if count:
        sizes = np.bincount(labels.ravel())
        keep = np.zeros(len(sizes), dtype=bool)
        keep[1:] = sizes[1:] >= minimum_cells
        mask &= keep[labels]
    return mask


def reconnect_water_corridors(full_ocean, proposed, spacing, *, smoothing_m=30):
    """Open limited, locally smoothed corridors when growth disconnects water.

    0-1 shortest-path search operates ONLY inside an original 4-connected ocean
    component. Existing water costs zero, candidate extension cells cost one.
    A recovered path is widened by a small physical-metre radius. This
    reduces the offending extension locally rather than shrinking an entire
    coastline reach to 8% because of one narrow channel.

    The returned geometry is still checked by original native-vector topology.
    In difficult cases the caller can reduce the width or decline the whole
    derived barrier; there is no silent change to source masks or elevations.
    """
    from collections import deque

    original, noriginal = ndimage.label(
        full_ocean, structure=ndimage.generate_binary_structure(2, 1)
    )
    mask = proposed.copy()
    restored = 0
    h, w = full_ocean.shape
    # Large migrations could be expensive; fail closed after a bounded number.
    for _ in range(12):
        wet, _ = ndimage.label(
            full_ocean & ~mask, structure=ndimage.generate_binary_structure(2, 1)
        )
        conflict = None
        for component in range(1, noriginal + 1):
            labels, counts = np.unique(
                wet[(original == component) & (wet > 0)], return_counts=True
            )
            if len(labels) > 1:
                biggest = labels[np.argmax(counts)]
                candidates = np.flatnonzero(labels != biggest)
                smallest = labels[candidates[np.argmin(counts[candidates])]]
                conflict = (component, int(biggest), int(smallest))
                break
        if conflict is None:
            return mask, restored, True
        component, goal, start = conflict
        # 0-1 BFS minimizes the number of cells that must be returned to water.
        dist = np.full((h, w), np.iinfo("int32").max, dtype="int32")
        parent = np.full((h, w), -1, dtype="int64")
        q = deque()
        yy, xx = np.where(wet == start)
        for y, x in zip(yy, xx):
            dist[y, x] = 0
            q.append(int(y * w + x))
        end = -1
        while q:
            k = q.popleft()
            y, x = divmod(k, w)
            if wet[y, x] == goal:
                end = k
                break
            base = int(dist[y, x])
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if (
                    ny < 0
                    or nx < 0
                    or ny >= h
                    or nx >= w
                    or original[ny, nx] != component
                ):
                    continue
                cost = int(mask[ny, nx])
                if base + cost < dist[ny, nx]:
                    dist[ny, nx] = base + cost
                    parent[ny, nx] = k
                    (q.append if cost else q.appendleft)(int(ny * w + nx))
        if end < 0:
            return mask, restored, False
        path = np.zeros_like(mask)
        while end >= 0:
            y, x = divmod(end, w)
            path[y, x] = True
            end = int(parent[y, x])
        # A small smooth corridor retains open water along the recovered route;
        # do not introduce a one-cell jagged notch through the extension.
        clearance = max(min(spacing) * 1.1, min(8.0, smoothing_m / 4))
        near = ndimage.distance_transform_edt(~path, sampling=spacing) <= clearance
        newly = int(np.count_nonzero(mask & near))
        if not newly:
            return mask, restored, False
        mask[near] = False
        restored += newly
    return mask, restored, False


def reopen_separate_land_passages(land, proposal, spacing, *, smoothing_m=30):
    """Keep a water channel between independently connected original land.

    A nearest-original-land partition identifies the contact interface between
    two expansions owned by different original land components. Remove only
    proposal cells in a short corridor around that interface. Never touch
    original land or any observed sample. Native vector topology still
    provides the final independent guard.
    """
    eight = ndimage.generate_binary_structure(2, 2)
    labels, number = ndimage.label(land, structure=eight)
    if number < 2 or not proposal.any():
        return proposal.copy(), 0
    _, closest = ndimage.distance_transform_edt(
        ~land, sampling=spacing, return_indices=True
    )
    owner = labels[closest[0], closest[1]]
    seam = np.zeros_like(proposal)
    for sy, sx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        if sx >= 0:
            aa = (slice(None, -sy if sy else None), slice(None, -sx if sx else None))
            bb = (slice(sy, None), slice(sx, None))
        else:
            aa = (slice(None, -sy), slice(-sx, None))
            bb = (slice(sy, None), slice(None, sx))
        contact = (
            proposal[aa]
            & proposal[bb]
            & (owner[aa] != owner[bb])
            & (owner[aa] > 0)
            & (owner[bb] > 0)
        )
        seam[aa] |= contact
        seam[bb] |= contact
    if not seam.any():
        return proposal.copy(), 0
    # A physically bounded short corridor is preferred over a whole-window
    # retreat. Native grid cells require at least a 1-cell clearance.
    clearance = max(min(spacing) * 1.1, min(6.0, smoothing_m / 5.0))
    corridor = ndimage.distance_transform_edt(~seam, sampling=spacing) <= clearance
    restored = int(np.count_nonzero(proposal & corridor))
    result = proposal & ~corridor
    return result, restored


def _land_merger_contacts(land, proposal):
    """Locate only contact with minor original land bodies in a merged group.

    A global nearest-owner seam can extend a long way beyond the actual land
    contact and cause the *local* taper to reduce an entire coastal reach.
    Instead identify connected new land groups that contain multiple original
    components, and use candidate cells touching the smaller original bodies
    as the conflict sites. No source land is ever removed.
    """
    eight = ndimage.generate_binary_structure(2, 2)
    originals, _ = ndimage.label(land, structure=eight)
    joined, _ = ndimage.label(land | proposal, structure=eight)
    result = np.zeros_like(proposal)
    for group in np.unique(joined[land]):
        if not group:
            continue
        labels = originals[(joined == group) & land]
        labels = labels[labels > 0]
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) <= 1:
            continue
        major = unique[np.argmax(counts)]
        for label in unique:
            if label == major:
                continue
            adjacent = ndimage.binary_dilation(originals == label, structure=eight)
            result |= proposal & adjacent
    return result


def _screen_topology_candidate(land, full_ocean, proposal, spacing, *, smoothing_m=30):
    """Screen a proposal against ORIGINAL raster connectivity.

    Returns (is_valid, conflict_footprint, failure, land_cells, water_cells).
    Corridor re-openers are *locators*, not the published solution: their
    removed cells identify the reach where a smoothly tapered width is needed.
    Final geometry must be regenerated from a width field, not a pixel notch.
    """
    four = ndimage.generate_binary_structure(2, 1)
    eight = ndimage.generate_binary_structure(2, 2)
    original_land = ndimage.label(land, structure=eight)[1]
    original_water = ndimage.label(full_ocean, structure=four)[1]
    work = proposal.copy()
    failures = []
    nland = nwater = 0
    land_contacts = np.zeros_like(proposal)
    if ndimage.label(land | work, structure=eight)[1] != original_land:
        land_contacts = _land_merger_contacts(land, work)
        # This is an evidence locator only; the smooth width field is
        # retapered and regrown by the caller rather than cutting a seam.
        nland += int(np.count_nonzero(land_contacts))
        failures.append("land_merger")
    if ndimage.label(full_ocean & ~work, structure=four)[1] != original_water:
        modified, _, repaired = reconnect_water_corridors(
            full_ocean,
            work,
            spacing,
            smoothing_m=smoothing_m,
        )
        nwater += int(np.count_nonzero(work & ~modified))
        work = modified
        failures.append("water_split" if not repaired else "water_corridor")
    land_valid = ndimage.label(land | work, structure=eight)[1] == original_land
    water_valid = ndimage.label(full_ocean & ~work, structure=four)[1] == original_water
    # Counts must be checked on the ORIGINAL candidate, not just the carved
    # candidate: carving produces an angular seam, not a smooth barrier.
    candidate_valid = (
        ndimage.label(land | proposal, structure=eight)[1] == original_land
        and ndimage.label(full_ocean & ~proposal, structure=four)[1] == original_water
    )
    if candidate_valid:
        return True, np.zeros_like(proposal), None, 0, 0
    footprint = (proposal & ~work) | land_contacts
    if not (land_valid and water_valid):
        failures.append("unresolved")
    return False, footprint, "+".join(failures) or "topology", nland, nwater


def _local_width_taper(permitted, seeds, conflict, spacing, *, reach_m, minimum=0.05):
    """Bounded, continuously varying width retreat near a forbidden corridor.

    Missing profile evidence is not synthesized: only existing positive
    permitted widths may decrease. The taper is evaluated at the coast seeds,
    and smooth_growth constructs the final topology from these widths.
    """
    if not np.any(conflict):
        return permitted.copy(), 1.0, 0
    d = ndimage.distance_transform_edt(~conflict, sampling=spacing)
    t = np.clip(d / float(reach_m), 0.0, 1.0)
    smoothstep = t * t * (3.0 - 2.0 * t)
    factor = minimum + (1.0 - minimum) * smoothstep
    out = permitted.copy()
    affected = seeds & (permitted > 0) & (factor < 1.0 - 1e-6)
    out[affected] *= factor[affected]
    smallest = float(factor[affected].min()) if affected.any() else 1.0
    return out, smallest, int(np.count_nonzero(affected))


def topology_safe_smooth_growth(
    seeds,
    available_ocean,
    full_ocean,
    land,
    radius,
    spacing,
    *,
    smoothing_m=30,
    maximum_m=None,
    diagnostics=None,
    alongshore_smoothed=False,
):
    """Create a smooth, topology-safe barrier with *local* retreat first.

    All zero-crossing observations, source masks and the original refined
    coastline stay unchanged. First grow at the permitted width. If this
    closes a water passage or merges land, locate the conflict using the
    existing conservative corridor methods, then taper WIDTHS near that
    conflict and regenerate an entirely smooth boundary. No repaired raster
    notches are used directly as the published extension. Whole-window
    retreat remains a last resort and is labelled as such in diagnostics.
    """
    if diagnostics is not None:
        diagnostics.setdefault("attempts", [])
        diagnostics.update(
            restored_water_cells=0,
            restored_land_passage_cells=0,
            topology_mode="local_first",
            local_taper_applied=False,
            local_taper_iterations=0,
            local_taper_seed_cells=0,
            local_minimum_width_fraction=1.0,
            whole_window_fallback=False,
        )
    permitted = (
        np.minimum(radius, float(maximum_m)) if maximum_m is not None else radius.copy()
    )

    def growth(candidate_width):
        return smooth_growth(
            seeds,
            available_ocean,
            candidate_width,
            spacing,
            smoothing_m=smoothing_m,
            maximum_m=maximum_m,
            alongshore_smoothed=alongshore_smoothed,
        )

    def screen(candidate):
        return _screen_topology_candidate(
            land,
            full_ocean,
            candidate,
            spacing,
            smoothing_m=smoothing_m,
        )

    try:
        first = growth(permitted)
    except ValueError as exc:
        if str(exc) != "Smoothed coast extension has an abrupt alongshore width jump":
            raise
        first = None
        if diagnostics is not None:
            diagnostics["attempts"].append(
                dict(fraction=1.0, reason="width_discontinuity")
            )

    if first is not None and first.any():
        ok, conflict, reason, _, _ = screen(first)
        if ok:
            if diagnostics is not None:
                diagnostics["attempts"].append(dict(fraction=1.0, reason="accepted"))
            return first, 1.0
        if diagnostics is not None:
            diagnostics["attempts"].append(dict(fraction=1.0, reason=reason))

        # The fitted width is already smooth along each connected coast.
        # Keep the full width everywhere EXCEPT near a conflict corridor. A
        # gradually expanded physical-metre neighborhood increases chance of
        # a valid reconnection while leaving remote coast untouched.
        conflict_union = conflict.copy()
        reach = max(2 * smoothing_m, 1.5 * float(maximum_m or np.max(permitted)))
        for attempt in range(1, 7):
            if not conflict_union.any():
                break  # Existing reconnection did not identify a usable path.
            candidate_width, minimum_factor, affected = _local_width_taper(
                permitted,
                seeds,
                conflict_union,
                spacing,
                reach_m=reach * (1 + 0.4 * (attempt - 1)),
                minimum=0.05,
            )
            if not affected:
                break
            try:
                candidate = growth(candidate_width)
            except ValueError as exc:
                if (
                    str(exc)
                    != "Smoothed coast extension has an abrupt alongshore width jump"
                ):
                    raise
                if diagnostics is not None:
                    diagnostics["attempts"].append(
                        dict(
                            local_iteration=attempt,
                            reason="width_discontinuity",
                            affected_seed_cells=affected,
                        )
                    )
                continue
            if not candidate.any():
                break
            valid, new_conflict, reason, _, _ = screen(candidate)
            if diagnostics is not None:
                diagnostics["attempts"].append(
                    dict(
                        local_iteration=attempt,
                        reason="accepted_local" if valid else reason,
                        affected_seed_cells=affected,
                        local_minimum_width_fraction=round(minimum_factor, 4),
                    )
                )
            if valid:
                if diagnostics is not None:
                    diagnostics.update(
                        local_taper_applied=True,
                        local_taper_iterations=attempt,
                        local_taper_seed_cells=affected,
                        local_minimum_width_fraction=round(minimum_factor, 4),
                    )
                return candidate, 1.0
            conflict_union |= new_conflict

    # Explicit conservative fallback. No speculative geometry is published.
    if diagnostics is not None:
        diagnostics["whole_window_fallback"] = True
    for fraction in (0.8, 0.6, 0.4, 0.25, 0.15, 0.08):
        try:
            candidate = growth(permitted * fraction)
        except ValueError as exc:
            if (
                str(exc)
                != "Smoothed coast extension has an abrupt alongshore width jump"
            ):
                raise
            if diagnostics is not None:
                diagnostics["attempts"].append(
                    dict(fraction=fraction, reason="width_discontinuity")
                )
            continue
        if not candidate.any():
            continue
        valid, _, reason, _, _ = screen(candidate)
        if diagnostics is not None:
            diagnostics["attempts"].append(
                dict(
                    fraction=fraction,
                    reason="accepted_global" if valid else reason,
                )
            )
        if valid:
            return candidate, fraction
    return np.zeros_like(seeds, dtype=bool), 0.0
