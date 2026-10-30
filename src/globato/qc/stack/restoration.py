"""Cheap native-array ridge/valley restoration; no reference surface, DEM gap-fill, or I/O.

Removed cells supply locations and candidate ORIGINAL elevations, never fitting
support. No post-removal interpolated DEM is required. Outside the union of all source rejections the cached original survey
values are surviving measurements. Known surveys each supply one context layer;
unknown sources may supply one pooled terrain layer, never independent votes.
"""

import warnings
import numpy as np
from scipy import ndimage as ndi

POLICY = dict(
    version="retained_flanks_v2",
    core_cells=256,
    context_radius_m=45.0,
    minimum_trace_neighbors=8,
    minimum_major_std_m=6.0,
    maximum_minor_std_m=18.0,
    minimum_trace_coherence=0.4,
    maximum_removal_fraction=0.55,
    maximum_flank_distance_m=90.0,
    flank_sample_spacing_m=6.0,
    fit_tolerance_m=1.75,
    source_tolerance_m=1.75,
    minimum_context_relief_m=1.75,
    maximum_growth_m=6.0,
    minimum_along_support=1,
    along_distance_m=6.0,
)


def moments(mask, spacing, policy=POLICY):
    row_spacing, col_spacing = spacing
    window_rows = 2 * int(np.ceil(policy["context_radius_m"] / row_spacing)) + 1
    window_cols = 2 * int(np.ceil(policy["context_radius_m"] / col_spacing)) + 1
    y_m, x_m = np.indices(mask.shape, dtype="float64")
    x_m *= col_spacing
    y_m *= row_spacing
    weights = np.asarray(mask, dtype="float64")
    box = lambda a: (  # noqa: E731 -- frozen scientific expression
        ndi.uniform_filter(a, size=(window_rows, window_cols), mode="constant")
        * (window_rows * window_cols)
    )
    count = box(weights)
    denominator = np.maximum(count, 1)
    mean_x, mean_y = box(weights * x_m) / denominator, box(weights * y_m) / denominator
    variance_x = np.maximum(box(weights * x_m * x_m) / denominator - mean_x * mean_x, 0)
    variance_y = np.maximum(box(weights * y_m * y_m) / denominator - mean_y * mean_y, 0)
    covariance = box(weights * x_m * y_m) / denominator - mean_x * mean_y
    eigenvalue_gap = np.hypot(variance_x - variance_y, 2 * covariance)
    major_variance = np.maximum((variance_x + variance_y + eigenvalue_gap) / 2, 0)
    minor_variance = np.maximum((variance_x + variance_y - eigenvalue_gap) / 2, 0)
    coherence = np.divide(
        eigenvalue_gap,
        major_variance + minor_variance,
        out=np.zeros(mask.shape),
        where=major_variance + minor_variance > 1e-6,
    )
    angle = 0.5 * np.arctan2(2 * covariance, variance_x - variance_y)
    trace = (
        (count >= policy["minimum_trace_neighbors"])
        & (np.sqrt(major_variance) >= policy["minimum_major_std_m"])
        & (np.sqrt(minor_variance) <= policy["maximum_minor_std_m"])
        & (coherence >= policy["minimum_trace_coherence"])
        & (count / (window_rows * window_cols) <= policy["maximum_removal_fraction"])
    )
    return trace, angle, coherence


def sample(elevations, valid, rows, cols):
    coords = np.stack((rows, cols))
    weights = ndi.map_coordinates(
        valid.astype("float32"), coords, order=1, mode="constant", cval=0.0
    )
    values = ndi.map_coordinates(
        np.where(valid, elevations, 0.0), coords, order=1, mode="constant", cval=0.0
    )
    return np.where(weights >= 0.99999, values, np.nan)


def fit_flanks(context, removed, rows, cols, angle, spacing, policy=POLICY):
    """Fit the first three well-spaced surviving samples on each side of a gap.

    An upper (valley) / lower (ridge) envelope accommodates the banks as well
    as the vertex. Crossing must fall inside the removed footprint, between
    the nearest surviving flanks. No deleted observation is used by this fit.
    """
    row_spacing, col_spacing = spacing
    normal_rows, normal_cols = np.cos(angle) / row_spacing, -np.sin(angle) / col_spacing
    valid = np.isfinite(context) & ~removed
    candidate_count = len(rows)
    side_fits = []
    nearest_distances = []
    fit_errors = []
    step = min(row_spacing, col_spacing)
    distances = np.arange(step, policy["maximum_flank_distance_m"] + step * 0.01, step)
    for sign in (-1.0, 1.0):
        samples_z = np.full((3, candidate_count), np.nan)
        samples_t = np.full((3, candidate_count), np.nan)
        sample_counts = np.zeros(candidate_count, "uint8")
        last_distance = np.full(candidate_count, -np.inf)
        for distance in distances:
            pending = (sample_counts < 3) & (
                distance - last_distance >= policy["flank_sample_spacing_m"] - 1e-6
            )
            if not pending.any():
                continue
            pending_indices = np.flatnonzero(pending)
            sampled_z = sample(
                context,
                valid,
                rows[pending_indices] + normal_rows[pending_indices] * distance * sign,
                cols[pending_indices] + normal_cols[pending_indices] * distance * sign,
            )
            good = np.isfinite(sampled_z)
            accepted_indices = pending_indices[good]
            samples_z[sample_counts[accepted_indices], accepted_indices] = sampled_z[
                good
            ]
            samples_t[sample_counts[accepted_indices], accepted_indices] = (
                distance * sign
            )
            last_distance[accepted_indices] = distance
            sample_counts[accepted_indices] += 1
        # Analytic ordinary least squares; all samples must be present.
        mean_distance = np.mean(samples_t, axis=0)
        mean_elevation = np.mean(samples_z, axis=0)
        distance_variance = np.sum((samples_t - mean_distance) ** 2, axis=0)
        slope = np.divide(
            np.sum((samples_t - mean_distance) * (samples_z - mean_elevation), axis=0),
            distance_variance,
            out=np.full(candidate_count, np.nan),
            where=distance_variance > 1e-8,
        )
        intercept = mean_elevation - slope * mean_distance
        rms = np.sqrt(np.mean((samples_z - intercept - slope * samples_t) ** 2, axis=0))
        side_fits.append((intercept, slope))
        nearest_distances.append(samples_t[0])
        fit_errors.append(rms)
    (left_intercept, left_slope), (right_intercept, right_slope) = side_fits
    valley = (left_slope < 0) & (right_slope > 0)
    ridge = (left_slope > 0) & (right_slope < 0)
    intersection_distance = np.divide(
        right_intercept - left_intercept,
        left_slope - right_slope,
        out=np.full(candidate_count, np.nan),
        where=abs(left_slope - right_slope) > 1e-8,
    )
    between_flanks = (intersection_distance >= nearest_distances[0]) & (
        intersection_distance <= nearest_distances[1]
    )
    safe_intersection = np.nan_to_num(intersection_distance, nan=0.0)
    inside_removed = (
        ndi.map_coordinates(
            removed.astype("uint8"),
            np.stack(
                (
                    rows + normal_rows * safe_intersection,
                    cols + normal_cols * safe_intersection,
                )
            ),
            order=0,
            mode="constant",
            cval=0,
        )
        > 0
    )
    good = (
        np.isfinite(intersection_distance)
        & between_flanks
        & inside_removed
        & (valley | ridge)
        & (np.maximum(*fit_errors) <= policy["fit_tolerance_m"])
    )
    prediction = np.where(
        valley,
        np.maximum(left_intercept, right_intercept),
        np.minimum(left_intercept, right_intercept),
    )
    shape_kind = np.where(valley, 1, 2).astype("uint8")
    shape_kind[~good] = 0
    prediction[~good] = np.nan
    return prediction, shape_kind, np.maximum(*fit_errors)


def tile_model(survey_values, removed, spacing, policy=POLICY):
    """One shared model per bounded tile, independent of candidate-source values."""
    if np.asarray(survey_values).ndim != 3 or survey_values.shape[1:] != removed.shape:
        raise ValueError("Survey/footprint grids differ")
    if not all(np.isfinite(x) and x > 0 for x in spacing):
        raise ValueError("Positive native metre spacing required")
    retained = np.asarray(survey_values, dtype="float32").copy()
    retained[:, removed] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        context = np.nanmedian(retained, axis=0)
    valid = np.isfinite(context)
    dy, dx = spacing
    shape = (
        2 * int(np.ceil(policy["context_radius_m"] / dy)) + 1,
        2 * int(np.ceil(policy["context_radius_m"] / dx)) + 1,
    )
    box = lambda a: ndi.uniform_filter(a, size=shape, mode="constant")  # noqa: E731
    w = box(valid.astype("float64"))
    origin = float(context[valid][0]) if valid.any() else 0.0
    z = np.where(valid, context - origin, 0.0)
    mean = np.divide(box(z), w, out=np.zeros(context.shape), where=w > 0)
    var = (
        np.divide(box(z * z), w, out=np.zeros(context.shape), where=w > 0) - mean * mean
    )
    relief = np.sqrt(np.maximum(var, 0.0))
    trace, angle, coherence = moments(removed, spacing, policy)
    allowed = removed & (relief >= policy["minimum_context_relief_m"]) & (w >= 0.25)
    seed = allowed & trace
    # Limited extension stays inside the real removed footprint. Every grown
    # location still needs a full two-flank fit and original-source check.
    grown = seed.copy()
    iterations = int(np.ceil(policy["maximum_growth_m"] / max(spacing)))
    for _ in range(iterations):
        grown |= ndi.binary_dilation(grown, structure=np.ones((3, 3), bool)) & allowed
    grown &= (
        ndi.distance_transform_edt(~seed, sampling=spacing)
        <= policy["maximum_growth_m"]
    )
    rr, cc = np.nonzero(grown)
    pred, kind, rms = fit_flanks(
        context, removed, rr, cc, angle[rr, cc], spacing, policy
    )
    supported = np.zeros(len(rr), "uint8")
    # Neighbouring stations use the SAME surviving context, never the newly
    # restored candidate. This checks longitudinal continuation of the shape.
    for sign in (-1.0, 1.0):
        ar = rr + sign * np.sin(angle[rr, cc]) * policy["along_distance_m"] / dy
        ac = cc + sign * np.cos(angle[rr, cc]) * policy["along_distance_m"] / dx
        ap, ak, _ = fit_flanks(context, removed, ar, ac, angle[rr, cc], spacing, policy)
        supported += (ak == kind) & (kind > 0) & np.isfinite(ap)
    kind[supported < policy["minimum_along_support"]] = 0
    pred[kind == 0] = np.nan
    out = np.full(removed.shape, np.nan, "float32")
    k = np.zeros(removed.shape, "uint8")
    err = np.full_like(out, np.nan)
    out[rr, cc] = pred
    k[rr, cc] = kind
    err[rr, cc] = rms
    return dict(
        prediction=out,
        kind=k,
        flank_rms=err,
        trace_candidate=grown,
        context=context,
        coherence=coherence.astype("float32"),
    )


def context_halo(spacing, policy=POLICY):
    """Bound all neighborhood/flank/continuation dependencies in one place."""
    distance = sum(
        policy[key]
        for key in (
            "context_radius_m",
            "maximum_flank_distance_m",
            "along_distance_m",
            "maximum_growth_m",
        )
    )
    return int(np.ceil(distance / min(spacing))) + 3


def model(survey_values, removed, spacing, policy=POLICY):
    """Bounded work arrays; survey layers are a read-only memmap in production."""
    removed = np.asarray(removed, bool)
    shape = removed.shape
    out = dict(
        prediction=np.full(shape, np.nan, "float32"),
        kind=np.zeros(shape, "uint8"),
        flank_rms=np.full(shape, np.nan, "float32"),
        trace_candidate=np.zeros(shape, bool),
    )
    halo = context_halo(spacing, policy)
    core = policy["core_cells"]
    for y in range(0, shape[0], core):
        for x in range(0, shape[1], core):
            ys = slice(y, min(y + core, shape[0]))
            xs = slice(x, min(x + core, shape[1]))
            if not removed[ys, xs].any():
                continue
            sy = slice(max(0, y - halo), min(shape[0], y + core + halo))
            sx = slice(max(0, x - halo), min(shape[1], x + core + halo))
            inner = (
                slice(y - sy.start, ys.stop - sy.start),
                slice(x - sx.start, xs.stop - sx.start),
            )
            block = tile_model(
                survey_values[:, sy, sx],
                removed[sy, sx],
                spacing,
                policy,
            )
            for name in out:
                out[name][ys, xs] = block[name][inner]
    return out


def accept_source(
    original, valid, rejected, prediction, *, clear_outlier=None, policy=POLICY
):
    if any(a.shape != rejected.shape for a in (original, valid, prediction)):
        raise ValueError("Source grids differ")
    out = (
        rejected
        & valid
        & np.isfinite(original)
        & np.isfinite(prediction)
        & (abs(original - prediction) <= policy["source_tolerance_m"])
    )
    if clear_outlier is not None:
        out &= ~clear_outlier
    return out
