"""Bounded working graph; reuse each tile's path masses for every profile.

The experiment harness supplies H11891 arrays. Production callers can use
window_selections directly with raster-window reads and stream core results.
"""

import time
import numpy as np
from affine import Affine
from .graph import build, metric, Rule, grid_edge_limit
from .evidence import measure, decide
from .fragments import complete


def window_selections(
    z, first, gross, profile, peers, profiles, timings=None, max_gap_m=None
):
    if not np.any(first == 1):
        return {
            key: (np.zeros(z.shape, bool), np.zeros(z.shape, bool)) for key in profiles
        }
    protections = {}
    eligible = np.zeros(z.shape, bool)
    for key, param in profiles.items():
        protected = np.zeros(z.shape, bool)
        for peer in peers:
            protected |= (
                np.isfinite(z)
                & np.isfinite(peer)
                & (abs(z - peer) <= param["agreement_m"])
            )
        protections[key] = protected
        eligible |= (first == 0) & ~protected
    if not eligible.any():
        return {
            key: (np.zeros(z.shape, bool), np.zeros(z.shape, bool)) for key in profiles
        }
    started = time.perf_counter()
    graph = build(z, profile, Rule(max_gap_m=max_gap_m or grid_edge_limit(profile)))
    built = time.perf_counter()
    features = measure(graph, first, profile, eligible_mask=eligible)
    measured = time.perf_counter()
    result = {}
    if timings is not None:
        timings["graph_seconds"] = built - started
        timings["path_seconds"] = measured - built
    if timings is not None:
        timings["shared_geometry_and_paths_seconds"] = time.perf_counter() - started
    for key, param in profiles.items():
        started = time.perf_counter()
        protected = protections[key]
        primary = decide(
            graph, first, features, protected, rejection_ratio=param["rejection_ratio"]
        )
        decided = time.perf_counter()
        extra, _ = (
            complete(graph, first, gross, primary, protected, profile)
            if param["fragment"]
            else (np.zeros(z.shape, bool), {})
        )
        result[key] = (primary, extra)
        if timings is not None:
            timings["decision_seconds"] = decided - started
            timings["fragment_seconds"] = time.perf_counter() - decided
        if timings is not None:
            timings[key] = time.perf_counter() - started
    return result


def selections(z, first, gross, profile, peers, profiles, tile_size=512):
    if not isinstance(tile_size, int) or tile_size < 1:
        raise ValueError("Positive tile size required")
    h, w = z.shape
    dx, dy = metric(profile, np.array([0, h - 1]), np.array([0, h - 1]))
    edge_limit = grid_edge_limit(profile)
    halo = int(np.ceil((400 + 2 * edge_limit) / min(float(dx.min()), dy))) + 1
    result = {
        k: (np.zeros(z.shape, bool), {"primary": 0, "fragment_added": 0})
        for k in profiles
    }
    started = time.perf_counter()
    tiles = 0
    shared_seconds = 0.0
    profile_seconds = {k: 0.0 for k in profiles}
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y1, x1 = min(h, y + tile_size), min(w, x + tile_size)
            if not np.isfinite(z[y:y1, x:x1]).any():
                continue
            a, b, c, d = (
                max(0, y - halo),
                min(h, y1 + halo),
                max(0, x - halo),
                min(w, x1 + halo),
            )
            ext = np.s_[a:b, c:d]
            inner = np.s_[y - a : y1 - a, x - c : x1 - c]
            p = {
                **profile,
                "height": b - a,
                "width": d - c,
                "transform": profile["transform"] * Affine.translation(c, a),
            }
            tile_times = {}
            values = window_selections(
                z[ext],
                first[ext],
                gross[ext],
                p,
                [peer[ext] for peer in peers],
                profiles,
                tile_times,
                max_gap_m=edge_limit,
            )
            tiles += 1
            shared_seconds += tile_times.get("shared_geometry_and_paths_seconds", 0.0)
            for key in profiles:
                profile_seconds[key] += tile_times.get(key, 0.0)
            for key, (primary, extra) in values.items():
                result[key][0][y:y1, x:x1] = primary[inner] | extra[inner]
                result[key][1]["primary"] += int(primary[inner].sum())
                result[key][1]["fragment_added"] += int(extra[inner].sum())
    return result, {
        "filter_seconds_all_profiles": time.perf_counter() - started,
        "occupied_tiles": tiles,
        "core_size_cells": tile_size,
        "halo_cells": halo,
        "path_traces_per_occupied_tile": 1,
        "feature_cache_written": False,
        "shared_geometry_and_paths_seconds": shared_seconds,
        "decision_fragment_seconds_by_profile": profile_seconds,
    }
