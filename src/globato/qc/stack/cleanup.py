"""Exact-rule cleanup with sparse peer preparation and source-major selection."""

import json
import logging
import time
from pathlib import Path
import numpy as np
import rasterio
from .io import sha, write as json_write, raster_profile
from .sources import MaskWriter
from .connectivity.tiled import window_selections
from .connectivity.graph import metric
from .cleanup_peers import CleanupPeers

logger = logging.getLogger(__name__)


def run(pool, first, gross, final, out, tile_size=512, detection_passes=2):
    if not isinstance(tile_size, int) or tile_size < 1:
        raise ValueError("Positive tile size required")
    started = time.perf_counter()
    inv = pool.inv
    first = json.loads(Path(first).read_text())
    gross = json.loads(Path(gross).read_text())
    parent = json.loads(Path(final).read_text())
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ids = {r["source_id"] for r in inv["sources"]}
    for rec in inv["sources"]:
        stat = Path(rec["path"]).stat()
        if rec.get("fingerprint") and (
            stat.st_size != rec["fingerprint"]["size"]
            or stat.st_mtime_ns != rec["fingerprint"]["mtime_ns"]
        ):
            raise ValueError("Original native source changed")
    for manifest in [first, gross, parent]:
        if set(manifest["masks"]) != ids:
            raise ValueError("Cleanup source inventory mismatch")
        for row in manifest["masks"].values():
            if sha(row["mask"]) != row["mask_sha256"]:
                raise ValueError("Input mask changed")
    phases = {
        "input_validation": time.perf_counter() - started,
        "peer_medians": 0.0,
        "native_reads": 0.0,
        "peer_reads": 0.0,
        "graph_seconds": 0.0,
        "path_seconds": 0.0,
        "decision_seconds": 0.0,
        "fragment_seconds": 0.0,
        "mask_apply": 0.0,
    }
    writer = MaskWriter(out / "MASKS", pool)
    p = raster_profile(inv["grid"])
    h, w = inv["grid"]["shape"]
    peers = None
    observed = {}
    counts = {}
    added = 0
    per_source = {}
    tile_count = 0

    def source(rec, bounds, manifest):
        tick = time.perf_counter()
        result = pool.cleanup_source(rec, bounds, manifest)
        phases["native_reads"] += time.perf_counter() - tick
        return result

    try:
        rows = np.array([0, h - 1])
        dx, dy = metric(p, rows, rows)
        halo = int(np.ceil(460 / min(float(dx.min()), dy))) + 1
        logger.info("CLEANUP preparing peer evidence only for candidate windows")
        tick = time.perf_counter()
        peers = CleanupPeers(
            pool, first, p, out / "peer_medians", source, halo=halo, tile_size=tile_size
        )
        phases["peer_medians"] = time.perf_counter() - tick
        for rec in inv["sources"]:
            start = time.perf_counter()
            sid = rec["source_id"]
            observed[sid] = 0
            source_added = 0
            for y in range(0, h, tile_size):
                for x in range(0, w, tile_size):
                    y1, x1 = min(h, y + tile_size), min(w, x + tile_size)
                    core = (slice(y, y1), slice(x, x1))
                    z0, valid0, _, old = source(rec, core, parent)
                    proposal = np.zeros(valid0.shape, bool)
                    if valid0.any():
                        a, b, c, d = (
                            max(0, y - halo),
                            min(h, y1 + halo),
                            max(0, x - halo),
                            min(w, x1 + halo),
                        )
                        bounds = (slice(a, b), slice(c, d))
                        inner = np.s_[y - a : y1 - a, x - c : x1 - c]
                        z, valid, _, f = source(rec, bounds, first)
                        if np.any(valid0 & ~f[inner]) and np.any(f):
                            gr = pool.rejection_mask(rec, bounds, gross)
                            fp = np.where(valid, f.astype(float), np.nan)
                            gp = np.where(valid, gr.astype(float), np.nan)
                            pp = {
                                **p,
                                "height": b - a,
                                "width": d - c,
                                "transform": p["transform"]
                                * rasterio.Affine.translation(c, a),
                            }
                            timings = {}
                            selected = window_selections(
                                z,
                                fp,
                                gp,
                                pp,
                                peers.views(rec["group"], bounds),
                                {
                                    "selected": {
                                        "agreement_m": 7,
                                        "rejection_ratio": 8,
                                        "fragment": True,
                                    }
                                },
                                timings=timings,
                                max_gap_m=30.0,
                            )["selected"]
                            proposal = (selected[0] | selected[1])[inner] & valid0
                            tile_count += 1
                            for name in [
                                "graph_seconds",
                                "path_seconds",
                                "decision_seconds",
                                "fragment_seconds",
                            ]:
                                phases[name] += timings.get(name, 0.0)
                    tick = time.perf_counter()
                    observed[sid] += int(valid0.sum())
                    new = old | proposal
                    source_added += int((new & ~old).sum())
                    writer.put(rec, core, new, old)
                    phases["mask_apply"] += time.perf_counter() - tick
            counts[sid] = int(writer.counts[sid])
            added += source_added
            per_source[sid] = time.perf_counter() - start
            logger.info(
                "CLEANUP %s added %s",
                sid.split("/")[-1].split("|")[0],
                source_added,
            )
        phases["peer_reads"] = peers.stats["peer_read_seconds"]
        manifest, _ = writer.finish(
            {
                "source_counts": {
                    sid: {"observed": observed[sid], "rejected": counts[sid]}
                    for sid in observed
                },
                "method": "connectivity",
                "policy": "t7_b8_post_r8",
                "additional_exclusions": added,
                "BAG_used": False,
            }
        )
        elapsed = time.perf_counter() - started
        receipt = {
            "manifest": str(manifest),
            "manifest_sha256": sha(manifest),
            "additional_exclusions": added,
            "processing_seconds": elapsed,
            "standalone_processing_seconds": elapsed,
            "evidence_reused": False,
            "proposal_cache_written": False,
            "candidate_detection_passes": detection_passes,
            "phase_seconds": phases,
            "phase_note": "peer_medians includes planning and some native_reads; do not sum overlapping phase counters",
            "peer_io": peers.stats,
            "cleanup_implementation": "v44_sparse_peer_preparation_source_major",
            "processed_tiles": tile_count,
            "per_source_seconds": per_source,
            "parameters": {
                "agreement_m": 7,
                "rejection_ratio": 8,
                "restoration_match_m": 8,
                "fragment": True,
                "score": 0.7,
                "contacts": 12,
                "path_m": 400,
                "edge_m": 30,
            },
        }
        json_write(out / "DONE.json", receipt)
        return receipt
    finally:
        if peers is not None:
            peers.close()
        for reader in writer.open.values():
            reader.close()
        writer.open.clear()
        pool.close()
