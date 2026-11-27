"""Prepare only peer evidence that cleanup can read, using bounded windows.

The query plan is conservative: mask bits and source footprints may request
extra evidence, but never omit a window that the historical selection can use.
Survey medians retain the original source order, float64 dtype and nanmedian.
"""

from collections import OrderedDict
import hashlib
from pathlib import Path
import time
import warnings

import numpy as np
import rasterio
from rasterio.windows import Window


def median_observed_columns(values, observed):
    """Same per-cell float64 nanmedian; omit columns known to be all NaN."""
    median = np.full(observed.shape, np.nan, dtype="float64")
    if observed.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median[observed] = np.nanmedian(
                np.stack([value[observed] for value in values]), axis=0
            )
    return median


class CleanupPeers:
    block_height = 128
    block_width = 512

    def __init__(self, pool, first, profile, folder, source, *, halo, tile_size):
        self.pool = pool
        self.first = first
        self.profile = profile
        self.source = source
        self.folder = Path(folder)
        self.paths = {}
        self.coverage = {}
        self._writers = OrderedDict()
        self._readers = OrderedDict()
        self.stats = dict(
            planning_seconds=0.0,
            candidate_windows=0,
            requested_blocks=0,
            median_source_reads=0,
            median_tiles_written=0,
            median_tiles_considered=0,
            median_columns_available=0,
            median_columns_evaluated=0,
            peer_reads=0,
            peer_empty_skips=0,
            peer_read_seconds=0.0,
            peak_peer_window_bytes=0,
        )
        try:
            # No independent survey can protect a point in a one-survey run.
            if len(pool.groups) < 2:
                return
            started = time.perf_counter()
            requests = self._plan(halo, tile_size)
            self.stats["planning_seconds"] = time.perf_counter() - started
            self.stats["requested_blocks"] = len(requests)
            if requests:
                self.folder.mkdir(parents=True, exist_ok=True)
                self._prepare(requests)
        except BaseException:
            self.close()
            raise
        finally:
            self._close(self._writers)

    def _blocks(self, bounds):
        ys, xs = bounds
        for row in range(
            ys.start // self.block_height, (ys.stop - 1) // self.block_height + 1
        ):
            for col in range(
                xs.start // self.block_width, (xs.stop - 1) // self.block_width + 1
            ):
                yield row, col

    def _plan(self, halo, tile_size):
        h, w = self.profile["height"], self.profile["width"]
        requests = {}
        # Keep planning source-major too. Only the first-pass masks are read.
        # Do not trust rejection-count metadata as a substitute for mask bits.
        for rec in self.pool.records:
            group = rec["group"]
            if not group:
                continue  # Historical cleanup has no peer votes for unknowns.
            for y in range(0, h, tile_size):
                for x in range(0, w, tile_size):
                    y1, x1 = min(h, y + tile_size), min(w, x + tile_size)
                    core = (slice(y, y1), slice(x, x1))
                    dy, dx = rec["offset"]
                    rh, rw = rec["grid"]["shape"]
                    if (
                        dy >= y1
                        or dy + rh <= y
                        or dx >= x1
                        or dx + rw <= x
                        or not self.pool.intersects(rec, core)
                    ):
                        continue
                    a, b = max(0, y - halo), min(h, y1 + halo)
                    c, d = max(0, x - halo), min(w, x1 + halo)
                    bounds = (slice(a, b), slice(c, d))
                    rejected = self.pool.rejection_mask(rec, bounds, self.first)
                    # Omitting validity makes this a superset of the actual
                    # test, including holes and unobserved source boundaries.
                    if (
                        not rejected.any()
                        or rejected[y - a : y1 - a, x - c : x1 - c].all()
                    ):
                        continue
                    self.stats["candidate_windows"] += 1
                    for block in self._blocks(bounds):
                        requests.setdefault(block, set()).add(group)
        return requests

    def _writer(self, group):
        if group not in self.paths:
            path = self.folder / (
                hashlib.sha256(group.encode()).hexdigest()[:16] + ".tif"
            )
            self.paths[group] = path
            self.coverage[group] = set()
            with rasterio.open(
                path,
                "w",
                **{
                    **self.profile,
                    "compress": "deflate",
                    "tiled": True,
                    "blockxsize": self.block_width,
                    "blockysize": self.block_height,
                    "dtype": "float64",
                    "SPARSE_OK": "TRUE",
                    "BIGTIFF": "IF_SAFER",
                },
            ):
                pass
        if group not in self._writers:
            self._writers[group] = rasterio.open(self.paths[group], "r+")
            if len(self._writers) > 8:
                self._writers.popitem(last=False)[1].close()
        self._writers.move_to_end(group)
        return self._writers[group]

    def _prepare(self, requests):
        h, w = self.profile["height"], self.profile["width"]
        # Traverse blocks once; group records keep native inventory order.
        for (row, col), requesters in sorted(requests.items()):
            y, x = row * self.block_height, col * self.block_width
            y1, x1 = min(h, y + self.block_height), min(w, x + self.block_width)
            bounds = (slice(y, y1), slice(x, x1))
            grouped = {}
            for rec in self.pool.active_records(bounds):
                group = rec["group"]
                if group is not None and (
                    len(requesters) > 1 or group not in requesters
                ):
                    grouped.setdefault(group, []).append(rec)
            for group, records in grouped.items():
                self.stats["median_tiles_considered"] += 1
                values = []
                observed = np.zeros((y1 - y, x1 - x), dtype=bool)
                for rec in records:
                    z, valid, _, rejected = self.source(rec, bounds, self.first)
                    accepted = valid & ~rejected
                    observed |= accepted
                    values.append(np.where(accepted, z, np.nan).astype("float64"))
                    self.stats["median_source_reads"] += 1
                self.stats["median_columns_available"] += observed.size
                self.stats["median_columns_evaluated"] += int(observed.sum())
                median = median_observed_columns(values, observed)
                data = np.where(np.isfinite(median), median, -9999).astype("float64")
                # -9999 is also masked by the historical reader. An omitted
                # sparse block reads as this same NoData value.
                if np.any(data != -9999):
                    self._writer(group).write(
                        data, 1, window=Window(x, y, x1 - x, y1 - y)
                    )
                    self.coverage[group].add((row, col))
                    self.stats["median_tiles_written"] += 1

    def views(self, group, bounds):
        if not group:
            return
        # Same ordering as the old readers dictionary. Stream one array at a
        # time instead of retaining every survey surface for this halo window.
        for peer in self.pool.groups:
            if peer == group:
                continue
            if peer not in self.coverage or not any(
                block in self.coverage[peer] for block in self._blocks(bounds)
            ):
                self.stats["peer_empty_skips"] += 1
                continue
            started = time.perf_counter()
            if peer not in self._readers:
                self._readers[peer] = rasterio.open(self.paths[peer])
                if len(self._readers) > 32:
                    self._readers.popitem(last=False)[1].close()
            self._readers.move_to_end(peer)
            ys, xs = bounds
            value = (
                self._readers[peer]
                .read(
                    1,
                    window=Window(
                        xs.start, ys.start, xs.stop - xs.start, ys.stop - ys.start
                    ),
                    masked=True,
                )
                .astype(float)
                .filled(np.nan)
            )
            self.stats["peer_reads"] += 1
            self.stats["peer_read_seconds"] += time.perf_counter() - started
            self.stats["peak_peer_window_bytes"] = max(
                self.stats["peak_peer_window_bytes"], value.nbytes
            )
            yield value

    @staticmethod
    def _close(readers):
        for reader in readers.values():
            reader.close()
        readers.clear()

    def close(self):
        self._close(self._writers)
        self._close(self._readers)
