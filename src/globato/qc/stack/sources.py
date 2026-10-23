"""Read native source windows and write separate rejection masks.

Known acquisition groups are pooled once per survey. Sources without survey IDs
can supply one pooled local-terrain layer, never independent survey votes.
"""

from collections import OrderedDict
from pathlib import Path
import hashlib
import numpy as np
import rasterio
from rasterio.windows import Window
from .io import write, sha, digest, windows


class SourcePool:
    def __init__(self, inventory):
        self.inv = inventory
        self.shape = tuple(inventory["grid"]["shape"])
        self.records = inventory["sources"]
        self.groups = sorted(
            {r["group"] for r in self.records if r["group"] is not None}
        )
        self.group_index = {g: i for i, g in enumerate(self.groups)}
        self.by_group = {
            g: [r for r in self.records if r["group"] == g]
            for g in [*self.groups, None]
        }
        self.open = OrderedDict()
        # At most 32 MiB of decoded source measurements, reused only inside one
        # processing window. A different window clears this cache entirely.
        self._window_key = None
        self._window_cache = OrderedDict()
        self._window_cache_bytes = 0
        self._window_cache_limit = 32 * 1024 * 1024
        self.source_read_count = 0
        self.source_cache_hits = 0
        # Native context blocks are 256x256. Missing index means ordinary reads.
        self.occupied = {
            r["source_id"]: {tuple(map(int, b.split(","))) for b in r["active_blocks"]}
            for r in self.records
            if "active_blocks" in r
        }

        # The native context already records which 256-cell blocks had source
        # measurements. Index those once, in original source order, instead of
        # checking every record for every window on large CUDEM grids.
        self._block_sources = {}
        self._unindexed = []
        for position, r in enumerate(self.records):
            blocks = self.occupied.get(r["source_id"])
            if blocks is None:
                self._unindexed.append(position)
            else:
                for block in blocks:
                    self._block_sources.setdefault(block, set()).add(position)
        self._active_window_key = None
        self._active_window_records = None

    @staticmethod
    def _key(s):
        return tuple((axis.start, axis.stop) for axis in s)

    def active_records(self, s):
        """Candidate sources for a window, always in native source order."""
        key = self._key(s)
        if self._active_window_key == key:
            return self._active_window_records
        positions = set(self._unindexed)
        for y in range(s[0].start // 256, (s[0].stop - 1) // 256 + 1):
            for x in range(s[1].start // 256, (s[1].stop - 1) // 256 + 1):
                positions.update(self._block_sources.get((y, x), ()))
        active = []
        for i in sorted(positions):
            r = self.records[i]
            dy, dx = r["offset"]
            height, width = r["grid"]["shape"]
            if (
                dy < s[0].stop
                and dy + height > s[0].start
                and dx < s[1].stop
                and dx + width > s[1].start
            ):
                active.append(r)
        self._active_window_key = key
        self._active_window_records = active
        return active

    def intersects(self, r, s):
        blocks = self.occupied.get(r["source_id"])
        return blocks is None or any(
            (y, x) in blocks
            for y in range(s[0].start // 256, (s[0].stop - 1) // 256 + 1)
            for x in range(s[1].start // 256, (s[1].stop - 1) // 256 + 1)
        )

    def ds(self, path, mode="r"):
        key = (str(path), mode)
        if key not in self.open:
            self.open[key] = rasterio.open(path, mode)
            while len(self.open) > 40:
                self.open.popitem(last=False)[1].close()
        self.open.move_to_end(key)
        return self.open[key]

    def close(self):
        for ds in self.open.values():
            ds.close()
        self.open.clear()
        self._window_cache.clear()
        self._window_cache_bytes = 0
        self._window_key = None
        self._active_window_key = None
        self._active_window_records = None

    def source(self, r, s, manifest=None):
        """One source's observations. Never cache a rejection decision."""
        key = self._key(s)
        if key != self._window_key:
            self._window_cache.clear()
            self._window_cache_bytes = 0
            self._window_key = key
        sid = r["source_id"]
        cached = self._window_cache.get(sid)
        if cached is not None:
            self._window_cache.move_to_end(sid)
            self.source_cache_hits += 1
            mean, valid, weight = cached
        else:
            sh = tuple(i.stop - i.start for i in s)
            mean = np.full(sh, np.nan, "float32")
            weight = np.zeros(sh, "float32")
            dy, dx = r["offset"]
            height, width = r["grid"]["shape"]
            y0, x0 = max(s[0].start, dy), max(s[1].start, dx)
            y1, x1 = min(s[0].stop, dy + height), min(s[1].stop, dx + width)
            if y1 > y0 and x1 > x0 and self.intersects(r, s):
                ds = self.ds(r["path"])
                w = Window(x0 - dx, y0 - dy, x1 - x0, y1 - y0)
                dst = (
                    slice(y0 - s[0].start, y1 - s[0].start),
                    slice(x0 - s[1].start, x1 - s[1].start),
                )
                b = (
                    ds.read([1, 2, 3], window=w, masked=True)
                    .astype("float32")
                    .filled(np.nan)
                )
                self.source_read_count += 1
                ok = np.isfinite(b).all(axis=0) & (b[1] > 0) & (b[2] > 0)
                mean[dst] = np.divide(
                    b[0], b[2], out=np.full(ok.shape, np.nan, "float32"), where=ok
                )
                weight[dst] = np.where(ok, b[2], 0)
            valid = np.isfinite(mean)
            size = mean.nbytes + valid.nbytes + weight.nbytes
            if size <= self._window_cache_limit:
                while (
                    self._window_cache
                    and self._window_cache_bytes + size > self._window_cache_limit
                ):
                    old = self._window_cache.popitem(last=False)[1]
                    self._window_cache_bytes -= sum(a.nbytes for a in old)
                self._window_cache[sid] = (mean, valid, weight)
                self._window_cache_bytes += size
        rejected = np.zeros(mean.shape, bool)
        if manifest is not None:
            dy, dx = r["offset"]
            height, width = r["grid"]["shape"]
            y0, x0 = max(s[0].start, dy), max(s[1].start, dx)
            y1, x1 = min(s[0].stop, dy + height), min(s[1].stop, dx + width)
            if y1 > y0 and x1 > x0 and self.intersects(r, s):
                w = Window(x0 - dx, y0 - dy, x1 - x0, y1 - y0)
                dst = (
                    slice(y0 - s[0].start, y1 - s[0].start),
                    slice(x0 - s[1].start, x1 - s[1].start),
                )
                rejected[dst] = (
                    self.ds(manifest["masks"][sid]["mask"]).read(1, window=w) == 1
                )
        # Return independent arrays, as before this optimization: downstream
        # callers may safely modify them without corrupting cached evidence.
        return mean.copy(), valid.copy(), weight.copy(), rejected

    def rejection_mask(self, r, s, manifest):
        """Read only rejection flags when elevation values are not needed."""
        sh = tuple(i.stop - i.start for i in s)
        rejected = np.zeros(sh, bool)
        dy, dx = r["offset"]
        height, width = r["grid"]["shape"]
        y0, x0 = max(s[0].start, dy), max(s[1].start, dx)
        y1, x1 = min(s[0].stop, dy + height), min(s[1].stop, dx + width)
        if y1 > y0 and x1 > x0 and self.intersects(r, s):
            w = Window(x0 - dx, y0 - dy, x1 - x0, y1 - y0)
            dst = (
                slice(y0 - s[0].start, y1 - s[0].start),
                slice(x0 - s[1].start, x1 - s[1].start),
            )
            rejected[dst] = (
                self.ds(manifest["masks"][r["source_id"]]["mask"]).read(1, window=w)
                == 1
            )
        return rejected

    def values(self, s, manifest=None, *, local_context=False):
        sh = tuple(i.stop - i.start for i in s)
        groups = (
            [*self.groups, None]
            if local_context and self.by_group[None]
            else self.groups
        )
        values = np.full((len(groups), *sh), np.nan, "float32")
        unknown = np.zeros(sh, bool)
        # Only visit sources intersecting this window. Keep group and source
        # reduction order identical to the original implementation.
        active_groups = {g: [] for g in [*self.groups, None]}
        for r in self.active_records(s):
            active_groups[r["group"]].append(r)
        for g in [*self.groups, None]:
            if not active_groups[g]:
                continue
            total = np.zeros(sh, "float64")
            weights = np.zeros(sh, "float64")
            for r in active_groups[g]:
                z, valid, w, reject = self.source(r, s, manifest)
                valid = valid & ~reject
                if g is None:
                    unknown |= valid
                if g is not None or local_context:
                    total += np.where(valid, z, 0.0) * w
                    weights += np.where(valid, w, 0.0)
            if g in groups:
                np.divide(
                    total, weights, out=values[groups.index(g)], where=weights > 0
                )
        return values, unknown


class MaskWriter:
    """Sparse 0/1 rasters on EXACT original source grids; no seven-band copies."""

    def __init__(self, out, pool):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.pool = pool
        self.paths = {}
        self.hashes = {}
        self.counts = {}
        self.new = {}
        self.open = OrderedDict()
        for r in pool.records:
            sid = r["source_id"]
            p = self.out / (hashlib.sha256(sid.encode()).hexdigest()[:20] + ".tif")
            self.paths[sid] = p
            with rasterio.open(r["path"]) as src:
                pr = src.profile.copy()
                pr.update(
                    count=1,
                    dtype="uint8",
                    nodata=None,
                    compress="deflate",
                    predictor=1,
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                    BIGTIFF="IF_SAFER",
                    SPARSE_OK="TRUE",
                )
                with rasterio.open(p, "w", **pr) as ds:
                    ds.update_tags(
                        mask_semantics="0=retain;1=zero all seven original bands",
                        source_id=sid,
                    )
            self.hashes[sid] = hashlib.sha256()
            self.counts[sid] = 0
            self.new[sid] = 0

    def put(self, r, s, mask, previous=None):
        sid = r["source_id"]
        self.hashes[sid].update(np.packbits(mask).tobytes())
        self.counts[sid] += int(mask.sum())
        self.new[sid] += (
            int((mask & ~previous).sum()) if previous is not None else int(mask.sum())
        )
        if not mask.any():
            return
        dy, dx = r["offset"]
        h, w = r["grid"]["shape"]
        y0 = max(s[0].start, dy)
        x0 = max(s[1].start, dx)
        y1 = min(s[0].stop, dy + h)
        x1 = min(s[1].stop, dx + w)
        if y1 <= y0 or x1 <= x0:
            return
        inside = (
            slice(y0 - s[0].start, y1 - s[0].start),
            slice(x0 - s[1].start, x1 - s[1].start),
        )
        if sid not in self.open:
            self.open[sid] = rasterio.open(self.paths[sid], "r+")
            while len(self.open) > 24:
                self.open.popitem(last=False)[1].close()
        self.open.move_to_end(sid)
        self.open[sid].write(
            mask[inside].astype("uint8"),
            1,
            window=Window(x0 - dx, y0 - dy, x1 - x0, y1 - y0),
        )

    def finish(self, extra=None):
        for ds in self.open.values():
            ds.close()
        self.open.clear()
        masks = {}
        for r in self.pool.records:
            sid = r["source_id"]
            masks[sid] = dict(
                mask=str(self.paths[sid].resolve()),
                mask_sha256=sha(self.paths[sid]),
                source_path=r["path"],
                source_sha256=r["source_sha256"],
                mask_bits_sha256=self.hashes[sid].hexdigest(),
                cumulative_rejected_cells=self.counts[sid],
                newly_rejected_cells=self.new[sid],
            )
        d = dict(
            schema="stack_qc_masks_v1",
            masks=masks,
            rejected_source_cells=sum(self.counts.values()),
            newly_rejected_source_cells=sum(self.new.values()),
            signature=digest(
                [
                    [r["source_id"], masks[r["source_id"]]["mask_bits_sha256"]]
                    for r in self.pool.records
                ]
            ),
            **(extra or {}),
        )
        p = self.out / "MANIFEST.json"
        write(p, d)
        return p, d


def zero(pool, out):
    m = MaskWriter(out, pool)
    for s, _, _ in windows(pool.shape):
        z = np.zeros(tuple(i.stop - i.start for i in s), bool)
        for r in pool.records:
            m.put(r, s, z)
    return m.finish()
