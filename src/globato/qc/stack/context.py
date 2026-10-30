"""In-process working DEMs for Stack QC.

Each filter pass compares source cells with an interpolated working DEM built
from the module's own sources: an ordinary MultiStack accumulator followed by
the configured ``ms_binary_cudem`` interpolator. Everything runs in the
calling process, so the fetchez registries, caches, patches and logging that
the main build uses also apply to the working DEMs (barrier lookup,
dist2coast reuse, errors in the main log).
"""

import contextlib
import logging
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import rasterio

from .io import read, write, sha, fingerprint
from .stacking import rebuild

logger = logging.getLogger(__name__)


class WorkingSurface:
    """Build or reuse the working DEM for one rejection-mask state.

    ``stack`` is a MultiStackHook used only for ``create_accumulator``.
    ``interpolators`` is a factory returning fresh raster hooks for each build,
    so no runtime state is carried from one build to the next. ``cache_root``
    is where raster hooks resolve shared dependencies (``auto_barriers``,
    dist2coast), the same place the main build uses.
    """

    def __init__(self, stack, interpolators, sources, region, cache_root, workers=1):
        self.stack = stack
        self.interpolators = interpolators
        self.sources = sources
        self.region = region
        self.cache_root = str(cache_root)
        self.workers = workers
        self.baseline = None
        self.previous = None

    def __call__(self, manifest, output):
        if manifest and read(manifest)["rejected_source_cells"] == 0 and self.baseline:
            return self.baseline
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        done = output / "COMPLETE.json"
        if done.exists():
            d = read(done)
            if sha(d["dem"]) != d["sha256"]:
                raise RuntimeError(f"Cached working DEM changed: {d['dem']}")
            if fingerprint(d["state"]) != d["state_fingerprint"]:
                raise RuntimeError(f"Cached working FusionState changed: {d['state']}")
            self.previous = d
            if manifest is None:
                self.baseline = d["dem"]
            return d["dem"]
        # An interrupted build cannot be resumed with a different mask; keep
        # its files for inspection and start clean.
        work = output / "work"
        if work.exists():
            work.rename(output / ("interrupted_" + str(time.time_ns())))
        work.mkdir()
        d = self._build(manifest, work.resolve())
        for field, filename in (("dem", "dem.tif"), ("state", "fusion.tif")):
            source = Path(d[field]).resolve()
            if source.is_relative_to(work.resolve()):
                target = output / filename
                shutil.move(source, target)
                d[field] = str(target.resolve())
        d["state_fingerprint"] = fingerprint(d["state"])
        shutil.rmtree(work)
        write(done, d)
        self.previous = d
        logger.info(
            "Stack QC working DEM ready: %s; interpolation %.1f s",
            d["timings"],
            d["interpolation_seconds"],
        )
        if manifest is None:
            self.baseline = d["dem"]
        return d["dem"]

    def _build(self, manifest, work):
        masks = read(manifest) if manifest else None
        previous = self.previous
        if previous and fingerprint(previous["state"]) != previous["state_fingerprint"]:
            raise RuntimeError("Cached working FusionState changed")
        for rec in self.sources:
            if fingerprint(rec["path"]) != rec["fingerprint"]:
                raise RuntimeError(f"QC source grid changed during QC: {rec['path']}")
            if masks and (
                sha(masks["masks"][rec["source_id"]]["mask"])
                != masks["masks"][rec["source_id"]]["mask_sha256"]
            ):
                raise RuntimeError("QC rejection mask changed during QC")
        state = str(work / "fusion.tif")
        accumulator = self.stack.create_accumulator(
            self.region, output=str(work / "stack.tif"), state=state, sparse_state=True
        )
        try:
            details = rebuild(
                accumulator, self.sources, masks, previous, workers=self.workers
            )
            details.update(state=state, masks=masks)
            if previous and details["timings"]["changed_blocks"] == 0:
                accumulator.close()
                Path(state).unlink()
                details.update(
                    state=previous["state"],
                    dem=previous["dem"],
                    sha256=previous["sha256"],
                    interpolation_seconds=0.0,
                )
                logger.info(
                    "Stack QC: stack unchanged; reusing the previous working DEM"
                )
                return details
            started = time.monotonic()
            stack_path = accumulator.finalize(statistics=False)
            details["timings"]["finalizing_seconds"] = time.monotonic() - started
        finally:
            accumulator.close()
        details.update(self._interpolate(stack_path, work))
        return details

    def _interpolate(self, stack_path, work):
        """Run the interpolator chain on the working stack, inside ``work``."""
        # Raster hooks write scratch files under ./tmp and resolve shared
        # dependencies below the module's outdir. Give them the work folder
        # and the shared cache root.
        mod = SimpleNamespace(
            region=self.region, name="stack_qc_context", outdir=self.cache_root
        )
        entries = [
            (mod, dict(dst_fn=stack_path, artifacts={"multi_stack": stack_path}))
        ]
        hooks = self.interpolators()
        started = time.monotonic()
        try:
            with contextlib.chdir(work):
                for i, hook in enumerate(hooks):
                    for field in ("output", "output_path"):
                        if hasattr(hook, field):
                            setattr(
                                hook,
                                field,
                                str(work / f"{i:02}_{hook.name.replace('-', '_')}.tif"),
                            )
                    if hasattr(hook, "keep_steps"):
                        hook.keep_steps = False
                    entries = hook.run(entries)
        finally:
            for hook in hooks:
                teardown = getattr(hook, "teardown", None)
                if callable(teardown):
                    teardown()
        if len(entries) != 1:
            raise RuntimeError("Stack QC working DEM chain must produce one DEM")
        dem = Path(entries[0][1]["dst_fn"]).resolve()
        if not dem.is_file():
            raise RuntimeError(f"Stack QC working DEM was not written: {dem}")
        with rasterio.open(dem) as ds:
            if ds.count != 1:
                raise RuntimeError("Stack QC working DEM is not a single-band DEM")
        seconds = time.monotonic() - started
        logger.info("Stack QC working DEM interpolated in %.1f s", seconds)
        return dict(dem=str(dem), sha256=sha(dem), interpolation_seconds=seconds)


def cache_root_for(module):
    """The shared cache root raster hooks use for barriers and dist2coast."""
    outdir = getattr(module, "outdir", None)
    if outdir:
        return os.path.abspath(outdir)
    return os.path.dirname(os.path.abspath(module._outdir))
