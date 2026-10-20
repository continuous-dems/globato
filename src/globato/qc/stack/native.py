"""Native Globato interpolation callback and seven-band mask application.

Temporary native work runs in a spawned process: native hooks may change the
working directory or use relative scratch paths. No custom interpolator is used.
"""

import copy
import logging
import multiprocessing
import os
import shutil
import time
import traceback
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import rasterio
from .io import read, write, sha, fingerprint
from .stacking import rebuild

logger = logging.getLogger(__name__)


def masked_stream(path, mask_path=None):
    from globato.hooks.sinks.side_stack import read_fusion_state_stream

    stream = read_fusion_state_stream(path)
    if mask_path is None:
        yield from stream
        return
    with rasterio.open(mask_path) as mask:
        profile = next(stream)
        if (
            profile["height"],
            profile["width"],
            profile["transform"],
            rasterio.crs.CRS.from_user_input(profile["crs"]),
        ) != (mask.height, mask.width, mask.transform, mask.crs):
            raise ValueError("QC mask does not match the original seven-band grid")
        yield profile
        for item in stream:
            win, buffer, data, ndv, transform = item[:5]
            flags = mask.read(1, window=win)
            if data.ndim != 3 or data.shape[0] != 7 or flags.shape != data.shape[1:]:
                raise ValueError("Invalid FusionState chunk")
            if np.any((flags != 0) & (flags != 1)):
                raise ValueError("Invalid source rejection flags")
            if flags.any():
                data = data.copy()
                data[:, flags == 1] = 0.0
            yield (win, buffer, data, ndv, transform, *item[5:])


def _worker(stack, post, sources, region, manifest, work, previous):
    work = Path(work)
    try:
        os.chdir(work)
        with (
            (work / "native.log").open("w") as log,
            redirect_stdout(log),
            redirect_stderr(log),
            ExitStack() as cleanup,
        ):
            logging.basicConfig(level=logging.INFO, stream=log, force=True)
            masks = read(manifest) if manifest else None
            if (
                previous
                and fingerprint(previous["state"]) != previous["state_fingerprint"]
            ):
                raise RuntimeError("Cached native QC FusionState changed")
            for rec in sources:
                if fingerprint(rec["path"]) != rec["fingerprint"]:
                    raise RuntimeError("Prepared source changed during QC")
                if (
                    masks
                    and sha(masks["masks"][rec["source_id"]]["mask"])
                    != masks["masks"][rec["source_id"]]["mask_sha256"]
                ):
                    raise RuntimeError("Prepared rejection mask changed")
            state = str(work / "fusion.tif")
            accumulator = stack.create_accumulator(
                region, output=str(work / "stack.tif"), state=state
            )
            cleanup.callback(accumulator.close)
            details = rebuild(accumulator, sources, masks, previous)
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
                    "QC stack unchanged: reusing preceding context; interpolation skipped"
                )
                write(work / "COMPLETE.json", details)
                return
            finalizing = time.monotonic()
            stack_path = accumulator.finalize(statistics=False)
            details["timings"]["finalizing_seconds"] = time.monotonic() - finalizing
            # The configured native raster operators consume a normal MultiStack.
            mod = SimpleNamespace(
                region=region, name="stack_qc_context", outdir=str(work)
            )
            one = [
                (mod, dict(dst_fn=stack_path, artifacts={"multi_stack": stack_path}))
            ]
            interpolation_started = time.monotonic()
            logger.info("QC interpolation starting: configured native raster chain")
            for i, hook in enumerate(post):
                cleanup.callback(hook.teardown)
                for field in ("output", "output_path"):
                    if hasattr(hook, field):
                        setattr(
                            hook,
                            field,
                            str(work / f"{i:02}_{hook.name.replace('-', '_')}.tif"),
                        )
                if hasattr(hook, "keep_steps"):
                    hook.keep_steps = False
                one = hook.run(one)
            if len(one) != 1:
                raise RuntimeError("Native QC context must produce one DEM")
            dem = Path(one[0][1]["dst_fn"]).resolve()
            with rasterio.open(dem) as ds:
                if ds.count != 1:
                    raise RuntimeError("Native QC context is not an interpolated DEM")
            details.update(
                dem=str(dem),
                sha256=sha(dem),
                interpolation_seconds=time.monotonic() - interpolation_started,
            )
            logger.info(
                "QC interpolation complete: %.2f seconds",
                details["interpolation_seconds"],
            )
            write(work / "COMPLETE.json", details)
    except BaseException:
        (work / "ERROR.txt").write_text(traceback.format_exc())
        raise


class NativeBuilder:
    """Uses configured native owners; never discovers or reprocesses sources."""

    def __init__(self, stack, post, sources, region):
        self.stack = stack
        self.post = post
        self.sources = sources
        self.region = region
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
                raise RuntimeError("Cached native QC DEM changed")
            if fingerprint(d["state"]) != d["state_fingerprint"]:
                raise RuntimeError("Cached native QC FusionState changed")
            self.previous = d
            return d["dem"]
        # Each state owns its scratch files; an interrupted state cannot resume
        # an accumulator with a different source mask. Preserve its log first.
        work = output / "work"
        if work.exists():
            work.rename(output / ("interrupted_" + str(time.time_ns())))
        work.mkdir()
        ctx = multiprocessing.get_context("spawn")

        def configured(hook):
            clone = copy.copy(hook)
            # A reused native RasterHook may retain the last module/stream in
            # current_mod. That is runtime state, not interpolation configuration.
            clone.__dict__ = copy.deepcopy(
                {
                    k: v
                    for k, v in vars(hook).items()
                    if k not in {"current_mod", "_accumulator"}
                }
            )
            return clone

        process = ctx.Process(
            target=_worker,
            args=(
                self.stack,
                [configured(h) for h in self.post],
                self.sources,
                self.region,
                str(manifest) if manifest else None,
                str(work.resolve()),
                self.previous,
            ),
        )
        process.start()
        started = time.monotonic()
        try:
            while process.is_alive():
                process.join(10)
                if process.is_alive():
                    latest = "awaiting native stage message"
                    logfile = work / "native.log"
                    if logfile.exists():
                        with logfile.open("rb") as stream:
                            stream.seek(max(0, logfile.stat().st_size - 2048))
                            lines = (
                                stream.read()
                                .decode("utf-8", errors="replace")
                                .splitlines()
                            )
                        if lines:
                            latest = lines[-1]
                    logger.info(
                        "Native QC context active: %.0f s; %s; %s",
                        time.monotonic() - started,
                        latest,
                        logfile,
                    )
        except BaseException:
            process.terminate()
            process.join()
            raise
        if process.exitcode:
            raise RuntimeError(f"Native QC interpolation failed: {work / 'ERROR.txt'}")
        d = read(work / "COMPLETE.json")
        for field, filename in (("dem", "dem.tif"), ("state", "fusion.tif")):
            source = Path(d[field]).resolve()
            if source.is_relative_to(work.resolve()):
                target = output / filename
                shutil.move(source, target)
                d[field] = str(target.resolve())
        d["state_fingerprint"] = fingerprint(d["state"])
        if (work / "native.log").exists():
            shutil.move(work / "native.log", output / "native.log")
        shutil.rmtree(work)
        write(done, d)
        self.previous = d
        logger.info(
            "QC context ready: %s; interpolation %.2f s",
            d["timings"],
            d["interpolation_seconds"],
        )
        if manifest is None:
            self.baseline = d["dem"]
        return d["dem"]
