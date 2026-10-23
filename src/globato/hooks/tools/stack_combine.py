#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.hooks.tools.stack_combine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assemble a MultiStack from completed, compatible side-stack FusionStates.

This is a collection-stage hook: source processing and side-stack caching must
finish before it runs. It deliberately reads raster artifacts directly rather
than routing them through another point/raster stream.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import rasterio
from rasterio.crs import CRS

from fetchez.hooks import FetchHook
from fetchez.spatial import parse_increments
from fetchez.utils import str2bool

from ..sinks.multi_stack import MultiStackAccumulator, VALID_STRATEGIES
from ..transforms.point_pixels import FUSION_BANDS, FUSION_STATE_VERSION

logger = logging.getLogger(__name__)


class StackCombine(FetchHook):
    """Build a finalized MultiStack and its FusionState from side-stack artifacts.

    All inputs must match the requested destination grid exactly. Both the
    additive and priority-based strategies reuse MultiStackAccumulator's
    reduction logic. Rebuilds from side-stacks by default (no resume/deltas).

    Outputs are available immediately after ``run()`` so later collection hooks
    (including focus-sink) can consume them; ``teardown()`` only cleans up.
    """

    name = "stack_combine"
    meta_stage = "collection"
    meta_category = "system"
    meta_desc = "Combine cached side-stack FusionStates into a MultiStack."

    def __init__(
        self,
        res="1s",
        output="multi_stack_output.tif",
        *,
        state=None,
        crs="EPSG:4326",
        strategy="mean",
        weight_threshold="1",
        compress_state=True,
        ndv=-9999,
        source_artifact="side-stack",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.res = res
        self.output = output
        self.state = state
        self.crs = crs
        self.strategy = str(strategy).lower()
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(f"Invalid stack strategy {strategy!r}")
        self.weight_threshold = weight_threshold
        self.compress_state = str2bool(compress_state)
        self.ndv = float(ndv)
        self.source_artifact = source_artifact
        self._accumulator = None
        self._temporary_paths = []

    @staticmethod
    def _region_name(path, region):
        """Append the region suffix once, preserving the original configured path."""
        path = Path(path)
        suffix = region.format("fn")
        if suffix not in path.stem:
            path = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
        return path.absolute()

    @staticmethod
    def _temp_path(destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(
            prefix=f".{destination.stem}-", suffix=".tif", dir=destination.parent
        )
        os.close(fd)
        os.unlink(name)
        return name

    @staticmethod
    def _sources(entries, artifact):
        """Pick declared side-stack artifacts, ignoring repeated references."""
        selected = []
        seen = set()
        for _, entry in entries:
            path = (entry.get("artifacts") or {}).get(artifact)
            if path is None:
                continue
            if not isinstance(path, (str, os.PathLike)):
                raise TypeError("side-stack artifact must be one raster filename")
            canonical = str(Path(path).resolve())
            if canonical not in seen:
                seen.add(canonical)
                selected.append(canonical)
        return selected

    @staticmethod
    def _validate_source(src, path, accumulator):
        """Reject non-FusionState rasters or incompatible grid definitions."""
        problems = []
        if src.count != len(FUSION_BANDS):
            problems.append("band count")
        if tuple(src.descriptions) != tuple(FUSION_BANDS):
            problems.append("FusionState band descriptions")
        if src.tags().get("GLOBATO_DATATYPE") != "FUSION_STATE":
            problems.append("GLOBATO_DATATYPE")
        if src.tags().get("GLOBATO_FUSION_VERSION") != str(FUSION_STATE_VERSION):
            problems.append("FusionState version")
        if (src.width, src.height) != (accumulator.xcount, accumulator.ycount):
            problems.append("dimensions")
        if src.transform != accumulator.transform:
            problems.append("geotransform")
        expected_crs = CRS.from_user_input(accumulator.crs) if accumulator.crs else None
        if src.crs != expected_crs:
            problems.append("CRS")
        if any(dtype != "float64" for dtype in src.dtypes):
            problems.append("dtype (expected float64)")
        if problems:
            raise ValueError(f"Incompatible side-stack {path!r}: {', '.join(problems)}")

    def _grid(self, region):
        inc = parse_increments(self.res, crs=self.crs)
        if inc is None or inc <= 0:
            raise ValueError(f"Invalid stack resolution: {self.res!r}")
        return inc

    def run(self, entries):
        sources = self._sources(entries, self.source_artifact)
        if not sources:
            logger.info("stack_combine: no side-stack artifacts; nothing to assemble")
            return entries

        region = next(
            (mod.region for mod, _ in entries if getattr(mod, "region", None)),
            None,
        )
        if region is None:
            raise ValueError("stack_combine requires a module region")

        output_path = self._region_name(self.output, region)
        state_path = (
            self._region_name(self.state, region)
            if self.state is not None
            else output_path.with_name(output_path.stem + ".fusion.tif")
        )
        if output_path == state_path:
            raise ValueError("stack_combine output and state paths must differ")

        # Avoid changing existing outputs until every input has been reduced and
        # the finalized MultiStack has been successfully written.
        work_output = self._temp_path(output_path)
        work_state = self._temp_path(state_path)
        self._temporary_paths.extend((work_output, work_state))
        try:
            accumulator = MultiStackAccumulator(
                region=region,
                x_inc=self._grid(region),
                y_inc=self._grid(region),
                output_fn=work_output,
                state_fn=work_state,
                resume=False,
                strategy=self.strategy,
                weight_threshold=self.weight_threshold,
                crs=self.crs,
                compress_state=self.compress_state,
            )
            self._accumulator = accumulator

            for path in sources:
                with rasterio.open(path) as src:
                    self._validate_source(src, path, accumulator)
                    logger.debug("Combining side-stack: %s", path)
                    for _, window in src.block_windows(1):
                        data = src.read(window=window)
                        state = {
                            name: data[index] for index, name in enumerate(FUSION_BANDS)
                        }
                        accumulator.update_state(state, window)
                    # Use the side-stack's immutable cache identity when present.
                    # Path is the fallback for externally generated FusionStates.
                    source_key = src.tags().get("GLOBATO_SIDESTACK_HASH") or path
                    accumulator.mark_registered(source_key)

            accumulator.finalize(ndv=self.ndv)
            accumulator.close()
            self._accumulator = None
            os.replace(work_state, state_path)
            os.replace(work_output, output_path)
            self._temporary_paths.clear()
        finally:
            self.teardown()

        for _, entry in entries:
            artifacts = entry.setdefault("artifacts", {})
            artifacts[self.name] = str(output_path)
            artifacts["multi_stack"] = str(output_path)
            artifacts["fusion-state"] = str(state_path)

        logger.info(
            "Combined %d side-stacks: %s (state: %s)",
            len(sources),
            output_path,
            state_path,
        )
        return entries

    def teardown(self):
        """Release resources and remove staging files (safe after run/failure)."""
        if self._accumulator is not None:
            self._accumulator.close()
            self._accumulator = None
        for path in self._temporary_paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        self._temporary_paths.clear()
