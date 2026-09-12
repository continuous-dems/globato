#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.hooks.transforms.dynamic_weights
~~~~~~~~~~~~~

Adjust weight values based on point-stream array fields

Available methods
-----------------
inverse
    w = scale / (src + offset)
    Inverse-confidence weighting: larger source values -> larger weights.
    `offset` prevents division by zero (set it just above -min(src)).

inverse_squared
    w = scale / (src**2 + offset)
    Common for uncertainty fields: w = 1/U**2. Larger values get much
    smaller weights. `offset` keeps the denominator positive.

linear_invert
    w = offset - (src * scale)
    Linearly decreasing weights. Negative results are clipped to the
    floor, so this is effectively a sawtooth/cliff profile.

linear
    w = src * scale
    Plain linear scaling.

chronological
    w = exp(-ln(2) * (seed - src) / half_life), with half_life = scale
    Exponential recency decay around the anchor year `seed` (typically the
    current year). Entries `scale` years old receive half weight; entries
    older than that decay smoothly toward (but never reach) zero, while
    entries newer than `anchor` scale *up* exponentially. Recommended
    combo: `seed=<current year>`, `scale=5` (5-year half-life),
    `offset=0.01` (acts as the weight floor), `cap=0.5` (weight ceiling).

In all cases the result is clamped to [floor, cap] where:
    floor = offset      for 'chronological' (offset doubles as the floor)
    floor = 0.0001      for all other methods
    cap   = cap         (None disables the ceiling)

:copyright: (c) 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import numpy as np
import logging
from fetchez.hooks import FetchHook
from fetchez.utils import str2bool, float_or

logger = logging.getLogger(__name__)


class DynamicWeight(FetchHook):
    """Dynamically calculates point weights based on confidence or uncertainty fields."""

    name = "dynamic-weight"
    meta_stage = "stream"
    meta_category = "stream-transform"
    meta_desc = "Calculate weights dynamically from other stream columns."

    def __init__(
        self,
        source_field="confidence",
        method="inverse",
        offset=1.0,
        scale=1.0,
        seed=0,
        cap=None,
        on_entry=False,
        **kwargs,
    ):
        """
        Args:
            source_field: The column to read from (e.g., 'confidence', 'u', 'z').
            method: 'inverse', 'inverse_squared', 'linear_invert', 'linear',
                    or 'chronological'.
            offset: An offset to prevent division by zero, or a ceiling/floor.
                    For 'chronological' this doubles as the minimum weight
                    (recommend a small value such as 0.01).
            scale: A multiplier for the resulting weight. For 'chronological'
                   this is the half-life in years (recommend 5 for a
                   5-year decay).
            seed: Anchor year for the chronological method (typically the
                  current year).
            cap: Maximum allowed weight (e.g. 0.5). None disables the ceiling.
            on_entry: Perform dynamic weighting on an entry instead of a stream.
                      When set, source_field is the entry key.
        """
        super().__init__(**kwargs)
        self.source_field = source_field
        self.method = method
        self.offset = float(offset)
        self.scale = float(scale)
        self.seed = float(seed)
        self.cap = float_or(cap)
        self.on_entry = str2bool(on_entry)

        if self.on_entry:
            self.stage = "manifest"

    def _process_arr_or_val(self, arr_or_val):
        arr = np.asarray(arr_or_val, dtype=np.float32)
        eps = np.finfo(np.float32).eps

        if self.method == "inverse":
            # w = scale / (src + offset)
            denom = arr + self.offset
            denom = np.where(np.abs(denom) < eps, eps, denom)
            new_w = self.scale / denom

        elif self.method == "inverse_squared":
            # Common for uncertainty: w = 1 / (U^2)
            denom = (arr**2) + self.offset
            denom = np.where(denom < eps, eps, denom)
            new_w = self.scale / denom

        elif self.method == "linear_invert":
            # w = offset - (src * scale)
            new_w = self.offset - (arr * self.scale)

        elif self.method == "chronological":
            # Exponential recency decay: w = exp(-ln(2) * (seed - src) / half_life)
            # 'seed' is the anchor/current year and 'scale' the half-life in
            # years. Newer-than-anchor values grow exponentially and are
            # bounded by `cap`; older values decay smoothly toward zero.
            half_life = max(self.scale, eps)
            new_w = np.exp(-np.log(2.0) * (self.seed - arr) / half_life)

        else:  # linear
            new_w = arr * self.scale

        # Clamp to [floor, cap]: strictly positive floor guarantees w > 0,
        # and the (optional) cap prevents any single entry from dominating.
        floor = self.offset if self.method == "chronological" else 0.0001
        if self.cap is not None:
            new_w = np.clip(new_w, floor, self.cap)
        else:
            new_w = np.clip(new_w, floor, None)

        return new_w

    def _process_stream(self, stream):
        for chunk in stream:
            if chunk is None or len(chunk) == 0:
                continue

            if self.source_field not in chunk.dtype.names:
                yield chunk
                continue

            src_arr = chunk[self.source_field].astype(np.float32)
            new_w = self._process_arr_or_val(src_arr)

            chunk["w"] *= new_w

            yield chunk

    def _process_entry(self, entry):
        entry_val = entry.get(self.source_field)
        if entry_val is None and "metadata" in entry:
            entry_val = entry["metadata"].get(self.source_field)

        logger.debug("entry '%s' = %r", self.source_field, entry_val)
        if not float_or(entry_val):
            return None
        return self._process_arr_or_val(float(entry_val))

    def run(self, entries):
        for mod, entry in entries:
            if self.on_entry:
                entry_weight = self._process_entry(entry)
                logger.debug("computed entry weight: %r", entry_weight)
                if entry_weight is not None:
                    entry["weight"] = entry_weight

            elif self.is_point_stream(entry):
                stream = entry.get("stream")
                entry["stream"] = self._process_stream(stream)
        return entries
