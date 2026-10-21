#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.hooks.filters.dropclass
~~~~~~~~~~~~~

Drops the classification from the point stream

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import numpy as np
from .base import GlobatoFilter

logger = logging.getLogger(__name__)


class DropClass(GlobatoFilter):
    """Destructive filter: Removes points with specific classifications.

    Usage: --hook drop_class:classes=7/18
    """

    name = "drop-class"
    meta_desc = "Drop specified classes from the point stream"
    meta_aliases = ["drop_class"]

    def __init__(self, classes="7/12", stack_partition=None, **kwargs):
        super().__init__(**kwargs)
        self.target_classes = [int(x) for x in str(classes).split("/")]
        self.stack_partition = (
            str(stack_partition).strip() if stack_partition is not None else None
        )
        if self.stack_partition == "":
            raise ValueError("stack_partition must be a nonempty label")

    def run(self, entries):
        # Tag the lazy stream before the sink checks its duplicate registry.
        # Unlabelled global class cleanup must preserve a local branch label.
        if self.stack_partition is not None:
            for _mod, entry in entries:
                if self.is_point_stream(entry):
                    entry["stack_partition"] = self.stack_partition
        return super().run(entries)

    def filter_chunk(self, chunk):
        mask = np.isin(chunk["classification"], self.target_classes)

        if self.invert:
            # Keep these classes
            keep_mask = mask
        else:
            # Drop these classes
            keep_mask = ~mask

        logger.debug(f"Dropped {np.count_nonzero(~keep_mask)} points")
        if np.count_nonzero(keep_mask) > 0:
            return chunk[keep_mask]
        else:
            return chunk[0:0]
