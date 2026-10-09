#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.streams.base
~~~~~~~~~~~~~~~~~~~~

:copyright: (c) 2025 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
from typing import cast, Optional

from fetchez.streams.base import BaseStream

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class GlobatoStream(BaseStream):
    """The Globato point stream."""

    def __init__(
        self, modules, region=None, target_srs=None, ignore_failures=False, **kwargs
    ):
        super().__init__(modules, region=region, ignore_failures=ignore_failures)
        self.target_srs = target_srs
        self.stream_type = "point"

        if self.target_srs:
            self.pipe(
                {"name": "stream-reproject", "args": {"dst_srs": self.target_srs}}
            )

    def to_raster(self, increment, want_sums=False):
        """Transitions the point stream into a GlobatoRasterStream."""

        self.pipe(
            {
                "name": "points2pixels",
                "args": {
                    "x_inc": increment,
                    "y_inc": increment,
                    "want_sums": want_sums,
                },
            }
        )

        # Mutate this instance into a RasterStream!
        self.__class__ = GlobatoRasterStream
        self.increment = increment
        self.stream_type = "raster"
        return self

    # --- Terminal Sinks ---
    def to_dataframe(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Consumes the pipeline and returns a Pandas DataFrame."""

        chunks = []
        count = 0

        # Iterating over 'self' triggers the background Thread/Queue!
        for chunk in self:
            if chunk is None or len(chunk) == 0:
                continue
            chunks.append(pd.DataFrame(chunk))
            count += len(chunk)
            if limit and count >= limit:
                break

        if not chunks:
            return pd.DataFrame()

        df = pd.concat(chunks, ignore_index=True)
        return df.head(limit) if limit else df

    def to_numpy(self) -> np.recarray:
        """Consumes the pipeline and returns a Numpy Recarray."""
        chunks = list(self)
        if not chunks:
            return cast(
                np.recarray, np.array([], dtype=[("x", "f8"), ("y", "f8"), ("z", "f4")])
            )
        return cast(np.recarray, np.concatenate(chunks))


class GlobatoRasterStream(BaseStream):
    """A gridded raster stream for Globato."""

    def to_points(self):
        """Transitions the raster stream back into a GlobatoStream."""

        self.pipe({"name": "pixels2points", "args": {}})

        # Mutate back to a point stream!
        self.__class__ = GlobatoStream
        self.stream_type = "point"
        return self

    def write_dem(self, output_path):
        """Terminal sink that writes the raster chunks to a GeoTIFF."""

        self.pipe({"name": "write_raster", "args": {"output_path": output_path}})

        # Exhaust the stream to force execution
        for _chunk in self:
            pass

        return output_path
