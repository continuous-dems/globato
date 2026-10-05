#!/usr/bin/env python

"""
globato.hooks.filters.vector_crop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Crops stream data using a polygon vector mask.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging

import numpy as np
import shapely
from fetchez.utils import str2bool
from pyogrio.raw import read
from shapely.geometry import MultiPolygon
from shapely.vectorized import contains as vec_contains

from .base import GlobatoFilter

logger = logging.getLogger(__name__)


class VectorCrop(GlobatoFilter):
    """Crops the point stream using a vector polygon.

    Usage:
      --hook vector_crop:vector=mask.shp                (Hard: Deletes points outside)
      --hook vector_crop:vector=mask.shp:soft=True      (Soft: Classifies points outside as 7)
    """

    name = "vector_crop"
    meta_desc = "Crop stream data using a polygon vector."

    def __init__(self, vector=None, soft=False, **kwargs):
        super().__init__(**kwargs)
        self.vector_path = vector
        self.soft = str2bool(soft)
        self.geometry = None

    def setup(self, mod, entry):
        """Load the vector geometry once before the stream begins."""

        if not self.vector_path:
            logger.warning("No vector path provided to vector_crop. Skipping.")
            return False

        try:
            meta, geometry_wkb, field_data = read(self.vector_path)
            geoms = shapely.from_wkb(geometry_wkb)

            valid_geoms = [g for g in geoms if g and not g.is_empty]

            if not valid_geoms:
                logger.warning(
                    f"Vector file {self.vector_path} contains no valid geometries."
                )
                return False

            self.geometry = (
                MultiPolygon(valid_geoms) if len(valid_geoms) > 1 else valid_geoms[0]
            )

        except Exception as e:
            logger.error(f"Failed to load vector {self.vector_path}: {e}")
            return False

        return True

    def filter_chunk(self, chunk):
        """Process the chunk of stream data."""

        if len(chunk) == 0:
            return chunk if not self.soft else np.zeros(0, dtype=bool)

        # True if inside, False if outside
        inside_mask = vec_contains(self.geometry, chunk["x"], chunk["y"])

        if self.soft:
            return ~inside_mask
        else:
            return chunk[inside_mask]
