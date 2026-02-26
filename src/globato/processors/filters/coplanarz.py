#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.filters.coplanarz
~~~~~~~~~~~~~

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import numpy as np
from fetchez.utils import float_or, int_or
from .base import GlobatoFilter

logger = logging.getLogger(__name__)


class CoplanarZ(GlobatoFilter):
    """Filter points that deviate from a locally fitted plane.
    Useful for removing noise from generally flat features (roads, water, plains).
    """

    name = "coplanarz"
    desc = "filter outliers that deviate from a fitted plane"

    def __init__(self, radius=10, threshold=0.5, min_neighbors=3, **kwargs):
        super().__init__(**kwargs)
        self.radius = float_or(radius, 10)
        self.threshold = float_or(threshold, 0.5)
        self.min_neighbors = int_or(min_neighbors, 3)

    def filter_chunk(self, chunk):
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            logger.error("scipy.spatial.cKDTree required for coplanar filter.")
            return None

        # Build KDTree for efficient neighbor search
        coords = np.column_stack((chunk['x'], chunk['y']))
        tree = cKDTree(coords)

        # Query neighbors within radius
        logger.info(f"Querying neighbors (radius={self.radius})...")
        indices_list = tree.query_ball_point(coords, self.radius)

        outliers = np.zeros(len(chunk), dtype=bool)
        z_vals = chunk['z']

        with tqdm(total=len(chunk), desc='Plane Fitting', leave=False) as pbar:
            for i, neighbors in enumerate(indices_list):
                pbar.update()

                # Check neighbor count (including self)
                if len(neighbors) < self.min_neighbors + 1:
                    # Treat isolated points as outliers (noise)
                    outliers[i] = True
                    continue

                ## Get neighbor coordinates
                nb_coords = coords[neighbors]
                nb_z = z_vals[neighbors]

                center_x, center_y = coords[i]

                # Setup Least Squares: Z = a*X + b*Y + c
                # A matrix columns: [x_rel, y_rel, 1]
                A = np.column_stack((
                    nb_coords[:, 0] - center_x,
                    nb_coords[:, 1] - center_y,
                    np.ones(len(neighbors))
                ))

                try:
                    # Fit plane
                    # c (coeffs[2]) is the fitted Z at (0,0) relative coordinates (the query point)
                    coeffs, residuals, rank, s = np.linalg.lstsq(A, nb_z, rcond=None)

                    fitted_z = coeffs[2]

                    # Calculate deviation of the point from the fitted plane
                    deviation = abs(z_vals[i] - fitted_z)

                    if deviation > self.threshold:
                        outliers[i] = True

                except np.linalg.LinAlgError:
                    outliers[i] = True
        return outliers
