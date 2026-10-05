#!/usr/bin/env python

"""
vizdem.modules.colorbar
~~~~~~~~~~~~~~~~~~~~~~~

Generate a color bar.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import cm

from .geohillshade import GeoHillshade

logger = logging.getLogger(__name__)


class ColorBar(GeoHillshade):
    """Generate a standalone colorbar image."""

    name = "viz_colorbar"
    default_suffix = "_colorbar"
    meta_category = "Raster-Stream"

    def __init__(
        self,
        label="Elevation (m)",
        orientation="horizontal",
        width=6,
        height=1,
        dpi=300,
        engine="matplotlib",
        min_z=None,
        max_z=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.label = label
        self.orientation = orientation
        self.width = float(width)
        self.height = float(height)
        self.dpi = int(dpi)
        self.engine = engine.lower()

    def run_matplotlib(self, outfile):
        """Generate colorbar using Matplotlib."""

        # Create Figure
        fig = plt.figure(figsize=(self.width, self.height))

        if self.orientation == "horizontal":
            ax = fig.add_axes([0.05, 0.5, 0.9, 0.15])
        else:
            ax = fig.add_axes([0.5, 0.05, 0.15, 0.9])

        norm = mcolors.Normalize(vmin=self.z_min, vmax=self.z_max)

        plt.colorbar(
            cm.ScalarMappable(norm=norm, cmap=self.cm),
            cax=ax,
            orientation=self.orientation,
            label=self.label,
        )

        logger.info(f"Saving Matplotlib colorbar to {outfile}")
        fig.savefig(outfile, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def run_pygmt(self):
        """Generate colorbar using PyGMT."""

        try:
            import pygmt
        except ImportError:
            logger.error(
                "Error: PyGMT not installed. Install it or use --engine matplotlib"
            )
            return

        vmin, vmax = self._get_z_range()
        fig = pygmt.Figure()
        # pygmt.makecpt(cmap=self.cmap_name, series=[vmin, vmax])
        pos_str = (
            f"JTC+w{self.width}i/{self.height}i+h"
            if self.orientation == "horizontal"
            else f"JML+w{self.width}i/{self.height}i"
        )

        fig.colorbar(
            cmap=self.cmap_name,
            position=pos_str,
            frame=[f"x+l{self.label}", "y+1m"],
            region=[0, 10, 0, 10],
            projection="X1i/1i",
        )

        logger.info(f"Saving PyGMT colorbar to {self.outfile}")
        fig.savefig(self.outfile)

    def process_raster(self, src_path, dst_path, entry):
        if self.z_min is None or self.z_max is None:
            self._auto_detect_z_limits(src_path)

        if self.cm is None:
            self.cm = self._resolve_colormap()

        # if self.engine == 'pygmt':
        #     self.run_pygmt()
        # else:
        self.run_matplotlib(outfile=dst_path)
        # return self.outfile
        return True
