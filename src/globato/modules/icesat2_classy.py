#!/usr/bin/env python

"""
globato.modules.icesat2_classy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ICESat-2 Data Parser (ATL03, ATL24) ported from CUDEM for Fetchez-Globato.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging

from fetchez.cli import cli_opts
from fetchez.core import run_fetchez
from fetchez.hooks.copy_artifact import CopyArtifactHook
from fetchez.modules import FetchModule
from fetchez.modules.earthdata import IceSat2

# from globato.streams.readers.icesat2 import IceSat2Stream
from globato.hooks.sinks.xyz_writer import XYZWrite

logger = logging.getLogger(__name__)


@cli_opts(
    help_text="Fetches ATL03 data, applies legacy classification, and exports to NetCDF.",
    classes="Slash-separated list of target classes to retain (e.g., '40/41/42').",
    use_dbscan="If True, runs DBSCAN to identify bathymetry.",
)
class ICESat2ClassyModule(FetchModule):
    """Bridge Super-Module for IVERT ICESat-2 delivery."""

    name = "icesat2_classy"
    meta_desc = "Legacy ICESat-2 Classifier and Exporter"
    meta_category = "Globato"
    meta_tags = ["icesat", "global", "sattelie", "globato"]

    def __init__(self, classes=None, use_dbscan=False, **kwargs):
        super().__init__(name="icesat2_classy", **kwargs)
        self.classes = classes
        self.use_dbscan = str(use_dbscan).lower() in ["true", "1", "t", "yes"]

    def run(self):
        if not self.region:
            logger.error(f"[{self.name}] Requires a bounding box region to run.")
            return

        logger.info(f"[{self.name}] Orchestrating ATL03 Fetch & Classification...")

        fetcher = IceSat2(
            src_region=self.region, short_name="ATL03", outdir=self._outdir
        )

        # TODO: fix this to use DataStream from fetchez.hooks.stream_init
        # fetcher.add_hook(
        #     IceSat2Stream(
        #         cache_dir=self._outdir,
        #         classes=self.classes,
        #         use_dbscan=self.use_dbscan,
        #     )
        # )

        fetcher.add_hook(XYZWrite())

        fetcher.add_hook(
            CopyArtifactHook(target_dir="../_icesat2_deliverables", match=[".nc"])
        )

        fetcher.run()
        run_fetchez([fetcher])

        self.results = fetcher.results
        return self
