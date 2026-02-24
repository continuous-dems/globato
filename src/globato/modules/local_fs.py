#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.modules.local_fs
~~~~~~~~~~~~~~~~~~~~~~~~

Recursively crawl local directories, spatially filter files using .inf sidecars.
If an .inf is missing, it utilizes Globato's native stream factory to parse
the file (XYZ, LAS, TIF, etc.), calculate bounds, and generate the sidecar.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import json
import glob
import logging

from fetchez import core
from fetchez import cli
from transformez.spatial import TransRegion as Region

from ..processors.formats.stream_factory import DataStream
from ..processors.metadata.globato_inf import GlobatoInfo

logger = logging.getLogger(__name__)


@cli.cli_opts(
    help_text="Crawl and spatially filter a local directory of data.",
    path="The root directory to crawl.",
    ext="File extension to match (e.g., '.tif', '.bag', '.xyz').",
    datatype="Data type tag for downstream hooks (default: 'raster').",
    gen_inf="Boolean: Write a full .inf sidecar via stream if missing (default: True)."
)
class LocalFS(core.FetchModule):
    """The Modern Datalist."""

    def __init__(self, path=".", ext=".tif", datatype="raster", gen_inf=True, **kwargs):
        super().__init__(name="local_fs", **kwargs)
        self.path = os.path.abspath(path)
        self.ext = ext if ext.startswith('.') else f".{ext}"
        self.datatype = datatype
        self.gen_inf = gen_inf

    def _read_inf(self, inf_path):
        """Attempt to parse an existing .inf file for spatial bounds."""

        try:
            with open(inf_path, 'r') as f:
                data = json.load(f)
                if all(k in data for k in ["min_x", "max_x", "min_y", "max_y"]):
                    return Region.from_list([
                        data["min_x"], data["max_x"],
                        data["min_y"], data["max_y"]
                    ])
        except Exception:
            pass
        return None

    def _generate_inf_via_stream(self, filepath):
        """Creates a temporary mini-pipeline to generate the .inf sidecar."""

        try:
            dummy_entry = {
                "url": f"file://{filepath}",
                'dst_fn': filepath,
                "data_type": self.datatype,
                "status": 0
            }
            entries = [(self, dummy_entry)]

            ds_hook = DataStream()
            entries = ds_hook.run(entries)

            info_hook = GlobatoInfo()
            entries = info_hook.run(entries)

            for mod, entry in entries:
                stream = entry.get('stream')
                if stream:
                    for chunk in stream:
                        pass

            info_hook.teardown()
            ds_hook.teardown()

            return True
        except Exception as e:
            logger.warning(f"Failed to generate .inf via stream for {os.path.basename(filepath)}: {e}")
            return False

    def run(self):
        if not os.path.exists(self.path):
            logger.error(f"LocalFS path does not exist: {self.path}")
            return self

        target_region = Region.from_list(self.region) if self.region else Region.from_list([-180, 180, -90, 90])
        search_pattern = os.path.join(self.path, f"**/*{self.ext}")
        matched_files = 0

        logger.info(f"Crawling {self.path} for '{self.ext}' files...")

        for filepath in glob.iglob(search_pattern, recursive=True):
            file_region = None
            inf_path = filepath + ".inf"

            if os.path.exists(inf_path):
                file_region = self._read_inf(inf_path)

            if file_region is None and self.gen_inf:
                logger.info(f"Generating missing .inf for {os.path.basename(filepath)}...")
                success = self._generate_inf_via_stream(filepath)
                if success and os.path.exists(inf_path):
                    file_region = self._read_inf(inf_path)

            if file_region:
                if target_region.intersects(file_region):
                    self.add_entry_to_results(
                        url=f"file://{filepath}",
                        dst_fn=filepath,
                        data_type=self.datatype,
                        status=0
                    )
                    matched_files += 1

        logger.info(f"LocalFS matched {matched_files} files in {target_region}")
        return self
