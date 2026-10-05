#!/usr/bin/env python

"""
globato.hooks.tools.cleanup_tmp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cleanup the 'tmp' directory in the collection stage.

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import os
import shutil
from pathlib import Path

from fetchez.hooks import FetchHook
from fetchez.utils import str2bool

logger = logging.getLogger(__name__)


class CleanupTmpHook(FetchHook):
    """Deletes the temporary directory and all intermediate files at the end of the pipeline.
    Runs at the 'collection' stage and logs the total storage space freed.
    """

    name = "cleanup_tmp"
    meta_stage = "collection"
    meta_category = "system"

    def __init__(self, keep=False, target_dir="tmp", **kwargs):
        super().__init__(**kwargs)
        self.keep = str2bool(keep)
        self.target_dir = target_dir

    def run(self, entries):
        tmp_path = os.path.abspath(self.target_dir)

        if not os.path.exists(tmp_path) or not os.path.isdir(tmp_path):
            return entries

        total_size_bytes = 0
        try:
            p = Path(tmp_path)
            total_size_bytes = sum(
                f.stat().st_size for f in p.rglob("*") if f.is_file()
            )
        except Exception as e:
            logger.debug(f"[{self.name}] Failed to calculate temp dir size: {e}")

        total_size_mb = total_size_bytes / (1024 * 1024)

        if self.keep:
            logger.info(
                f"[{self.name}] Retaining temporary directory: {tmp_path} "
                f"({total_size_mb:.2f} MB of intermediate data)"
            )
            return entries

        try:
            shutil.rmtree(tmp_path)
            logger.info(
                f"[{self.name}] ✨ Cleaned up {total_size_mb:.2f} MB of temporary files."
            )
        except Exception as e:
            logger.error(f"[{self.name}] Failed to clean up {tmp_path}: {e}")

        return entries
