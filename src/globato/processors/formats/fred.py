#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.formats.fred
~~~~~~~~~~~~~

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import logging
from fetchez.hooks import FetchHook
from fetchez.utils import this_date

try:
    from ..fred import FredIndexer as FRED
except ImportError:
    from fetchez.fred import FRED

logger = logging.getLogger(__name__)

class FredGenerator(FetchHook):
    """Post-Hook: Generates a FRED Index (GeoJSON) of all successful entries.

    Usage:
      --hook fred_export:name=my_mission_data
    """

    name = "fred_export"
    stage = "post"
    desc = "generate a fred geojson index of the output data"

    def __init__(self, name="output", output_dir=None, scan=True, **kwargs):
        """Args:
            name (str): Filename for the index (e.g. 'fred' -> 'fred.geojson')
            output_dir (str): Where to save it. Defaults to current dir.
            scan (bool): If True, attempts to open files to get precise bounds.
                         If False, uses the Module's requested region (faster).
        """

        super().__init__(**kwargs)
        self.index_name = name
        self.output_dir = output_dir or os.getcwd()
        self.scan_files = str(scan).lower() == 'true'


    def run(self, entries):
        full_path = os.path.join(self.output_dir, self.index_name)
        fred = FRED(full_path, local=True)

        count = 0

        for mod, entry in entries:
            if entry.get('status', -1) != 0:
                continue

            dst_fn = entry.get('dst_fn')
            if not dst_fn or not os.path.exists(dst_fn):
                continue

            geom = None
            meta = {}

            if self.scan_files and hasattr(fred, '_extract_file_metadata'):
                try:
                    bbox, f_meta = fred._extract_file_metadata(dst_fn)
                    if bbox:
                        w, e, s, n = bbox
                        geom = {
                            'type': 'Polygon',
                            'coordinates': [[[w, s], [e, s], [e, n], [w, n], [w, s]]]
                        }
                        meta.update(f_meta)
                except Exception:
                    pass

            if not geom and mod.region:
                w, e, s, n = mod.region
                geom = {
                    'type': 'Polygon',
                    'coordinates': [[[w, s], [e, s], [e, n], [w, n], [w, s]]]
                }

            if not geom:
                logger.warning(f"Skipping FRED entry for {dst_fn}: No spatial bounds found.")
                continue

            props = {
                'Name': os.path.basename(dst_fn),
                'DataLink': f"file://{os.path.abspath(dst_fn)}",
                'DataType': entry.get('data_type', 'unknown'),
                'DataSource': mod.name,
                'Agency': getattr(mod, 'agency', 'Fetchez'),
                'Date': meta.get('date', this_date()),
                'Resolution': meta.get('resolution'),
                'HorizontalDatum': meta.get('h_datum'),
                'VerticalDatum': meta.get('v_datum')
            }

            fred.add_survey(geom, **props)
            count += 1

        if count > 0:
            fred.save()
            logger.info(f"Generated FRED index '{self.index_name}.geojson' with {count} items.")

        return entries
