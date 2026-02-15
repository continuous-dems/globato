#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.modules.gebco
~~~~~~~~~~~~~

Get gebco as a cog

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

from fetchez.modules.gebco import GEBCO as CoreGEBCO
from ..processors.formats.cog import COGSubset

# Source Cooperative (Alex Leith) - Cloud Optimized GeoTIFFs
GEBCO_COG_URLS = {
    'grid': 'https://data.source.coop/alexgleith/gebco-2024/GEBCO_2024.tif',
    'tid': 'https://data.source.coop/alexgleith/gebco-2024/GEBCO_2024_TID.tif',
    'sub_ice': 'https://data.source.coop/alexgleith/gebco-2024/GEBCO_2024_sub_ice_topo.tif'
}

class GEBCO_COG(CoreGEBCO):
    """Globato Wrapper for GEBCO that uses Cloud Optimized GeoTIFFs
    to fetch ONLY the requested region.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_hook(COGSubset())

    def run(self):
        """Setup for Cloud Optimized GeoTIFF subsetting."""

        url = GEBCO_COG_URLS.get(self.layer)
        if not url:
            logger.error(f"No COG URL available for layer '{self.layer}'.")
            return

        if self.region:
            w, e, s, n = self.region
            dst_fn = f"gebco_2024_{self.layer}_{w}_{e}_{s}_{n}.tif"
        else:
            dst_fn = f"gebco_2024_{self.layer}_subset.tif"

        self.add_entry_to_results(
            url=url,
            dst_fn=dst_fn,
            data_type="raster",
            cog=True
        )
