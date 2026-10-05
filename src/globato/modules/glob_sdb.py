#!/usr/bin/env python

"""
globato.modules.glob_sdb
~~~~~~~~~~~~~~~~~~~~~~~

SDB generation
"""

import logging
import os

from fetchez.hooks.set_datatype import SetDatatype
from fetchez.modules import FetchModule

logger = logging.getLogger(__name__)


class GlobSDB(FetchModule):
    """Curated SDB Super-Module.

    Fetches ICESat-2 (or MBDB) for training, Sentinel-2 for prediction,
    and yields a fully generated SDB raster DEM.
    """

    name = "glob_sdb"
    meta_tags = [
        "bathymetry",
        "sdb",
        "sentinel2",
        "icesat2",
        "machine-learning",
        "globato",
    ]
    meta_category = "Globato"
    meta_agency = "Globato"

    def __init__(self, train_source="icesat2", cloud_cover=10, max_depth=-25, **kwargs):
        super().__init__(**kwargs)
        self.train_source = train_source
        self.cloud_cover = cloud_cover
        self.max_depth = max_depth
        self.datatype = "raster"

        self.add_hook(SetDatatype(datatype="raster"))

    def run(self):
        from globato.cli.recipe import Recipe
        from globato.hooks.rasters.sdb_interp import SDBInterpolation

        logger.info(
            f"Initializing SDB Super-Module. Training source: {self.train_source}"
        )

        # Generate a 'micro-recipe' to train the data
        train_dem_path = os.path.join(self._outdir, "temp_sdb_train_stack.tif")
        micro_config = {
            "project": {"name": "sdb_trainer"},
            "region": [
                self.region.xmin,
                self.region.xmax,
                self.region.ymin,
                self.region.ymax,
            ],
            "modules": [{"module": self.train_source}],
            "global_hooks": [
                {
                    "name": "multi_stack",
                    "args": {
                        "res": ".111111111s",
                        "output": train_dem_path,
                        "nodata": -9999,
                        "crs": "EPSG:4326",
                    },
                }
            ],
        }

        logger.info(f"Fetching and gridding training data via {self.train_source}...")
        Recipe.from_file(micro_config).run()

        if not os.path.exists(train_dem_path):
            logger.error("Failed to generate training DEM.")
            return self

        # auto-fetch Sentinel-2!
        sdb_hook = SDBInterpolation(
            sat_image=None, cloud_cover=self.cloud_cover, max_depth=self.max_depth
        )

        final_sdb_path = "glob_sdb_output.tif"

        success = sdb_hook.process_raster(train_dem_path, final_sdb_path, {})
        if success and os.path.exists(final_sdb_path):
            self.add_entry_to_results(
                url=f"file://{final_sdb_path}",
                dst_fn=final_sdb_path,
                data_type=self.datatype,
                status=0,
            )

        return self
