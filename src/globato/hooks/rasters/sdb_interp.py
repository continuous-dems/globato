#!/usr/bin/env python

"""
globato.hooks.raster.sdb_interp
~~~~~~~~~~~~~~~~~~~~~~~

SDB interpolation hook
"""

import logging
import os

import numpy as np
import rasterio
from fetchez.core import run_fetchez
from fetchez.spatial import Region
from rasterio.warp import Resampling, reproject

from .base import RasterGlobalHook

try:
    from sklearn.ensemble import RandomForestRegressor

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class SDBInterpolation(RasterGlobalHook):
    """Satellite Derived Bathymetry (SDB) Gap Filler.

    Auto-fetches Sentinel-2 data via CDSE if a local image is not provided.
    """

    name = "interp_sdb"
    default_suffix = "_sdb"
    meta_desc = "Interpolate NoData voids using SDB."
    meta_tags = ["globato", "interpolation", "multi-stack"]

    def __init__(
        self, sat_image=None, n_trees=100, max_depth=-0.5, cloud_cover=10, **kwargs
    ):
        super().__init__(**kwargs)
        self.sat_image = sat_image
        self.n_trees = int(n_trees)
        self.max_depth = float(max_depth)
        self.cloud_cover = int(cloud_cover)

    def _fetch_sentinel2(self, bounds, crs):
        """Dynamically fetches and stacks Sentinel-2 bands via Fetchez."""

        from fetchez.modules.cdse import Sentinel2_CDSE

        w, s, e, n = bounds
        region = Region.from_list([w, e, s, n])

        logger.info(
            "No local sat_image provided. Auto-fetching Sentinel-2 data from CDSE..."
        )
        s2_fetcher = Sentinel2_CDSE(src_region=region, cloud_cover=self.cloud_cover)
        s2_fetcher.run()
        run_fetchez([s2_fetcher])

        # bands we need for SDB (B02=Blue, B03=Green, B04=Red, B08=NIR)
        bands = {}
        for entry in s2_fetcher.results:
            fn = entry.get("dst_fn", "")
            if "B02" in fn:
                bands[1] = fn
            elif "B03" in fn:
                bands[2] = fn
            elif "B04" in fn:
                bands[3] = fn
            elif "B08" in fn:
                bands[4] = fn

        if len(bands) < 4:
            logger.error("Failed to download all required Sentinel-2 bands.")
            return None

        # Stack the JP2s into a single temporary TIF matching the DEM bounds
        stack_path = "temp_s2_stack.tif"
        logger.info(f"Stacking Sentinel-2 bands into {stack_path}...")

        with rasterio.open(bands[1]) as src:
            profile = src.profile.copy()
            profile.update(count=4, driver="GTiff")

            with rasterio.open(stack_path, "w", **profile) as dst:
                for i in range(1, 5):
                    with rasterio.open(bands[i]) as b_src:
                        dst.write(b_src.read(1), i)

        return stack_path

    def process_raster(self, src_path, dst_path, entry):
        if not HAS_SKLEARN:
            logger.error("scikit-learn is required for SDB interpolation.")
            return False

        with rasterio.open(src_path) as src_dem:
            dem_data = src_dem.read(1)
            nodata = src_dem.nodata if src_dem.nodata is not None else -9999.0

            sat_path = self.sat_image
            if not sat_path or not os.path.exists(sat_path):
                sat_path = self._fetch_sentinel2(src_dem.bounds, src_dem.crs)
                if not sat_path:
                    return False

            with rasterio.open(sat_path) as src_sat:
                n_bands = src_sat.count

                # Reproject the Sat Data to match the DEM
                sat_data = np.zeros(
                    (n_bands, src_dem.height, src_dem.width), dtype=np.float32
                )

                logger.info("Aligning Sentinel-2 bands to DEM grid...")
                reproject(
                    source=rasterio.band(src_sat, [1, 2, 3, 4]),
                    destination=sat_data,
                    src_transform=src_sat.transform,
                    src_crs=src_sat.crs,
                    dst_transform=src_dem.transform,
                    dst_crs=src_dem.crs,
                    resampling=Resampling.bilinear,
                )

                dem_flat = dem_data.flatten()
                sat_flat = sat_data.reshape(n_bands, -1).T

                valid_dem = (
                    (dem_flat != nodata)
                    & (~np.isnan(dem_flat))
                    & (dem_flat < self.max_depth)
                )
                valid_sat = np.all(sat_flat > 0, axis=1)

                train_mask = valid_dem & valid_sat

                if not np.any(train_mask):
                    logger.error("No intersecting data to train SDB Random Forest!")
                    return False

                X_train = sat_flat[train_mask]
                y_train = dem_flat[train_mask]

                # Subsample the training data
                MAX_TRAIN_POINTS = 100000
                if len(X_train) > MAX_TRAIN_POINTS:
                    logger.info(
                        f"Subsampling SDB training data from {len(X_train):,} down to {MAX_TRAIN_POINTS:,} points..."
                    )
                    idx = np.random.choice(
                        len(X_train), MAX_TRAIN_POINTS, replace=False
                    )
                    X_train = X_train[idx]
                    y_train = y_train[idx]

                logger.info(f"Training SDB Random Forest ({self.n_trees} trees)...")
                rf = RandomForestRegressor(
                    n_estimators=self.n_trees, n_jobs=-1, random_state=42
                )
                rf.fit(X_train, y_train)

                # The gap mask: Where we have satellite data, but no DEM data
                gap_mask = (~valid_dem) & valid_sat

                logger.info(
                    f"Predicting bathymetry for {np.sum(gap_mask):,} shallow water pixels..."
                )

                # Predict only on the gaps to save time
                dem_flat[gap_mask] = rf.predict(sat_flat[gap_mask])
                result_arr = dem_flat.reshape(src_dem.height, src_dem.width)

                profile = src_dem.profile.copy()
                with rasterio.open(dst_path, "w", **profile) as dst:
                    dst.write(result_arr.astype(profile["dtype"]), 1)

                return True
