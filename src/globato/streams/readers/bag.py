#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.streams.readers.bag
~~~~~~~~~~~~~~~~~~~

Dedicated BAG (Bathymetric Attributed Grid) Reader.
Handles VR-BAGs, standard BAGs, uncertainty bands, etc.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging

import rasterio

from fetchez.spatial import Region

from .rio import RasterioReader

logger = logging.getLogger(__name__)


class BAGReader(RasterioReader):
    """Specialized Reader for BAG files.

    - Automatically handles Variable Resolution (VR) via GDAL Open Options.
    - Reads Band 2 as Uncertainty ('u').
    - Calculates weight based on resolution.
    - MODE=[LOW_RES_GRID​/​LIST_SUPERGRIDS​/​RESAMPLED_GRID​/​INTERPOLATED​/​AUTO]: Defaults to AUTO.
    """

    name = "bag-point-reader"
    meta_category = "point-stream"
    meta_dtype = ["bag-raster", "bag"]
    meta_desc = "Read BAG data through rasterio into a point stream"
    meta_extensions = ["bag"]

    def __init__(
        self,
        path,
        mode="RESAMPLED_GRID",
        min_weight=0.25,
        auto_weight=True,
        uncertainty_scale=5.0,
        **kwargs,
    ):
        super().__init__(
            path, auto_weight=auto_weight, uncertainty_scale=uncertainty_scale, **kwargs
        )
        self.modes = [
            "LOW_RES_GRID",
            "LIST_SUPERGRIDS",
            "RESAMPLED_GRID",
            "INTERPOLATED",
            "AUTO",
        ]
        self.mode = mode if mode.upper() in self.modes else "AUTO"

    def _calculate_bag_weight(self, transform):
        """Weight scales directly with physical resolution.
        Targeting 32 / res yields:
        ~30m -> 1.0 (1s)
        ~64m -> 0.5 (3s)
        ~128m -> 0.25 (9s)
        """
        x_res = abs(transform.a)

        if x_res == 0:
            return 1.0

        return 32.0 / x_res

    def _yield_raw_chunks(self):
        env_opts = {
            "GDAL_IGNORE_BAG_XML_METADATA": "YES",
            "OGR_BAG_MIN_VERSION": "1.0",
            "CPL_MIN_LOG_LEVEL": rasterio.logging.ERROR,
        }

        is_vr = False

        # temp hack; if world_region matches self.region,
        # it means that self.region wasn't set and should
        # be reset to None.
        world_region = Region(-180, 180, -90, 90)
        if world_region == self.region:
            self.region = None

        try:
            with rasterio.Env(**env_opts):
                with rasterio.open(self.src_fn) as src:
                    tags = src.tags(ns="IMAGE_STRUCTURE")
                    if tags.get("HAS_SUPERGRIDS") == "TRUE":
                        is_vr = True

                    if src.tags().get("MIN_RESOLUTION_X"):
                        is_vr = True

                    self.weight = self._calculate_bag_weight(src.transform)

            if not is_vr:
                self.u_band = 2
                yield from self._process_rio_dataset()
                return

            elif is_vr:
                logger.debug(
                    f"Detected VR-BAG, re-opening in {self.mode} mode: {self.src_fn}"
                )
                vr_opts = {"MODE": self.mode, "RES_STRATEGY": "MIN"}

                try:
                    with rasterio.Env(**env_opts):
                        self.u_band = 2
                        with rasterio.open(self.src_fn, **vr_opts) as src:
                            logger.debug(f"Dataset Driver: {src.driver}")
                            logger.debug(f"Dimensions: {src.width}x{src.height}")
                            self.weight = self._calculate_bag_weight(src.transform)
                            yield from self._process_rio_dataset(src=src)
                except Exception as e:
                    logger.error(f"Failed to read VR-BAG {self.src_fn}: {e}")
        except Exception as e:
            logger.error(f"Failed to probe BAG {self.src_fn}: {e}")
            return
