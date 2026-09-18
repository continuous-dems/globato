#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.hooks.viz.geoshade
~~~~~~~~~~~~~~~~~~~~~~~

Stream-based Georeferenced Hillshade generator.
Supports Matplotlib colormaps and dynamic CPT fetching.
"""

import os
import logging
import numpy as np

try:
    import matplotlib.colors as mcolors

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from globato.hooks.rasters.base import RasterStreamHook, write_cog
from . import cpt as cpt_utils

logger = logging.getLogger(__name__)


class GeoHillshade(RasterStreamHook):
    """Generate a Georeferenced Hillshade/Relief chunk-by-chunk.

    Usage:
      --hook viz_geoshade:cmap=etopo:vert_exag=3:blend_mode=soft_light:split_cpt=0
    """

    name = "viz_geoshade"
    default_suffix = "_hillshade"
    meta_desc = "Generate a Hillshade image of a raster."
    meta_category = "Raster-Stream"

    def __init__(
        self,
        azimuth=315,
        altitude=45,
        vert_exag=1,
        cmap="etopo",
        blend_mode="multiply",
        alpha=False,
        gamma=None,
        z_min=None,
        z_max=None,
        scale=111120.0,  # Degrees to Meters conversion
        split_cpt=0,
        cog=False,
        **kwargs,
    ):
        kwargs.setdefault("buffer", 2)
        super().__init__(**kwargs)

        self.azimuth = float(azimuth)
        self.altitude = float(altitude)
        self.vert_exag = float(vert_exag)
        self.cmap_name = cmap
        self.blend_mode = blend_mode
        self.alpha = str(alpha).lower() in ["true", "1", "yes"]
        self.gamma = float(gamma) if gamma else None
        self.cog = str(cog).lower() in ["true", "1", "yes"]

        self.z_min = float(z_min) if z_min is not None else None
        self.z_max = float(z_max) if z_max is not None else None
        self.scale = float(scale)
        self.split_cpt = float(split_cpt) if split_cpt is not None else None

        self.azrad = np.radians(360.0 - self.azimuth + 90.0)
        self.altrad = np.radians(self.altitude)

        # We hold off on resolving the colormap until we know the global Z-limits!
        self.cm = None

    def _validate_deps(self):
        if not HAS_MATPLOTLIB:
            return False, "matplotlib is required to generate hillshades."
        return True, ""

    def _process_file_fallback(self, src_path, dst_path, entry):
        success = super()._process_file_fallback(src_path, dst_path, entry)
        # The chunk writer produces a plain GeoTIFF (no overviews, no COG layout).
        # Only applies when this hook writes the file, not when it wraps a stream.
        if success and self.cog:
            write_cog(dst_path, dst_path)
        return success

    def modify_profile(self, profile):
        count = 4 if self.alpha else 3
        profile.update(dtype="uint8", count=count, nodata=None, photometric="RGB")
        return profile

    def _auto_detect_z_limits(self, src_fn):
        if not src_fn or not os.path.exists(src_fn):
            self.z_min, self.z_max = 0, 1
            return

        import rasterio

        try:
            with rasterio.open(src_fn) as src:
                if hasattr(src, "crs") and not src.crs.is_geographic:
                    self.scale = 1.0

                tags = src.tags(1)
                if "STATISTICS_MINIMUM" in tags and "STATISTICS_MAXIMUM" in tags:
                    self.z_min = float(tags["STATISTICS_MINIMUM"])
                    self.z_max = float(tags["STATISTICS_MAXIMUM"])
                    logger.info(
                        f"[{self.name}] Locked Z-limits from metadata: {self.z_min} to {self.z_max}"
                    )
                    return

                logger.info(
                    f"[{self.name}] No Z-stats in metadata. Calculating global limits for colormap..."
                )
                data = src.read(1, out_shape=(src.height // 10, src.width // 10))
                valid_mask = (
                    (data != src.nodata) if src.nodata is not None else ~np.isnan(data)
                )

                if np.any(valid_mask):
                    self.z_min = float(np.nanmin(data[valid_mask]))
                    self.z_max = float(np.nanmax(data[valid_mask]))
                else:
                    self.z_min, self.z_max = 0, 1

        except Exception as e:
            logger.warning(f"[{self.name}] Failed to auto-detect Z limits: {e}")
            self.z_min, self.z_max = 0, 1

    def _resolve_colormap(self):
        """Resolves and stretches the colormap to z_min/z_max."""

        import matplotlib.pyplot as plt

        try:
            cmap = plt.get_cmap(self.cmap_name)
            self._is_native_cmap = True
            return cmap
        except ValueError:
            self._is_native_cmap = False
            pass
        # try:
        #     return plt.get_cmap(self.cmap_name)
        # except ValueError:
        #     pass

        cpt_path = self.cmap_name
        is_pre_stretched = False

        if self.cmap_name.lower() == "etopo":
            cpt_path = cpt_utils.generate_etopo_cpt(self.z_min, self.z_max)
            is_pre_stretched = True

        elif self.cmap_name.lower() == "coastal_relief":
            cpt_path = cpt_utils.generate_coastal_relief_cpt(self.z_min, self.z_max)
            is_pre_stretched = True

        elif not os.path.exists(self.cmap_name):
            logger.info(f"[{self.name}] Fetching CPT from fetchez: {self.cmap_name}")
            cpt_path = cpt_utils.fetch_cpt_city(self.cmap_name)

        if not cpt_path or not os.path.exists(cpt_path):
            logger.warning(
                f"[{self.name}] Colormap '{self.cmap_name}' not found. Defaulting to 'terrain'."
            )
            return plt.get_cmap("terrain")

        if is_pre_stretched:
            stretched_cpt = cpt_path
        else:
            logger.info(
                f"[{self.name}] Stretching CPT to [{self.z_min:.2f}, {self.z_max:.2f}] (Split: {self.split_cpt})"
            )
            stretched_cpt = cpt_utils.process_cpt(
                cpt_path,
                gmin=self.z_min,
                gmax=self.z_max,
                split_cpt=self.split_cpt,
                gdal=False,
            )

        if stretched_cpt and os.path.exists(stretched_cpt):
            cm = cpt_utils.load_cmap(stretched_cpt)
            if not is_pre_stretched:
                os.remove(stretched_cpt)

            # Cleanup etopo base if generated
            if self.cmap_name.lower() in ["etopo", "coastal_relief"] and os.path.exists(
                cpt_path
            ):
                os.remove(cpt_path)
            if cm:
                return cm

        return plt.get_cmap("terrain")

    def _apply_gamma(self, arr):
        if self.gamma is None:
            return arr
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr / 255.0
        return arr ** (1 / self.gamma)

    def _blend_arrays(self, hs_norm, rgb_norm):
        H = hs_norm[..., np.newaxis]
        C = rgb_norm

        if self.blend_mode == "multiply":
            return H * C

        elif self.blend_mode == "screen":
            return 1 - (1 - H) * (1 - C)

        elif self.blend_mode == "overlay":
            return np.where(H < 0.5, 2 * H * C, 1 - 2 * (1 - H) * (1 - C))

        elif self.blend_mode == "hard_light":
            # Hard light is the same as overlay, but hinges on C instead of H
            return np.where(C < 0.5, 2 * H * C, 1 - 2 * (1 - C) * (1 - H))

        elif self.blend_mode == "soft_light":
            return (1 - 2 * H) * (C**2) + 2 * H * C

        return H * C

    def process_chunk(self, data, ndv, entry, transform=None, window=None):
        if not HAS_MATPLOTLIB:
            logger.error(
                "You must have matplotlib installed; get it with `pip install matplotlib`"
            )

            return data

        if self.z_min is None or self.z_max is None:
            self._auto_detect_z_limits(entry.get("src_fn"))

        if self.cm is None:
            self.cm = self._resolve_colormap()

        z = data[0] if data.ndim == 3 else data
        z_masked = z.copy()

        if ndv is not None:
            z_masked[z_masked == ndv] = np.nan

        dx = abs(transform[0]) * self.scale
        dy = abs(transform[4]) * self.scale

        dy_grad, dx_grad = np.gradient(z_masked * self.vert_exag, dy, dx)
        slope = 0.5 * np.pi - np.arctan(np.hypot(dx_grad, dy_grad))
        aspect = np.arctan2(dy_grad, dx_grad)

        hs = np.sin(self.altrad) * np.sin(slope) + np.cos(self.altrad) * np.cos(
            slope
        ) * np.cos(self.azrad - aspect)
        hs = np.clip(hs, 0.0, 1.0)

        if (
            self.split_cpt is not None
            and getattr(self, "_is_native_cmap", False)
            and self.z_min < self.split_cpt < self.z_max
        ):
            norm = mcolors.TwoSlopeNorm(
                vmin=self.z_min, vcenter=self.split_cpt, vmax=self.z_max
            )
        else:
            norm = mcolors.Normalize(vmin=self.z_min, vmax=self.z_max, clip=True)
        rgba = self.cm(norm(z_masked))
        rgb_colors = rgba[..., :3]

        if self.gamma:
            hs = self._apply_gamma(hs)
            rgb_colors = self._apply_gamma(rgb_colors)

        blended = self._blend_arrays(hs, rgb_colors)
        blended = np.nan_to_num(blended, nan=1.0)
        blended = np.clip(blended, 0.0, 1.0)
        blended_uint8 = (blended * 255).astype(np.uint8)

        write_data = np.transpose(blended_uint8, (2, 0, 1))

        if self.alpha:
            alpha_band = np.where(np.isnan(z_masked), 0, 255).astype(np.uint8)
            write_data = np.concatenate([write_data, alpha_band[np.newaxis, ...]])

        return write_data
