#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.hooks.filters.rq
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reference Quality (RQ) Filter.
Fetches a reference raster (e.g. GEBCO) and filters points that deviate from it.

:copyright: (c) 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import json
import hashlib
import logging
from pathlib import Path
import threading
import numpy as np
import rasterio
from scipy.ndimage import map_coordinates

import fetchez
from fetchez.utils import str2inc, parse_arg_to_list

try:
    from fetchez.entry import entry_id
except ImportError:  # pragma: no cover - compatibility with older Fetchez
    entry_id = None

from .base import GlobatoFilter

try:
    from osgeo import gdal

    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False

try:
    from transformez.grid.engine import GridEngine
    from transformez.grid.io import GridWriter

    HAS_GRID_ENGINE = True
except ImportError:
    HAS_GRID_ENGINE = False

from transformez.srs import SRSParser

logger = logging.getLogger(__name__)

T_LOCK = threading.Lock()


class ReferenceQuality(GlobatoFilter):
    """Filters points by comparing Z values to a Reference Raster (RQ).

    Builder Modes:
      - 'vrt': Uses GDAL to build a Virtual Raster.
      - 'grid': Uses GridEngine (transformez) to mosaic/interpolate/fill a solid GeoTIFF.

    Modes:
      - 'diff': Absolute difference
      - 'percent': Relative difference (default)
      - 'iho_1' / 'iho': IHO S-44 Order 1 TVU (a=0.5, b=0.013)
      - 'iho_2': IHO S-44 Order 2 TVU (a=1.0, b=0.023)

    Args:
        reference (str): Fetchez Module Name (default: 'gmrt').
        threshold (float): Max allowed difference.
        mode (str): 'diff' (absolute) or 'percent' (relative).
        builder (str): 'vrt' or 'grid'.
        res (float): Resolution for 'grid' builder (default: 0.000833333 ~3 arc-seconds).
    """

    name = "rq"
    meta_stage = "stream"
    meta_desc = "Filter points by comparing z values to a reference raster."

    def __init__(
        self,
        reference="gmrt",
        threshold=50,
        mode="percent",
        builder="grid",
        res=0.0008333333333333334,
        target_srs=None,
        iho_order="1",
        overwrite=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ref_sources = parse_arg_to_list(reference, str)
        self.threshold = float(threshold)
        self.mode = mode.lower()
        self.builder = builder.lower()
        self.res = str2inc(res)
        self.ref_fn = None
        self.overwrite = overwrite

        self.target_srs = target_srs
        self._transformer = None

        self.total_points = 0
        self.dropped_points = 0
        self.invalid_reference_points = 0
        self.out_of_bounds_points = 0

        # IHO S-44 Parameters (a, b)
        self.iho_order = str(iho_order).lower()
        if self.iho_order == "special":
            self.iho_a, self.iho_b = 0.25, 0.0075
        elif self.iho_order == "2":
            self.iho_a, self.iho_b = 1.0, 0.023
        else:  # Default to Order 1
            self.iho_a, self.iho_b = 0.5, 0.013

    def setup(self, mod, entry):
        """Called once before stream processing starts."""

        if not getattr(mod, "region", None):
            return False

        region = getattr(mod, "region")
        region = mod.region.copy()
        if not self.target_srs:
            self.target_srs = region.srs

        # The grid must be built in WGS84 to prevent massive memory allocations
        # when using geographic resolutions with projected (UTM) boundaries!
        # If the incoming region is projected, warp a copy to WGS84.
        self.wgs_region = region.copy()
        if self.wgs_region.srs and self.wgs_region.srs.upper() != "EPSG:4326":
            self.wgs_region.warp(dst_srs="EPSG:4326")
        self.wgs_region.buffer(pct=5)

        # self.target_region = self.wgs_region.buffer(pct=5)

        outdir = Path(getattr(mod, "_outdir"))
        if not self.ref_fn:
            files = self._fetch_reference_files(region, outdir)

            if not files:
                logger.error(
                    f"[RQ] No valid reference data found for {region}. Disabling RQ filter to prevent crash!"
                )
                return False

            if self.builder == "grid" and HAS_GRID_ENGINE:
                self.ref_fn = self._build_grid(files, region, outdir)
            else:
                self.ref_fn = self._build_vrt(files, region, outdir)

            if not self.ref_fn or not Path(self.ref_fn).exists():
                logger.error(
                    "[RQ] Builder failed to generate a reference surface. Disabling RQ filter."
                )
                return False

        try:
            self.src = rasterio.open(self.ref_fn)
            # Store the inverse transform matrix to map points to fractional pixels natively
            self.inv_transform = ~self.src.transform
            ref_raw = self.src.read(1).astype("float64")
            nodata = self.src.nodata

            # Standardize NoData to NaN so the bilinear interpolator ignores voids cleanly.
            if nodata is None:
                self.ref_data = np.where(np.isnan(ref_raw), np.nan, ref_raw)
            elif np.isnan(nodata):
                self.ref_data = np.where(np.isnan(ref_raw), np.nan, ref_raw)
            else:
                self.ref_data = np.where(
                    np.isclose(ref_raw, nodata, atol=1e-6), np.nan, ref_raw
                )
            # self.ref_data = self.src.read(1)
        except Exception as e:
            logger.error(
                f"[RQ] Failed to open generated reference surface: {e}. Disabling RQ filter."
            )
            return False

        # if target_region:
        #     self.target_srs = target_region.srs

        # if self.target_srs:

        # Check the stream's current SRS first, fallback to the module's region SRS
        current_stream_srs = (
            entry.get("src_srs") or region.srs or "EPSG:4326+global:mss"
        )

        try:
            if self._transformer is None:
                self._transformer, _ = SRSParser(
                    current_stream_srs,
                    self.wgs_region.srs,
                    region=self.wgs_region,
                ).get_components()
            if self._transformer is None:
                self._transformer, _ = SRSParser(
                    region.srs,
                    self.wgs_region.srs or "epsg:4326+global:mss",
                    region=self.wgs_region,
                ).get_components()
        except Exception as e:
            logger.exception(f"Could not perform vertical transformation: {e}")
            return False

        return True

    def _fetch_reference_files(self, region, outdir):
        """Downloads multiple reference datasets and stacks them by resolution."""

        valid_files = []

        for source in self.ref_sources:
            if Path(source).is_file():
                valid_files.append(source)
                continue

            logger.debug(f"[RQ] Fetching reference tier: {source}...")
            try:
                files = fetchez.get(
                    source,
                    region=region.copy().buffer(pct=5).to_list(),
                    region_srs=region.srs,
                    outdir=str(outdir),
                    use_cache=True,
                )
                # if files:
                for f in files:
                    if Path(f).is_file() and Path(f).stat().st_size > 0:
                        valid_files.append(f)
            except Exception as e:
                logger.warning(f"[RQ] Fetch failed for {source}: {e}")

        if not valid_files and "gebco" not in self.ref_sources:
            logger.warning("[RQ] Primary references failed. Falling back to GEBCO...")
            try:
                fallback = fetchez.get(
                    "gebco",
                    region=region.copy().buffer(pct=5).to_list(),
                    use_cache=True,
                )
                if fallback:
                    valid_files.extend(
                        [
                            f
                            for f in fallback
                            if Path(f).is_file() and Path(f).stat().st_size > 2000
                        ]
                    )
            except Exception as e:
                logger.error(f"[RQ] Fallback to GEBCO failed: {e}")

        if not valid_files:
            return []

        file_resolutions = []
        for f in valid_files:
            try:
                with rasterio.open(f) as src:
                    res = src.res[0]
                    file_resolutions.append((res, f))
            except Exception:
                pass

        file_resolutions.sort(key=lambda x: x[0], reverse=True)
        sorted_files = [f[1] for f in file_resolutions]
        logger.debug(
            f"[RQ] Stacked {len(sorted_files)} reference files for VRT/Grid engine."
        )
        return sorted_files

    @staticmethod
    def _file_identity(filename):
        """Return stable semantic metadata for a source raster artifact.

        RQ should distinguish regenerated source artifacts even when their
        filesystem names are unchanged. File contents are intentionally not
        hashed here because reference rasters can be large; the source path,
        stat information, and raster geometry/metadata form the identity input.
        """
        path = Path(filename).resolve()
        stat = path.stat()
        identity = {
            "name": path.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

        try:
            with rasterio.open(path) as src:
                identity.update(
                    {
                        "driver": src.driver,
                        "width": src.width,
                        "height": src.height,
                        "count": src.count,
                        "dtype": src.dtypes[0],
                        "crs": src.crs.to_string() if src.crs else None,
                        "transform": tuple(src.transform),
                        "bounds": tuple(src.bounds),
                        "nodata": src.nodata,
                    }
                )
        except Exception as exc:
            identity["raster_error"] = str(exc)

        return identity

    def _reference_identity(self, files):
        """Return the Fetchez entry identity for the derived RQ reference."""
        state = {
            "requested_references": list(self.ref_sources),
            "references": [self._file_identity(f) for f in files],
            "region": {
                "w": float(self.wgs_region.xmin),
                "e": float(self.wgs_region.xmax),
                "s": float(self.wgs_region.ymin),
                "n": float(self.wgs_region.ymax),
                "srs": self.wgs_region.srs,
            },
            "resolution": float(self.res),
            "builder": self.builder,
        }

        # Use Fetchez's canonical entry identity machinery whenever available.
        if entry_id is not None:
            return entry_id(
                {
                    "url": "globato://rq-reference",
                    "dst_fn": "rq-reference.tif",
                    "data_type": "raster",
                    "profile": "rq/v1",
                    "src_srs": self.wgs_region.srs,
                    "metadata": state,
                }
            )

        # Compatibility fallback for older Fetchez installations.
        payload = json.dumps(
            state,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _reference_is_valid(path):
        """Return True when a cached reference surface can be opened."""
        path = Path(path)
        if not path.is_file() or path.stat().st_size == 0:
            return False

        try:
            with rasterio.open(path) as src:
                return src.width > 0 and src.height > 0 and src.count > 0
        except Exception:
            return False

    def _build_vrt(self, files, region, outdir):
        """Build or reuse a content-addressed VRT using GDAL."""

        if not HAS_GDAL:
            logger.error("[RQ] GDAL required for 'vrt' builder.")
            return None

        cache_dir = Path(outdir) / "rq"
        cache_dir.mkdir(parents=True, exist_ok=True)
        identity = self._reference_identity(files)
        vrt_path = cache_dir / f"rq_ref_{identity[:24]}.vrt"

        with T_LOCK:
            if vrt_path.exists() and not self.overwrite:
                if self._reference_is_valid(vrt_path):
                    logger.debug(f"[RQ] Reusing cached reference surface: {vrt_path}")
                    return str(vrt_path)
                logger.warning(
                    f"[RQ] Cached reference surface is invalid; rebuilding: {vrt_path}"
                )
                try:
                    vrt_path.unlink()
                except OSError:
                    pass

            try:
                if self.overwrite and vrt_path.exists():
                    vrt_path.unlink()
                vrt_options = gdal.BuildVRTOptions(resampleAlg="bilinear")
                gdal.BuildVRT(str(vrt_path), files, options=vrt_options)
            except Exception as e:
                logger.warning(f"[RQ] VRT Build failed: {e}. Using first file.")
                try:
                    if vrt_path.exists():
                        vrt_path.unlink()
                except OSError:
                    pass
                return files[0]

        if self._reference_is_valid(vrt_path):
            logger.debug(f"[RQ] Wrote reference surface: {vrt_path}")
            return str(vrt_path)

        logger.warning(f"[RQ] VRT builder produced no usable reference: {vrt_path}")
        return None

    def _build_grid(self, files, region, outdir):
        """Build or reuse a content-addressed GeoTIFF reference mosaic."""

        if not HAS_GRID_ENGINE:
            logger.error("[RQ] transformez.grid_engine required for 'grid' builder.")
            return None

        # nx = int(np.ceil((self.wgs_region[1] - self.wgs_region[0]) / self.res))
        # ny = int(np.ceil((self.wgs_region[3] - self.wgs_region[2]) / self.res))
        nx = max(1, int(np.floor(self.wgs_region.width / self.res)))
        ny = max(1, int(np.floor(self.wgs_region.height / self.res)))
        logger.debug(
            f"[RQ] Gridding geographic reference surface ({nx}x{ny}) from {len(files)} files..."
        )

        cache_dir = Path(outdir) / "rq"
        cache_dir.mkdir(parents=True, exist_ok=True)
        identity = self._reference_identity(files)
        out_path = cache_dir / f"rq_ref_{identity[:24]}.tif"

        with T_LOCK:
            if out_path.exists() and not self.overwrite:
                if self._reference_is_valid(out_path):
                    logger.debug(f"[RQ] Reusing cached reference surface: {out_path}")
                    return str(out_path)
                logger.warning(
                    f"[RQ] Cached reference surface is invalid; rebuilding: {out_path}"
                )
                try:
                    out_path.unlink()
                except OSError:
                    pass

            try:
                if self.overwrite and out_path.exists():
                    out_path.unlink()
                grid_data = GridEngine.load_and_interpolate(
                    files, self.wgs_region, nx, ny
                )
                GridWriter.write(str(out_path), grid_data, self.wgs_region)
            except Exception as exc:
                logger.exception(f"[RQ] Failed to build reference grid: {exc}")
                try:
                    if out_path.exists():
                        out_path.unlink()
                except OSError:
                    pass
                return None

        if self._reference_is_valid(out_path):
            logger.debug(f"[RQ] Wrote reference surface: {out_path}")
            return str(out_path)

        logger.error(f"[RQ] Grid writer produced no usable reference: {out_path}")
        return None

    def filter_chunk(self, chunk):
        """Return a boolean mask identifying points rejected by RQ."""
        rx, ry, rz = chunk["x"], chunk["y"], chunk["z"]

        if self._transformer:
            rx, ry, rz = self._transformer.transform(rx, ry, rz)

        cols, rows = self.inv_transform * (rx, ry)

        # Convert rasterio transform coordinates (cell corners) into SciPy
        # array coordinates (cell centers).
        cols -= 0.5
        rows -= 0.5

        in_bounds = (
            np.isfinite(cols)
            & np.isfinite(rows)
            & (rows >= 0)
            & (rows <= self.src.height - 1)
            & (cols >= 0)
            & (cols <= self.src.width - 1)
        )

        ref_vals = np.full(len(chunk), np.nan, dtype=np.float64)
        if np.any(in_bounds):
            ref_vals[in_bounds] = map_coordinates(
                self.ref_data,
                [rows[in_bounds], cols[in_bounds]],
                order=1,
                mode="constant",
                cval=np.nan,
                prefilter=False,
            )

        valid_ref = in_bounds & np.isfinite(ref_vals)
        self.out_of_bounds_points += int(np.count_nonzero(~in_bounds))
        self.invalid_reference_points = int(np.count_nonzero(np.isnan(self.ref_data)))

        diff = np.abs(rz - ref_vals)
        is_outlier = np.zeros(len(chunk), dtype=bool)

        if self.mode == "iho":
            # IHO S-44 Formula: TVU = sqrt(a^2 + (b * depth)^2)
            allowable_error = np.sqrt(self.iho_a**2 + (self.iho_b * ref_vals) ** 2)
            is_outlier = (diff > allowable_error) & valid_ref
        elif self.mode == "percent":
            with np.errstate(divide="ignore", invalid="ignore"):
                pct_diff = (diff / np.abs(ref_vals)) * 100
                is_outlier = (pct_diff > self.threshold) & valid_ref
        else:
            is_outlier = (diff > self.threshold) & valid_ref

        chunk_drops = int(np.sum(is_outlier))
        self.dropped_points += chunk_drops
        self.total_points += len(chunk)

        if self.total_points > 0 and self.total_points % 1000000 < len(chunk):
            logger.debug(
                f"[RQ] Heartbeat: Filtered {self.dropped_points:,} outliers out of "
                f"{self.total_points:,} points evaluated..."
            )

        return is_outlier

    def teardown(self):
        if self.total_points > 0:
            pct_dropped = (self.dropped_points / self.total_points) * 100
            logger.debug(
                f"[RQ] Complete: Removed {self.dropped_points:,} outliers "
                f"({pct_dropped:.2f}%) from {self.total_points:,} total points. "
                f"Reference invalid: {self.invalid_reference_points:,}; "
                f"out of bounds: {self.out_of_bounds_points:,}."
            )

        if hasattr(self, "src"):
            self.src.close()

        # RQ reference surfaces are persistent derived artifacts in the RQ
        # cache.
        if hasattr(super(), "teardown"):
            super().teardown()
