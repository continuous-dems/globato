#!/usr/bin/env python

import logging
import threading

import numpy as np

from .base import BaseGlobatoReader

logger = logging.getLogger(__name__)

NC_LOCK = threading.Lock()


class NetCDFReader(BaseGlobatoReader):
    """Generic format reader to stream point data from NetCDF files."""

    name = "netcdf_reader"
    meta_dtype = "netcdf"
    meta_extensions = ["nc", "nc4"]
    meta_desc = "Read NetCDF data into a point stream using only netCDF4 and numpy."
    meta_category = "point-stream"

    def __init__(
        self,
        path,
        region=None,
        x_var=None,
        y_var=None,
        z_var=None,
        weight_var=None,
        unc_var=None,
        class_var=None,
        conf_var=None,
        chunk_size=100000,
        **kwargs,
    ):
        super().__init__(path, **kwargs)
        self.region = region
        self.x_var = x_var
        self.y_var = y_var
        self.z_var = z_var
        self.weight_var = weight_var
        self.unc_var = unc_var
        self.class_var = class_var
        self.conf_var = conf_var
        self.chunk_size = int(chunk_size)

    def _yield_raw_chunks(self):
        try:
            from h5netcdf.legacyapi import Dataset
        except ImportError:
            logger.error(f"[{self.name}] The 'h5netcdf' python library is required.")
            return

        try:
            with NC_LOCK:
                with Dataset(self.path, "r") as nc:
                    vars_dict = nc.variables.keys()

                    x_col = self.x_var or next(
                        (
                            c
                            for c in vars_dict
                            if c.lower() in ["x", "lon", "longitude"]
                        ),
                        None,
                    )
                    y_col = self.y_var or next(
                        (c for c in vars_dict if c.lower() in ["y", "lat", "latitude"]),
                        None,
                    )
                    z_col = self.z_var or next(
                        (
                            c
                            for c in vars_dict
                            if c.lower()
                            in ["z", "elev", "elevation", "height", "depth"]
                        ),
                        None,
                    )

                    if not (x_col and y_col and z_col):
                        logger.error(
                            f"[{self.name}] Could not resolve coordinates in {self.path}"
                        )
                        return

                    x_full = nc.variables[x_col][:]
                    y_full = nc.variables[y_col][:]

                    if self.region and x_full.ndim == 1 and y_full.ndim == 1:
                        w, e, s, n = self.region

                        x_mask = (x_full >= w) & (x_full <= e)
                        y_mask = (y_full >= s) & (y_full <= n)

                        if not np.any(x_mask) or not np.any(y_mask):
                            logger.debug(
                                f"[{self.name}] Region {self.region} is entirely outside {self.path}"
                            )
                            return

                        x_idx, y_idx = np.where(x_mask)[0], np.where(y_mask)[0]

                        x1, x2 = x_idx[0], x_idx[-1] + 1
                        y1, y2 = y_idx[0], y_idx[-1] + 1

                        x_data = x_full[x1:x2]
                        y_data = y_full[y1:y2]
                        z_data = nc.variables[z_col][y1:y2, x1:x2]

                        opt_data = {}
                        for v_attr, v_name in [
                            ("w", self.weight_var),
                            ("u", self.unc_var),
                            ("classification", self.class_var),
                            ("confidence", self.conf_var),
                        ]:
                            if v_name and v_name in vars_dict:
                                opt_data[v_attr] = nc.variables[v_name][y1:y2, x1:x2]

                    else:
                        x_data = x_full
                        y_data = y_full
                        z_data = nc.variables[z_col][:]

                        opt_data = {}
                        for v_attr, v_name in [
                            ("w", self.weight_var),
                            ("u", self.unc_var),
                            ("classification", self.class_var),
                            ("confidence", self.conf_var),
                        ]:
                            if v_name and v_name in vars_dict:
                                opt_data[v_attr] = nc.variables[v_name][:]

        except Exception as e:
            logger.error(f"[{self.name}] Failed to open/read NetCDF {self.path}: {e}")
            return

        # =================================================================
        # YIELD Data
        # =================================================================
        try:
            if z_data.ndim == 2 and x_data.ndim == 1 and y_data.ndim == 1:
                x_data, y_data = np.meshgrid(x_data, y_data)

            x_flat = x_data.flatten()
            y_flat = y_data.flatten()
            z_flat = z_data.flatten()

            if np.ma.isMaskedArray(z_flat):
                valid_mask = ~z_flat.mask & ~np.isnan(z_flat.data)
                z_flat = z_flat.data
            else:
                valid_mask = ~np.isnan(z_flat)

            chunk_arrays = {
                "x": x_flat[valid_mask].astype(np.float64),
                "y": y_flat[valid_mask].astype(np.float64),
                "z": z_flat[valid_mask].astype(np.float32),
            }
            dtypes = [("x", "f8"), ("y", "f8"), ("z", "f4")]

            type_map = {
                "w": "f4",
                "u": "f4",
                "classification": "u1",
                "confidence": "i2",
            }
            for v_attr, data in opt_data.items():
                if data.ndim == 2 and x_data.ndim == 2:
                    data = data.flatten()

                if np.ma.isMaskedArray(data):
                    data = data.data

                chunk_arrays[v_attr] = data[valid_mask].astype(type_map[v_attr])
                dtypes.append((v_attr, type_map[v_attr]))

            total_points = len(chunk_arrays["z"])
            for i in range(0, total_points, self.chunk_size):
                end = min(i + self.chunk_size, total_points)
                size = end - i

                chunk = np.zeros(size, dtype=dtypes)
                for name in chunk_arrays:
                    chunk[name] = chunk_arrays[name][i:end]

                yield chunk

        except Exception as e:
            logger.error(
                f"[{self.name}] Failed to process array chunks for {self.path}: {e}"
            )
