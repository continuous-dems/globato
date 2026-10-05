#!/usr/bin/env python

"""
globato.streams.readers.xyz
~~~~~~~~~~~~~

Process XYZ/ASCII files

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging
import warnings

import numpy as np
from fetchez.utils import float_or, int_or

from .base import BaseGlobatoReader

logger = logging.getLogger(__name__)


class XYZReader(BaseGlobatoReader):
    """Chunked reader for ASCII XYZ data.

    Adapted from cudem.xyzfile to use numpy for speed.
    """

    name = "xyz-point-reader"
    meta_category = "point-stream"
    meta_desc = "Read ASCII xyz data into a point stream (xyz-recarray)"
    meta_dtype = "xyz"
    meta_extensions = ["xyz", "dat"]

    KNOWN_DELIMS = [",", " ", "\t", ";", "|"]

    def __init__(
        self,
        path: str,
        xpos=0,
        ypos=1,
        zpos=2,
        wpos=None,
        upos=None,
        skiprows=0,
        delimiter=None,
        x_scale=1,
        y_scale=1,
        z_scale=1,
        x_offset=0,
        y_offset=0,
        chunk_size=100_000,
        **kwargs,
    ):
        """Accepts generic args like `skiprows` and `delimiter` to match StreamFactory profiles,
        while maintaining legacy cudem compatability (`xpos`, `skip`, `delim`).
        """

        super().__init__(path, **kwargs)

        self.src_fn = path

        self.xpos = int_or(
            kwargs.get("usecols", [xpos])[0] if "usecols" in kwargs else xpos, 0
        )
        self.ypos = int_or(
            kwargs.get("usecols", [0, ypos])[1]
            if "usecols" in kwargs and len(kwargs["usecols"]) > 1
            else ypos,
            1,
        )
        self.zpos = int_or(
            kwargs.get("usecols", [0, 1, zpos])[2]
            if "usecols" in kwargs and len(kwargs["usecols"]) > 2
            else zpos,
            2,
        )

        self.wpos = int_or(wpos)
        self.upos = int_or(upos)

        self.skip = int_or(kwargs.get("skip", skiprows), 0)
        self.delim = kwargs.get("delim", delimiter)

        self.x_scale = float_or(x_scale, 1)
        self.y_scale = float_or(y_scale, 1)
        self.z_scale = float_or(z_scale, 1)
        self.x_offset = x_offset
        self.rem = False
        if self.x_offset == "REM":
            self.x_offset = 0
            self.rem = True
        else:
            self.x_offset = float_or(x_offset, 0)
        self.y_offset = float_or(y_offset, 0)
        self.chunk_size = chunk_size
        self.transform = (
            self.x_scale != 1
            or self.y_scale != 1
            or self.z_scale != 1
            or self.x_offset != 0
            or self.y_offset != 0
        )

        # Detect delimiter if not provided
        if self.delim in [" ", "\t", ""]:
            self.delim = None

        if self.delim is None:
            self.delim = self._guess_delim()

    def _guess_delim(self):
        """Peek at the file to guess the delimiter."""

        try:
            with open(self.src_fn, "r") as f:
                for _ in range(self.skip):
                    f.readline()

                for _ in range(5):
                    line = f.readline()
                    if not line:
                        break

                    if line.strip().startswith("#"):
                        continue

                    for d in [",", ";", "|"]:
                        if len(line.split(d)) > 2:
                            return d

                    if len(line.split()) > 2:
                        return None

                    # for d in self.KNOWN_DELIMS:
                    #     if len(line.split(d)) > 2:
                    #         return d
        except Exception:
            pass
        return None

    def _yield_raw_chunks(self):
        """Stream read the source and yield standardized XYZ recarrays."""

        cols_to_extract = [self.xpos, self.ypos, self.zpos]
        if self.wpos is not None:
            cols_to_extract.append(self.wpos)

        if self.upos is not None:
            cols_to_extract.append(self.upos)

        sorted_cols = sorted(list(set(cols_to_extract)))

        col_map = {
            physical_idx: numpy_idx
            for numpy_idx, physical_idx in enumerate(sorted_cols)
        }

        try:
            with open(self.src_fn, "r") as f_in:
                # Apply skip only to the very first chunk
                for _ in range(self.skip):
                    f_in.readline()

                while True:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            # Load a chunk of data into memory
                            chunk_data = np.loadtxt(
                                f_in,
                                delimiter=self.delim,
                                comments="#",
                                usecols=sorted_cols,
                                ndmin=2,
                                max_rows=self.chunk_size,
                            )

                            if chunk_data.size == 0:
                                break

                            x = chunk_data[:, col_map[self.xpos]]
                            y = chunk_data[:, col_map[self.ypos]]
                            z = chunk_data[:, col_map[self.zpos]]

                            # logger.info(f"file: {self.src_fn} | z: {z}")
                            if self.transform:
                                x = (x + self.x_offset) * self.x_scale
                                y = (y + self.y_offset) * self.y_scale
                                z = z * self.z_scale
                            # logger.info(f"file: {self.src_fn} | z_scaled: {z}")
                            if self.rem:
                                x = np.fmod(x + 180, 360) - 180

                            w = np.ones_like(z)
                            u = np.zeros_like(z)

                            if self.wpos is not None:
                                w = chunk_data[:, col_map[self.wpos]]

                            if self.upos is not None:
                                u = chunk_data[:, col_map[self.upos]]

                            out_chunk = np.core.records.fromarrays(
                                [x, y, z, w, u], names=["x", "y", "z", "w", "u"]
                            )
                            yield out_chunk

                    except StopIteration:
                        break
                    except ValueError as e:
                        if "lines" in str(e):
                            break
                        raise e

        except Exception as e:
            logger.error(f"XYZ processing failed for {self.src_fn}: {e}")
            return None
