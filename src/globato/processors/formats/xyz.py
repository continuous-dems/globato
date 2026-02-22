#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.formats.xyz
~~~~~~~~~~~~~

Process XYZ/ASCII files

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import sys
import logging
import warnings
import numpy as np
from typing import Optional, List, Union

from fetchez.hooks import FetchHook
from fetchez.utils import int_or, float_or

logger = logging.getLogger(__name__)


class XYZReader:
    """Chunked reader for ASCII XYZ data.

    Adapted from cudem.xyzfile to use numpy for speed.
    """

    KNOWN_DELIMS = [',', ' ', '\t', ';', '|']

    def __init__(self,
                 src_fn: str,
                 xpos=0, ypos=1, zpos=2, wpos=None, upos=None,
                 skip=0, delim=None, x_scale=1, y_scale=1, z_scale=1,
                 x_offset=0, y_offset=0, chunk_size=100_000):

        self.src_fn = src_fn
        self.xpos = int_or(xpos, 0)
        self.ypos = int_or(ypos, 1)
        self.zpos = int_or(zpos, 2)
        self.wpos = int_or(wpos)
        self.upos = int_or(upos)
        self.skip = int_or(skip, 0)
        self.delim = delim

        self.x_scale = float_or(x_scale, 1)
        self.y_scale = float_or(y_scale, 1)
        self.z_scale = float_or(z_scale, 1)
        self.x_offset = float_or(x_offset, 0)
        self.y_offset = float_or(y_offset, 0)

        self.chunk_size = chunk_size
        self.transform = (self.x_scale != 1 or self.y_scale != 1 or self.z_scale != 1 or
                          self.x_offset != 0 or self.y_offset != 0)

        # Detect delimiter if not provided
        if self.delim is None:
            self.delim = self._guess_delim()


    def _guess_delim(self):
        """Peek at the file to guess the delimiter."""

        try:
            with open(self.src_fn, 'r') as f:
                for _ in range(self.skip):
                    f.readline()

                for _ in range(5):
                    line = f.readline()
                    if not line: break
                    if line.strip().startswith('#'): continue

                    for d in self.KNOWN_DELIMS:
                        if len(line.split(d)) > 2:
                            return d
        except Exception:
            pass
        return None

    def yield_chunks(self):
        """Stream read the source and write to dst_fn in standard XYZ format.

        Returns the path if successful, None otherwise.
        """

        cols = [self.xpos, self.ypos, self.zpos]
        names = ['x', 'y', 'z']

        if self.wpos is not None:
            cols.append(self.wpos)
            names.append('w')

        if self.upos is not None:
            cols.append(self.upos)
            names.append('u')

        try:
            with open(self.src_fn, 'r') as f_in:
                current_skip = self.skip
                while True:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            chunk_data = np.loadtxt(
                                f_in,
                                delimiter=self.delim,
                                comments='#',
                                usecols=cols,
                                skiprows=current_skip,
                                ndmin=2,
                                max_rows=self.chunk_size
                            )
                            current_skip = 0

                            if chunk_data.size == 0:
                                break

                            if self.transform:
                                chunk_data[:, 0] = (chunk_data[:, 0] + self.x_offset) * self.x_scale
                                chunk_data[:, 1] = (chunk_data[:, 1] + self.y_offset) * self.y_scale
                                chunk_data[:, 2] *= self.z_scale

                            x = chunk_data[:, 0]
                            y = chunk_data[:, 1]
                            z = chunk_data[:, 2]

                            w = np.ones_like(z)
                            u = np.zeros_like(z)

                            if self.wpos is not None: w = chunk_data[:, 3]
                            if self.upos is not None: u = chunk_data[:, 4]

                            out_chunk = np.rec.fromarrays([x, y, z, w, u], names=['x','y','z','w','u'])
                            yield out_chunk

                    except StopIteration:
                        break
                    except ValueError as e:
                        if "lines" in str(e): break
                        raise e

        except Exception as e:
            logger.error(f'XYZ processing failed for {self.src_fn}: {e}')
            return None


class XYZStream(FetchHook):
    """Standardize ASCII XYZ data.

    Can reorder columns, handle delimiters, skip headers, and rescale units.

    Usage:
      --hook process_xyz:xpos=1,ypos=0,skip=1,z_scale=0.3048
    """

    name = "xyz_stream"
    stage = "file"
    desc = "Standardize ASCII XYZ data"
    category = "format-stream"

    def __init__(self, xpos=0, ypos=1, zpos=2, wpos=None, upos=None,
                 skip=0, delim=None, x_scale=1, y_scale=1, z_scale=1,
                 x_offset=0, y_offset=0, keep_raw=False, **kwargs):
        super().__init__(**kwargs)

        self.params = {
            'xpos': xpos, 'ypos': ypos, 'zpos': zpos,
            'wpos': wpos, 'upos': upos, 'skip': skip, 'delim': delim,
            'x_scale': x_scale, 'y_scale': y_scale, 'z_scale': z_scale,
            'x_offset': x_offset, 'y_offset': y_offset
        }
        self.keep_raw = str(keep_raw).lower() == 'true'

    def run(self, entries):
        new_entries = []

        for mod, entry in entries:
            src = entry.get('dst_fn')

            if not src or not os.path.exists(src):
                new_entries.append((mod, entry))
                continue

            try:
                reader = XYZReader(src, **self.params)
                entry['stream'] = reader.yield_chunks()
                entry['stream_type'] = 'xyz_recarray'
            except Exception as e:
                logger.warning(f"XYZStream failed for {src}: {e}")

        return entries


# class XYZProcessor(FetchHook):
#     """Standardize ASCII XYZ data.

#     Can reorder columns, handle delimiters, skip headers, and rescale units.
#     Outputs a clean, space-delimited .xyz file.

#     Usage:
#       --hook process_xyz:xpos=1,ypos=0,skip=1,z_scale=0.3048
#     """

#     name = "process_xyz"
#     stage = "file"
#     desc = "Standardize ASCII XYZ data"

#     def __init__(self, xpos=0, ypos=1, zpos=2, skip=0, delim=None,
#                  x_scale=1, y_scale=1, z_scale=1,
#                  x_offset=0, y_offset=0, keep_raw=False, **kwargs):
#         super().__init__(**kwargs)

#         self.params = {
#             'xpos': xpos, 'ypos': ypos, 'zpos': zpos,
#             'skip': skip, 'delim': delim,
#             'x_scale': x_scale, 'y_scale': y_scale, 'z_scale': z_scale,
#             'x_offset': x_offset, 'y_offset': y_offset
#         }
#         self.keep_raw = str(keep_raw).lower() == 'true'

#     def run(self, entries):
#         new_entries = []

#         for mod, entry in entries:
#             src = entry.get('dst_fn')

#             if not src or not os.path.exists(src):
#                 new_entries.append((mod, entry))
#                 continue

#             dst = f"{src}_clean.xyz"

#             reader = XYZReader(src, **self.params)
#             result = reader.process(dst)

#             if result and os.path.exists(result) and os.path.getsize(result) > 0:
#                 entry['dst_fn'] = result
#                 entry['raw_fn'] = src
#                 entry['data_type'] = 'xyz'

#                 if not self.keep_raw:
#                     try:
#                         os.remove(src)
#                     except OSError:
#                         pass
#             else:
#                 if result and os.path.exists(result):
#                     os.remove(result)

#             new_entries.append((mod, entry))

#         return new_entries
