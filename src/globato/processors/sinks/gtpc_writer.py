#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.sinks.gtpc_writer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Writes streams to .gtpc (Globato Point Cloud) HDF5 format.
Supports optional spatial binning to reduce file size.
"""

import os
import logging
import numpy as np
import h5py
from fetchez.hooks import FetchHook
from fetchez import utils
from ..transforms.point_pixels import PointPixels

logger = logging.getLogger(__name__)

class WriteGTPC(FetchHook):
    """Saves stream to .gtpc HDF5.

    Args:
        res (str/float): If set (e.g. "1s", "0.0003"), bins data to this resolution
                         BEFORE writing. Drastically reduces size for dense sources.
        mode (str): Binning mode ('mean', 'min', 'max'). Default 'mean'.
        compression (str): HDF5 compression ('gzip', 'lzf').
    """

    name = "write_gtpc"
    stage = "file"
    category = "stream sink"

    def __init__(self, res=None, mode="mean", compression="gzip", **kwargs):
        super().__init__(**kwargs)
        self.res = res
        self.mode = mode
        self.compression = compression

    def _parse_res(self, res_str):
        """Parse '1s' or '0.001' to float degrees."""
        if not res_str: return None
        if isinstance(res_str, (int, float)): return float(res_str)
        if isinstance(res_str, str) and res_str.endswith('s'):
            return float(res_str[:-1]) / 3600.0
        return float(res_str)

    def run(self, entries):
        for mod, entry in entries:
            stream = entry.get('stream')
            if not stream: continue

            src_fn = entry.get('dst_fn')
            base, _ = os.path.splitext(src_fn)
            out_fn = f"{base}.gtpc"

            binner = None
            if self.res:
                try:
                    inc = self._parse_res(self.res)

                    w, e, s, n = mod.region
                    width = int(np.ceil((e - w) / inc))
                    height = int(np.ceil((n - s) / inc))

                    binner = PointPixels(
                        src_region=mod.region, # PointPixels expects TransRegion or list
                        x_size=width,
                        y_size=height,
                        verbose=False
                    )
                except Exception as e:
                    logger.warning(f"Could not init binning for {out_fn}: {e}. Writing raw.")
                    binner = None

            entry['stream'] = self._write_stream(stream, out_fn, binner)

        return entries

    def _write_stream(self, stream, out_fn, binner=None):
        with h5py.File(out_fn, 'w') as f:
            grp = f.create_group("points")
            datasets = {}
            total_pts = 0

            for chunk in stream:
                if chunk is None or len(chunk) == 0: continue

                if binner:
                    arrays, _, _ = binner(chunk, mode=self.mode)
                    valid = arrays['count'] > 0
                    if not np.any(valid): continue

                    data_to_write = {
                        'x': arrays['x'][valid],
                        'y': arrays['y'][valid],
                        'z': arrays['z'][valid],
                        'w': arrays['weight'][valid],
                        'u': arrays['uncertainty'][valid]
                    }

                    n_pts = np.count_nonzero(valid)
                    dt = [('x', 'f8'), ('y', 'f8'), ('z', 'f4'), ('w', 'f4'), ('u', 'f4')]
                    chunk_out = np.rec.fromarrays([data_to_write[k] for k, _ in dt], dtype=dt)

                else:
                    chunk_out = chunk

                n_new = len(chunk_out)
                if not datasets:
                    for field in chunk_out.dtype.names:
                        datasets[field] = grp.create_dataset(
                            field, shape=(0,), maxshape=(None,),
                            dtype=chunk_out[field].dtype, compression=self.compression
                        )

                for field in datasets:
                    if field in chunk_out.dtype.names:
                        dset = datasets[field]
                        dset.resize((total_pts + n_new,))
                        dset[total_pts : total_pts + n_new] = chunk_out[field]

                total_pts += n_new
                # Matbe yield the newly binned chunk instead?
                yield chunk

            if total_pts > 0:
                logger.info(f"Saved {total_pts} points to {out_fn}")
