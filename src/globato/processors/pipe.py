#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.pipe
~~~~~~~~~~~~~~~~~~~~~~~

pipe the stream to xyz (stdout)

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import sys
import logging
import numpy as np
from fetchez.hooks import FetchHook

#logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger(__name__)

class XYZPrinter(FetchHook):
    """Sink Hook: Prints the XYZ stream to stdout.
    Useful for piping to other tools like GMT, MB-System, or text files.
    If the input stream is 'xyz_recarray', it prints the points,
    if the input stream is 'pointz_pixels_arrays' it prints the y/x/z pixel locations.
    
    Usage:
      dlim ... --hook pipe_xyz
      dlim ... --hook pipe_xyz:fmt=%.4f:delimiter=,
    """
    
    name = "pipe_xyz"
    stage = "file"

    def __init__(self, fmt='%.6f', delimiter=' ', **kwargs):
        super().__init__(**kwargs)
        self.fmt = fmt
        self.delimiter = delimiter

        
    def run(self, entries):
        for mod, entry in entries:
            stream = entry.get('stream')
            stream_type = entry.get('stream_type')
            if not stream:
                continue

            try:
                if stream_type == 'xyz_recarray':
                    for chunk in stream:
                        columns = [chunk['x'], chunk['y'], chunk['z']]

                        if 'w' in chunk.dtype.names:
                            columns.append(chunk['w'])
                        if 'u' in chunk.dtype.names:
                            columns.append(chunk['u'])

                        data = np.column_stack(columns)
                        np.savetxt(sys.stdout, data, fmt=self.fmt, delimiter=self.delimiter)
                        
                elif stream_type == 'pointz_pixels_arrays':
                    for arrs, srcwin, gt in stream:

                        x_vals = arrs['pixel_x'].astype(int)
                        y_vals = arrs['pixel_y'].astype(int)
                        z_vals = arrs['z'][y_vals, x_vals]

                        columns = [y_vals, x_vals, z_vals]

                        data = np.column_stack(columns)
                        self.fmt = ['%d', '%d', '%.6f']
                        np.savetxt(sys.stdout, data, fmt=self.fmt, delimiter=self.delimiter)
                    
            except IOError:
                try:
                    sys.stdout.close()
                except:
                    pass
                return entries
            except Exception as e:
                logger.error(f"Error piping XYZ: {e}\n")
                
            del entry['stream']
            del entry['stream_type']
            
        return entries
