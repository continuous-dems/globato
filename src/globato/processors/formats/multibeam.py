#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.formats.multibeam
~~~~~~~~~~~~~~~~~~~

Multibeam Reader.
Require MB-System to be installed on system.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import io
import json
import subprocess
import logging
import requests
import numpy as np
import pandas as pd
from shapely.geometry import box, mapping

from fetchez.hooks import FetchHook
from fetchez.core import FetchModule
from fetchez.utils import int_or, float_or
from fetchez.presets import register_global_preset
from fetchez.presets import register_module_preset

from ..utils import yield_cmd

logger = logging.getLogger(__name__)

class MBSReader:
    """Providing an mbsystem parser.

    Process MB-System supported multibeam data files.
    """

    def __init__(self,
                 src_fn: str,
                 region=None,
                 mb_fmt=None,
                 mb_exclude='A',
                 want_mbgrid=False,
                 want_binned=False,
                 min_year=None,
                 auto_weight=True,
                 auto_uncertainty=True,
                 want_filtered=False,
                 **kwargs):

        self.src_fn = src_fn

        self.region = region
        self.mb_fmt = mb_fmt
        self.mb_exclude = mb_exclude
        self.want_mbgrid = want_mbgrid
        self.want_binned = want_binned
        self.min_year = min_year
        self.auto_weight = auto_weight
        self.auto_uncertainty = auto_uncertainty
        self.want_filtered = want_filtered

        self.weight = 1
        # if self.src_srs is None:
        #     self.src_srs = 'epsg:4326'


    def _get_mbs_meta(self, src_inf):
        """Extract metadata from mbsystem inf file."""

        meta = {'format': None, 'date': None, 'perc_good': None}

        if not os.path.exists(src_inf): return meta

        try:
            with open(src_inf, errors='ignore') as f:
                for line in f:
                    parts = line.split()
                    if not parts: continue

                    if parts[0].strip() == 'MBIO':
                        meta['format'] = parts[4]
                    elif parts[0] == 'Time:':
                        meta['date'] = parts[3]

                    if ':' in line:
                        p = line.split(':')
                        if p[0].strip() == 'Number of Good Beams':
                            meta['perc_good'] = p[1].split()[-1].split('%')[0]
        except Exception:
            pass

        return meta


    def read_mblist_ds(self):
        """Reads mblist data into a DataFrame, calculates uncertainty/weights,
        and filters noise.
        """

        # Determine format/metadata
        src_inf = f'{self.src_fn}.inf'
        meta = self._get_mbs_meta(src_inf)

        mb_format = meta['format']
        if self.src_fn.endswith('.fbt'): mb_format = None

        # Base Weight Calculation (Age Decay)
        age_weight = 1.0
        if self.auto_weight:
            if meta['date']:
                # Decay weight based on age (1980 baseline)
                age_weight = min(0.99, max(0.01, 1 - ((2024 - int(meta['date'])) / (2024 - 1980))))

            if self.weight is not None:
                self.weight *= age_weight

        # Build mblist Command
        #mb_region = None
        if self.region is not None:
            #mb_region = self.region.copy()
            #mb_region.buffer(pct=5)
            w, e, s, n = self.region

            #region_arg = f" {mb_region.format('gmt')}" if mb_region else ""
            region_arg = f' -R{w}/{e}/{s}/{n}'
        else:
            region_arg = ''

        fmt_arg = f' -F{mb_format}' if mb_format else ''

        # O-flags: XYZ, Distance(D), Angle(A), Grazing(G), Flag(g),
        # Pitch(P), p(draft), Roll(R), r(heave), Speed(S), Course(C), c(headings), etc.
        cmd_full = f'mblist -M{self.mb_exclude} -OXYZDAGgFPpRrSCcELH#{region_arg} -I{self.src_fn}{fmt_arg}'

        column_names = ['x', 'y', 'z', 'crosstrack_distance', 'crosstrack_slope',
                        'flat_bottom_grazing_angle', 'seafloor_grazing_angle',
                        'beamflag', 'pitch', 'draft', 'roll', 'heave', 'speed',
                        'sonar_alt', 'sonar_depth', 'alongtrack_distance', 'cumulative_alongtrack_distance',
                        'heading', 'beam_number', 'w', 'u']

        # Execute mblist and Parse
        try:
            raw_data = [
                [float(x) for x in line.strip().split('\t')]
                for line in yield_cmd(cmd_full, verbose=False)
            ]
        except ValueError:
            logger.info('Parsed invalid data in mblist output.')
            return pd.DataFrame(columns=column_names)

        if not raw_data:
            return pd.DataFrame(columns=column_names)

        df = pd.DataFrame(raw_data)

        # ==============================================
        # Mapping columns explicitly based on -O flags:
        # X Y Z D A G g F P p R r S C c E L H #
        # 0:x, 1:y, 2:z, 3:xtrack, 4:xtrack_slope, 5:flat_angle, 6:seafloor_angle,
        # 7:beamflag, 8:pitch, 9:draft, 10:roll, 11:heave, 12:speed,
        # 13:sonar_alt, 14:sonar_depth, 15:along_dist, 16:cum_along, 17:heading, 18:beam_num
        # ==============================================
        rename_map = {
            0: 'x', 1: 'y', 2: 'z', 3: 'crosstrack_distance', 4: 'crosstrack_slope',
            5: 'flat_bottom_grazing_angle', 6: 'seafloor_grazing_angle', 7: 'beamflag',
            8: 'pitch', 9: 'draft', 10: 'roll', 11: 'heave', 12: 'speed',
            13: 'sonar_alt', 14: 'sonar_depth', 15: 'alongtrack_distance',
            16: 'cumulative_alongtrack_distance', 17: 'heading', 18: 'beam_number'
        }
        df.rename(columns=rename_map, inplace=True)
        df = df[rename_map.values()]

        # Calculate Uncertainty and Weight
        if self.auto_weight:
            # U_depth = 0.51 * (0.25 + 0.02 * depth)
            u_depth = (0.25 + (0.02 * df['z'].abs())) * 0.51

            # U_xtrack = 0.005 * abs(xtrack)
            u_xtrack = 0.005 * df['crosstrack_distance'].abs()

            # U_speed: tmp_speed = min(14, abs(speed - 14)) * 0.51
            speed_diff = (df['speed'] - 14).abs()
            tmp_speed = np.minimum(14, speed_diff)
            u_speed = tmp_speed * 0.51

            # Total Uncertainty (TVU)
            df['u'] = np.sqrt(u_depth**2 + u_xtrack**2 + u_speed**2)

            # Weight = 1/U (avoiding divide by zero)
            df['w'] = np.where(df['u'] > 0, 1.0 / df['u'], 1.0)

            # Apply the Age Decay weight calculated earlier
            df['w'] *= age_weight
        else:
            df['u'] = 0.0
            df['w'] = self.weight if self.weight else 1.0

        # Apply beam filters
        if self.want_filtered:
            df = self._filter_mbs_data(df)

        return df


    def _filter_mbs_data(self, df):
        """Internal filter to clean noise based on aux columns."""

        initial_count = len(df)

        # Beamflag (should be 0)
        if 'beamflag' in df.columns:
            df = df[df['beamflag'] == 0]

        # Speed (Remove stationary data, usually burns/noise)
        if 'speed' in df.columns:
             df = df[df['speed'] > 2.0]

        # Roll/Pitch (Remove excessive motion)
        if 'roll' in df.columns:
            df = df[df['roll'].abs() < 10.0]

        # Grazing Angle (Remove outer beam spectral noise)
        # Keep data between 20 and 160 degrees (0-90 on either side)
        if 'seafloor_grazing_angle' in df.columns:
            df = df[df['seafloor_grazing_angle'].abs() > 20.0]

        # Slope (Remove spikes)
        if 'crosstrack_slope' in df.columns:
            df = df[df['crosstrack_slope'].abs() < 50.0]

        removed_count = initial_count - len(df)
        if self.verbose and initial_count > 0:
            perc_removed = (removed_count / initial_count) * 100
            logger.info(
                f'Removed {removed_count} of {initial_count} points '
                f'({perc_removed:.2f}%) based on quality metrics.'
            )

        return df


    def yield_points(self):
        dataset = self.read_mblist_ds()
        yield dataset


    def yield_chunks(self):
        dataset = self.read_mblist_ds()
        if hasattr(dataset, 'to_records'):
            dataset = dataset.to_records(index=False)
        yield dataset
