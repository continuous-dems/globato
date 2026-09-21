#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.streams.readers.multibeam
~~~~~~~~~~~~~~~~~~~

Multibeam Reader.
Provides a native Python reader for MB-System Format 71 (.fbt).
Falls back to MB-System subprocess calls for raw vendor formats.

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import struct
import logging
import numpy as np
import pandas as pd

from .base import BaseGlobatoReader

from globato.utils import yield_cmd
from fetchez.utils import float_or

logger = logging.getLogger(__name__)


class MBSReader(BaseGlobatoReader):
    """Providing an mbsystem parser.

    Process MB-System supported multibeam data files.
    Prefers native python struct parsing for .fbt files.
    """

    name = "multibeam-point-reader"
    meta_category = "point-stream"
    meta_dtype = "multibeam (mbs)"
    meta_desc = "Read multibeam data into a point stream"
    meta_extensions = ["fbt"]

    def __init__(
        self,
        path: str,
        region=None,
        mb_fmt=None,
        mb_exclude="A",
        want_mbgrid=False,
        want_binned=False,
        min_year=None,
        auto_weight=True,
        auto_uncertainty=True,
        want_filtered=False,
        want_flagged=False,
        rot_threshold=2.0,
        roll_threshold=5.0,
        pitch_threshold=5.0,
        kinematic_penalty=0.2,
        amp_penalty=0.5,
        **kwargs,
    ):
        super().__init__(path, **kwargs)
        self.src_fn = path

        self.region = region
        self.mb_fmt = mb_fmt
        self.mb_exclude = mb_exclude
        self.want_mbgrid = want_mbgrid
        self.want_binned = want_binned
        self.min_year = min_year
        self.auto_weight = auto_weight
        self.auto_uncertainty = auto_uncertainty
        self.want_filtered = want_filtered
        self.want_flagged = want_flagged

        self.rot_threshold = float_or(rot_threshold, 2.0)
        self.roll_threshold = float_or(roll_threshold, 5.0)
        self.pitch_threshold = float_or(pitch_threshold, 5.0)
        self.kinematic_penalty = float_or(kinematic_penalty, 0.2)
        self.amp_penalty = float_or(amp_penalty, 0.5)

        self.weight = 1
        # if self.src_srs is None:
        #     self.src_srs = 'epsg:4326'

    def _get_mbs_meta(self, src_inf):
        """Extract metadata from mbsystem inf file."""

        meta = {"format": None, "date": None, "perc_good": None}

        if not os.path.exists(src_inf):
            return meta

        try:
            with open(src_inf, errors="ignore") as f:
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue

                    if parts[0].strip() == "MBIO":
                        meta["format"] = parts[4]
                    elif parts[0] == "Time:":
                        meta["date"] = parts[3]

                    if ":" in line:
                        p = line.split(":")
                        if p[0].strip() == "Number of Good Beams":
                            meta["perc_good"] = p[1].split()[-1].split("%")[0]
        except Exception:
            pass

        return meta

    def _dynamic_weight_factors(self, df, meta, kinematic_factor, amp_factor):
        import datetime

        current_year = datetime.datetime.now().year
        file_year = int(meta["date"]) if meta.get("date") else current_year - 10

        age_factor = np.clip(
            0.5 + 0.5 * ((file_year - 1980) / (current_year - 1980)),
            0.25,
            1.0,
        )

        quality_factor = 1.0
        if meta.get("perc_good"):
            quality_factor = float(meta["perc_good"]) / 100.0

        theta = np.arctan2(
            df["crosstrack_distance"].abs(),
            df["z"].abs(),
        )
        acoustic_factor = np.clip(np.cos(theta) ** 2, 0.4, 1.0)

        soft_quality = np.power(
            age_factor * quality_factor * acoustic_factor * amp_factor,
            0.25,
        )

        return soft_quality * kinematic_factor * (self.weight or 1.0)

    def read_native_fbt(self):
        """Natively reads MB-System Format 71 (.fbt) binary files.
        Supports modern V4/V5 formats and legacy V1/V2/V3 formats.
        """

        logger.debug(f"Attempting native Python binary read for {self.src_fn}")

        all_x, all_y, all_z, all_flags, all_xtrack = [], [], [], [], []
        all_heading, all_ping, all_amp = [], [], []
        ping_counter = 0

        # MB-System format ID constants
        ID_V1, ID_V2, ID_V3 = 25700, 28270, 17476
        ID_V4, ID_V5 = 22068, 22069
        ID_COMMENT1, ID_COMMENT2 = 8995, 25443

        try:
            with open(self.src_fn, "rb") as f:
                while True:
                    flag_bytes = f.read(2)
                    if not flag_bytes or len(flag_bytes) < 2:
                        break  # EOF

                    le_flag = struct.unpack("<H", flag_bytes)[0]
                    be_flag = struct.unpack(">H", flag_bytes)[0]

                    # Check Asymmetric Modern Flags
                    if le_flag in [ID_V4, ID_V5]:
                        endian = "<"
                        flag = le_flag
                    elif be_flag in [ID_V4, ID_V5]:
                        endian = ">"
                        flag = be_flag

                    # Check Symmetric Legacy Flags
                    elif le_flag in [ID_V1, ID_V2, ID_V3, ID_COMMENT1, ID_COMMENT2]:
                        flag = le_flag
                        endian = "<"  # Default for comments

                        # Peek at the year to determine the true endianness of the data!
                        if flag in [ID_V1, ID_V2, ID_V3]:
                            year_bytes = f.read(2)
                            f.seek(-2, os.SEEK_CUR)  # Rewind the peek

                            # If Little Endian yields a sane year (0-2050), it's Little Endian.
                            # (Note: Old files often stored 2-digit years like '98')
                            y_le = struct.unpack("<h", year_bytes)[0]
                            if 0 <= y_le <= 2050:
                                endian = "<"
                            else:
                                endian = ">"
                    else:
                        # logger.debug("Unknown FBT flag. Falling back to mblist.")
                        # return None
                        continue

                    # Process Comments
                    if flag == ID_COMMENT2:
                        f.read(128)
                        continue
                    elif flag == ID_COMMENT1:
                        f.read(36 + 128)
                        continue

                    # Read Data Headers
                    if flag == ID_V5:
                        h_size = 96
                        h_bytes = f.read(h_size)
                        if len(h_bytes) < h_size:
                            break
                        h = struct.unpack(endian + "dddddfffffffiiiiffbBBB", h_bytes)
                        lon, lat, sensor_depth = h[1], h[2], h[3]
                        heading = h[5]
                        beams_bath, beams_amp, pixels_ss = h[12], h[13], h[14]
                        depth_scale, distance_scale = h[16], h[17]

                    elif flag == ID_V4:
                        h_size = 88
                        h_bytes = f.read(h_size)
                        if len(h_bytes) < h_size:
                            break
                        h = struct.unpack(endian + "dddddfffffffhhhhffbBBB", h_bytes)
                        lon, lat, sensor_depth = h[1], h[2], h[3]
                        heading = h[5]
                        beams_bath, beams_amp, pixels_ss = h[12], h[13], h[14]
                        depth_scale, distance_scale = h[16], h[17]

                    elif flag == ID_V3:
                        h_size = 46
                        h_bytes = f.read(h_size)
                        if len(h_bytes) < h_size:
                            break
                        h = struct.unpack(endian + "hhhhhHHHHHHhhhhhiihhh", h_bytes)
                        lon = (h[5] / 60.0) + (h[6] / 600000.0)
                        lat = (h[7] / 60.0) + (h[8] / 600000.0) - 90.0
                        heading = h[9] * 360.0 / 65536.0
                        beams_bath, beams_amp, pixels_ss = h[11], h[12], h[13]
                        depth_scale, distance_scale, sensor_depth = (
                            h[14] / 1000.0,
                            h[15] / 1000.0,
                            h[16] / 1000.0,
                        )

                    elif flag == ID_V2:
                        h_size = 42
                        h_bytes = f.read(h_size)
                        if len(h_bytes) < h_size:
                            break
                        h = struct.unpack(endian + "hhhhhHHHHHHhhhhhhhhhh", h_bytes)
                        lon = (h[5] / 60.0) + (h[6] / 600000.0)
                        lat = (h[7] / 60.0) + (h[8] / 600000.0) - 90.0
                        heading = h[9] * 360.0 / 65536.0
                        beams_bath, beams_amp, pixels_ss = h[11], h[12], h[13]
                        depth_scale, distance_scale, sensor_depth = (
                            h[14] / 1000.0,
                            h[15] / 1000.0,
                            0.0,
                        )

                    elif flag == ID_V1:
                        h_size = 36
                        h_bytes = f.read(h_size)
                        if len(h_bytes) < h_size:
                            break
                        h = struct.unpack(endian + "hhhhhHHHHHHhhhhhhh", h_bytes)
                        lon = (h[5] / 60.0) + (h[6] / 600000.0)
                        lat = (h[7] / 60.0) + (h[8] / 600000.0) - 90.0
                        heading = h[9] * 360.0 / 65536.0
                        beams_bath, beams_amp, pixels_ss = h[11], h[12], h[13]
                        depth_scale, distance_scale, sensor_depth = (
                            h[14] / 1000.0,
                            h[15] / 1000.0,
                            0.0,
                        )

                    if (
                        beams_bath < 0
                        or beams_amp < 0
                        or pixels_ss < 0
                        or beams_bath > 10000
                    ):
                        logger.warning(
                            "Corrupt array lengths parsed. Falling back to mblist."
                        )
                        continue
                        # return None

                    if lon > 180:
                        lon -= 360

                    # Read Data Arrays
                    b_flags = f.read(beams_bath)
                    if len(b_flags) < beams_bath:
                        break
                    beamflags = np.frombuffer(b_flags, dtype=np.uint8)

                    b_bath = f.read(beams_bath * 2)
                    if len(b_bath) < beams_bath * 2:
                        break
                    bath_raw = np.frombuffer(b_bath, dtype=f"{endian}i2")

                    b_xtrack = f.read(beams_bath * 2)
                    if len(b_xtrack) < beams_bath * 2:
                        break
                    acrosstrack_raw = np.frombuffer(b_xtrack, dtype=f"{endian}i2")

                    b_ltrack = f.read(beams_bath * 2)
                    if len(b_ltrack) < beams_bath * 2:
                        break
                    alongtrack_raw = np.frombuffer(b_ltrack, dtype=f"{endian}i2")

                    if beams_amp > 0:
                        b_amp = f.read(beams_amp * 2)
                        if len(b_amp) < beams_amp * 2:
                            break
                        amp_raw = np.frombuffer(b_amp, dtype=f"{endian}i2")
                    else:
                        amp_raw = np.zeros(beams_bath)

                    if pixels_ss > 0:
                        f.read(pixels_ss * 2 * 3)

                    # if beams_amp > 0:
                    #     f.read(beams_amp * 2)
                    # if pixels_ss > 0:
                    #     f.read(pixels_ss * 2 * 3)

                    # Apply Scaling
                    bath = ((bath_raw * depth_scale) + sensor_depth) * -1
                    xtrack = acrosstrack_raw * distance_scale
                    ltrack = alongtrack_raw * distance_scale

                    heading_rad = np.radians(heading)
                    cos_h = np.cos(heading_rad)
                    sin_h = np.sin(heading_rad)

                    # Calculate physical offsets in meters
                    # (MB-System standard: X-Track is Starboard, L-Track is Forward)
                    delta_x = (ltrack * sin_h) + (xtrack * cos_h)
                    delta_y = (ltrack * cos_h) - (xtrack * sin_h)

                    # Convert meters to degrees
                    x_pos = lon + (delta_x / (111111.0 * np.cos(np.radians(lat))))
                    y_pos = lat + (delta_y / 111111.0)

                    all_x.append(x_pos)
                    all_y.append(y_pos)
                    all_z.append(bath)
                    all_flags.append(beamflags)
                    all_xtrack.append(xtrack)

                    all_heading.append(np.full(beams_bath, heading))
                    all_ping.append(np.full(beams_bath, ping_counter))

                    if len(amp_raw) == beams_bath:
                        all_amp.append(amp_raw)
                    else:
                        all_amp.append(np.zeros(beams_bath))

                    ping_counter += 1

        except Exception as e:
            logger.debug(f"Native Python FBT read failed: {e}. Falling back to mblist.")
            return None

        if not all_x:
            return None

        df = pd.DataFrame(
            {
                "x": np.concatenate(all_x),
                "y": np.concatenate(all_y),
                "z": np.concatenate(all_z),
                "beamflag": np.concatenate(all_flags),
                "crosstrack_distance": np.concatenate(all_xtrack),
                "heading": np.concatenate(all_heading),
                "ping": np.concatenate(all_ping),
                "amplitude": np.concatenate(all_amp),
            }
        )

        # Remove points where the vessel's GPS dropped out (Null Island)
        df = df[~((df["x"].abs() < 0.1) & (df["y"].abs() < 0.1))]

        if not self.want_flagged:
            df = df[df["beamflag"] == 0]

        if self.auto_uncertainty:
            u_depth = df["z"].abs() * 0.01  # 1% of depth
            # u_xtrack = df["xtrack"].abs() * 0.05 if "xtrack" in df.columns else 0.0
            u_xtrack = (
                df["crosstrack_distance"].abs() * 0.05
                if "crosstrack_distance" in df.columns
                else 0.0
            )
            # print(u_xtrack)
            # u_speed = df["speed"] * 0.51 if "speed" in df.columns else 0.0

            df["u"] = np.sqrt(u_depth**2 + u_xtrack**2)  # + u_speed**2)
        else:
            df["u"] = 0.0

        df["w"] = self.weight if self.weight else 1.0
        if self.auto_weight:
            src_inf = self.src_fn.replace(".fbt", ".inf")
            meta = self._get_mbs_meta(src_inf)

            ping_headings = df.groupby("ping")["heading"].first()
            heading_diffs = ping_headings.diff().abs()
            heading_diffs = np.minimum(heading_diffs, 360 - heading_diffs)

            df["rot"] = df["ping"].map(heading_diffs.fillna(0))

            kinematic_factor = np.where(
                df["rot"] > self.rot_threshold, self.kinematic_penalty, 1.0
            )

            if (
                "amplitude" in df.columns
                and df["amplitude"].max() > df["amplitude"].min()
            ):
                # (5th to 95th percentile) to ignore anomalous spikes
                p5 = df["amplitude"].quantile(0.05)
                p95 = df["amplitude"].quantile(0.95)

                # Normalize the amplitude from 0.0 to 1.0
                amp_norm = (df["amplitude"] - p5) / (p95 - p5 + 1e-5)

                # Map the 0.0 -> 1.0 range onto our penalty floor -> 1.0
                amp_factor = np.clip(
                    self.amp_penalty + (amp_norm * (1.0 - self.amp_penalty)),
                    self.amp_penalty,
                    1.0,
                )
            else:
                amp_factor = 1.0

            df["w"] = self._dynamic_weight_factors(
                df, meta, kinematic_factor, amp_factor
            )

        else:
            df["w"] = self.weight or 1.0

        df.drop(columns=["heading", "ping", "rot"], inplace=True, errors="ignore")

        return df

    def _yield_raw_chunks(self):
        """Yield data, attempting native reader for .fbt files with a subprocess fallback."""
        dataset = None

        if self.src_fn.lower().endswith(".fbt"):
            dataset = self.read_native_fbt()

        if dataset is None:
            dataset = self.read_mblist_ds()

        if dataset is not None and not dataset.empty:
            yield dataset.to_records(index=False)

    def read_mblist_ds(self):
        """Reads mblist data into a DataFrame, calculates uncertainty/weights,
        and filters noise.
        """

        src_inf = f"{self.src_fn}.inf"
        meta = self._get_mbs_meta(src_inf)

        mb_format = meta["format"]
        if self.src_fn.endswith(".fbt"):
            mb_format = None

        # Base Weight Calculation (Age Decay)
        age_weight = 1.0
        if self.auto_weight:
            if meta["date"]:
                # Decay weight based on age (1980 baseline)
                age_weight = min(
                    0.99, max(0.01, 1 - ((2024 - int(meta["date"])) / (2024 - 1980)))
                )

            if self.weight is not None:
                self.weight *= age_weight

        # Build mblist Command
        if self.region is not None:
            w, e, s, n = self.region
            region_arg = f" -R{w}/{e}/{s}/{n}"
        else:
            region_arg = ""

        fmt_arg = f" -F{mb_format}" if mb_format else ""

        # O-flags: XYZ, Distance(D), Angle(A), Grazing(G), Flag(g),
        # Pitch(P), p(draft), Roll(R), r(heave), Speed(S), Course(C), c(headings), etc.
        # cmd_full = f"mblist -M{self.mb_exclude} -OXYZDAGgFPpRrSCcELH#{region_arg} -I{self.src_fn}{fmt_arg}"
        cmd_full = f"mblist -M{self.mb_exclude} -OXYZDAGgFPpRrSCcELH#a{region_arg} -I{self.src_fn}{fmt_arg}"

        column_names = [
            "x",
            "y",
            "z",
            "crosstrack_distance",
            "crosstrack_slope",
            "flat_bottom_grazing_angle",
            "seafloor_grazing_angle",
            "beamflag",
            "pitch",
            "draft",
            "roll",
            "heave",
            "speed",
            "sonar_alt",
            "sonar_depth",
            "alongtrack_distance",
            "cumulative_alongtrack_distance",
            "heading",
            "beam_number",
            "w",
            "u",
            "amplitude",
        ]

        # Execute mblist and Parse
        try:
            raw_data = [
                [float(x) for x in line.strip().split("\t")]
                for line in yield_cmd(cmd_full, verbose=False)
            ]
        except ValueError:
            logger.error("Parsed invalid data in mblist output.")
            return pd.DataFrame(columns=column_names)

        if not raw_data:
            return pd.DataFrame(columns=column_names)

        df = pd.DataFrame(raw_data)

        rename_map = {
            0: "x",
            1: "y",
            2: "z",
            3: "crosstrack_distance",
            4: "crosstrack_slope",
            5: "flat_bottom_grazing_angle",
            6: "seafloor_grazing_angle",
            7: "beamflag",
            8: "'pitch",
            9: "draft",
            10: "roll",
            11: "heave",
            12: "speed",
            13: "sonar_alt",
            14: "sonar_depth",
            15: "alongtrack_distance",
            16: "cumulative_alongtrack_distance",
            17: "heading",
            18: "beam_number",
            19: "amplitude",
        }
        df.rename(columns=rename_map, inplace=True)
        df = df[rename_map.values()]

        if self.auto_weight:
            kinematic_factor = np.ones(len(df))

            if "roll" in df.columns:
                kinematic_factor = np.where(
                    df["roll"].abs() > self.roll_threshold,
                    self.kinematic_penalty,
                    kinematic_factor,
                )
            if "pitch" in df.columns:
                kinematic_factor = np.where(
                    df["pitch"].abs() > self.pitch_threshold,
                    self.kinematic_penalty,
                    kinematic_factor,
                )

            df["ping_boundary"] = (df["heading"].diff().abs() > 1e-4).cumsum()
            ping_rot = df.groupby("ping_boundary")["heading"].first().diff().abs()
            ping_rot = np.minimum(ping_rot, 360 - ping_rot).fillna(0)
            df["rot"] = df["ping_boundary"].map(ping_rot)

            kinematic_factor = np.where(
                df["rot"] > self.rot_threshold, self.kinematic_penalty, kinematic_factor
            )

            if (
                "amplitude" in df.columns
                and df["amplitude"].max() > df["amplitude"].min()
            ):
                p5 = df["amplitude"].quantile(0.05)
                p95 = df["amplitude"].quantile(0.95)
                amp_norm = (df["amplitude"] - p5) / (p95 - p5 + 1e-5)
                amp_factor = np.clip(
                    self.amp_penalty + (amp_norm * (1.0 - self.amp_penalty)),
                    self.amp_penalty,
                    1.0,
                )
            else:
                amp_factor = 1.0

            df["w"] = self._dynamic_weight_factors(
                df, meta, kinematic_factor, amp_factor
            )

        else:
            df["w"] = self.weight or 1.0

        if self.want_filtered:
            df = self._filter_mbs_data(df)

        return df

    def _filter_mbs_data(self, df):
        """Internal filter to clean noise based on aux columns."""

        initial_count = len(df)

        if "beamflag" in df.columns:
            df = df[df["beamflag"] == 0]
        if "speed" in df.columns:
            df = df[df["speed"] > 2.0]
        if "roll" in df.columns:
            df = df[df["roll"].abs() < 10.0]
        if "seafloor_grazing_angle" in df.columns:
            df = df[df["seafloor_grazing_angle"].abs() > 20.0]
        if "crosstrack_slope" in df.columns:
            df = df[df["crosstrack_slope"].abs() < 50.0]

        removed_count = initial_count - len(df)
        if (
            hasattr(self, "verbose")
            and getattr(self, "verbose", False)
            and initial_count > 0
        ):
            perc_removed = (removed_count / initial_count) * 100
            logger.info(
                f"Removed {removed_count} of {initial_count} points "
                f"({perc_removed:.2f}%) based on quality metrics."
            )

        return df

    def yield_points(self):
        dataset = self.read_mblist_ds()
        yield dataset
