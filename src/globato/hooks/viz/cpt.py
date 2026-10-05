#!/usr/bin/env python

"""
globato.hooks.viz.cpt
~~~~~~~~~~~~~~~~~~~~~~~

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging

try:
    from matplotlib.colors import LinearSegmentedColormap

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from fetchez import core, registry
from fetchez.utils import float_or, int_or

logger = logging.getLogger(__name__)

## CPT Colors dictionary
CPT_COLORS = {
    "black": [0, 0, 0],
    "white": [255, 255, 255],
    "red": [255, 0, 0],
    "green": [0, 255, 0],
    "blue": [0, 0, 255],
    "yellow": [255, 255, 0],
    "cyan": [0, 255, 255],
    "magenta": [255, 0, 255],
    "gray": [128, 128, 128],
    "lightgray": [211, 211, 211],
    "darkgray": [169, 169, 169],
    "orange": [255, 165, 0],
    "purple": [128, 0, 128],
    "brown": [165, 42, 42],
}


## The following scale_el_* functions are depreciated.
def scale_el_simple(value, gmin, gmax, tr):
    """Simple scaling of elevation based on predefined ranges."""

    if value > 0 and gmax > 0:
        return (gmax * tr) / 8000
    elif value < 0 and gmin < 0:
        return (gmin * tr) / -11000
    elif value == 0:
        return 0
    else:
        print(value)
        return None


def scale_el_relative(value, gmin, gmax, tr, trs):
    """Linearly scales 'tr' from the range [min(trs), max(trs)] to [gmin, gmax].
    Lowest input -> gmin
    Highest input -> gmax
    """

    input_min = min(trs)
    input_max = max(trs)

    input_range = input_max - input_min
    output_range = gmax - gmin

    if input_range == 0:
        return gmin

    percentage = (tr - input_min) / input_range
    return gmin + (percentage * output_range)


def scale_el_relative_etopo(value, gmin, gmax, tr, trs):
    """Scaling relative to the max/min of the input ranges (trs)."""

    if value > 0 and gmax > 0:
        return (gmax * tr) / max(trs)
    elif value < 0 and gmin < 0:
        if min(trs) == 0:
            return gmin * tr
        else:
            return (gmin * tr) / min(trs)
    elif value == 0:
        return gmin
    else:
        return None


def scale_el_linear(value, gmin, gmax, tr, trs):
    """Linear scaling calculation."""

    p = (tr - min(trs)) / (max(trs) - min(trs))
    v = (1 - p) * (gmin - gmax) + gmax
    return v


def generate_etopo_cpt(gmin, gmax, output_file="tmp.cpt"):
    """Generates a CPT based on ETOPO1 color steps scaled to gmin/gmax."""

    trs = [
        -11000,
        -10500,
        -10000,
        -9500,
        -9000,
        -8500,
        -8000,
        -7500,
        -7000,
        -6500,
        -6000,
        -5500,
        -5000,
        -4500,
        -4000,
        -3500,
        -3000,
        -2500,
        -2000,
        -1500,
        -1000,
        -500,
        -0.001,
        0,
        100,
        200,
        500,
        1000,
        1500,
        2000,
        2500,
        3000,
        3500,
        4000,
        4500,
        5000,
        5500,
        6000,
        6500,
        7000,
        7500,
        8000,
    ]
    colors = (
        [10, 0, 121],
        [26, 0, 137],
        [38, 0, 152],
        [27, 3, 166],
        [16, 6, 180],
        [5, 9, 193],
        [0, 14, 203],
        [0, 22, 210],
        [0, 30, 216],
        [0, 39, 223],
        [12, 68, 231],
        [26, 102, 240],
        [19, 117, 244],
        [14, 133, 249],
        [21, 158, 252],
        [30, 178, 255],
        [43, 186, 255],
        [55, 193, 255],
        [65, 200, 255],
        [79, 210, 255],
        [94, 223, 255],
        [138, 227, 255],
        [138, 227, 255],
        [51, 102, 0],
        [51, 204, 102],
        [187, 228, 146],
        [255, 220, 185],
        [243, 202, 137],
        [230, 184, 88],
        [217, 166, 39],
        [168, 154, 31],
        [164, 144, 25],
        [162, 134, 19],
        [159, 123, 13],
        [156, 113, 7],
        [153, 102, 0],
        [162, 89, 89],
        [178, 118, 118],
        [183, 147, 147],
        [194, 176, 176],
        [204, 204, 204],
        [229, 229, 229],
        [138, 227, 255],
        [51, 102, 0],
    )
    new_elevs = []
    split_val = 0
    t_min, t_max = min(trs), max(trs)

    for t in trs:
        if t <= split_val:
            if t_min == split_val:
                pct = 0
            else:
                pct = (t - t_min) / (split_val - t_min)
            val = gmin + pct * (0 - gmin)
        else:
            if t_max == split_val:
                pct = 0
            else:
                pct = (t - split_val) / (t_max - split_val)
            val = 0 + pct * (gmax - 0)
        new_elevs.append(val)

    with open(output_file, "w") as cpt:
        for i in range(len(new_elevs) - 1):
            elev_curr = new_elevs[i]
            elev_next = new_elevs[i + 1]
            c1 = colors[i]
            cpt.write(
                f"{elev_curr} {c1[0]} {c1[1]} {c1[2]} {elev_next} {c1[0]} {c1[1]} {c1[2]}\n"
            )
    return output_file


def generate_coastal_relief_cpt(gmin, gmax, output_file="tmp.cpt"):
    """Generates a CPT based on a custom QGIS Coastal Topobathy colormap,
    scaled to gmin/gmax while anchoring the coastline (0m) exactly at 0.
    """

    trs = [
        -6000,
        -4000,
        -2500,
        -1500,
        -750,
        -25,
        -10,
        -5,
        0,
        1,
        2.5,
        5,
        10,
        25,
        50,
        100,
        250,
        500,
        1000,
    ]

    colors = [
        [18, 32, 89],
        [28, 55, 125],
        [38, 84, 158],
        [48, 115, 184],
        [68, 150, 199],
        [106, 185, 209],
        [145, 210, 217],
        [181, 225, 219],
        [214, 235, 214],
        [211, 229, 176],
        [180, 214, 143],
        [145, 190, 112],
        [111, 162, 88],
        [130, 137, 82],
        [157, 126, 83],
        [181, 143, 103],
        [199, 174, 137],
        [218, 205, 177],
        [242, 239, 230],
    ]

    new_elevs = []
    split_val = 0
    t_min, t_max = min(trs), max(trs)

    # Stretch the fixed elevation bins to fit the actual grid's Min/Max
    # while strictly pinning 0 (the coastline) to 0.
    for t in trs:
        if t <= split_val:
            if t_min == split_val:
                pct = 0
            else:
                pct = (t - t_min) / (split_val - t_min)
            val = gmin + pct * (0 - gmin)
        else:
            if t_max == split_val:
                pct = 0
            else:
                pct = (t - split_val) / (t_max - split_val)
            val = 0 + pct * (gmax - 0)
        new_elevs.append(val)

    with open(output_file, "w") as cpt:
        for i in range(len(new_elevs) - 1):
            elev_curr = new_elevs[i]
            elev_next = new_elevs[i + 1]
            c1 = colors[i]
            c2 = colors[i + 1]

            # Write Gradient CPT format (z0 r0 g0 b0 z1 r1 g1 b1)
            cpt.write(
                f"{elev_curr} {c1[0]} {c1[1]} {c1[2]} {elev_next} {c2[0]} {c2[1]} {c2[2]}\n"
            )

    return output_file


def process_cpt(cpt_file, gmin, gmax, gdal=False, split_cpt=None):
    """Stretches an existing CPT to global limits, preserving the split hinge point."""

    if cpt_file is None:
        return None

    trs, colors = [], []
    with open(cpt_file, "r") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue

            if float_or(parts[0]) is not None:
                trs.append(float(parts[0]))
                if int_or(parts[1]) is not None:
                    colors.append(
                        [
                            int(float(parts[1])),
                            int(float(parts[2])),
                            int(float(parts[3])),
                        ]
                    )
                elif parts[1] in CPT_COLORS:
                    colors.append(CPT_COLORS[parts[1]])
                elif "/" in parts[1]:
                    colors.append([int(float(x)) for x in parts[1].split("/")])

    if not trs:
        return None

    new_elevs = []
    t_min, t_max = min(trs), max(trs)

    if split_cpt is not None:
        split_val = float(split_cpt)
        for t in trs:
            if t <= split_val:
                if split_val == t_min:
                    val = gmin
                else:
                    pct = (t - t_min) / (split_val - t_min)
                    val = gmin + pct * (split_val - gmin)
            else:
                if t_max == split_val:
                    val = gmax
                else:
                    pct = (t - split_val) / (t_max - split_val)
                    val = split_val + pct * (gmax - split_val)
            new_elevs.append(val)
    else:
        for t in trs:
            if t_max == t_min:
                val = gmin
            else:
                pct = (t - t_min) / (t_max - t_min)
                val = gmin + pct * (gmax - gmin)
            new_elevs.append(val)

    output_fn = "tmp_stretched.cpt"
    with open(output_fn, "w") as f_out:
        for i in range(len(new_elevs) - 1):
            elev_curr = new_elevs[i]
            elev_next = new_elevs[i + 1]
            c = colors[i]
            if not gdal:
                f_out.write(
                    f"{elev_curr} {c[0]} {c[1]} {c[2]} {elev_next} {c[0]} {c[1]} {c[2]}\n"
                )
            else:
                f_out.write(f"{elev_curr} {c[0]} {c[1]} {c[2]} 255\n")

        if gdal and len(new_elevs) > 0:
            last_c = colors[-1]
            f_out.write(
                f"{new_elevs[-1]} {last_c[0]} {last_c[1]} {last_c[2]} 255\nnv 0 0 0 0\n"
            )

    return output_fn


def fetch_cpt_city(query="grass/haxby", out_dir=None):
    """Wraps fetchez to get the data."""

    registry.ModuleRegistry.load_builtins()
    CPTCityModule = registry.ModuleRegistry.get_class("cpt_city")
    if not CPTCityModule:
        return None

    fetcher = CPTCityModule(query=query, outdir=out_dir)
    fetcher.run()
    if not fetcher.results:
        return None

    core.run_fetchez([fetcher], threads=1)
    return fetcher.results[0]["dst_fn"]


def load_cmap(cpt_file, name="globato_cpt"):
    """Reads a CPT file and converts it to a Matplotlib Colormap respecting irregular Z spacing!"""

    try:
        with open(cpt_file, "r") as f:
            lines = f.readlines()

        z_vals, colors = [], []
        for line in lines:
            parts = line.split()
            if len(parts) >= 8:
                try:
                    z0 = float(parts[0])
                    r0, g0, b0 = (
                        float(parts[1]) / 255.0,
                        float(parts[2]) / 255.0,
                        float(parts[3]) / 255.0,
                    )
                    z1 = float(parts[4])
                    r1, g1, b1 = (
                        float(parts[5]) / 255.0,
                        float(parts[6]) / 255.0,
                        float(parts[7]) / 255.0,
                    )

                    if not z_vals:
                        z_vals.append(z0)
                        colors.append((r0, g0, b0))
                    elif z0 != z_vals[-1] or (r0, g0, b0) != colors[-1]:
                        # elif z0 != z_vals[-1]:
                        # Handle discontinuous colormaps (sharp breaks)
                        z_vals.append(z0)
                        colors.append((r0, g0, b0))

                    z_vals.append(z1)
                    colors.append((r1, g1, b1))
                except ValueError:
                    continue

        if not colors:
            return None

        z_min, z_max = min(z_vals), max(z_vals)
        z_range = z_max - z_min

        cdict = {"red": [], "green": [], "blue": []}
        if z_range == 0:
            return LinearSegmentedColormap.from_list(name, colors, N=256)

        unique_x = []
        c_left = []
        c_right = []

        for z, color in zip(z_vals, colors):
            x = (z - z_min) / z_range
            x = max(0.0, min(1.0, x))

            if unique_x:
                x = max(x, unique_x[-1])

                if x == unique_x[-1]:
                    c_right[-1] = color
                    continue

            unique_x.append(x)
            c_left.append(color)
            c_right.append(color)

        unique_x[0] = 0.0
        unique_x[-1] = 1.0

        for x, cl, cr in zip(unique_x, c_left, c_right):
            cdict["red"].append((x, cl[0], cr[0]))
            cdict["green"].append((x, cl[1], cr[1]))
            cdict["blue"].append((x, cl[2], cr[2]))

        # return LinearSegmentedColormap(name, cdict)
        return LinearSegmentedColormap(name, cdict, N=65536)

    except Exception as e:
        logger.error(f"Failed to load CPT to cmap: {e}")
        return None
