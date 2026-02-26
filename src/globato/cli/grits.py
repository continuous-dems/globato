#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The GRITS Command Line Interface.
Direct access to globato raster processors without the pipeline overhead.
"""

import sys
import argparse
import logging
import os

# Import hooks

# from globato.processors.rasters.zscore import RasterZScore

# For Region parsing
from transformez.spatial import TransRegion

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("grits")


def run_hook(hook_instance, src, dst, region=None):
    """Helper to execute a hook in standalone mode."""

    if region:
        try:
            r_vals = [float(x) for x in region.split('/')]
            if len(r_vals) == 4:
                hook_instance.region = TransRegion(r_vals)
            else:
                logger.error("Region must be W/E/S/N")
                return
        except Exception as e:
            logger.error(f"Invalid region format: {e}")
            return

    entry = {
        'src_fn': src,
        'dst_fn': dst,
        'weight': 1.0 # Default weight if hook checks it
    }

    logger.info(f"Running {hook_instance.name}...")
    try:
        if os.path.exists(dst):
            logger.warning(f"Overwriting {dst}")

        success = hook_instance.process_raster(src, dst, entry)

        if success:
            logger.info(f"Success: {dst}")
        else:
            logger.error("Operation failed (hook returned False)")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        prog="grits",
        description="GRITS: Grid Transformation & Processing System"
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command")

    # --- Common Arguments (Parent Parser) ---
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("src", help="Input Raster")
    parent.add_argument("dst", help="Output Raster")
    parent.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parent.add_argument("--strip-bands", action="store_true", help="Strip extra bands in the output.")

    # --- DIFF ---
    p_diff = subparsers.add_parser("diff", parents=[parent], help="Calculate difference (Src - Aux)")
    p_diff.add_argument("--aux", required=True, help="Auxiliary/Reference Raster")
    p_diff.add_argument("--mode", choices=["difference", "filter"], default="difference")
    p_diff.add_argument("--threshold", type=float, help="Filter threshold")

    # --- SLOPE ---
    p_slope = subparsers.add_parser("slope", parents=[parent], help="Filter by Slope")
    p_slope.add_argument("--min", type=float, help="Min Slope")
    p_slope.add_argument("--max", type=float, help="Max Slope")

    # --- CUT ---
    p_cut = subparsers.add_parser("cut", parents=[parent], help="Cut/Mask to Region")
    p_cut.add_argument("-R", "--region", required=True, help="Region W/E/S/N")

    # --- FLATS ---
    p_flats = subparsers.add_parser("flats", parents=[parent], help="Remove Flat-Zones")
    p_flats.add_argument("--threshold", type=float, default=1, help="Minimum size of a flat-zone")

    # --- FILL ---
    p_fill = subparsers.add_parser("fill", parents=[parent], help="Fill NoData (IDW)")
    p_fill.add_argument("--dist", type=float, default=100, help="Max search distance")
    p_fill.add_argument("--smooth", type=int, default=0, help="Smoothing iterations")

    # --- MORPH ---
    p_morph = subparsers.add_parser("morph", parents=[parent], help="Morphology Ops")
    p_morph.add_argument("--op", choices=["erosion", "dilation", "opening", "closing"], default="erosion")
    p_morph.add_argument("--kernel", type=int, default=3, help="Kernel size")

    # --- INTERP ---
    p_interp = subparsers.add_parser("interp", parents=[parent], help="Interpolate Gaps")
    p_interp.add_argument("--method", choices=["linear", "cubic", "nearest"], default="linear")

    # --- BLEND ---
    p_blend = subparsers.add_parser("blend", parents=[parent], help="Blend rasters (Src -> Aux)")
    p_blend.add_argument("--aux", required=True, help="Auxiliary/Reference Raster")
    p_blend.add_argument("--blend_dist", type=float, default=20, help="Max blend distance")
    p_blend.add_argument("--core_dist", type=float, default=5, help="Max core blend distance")
    p_blend.add_argument("--slope_scale", type=float, default=.5, help="Normalize the slope-gate")
    p_blend.add_argument("--random_scale", type=float, default=.05, help="Density of random points for buffer")

    # --- ZSCORE ---
    p_zscore = subparsers.add_parser("zscore", parents=[parent], help="Filter based on zscore")
    p_zscore.add_argument("--threshold", type=float, default=3.0, help="Mask zscore over this threshold")
    p_zscore.add_argument("--size", type=int, default=5, help="The size of the neighborhood window.")

    args = parser.parse_args()

    hook = None
    region = None

    if args.command == "diff":
        from globato.processors.rasters.diff import RasterDiff

        hook = RasterDiff(
            aux_path=args.aux,
            mode=args.mode,
            threshold=args.threshold
        )

    elif args.command == "slope":
        from globato.processors.rasters.slope import RasterSlopeFilter

        hook = RasterSlopeFilter(
            min_val=args.min,
            max_val=args.max
        )

    elif args.command == "cut":
        from globato.processors.rasters.cut import RasterCut

        # RasterCut needs 'region' injected
        hook = RasterCut()
        region = args.region

    elif args.command == "flats":
        from globato.processors.rasters.flats import RasterFlats

        hook = RasterFlats(
            size_threshold=args.threshold,
        )

    elif args.command == "fill":
        from globato.processors.rasters.fill import RasterFill

        hook = RasterFill(
            max_dist=args.dist,
            smoothing=args.smooth
        )

    elif args.command == "morph":
        from globato.processors.rasters.morphology import RasterMorphology

        hook = RasterMorphology(
            op=args.op,
            kernel=args.kernel
        )

    elif args.command == "interp":
        from globato.processors.rasters.scipy_griddata import ScipyInterp

        hook = ScipyInterp(
            method=args.method
        )

    elif args.command == "blend":
        from globato.processors.rasters.blend import RasterBlend

        hook = RasterBlend(
            aux_path=args.aux,
            blend_dist=args.blend_dist,
            core_dist=args.core.dist,
            slope_scale=args.slope_scale,
            random_scale=args.random_scale,
        )

    elif args.command == "zscore":
        from globato.processors.rasters.zscore import RasterZScore

        hook = RasterZScore(threshold=args.threshold, kernel_size=args.size)

    if args.strip_bands:
        hook.strip_bands = True

    run_hook(hook, args.src, args.dst, region=region)

if __name__ == "__main__":
    main()
