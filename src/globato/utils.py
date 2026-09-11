#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.utils
~~~~~~~~~~~~~

Some utility functions for globato. Taken from cudem.utils

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import sys
import shutil
import subprocess
import logging
import io

from tqdm import tqdm
import numpy as np
from numpy.lib.recfunctions import append_fields

from fetchez.core import run_fetchez
from fetchez.registry import ModuleRegistry, BundleRegistry
from fetchez.utils import str2inc
from fetchez.recipe import Recipe

# from fetchez.utils import parse_source_string as fetchez_parse_source

from transformez.utils import cmd_exists

logger = logging.getLogger(__name__)


def run_cmd(cmd, data_fun=None, verbose=False, cwd="."):
    """Run a system command while optionally passing data.

    `data_fun` should be a function to write to a file-port:
    >> data_fun = lambda p: datalist_dump(wg, dst_port = p, ...)
    """

    out = None
    cols, _ = shutil.get_terminal_size()
    width = cols - 55

    with tqdm(desc=f"`{cmd.rstrip()[:width]}...`", leave=verbose) as pbar:
        pipe_stdin = subprocess.PIPE if data_fun is not None else None

        p = subprocess.Popen(
            cmd,
            shell=True,
            stdin=pipe_stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            cwd=cwd,
        )

        if data_fun is not None:
            if verbose:
                logger.info("Piping data to cmd subprocess...")
            data_fun(p.stdin)
            p.stdin.close()

        io_reader = io.TextIOWrapper(p.stderr, encoding="utf-8")
        while p.poll() is None:
            err_line = io_reader.readline()
            if verbose and err_line:
                pbar.write(err_line.rstrip())
                sys.stderr.flush()
            pbar.update()

        out = p.stdout.read()
        p.stderr.close()
        p.stdout.close()

        if verbose:
            logger.info(f"Ran cmd {cmd.rstrip()} and returned {p.returncode}")

    return out, p.returncode


def yield_cmd(cmd, data_fun=None, verbose=False, cwd="."):
    """Yield output from a system command.

    `data_fun` should be a function to write to a file-port:
    >> data_fun = lambda p: datalist_dump(wg, dst_port = p, ...)
    """

    if verbose:
        logger.info(f"Running cmd {cmd.rstrip()}...")

    pipe_stdin = subprocess.PIPE if data_fun is not None else None

    p = subprocess.Popen(
        cmd,
        shell=True,
        stdin=pipe_stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        cwd=cwd,
    )

    if data_fun is not None:
        if verbose:
            logger.info("Piping data to cmd subprocess...")
        data_fun(p.stdin)
        p.stdin.close()

    io_reader = io.TextIOWrapper(p.stderr, encoding="utf-8")
    while p.poll() is None:
        err_line = io_reader.readline()
        if verbose and err_line:
            logger.error(err_line.rstrip())
            sys.stderr.flush()

        line = p.stdout.readline().decode("utf-8")
        if not line:
            break
        yield line

    p.stdout.close()
    p.stderr.close()
    if verbose:
        logger.info(f"Ran cmd {cmd.rstrip()}, returned {p.returncode}.")


def cmd_check(cmd_str, cmd_vers_str):
    """check system for availability of 'cmd_str'"""

    if cmd_exists(cmd_str):
        cmd_vers, status = run_cmd(f"{cmd_vers_str}")
        return cmd_vers.rstrip()
    return b"0"


def add_field_to_recarray(rec, name, dtype, default_val):
    """Append a new field to a structured array/recarray."""

    if name not in rec.dtype.names:
        new_col = np.full(len(rec), default_val, dtype=dtype)

        return append_fields(rec, name, new_col, usemask=False, asrecarray=True)
    return rec


# --- Region parsing ---


# --- Source and Hook parsing ---
def globatize_modules(modules, shared_cache=None, crs=None, res=None):
    cache_dir = shared_cache

    ModuleRegistry.load_all()
    BundleRegistry.load_all()

    # Expand the Modules & Bundles
    modules = Recipe({})._expand_modules(modules)
    res = str2inc(res)
    for mod in modules:
        hooks = mod.setdefault("hooks", [])

        # -- Shared Cache Directory --
        if cache_dir and mod.get("module") not in ["file", "local_fs", "stdin"]:
            mod.setdefault("args", {})["outdir"] = cache_dir

        # --- Insert the target crs into stream-reproject etc. ---
        if crs:
            reproject_hook = None
            for h in hooks:
                if h.get("name") in ["stream_reproject", "stream-reproject"]:
                    reproject_hook = h
                    break

            if reproject_hook:
                reproject_hook.setdefault("args", {})["dst_srs"] = crs
                if cache_dir:
                    reproject_hook.setdefault("args", {})["cache_dir"] = cache_dir
            else:
                hooks.insert(
                    0,
                    {
                        "name": "stream_reproject",
                        "args": {"dst_srs": crs, "cache_dir": cache_dir},
                    },
                )
        else:
            # -- Make sure the source has a stream initiator ---
            has_stream = any(
                h.get("name") in ["stream-init", "stream_data"] for h in hooks
            )
            if not has_stream:
                hooks.insert(0, {"name": "stream-init"})
                logger.debug(
                    f"Auto-injected 'stream-init' into module '{mod.get('module')}'"
                )
        if res:
            for h in hooks:
                h_args = h.get("args", {})
                for arg in h_args.keys():
                    if arg == "res":
                        h_args[arg] = res

    return modules


# --- Recipe building ---
def make_recipe_config(name, r_str, modules, hooks, crs="EPSG:4326", threads=4):
    config = {
        "project": {"name": name},
        "region": r_str,  # Provide the buffered region to the modules
        "region_srs": crs,  # The region srs
        "modules": modules,  # Use our compiled modules list
        "global_hooks": hooks,  # Use compiled global dem-building hooks
        "execution": {"threads": threads},
    }

    return config


# -- rasterio helpers ---
def is_valid_window(window_tuple):
    """Safeguard against Rasterio's zero-width truncation quirk.
    Accepts a tuple of (col_off, row_off, width, height) or a Rasterio Window.
    """

    from rasterio.windows import Window

    if isinstance(window_tuple, Window):
        w, h = window_tuple.width, window_tuple.height
    else:
        _, _, w, h = window_tuple

    return w > 0 and h > 0


def safe_window_read(src, window):
    """Reads a window from a Rasterio dataset safely.
    Prevents the GDAL/NumPy broadcasting crash on edge chunks.
    """

    if not is_valid_window(window):
        return None

    data = src.read(window=window)

    # Rasterio can still truncate at the exact file edge, so we verify
    # the returned array actually has data to broadcast against.
    if 0 in data.shape:
        return None

    return data


def _generate_barrier_hash(
    region,
    res,
    include_rivers,
    include_lakes,
    include_reefs,
    include_wetlands,
    include_breakwaters,
    include_estuaries,
    target_crs,
    mode,
):
    """Generates a short, unique 8-character MD5 hash based on spatial parameters."""

    import hashlib

    crs_str = str(target_crs).replace(":", "_").replace(" ", "_")
    region_str = (
        f"{region.xmin}_{region.xmax}_{region.ymin}_{region.ymax}"
        if region
        else "global"
    )
    hash_seed = f"{region_str}_{res}_{include_rivers}_{include_lakes}_{include_reefs}_{include_wetlands}_{include_breakwaters}_{include_estuaries}_{crs_str}{mode}"
    return hashlib.md5(hash_seed.encode("utf-8")).hexdigest()[:8]


def _get_crs(crs_obj):
    """Extracts an EPSG/WKT string from CRS objects."""

    import pyproj

    if not crs_obj:
        return "EPSG:4326"
    try:
        return pyproj.CRS.from_user_input(crs_obj).to_string()
    except Exception:
        return "EPSG:4326"


def resolve_barrier(
    barrier_str,
    region,
    outdir=None,
    res="1s",
    include_water=False,
    include_rivers=True,
    include_lakes=False,
    include_reefs=False,
    include_wetlands=True,
    include_breakwaters=True,
    include_estuaries=True,
    output_mode="binary",
    output_type="raster",
    target_crs="EPSG:4326",
):
    """Resolves a barrier string into a valid file path of the requested type and CRS.

    Auto-generates magic keywords or seamlessly converts/reprojects existing files.
    """

    import pyproj
    import rasterio
    from rasterio.features import rasterize, shapes
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.transform import from_bounds
    import shapely
    from shapely.geometry import shape
    from shapely.ops import transform as shapely_transform
    from pyogrio.raw import read as pyogrio_read, write as pyogrio_write

    if not barrier_str:
        return None

    if outdir is None:
        outdir = os.path.join(os.getcwd(), "auto_barriers")
    os.makedirs(outdir, exist_ok=True)

    barrier_lower = str(os.path.basename(barrier_str)).lower()
    magic_keywords = ["coastline", "landmask", "osm", "glob_coast"]
    resolved_path = None
    native_type = None

    # --- Resolve to a native file path (Existing or Auto-Generated) ---
    if barrier_lower not in magic_keywords:
        if os.path.exists(barrier_str):
            resolved_path = barrier_str
            native_type = (
                "vector"
                if resolved_path.endswith((".shp", ".geojson", ".gpkg", ".json"))
                else "raster"
            )
        else:
            logger.error(f"Barrier file not found: {barrier_str}")
            return None
    else:
        if not region:
            logger.error("Region is required to auto-generate a coastline barrier.")
            return None

        target_mod_name = (
            "osm_landmask"
            if barrier_lower in ["osm", "landmask", "coastline"]
            else "glob_coast"
        )
        logger.debug(f"Auto-generating barrier using {target_mod_name}...")

        generator_mod = ModuleRegistry.get_class(target_mod_name)
        if generator_mod:
            gen_instance = generator_mod(
                src_region=region,
                outdir=outdir,
                res=res,
                include_rivers=include_rivers,
                include_lakes=include_lakes,
                include_reefs=include_reefs,
                include_wetlands=include_wetlands,
                include_breakwaters=include_breakwaters,
                include_estuaries=include_estuaries,
                output_mode=output_mode,
            )
            gen_instance.run()
            run_fetchez([gen_instance])

            if gen_instance.results:
                for r in gen_instance.results:
                    artifacts = r.get("artifacts", {})

                    if output_type == "vector" and "vector_fill_holes" in artifacts:
                        resolved_path = artifacts["vector_fill_holes"]
                        native_type = "vector"
                        break
                    elif (
                        output_type == "raster"
                        and r.get("data_type") == "coastline_mask"
                    ):
                        resolved_path = r.get("dst_fn")
                        native_type = "raster"
                        break

                if not resolved_path:
                    resolved_path = gen_instance.results[0].get("dst_fn")

                    native_type = (
                        "vector"
                        if resolved_path.endswith(
                            (".shp", ".geojson", ".gpkg", ".json")
                        )
                        else "raster"
                    )

    if not resolved_path or not os.path.exists(resolved_path):
        logger.error("Failed to resolve or generate the barrier.")
        return None

    # --- Extract Native CRS via Pyogrio ---
    if native_type == "raster":
        with rasterio.open(resolved_path) as src:
            native_crs = _get_crs(src.crs)
    else:
        meta, _, _, _ = pyogrio_read(resolved_path)
        native_crs = _get_crs(meta.get("crs"))

    # Fast-path: Identity check
    if native_type == output_type and native_crs == target_crs:
        return resolved_path

    logger.debug(
        f"Barrier mismatch: Converting {native_type}({native_crs}) -> {output_type}({target_crs})"
    )

    spatial_hash = _generate_barrier_hash(
        region,
        res,
        include_rivers,
        include_lakes,
        include_reefs,
        include_wetlands,
        include_breakwaters,
        include_estuaries,
        target_crs,
        output_mode,
    )
    base_name = os.path.splitext(os.path.basename(resolved_path))[0]

    needs_reproject = native_crs != target_crs
    transformer = None
    if needs_reproject:
        transformer = pyproj.Transformer.from_crs(
            native_crs, target_crs, always_xy=True
        )

    # --- OUTPUT: VECTOR ---
    if output_type == "vector":
        out_vec_path = os.path.join(outdir, f"{base_name}_{spatial_hash}.geojson")
        if os.path.exists(out_vec_path):
            return out_vec_path

        if native_type == "raster":
            with rasterio.open(resolved_path) as src:
                image = src.read(1)
                mask = image != src.nodata if src.nodata is not None else image > 0
                geoms = [
                    shape(s)
                    for s, v in shapes(image, mask=mask, transform=src.transform)
                    if v > 0
                ]
            if needs_reproject:
                geoms = [shapely_transform(transformer.transform, g) for g in geoms]

            # Write new vector with Pyogrio
            geometry_wkb = shapely.to_wkb(geoms)
            field_data = [np.array([1] * len(geoms), dtype="int32")]
            pyogrio_write(
                out_vec_path,
                geometry_wkb,
                field_data,
                fields=["val"],
                geometry_type="Polygon",
                crs=target_crs,
                driver="GeoJSON",
            )

        else:
            # Vector to Vector: Preserve ALL attributes!
            meta, fids, geometry_wkb, fields = pyogrio_read(resolved_path)

            if needs_reproject:
                geoms = shapely.from_wkb(geometry_wkb)
                geoms = [
                    shapely_transform(transformer.transform, g) if g else None
                    for g in geoms
                ]
                geometry_wkb = shapely.to_wkb(geoms)

            # Write directly preserving the original fields (like 'class'!)
            pyogrio_write(
                out_vec_path,
                geometry_wkb,
                fields,
                fields=meta.get("fields", []),
                geometry_type=meta.get("geometry_type", "Unknown"),
                crs=target_crs,
                driver="GeoJSON",
            )

        return out_vec_path

    # --- OUTPUT: RASTER ---
    elif output_type == "raster":
        out_ras_path = os.path.join(outdir, f"{base_name}_{spatial_hash}.tif")
        if os.path.exists(out_ras_path):
            return out_ras_path

        # Vector -> Raster (Rasterize directly into target space)
        if native_type == "vector":
            if not region:
                logger.error("Region required to rasterize vector barrier.")
                return None

            meta, fids, geometry_wkb, fields = pyogrio_read(resolved_path)
            geoms = shapely.from_wkb(geometry_wkb)
            geoms = [g for g in geoms if g is not None]

            if needs_reproject:
                geoms = [shapely_transform(transformer.transform, g) for g in geoms]

            inc = str2inc(res)
            width = int(round(region.width / inc))
            height = int(round(region.height / inc))
            transform = from_bounds(
                region.xmin, region.ymin, region.xmax, region.ymax, width, height
            )

            if not geoms:
                # If there are no geometries (e.g., all water), yield an array of zeros
                mask_arr = np.zeros((height, width), dtype="uint8")
            else:
                mask_arr = rasterize(
                    geoms,
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    default_value=1,
                    dtype="uint8",
                )
            profile = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": 1,
                "dtype": "uint8",
                "crs": target_crs,
                "transform": transform,
                "nodata": 0,
                "compress": "deflate",
            }

            with rasterio.open(out_ras_path, "w", **profile) as dst:
                dst.write(mask_arr, 1)

            return out_ras_path

        # Raster -> Raster (Warp)
        else:
            with rasterio.open(resolved_path) as src:
                transform, width, height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds
                )
                profile = src.profile.copy()
                profile.update(
                    {
                        "crs": target_crs,
                        "transform": transform,
                        "width": width,
                        "height": height,
                    }
                )

                with rasterio.open(out_ras_path, "w", **profile) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest,
                    )
            return out_ras_path

    return None
