"""Identity and integrity checks for completed native point captures."""

from functools import lru_cache
from importlib.metadata import version
import re
from pathlib import Path

import h5py
import numpy as np

from ._io import digest, identity, serializable

# Only known deterministic, non-diagnostic stream prefixes are reusable.
# Unrecognised/custom hooks still execute normally and produce fresh captures.
_PREFIX_CLASSES = {
    "fetchez.hooks.stream_init.DataStream",
    "globato.hooks.transforms.reproject.StreamReproject",
    "globato.hooks.filters.claim_grid.ClaimGridFilter",
    "globato.hooks.filters.spatial_crop.SpatialCrop",
    "globato.hooks.filters.dropclass.DropClass",
    "globato.hooks.filters.rangez.RangeZ",
    "globato.hooks.filters.point_raster_mask.PointRasterMask",
}


@lru_cache(maxsize=1)
def runtime_identity():
    """Seal reader implementations and the numerical library versions once/run."""
    import fetchez
    import globato
    import pyproj

    code = {}
    for package in (fetchez, globato):
        root = Path(package.__file__).parent
        folders = ("streams", "datatypes")
        if package is globato:
            folders += ("hooks/filters", "hooks/transforms")
        for folder in folders:
            for path in sorted((root / folder).rglob("*")):
                if path.is_file() and path.suffix in (".py", ".yaml", ".yml"):
                    code[f"{package.__name__}/{path.relative_to(root)}"] = digest(path)
    for filename in (
        "coast_refine/capture.py",
        "coast_refine/integration.py",
        "coast_refine/consumer.py",
        "hooks/sinks/gtpc_writer.py",
    ):
        code[filename] = digest(Path(globato.__file__).parent / filename)
    return {
        "readers": code,
        "libraries": {
            name: version(name)
            for name in (
                "numpy",
                "rasterio",
                "pyproj",
                "shapely",
                "h5py",
                "transformez",
            )
        },
        "proj": pyproj.proj_version_str,
        "epsg": pyproj.database.get_database_metadata("EPSG.VERSION"),
    }


def capture_identity(module, entry, request, compression):
    reader = entry.get("stream_reader") or {}
    accepted_readers = {"globato.streams.readers.rio.RasterioReader"}
    if request.get("point_source"):
        accepted_readers.add("globato.streams.readers.lidar.LASReader")
    if reader.get("implementation") not in accepted_readers:
        return None, "no verified native raster-reader identity"
    prefix = []
    entry_inputs = {}
    for item in entry.get("history", []):
        if item.get("stage") != "stream":
            continue
        if (
            item.get("implementation") not in _PREFIX_CLASSES
            or "configuration" not in item
            or not item.get("implementation_sha256")
        ):
            return None, "unrecorded or unsupported upstream hook " + item.get(
                "hook", "?"
            )
        if (
            item["implementation"]
            == "globato.hooks.filters.point_raster_mask.PointRasterMask"
        ):
            if (
                not request.get("point_source")
                or item["configuration"].get("barrier") != "coast_refine"
            ):
                return None, "ordinary point mask is not a deferred authority binding"
        if item["implementation"] == "globato.hooks.filters.claim_grid.ClaimGridFilter":
            config = item["configuration"]
            for argument, default in (
                ("accepted_key", "accepted_geometry"),
                ("excluded_key", "excluded_geometry"),
                ("required_key", "claim_required"),
            ):
                key = config.get(argument, default)
                entry_inputs[key] = entry.get(key)
        prefix.append(
            {
                key: item[key]
                for key in (
                    "hook",
                    "implementation",
                    "implementation_sha256",
                    "configuration",
                )
            }
        )
    if not prefix or not any(
        p["implementation"] == "fetchez.hooks.stream_init.DataStream" for p in prefix
    ):
        return None, "no recorded stream-init configuration"
    if any(
        p["implementation"] == "globato.hooks.transforms.reproject.StreamReproject"
        for p in prefix
    ):
        if not entry.get("stream_transform_identity"):
            return None, "no resolved coordinate-operation identity"
    try:
        record = serializable(
            {
                "schema": 1,
                "source": request["source"],
                "region": list(module.region),
                "region_crs": str(module.region.srs),
                "stream_crs": entry.get("src_srs"),
                "contribution": entry.get("contribution_id"),
                "data_type": entry.get("data_type"),
                "profile": entry.get("profile"),
                "reader": reader,
                "uncertainty": entry.get("uncertainty", 0),
                "module_uncertainty": getattr(module, "uncertainty", 0),
                "prefix": prefix,
                "entry_inputs": entry_inputs,
                "transforms": entry.get("stream_transform_identity"),
                "runtime": runtime_identity(),
                "compression": compression,
            }
        )
        return (identity(record), record), None
    except (TypeError, ValueError):
        return None, "upstream configuration has unsupported runtime values"


def inventory(path):
    """Inspect schema/completion without decoding all saved observations."""
    with h5py.File(path, "r") as handle:
        fields = handle["points"]
        names = sorted(fields)
        if not names:
            count = 0
        else:
            if not {"x", "y", "z"}.issubset(names):
                raise ValueError("Point capture is missing coordinate fields")
            count = len(fields["x"])
            if any(len(fields[name]) != count for name in names):
                raise ValueError("Point capture field lengths differ")
        lengths = np.asarray(handle["chunk_lengths"], dtype="int64")
        if np.any(lengths <= 0) or int(lengths.sum()) != count:
            raise ValueError("Point capture chunk inventory is incomplete")
        return {
            "points": count,
            "chunks": len(lengths),
            "fields": {name: str(fields[name].dtype) for name in names},
        }


def read_capture(root, key, record):
    """A completed receipt is trusted only after checking all identities/bytes."""
    import json

    receipt = root / "receipts" / (key + ".json")
    if receipt.is_symlink():
        raise ValueError("Point capture receipt is a symlink")
    if not receipt.exists():
        return None
    saved = json.loads(receipt.read_text())
    checksum = saved.get("sha256", "")
    if saved.get("inputs") != record or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("Point capture receipt identity mismatch")
    target = root / (checksum + ".gtpc")
    if target.is_symlink() or not target.is_file() or digest(target) != checksum:
        raise ValueError("Point capture checksum mismatch or missing artifact")
    if inventory(target) != saved.get("inventory"):
        raise ValueError("Point capture schema/completion mismatch")
    return target, saved
