#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""globato.source_mask_grouping
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Group source masks by shared dataset metadata without loading full rasters.

:license: MIT, see LICENSE for more details.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio


def parse_group_fields(group_by) -> tuple[str, ...]:
    if not group_by:
        return ()
    if isinstance(group_by, str):
        values = re.split(r"[/,]", group_by)
    else:
        values = list(group_by)
    fields = tuple(str(v).strip().upper() for v in values if str(v).strip())
    return tuple(dict.fromkeys(fields))


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_.-")
    return value[:180] or "group"


def _has_data(src) -> bool:
    return any(
        np.any(src.read(1, window=window) > 0) for _, window in src.block_windows(1)
    )


def group_source_mask_files(
    mask_files: Iterable[str | os.PathLike],
    group_by="MODULE/DATASET/WEIGHT",
    output_dir: str | os.PathLike | None = None,
) -> list[str]:
    """Union retained masks on their common grid, one raster block at a time."""
    fields = parse_group_fields(group_by)
    if not fields:
        return []

    groups = defaultdict(list)
    for path in mask_files:
        path = str(path)
        if not os.path.isfile(path):
            raise RuntimeError(f"Missing requested source mask: {path}")
        with rasterio.open(path) as src:
            if not _has_data(src):
                continue
            tags = {str(k).upper(): str(v) for k, v in src.tags().items()}
            if any(not tags.get(field) for field in fields):
                raise RuntimeError(f"Source mask missing grouping metadata: {path}")
            key = tuple(tags[field] for field in fields)
            groups[key].append((path, tags, src.profile.copy()))

    if not groups:
        return []

    if output_dir is None:
        first = Path(next(iter(groups.values()))[0][0])
        output_dir = first.parent / "grouped"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for key, members in sorted(groups.items()):
        first_profile = members[0][2]
        for _, _, profile in members[1:]:
            if any(
                profile.get(field) != first_profile.get(field)
                for field in ("width", "height", "transform", "crs")
            ):
                raise RuntimeError(
                    "Cannot group source masks on different raster grids"
                )

        # Only metadata shared by every member describes the union. Keep source
        # URLs, acquisition dates and per-file statistics on individual masks.
        tags = {
            name: value
            for name, value in members[0][1].items()
            if name
            not in {
                "URL",
                "TITLE",
                "DATE",
                "GROUPED_BY",
                "GROUPED_SOURCE_COUNT",
                "GROUPED_MEMBERS",
            }
            and not name.startswith("STATISTICS_")
            and all(member_tags.get(name) == value for _, member_tags, _ in members[1:])
        }
        tags["GROUPED_BY"] = "/".join(fields)
        tags["GROUPED_SOURCE_COUNT"] = str(len(members))
        tags["GROUPED_MEMBERS"] = json.dumps(
            [os.path.relpath(path, output_dir) for path, _, _ in members]
        )
        # Keep the filename readable and bounded; the digest preserves the
        # complete unsanitized grouping identity, including field names.
        prefix = (
            "__".join(_safe_name(value) for value in key)[:160].rstrip("_.-") or "group"
        )
        identity = json.dumps([fields, key], ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        label = f"{prefix}__{digest}"
        out = output_dir / f"{label}_mask.tif"
        profile = first_profile.copy()
        profile.update(count=1, dtype="uint8", nodata=0)
        if not profile.get("tiled"):
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
        with rasterio.open(out, "w+", **profile) as dst:
            for _, window in dst.block_windows(1):
                dst.write(
                    np.zeros((int(window.height), int(window.width)), dtype="uint8"),
                    1,
                    window=window,
                )
            for path, _, _ in members:
                with rasterio.open(path) as src:
                    for _, window in src.block_windows(1):
                        union = dst.read(1, window=window)
                        union |= (src.read(1, window=window) > 0).astype("uint8")
                        dst.write(union, 1, window=window)
            dst.set_band_description(1, label)
            dst.update_tags(**tags)
        outputs.append(str(out))

    return outputs
