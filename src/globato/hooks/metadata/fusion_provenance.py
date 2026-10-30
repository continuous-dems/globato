#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
globato.hooks.metadata.fusion_provenance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Collection-stage provenance for side-stack + stack_combine workflows.

Produces persistent per-contribution coverage masks, then renders VRT/GPKG
independently. No point-stream interception. Only MultiStack lineage is supported;
interpolation (binary_cudem) lineage is not inferred.

Initial implementation: ``mode=build`` recomputes mask state from selected
side-stacks; ``mode=render`` reuses stored masks and metadata to change presentation.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import rasterio

from fetchez.hooks import FetchHook
from fetchez.spatial import Region
from fetchez.utils import str2bool
from fetchez.entry import dataset_id
from ..transforms.point_pixels import (
    FUSION_BAND_MAP,
    FUSION_BANDS,
    FUSION_STATE_VERSION,
)
from .provenance import MaskSet

logger = logging.getLogger(__name__)
MANIFEST_VERSION = 1


def _fingerprint(path):
    """A cheap change detector, NOT a content hash; see production caveats."""
    stat = os.stat(path)
    return {
        "path": str(Path(path).resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _tags(mod, entry, src):
    tags = {
        "MODULE": getattr(mod, "name", None),
        "DATASET": getattr(mod, "title", getattr(mod, "name", None)),
        "CATEGORY": getattr(mod, "meta_category", None),
        "AGENCY": getattr(mod, "meta_agency", None),
        "DATATYPE": entry.get("data_type"),
        "RESOLUTION": getattr(mod, "meta_resolution", None),
        "URL": entry.get("url"),
        "WEIGHT": getattr(mod, "weight", None),
    }
    if isinstance(entry.get("metadata"), dict):
        tags.update({str(k).upper(): v for k, v in entry["metadata"].items()})

    # Explicit source metadata in the side-stack takes precedence if present.
    for key in ("GLOBATO_SOURCE_ID", "GLOBATO_SURVEY_ID"):
        if key in src.tags():
            tags[key] = src.tags()[key]

    return {
        str(k): str(v)
        for k, v in tags.items()
        if v is not None and str(v) not in ("", "None", "Unknown")
    }


def _validate(src, expected):
    if src.tags().get("GLOBATO_DATATYPE") != "FUSION_STATE":
        raise ValueError(f"Not a FusionState: {src.name}")
    if src.tags().get("GLOBATO_FUSION_VERSION") != str(FUSION_STATE_VERSION):
        raise ValueError(f"FusionState version mismatch: {src.name}")
    if src.count != len(FUSION_BANDS) or tuple(src.descriptions) != tuple(FUSION_BANDS):
        raise ValueError(f"FusionState band schema mismatch: {src.name}")
    if any(dtype != "float64" for dtype in src.dtypes):
        raise ValueError(f"FusionState dtype mismatch: {src.name}")
    if (src.width, src.height) != (expected.width, expected.height):
        raise ValueError(f"FusionState dimensions mismatch: {src.name}")
    if src.transform != expected.transform or src.crs != expected.crs:
        raise ValueError(f"FusionState CRS/geotransform mismatch: {src.name}")


def _strategy(fusion):
    strategy = fusion.tags().get("GLOBATO_STACK_STRATEGY", "").lower()
    if strategy not in ("mixed", "mean", "weighted_mean"):
        raise ValueError(
            f"Unsupported provenance reducer {strategy!r}; "
            "supercede requires explicit winner decisions"
        )

    try:
        tiers = np.sort(
            np.asarray(
                json.loads(fusion.tags().get("GLOBATO_WEIGHT_TIERS", "[]")),
                dtype=np.float64,
            )
        )
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid GLOBATO_WEIGHT_TIERS") from exc

    if strategy == "mixed" and (tiers.size >= 255 or not np.all(np.isfinite(tiers))):
        raise ValueError("Invalid mixed weight tiers (maximum 254 thresholds)")

    return strategy, tiers


def _codes(count, weight_sum, strategy, tiers):
    out = np.zeros(count.shape, dtype=np.uint8)
    valid = count > 0
    if strategy != "mixed":
        out[valid] = 1
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_weight = np.divide(
                weight_sum,
                count,
                out=np.zeros_like(weight_sum, dtype="float64"),
                where=valid,
            )

        out[valid] = np.digitize(mean_weight[valid], tiers).astype(np.uint8) + 1

    return out


class FusionProvenance(FetchHook):
    """Generate source or reduced-stack lineage from selected FusionStates.

    ``mode=build`` rebuilds mask state (no unsafe resume), then renders requested
    outputs. ``mode=render`` uses saved state and never reads side-stack inputs.

    ``stage=source``: selected side-stack count > 0.
    ``stage=stack``: sources surviving stack_combine's mean/weighted_mean/mixed
    strategy, using the same tier reconstruction as stack_provenance.
    """

    name = "fusion_provenance"
    meta_stage = "collection"
    meta_category = "metadata"
    meta_desc = "Build and render collection-stage FusionState provenance."

    def __init__(
        self,
        stage="stack",
        mode="build",
        source_artifact="side-stack",
        state_dir="fusion_provenance_state",
        fusion_state=None,
        output="fusion_sources.vrt",
        vector_output=None,
        missing="error",
        group_by=None,
        vector_max_size=2048,
        vector_simplify=0.0,
        qgis_style=True,
        qgis_style_field="SOURCE_ID",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.provenance_stage = str(stage).lower()
        self.mode = str(mode).lower()
        self.source_artifact = source_artifact
        self.state_dir = Path(state_dir).resolve()
        self.fusion_state = fusion_state
        self.output = output
        self.vector_output = vector_output
        self.missing = str(missing).lower()
        self.group_by = group_by
        self.vector_max_size = int(vector_max_size)
        self.vector_simplify = float(vector_simplify)
        self.qgis_style = str2bool(qgis_style)
        self.qgis_style_field = qgis_style_field
        if self.provenance_stage not in ("source", "stack"):
            raise ValueError("stage must be 'source' or 'stack' (DEM lineage pending)")
        if self.mode not in ("build", "render"):
            raise ValueError("mode must be 'build' or 'render'")
        if self.missing not in ("error", "skip"):
            raise ValueError("missing must be 'error' or 'skip'")

    @property
    def _manifest_path(self):
        return self.state_dir / "manifest.json"

    def _selected(self, entries):
        """Match stack_combine's path deduplication and preserve source metadata."""
        selected = []
        seen_paths = set()
        seen_ids = set()
        for mod, entry in entries:
            path = (entry.get("artifacts") or {}).get(self.source_artifact)
            if path is None:
                if self.missing == "error":
                    raise ValueError(
                        f"Missing {self.source_artifact!r} for {entry.get('dst_fn')!r}"
                    )
                continue

            path = str(Path(path).resolve())
            if path in seen_paths:
                continue

            seen_paths.add(path)
            # sid = str(source_id(entry))
            sid = str(dataset_id(entry))
            if sid in seen_ids:
                raise ValueError(
                    f"Two distinct selected artifacts share source ID {sid!r}; "
                    "needs an explicit contribution identity"
                )

            seen_ids.add(sid)
            selected.append((mod, entry, path, sid))
        return selected

    def _fusion_path(self, entries):
        paths = {
            str(Path(e.get("artifacts", {})["fusion-state"]).resolve())
            for _, e in entries
            if e.get("artifacts", {}).get("fusion-state")
        }
        if self.fusion_state:
            paths.add(str(Path(self.fusion_state).resolve()))
        if len(paths) != 1:
            raise ValueError(
                "Exactly one finalized fusion-state is required after stack_combine"
            )
        return paths.pop()

    def _maskset(self, fusion, *, output, resume):
        bounds = fusion.bounds
        region = Region(bounds.left, bounds.right, bounds.bottom, bounds.top)
        if fusion.crs:
            region.srs = fusion.crs.to_string()
        return MaskSet(
            region,
            abs(fusion.transform.a),
            output,
            output_dir=str(self.state_dir / "masks"),
            vector_output=self.vector_output,
            mask_type="FUSION_PROVENANCE_MASK",
            resume=resume,
            vector_max_size=self.vector_max_size,
            vector_simplify=self.vector_simplify,
            qgis_style=self.qgis_style,
            qgis_style_field=self.qgis_style_field,
            group_by=self.group_by,
            dst_gt=fusion.transform.to_gdal(),
            xcount=fusion.width,
            ycount=fusion.height,
        )

    def _build(self, entries):
        selected = self._selected(entries)
        if not selected:
            raise ValueError("No selected side-stack contributions")

        fusion_path = self._fusion_path(entries)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # Build into fresh masks. State is not incrementally OR-ed across reruns.
        mask_dir = self.state_dir / "masks"
        mask_dir.mkdir(exist_ok=True)
        records = []
        with rasterio.open(fusion_path) as fusion:
            _validate(fusion, fusion)  # includes tagged FusionState schema
            strategy, tiers = (
                _strategy(fusion) if self.provenance_stage == "stack" else (None, None)
            )
            masks = self._maskset(fusion, output=self.output, resume=False)
            used_paths = set()
            for mod, entry, path, sid in selected:
                with rasterio.open(path) as src:
                    _validate(src, fusion)
                    tags = _tags(mod, entry, src)
                    description = Path(entry.get("dst_fn") or path).stem
                    mask_path = masks.register(sid, description=description, tags=tags)
                    used_paths.add(str(Path(mask_path).resolve()))
                    with rasterio.open(mask_path, "r+") as dst:
                        for _, window in src.block_windows(1):
                            count = src.read(FUSION_BAND_MAP["count"], window=window)
                            if not np.any(count > 0):
                                continue

                            if self.provenance_stage == "source":
                                keep = count > 0
                            else:
                                weights = src.read(
                                    FUSION_BAND_MAP["weight_sum"], window=window
                                )
                                source_codes = _codes(count, weights, strategy, tiers)
                                final_count = fusion.read(
                                    FUSION_BAND_MAP["count"], window=window
                                )
                                final_weight = fusion.read(
                                    FUSION_BAND_MAP["weight_sum"], window=window
                                )
                                final_codes = _codes(
                                    final_count, final_weight, strategy, tiers
                                )
                                keep = (
                                    source_codes > 0
                                    if strategy != "mixed"
                                    else (
                                        (source_codes > 0)
                                        & (source_codes == final_codes)
                                    )
                                )

                            if np.any(keep):
                                dst.write(keep.astype("uint8"), 1, window=window)

                    records.append(
                        {
                            "source_id": sid,
                            "path": path,
                            "fingerprint": _fingerprint(path),
                            "description": description,
                            "tags": tags,
                        }
                    )
            # Remove orphan masks from a preceding build: only this selection is valid.
            for old in mask_dir.glob("*_mask.tif"):
                if str(old.resolve()) not in used_paths:
                    old.unlink()

            manifest = {
                "version": MANIFEST_VERSION,
                "stage": self.provenance_stage,
                "fusion_state": _fingerprint(fusion_path),
                "strategy": strategy,
                "tiers": tiers.tolist() if tiers is not None else None,
                "sources": records,
            }
            tmp = self._manifest_path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf8") as f:
                json.dump(manifest, f, indent=2, sort_keys=True)

            os.replace(tmp, self._manifest_path)
        return manifest

    def _render(self):
        with open(self._manifest_path, encoding="utf8") as f:
            manifest = json.load(f)

        if manifest.get("version") != MANIFEST_VERSION:
            raise ValueError("Incompatible fusion provenance manifest version")

        if manifest.get("stage") != self.provenance_stage:
            raise ValueError("Requested stage differs from persisted provenance stage")

        fusion_path = manifest["fusion_state"]["path"]
        # Render does not trust changed/removed original fusion grids.
        if _fingerprint(fusion_path) != manifest["fusion_state"]:
            raise ValueError("Final FusionState changed since provenance was built")

        with rasterio.open(fusion_path) as fusion:
            masks = self._maskset(fusion, output=self.output, resume=True)
            for record in manifest["sources"]:
                path = masks._path(record["source_id"])
                # Never let MaskSet.register create an empty replacement on render.
                if not os.path.isfile(path) or not masks._validate(
                    path, record["source_id"]
                ):
                    raise ValueError(f"Missing or invalid persisted mask: {path}")
                masks.register(
                    record["source_id"],
                    description=record["description"],
                    tags=record["tags"],
                )
            masks.finalize()
        return manifest

    def run(self, entries):
        if self.mode == "build":
            self._build(entries)
        self._render()
        for _, entry in entries:
            artifacts = entry.setdefault("artifacts", {})
            artifacts[self.name] = str(self._manifest_path)
            if self.output:
                artifacts["fusion-provenance-vrt"] = str(Path(self.output).resolve())
            if self.vector_output:
                artifacts["fusion-provenance-vector"] = str(
                    Path(self.vector_output).resolve()
                )
        return entries
