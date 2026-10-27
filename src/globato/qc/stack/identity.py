"""Deterministic configuration identity for hooks in Stack QC cache keys.

Stack QC keys its filter cache on the hooks that shape each source and on the
final stacking configuration. Side-stacks use fetchez's ``dataset_id`` instead;
Stack QC passes its source-pipeline receipt to them as a ``contribution_id``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from rasterio.crs import CRS

VOLATILE_HOOK_FIELDS = {
    "logger",
    "results",
    "stream",
    "raster_stream",
    "session",
    "dataset",
    "_accumulator",
}


def _stable_value(value: Any) -> Any:
    """Convert configuration values to deterministic JSON-compatible values."""

    if value is None or isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        return value if np.isfinite(value) else repr(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, CRS):
        return value.to_wkt()

    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }

    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]

    if isinstance(value, set):
        return sorted(_stable_value(item) for item in value)

    if hasattr(value, "export_as_list"):
        try:
            return _stable_value(value.export_as_list())
        except Exception:
            pass

    if hasattr(value, "format"):
        try:
            return str(value.format("fn"))
        except Exception:
            pass

    return str(value)


def _hook_name(hook: Any) -> str:
    if isinstance(hook, Mapping):
        return str(hook.get("name", ""))
    return str(getattr(hook, "name", hook.__class__.__name__))




def hook_identity(hook: Any, entry: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return deterministic configuration identity for one upstream hook."""

    name = _hook_name(hook)

    if isinstance(hook, Mapping):
        config = {
            str(key): value
            for key, value in hook.items()
            if key not in {"name", "affects_fusion_state"}
        }
        return {"name": name, "config": _stable_value(config)}

    explicit = getattr(hook, "sidestack_identity", None)
    if callable(explicit):
        explicit = explicit()
    if explicit is not None:
        return {"name": name, "config": _stable_value(explicit)}

    config = {}
    for key, value in vars(hook).items():
        if key.startswith("_") or key in VOLATILE_HOOK_FIELDS:
            continue
        config[key] = value

    entry_identity = getattr(hook, "sidestack_entry_identity", None)
    if entry is not None and callable(entry_identity):
        config["entry_context"] = entry_identity(entry)
    return {"name": name, "config": _stable_value(config)}

