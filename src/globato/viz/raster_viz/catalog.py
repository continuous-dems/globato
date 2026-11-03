"""Producer catalog publisher for any raster-producing workflow.

This module and schema.py use only the Python standard library. The caller
chooses its own keys, labels, stages, styles, and comparison pairs. Completed
paths are validated before publication; JSON is written with an atomic rename.
One catalog writer per file (or supply your own cross-process synchronization).
"""

from __future__ import annotations
import json
import math
import os
import tempfile

from .schema import validate_document
from pathlib import Path

_STYLES = {"elevation", "difference", "scalar", "grayscale", "categorical", "rgb"}


class ProductCatalog:
    """Prepare a complete producer-defined `products.json`.

    Example::
        with ProductCatalog(run / 'viz_products/products.json', title='Survey QA') as c:
            c.add_raster('input', run / 'input.tif', label='Original')
            c.add_raster('output', run / 'output.tif', label='Result')
            c.compare('input', 'output')
    """

    def __init__(self, path, *, title="Raster visualization", area="default"):
        self.path = Path(path).resolve()
        self.title = str(title)
        self.area = str(area)
        self.products = []
        self.defaults = {}
        self.metrics = {}

    def _source(self, source):
        path = Path(source)
        resolved = (path if path.is_absolute() else Path.cwd() / path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError("Publish only completed products: " + str(resolved))
        return os.path.relpath(resolved, self.path.parent)

    def add_raster(
        self,
        key,
        path,
        *,
        label=None,
        stage=None,
        group=None,
        style="elevation",
        band=1,
        limits=None,
        area=None,
        metadata=None,
        views=None,
        water_level=None,
    ):
        if style not in _STYLES:
            raise ValueError("Unknown raster style: " + str(style))
        if not isinstance(band, int) or isinstance(band, bool) or band < 1:
            raise ValueError("band must be positive integer")
        # The default group follows the raster type so a difference layer
        # cannot accidentally share an elevation palette and fail validation.
        if group is None:
            group = "imagery" if style == "rgb" else style
        item = {
            "type": "raster",
            "key": str(key),
            "path": self._source(path),
            "label": str(label or key),
            "area": str(area or self.area),
            "group": str(group),
            "style": style,
            "band": band,
        }
        if stage is not None:
            item["stage"] = str(stage)
        if views is not None:
            if (
                not isinstance(views, (tuple, list))
                or "2d" not in views
                or any(v not in ("2d", "3d") for v in views)
                or len(views) != len(set(views))
            ):
                raise ValueError("views must include 2d and may include 3d")
            if "3d" in views and style not in ("elevation", "difference"):
                raise ValueError("3d views require elevation or difference style")
            item["views"] = list(views)
        if limits is not None:
            if style == "rgb":
                raise ValueError("RGB is a display image, not a scaled scalar raster")
            if (
                len(limits) != 2
                or not all(
                    isinstance(x, (int, float))
                    and not isinstance(x, bool)
                    and math.isfinite(x)
                    for x in limits
                )
                or limits[0] >= limits[1]
            ):
                raise ValueError("limits must be two increasing finite numbers")
            item["limits"] = [float(x) for x in limits]
        if water_level is not None:
            if (
                style != "elevation"
                or type(water_level) not in (int, float)
                or not math.isfinite(water_level)
            ):
                raise ValueError(
                    "water_level is a finite numeric elevation-style color threshold"
                )
            item["water_level"] = float(water_level)
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise TypeError("metadata must be a dictionary")
            item["metadata"] = metadata
        self._add(item)
        return self

    def add_vector(
        self,
        key,
        path,
        *,
        label=None,
        stage=None,
        area=None,
        metadata=None,
        display=None,
        layer=None,
    ):
        item = {
            "key": str(key),
            "path": self._source(path),
            "type": "vector",
            "label": str(label or key),
            "area": str(area or self.area),
        }
        if stage is not None:
            item["stage"] = str(stage)
        if layer is not None:
            if not isinstance(layer, str) or not layer.strip():
                raise ValueError("vector layer must be a nonempty string")
            item["vector_layer"] = layer
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise TypeError("metadata must be a dictionary")
            item["metadata"] = metadata
        if display is not None:
            if not isinstance(display, dict):
                raise TypeError("vector display must be a dictionary")
            item["display"] = display
        self._add(item)
        return self

    def _add(self, item):
        for field in ("key", "path", "area", "label"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"{field} must be a nonempty string")
        identity = (item["area"], item["key"])
        if any((p["area"], p["key"]) == identity for p in self.products):
            raise ValueError("Duplicate product: " + str(identity))
        self.products.append(item)

    def compare(self, left, right, *, area=None):
        target = str(area or self.area)
        declared = {
            (p["area"], p["key"])
            for p in self.products
            if p.get("type", "raster") == "raster"
            and p.get("style", "elevation") != "rgb"
        }
        if (
            not isinstance(left, str)
            or not isinstance(right, str)
            or (target, left) not in declared
            or (target, right) not in declared
        ):
            raise ValueError(
                "Comparison requires two declared raster keys in the same area"
            )
        self.defaults[target] = [left, right]
        return self

    def publish(self, *, running=False, status=None, rendering_enabled=True):
        data = {
            "schema_version": 1,
            "title": self.title,
            "running": running,
            "products": self.products,
            "defaults": self.defaults,
        }
        if rendering_enabled is not True:
            data["rendering_enabled"] = rendering_enabled
        if status is not None:
            data["status"] = str(status)
        if self.metrics:
            data["metrics"] = self.metrics
        # The consumer uses the same contract; invalid updates preserve the prior file.
        validate_document(data)
        raw = json.dumps(data, allow_nan=False, indent=2) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=".products-", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf8") as fh:
                fh.write(raw)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        finally:
            Path(tmp).unlink(missing_ok=True)
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.publish()
