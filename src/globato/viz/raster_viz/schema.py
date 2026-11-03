"""Shared, stdlib-only schema-v1 contract for publishers and consumers."""

import math

STYLES = {"elevation", "difference", "categorical", "grayscale", "scalar", "rgb"}
PRODUCT_FIELDS = {
    "key",
    "path",
    "type",
    "label",
    "area",
    "group",
    "stage",
    "band",
    "style",
    "limits",
    "water_level",
    "metadata",
    "method",
    "layer",
    "vector_layer",
    "views",
    "display",
}


def product_group(item):
    """The same defaults as the standalone catalog publisher."""
    return item.get(
        "group",
        "imagery"
        if item.get("style", "elevation") == "rgb"
        else item.get("style", "elevation"),
    )


def validate_document(data):
    """Validate schema v1 without opening any source file or importing renderers."""
    if not isinstance(data, dict) or not isinstance(data.get("products"), list):
        raise ValueError("Catalog requires a products array (which may be empty)")
    if (
        type(data.get("schema_version", 1)) is not int
        or data.get("schema_version", 1) != 1
    ):
        raise ValueError("Unsupported product catalog schema")
    for name in ("title", "status"):
        if name in data and not isinstance(data[name], str):
            raise ValueError(name + " must be a string")
    for name in ("running", "rendering_enabled"):
        if name in data and not isinstance(data[name], bool):
            raise ValueError(name + " must be boolean")
    seen = set()
    raster_keys = set()
    for i, item in enumerate(data["products"]):
        if not isinstance(item, dict) or set(item) - PRODUCT_FIELDS:
            raise ValueError(f"Product {i}: unknown fields or invalid object")
        for name in ("key", "path"):
            if not isinstance(item.get(name), str) or not item[name].strip():
                raise ValueError(f"Product {i} requires a nonempty {name}")
        for name in ("area", "group", "label", "stage", "method", "layer"):
            if name in item and (not isinstance(item[name], str) or not item[name]):
                raise ValueError(f"Product {i}: {name} must be a nonempty string")
        identity = (item.get("area", "default"), item["key"])
        if identity in seen:
            raise ValueError("Duplicate area/product key: " + str(identity))
        seen.add(identity)
        if item.get("type", "raster") not in {"raster", "vector"}:
            raise ValueError("Product type must be raster or vector")
        if (
            item.get("type", "raster") == "raster"
            and item.get("style", "elevation") != "rgb"
        ):
            raster_keys.add(identity)
        if item.get("style", "elevation") not in STYLES:
            raise ValueError("Unknown raster style")
        if "water_level" in item:
            level = item["water_level"]
            if (
                item.get("type", "raster") != "raster"
                or item.get("style", "elevation") != "elevation"
                or type(level) not in (int, float)
                or not math.isfinite(level)
            ):
                raise ValueError(
                    "water_level is allowed only on elevation rasters and must be finite"
                )
        if "band" in item and (type(item["band"]) is not int or item["band"] < 1):
            raise ValueError("band must be a positive integer")
        if "views" in item:
            views = item["views"]
            if (
                not isinstance(views, list)
                or not views
                or len(views) != len(set(map(str, views)))
                or any(v not in ("2d", "3d") for v in views)
                or "2d" not in views
            ):
                raise ValueError("views must be a list containing 2d and optionally 3d")
            if "3d" in views and item.get("style", "elevation") not in (
                "elevation",
                "difference",
            ):
                raise ValueError("3d views require elevation or difference style")
        if "display" in item:
            display = item["display"]
            if item.get("type", "raster") != "vector" or not isinstance(display, dict):
                raise ValueError("display is a vector-only object")
            if set(display) - {"visible", "color", "width", "dash"}:
                raise ValueError("Unknown vector display field")
            if "visible" in display and type(display["visible"]) is not bool:
                raise ValueError("Vector display.visible must be boolean")
            if "color" in display:
                color = display["color"]
                if (
                    not isinstance(color, str)
                    or len(color) != 7
                    or color[0] != "#"
                    or any(c not in "0123456789abcdefABCDEF" for c in color[1:])
                ):
                    raise ValueError(
                        "Vector display.color must be a six-digit hex color"
                    )
            if "width" in display and (
                type(display["width"]) not in (int, float)
                or not math.isfinite(display["width"])
                or not 0.5 <= display["width"] <= 6
            ):
                raise ValueError("Vector display.width must be between 0.5 and 6")
            if "dash" in display and (
                not isinstance(display["dash"], list)
                or len(display["dash"]) > 4
                or any(
                    type(x) not in (int, float)
                    or not math.isfinite(x)
                    or x <= 0
                    or x > 24
                    for x in display["dash"]
                )
            ):
                raise ValueError(
                    "Vector display.dash must be a list of up to four positive lengths"
                )
        if not isinstance(item.get("metadata", {}), dict):
            raise ValueError("Product metadata must be an object")
        if "vector_layer" in item:
            if (
                item.get("type", "raster") != "vector"
                or not isinstance(item["vector_layer"], str)
                or not item["vector_layer"].strip()
            ):
                raise ValueError(
                    "vector_layer must be a nonempty string on vector products"
                )
        if item.get("style") == "rgb" and "limits" in item:
            raise ValueError("RGB products must not have elevation/colorbar limits")
        if "limits" in item:
            limits = item["limits"]
            if (
                not isinstance(limits, list)
                or len(limits) != 2
                or not all(
                    isinstance(x, (float, int))
                    and not isinstance(x, bool)
                    and math.isfinite(x)
                    for x in limits
                )
                or limits[0] >= limits[1]
            ):
                raise ValueError("limits must be two increasing finite numbers")
    if not isinstance(data.get("defaults", {}), dict):
        raise ValueError("defaults must be an area-to-product-pair object")
    for area, pair in data.get("defaults", {}).items():
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(k, str) and (area, k) in raster_keys for k in pair)
        ):
            raise ValueError(
                f"Default comparison for {area} must name two declared raster products"
            )
    for name in ("metrics", "tables"):
        if not isinstance(data.get(name, {}), dict):
            raise ValueError(name + " must be an object")
    for table in data.get("tables", {}).values():
        if (
            not isinstance(table, dict)
            or not isinstance(table.get("columns"), list)
            or not isinstance(table.get("rows"), list)
        ):
            raise ValueError("Tables require columns and rows arrays")

    # Reject non-standard JSON constants (NaN/Infinity) before writing a review.
    # Python's json.loads accepts them by default; browsers and safe_json do not.
    def finite_tree(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Catalog must not contain NaN or Infinity")
        if isinstance(value, dict):
            for child in value.values():
                finite_tree(child)
        elif isinstance(value, list):
            for child in value:
                finite_tree(child)

    finite_tree(data)
    # Limits are shared within a comparison group, including products without
    # an explicit limit. Conflicting declarations are rejected before rendering.
    groups = {}
    for item in data["products"]:
        if item.get("type", "raster") == "vector":
            continue
        identity = (item.get("area", "default"), product_group(item))
        style = item.get("style", "elevation")
        config = groups.setdefault(
            identity, {"style": style, "limits": None, "water_level": None}
        )
        if config["style"] != style:
            raise ValueError("A comparison group must use one rendering style")
        if "water_level" in item:
            if config["water_level"] is not None and config["water_level"] != float(
                item["water_level"]
            ):
                raise ValueError("Conflicting water_level in comparison group")
            config["water_level"] = float(item["water_level"])
        if "limits" in item:
            if config["limits"] is not None and config["limits"] != item["limits"]:
                raise ValueError("Conflicting color limits in comparison group")
            config["limits"] = item["limits"]
    return data, groups
