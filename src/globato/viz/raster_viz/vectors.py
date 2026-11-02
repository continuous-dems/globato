"""Dependency-lazy, bounded vector reading: Pyogrio first, Fiona fallback."""


def vector_layer(item):
    """Select a real layer without repurposing legacy opaque ``layer`` labels."""
    if "vector_layer" in item:
        return item["vector_layer"]
    from pathlib import Path

    hint = item.get("metadata", {}).get("layer")
    if (
        Path(item["path"]).suffix.lower() == ".gpkg"
        and isinstance(hint, str)
        and hint.strip()
    ):
        return hint
    return None


def inspect_vector(path, layer=None):
    """Read layer metadata without loading its geometry into memory."""
    try:
        import pyogrio
    except (ImportError, OSError):
        pyogrio = None
    if pyogrio is not None:
        info = pyogrio.read_info(path, layer=layer)
        return info.get("crs"), info.get("features", -1)
    import fiona

    with fiona.open(path, layer=layer) as src:
        return src.crs, len(src)


def read_features(path, max_features, layer=None):
    from shapely.geometry import shape
    from shapely.wkb import loads

    try:
        import pyogrio
    except (ImportError, OSError):
        pyogrio = None
    if pyogrio is not None:
        info = pyogrio.read_info(path, layer=layer)
        count = info.get("features", -1)
        if count > max_features:
            raise ValueError(
                f"Vector has {count} features (limit {max_features}); select/simplify first"
            )
        meta, _, geometries, _ = pyogrio.raw.read(
            path, layer=layer, columns=[], max_features=max_features + 1
        )
        if len(geometries) > max_features:
            raise ValueError(
                f"Vector exceeds {max_features} features; select/simplify first"
            )
        return (
            meta.get("crs"),
            [loads(g) for g in geometries if g is not None],
            "pyogrio",
        )
    try:
        import fiona
    except (ImportError, OSError) as exc:
        raise ImportError(
            "Vector overlays require the optional vectors extra (Pyogrio + Shapely), or Fiona + Shapely"
        ) from exc
    with fiona.open(path, layer=layer) as src:
        if len(src) > max_features:
            raise ValueError(
                f"Vector has {len(src)} features (limit {max_features}); select/simplify first"
            )
        return (
            src.crs,
            [shape(f["geometry"]) for f in src if f.get("geometry")],
            "fiona",
        )
