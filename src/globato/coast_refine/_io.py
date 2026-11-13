from pathlib import Path
import hashlib
import json
import os
import tempfile
import numpy as np
import pyogrio.raw
import geopandas as gpd
import shapely


def digest(path, *, progress=None):
    h = hashlib.sha256()
    total = Path(path).stat().st_size if progress else 0
    done = 0
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(chunk)
            if progress:
                done += len(chunk)
                progress(done, total)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def raster_files(path, *, progress=None):
    """Include external validity and raster-data sidecars in processing identity."""
    import rasterio

    with rasterio.open(path) as src:
        files = sorted(Path(name).resolve(strict=True) for name in src.files)
    return [{"name": p.name, "sha256": digest(p, progress=progress)} for p in files]


def vector_files(path):
    """Fingerprint a vector input and all required shapefile components.

    A bare .shp digest does not represent the attributes, CRS or geometry
    indices actually consumed by GDAL. Single-file formats use one hash.
    """
    source = Path(path).expanduser().resolve(strict=True)
    files = [source]
    if source.suffix.lower() == ".shp":
        for suffix in (".shx", ".dbf", ".prj", ".cpg"):
            sidecar = source.with_suffix(suffix)
            if suffix in (".shx", ".dbf") and not sidecar.is_file():
                raise ValueError(f"Missing required shapefile component: {sidecar}")
            if sidecar.is_file():
                files.append(sidecar)
    return [{"name": p.name, "sha256": digest(p)} for p in files]


def polygonal_geometry(geometry):
    """Return only the area of a valid constructive geometry, without repair.

    Polygon clipping can also return boundary-touch lines/points. Retention
    uses strict polygon interiors, so those zero-area pieces cannot select
    observations. Preserve ordinary polygons unchanged and keep every polygon
    component, including arbitrarily small ones; never buffer or simplify.
    """
    import shapely

    if geometry is None or not geometry.is_valid:
        raise ValueError("Missing or invalid polygonal geometry")
    if geometry.geom_type in ("Polygon", "MultiPolygon"):
        return geometry
    parts = []

    def collect(value):
        if value.geom_type in ("Polygon", "MultiPolygon"):
            if not value.is_empty:
                parts.append(value)
        elif hasattr(value, "geoms"):
            for child in value.geoms:
                collect(child)

    collect(geometry)
    return shapely.union_all(parts) if parts else shapely.Polygon()


def read_polygon(path, layer="mask", *, crs="EPSG:4326", require_native=False):
    data = gpd.read_file(path, layer=layer)
    if data.crs is None:
        raise ValueError(f"Polygon product has no CRS: {path}")
    if require_native and data.crs != gpd.GeoSeries([], crs=crs).crs:
        raise ValueError(
            f"{Path(path).stem} CRS disagrees with native application grid {crs}"
        )
    from .geometry_crs import horizontal_frame
    from .selection import horizontal_crs

    geoms = list(
        (
            data
            if require_native
            else horizontal_frame(data).to_crs(horizontal_crs(crs))
        ).geometry
    )
    if any(
        g is None or not g.is_valid or g.geom_type not in ("Polygon", "MultiPolygon")
        for g in geoms
    ):
        raise ValueError(f"Invalid polygon product: {path}")
    result = shapely.union_all(geoms)
    if not result.is_valid:
        raise ValueError(f"Invalid polygon union: {path}")
    return result


def write_polygon(path, geom, crs="EPSG:4326"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if geom.is_empty:
        geometry = np.array([], dtype=object)
        field_data = [np.array([], dtype=object)]
    else:
        if geom.geom_type == "Polygon":
            geom = shapely.MultiPolygon([geom])
        if geom.geom_type != "MultiPolygon" or not geom.is_valid:
            raise ValueError("Expected valid polygon geometry")

        geometry = np.array([shapely.to_wkb(geom)], dtype=object)
        field_data = [np.array([path.stem], dtype=object)]

    pyogrio.raw.write(
        path,
        geometry=geometry,
        field_data=field_data,
        fields=["role"],
        layer="mask",
        driver="GPKG",
        geometry_type="MultiPolygon",
        crs=crs,
    )
    return path


def json_write(path, value):
    """Commit a complete receipt atomically; never leave a truncated JSON file."""
    path = Path(path)
    body = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix="." + path.name + "_", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def serializable(value):
    """Stable receipt values for native hook settings and spatial claims."""
    import numpy as np
    import shapely

    if isinstance(value, shapely.geometry.base.BaseGeometry):
        return {"wkb": value.normalize().wkb_hex}
    if isinstance(value, bytes):
        return {"hex": value.hex()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    return value
