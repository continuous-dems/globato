"""2D geometry uses horizontal CRS components; elevations remain source-owned."""

from pathlib import Path
import numpy as np
import pyproj
import rasterio
from .selection import horizontal_crs


def horizontal_frame(frame):
    if frame.crs is None:
        raise ValueError("Geometry has no CRS")
    horizontal = horizontal_crs(frame.crs)
    return (
        frame.set_crs(horizontal, allow_override=True)
        if pyproj.CRS(frame.crs).is_compound
        else frame
    )


def operation_record(source, target, point):
    src = horizontal_crs(source)
    dst = horizontal_crs(target)
    transform = pyproj.Transformer.from_crs(src, dst, always_xy=True)
    xy = transform.transform(*point, errcheck=True)
    try:
        used = transform.get_last_used_operation()
    except RuntimeError:
        used = transform
    return dict(
        source_horizontal_crs=src.to_string(),
        target_horizontal_crs=dst.to_string(),
        pipeline=used.definition,
        description=used.description,
        accuracy_m=used.accuracy,
        sample_input=list(point),
        sample_output=list(xy),
        proj_version=pyproj.proj_version_str,
        pyproj_version=pyproj.__version__,
        vertical_operation=False,
    )


def science_horizontal_view(source, target):
    """An explicit 2D metadata view, with identical pixels and validity mask.

    The immutable science receives metre elevations already decoded upstream.
    Removing the vertical CRS component prevents a 2D footprint operation from
    selecting a vertical/compound datum route. No value or datum conversion runs.
    """
    source = Path(source)
    target = Path(target)
    with rasterio.open(source) as src:
        profile = src.profile.copy()
        profile.update(crs=horizontal_crs(src.crs).to_wkt())
        with rasterio.open(target, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                dst.write(src.read(window=window), window=window)
                dst.write_mask(src.dataset_mask(window=window), window=window)
            dst.descriptions = src.descriptions
            dst.scales = src.scales
            dst.offsets = src.offsets
            dst.update_tags(
                **src.tags(),
                COAST_MASK_GEOMETRY_VIEW="horizontal_only_no_value_transform",
            )
        with rasterio.open(target) as view:
            for _, window in src.block_windows(1):
                if src.read(window=window).tobytes() != view.read(
                    window=window
                ).tobytes() or not np.array_equal(
                    src.dataset_mask(window=window), view.dataset_mask(window=window)
                ):
                    raise ValueError(
                        "2D science view changed evidence values or validity"
                    )
    return target


def geometry_environment():
    import shapely
    import geopandas

    return dict(
        pyproj=pyproj.__version__,
        proj=pyproj.proj_version_str,
        shapely=shapely.__version__,
        geos=shapely.geos_version_string,
        geopandas=geopandas.__version__,
        gdal=rasterio.__gdal_version__,
    )
