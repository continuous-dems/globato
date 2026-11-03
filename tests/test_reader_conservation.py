"""Original raster validity and edge rows survive the native reader."""

import numpy as np
import rasterio
from rasterio.transform import from_origin
from fetchez.spatial import Region
from globato.streams.readers.rio import RasterioReader, _window_index


def test_pixel_boundary_roundoff_does_not_change_real_fractional_crop():
    assert _window_index(695.9999999998) == 696
    assert _window_index(371.9999999998) == 372
    assert _window_index(1.25) == 1
    assert _window_index(695.75) == 695


def test_full_native_extent_reads_every_valid_edge_cell(tmp_path):
    source = tmp_path / "native.tif"
    shape = (696, 372)
    transform = from_origin(
        -124.21074074066667, 41.920740740666666, 1 / 32400, 1 / 32400
    )
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=1,
        dtype="float32",
        crs=4326,
        transform=transform,
        nodata=-9999,
    ) as ds:
        ds.write(np.full(shape, 3.0, dtype="float32"), 1)
        b = ds.bounds
    region = Region(b.left, b.right, b.bottom, b.top)
    region.srs = "EPSG:4326"
    reader = RasterioReader(str(source), region=region)
    chunks = list(reader._yield_raw_chunks())
    assert sum(len(x) for x in chunks) == shape[0] * shape[1]


def test_reader_uses_exact_nodata_and_gdal_mask(tmp_path):
    source = tmp_path / "validity.tif"
    values = np.array(
        [[-9999, -9998.95, -2, 0], [np.nan, np.inf, 3, 4]], dtype="float32"
    )
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        height=2,
        width=4,
        count=1,
        dtype="float32",
        crs=4326,
        transform=from_origin(0, 2, 1, 1),
        nodata=-9999,
    ) as ds:
        ds.write(values, 1)
    chunks = list(RasterioReader(str(source))._yield_raw_chunks())
    z = np.concatenate([a["z"] for a in chunks])
    assert np.array_equal(z, values[np.isfinite(values) & (values != -9999)])
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False):
        with rasterio.open(source, "r+") as ds:
            valid = (np.isfinite(values) & (values != -9999)).astype("uint8") * 255
            valid[1, 3] = 0
            ds.write_mask(valid)
    chunks = list(RasterioReader(str(source))._yield_raw_chunks())
    z = np.concatenate([a["z"] for a in chunks])
    assert np.array_equal(z, values[valid > 0])
