import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from fetchez.spatial import Region
from globato.streams.readers.rio import RasterioReader

# A 2 km UTM 11N grid off Santa Barbara, and a lon/lat region around it.
UTM_WEST, UTM_NORTH, CELL, SIZE = 250000.0, 3802000.0, 20.0, 100
REGION = [-119.73, -119.68, 34.30, 34.34]


def _write_grid(path, crs=None):
    z = np.arange(SIZE * SIZE, dtype="float32").reshape(SIZE, SIZE) * -0.01
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=SIZE,
        height=SIZE,
        count=1,
        dtype="float32",
        crs=crs,
        nodata=-9999,
        transform=from_origin(UTM_WEST, UTM_NORTH, CELL, CELL),
    ) as dst:
        dst.write(z, 1)
    return str(path)


def _count(path, **kwargs):
    region = Region.from_list(REGION)
    reader = RasterioReader(path, region=region, **kwargs)
    return sum(len(c) for c in reader.yield_chunks() if c is not None)


@pytest.mark.parametrize(
    "src_srs", ["EPSG:26911+5703", "EPSG:32611+vdatum:mllw", "EPSG:26911"]
)
def test_src_srs_crops_grid_without_crs(tmp_path, src_srs):
    """A grid with no CRS of its own is cropped using the horizontal part of src_srs."""

    path = _write_grid(tmp_path / "no_crs.tif")
    assert _count(path, src_srs=src_srs) == SIZE * SIZE


def test_unparseable_src_srs_falls_back_to_file_crs(tmp_path):
    path = _write_grid(tmp_path / "utm.tif", crs="EPSG:26911")
    assert _count(path, src_srs="auto-utm+vdatum:mllw") == SIZE * SIZE


def test_crop_crs_keeps_proj_and_wkt_strings(tmp_path):
    path = _write_grid(tmp_path / "utm.tif", crs="EPSG:26911")
    with rasterio.open(path) as src:
        for srs in ("+proj=utm +zone=11 +datum=NAD83", src.crs.to_wkt()):
            reader = RasterioReader(path, src_srs=srs)
            assert reader._crop_crs(src).to_epsg() == 26911
