"""GDALReader.get_read_window(): the raster window covering the requested region.

GDAL is an optional dependency, and the reader refuses to start without it, so
these tests build the reader without __init__ and give it a stand-in dataset.
"""

from types import SimpleNamespace

import pytest
from fetchez.spatial import Region
from pyproj import Transformer

from globato.streams.readers.gdal_proc import GDALReader


def _reader(region, file_srs):
    reader = GDALReader.__new__(GDALReader)
    reader.req_region_bounds = region
    reader.get_srs = lambda: file_srs
    return reader


def _dataset(geo_transform, nx, ny):
    return SimpleNamespace(
        RasterXSize=nx, RasterYSize=ny, GetGeoTransform=lambda: geo_transform
    )


# 100 x 100 cells of 0.01 degrees, from (0, 0) to (1, 1).
LONLAT = _dataset((0.0, 0.01, 0.0, 1.0, 0.0, -0.01), 100, 100)


@pytest.mark.parametrize(
    "region",
    [[0.2, 0.5, 0.3, 0.6], Region(w=0.2, e=0.5, s=0.3, n=0.6, srs="epsg:4326")],
)
def test_a_lonlat_region_over_a_lonlat_raster(region):
    """A bounds list or a Region gives the cells from x 0.2-0.5 and y 0.3-0.6.

    fetchez's Region.srcwin() includes the cells on both edges, so 0.3 degrees
    of 0.01-degree cells is 31 of them.
    """
    window = _reader(region, "EPSG:4326").get_read_window(LONLAT)

    assert window == (20, 40, 31, 31)


def test_the_region_is_warped_to_a_projected_raster():
    """A raster in Web Mercator that covers exactly the region is read whole."""
    to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xmin, ymin = to_merc.transform(0.2, 0.3)
    xmax, ymax = to_merc.transform(0.5, 0.6)
    nx, ny = 50, 50
    raster = _dataset(
        (xmin, (xmax - xmin) / nx, 0.0, ymax, 0.0, -(ymax - ymin) / ny), nx, ny
    )

    window = _reader([0.2, 0.5, 0.3, 0.6], "EPSG:3857").get_read_window(raster)

    assert window == (0, 0, nx, ny)


def test_a_region_object_is_not_changed():
    """The window is found without warping the caller's Region."""
    region = Region(w=0.2, e=0.5, s=0.3, n=0.6, srs="epsg:4326")

    _reader(region, "EPSG:3857").get_read_window(LONLAT)

    assert (region.xmin, region.xmax, region.srs) == (0.2, 0.5, "epsg:4326")
