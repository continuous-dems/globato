# tests/test_format_cog.py

import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from globato.hooks.rasters.base import RasterCOG

# Bigger than one 256 px block: a single-block file with no overviews is
# trivially in COG order, which would hide the bug these tests are for.
SIZE = 700


def _write_raster(path, dtype="float32", count=1):
    rng = np.random.default_rng(0)
    if dtype == "uint8":
        data = rng.integers(0, 255, (count, SIZE, SIZE), dtype="uint8")
    else:
        data = rng.normal(0, 50, (count, SIZE, SIZE)).astype(dtype)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=SIZE,
        width=SIZE,
        count=count,
        dtype=dtype,
        crs="EPSG:4326",
        transform=from_origin(-120.0, 36.0, 0.001, 0.001),
        nodata=-9999.0 if dtype != "uint8" else None,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(data)
    return data


def _first_block(src, ovr=None):
    kwargs = {"bidx": 1} if ovr is None else {"bidx": 1, "ovr": ovr}
    return int(src.get_tag_item("BLOCK_OFFSET_0_0", "TIFF", **kwargs))


def _assert_is_cog(path):
    with rasterio.open(path) as src:
        overviews = src.overviews(1)
        assert overviews, "no overviews"
        assert src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
        # The defining layout rule: overview data sits before the full-res data.
        assert _first_block(src, ovr=len(overviews) - 1) < _first_block(src)
        assert src.block_shapes[0] == (256, 256)


@pytest.fixture
def tile_dir(tmp_path, monkeypatch):
    # Raster hooks keep their scratch files in ./tmp, like a recipe's tile directory.
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _entry(path):
    return {"dst_fn": str(path), "artifacts": {"ms_binary_cudem": str(path)}}


def test_format_cog_converts_the_input_in_place(tile_dir):
    dem = tile_dir / "name_tile.tif"
    data = _write_raster(dem)

    entries = RasterCOG().run([(None, _entry(dem))])

    entry = entries[0][1]
    # Later hooks and copy_artifact must find the COG under the DEM's own name,
    # not in tmp/, which cleanup_tmp removes before anything is delivered.
    assert entry["dst_fn"] == str(dem)
    assert entry["artifacts"]["format-cog"] == str(dem)
    _assert_is_cog(dem)

    with rasterio.open(dem) as src:
        assert np.array_equal(src.read(), data)

    leftovers = [f for f in os.listdir(tile_dir) if f != dem.name and f != "tmp"]
    assert leftovers == []
    assert os.listdir(tile_dir / "tmp") == []


def test_format_cog_with_an_output_writes_there_and_keeps_the_input(tile_dir):
    dem = tile_dir / "name_tile.tif"
    out = tile_dir / "name_tile_cog.tif"
    _write_raster(dem)

    entries = RasterCOG(output=str(out)).run([(None, _entry(dem))])

    assert entries[0][1]["dst_fn"] == str(out)
    assert dem.exists()
    _assert_is_cog(out)


def test_format_cog_output_may_be_the_input_itself(tile_dir):
    dem = tile_dir / "name_tile.tif"
    _write_raster(dem)

    RasterCOG(output=str(dem)).run([(None, _entry(dem))])

    _assert_is_cog(dem)


def test_format_cog_handles_a_uint8_hillshade(tile_dir):
    """PREDICTOR=3 is floating point only, so it cannot be hard-coded."""
    hillshade = tile_dir / "name_tile_hs.tif"
    data = _write_raster(hillshade, dtype="uint8", count=3)

    RasterCOG().run([(None, _entry(hillshade))])

    _assert_is_cog(hillshade)
    with rasterio.open(hillshade) as src:
        assert src.count == 3
        assert np.array_equal(src.read(), data)
