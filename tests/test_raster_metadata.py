# tests/test_raster_metadata.py

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from globato.hooks.metadata.metadata import RasterMetadataHook
from globato.hooks.rasters.base import RasterCOG, carry_metadata
from globato.hooks.rasters.crop import RasterCrop
from globato.hooks.rasters.cut import RasterCut
from globato.hooks.viz.geohillshade import GeoHillshade

SIZE = 700
TAGS = {"Project": "Coastal Relief Model", "Version": "6", "Author": "NCEI"}
BAND = "Elevation (meters)"


def _write_dem(path, count=1):
    data = np.random.default_rng(0).normal(0, 50, (count, SIZE, SIZE)).astype("float32")
    # A NoData moat, so that raster_crop has something to remove.
    data[:, :40, :] = -9999.0
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=SIZE,
        width=SIZE,
        count=count,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-120.0, 36.0, 0.001, 0.001),
        nodata=-9999.0,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(data)


def _metadata(path):
    with rasterio.open(path) as src:
        tags = {k: v for k, v in src.tags().items() if k != "AREA_OR_POINT"}
        return tags, src.descriptions, src.units


def _tagged_dem(path):
    _write_dem(path)
    RasterMetadataHook(
        tags=",".join(f"{k}={v}" for k, v in TAGS.items()), bands=BAND
    ).run([(None, {"dst_fn": str(path)})])
    with rasterio.open(path, "r+") as dst:
        dst.set_band_unit(1, "metre")
        dst.update_tags(1, STATISTICS_MEAN="1.5", Source="lidar")


@pytest.fixture
def tile_dir(tmp_path, monkeypatch):
    # Raster hooks keep their scratch files in ./tmp, like a recipe's tile directory.
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(hook, path):
    entries = hook.run([(None, {"dst_fn": str(path), "artifacts": {}})])
    return entries[0][1]["dst_fn"]


def test_raster_cut_keeps_tags_band_description_and_units(tile_dir):
    dem = tile_dir / "dem.tif"
    _tagged_dem(dem)

    out = _run(RasterCut(region=[-119.9, -119.4, 35.4, 35.9]), dem)

    assert out != str(dem)
    assert _metadata(out) == (TAGS, (BAND,), ("metre",))


def test_raster_crop_keeps_tags_band_description_and_units(tile_dir):
    dem = tile_dir / "dem.tif"
    _tagged_dem(dem)

    out = _run(RasterCrop(output=str(tile_dir / "cropped.tif")), dem)

    with rasterio.open(out) as src:
        assert src.height < SIZE
    assert _metadata(out) == (TAGS, (BAND,), ("metre",))


def test_metadata_survives_the_whole_delivery_chain(tile_dir):
    """The order buffer-and-cut produces: raster_metadata, then cut, crop and format_cog."""
    dem = tile_dir / "dem.tif"
    _tagged_dem(dem)

    out = str(dem)
    for hook in (
        RasterCut(region=[-119.9, -119.4, 35.4, 35.9]),
        RasterCrop(output=str(dem)),
        RasterCOG(),
    ):
        out = _run(hook, out)

    assert _metadata(out) == (TAGS, (BAND,), ("metre",))


def test_band_tags_are_kept_but_stale_statistics_are_not(tile_dir):
    dem = tile_dir / "dem.tif"
    _tagged_dem(dem)

    out = _run(RasterCut(region=[-119.9, -119.4, 35.4, 35.9]), dem)

    with rasterio.open(out) as src:
        assert src.tags(1).get("Source") == "lidar"
        # The cut changes the pixels, so statistics of the input no longer apply.
        assert "STATISTICS_MEAN" not in src.tags(1)


def test_a_derived_product_with_different_bands_starts_clean(tile_dir):
    """An RGB hillshade is not the DEM, and a DEM is not the 7-band stack it came from.

    Labels such as GLOBATO_DATATYPE=MULTI_STACK must not follow the data into them.
    """
    dem = tile_dir / "dem.tif"
    _tagged_dem(dem)

    hillshade = _run(GeoHillshade(output=str(tile_dir / "hs.tif")), dem)
    assert _metadata(hillshade) == ({}, (None, None, None), (None, None, None))

    stack, product = tile_dir / "stack.tif", tile_dir / "product.tif"
    _write_dem(stack, count=7)
    _write_dem(product)
    with rasterio.open(stack, "r+") as dst:
        dst.update_tags(GLOBATO_DATATYPE="MULTI_STACK")

    carry_metadata(str(stack), str(product))

    assert _metadata(product)[0] == {}


def test_the_dem_stripped_out_of_a_stack_does_not_inherit_the_stacks_labels(tile_dir):
    """ms_binary_cudem and ms_cudem interpolate a 7-band stack, then keep only band 1."""
    stack = tile_dir / "stack.tif"
    _write_dem(stack, count=7)
    with rasterio.open(stack, "r+") as dst:
        dst.update_tags(GLOBATO_DATATYPE="MULTI_STACK")

    out = _run(RasterCut(region=[-119.9, -119.4, 35.4, 35.9], strip_bands=True), stack)

    with rasterio.open(out) as src:
        assert src.count == 1
    assert _metadata(out)[0] == {}


def test_a_hooks_own_metadata_is_not_overwritten(tile_dir):
    src_fn, dst_fn = tile_dir / "src.tif", tile_dir / "dst.tif"
    _tagged_dem(src_fn)
    _write_dem(dst_fn)
    with rasterio.open(dst_fn, "r+") as dst:
        dst.update_tags(Project="Set by the hook")
        dst.set_band_description(1, "Set by the hook")

    carry_metadata(str(src_fn), str(dst_fn))

    tags, descriptions, _ = _metadata(dst_fn)
    assert tags == {**TAGS, "Project": "Set by the hook"}
    assert descriptions == ("Set by the hook",)


def test_an_output_with_nothing_missing_is_left_untouched(tile_dir):
    """Reopening a file in r+ rewrites it, which would break a COG's layout."""
    src_fn, dst_fn = tile_dir / "src.tif", tile_dir / "dst.tif"
    _tagged_dem(src_fn)
    dst_fn.write_bytes(src_fn.read_bytes())
    before = dst_fn.read_bytes()

    carry_metadata(str(src_fn), str(dst_fn))

    assert dst_fn.read_bytes() == before
