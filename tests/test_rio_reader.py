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


# 2100 rows x 500 cols: one-row strips would mean 2100 chunks; bands of
# ceil(500_000 / 500) = 1000 rows make it 3.
STRIP_ROWS, STRIP_COLS = 2100, 500


def _grid():
    """Depths of -1 to -100.9 m; well clear of the -9999 nodata value."""

    z = np.arange(STRIP_ROWS * STRIP_COLS) % 1000 * -0.1 - 1
    return z.astype("float32").reshape(STRIP_ROWS, STRIP_COLS)


def _write_asc(path):
    with open(path, "w") as f:
        f.write(
            f"ncols {STRIP_COLS}\nnrows {STRIP_ROWS}\n"
            f"xllcorner {UTM_WEST}\nyllcorner {UTM_NORTH - STRIP_ROWS * CELL}\n"
            f"cellsize {CELL}\nNODATA_value -9999\n"
        )
        np.savetxt(f, _grid(), fmt="%.2f")
    return str(path)


def _write_tif(path, **layout):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=STRIP_COLS,
        height=STRIP_ROWS,
        count=1,
        dtype="float32",
        nodata=-9999,
        transform=from_origin(UTM_WEST, UTM_NORTH, CELL, CELL),
        **layout,
    ) as dst:
        dst.write(_grid(), 1)
    return str(path)


def _chunks(path, **kwargs):
    reader = RasterioReader(path, src_srs="EPSG:26911", **kwargs)
    return [c for c in reader.yield_chunks() if c is not None]


def _points(chunks):
    return np.sort(np.concatenate(chunks)[["x", "y", "z"]].copy(), order=["x", "y"])


@pytest.mark.parametrize("fmt", ["asc", "striped_tif"])
def test_thin_strips_read_in_bands(tmp_path, fmt):
    if fmt == "asc":
        path = _write_asc(tmp_path / "grid.asc")
    else:
        path = _write_tif(tmp_path / "grid.tif", blockysize=1)
    with rasterio.open(path) as src:
        assert src.block_shapes[0] == (1, STRIP_COLS)

    chunks = _chunks(path)
    assert len(chunks) == 3
    assert sum(len(c) for c in chunks) == STRIP_ROWS * STRIP_COLS


def test_banded_strips_give_same_points_as_single_strips(tmp_path, monkeypatch):
    path = _write_tif(tmp_path / "grid.tif", blockysize=1)
    banded = _chunks(path)

    # The old behaviour: one chunk per one-row strip.
    monkeypatch.setattr(RasterioReader, "STRIP_CHUNK_CELLS", 1)
    single = _chunks(path)
    assert len(single) == STRIP_ROWS

    np.testing.assert_array_equal(_points(banded), _points(single))


def test_explicit_chunk_size_wins(tmp_path):
    path = _write_asc(tmp_path / "grid.asc")
    assert len(_chunks(path, chunk_size=1000)) == 3  # 3 row bands x 1 column band
    assert len(_chunks(path, chunk_size=300)) == 7 * 2


def test_tiled_geotiff_keeps_its_blocks(tmp_path):
    path = _write_tif(
        tmp_path / "tiled.tif", tiled=True, blockxsize=256, blockysize=256
    )
    assert len(_chunks(path)) == 9 * 2  # ceil(2100 / 256) x ceil(500 / 256)


def test_thick_strips_keep_their_blocks(tmp_path):
    path = _write_tif(tmp_path / "thick.tif", blockysize=1024)
    assert len(_chunks(path)) == 3  # 1024-row strips already hold >= 500k cells
