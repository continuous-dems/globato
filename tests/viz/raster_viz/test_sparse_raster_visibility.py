"""Sparse raster symbols survive preview reduction and retain source colors."""

import hashlib

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin

from globato.viz.raster_viz.palette import elevation_rgba
from globato.viz.raster_viz.review import Review, sparse_elevation_cells


def write_raster(path, data):
    if data.ndim == 2:
        data = data[None]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[2],
        height=data.shape[1],
        count=data.shape[0],
        dtype="float32",
        nodata=-9999,
        crs="EPSG:3857",
        transform=from_origin(1000, 20000, 10, 10),
    ) as dst:
        dst.write(data.astype("float32"))


def test_sparse_cells_survive_downsampling_with_unshaded_raster_colors(tmp_path):
    source = tmp_path / "historical_cells.tif"
    a = np.full((1024, 1024), -9999, dtype="float32")
    a[101, 101], a[901, 901] = -20, 20
    write_raster(source, a)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    review = Review(tmp_path / "viz", max_side=128)
    record = review.add_raster("soundings", source, limits=[-30, 30])
    image = np.asarray(Image.open(review.output / record["image"]))
    assert record["sparse_display"]["valid_source_cells"] == 2
    assert record["source"]["preview_rendering"] == "valid_source_cells_as_visible_dots"
    assert image.shape == (128, 128, 4)
    # Neither isolated source cell is selected by the ordinary nearest preview.
    assert image[..., 3].sum() > 2 * 255
    for row, col, value in [(101, 101, -20), (901, 901, 20)]:
        x, y = int((col + 0.5) / 8), int((row + 0.5) / 8)
        expected = (elevation_rgba(np.array([value]), [-30, 30])[0] * 255).astype(
            "uint8"
        )
        np.testing.assert_array_equal(image[y, x], expected)
    assert record["legend"]["ramp"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


def test_sparse_scan_uses_selected_band_and_excludes_nodata_and_nan(tmp_path):
    source = tmp_path / "two_bands.tif"
    a = np.full((2, 256, 256), -9999, dtype="float32")
    rng = np.random.default_rng(17)
    cells = rng.choice(256 * 256, 454, replace=False)
    a[1].flat[cells] = np.arange(454)
    a[1, 255, 255] = np.nan
    write_raster(source, a)
    with rasterio.open(source) as src:
        assert sparse_elevation_cells(src, 1) is None
        x, y, z = sparse_elevation_cells(src, 2)
    assert len(z) == 454
    np.testing.assert_array_equal(np.sort(z), np.arange(454))
    rows, cols = np.nonzero(np.isfinite(a[1]) & (a[1] != -9999))
    np.testing.assert_array_equal(x, 1000 + (cols + 0.5) * 10)
    np.testing.assert_array_equal(y, 20000 - (rows + 0.5) * 10)


def test_dense_raster_keeps_existing_preview_path(tmp_path):
    source = tmp_path / "dense.tif"
    write_raster(source, np.arange(100).reshape(10, 10))
    with rasterio.open(source) as src:
        assert sparse_elevation_cells(src, 1) is None
    record = Review(tmp_path / "viz").add_raster("dem", source)
    assert "sparse_display" not in record
    assert (
        record["source"]["preview_rendering"]
        == "shade_on_source_grid_then_project_RGBA"
    )
