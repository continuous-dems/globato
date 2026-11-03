"""The optional maps cannot allocate a full quarter-degree grid for display."""

import numpy as np
import rasterio
from rasterio.transform import from_origin

from globato.qc.stack.review import visualize as viz


def tif(path, data, *, transform=None, native=False):
    if data.ndim == 2:
        data = data[None]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[-2],
        width=data.shape[-1],
        count=data.shape[0],
        dtype=data.dtype,
        crs="EPSG:32611",
        transform=transform or from_origin(350000, 3750000, 3, 3),
        nodata=None,
    ) as ds:
        ds.write(data)
        if native:
            ds.update_tags(GLOBATO_DATATYPE="MULTI_STACK")
    return path


def test_thumbnail_surface_and_footprint_preserve_isolated_rejected_and_restored_cells(
    tmp_path,
):
    orig = np.full((90, 90), -100, dtype="float32")
    dem = tif(tmp_path / "dem.tif", orig)
    before = np.zeros((90, 90), dtype="uint8")
    after = before.copy()
    before[41, 44] = 1
    before[50, 55] = 1
    after[41, 44] = 1  # this source location remains rejected
    b = tif(tmp_path / "before.tif", before)
    a = tif(tmp_path / "after.tif", after)
    bm = {"masks": {"source": {"mask": str(b)}}}
    am = {"masks": {"source": {"mask": str(a)}}}
    z, (dx, dy), _, _ = viz.surface(dem, max_pixels=100)
    assert z.shape == (10, 10) and abs(dx - 27) < 0.1 and abs(dy - 27) < 0.1
    rejected = viz.removed(am, dem, max_pixels=100)
    restored = viz.removed(am, dem, bm, max_pixels=100)
    assert rejected.shape == restored.shape == z.shape
    assert rejected[4, 4] and not restored[4, 4]
    assert restored[5, 6] and not rejected[5, 6]
    with rasterio.open(a) as ds:
        assert ds.read(1).sum() == 1  # scientific mask unchanged


def test_thumbnail_mask_clip_and_overlay_independent_sources(tmp_path):
    dem = tif(tmp_path / "full.tif", np.full((90, 90), -50, "float32"))
    offset = from_origin(350000 + 12 * 3, 3750000 - 18 * 3, 3, 3)
    m = np.zeros((15, 15), "uint8")
    m[3, 3] = 1
    path = tif(tmp_path / "smaller_mask.tif", m, transform=offset)
    a = viz.removed({"masks": {"a": {"mask": str(path)}}}, dem, max_pixels=100)
    assert a.sum() >= 1
    assert a[2, 1]


def test_thumbnail_measured_surface_selection_and_image(tmp_path):
    grid = np.zeros((7, 90, 90), "float32")
    grid[0] = -120
    grid[1, 44, 44] = 1
    stack = tif(tmp_path / "stack.tif", grid, native=True)
    z, measured, _, _ = viz.measured_cells(stack, max_pixels=100)
    assert z.shape == measured.shape == (10, 10)
    assert measured[4, 4]
    baseline = tif(tmp_path / "baseline.tif", np.full((90, 90), -100, "float32"))
    orig = viz.MAX_REVIEW_PIXELS
    try:
        viz.MAX_REVIEW_PIXELS = 100
        viz.render_measurements(
            stack, baseline, tmp_path / "measured.png", restored=measured
        )
    finally:
        viz.MAX_REVIEW_PIXELS = orig
    assert (tmp_path / "measured.png").stat().st_size > 1000


def test_chunked_large_mask_restoration_and_thumbnail_match_full_warp(tmp_path):
    """Long masks are chunked; sparse marks must not vanish at 512-cell seams."""
    from rasterio.warp import reproject
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from globato.qc.stack.review.visualize import removed

    size = 1400  # > 512-cell source chunks in both dimensions
    arr = np.zeros((size, size), dtype="uint8")
    spots = [(17, 7), (511, 511), (512, 512), (1023, 500), (1024, 1024), (1338, 1339)]
    for r, c in spots:
        arr[r, c] = 1
    # A second source overlaps the same display grid with a nonzero native offset.
    shifted = np.zeros((size - 200, size - 500), dtype="uint8")
    shifted[511, 211] = 1
    shifted[512, 211] = 1
    dem = tif(tmp_path / "large_dem.tif", np.full((size, size), -200, "float32"))
    source = tif(tmp_path / "large_source.tif", arr)
    offs = from_origin(350000 + 500 * 3, 3750000 - 200 * 3, 3, 3)
    other = tif(tmp_path / "other.tif", shifted, transform=offs)
    manifest = {"masks": {"one": {"mask": str(source)}, "two": {"mask": str(other)}}}
    actual = removed(manifest, dem, max_pixels=10_000)
    assert actual.shape == (100, 100)
    # Whole-source warp is only a reference in this small synthetic test.
    expected = np.zeros((100, 100), dtype="uint8")
    with rasterio.open(dem) as d:
        for src in (source, other):
            with rasterio.open(src) as s:
                tmp = np.zeros_like(expected)
                reproject(
                    source=s.read(1),
                    destination=tmp,
                    src_transform=s.transform,
                    src_crs=s.crs,
                    dst_transform=d.transform * Affine.scale(size / 100),
                    dst_crs=d.crs,
                    src_nodata=0,
                    dst_nodata=0,
                    resampling=Resampling.max,
                )
                expected |= tmp
    assert np.array_equal(actual, expected.astype(bool))
    after = np.zeros((size, size), dtype="uint8")
    after[511, 511] = 1
    fin = tif(tmp_path / "after.tif", after)
    restored = removed(
        {"masks": {"one": {"mask": str(fin)}}},
        dem,
        {"masks": {"one": {"mask": str(source)}}},
        max_pixels=10_000,
    )
    assert restored.any()
    assert restored[36, 36]  # adjacent restored cell shares this thumbnail pixel
    empty = removed(
        {"masks": {"one": {"mask": str(source)}}},
        dem,
        {"masks": {"one": {"mask": str(source)}}},
        max_pixels=10_000,
    )
    assert not empty.any()  # no changes cannot produce fake restored marks
    # Scientific source raster data were not edited by thumbnail preparation.
    with rasterio.open(source) as src:
        assert int(src.read(1).sum()) == len(spots)
