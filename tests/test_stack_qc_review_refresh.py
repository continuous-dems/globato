"""Review products must reflect the current run and scale to large grids."""

import numpy as np
import rasterio
from matplotlib.colors import LightSource
from rasterio.transform import from_origin

from globato.qc.stack.review.final_stages import (
    module_interpolated_preview,
    write_hillshade_tif,
    write_measured_surface,
)
from globato.qc.stack.review.review_page import publish


def native(path, mean, count):
    a = np.zeros((7, *mean.shape), "float32")
    a[0], a[1] = mean, count
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=mean.shape[0],
        width=mean.shape[1],
        count=7,
        dtype="float32",
        crs="EPSG:32611",
        transform=from_origin(350000, 3750000, 3, 3),
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(a)
        dst.update_tags(GLOBATO_DATATYPE="MULTI_STACK")


def test_review_rerun_replaces_stale_products(tmp_path):
    y, x = np.indices((520, 530))
    mean = (-200 + x * 0.04 + y * 0.02 + 12 * np.sin(x / 31)).astype("float32")
    count = np.ones(mean.shape, "float32")
    count[20:70, 20:70] = 0
    src = tmp_path / "stack.tif"
    measured = tmp_path / "measured.tif"
    hillshade = tmp_path / "hillshade.tif"
    preview = tmp_path / "preview.tif"
    native(src, mean, count)
    write_measured_surface(src, measured)
    write_hillshade_tif(measured, hillshade)
    module_interpolated_preview(measured, preview)
    with rasterio.open(measured) as ds:
        assert ds.read(1)[30, 30] == ds.nodata
        before = ds.read(1)[250, 250]
    with rasterio.open(preview) as ds:
        original_preview = ds.read(1)[250, 250]
    with rasterio.open(hillshade) as ds:
        first_hillshade = ds.read(1).copy()
    # Same filenames, different legitimate QC result: all products must refresh.
    mean += 40 * np.exp(-((x - 250) ** 2 + (y - 250) ** 2) / 1000)
    native(src, mean, count)
    write_measured_surface(src, measured)
    write_hillshade_tif(measured, hillshade)
    module_interpolated_preview(measured, preview)
    with rasterio.open(measured) as ds:
        assert ds.read(1)[250, 250] > before + 30
    with rasterio.open(preview) as ds:
        assert ds.read(1)[250, 250] > original_preview + 30
    with rasterio.open(hillshade) as ds:
        assert not np.array_equal(ds.read(1), first_hillshade)
        assert np.all(ds.read(1)[20:70, 20:70] == 0)
    assert not list(tmp_path.glob("*.partial.tif"))


def test_windowed_hillshade_matches_full_grid_on_valid_surface(tmp_path):
    y, x = np.indices((520, 530))
    z = (-200 + x * 0.04 + y * 0.02 + 12 * np.sin(x / 31)).astype("float32")
    src = tmp_path / "z.tif"
    with rasterio.open(
        src,
        "w",
        driver="GTiff",
        height=z.shape[0],
        width=z.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:32611",
        transform=from_origin(350000, 3750000, 3, 3),
        nodata=-9999.0,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(z, 1)
    dst_path = write_hillshade_tif(src, tmp_path / "hill.tif")
    gy, gx = np.gradient(z.astype("float64"), -3, 3)
    direction = LightSource(azdeg=315, altdeg=45).direction
    expected = np.clip(
        (-gx * direction[0] - gy * direction[1] + direction[2])
        / np.sqrt(1 + gx * gx + gy * gy),
        0,
        1,
    )
    expected = np.clip(np.round(expected * 254) + 1, 1, 255).astype("uint8")
    with rasterio.open(dst_path) as ds:
        # Floating-point gradient evaluation may change the final 8-bit
        # rounding by one gray level; it must not create block-edge seams.
        assert (
            np.max(np.abs(ds.read(1).astype("int16") - expected.astype("int16"))) <= 1
        )
        assert ds.nodata == 0


def test_html_links_to_both_review_geotiffs(tmp_path):
    publish(
        tmp_path,
        [
            dict(
                label="Module measurements",
                status="Done",
                image="review/measured.png",
                downloads=[
                    (
                        "Cleaned measurements (GeoTIFF)",
                        "review/module_final_measured.tif",
                    ),
                    (
                        "Measured terrain hillshade (GeoTIFF)",
                        "review/module_final_measured_hillshade.tif",
                    ),
                ],
            )
        ],
        complete=True,
    )
    html = (tmp_path / "review.html").read_text()
    assert 'href="review/module_final_measured.tif"' in html
    assert 'href="review/module_final_measured_hillshade.tif"' in html


def test_failed_review_does_not_leave_stale_success_page(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from globato.qc.stack.review.publisher import ReviewPublisher
    import globato.qc.stack.review.visualize as visualization

    final = tmp_path / "normal_final.tif"
    final.write_bytes(b"production result already finished")
    publisher = ReviewPublisher(
        cache=tmp_path,
        result={},
        stack=None,
        region=None,
        interpolator=SimpleNamespace(output=str(final)),
        module_name="mbdb",
    )
    (tmp_path / "review.html").write_text("old run - do not show")
    (tmp_path / "review_ERROR.txt").write_text("old error")

    def fail(*args, **kwargs):
        raise ValueError("intentional failed review")

    monkeypatch.setattr(visualization, "completed", fail)
    original = [("module", dict(dst_fn=str(final)))]
    assert publisher.run(original) == original
    assert not (tmp_path / "review.html").exists()
    assert "intentional failed review" in (tmp_path / "review_ERROR.txt").read_text()
    assert final.read_bytes() == b"production result already finished"

    def pass_review(run, **kwargs):
        (run / "review.html").write_text("current review")
        return []

    monkeypatch.setattr(visualization, "completed", pass_review)
    assert publisher.run(original) == original
    assert (tmp_path / "review.html").read_text() == "current review"
    assert not (tmp_path / "review_ERROR.txt").exists()
