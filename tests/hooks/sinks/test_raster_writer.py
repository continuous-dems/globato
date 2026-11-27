"""Canonical/legacy native raster streams preserve pixels and metadata."""

import numpy as np
import pytest
from globato.hooks.sinks.raster_writer import RasterWrite


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("legacy_field", [False, True])
def test_native_raster_writer_consumes_stream_and_preserves_metadata(
    tmp_path, inline, legacy_field
):
    import rasterio
    from affine import Affine
    from rasterio.windows import Window

    source, output = tmp_path / "source.tif", tmp_path / "checkpoint.tif"
    profile = dict(
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        transform=Affine(3, 0, 350000, 0, -3, 3750000),
        crs="EPSG:32611",
    )
    with rasterio.open(source, "w", **profile) as dst:
        dst.write(np.ones((1, 2, 2), "float32"))
        dst.update_tags(survey_label="measured")
        dst.set_band_description(1, "elevation")
        dst.set_band_unit(1, "m")

    def stream():
        yield profile
        window = Window(0, 0, 2, 2)
        yield (
            window,
            window,
            np.full((1, 2, 2), 2.0, "float32"),
            None,
            profile["transform"],
        )

    entry = dict(
        dst_fn=str(source), stream_type="raster" if legacy_field else "raster-stream"
    )
    entry["raster_stream" if legacy_field else "stream"] = stream()
    writer = RasterWrite(output_path=str(output), inline=inline)
    writer.run([(None, entry)])
    if inline:
        list(entry["stream"])
    with rasterio.open(output) as dst:
        assert (dst.read() == 2).all()  # The modified stream, not an old file copy.
        assert dst.tags()["survey_label"] == "measured"
        assert dst.descriptions == ("elevation",)
        assert dst.units == ("m",)
    with rasterio.open(source) as src:
        assert (src.read() == 1).all()
