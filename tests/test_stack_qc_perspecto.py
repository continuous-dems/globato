"""Review GeoTIFFs use Perspecto, not a second hillshade algorithm."""

import sys
import types
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from globato.qc.stack.review.perspecto import colored_hillshade, reference_limits


def sample(path, value=-120):
    z = np.linspace(value - 10, value + 10, 30 * 40, dtype="float32").reshape(30, 40)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=40,
        height=30,
        count=1,
        dtype="float32",
        crs="EPSG:4269",
        transform=from_origin(-118.75, 34, 0.001, 0.001),
        nodata=-9999,
    ) as ds:
        ds.write(z, 1)
    return path


def test_perspecto_review_uses_same_color_range_and_preserves_geo_tags(
    tmp_path, monkeypatch
):
    called = []

    class FakePerspecto:
        def __init__(self, **kwargs):
            called.append(kwargs)

        def _process_file_fallback(self, source, dst, entry):
            with rasterio.open(source) as src:
                profile = src.profile.copy()
                profile.update(count=4, dtype="uint8", nodata=None)
                with rasterio.open(dst, "w", **profile) as output:
                    output.write(np.full((4, src.height, src.width), 127, "uint8"))
            return True

    module = types.ModuleType("globato.hooks.viz.geohillshade")
    module.GeoHillshade = FakePerspecto
    monkeypatch.setitem(sys.modules, module.__name__, module)
    source = sample(tmp_path / "input.tif")
    limits = reference_limits(source)
    for stem in ("initial", "module", "globato"):
        path = tmp_path / (stem + ".tif")
        colored_hillshade(source, path, color_limits=limits)
        with rasterio.open(path) as ds:
            assert ds.crs.to_epsg() == 4269 and ds.count == 4
            assert ds.tags()["DISPLAY_COLOR_MIN_M"] == str(limits[0])
            assert ds.tags()["DISPLAY_COLOR_MAX_M"] == str(limits[1])
            assert ds.tags()["STACK_QC_REVIEW"] == "true"
    assert len(called) == 3
    assert all(q["z_min"] == limits[0] and q["z_max"] == limits[1] for q in called)
    assert all(q["alpha"] for q in called)
    assert all(q["cmap"] == "coastal_relief" for q in called)


def test_perspecto_failure_leaves_no_new_partial_output(tmp_path, monkeypatch):
    class Failure:
        def __init__(self, **kwargs):
            pass

        def _process_file_fallback(self, *args):
            raise RuntimeError("simulated failure")

    module = types.ModuleType("globato.hooks.viz.geohillshade")
    module.GeoHillshade = Failure
    monkeypatch.setitem(sys.modules, module.__name__, module)
    source = sample(tmp_path / "input.tif")
    destination = tmp_path / "result.tif"
    with pytest.raises(RuntimeError, match="simulated failure"):
        colored_hillshade(source, destination, color_limits=(-150.0, -90.0))
    assert not destination.exists()
    assert not (tmp_path / "result.partial.tif").exists()
