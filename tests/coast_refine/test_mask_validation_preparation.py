"""Prepare cached-mask geometry without weakening the independent pixel audit."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import shapely
from shapely.geometry import Polygon

from globato.coast_refine import masking


@pytest.mark.parametrize("mode", ["direct", "cache"])
def test_validation_prepares_fresh_geometry_and_still_rejects_corruption(
    tmp_path, monkeypatch, mode
):
    source = tmp_path / "native.tif"
    values = np.arange(64, dtype="float32").reshape(8, 8) - 20
    values[0, 0] = -9999
    values[2, 2] = -0.0
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=1,
        dtype="float32",
        nodata=-9999,
        crs=4326,
        transform=from_origin(0, 8, 1, 1),
    ) as dst:
        dst.write(values, 1)
    keep = Polygon(
        [(0.5, 0.5), (6.5, 0.5), (6.5, 7.5), (0.5, 7.5)],
        holes=[[(3, 3), (5, 3), (5, 5), (3, 5)]],
    )
    args = dict(
        provenance=dict(
            source_identity={"url": "fixture"},
            source_weight=1,
            module_weight=2,
            application_geometry_selection="source_retention_native_centres",
        ),
        mask_manifest="fixture-manifest.json",
        preparation_id="fixture-preparation",
    )
    receipt = masking.publish_masked_source(source, keep, tmp_path / "derived", **args)
    fresh = shapely.from_wkb(keep.wkb)
    original_wkb = fresh.wkb
    assert not shapely.is_prepared(fresh)
    original_classification = masking._native_classification
    checks = []

    def checked(native, geographic, window, transformer=None):
        assert shapely.is_prepared(geographic), (
            "Validation skipped geometry preparation"
        )
        checks.append(window)
        return original_classification(native, geographic, window, transformer)

    monkeypatch.setattr(masking, "_native_classification", checked)
    if mode == "cache":
        assert (
            masking.publish_masked_source(source, fresh, tmp_path / "derived", **args)
            == receipt
        )
    else:
        counts = masking.validate_product(source, receipt["masked"], fresh)
        assert counts["included_cells"] == receipt["included_cells"]
    assert checks
    assert fresh.wkb == original_wkb
    with rasterio.open(receipt["masked"]) as dst:
        output = dst.read(1)
    retained = np.isfinite(output)
    assert values[retained].tobytes() == output[retained].tobytes()
    # Exact edge ties and the hole remain excluded; negative zero is retained.
    assert not np.isfinite(output[0, 1])
    assert not np.isfinite(output[3, 3])
    assert np.signbit(output[2, 2])
    with rasterio.open(receipt["masked"], "r+") as dst:
        damaged = output.copy()
        damaged[2, 2] = 0.0  # same numeric value, different original bits
        dst.write(damaged, 1)
    with pytest.raises(ValueError, match="elevation bits"):
        masking.validate_product(source, receipt["masked"], fresh)
    with pytest.raises(ValueError, match="identity/hash"):
        masking.publish_masked_source(source, fresh, tmp_path / "derived", **args)
