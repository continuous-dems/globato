"""The established native-cell-centre convention excludes exact edge ties."""

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
from globato.coast_refine.masking import write_masked_tiff


def test_original_cell_centres_on_keep_boundary_are_excluded(tmp_path):
    source = tmp_path / "source.tif"
    output = tmp_path / "masked.tif"
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="float32",
        crs=4326,
        transform=from_origin(0, 4, 1, 1),
    ) as dst:
        dst.write(np.ones((4, 4), "float32"), 1)
    stats = write_masked_tiff(source, output, box(0.5, 0.5, 2.5, 3.5))
    expected = np.zeros((4, 4), bool)
    expected[1:3, 1] = True
    with rasterio.open(output) as ds:
        np.testing.assert_array_equal(np.isfinite(ds.read(1)), expected)
    assert stats["included_cells"] == 2
