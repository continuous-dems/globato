"""Grid-native terrain authorities retain the authoritative input contract."""

from types import SimpleNamespace
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from fetchez.spatial import Region
from globato.coast_refine.integration import CoastMask


@pytest.mark.parametrize(
    "encoding", ["packed", "integer", "feet", "multiband", "valid"]
)
def test_producer_encoding_is_checked_before_streaming(tmp_path, encoding):
    source = tmp_path / "terrain.tif"
    dtype = "int16" if encoding == "integer" else "float32"
    count = 2 if encoding == "multiband" else 1
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=count,
        dtype=dtype,
        crs=4326,
        transform=from_origin(0, 16, 1, 1),
    ) as ds:
        ds.write(np.ones((count, 16, 16), dtype=dtype))
        if encoding == "packed":
            ds.scales = (0.01,)
        if encoding == "feet":
            ds.set_band_unit(1, "ft")
    module = SimpleNamespace(
        name="terrain", weight=1, region=Region(0, 16, 0, 16), hooks=[]
    )
    entry = {"dst_fn": str(source), "data_type": "raster"}
    hook = CoastMask(
        "authority", mask="local_water.gpkg", polygon_class="water", remove="water"
    )
    if encoding == "valid":
        assert (
            hook.run([(module, entry)])[0][1]["coast_refine_request"]["point_source"]
            is False
        )
    else:
        with pytest.raises(ValueError):
            hook.run([(module, entry)])
        assert "coast_refine_request" not in entry
