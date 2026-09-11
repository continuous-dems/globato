# tests/test_utils.py

import os
import rasterio
import numpy as np
from fetchez.spatial import Region
from globato.utils import resolve_barrier


def test_resolve_barrier_empty_geojson_yields_blank_raster(tmp_path):
    """Prove that an empty GeoJSON safely rasterizes to a zero-filled mask."""
    # Create a valid but empty GeoJSON
    empty_geojson = tmp_path / "empty_mask.geojson"
    empty_geojson.write_text('{"type": "FeatureCollection", "features": []}')

    # Define a dummy region to provide rasterization bounds
    region = Region(-124.5, -124.25, 41.5, 41.75, srs="EPSG:4326")

    # Resolve the barrier into a raster
    raster_path = resolve_barrier(
        barrier_str=str(empty_geojson),
        region=region,
        outdir=str(tmp_path),
        res="1s",
        output_type="raster",
        target_crs="EPSG:4326",
    )

    # Verify the raster was successfully created
    assert raster_path is not None
    assert os.path.exists(raster_path)

    # Read the output raster and ensure it contains absolutely no land (1s)
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        assert np.max(data) == 0
        assert np.all(data == 0)
