"""Performance-path regressions: only change work, never measurements."""

import numpy as np
import rasterio
from rasterio.transform import from_origin

from globato.qc.stack.sources import SourcePool


def test_occupied_window_lookup_and_bounded_read_cache(tmp_path):
    records = []
    for index, tile in enumerate((0, 1, 2)):
        a = np.zeros((7, 32, 32), dtype="float32")
        a[0] = -120 + index
        a[1] = 1
        a[2] = 1
        path = tmp_path / f"source{index}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=32,
            height=32,
            count=7,
            dtype="float32",
            crs="EPSG:32611",
            transform=from_origin(tile * 256, 128, 1, 1),
        ) as dst:
            dst.write(a)
        record = dict(
            source_id=f"s{index}",
            path=str(path),
            group="g",
            offset=[0, tile * 256],
            grid=dict(shape=[32, 32]),
        )
        if index != 2:
            record["active_blocks"] = [f"0,{tile}"]
        records.append(record)
    pool = SourcePool(dict(grid=dict(shape=[256, 768]), sources=records))
    try:
        s0 = (slice(0, 128), slice(0, 128))
        assert [r["source_id"] for r in pool.active_records(s0)] == ["s0"]
        first = pool.source(records[0], s0)
        pool.source(records[0], s0)
        assert pool.source_read_count == 1
        assert pool.source_cache_hits == 1
        first[0][:] = 123  # cache remains safe even if a caller modifies its arrays
        again = pool.source(records[0], s0)
        assert again[0][0, 0] == -120
        assert pool._window_cache_bytes <= 32 * 1024 * 1024
        s2 = (slice(0, 128), slice(512, 640))
        assert [r["source_id"] for r in pool.active_records(s2)] == ["s2"]
        value, valid, weight, flag = pool.source(records[2], s2)
        assert np.isfinite(value[0, 0])
        assert pool.source_read_count == 2
        assert pool._window_cache_bytes <= 32 * 1024 * 1024
    finally:
        pool.close()
