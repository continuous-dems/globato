"""Coast Mask consumes native receipts and rejects stale policy or state."""

import pytest
import rasterio
from globato.coast_refine.source_grid_binding import bind_source_grid
from globato.hooks.sinks.coast_mask_side_stack import CoastMaskSideStack
from types import SimpleNamespace
import numpy as np
from fetchez.spatial import Region


def source(tmp_path):
    asset = tmp_path / "source.las"
    asset.write_bytes(b"asset fixture")
    points = np.array(
        [(0.5, 1.5, 10.0, 2.0, 0.1)],
        dtype=[(key, "f8") for key in ("x", "y", "z", "w", "u")],
    )
    module = SimpleNamespace(
        name="local", region=Region(0, 2, 0, 2), weight=2, uncertainty=0, hooks=[]
    )
    entry = dict(
        url=asset.as_uri(),
        dst_fn=str(asset),
        src_srs="EPSG:4326",
        stream_type="point-stream",
        data_type="lidar",
        stream=iter([points]),
    )
    return module, entry


@pytest.mark.parametrize("drift", ["none", "policy", "weight", "file", "identity"])
def test_binding_rejects_mutation_after_native_registration(tmp_path, drift):
    pair = source(tmp_path)
    entry = pair[1]
    entry["coast_refine_consumer"] = dict(authority="a", remove="land")
    CoastMaskSideStack(res=1, cache_dir=tmp_path).run([pair])
    list(entry["stream"])
    if drift == "policy":
        entry["coast_refine_consumer"]["remove"] = "water"
    elif drift == "weight":
        entry["weight"] = 7
    elif drift == "identity":
        entry["_sidestack_processing_identity"]["selectors"]["module"] = "other"
    elif drift == "file":
        with rasterio.open(entry["artifacts"]["side-stack"], "r+") as ds:
            value = ds.read(1)
            value[0, 0] += 1
            ds.write(value, 1)
    if drift == "none":
        receipt = bind_source_grid(entry, expected_crs=4326)
        assert (
            receipt["fresh_sink_receipt"]
            and receipt["identity_mode"] == "native_fresh_receipt"
        )
    else:
        with pytest.raises(ValueError, match="sink receipt"):
            bind_source_grid(entry, expected_crs=4326)
