"""Release-review regressions for ordinary hook behavior and isolated contexts."""

from concurrent.futures import ThreadPoolExecutor
import threading
import pytest
from globato.qc.stack import io, processing, restoration


def test_cached_identical_surface_needs_no_full_raster_read(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Unchanged context must not reread two full DEMs")

    monkeypatch.setattr(processing.rasterio, "open", unexpected)
    assert processing.changes(
        tmp_path / "context.tif", tmp_path / "context.tif"
    ) == dict(maximum_absolute_change_m=0.0, coverage_changed_cells=0)


def test_concurrent_hash_receipts_publish_complete_json(tmp_path, monkeypatch):
    original = io.os.replace
    barrier = threading.Barrier(2)

    def simultaneous(source, destination):
        barrier.wait(timeout=10)
        original(source, destination)

    monkeypatch.setattr(io.os, "replace", simultaneous)
    receipt = tmp_path / "hash.json"
    with ThreadPoolExecutor(2) as workers:
        jobs = [
            workers.submit(io.write, receipt, dict(sha256=str(index)))
            for index in range(2)
        ]
        for job in jobs:
            job.result()
    assert io.read(receipt) in [dict(sha256="0"), dict(sha256="1")]
    assert not list(tmp_path.glob("*.partial"))


def test_failed_json_write_preserves_existing_receipt_and_cleans_temp(tmp_path):
    receipt = tmp_path / "hash.json"
    io.write(receipt, dict(sha256="original"))
    with pytest.raises(ValueError):
        io.write(receipt, dict(broken=float("nan")))
    assert io.read(receipt) == dict(sha256="original")
    assert not list(tmp_path.glob("*.partial"))


def test_restoration_halo_uses_its_scientific_policy():
    assert restoration.context_halo((3, 3)) == 52
    policy = dict(restoration.POLICY, maximum_flank_distance_m=120)
    assert restoration.context_halo((3, 3), policy) == 62
