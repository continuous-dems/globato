"""Release-review regressions for ordinary hook behavior and isolated contexts."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import sys
import threading
import numpy as np
import pytest
from test_stack_qc_native import run_plan as run_fetchez
from fetchez.hooks import FetchHook
from globato.hooks.rasters.raster_stream import RasterStreamInit
from globato.hooks.sinks.raster_writer import RasterWrite
from globato.qc.stack import io, preparation, processing, restoration
from globato.qc.stack.native import NativeBuilder
from test_stack_qc_native import setup
from test_stack_qc_science import pool


class FinalNotice(FetchHook):
    name = "fixture_final_notice"
    meta_stage = "collection"

    def __init__(self, path):
        super().__init__()
        self.path = str(path)

    def run(self, entries):
        with Path(self.path).open("a") as stream:
            stream.write("final build\n")
        return entries


class SourceAudit(FetchHook):
    name = "audit"
    meta_stage = "file"

    def __init__(self, path):
        super().__init__()
        self.path = str(path)

    def run(self, entries):
        with Path(self.path).open("a") as stream:
            for _, entry in entries:
                stream.write(Path(entry["dst_fn"]).name + "\n")
        return entries


class ContextResource(RasterStreamInit):
    """A lazy native raster hook whose resource must live through consumption."""

    def run(self, entries):
        self.closed = False
        entries = super().run(entries)
        for _, entry in entries:
            original = entry["raster_stream"]

            def guarded(original=original):
                for chunk in original:
                    if self.closed:
                        raise RuntimeError(
                            "Context resource was closed before stream consumption"
                        )
                    yield chunk

            entry["raster_stream"] = guarded()
        return entries

    def teardown(self):
        self.closed = True


def test_normal_file_audit_runs_once_and_notification_is_final_only(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    module, _, hooks = setup(tmp_path)
    audit, notice = tmp_path / "audit.txt", tmp_path / "notice.txt"
    run_fetchez(
        [module],
        threads=1,
        ignore_failures=False,
        global_hooks=[SourceAudit(audit), FinalNotice(notice), *hooks],
    )
    assert sorted(audit.read_text().splitlines()) == ["points.inf", "points.xyz"]
    assert notice.read_text().splitlines() == ["final build"]


def test_context_keeps_native_raster_chain_and_skips_generic_collection_hooks(tmp_path):
    _, _, hooks = setup(tmp_path)
    stream = RasterStreamInit(stage="collection")
    checkpoint = RasterWrite(output_path=str(tmp_path / "production_checkpoint.tif"))
    notice = FinalNotice(tmp_path / "notice.txt")
    interpolator = hooks[-1]
    chain = [notice, hooks[1], stream, checkpoint, interpolator]
    assert preparation.native_context_hooks(chain, interpolator) == chain[1:]


@pytest.mark.parametrize("stream_class", [RasterStreamInit, ContextResource])
def test_native_checkpoint_cannot_overwrite_production_output(
    tmp_path, monkeypatch, stream_class
):
    monkeypatch.chdir(tmp_path)
    module, _, hooks = setup(tmp_path)
    z = np.full((64, 64), -100.0, "float32")
    sources, _ = pool(tmp_path, z)
    records = sources.records
    for record in records:
        record.update(
            weight=1.0,
            module="local_elevation",
            cache_dir=str(tmp_path),
            entry=dict(
                url=Path(record["path"]).as_uri(),
                dst_fn=record["path"],
                src_srs="EPSG:32611",
            ),
        )
    production = tmp_path / "production_checkpoint.tif"
    production.write_bytes(b"existing production artifact")
    builder = NativeBuilder(
        hooks[0],
        [
            stream_class(stage="collection"),
            RasterWrite(output_path=str(production)),
            hooks[-1],
        ],
        records,
        module.region,
    )
    result = builder(None, tmp_path / "context")
    assert Path(result).is_file()
    assert production.read_bytes() == b"existing production artifact"
    sources.close()


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


def test_context_identity_covers_every_native_operator_implementation(tmp_path):
    module, qc, hooks = setup(tmp_path)
    operator = tmp_path / "fixture_native_operator.py"
    operator.write_text(
        'from fetchez.hooks import FetchHook\nclass Operator(FetchHook):\n    name="fixture_operator"\n'
    )
    spec = importlib.util.spec_from_file_location("fixture_native_operator", operator)
    fixture = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixture
    try:
        spec.loader.exec_module(fixture)
        post = [fixture.Operator(), hooks[-1]]
        records = [dict(source_id="source", source_sha256="original", group=None)]
        before = preparation.context_key(qc, module, records, hooks[0], post)
        operator.write_text(
            operator.read_text() + "# changed operator implementation\n"
        )
        after = preparation.context_key(qc, module, records, hooks[0], post)
        assert before != after
        checkpoint = RasterWrite(output_path="first-output.tif")
        post = [checkpoint, hooks[-1]]
        first = preparation.context_key(qc, module, records, hooks[0], post)
        checkpoint.output_path = "another-output.tif"
        assert first == preparation.context_key(qc, module, records, hooks[0], post)
    finally:
        sys.modules.pop(spec.name, None)


def test_restoration_halo_uses_its_scientific_policy():
    assert restoration.context_halo((3, 3)) == 52
    policy = dict(restoration.POLICY, maximum_flank_distance_m=120)
    assert restoration.context_halo((3, 3), policy) == 62


def test_normal_globato_command_returns_failure_for_invalid_hook_settings(tmp_path):
    import subprocess
    import yaml

    module, _, _ = setup(tmp_path)
    recipe = dict(
        project=dict(name="invalid_qc"),
        region=module.region.to_list(),
        region_srs="EPSG:32611",
        modules=[
            dict(
                module="local_fs",
                args=dict(path=module.path, data_type="xyz"),
                hooks=[dict(name="stack_qc", args=dict(max_passes=11))],
            )
        ],
    )
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(recipe))
    result = subprocess.run(
        [
            str(Path(sys.executable).with_name("globato")),
            "run",
            str(path),
            "--outdir",
            str(tmp_path / "output"),
            "--fail-fast",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "Invalid max_passes" in result.stdout + result.stderr
    assert "Successfully executed" not in result.stdout + result.stderr
