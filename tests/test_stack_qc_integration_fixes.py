"""Regression cases found during the independent stock-Fetchez review."""

from types import SimpleNamespace
import pytest
from globato.qc.stack import pipeline
from globato.hooks.sinks.side_stack import SideStackCheck


class Hook:
    def __init__(self, name, stage, action=None):
        self.name, self.stage, self.action = name, stage, action
        self.closed = 0

    def run(self, entries):
        return self.action(entries) if self.action else entries

    def teardown(self):
        self.closed += 1


class Module:
    def __init__(self, name, entries, hooks):
        self.name, self.results = name, entries
        self.internal_hooks, self.external_hooks = [], hooks

    @property
    def hooks(self):
        return self.internal_hooks + self.external_hooks

    def run(self):
        pass


def qc():
    hook = Hook("stack_qc", "collection")
    hook.prepare_module = lambda mod, hooks, threads: pipeline.PreparedSources(
        mod,
        [dict(e, stream=iter([1]), stream_type="raster-stream") for e in mod.results],
    )
    return hook


def test_native_cache_check_is_deferred_only_for_qc(monkeypatch):
    measured = Module("qc", [{"id": "q"}], [qc()])
    ordinary = Module("ordinary", [{"id": "n"}], [])
    check = SideStackCheck()
    seen = []
    monkeypatch.setattr(
        check,
        "run",
        lambda entries: seen.extend(e["id"] for _, e in entries) or entries,
    )

    def consume(modules, **kw):
        return []

    monkeypatch.setattr(pipeline, "run_fetchez", consume)
    pipeline.StackQCPlan(
        config={}, modules=[measured, ordinary], global_hooks=[check]
    ).execute()
    assert seen == ["n"]  # QC's lookup belongs to source preparation after selection.


def test_manifest_global_order_reaches_native_submission(monkeypatch):
    first = Module("qc", [{"id": "a"}, {"id": "b"}], [qc()])
    other = Module("ordinary", [{"id": "c"}], [])
    selection = Hook(
        "reverse", "manifest", lambda entries: [entries[2], entries[0], entries[1]]
    )
    seen = []

    def consume(modules, *, global_hooks, **kw):
        entries = [(m, e) for m in modules for e in m.results]
        for hook in global_hooks:
            if hook.stage == "manifest":
                entries = hook.run(entries)
        seen.extend(e["id"] for _, e in entries)
        return entries

    monkeypatch.setattr(pipeline, "run_fetchez", consume)
    pipeline.StackQCPlan(
        config={}, modules=[first, other], global_hooks=[selection]
    ).execute()
    assert seen == ["c", "a", "b"]


def test_selection_failure_cleans_initialized_hooks():
    def fail(_):
        raise RuntimeError("selection failed")

    selection = Hook("broken", "manifest", fail)
    sink = Hook("multi_stack", "stream")
    module = Module("qc", [{"id": "a"}], [qc()])
    with pytest.raises(RuntimeError, match="selection failed"):
        pipeline.StackQCPlan(
            config={}, modules=[module], global_hooks=[selection, sink]
        ).execute()
    assert selection.closed == sink.closed == 1


def test_module_attached_consumer_rejected_before_source_processing(tmp_path):
    from globato.qc.stack.preparation import source_pipeline

    module = Module("qc", [{"id": "a"}], [qc(), Hook("multi_stack", "stream")])
    with pytest.raises(ValueError, match="global_hooks"):
        source_pipeline(
            module.hooks[0], module, [], SimpleNamespace(res="3", crs="EPSG:32611")
        )


def test_qc_rebuilds_persistent_final_state_instead_of_reusing_old_masks(
    tmp_path, monkeypatch
):
    import numpy as np
    import rasterio
    from fetchez.spatial import Region
    from globato.hooks.sinks.multi_stack import MultiStackHook

    monkeypatch.chdir(tmp_path)
    state = tmp_path / "final.state.tif"
    for depth in (-10.0, -20.0):
        points = np.array(
            [(350001.5, 3749998.5, depth)],
            dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")],
        )
        hook = qc()
        hook.prepare_module = lambda mod, hooks, threads: pipeline.PreparedSources(
            mod,
            [
                dict(e, stream=iter([points]), stream_type="point-stream")
                for e in mod.results
            ],
        )
        module = Module(
            "qc",
            [{"id": "a", "url": "https://example.invalid/source", "dst_fn": "source"}],
            [hook],
        )
        module.region = Region(350000, 350012, 3749988, 3750000, srs="EPSG:32611")
        module.weight = 1.0
        stack = MultiStackHook(
            res="3",
            crs="EPSG:32611",
            strategy="mean",
            state=str(state),
            resume=True,
            output=str(tmp_path / "final.tif"),
        )
        pipeline.StackQCPlan(
            config={}, modules=[module], global_hooks=[stack]
        ).execute()
        with rasterio.open(stack.output) as ds:
            assert ds.read(1, masked=True).compressed().tolist() == [depth]
        assert stack.resume is True  # Public hook configuration is restored.


def test_prepared_source_identity_survives_native_consumer_and_provenance(tmp_path):
    from globato.hooks.sinks.multi_stack import MultiStackHook
    from globato.hooks.metadata.provenance import source_id

    a = tmp_path / "a" / "source.xyz"
    b = tmp_path / "b" / "source.xyz"
    for path in (a, b):
        path.parent.mkdir()
        path.write_bytes(b"same size")
    first = {
        "dst_fn": str(a),
        "url": a.as_uri(),
        "_stack_qc_source_id": "source|sidestack=aaa",
    }
    second = {
        "dst_fn": str(b),
        "url": b.as_uri(),
        "_stack_qc_source_id": "source|sidestack=bbb",
    }
    assert MultiStackHook._dataset_id(first) != MultiStackHook._dataset_id(second)
    assert source_id(first) != source_id(second)
    # Inputs without QC retain the upstream identity convention.
    assert (
        MultiStackHook._dataset_id({"dst_fn": str(a), "url": a.as_uri()})
        == "source.xyz|9B|w=1.0"
    )


def test_deferred_manifest_and_shared_hook_teardown_happen_once(monkeypatch):
    transform = Hook("transform", "stream")
    selection = Hook("select", "manifest")
    module = Module("qc", [{"id": "a"}], [qc(), selection])
    ordinary = Module("normal", [{"id": "b"}], [transform])

    def consume(modules, *, global_hooks, **kw):
        hooks = [*global_hooks, *(h for m in modules for h in m.hooks)]
        seen = set()
        for h in hooks:
            if id(h) not in seen and hasattr(h, "teardown"):
                h.teardown()
                seen.add(id(h))
        return []

    monkeypatch.setattr(pipeline, "run_fetchez", consume)
    pipeline.StackQCPlan(
        config={}, modules=[module, ordinary], global_hooks=[selection, transform]
    ).execute()
    assert selection.closed == transform.closed == 1


@pytest.mark.parametrize("qc_res", [None, 1.0])
def test_accepted_qc_sources_reach_optional_coverage_consumers(
    tmp_path, monkeypatch, qc_res
):
    """Coverage must match accepted source counts on the final native grid."""
    import numpy as np
    import rasterio
    from test_stack_qc_native import setup, run_plan
    from globato.hooks.metadata.provenance import ProvenanceHook, SourceMasks
    from globato.hooks.metadata.stack_provenance import StackProvenance

    monkeypatch.chdir(tmp_path)
    mod, owner, hooks = setup(tmp_path)
    owner.qc_res = qc_res
    module_mask = ProvenanceHook(res=3, output=str(tmp_path / "modules.tif"))
    source_masks = SourceMasks(
        res=3,
        output=str(tmp_path / "sources.vrt"),
        output_dir=str(tmp_path / "masks"),
        resume=True,
    )
    # Masks describe accepted observations before native strategy reduction.
    hooks[0:0] = [module_mask, source_masks]
    surviving = StackProvenance(output=str(tmp_path / "surviving.vrt"), resume=True)
    hooks.insert(3, surviving)  # Observe the same accepted stream after multi_stack.
    run_plan([mod], global_hooks=hooks)
    stack = next(h for h in hooks if h.name == "multi_stack")
    with rasterio.open(stack.output) as ds:
        expected = ds.read(2) > 0
        target = (ds.shape, ds.transform, ds.crs)
    assert expected.any()
    with rasterio.open(module_mask.output) as ds:
        assert (ds.shape, ds.transform, ds.crs) == target
        np.testing.assert_array_equal(ds.read(1) > 0, expected)
        assert ds.tags(1)["MOD_mbdb"] == "1"
    assert len(source_masks._masks.masks) == 1
    with rasterio.open(source_masks.output) as ds:
        assert ds.count == 1 and (ds.shape, ds.transform, ds.crs) == target
        np.testing.assert_array_equal(ds.read(1) > 0, expected)
    with rasterio.open(surviving.output) as ds:
        np.testing.assert_array_equal(ds.read(1) > 0, expected)
    assert surviving.resume is source_masks.resume is True  # Configuration restored.
