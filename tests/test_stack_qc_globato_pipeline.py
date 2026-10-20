"""Globato-only contract: no Fetchez preparation extension is required."""

import importlib.util
import sys
import types
from pathlib import Path

SRC = Path(__file__).parents[1] / "src" / "globato" / "qc" / "stack" / "pipeline.py"


def _load(monkeypatch):
    # Import real Fetchez Recipe/PipelinePlan when available, while keeping the
    # isolated test independent of optional GIS/transformez runtime libraries.
    for name in ("globato", "globato.qc", "globato.qc.stack"):
        m = types.ModuleType(name)
        m.__path__ = [str(SRC.parent)]
        monkeypatch.setitem(sys.modules, name, m)
    prep = types.ModuleType("globato.qc.stack.preparation")
    prep.SOURCE_CONSUMERS = {"multi_stack", "provenance", "source_masks"}
    monkeypatch.setitem(sys.modules, prep.__name__, prep)
    name = "globato.qc.stack.pipeline"
    spec = importlib.util.spec_from_file_location(name, SRC)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


class Hook:
    def __init__(self, name, stage, calls):
        self.name, self.stage, self.calls = name, stage, calls

    def run(self, entries):
        self.calls.append((self.name, [(m.name, e["id"]) for m, e in entries]))
        return entries


class QC(Hook):
    def __init__(self, calls):
        super().__init__("stack_qc", "collection", calls)

    def prepare_module(self, module, hooks, *, threads):
        self.calls.append(("prepare", [(module.name, e["id"]) for e in module.results]))
        from globato.qc.stack.pipeline import PreparedSources

        return PreparedSources(
            module,
            [
                dict(e, stream=iter((1,)), stream_type="raster-stream")
                for e in module.results
            ],
        )


class Mod:
    def __init__(self, name, hooks):
        self.name = name
        self.internal_hooks = []
        self.external_hooks = hooks
        self.results = []

    @property
    def hooks(self):
        return self.internal_hooks + self.external_hooks

    def run(self):
        self.results = [{"id": self.name}]


def test_mixed_prepared_and_regular_preserve_selection_and_order(monkeypatch):
    m = _load(monkeypatch)
    calls = []
    qc = QC(calls)
    modules = [
        Mod("qc", [Hook("local-manifest", "manifest", calls), qc]),
        Mod(
            "ordinary",
            [
                Hook("local-manifest", "manifest", calls),
                Hook("local-stream", "stream", calls),
            ],
        ),
    ]
    hooks = [
        Hook("shared-manifest", "manifest", calls),
        Hook("global-file", "file", calls),
        Hook("global-transform", "stream", calls),
        Hook("multi_stack", "stream", calls),
        Hook("last", "collection", calls),
    ]

    def fake_run_fetchez(mods, threads, global_hooks, ignore_failures):
        assert [h.name for h in global_hooks if h.stage == "manifest"] == [
            "stack_qc_delivery_order"
        ]
        assert [type(x).__name__ for x in mods] == ["PreparedSources", "Mod"]
        assert all(not any(h.stage == "manifest" for h in x.hooks) for x in mods)
        for mod in mods:
            for stage in ("file", "stream"):
                for hook in global_hooks:
                    if hook.stage == stage:
                        hook.run([(mod, mod.results[0])])
        return ["finished"]

    monkeypatch.setattr(m, "run_fetchez", fake_run_fetchez)
    p = m.StackQCPlan(config={}, modules=modules, global_hooks=hooks, threads=2)
    assert p.execute() == ["finished"]
    assert sum(x[0] == "prepare" for x in calls) == 1
    assert sum(x[0] == "shared-manifest" for x in calls) == 1
    assert sum(x[0] == "local-manifest" for x in calls) == 2
    assert [z[0] for z in calls if z[0] == "global-file"] == ["global-file"]
    assert [z[0] for z in calls if z[0] == "global-transform"] == ["global-transform"]
    assert len([x for x in calls if x[0] == "multi_stack"]) == 2
    assert len(modules[0].hooks) == 2
    assert len(modules[1].hooks) == 2


def test_prepared_sources_do_not_autoinitialize_metadata(monkeypatch):
    m = _load(monkeypatch)
    original = Mod("measured", [])
    prepared = m.PreparedSources(original, [{"id": "metadata-only"}])
    assert prepared.results[0]["stream_type"] == "raster-stream"
    assert list(prepared.results[0]["stream"]) == []
    assert prepared.fetch_entry(prepared.results[0]) == 0


def test_unmodified_recipe_stays_native(monkeypatch):
    m = _load(monkeypatch)
    ordinary = Mod("ordinary", [])
    from fetchez.recipe import PipelinePlan

    base = PipelinePlan(config={}, modules=[ordinary], global_hooks=[])
    assert not any(m._qc_owner(x) for x in base.modules)


def test_multiple_qc_owners_error(monkeypatch):
    m = _load(monkeypatch)
    module = Mod("duplicate", [QC([]), QC([])])
    import pytest

    with pytest.raises(ValueError, match="multiple Stack QC"):
        m._qc_owner(module)


def test_real_stock_fetchez_streams_only_accepted_contributions(monkeypatch):
    """Exercise real run_fetchez (not a simulated call), with no preparation API."""
    m = _load(monkeypatch)
    calls = []

    class DataHook(Hook):
        def run(self, entries):
            for owner, entry in entries:
                entry["stream"] = iter([7])
                entry["stream_type"] = "raster-stream"
            calls.append(("init", entries[0][0].name))
            return entries

    class TransformHook(Hook):
        def run(self, entries):
            calls.append(("transform", entries[0][0].name))
            return entries

    class StackConsumer(Hook):
        def run(self, entries):
            for owner, entry in entries:
                calls.append(("stack", owner.name, list(entry["stream"])))
                entry["stream"] = iter(())
            return entries

    qc = QC(calls)
    first = Mod("qc", [qc])
    second = Mod("ordinary", [DataHook("native-stream", "stream", calls)])
    # The native FetchModule contract is no-op here because the simulated
    # cached source has already been materialized (no network needed).
    for module in (first, second):
        module.fetch_entry = lambda entry, **kwargs: 0
        module.run = lambda mod=module: setattr(
            mod,
            "results",
            [{"id": mod.name, "url": "local:", "dst_fn": mod.name + ".bin"}],
        )

    # Keep the source-specific accepted stream, native collection consumer and
    # original unmodified Fetchez fetch/stream/collection lifecycle.
    plan = m.StackQCPlan(
        config={},
        modules=[first, second],
        threads=1,
        global_hooks=[
            TransformHook("transform", "stream", calls),
            StackConsumer("multi_stack", "stream", calls),
        ],
    )
    result = plan.execute()
    assert result is not None
    assert ("stack", "qc", [1]) in calls
    assert ("stack", "ordinary", [7]) in calls
    assert ("transform", "ordinary") in calls
    assert ("transform", "qc") not in calls
    assert ("init", "ordinary") in calls
    assert len([x for x in calls if x[0] == "prepare"]) == 1


def test_regular_recipe_compile_returns_the_exact_stock_plan(monkeypatch):
    m = _load(monkeypatch)
    from fetchez.recipe import Recipe, PipelinePlan

    stock = PipelinePlan(config={}, modules=[Mod("ordinary", [])], global_hooks=[])
    monkeypatch.setattr(Recipe, "compile", lambda *a, **k: stock)
    assert m.StackQCRecipe({}).compile() is stock
