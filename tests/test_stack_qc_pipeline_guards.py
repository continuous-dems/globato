"""Fail-closed boundaries for the Globato-only preparation adapter."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).parents[1] / "src" / "globato" / "qc" / "stack" / "pipeline.py"


def adapter(monkeypatch):
    for name in ("globato", "globato.qc", "globato.qc.stack"):
        mod = types.ModuleType(name)
        mod.__path__ = [str(SRC.parent)]
        monkeypatch.setitem(sys.modules, name, mod)
    prep = types.ModuleType("globato.qc.stack.preparation")
    prep.SOURCE_CONSUMERS = {"multi_stack"}
    monkeypatch.setitem(sys.modules, prep.__name__, prep)
    spec = importlib.util.spec_from_file_location("globato.qc.stack.pipeline", SRC)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


class Hook:
    def __init__(self, name, stage, result=None, deferred=False):
        self.name, self.stage, self.result = name, stage, result
        self.defer_to_preparation = deferred
        self.calls = 0
        self.closed = 0

    def run(self, entries):
        self.calls += 1
        if callable(self.result):
            return self.result(entries)
        return entries

    def teardown(self):
        self.closed += 1


class Module:
    def __init__(self, name, hooks, ids=("1",)):
        self.name = name
        self.internal_hooks = []
        self.external_hooks = hooks
        self.results = []
        self.ids = ids

    @property
    def hooks(self):
        return self.internal_hooks + self.external_hooks

    def run(self):
        self.results = [{"id": v} for v in self.ids]


class QC(Hook):
    def __init__(self):
        super().__init__("stack_qc", "collection")

    def prepare_module(self, module, hooks, threads):
        raise AssertionError("Preparation must not run after invalid manifests")


def test_deferred_local_manifest_fails_before_preparation(monkeypatch):
    m = adapter(monkeypatch)
    mod = Module("qc", [Hook("deferred", "manifest", deferred=True), QC()])
    plan = m.StackQCPlan(config={}, modules=[mod], global_hooks=[])
    with pytest.raises(ValueError, match="deferred module manifest"):
        plan.execute()
    assert mod.results == []  # Fail before discovery or preparation.


def test_deferred_global_manifest_fails_before_preparation(monkeypatch):
    m = adapter(monkeypatch)
    mod = Module("qc", [QC()])
    hook = Hook("deferred", "manifest", deferred=True)
    plan = m.StackQCPlan(config={}, modules=[mod], global_hooks=[hook])
    with pytest.raises(ValueError, match="deferred global manifest"):
        plan.execute()
    assert hook.calls == 0


def test_global_manifest_preserves_cross_module_order(monkeypatch):
    m = adapter(monkeypatch)
    first = Module("a", [], ["a", "b"])
    second = Module("b", [], ["c"])
    seen = []

    def collect(entries):
        seen.extend((o.name, e["id"]) for o, e in entries)
        return list(reversed(entries))

    hook = Hook("select", "manifest", collect)
    plan = m.StackQCPlan(config={}, modules=[first, second], global_hooks=[hook])
    plan._select_sources()
    assert seen == [("a", "a"), ("a", "b"), ("b", "c")]
    assert [e["id"] for e in first.results] == ["b", "a"]
    assert [e["id"] for e in second.results] == ["c"]
    assert hook.calls == 1


def test_stream_gate_keeps_prepared_positions_and_tears_down(monkeypatch):
    m = adapter(monkeypatch)
    original = Module("qc", [])
    prepared = m.PreparedSources(original, [{"id": "q", "stream": iter(())}])
    ordinary = Module("normal", [])
    hook = Hook(
        "transform", "stream", lambda es: [(o, {**e, "ok": True}) for o, e in es]
    )
    gate = m.StreamGate(hook)
    entries = [
        (prepared, prepared.results[0]),
        (ordinary, {"id": "n"}),
        (prepared, prepared.results[0]),
    ]
    result = gate.run(entries)
    assert [x[0] for x in result] == [prepared, ordinary, prepared]
    assert result[1][1]["ok"] is True
    assert hook.calls == 1
    gate.teardown()
    assert hook.closed == 1


def test_stream_gate_rejects_ambiguous_mixed_cardinality(monkeypatch):
    m = adapter(monkeypatch)
    original = Module("qc", [])
    prepared = m.PreparedSources(original, [{"id": "q", "stream": iter(())}])
    ordinary = Module("normal", [])
    hook = Hook("split", "stream", lambda es: es + es)
    with pytest.raises(ValueError, match="changed the number"):
        m.StreamGate(hook).run(
            [(prepared, prepared.results[0]), (ordinary, {"id": "n"})]
        )


def test_original_manifest_restored_after_native_error(monkeypatch):
    m = adapter(monkeypatch)
    qc = QC()
    qc.prepare_module = lambda mod, hooks, threads: m.PreparedSources(
        mod, [dict(mod.results[0], stream=iter(()))]
    )
    local = Hook("module-manifest", "manifest")
    mod = Module("qc", [local, qc])

    def fail(*args, **kwargs):
        assert all(h.stage != "manifest" for h in mod.hooks)
        raise RuntimeError("native-fail")

    monkeypatch.setattr(m, "run_fetchez", fail)
    with pytest.raises(RuntimeError, match="native-fail"):
        m.StackQCPlan(config={}, modules=[mod], global_hooks=[]).execute()
    assert local in mod.hooks
    assert local.calls == 1
