import importlib.metadata

from fetchez.registry import BundleRegistry
from fetchez.utils import compile_sources
from globato.utils import globatize_modules


def _disable_installed_bundle_entry_points(monkeypatch):
    # Reproduce dem-devel's isolated exact-source runtime: the source tree is
    # importable, but installed package entry-point metadata is unavailable.
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda *args, **kwargs: [])
    isolated_registry = {}
    monkeypatch.setattr(BundleRegistry, "get_registry", lambda: isolated_registry)
    assert BundleRegistry.get_yaml("glob-tnm") is None


def test_glob_tnm_public_products_shorthand_reaches_real_compile_and_globatize_path(
    monkeypatch,
):
    _disable_installed_bundle_entry_points(monkeypatch)

    # Exercise the same public-string compilation used by globato.api.build().
    compiled = compile_sources(["glob-tnm:products=1_3as/1m"])
    assert compiled[0]["module"] == "glob-tnm"

    modules = globatize_modules(compiled)
    tnm = [m for m in modules if m.get("module") == "tnm"]
    assert [m["args"]["products"] for m in tnm] == ["1m", "1_3as"]
    # The TNM policy owns canonical entry weights. Module weights remain neutral.
    assert [float(m["args"].get("weight", 1.0)) for m in tnm] == [1.0, 1.0]
    for module in tnm:
        names = [h.get("name") for h in module.get("hooks", [])]
        assert "tnm-policy" in names
        assert "claim-grid-filter" in names


def test_bare_glob_tnm_reaches_real_compile_and_globatize_path_without_entry_points(
    monkeypatch,
):
    _disable_installed_bundle_entry_points(monkeypatch)

    compiled = compile_sources(["glob-tnm"])
    assert compiled[0]["module"] == "glob-tnm"

    modules = globatize_modules(compiled)
    tnm = [m for m in modules if m.get("module") == "tnm"]
    assert [m["args"]["products"] for m in tnm] == [
        "s1m",
        "1m",
        "1_9as",
        "5m",
        "1_3as",
        "1_as",
        "2_as",
    ]


def test_real_module_source_is_not_reinterpreted_as_bundle(monkeypatch):
    _disable_installed_bundle_entry_points(monkeypatch)

    compiled = compile_sources(["tnm:products=1m"])
    modules = globatize_modules(compiled)
    assert len(modules) == 1
    assert modules[0]["module"] == "tnm"
    assert modules[0]["args"]["products"] == "1m"
