from types import SimpleNamespace

import pytest

import numpy as np

from fetchez.spatial import Region
from globato.hooks.metadata.provenance import SourceMasks


def _stream():
    arr = np.zeros(1, dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")])
    arr["x"] = 0.5
    arr["y"] = 0.5
    arr["z"] = 1.0
    yield arr


def test_source_mask_grouping_is_per_entry_opt_in(tmp_path):
    hook = SourceMasks(res=1, output=str(tmp_path / "sources.vrt"))
    mod = SimpleNamespace(
        name="tnm",
        title="TNM",
        meta_category="elevation",
        meta_agency="USGS",
        meta_resolution="1 m",
        weight=1.0,
        region=Region(0, 1, 0, 1),
    )
    ordinary = {
        "stream": _stream(),
        "stream_type": "point-stream",
        "dst_fn": str(tmp_path / "ordinary.tif"),
        "metadata": {"dataset": "ordinary", "weight": 1.0},
    }
    grouped = {
        "stream": _stream(),
        "stream_type": "point-stream",
        "dst_fn": str(tmp_path / "grouped.tif"),
        "metadata": {"dataset": "TNM 1 m: Project", "weight": 5.0},
        "source_mask_group_by": "MODULE/DATASET/WEIGHT",
    }
    hook.run([(mod, ordinary), (mod, grouped)])
    assert len(hook.group_requests) == 1
    assert next(iter(hook.group_requests.values())) == "MODULE/DATASET/WEIGHT"


def test_spatial_claim_global_hook_is_only_added_when_claim_grid_is_present(
    monkeypatch,
):
    import globato.api as api

    captured = []
    failure_modes = []

    class FakeRecipe:
        def __init__(self, config):
            self.config = config

        def run(self, **kwargs):
            captured.append(self.config)
            failure_modes.append(kwargs["ignore_failures"])
            return iter(())

    monkeypatch.setattr(api.HookRegistry, "load_all", lambda: None)
    monkeypatch.setattr(api, "compile_sources", lambda sources: sources)
    monkeypatch.setattr(api.Recipe, "from_dict", lambda config: FakeRecipe(config))

    monkeypatch.setattr(
        api,
        "globatize_modules",
        lambda *args, **kwargs: [{"module": "file", "args": {}, "hooks": []}],
    )
    list(api.build(["file:x"], region="0/1/0/1", increment="1s", export=False))
    names = [hook["name"] for hook in captured[-1]["global_hooks"]]
    assert "spatial-claim" not in names
    assert failure_modes[-1] is True  # Unrelated builds retain their default.

    monkeypatch.setattr(
        api,
        "globatize_modules",
        lambda *args, **kwargs: [
            {
                "module": "tnm",
                "args": {},
                "hooks": [{"name": "claim-grid-filter", "args": {"res": 1}}],
            }
        ],
    )
    list(api.build(["glob-tnm"], region="0/1/0/1", increment="1s", export=False))
    names = [hook["name"] for hook in captured[-1]["global_hooks"]]
    assert names.count("spatial-claim") == 1
    assert failure_modes[-1] is False  # Public default must be strict.


def test_public_claiming_build_propagates_discovery_failure_by_default(
    monkeypatch, tmp_path
):
    import globato.api as api
    from fetchez.recipe import Recipe

    monkeypatch.setattr(api.HookRegistry, "load_all", lambda: None)
    monkeypatch.setattr(api, "compile_sources", lambda sources: sources)
    monkeypatch.setattr(
        api,
        "globatize_modules",
        lambda *args, **kwargs: [
            {"module": "tnm", "hooks": [{"name": "claim-grid-filter"}]}
        ],
    )
    monkeypatch.setattr(Recipe, "_check_integrity", lambda self: None)
    monkeypatch.setattr(Recipe, "_expand_hooks", lambda self, hooks: hooks)
    monkeypatch.setattr(Recipe, "_expand_modules", lambda self, modules: modules)
    monkeypatch.setattr(Recipe, "_init_modifiers", lambda self, values: [])
    monkeypatch.setattr(Recipe, "_init_schemas", lambda self, values: [])
    monkeypatch.setattr(Recipe, "_init_hooks", lambda self, values: [])

    class UnavailableHigherTier:
        name = "tnm"

        def run(self):
            raise RuntimeError("injected strict TNM discovery outage")

    monkeypatch.setattr(
        Recipe, "_init_modules", lambda self, *args, **kwargs: [UnavailableHigherTier()]
    )
    with pytest.raises(RuntimeError, match="injected strict TNM discovery outage"):
        list(
            api.build(
                ["glob-tnm"],
                region="0/1/0/1",
                increment="1s",
                outdir=str(tmp_path),
            )
        )
