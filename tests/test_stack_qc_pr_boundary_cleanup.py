"""Protect ordinary Globato behavior and Stack QC's temporary integration state.

These tests exercise the actual production entry points with in-memory objects;
no provider APIs, local surveys, or network resources are required.
"""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from fetchez.recipe import Recipe as FetchezRecipe, PipelinePlan
from globato.qc.stack import pipeline
from globato.cli import run as run_module


class Hook:
    def __init__(self, name, stage):
        self.name, self.stage = name, stage
        self.teardowns = 0

    def teardown(self):
        self.teardowns += 1


class Module:
    def __init__(self, qc=False):
        self.name = "measured" if qc else "ordinary"
        self.internal_hooks = []
        self.external_hooks = [Hook("stack_qc", "collection")] if qc else []
        self.results = [{"id": self.name}]

    @property
    def hooks(self):
        return self.internal_hooks + self.external_hooks

    def run(self):
        raise AssertionError("No data discovery should occur in this test")


@pytest.mark.parametrize("with_qc", [False, True])
def test_stock_recipe_dispatch_is_a_true_noop_without_stack_qc(monkeypatch, with_qc):
    """No QC: preserve the stock compiled plan object, hooks and options."""
    original = PipelinePlan(config={}, modules=[Module(qc=with_qc)], global_hooks=[])
    monkeypatch.setattr(FetchezRecipe, "compile", lambda self, *a, **k: original)
    recipe = object.__new__(pipeline.StackQCRecipe)
    result = recipe.compile()
    if with_qc:
        assert type(result) is pipeline.StackQCPlan
        assert result is not original
    else:
        assert result is original
        assert type(result) is PipelinePlan
    assert len(original.global_hooks) == 0
    assert len(original.modules) == 1


@pytest.mark.parametrize("native_fails", [False, True])
def test_transient_review_publication_does_not_pollute_reusable_plan(
    monkeypatch, native_fails
):
    """Review publishers and rewritten global hook state must never leak."""
    normal = Hook("multi_stack", "stream")
    published = Hook("stack_qc_review", "collection")
    plan = pipeline.StackQCPlan(config={}, modules=[], global_hooks=[normal])

    def fake_build():
        plan.global_hooks.append(published)
        if native_fails:
            raise RuntimeError("simulated preparation failure")
        plan._native_started = True
        plan._native_hook_ids = {id(normal), id(published)}
        return "completed"

    monkeypatch.setattr(plan, "_execute_prepared", fake_build)
    if native_fails:
        with pytest.raises(RuntimeError, match="simulated preparation failure"):
            plan.execute()
        assert normal.teardowns == published.teardowns == 1
    else:
        assert plan.execute() == "completed"
    assert plan.global_hooks == [normal]


def test_registered_recipe_cli_overrides_do_not_modify_registry(monkeypatch):
    """A registered recipe can be run again without inheriting the prior region."""
    config = dict(
        region="old",
        region_srs="EPSG:4326",
        global_hooks=[dict(name="multi_stack", args={"region": "old"})],
        modules=[],
    )
    pristine = deepcopy(config)
    captures = []

    class Registry:
        @staticmethod
        def load_all():
            pass

        @staticmethod
        def get_yaml(name):
            return {"config": config}

    class TestRecipe:
        @staticmethod
        def from_dict(value):
            captures.append(deepcopy(value))
            return SimpleNamespace(run=lambda **kwargs: iter(()))

    monkeypatch.setattr(run_module, "RecipeRegistry", Registry)
    monkeypatch.setattr(run_module, "Recipe", TestRecipe)
    kwargs = dict(
        name="registered_test",
        region="1/2/3/4",
        region_srs="EPSG:4269",
        outdir=None,
        shared_cache=None,
        modifier=None,
        schema=None,
        refresh=False,
        fail_fast=True,
    )
    run_module.run_recipe.callback(**kwargs)
    assert config == pristine
    assert captures[0]["region"] == "1/2/3/4"
    assert captures[0]["global_hooks"][0]["args"]["region"] == "1/2/3/4"
    kwargs["region"] = None
    run_module.run_recipe.callback(**kwargs)
    assert captures[1] == pristine
    assert config == pristine
