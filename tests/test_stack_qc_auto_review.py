"""Regression tests for Stack QC review insertion via Fetchez ensure-hook.

The modifier changes the expanded recipe, and the reporter runs at finalize,
not through the removed StackQC.final_hooks()/prepare_final_hooks() API.
No remote data access or actual QC/DEM calculation is needed here.
"""

from copy import deepcopy

import pytest

from fetchez.core import FetchContext, _run_grouped_stage
from fetchez.recipe import Recipe
from fetchez.recipes.modifiers.ensure_hook import EnsureHook


REVIEW_RULE = {
    "name": "ensure-hook",
    "args": {
        "when": {"hook": "stack_qc", "args": {"viz": True}},
        "ensure": {"name": "stack_qc_review", "scope": "global", "position": "append"},
    },
}


def _names(specs):
    return [
        (spec.get("name") if isinstance(spec, dict) else spec.name).replace("-", "_")
        for spec in specs
    ]


def _config(*, viz=True, global_hooks=None, modules=None, modifiers=None):
    return {
        "region": [0, 1, 0, 1],
        "global_hooks": (
            [{"name": "stack_qc", "args": {"viz": viz}}]
            if global_hooks is None
            else global_hooks
        ),
        "modules": [] if modules is None else modules,
        "modifiers": [deepcopy(REVIEW_RULE)] if modifiers is None else modifiers,
    }


def _apply(config, *, times=1):
    result = deepcopy(config)
    rule = EnsureHook(**REVIEW_RULE["args"])
    for _ in range(times):
        result = rule.apply(result)
    return result


@pytest.mark.parametrize("viz", [False, True])
def test_stack_qc_review_injection(viz):
    # An actual compiled plan verifies registration and finalize-stage metadata.
    plan = Recipe.from_dict(_config(viz=viz)).compile()
    reviews = [
        h for h in plan.global_hooks if h.name.replace("-", "_") == "stack_qc_review"
    ]
    assert len(reviews) == int(viz)
    if viz:
        assert reviews[0].stage == "finalize"
        assert plan.global_hooks[-1] is reviews[0]


def test_explicit_review_configuration_is_preserved_once():
    explicit = {
        "name": "stack_qc_review",
        "args": {"dem": "custom.tif", "output": "reports"},
    }
    config = _config(
        global_hooks=[{"name": "stack_qc", "args": {"viz": True}}, explicit]
    )
    resolved = _apply(config)
    reporters = [
        spec for spec in resolved["global_hooks"] if spec["name"] == "stack_qc_review"
    ]
    assert reporters == [explicit]
    assert resolved["global_hooks"][1]["args"] == explicit["args"]
    # The compiled object must retain the user-specified constructor settings.
    plan = Recipe.from_dict(config).compile()
    review = next(h for h in plan.global_hooks if h.name == "stack_qc_review")
    assert review.stage == "finalize"
    assert str(review.dem).endswith("custom.tif")
    assert str(review.output).endswith("reports")


@pytest.mark.parametrize("name", ["stack_qc", "stack-qc"])
def test_module_local_requester_inserts_global_reporter(name):
    config = _config(
        global_hooks=[],
        modules=[
            {
                "module": "synthetic",
                "hooks": [{"name": name, "args": {"viz": True}}],
            }
        ],
    )
    resolved = _apply(config)
    assert _names(resolved["global_hooks"]) == ["stack_qc_review"]
    assert _names(resolved["modules"][0]["hooks"]) == ["stack_qc"]


def test_module_local_viz_false_does_not_insert_reporter():
    config = _config(
        global_hooks=[],
        modules=[
            {
                "module": "synthetic",
                "hooks": [{"name": "stack_qc", "args": {"viz": False}}],
            }
        ],
    )
    assert not _apply(config)["global_hooks"]


def test_ensure_hook_is_idempotent_even_when_declared_twice():
    config = _config(modifiers=[deepcopy(REVIEW_RULE), deepcopy(REVIEW_RULE)])
    resolved = _apply(config, times=2)
    assert _names(resolved["global_hooks"]).count("stack_qc_review") == 1
    assert _names(config["global_hooks"]) == ["stack_qc"]  # original untouched


def test_multiple_viz_requesters_share_one_unscoped_reporter():
    config = _config(
        global_hooks=[
            {"name": "stack_qc", "id": "first", "args": {"viz": True}},
            {"name": "stack_qc", "id": "second", "args": {"viz": True}},
        ]
    )
    assert _names(_apply(config)["global_hooks"]).count("stack_qc_review") == 1


def test_other_visualization_hooks_do_not_trigger_stack_qc_review():
    config = _config(global_hooks=[{"name": "raster_viz", "args": {"viz": True}}])
    assert _names(_apply(config)["global_hooks"]) == ["raster_viz"]


def test_finalize_runs_after_collection_and_receives_its_output():
    # Exercising the native grouped-stage runner catches accidental reporter
    # migration back to collection, without invoking expensive DEM processing.
    events = []
    source_entry = {"dst_fn": "input.xyz", "artifacts": {}}

    class SimpleModule:
        name = "sample"
        hooks = []

    owner = SimpleModule()
    source = [(owner, source_entry)]

    class Hook:
        def __init__(self, name, stage):
            self.name = name
            self.stage = stage

        def run(self, entries):
            events.append(self.name)
            if self.name == "final_dem":
                entries[0][1]["dst_fn"] = "dem.tif"
                entries[0][1]["artifacts"]["stack_qc"] = "qc.json"
            if self.name == "stack_qc_review":
                assert entries[0][1]["dst_fn"] == "dem.tif"
                assert entries[0][1]["artifacts"]["stack_qc"] == "qc.json"
            return entries

        def teardown(self):
            pass

    hooks = [Hook("final_dem", "collection"), Hook("stack_qc_review", "finalize")]
    context = FetchContext()
    collection = _run_grouped_stage(
        "collection",
        [owner],
        hooks,
        source_entries=source,
        context=context,
    )
    finished = _run_grouped_stage(
        "finalize",
        [owner],
        hooks,
        source_entries=source,
        global_entries=collection,
        merge_local=False,
        context=context,
    )
    assert events == ["final_dem", "stack_qc_review"]
    assert finished is collection


def test_final_stage_is_metadata_of_registered_review_hook():
    from globato.hooks.filters.stack_qc_review import StackQCReview
    from globato.hooks.filters.stack_qc import StackQC

    assert StackQCReview.meta_stage == "finalize"
    assert StackQCReview().stage == "finalize"
    assert not hasattr(StackQC, "final_hooks")
