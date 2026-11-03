"""One viz flag publishes the final build without another recipe hook."""

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from fetchez.core import prepare_final_hooks
from fetchez.recipe import Recipe, RecipeContext

from globato.hooks.filters import stack_qc
from globato.hooks.filters.stack_qc import StackQC
from globato.hooks.filters.stack_qc_review import StackQCReview
from test_stack_qc_hook import build


@pytest.fixture(autouse=True)
def clear_results():
    stack_qc.RESULTS.clear()
    yield
    stack_qc.RESULTS.clear()


def test_viz_flag_publishes_catalog_after_native_dem_without_changing_science(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    without = build(tmp_path, tag="off")
    assert not list(tmp_path.glob("off/stack_qc_review/*/products.json"))
    automatic = build(tmp_path, tag="on", qc_args=dict(viz=True))
    (catalog,) = tmp_path.glob("on/stack_qc_review/*/products.json")
    doc = json.loads(catalog.read_text())
    assert {p["stage"] for p in doc["products"]} == {
        "before_measured",
        "after_measured",
        "before_interpolated",
        "after_interpolated",
    }
    assert (catalog.parent / "review.html").is_file()
    assert not (catalog.parent / "review_ERROR.txt").exists()
    before = Path(without[0][1]["dst_fn"])
    after = Path(automatic[0][1]["dst_fn"])
    with rasterio.open(before) as a, rasterio.open(after) as b:
        assert a.transform == b.transform and a.crs == b.crs
        np.testing.assert_array_equal(a.read(), b.read())
        np.testing.assert_array_equal(a.read_masks(), b.read_masks())
    final = next(p for p in doc["products"] if p["stage"] == "after_interpolated")
    with rasterio.open(catalog.parent / final["path"]) as a, rasterio.open(after) as b:
        np.testing.assert_array_equal(a.read(), b.read())


@pytest.mark.parametrize("viz", [False, True])
def test_compiled_recipe_orders_automatic_review_after_final_crop(tmp_path, viz):
    recipe = Recipe.from_dict(
        dict(
            region=[0, 1, 0, 1],
            modules=[
                dict(module="scratch", args=dict(path=str(tmp_path / "points.xyz")))
            ],
            modifiers=[
                dict(
                    name="buffer-and-cut",
                    args=dict(buffer_cells=4, inc="1s", outname="dem"),
                )
            ],
            global_hooks=[
                dict(name="stack_qc", args=dict(viz=viz)),
                dict(name="ms_binary_cudem", args=dict(output="dem.tif")),
            ],
        )
    )
    plan = recipe.compile(
        RecipeContext(name="test", region_srs="EPSG:4326", tile_dir=tmp_path)
    )
    names = [h.name.replace("-", "_") for h in plan.global_hooks]
    if viz:
        assert names[-3:] == ["raster_cut", "raster_crop", "stack_qc_review"]
        assert len(prepare_final_hooks(plan.modules, plan.global_hooks)) == len(
            plan.global_hooks
        )
    else:
        assert names[-2:] == ["raster_cut", "raster_crop"]
    assert "stack_qc_review" not in [h["name"] for h in recipe.config["global_hooks"]]


def test_explicit_review_configuration_is_preserved_once(tmp_path):
    qc = StackQC(viz=True)
    explicit = StackQCReview(dem="custom.tif", output=str(tmp_path))
    prepared = prepare_final_hooks([], [qc, explicit])
    assert prepared == [qc, explicit]
    assert prepared[-1].dem == "custom.tif"


def test_automatic_publishers_are_scoped_to_their_qc_results():
    first, second = StackQC(viz=True), StackQC(viz=True)
    prepared = prepare_final_hooks([], [first, second])
    assert prepared[-2]._qc_hooks[0] is first
    assert prepared[-1]._qc_hooks[0] is second
    assert StackQC(viz=False).final_hooks() == ()
