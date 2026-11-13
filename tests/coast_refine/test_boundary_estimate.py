"""Native complete-stack zero estimate, optional consumer and exact fallback."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import yaml
from shapely.geometry import box

from globato.coast_refine._io import digest
from globato.coast_refine.artifact import MaskArtifact
from globato.coast_refine.boundary import CoastMaskBoundary
from globato.coast_refine.extension import build, validate
from globato.coast_refine.global_refinement import select_group_barrier
from test_global_refinement import global_recipe, invoke


def boundary_recipe(root, *, interpolation=False):
    recipe, _, config = global_recipe(root, interpolation=interpolation)
    authorities = [
        Path(m["args"]["path"])
        for m in config["modules"]
        if str(m["args"]["path"]).endswith(".tif")
    ]
    with rasterio.open(authorities[0]) as src:
        profile = src.profile.copy()
        cell = src.transform.a
    rows, cols = np.indices((profile["height"], profile["width"]))
    for path in authorities:
        with rasterio.open(path) as src:
            present = src.read(1) != -9999
        values = np.where(present & (cols < 40), 8.0, -9999).astype("float32")
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(values, 1)
    gpd.GeoDataFrame(geometry=[box(40 * cell, -1, 1, 1)], crs=4326).to_file(
        root / "water.gpkg", driver="GPKG"
    )
    reference = root / "ordinary_reference.tif"
    values = ((47.5 - cols) * 0.3).astype("float32")
    values[(cols >= 40) & (cols < 55)] = -9999
    with rasterio.open(reference, "w", **profile) as dst:
        dst.write(values, 1)
    config["modules"].append(
        dict(
            module="local_fs",
            args=dict(path=str(reference), data_type="raster", weight=5),
            hooks=[dict(name="stream-init", args=dict(profile="rio"))],
        )
    )
    for i, hook in enumerate(config["global_hooks"]):
        if hook["name"] == "stack_combine":
            hook["args"].update(strategy="mixed", weight_threshold="1/2/3/4/5")
            if not interpolation:
                config["global_hooks"].insert(
                    i + 1,
                    dict(
                        name="coast_mask_boundary",
                        args=dict(group="authority", max_distance_m=40, min_weight=2),
                    ),
                )
            break
    if interpolation:
        for hook in config["global_hooks"]:
            if hook["name"] == "ms_binary_cudem":
                hook["args"].update(
                    estimate_zero=True, zero_max_distance_m=40, regime_guidance="off"
                )
    recipe.write_text(yaml.safe_dump(config, sort_keys=False))
    return recipe, config


@pytest.fixture
def completed(tmp_path):
    recipe, config = boundary_recipe(tmp_path)
    collection = invoke(tmp_path, recipe)
    index = tmp_path / "out/coast_mask/coast_mask_groups_index.json"
    group = json.loads(index.read_text())["groups"]["authority"]
    estimate = group["estimated_zero"]
    return tmp_path, collection, index, group, estimate


def test_native_estimate_uses_complete_stack_and_preserves_masked_sources(completed):
    root, collection, index, group, estimate = completed
    record = json.loads(Path(estimate["receipt"]).read_text())
    assert record["inputs"]["land_evidence"] == "complete_accepted_unfilled_stack"
    assert record["inputs"]["support_stack_sha256"] == digest(estimate["support_stack"])
    assert record["counts"]["extension_cells"] > 0, record
    assert record["source_masks_changed"] is False
    # An ordinary module, not an authority, supplies the descending reference.
    assert any(
        row["group"] is None and row["original_file"].endswith("ordinary_reference.tif")
        for row in collection["sources"]
    )
    assert all(
        digest(row["accepted_state"]) == row["sha256"] for row in collection["sources"]
    )
    entry = {"artifacts": {"coast_mask_groups_index": str(index)}}
    base = select_group_barrier(entry, "authority", estimate["support_stack"])
    chosen = select_group_barrier(
        entry, "authority", estimate["support_stack"], boundary="estimated_zero"
    )
    assert base != chosen and chosen == estimate["topology"]
    # The default clears a previously selected estimate and restores the base.
    assert select_group_barrier(entry, "authority", estimate["support_stack"]) == base
    assert "coast_refine_barrier_extension" not in entry["artifacts"]
    catalog = json.loads(
        next((root / "out/coast_mask/products").glob("*/products.json")).read_text()
    )
    assert len(catalog["products"]) == 6
    assert catalog["products"][-1]["key"] == "estimated_zero"
    assert catalog["products"][-1]["display"]["visible"] is False
    assert catalog["defaults"] == {"default": ["before", "after"]}
    with rasterio.open(estimate["support_stack"]) as stack:
        from rasterio.features import rasterize

        base_frame = gpd.read_file(base, layer="topology")
        new_frame = gpd.read_file(chosen, layer="topology")

        def water(frame):
            return rasterize(
                [(g, 1) for g in frame.loc[frame["class"] == "ocean"].geometry],
                out_shape=stack.shape,
                transform=stack.transform,
                fill=0,
            ).astype(bool)

        z = stack.read(1, masked=True)
        changed = water(base_frame) & ~water(new_frame)
        assert not np.any(changed & ~np.ma.getmaskarray(z) & (z.data <= 0))


def test_low_weight_evidence_abstains_exactly_and_stale_support_fails(completed):
    _, _, _, group, estimate = completed
    artifact = MaskArtifact(group["manifest"])
    receipt = build(
        artifact,
        40,
        method="slope_smooth",
        support_stack=estimate["support_stack"],
        support_min_weight=100,
        use_stack_for_land=True,
    )
    record = json.loads(receipt.read_text())
    assert record["counts"]["extension_cells"] == 0
    path = validate(artifact, receipt)
    before = gpd.read_file(artifact.products["refined_topology"], layer="topology")
    after = gpd.read_file(path, layer="topology")
    for label in before["class"].unique():
        assert (
            before.loc[before["class"] == label]
            .geometry.union_all()
            .equals(after.loc[after["class"] == label].geometry.union_all())
        )
    with rasterio.open(estimate["support_stack"], "r+") as dst:
        band = dst.read(1)
        band[10, 10] += 1
        dst.write(band, 1)
    with pytest.raises(ValueError, match="support stack changed"):
        validate(artifact, Path(estimate["receipt"]))


def test_registered_native_solver_can_select_estimated_boundary(tmp_path):
    recipe, _ = boundary_recipe(tmp_path, interpolation=True)
    invoke(tmp_path, recipe)
    # Public and authoritative runtimes retain their native solver behavior.
    assert (tmp_path / "out/dem.tif").is_file()
    index = json.loads(
        (tmp_path / "out/coast_mask/coast_mask_groups_index.json").read_text()
    )
    estimate = index["groups"]["authority"]["estimated_zero"]
    assert (
        json.loads(Path(estimate["receipt"]).read_text())["counts"]["extension_cells"]
        > 0
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(max_distance_m=0, min_weight=2),
        dict(max_distance_m=40, min_weight=0),
        dict(max_distance_m=40, min_weight=2, method="fixed"),
    ],
)
def test_boundary_requires_explicit_usable_evidence_policy(kwargs):
    with pytest.raises((ValueError, TypeError)):
        CoastMaskBoundary("test", **kwargs)


def test_bracketed_plus_two_minus_three_linear_profile_crosses_at_twenty_metres():
    from globato.coast_refine.shore_profile import profile_zero

    t = np.arange(-30, 65, 5, dtype=float)
    land = np.where(t < 0, 2 - 0.1 * t, np.nan)
    water = np.where(t >= 50, 2 - 0.1 * t, np.nan)
    distance, kind, _, _ = profile_zero(t, land, water, 2, 40, 30)
    assert kind == "bracketed"
    assert distance == pytest.approx(20, abs=0.02)
