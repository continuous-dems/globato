"""Native multi-module global refinement and explicit interpolation selection."""

from pathlib import Path
import json
import copy
import numpy as np
import pytest
import rasterio
import yaml
from click.testing import CliRunner
from globato.cli import cli
from globato.coast_refine._io import digest
from globato.coast_refine.global_refinement import (
    CoastMaskSource,
    CoastMaskRefine,
    select_group_barrier,
)
from test_coast_mask_review import native_recipe


def global_recipe(root, *, interpolation=False, groups=False):
    recipe, terrain, _ = native_recipe(root, interpolation=interpolation)
    obj = yaml.safe_load(recipe.read_text())
    with rasterio.open(terrain) as src:
        original = src.read(1)
        profile = src.profile.copy()
    east = root / "east.tif"
    west = original.copy()
    right = original.copy()
    west[:, 48:] = -9999
    right[:, :48] = -9999
    with rasterio.open(terrain, "w", **profile) as ds:
        ds.write(west, 1)
    with rasterio.open(east, "w", **profile) as ds:
        ds.write(right, 1)
    producer = None
    for module in obj["modules"]:
        for hook in module["hooks"]:
            if hook["name"] == "coast_mask":
                hook.update(
                    name="coast_mask_source",
                    args=dict(group="authority", evidence=True, remove="water"),
                )
                producer = module
            elif hook["name"] == "coast_mask_consumer":
                hook.update(
                    name="coast_mask_source",
                    args=dict(group="authority", remove=hook["args"]["remove"]),
                )
    other = copy.deepcopy(producer)
    other["args"]["path"] = str(east)
    other["hooks"][0]["args"].update(
        group="other" if groups else "authority", remove=None
    )
    obj["modules"].append(other)
    for hook in obj["global_hooks"]:
        if hook["name"] == "coast_mask_finalize":
            hook["name"] = "coast_mask_refine"
            hook["args"].update(
                mask=str(root / "water.gpkg"),
                polygon_class="ocean",
                viz=True,
                output_dir="%tile_dir%/coast_mask",
            )
        if hook["name"] == "ms_binary_cudem":
            hook["args"].update(coast_mask_group="authority")
            hook["args"].pop("barrier", None)
    recipe.write_text(yaml.safe_dump(obj, sort_keys=False))
    return recipe, original, obj


def invoke(root, recipe):
    result = CliRunner().invoke(
        cli,
        [
            "run",
            str(recipe),
            "--outdir",
            str(root / "out"),
            "--shared-cache",
            str(root / "cache"),
            "--refresh",
        ],
    )
    (root / "native_cli.log").write_text(result.output)
    assert result.exit_code == 0, result.output
    indexes = list((root / "out").rglob("accepted_collection.json"))
    assert len(indexes) == 1, result.output
    return json.loads(indexes[0].read_text())


def test_split_authorities_build_one_unfilled_mosaic_and_preserve_evidence_only(
    tmp_path,
):
    recipe, expected, obj = global_recipe(tmp_path)
    original_hashes = {
        Path(m["args"]["path"]): digest(m["args"]["path"]) for m in obj["modules"]
    }
    index = invoke(tmp_path, recipe)
    assert len(index["sources"]) == 5
    assert all(s["contribution_id"] for s in index["sources"])
    assert (
        len(
            {s["native_source_identity"]["native_dataset_id"] for s in index["sources"]}
        )
        == 5
    )
    assert all(
        s["native_source_identity"]["identity_owner"] == "fetchez.entry.dataset_id"
        for s in index["sources"]
    )
    assert sum(s["evidence"] for s in index["sources"]) == 2
    unmasked = [s for s in index["sources"] if s["group"] is None]
    assert (
        len(unmasked) == 1
        and unmasked[0]["lineage_assurance"] == "unmasked_native_entry_only"
    )
    evidence_only = next(
        s for s in index["sources"] if s["evidence"] and s["remove"] is None
    )
    record = json.loads(Path(evidence_only["source_receipt"]).read_text())
    assert record["selection"] == "unmasked_evidence"
    assert evidence_only["sha256"] == record["original_grid_binding"]["source_sha256"]
    artifact = Path(record["mask_manifest"]).parent
    with rasterio.open(artifact / "unmasked_evidence.tif") as ds:
        assert ds.shape == expected.shape
        assert np.array_equal(ds.read(1), expected)
    assert all(digest(p) == h for p, h in original_hashes.items())
    catalog = json.loads(
        next((tmp_path / "out").rglob("coast_mask_groups_index.json")).read_text()
    )
    assert catalog["selected_legacy_barrier_group"] is None
    assert list(catalog["groups"]) == ["authority"]


def test_independent_groups_need_no_barrier_selection_until_interpolation(tmp_path):
    recipe, _, _ = global_recipe(tmp_path, groups=True)
    invoke(tmp_path, recipe)
    index = next((tmp_path / "out").rglob("coast_mask_groups_index.json"))
    obj = json.loads(index.read_text())
    assert set(obj["groups"]) == {"authority", "other"}
    entry = {"artifacts": {"coast_mask_groups_index": str(index)}}
    with pytest.raises(ValueError, match="Unknown/incomplete"):
        select_group_barrier(
            entry, "absent", next((tmp_path / "out").glob("combined*.tif"))
        )
    for group in ("authority", "other"):
        selected = select_group_barrier(
            entry, group, next((tmp_path / "out").glob("combined*.tif"))
        )
        assert Path(selected).name == "refined_topology.gpkg"
    manifest = Path(obj["groups"]["authority"]["manifest"])
    record = json.loads(manifest.read_text())
    record["extra"] = "changed"
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="manifest changed"):
        select_group_barrier(
            entry, "authority", next((tmp_path / "out").glob("combined*.tif"))
        )


def test_global_configuration_cannot_be_put_on_source_declaration():
    with pytest.raises(TypeError):
        CoastMaskSource("a", evidence=True, mask="osm")
    with pytest.raises(ValueError):
        CoastMaskSource("a")
    with pytest.raises(TypeError):
        CoastMaskRefine(binding_group="a")


def test_named_barrier_runs_existing_native_interpolator(tmp_path):
    recipe, _, obj = global_recipe(tmp_path)
    cell = 1 / 32400
    obj["global_hooks"].extend(
        [
            {"name": "focus_sink", "args": {"target": "multi_stack"}},
            {
                "name": "raster_stream",
                "args": {"stream_type": "raster", "stage": "collection"},
            },
            {
                "name": "ms_binary_cudem",
                "args": {
                    "steps": 0,
                    "resolutions": str(cell),
                    "weights": "0",
                    "algos": ["raster_fill"],
                    "blend_dists": "0",
                    "coast_mask_group": "authority",
                    "bathy_max_z": None,
                    "inland_decay_dist": 0,
                    "output": "%tile_dir%/dem.tif",
                },
            },
        ]
    )
    recipe.write_text(yaml.safe_dump(obj, sort_keys=False))
    invoke(tmp_path, recipe)
    assert (tmp_path / "out/dem.tif").is_file(), (
        tmp_path / "native_cli.log"
    ).read_text()
