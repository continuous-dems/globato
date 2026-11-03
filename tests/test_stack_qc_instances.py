"""Recipe instance selection and coverage must preserve datatype/survey identity."""

from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import rasterio
from fetchez.spatial import Region
from globato.hooks.filters.stack_qc import _selected
from globato.hooks.metadata.provenance import ProvenanceHook, source_tags
from globato.hooks.transforms.point_pixels import FUSION_BANDS, FUSION_STATE_VERSION


def owner(name, identity, **attrs):
    return SimpleNamespace(
        name=name,
        module_id=lambda: identity,
        region=Region(0, 4, 0, 4, srs="EPSG:4326"),
        weight=1.0,
        **attrs,
    )


def test_recipe_selectors_split_nos_types_and_dav_surveys():
    modules = [
        owner("nos_hydro", "xyz", datatype="xyz"),
        owner("nos_hydro", "bag", datatype="bag"),
        owner("dav", "lidar1", survey_id=4912),
        owner("dav", "lidar2", survey_id=10247),
    ]
    for name, args, expected in [
        ("nos_hydro", {"datatype": "xyz"}, [True, False, False, False]),
        ("nos_hydro", {"datatype": "bag"}, [False, True, False, False]),
        ("dav", {"survey_id": "4912"}, [False, False, True, False]),
    ]:
        assert [
            _selected(mod, [dict(module=name, args=args)]) for mod in modules
        ] == expected


def native(path, region, column):
    data = np.zeros((7, 4, 4), dtype="float64")
    data[0, :, column] = -100
    data[1:3, :, column] = 1
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=7,
        dtype="float64",
        crs="EPSG:4326",
        transform=rasterio.transform.from_origin(0, 4, 1, 1),
    ) as dst:
        dst.write(data)
        dst.descriptions = tuple(FUSION_BANDS)
        dst.update_tags(
            GLOBATO_DATATYPE="FUSION_STATE",
            GLOBATO_FUSION_VERSION=str(FUSION_STATE_VERSION),
        )


@pytest.mark.parametrize("reverse", [False, True])
def test_instance_provenance_has_distinct_deterministic_bits(tmp_path, reverse):
    mods = [
        owner("nos_hydro", "b", datatype="xyz"),
        owner("nos_hydro", "a", datatype="bag"),
        owner("dav", "d", survey_id=4912),
        owner("dav", "c", survey_id=10247),
    ]
    entries = []
    for column, mod in enumerate(mods):
        path = tmp_path / f"{column}.tif"
        native(path, mod.region, column)
        entries.append(
            (
                mod,
                dict(
                    artifacts={"qc-side-stack": str(path)}, url=f"https://test/{column}"
                ),
            )
        )
    if reverse:
        entries.reverse()
    out = tmp_path / "instances.tif"
    hook = ProvenanceHook(
        res="1", output=str(out), source_artifact="qc-side-stack", group_by="module_id"
    )
    hook.run(entries)
    hook.teardown()
    legend = json.loads(Path(str(out) + ".json").read_text())
    assert len(legend["groups"]) == 4
    bits = {g["module_id"]: g["bit"] for g in legend["groups"]}
    assert bits == {"a": 1, "b": 2, "c": 4, "d": 8}
    with rasterio.open(out) as ds:
        for column, mod in enumerate(mods):
            assert np.all(ds.read(1)[:, column] == bits[mod.module_id()])
        assert json.loads(ds.tags(1)["GROUPS"]) == legend
    labels = {
        (g["module"], g.get("datatype"), g.get("survey_id")) for g in legend["groups"]
    }
    assert ("nos_hydro", "bag", None) in labels and ("dav", None, "4912") in labels
    assert source_tags(mods[0], {})["MODULE_DATATYPE"] == "xyz"


def test_instance_provenance_rejects_stream_mode_and_overflow(tmp_path):
    with pytest.raises(ValueError, match="collection-stage"):
        ProvenanceHook(group_by="module_id")
    hook = ProvenanceHook(
        group_by="module_id",
        source_artifact="side-stack",
        output=str(tmp_path / "x.tif"),
    )
    with pytest.raises(ValueError, match="32 module"):
        hook.run([(owner("dav", str(i), survey_id=i), {}) for i in range(33)])


def test_recipe_entry_ids_are_unique_without_changing_provider_cache(
    tmp_path, monkeypatch
):
    from fetchez.recipe import Recipe

    monkeypatch.chdir(tmp_path)
    recipe = Recipe(dict(modules=[]))
    definitions = [
        dict(id="xyz_input", module="nos_hydro", args={"datatype": "xyz"}),
        dict(id="xyz_reference", module="nos_hydro", args={"datatype": "xyz"}),
        dict(id="bag_reference", module="nos_hydro", args={"datatype": "bag"}),
    ]
    mods = recipe._init_modules(definitions)
    assert [m.recipe_id for m in mods] == [
        "xyz_input",
        "xyz_reference",
        "bag_reference",
    ]
    assert mods[0].module_id() == mods[1].module_id()
    assert [_selected(m, ["xyz_input"]) for m in mods] == [True, False, False]
    assert [_selected(m, [{"id": "xyz_reference"}]) for m in mods] == [
        False,
        True,
        False,
    ]
    assert [
        _selected(m, [{"module": "nos_hydro", "args": {"datatype": "xyz"}}])
        for m in mods
    ] == [True, True, False]
    with pytest.raises(ValueError, match="Duplicate module entry id"):
        recipe._init_modules([definitions[0], definitions[0]])
    auto = recipe._init_modules([dict(module="nos_hydro"), dict(module="nos_hydro")])
    assert [m.recipe_id for m in auto] == ["entry_1", "entry_2"]


def test_recipe_id_takes_priority_over_a_provider_shorthand():
    a = owner("nos_hydro", "a", datatype="xyz", recipe_id="nos_hydro")
    b = owner("nos_hydro", "b", datatype="bag", recipe_id="bag")
    a._recipe_ids = b._recipe_ids = frozenset(["nos_hydro", "bag"])
    assert _selected(a, ["nos_hydro"]) and not _selected(b, ["nos_hydro"])
    assert _selected(a, [{"module": "nos_hydro"}]) and _selected(
        b, [{"module": "nos_hydro"}]
    )


def test_recipe_provenance_separates_identical_provider_configurations(tmp_path):
    mods = [
        owner("nos_hydro", "same-config", datatype="xyz", recipe_id="one"),
        owner("nos_hydro", "same-config", datatype="xyz", recipe_id="two"),
    ]
    entries = []
    for column, mod in enumerate(mods):
        path = tmp_path / f"{column}.tif"
        native(path, mod.region, column)
        entries.append(
            (
                mod,
                dict(
                    artifacts={"qc-side-stack": str(path)}, url=f"https://test/{column}"
                ),
            )
        )
    out = tmp_path / "recipe_entries.tif"
    hook = ProvenanceHook(
        res="1", output=str(out), source_artifact="qc-side-stack", group_by="recipe_id"
    )
    hook.run(entries)
    hook.teardown()
    legend = json.loads(Path(str(out) + ".json").read_text())
    assert {g["recipe_id"] for g in legend["groups"]} == {"one", "two"}
    with rasterio.open(out) as ds:
        assert set(np.unique(ds.read(1))) == {0, 1, 2}


def test_identical_recipe_entries_can_have_different_qc_roles(tmp_path, monkeypatch):
    from fetchez.recipe import Recipe
    from fetchez.core import run_fetchez
    from test_stack_qc_hook import FixtureMB, points, INTERPOLATION
    from globato.hooks.filters.stack_qc import StackQC, StackQCGrid
    from globato.hooks.sinks.side_stack import SideStackGenerate
    from globato.qc.stack import io

    monkeypatch.chdir(tmp_path)
    region = Region(350000, 350192, 3749808, 3750000, srs="EPSG:32611")
    path = points(tmp_path)
    mods = Recipe(dict(modules=[]))._init_modules(
        [
            dict(
                id="target_copy",
                module=FixtureMB,
                args=dict(path=str(path), outdir=str(tmp_path / "sources")),
            ),
            dict(
                id="reference_copy",
                module=FixtureMB,
                args=dict(path=str(path), outdir=str(tmp_path / "sources")),
            ),
        ],
        target_region=region,
        global_region_srs="EPSG:32611",
    )
    assert mods[0].module_id() == mods[1].module_id()
    for mod in mods:
        mod.run()
    shared = dict(
        modules=["target_copy"],
        add_ref_modules=["reference_copy"],
        res="3",
        crs="EPSG:32611",
    )
    run_fetchez(
        mods,
        threads=1,
        ignore_failures=False,
        global_hooks=[
            StackQCGrid(**shared),
            SideStackGenerate(res="3", crs="EPSG:32611"),
            StackQC(
                **shared,
                max_passes=5,
                stop_when_stable=False,
                terrain_restoration=2,
                interpolation=INTERPOLATION,
                cache_dir=str(tmp_path / "qc"),
                workers=1,
            ),
        ],
    )
    inv = io.read(next((tmp_path / "qc").glob("*/SOURCES.json")))
    assert inv["sources"][0]["recipe_id"] == "target_copy"
    assert inv["reference_sources"][0]["recipe_id"] == "reference_copy"
    assert inv["sources"][0]["source_id"] != inv["reference_sources"][0]["source_id"]
    assert {r["group"] for r in [*inv["sources"], *inv["reference_sources"]]} == {
        "synthetic-acquisition"
    }
    receipt = io.read(next((tmp_path / "qc").glob("*/contributions_*.json")))
    assert {(r["recipe_id"], r["role"]) for r in receipt["sources"]} == {
        ("target_copy", "filter"),
        ("reference_copy", "reference"),
    }


def test_dav_class_branches_reuse_native_contribution_identity(tmp_path, monkeypatch):
    from fetchez.recipe import Recipe
    from fetchez.core import run_fetchez
    from fetchez.entry import dataset_id
    from test_stack_qc_hook import FixtureMB, INTERPOLATION
    from globato.hooks.filters.dropclass import DropClass
    from globato.hooks.filters.stack_qc import StackQC, StackQCGrid
    from globato.hooks.sinks.side_stack import SideStackGenerate
    from globato.qc.stack import io

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "same_lidar.xyz"
    path.write_text("Prepared fixture stream; file ownership retained")
    region = Region(350000, 350024, 3749976, 3750000, srs="EPSG:32611")
    definitions = [
        dict(
            id="ground",
            module=FixtureMB,
            args=dict(path=str(path)),
            hooks=[
                dict(
                    name=DropClass,
                    args=dict(classes="2", invert=True, stack_partition="ground"),
                )
            ],
        ),
        dict(
            id="bottom",
            module=FixtureMB,
            args=dict(path=str(path)),
            hooks=[
                dict(
                    name=DropClass,
                    args=dict(classes="29", invert=True, stack_partition="bottom"),
                )
            ],
        ),
    ]
    mods = Recipe({"modules": []})._init_modules(
        definitions, target_region=region, global_region_srs="EPSG:32611"
    )
    y, x = np.indices((8, 8))
    data = np.zeros(
        64,
        dtype=[
            ("x", "f8"),
            ("y", "f8"),
            ("z", "f8"),
            ("w", "f8"),
            ("u", "f8"),
            ("classification", "u1"),
            ("confidence", "i2"),
        ],
    )
    data["x"] = (350000 + x * 3 + 1.5).ravel()
    data["y"] = (3750000 - y * 3 - 1.5).ravel()
    data["z"] = -100
    data["w"] = 1
    data["classification"] = np.where(y.ravel() % 2, 29, 2)
    for mod in mods:
        mod.run()
        mod.results[0].update(
            stream=iter([data.copy()]),
            stream_type="point-stream",
            contribution_id={"reader_profile": "original"},
        )
    config = dict(
        modules=["ground"], add_ref_modules=["bottom"], res="3", crs="EPSG:32611"
    )
    entries = run_fetchez(
        mods,
        threads=1,
        ignore_failures=False,
        global_hooks=[
            StackQCGrid(**config),
            SideStackGenerate(res="3", crs="EPSG:32611"),
            StackQC(
                **config,
                max_passes=5,
                stop_when_stable=False,
                terrain_restoration=2,
                interpolation=INTERPOLATION,
                cache_dir=str(tmp_path / "qc"),
                workers=1,
            ),
        ],
    )
    assert {entry["contribution_id"]["recipe"] for _, entry in entries} == {
        "ground",
        "bottom",
    }
    assert len({dataset_id(entry) for _, entry in entries}) == 2
    assert all(
        entry["contribution_id"]["reader_profile"] == "original" for _, entry in entries
    )
    inv = io.read(next((tmp_path / "qc").glob("*/SOURCES.json")))
    for rec in [*inv["sources"], *inv["reference_sources"]]:
        with rasterio.open(rec["path"]) as ds:
            assert ds.read(2).sum() == 32
    assert (
        inv["sources"][0]["recipe_id"] == "ground"
        and inv["reference_sources"][0]["recipe_id"] == "bottom"
    )


def test_unknown_recipe_labels_fail_before_streams_are_consumed():
    from globato.hooks.filters.stack_qc import StackQC, StackQCGrid

    mod = owner("nos_hydro", "config", datatype="xyz", recipe_id="xyz")
    mod._recipe_ids = frozenset(["xyz"])
    mod._recipe_providers = frozenset(["nos_hydro"])
    for hook in [StackQC(modules=["typo"]), StackQCGrid(modules=["typo"])]:
        with pytest.raises(ValueError, match="Unknown QC recipe"):
            hook.run([(mod, {})])


def test_generic_catalog_keeps_entry_labels_and_measured_pass_semantics():
    from globato.qc.stack.review.catalog import document

    stages = [
        dict(
            role="entry_before_measured",
            path="before.tif",
            label="nos_xyz before",
            entry_id="nos_xyz",
            interpolation=False,
            scope="recipe-entry",
        ),
        dict(
            role="entry_after_measured",
            path="after.tif",
            label="nos_xyz after",
            entry_id="nos_xyz",
            interpolation=False,
            scope="recipe-entry",
        ),
        dict(
            role="pass_2_measured",
            path="pass2.tif",
            label="After pass 2 measured",
            interpolation=False,
            scope="qc-context",
        ),
        dict(
            role="pass_2_review",
            path="pass2.png",
            label="Figure",
            interpolation=False,
            scope="review",
        ),
        dict(
            role="entry_before_hillshade",
            path="shade.tif",
            label="RGB hillshade",
            interpolation=False,
            scope="recipe-entry",
        ),
    ]
    result = document(
        stages, title="QC study", method="connectivity", preferred_entry="nos_xyz"
    )
    assert result["schema_version"] == 1 and len(result["products"]) == 3
    assert result["defaults"]["default"] == [
        result["products"][0]["key"],
        result["products"][1]["key"],
    ]
    assert all(p["metadata"]["interpolation"] is False for p in result["products"])
    assert result["stages"] == stages


def test_source_masks_reuse_contribution_id_for_same_file_views(tmp_path):
    from globato.hooks.metadata.provenance import SourceMasks

    mods = [
        owner("dav", "same", recipe_id="ground", survey_id=4912),
        owner("dav", "same", recipe_id="bottom", survey_id=4912),
    ]
    entries = []
    for column, mod in enumerate(mods):
        path = tmp_path / f"state{column}.tif"
        native(path, mod.region, column)
        entries.append(
            (
                mod,
                dict(
                    url="https://test/same.laz",
                    dst_fn="same.laz",
                    contribution_id={"recipe": mod.recipe_id},
                    artifacts={"qc-side-stack": str(path)},
                ),
            )
        )
    output = tmp_path / "sources.vrt"
    hook = SourceMasks(
        res="1",
        output=str(output),
        output_dir=str(tmp_path / "masks"),
        source_artifact="qc-side-stack",
        resume=False,
        qgis_style=False,
    )
    hook.run(entries)
    hook.teardown()
    assert len({entry["artifacts"]["source-masks"] for _, entry in entries}) == 2
    with rasterio.open(output) as ds:
        assert ds.count == 2 and len(set(ds.descriptions)) == 2
        assert {ds.tags(i)["MODULE_RECIPE_ID"] for i in [1, 2]} == {"ground", "bottom"}
