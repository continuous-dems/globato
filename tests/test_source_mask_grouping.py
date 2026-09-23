import json
from pathlib import Path

import numpy as np
import rasterio
import pytest
from rasterio.transform import from_origin

from globato.source_mask_grouping import group_source_mask_files, parse_group_fields


def _mask(path, data, **tags):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(0, data.shape[0], 1, 1),
        nodata=0,
    ) as dst:
        dst.write(data.astype("uint8"), 1)
        dst.update_tags(**tags)


def test_group_fields_are_normalized():
    assert parse_group_fields("module/Dataset,weight") == (
        "MODULE",
        "DATASET",
        "WEIGHT",
    )


def test_group_source_masks_makes_one_project_mask(tmp_path):
    a = np.zeros((4, 4), dtype="uint8")
    b = np.zeros((4, 4), dtype="uint8")
    a[:2, :2] = 1
    b[2:, :2] = 1
    common = {
        "MODULE": "tnm",
        "DATASET": "TNM 1 m: WESM PROJECT A",
        "WEIGHT": "5.0",
        "CATEGORY": "elevation",
        "AGENCY": "USGS",
    }
    _mask(tmp_path / "a_mask.tif", a, **common)
    _mask(tmp_path / "b_mask.tif", b, **common)

    outputs = group_source_mask_files(
        [tmp_path / "a_mask.tif", tmp_path / "b_mask.tif"],
        output_dir=tmp_path / "groups",
    )
    assert len(outputs) == 1
    with rasterio.open(outputs[0]) as src:
        out = src.read(1)
        tags = src.tags()
    assert int(out.sum()) == 8
    assert tags["DATASET"] == "TNM 1 m: WESM PROJECT A"
    assert tags["GROUPED_BY"] == "MODULE/DATASET/WEIGHT"
    assert tags["GROUPED_SOURCE_COUNT"] == "2"
    assert sorted(json.loads(tags["GROUPED_MEMBERS"])) == [
        "../a_mask.tif",
        "../b_mask.tif",
    ]


def test_group_does_not_attribute_union_to_first_source(tmp_path):
    arr = np.ones((2, 2), dtype="uint8")
    _mask(
        tmp_path / "a_mask.tif",
        arr,
        MODULE="tnm",
        DATASET="TNM 1 m: A",
        WEIGHT="5.0",
        AGENCY="USGS",
        URL="https://example.org/a.tif",
        TITLE="a",
        DATE="2020-01-01",
        STATISTICS_VALID_PERCENT="25",
    )
    _mask(
        tmp_path / "b_mask.tif",
        arr,
        MODULE="tnm",
        DATASET="TNM 1 m: A",
        WEIGHT="5.0",
        AGENCY="USGS",
        URL="https://example.org/b.tif",
        TITLE="b",
        DATE="2021-01-01",
        STATISTICS_VALID_PERCENT="75",
    )
    outputs = group_source_mask_files(
        [tmp_path / "a_mask.tif", tmp_path / "b_mask.tif"],
        output_dir=tmp_path / "groups",
    )
    with rasterio.open(outputs[0]) as src:
        tags = src.tags()
    assert tags["DATASET"] == "TNM 1 m: A"
    assert tags["AGENCY"] == "USGS"
    assert tags["GROUPED_SOURCE_COUNT"] == "2"
    for name in ("URL", "TITLE", "DATE", "STATISTICS_VALID_PERCENT"):
        assert name not in tags


def test_different_datasets_stay_separate(tmp_path):
    arr = np.ones((2, 2), dtype="uint8")
    files = []
    for idx, dataset in enumerate(("TNM 1/9 arc-second", "TNM 1/3 arc-second")):
        path = tmp_path / f"{idx}_mask.tif"
        _mask(path, arr, MODULE="tnm", DATASET=dataset, WEIGHT=str(3 - idx * 2))
        files.append(path)
    outputs = group_source_mask_files(files, output_dir=tmp_path / "groups")
    assert len(outputs) == 2


def test_group_names_preserve_complete_identity_without_overwrite(tmp_path):
    """Different metadata must never converge on one grouped raster path."""
    first = np.zeros((2, 2), dtype="uint8")
    second = first.copy()
    first[0, 0] = 1
    second[1, 1] = 1
    prefix = "TNM 1 m: " + "a" * 220
    datasets = (prefix + "_first", prefix + "_second", "A/B", "A_B")
    expected = (first, second, first, second)
    paths = []
    for idx, (dataset, data) in enumerate(zip(datasets, expected)):
        path = tmp_path / f"source_{idx}.tif"
        _mask(path, data, MODULE="tnm", DATASET=dataset, WEIGHT="5.0")
        paths.append(path)

    outputs = group_source_mask_files(paths, output_dir=tmp_path / "groups")
    assert len(outputs) == len(datasets)
    assert len(set(outputs)) == len(datasets)
    assert all(len(path.name) < 255 for path in map(Path, outputs))
    actual = {}
    for path in outputs:
        with rasterio.open(path) as src:
            tags = src.tags()
            actual[tags["DATASET"]] = src.read(1)
            assert tags["GROUPED_SOURCE_COUNT"] == "1"
    assert set(actual) == set(datasets)
    for dataset, data in zip(datasets, expected):
        np.testing.assert_array_equal(actual[dataset], data)
    assert outputs == group_source_mask_files(paths, output_dir=tmp_path / "groups")


def test_group_filename_includes_grouping_fields(tmp_path):
    data = np.ones((2, 2), dtype="uint8")
    source = tmp_path / "source.tif"
    _mask(source, data, MODULE="tnm", DATASET="same", WEIGHT="5.0")
    destination = tmp_path / "groups"
    by_dataset = group_source_mask_files(
        [source], group_by="DATASET", output_dir=destination
    )
    by_module = group_source_mask_files(
        [source], group_by="MODULE", output_dir=destination
    )
    assert by_dataset != by_module
    with rasterio.open(by_dataset[0]) as src:
        assert src.tags()["GROUPED_BY"] == "DATASET"
    with rasterio.open(by_module[0]) as src:
        assert src.tags()["GROUPED_BY"] == "MODULE"


def test_requested_missing_source_mask_is_fatal(tmp_path):
    with pytest.raises(RuntimeError, match="Missing requested source mask"):
        group_source_mask_files([tmp_path / "missing_mask.tif"])


def test_nonempty_source_without_group_fields_is_fatal(tmp_path):
    path = tmp_path / "unlabeled_mask.tif"
    _mask(path, np.ones((2, 2), dtype="uint8"), MODULE="tnm")
    with pytest.raises(RuntimeError, match="missing grouping metadata"):
        group_source_mask_files([path])


def test_grouping_after_source_masks_teardown_skips_removed_empty_mask(tmp_path):
    """Test VRT cleanup and grouping in the same SourceMasks lifecycle."""
    from types import SimpleNamespace

    from fetchez.spatial import Region
    from globato.hooks.metadata.provenance import SourceMasks

    region = Region(0, 2, 0, 2)
    mod = SimpleNamespace(
        name="tnm",
        title="TNM",
        meta_category="elevation",
        meta_agency="USGS",
        meta_resolution="1 m",
        weight=5.0,
        region=region,
    )
    points = np.array([(0.5, 1.5, 10.0)], dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")])

    def entry(name, stream):
        return {
            "stream": stream,
            "stream_type": "point-stream",
            "dst_fn": str(tmp_path / f"{name}.tif"),
            "source_mask_group_by": "MODULE/DATASET/WEIGHT",
            "metadata": {"DATASET": "TNM 1 m: Project A", "WEIGHT": "5.0"},
        }

    hook = SourceMasks(
        res=1, output=str(tmp_path / "sources.vrt"), output_dir=str(tmp_path / "masks")
    )
    rows = [
        (mod, entry("contributing", iter((points,)))),
        (mod, entry("empty", iter(()))),
    ]
    hook.run(rows)
    for _, source in rows:
        list(source["stream"])
    hook.teardown()

    assert len(hook.group_requests) == 2
    assert not list((tmp_path / "masks").glob("empty*_mask.tif"))
    outputs = list((tmp_path / "masks" / "grouped").rglob("*_mask.tif"))
    assert len(outputs) == 1
    with rasterio.open(outputs[0]) as src:
        assert src.read(1).sum() == 1
        assert src.tags()["GROUPED_SOURCE_COUNT"] == "1"


def test_grouping_unions_tiled_masks_in_blocks(tmp_path):
    first = np.zeros((64, 64), dtype="uint8")
    second = first.copy()
    first[:20, :25] = 1
    second[30:, 30:] = 1
    files = []
    for name, data in (("first", first), ("second", second)):
        path = tmp_path / f"{name}_mask.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=64,
            height=64,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=from_origin(0, 64, 1, 1),
            nodata=0,
            tiled=True,
            blockxsize=16,
            blockysize=16,
        ) as dst:
            dst.write(data, 1)
            dst.update_tags(MODULE="tnm", DATASET="TNM 1/3 arc-second", WEIGHT="1.0")
        files.append(path)

    outputs = group_source_mask_files(files, output_dir=tmp_path / "grouped")
    with rasterio.open(outputs[0]) as src:
        np.testing.assert_array_equal(src.read(1), first | second)
