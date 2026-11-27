# tests/metdata/test_provenance.py

import pytest

import numpy as np
import rasterio
from rasterio.windows import Window

from globato.hooks.metadata.provenance import MaskSet
from fetchez.spatial import Region


def test_maskset_valid_masks_are_sorted_by_dataset_id(tmp_path):
    region = Region(0, 4, 0, 4)

    masks = MaskSet(
        region,
        1,
        str(tmp_path / "sources.vrt"),
        output_dir=str(tmp_path / "masks"),
        qgis_style=False,
    )

    registered = {}

    # Deliberately register in a non-sorted order.
    for dataset_id in ("source-c", "source-a", "source-b"):
        path = masks.register(dataset_id)
        registered[dataset_id] = path

        MaskSet.update(
            path,
            Window(0, 0, 1, 1),
            np.array([[True]]),
        )

    valid = masks._valid_masks()

    assert [dataset_id for dataset_id, _ in valid] == [
        "source-a",
        "source-b",
        "source-c",
    ]

    assert dict(valid) == registered


def test_maskset_vrt_band_order_is_deterministic(tmp_path):
    region = Region(0, 4, 0, 4)

    masks = MaskSet(
        region,
        1,
        str(tmp_path / "sources.vrt"),
        output_dir=str(tmp_path / "masks"),
        qgis_style=False,
    )

    for dataset_id in ("source-c", "source-a", "source-b"):
        path = masks.register(
            dataset_id,
            description=dataset_id,
        )
        MaskSet.update(
            path,
            Window(0, 0, 1, 1),
            np.array([[True]]),
        )

    masks.build_vrt()

    with rasterio.open(masks.output) as src:
        assert src.count == 3
        assert src.descriptions == (
            "source-a",
            "source-b",
            "source-c",
        )

        source_ids = [
            src.tags(index)["GLOBATO_SOURCE_ID"] for index in range(1, src.count + 1)
        ]

    assert source_ids == [
        "source-a",
        "source-b",
        "source-c",
    ]


@pytest.mark.parametrize(
    ("field", "values", "expected"),
    [
        (
            "DATASET",
            ["Zulu", "Alpha", "Mike", "Alpha"],
            "Alpha, Mike, Zulu",
        ),
        (
            "PUBLICATION_DATE",
            ["2025-04-01", "2022-01-01", "2024-06-01"],
            "2022-01-01 - 2025-04-01",
        ),
        (
            "URL",
            [
                "https://z.example.test/a",
                "https://a.example.test/b",
                "https://z.example.test/c",
            ],
            "a.example.test, z.example.test",
        ),
    ],
)
def test_maskset_aggregate_values_is_deterministic(field, values, expected):
    import pandas as pd

    forward = pd.Series(values)
    reverse = pd.Series(list(reversed(values)))

    assert MaskSet._aggregate_values(forward, field) == expected
    assert MaskSet._aggregate_values(reverse, field) == expected


def _build_grouped_provenance(tmp_path, registration_order):
    import geopandas as gpd

    region = Region(0, 4, 0, 4)

    masks = MaskSet(
        region,
        1,
        str(tmp_path / "sources.vrt"),
        output_dir=str(tmp_path / "masks"),
        vector_output=str(tmp_path / "sources.gpkg"),
        group_by="MODULE/WEIGHT",
        qgis_style=False,
    )

    metadata = {
        "source-a": {
            "MODULE": "tnm",
            "WEIGHT": "1.0",
            "DATASET": "Alpha",
            "URL": "https://z.example.test/a",
        },
        "source-b": {
            "MODULE": "tnm",
            "WEIGHT": "1.0",
            "DATASET": "Beta",
            "URL": "https://a.example.test/b",
        },
        "source-c": {
            "MODULE": "copernicus",
            "WEIGHT": "0.5",
            "DATASET": "Copernicus",
            "URL": "https://copernicus.example.test/c",
        },
    }

    windows = {
        "source-a": Window(0, 0, 1, 1),
        "source-b": Window(1, 0, 1, 1),
        "source-c": Window(2, 0, 1, 1),
    }

    for dataset_id in registration_order:
        path = masks.register(
            dataset_id,
            description=dataset_id,
            tags=metadata[dataset_id],
        )

        MaskSet.update(
            path,
            windows[dataset_id],
            np.array([[True]]),
        )

    masks.finalize()

    return gpd.read_file(masks.vector_output)


def test_grouped_vector_output_is_independent_of_registration_order(tmp_path):
    forward = _build_grouped_provenance(
        tmp_path / "forward",
        ["source-a", "source-b", "source-c"],
    )

    reverse = _build_grouped_provenance(
        tmp_path / "reverse",
        ["source-c", "source-b", "source-a"],
    )

    assert forward["GROUP_ID"].tolist() == reverse["GROUP_ID"].tolist()

    assert forward["GROUP_ID"].tolist() == [
        "copernicus | 0.5",
        "tnm | 1.0",
    ]

    # Compare stable semantic metadata, not raw GPKG bytes.
    columns = [
        "GROUP_ID",
        "SOURCE_COUNT",
        "MODULE",
        "WEIGHT",
        "DATASET",
        "URL",
    ]

    assert forward[columns].to_dict("records") == reverse[columns].to_dict("records")

    # TNM metadata should itself be ordered deterministically.
    tnm = forward.loc[forward["GROUP_ID"] == "tnm | 1.0"].iloc[0]

    assert tnm["SOURCE_COUNT"] == 2
    assert tnm["DATASET"] == "Alpha, Beta"
    assert tnm["URL"] == "a.example.test, z.example.test"


@pytest.mark.parametrize("consumer", ["provenance", "source_masks"])
def test_fusion_coverage_retains_stream_and_horizontal_compound_crs(tmp_path, consumer):
    from types import SimpleNamespace
    from globato.hooks.metadata.provenance import ProvenanceHook, SourceMasks
    from rasterio.transform import from_origin

    region = Region(-118, -118 + 4 / 3600, 34 - 4 / 3600, 34, srs="EPSG:4269")
    profile = dict(
        width=4,
        height=4,
        count=7,
        crs="EPSG:4269+5703",
        transform=from_origin(-118, 34, 1 / 3600, 1 / 3600),
    )
    data = np.zeros((7, 2, 2))
    data[1] = [[1, 0], [0, 3]]
    chunk = (
        Window(1, 1, 2, 2),
        Window(1, 1, 2, 2),
        data,
        None,
        rasterio.windows.transform(Window(1, 1, 2, 2), profile["transform"]),
    )
    hook = (
        ProvenanceHook(res="1s", output=str(tmp_path / "modules.tif"))
        if consumer == "provenance"
        else SourceMasks(res="1s", output=str(tmp_path / "sources.vrt"))
    )
    entry = dict(
        stream=iter([profile, chunk]),
        stream_type="raster-stream",
        data_type="fusion-state",
        _stack_qc_source_id="accepted-source",
    )
    hook.run([(SimpleNamespace(region=region, name="fixture"), entry)])
    emitted = list(entry["stream"])
    assert emitted[0] is profile and emitted[1] is chunk
    hook.teardown()
    with rasterio.open(hook.output) as ds:
        expected = np.zeros((4, 4), bool)
        expected[1, 1] = expected[2, 2] = True
        np.testing.assert_array_equal(ds.read(1) > 0, expected)


def test_fusion_coverage_refuses_misaligned_grid_and_invalid_counts():
    from globato.hooks.metadata.provenance import fusion_coverage
    from rasterio.transform import from_origin

    profile = dict(
        width=4,
        height=4,
        count=7,
        crs="EPSG:32611",
        transform=from_origin(350000, 3750000, 3, 3),
    )
    bad = dict(profile, transform=from_origin(350001, 3750000, 3, 3))
    with pytest.raises(ValueError, match="same horizontal CRS"):
        list(fusion_coverage(iter([bad]), profile))
    data = np.zeros((7, 1, 1))
    data[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="coverage count"):
        list(
            fusion_coverage(iter([profile, (Window(0, 0, 1, 1), None, data)]), profile)
        )


def test_module_coverage_empty_teardown_is_valid(tmp_path):
    from globato.hooks.metadata.provenance import ProvenanceHook

    hook = ProvenanceHook(res=3, output=str(tmp_path / "empty.tif"))
    hook._init_raster(Region(0, 12, 0, 12))
    hook.teardown()  # No measured source: legend can be empty.
    with rasterio.open(hook.output) as ds:
        assert not ds.read(1).any()


@pytest.mark.parametrize("value", [10, 30, "10", "30"])
def test_numeric_coverage_resolutions_retain_final_zero(value):
    from globato.hooks.metadata.provenance import ProvenanceHook, SourceMasks

    assert ProvenanceHook(res=value).res == float(value)
    assert SourceMasks(res=value).res == float(value)
