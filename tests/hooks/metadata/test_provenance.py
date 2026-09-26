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
