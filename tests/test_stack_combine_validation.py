"""Additional regressions for stack_combine FusionState validation.

Place beside the existing test_stack_combine.py tests.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from globato.hooks.tools.stack_combine import StackCombine
from globato.hooks.transforms.point_pixels import FUSION_BANDS, FUSION_STATE_VERSION


def _write(path, *, cache_hash=None, version=None, shifted=False):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=len(FUSION_BANDS),
        dtype="float64",
        crs="EPSG:4326",
        transform=from_origin(0.5 if shifted else 0, 2, 1, 1),
    ) as dst:
        data = np.zeros((len(FUSION_BANDS), 2, 2), dtype="float64")
        data[FUSION_BANDS.index("count")] = 1
        data[FUSION_BANDS.index("weight_sum")] = 1
        dst.write(data)
        for idx, name in enumerate(FUSION_BANDS, 1):
            dst.set_band_description(idx, name)
        tags = {
            "GLOBATO_DATATYPE": "FUSION_STATE",
            "GLOBATO_FUSION_VERSION": str(
                FUSION_STATE_VERSION if version is None else version
            ),
        }
        if cache_hash is not None:
            tags["GLOBATO_SIDESTACK_HASH"] = cache_hash
        dst.update_tags(**tags)


def _accumulator():
    return SimpleNamespace(
        xcount=2, ycount=2, transform=from_origin(0, 2, 1, 1), crs="EPSG:4326"
    )


def test_new_cache_hash_does_not_affect_assembly_validation(tmp_path):
    path = tmp_path / "test_sidestack.tif"
    _write(path, cache_hash="new-fetchez-dataset-id-digest")
    with rasterio.open(path) as src:
        StackCombine._validate_source(src, str(path), _accumulator())


def test_qc_fusion_artifact_can_lack_original_cache_hash(tmp_path):
    path = tmp_path / "qc.tif"
    _write(path)
    with rasterio.open(path) as src:
        StackCombine._validate_source(src, str(path), _accumulator())


@pytest.mark.parametrize(
    "variant, expected",
    [
        ("version", "FusionState version"),
        ("grid", "geotransform"),
    ],
)
def test_scientific_incompatibility_still_rejected(tmp_path, variant, expected):
    path = tmp_path / "invalid.tif"
    _write(
        path,
        version=FUSION_STATE_VERSION + 1 if variant == "version" else None,
        shifted=variant == "grid",
    )
    with rasterio.open(path) as src:
        with pytest.raises(ValueError, match=expected):
            StackCombine._validate_source(src, str(path), _accumulator())
