import threading
from types import SimpleNamespace

import rasterio

from globato.hooks.metadata.provenance import SourceMasks


def test_tnm_audit_fields_do_not_become_source_mask_tags(tmp_path):
    hook = SourceMasks(
        res="1d",
        output_dir=str(tmp_path / "masks"),
        output=str(tmp_path / "masks.vrt"),
    )
    hook._init_grid(
        SimpleNamespace(
            geo_transform=lambda **kwargs: (1, 1, (0, 1, 0, 1, 0, -1)),
            xmin=0,
            ymax=1,
            srs="EPSG:4326",
        )
    )
    hook.lock = threading.Lock()
    module = SimpleNamespace(
        name="tnm",
        title="TNM",
        meta_category="Topography",
        meta_agency="USGS",
        meta_resolution="Varies",
        weight=5,
        region=None,
    )
    entry = {
        "stream": iter(()),
        "dst_fn": "project-a.tif",
        "data_type": "rio",
        "url": "https://example.test/project-a.tif",
        "tnm_project": "Project A",
        "tnm_resolution_tier": "1m",
        "tnm_source_coverage": [{"geometry": "POLYGON EMPTY", "year": 2024}],
        "metadata": {
            "category": "elevation",
            "dataset": "TNM 1 m: Project A",
            "resolution": "1 m",
        },
    }
    hook.is_point_stream = lambda value: True
    hook._intercept = lambda stream, path, region: stream

    hook.run([(module, entry)])

    with rasterio.open(hook.tifs[0]) as src:
        tags = src.tags()
    assert tags["MODULE"] == "tnm"
    assert tags["CATEGORY"] == "elevation"
    assert tags["AGENCY"] == "USGS"
    assert tags["DATASET"] == "TNM 1 m: Project A"
    assert tags["RESOLUTION"] == "1 m"
    assert tags["WEIGHT"] == "5"
    assert not any(key.startswith("TNM_") for key in tags)
