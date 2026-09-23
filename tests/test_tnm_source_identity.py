"""Live owner regressions for distinct TNM inputs with identical basenames."""

from pathlib import Path
import hashlib
import os
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from fetchez.spatial import Region
from fetchez.utils import inc2str

from globato.hooks.rasters.flats import RasterFlats
from globato.hooks.metadata.provenance import SourceMasks


def _source(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        nodata=-9999,
        transform=rasterio.transform.from_origin(0, 2, 1, 1),
    ) as dst:
        dst.write(np.full((2, 2), value, dtype="float32"), 1)
        dst.update_tags(PROJECT=f"project-{int(value)}")


def test_raster_flats_keeps_same_basename_sources_distinct(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = tmp_path / "download_a" / "tile.tif"
    b = tmp_path / "download_b" / "tile.tif"
    _source(a, 12)
    _source(b, 34)
    mod = SimpleNamespace(name="tnm", region=Region(0, 2, 0, 2))
    entries = [
        (
            mod,
            {
                "dst_fn": str(path),
                "tnm_product": "1m",
                "url": f"https://test/{i}/tile.tif",
            },
        )
        for i, path in enumerate((a, b))
    ]
    resulting = RasterFlats(size_threshold=1000).run(entries)
    out_a, out_b = (Path(entry["dst_fn"]) for _, entry in resulting)
    assert out_a != out_b
    assert out_a.is_file() and out_b.is_file()
    with rasterio.open(out_a) as src_a, rasterio.open(out_b) as src_b:
        assert (src_a.read(1) == 12).all()
        assert (src_b.read(1) == 34).all()
        assert src_a.tags()["PROJECT"] == "project-12"
        assert src_b.tags()["PROJECT"] == "project-34"


def test_source_masks_preserve_urls_pixels_and_project_grouping(tmp_path):
    region = Region(0, 2, 0, 2)
    mod = SimpleNamespace(
        name="tnm",
        title="TNM",
        region=region,
        meta_category="elevation",
        meta_agency="USGS",
        meta_resolution="1 m",
        weight=5.0,
    )
    dtype = [("x", "f8"), ("y", "f8"), ("z", "f8")]
    points = (
        np.array([(0.5, 1.5, 12)], dtype=dtype),
        np.array([(1.5, 0.5, 34)], dtype=dtype),
    )
    entries = []
    for idx in range(2):
        entry = {
            "dst_fn": str(tmp_path / f"download_{idx}" / "tile.tif"),
            "stream": iter([points[idx]]),
            "stream_type": "point-stream",
            "url": f"https://test/{idx}/tile.tif",
            "tnm_product": "1m",
            "source_mask_group_by": "MODULE/DATASET/WEIGHT",
            "metadata": {"dataset": f"TNM 1 m: Project {idx}", "weight": "5.0"},
        }
        entries.append((mod, entry))
    hook = SourceMasks(
        res=1, output=str(tmp_path / "sources.vrt"), output_dir=str(tmp_path / "masks")
    )
    hook.run(entries)
    paths = [Path(entry["artifacts"][hook.name]) for _, entry in entries]
    assert len(set(paths)) == 2
    for idx, path in enumerate(paths):
        with rasterio.open(path) as src:
            assert src.tags()["URL"] == entries[idx][1]["url"]
            assert src.tags()["DATASET"] == f"TNM 1 m: Project {idx}"
    for _, entry in entries:
        list(entry["stream"])
    with rasterio.open(paths[0]) as first, rasterio.open(paths[1]) as second:
        a, b = first.read(1), second.read(1)
    assert a.sum() == b.sum() == 1
    assert a[0, 0] == 1 and b[1, 1] == 1
    hook.teardown()
    with rasterio.open(tmp_path / "sources.vrt") as vrt:
        assert vrt.count >= 2
    grouped = list((tmp_path / "masks" / "grouped").rglob("*_mask.tif"))
    assert len(grouped) == 2
    assert {rasterio.open(path).tags()["DATASET"] for path in grouped} == {
        "TNM 1 m: Project 0",
        "TNM 1 m: Project 1",
    }


def test_source_masks_never_silently_overwrite_duplicate_identity(tmp_path):
    region = Region(0, 2, 0, 2)
    mod = SimpleNamespace(name="tnm", region=region)
    source = {
        "dst_fn": str(tmp_path / "tile.tif"),
        "url": "https://test/tile.tif",
        "tnm_product": "1m",
        "stream_type": "point-stream",
        "stream": iter([]),
    }
    other = {**source, "stream": iter([])}
    hook = SourceMasks(
        res=1, output=str(tmp_path / "sources.vrt"), output_dir=str(tmp_path / "masks")
    )
    with pytest.raises(RuntimeError, match="overwrite another source"):
        hook.run([(mod, source), (mod, other)])


def test_source_masks_allow_preexisting_tnm_mask_from_prior_run(tmp_path):
    region = Region(0, 2, 0, 2)
    mod = SimpleNamespace(name="tnm", region=region)
    source = {
        "dst_fn": str(tmp_path / "tile.tif"),
        "url": "https://test/tile.tif",
        "tnm_product": "1m",
        "stream_type": "point-stream",
        "stream": iter([]),
    }
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()

    source_path = os.path.abspath(source["dst_fn"])
    identity = f"{source['url']}\0{source_path}"
    token = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    prior = mask_dir / f"tile_{token}_{inc2str(1)}_mask.tif"
    prior.touch()

    hook = SourceMasks(
        res=1,
        output=str(tmp_path / "sources.vrt"),
        output_dir=str(mask_dir),
    )
    hook.run([(mod, source)])
    assert Path(source["artifacts"][hook.name]) == prior
