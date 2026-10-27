"""Collection-stage provenance from side-stack artifacts (``source_artifact``)."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from fetchez.spatial import Region
from globato.hooks.metadata.provenance import ProvenanceHook, SourceMasks
from globato.hooks.metadata.stack_provenance import StackProvenance
from globato.hooks.transforms.point_pixels import FUSION_BAND_MAP, FUSION_BANDS

SIZE = 4
CRS = "EPSG:32611"
TRANSFORM = from_origin(0, SIZE, 1, 1)


def _region():
    return Region(0, SIZE, 0, SIZE, srs=CRS)


def _module(name):
    return SimpleNamespace(
        name=name,
        title=name.upper(),
        meta_category="test",
        meta_agency="TEST",
        weight=1.0,
        region=_region(),
    )


def _write_side_stack(path, weights, *, transform=TRANSFORM, extra_tags=None):
    """A FusionState side-stack: one sounding per cell with weight > 0."""
    data = np.zeros((len(FUSION_BANDS), SIZE, SIZE), dtype=np.float64)
    data[FUSION_BAND_MAP["count"] - 1] = (weights > 0).astype(np.float64)
    data[FUSION_BAND_MAP["weight_sum"] - 1] = weights
    profile = dict(
        driver="GTiff",
        dtype="float64",
        count=len(FUSION_BANDS),
        width=SIZE,
        height=SIZE,
        transform=transform,
        crs=CRS,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for name, index in FUSION_BAND_MAP.items():
            dst.set_band_description(index, name)
        dst.update_tags(GLOBATO_DATATYPE="FUSION_STATE", **(extra_tags or {}))
    return str(path)


def _cells(*cells):
    w = np.zeros((SIZE, SIZE), dtype=np.float64)
    for row, col in cells:
        w[row, col] = 1.0
    return w


def _entries(tmp_path):
    """Two sources; source a's QC artifact drops one of its two cells."""
    a = _write_side_stack(tmp_path / "a.tif", _cells((0, 0), (0, 1)))
    a_qc = _write_side_stack(tmp_path / "a_qc.tif", _cells((0, 0)))
    b = _write_side_stack(tmp_path / "b.tif", _cells((3, 3)))
    return [
        (
            _module("a"),
            dict(
                url="https://x/a.xyz",
                dst_fn="a.xyz",
                data_type="xyz",
                artifacts={"side-stack": a, "qc-side-stack": a_qc},
            ),
        ),
        (
            _module("b"),
            dict(
                url="https://x/b.xyz",
                dst_fn="b.xyz",
                data_type="xyz",
                artifacts={"side-stack": b, "qc-side-stack": b},
            ),
        ),
    ]


def _read(path):
    with rasterio.open(path) as src:
        return src.read(1)


def test_source_artifact_moves_hooks_to_the_collection_stage():
    assert SourceMasks(source_artifact="side-stack").stage == "collection"
    assert ProvenanceHook(source_artifact="side-stack").stage == "collection"
    assert StackProvenance(source_artifact="side-stack").stage == "collection"
    assert SourceMasks().stage == "stream"
    with pytest.raises(ValueError, match="collection stage"):
        SourceMasks(source_artifact="side-stack", stage="stream")


@pytest.mark.parametrize("artifact", ["side-stack", "qc-side-stack"])
def test_source_masks_follow_the_selected_artifact(tmp_path, artifact):
    entries = _entries(tmp_path)
    hook = SourceMasks(
        res="1",
        output=str(tmp_path / f"sources_{artifact}.vrt"),
        source_artifact=artifact,
        resume=False,
        qgis_style=False,
    )
    hook.run(entries)
    hook.teardown()

    mask_a = _read(entries[0][1]["artifacts"]["source-masks"])
    mask_b = _read(entries[1][1]["artifacts"]["source-masks"])
    expected_a = (0, 0), (0, 1)
    if artifact == "qc-side-stack":
        expected_a = ((0, 0),)
    np.testing.assert_array_equal(mask_a, _cells(*expected_a).astype(np.uint8))
    np.testing.assert_array_equal(mask_b, _cells((3, 3)).astype(np.uint8))
    with rasterio.open(entries[0][1]["artifacts"]["source-masks"]) as src:
        assert src.tags()["MODULE"] == "a"
        assert src.tags()["GLOBATO_SOURCE_ID"] == "https://x/a.xyz"


def test_provenance_bitmask_reads_side_stacks(tmp_path):
    entries = _entries(tmp_path)
    output = tmp_path / "provenance.tif"
    hook = ProvenanceHook(res="1", output=str(output), source_artifact="qc-side-stack")
    hook.run(entries)
    hook.teardown()

    bits = _read(output)
    expected = np.zeros((SIZE, SIZE), dtype=np.uint32)
    expected[0, 0] = hook.module_bits["a"]
    expected[3, 3] = hook.module_bits["b"]
    np.testing.assert_array_equal(bits, expected)
    assert entries[0][1]["artifacts"]["provenance"] == str(output)


def test_side_stack_on_another_grid_is_refused(tmp_path):
    shifted = _write_side_stack(
        tmp_path / "shifted.tif", _cells((0, 0)), transform=from_origin(1, SIZE, 1, 1)
    )
    entries = [(_module("a"), dict(url="u", artifacts={"side-stack": shifted}))]
    hook = SourceMasks(
        res="1", output=str(tmp_path / "s.vrt"), source_artifact="side-stack"
    )
    with pytest.raises(ValueError, match="differs from the provenance grid"):
        hook.run(entries)


def test_repeated_references_are_recorded_once(tmp_path):
    a = _write_side_stack(tmp_path / "a.tif", _cells((1, 1)))
    entry = dict(url="https://x/a.xyz", artifacts={"side-stack": a})
    hook = SourceMasks(
        res="1",
        output=str(tmp_path / "s.vrt"),
        source_artifact="side-stack",
        resume=False,
        qgis_style=False,
    )
    hook.run([(_module("a"), entry), (_module("a"), dict(entry))])
    assert len(hook._masks.masks) == 1


def _fusion_state(path, count, weight_sum):
    data = np.zeros((len(FUSION_BANDS), SIZE, SIZE), dtype=np.float64)
    data[FUSION_BAND_MAP["count"] - 1] = count
    data[FUSION_BAND_MAP["weight_sum"] - 1] = weight_sum
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        dtype="float64",
        count=len(FUSION_BANDS),
        width=SIZE,
        height=SIZE,
        transform=TRANSFORM,
        crs=CRS,
    ) as dst:
        dst.write(data)
        dst.update_tags(
            GLOBATO_DATATYPE="FUSION_STATE",
            GLOBATO_STACK_STRATEGY="mixed",
            GLOBATO_WEIGHT_TIERS="[0.25, 0.5, 1.0]",
        )
    return str(path)


def _mixed_sources():
    # [0,0] b(1.0) beats a(0.5); [0,1] a and b merge at 0.5; [1,0] only a.
    a = np.zeros((SIZE, SIZE))
    a[0, 0], a[0, 1], a[1, 0] = 0.5, 0.5, 0.25
    b = np.zeros((SIZE, SIZE))
    b[0, 0], b[0, 1] = 1.0, 0.5
    count = np.zeros((SIZE, SIZE))
    wsum = np.zeros((SIZE, SIZE))
    count[0, 0], wsum[0, 0] = 1, 1.0
    count[0, 1], wsum[0, 1] = 2, 1.0
    count[1, 0], wsum[1, 0] = 1, 0.25
    return a, b, count, wsum


def _stack_masks(hook):
    return {
        dataset_id: _read(path)
        for dataset_id, path in hook._final_masks.masks.items()
        if os.path.exists(path)
    }


def test_stack_provenance_from_side_stacks_matches_the_stream_stage(tmp_path):
    a, b, count, wsum = _mixed_sources()
    fusion = _fusion_state(tmp_path / "fusion.tif", count, wsum)
    side_a = _write_side_stack(tmp_path / "a.tif", a)
    side_b = _write_side_stack(tmp_path / "b.tif", b)

    def entries(with_stream):
        out = []
        for name, weights, side in (("a", a, side_a), ("b", b, side_b)):
            entry = dict(
                checksum=f"source-{name}",
                dst_fn=f"{name}.tif",
                data_type="test",
                artifacts={"fusion-state": fusion, "side-stack": side},
            )
            if with_stream:
                data = np.zeros((len(FUSION_BANDS), SIZE, SIZE))
                data[FUSION_BAND_MAP["count"] - 1] = (weights > 0).astype(float)
                data[FUSION_BAND_MAP["weight_sum"] - 1] = weights
                window = Window(0, 0, SIZE, SIZE)
                entry["stream"] = iter([(window, window, data, None, TRANSFORM)])
            out.append((_module(name), entry))
        return out

    def build(tag, **kwargs):
        return StackProvenance(
            output=str(tmp_path / f"stack_{tag}.vrt"),
            state_dir=str(tmp_path / f"state_{tag}"),
            storage="memory",
            qgis_style=False,
            **kwargs,
        )

    stream_hook = build("stream")
    for _, entry in stream_hook.run(entries(True)):
        list(entry["stream"])
    stream_hook.teardown()

    side_hook = build("side", source_artifact="side-stack")
    side_hook.run(entries(False))
    side_hook.teardown()

    stream_masks = _stack_masks(stream_hook)
    side_masks = _stack_masks(side_hook)
    assert side_masks.keys() == stream_masks.keys() == {"source-a", "source-b"}
    for key in stream_masks:
        np.testing.assert_array_equal(side_masks[key], stream_masks[key])
    assert side_masks["source-a"][0, 0] == 0  # b's higher tier wins there
    assert side_masks["source-a"][1, 0] == 1


def test_stack_provenance_needs_stack_combine_first(tmp_path):
    side = _write_side_stack(tmp_path / "a.tif", _cells((0, 0)))
    hook = StackProvenance(
        output=str(tmp_path / "s.vrt"), source_artifact="side-stack", storage="memory"
    )
    with pytest.raises(ValueError, match="after stack_combine"):
        hook.run([(_module("a"), dict(url="u", artifacts={"side-stack": side}))])


def test_real_side_stack_and_stack_combine_grids_line_up(tmp_path, monkeypatch):
    """side_stack -> stack_combine -> provenance hooks on a geographic 1" grid."""
    from globato.hooks.sinks.side_stack import SideStackGenerate
    from globato.hooks.tools.stack_combine import StackCombine

    monkeypatch.chdir(tmp_path)
    region = Region(-118.5, -118.49, 34.0, 34.01, srs="EPSG:4326")

    def points(x, y, w):
        data = np.zeros(
            1, dtype=[("x", "f8"), ("y", "f8"), ("z", "f8"), ("w", "f8"), ("u", "f8")]
        )
        data["x"], data["y"], data["z"], data["w"] = x, y, -10.0, w
        return data

    entries = []
    for name, x, y, w in (
        ("a", -118.4995, 34.0095, 1.0),
        ("b", -118.4905, 34.0005, 0.5),
    ):
        mod = SimpleNamespace(
            name=name,
            title=name,
            weight=w,
            uncertainty=0,
            hooks=[],
            region=region,
            _outdir=str(tmp_path),
        )
        entry = dict(
            url=f"https://x/{name}.xyz",
            dst_fn=str(tmp_path / f"{name}.xyz"),
            src_srs="EPSG:4326",
            data_type="xyz",
            stream_type="point-stream",
            stream=iter([points(x, y, w)]),
        )
        entries.append((mod, entry))

    SideStackGenerate(res="1s", crs="EPSG:4326").run(entries)
    for _, entry in entries:
        list(entry["stream"])
    StackCombine(
        res="1s",
        crs="EPSG:4326",
        strategy="mixed",
        weight_threshold="0.5/1",
        output=str(tmp_path / "stack.tif"),
    ).run(entries)

    hooks = [
        SourceMasks(
            res="1s",
            output="sources.vrt",
            source_artifact="side-stack",
            resume=False,
            qgis_style=False,
        ),
        ProvenanceHook(res="1s", output="provenance.tif", source_artifact="side-stack"),
        StackProvenance(
            output="stack_sources.vrt",
            source_artifact="side-stack",
            storage="memory",
            qgis_style=False,
        ),
    ]
    for hook in hooks:
        hook.run(entries)
        hook.teardown()

    masks = {k: _read(p) for k, p in hooks[0]._masks.masks.items()}
    assert {k: int(m.sum()) for k, m in masks.items()} == {
        "https://x/a.xyz": 1,
        "https://x/b.xyz": 1,
    }
    assert np.count_nonzero(_read(tmp_path / "provenance.tif")) == 2
    final = _stack_masks(hooks[2])
    assert {k: int(m.sum()) for k, m in final.items()} == {
        "https://x/a.xyz": 1,
        "https://x/b.xyz": 1,
    }
