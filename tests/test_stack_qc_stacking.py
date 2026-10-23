"""Exact cached block rebuild versus the existing native source-stream path."""

from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
from fetchez.spatial import Region
from globato.hooks.sinks.multi_stack import MultiStackHook
from globato.hooks.transforms.point_pixels import FUSION_BANDS
from globato.qc.stack import io
from globato.qc.stack.context import WorkingSurface
from globato.qc.stack.native import masked_stream
from globato.qc.stack.stacking import rebuild


def fixture(tmp_path, strategy):
    region = Region(350000, 351536, 3748464, 3750000, srs="EPSG:32611")
    stack = MultiStackHook(
        res="3", crs="EPSG:32611", strategy=strategy, weight_threshold=".5/1.5"
    )
    rng = np.random.default_rng(27)
    records = []
    for index, weight in enumerate((0.2, 1.0, 2.0, 1.0)):
        path = tmp_path / f"source{index}.tif"
        shape = (512, 512) if index != 3 else (256, 256)
        transform = from_origin(350000, 3750000, 3, 3)
        if index == 3:
            # Integer source offset crosses global blocks in both directions.
            transform = from_origin(350381, 3749619, 3, 3)
        count = rng.integers(1, 10, size=shape).astype("float64")
        if index == 2:
            count[256:] = 0
        z = rng.normal(-100 - index * 5, 2, size=shape)
        w = count * weight
        data = np.stack((z * w, count, w, z * z * w, w * 0.1, w * 350000, w * 3750000))
        data[:, count == 0] = 0
        profile = dict(
            driver="GTiff",
            width=shape[1],
            height=shape[0],
            count=7,
            dtype="float64",
            transform=transform,
            crs="EPSG:32611",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="lzw",
        )
        with rasterio.open(path, "w", **profile) as ds:
            ds.write(data)
            for i, name in enumerate(FUSION_BANDS, 1):
                ds.set_band_description(i, name)
        records.append(
            dict(
                source_id=str(index),
                path=str(path),
                group=str(index),
                grid=dict(shape=list(shape)),
                fingerprint=io.fingerprint(path),
            )
        )
    return stack, region, records


def mask(tmp_path, records, suffix, *, cell=(10, 10), source=0):
    entries = {}
    for rec in records:
        path = tmp_path / f"mask{suffix}_{rec['source_id']}.tif"
        with rasterio.open(rec["path"]) as src:
            profile = src.profile.copy()
        profile.update(count=1, dtype="uint8")
        data = np.zeros((profile["height"], profile["width"]), "uint8")
        if int(rec["source_id"]) == source:
            data[cell] = 1
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)
        entries[rec["source_id"]] = dict(
            mask=str(path),
            mask_sha256=io.sha(path),
            mask_bits_sha256=io.digest(data.tolist()),
        )
    return dict(masks=entries, rejected_source_cells=1)


def reference(stack, region, records, manifest, path):
    acc = stack.create_accumulator(
        region, output=str(path), state=str(path.with_suffix(".state.tif"))
    )
    try:
        for rec in records:
            stream = masked_stream(
                rec["path"],
                manifest["masks"][rec["source_id"]]["mask"] if manifest else None,
            )
            profile = next(stream)
            from globato.qc.stack.grid import grid_offset
            from types import SimpleNamespace

            dy, dx = grid_offset(SimpleNamespace(**profile), acc.dataset)
            for item in stream:
                w, _, data, *_ = item
                acc.update_raster_state(
                    (
                        Window(w.col_off + dx, w.row_off + dy, w.width, w.height),
                        None,
                        data,
                    )
                )
        return acc.finalize(statistics=False)
    finally:
        acc.close()


@pytest.mark.parametrize("strategy", ["mean", "weighted_mean", "mixed", "supercede"])
@pytest.mark.parametrize("workers", [1, 3])
def test_rebuilt_all_bands_exactly_match_native_replay_and_reuse_unchanged_blocks(
    tmp_path, strategy, workers
):
    stack, region, records = fixture(tmp_path, strategy)
    a = stack.create_accumulator(
        region,
        output=str(tmp_path / "base.tif"),
        state=str(tmp_path / "base.state.tif"),
    )
    details = rebuild(a, records, None, workers=workers)
    a.finalize(statistics=False)
    previous = dict(**details, state=a.state_fn, masks=None)
    current = mask(tmp_path, records, "one", source=2)
    b = stack.create_accumulator(
        region,
        output=str(tmp_path / "update.tif"),
        state=str(tmp_path / "update.state.tif"),
    )
    updated = rebuild(b, records, current, previous, workers=workers)
    b.finalize(statistics=False)
    assert updated["timings"]["rebuilt_blocks"] == 1
    assert updated["timings"]["changed_blocks"] == 1
    assert updated["timings"]["total_blocks"] == 4
    assert (
        updated["timings"]["source_block_reads"]
        < details["timings"]["source_block_reads"]
    )
    expected = reference(stack, region, records, current, tmp_path / "reference.tif")
    with rasterio.open(b.output_fn) as actual, rasterio.open(expected) as wanted:
        np.testing.assert_array_equal(actual.read(), wanted.read())
    # Restoration-like reversal and a moved rejection must also be exact.
    moved = mask(tmp_path, records, "two", cell=(10, 400), source=2)
    c = stack.create_accumulator(
        region,
        output=str(tmp_path / "moved.tif"),
        state=str(tmp_path / "moved.state.tif"),
    )
    more = rebuild(
        c,
        records,
        moved,
        dict(**updated, state=b.state_fn, masks=current),
        workers=workers,
    )
    c.finalize(statistics=False)
    assert more["timings"]["rebuilt_blocks"] == 2
    expected = reference(stack, region, records, moved, tmp_path / "reference2.tif")
    with rasterio.open(c.output_fn) as actual, rasterio.open(expected) as wanted:
        np.testing.assert_array_equal(actual.read(), wanted.read())


def interpolators():
    from globato.hooks.rasters.binary_cudem import BinaryCudemStepDown

    return [
        BinaryCudemStepDown(
            steps=1,
            resolutions="3/9",
            weights=".5",
            algos="raster_fill/raster_fill",
            barrier=None,
            inland_decay_dist=0,
        )
    ]


def test_unchanged_winning_state_skips_native_interpolation(tmp_path):
    # In supercede mode the lower-weight source is hidden by better observations.
    stack, region, records = fixture(tmp_path, "supercede")
    builder = WorkingSurface(stack, interpolators, records, region, tmp_path)
    baseline = builder(None, tmp_path / "baseline")
    manifest = tmp_path / "manifest.json"
    io.write(manifest, mask(tmp_path, records, "loser"))
    updated = builder(manifest, tmp_path / "updated")
    assert updated == baseline
    receipt = io.read(tmp_path / "updated/COMPLETE.json")
    assert receipt["interpolation_seconds"] == 0
    assert receipt["timings"]["changed_blocks"] == 0
    assert Path(baseline).is_file()
    # Reloading this cached state must preserve the earlier files.
    assert builder(manifest, tmp_path / "updated") == baseline


def test_changed_fusion_state_invalidates_cached_context(tmp_path):
    stack, region, records = fixture(tmp_path, "mean")
    builder = WorkingSurface(stack, interpolators, records, region, tmp_path)
    builder(None, tmp_path / "baseline")
    with Path(builder.previous["state"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(RuntimeError, match="FusionState changed"):
        builder(None, tmp_path / "baseline")


def test_sparse_source_index_skips_empty_science_window_without_reading(
    tmp_path, monkeypatch
):
    from globato.qc.stack.sources import SourcePool

    stack, region, records = fixture(tmp_path, "mean")
    records[2].update(offset=[0, 0], active_blocks=["0,0", "0,1"])
    pool = SourcePool(dict(grid=dict(shape=[512, 512]), sources=[records[2]]))

    def unexpected(*args, **kwargs):
        raise AssertionError("An empty indexed block should not open source files")

    monkeypatch.setattr(pool, "ds", unexpected)
    data = pool.source(records[2], (slice(256, 512), slice(0, 256)))
    assert not data[1].any() and not data[2].any()
    pool.close()


def test_unaligned_mask_is_rejected_before_incremental_rebuild(tmp_path):
    stack, region, records = fixture(tmp_path, "mean")
    manifest = mask(tmp_path, records, "unaligned")
    path = manifest["masks"]["0"]["mask"]
    with rasterio.open(path, "r+") as ds:
        ds.transform = from_origin(350001, 3750000, 3, 3)
    acc = stack.create_accumulator(
        region, output=str(tmp_path / "bad.tif"), state=str(tmp_path / "bad.state.tif")
    )
    try:
        with pytest.raises(ValueError, match="noninteger cell offset"):
            rebuild(acc, records, manifest)
    finally:
        acc.close()


def test_sparse_index_preserves_windows_that_cross_block_boundaries(tmp_path):
    from copy import deepcopy
    from globato.qc.stack.sources import SourcePool

    stack, region, records = fixture(tmp_path, "mean")
    acc = stack.create_accumulator(
        region,
        output=str(tmp_path / "indexed.tif"),
        state=str(tmp_path / "indexed.state.tif"),
    )
    try:
        details = rebuild(acc, records, None)
    finally:
        acc.close()
    indexed = deepcopy(records)
    for record in indexed:
        record["active_blocks"] = details["source_blocks"][record["source_id"]]
    plain = SourcePool(dict(grid=dict(shape=[512, 512]), sources=records))
    sparse = SourcePool(dict(grid=dict(shape=[512, 512]), sources=indexed))
    try:
        for s in (
            (slice(250, 270), slice(250, 270)),
            (slice(256, 512), slice(0, 256)),
            (slice(0, 100), slice(0, 100)),
        ):
            for a, b in zip(records, indexed):
                for actual, expected in zip(sparse.source(b, s), plain.source(a, s)):
                    np.testing.assert_array_equal(actual, expected)
    finally:
        plain.close()
        sparse.close()


def _sparse_source(path, blocks, value=-50.0):
    """A sparse 7-band 1024x1024 FusionState with data only in ``blocks``."""
    profile = dict(
        driver="GTiff",
        width=1024,
        height=1024,
        count=7,
        dtype="float64",
        nodata=None,
        crs="EPSG:32611",
        transform=from_origin(350000, 3750000, 3, 3),
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="lzw",
        sparse_ok=True,
    )
    with rasterio.open(path, "w", **profile) as dst:
        for row, col in blocks:
            data = np.zeros((7, 256, 256))
            data[1] = 2.0
            data[2] = 2.0
            data[0] = value * 2.0
            data[3] = value * value * 2.0
            dst.write(data, window=Window(col * 256, row * 256, 256, 256))
    with rasterio.open(path) as ds:
        g = io.grid(ds)
    return dict(
        source_id=path.name,
        group=None,
        path=str(path),
        grid=g,
        offset=[0, 0],
        source_sha256=io.sha(path),
        fingerprint=io.fingerprint(path),
    )


def test_unwritten_sparse_blocks_are_skipped_with_identical_results(
    tmp_path, monkeypatch
):
    import globato.qc.stack.stacking as stacking

    region = Region(350000, 353072, 3746928, 3750000, srs="EPSG:32611")
    stack = MultiStackHook(res="3", crs="EPSG:32611", strategy="mean")
    records = [
        _sparse_source(tmp_path / "a.tif", [(0, 0), (2, 3)]),
        _sparse_source(tmp_path / "b.tif", [(1, 1)], value=-60.0),
    ]
    info = stacking.written_blocks(records[0]["path"])
    assert info is not None and info[2] == {(0, 0), (2, 3)}

    def build(name):
        acc = stack.create_accumulator(
            region,
            output=str(tmp_path / f"{name}.tif"),
            state=str(tmp_path / f"{name}.state.tif"),
        )
        details = rebuild(acc, [dict(r) for r in records], None, workers=2)
        acc.close()
        with rasterio.open(acc.state_fn) as ds:
            return ds.read(), details

    skipped, d1 = build("skipped")
    monkeypatch.setattr(stacking, "written_blocks", lambda path: None)
    read_all, d2 = build("read_all")
    np.testing.assert_array_equal(skipped, read_all)
    assert d1["source_blocks"] == d2["source_blocks"]
    assert d1["timings"]["source_block_reads"] == d2["timings"]["source_block_reads"]


def test_side_stacks_are_written_sparse(tmp_path):
    """Blocks no source chunk touches are not stored."""
    import globato.qc.stack.stacking as stacking
    from globato.hooks.sinks.side_stack import SideStackGenerate

    profile = SideStackGenerate._creation_profile(
        dict(
            driver="GTiff",
            width=1024,
            height=1024,
            count=7,
            crs="EPSG:32611",
            transform=from_origin(350000, 3750000, 3, 3),
        )
    )
    assert profile["sparse_ok"] is True
    path = tmp_path / "s.tif"
    with rasterio.open(path, "w+", **profile) as dst:
        dst.write(
            dst.read(window=Window(0, 0, 256, 256)) + 1, window=Window(0, 0, 256, 256)
        )
    assert stacking.written_blocks(path)[2] == {(0, 0)}
