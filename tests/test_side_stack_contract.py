"""Native sink guarantees, exercised without Coast Mask or a regional runner."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import hashlib

import numpy as np
import pytest
import rasterio
from fetchez.entry import dataset_id
from fetchez.spatial import Region
from globato.hooks.sinks.side_stack import (
    SideStackGenerate,
    SideStackCheck,
    _resolve_grid,
    sidestack_cache_key,
)
from globato.hooks.transforms.point_pixels import FUSION_BANDS, FUSION_STATE_VERSION


def source(tmp_path, *, z=10, weight=2, chunks=2):
    asset = tmp_path / "same-source.las"
    if not asset.exists():
        asset.write_bytes(b"one asset used by independent processing branches")
    points = np.array(
        [(0.5, 1.5, z, weight, 0.1)],
        dtype=[(key, "f8") for key in ("x", "y", "z", "w", "u")],
    )
    module = SimpleNamespace(
        name="local", region=Region(0, 2, 0, 2), weight=weight, uncertainty=0, hooks=[]
    )
    entry = dict(
        url=asset.as_uri(),
        dst_fn=str(asset),
        src_srs="EPSG:4326",
        stream_type="point-stream",
        data_type="lidar",
        stream=iter([points.copy() for _ in range(chunks)]),
    )
    return module, entry


def consume(sink, pair):
    sink.run([pair])
    entry = pair[1]
    chunks = list(entry["stream"])
    with rasterio.open(entry["artifacts"]["side-stack"]) as ds:
        state = ds.read()
    return entry, chunks, state


def test_default_writer_adds_overlapping_chunks_and_passes_each_chunk_unchanged(
    tmp_path,
):
    entry, chunks, state = consume(
        SideStackGenerate(res=1, cache_dir=tmp_path), source(tmp_path)
    )
    assert len(chunks) == 3
    expected = chunks[1][2] + chunks[2][2]
    active = state[1] > 0
    np.testing.assert_array_equal(state[:, active].ravel(), expected.ravel())
    assert state[1].sum() == 2 and state[2].sum() == 4
    assert "_sidestack_receipt" not in entry


def test_fresh_receipt_requires_natural_eof_and_freezes_consumer_context(tmp_path):
    pair = source(tmp_path)
    pair[1]["_sidestack_context"] = {"consumer": {"policy": "keep"}}
    sink = SideStackGenerate(res=1, cache_dir=tmp_path, fresh=True)
    sink.run([pair])
    entry = pair[1]
    entry["_sidestack_context"]["consumer"]["policy"] = "changed"
    next(entry["stream"])
    next(entry["stream"])
    assert "_sidestack_receipt" not in entry
    assert not Path(entry["artifacts"]["side-stack"]).exists()
    list(entry["stream"])
    receipt = entry["_sidestack_receipt"]
    assert receipt["context"] == {"consumer": {"policy": "keep"}}
    assert (
        receipt["sha256"]
        == hashlib.sha256(Path(receipt["path"]).read_bytes()).hexdigest()
    )
    assert (
        receipt["processing_identity"]["identity_owner"] == "fetchez.entry.dataset_id"
    )


@pytest.mark.parametrize("failure", ["close", "error"])
def test_partial_or_failed_stream_never_publishes_a_completed_cache(tmp_path, failure):
    pair = source(tmp_path)
    if failure == "error":
        original = pair[1]["stream"]

        def broken():
            yield next(original)
            raise RuntimeError("source failed")

        pair[1]["stream"] = broken()
    sink = SideStackGenerate(res=1, cache_dir=tmp_path, fresh=True)
    sink.run([pair])
    stream = pair[1]["stream"]
    next(stream)
    next(stream)
    if failure == "close":
        stream.close()
    else:
        with pytest.raises(RuntimeError, match="source failed"):
            list(stream)
    assert "_sidestack_receipt" not in pair[1]
    assert not list(tmp_path.rglob("*.tif"))


def test_empty_source_is_a_completed_zero_state(tmp_path):
    entry, _, state = consume(
        SideStackGenerate(res=1, cache_dir=tmp_path, fresh=True),
        source(tmp_path, chunks=0),
    )
    assert not state.any() and entry["_sidestack_receipt"]["sha256"]


def test_shared_native_sink_isolates_branches_and_repeated_runs(tmp_path):
    sink = SideStackGenerate(res=1, cache_dir=tmp_path, fresh=True)
    pairs = [source(tmp_path, z=z, weight=z) for z in (2, 3, 4, 2)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda pair: consume(sink, pair), pairs))
    paths = [entry["artifacts"]["side-stack"] for entry, _, _ in results]
    ids = [
        entry["_sidestack_processing_identity"]["native_dataset_id"]
        for entry, _, _ in results
    ]
    assert len(set(paths)) == 4 and len(set(ids)) == 3
    assert sink.cache_dir == tmp_path
    for z, (_, _, state) in zip((2, 3, 4, 2), results):
        assert state[1].sum() == 2 and state[2].sum() == 2 * z
        assert state[0].sum() == 2 * z * z


def test_normal_cache_lookup_reuses_corrected_writer_and_rejects_old_identity(tmp_path):
    pair = source(tmp_path)
    entry, _, state = consume(SideStackGenerate(res=1, cache_dir=tmp_path), pair)
    hit = source(tmp_path)
    SideStackCheck(res=1, cache_dir=tmp_path).run([hit])
    assert hit[1]["_sidestack_hit"] is True
    cached = list(hit[1]["stream"])
    np.testing.assert_array_equal(cached[1][2], state)
    grid = _resolve_grid(pair[0].region, 1, "EPSG:4326")
    legacy = hashlib.sha256(
        dataset_id(entry, grid=grid, fusion_version=FUSION_STATE_VERSION).encode()
    ).hexdigest()[:16]
    assert legacy != sidestack_cache_key(entry, grid)
    assert tuple(FUSION_BANDS) == (
        "z_weighted_sum",
        "count",
        "weight_sum",
        "z2_weighted_sum",
        "weighted_uncertainty_sq",
        "x_weighted_sum",
        "y_weighted_sum",
    )


def test_native_sink_rejects_crs_relabelling_and_fresh_cache_binding(tmp_path):
    pair = source(tmp_path)
    pair[1]["src_srs"] = "EPSG:32611"
    with pytest.raises(ValueError, match="stream_reproject"):
        SideStackGenerate(res=1, cache_dir=tmp_path).run([pair])
    pair = source(tmp_path)
    pair[1]["_sidestack_path"] = str(tmp_path / "old.tif")
    with pytest.raises(ValueError, match="unbound"):
        SideStackGenerate(res=1, cache_dir=tmp_path, fresh=True).run([pair])


def test_identity_does_not_silently_alias_deep_or_cyclic_hook_settings(tmp_path):
    from globato.hooks.sinks.side_stack import source_processing_identity

    pair = source(tmp_path)
    nested = []
    nested.append(nested)
    pair[0].hooks = [SimpleNamespace(name="custom", settings=nested)]
    with pytest.raises(TypeError, match="sidestack_identity"):
        source_processing_identity(*pair)


def completed_pair(tmp_path, **kwargs):
    pair = source(tmp_path, **kwargs)
    consume(SideStackGenerate(res=1, cache_dir=tmp_path, fresh=True), pair)
    return pair


def test_native_collection_selects_exact_url_alias_once_and_keeps_lineage(tmp_path):
    from globato.hooks.sinks.side_stack import unique_source_contributions

    a = completed_pair(tmp_path)
    b = completed_pair(tmp_path)
    b[1]["url"] = "https://example.invalid/other-provider-directory/same-source.las"
    before = [Path(p[1]["artifacts"]["side-stack"]).read_bytes() for p in (a, b)]
    chosen = unique_source_contributions([a, a, b])
    assert chosen == [a]
    assert a[1]["_sidestack_aliases"][0]["source_url"] == b[1]["url"]
    assert a[1]["_sidestack_aliases"][0]["comparison"] == "exact_seven_band_state"
    assert before == [
        Path(p[1]["artifacts"]["side-stack"]).read_bytes() for p in (a, b)
    ]


@pytest.mark.parametrize("selector", ["weight", "class", "partition"])
def test_distinct_native_processing_selectors_are_never_merged(tmp_path, selector):
    from globato.hooks.sinks.side_stack import unique_source_contributions

    a = source(tmp_path)
    b = source(tmp_path, weight=3 if selector == "weight" else 2)
    if selector == "class":
        a[0].hooks = [SimpleNamespace(name="drop-class", classes=[2])]
        b[0].hooks = [SimpleNamespace(name="drop-class", classes=[40])]
    if selector == "partition":
        a[1]["stack_partition"] = "a"
        b[1]["stack_partition"] = "b"
    for pair in (a, b):
        consume(SideStackGenerate(res=1, cache_dir=tmp_path, fresh=True), pair)
    assert unique_source_contributions([a, b]) == [a, b]


def test_same_processing_identity_with_conflicting_statistics_fails(tmp_path):
    from globato.hooks.sinks.side_stack import unique_source_contributions

    a = completed_pair(tmp_path, z=10)
    b = completed_pair(tmp_path, z=20)
    with pytest.raises(ValueError, match="Conflicting statistics"):
        unique_source_contributions([a, b])


def test_duplicate_without_completed_native_receipt_fails(tmp_path):
    from globato.hooks.sinks.side_stack import unique_source_contributions

    a = completed_pair(tmp_path)
    b = completed_pair(tmp_path)
    b[1].pop("_sidestack_receipt")
    with pytest.raises(ValueError, match="completed native side-stack receipt"):
        unique_source_contributions([a, b])
