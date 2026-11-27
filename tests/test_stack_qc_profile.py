"""Phase receipts describe existing work without entering scientific state."""

from pathlib import Path

import pytest

from globato.qc.stack import io
from globato.qc.stack.profile import SCHEMA, timed_stream
from test_stack_qc_hook import build, FixtureMB


def test_stream_excludes_consumer_time_and_closes(monkeypatch):
    from globato.qc.stack import profile

    clock = [0.0]
    closed = []
    monkeypatch.setattr(profile, "perf_counter", lambda: clock[0])

    def source():
        try:
            clock[0] += 2
            yield "profile"
            clock[0] += 3
            yield "data"
        finally:
            closed.append(True)

    receipt = {}
    stream = timed_stream(source(), receipt)
    assert next(stream) == "profile"
    clock[0] += 100  # downstream processing is not producer work
    assert next(stream) == "data"
    list(stream)
    assert receipt == dict(seconds=5.0, status="complete")
    assert closed == [True]


def test_failed_stream_does_not_claim_completion():
    def broken():
        yield "profile"
        raise ValueError("source failed")

    receipt = {}
    with pytest.raises(ValueError, match="source failed"):
        list(timed_stream(broken(), receipt))
    assert receipt["status"] == "failed"
    assert receipt["seconds"] >= 0


def test_nested_science_invalidates_cache_but_review_does_not(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from globato.qc.stack import preparation
    from globato.qc.stack.settings import Settings

    nested = tmp_path / "connectivity" / "evidence.py"
    nested.parent.mkdir()
    nested.write_text("VALUE = 1\n")
    sibling = tmp_path / "evidence.py"
    sibling.write_text("VALUE = 0\n")
    review = tmp_path / "review" / "visualize.py"
    review.parent.mkdir()
    review.write_text("VALUE = 1\n")
    monkeypatch.setattr(preparation, "__file__", str(tmp_path / "preparation.py"))

    qc = SimpleNamespace(
        settings=Settings(method="connectivity", max_passes=2, stop_when_stable=False)
    )
    module = SimpleNamespace(weight=1)
    region = SimpleNamespace(to_list=lambda: [0, 1, 0, 1])
    sources = [dict(source_id="fixed", source_sha256="fixed-content", group="survey")]

    def key():
        return preparation.cache_key(qc, module, sources, region, {})

    hashes = preparation._code_hashes()
    assert hashes["connectivity/evidence.py"] == io.sha(nested)
    assert hashes["evidence.py"] == io.sha(sibling)
    assert "review/visualize.py" not in hashes
    before = key()
    nested.write_text("VALUE = 2\n")
    after = key()
    assert after != before
    review.write_text("VALUE = 2\n")
    assert key() == after


@pytest.mark.parametrize("method,passes", [("geometry", 5), ("connectivity", 2)])
def test_profile_current_vs_cached_execution(tmp_path, monkeypatch, method, passes):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(FixtureMB, "name", "fixture_points")
    kwargs = dict(method=method, max_passes=passes, terrain_restoration=2)
    build(tmp_path, qc_args=kwargs)
    (receipt,) = (tmp_path / "qc").glob("*/profile_*.json")
    cold = io.read(receipt)
    (result_path,) = (tmp_path / "qc").glob("*/result_*.json")
    result_bytes = result_path.read_bytes()
    result = io.read(result_path)
    manifest_bytes = Path(result["manifest"]).read_bytes()
    mask_hashes = {
        sid: io.sha(r["mask"])
        for sid, r in io.read(result["manifest"])["masks"].items()
    }
    assert cold["schema"] == SCHEMA
    assert not cold["cache"]["filter_hit"]
    assert cold["settings"]["method"] == method
    assert cold["settings"]["max_passes"] == passes
    assert cold["engine"]["origin"] == "current_execution"
    assert len(cold["inputs"]) == 1
    assert cold["working_surfaces"]
    assert all(v >= 0 for v in cold["phases_seconds"].values())
    assert sum(cold["phases_seconds"].values()) == pytest.approx(cold["total_seconds"])
    build(tmp_path, qc_args=kwargs, tag="b")
    warm = io.read(receipt)
    assert warm["cache"]["filter_hit"]
    assert warm["engine"]["origin"] == "cached_execution"
    assert warm["working_surfaces"] == []
    assert warm["engine"]["timing_seconds"] == cold["engine"]["timing_seconds"]
    assert result_path.read_bytes() == result_bytes
    assert Path(result["manifest"]).read_bytes() == manifest_bytes
    assert mask_hashes == {
        sid: io.sha(r["mask"])
        for sid, r in io.read(result["manifest"])["masks"].items()
    }
    assert list((tmp_path / "b").glob("*.profile.json"))
