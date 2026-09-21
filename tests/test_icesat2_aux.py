# tests/test_icesat2_aux.py

import pytest

from fetchez.modules import earthdata
from globato.streams.readers.icesat2 import ATL03Reader

ATL03 = "ATL03_20241107234251_08052501_007_01_subsetted.h5"


class _NoResults:
    """Stands in for fetchez's IceSat2 module so no test touches the network."""

    searched = []

    def __init__(self, **kwargs):
        self.results = []
        type(self).searched.append((kwargs["short_name"], kwargs["filename_filter"]))

    def run(self):
        pass


@pytest.fixture
def offline(monkeypatch):
    _NoResults.searched = []
    monkeypatch.setattr(earthdata, "IceSat2", _NoResults)
    return _NoResults


def _reader(tmp_path, *names, **kwargs):
    for name in (ATL03, *names):
        (tmp_path / name).touch()
    return ATL03Reader(str(tmp_path / ATL03), cache_dir=str(tmp_path), **kwargs)


def test_cached_atl24_prefers_the_newest_version(tmp_path, offline):
    reader = _reader(
        tmp_path,
        "ATL24_20241107234251_08052501_006_01_001_01.h5",
        "ATL24_20241107234251_08052501_006_01_002_01.h5",
    )

    found = reader.fetch_atlxx(reader.fn, "ATL24")

    assert found.endswith("_006_01_002_01.h5")


def test_atl24_may_come_from_another_release(tmp_path, offline):
    reader = _reader(tmp_path, "ATL24_20241107234251_08052501_006_01_002_01.h5")

    assert reader.fetch_atlxx(reader.fn, "ATL24") is not None


def test_atl08_must_match_the_atl03_release(tmp_path, offline):
    reader = _reader(tmp_path, "ATL08_20241107234251_08052501_006_01.h5")

    assert reader.fetch_atlxx(reader.fn, "ATL08") is None
    # Only the release-specific search ran; no fallback to timestamp and track.
    assert offline.searched == [("ATL08", "20241107234251_08052501_007")]


def test_atl08_of_the_same_release_is_used(tmp_path, offline):
    reader = _reader(tmp_path, "ATL08_20241107234251_08052501_007_01.h5")

    assert reader.fetch_atlxx(reader.fn, "ATL08").endswith("_007_01.h5")
