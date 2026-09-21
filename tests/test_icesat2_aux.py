# tests/test_icesat2_aux.py

import h5py
import numpy as np
import pandas as pd
import pytest

from fetchez.modules import earthdata
from globato.streams.readers.icesat2 import ATL03Reader, _atl24_rows_in_atl03

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


@pytest.mark.parametrize("requested", ["008", 8, "8"])
def test_wrong_atl03_release_is_skipped(tmp_path, offline, requested):
    # The ATL03 file is empty, so reaching h5py would raise: yielding nothing
    # shows the release check turned the file away first.
    reader = _reader(tmp_path, atl_version=requested)

    assert list(reader.yield_chunks()) == []


def test_matching_atl03_release_is_read(tmp_path, offline):
    reader = _reader(tmp_path, atl_version="007", classes="1")

    # Past the release check, the empty file fails to open as HDF5.
    with pytest.raises(OSError):
        list(reader.yield_chunks())


# ---------------------------------------------------------------------------
# ATL24 -> ATL03 photon join
# ---------------------------------------------------------------------------
EPOCH = 1198800018.0
OFFSET = 72_472  # rows of the full granule that come before the subset
SEAFLOOR_ROWS = [2, 9, 10]  # one photon of a 3-photon pulse, two of a 4-photon pulse


def _atl24_time(atl03_dt):
    """delta_time as ATL24 stores it: through absolute nanoseconds in a float64."""
    return ((atl03_dt + EPOCH) * 1e9) / 1e9 - EPOCH


def _atl03_delta_time():
    """One subsetted beam: 8 pulses 1e-4 s apart returning 1-4 photons each."""
    photons_per_pulse = [1, 3, 2, 1, 4, 1, 2, 1]
    pulse_times = 181_635_480.123456789 + 1e-4 * np.arange(len(photons_per_pulse))
    return np.repeat(pulse_times, photons_per_pulse)


def _atl24_beam(atl03_dt, offset=OFFSET, omit=(4, 12)):
    """ATL24's view of that beam: it omits a few photons, and it covers the whole
    granule, so it also has photons from before and after the subset."""
    rows = np.array([r for r in range(len(atl03_dt)) if r not in omit])
    outside = _atl24_time(atl03_dt[[0, -1]] + [-0.5, 0.5])
    delta_time = np.concatenate(
        ([outside[0]], _atl24_time(atl03_dt[rows]), [outside[1]])
    )
    index_ph = np.concatenate(([offset - 5000], rows + offset, [offset + 5000]))
    return rows, delta_time, index_ph


def test_atl24_rows_are_found_in_a_subsetted_atl03():
    atl03_dt = _atl03_delta_time()
    rows, atl24_dt, index_ph = _atl24_beam(atl03_dt)

    in_file, found = _atl24_rows_in_atl03(atl03_dt, atl24_dt, index_ph, EPOCH)

    # The two products disagree on the raw delta_time, so equality on it would
    # not have joined them.
    assert not np.array_equal(atl24_dt[1:-1], atl03_dt[rows])
    assert in_file.tolist() == [False] + [True] * len(rows) + [False]
    assert found.tolist() == rows.tolist()


def test_atl24_rows_are_found_in_a_full_atl03_granule():
    atl03_dt = _atl03_delta_time()
    rows, atl24_dt, index_ph = _atl24_beam(atl03_dt, offset=0)

    _, found = _atl24_rows_in_atl03(atl03_dt, atl24_dt, index_ph, EPOCH)

    assert found.tolist() == rows.tolist()


def test_atl24_with_no_photons_in_the_atl03_file_finds_nothing():
    atl03_dt = _atl03_delta_time()
    _, atl24_dt, index_ph = _atl24_beam(atl03_dt)

    in_file, found = _atl24_rows_in_atl03(
        atl03_dt, atl24_dt[[0, -1]], index_ph[[0, -1]], EPOCH
    )

    assert not in_file.any()
    assert len(found) == 0


def test_atl24_rows_are_refused_when_the_photons_do_not_line_up():
    # An ATL03 release holding a photon that ATL24's release did not: every row
    # after it is off by one, so no single offset fits.
    atl03_dt = _atl03_delta_time()
    _, atl24_dt, index_ph = _atl24_beam(atl03_dt)
    with_extra_photon = np.insert(atl03_dt, 7, atl03_dt[7])

    assert _atl24_rows_in_atl03(with_extra_photon, atl24_dt, index_ph, EPOCH) is None


def _write_atl24(path, atl03_dt, laser="gt1l"):
    rows, delta_time, index_ph = _atl24_beam(atl03_dt)
    class_ph = np.full(len(delta_time), 41, dtype=np.int8)
    class_ph[np.isin(index_ph - OFFSET, SEAFLOOR_ROWS)] = 40
    class_ph[0] = 40  # a seafloor photon outside the ATL03 file
    n = len(delta_time)
    with h5py.File(path, "w") as f:
        f["ancillary_data/atlas_sdp_gps_epoch"] = np.array([EPOCH])
        f[f"{laser}/delta_time"] = delta_time
        f[f"{laser}/index_ph"] = index_ph.astype(np.int32)
        f[f"{laser}/class_ph"] = class_ph
        f[f"{laser}/confidence"] = np.where(
            index_ph - OFFSET == SEAFLOOR_ROWS[0], 0.4, 0.9
        )
        f[f"{laser}/lat_ph"] = 24.0 + 1e-6 * np.arange(n)
        f[f"{laser}/lon_ph"] = -81.0 - 1e-6 * np.arange(n)
        f[f"{laser}/ortho_h"] = (-5.0 - 0.1 * np.arange(n)).astype(np.float32)
    return index_ph - OFFSET


def _atl03_frame(atl03_dt):
    n = len(atl03_dt)
    return pd.DataFrame(
        {
            "latitude": np.full(n, 25.0),
            "longitude": np.full(n, -80.0),
            "photon_height": np.zeros(n, dtype=np.float32),
            "delta_time": atl03_dt,
            "photon_geoid": np.full(n, -25.0, dtype=np.float32),
            "photon_f2m": np.zeros(n, dtype=np.float32),
            "photon_tide_f2m": np.zeros(n, dtype=np.float32),
            "ph_h_classed": -1,
            "bathy_confidence": -1.0,
        }
    )


def test_atl24_labels_only_the_seafloor_photon_of_a_pulse(tmp_path, offline):
    atl03_dt = _atl03_delta_time()
    atl24_fn = tmp_path / "ATL24_20241107234251_08052501_006_01_002_01.h5"
    atl24_rows = _write_atl24(atl24_fn, atl03_dt)
    reader = _reader(tmp_path)

    df = reader.apply_atl24_classifications(
        _atl03_frame(atl03_dt), str(atl24_fn), "gt1l", None, None
    )

    assert np.flatnonzero(df["ph_h_classed"] == 40).tolist() == SEAFLOOR_ROWS
    assert (df["ph_h_classed"].drop(SEAFLOOR_ROWS) == -1).all()
    # Each labelled photon takes the position and height of its own ATL24 photon.
    with h5py.File(atl24_fn) as f:
        at = [int(np.flatnonzero(atl24_rows == r)[0]) for r in SEAFLOOR_ROWS]
        assert (
            df.loc[SEAFLOOR_ROWS, "latitude"].tolist()
            == f["gt1l/lat_ph"][...][at].tolist()
        )
        assert (
            df.loc[SEAFLOOR_ROWS, "photon_height"].tolist()
            == f["gt1l/ortho_h"][...][at].tolist()
        )
    assert (df["latitude"].drop(SEAFLOOR_ROWS) == 25.0).all()


def test_atl24_min_bathy_confidence_is_applied(tmp_path, offline):
    atl03_dt = _atl03_delta_time()
    atl24_fn = tmp_path / "ATL24_20241107234251_08052501_006_01_002_01.h5"
    _write_atl24(atl24_fn, atl03_dt)
    reader = _reader(tmp_path, min_bathy_confidence=0.6)

    df = reader.apply_atl24_classifications(
        _atl03_frame(atl03_dt), str(atl24_fn), "gt1l", None, None
    )

    assert np.flatnonzero(df["ph_h_classed"] == 40).tolist() == SEAFLOOR_ROWS[1:]


def test_atl24_that_does_not_line_up_changes_nothing(tmp_path, offline):
    atl03_dt = _atl03_delta_time()
    atl24_fn = tmp_path / "ATL24_20241107234251_08052501_006_01_002_01.h5"
    _write_atl24(atl24_fn, atl03_dt)
    reader = _reader(tmp_path)
    before = _atl03_frame(np.insert(atl03_dt, 7, atl03_dt[7]))

    after = reader.apply_atl24_classifications(
        before.copy(), str(atl24_fn), "gt1l", None, None
    )

    pd.testing.assert_frame_equal(after, before)
