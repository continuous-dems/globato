# tests/test_icesat2.py

import logging

import json

import h5py
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.strtree import STRtree

import fetchez
from fetchez.modules import earthdata
from globato.streams.readers import icesat2
from globato.streams.readers.icesat2 import (
    ATL03Reader,
    _as_atl24_time,
    _atl24_release_shift,
    _atl24_rows_in_atl03,
    _per_photon,
    _photon_index_within_segment,
    _points_in_tree,
    _read_atl24_block,
)

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
CHUNK = 4  # photons per storage chunk in the ATL24 files written here
SEAFLOOR_ROWS = [2, 9, 10]  # one photon of a 3-photon pulse, two of a 4-photon pulse
GEOID = -25.0
# Where ATL24 has every photon, relative to the ATL03 file being read, because it
# was built from another ATL03 release: (degrees latitude, degrees longitude, metres).
RELEASE_SHIFT = (3e-6, -7e-6, 0.02)
# What ATL24's refraction correction adds to that for a seafloor photon.
REFRACTION = (1e-7, 2e-7, 1.5)


def _atl24_time(atl03_dt):
    """delta_time as ATL24 stores it: through absolute nanoseconds in a float64."""
    return ((atl03_dt + EPOCH) * 1e9) / 1e9 - EPOCH


def _atl03_delta_time():
    """One subsetted beam: 10 pulses 1e-4 s apart returning 1-4 photons each."""
    photons_per_pulse = [1, 3, 2, 1, 4, 1, 2, 1, 2, 3]
    pulse_times = 181_635_480.123456789 + 1e-4 * np.arange(len(photons_per_pulse))
    return np.repeat(pulse_times, photons_per_pulse)


def _atl24_beam(atl03_dt, offset=OFFSET, omit=(4, 12), outside=1):
    """ATL24's view of that beam: it omits a few photons, and it covers the whole
    granule, so it also has `outside` photons from before and after the subset."""
    rows = np.array([r for r in range(len(atl03_dt)) if r not in omit])
    away = 0.5 + 1e-4 * np.arange(outside)
    before = _atl24_time(atl03_dt[0] - away[::-1])
    after = _atl24_time(atl03_dt[-1] + away)
    delta_time = np.concatenate((before, _atl24_time(atl03_dt[rows]), after))
    index_ph = np.concatenate(
        (
            offset - 5000 - np.arange(outside)[::-1],
            rows + offset,
            offset + 5000 + np.arange(outside),
        )
    )
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


def test_atl24_rows_survive_a_last_bit_change_in_delta_time():
    # Between ATL03 releases a pulse's delta_time can differ in its last bit,
    # which can move it to the next value ATL24 is able to store.
    atl03_dt = _atl03_delta_time()
    rows, atl24_dt, index_ph = _atl24_beam(atl03_dt)
    pulse = atl03_dt == atl03_dt[9]
    for last_bits in range(1, 20):
        nudged = np.where(pulse, atl03_dt + last_bits * np.spacing(atl03_dt), atl03_dt)
        if _atl24_time(nudged[9]) != _atl24_time(atl03_dt[9]):
            break
    else:
        pytest.fail("could not move the pulse onto another ATL24 value")

    _, found = _atl24_rows_in_atl03(nudged, atl24_dt, index_ph, EPOCH)

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


def _atl03_photons(n):
    """Latitude, longitude and ellipsoid height of the ATL03 photons: a sea surface
    at the geoid, with the seafloor photons 6 m below it before refraction."""
    h_ph = np.full(n, GEOID)
    h_ph[SEAFLOOR_ROWS] -= 6.0
    return 25.0 + 1e-5 * np.arange(n), -80.0 + 1e-6 * np.arange(n), h_ph


def _write_atl24(path, atl03_dt, laser="gt1l", outside=1, order=None, sea_surface=True):
    """Write an ATL24 file in small chunks; `order` rearranges its photons. Without
    `sea_surface`, the photons that are not seafloor are unclassified ones under
    the surface, which ATL24 refracts too."""
    rows, delta_time, index_ph = _atl24_beam(atl03_dt, outside=outside)
    n = len(delta_time)
    atl03_row = np.clip(index_ph - OFFSET, 0, len(atl03_dt) - 1)
    is_seafloor = np.isin(index_ph - OFFSET, SEAFLOOR_ROWS)
    class_ph = np.full(n, 41 if sea_surface else 0, dtype=np.int8)
    class_ph[is_seafloor] = 40
    class_ph[0] = 40  # a seafloor photon outside the ATL03 file

    lat, lon, h_ph = (v[atl03_row] for v in _atl03_photons(len(atl03_dt)))
    lat = lat + RELEASE_SHIFT[0] + REFRACTION[0] * is_seafloor
    lon = lon + RELEASE_SHIFT[1] + REFRACTION[1] * is_seafloor
    ellipse_h = h_ph + RELEASE_SHIFT[2] + REFRACTION[2] * is_seafloor
    columns = {
        "delta_time": delta_time,
        "index_ph": index_ph.astype(np.int32),
        "class_ph": class_ph,
        "confidence": np.where(index_ph - OFFSET == SEAFLOOR_ROWS[0], 0.4, 0.9),
        "lat_ph": lat,
        "lon_ph": lon,
        "ellipse_h": ellipse_h.astype(np.float32),
        "ortho_h": (ellipse_h - GEOID).astype(np.float32),
        "surface_h": np.full(n, 1.0 if not sea_surface else 0.0, dtype=np.float32),
    }
    order = np.arange(n) if order is None else order
    with h5py.File(path, "w") as f:
        f["ancillary_data/atlas_sdp_gps_epoch"] = np.array([EPOCH])
        for name, values in columns.items():
            f.create_dataset(f"{laser}/{name}", data=values[order], chunks=(CHUNK,))
    return (index_ph - OFFSET)[order]


def _atl03_frame(atl03_dt):
    n = len(atl03_dt)
    lat, lon, h_ph = _atl03_photons(n)
    return pd.DataFrame(
        {
            "latitude": lat,
            "longitude": lon,
            "photon_height": (h_ph - GEOID).astype(np.float32),
            "delta_time": atl03_dt,
            "photon_meantide": (h_ph - GEOID).astype(np.float32),
            "photon_geoid": np.full(n, GEOID, dtype=np.float32),
            "photon_f2m": np.zeros(n, dtype=np.float32),
            "photon_tide_f2m": np.zeros(n, dtype=np.float32),
            "ph_h_classed": -1,
            "bathy_confidence": -1.0,
        }
    )


def test_atl24_labels_only_the_seafloor_photon_of_a_pulse(tmp_path, offline):
    atl03_dt = _atl03_delta_time()
    atl24_fn = tmp_path / "ATL24_20241107234251_08052501_006_01_002_01.h5"
    _write_atl24(atl24_fn, atl03_dt)
    reader = _reader(tmp_path)
    before = _atl03_frame(atl03_dt)

    df = reader.apply_atl24_classifications(
        before.copy(), str(atl24_fn), "gt1l", None, None
    )

    assert np.flatnonzero(df["ph_h_classed"] == 40).tolist() == SEAFLOOR_ROWS
    others = df.index.difference(SEAFLOOR_ROWS)
    pd.testing.assert_frame_equal(df.loc[others], before.loc[others])


def test_atl24_bathymetry_keeps_the_refraction_but_not_the_release_shift(
    tmp_path, offline
):
    # ATL24 has every photon displaced by RELEASE_SHIFT, and the seafloor photons by
    # REFRACTION as well. Only the refraction belongs in the output.
    atl03_dt = _atl03_delta_time()
    atl24_fn = tmp_path / "ATL24_20241107234251_08052501_006_01_002_01.h5"
    _write_atl24(atl24_fn, atl03_dt)
    before = _atl03_frame(atl03_dt).loc[SEAFLOOR_ROWS]

    df = _reader(tmp_path).apply_atl24_classifications(
        _atl03_frame(atl03_dt), str(atl24_fn), "gt1l", None, None
    )
    after = df.loc[SEAFLOOR_ROWS]

    assert after["latitude"].to_numpy() == pytest.approx(
        before["latitude"].to_numpy() + REFRACTION[0], abs=1e-10
    )
    assert after["longitude"].to_numpy() == pytest.approx(
        before["longitude"].to_numpy() + REFRACTION[1], abs=1e-10
    )
    assert after["photon_height"].to_numpy() == pytest.approx(
        before["photon_height"].to_numpy() + REFRACTION[2], abs=1e-4
    )


def test_atl24_positions_are_kept_when_nothing_measures_the_shift(tmp_path, offline):
    # No sea surface photons here, so nothing ATL24 left un-refracted to compare.
    atl03_dt = _atl03_delta_time()
    atl24_fn = tmp_path / "ATL24_20241107234251_08052501_006_01_002_01.h5"
    _write_atl24(atl24_fn, atl03_dt, sea_surface=False)
    before = _atl03_frame(atl03_dt).loc[SEAFLOOR_ROWS]

    df = _reader(tmp_path).apply_atl24_classifications(
        _atl03_frame(atl03_dt), str(atl24_fn), "gt1l", None, None
    )

    assert df.loc[SEAFLOOR_ROWS, "latitude"].to_numpy() == pytest.approx(
        before["latitude"].to_numpy() + RELEASE_SHIFT[0] + REFRACTION[0], abs=1e-10
    )


def test_release_shift_follows_a_drift_along_the_track():
    times = np.linspace(100.0, 110.0, 2001)
    drift = 0.5 + 0.01 * (times - 100.0)
    at = np.array([102.0, 105.0, 108.0])

    shift = _atl24_release_shift(times, {"h": drift}, at)

    assert shift["h"] == pytest.approx(0.5 + 0.01 * (at - 100.0), abs=1e-3)


def test_release_shift_widens_its_window_where_photons_are_sparse():
    # Five reference photons at t=100 and five at t=103. No one-second window
    # holds ten, so every estimate has to grow until it spans both groups, and
    # each then sees the same ten photons.
    times = np.concatenate((np.full(5, 100.0), np.full(5, 103.0)))
    values = np.concatenate((np.full(5, 0.4), np.full(5, 0.6)))
    at = np.array([99.0, 100.0, 101.5, 103.0, 104.0])

    shift = _atl24_release_shift(times, {"h": values}, at)

    assert shift["h"] == pytest.approx(np.full(5, 0.5))


def test_release_shift_stays_local_where_photons_are_plentiful():
    # Twenty photons per second with a step in the shift at t=105: an estimate
    # on either side sees only its own side.
    times = np.linspace(100.0, 110.0, 201)
    values = np.where(times < 105.0, 0.2, 0.8)
    at = np.array([102.0, 108.0])

    shift = _atl24_release_shift(times, {"h": values}, at)

    assert shift["h"] == pytest.approx([0.2, 0.8])


def test_release_shift_needs_enough_photons():
    times = np.linspace(100.0, 100.5, 9)

    assert _atl24_release_shift(times, {"h": np.ones(9)}, times) is None


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


def test_atl24_file_that_cannot_be_read_changes_nothing(tmp_path, offline):
    atl03_dt = _atl03_delta_time()
    atl24_fn = tmp_path / "ATL24_20241107234251_08052501_006_01_002_01.h5"
    atl24_fn.write_bytes(b"not an HDF5 file")
    before = _atl03_frame(atl03_dt)

    after = _reader(tmp_path).apply_atl24_classifications(
        before.copy(), str(atl24_fn), "gt1l", None, None
    )

    pd.testing.assert_frame_equal(after, before)


def test_atl24_file_missing_a_dataset_changes_nothing(tmp_path, offline):
    atl03_dt = _atl03_delta_time()
    atl24_fn = tmp_path / "ATL24_20241107234251_08052501_006_01_002_01.h5"
    _write_atl24(atl24_fn, atl03_dt)
    with h5py.File(atl24_fn, "a") as f:
        del f["gt1l/confidence"]
    before = _atl03_frame(atl03_dt)

    after = _reader(tmp_path).apply_atl24_classifications(
        before.copy(), str(atl24_fn), "gt1l", None, None
    )

    pd.testing.assert_frame_equal(after, before)


def test_atl24_errors_in_the_join_itself_are_logged_with_a_traceback(
    tmp_path, offline, cap_globato
):
    # A frame without the columns the join needs is a caller's mistake. The
    # granule still comes through, without bathymetry, but the error and where
    # it happened go on record.
    atl03_dt = _atl03_delta_time()
    atl24_fn = tmp_path / "ATL24_20241107234251_08052501_006_01_002_01.h5"
    _write_atl24(atl24_fn, atl03_dt)
    before = _atl03_frame(atl03_dt).drop(columns=["photon_meantide"])

    with cap_globato.at_level(
        logging.WARNING, logger="globato.streams.readers.icesat2"
    ):
        after = _reader(tmp_path).apply_atl24_classifications(
            before.copy(), str(atl24_fn), "gt1l", None, None
        )

    pd.testing.assert_frame_equal(after, before)
    (record,) = [r for r in cap_globato.records if r.exc_info is not None]
    assert record.exc_info[0] is KeyError


def _span(atl03_dt):
    times = _as_atl24_time(atl03_dt, EPOCH)
    return times.min(), times.max()


def test_atl24_block_covers_the_atl03_span(tmp_path):
    atl03_dt = _atl03_delta_time()
    atl24_fn = tmp_path / "atl24.h5"
    _write_atl24(atl24_fn, atl03_dt, outside=1000)
    first, last = _span(atl03_dt)

    with h5py.File(atl24_fn) as f:
        everything = f["gt1l/delta_time"][...]
        start, block = _read_atl24_block(f["gt1l/delta_time"], first, last)

    wanted = np.flatnonzero((everything >= first) & (everything <= last))
    assert start <= wanted[0] and wanted[-1] < start + len(block)
    assert block.tolist() == everything[start : start + len(block)].tolist()
    assert len(block) < len(everything) / 10


def test_atl24_block_is_widened_until_both_ends_pass(tmp_path):
    # 992 photons before the ones in the file puts the first of them on a chunk
    # boundary, so the chunk it is in does not show where the range begins.
    atl03_dt = _atl03_delta_time()
    atl24_fn = tmp_path / "atl24.h5"
    _write_atl24(atl24_fn, atl03_dt, outside=992)
    first, last = _span(atl03_dt)

    with h5py.File(atl24_fn) as f:
        start, block = _read_atl24_block(f["gt1l/delta_time"], first, last)

    assert start == 992 - CHUNK
    assert block[0] < first and block[-1] > last


def test_atl24_block_is_refused_when_not_in_time_order(tmp_path):
    atl03_dt = _atl03_delta_time()
    atl24_fn = tmp_path / "atl24.h5"
    n = 2 * 1000 + len(_atl24_beam(atl03_dt)[0])
    _write_atl24(atl24_fn, atl03_dt, outside=1000, order=np.arange(n)[::-1])

    with h5py.File(atl24_fn) as f:
        assert _read_atl24_block(f["gt1l/delta_time"], *_span(atl03_dt)) is None


@pytest.mark.parametrize("in_time_order", [True, False])
def test_atl24_in_a_long_file_labels_the_same_photons(tmp_path, offline, in_time_order):
    atl03_dt = _atl03_delta_time()
    atl24_fn = tmp_path / "ATL24_20241107234251_08052501_006_01_002_01.h5"
    n = 2 * 1000 + len(_atl24_beam(atl03_dt)[0])
    order = None if in_time_order else np.arange(n)[::-1]
    _write_atl24(atl24_fn, atl03_dt, outside=1000, order=order)
    reader = _reader(tmp_path)

    df = reader.apply_atl24_classifications(
        _atl03_frame(atl03_dt), str(atl24_fn), "gt1l", None, None
    )

    assert np.flatnonzero(df["ph_h_classed"] == 40).tolist() == SEAFLOOR_ROWS


# ---------------------------------------------------------------------------
# Per-segment values spread over photons
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("short_by", [0, 7])
def test_per_photon_values_and_ordinals_follow_the_segment_counts(short_by):
    # Segment photon counts including empty segments; the file may hold fewer
    # photons than the counts add up to.
    rng = np.random.default_rng(1)
    seg_ph_cnt = rng.integers(0, 6, size=40).astype(np.int32)
    seg_id = 1_000_000 + np.arange(len(seg_ph_cnt))
    values = rng.normal(size=len(seg_ph_cnt)).astype(np.float32)
    n = int(seg_ph_cnt.sum()) - short_by

    # The definitions these replace: a dict lookup per photon, an arange per segment.
    ph_seg_ids = np.repeat(seg_id, seg_ph_cnt)[:n]
    by_segment = dict(zip(seg_id, values))
    expected_values = np.array([by_segment[s] for s in ph_seg_ids])
    expected_ordinals = np.concatenate([np.arange(1, c + 1) for c in seg_ph_cnt])[:n]

    got = _per_photon(values, seg_ph_cnt, n)
    assert np.array_equal(got, expected_values) and got.dtype == np.float32
    ordinals = _photon_index_within_segment(seg_ph_cnt, n)
    assert np.array_equal(ordinals, expected_ordinals)


def _atl08_frame(seg_id, seg_ph_cnt):
    return pd.DataFrame(
        {
            "ph_segment_id": np.repeat(seg_id, seg_ph_cnt),
            "ph_h_classed": -1,
        }
    )


def test_atl08_classes_land_on_their_photons(tmp_path):
    seg_id = np.array([500, 501, 503, 504])  # ATL03 has no segment 502
    seg_ph_cnt = np.array([3, 2, 4, 1])
    seg_starts = np.concatenate(([0], np.cumsum(seg_ph_cnt)[:-1]))
    df = _atl08_frame(seg_id, seg_ph_cnt)

    # (segment, ordinal within it, class): segment 502 is not in the file.
    rows = [(500, 2, 1), (501, 1, 2), (502, 1, 3), (503, 4, 1), (504, 1, 3)]
    atl08 = tmp_path / "ATL08_test.h5"
    with h5py.File(atl08, "w") as f:
        g = f.create_group("gt1l/signal_photons")
        g["ph_segment_id"] = np.array([r[0] for r in rows])
        g["classed_pc_indx"] = np.array([r[1] for r in rows], dtype=np.int32)
        g["classed_pc_flag"] = np.array([r[2] for r in rows], dtype=np.int8)

    reader = ATL03Reader(str(tmp_path / ATL03), cache_dir=str(tmp_path))
    out = reader.apply_atl08_classifications(df, str(atl08), "gt1l", seg_id, seg_starts)

    expected = np.full(len(df), -1)
    expected[0 + 1] = 1  # segment 500, 2nd photon
    expected[3 + 0] = 2  # segment 501, 1st photon
    expected[5 + 3] = 1  # segment 503, 4th photon
    expected[9 + 0] = 3  # segment 504, 1st photon
    assert np.array_equal(out["ph_h_classed"].to_numpy(), expected)


# ---------------------------------------------------------------------------
# External masks
# ---------------------------------------------------------------------------
@pytest.fixture
def mask_cache():
    icesat2._MASK_TREE_CACHE.clear()
    yield icesat2._MASK_TREE_CACHE
    icesat2._MASK_TREE_CACHE.clear()


def _polygons(rng, n, big=False):
    """Random polygons: big ones are jagged with many vertices, like a coastline."""
    polys = []
    for _ in range(n):
        cx, cy = rng.uniform(-79.9, -79.1), rng.uniform(25.1, 25.9)
        k = 400 if big else 6
        angles = np.sort(rng.uniform(0, 2 * np.pi, k))
        radius = (0.15 if big else 0.02) * (1 + 0.6 * rng.uniform(-1, 1, k))
        polys.append(
            shapely.polygons(
                np.column_stack(
                    (cx + radius * np.cos(angles), cy + radius * np.sin(angles))
                )
            )
        )
    return polys


@pytest.mark.parametrize("prepared", [False, True])
def test_points_in_tree_agrees_with_the_predicate_query(prepared):
    rng = np.random.default_rng(3)
    geoms = _polygons(rng, 5, big=True) + _polygons(rng, 30)
    if prepared:
        shapely.prepare(geoms)
    tree = STRtree(geoms)
    x = rng.uniform(-80.0, -79.0, 20_000)
    y = rng.uniform(25.0, 26.0, 20_000)

    expected = np.zeros(len(x), dtype=bool)
    expected[tree.query(shapely.points(x, y), predicate="intersects")[0]] = True

    assert np.array_equal(_points_in_tree(tree, x, y), expected)
    assert expected.any() and not expected.all()


def test_points_in_tree_with_nothing_to_hit():
    tree = STRtree([shapely.box(0, 0, 1, 1)])
    assert not _points_in_tree(tree, [5.0, 6.0], [5.0, 6.0]).any()
    assert len(_points_in_tree(tree, [], [])) == 0
    assert len(_points_in_tree(None, [0.5], [0.5])) == 1


def _write_geojson(path, boxes):
    features = [
        {
            "type": "Feature",
            "properties": {"height": 3.0},
            "geometry": json.loads(shapely.to_geojson(shapely.box(*b))),
        }
        for b in boxes
    ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return str(path)


REGION = "-80.0/-79.0/25.0/26.0"


def test_building_tree_holds_the_footprints_of_every_file(tmp_path, monkeypatch):
    # Two files, as two Bing quadkey tiles; one footprint lies outside the region.
    first = _write_geojson(tmp_path / "a.geojson", [(-79.9, 25.1, -79.8, 25.2)])
    second = _write_geojson(
        tmp_path / "b.geojson",
        [(-79.5, 25.5, -79.4, 25.6), (-78.5, 25.5, -78.4, 25.6)],
    )
    monkeypatch.setattr(fetchez, "get", lambda *a, **k: [first, second])

    tree = _reader(tmp_path, region=REGION)._build_bldg_tree("bing")

    assert len(tree.geometries) == 2
    assert not shapely.is_prepared(tree.geometries).any()


def test_land_tree_is_prepared(tmp_path, monkeypatch):
    path = _write_geojson(tmp_path / "land.geojson", [(-79.9, 25.1, -79.5, 25.9)])
    monkeypatch.setattr(fetchez, "get", lambda *a, **k: [path])

    tree = _reader(tmp_path, region=REGION)._build_land_tree()

    assert shapely.is_prepared(tree.geometries).all()
    assert _points_in_tree(tree, [-79.7, -79.2], [25.5, 25.5]).tolist() == [True, False]


def test_mask_trees_are_built_once_per_region(tmp_path, monkeypatch, mask_cache):
    path = _write_geojson(tmp_path / "land.geojson", [(-79.9, 25.1, -79.5, 25.9)])
    calls = []

    def fake_get(module, **kwargs):
        calls.append((module, tuple(kwargs["region"])))
        return [path]

    monkeypatch.setattr(fetchez, "get", fake_get)

    one = _reader(tmp_path, region=REGION)
    two = _reader(tmp_path, "ATL03_other_subsetted.h5", region=REGION)
    assert one._get_land_tree() is two._get_land_tree()
    assert one._get_bldg_tree("bing") is two._get_bldg_tree("bing")
    assert len(calls) == 2  # one landmask fetch, one building fetch

    elsewhere = _reader(tmp_path, region="-81.0/-80.0/25.0/26.0")
    assert elsewhere._get_land_tree() is not one._get_land_tree()
    assert len(calls) == 3


def test_only_the_most_recent_mask_trees_are_kept(tmp_path, monkeypatch, mask_cache):
    path = _write_geojson(tmp_path / "land.geojson", [(-79.9, 25.1, -79.5, 25.9)])
    monkeypatch.setattr(fetchez, "get", lambda *a, **k: [path])
    monkeypatch.setattr(icesat2, "_MASK_TREE_CACHE_SIZE", 2)

    first = _reader(tmp_path, region="-80.0/-79.0/25.0/26.0")._get_land_tree()
    for w in (-81.0, -82.0):
        _reader(tmp_path, region=f"{w}/{w + 1}/25.0/26.0")._get_land_tree()

    assert len(mask_cache) == 2
    assert (
        _reader(tmp_path, region="-80.0/-79.0/25.0/26.0")._get_land_tree() is not first
    )


def test_prebuilt_trees_are_used_as_given(tmp_path, monkeypatch, offline):
    def no_fetch(*args, **kwargs):
        raise AssertionError("a tree was rebuilt")

    monkeypatch.setattr(fetchez, "get", no_fetch)
    tree = STRtree([shapely.box(-79.9, 25.1, -79.5, 25.9)])
    reader = _reader(
        tmp_path,
        region=REGION,
        classes="1",
        use_external_masks=True,
        bldg_tree=tree,
        land_tree=tree,
    )

    # Past the masks, the empty file fails to open as HDF5.
    with pytest.raises(OSError):
        list(reader.yield_chunks())
