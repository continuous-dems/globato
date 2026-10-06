import numpy as np
import pytest
from fetchez.registry import ProfileRegistry, ReaderRegistry

from globato.streams.readers.xyz import XYZReader

NOS_ROWS = """survey_id,lat,long,depth,quality_code,active
H00001,33.0,-119.0,10.0,1,1
H00001,33.1,-119.1,20.0,17,0
H00001,33.2,-119.2,30.0,1,1
H00001,33.3,-119.3,40.0,2,0
"""


@pytest.fixture
def nos_file(tmp_path):
    src = tmp_path / "H00001.xyz"
    src.write_text(NOS_ROWS)
    return str(src)


def _z(reader):
    chunks = list(reader.yield_chunks())
    return np.concatenate(chunks)["z"] if chunks else np.empty(0)


def test_xyz_reads_every_row_by_default(nos_file):
    reader = XYZReader(nos_file, delimiter=",", skiprows=1, usecols=[2, 1, 3])
    assert np.allclose(_z(reader), [10.0, 20.0, 30.0, 40.0])


@pytest.mark.parametrize("chunk_size", [1, 100_000])
def test_xyz_keep_rows(nos_file, chunk_size):
    reader = XYZReader(
        nos_file,
        delimiter=",",
        skiprows=1,
        usecols=[2, 1, 3],
        keep_pos=5,
        keep_values=[1],
        chunk_size=chunk_size,
    )
    assert np.allclose(_z(reader), [10.0, 30.0])


def test_xyz_keep_values_from_cli_string(nos_file):
    reader = XYZReader(
        nos_file,
        delimiter=",",
        skiprows=1,
        usecols=[2, 1, 3],
        keep_pos=4,
        keep_values="2/17",
    )
    assert np.allclose(_z(reader), [20.0, 40.0])


def test_nos_xyz_profile_drops_inactive(nos_file):
    ReaderRegistry.load_all()
    ProfileRegistry.load_all()
    reader = ReaderRegistry.get_reader(nos_file, "nos-xyz")
    assert np.allclose(_z(reader), [-10.0, -30.0])


@pytest.mark.parametrize("override", [{"keep_pos": None}, {"keep_pos": "None"}])
def test_nos_xyz_profile_keep_inactive(nos_file, override):
    ReaderRegistry.load_all()
    ProfileRegistry.load_all()
    reader = ReaderRegistry.get_reader(nos_file, "nos-xyz", **override)
    assert np.allclose(_z(reader), [-10.0, -20.0, -30.0, -40.0])
