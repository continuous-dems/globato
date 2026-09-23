import os
from types import SimpleNamespace

import pytest
import shapely

from globato.hooks.filters import tnm_wesm


def _entry(project="CA_Orange_County_2016", bounds=(0, 1, 0, 1)):
    return {
        "tnm_product": "1m",
        "tnm_project": project,
        "bounds": bounds,
        "url": f"https://example.test/Projects/{project}/tile.tif",
    }


def test_provider_project_name_is_preferred():
    entry = _entry()
    assert tnm_wesm.project_name(entry) == "CA_Orange_County_2016"


def test_directionless_wesm_alias_is_accepted_when_unambiguous():
    aliases = {"CA_SanDiegoCo_2016": {"CA_SanDiegoCo_2016"}}
    row = {"workunit": "CA_E_SanDiegoCo_2016", "project": None, "sourcedem_link": None}
    assert tnm_wesm._row_project(row, aliases) == "CA_SanDiegoCo_2016"


def test_ambiguous_alias_fails_closed():
    aliases = {"a": {"CA_E_Test_2020"}, "b": {"CA_W_Test_2020"}}
    row = {"workunit": "CA_Test_2020", "project": None, "sourcedem_link": None}
    with pytest.raises(RuntimeError, match="multiple TNM projects"):
        tnm_wesm._row_project(row, aliases)


def test_collection_year_uses_end_then_start():
    assert (
        tnm_wesm.collection_year({"collect_end": "2024-06-01", "collect_start": "2023"})
        == 2024
    )
    assert (
        tnm_wesm.collection_year({"collect_end": None, "collect_start": "2019-01-01"})
        == 2019
    )


def test_hook_only_targets_one_meter_and_preserves_other_products(monkeypatch):
    one = _entry()
    coarse = {"tnm_product": "1_3as", "bounds": (0, 1, 0, 1)}
    mod = SimpleNamespace(wgs_region=(0, 1, 0, 1))

    def fake(entries, region, require_year=False):
        assert entries == [one]
        assert require_year is True
        one["tnm_source_coverage"] = [
            {
                "geometry": shapely.to_wkt(shapely.box(0, 0, 1, 1)),
                "year": 2024,
                "project": "WESM Project",
            }
        ]
        return entries

    monkeypatch.setattr(tnm_wesm.WESM, "add_source_coverage", fake)
    result = tnm_wesm.TNMWESMCoverage().run([(mod, one), (mod, coarse)])
    assert [entry["tnm_product"] for _, entry in result] == ["1m", "1_3as"]
    assert one["tnm_source_coverage"][0]["year"] == 2024


def test_hook_drops_one_meter_entry_known_not_to_intersect(monkeypatch):
    one = _entry()
    mod = SimpleNamespace(wgs_region=(0, 1, 0, 1))
    monkeypatch.setattr(tnm_wesm.WESM, "add_source_coverage", lambda *a, **k: [])
    assert tnm_wesm.TNMWESMCoverage().run([(mod, one)]) == []


def test_hook_fails_closed_when_wesm_matching_fails(monkeypatch):
    one = _entry()
    mod = SimpleNamespace(wgs_region=(0, 1, 0, 1))

    def fail(*args, **kwargs):
        raise RuntimeError("USGS WESM has no matching work-unit identity")

    monkeypatch.setattr(tnm_wesm.WESM, "add_source_coverage", fail)
    with pytest.raises(RuntimeError, match="no matching work-unit"):
        tnm_wesm.TNMWESMCoverage().run([(mod, one)])


def test_wesm_gdal_environment_is_restored(monkeypatch):
    monkeypatch.setenv("AWS_NO_SIGN_REQUEST", "NO")
    monkeypatch.delenv("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", raising=False)
    with tnm_wesm.WESM._gdal_env():
        assert os.environ["AWS_NO_SIGN_REQUEST"] == "YES"
        assert os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] == ".gpkg"
        with tnm_wesm.WESM._gdal_env():
            assert os.environ["AWS_NO_SIGN_REQUEST"] == "YES"
        assert os.environ["AWS_NO_SIGN_REQUEST"] == "YES"
    assert os.environ["AWS_NO_SIGN_REQUEST"] == "NO"
    assert "CPL_VSIL_CURL_ALLOWED_EXTENSIONS" not in os.environ


def test_retryable_remote_gpkg_read_retries_with_backoff(monkeypatch):
    attempts = []
    sleeps = []

    def fail(*args, **kwargs):
        attempts.append((args, kwargs))
        raise RuntimeError("database disk image is malformed")

    monkeypatch.setattr(tnm_wesm, "read", fail)
    monkeypatch.setattr(tnm_wesm.time, "sleep", sleeps.append)

    with pytest.raises(
        RuntimeError, match="Unable to read USGS WESM GeoPackage geometry"
    ):
        tnm_wesm.WESM._read()

    assert len(attempts) == tnm_wesm.WESM_RETRIES
    assert sleeps == [1, 2, 4, 8]


def test_nonretryable_gpkg_read_fails_closed_without_sleep(monkeypatch):
    attempts = []
    sleeps = []

    def fail(*args, **kwargs):
        attempts.append((args, kwargs))
        raise RuntimeError("layer does not exist")

    monkeypatch.setattr(tnm_wesm, "read", fail)
    monkeypatch.setattr(tnm_wesm.time, "sleep", sleeps.append)

    with pytest.raises(
        RuntimeError, match="Unable to read USGS WESM GeoPackage geometry"
    ):
        tnm_wesm.WESM._read()

    assert len(attempts) == 1
    assert sleeps == []
