import json

import geopandas as gpd
import pytest

from fetchez.spatial import Region
from globato.modules import osm_landmask as om

REGION = [0.0, 1.0, 0.0, 1.0]  # w, e, s, n


def _way(way_id, coords, tags):
    return {
        "type": "way",
        "id": way_id,
        "tags": tags,
        "geometry": [{"lon": x, "lat": y} for x, y in coords],
    }


def _ring(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def _osm_reply(path):
    """A lake whose outline crosses itself, plus a water relation with an inner
    ring (an island), so the island subtraction runs over the invalid lake."""
    bowtie_lake = [(0.1, 0.1), (0.3, 0.3), (0.3, 0.1), (0.1, 0.3), (0.1, 0.1)]
    relation = {
        "type": "relation",
        "id": 10,
        "tags": {"natural": "water"},
        "members": [
            {
                "type": "way",
                "ref": 2,
                "role": "outer",
                "geometry": [
                    {"lon": x, "lat": y} for x, y in _ring(0.5, 0.5, 0.9, 0.9)
                ],
            },
            {
                "type": "way",
                "ref": 3,
                "role": "inner",
                "geometry": [
                    {"lon": x, "lat": y} for x, y in _ring(0.6, 0.6, 0.7, 0.7)
                ],
            },
        ],
    }
    data = {"elements": [_way(1, bowtie_lake, {"natural": "water"}), relation]}
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture
def module(tmp_path, monkeypatch):
    monkeypatch.setattr(
        om.OSMLandmaskModule, "_is_land_by_gmrt", lambda self, geom: True
    )

    def no_fallback(self, dst_file, region):
        raise AssertionError("fell back instead of building the mask")

    monkeypatch.setattr(om.OSMLandmaskModule, "_handle_fallback", no_fallback)
    w, e, s, n = REGION
    return om.OSMLandmaskModule(
        src_region=Region(w=w, e=e, s=s, n=n),
        outdir=str(tmp_path),
        include_lakes=True,
        output_mode="topology",
    )


def test_self_intersecting_water_polygon_is_repaired(module, tmp_path):
    out = tmp_path / "mask.geojson"
    module._polygonize(_osm_reply(tmp_path / "osm.json"), str(out), REGION)

    mask = gpd.read_file(out)
    assert mask.is_valid.all()
    lakes = mask[mask["class"] == "lake"]
    # both lobes of the bowtie survive the repair, next to the relation's lake
    assert lakes.intersects(gpd.points_from_xy([0.15, 0.25], [0.2, 0.2])[0]).any()
    assert lakes.intersects(gpd.points_from_xy([0.25], [0.2])[0]).any()


def test_temp_download_name_uses_north_edge(module, monkeypatch):
    names = []

    class FakeFetch:
        def __init__(self, url, headers=None):
            pass

        def fetch_file(self, dest, **kwargs):
            names.append(dest)
            return 1

    monkeypatch.setattr(om, "Fetch", FakeFetch)
    module._fetch_osm([0.0, 1.0, 0.25, 0.75])
    assert names[0].split("/")[-1].startswith("temp_osm_0.0_0.25_1.0_0.75")
