from types import SimpleNamespace

import pytest
import shapely

from globato.hooks.filters.tnm_policy import (
    TNMPolicy,
    TNM_CANONICAL_ORDER,
    TNM_PRODUCT_POLICY,
)
from fetchez.hooks.spatial_claim import SpatialClaimHook


def _claim(geometry, year=None, project=None, workunit=None):
    value = {"geometry": shapely.to_wkt(geometry)}
    if year is not None:
        value["year"] = year
    if project is not None:
        value["project"] = project
    if workunit is not None:
        value["workunit"] = workunit
    return value


def _entry(product, bounds, *, project=None, coverage=None, geometry=None):
    value = {
        "tnm_product": product,
        "bounds": bounds,
        "metadata": {},
    }
    if geometry is not None:
        value["geometry"] = geometry
        footprint_hook = {
            "s1m": "remote_raster_footprint",
            "1_9as": "remote_archive_footprint",
            "5m": "remote_raster_footprint",
        }.get(product)
        if footprint_hook:
            value["history"] = [{"hook": footprint_hook, "stage": "manifest"}]
    if project is not None:
        value["tnm_project"] = project
    if coverage is not None:
        value["tnm_source_coverage"] = coverage
    return value


def _geom(entry, key="claim_geometry"):
    return shapely.from_wkt(entry[key])


def test_canonical_policy_matches_approved_weight_classes():
    assert TNM_CANONICAL_ORDER == (
        "s1m",
        "1m",
        "1_9as",
        "5m",
        "1_3as",
        "1_as",
        "2_as",
    )
    assert {key: value.weight for key, value in TNM_PRODUCT_POLICY.items()} == {
        "s1m": 10.0,
        "1m": 5.0,
        "1_9as": 3.0,
        "5m": 3.0,
        "1_3as": 1.0,
        "1_as": 0.5,
        "2_as": 0.25,
    }
    assert TNM_PRODUCT_POLICY["1_9as"].rank > TNM_PRODUCT_POLICY["5m"].rank


def test_newer_one_meter_project_supersedes_older_overlap():
    old = _entry(
        "1m",
        (0, 2, 0, 1),
        project="old",
        coverage=[_claim(shapely.box(0, 0, 2, 1), 2018, "Old Project")],
    )
    new = _entry(
        "1m",
        (1, 3, 0, 1),
        project="new",
        coverage=[_claim(shapely.box(1, 0, 3, 1), 2024, "New Project")],
    )

    rows = TNMPolicy().run([(SimpleNamespace(), old), (SimpleNamespace(), new)])
    SpatialClaimHook().run(rows)

    assert _geom(old).equals(shapely.box(0, 0, 2, 1))
    assert _geom(new).equals(shapely.box(1, 0, 3, 1))
    assert shapely.from_wkt(old["excluded_geometry"]).equals(shapely.box(1, 0, 2, 1))
    assert (
        _geom(old, "excluded_geometry")
        .intersection(_geom(old, "accepted_geometry"))
        .area
        == 0
    )
    assert _geom(old).symmetric_difference(
        _geom(old, "accepted_geometry").union(_geom(old, "excluded_geometry"))
    ).area == pytest.approx(0)
    assert old["metadata"]["dataset"] == "TNM 1 m: Old Project"
    assert new["metadata"]["dataset"] == "TNM 1 m: New Project"


def test_same_year_projects_are_not_ranked():
    left = _entry(
        "1m",
        (0, 2, 0, 1),
        project="left",
        coverage=[_claim(shapely.box(0, 0, 2, 1), 2024, "Left")],
    )
    right = _entry(
        "1m",
        (1, 3, 0, 1),
        project="right",
        coverage=[_claim(shapely.box(1, 0, 3, 1), 2024, "Right")],
    )

    TNMPolicy().run([(SimpleNamespace(), left), (SimpleNamespace(), right)])

    assert _geom(left).intersection(_geom(right)).area == pytest.approx(1.0)
    assert "excluded_geometry" not in left
    assert "excluded_geometry" not in right


def test_product_hierarchy_uses_authoritative_source_footprints_not_valid_data():
    s1m = _entry(
        "s1m",
        (0, 2, 0, 1),
        geometry=shapely.box(0, 0, 2, 1),
    )
    one_ninth = _entry(
        "1_9as",
        (0, 4, 0, 1),
        project="p",
        geometry=shapely.box(0, 0, 4, 1),
    )
    one_third = _entry("1_3as", (0, 5, 0, 1))

    entries = TNMPolicy().run(
        [
            (SimpleNamespace(), one_third),
            (SimpleNamespace(), one_ninth),
            (SimpleNamespace(), s1m),
        ]
    )
    SpatialClaimHook().run(entries)

    assert shapely.from_wkt(s1m["accepted_geometry"]).equals(shapely.box(0, 0, 2, 1))
    assert shapely.from_wkt(one_ninth["accepted_geometry"]).equals(
        shapely.box(2, 0, 4, 1)
    )
    assert shapely.from_wkt(one_third["accepted_geometry"]).equals(
        shapely.box(4, 0, 5, 1)
    )


def test_one_meter_chronology_exclusion_survives_higher_product_claim():
    s1m = _entry(
        "s1m",
        (1.5, 2.5, 0, 1),
        geometry=shapely.box(1.5, 0, 2.5, 1),
    )
    old = _entry(
        "1m",
        (0, 2, 0, 1),
        project="old",
        coverage=[_claim(shapely.box(0, 0, 2, 1), 2018, "Old")],
    )
    new = _entry(
        "1m",
        (1, 3, 0, 1),
        project="new",
        coverage=[_claim(shapely.box(1, 0, 3, 1), 2024, "New")],
    )

    entries = TNMPolicy().run(
        [(SimpleNamespace(), old), (SimpleNamespace(), s1m), (SimpleNamespace(), new)]
    )
    SpatialClaimHook().run(entries)

    # Old project was already excluded by New over 1..2. That history must not
    # be lost when S1M also claims 1.5..2.5.
    excluded = shapely.from_wkt(old["excluded_geometry"])
    assert excluded.covers(shapely.box(1, 0, 2, 1))
    assert shapely.from_wkt(old["accepted_geometry"]).equals(shapely.box(0, 0, 1, 1))


def test_standard_tnm_provenance_and_entry_weight_are_set():
    entry = _entry("1_as", (0, 1, 0, 1))
    TNMPolicy().run([(SimpleNamespace(), entry)])
    assert entry["weight"] == 0.5
    assert entry["tnm_resolution_tier"] == "1_as"
    assert entry["claim_product"] == "1_as"
    assert entry["metadata"] == {
        "category": "elevation",
        "agency": "USGS",
        "dataset": "TNM 1 arc-second",
        "resolution": "1 arc-second",
        "weight": 0.5,
    }
    assert entry["source_mask_group_by"] == "MODULE/DATASET/WEIGHT"


def test_products_requiring_authoritative_coverage_fail_closed():
    entry = _entry("1_9as", (0, 1, 0, 1), project="p")
    with pytest.raises(
        RuntimeError, match="requires completed remote_archive_footprint"
    ):
        TNMPolicy().run([(SimpleNamespace(), entry)])


def test_non_tnm_entries_pass_through():
    other = {"title": "other"}
    result = TNMPolicy().run([(SimpleNamespace(), other)])
    assert result[0][1] is other


def test_claims_are_clipped_to_module_wgs_roi():
    entry = _entry(
        "1_9as",
        (0, 4, 0, 1),
        project="p",
        geometry=shapely.box(0, 0, 4, 1),
    )
    module = SimpleNamespace(wgs_region=SimpleNamespace(w=1, e=3, s=0, n=1))
    TNMPolicy().run([(module, entry)])
    assert _geom(entry).equals(shapely.box(1, 0, 3, 1))


def test_wesm_dataset_name_uses_project_not_workunit():
    entry = _entry(
        "1m",
        (0, 1, 0, 1),
        project="provider-project",
        coverage=[_claim(shapely.box(0, 0, 1, 1), 2024, "WESM Project", "Workunit 17")],
    )
    TNMPolicy().run([(SimpleNamespace(), entry)])
    assert entry["metadata"]["dataset"] == "TNM 1 m: WESM Project"


def test_generic_footprint_geometry_can_be_wkt_or_shapely():
    a = _entry("s1m", (0, 2, 0, 1), geometry=shapely.box(0, 0, 1.5, 1))
    b = _entry("5m", (0, 3, 0, 1), geometry=shapely.to_wkt(shapely.box(0, 0, 3, 1)))
    result = TNMPolicy().run([(SimpleNamespace(), a), (SimpleNamespace(), b)])
    SpatialClaimHook().run(result)
    assert shapely.from_wkt(a["accepted_geometry"]).equals(shapely.box(0, 0, 1.5, 1))
    assert shapely.from_wkt(b["accepted_geometry"]).equals(shapely.box(1.5, 0, 3, 1))


def test_metadata_weight_matches_module_times_entry_weight():
    entry = _entry(
        "1m",
        (0, 1, 0, 1),
        project="p",
        coverage=[_claim(shapely.box(0, 0, 1, 1), 2024, "P")],
    )
    TNMPolicy().run([(SimpleNamespace(weight=2.0), entry)])
    assert entry["weight"] == 5.0
    assert entry["metadata"]["weight"] == 10.0


def test_entry_outside_module_roi_is_dropped_before_claim_policy_is_marked():
    entry = _entry("1_3as", (0, 1, 0, 1))
    module = SimpleNamespace(wgs_region=SimpleNamespace(w=2, e=3, s=0, n=1))

    result = TNMPolicy().run([(module, entry)])

    assert result == []
    assert "claim_geometry" not in entry
    assert "claim_priority" not in entry
    assert "claim_required" not in entry


def test_authoritative_footprint_outside_source_bounds_is_dropped_before_claim_policy():
    entry = _entry(
        "1_9as",
        (0, 1, 0, 1),
        project="p",
        geometry=shapely.box(2, 0, 3, 1),
    )

    result = TNMPolicy().run([(SimpleNamespace(), entry)])

    assert result == []
    assert "claim_geometry" not in entry
    assert "claim_priority" not in entry
    assert "claim_required" not in entry


def test_empty_discovered_entry_cannot_poison_global_spatial_claim():
    inside = _entry("1_3as", (0, 1, 0, 1))
    outside = _entry("1_3as", (3, 4, 0, 1))
    module = SimpleNamespace(wgs_region=SimpleNamespace(w=0, e=2, s=0, n=1))

    result = TNMPolicy().run([(module, inside), (module, outside)])
    claimed = SpatialClaimHook().run(result)

    assert [entry for _, entry in claimed] == [inside]
    assert _geom(inside, "accepted_geometry").equals(shapely.box(0, 0, 1, 1))
    assert "claim_priority" not in outside
    assert "claim_required" not in outside


def _seamless(product, tile, edition, left, right):
    """Dated versions of one USGS seamless tile with partly distinct coverage."""
    entry = _entry(product, (left, right, 0, 1))
    entry["url"] = f"https://example.test/USGS_{tile}_{edition}.tif"
    entry["tnm_publication_date"] = f"{edition[:4]}-{edition[4:6]}-{edition[6:]}"
    return entry


def test_seamless_versions_exclude_old_overlap_but_keep_unique_old_coverage():
    old = _seamless("1_3as", "13_n34w118", "20190917", 0, 3)
    new = _seamless("1_3as", "13_n34w118", "20260915", 1, 4)
    low = _entry("1_as", (0, 5, 0, 1))
    rows = TNMPolicy().run(
        [(SimpleNamespace(), old), (SimpleNamespace(), low), (SimpleNamespace(), new)]
    )
    rows = SpatialClaimHook().run(rows)
    assert _geom(new, "accepted_geometry").equals(shapely.box(1, 0, 4, 1))
    assert _geom(old, "accepted_geometry").equals(shapely.box(0, 0, 1, 1))
    assert _geom(old, "excluded_geometry").equals(shapely.box(1, 0, 3, 1))
    assert _geom(low, "accepted_geometry").equals(shapely.box(4, 0, 5, 1))
    assert new["claim_priority"] == old["claim_priority"] == 300
    # The claim must remain the actual original tile footprint, not the
    # edition-trimmed remainder: claim = accepted + excluded for the audit.
    assert _geom(old).equals(shapely.box(0, 0, 3, 1))
    assert (
        _geom(old, "accepted_geometry")
        .union(_geom(old, "excluded_geometry"))
        .equals(_geom(old))
    )


def test_seamless_edition_audit_partitions_original_source_claim(tmp_path):
    import json
    from shapely.geometry import shape

    old = _seamless("1_as", "1_n45w068", "20190930", 0, 3)
    new = _seamless("1_as", "1_n45w068", "20260520", 1, 4)
    high = _entry("1_9as", (2, 2.5, 0, 1), geometry=shapely.box(2, 0, 2.5, 1))
    lower = _entry("2_as", (0, 5, 0, 1))
    rows = TNMPolicy().run(
        [
            (SimpleNamespace(), old),
            (SimpleNamespace(), new),
            (SimpleNamespace(), high),
            (SimpleNamespace(), lower),
        ]
    )
    audit = tmp_path / "maine_spatial_claim.geojson"
    SpatialClaimHook(audit_output=str(audit)).run(rows)
    features = json.loads(audit.read_text())["features"]
    old_record = next(
        feature for feature in features if feature["properties"]["url"] == old["url"]
    )
    claim = shape(old_record["geometry"])
    accepted = shapely.from_wkt(old_record["properties"]["accepted_wkt"])
    excluded = shapely.from_wkt(old_record["properties"]["excluded_wkt"])
    assert claim.equals(shapely.box(0, 0, 3, 1))
    assert accepted.equals(shapely.box(0, 0, 1, 1))
    assert excluded.equals(shapely.box(1, 0, 3, 1))
    assert accepted.intersection(excluded).area == 0
    assert excluded.difference(claim).area == 0
    assert claim.difference(accepted.union(excluded)).area == 0
    assert _geom(lower, "accepted_geometry").equals(shapely.box(4, 0, 5, 1))


def test_fully_superseded_seamless_edition_is_not_streamed():
    old = _seamless("2_as", "2_n59w135", "20170407", 0, 2)
    new = _seamless("2_as", "2_n59w135", "20260611", 0, 2)
    rows = TNMPolicy().run([(SimpleNamespace(), old), (SimpleNamespace(), new)])
    assert [entry for _, entry in rows] == [new]
    assert "claim_required" not in old


def test_different_seamless_tiles_are_peers_and_one_meter_projects_stay_separate():
    a = _seamless("1_3as", "13_n34w118", "20260915", 0, 2)
    b = _seamless("1_3as", "13_n34w119", "20120201", 1, 3)
    rows = TNMPolicy().run([(SimpleNamespace(), a), (SimpleNamespace(), b)])
    SpatialClaimHook().run(rows)
    assert _geom(a, "accepted_geometry").intersection(
        _geom(b, "accepted_geometry")
    ).area == pytest.approx(1)


def test_ambiguous_same_day_seamless_editions_fail_instead_of_being_averaged():
    first = _seamless("1_3as", "13_n34w118", "20260915", 0, 2)
    second = _seamless("1_3as", "13_n34w118", "20260915", 1, 3)
    with pytest.raises(RuntimeError, match="Ambiguous same-day"):
        TNMPolicy().run([(SimpleNamespace(), first), (SimpleNamespace(), second)])


def test_repeated_tile_with_missing_publication_date_fails_closed():
    first = _seamless("1_3as", "13_n34w118", "20260915", 0, 2)
    second = _seamless("1_3as", "13_n34w118", "20190917", 1, 3)
    second.pop("tnm_publication_date")
    with pytest.raises(RuntimeError, match="valid publication date"):
        TNMPolicy().run([(SimpleNamespace(), first), (SimpleNamespace(), second)])


def test_tnm_rejects_unverified_provider_bbox_even_if_geometry_is_valid():
    entry = _entry("s1m", (0, 2, 0, 1), geometry=shapely.box(0, 0, 2, 1))
    entry.pop("history")
    with pytest.raises(
        RuntimeError, match="requires completed remote_raster_footprint"
    ):
        TNMPolicy().run([(SimpleNamespace(), entry)])
    entry["history"] = [{"hook": "remote_raster_footprint", "stage": "file"}]
    with pytest.raises(
        RuntimeError, match="requires completed remote_raster_footprint"
    ):
        TNMPolicy().run([(SimpleNamespace(), entry)])


def test_remote_archive_footprint_must_have_completed_before_tnm_policy():
    entry = _entry("1_9as", (0, 2, 0, 1), geometry=shapely.box(0, 0, 2, 1))
    entry.pop("history")
    with pytest.raises(
        RuntimeError, match="requires completed remote_archive_footprint"
    ):
        TNMPolicy().run([(SimpleNamespace(), entry)])


def _ned19(name, published, left, right, *, coverage=None):
    """Observed TNM NED 1/9 archive naming, with an authoritative archive outline."""
    entry = _entry(
        "1_9as",
        (left, right, 0, 1),
        geometry=coverage if coverage is not None else shapely.box(left, 0, right, 1),
    )
    entry["url"] = f"https://prd-tnm.s3.amazonaws.com/test/{name}"
    if published is not None:
        entry["tnm_publication_date"] = published
    return entry


def test_ned19_newest_edition_controls_overlapping_authoritative_footprint():
    old = _ned19("ned19_n33x50_w117x75_ca_orangeco_2011.zip", "2012-04-01", 0, 3)
    new = _ned19("ned19_n33x50_w117x75_ca_orangeco_2025.zip", "2026-02-01", 1, 4)
    third = _entry("1_3as", (0, 5, 0, 1))
    rows = TNMPolicy().run(
        [(SimpleNamespace(), old), (SimpleNamespace(), third), (SimpleNamespace(), new)]
    )
    SpatialClaimHook().run(rows)
    assert _geom(old).equals(shapely.box(0, 0, 3, 1))
    assert _geom(new, "accepted_geometry").equals(shapely.box(1, 0, 4, 1))
    assert _geom(old, "accepted_geometry").equals(shapely.box(0, 0, 1, 1))
    assert _geom(old, "excluded_geometry").equals(shapely.box(1, 0, 3, 1))
    assert _geom(old).symmetric_difference(
        _geom(old, "accepted_geometry").union(_geom(old, "excluded_geometry"))
    ).area == pytest.approx(0)
    assert _geom(third, "accepted_geometry").equals(shapely.box(4, 0, 5, 1))
    assert old["claim_priority"] == new["claim_priority"] == 500
    assert old["weight"] == new["weight"] == 3.0


def test_ned19_archive_footprint_not_provider_bbox_controls_edition_exclusion():
    # Both provider bounds cover 0..4. Their actual archive outlines do not.
    old = _ned19(
        "ned19_n33x50_w117x75_ca_orangeco_2011.zip",
        "2012-04-01",
        0,
        4,
        coverage=shapely.box(0, 0, 3, 1),
    )
    new = _ned19(
        "ned19_n33x50_w117x75_ca_orangeco_2025.zip",
        "2026-02-01",
        0,
        4,
        coverage=shapely.box(1, 0, 4, 1),
    )
    rows = SpatialClaimHook().run(
        TNMPolicy().run([(SimpleNamespace(), old), (SimpleNamespace(), new)])
    )
    assert len(rows) == 2
    assert _geom(old, "accepted_geometry").equals(shapely.box(0, 0, 1, 1))
    assert _geom(old, "excluded_geometry").equals(shapely.box(1, 0, 3, 1))
    assert _geom(new, "accepted_geometry").equals(shapely.box(1, 0, 4, 1))


def test_ned19_newer_publication_metadata_controls_not_filename_year():
    # The year in a filename can describe collection, not TNM publication.
    a = _ned19("ned19_n33x50_w117x75_ca_orangeco_2011.zip", "2026-04-01", 0, 2)
    b = _ned19("ned19_n33x50_w117x75_ca_orangeco_2025.zip", "2025-01-01", 1, 3)
    SpatialClaimHook().run(
        TNMPolicy().run([(SimpleNamespace(), a), (SimpleNamespace(), b)])
    )
    assert _geom(a, "accepted_geometry").equals(shapely.box(0, 0, 2, 1))
    assert _geom(b, "accepted_geometry").equals(shapely.box(2, 0, 3, 1))
    assert _geom(b, "excluded_geometry").equals(shapely.box(1, 0, 2, 1))


def test_ned19_fully_superseded_archive_is_not_streamed():
    old = _ned19("ned19_n33x50_w117x75_ca_orangeco_2011.zip", "2012-01-01", 0, 2)
    new = _ned19("ned19_n33x50_w117x75_ca_orangeco_2025.zip", "2026-01-01", 0, 2)
    rows = TNMPolicy().run([(SimpleNamespace(), old), (SimpleNamespace(), new)])
    assert [entry for _, entry in rows] == [new]
    assert "claim_required" not in old


def test_ned19_different_named_series_or_tiles_remain_peers():
    old = _ned19("ned19_n33x50_w117x75_ca_orangeco_2011.zip", "2012-01-01", 0, 2)
    separate = _ned19("ned19_n33x50_w117x75_ca_sandiego_2025.zip", "2026-01-01", 1, 3)
    other_tile = _ned19("ned19_n33x75_w117x75_ca_orangeco_2025.zip", "2026-01-01", 1, 3)
    rows = TNMPolicy().run(
        [
            (SimpleNamespace(), old),
            (SimpleNamespace(), separate),
            (SimpleNamespace(), other_tile),
        ]
    )
    SpatialClaimHook().run(rows)
    assert all(
        "excluded_geometry" not in entry for entry in (old, separate, other_tile)
    )
    assert _geom(old, "accepted_geometry").intersection(
        _geom(separate, "accepted_geometry")
    ).area == pytest.approx(1)


def test_ned19_ambiguous_same_publication_date_fails_closed():
    old = _ned19("ned19_n33x50_w117x75_ca_orangeco_2011.zip", "2026-02-01", 0, 2)
    new = _ned19("ned19_n33x50_w117x75_ca_orangeco_2025.zip", "2026-02-01", 1, 3)
    with pytest.raises(RuntimeError, match="Ambiguous same-day"):
        TNMPolicy().run([(SimpleNamespace(), old), (SimpleNamespace(), new)])


def test_ned19_repeated_series_missing_publication_date_fails_closed():
    old = _ned19("ned19_n33x50_w117x75_ca_orangeco_2011.zip", None, 0, 2)
    new = _ned19("ned19_n33x50_w117x75_ca_orangeco_2025.zip", "2026-02-01", 1, 3)
    with pytest.raises(RuntimeError, match="valid publication date"):
        TNMPolicy().run([(SimpleNamespace(), old), (SimpleNamespace(), new)])


def test_ned19_unknown_archive_identity_is_not_guessed_from_bounds():
    old = _ned19("unidentified_n33x50_w117x75_2011.zip", "2012-01-01", 0, 2)
    new = _ned19("unidentified_n33x50_w117x75_2025.zip", "2026-01-01", 1, 3)
    SpatialClaimHook().run(
        TNMPolicy().run([(SimpleNamespace(), old), (SimpleNamespace(), new)])
    )
    assert "excluded_geometry" not in old
    assert _geom(old, "accepted_geometry").intersection(
        _geom(new, "accepted_geometry")
    ).area == pytest.approx(1)
