"""Representative terrain, claims and hydroflattened-water contracts."""

import geopandas as gpd


import numpy as np


import rasterio


from rasterio.transform import from_origin


from rasterio.warp import transform_bounds


from shapely.geometry import box


from globato.coast_refine._science.coast_refine_v6_6_1.coast_refine_core import (
    extract_positive_evidence,
    classify_connectivity,
)


def make_plan(tmp_path):
    p = tmp_path / "plan.gpkg"
    gpd.GeoDataFrame(
        {"id": [1]}, geometry=[box(40, 0, 50, 100)], crs="EPSG:32611"
    ).to_file(p, layer="landward_context", driver="GPKG")
    gpd.GeoDataFrame(
        {"id": [1]}, geometry=[box(50, 0, 75, 100)], crs="EPSG:32611"
    ).to_file(p, layer="seaward_search", driver="GPKG")
    return p


def make_raster(tmp_path):
    p = tmp_path / "r.tif"
    a = np.full((100, 100), -1, dtype="float32")
    a[:, :55] = 2
    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=1,
        dtype="float32",
        crs="EPSG:32611",
        transform=from_origin(0, 100, 1, 1),
        nodata=-9999,
    ) as d:
        d.write(a, 1)
    b = transform_bounds("EPSG:32611", "EPSG:4326", 0, 0, 100, 100)
    return p, (b[0], b[2], b[1], b[3])


def _screen_then_connect(evidence, *, osm=None):
    from globato.coast_refine._science.coast_refine_v6_6_1.coast_refine_core import (
        screen_hydroflat_transition_fringe,
        local_terrain_support_screen,
    )

    screened, seeds, rejected, protected, stats = screen_hydroflat_transition_fringe(
        evidence, metric_crs="EPSG:32611", osm_evidence=osm, min_valid_cells=4
    )
    support, review = local_terrain_support_screen(
        screened, local_window_m=15, min_local_relief_m=0.10, max_flat_bridge_m=5
    )
    context = screened[screened["kind"] == "land_context_positive"].copy()
    frames = [g for g in (context, support) if g is not None and not g.empty]
    if frames:
        filtered = gpd.GeoDataFrame(
            __import__("pandas").concat(frames, ignore_index=True),
            geometry="geometry",
            crs=screened.crs,
        )
    else:
        filtered = gpd.GeoDataFrame(
            columns=screened.columns, geometry="geometry", crs=screened.crs
        )
    accepted, detached, union = classify_connectivity(filtered, "EPSG:32611")
    return (
        screened,
        seeds,
        rejected,
        protected,
        stats,
        support,
        review,
        accepted,
        detached,
        union,
    )


def test_connected_positive_terrain_is_accepted(tmp_path):
    plan = make_plan(tmp_path)
    raster, bbox = make_raster(tmp_path)
    evidence, _ = extract_positive_evidence(
        raster_path=raster,
        planning_gpkg=plan,
        bbox_wgs84=bbox,
        accepted_geometry_wkt=None,
        threshold=0.01,
        min_seaward_cells=4,
        out_crs="EPSG:32611",
    )
    accepted, detached, union = classify_connectivity(evidence, "EPSG:32611")
    assert len(accepted) == 1 and len(detached) == 0
    assert union.area == 500


def test_accepted_claim_blocks_superseded_coverage(tmp_path):
    plan = make_plan(tmp_path)
    raster, bbox = make_raster(tmp_path)
    # Claim only the landward half; positive seaward cells must be invisible.
    claim = box(0, 0, 50, 100)
    evidence, _ = extract_positive_evidence(
        raster_path=raster,
        planning_gpkg=plan,
        bbox_wgs84=bbox,
        accepted_geometry_wkt=claim.wkt,
        accepted_geometry_srs="EPSG:32611",
        threshold=0.01,
        min_seaward_cells=4,
        out_crs="EPSG:32611",
    )
    assert (evidence["kind"] == "seaward_positive").sum() == 0


def test_local_terrain_support_does_not_flood_broad_flat_harbor(tmp_path):
    from globato.coast_refine._science.coast_refine_v6_6_1.coast_refine_core import (
        local_terrain_support_screen,
    )

    plan = make_plan(tmp_path)
    p = tmp_path / "mixed_harbor.tif"
    a = np.full((100, 100), -1, dtype="float32")
    # Land context positive. Seaward water is +0.20 and flat, except a narrow
    # shore-edge ramp (quay/structure-like relief) that would make whole-component
    # statistics pass in v5.5.
    a[:, :75] = 0.20
    a[:, 50:53] = np.array([0.9, 0.6, 0.3], dtype="float32")[None, :]
    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=1,
        dtype="float32",
        crs="EPSG:32611",
        transform=from_origin(0, 100, 1, 1),
        nodata=-9999,
    ) as d:
        d.write(a, 1)
    b = transform_bounds("EPSG:32611", "EPSG:4326", 0, 0, 100, 100)
    evidence, _ = extract_positive_evidence(
        raster_path=p,
        planning_gpkg=plan,
        bbox_wgs84=(b[0], b[2], b[1], b[3]),
        accepted_geometry_wkt=None,
        threshold=0.01,
        min_seaward_cells=4,
        out_crs="EPSG:32611",
    )
    support, review = local_terrain_support_screen(
        evidence, local_window_m=15, min_local_relief_m=0.10, max_flat_bridge_m=5
    )
    assert not support.empty
    assert not review.empty
    # The far side of the 25 m-wide flat harbor must remain unsupported.
    assert support.total_bounds[2] < 70.0
    assert review.total_bounds[2] >= 74.0


def test_local_terrain_support_retains_real_slope(tmp_path):
    from globato.coast_refine._science.coast_refine_v6_6_1.coast_refine_core import (
        local_terrain_support_screen,
    )

    plan = make_plan(tmp_path)
    p = tmp_path / "local_slope.tif"
    a = np.full((100, 100), -1, dtype="float32")
    vals = np.linspace(2.0, 0.05, 75, dtype="float32")
    a[:, :75] = vals[None, :]
    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=1,
        dtype="float32",
        crs="EPSG:32611",
        transform=from_origin(0, 100, 1, 1),
        nodata=-9999,
    ) as d:
        d.write(a, 1)
    b = transform_bounds("EPSG:32611", "EPSG:4326", 0, 0, 100, 100)
    evidence, _ = extract_positive_evidence(
        raster_path=p,
        planning_gpkg=plan,
        bbox_wgs84=(b[0], b[2], b[1], b[3]),
        accepted_geometry_wkt=None,
        threshold=0.01,
        min_seaward_cells=4,
        out_crs="EPSG:32611",
    )
    support, review = local_terrain_support_screen(
        evidence, local_window_m=15, min_local_relief_m=0.10, max_flat_bridge_m=5
    )
    assert not support.empty
    # Most of the 25m seaward slope should remain supported.
    assert support.geometry.area.sum() > 1800


def test_hydroflat_flat_plateau_smooth_ramp_is_not_accepted(tmp_path):
    """Regression shape for the demonstrated Los Angeles hydroflat/TIN fringe."""
    plan = make_plan(tmp_path)
    p = tmp_path / "hydroflat_ramp.tif"
    a = np.full((100, 100), -1, dtype="float32")
    # Positive inland terrain reaches the baseline shoreline smoothly.
    a[:, 40:50] = np.linspace(1.6, 1.15, 10, dtype="float32")[None, :]
    # The invalid seaward extension is a smooth positive interpolation ramp into
    # a broad hydroflattened +0.40 m water plateau.  Every value is > +0.01 m.
    # The plateau is wider than the 15 m seed window so it contains a genuine
    # broad flat core, while the ramp steps remain below the 10% propagation gate.
    a[:, 50:60] = np.linspace(1.10, 0.40, 10, dtype="float32")[None, :]
    a[:, 60:75] = 0.40
    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=1,
        dtype="float32",
        crs="EPSG:32611",
        transform=from_origin(0, 100, 1, 1),
        nodata=-9999,
    ) as d:
        d.write(a, 1)
    b = transform_bounds("EPSG:32611", "EPSG:4326", 0, 0, 100, 100)
    evidence, _ = extract_positive_evidence(
        raster_path=p,
        planning_gpkg=plan,
        bbox_wgs84=(b[0], b[2], b[1], b[3]),
        accepted_geometry_wkt=None,
        threshold=0.01,
        min_seaward_cells=4,
        out_crs="EPSG:32611",
    )
    assert not evidence[evidence["kind"] == "seaward_positive"].empty
    result = _screen_then_connect(evidence)
    (
        screened,
        seeds,
        rejected,
        protected,
        stats,
        support,
        review,
        accepted,
        detached,
        union,
    ) = result
    assert stats["rows_with_seed"] == 1
    assert stats["seed_cells"] >= 400
    assert stats["rejected_cells"] > 1500
    # The smooth flat-water-connected seaward surface must not survive as
    # land-connected accepted terrain.
    assert accepted.empty
    assert union.is_empty
    assert not rejected.empty


def test_hydroflat_screen_leaves_real_slope_alone_when_no_broad_flat_seed(tmp_path):
    plan = make_plan(tmp_path)
    p = tmp_path / "real_slope_no_hydroflat.tif"
    a = np.full((100, 100), -1, dtype="float32")
    vals = np.linspace(2.0, 0.05, 75, dtype="float32")
    a[:, :75] = vals[None, :]
    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=1,
        dtype="float32",
        crs="EPSG:32611",
        transform=from_origin(0, 100, 1, 1),
        nodata=-9999,
    ) as d:
        d.write(a, 1)
    b = transform_bounds("EPSG:32611", "EPSG:4326", 0, 0, 100, 100)
    evidence, _ = extract_positive_evidence(
        raster_path=p,
        planning_gpkg=plan,
        bbox_wgs84=(b[0], b[2], b[1], b[3]),
        accepted_geometry_wkt=None,
        threshold=0.01,
        min_seaward_cells=4,
        out_crs="EPSG:32611",
    )
    result = _screen_then_connect(evidence)
    (
        screened,
        seeds,
        rejected,
        protected,
        stats,
        support,
        review,
        accepted,
        detached,
        union,
    ) = result
    assert stats["rows_with_seed"] == 0
    assert seeds.empty
    assert rejected.empty
    assert not accepted.empty
    assert union.area > 1800
