"""Keep the application lattice while enclosing the actual preparation AOI."""

import json

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from globato.coast_refine.prepare import prepare_from_baseline


@pytest.mark.parametrize("uniform_water", [False, True])
@pytest.mark.parametrize("eps", [2e-10, 2e-12, 2e-13])
def test_native_anchor_does_not_discard_geographic_preparation_scope(
    tmp_path, uniform_water, eps
):
    evidence = tmp_path / "evidence.tif"
    transform = from_origin(-118.70, 34.045, 1 / 32400, 1 / 32400)
    data = np.arange(64 * 64, dtype="float32").reshape(64, 64)
    with rasterio.open(
        evidence,
        "w",
        driver="GTiff",
        width=64,
        height=64,
        count=1,
        dtype="float32",
        nodata=-9999,
        crs=5498,
        transform=transform,
    ) as ds:
        ds.write(data, 1)
    with rasterio.open(evidence) as ds:
        native = tuple(ds.bounds)
    # A preparation AOI can overhang the native anchor after horizontal
    # projection/chord construction or the existing GEOS scope guard.
    # This is larger than ULP noise, but much smaller than one native cell.
    aoi = box(native[0] - eps, native[1] - eps, native[2] + eps, native[3] + eps)
    area = tmp_path / "aoi.gpkg"
    gpd.GeoDataFrame(geometry=[aoi], crs=4326).to_file(area)
    polygon = aoi.buffer(0.001) if uniform_water else box(-118, 33, -117, 34)
    mask = tmp_path / "water.gpkg"
    gpd.GeoDataFrame(geometry=[polygon], crs=4326).to_file(mask)
    original = evidence.read_bytes()
    artifact = prepare_from_baseline(
        evidence,
        area,
        tmp_path / "artifact",
        tmp_path / "cache",
        mask=str(mask),
        polygon_class="ocean",
        native_bounds=native,
    )
    assert artifact._coverage_covers(aoi)
    assert artifact.barrier_for(evidence).is_file()
    assert evidence.read_bytes() == original
    grid = json.loads((artifact.root / "application_grid.json").read_text())
    if uniform_water:
        assert grid["removed_subcell_fragment_count"] == 0
    affine = rasterio.Affine(*grid["transform"])
    assert affine.a == transform.a and affine.e == transform.e
    padding = 1 if eps > 1e-12 else 0
    assert (affine.c - transform.c) / transform.a == pytest.approx(-padding)
    assert (affine.f - transform.f) / transform.e == pytest.approx(-padding)
    assert grid["width"] == grid["height"] == 64 + 2 * padding
    # Padding the preparation must not disable the strict working-grid gate.
    outside = tmp_path / "outside.tif"
    with rasterio.open(
        outside,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs=5498,
        transform=from_origin(
            native[0] - 10 * transform.a, native[3], transform.a, -transform.e
        ),
    ) as ds:
        ds.write(np.ones((2, 2), dtype="float32"), 1)
    with pytest.raises(ValueError, match="does not cover"):
        artifact.barrier_for(outside)


def test_failed_preparation_keeps_diagnostics_without_publishing_acceptance(
    tmp_path, monkeypatch
):
    from fetchez.recipe import Recipe, RecipeContext
    from fetchez.spatial import Region
    from globato.coast_refine import prepare
    from test_global_refinement import global_recipe
    from fetchez.core import HookExecutionError

    recipe, _, _ = global_recipe(tmp_path)

    def fail(evidence, area, output_dir, *args, **kwargs):
        output_dir.mkdir()
        (output_dir / "failure-marker.txt").write_text("unaccepted scratch")
        raise ValueError("forced preparation failure")

    monkeypatch.setattr(prepare, "prepare_from_baseline", fail)
    output = tmp_path / "out"
    output.mkdir()
    monkeypatch.chdir(output)
    plan = Recipe.from_file(recipe).compile(
        RecipeContext(
            name="failure_test",
            region=Region(0, 96 / 32400, 0, 96 / 32400),
            outdir=output,
            tile_dir=output,
            shared_cache=tmp_path / "cache",
        )
    )
    with pytest.raises(
        HookExecutionError,
        match="forced preparation failure",
    ) as exc_info:
        plan.execute()

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert str(exc_info.value.__cause__) == "forced preparation failure"

    failures = list((tmp_path / "out").rglob("failure.json"))
    assert len(failures) == 1
    receipt = json.loads(failures[0].read_text())
    assert receipt["status"] == "FAILED_NOT_ACCEPTED"
    assert (
        failures[0].parent / "artifact/failure-marker.txt"
    ).read_text() == "unaccepted scratch"
    assert not list((tmp_path / "out").rglob("accepted_collection.json"))
