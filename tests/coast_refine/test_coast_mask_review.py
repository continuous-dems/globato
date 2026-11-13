"""Source retention parity, native standalone lifecycle, lineage and concurrency."""

from types import SimpleNamespace
import hashlib
import json
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
import laspy
from pyproj import CRS
from shapely.geometry import box, Polygon
from click.testing import CliRunner
import yaml

from fetchez.spatial import Region
from globato.cli import cli
from globato.coast_refine._io import digest, json_write
from globato.coast_refine.artifact import MaskArtifact, MODES
from globato.coast_refine.integration import (
    CoastMask,
    CoastMaskConsumer,
    CoastRefineFinalize,
)
from globato.coast_refine.grid_native_fusion import (
    mask_fusion_state,
    inspect_fusion_state,
)
from globato.coast_refine.masking import write_masked_tiff
from globato.coast_refine.source_receipt import verify_accepted_source
from globato.hooks.transforms.point_pixels import FUSION_BANDS


def state_file(path, transform, crs, shape=(16, 24), values=None):
    rows, cols = np.indices(shape)
    z = np.full(shape, -2.0) if values is None else np.asarray(values, float)
    x, y = transform * (cols + 0.5, rows + 0.5)
    state = np.stack(
        [
            z * 2,
            np.ones(shape),
            np.full(shape, 2.0),
            z * z * 2,
            np.full(shape, 0.04),
            x * 2,
            y * 2,
        ]
    ).astype("float64")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=shape[1],
        height=shape[0],
        count=7,
        dtype="float64",
        transform=transform,
        crs=crs,
    ) as ds:
        ds.write(state)
        ds.descriptions = FUSION_BANDS
        ds.update_tags(
            GLOBATO_DATATYPE="FUSION_STATE",
            GLOBATO_FUSION_VERSION="1",
            COAST_MASK_GRID_ROLE="source_unfilled",
        )
    return state


@pytest.mark.parametrize(
    "crs,transform,keep",
    [
        ("EPSG:4326", from_origin(0, 16, 1, 1), box(0.5, 0.5, 12.5, 15.5)),
        (
            "EPSG:32610",
            from_origin(500000, 4650000, 2, 3),
            box(500001, 4649953, 500025, 4649998.5),
        ),
    ],
)
def test_native_tiff_and_fusionstate_use_identical_centres_and_boundary_ties(
    tmp_path, crs, transform, keep
):
    source = tmp_path / "raw.fusion.tif"
    values = np.arange(384).reshape(16, 24) - 200.0
    state = state_file(source, transform, crs, values=values)
    with rasterio.open(source) as ds:
        profile = ds.profile.copy()
        profile.update(count=1)
    single = tmp_path / "native.tif"
    with rasterio.open(single, "w", **profile) as ds:
        ds.write(values, 1)
    geo = gpd.GeoSeries([keep], crs=crs).to_crs(4326).iloc[0]
    write_masked_tiff(single, tmp_path / "native_masked.tif", geo)
    mask_fusion_state(source, tmp_path / "masked.fusion.tif", geo, 4326)
    with (
        rasterio.open(tmp_path / "native_masked.tif") as a,
        rasterio.open(tmp_path / "masked.fusion.tif") as b,
    ):
        native = np.isfinite(a.read(1))
        masked = b.read()
        assert np.array_equal(native, masked[1] > 0)
        assert masked[:, native].tobytes() == state[:, native].tobytes()
        assert np.all(masked[:, ~native] == 0)


def test_all_removed_source_is_a_valid_zero_state(tmp_path):
    source = tmp_path / "source.tif"
    state_file(source, from_origin(0, 16, 1, 1), 4326)
    before = digest(source)
    result = mask_fusion_state(source, tmp_path / "empty.tif", Polygon(), 4326)
    assert (
        result["outcome"] == "EMPTY_AFTER_MASK" and result["counts"]["after_cells"] == 0
    )
    with rasterio.open(tmp_path / "empty.tif") as ds:
        assert np.all(ds.read() == 0)
    assert digest(source) == before


@pytest.mark.parametrize(
    "kind", ["nan", "negative_count", "negative_weight", "phantom_empty"]
)
def test_invalid_sufficient_statistics_fail_closed(tmp_path, kind):
    source = tmp_path / "source.tif"
    a = state_file(source, from_origin(0, 16, 1, 1), 4326)
    if kind == "nan":
        a[0, 0, 0] = np.nan
    if kind == "negative_count":
        a[1, 0, 0] = -1
    if kind == "negative_weight":
        a[2, 0, 0] = -1
    if kind == "phantom_empty":
        a[1, 0, 0] = 0
    with rasterio.open(source, "r+") as ds:
        ds.write(a)
    with pytest.raises(ValueError):
        inspect_fusion_state(source)


def test_atomic_receipt_preserves_prior_file_on_serialization_failure(tmp_path):
    p = tmp_path / "receipt.json"
    json_write(p, {"status": "completed"})
    before = p.read_bytes()
    with pytest.raises(ValueError):
        json_write(p, {"bad": np.nan})
    assert p.read_bytes() == before and not list(tmp_path.glob("*.tmp"))


def test_group_labels_and_one_registration_contract(tmp_path):
    p = tmp_path / "source.tif"
    state_file(p, from_origin(0, 16, 1, 1), 4326)
    m = SimpleNamespace(name="fixture", weight=1, region=Region(0, 24, 0, 16), hooks=[])
    e = {"dst_fn": str(p)}
    CoastMaskConsumer(" authority ", "water").run([(m, e)])
    assert e["coast_refine_consumer"]["authority"] == "authority"
    with pytest.raises(ValueError, match="already registered"):
        CoastMask(" authority ", remove="water").run([(m, e)])
    assert CoastMask(" authority ", remove="water").settings["group"] == "authority"
    assert CoastMask("authority", remove="water").settings["grid_native"] is True


@pytest.mark.parametrize("remove", MODES)
def test_consumer_all_removal_policies_share_artifact_retention_not_barrier(
    tmp_path, remove
):
    # Different polygons deliberately represent the source-retention and signed
    # interpolation roles. The REAL MaskArtifact method owns source selection.
    artifact = MaskArtifact.__new__(MaskArtifact)
    artifact.aoi = box(0, 0, 24, 16)
    artifact.coverage = artifact.aoi
    artifact.path = tmp_path / "manifest.json"
    json_write(artifact.path, {"fixture": True})
    artifact.manifest_hash = digest(artifact.path)
    retained = box(12, 0, 24, 16)
    artifact._classes = {
        "retention_" + k: retained
        for k in (
            "water",
            "marine_water",
            "inland_water",
            "unresolved_water",
            "classified_ocean",
            "classified_inland_water",
        )
    }
    artifact._classes["structures"] = Polygon()
    artifact._classes["refined_water"] = box(
        6, 0, 24, 16
    )  # Must NOT mask source cells 6..11.
    source = tmp_path / "original.fusion.tif"
    raw = state_file(source, from_origin(0, 16, 1, 1), 4326)
    canonical = json.dumps(
        {"source": "fixture", "branch": remove}, sort_keys=True, separators=(",", ":")
    )
    token = hashlib.sha256(canonical.encode()).hexdigest()
    with rasterio.open(source, "r+") as ds:
        ds.update_tags(
            GLOBATO_SIDESTACK_HASH=token, GLOBATO_SIDESTACK_IDENTITY=canonical
        )
    m = SimpleNamespace(name="fixture", weight=1, region=Region(0, 24, 0, 16), hooks=[])
    e = {
        "dst_fn": str(tmp_path / "original.xyz"),
        "stream_type": "raster-stream",
        "data_type": "fusion-state",
        "_sidestack_hash": token,
        "_sidestack_identity": canonical,
        "_sidestack_path": str(source),
        "artifacts": {"side-stack": str(source)},
        "coast_refine_consumer": {
            "authority": "a",
            "remove": remove,
            "output_dir": str(tmp_path),
            "registered_source_metadata": {},
        },
    }
    f = CoastRefineFinalize(res=1, crs=4326, grid_native=True)
    f._finish_grid_consumers([(m, e)], {"a": artifact})
    expected = artifact.keep_geometry(remove, artifact.scope_for(m))
    with rasterio.open(e["artifacts"]["side-stack"]) as ds:
        actual = ds.read()
        rr, cc = np.indices(ds.shape)
        x, y = ds.transform * (cc + 0.5, rr + 0.5)
        import shapely

        on = shapely.contains_xy(expected, x, y)
        assert np.array_equal(actual[:, on], raw[:, on]) and np.all(actual[:, ~on] == 0)
    receipt = verify_accepted_source(e["artifacts"]["coast_mask_source_receipt"])
    assert receipt["selection"] == "source_retention_native_centres"
    assert receipt["source_id"] is None and not receipt["physical_guidance_generated"]


def test_consumer_outside_prepared_coverage_fails_before_masking(tmp_path):
    artifact = MaskArtifact.__new__(MaskArtifact)
    artifact.aoi = box(0, 0, 10, 10)
    artifact.coverage = artifact.aoi
    artifact._classes = {"retention_water": box(0, 0, 10, 10), "structures": Polygon()}
    with pytest.raises(ValueError, match="exceeds prepared"):
        artifact.keep_geometry("land", box(0, 0, 20, 20))


def native_recipe(tmp_path, *, compressed=False, interpolation=False):
    cell = 1 / 32400
    size = 96
    span = size * cell
    terrain = tmp_path / "terrain.tif"
    water = tmp_path / "water.gpkg"
    with rasterio.open(
        terrain,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="float32",
        nodata=-9999,
        crs=4326,
        transform=from_origin(0, span, cell, cell),
    ) as ds:
        ds.write(
            np.tile((47.5 - np.arange(size, dtype="float32")) * 0.25, (size, 1)), 1
        )
    gpd.GeoDataFrame(
        geometry=[box(60 * cell, -span, 2 * span, 2 * span)], crs=4326
    ).to_file(water, driver="GPKG")
    source = tmp_path / ("survey.laz" if compressed else "survey.las")
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = [1e-9, 1e-9, 1e-4]
    header.add_crs(CRS.from_epsg(4326))
    data = laspy.LasData(header)
    data.x = np.array([75.5, 20.5, 49.5, 20.5, 75.5, 85.5]) * cell
    data.y = np.full(6, 40.5 * cell)
    data.z = [-0.1, -1.0, -1.0, 0.0, 1.0, -9.0]
    data.classification = [29, 29, 2, 2, 2, 7]
    data.write(source)
    modules = []
    for cls, remove, role, weight in [
        ("29", "land", "bathy", 2),
        ("2", "water", "topo", 3),
        ("7", None, "unmasked", 1),
    ]:
        hooks = [
            {
                "name": "set-entry",
                "args": {
                    "key": "metadata",
                    "value": {"survey_id": "fixture_survey", "class_branch": role},
                },
            }
        ]
        if remove:
            hooks.append(
                {
                    "name": "coast_mask_consumer",
                    "args": {
                        "authority": "authority",
                        "remove": remove,
                        "output_dir": "%tile_dir%/coast_mask",
                    },
                }
            )
        hooks.extend(
            [
                {
                    "name": "stream-init",
                    "args": {"data_type": "lidar", "classes": cls, "chunk_size": 2},
                },
                {
                    "name": "drop-class",
                    "args": {"classes": cls, "invert": True, "stack_partition": role},
                },
            ]
        )
        modules.append(
            {
                "module": "local_fs",
                "args": {"path": str(source), "data_type": "lidar", "weight": weight},
                "hooks": hooks,
            }
        )
    modules.append(
        {
            "module": "local_fs",
            "args": {"path": str(terrain), "data_type": "raster", "weight": 2},
            "hooks": [
                {
                    "name": "coast_mask",
                    "args": {
                        "group": "authority",
                        "mask": str(water),
                        "polygon_class": "ocean",
                        "remove": "water",
                        "viz": True,
                        "output_dir": "%tile_dir%/coast_mask",
                    },
                },
                {"name": "stream-init", "args": {"profile": "rio"}},
            ],
        }
    )
    tiers = "0.1111111s/0.3333333s/1s/3s/9s/15s"
    global_hooks = [
        {
            "name": "coast_mask_side_stack",
            "args": {
                "res": str(cell),
                "crs": "EPSG:4326",
                "cache_dir": "%tile_dir%/source_states",
            },
        },
        {
            "name": "coast_mask_finalize",
            "args": {
                "res": str(cell),
                "crs": "EPSG:4326",
                "resolutions": tiers if interpolation else None,
            },
        },
        {
            "name": "stack_combine",
            "args": {
                "res": str(cell),
                "crs": "EPSG:4326",
                "strategy": "mixed",
                "weight_threshold": "1/2/3",
                "missing": "error",
                "output": "%tile_dir%/combined.tif",
            },
        },
    ]
    if interpolation:
        global_hooks.extend(
            [
                {"name": "focus_sink", "args": {"target": "multi_stack"}},
                {
                    "name": "raster_stream",
                    "args": {"stream_type": "raster", "stage": "collection"},
                },
                {
                    "name": "ms_binary_cudem",
                    "args": {
                        "steps": 5,
                        "resolutions": tiers,
                        "weights": "3/2/1/.5/.1/0",
                        "blend_dists": "0/0/0/0/0/0",
                        "barrier": "coast_refine",
                        "bathy_max_z": None,
                        "algos": ["raster_fill"] * 6,
                        "inland_decay_dist": 0,
                        "previous_tier_mode": "points",
                        "regime_guidance": "adaptive",
                        "output": "%tile_dir%/dem.tif",
                    },
                },
            ]
        )
    obj = {
        "project": {"name": "coast_mask_native"},
        "region": f"0/{span}/0/{span}",
        "region_srs": "EPSG:4326",
        "modules": modules,
        "global_hooks": global_hooks,
        "execution": {"threads": 2},
    }
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump(obj, sort_keys=False))
    return recipe, terrain, source


@pytest.mark.parametrize("compressed", [False, True], ids=["LAS", "LAZ"])
def test_registered_standalone_cli_masks_each_class_branch_and_reruns(
    tmp_path, compressed
):
    recipe, terrain, source = native_recipe(tmp_path, compressed=compressed)
    hashes = {p: digest(p) for p in (terrain, source)}
    args = [
        "run",
        str(recipe),
        "--outdir",
        str(tmp_path / "out"),
        "--shared-cache",
        str(tmp_path / "cache"),
        "--refresh",
    ]
    first = CliRunner().invoke(cli, args)
    assert first.exit_code == 0, first.output
    receipts = list((tmp_path / "out").rglob("*.source.json"))
    assert len(receipts) == 3, first.output
    verified = [verify_accepted_source(p) for p in receipts]
    consumers = [
        r
        for r in verified
        if r["registered_source_metadata"].get("class_branch") in ("bathy", "topo")
    ]
    assert len(consumers) == 2 and len({r["accepted_state"] for r in consumers}) == 2
    assert {r["survey_id"] for r in consumers} == {"fixture_survey"}
    bathy = next(r for r in consumers if r["remove"] == "land")
    topo = next(r for r in consumers if r["remove"] == "water")
    assert bathy["counts"]["after_observations"] == 1
    assert (
        topo["counts"]["after_observations"] == 2
    )  # Includes negative terrain in signed barrier water.
    assert all(digest(p) == h for p, h in hashes.items())
    stacks = [
        p for p in (tmp_path / "out").glob("combined*.tif") if ".fusion." not in p.name
    ]
    assert len(stacks) == 1
    with rasterio.open(stacks[0]) as ds:
        assert ds.count == 7 and np.any(ds.read(1) == -9.0)  # Unmasked source retained.
    assert not list(
        (tmp_path / "out").glob("dem*.tif")
    )  # Independently useful, no interpolation.
    catalogs = list((tmp_path / "out/coast_mask/products").rglob("products.json"))
    assert len(catalogs) == 1
    catalog = json.loads(catalogs[0].read_text())
    assert not catalog["running"] and len(catalog["products"]) == 5
    for product in catalog["products"]:
        assert (catalogs[0].parent / product["path"]).is_file()
        assert product["metadata"]["interpolated"] is False
    assert catalog["defaults"] == {"default": ["before", "after"]}
    raster_products = [
        p for p in catalog["products"] if p.get("type", "raster") == "raster"
    ]
    assert [p["key"] for p in raster_products] == ["before", "after"]
    assert all(p["metadata"]["participant_count"] == 3 for p in raster_products)
    with rasterio.open(catalogs[0].parent / "after.tif") as accepted:
        values = accepted.read(1, masked=True).compressed()
        assert -9.0 not in values  # Ordinary unmasked control is not a participant.
    second = CliRunner().invoke(cli, args)
    assert second.exit_code == 0, second.output
    assert all(digest(p) == h for p, h in hashes.items())
