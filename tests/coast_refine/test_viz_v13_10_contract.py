"""Visualization publisher contract tests using Raster Viz v0.13.10.

Load the publisher in isolation so the tests never invoke scientific Coast Mask,
Globato/Fetchez import graphs, or the external legacy viewer.
"""

from pathlib import Path
import importlib.util
import json
import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import from_origin


from globato.viz.raster_viz.manifest import read_catalog
from globato.viz.raster_viz.validate import check_catalog


def publisher():
    base = Path(__file__).resolve().parents[2] / "src/globato/coast_refine"
    package = "isolated_coast_refine_viz"
    pkg = types.ModuleType(package)
    pkg.__path__ = [str(base)]
    io = types.ModuleType(package + "._io")

    def json_write(path, data):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, allow_nan=False))

    io.identity = lambda content: "test-key"
    io.json_write = json_write
    sys.modules[package] = pkg
    sys.modules[package + "._io"] = io
    spec = importlib.util.spec_from_file_location(
        package + ".viz_products", base / "viz_products.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def tif(path):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=1,
        dtype="float32",
        crs="EPSG:5498",
        transform=from_origin(-118.2, 34.1, 0.001, 0.001),
    ) as dst:
        dst.write(np.ones((8, 8), dtype="float32"), 1)
    return path


def vector(path):
    # Raster Viz supports polygons and lines from GPKG/GeoJSON.
    import fiona
    from fiona.crs import CRS

    with fiona.open(
        path,
        "w",
        driver="GeoJSON",
        schema={"geometry": "Polygon", "properties": {}},
        crs=CRS.from_epsg(4326),
    ) as dst:
        dst.write(
            {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-118.2, 34.1],
                            [-118.19, 34.1],
                            [-118.19, 34.09],
                            [-118.2, 34.09],
                            [-118.2, 34.1],
                        ]
                    ],
                },
                "properties": {},
            }
        )
    return path


def test_source_and_final_catalog_stay_standalone_and_portable(tmp_path):
    mod = publisher()
    products = tmp_path / "data"
    products.mkdir()
    tif(products / "unmasked_evidence.tif")
    vector(products / "topology.geojson")
    artifact = SimpleNamespace(
        preparation_id="prep123",
        root=products,
        products={"refined_topology": str(products / "topology.geojson")},
    )
    request = {"coast_refine_request": {"output_dir": str(tmp_path / "run")}}
    published = mod.publish_source_group(
        "group-a", [(object(), request)], artifact, deferred=True
    )
    data, groups = read_catalog(published)
    assert data["running"] is True
    assert len(data["products"]) == 2 and not data["defaults"]
    assert data["products"][1]["display"]["visible"] is True
    assert all(not os.path.isabs(p["path"]) for p in data["products"])
    assert check_catalog(published)["ok"]
    # The optional final global publication appends, preserving the source overview.
    final = tif(products / "final.tif")
    stack = tif(products / "source_stack.tif")
    barrier = vector(products / "barrier.geojson")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"source_stack_path": str(stack)}))
    entry = {
        "coast_refine_viz_catalog": str(published),
        "dst_fn": str(final),
        "artifacts": {
            "coast_refine_barrier_audit": str(audit),
            "coast_refine_barrier": str(barrier),
        },
    }
    mod.publish_global_result(entry)
    data, groups = read_catalog(published)
    assert data["running"] is False
    assert len(data["products"]) == 5
    assert len(data["defaults"]) == 1
    assert data["products"][1]["display"]["visible"] is True
    assert (
        sum(bool(p.get("display", {}).get("visible")) for p in data["products"]) == 2
    )  # one per area
    assert all(not os.path.isabs(p["path"]) for p in data["products"])
    assert check_catalog(published)["ok"]


def test_grid_native_five_products_and_independent_vectors(tmp_path):
    # Only the integration/publication seam is mocked: the actual provider and
    # native stacking algorithms are deliberately not exercised in this test.
    import hashlib

    base = Path(__file__).resolve().parents[2] / "src/globato/coast_refine"
    package = "isolated_coast_mask_native_viz"
    pkg = types.ModuleType(package)
    pkg.__path__ = [str(base)]
    io = types.ModuleType(package + "._io")
    io.identity = lambda content: "group-viz-fixture"
    io.json_write = lambda path, data: (
        Path(path).parent.mkdir(parents=True, exist_ok=True),
        Path(path).write_text(json.dumps(data, allow_nan=False)),
    )
    io.digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    sys.modules[package] = pkg
    sys.modules[package + "._io"] = io
    bands = ("sum_z", "count", "weight", "sum_z2", "variance", "x", "y")
    globato = types.ModuleType("globato")
    globato.__path__ = []
    hooks = types.ModuleType("globato.hooks")
    hooks.__path__ = []
    transforms = types.ModuleType("globato.hooks.transforms")
    transforms.__path__ = []
    point_pixels = types.ModuleType("globato.hooks.transforms.point_pixels")
    point_pixels.FUSION_BANDS = bands
    point_pixels.finalize_fusion_state = lambda state: SimpleNamespace(
        valid=state["count"] > 0, z=state["sum_z"] / np.maximum(1, state["count"])
    )
    tools = types.ModuleType("globato.hooks.tools")
    tools.__path__ = []
    stack_combine = types.ModuleType("globato.hooks.tools.stack_combine")

    class FakeStackCombine:
        def __init__(self, **kwargs):
            pass

        def run(self, copied):
            copied[0][1]["artifacts"]["fusion-state"] = copied[0][1]["artifacts"][
                "side-stack"
            ]

        def teardown(self):
            pass

    stack_combine.StackCombine = FakeStackCombine
    modules = {
        "globato": globato,
        "globato.hooks": hooks,
        "globato.hooks.transforms": transforms,
        "globato.hooks.transforms.point_pixels": point_pixels,
        "globato.hooks.tools": tools,
        "globato.hooks.tools.stack_combine": stack_combine,
    }
    old = {k: sys.modules.get(k) for k in modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location(
            package + ".source_products", base / "source_products.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        raw = tmp_path / "raw_state.tif"
        accepted = tmp_path / "accepted_state.tif"
        for p, z in ((raw, -3.0), (accepted, -2.0)):
            with rasterio.open(
                p,
                "w",
                driver="GTiff",
                width=8,
                height=8,
                count=7,
                dtype="float64",
                crs="EPSG:5498",
                transform=from_origin(-118.2, 34.1, 0.001, 0.001),
            ) as dst:
                a = np.zeros((7, 8, 8), dtype="float64")
                a[0] = z
                a[1] = 1
                dst.write(a)
        geometry = {}
        for name in ("original_water", "retention_water", "refined_topology"):
            p = tmp_path / f"{name}_input.gpkg"
            import fiona
            from fiona.crs import CRS

            with fiona.open(
                p,
                "w",
                driver="GPKG",
                schema={"geometry": "Polygon", "properties": {}},
                crs=CRS.from_epsg(4326),
                layer="boundary",
            ) as dst:
                dst.write(
                    {
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [-118.2, 34.1],
                                    [-118.19, 34.1],
                                    [-118.19, 34.09],
                                    [-118.2, 34.09],
                                    [-118.2, 34.1],
                                ]
                            ],
                        },
                        "properties": {},
                    }
                )
            geometry[name] = str(p)
        artifact = SimpleNamespace(
            products=geometry, layers={k: "boundary" for k in geometry}
        )
        entries = [
            (
                "module",
                {
                    "coast_refine_request": {
                        "group": "a",
                        "output_dir": str(tmp_path / "run"),
                    },
                    "_coast_mask_source_grid_binding": {"source_path": str(raw)},
                    "artifacts": {"side-stack": str(accepted)},
                },
            )
        ]
        catalog = mod.publish_grid_products(
            "a",
            entries,
            res=1,
            crs="EPSG:5498",
            strategy="priority",
            weight_threshold=1,
            artifact=artifact,
        )
        data, groups = read_catalog(catalog)
        assert not data["running"]
        assert data["defaults"] == {"default": ["before", "after"]}
        assert len(data["products"]) == 5
        assert [p["key"] for p in data["products"] if p.get("type") == "vector"] == [
            "original_water",
            "retention_water",
            "refined_topology",
        ]
        assert (
            sum(p.get("display", {}).get("visible", False) for p in data["products"])
            == 1
        )
        assert all(not os.path.isabs(p["path"]) for p in data["products"])
        assert check_catalog(catalog)["ok"]
    finally:
        for k, value in old.items():
            if value is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = value
