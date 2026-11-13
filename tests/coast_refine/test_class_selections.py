"""Authoritative subtypes and ocean connectivity share one native mask pipeline."""

from types import SimpleNamespace
import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
import shapely
from globato.coast_refine.artifact import MODES
from globato.coast_refine.prepare import prepare_from_baseline


@pytest.fixture(scope="module")
def river_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("connected_river")
    size = 64
    cell = 1 / 32400
    span = size * cell
    scope = box(0, 0, span, span)
    context = box(-span, -span, 2 * span, 2 * span)
    ocean = box(-span, -span, 20 * cell, 2 * span)
    river = box(20 * cell, 30 * cell, 44 * cell, 34 * cell)
    lake = box(48 * cell, 45 * cell, 54 * cell, 51 * cell)
    water = shapely.union_all([ocean, river, lake])
    land = context.difference(water)
    frame = gpd.GeoDataFrame(
        {"class": ["ocean", "river", "lake", "land"]},
        geometry=[ocean, river, lake, land],
        crs=4326,
    )
    peer = root / "osm.gpkg"
    frame.to_file(peer)
    source = root / "original.tif"
    y, x = np.indices((size, size))
    wet = (
        (x < 20)
        | ((x >= 20) & (x < 44) & (y >= 30) & (y < 34))
        | ((x >= 48) & (x < 54) & (y >= 13) & (y < 19))
    )
    # Raster row order is north to south. The connected river occupies y=30..34
    # here too because the grid has 64 rows.
    z = np.where(wet, -5.0, 5.0).astype("float32")
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        count=1,
        width=size,
        height=size,
        dtype="float32",
        crs=4326,
        transform=from_origin(0, span, cell, cell),
        nodata=np.nan,
    ) as ds:
        ds.write(z, 1)
        ds.set_band_unit(1, "m")
    area = root / "area.gpkg"
    gpd.GeoDataFrame(geometry=[scope], crs=4326).to_file(area)
    import globato.utils

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(globato.utils, "resolve_barrier", lambda *a, **k: str(peer))
        artifact = prepare_from_baseline(
            source,
            area,
            root / "prepared",
            root / "cache",
            mask="osm",
            context_region=f"{-span}/{2 * span}/{-span}/{2 * span}",
        )
    return SimpleNamespace(
        root=root,
        source=source,
        area=area,
        peer=peer,
        context=f"{-span}/{2 * span}/{-span}/{2 * span}",
        artifact=artifact,
        z=z,
        wet=wet,
        x=x,
        y=y,
    )


@pytest.mark.parametrize("selection", MODES)
def test_five_selections_share_osm_and_local_attribute_logic(
    river_case, tmp_path, selection
):
    from globato.coast_refine.masking import write_masked_tiff

    c = river_case
    ocean = c.x < 20
    river = (c.x >= 20) & (c.x < 44) & (c.y >= 30) & (c.y < 34)
    lake = (c.x >= 48) & (c.x < 54) & (c.y >= 13) & (c.y < 19)
    remove = {
        "ocean": ocean,
        "ocean_connected": ocean | river,
        "inland_water": river | lake,
        "water": ocean | river | lake,
        "land": ~(ocean | river | lake),
    }[selection]
    local = prepare_from_baseline(
        c.source,
        c.area,
        tmp_path / "local",
        tmp_path / "cache",
        mask=c.peer,
        polygon_class="osm",
        context_region=c.context,
    )
    for name, artifact in [("osm", c.artifact), ("local", local)]:
        output = tmp_path / (name + ".tif")
        write_masked_tiff(c.source, output, artifact.keep_geometry(selection))
        with rasterio.open(output) as ds:
            valid = (ds.read_masks(1) > 0) & np.isfinite(ds.read(1))
        np.testing.assert_array_equal(valid, ~remove)
