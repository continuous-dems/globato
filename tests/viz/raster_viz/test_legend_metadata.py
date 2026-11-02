"""GeoTIFF-derived legend labels, without inventing a datum from horizontal CRS."""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from globato.viz.raster_viz.review import Review
from globato.viz.raster_viz.legends import raster_units


def tif(path: Path, *, crs="EPSG:5498", tags=None, band_tags=None, band_unit=None):
    z = np.arange(144, dtype="float32").reshape(12, 12)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=12,
        width=12,
        count=1,
        dtype="float32",
        crs=crs,
        transform=from_origin(-88, 46, 0.001, 0.001),
    ) as dst:
        dst.write(z, 1)
        if tags:
            dst.update_tags(**tags)
        if band_tags:
            dst.update_tags(1, **band_tags)
        if band_unit:
            dst.set_band_unit(1, band_unit)
    return path


def test_compound_crs_derives_navd88_but_horizontal_only_does_not(tmp_path):
    compound = tif(tmp_path / "compound.tif")
    with rasterio.open(compound) as src:
        assert raster_units(src, {}, 1.0, "elevation") == ("m", "NAVD88")
    # EPSG:4326 has angular horizontal units; it does not imply vertical metres.
    horizontal = tif(tmp_path / "horizontal.tif", crs="EPSG:4326")
    with rasterio.open(horizontal) as src:
        assert raster_units(src, {}, 1.0, "elevation") == ("", "")
        assert raster_units(src, {}, 1.0, "difference") == ("", "")


def test_tiff_tags_and_band_unit_are_used_without_guessing(tmp_path):
    f = tif(
        tmp_path / "tags.tif",
        crs="EPSG:4326",
        tags={"VERTICAL_DATUM": "NAVD88"},
        band_unit="m",
    )
    with rasterio.open(f) as src:
        assert raster_units(src, {}, 1.0, "elevation") == ("m", "NAVD88")
        assert raster_units(src, {}, 1.0, "difference") == ("m", "")
    g = tif(
        tmp_path / "band.tif",
        crs="EPSG:4326",
        band_tags={
            "Z_UNITS": "US survey foot",
            "VERTICAL_REFERENCE": "local bench mark",
        },
    )
    with rasterio.open(g) as src:
        assert raster_units(src, {}, 1.0, "elevation") == ("ftUS", "local bench mark")


def test_explicit_catalog_metadata_precedence_and_scalar_safety(tmp_path):
    f = tif(tmp_path / "z.tif", crs="EPSG:5498")
    with rasterio.open(f) as src:
        assert raster_units(
            src, {"value_units": "ft", "vertical_datum": "MLLW"}, 1.0, "elevation"
        ) == ("ft", "MLLW")
        assert raster_units(src, {}, 1.0, "scalar") == ("", "")
        assert raster_units(
            src,
            {"value_units": "probability", "vertical_datum": "NAVD88"},
            1.0,
            "scalar",
        ) == ("probability", "")
        assert raster_units(
            src, {"value_units": "m", "vertical_datum": "NAVD88"}, 1.0, "difference"
        ) == ("m", "")


def test_published_legend_and_minimal_viewer_title(tmp_path):
    f = tif(tmp_path / "navd88.tif")
    viewer = Review(tmp_path / "viz", render_3d=False)
    rec = viewer.add_raster("dem", f)
    assert rec["legend"]["units"] == "m"
    assert rec["legend"]["datum"] == "NAVD88"
    html = (tmp_path / "viz" / "viz.html").read_text()
    assert (
        "heading.textContent=info.title+(info.units?' ('+info.units+')':'')+(info.datum?' · '+info.datum:'');"
        in html
    )
    assert "' · Matplotlib '" not in html
    assert "' · Shared scale'" not in html
    rec2 = viewer.add_raster(
        "delta",
        f,
        group="difference",
        style="difference",
        metadata={"vertical_datum": "NAVD88"},
    )
    assert rec2["legend"]["datum"] == ""
    assert "datum:product.style==='elevation'" in html


def test_differences_are_not_auto_3d_terrain(tmp_path):
    """Height differences can be viewed, but not mistaken for absolute terrain."""
    f = tif(tmp_path / "height.tif")
    viewer = Review(tmp_path / "viz", render_3d=True, mesh_side=12)
    delta = viewer.add_raster("delta", f, style="difference", group="changes")
    assert "terrain_scene" not in delta
    assert "terrain_3d_status" not in delta
    elevation = viewer.add_raster("dem", f, style="elevation", group="elev")
    assert elevation["terrain_3d_status"] == "ready"
