"""Clipped area plus boundary contacts must retain strict polygon interiors."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, MultiPolygon, Polygon

from globato.coast_refine._io import digest, polygonal_geometry
from globato.coast_refine.artifact import MaskArtifact
from globato.coast_refine.grid_native_fusion import mask_fusion_state
from globato.coast_refine.masking import write_masked_tiff
from test_coast_mask_review import state_file


@pytest.mark.parametrize("contact", ["mixed", "line", "point"])
def test_source_scope_contacts_preserve_raster_and_fusion_contract(tmp_path, contact):
    scope = box(0, 0, 3.5, 4)
    touching = box(3.5, 0, 5, 1) if contact != "point" else box(3.5, 4, 5, 5)
    water = (
        MultiPolygon([box(0.5, 0.5, 2.5, 2.5), touching])
        if contact == "mixed"
        else touching
    )
    clipped = scope.intersection(water)
    assert clipped.geom_type in ("GeometryCollection", "LineString", "Point")
    artifact = object.__new__(MaskArtifact)
    artifact.aoi = box(-1, -1, 6, 6)
    artifact._classes = {"retention_water": water, "structures": Polygon()}
    artifact._coverage_covers = lambda *args: True
    keep = artifact.keep_geometry("land", scope)
    assert keep.geom_type in ("Polygon", "MultiPolygon")
    expected = np.zeros((4, 4), bool)
    if contact == "mixed":
        expected[2, 1] = True
        assert keep.equals(box(0.5, 0.5, 2.5, 2.5))
    else:
        assert keep.is_empty
    raw = tmp_path / "source.fusion.tif"
    data = state_file(raw, from_origin(0, 4, 1, 1), 4326, shape=(4, 4))
    checksum = digest(raw)
    with rasterio.open(raw) as src:
        profile = src.profile.copy()
    profile.update(count=1)
    scalar = tmp_path / "source.tif"
    with rasterio.open(scalar, "w", **profile) as dst:
        dst.write(np.full((4, 4), -2.0), 1)
    write_masked_tiff(scalar, tmp_path / "scalar_masked.tif", keep)
    receipt = mask_fusion_state(raw, tmp_path / "state_masked.tif", keep, 4326)
    with (
        rasterio.open(tmp_path / "scalar_masked.tif") as a,
        rasterio.open(tmp_path / "state_masked.tif") as b,
    ):
        assert np.array_equal(np.isfinite(a.read(1)), expected)
        result = b.read()
        assert np.array_equal(result[1] > 0, expected)
        assert result[:, expected].tobytes() == data[:, expected].tobytes()
        assert np.all(result[:, ~expected] == 0)
    assert digest(raw) == checksum
    assert receipt["counts"]["after_cells"] == int(expected.sum())


def test_polygon_normalization_does_not_repair_or_discard_small_area():
    from shapely.geometry import GeometryCollection, LineString

    tiny = box(1, 1, 1 + 1e-9, 1 + 1e-9)
    assert polygonal_geometry(tiny) is tiny
    mixed = GeometryCollection([tiny, LineString([(5, 5), (6, 6)])])
    assert polygonal_geometry(mixed).equals(tiny)
    with pytest.raises(ValueError, match="invalid"):
        polygonal_geometry(Polygon([(0, 0), (1, 1), (0, 1), (1, 0), (0, 0)]))
