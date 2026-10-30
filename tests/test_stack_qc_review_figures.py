"""Small maps used for review; they must not alter any Stack QC decisions."""

from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from globato.qc.stack.review.visualize import (
    expanded_cells,
    footprint_colors,
    measured_cells,
    removed,
    render_measurements,
)
from globato.qc.stack.review.review_page import publish


def raster(path, data, *, native=False):
    path = Path(path)
    if data.ndim == 2:
        data = data[None, ...]
    profile = dict(
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=str(data.dtype),
        crs="EPSG:32611",
        transform=from_origin(400000, 3700000, 10, 10),
    )
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(data)
        if native:
            ds.update_tags(GLOBATO_DATATYPE="MULTI_STACK")
    return path


def test_restored_black_symbol_covers_red_but_original_flags_are_untouched():
    red = np.zeros((30, 30), dtype=bool)
    red[14, 15] = True
    red[7, 8] = True
    restored = np.zeros_like(red)
    restored[14, 15] = True
    original_red = red.copy()
    rgba = footprint_colors(red, restored)
    np.testing.assert_array_equal(rgba[14, 15], [0, 0, 0, 1])
    np.testing.assert_array_equal(rgba[11, 15], [0, 0, 0, 1])
    np.testing.assert_allclose(rgba[10, 15], [1, 1, 1, 0.93], atol=1e-6)
    assert rgba[7, 8, 0] == 1.0  # unrelated rejected location stays red
    np.testing.assert_array_equal(red, original_red)
    assert expanded_cells(restored, 3).sum() == 49  # picture only, 7 × 7 mark


def test_restoration_visible_where_another_source_is_still_rejected(tmp_path):
    a = np.zeros((30, 30), dtype="uint8")
    a[14, 15] = 1
    no = np.zeros_like(a)
    before = {
        "masks": {
            "a": {"mask": str(raster(tmp_path / "before_a.tif", a))},
            "b": {"mask": str(raster(tmp_path / "before_b.tif", a))},
        }
    }
    after = {
        "masks": {
            "a": {"mask": str(raster(tmp_path / "after_a.tif", a))},
            "b": {"mask": str(raster(tmp_path / "after_b.tif", no))},
        }
    }
    baseline = raster(tmp_path / "dem.tif", a.astype("float32"))
    red = removed(after, baseline)
    restored = removed(after, baseline, restored_from=before)
    assert red[14, 15] and restored[14, 15]
    np.testing.assert_array_equal(footprint_colors(red, restored)[14, 15], [0, 0, 0, 1])


def test_final_measurements_are_selected_by_count_not_a_dem_fill(tmp_path):
    final = np.zeros((7, 18, 18), dtype="float32")
    final[0] = -120.0  # valid numeric elevation by itself does not prove a measurement
    final[1, 3, 3] = 1.0
    final[0, 3, 3] = -110.0
    final[1, 5, 8] = 2.0
    path = raster(tmp_path / "multistack.tif", final, native=True)
    z, coverage, _, _ = measured_cells(path)
    assert z[10, 10] == -120.0 and not coverage[10, 10]
    assert coverage.sum() == 2
    assert coverage[3, 3] and coverage[5, 8]
    raster(tmp_path / "not_multistack.tif", final)
    with pytest.raises(ValueError, match="seven-band MultiStack"):
        measured_cells(tmp_path / "not_multistack.tif")


def test_restored_only_and_all_measured_maps_with_empty_case(tmp_path):
    final = np.zeros((7, 18, 18), dtype="float32")
    final[0] = -120
    final[0, 3, 3] = -110
    final[1, 3, 3] = 1
    final[1, 5, 8] = 1
    stack = raster(tmp_path / "multistack.tif", final, native=True)
    terrain = raster(
        tmp_path / "baseline.tif",
        np.linspace(-180, -80, 324).reshape(18, 18).astype("float32"),
    )
    restored = np.zeros((18, 18), bool)
    restored[3, 3] = True
    selected = tmp_path / "92.png"
    all_cells = tmp_path / "93.png"
    render_measurements(stack, terrain, selected, restored=restored)
    render_measurements(stack, terrain, all_cells)
    assert selected.is_file() and selected.stat().st_size > 1000
    assert all_cells.is_file() and all_cells.stat().st_size > 1000
    render_measurements(
        stack, terrain, tmp_path / "none.png", restored=np.zeros_like(restored)
    )
    assert (tmp_path / "none.png").is_file()


def test_review_html_can_compare_footprints_and_only_restored(tmp_path):
    publish(
        tmp_path,
        [
            dict(
                label="After pass 2",
                status="Complete",
                image="review/02.png",
                overlay="review/02_footprint.png",
            ),
            dict(
                label="Only the restored measured cells",
                status="Ready",
                image="review/92_restored_cells_only.png",
            ),
            dict(
                label="All final measured cells",
                status="Ready",
                image="review/93_all_final_measured_cells.png",
            ),
        ],
        complete=True,
    )
    html = (tmp_path / "review.html").read_text()
    assert "review/92_restored_cells_only.png" in html
    assert "review/93_all_final_measured_cells.png" in html
    assert "review/02_footprint.png" in html
    assert "After pass 2 — removed / restored sites" in html
    assert 'http-equiv="refresh"' not in html


def test_completed_review_has_both_new_maps_and_comparable_pass_footprints(
    tmp_path, monkeypatch
):
    import json
    import sys
    import types
    from globato.qc.stack.review.visualize import completed

    shape = (20, 20)
    original = np.linspace(-200, -50, 400).reshape(shape).astype("float32")
    baseline = raster(tmp_path / "baseline.tif", original)
    final_dem = raster(tmp_path / "final_dem.tif", original + 0.5)
    native = np.zeros((7, *shape), "float32")
    native[0] = original
    native[1, 2:18, 2:18] = 1
    stack = raster(tmp_path / "production_stack.tif", native, native=True)
    removed_flags = np.zeros(shape, "uint8")
    removed_flags[9:12, 9:12] = 1
    final_flags = removed_flags.copy()
    final_flags[10, 10] = 0  # One source contribution restored
    first_mask = raster(tmp_path / "removed.tif", removed_flags)
    returned_mask = raster(tmp_path / "returned.tif", final_flags)
    root = tmp_path / "work"
    (root / "baseline").mkdir(parents=True)
    (root / "zero").mkdir()
    (root / "dems" / "pass1").mkdir(parents=True)
    (root / "baseline/COMPLETE.json").write_text(json.dumps(dict(dem=str(baseline))))
    (root / "zero/MANIFEST.json").write_text(json.dumps(dict(signature="start")))
    (root / "dems/pass1/COMPLETE.json").write_text(json.dumps(dict(dem=str(baseline))))
    manifest_before = tmp_path / "pass1.json"
    manifest_after = tmp_path / "final.json"
    manifest_before.write_text(
        json.dumps(
            dict(
                signature="pass1",
                masks={"a": dict(mask=str(first_mask))},
                newly_rejected_source_cells=9,
                rejected_source_cells=9,
            )
        )
    )
    manifest_after.write_text(
        json.dumps(dict(signature="final", masks={"a": dict(mask=str(returned_mask))}))
    )

    # This is the completed native product: review may read it, never recreate it.
    fake = types.ModuleType("globato.qc.stack.review.final_stages")
    fake.final_measured_stack = lambda *args: stack
    fake.write_measured_surface = lambda stack_path, output: raster(
        output, original.astype("float32")
    )
    fake.write_hillshade_tif = lambda surface_path, output: raster(
        output, np.ones(shape, dtype="uint8")
    )
    fake.module_interpolated_preview = lambda *args: raster(
        tmp_path / "module_preview.tif", original + 0.25
    )

    def no_extra_interpolation(*args):
        raise AssertionError("A completed review must not rebuild a DEM")

    fake.last_pass_stack = no_extra_interpolation
    monkeypatch.setitem(sys.modules, fake.__name__, fake)

    class Region:
        def to_list(self):
            return [-118.55, -118.50, 33.77, 33.81]

    result = dict(
        settings=dict(max_passes=1, aggressiveness=2, terrain_restoration=2),
        passes_completed=1,
        history=[dict(manifest=str(manifest_before))],
        manifest=str(manifest_after),
        restored_source_cells=1,
    )
    files = completed(
        tmp_path,
        cache=root,
        result=result,
        stack=object(),
        region=Region(),
        final=final_dem,
    )
    names = {p.name for p in files}
    assert {
        "01_pass_01_footprint.png",
        "91_restoration_footprint.png",
        "92_restored_cells_only.png",
        "93_all_final_measured_cells.png",
        "94_module_only_interpolated.png",
        "99_final_dem.png",
    } <= names
    html = (tmp_path / "review.html").read_text()
    assert "1 restored sites with final measurements" in html
    assert "256 final grid cells with measurements" in html
    assert "removed / restored sites" in html
    assert "module-only interpolated review DEM".lower() in html.lower()
