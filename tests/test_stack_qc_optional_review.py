"""Opt-in review remains downstream of the ordinary, native final DEM build."""

from pathlib import Path
import rasterio
from test_stack_qc_native import run_plan as run_fetchez
from globato.hooks.filters.stack_qc import StackQC
from test_stack_qc_native import setup


def test_review_defaults_off_and_enables_with_bool_or_yaml_string():
    assert StackQC().review is False
    assert StackQC(review=False).review is False
    assert StackQC(review="false").review is False
    assert StackQC(review=True).review is True
    assert StackQC(review="true").review is True


def test_review_off_generates_no_png_or_html(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module, qc, hooks = setup(tmp_path)
    assert qc.review is False
    run_fetchez([module], threads=1, global_hooks=hooks, ignore_failures=False)
    assert (tmp_path / "normal_dem.tif").exists() or any(
        tmp_path.glob("normal_dem*.tif")
    )
    assert not (tmp_path / "review.html").exists()
    assert not (tmp_path / "review").exists()


def test_review_on_publishes_after_native_final_dem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module, qc, hooks = setup(tmp_path)
    qc.review = True
    original = Path(module.path).read_bytes()
    original_hooks = list(hooks)
    from globato.qc.stack.review.publisher import ReviewPublisher

    published_after_final = []
    publish = ReviewPublisher.run

    def assert_final_exists_at_publication(self, entries):
        assert self.final.is_file()
        published_after_final.append(self.final)
        return publish(self, entries)

    monkeypatch.setattr(ReviewPublisher, "run", assert_final_exists_at_publication)
    run_fetchez([module], threads=1, global_hooks=hooks, ignore_failures=False)
    # The source and scientific settings are identical to default-off.
    assert Path(module.path).read_bytes() == original
    assert (tmp_path / "review.html").exists()
    assert (tmp_path / "review/00_original.png").exists()
    assert (tmp_path / "review/99_final_dem.png").exists()
    assert (tmp_path / "review/91_restoration_footprint.png").exists()
    assert (tmp_path / "review/92_restored_cells_only.png").exists()
    assert (tmp_path / "review/93_all_final_measured_cells.png").exists()
    assert (tmp_path / "review/94_module_only_interpolated.png").exists()
    assert (tmp_path / "review/module_final_measured.tif").exists()
    assert (tmp_path / "review/module_final_measured_hillshade.tif").exists()
    assert (tmp_path / "review/module_cleaned_interpolated.tif").exists()
    assert (tmp_path / "review/module_cleaned_interpolated_hillshade.tif").exists()
    with rasterio.open(tmp_path / "normal_dem.tif") as ds:
        assert ds.count == 1
    html = (tmp_path / "review.html").read_text()
    assert "review/00_original.png" in html
    assert "review/99_final_dem.png" in html
    assert "review/92_restored_cells_only.png" in html
    assert "review/93_all_final_measured_cells.png" in html
    assert "review/94_module_only_interpolated.png" in html
    # The publisher is temporary: the compiled plan must be reusable.
    assert published_after_final == [(tmp_path / "normal_dem.tif").resolve()]
    assert hooks == original_hooks


def test_review_toggle_does_not_change_masks_or_final_dem(tmp_path, monkeypatch):
    """Same original observations, independent runs, identical scientific outputs."""
    import json
    import numpy as np

    shared = tmp_path / "inputs"
    shared.mkdir()
    shared_module, _, _ = setup(shared)

    def run_case(name, enabled):
        work = tmp_path / name
        work.mkdir()
        monkeypatch.chdir(work)
        module, qc, hooks = setup(work, passes=2, restore=2)
        module.path = shared_module.path
        module._discovered = False
        module.run()
        qc.review = enabled
        run_fetchez([module], threads=1, global_hooks=hooks, ignore_failures=False)
        report = json.loads(next((work / "qc").glob("*/result_*.json")).read_text())
        manifest = json.loads(Path(report["manifest"]).read_text())
        masks = {}
        for key, record in manifest["masks"].items():
            with rasterio.open(record["mask"]) as ds:
                masks[key] = ds.read(1)
        with rasterio.open(work / "normal_dem.tif") as ds:
            dem = ds.read(1, masked=True).filled(np.nan)
        return report, masks, dem

    before, before_masks, before_dem = run_case("off", False)
    after, after_masks, after_dem = run_case("on", True)
    assert before["passes_completed"] == after["passes_completed"]
    assert before["rejected_source_cells"] == after["rejected_source_cells"]
    assert before["restored_source_cells"] == after["restored_source_cells"]
    assert before_masks.keys() == after_masks.keys()
    for key in before_masks:
        np.testing.assert_array_equal(before_masks[key], after_masks[key])
    np.testing.assert_allclose(before_dem, after_dem, equal_nan=True, rtol=0, atol=0)


def test_two_reviewed_modules_publish_separate_pages(tmp_path, monkeypatch):
    """Both attachments review their own sources; Globato still owns one final DEM."""
    from test_stack_qc_native import run_plan as run_fetchez

    monkeypatch.chdir(tmp_path)
    first, first_qc, hooks = setup(tmp_path, passes=1, restore=0)
    first_qc.review = True
    second, second_qc, _ = setup(tmp_path, passes=1, restore=0)
    second_qc.review = True
    # The test module names are synthetic; each owns a separate source path.
    second_file = tmp_path / "second_points.xyz"
    second_file.write_bytes(Path(first.path).read_bytes())
    second.path = str(second_file)
    second.results[0]["dst_fn"] = str(second_file)
    second.results[0]["url"] = second_file.as_uri()
    second._outdir = str(tmp_path / "second_module")
    run_fetchez([first, second], threads=1, global_hooks=hooks, ignore_failures=False)
    pages = list(tmp_path.rglob("review.html"))
    assert len(pages) == 2
    assert len({p.parent for p in pages}) == 2
    for page in pages:
        html = page.read_text()
        assert "Module-only interpolated review DEM" in html
        assert "Final combined Globato DEM" in html
        assert (page.parent / "review/module_final_measured.tif").exists()
