"""Small contract checks that need no native DEM executable or external data."""

import pytest

from globato.hooks.filters.stack_qc import StackQC
from globato.qc.stack.review.publisher import ReviewPublisher


PUBLIC = {
    "aggressiveness",
    "terrain_restoration",
    "max_passes",
    "stop_when_stable",
    "review",
    "qc_res",
}


def test_six_settings_and_science_defaults_fixed():
    qc = StackQC()
    assert set(StackQC._cli_arg_help) == PUBLIC
    assert qc.stage == "collection"
    assert qc.review is False
    for obsolete in (
        "consensus",
        "survey_exclusion",
        "stable_changed_fraction",
        "stable_dem_change_m",
        "stable_consecutive_passes",
        "cache_dir",
        "stage",
    ):
        with pytest.raises(TypeError, match="accepts only"):
            StackQC(**{obsolete: None})
    defaults = qc.settings.to_dict()
    assert defaults["consensus"] is False
    assert defaults["survey_exclusion"] == "off"
    assert defaults["stable_changed_fraction"] == 0.00001
    assert defaults["stable_dem_change_m"] == 0.05
    assert defaults["stable_consecutive_passes"] == 2
    assert qc.cache_dir is None


def test_review_toggle_does_not_change_scientific_settings():
    off = StackQC(
        aggressiveness=3,
        terrain_restoration=1,
        max_passes=2,
        stop_when_stable=False,
        review=False,
    )
    on = StackQC(
        aggressiveness=3,
        terrain_restoration=1,
        max_passes=2,
        stop_when_stable=False,
        review=True,
    )
    assert on.review and not off.review
    assert off.settings.to_dict() == on.settings.to_dict()


def test_module_review_publishers_keep_separate_output_directories(
    tmp_path, monkeypatch
):
    """Each module publishes its own page without modifying the input entries."""
    from globato.qc.stack.review import visualize

    final = tmp_path / "combined_dem.tif"
    final.write_bytes(b"successfully finished production DEM")
    completed_calls = []

    def publish(run, **kwargs):
        completed_calls.append((run, kwargs["cache"], kwargs["final"]))
        (run / "review.html").write_text("Module review")
        return []

    monkeypatch.setattr(visualize, "completed", publish)

    class Interpolator:
        output = str(final)

    class Module:
        pass

    one = ReviewPublisher(
        cache=tmp_path / "cache_mbdb",
        result={},
        stack=Module(),
        region=Module(),
        interpolator=Interpolator(),
        module_name="mbdb",
    )
    two = ReviewPublisher(
        cache=tmp_path / "cache_other",
        result={},
        stack=Module(),
        region=Module(),
        interpolator=Interpolator(),
        module_name="other",
    )
    one.output = tmp_path / "reviews" / "mbdb"
    two.output = tmp_path / "reviews" / "other"
    original = [("entry", {"dst_fn": "source"})]
    assert one.run(original) is original
    assert two.run(original) is original
    assert (one.output / "review.html").exists()
    assert (two.output / "review.html").exists()
    assert len(completed_calls) == 2
    assert completed_calls[0][1] != completed_calls[1][1]
    assert all(call[2] == final for call in completed_calls)
    assert final.read_bytes() == b"successfully finished production DEM"
