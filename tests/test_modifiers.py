# tests/test_modifiers.py

from pathlib import Path

import yaml

import globato
from globato.recipes.modifiers.buffer_and_cut import RegionBufferModifier


def _config():
    return {
        "region": [-120.5, -120.25, 35.75, 36.0],
        "global_hooks": [{"name": "raster_metadata"}, {"name": "format_cog"}],
    }


def _crop_output(config):
    crop = [h for h in config["global_hooks"] if h["name"] == "raster_crop"]
    assert len(crop) == 1
    return crop[0]["args"]["output"]


def test_buffer_and_cut_default_outname_is_not_none():
    """Prove that omitting 'outname' does not put a literal 'None' in the output name."""
    config = RegionBufferModifier(pct=20).apply(_config())

    output = _crop_output(config)
    assert "None" not in output
    # The recipe runner fills these in for each tile after modifiers are applied.
    assert output == "%name%_%batch_name%.tif"


def test_buffer_and_cut_empty_outname_falls_back_to_default():
    config = RegionBufferModifier(pct=20, outname="").apply(_config())

    assert _crop_output(config) == "%name%_%batch_name%.tif"


def test_buffer_and_cut_explicit_outname_is_kept():
    config = RegionBufferModifier(pct=20, outname="my_dem").apply(_config())

    assert _crop_output(config) == "my_dem.tif"


def test_buffer_and_cut_injects_cut_then_crop_before_format_cog():
    config = RegionBufferModifier(pct=20).apply(_config())

    names = [h["name"] for h in config["global_hooks"]]
    assert names == ["raster_metadata", "raster_cut", "raster_crop", "format_cog"]


def test_buffer_and_cut_injects_before_the_first_format_cog():
    """With several format_cog hooks, everything after the DEM must see the cropped DEM."""
    config = _config()
    config["global_hooks"] += [{"name": "viz_geoshade"}, {"name": "format-cog"}]

    names = [
        h["name"] for h in RegionBufferModifier(pct=20).apply(config)["global_hooks"]
    ]
    assert names == [
        "raster_metadata",
        "raster_cut",
        "raster_crop",
        "format_cog",
        "viz_geoshade",
        "format-cog",
    ]


def _names_after_apply(hook_names):
    config = {
        "region": _config()["region"],
        "global_hooks": [{"name": n} for n in hook_names],
    }
    return [
        h["name"] for h in RegionBufferModifier(pct=20).apply(config)["global_hooks"]
    ]


def test_buffer_and_cut_does_not_need_a_format_cog():
    """Without format_cog, the cut/crop must still precede whatever uses the DEM.

    Appended at the end they ran after copy_artifact, so the buffered DEM and a
    buffered hillshade were delivered.
    """
    names = _names_after_apply(
        ["ms_binary_cudem", "viz_geoshade", "cleanup_tmp", "copy_artifact"]
    )
    assert names == [
        "ms_binary_cudem",
        "raster_cut",
        "raster_crop",
        "viz_geoshade",
        "cleanup_tmp",
        "copy_artifact",
    ]


def test_buffer_and_cut_precedes_a_lone_copy_artifact():
    names = _names_after_apply(["ms_binary_cudem", "copy-artifact"])
    assert names == ["ms_binary_cudem", "raster_cut", "raster_crop", "copy-artifact"]


def test_buffer_and_cut_is_appended_when_nothing_uses_the_dem():
    names = _names_after_apply(["multi_stack", "focus_sink", "ms_blend"])
    assert names == [
        "multi_stack",
        "focus_sink",
        "ms_blend",
        "raster_cut",
        "raster_crop",
    ]


def test_buffer_and_cut_position_in_mr_globato_is_unchanged():
    """format_cog is the first hook in the preset that uses the DEM."""
    preset_fn = Path(globato.__file__).parent / "hooks" / "presets" / "mr_globato.yaml"
    hooks = yaml.safe_load(preset_fn.read_text())["hooks"]

    names = _names_after_apply([h["name"] for h in hooks])
    i = names.index("raster_cut")
    assert names[i - 1 : i + 3] == [
        "raster_metadata",
        "raster_cut",
        "raster_crop",
        "format_cog",
    ]


def test_buffer_and_cut_still_sees_a_raster_cut_after_format_cog():
    config = _config()
    config["global_hooks"].append({"name": "raster_cut"})

    config = RegionBufferModifier(pct=20).apply(config)

    assert config["region"] == _config()["region"]
    assert "raster_crop" not in [h["name"] for h in config["global_hooks"]]


def test_buffer_and_cut_keeps_its_hooks_when_the_config_has_no_global_hooks():
    """A buffered region with no cut/crop to undo it would deliver an oversize DEM."""
    for config in ({}, {"global_hooks": None}):
        config["region"] = _config()["region"]

        config = RegionBufferModifier(pct=20).apply(config)

        assert config["region"] != _config()["region"]
        assert [h["name"] for h in config["global_hooks"]] == [
            "raster_cut",
            "raster_crop",
        ]


def test_buffer_and_cut_without_a_region_leaves_the_config_alone(caplog):
    """A recipe with no region has nothing to buffer; it must warn, not raise."""
    for config in ({"global_hooks": []}, {"region": None, "global_hooks": []}):
        expected = dict(config)
        caplog.clear()

        with caplog.at_level("WARNING"):
            assert RegionBufferModifier(pct=20).apply(config) == expected

        assert "No region set in the recipe" in caplog.text


def test_buffer_and_cut_with_a_region_does_not_warn_about_it(caplog):
    with caplog.at_level("WARNING"):
        RegionBufferModifier(pct=20).apply(_config())

    assert "No region set" not in caplog.text


def test_buffer_and_cut_defaults_to_a_5_pct_buffer_when_none_is_given():
    no_buffer = RegionBufferModifier().apply(_config())
    five_pct = RegionBufferModifier(pct=5).apply(_config())

    assert no_buffer["region"] != _config()["region"]
    assert no_buffer["region"] == five_pct["region"]


def test_buffer_and_cut_explicit_zero_buffer_is_respected():
    for kwargs in ({"pct": 0}, {"cells": 0}, {"pct": "0", "cells": "0"}):
        config = RegionBufferModifier(**kwargs).apply(_config())

        assert config["region"] == _config()["region"]


def test_buffer_and_cut_default_replaces_the_preset_dem():
    """The cropped DEM must overwrite the buffered DEM that mr-globato delivers.

    copy_artifact matches on the DEM's name, so if the crop wrote anywhere
    else the un-cropped (buffered) DEM would be the one delivered.
    """
    preset_fn = Path(globato.__file__).parent / "hooks" / "presets" / "mr_globato.yaml"
    hooks = yaml.safe_load(preset_fn.read_text())["hooks"]

    dem_output = [h for h in hooks if h["name"] == "ms_binary_cudem"][0]["args"][
        "output"
    ]
    delivered = [
        m for h in hooks if h["name"] == "copy_artifact" for m in h["args"]["match"]
    ]

    config = RegionBufferModifier(pct=20).apply(_config())

    assert _crop_output(config) == dem_output
    assert _crop_output(config) in delivered
