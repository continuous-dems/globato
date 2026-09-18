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


def test_buffer_and_cut_without_a_region_leaves_the_config_alone():
    """A recipe with no region has nothing to buffer; it must not raise."""
    for config in ({"global_hooks": []}, {"region": None, "global_hooks": []}):
        expected = dict(config)

        assert RegionBufferModifier(pct=20).apply(config) == expected


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
