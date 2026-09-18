# tests/test_modifiers.py

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
    assert output == "%name%_%batch_name%_crop.tif"


def test_buffer_and_cut_empty_outname_falls_back_to_default():
    config = RegionBufferModifier(pct=20, outname="").apply(_config())

    assert _crop_output(config) == "%name%_%batch_name%_crop.tif"


def test_buffer_and_cut_explicit_outname_is_kept():
    config = RegionBufferModifier(pct=20, outname="my_dem").apply(_config())

    assert _crop_output(config) == "my_dem_crop.tif"


def test_buffer_and_cut_injects_cut_then_crop_before_format_cog():
    config = RegionBufferModifier(pct=20).apply(_config())

    names = [h["name"] for h in config["global_hooks"]]
    assert names == ["raster_metadata", "raster_cut", "raster_crop", "format_cog"]
