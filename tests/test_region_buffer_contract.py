"""Processing margins extend the native modifier without changing delivery."""

import pytest
from globato.recipes.modifiers.buffer_and_cut import RegionBufferModifier


def recipe():
    return dict(
        region=[0, 1, 0, 1],
        global_hooks=[dict(name="ms_binary_cudem"), dict(name="stack_qc_review")],
    )


def test_processing_buffer_preserves_delivery_and_crops_before_review():
    result = RegionBufferModifier(buffer_cells=40, cells=0, inc="1s").apply(recipe())
    margin = 40 / 3600
    assert result["region"][:4] == pytest.approx(
        [-margin, 1 + margin, -margin, 1 + margin]
    )
    hooks = result["global_hooks"]
    assert [h["name"] for h in hooks] == [
        "ms_binary_cudem",
        "raster_cut",
        "raster_crop",
        "stack_qc_review",
    ]
    assert hooks[1]["args"]["region"][:4] == [0, 1, 0, 1]


def test_legacy_delivery_overlap_keeps_native_semantics():
    result = RegionBufferModifier(cells=2, inc="1s").apply(recipe())
    margin = 2 / 3600
    expected = [-margin, 1 + margin, -margin, 1 + margin]
    # Legacy cells affects delivery only; processing bounds stay as supplied.
    assert result["region"][:4] == [0, 1, 0, 1]
    assert result["global_hooks"][1]["args"]["region"][:4] == pytest.approx(expected)


@pytest.mark.parametrize("value", [0, -1, 1.5, float("nan")])
def test_processing_buffer_rejects_invalid_cell_count(value):
    with pytest.raises(ValueError, match="positive integer"):
        RegionBufferModifier(buffer_cells=value)


def test_processing_buffer_rejects_conflicting_margins():
    with pytest.raises(ValueError, match="pct or buffer_cells"):
        RegionBufferModifier(buffer_cells=40, pct=5)
    with pytest.raises(ValueError, match="exceed delivery"):
        RegionBufferModifier(buffer_cells=40, cells=40).apply(recipe())
