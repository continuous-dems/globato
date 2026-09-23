"""Source-mask pixels must be georeferenced on their actual PointPixels grid."""

import numpy as np
import pytest
import rasterio
from fetchez.spatial import Region
from globato.hooks.metadata.provenance import SourceMasks
from globato.hooks.transforms.point_pixels import PointPixels
from rasterio.features import rasterize
from shapely.geometry import box, mapping


@pytest.mark.parametrize("extent", [(3.2, 2.2), (3.0, 2.0)])
def test_source_mask_georeferencing_matches_point_binner(tmp_path, extent):
    region = Region(0.0, extent[0], 0.0, extent[1])
    hook = SourceMasks(res=1, output=str(tmp_path / "sources.vrt"))
    hook._init_grid(region)

    binner = PointPixels(src_region=region, x_size=hook.xcount, y_size=hook.ycount)
    binner.init_gt()
    assert hook.transform == rasterio.Affine.from_gdal(*binner.dst_gt)

    # When the extent does not divide evenly into whole nominal-size cells,
    # the former from_origin(..., 1, 1) TIFF misregistered right/bottom cells.
    x, y = extent[0] - 0.02, 0.02
    points = np.array([(x, y, 4.0)], dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")])
    tif = tmp_path / "source_mask.tif"
    with rasterio.open(tif, "w", **hook.profile) as dst:
        dst.write(np.zeros((hook.ycount, hook.xcount), dtype="uint8"), 1)
    list(hook._intercept(iter((points,)), str(tif), region))

    with rasterio.open(tif) as src:
        pixels = src.read(1) > 0
        row, col = rasterio.transform.rowcol(src.transform, x, y)
        assert int(pixels.sum()) == 1
        assert pixels[row, col]
        accepted = box(x - 1e-5, y - 1e-5, x + 1e-5, y + 1e-5)
        allowed = (
            rasterize(
                [(mapping(accepted), 1)],
                out_shape=src.shape,
                transform=src.transform,
                all_touched=True,
                dtype="uint8",
            )
            > 0
        )
        assert not np.any(pixels & ~allowed)
