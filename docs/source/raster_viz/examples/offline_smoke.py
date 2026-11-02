"""Create synthetic display fixtures; this does not run or validate a science hook.

Run: python docs/source/raster_viz/examples/offline_smoke.py /path/to/smoke
Then: raster-viz /path/to/smoke/products.json --3d --mesh-side 100
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from globato.viz.raster_viz import ProductCatalog


def create(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[:48, :64]
    surface = (x - 36 + 4 * np.sin(y / 8)).astype("float32")
    profile = dict(
        driver="GTiff",
        width=64,
        height=48,
        count=1,
        dtype="float32",
        crs="EPSG:32610",
        transform=from_origin(500000, 4500000, 5, 5),
        nodata=-9999,
    )
    for name, values in (("before", surface), ("after", surface.copy())):
        values[10:18, 14:25] = 18 if name == "before" else -9999
        with rasterio.open(root / f"{name}.tif", "w", **profile) as dst:
            dst.write(values, 1)
    rgb = np.stack((100 + x, 80 + y, 150 + x // 2)).astype("uint8")
    with rasterio.open(
        root / "rgb.tif",
        "w",
        **{
            **profile,
            "count": 3,
            "dtype": "uint8",
            "nodata": None,
        },
    ) as dst:
        dst.write(rgb)
    gpd.GeoDataFrame(
        {"name": ["Illustrative boundary"]},
        geometry=[box(500070, 4499910, 500125, 4499950)],
        crs=profile["crs"],
    ).to_file(
        root / "boundaries.gpkg", layer="exclusion", driver="GPKG", engine="pyogrio"
    )
    c = ProductCatalog(root / "products.json", title="Synthetic display example")
    c.add_raster(
        "before", root / "before.tif", label="Illustrative input", group="measured"
    )
    c.add_raster(
        "after", root / "after.tif", label="Illustrative masked cells", group="measured"
    )
    c.add_raster("background", root / "rgb.tif", style="rgb")
    c.add_vector(
        "outline",
        root / "boundaries.gpkg",
        layer="exclusion",
        display={"visible": True, "color": "#dd7314"},
    )
    c.compare("before", "after")
    return c.publish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(create(parser.parse_args().output))  # noqa: T201 - CLI output: published catalog path
