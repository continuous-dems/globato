#!/usr/bin/env python3
"""Publish a minimal generic DEM + georeferenced RGB catalog (stdlib wrapper)."""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from globato.viz.raster_viz.catalog import ProductCatalog

p = argparse.ArgumentParser()
p.add_argument("--dem", type=Path, required=True)
p.add_argument("--rgb", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument(
    "--water-level",
    type=float,
    help="Optional elevation palette threshold (TIFF vertical units; default 0)",
)
a = p.parse_args()
with ProductCatalog(a.out, title="DEM with RGB background") as c:
    c.add_raster(
        "dem",
        a.dem,
        label="Elevation DEM",
        style="elevation",
        water_level=a.water_level,
    )
    c.add_raster("rgb", a.rgb, label="Satellite RGB", style="rgb")
    c.compare("dem", "dem")
print(a.out)  # noqa: T201 - CLI output: published catalog path
