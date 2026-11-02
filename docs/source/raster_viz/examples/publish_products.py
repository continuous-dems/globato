"""Generic hook-side adapter. Accepts ANY existing input/output GeoTIFFs.

This example never imports a science hook or attempts to create the rasters.
"""

import argparse
from pathlib import Path
from globato.viz.raster_viz.catalog import ProductCatalog

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("catalog", help="Where to atomically publish products.json")
    p.add_argument("before", help="Completed original TIFF")
    p.add_argument("after", help="Completed modified TIFF")
    p.add_argument("--final", help="Optional interpolated DEM TIFF")
    p.add_argument(
        "--vector",
        action="append",
        default=[],
        help="Optional vector overlay (GeoJSON, GeoPackage, or Shapefile); repeat for multiple layers",
    )
    p.add_argument("--units", default=None)
    p.add_argument("--datum", default=None)
    a = p.parse_args()
    metadata = {
        k: v
        for k, v in [("value_units", a.units), ("vertical_datum", a.datum)]
        if v is not None
    }
    catalog = ProductCatalog(a.catalog, title="Raster comparison")
    catalog.add_raster(
        "before", a.before, label="Before", stage="before", metadata=metadata
    )
    catalog.add_raster(
        "after", a.after, label="After", stage="after", metadata=metadata
    )
    if a.final:
        catalog.add_raster(
            "final",
            a.final,
            label="Interpolated result",
            stage="final",
            group="final_dem",
            metadata=metadata,
        )
    for n, vector in enumerate(a.vector, 1):
        catalog.add_vector(f"vector_{n}", vector, label=Path(vector).stem)
    catalog.compare("before", "after")
    print(catalog.publish())
