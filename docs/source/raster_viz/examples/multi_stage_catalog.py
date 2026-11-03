"""Pattern for any scientific producer with zero, one or many processing passes.

Producer supplies *completed* files. Never copy this script's sample stages as
mandatory names: name the products from your own processing pipeline.
"""

from globato.viz.raster_viz.catalog import ProductCatalog


def publish_stages(
    catalog_path,
    source,
    intermediate,
    result,
    *,
    final_dem=None,
    vector_overlay=None,
    units=None,
    vertical_datum=None,
):
    """Hook/workflow calls this after all paths exist. This file does no science."""
    md = {
        k: v
        for k, v in [("value_units", units), ("vertical_datum", vertical_datum)]
        if v is not None
    }
    out = ProductCatalog(catalog_path, title="Processing comparison")
    out.add_raster(
        "original", source, label="Original measurements", stage="before", metadata=md
    )
    for n, filename in enumerate(intermediate, 1):
        out.add_raster(
            f"pass_{n}",
            filename,
            label=f"After pass {n}",
            stage=f"pass {n}",
            metadata=md,
        )
    out.add_raster(
        "result", result, label="Retained measurements", stage="after", metadata=md
    )
    if final_dem is not None:
        out.add_raster(
            "interpolated",
            final_dem,
            label="Final interpolated DEM",
            stage="final",
            group="final_dem",
            metadata=md,
        )
    if vector_overlay is not None:
        out.add_vector("boundary", vector_overlay, label="Vector boundary")
    out.compare("original", "result")
    return out.publish()


# Example caller from your own hook: publish_stages(run/'products.json', orig, pass_rasters, final, final_dem=dem)
