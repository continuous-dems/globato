"""Colored GIS hillshades using Globato's existing Perspecto owner.

These are optional review artifacts, not part of Stack QC's scientific decisions.
All three surfaces use one color range for meaningful side-by-side comparison.
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling


def reference_limits(reference):
    """Estimate comparable display limits without loading a huge tile into RAM."""
    with rasterio.open(reference) as src:
        scale = min(1.0, (1_000_000 / (src.width * src.height)) ** 0.5)
        data = src.read(
            1,
            out_shape=(max(1, int(src.height * scale)), max(1, int(src.width * scale))),
            resampling=Resampling.nearest,
            masked=True,
        )
        vals = data.compressed().astype("float64")
        vals = vals[np.isfinite(vals)]
        if not vals.size:
            raise ValueError("Initial interpolated surface has no valid elevations")
        lo, hi = (float(v) for v in np.percentile(vals, (1, 99)))
        return lo, max(hi, lo + 1.0)


def colored_hillshade(source, destination, *, color_limits):
    """Reuse the actual Perspecto hillshade implementation, rather than fork it."""
    from globato.hooks.viz.geohillshade import GeoHillshade

    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.stem + ".partial.tif")
    temp.unlink(missing_ok=True)
    try:
        hook = GeoHillshade(
            azimuth=315,
            altitude=45,
            vert_exag=1,
            cmap="coastal_relief",
            blend_mode="soft_light",
            alpha=True,
            z_min=color_limits[0],
            z_max=color_limits[1],
        )
        entry = {"src_fn": str(source), "dst_fn": str(temp)}
        if not hook._process_file_fallback(str(source), str(temp), entry):
            raise RuntimeError(f"Perspecto could not shade {source}")
        with rasterio.open(temp, "r+") as dst:
            if dst.count not in (3, 4) or dst.dtypes[0] != "uint8":
                raise ValueError("Perspecto did not produce an RGB or RGBA GeoTIFF")
            dst.update_tags(
                STACK_QC_REVIEW="true",
                DISPLAY_COLOR_MIN_M=str(color_limits[0]),
                DISPLAY_COLOR_MAX_M=str(color_limits[1]),
                GENERATOR="globato.hooks.viz.geohillshade.GeoHillshade (Perspecto)",
            )
        temp.replace(destination)
        return destination
    finally:
        temp.unlink(missing_ok=True)
