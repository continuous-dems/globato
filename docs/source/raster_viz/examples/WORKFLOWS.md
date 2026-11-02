# Applying the same catalog to different data-producing workflows

Run the following **after** the listed files are completely written. Paths are illustrative filenames; substitute real products. `raster-viz-catalog` does not calculate a difference, rasterize points or interpolate an elevation grid.

## A. Globato final build (gridded and interpolated)

```bash
raster-viz-catalog --out /path/to/globato_run/products.json \
  --title 'Globato DEM products' \
  --raster measured=/path/to/globato_run/gridded_measured.tif \
  --raster dem=/path/to/globato_run/interpolated_dem.tif \
  --categorical provenance=/path/to/globato_run/provenance.tif \
  --compare measured,dem \
  --label measured='Measured cells' --label dem='Interpolated DEM'
raster-viz --manifest /path/to/globato_run/products.json --3d --out /path/to/globato_run/viz
```

The *producer* must supply gridded / interpolated rasters; these filenames do not imply Globato emits products under these exact names.

## B. Stack QC / Coast Refine before, passes and after

```bash
raster-viz-catalog --out /path/to/qc_run/products.json \
  --raster before=/path/to/qc_run/original.tif \
  --raster pass_1=/path/to/qc_run/pass1.tif \
  --raster pass_2=/path/to/qc_run/pass2.tif \
  --raster after=/path/to/qc_run/filtered.tif \
  --raster final=/path/to/qc_run/interpolated.tif \
  --compare before,after \
  --label before='Before QC' --label after='After QC'
raster-viz --manifest /path/to/qc_run/products.json --3d --out /path/to/qc_run/viz
```

If your QC runner does not save intermediate rasters, omit those flags. This viewer does not force scientific hooks to create any particular products.

## C. Lake Michigan / other Sentinel-2 + DEM pilot

```bash
raster-viz-catalog --out /path/to/sdb_run/products.json \
  --rgb s2=/path/to/RGB_10m.tif \
  --raster lidar=/path/to/lidar_measured.tif \
  --raster combined=/path/to/combined_dem.tif \
  --difference residual=/path/to/residual.tif \
  --compare lidar,combined --water-level 176.0
raster-viz --manifest /path/to/sdb_run/products.json --3d --mesh-side 500 \
  --out /path/to/sdb_run/viz
```

RGB is selectable as a georeferenced background plane in 2D or 3D, independent of the DEM mesh. The `176.0` water-level threshold above is only an illustrative placeholder. Determine the actual lake level in the **same vertical datum and units** as your DEM; this is strictly a color transition, not a water-level determination or vertical conversion.

The equivalent editable catalog is `examples/hook_templates/freshwater_topobathy_products.json` (replace paths and the illustrative threshold).

## D. Simple vector overlay

```bash
raster-viz-catalog --out /path/to/run/products.json \
  --raster topo=/path/to/topo.tif \
  --vector coastline=/path/to/coastline.gpkg \
  --compare topo,topo
raster-viz --manifest /path/to/run/products.json --out /path/to/run/viz
```

Globato already installs the vector reader dependencies. Overlay polygons are not an interpolation barrier, land/water mask or science processing step.

## E. Publish new results from any Python hook (no coupling)

```python
from globato.viz.raster_viz import ProductCatalog

# Called after writing each output file, e.g. once per QC pass.
products = ProductCatalog('/path/to/run/products.json', title='My workflow')
products.add_raster('raw', '/path/to/run/raw.tif', label='Original')
products.publish(running=True, status='Original available')

# Later in the same process:
products.add_raster('final', '/path/to/run/final.tif', label='Final DEM')
products.compare('raw', 'final')
products.publish(running=False, status='Complete')
```

The publisher and its schema validator use only the standard library, although importing via Globato requires its normal installation. A producer with no Globato dependency can write equivalent JSON using its own atomic writer. **Only one writer should update any given JSON path.**

## F. Explicit views for specialized difference surfaces

Differences are shown in 2D by default, including file names like `bathy_diff.tif`. A scientific producer that deliberately wants a 3D visualization of a difference surface can opt in without modifying a global viewer hook:

```python
from globato.viz.raster_viz import ProductCatalog
with ProductCatalog('/path/to/run/products.json') as c:
    c.add_raster('before', '/path/to/before.tif', label='Before')
    c.add_raster('after', '/path/to/after.tif', label='After')
    c.add_raster('change', '/path/to/difference.tif', label='Change',
                 style='difference', group='difference', views=['2d', '3d'])
    c.compare('before', 'after')
```

Invoke `raster-viz --manifest /path/to/run/products.json --3d --mesh-side 500 --out /path/to/run/viz`. This is a graphical representation of *differences*, not a claim that they are heights relative to NAVD88.

## G. Coast Mask: one visible boundary, two optional diagnostics

The standard template at `examples/hook_templates/coast_mask_products.json`
shows a 2D before/after comparison with three clearly named vectors.
Only the **source-exclusion boundary** is visible initially. Open
**Boundaries (1/3)** to reveal the original water boundary or
interpolation topology individually. Blue dashed, orange solid, and purple
dotted strokes distinguish overlapping geometries. This setting affects
**display only** and does not change masking or interpolation.
