# Raster Viz

Raster Viz displays completed georeferenced rasters in two linked panels, with optional RGB backgrounds, vector overlays and WebGL2 terrain. It runs as a separate command after any processing workflow. Its code lives at `src/globato/viz/raster_viz/`; it has no imports from Stack QC, Coast Mask or their scientific implementations.

## Install and run

From a Globato checkout using Python 3.12 or later:

```bash
python -m pip install -e .
raster-viz --help
raster-viz-catalog --help
raster-viz-check --help
```

These three commands are installed by Globato. There is no additional viewer distribution to install. Rasterio, PyProj, Shapely and Pyogrio are existing Globato dependencies; Pillow and Filelock are declared direct dependencies for display and output locking. The module form is also available:

```bash
python -m globato.viz.raster_viz /path/to/products.json --3d --mesh-side 500
```

## Use a hook's existing catalog

```bash
raster-viz-check /path/to/products.json
raster-viz /path/to/products.json --3d --mesh-side 500
```

`--manifest /path/to/products.json` remains supported. By default output goes in `viz/` beside the catalog; use `--out DIRECTORY` to choose another location. Open `viz/viz.html` if the browser does not open automatically. Keep the HTML together with its `images/` and `scenes/` folders for offline viewing. `--no-open` renders without launching a browser.

Stack QC and Coast Mask keep publishing schema-v1 JSON themselves. Neither hook needs to import this viewer, change its scientific settings or bundle the viewer source. Render catalogs from any other producer using exactly the same commands.

## Inspect a directory

```bash
raster-viz /path/to/finished/rasters --3d --mesh-side 500
```

Directory mode scans only the top level by default. `--recursive` includes subdirectories, and `--max-files` bounds discovery (default 250; 0 means unlimited). Filename/metadata classification is a display heuristic; use an explicit catalog for reproducible comparisons. Raw native FusionState files must be finalized by their producer before elevation display.

## Create a catalog from completed files

```bash
raster-viz-catalog --out /path/to/run/products.json \
  --title 'Processing comparison' \
  --raster before=/path/to/before.tif --raster after=/path/to/after.tif \
  --rgb background=/path/to/rgb.tif \
  --vector boundary=/path/to/boundaries.gpkg \
  --layer boundary=retention_water --visible boundary \
  --compare before,after
raster-viz-check /path/to/run/products.json
raster-viz /path/to/run/products.json --3d
```

Only supply products that exist. All paths are stored relative to the catalog. Use repeated `--raster`, `--difference`, `--scalar`, `--grayscale`, `--categorical`, `--rgb` and `--vector KEY=FILE` declarations. `--label KEY=TEXT`, `--stage KEY=TEXT` and `--group KEY=NAME` set display semantics. `--band KEY=NUMBER` selects a raster band. `--layer KEY=NAME` selects a vector layer, and `--visible KEY` / `--hidden KEY` set its initial checkbox state. `--scan DIRECTORY` is available for automatic catalog generation.

## Publish from Python

```python
from globato.viz.raster_viz import ProductCatalog

c = ProductCatalog('/path/to/run/products.json', title='My hook')
c.add_raster('before', '/path/to/run/before.tif', group='measured')
c.add_raster('after', '/path/to/run/after.tif', group='measured')
c.add_vector('boundary', '/path/to/run/boundaries.gpkg', layer='retention_water',
             display={'visible': True, 'color': '#dd7314'})
c.compare('before', 'after')
c.publish()
```

`catalog.py` and `schema.py` use the Python standard library; importing the publisher does not load the viewer renderer. Importing through the Globato namespace still requires Globato's normal installation. An independent producer can instead write ordinary schema-v1 JSON using its own atomic writer, with no Globato dependency. [CATALOG.md](CATALOG.md) documents every field. Unknown producer details belong inside `metadata`.

The publisher and consumer use the same schema validator. An invalid update is rejected before replacing the previous catalog. This checks declarations, not whether the producer has finished writing a file: close each scientific output first. One writer owns each catalog; concurrent workers should publish separate catalogs or serialize updates.

For a live workflow, publish the completed products so far with `running=True`, then finish with `running=False`. Run the viewer in a separate process:

```bash
raster-viz /path/to/run/products.json --watch --3d
```

It can start before the catalog exists. `publish(running=True, rendering_enabled=False)` pauses source reads during timing or processing while allowing status updates; republish with rendering enabled to resume previews. Keep rendering outside benchmark timing.

## Comparisons and display

Products in one `group` share a numeric scale and 3D camera extent. Use the same group only for compatible units, datums and scientific scope. `area` separates geographic cases. Keys, labels and stages are producer-defined; there are no required hook names or processing passes.

- Stack QC's four products are measured before/after, working interpolation and final combined DEM. Only the measured pair shares a group in the supplied hook.
- Coast Mask's native catalog contains participant measurements before/after and three optional boundary overlays. It does not imply a whole-DEM before/after comparison.
- Other hooks can publish any number of products, including differences, scalar statistics, categories, imagery and named vector layers. The templates under [examples](examples/WORKFLOWS.md) are examples, not mandatory output contracts.

RGB is a selectable background in 2D/3D and remains separate from DEM opacity. Vector checkboxes are 2D overlays; 3D vector draping is not implemented. Terrain meshes exclude source NoData triangles and use bounded previews. A modern browser with WebGL2 is needed for 3D.

The elevation legend reads height units and vertical reference from supported GeoTIFF metadata; unknown values remain unspecified. Explicit verified catalog metadata takes precedence. Horizontal CRS alone never establishes a vertical datum. Mixed units/datums are not converted automatically.

`water_level` is a color threshold in source elevation units, default 0. Set a verified lake reference elevation with `--water-level VALUE` or an elevation product's `water_level` field. One explicit value applies to its whole comparison group; conflicting values are rejected. `--vertical-unit-to-m` separately scales display hillshade/terrain heights (e.g. 0.3048 for international feet). Neither setting changes source rasters or performs a datum conversion.

## Validation and return codes

`raster-viz-check` validates the catalog, raster CRS/bands, RGB band count, vector readability/selected layer/CRS and shapefile companions. It rejects raw FusionStates. It reads vector metadata rather than rendering geometry; full vertex-budget and visual checks occur during rendering. An empty vector is a warning, since a legitimate hook result may have no boundary features.

```bash
raster-viz-check /path/to/products.json --json
raster-viz-check /path/to/products.json --allow-pending
```

Commands return 0 on success and 2 on invalid input or failed rendering. A completed catalog with missing products fails, including when a watch finishes. A running catalog may still have pending products; `--allow-pending` treats those as warnings during preflight. Directory mode returns 2 if any discovered source cannot be rendered. Partial display results remain available for diagnosis. Missing, removed or failed replacements cannot silently retain old catalog-owned previews.

From the Globato repository:

```bash
python -m pytest -q tests/viz/raster_viz
python -m compileall -q src/globato/viz
python -m build
```

Browser tests run when Playwright and Chromium are available; remaining tests exercise the catalog, installed command mapping, 2D/3D assets, RGB/opacity logic, layers, watch/refresh and source immutability. Real hook/provider runs and visual browser inspection remain release checks.

```{toctree}
:hidden:

CATALOG
examples/WORKFLOWS
```
