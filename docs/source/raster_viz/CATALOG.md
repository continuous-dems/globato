# Small producer-to-viewer contract

A producer writes one `products.json` per run or tile, using atomic rename. It can use `raster-viz-catalog` for one-shot catalogs or the stdlib-only `ProductCatalog` for progressive hook publication. All relative paths in `products` are resolved **relative to that JSON file**. Only publish a raster **after** it is closed and complete. RGB products can only be selected as backgrounds, not as the left/right default raster. No source module-specific names or required stages exist.

Minimum file:

```json
{"schema_version":1,"products":[{"key":"terrain","path":"../dem.tif"}]}
```

Recommended for meaningful comparisons:

```json
{
  "schema_version": 1,
  "title": "Example workflow",
  "running": false,
  "products": [
    {"key":"before", "path":"../results/before.tif", "label":"Before", "stage":"before", "group":"elevation", "style":"elevation"},
    {"key":"after", "path":"../results/after.tif", "label":"After", "stage":"after", "group":"elevation", "style":"elevation"},
    {"key":"final", "path":"../results/dem.tif", "label":"Interpolated DEM", "group":"elevation", "style":"elevation"}
  ],
  "defaults":{"default":["before","after"]}
}
```

Fields:

| Field | Purpose |
|---|---|
| `key` | Unique stable product identifier within an area (required) |
| `path` | Existing GeoTIFF or vector path; relative to JSON (required) |
| `label` | Friendly label for a drop-down |
| `style` | `elevation` (default), `difference`, `scalar`, `grayscale`, `categorical`, `rgb` (georeferenced 3-band imagery) |
| `stage` | Optional arbitrary producer label (`before`, `pass 1`, `after`, etc.) |
| `group` | Shared color range group within one area; use separate groups for unrelated quantities |
| `area` | Optional name for a tile/study area (default `default`) |
| `band` | 1-based raster band (default 1) |
| `views` | Optional `["2d"]` or `["2d", "3d"]`. Difference rasters are 2D unless 3D is explicitly requested; only elevation/difference can use 3D. |
| `water_level` | **Optional elevation-only** numeric threshold for switching from blue below to land colors at/above; default `0` in original TIFF vertical units. Shared by every product in the comparison `group`; does not modify elevations or datums. |
| `limits` | Optional `[min,max]` for reproducible shared colors; otherwise v13 estimates jointly across all available rasters in the same group, with 1st–99th percentiles; differences are centered at zero |
| `metadata` | Arbitrary additional dictionary; `value_units` and `vertical_datum` explicitly label the legend when known; missing values can also come from source GeoTIFF vertical CRS or relevant TIFF metadata |
| `vector_layer` | Optional real vector dataset layer name; the legacy `layer` field remains opaque metadata |
| `type` | `vector` for optional GeoPackage, GeoJSON, or Shapefile overlays (default `raster`); shown in 2D, toggle in viewer |
| `defaults` | `{area: [left_key,right_key]}` selects initial pair; no heuristic pairing/presets |
| `rendering_enabled` | Optional boolean, default true; false pauses source reads while status can update |
| `running` | `true` means producer is still publishing, `false` lets a watch process exit |
| `status`, `metrics`, `tables` | Optional human-readable diagnostics; viewer only displays them |

### Water/land display boundary (independent of water-mask science)

A coastal DEM generally uses the default threshold **0**. A freshwater-lake DEM may require a different elevation in its own vertical datum:

```json
{
  "schema_version": 1,
  "products": [
    {"key":"before", "path":"before.tif", "style":"elevation",
     "group":"elevation", "water_level":176.0},
    {"key":"after", "path":"after.tif", "style":"elevation",
     "group":"elevation"},
    {"key":"rgb", "path":"sentinel_rgb.tif", "style":"rgb"}
  ],
  "defaults": {"default":["before","after"]}
}
```

`176.0` is an illustrative value, **not an authoritative Great Lakes water level**. Supply the actual elevation for your survey period and vertical datum, in **the TIFF elevation units**; a `ft` TIFF expects a `ft` threshold. The threshold colors values below it blue, and values at/above it green/tan/cream. This never modifies a raster, infers shoreline, or converts NAVD88 / IGLD85. Only elevation products support this field; RGB/difference/scalar/categorical products must not declare it. One explicit declaration per comparison group is enough; all group members inherit the same value. Conflicting explicit values in the same group are errors. For directory mode, use `raster-viz --water-level VALUE`; for the catalog generator, use `raster-viz-catalog --water-level VALUE`.

**Recommended group practice:** put raster data with comparable units and meaning into the same group, e.g. `elevation` for both before and after. Use `density` or `difference` for independent statistics. If you know an authoritative range, include the **same** `limits` for every product in that group. Without provided limits, v13 estimates them jointly using bounded overviews; these may shift as files arrive during a live run and cause previews to be regenerated. Producer-defined explicit limits are best for fixed run-to-run comparisons.

**Producer integration, independent of any specific hook:**

1. The hook writes its own source rasters before/after each relevant stage and finishes the files.
2. Its own small publication adapter maps completed paths to keys, labels, styles, and group names, using `ProductCatalog` (or writes equivalent JSON).
3. It calls `compare('before','after')` for the initial pair and atomically publishes `products.json`.
4. The generic `raster-viz --manifest ...` process renders the declared files; it never executes or imports the hook.

The caller decides which products are scientifically meaningful. For pass-by-pass workflows, simply call `add_raster('pass_1', file, label='After pass 1')` or similar. For polygon masking, use `add_vector` only when an actual compatible GeoPackage/GeoJSON is present. Neither special case requires changing the viewer.

Concurrent hook workers must publish separate JSON files (e.g. one per tile) or serialize updates externally. This helper intentionally avoids a distributed publishing framework.

For vector layers, use `ProductCatalog.add_vector("boundary", polygon_path, label="Boundary")`.
The viewer shows a compact **Boundaries** menu (only in 2D), with a separate checkbox
and colored line key for each layer. Unconfigured layers remain visible by default,
so existing catalogs continue to work. For overlapping boundaries, explicitly set
which layer appears first:

```python
catalog.add_vector('original', original_water, label='Original water boundary',
                   display={'visible': False, 'color': '#2563eb', 'width': 1.8, 'dash': [7, 4]})
catalog.add_vector('exclusion', exclusion_water, label='Source-exclusion boundary',
                   display={'visible': True, 'color': '#dd7314', 'width': 2.2})
catalog.add_vector('topology', topology, label='Interpolation topology',
                   display={'visible': False, 'color': '#7c3aed', 'dash': [2, 4]})
```

An equivalent `products.json` vector entry can set `"display"` with optional
`"visible"` (boolean), `"color"` (`#RRGGBB`), `"width"` (0.5–6 screen pixels),
and `"dash"` (array of up to four positive on/off lengths; `[]` is solid).
These **display-only** settings never change the geometry or analysis.
The viewer remembers the user's selected layers per area. A new layer honors
its catalog default. Original vectors are read-only, and vectors need a valid
CRS plus the optional vector readers installed. This generic schema is not
specific to Coast Mask or any other producer.

## Georeferenced RGB underlay (v13.2)

Register RGB separately from elevation (with `style: "rgb"` and `group: "imagery"`):

```json
{
  "schema_version":1,
  "products":[
    {"key":"dem","path":"/data/dem.tif","style":"elevation","group":"elevation"},
    {"key":"s2_rgb","path":"/data/RGB_10m.tif","style":"rgb","group":"imagery"}
  ],
  "defaults":{"default":["dem","dem"]}
}
```

The RGB product needs georeferencing and >=3 bands (red, green, blue). Do not
provide numeric `limits` for RGB; these are elevation/scalar display scales,
not color balancing. The viewer auto-populates background selectors per area,
showing the RGB as a base layer under DEM opacity in 2D and optional terrain
colors in 3D. RGB is not gridded or incorporated into DEM science. Directory
mode recognizes clearly labeled RGB rasters, including `RGB_10m.tif`.

## Create a catalog without writing Python

```bash
raster-viz-catalog --out /path/to/run/products.json \
  --raster before=/path/to/before.tif --raster after=/path/to/after.tif \
  --rgb imagery=/path/to/RGB_10m.tif --compare before,after
```

Use `--label KEY=NAME`, `--stage KEY=NAME`, `--group KEY=GROUP`, `--area AREA`, or `--title TITLE` to customize. Or use `--scan DIRECTORY` for a bounded automatic catalog of *already finished* rasters. Explicit declarations are preferred because filename-based style classification is heuristic, not scientific provenance. See [examples/WORKFLOWS.md](examples/WORKFLOWS.md).

Catalog input paths are relative to the catalog file, not the current working directory. Paths supplied to `raster-viz-catalog` and `ProductCatalog.add_raster` are resolved from the process's current directory before being written as portable catalog-relative paths. A source being present does not guarantee it is complete: publish **after closing** each source.

## Elevation legend: source of units and vertical datum (v0.13.6)

The visible heading is limited to **Elevation (m) · NAVD88** when that information is available. The original Matplotlib palette and shared color-scale labels were removed from the UI **without removing shared numeric color limits**. A compound/vertical CRS such as EPSG:5498 can supply `NAVD88` and `m` automatically; a horizontal-only CRS cannot. TIFF band/dataset tags (`VERTICAL_DATUM`, `VERTICAL_REFERENCE`, `VDATUM`, `VERTICAL_UNITS`, `Z_UNITS`, `UNITTYPE`) and band unit strings are supported.

Producer-defined metadata overrides TIFF-derived labels, for example:

```python
from globato.viz.raster_viz import ProductCatalog
c = ProductCatalog('/path/to/run/products.json')
c.add_raster('dem', '/path/to/run/dem.tif', label='DEM',
             metadata={'value_units': 'm', 'vertical_datum': 'NAVD88'})
c.compare('dem', 'dem')
c.publish()
```

Only publish an explicit datum when the producing workflow has verified it. Differences carry units (e.g. **Elevation difference (m)**), not a height reference datum. Arbitrary scalar rasters do not inherit an elevation CRS's height units. Mixed vertical datums or units in one numeric color group are **not** automatically converted or reconciled. The `--vertical-unit-to-m` setting remains required for feet-based 3D/hillshading, since unit-label discovery does not perform a vertical conversion.

## Catalog refresh and viewer output reuse

A new catalog version is authoritative for its declared products. Removing a declaration, removing its source file, or replacing it with an unreadable source removes the older catalog-owned preview instead of leaving a misleading old result. Missing paths are reported as pending; unreadable paths are reported as failures. Refresh may update shared colors and 3D camera bounds for changed extents. This has no effect on original raster files.

In directory mode, vanished files are removed on the next scan of the *same root*, and only that root's discovered records are affected. Names including isolated `diff`, `difference` and `delta` tokens are heuristically displayed as difference maps, without automatic 3D. Directory classification is not a substitute for explicit science metadata.

Optional 3D for a difference surface requires explicit producer intent, for example:

```python
c.add_raster('change', '/path/to/change.tif', style='difference', group='difference',
             views=['2d', '3d'])
```

The normal difference style is 2D. This opt-in shows a visualization of the difference values as geometry; it does **not** turn a difference surface into physically absolute elevation.

## Hook integration: small, reproducible catalogs

The same schema v1 works for Coast Mask and Stack QC. The viewer does **not**
import the hook or infer its scientific role. Producers declare `area`, `key`,
`path`, `style`, `group`, and `defaults`, and publish after outputs are closed.
Keep `products.json` relative paths anchored at the **catalog's directory**.
Choose a shared color `group` only when products measure comparable values.
Prefer a four-product default overview over publishing every intermediate file.

**Coast Mask** emits the participant comparison (`before`, `after`, group
`participant_elevation`) and three geometry overlays. Its accepted all-source
stack and final DEM have a different scope and should use another color group.
The source-only group output must not be presented as a whole-tile before/after.

**Stack QC** emits four rasters (`before_measured`, `after_measured`,
`before_interpolated`, `after_interpolated`). Its `before_measured` and
`after_measured` share `qc_measured_elevation`; working-grid interpolation and
final Globato composite use independent groups. Per-pass details belong in the
legacy diagnostic HTML rather than overcrowding the viewer product selector.

The viewer infers elevation units and vertical reference from the raster when
available. **Do not publish guessed `vertical_datum` or `value_units` fields**:
a horizontal-only CRS is not enough to assert NAVD88 or meters.

Run the standalone preflight without starting the viewer:

```bash
raster-viz-check /path/to/products.json
# Publish-in-progress catalogs may have pending files:
raster-viz-check /path/to/products.json --allow-pending
# Structured report for CI:
raster-viz-check /path/to/products.json --json
```

`raster-viz-check` validates schema, declared files, GeoTIFF CRS and bands,
RGB band count, shapefile companions, and rejects raw FusionStates. It never
changes any source file. A green preflight means the files are consumable;
visual correctness and user-facing browser checks remain separate.

The minimal editable examples are
[`examples/hook_templates/coast_mask_products.json`](examples/hook_templates/coast_mask_products.json)
and [`examples/hook_templates/stack_qc_products.json`](examples/hook_templates/stack_qc_products.json).
The example paths are placeholders until you copy or generate completed files;
the production hooks generate their own catalogs and do not read these templates.

## Named vector layers and paused rendering

A vector entry may declare `"vector_layer": "retention_water"` to select a
GeoPackage layer explicitly. Existing native Coast Mask catalogs using
`"metadata": {"layer": "retention_water"}` are supported as GeoPackage hints.
An explicit `vector_layer` takes precedence. The older top-level `layer`
remains opaque producer metadata for both rasters and vectors; old source-overview
catalogs used it for semantic labels rather than actual dataset layer names.
Without a real selection, the reader uses its default layer. Use `band` for rasters.

`raster-viz-catalog --layer boundary=retention_water --hidden boundary` and
`add_vector(..., layer="retention_water", display={"visible": False})` publish the
same declaration. Preflight verifies actual vector readability, CRS and layer
existence; full geometry/vertex budgets are checked during rendering.

A running producer may set `"rendering_enabled": false`. The viewer then reads
only the catalog and updates status; it does not access scientific source files
for ranges, camera bounds or previews. `ProductCatalog.publish` accepts this flag.
The explicit checker still inspects files when asked to validate them.
