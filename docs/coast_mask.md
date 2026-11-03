# Coast Mask

Declare source participation with `coast_mask_source` before stream initialization. `group` names shared refinement; `evidence: true` contributes terrain; optional `remove` selects water, land, ocean, ocean_connected or inland_water for that source. Evidence without removal remains a normal DEM source.

Global hook order is `coast_mask_side_stack` → `coast_mask_refine` → ordinary `stack_combine` → optional `ms_binary_cudem`. The global refinement owns mask, polygon_class, grid and refinement settings. Multiple sources can cover different parts of one region. `ms_binary_cudem(coast_mask_group=...)` explicitly selects completed topology and validates actual tier coverage.

`coast_mask_side_stack` freezes source policy and delegates to native `side_stack(fresh=True)`. That reusable native mode publishes a completed-file receipt only after natural EOF. It does not use pre-bound side-stack cache hits. Ordinary side_stack/check retain their normal lifecycle, using a new cache version that excludes overwrite-era states.

Source retention and interpolation topology are separate products. Valid below-zero terrain is not automatically water. Ambiguous connectivity stays unresolved. The existing interpolator owns its numerical treatment; topology is not a new impermeable-domain solver.

Completed group publication records point to `refined_topology.gpkg`, refined water/land/coastline, independent `retention_water.gpkg` and a hash-checked manifest in the existing shared-cache preparation directory. Accepted-source receipts bind source branches to seven-band accepted states; they do not establish independent survey identity. `viz: true` emits a neutral products.json for a separate viewer.

Legacy source-attached hooks retain their adapters. Prepared-point capture requires its supported runtime. The new workflow uses native source grids. River guidance remains disabled; the optional zero-boundary estimate below requires explicit opt-in.

## Compact visualization contract

With `viz: true`, each completed grid-native group publishes a matched before/after participant composite and three portable vector overlays through schema-v1 `products.json`. `source_products.publish_grid_products` calls native `StackCombine` on copied entry dictionaries, using the same contributors and reduction settings on both sides. Source weights already present in FusionState are not applied again. Ordinary modules outside that group are excluded from this comparison and remain in the normal accepted collection. No source stream is replayed and production artifacts are not replaced.

The declarations use stable keys `before`/`after`, labels, `stage`, `area`, a shared elevation `group`, explicit `band: 1`, and `defaults`. Vectors distinguish starting water, source-exclusion water and independent interpolation topology. A downstream run publisher may add the normal accepted MultiStack and final delivery DEM; it should consume completed declarations rather than enumerate TIFFs. Detailed native states and per-source receipts remain authoritative. Raster Viz is optional and is not a production import.

## Optional zero-boundary estimate

Zero estimation is **off by default** (`estimate_zero: false`). After the complete accepted MultiStack exists, add these arguments to the normal `ms_binary_cudem` configuration:

```yaml
- name: ms_binary_cudem
  args:
    coast_mask_group: my_coast  # Match an existing completed refinement group.
    estimate_zero: true
    zero_max_distance_m: 40
```

The existing smoothed slope/profile estimator uses complete-stack land and bathymetric observations meeting the normal finest-tier weight cutoff. For example, compatible positive topography and negative bathymetry can support an estimated zero crossing in their gap. `zero_max_distance_m` is the maximum permitted seaward adjustment in meters (default 40), not a fixed buffer or a promise that the boundary reaches that crossing. Unsupported sections retain the refined boundary. Alongshore smoothing and topology safeguards constrain the result; those safeguards can substantially reduce or reject a proposed adjustment.

The estimator publishes a coastline line and polygon topology, and uses the polygon as the interpolation barrier. Source-exclusion masks and accepted measured data are unchanged. The estimate uses the native vertical datum and adjusts explicit ocean segments only. It is experimental: it does not establish a tidal shoreline, independent contour accuracy, or a fix for interpolation oscillation.

The returned v19 Malibu, Crescent South and Pelican Bay provider runs completed with source preservation and six-tier barrier consumption verified. Pelican Bay supplied the clearest useful visual example, consistent with the user's visual review. Malibu and Crescent South retained smaller, patchier adjustments after topology safeguards. These results support optional regional use and review; they do not establish uniform boundary smoothness or general accuracy. Inspect the estimated boundary against the unfilled accepted observations before adopting it for a region.

For vector-only publication, a `coast_mask_boundary` collection hook after `stack_combine` accepts `group`, `max_distance_m` (default 40) and `min_weight` (default 2). Leave the interpolator flag off to review the estimate with an unchanged baseline DEM. Both entry points call the same component. Files and the support checksum are recorded under `estimated_zero` in the existing group index. With Viz enabled, one extra green dashed boundary is published and initially hidden. The coastline line is for display; interpolation consumes `interpolation_topology.gpkg`.

## Polygonal retention at source scope boundaries

`MaskArtifact.keep_geometry` clips shared water geography to each source scope and returns polygonal area. Valid constructive intersections can also include boundary-touch lines or points; those zero-area parts are removed before native masking. Polygon components of every size are retained. Invalid geometry fails rather than being buffered or silently repaired. Native raster and seven-band state writers retain strict-interior cell-centre selection, including exclusion of exact boundary ties.

## Native source aliases

After native source grids complete and before coastline evidence or source masks are assembled, `CoastMaskRefine.run` calls `globato.hooks.sinks.side_stack.unique_source_contributions`. This extends the existing native source-completion contract. It uses the unchanged Fetchez processing identity, including class, weight and partition selectors. Two entries with that identity are selected once only if their completed native receipts, grid metadata and all seven statistic bands agree exactly. Conflicting states fail before reduction.

Alternate source URLs and their completed-state checksums remain on the selected entry and in `accepted_collection.json` under `source_aliases`. Ordinary entries without the opt-in identity keep existing behavior. Distinct processing branches are not merged. This is collection-time source selection; it adds no identity algorithm, registry, alternate reducer or cache. Native gridding and interpolation arithmetic are unchanged.

## Raster Viz integration (visualization only)

`viz: true` creates a schema-v1 `products.json`; the standalone viewer is
optional and is not imported by production Coast Mask processing. Native
source groups publish `before`/`after` measurements and three independently
selectable vectors; only the source-exclusion boundary is initially visible.
The alternative source-overview path also writes the same neutral schema; it
no longer invokes `globato.products.publisher` or an embedded viewer launcher.
When a global product catalog is explicitly updated, global DEM products
remain in their own comparison area, distinct from source-refinement evidence.

The viewer and catalog checker are included in Globato, following the merged Raster Viz PR #299. Check and render a completed catalog with:

```bash
raster-viz-check /path/to/products.json
raster-viz /path/to/products.json --3d --mesh-side 500 \
  --out /path/to/viz
```

No water level is guessed from the geographic location; elevation symbology
uses the viewer default `water_level=0` unless a user/workflow explicitly
sets a threshold in the catalog or viewer CLI. The chosen water level must
use the same vertical units and reference as the displayed TIFF.
