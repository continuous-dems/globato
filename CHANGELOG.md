# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### ADDED
* Added draft implementation of the new `fusion-provenance` hook for use in the new side-stack workflows.
* Added `globatize_sources` recipe modifier to set the res and dst_srs for modules in a recipe.
* Added the `globatize_sources` modifier as a default in `globato build`
* Added `sidestack` cli managment tool to the globato cli to inspect and clear existing sidestack files.
* A `py.typed` marker, so type checkers read globato's type annotations in code that uses it (#284). mypy now runs on every pull request, in a new "Typecheck" workflow.
* Added `stack_combine` to construct a MultiStack and its FusionState directly from completed side-stack artifacts, using the existing `MultiStackAccumulator` for reduction and finalization.
* Allow `stack_combine` to consume original or QC-adjusted side-stacks, with configurable handling of missing required artifacts.
* `source_masks`, `provenance` and `stack_provenance` take a `source_artifact` argument. With it they run at the collection stage and read each entry's FusionState side-stack artifact (original or QC-adjusted) instead of its stream, so they work in `side_stack` + `stack_combine` recipes and reflect collection-stage filtering.
* `stack_qc` hook: Stack QC source filtering and terrain restoration (from #285) as a collection-stage hook for `side_stack` + `stack_combine` recipes. It filters the named modules' side-stacks against working DEMs built in the same process and records the accepted measurements, on the DEM grid, as `qc-side-stack` (`output_artifact`); other modules' side-stacks pass through, so `stack_combine` reads `qc-side-stack`. Detection thresholds, window sizes, plane tolerance and restoration scales, and the working-DEM settings, are hook arguments. Results are cached on the filter's own inputs only.
* `stack_qc_grid` hook: with a finer `qc_res`, grids the filtered modules' points on Stack QC's grid during the stream stage (cached like side-stacks, recorded as `stack-qc-grid`) for `stack_qc`. Listed before `side_stack`, which then leaves those entries alone.
* `stack_qc_review` hook: optional HTML review (before/after hillshades) of Stack QC results, written after the final DEM.

### CHANGED
* Simplify cache identities using Fetchez's `dataset_id()` and contribution metadata instead of a separate Globato identity implementation. Retain grid, CRS, FusionState schema, and cache hash validation.
* Updated globato build overrides to propagate resolution, CRS, and stacking configuration to `stack_combine`.
* Updated `globato build` to use `ignore-failures` instead of `fail-fast`
* `mr-globato-crm` and `mr-globato-cudem` are complete hook lists that stack with `side_stack` + `stack_combine`, filter multibeam with `stack_qc` (CRM: `stack_qc_grid` at 1/9") and build provenance from the stacked side-stacks. As overrides of `mr-globato` their `side_stack` and `stack_combine` entries never ran, since preset overrides only change hooks the base preset has.
* `multi_stack` dataset ids include an entry's `contribution_id` when it has one; ids of other entries are unchanged.
* Requires fetchez 0.12.1.
* `crm-bathy-topo` no longer runs `rq` on NCEI multibeam (`mbdb`); the CRM preset filters it with `stack_qc` (1/9" QC for the 1" DEM) and the CUDEM preset on its own 1/9" grid. Other recipes using `global-bathy-topo` keep `rq`.
* `mr-globato-crm` tier weights are 0.6/0.4/0.2 (were 1/0.5/0.25), so multibeam (stack weights ~0.8-0.95) is gridded at 1" instead of 3". On the 330 Southern California CRM tiles no other dataset changed tier (module weights are >= 1 or <= 0.5).

### FIXED
* `side_stack` added point chunks that revisit cells by overwriting them, losing earlier soundings; it now adds them (side-stack cache version 2, so older caches are rebuilt). It also checks the input CRS and grid and publishes the cache file under a lock.
* `Point2PixelStream`'s "Pixelated" log line had three placeholders and two arguments, so every point stream logged a `--- Logging error ---` traceback in place of its point count, and tests that capture INFO logs failed with `not enough arguments for format string`.
* The `glob_sdb` module failed with `ModuleNotFoundError: No module named 'globato.cli.recipe'` as soon as it started building its training DEM (#284). It now uses `fetchez.recipe.Recipe`.
* `GDALReader.get_read_window()` failed with `ModuleNotFoundError: No module named 'globato.streams.spatial'` whenever a region was set (#284). It now uses `fetchez.spatial.Region`, and takes the region as a `Region` or as `[w, e, s, n]` in lon/lat.

## [0.5.0 - 10-07-2026]

### ADDED
* `globato.read()` takes `ignore_failures` (#274). With fetchez's stream fix (fetchez #472), an error a reader raises while the stream is read, such as `AuxiliaryDataError`, is raised to the code iterating the stream; before, it was lost and the stream just ended early. `ignore_failures=True` keeps the old behavior: the error is logged and the stream ends early. Before this, `read()` had no way to set it.

### FIXED
* `ClaimGridFilter` and `WaterSurfaceFilter` work with numpy 2.0.0, which returns `np.unique(..., axis=0, return_inverse=True)`'s inverse with an extra dimension. `ClaimGridFilter` failed with "non-broadcastable output operand" and `WaterSurfaceFilter` with "object too deep for desired array". Other numpy versions are unaffected.

### CHANGED
* Updated `api.read` to use `fetchez.recipe.RecipeContext`.
* Updated `streams.base` to coincide with the recent updates to `fetchez.streams.base`, which is sub-classes.
* Use `fetchez.logging.configure_logging` in cli instead of old `fetchez.cli.setup_logging`
* Update fetchez.cli imports.
* A failure to fetch or apply an ICESat-2 ATL08 or ATL24 granule raises the new `AuxiliaryDataError` instead of reading the ATL03 granule without it (#273). Before, `ATL03Reader.fetch_atlxx()` logged a failed lookup or download at debug level and returned `None`, as for a granule that has no ATL08 or ATL24, and ignored the download status fetchez returned, handing back the path of a file that was missing or incomplete; an incomplete file left in the cache was then reused by every later read. Applying an ATL08 or ATL24 file that could not be read, or whose ATL24 photons did not line up with the ATL03 granule, logged a warning and went on. Either way the photons came out without ground and canopy (ATL08) or seafloor (ATL24) classes, which looks the same as a granule that has none. A granule for which no ATL08 or ATL24 exists at all is still read without it, as before. An unreadable copy in the cache is deleted and fetched again, with a warning. A failed lookup at NASA is only detected with a fetchez that raises on a failed CMR search (continuous-dems/fetchez#470); with older fetchez it still looks like a missing granule. `ATL03Reader(strict_aux=False)` turns the error off: each of these failures is logged as a warning and the granule is read without that product.
* Every dependency now has a minimum version that installs on Python 3.12, globato's oldest supported Python: the first release with Python 3.12 wheels for compiled packages, or a release from around Python 3.12's own (fall 2023) for pure-Python ones. `numpy>1.24` becomes `numpy>=1.26.0` and `shapely>=2.0.0` becomes `shapely>=2.0.3`, since the older releases have no Python 3.12 wheels. `shapely`, `pandas` (`>=2.1.2`), `h5py` (`>=3.11.0`) and `pyogrio` (`>=0.8.0`, up from `>=0.7.0`) are also a little past their first Python 3.12 wheels, since the earlier releases install alongside numpy 2 but fail to import. CI now also runs the tests with each dependency at its minimum, and on Python 3.14.

## [0.4.0 - 10-07-2026]

### ADDED
* `XYZReader` takes `keep_pos` and `keep_values` to keep only rows whose value in one column is in a given set.
* Add new modifier 'ensure-spatial-claim' to insert the globato 'spatial-claim' hook into global hooks when 'claim-grid-filter' is present.

### FIXED
* The `nos-xyz` profile drops soundings NOS marks inactive (`active` == 0); pass `keep_pos=None` to read them all.
* Add `is_globato_source` to utils.

### CHANGED
* Update cli to use modern fetchez imports.
* The base raster hook now hases it's _dst_fn to avoid collision in certain instances.
* Remove 'gui' from cli as it's still testing

## [0.3.9 - 10-02-2026]

### ADDED
* Add a Changelog
* `ATL03Reader` takes an `atl_version` option (e.g. `"007"`) and skips any ATL03 granule of a different release.
* Add `stack_provenance` metadata hook to resolve the participating sources in a resulting `source_masks` output.
* Added resumable per-source stack provenance state with bounded-memory disk storage and an optional in-memory backend.
* Added grouped provenance vector output and QGIS styling, including configurable metadata-based grouping such as module and source weight.
* Added regression coverage for stack/source provenance alignment, boundary-pixel behavior, resume state, storage-backend parity, and deterministic derived outputs.
* Added the ms_quality raster hook(s); `ms_consistency_weight` and `ms_support_weight`

### CHANGED

* Refactored `PointPixels` around an explicit, associative `FusionState` contract for point-to-pixel accumulation.
* Reworked `multi_stack` to consume and persist FusionState directly, separating source stacking strategy from point aggregation and enabling resumable/incremental accumulation.
* Added explicit FusionState finalization for weighted elevation, coordinates, uncertainty, and elevation dispersion.
* Replaced legacy `PointPixels` aggregation modes with focused `accumulate()`, `count()`, and `coverage()` operations.
* Updated provenance/source-mask generation to use the shared point-to-pixel coverage API.
* Added invariant tests for FusionState associativity, finalization, persistence/resume behavior, and MultiStack stacking strategies.
* Updated DEM-generation documentation and added detailed documentation for the PointPixels/FusionState architecture.
* Reverted the global-bato.yaml bundle's multibeam reference to use the rq hook's 'percent' mode instead of 'iho_2' which was removing too much data in deep water.
* `ATL03Reader` reads only the part of an ATL24 granule that overlaps the ATL03 file, rather than every photon of the beam. ATL24 granules are never subsetted, so with a spatially subsetted ATL03 file this was most of the cost of applying ATL24; reading such a granule now takes roughly 20-55% less time. An ATL24 file that is not stored in time order is still read in full, and an ATL03 file whose time range falls entirely outside an ATL24 beam (ATL24 keeps only photons over coastal water, so a beam can end well before the subset) reads nothing from that beam and leaves its bathymetry unclassified.
* `ATL03Reader` builds the Bing-building and OSM-landmask trees once per region per process and reuses them for every granule read over that region, instead of re-parsing every footprint file on each read; prebuilt trees can also be passed in (`bldg_tree=`, `land_tree=`). Over a city, the rebuild took minutes per granule.
* `ATL03Reader` tests photons against the landmask with prepared polygons. A coastline polygon of tens of thousands of vertices was walked in full for every photon, which took minutes per granule; it now takes well under a second, with the same result.
* `ATL03Reader` spreads per-segment values (geoid, tides, DEM height, photon ordinal) over photons with array operations rather than a Python lookup per photon, and looks ATL08 segments up by bisection.
* `ATL03Reader` reads only the stretch of an ATL08 granule whose segments the ATL03 file spans, found by bisection as for ATL24, instead of every classed photon of the beam.
* `ATL03Reader` tests a photon against the landmask and building masks only when the test could change what the reader yields: photons already in a water class are not tested against the landmask, photons in a water class are not tested against the building mask, and when reading with a class filter that wants none of the classes a mask (or a step after it) can assign, photons of classes the filter drops are not tested either (noise photons always are, since the steps after the landmask read a photon's signal flag from its class). Output-identical; on a granule over a city this was about a third of the time to read it.
* `ATL03Reader`'s building classifier takes its running ground level, range and thickness from one sort of each window instead of a pandas rolling pass per statistic, and groups its candidates with array operations; its outlier, nearshore, inland-water and reflectance steps work on the columns they use instead of copies of the whole frame. Output-identical.
* Provenance and source-mask generation now use the exact authoritative raster transform for point-to-pixel binning, avoiding downstream reconstruction of grid geometry from bounds and resolution.
* Stack provenance now derives final source masks from completed FusionState tier information, separating numerical stacking decisions from provenance bookkeeping.
* Provenance VRT bands, grouped vector features, and aggregated metadata are emitted deterministically regardless of source processing order.
* Final provenance products now expose consistent semantic metadata across disk and memory state backends while keeping temporary storage details internal.

* `RasterioReader` reads rasters stored in thin full-width strips (ASCII grids, untiled GeoTIFFs) in bands of whole strips holding about 500k cells (`STRIP_CHUNK_CELLS`), instead of one chunk per strip; a `usgs_ds702` ASCII grid came through as 1,400-3,000 chunks. Tiled files, strips that already hold that many cells, and an explicit `chunk_size` are unchanged. Same points, in fewer chunks.

### BUGFIX

* `ATL03Reader` used only the last Bing tile's footprints for the building mask when a region spanned several tiles; footprints from every tile are now used.
* `ATL03Reader` flagged the wrong photons as inside a building or on land. The hits of an `STRtree` query pair a photon index with a polygon index, and both were read as photon indices, so for every photon that hit a polygon, the photon whose row number matched the polygon's index was flagged too.
* Fixed bug in binary_cudem that would wipe background data in the fine tier with a large `blend_dist`
* `ATL03Reader` now picks the newest cached ATL24 granule when several versions of one track are in the cache, matching how search results were already ranked. Before, whichever file the directory listing returned first was used.
* `ATL03Reader` no longer pairs an ATL03 granule with an ATL08 granule of a different release. ATL08 indexes photons by their position in one specific ATL03 release, so a mismatched pair can misclassify photons without any error. ATL24 may still come from another release: its join checks every photon against `delta_time`, which is the same in every release.
* `ATL03Reader` now finds the ATL03 photon behind each ATL24 seafloor photon instead of matching on `delta_time` alone. ATL24 stores `delta_time` with less precision than ATL03, so the two were bit-equal for only about 1 photon in 10 and the other 90% of ATL24's bathymetry was dropped without a message. `delta_time` is also shared by every photon of a transmit pulse, so each match labelled the whole pulse as seafloor and gave all of its photons the same position and height. Photons are now located through ATL24's `index_ph`, which works on spatially subsetted ATL03 files too; the transmit pulse is matched to within 5 µs, so a last-bit difference in `delta_time` between ATL03 releases does not lose a photon; and a beam whose photons do not line up is left unclassified with a warning.
* `ATL03Reader` no longer places ATL24 bathymetry at ATL24's own latitude, longitude and height as they stand. ATL24 takes its geolocation from the ATL03 release it was built from (006 for ATL24 V002), and against ATL03 release 007 that puts every photon of a granule 0.05 to 1.7 m away horizontally and up to 3 cm vertically, by an amount that differs from granule to granule. Bathymetry photons therefore sat that far from the rest of the photons they were read with. The offset is now measured from the photons ATL24 does not refract, over a window of track centred on each bathymetry photon and widened until it holds enough of them, and removed, so a bathymetry photon keeps ATL24's refraction correction and is otherwise where its own ATL03 file has it. Only a beam with fewer than ten such photons in the whole file keeps ATL24's values as before.
* `MultiStack` no longer iterates or even knows about the `source_masks` hook or output.
* Fixed stack-provenance pixel misalignment at raster cell boundaries that could leave valid MultiStack cells uncredited or assign provenance to an adjacent cell.
* Fixed resumed stack provenance omitting sources whose persistent provenance state was created in an earlier run.
* Fixed temporary provenance-state/backend metadata leaking into finalized source-mask products.
* Fixed source masks retaining contributions that had been superseded by higher-priority data in mixed stacking.
