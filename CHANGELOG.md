# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### ADDED
* Add a Changelog
* `ATL03Reader` takes an `atl_version` option (e.g. `"007"`) and skips any ATL03 granule of a different release.

### CHANGED

* Refactored `PointPixels` around an explicit, associative `FusionState` contract for point-to-pixel accumulation.
* Reworked `multi_stack` to consume and persist FusionState directly, separating source stacking strategy from point aggregation and enabling resumable/incremental accumulation.
* Added explicit FusionState finalization for weighted elevation, coordinates, uncertainty, and elevation dispersion.
* Replaced legacy `PointPixels` aggregation modes with focused `accumulate()`, `count()`, and `coverage()` operations.
* Updated provenance/source-mask generation to use the shared point-to-pixel coverage API.
* Added invariant tests for FusionState associativity, finalization, persistence/resume behavior, and MultiStack stacking strategies.
* Updated DEM-generation documentation and added detailed documentation for the PointPixels/FusionState architecture.
* Reverted the global-bato.yaml bundle's multibeam reference to use the rq hook's 'percent' mode instead of 'iho_2' which was removing too much data in deep water.

### BUGFIX

* Fixed bug in binary_cudem that would wipe background data in the fine tier with a large `blend_dist`
* `ATL03Reader` now picks the newest cached ATL24 granule when several versions of one track are in the cache, matching how search results were already ranked. Before, whichever file the directory listing returned first was used.
* `ATL03Reader` no longer pairs an ATL03 granule with an ATL08 granule of a different release. ATL08 indexes photons by their position in one specific ATL03 release, so a mismatched pair can misclassify photons without any error. ATL24 joins on `delta_time` and may still come from another release.
