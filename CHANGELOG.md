# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### ADDED
* Add a Changelog

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
