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
* `ATL03Reader` reads only the part of an ATL24 granule that overlaps the ATL03 file, rather than every photon of the beam. ATL24 granules are never subsetted, so with a spatially subsetted ATL03 file this was most of the cost of applying ATL24; reading such a granule now takes roughly 20-55% less time. An ATL24 file that is not stored in time order is still read in full.

### BUGFIX

* Fixed bug in binary_cudem that would wipe background data in the fine tier with a large `blend_dist`
* `ATL03Reader` now picks the newest cached ATL24 granule when several versions of one track are in the cache, matching how search results were already ranked. Before, whichever file the directory listing returned first was used.
* `ATL03Reader` no longer pairs an ATL03 granule with an ATL08 granule of a different release. ATL08 indexes photons by their position in one specific ATL03 release, so a mismatched pair can misclassify photons without any error. ATL24 may still come from another release: its join checks every photon against `delta_time`, which is the same in every release.
* `ATL03Reader` now finds the ATL03 photon behind each ATL24 seafloor photon instead of matching on `delta_time` alone. ATL24 stores `delta_time` with less precision than ATL03, so the two were bit-equal for only about 1 photon in 10 and the other 90% of ATL24's bathymetry was dropped without a message. `delta_time` is also shared by every photon of a transmit pulse, so each match labelled the whole pulse as seafloor and gave all of its photons the same position and height. Photons are now located through ATL24's `index_ph`, which works on spatially subsetted ATL03 files too; the transmit pulse is matched to within 5 µs, so a last-bit difference in `delta_time` between ATL03 releases does not lose a photon; and a beam whose photons do not line up is left unclassified with a warning.
* `ATL03Reader` no longer places ATL24 bathymetry at ATL24's own latitude, longitude and height as they stand. ATL24 takes its geolocation from the ATL03 release it was built from (006 for ATL24 V002), and against ATL03 release 007 that puts every photon of a granule 0.05 to 1.7 m away horizontally and up to 3 cm vertically, by an amount that differs from granule to granule. Bathymetry photons therefore sat that far from the rest of the photons they were read with. The offset is now measured along each beam, from the photons ATL24 does not refract, and removed, so a bathymetry photon keeps ATL24's refraction correction and is otherwise where its own ATL03 file has it. With too few such photons to measure from, ATL24's values are used as before.
