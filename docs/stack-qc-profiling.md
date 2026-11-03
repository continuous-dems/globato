# Stack QC phase receipts

Stack QC publishes `profile_<review-tag>.json` alongside its existing review receipt.
The review receipt's `profile` field and the accepted entries' `stack_qc_profile`
artifact identify it. No new YAML option or viewer dependency is required.

The `stack_qc_profile/1` schema records the chosen settings, working interpolation
identity, QC region, output grid, source SHA-256s and fingerprints, filter cache
identity/hit, output manifest/signature and accepted side-stack paths.

`phases_seconds` partitions `total_seconds`: inventory/identity, filtering or cache
validation, and accepted publication (including review metadata). These use
`perf_counter`. They exclude the profile JSON write itself. Do not add nested
engine or working-surface diagnostics to this total.

`engine.timing_seconds` preserves the existing engine's timings, without changing
its boundaries. `engine.origin` distinguishes `current_execution` from
`cached_execution`. In particular, Geometry pass timing includes its next context
build; Connectivity's pass 1 timer excludes the following context build. Their
individual pass times are not directly comparable CPU costs. Engine timing begins
after the orchestration's initial baseline construction; use working-surface events
to see that construction as well.

`working_surfaces` describes only calls made during this invocation, including
baseline hits, validated working-surface hits, unchanged-stack reuse, stacking
counters/times, finalization and interpolation. A cached filter result has no new
working-surface events; it must not inherit the original build costs as current work.

With `stack_qc_grid`, each input also has a lazy producer timing and cache outcome.
It times `next()` calls, excluding consumer pauses. On misses this combines
upstream source reading/reprojection and grid materialization: these are not
separable at this boundary. It excludes provider discovery before the stream and
cache lookup setup. Never interpret it as complete discovery or network timing.

For QC recipes, `stack_combine` and the final `ms_binary_cudem` publish adjacent
`.profile.json` sidecars and corresponding entry artifacts. Working interpolation
calls remain in the working-surface diagnostics. These sidecars contain references
to upstream receipts, not another copy of the same inclusive costs.

Reporting does not modify engine results, masks or raster tags. The existing
conservative code fingerprint includes reporting code, so installing this patch
invalidates old filter-result cache identities once. It does not weaken any cache
validation or remove method/settings from decision-cache keys.

Raster Viz is an optional external consumer. These receipts neither launch it nor
introduce regional or provider-specific behavior into production hooks.
