# Stack QC execution path

Only Globato owns collection preparation. Stock Fetchez discovers/downloads,
reads/transforms streams, tears down native consumers, and runs final sinks.

| Owner | Responsibility |
| --- | --- |
| `cli/__init__.py` → `cli/run.py:run_recipe` | Registers the actual `globato run` command; reuses stock CLI options; invokes `StackQCRecipe`; failures exit nonzero. The former 300-line command was unused. |
| `qc/stack/pipeline.py:StackQCRecipe.compile` | Ordinary recipes return the exact upstream plan. QC attachments select the Globato-only plan. Global QC attachments are rejected. |
| `StackQCPlan._select_sources` | Discover once; module manifests; global manifests once. Defer only native side-stack lookup on QC sources; reject unknown deferred selectors before discovery. Preserve selected ordering into native submission. Fetch completion remains concurrent, as upstream. |
| `preparation.py:source_pipeline`, `fetch_sources` | Copy source hooks; resolve the actual stack grid; check source caches; fetch/process misses once through stock Fetchez. Final consumers cannot be local to a QC module. |
| `collect_sources`, `context_key`, `run_qc` | Validate original seven-band schemas, identity, checksums/grid/claims and scientific owner code; sort scientific inputs stably; lock one context; build native context DEMs. |
| `processing.py:process` | Rejection passes and convergence; restoration once after filtering. Unchanged science; no provider/region logic. |
| `source_windows.py`, `sources.py`, `surveys.py`, `restoration.py`, `recovery.py` | Scientific evidence, masks, survey protection, restoration, bounded source-window reads. Frozen expressions unchanged. |
| `native.py:NativeBuilder`, `stacking.py:rebuild` | Spawn isolated native interpolation; reuse unchanged blocks/contexts; verify source/mask receipts; apply original seven-band contributions without subtractive approximation. |
| `resampling.py:context_hooks`, `accepted_stream` | Validate units and integer nesting; derive QC bounds from the resolved final grid; scale private context ladder; aggregate accepted seven-band source statistics into complete final cells with bounded reads. Default path skips aggregation. |
| `preparation.py:accepted_sources` | Apply final 0/1 masks lazily; retain raw URL/CRS/weights/group metadata. Carry processed-source IDs so equal local basenames cannot collide in final stack/provenance. |
| `StackQCPlan._execute_prepared` | Gate original file/transforms on accepted streams; ordinary modules keep native hooks; global consumers run once. Final additive consumers rebuild; raw/source/QC caches remain reusable. Restore original hook lists/settings on exit. |
| `review/publisher.py:ReviewPublisher.run` | Runs after the actual final DEM. Writes optional GIS/figures/HTML; reports display failures in `review_ERROR.txt`; never changes masks or DEM. |

The five established YAML controls are preserved; `qc_res` is the sixth optional argument. Scientific modules do not import
Fetchez. `pipeline.py` and `preparation.py` are application adapters, outside the
array-level filtering owners. Source consumers currently supported are native
`multi_stack`, `provenance`, `source_masks`, and `stack_provenance`.

Do not run simultaneous jobs into one final-output folder. Do not feed prepared
FusionState streams through original point transforms again. Keep examples,
comparisons, validation scripts, field results, and packaging outside `src/`.

The patch's generic changes have narrow owners: `multi_stack` exposes its existing
reduction/accumulator for context reuse; `side_stack` stores additive chunks and
validates actual grid/processing identity; claim filtering supplies footprint
identity; raster checkpoints preserve live streams/metadata; native interpolation
uses the decimator's actual output; provenance accepts processed-source identities.
No Fetchez source, lifecycle, or command globals are modified.


`resampling.nested_region` uses resolved final bounds for source preparation and
QC context, on a shallow module copy. Final consumers retain the original module.
`utils.parse_increment` preserves numeric increments ending in zero across point
binning, side-stack geometry and native interpolation; suffixed increments still
use stock Fetchez. This fixes a reproduced 30→3 parsing defect without changing
Fetchez or adding an alternative lifecycle. No scientific thresholds change.


The optional `provenance` and `source_masks` consumers read accepted coverage
from the FusionState count band and pass the original chunks through unchanged.
They require the final stack's grid and compare horizontal CRS components, so a
compound elevation CRS remains valid. `stack_provenance` consumes the original
additive bands after `multi_stack`. Keep these metadata consumers outside source
preparation; they must see the accepted stream only once.
