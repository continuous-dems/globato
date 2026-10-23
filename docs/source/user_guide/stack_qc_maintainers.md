# Stack QC execution path

`stack_qc` is an ordinary fetchez collection hook. There is no Stack QC
specific recipe class, plan or command: `globato run`, `fetchez run` and the
Python recipe API all run it the same way.

| Stage | Owner | What happens |
| --- | --- | --- |
| manifest, stream | stock fetchez + `side_stack_check` / `side_stack` | Sources are read and transformed by their module hooks and written as DEM-grid side-stacks, as for every module. |
| collection (module) | `hooks/filters/stack_qc.py:StackQC.run` | Groups the module's entries (or, as a global hook, the named modules') and calls `prepare`. |
| | `qc/stack/preparation.py:grid_sources` | With a finer `qc_res`: re-grids each source on the QC grid with the module's own file/stream hooks (stock `run_fetchez`, cached as ordinary side-stacks). The source CRS from before the stream stage is restored first (`stream_reproject` rewrites `src_srs`). Otherwise the module's own side-stacks are the QC grids. |
| | `collect_sources`, `cache_key` | Inventory and checksums; the key holds only the filter's inputs. |
| | `run_filter` → `processing.process` | Cached result, or the passes and restoration. |
| | `qc/stack/context.py:WorkingSurface` | Working DEMs, in process: `stacking.rebuild` (only changed blocks) into a MultiStack accumulator, then `ms_binary_cudem` in a scratch folder, with the module's shared cache for barriers and dist2coast. |
| | `source_windows.py`, `geometry.py`, `plane.py`, `sources.py`, `surveys.py`, `recovery.py`, `restoration.py` | The science. Tuning values come from `Settings`; defaults are the original constants. The geometry step runs its blocks in `workers` processes (results do not depend on the count). |
| | `write_accepted` | Accepted statistics aggregated per source to the DEM grid (`resampling.accepted_stream`) and written as a side-stack; the entry's `side-stack` artifact points to it. |
| collection (global) | `stack_combine`, `ms_binary_cudem`, ... | Unchanged. |
| collection (global, optional) | `hooks/filters/stack_qc_review.py` | Review pages after the final DEM, from the cached results. |

Filtered entries need no special identity in `multi_stack` or provenance:
`stack_combine` registers each accepted side-stack by its own hash tag, which
combines the source side-stack's hash with the QC result signature.
