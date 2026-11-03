# Stack QC — source quality control before stacking

Stack QC evaluates each source's gridded elevation contributions, removes
inconsistent cells, and restores original contributions supported by the
surrounding terrain and survey evidence. Source files are unchanged;
decisions are recorded as 0/1 masks on each source's grid.

`stack_qc` is a collection-stage hook for `side_stack` + `stack_combine`
recipes. List it in `global_hooks` with `modules:`, after `side_stack` and
before `stack_combine`. It filters the named modules' sources and records each
source's accepted measurements, on the DEM grid, as the entry's
`qc-side-stack` artifact. Entries of other modules get their own side-stack
there unchanged, so `stack_combine` reads `qc-side-stack` for everything. The
original side-stacks are left untouched.

To filter on a finer grid than the DEM (`qc_res`), add `stack_qc_grid` with
the same `modules`, `res`, `qc_res` and `crs`, before `side_stack`. During the
stream stage it grids those modules' points on the QC grid and caches that
like a side-stack. The entries then carry the QC-grid FusionState instead of
points, so `side_stack` (which only side-stacks point streams) leaves them
alone. This is the CRM layout (`mr-globato-crm`):

```yaml
global_hooks:
  - name: side_stack_check
    args: {res: 1s, crs: "EPSG:4326+3855"}
  - name: stack_qc_grid
    args: {modules: mbdb, res: 1s, qc_res: 1/9s, crs: "EPSG:4326+3855"}
  - name: side_stack
    args: {res: 1s, crs: "EPSG:4326+3855"}
  - name: stack_qc
    args:
      modules: mbdb
      res: 1s
      crs: "EPSG:4326+3855"
      qc_res: 1/9s
      max_passes: 5
      stop_when_stable: false
      interpolation:
        steps: 3
        resolutions: 1s/3s/9s/15s
        weights: [0.6, 0.4, 0.2]
        algos: raster_fill/raster_fill/raster_fill/interp_gmt:tension=1
        blend_dists: 32/16/8/4
        barrier: osm
  - name: stack_combine
    args: {res: 1s, crs: "EPSG:4326+3855", source_artifact: qc-side-stack}
```

For an explicit single-module recipe without `qc_res`, `stack_qc` can filter
DEM-grid side-stacks directly without `stack_qc_grid` (`mr-globato-cudem`).
Joint contexts require `stack_qc_grid` even when the two resolutions are equal. `stack_qc`
refuses a recipe that stacked with `multi_stack`.

The optional `stack_qc_review` hook writes an HTML review with before/after
hillshades. Put it in `global_hooks` after the final interpolation (see
[Review pages](#review-pages)).

## Shared context across modules

Use one global collection hook for all targets. The targets automatically supply
working-surface and survey evidence; `add_ref_modules` supplies additional fixed,
unfiltered evidence. For example, use these matching arguments on both grid and
QC hooks in an otherwise normal recipe:

```yaml
modules: [mbdb, csb]
add_ref_modules: [dav, nos_hydro]
res: 1s
qc_res: 1/9s
crs: "EPSG:4269+5703"
```

All target/reference modules must already be declared in the recipe with their
normal source transforms and weights. List `stack_qc_grid` before `side_stack`,
then `stack_qc` before `stack_combine` in `global_hooks`. Do not place
`side_stack_check` before the joint grid producer; it can replace current points
with an old side-stack. Modules outside these selections pass through final
stacking but do not influence QC. Overlapping target/reference roles are errors.

Each pass evaluates every target against the same parent state, then rebuilds
changed blocks for the next pass. References never receive rejection masks.
All targets share the invocation's method and pass policy. Geometry5 and
Connectivity2 detection/restoration rules remain unchanged; adding evidence can
change their decisions and requires separate validation from the frozen
single-module case.

Selectors also accept ordinary recipe-style mappings to distinguish provider
instances: `{module: dav, args: {survey_id: reference_2024}}`. Selector arguments
must match the actual invocation's attributes/arguments. Source weights enter
native FusionStates and within-survey pooling; independent survey votes retain
the original semantics. Module/source/invocation IDs, groups, roles, weights,
source hashes, outputs and target masks are recorded in a contribution JSON
artifact (`stack_qc_contributions`). Use the normal optional `provenance` hook
for pixel-level provenance products. References are retained in final assembly;
fine reference statistics are summed onto the DEM grid without filtering.

## Settings

Grid and working DEM:

| Setting | Default | Purpose |
| --- | --- | --- |
| `res` | `1s` | DEM grid of the accepted measurements; equal to the stack's `res` |
| `crs` | `EPSG:4326+3855` | Stack CRS; equal to the stack's `crs` |
| `modules` | all | Target names or recipe-style selectors; all targets also supply evidence |
| `add_ref_modules` | none | Additional unfiltered reference names/selectors, disjoint from targets |
| `qc_res` | none | Optional finer QC grid, `res / N` (e.g. `1/9s` for a 1s DEM) |
| `stack_strategy` | `mixed` | Working-DEM MultiStack strategy |
| `stack_weight_threshold` | `3/2/1/.75/.5/.4/.3/.2/.1` | Working-DEM MultiStack weight tiers |
| `interpolation` | CRM 1" ladder | `ms_binary_cudem` arguments for the working DEMs |
| `source_artifact` | `side-stack` | Each entry's DEM-grid side-stack (filtered without `qc_res`; passed through for other modules) |
| `grid_artifact` | `stack-qc-grid` | With `qc_res`: each entry's QC-grid FusionState from `stack_qc_grid` |
| `output_artifact` | `qc-side-stack` | Where the accepted DEM-grid side-stacks are recorded |

`stack_qc_grid` takes `modules`, `add_ref_modules`, `res`, `qc_res` and `crs` (as `stack_qc`),
plus `artifact` (default `stack-qc-grid`) and `cache_dir`.

The working DEMs are the surfaces each pass compares source cells with. They
are built from all selected targets and fixed references, in process, with
an ordinary MultiStack and `ms_binary_cudem`. Write `interpolation` as you
would for the final DEM grid; with `qc_res`, its resolution ladder is scaled
down so the first tier is the QC grid. Keep it in step with the final
`ms_binary_cudem` if the filter should see the same surface.

Filtering:

| Setting | Values | Default | Purpose |
| --- | --- | --- | --- |
| `aggressiveness` | `1`, `2`, `3` | `2` | Strength of anomaly filtering |
| `terrain_restoration` | `0`–`3` | `2` | Restore supported original terrain, or off at `0` |
| `max_passes` | `1`–`10` | `4` | Maximum rejection passes |
| `stop_when_stable` | boolean | `true` | Stop when passes stop changing |
| `consensus` | boolean | `false` | Also reject cells that disagree with other surveys |
| `survey_exclusion` | `off`, `near-complete`, `redundant` | `off` | Optional whole-survey exclusion |
| `stable_changed_fraction`, `stable_dem_change_m`, `stable_consecutive_passes` | numbers | `1e-5`, `0.05`, `2` | What counts as stable |

Detection and restoration tuning (defaults are the original constants):

| Setting | Default | Purpose |
| --- | --- | --- |
| `threshold_scale` | `1.0` | Multiply the detection thresholds (metres) |
| `median_windows` | `3/9/25` | Small/mid/large median windows, in QC cells |
| `flank_distances` | `4/6/9/12` | Flank distances, in QC cells |
| `plane_tolerance` | `1.75` | Metres a local plane may miss by and still protect a cell |
| `restoration_length_scale` | `1.0` | Multiply the restoration distances |
| `restoration_fit_scale` | `1.0` | Multiply the restoration fit tolerance |

Windows and flank distances are counted in grid cells, so the same values
cover nine times the ground on a 1" grid as on a 1/9" grid. In tests, filtering
directly on a 1" grid with the default constants removed real relief; either
filter on a finer `qc_res` or scale the thresholds up and the windows down.

Execution (results do not depend on these): `workers` (processes for the
geometry step, default 4) and `cache_dir` (default `<module cache>/stack_qc`).

## Caching

Each QC context's results are cached under `<cache_dir>/<key>/`. Keys bind
source contents and roles, per-source weights/invocations, region/grid, detection
and restoration settings, working interpolation and recursive QC code hashes.
Only changes outside the selected context can reuse decisions unchanged.
Existing pass-state and changed-block reuse remain in place. Final DEM tiers can
change without invalidating decisions when the working interpolation is unchanged.

Explicit legacy single-module grid caches retain the original behavior. New joint
grids materialize current streams into private per-entry paths before exact-content
decision-cache lookup. This avoids stale hits for different upstream transforms,
raw source changes or weights, and prevents concurrent views from overwriting
each other's input grids. Warm joint runs can reuse QC decisions but still read,
transform and grid the source streams. A universal immutable source-grid cache
needs upstream payload/transform/datum fingerprints and is not enabled here.
If current points cannot reach `stack_qc_grid`, the hook fails rather than silently
filtering an old side-stack. See [architecture](../../stack-qc-joint-context.md)
for ownership and remaining reuse work.

## Choose the QC grid

First choose the normal final DEM grid (`stack_combine.res`, the first
`ms_binary_cudem.resolutions` increment, and `stack_qc.res`). Then either omit `qc_res` to use that
grid, or set it to an equal or finer **integer subdivision**:

**QC increment = final DEM increment / N, where N = 1, 2, 3, ...**

| Final grid | qc_res | Result |
| --- | --- | --- |
| Geographic 1s | `"1s"` | Same grid |
| Geographic 1s | `"1/2s"` | 2×2 QC cells per DEM cell |
| Geographic 1s | `"1/3s"` | 3×3 QC cells per DEM cell |
| Geographic 1s | `"1/9s"` | 9×9 QC cells per DEM cell |
| Geographic 1s | `"0.4s"` | Error: 1 / 0.4 = 2.5 is not an integer |
| Geographic 1s | `"2s"` | Error: QC is coarser than the DEM |
| Geographic | `3` | Error: arcsecond suffix required |
| Projected 30 metres | `3` | 10×10 QC cells per DEM cell |
| Projected 30 metres | `4` | Error: 30 / 4 = 7.5 is not an integer |

This is a rule, not a fixed menu of supported resolutions. Any positive integer
subdivision is allowed, subject to matching grid geometry and practical cost.

For geographic grids, always use an explicit arcsecond suffix, such as
`qc_res: "1/9s"`. Decimal arcseconds such as `"0.5s"` also work. Bare numbers are
rejected, even if their numerical ratio would fit. For projected grids, use a
number in the CRS's linear units: `qc_res: 3` means 3 metres in a metre-based
CRS, or 3 feet in a feet-based CRS. Units follow the **stack grid CRS**, not raw
source coordinates. `3m` and `3t` are rejected as ambiguous. Do not use `1_9`:
YAML can read it as 19. No metre-to-degree approximation is performed.

## Extents and alignment

The resolved final DEM grid owns the bounds and origin. Native Globato may round
an arbitrary requested AOI to whole final-grid cells. QC uses **those resolved
bounds**, rather than independently rounding the original AOI at the finer
resolution. Its working width and height are exactly N times the DEM grid's
width and height. `stack_qc_grid` and `stack_qc` compute this grid with the
same function, so the QC grids always match what `stack_qc` expects.

Both grids must share the same CRS and north-up orientation. The AOI reference
must match the stack's horizontal CRS. Source side-stacks must exactly match
the expected QC grid; every final DEM cell contains a whole N×N block of fine
cells. For factor two, cell **edges** nest; coarse and fine cell centres do not
coincide. The algorithm sums cell contributions and does not require shared
sample centres. It does not interpolate or warp misaligned masks to repair them.

Invalid units or ratios fail when the hook is created, before any data is
fetched. Actual side-stack geometry is checked before QC; an
incompatible artifact raises an error. Only floating-point serialization noise
is tolerated in the subdivision ratio; accepted increments are canonicalized
as final increment / N to avoid accumulated edge drift.

## How accepted measurements reach the DEM

QC and restoration operate on each source's QC-grid side-stack. Accepted
seven-band statistics are summed **per source** into complete DEM cells and
written as a side-stack on the DEM grid and recorded as the entry's
`qc-side-stack` artifact. `stack_combine` (with `source_artifact:
qc-side-stack`) reduces the accepted side-stacks with every other module's. One
survey may be rejected while an overlapping survey remains. Interpolated
working-DEM values never enter the stack.

A factor of nine means up to 81 times the working cells; finer QC costs more.

## Provenance

Give `provenance`, `source_masks` and `stack_provenance`
`source_artifact: qc-side-stack` so they run at the collection stage on the
stacked side-stacks and record the filtered coverage (`stack_provenance`
after `stack_combine`). At the stream stage they would see the sources before
filtering.

## Review pages and standalone Raster Viz

Set `viz: true` on `stack_qc` to automatically publish the review after the
recipe has built and cropped its final DEM. No extra review hook is required:

```yaml
- name: stack_qc
  args:
    # Keep the recipe's existing source/grid/method/interpolation settings.
    viz: true
```

The final collection hook reads cached QC results and the finished DEM. It writes
`stack_qc_review/<module>_<key>/review.html` and a portable schema-v1
`products.json` beside the final DEM. The catalog contains before/after measured
surfaces, the initial QC working interpolation, and the final combined DEM.
Per-pass figures remain in the HTML review. Filtering and the final DEM are
unchanged by publication. Raster Viz remains a separate consumer:

```bash
raster-viz /path/to/stack_qc_review/<module>_<key>/products.json --3d
```

To customize the review's input DEM, location or placement, an explicit
`stack_qc_review` in `global_hooks` overrides automatic publication. Place it
after the final interpolation and crop. Existing explicit recipes remain valid
and do not get a second automatic review. `viz: false` (the default) adds no
automatic review; an explicitly requested HTML review can still be used.

Automatic setup uses Fetchez's generic `FetchHook.final_hooks()` contract,
provided by [Fetchez #525](https://github.com/continuous-dems/fetchez/pull/525). It works through `globato run`,
normal `Recipe.compile()/execute()` and the native `run_fetchez` pipeline,
including when Stack QC is attached to a source module. Automatic reviews use
only the receipts recorded by their requesting QC hook, avoiding stale results
from other builds in the same Python process. They never import Raster Viz.
Review failures are logged in `review_ERROR.txt` and do not modify the DEM.

## Limitations

Decisions apply to source grid cells, not individual sonar beams. Original
-sounding validation and memory/runtime on full tiles remain important checks.


## Named recipe entries

Data-module declarations may have an `id`, e.g. `id: nos_xyz` on a `nos_hydro`
XYZ declaration and `id: nos_bag` on a BAG declaration. Select these labels in
`modules` and `add_ref_modules`; use matching selections on `stack_qc_grid`.
Named recipe entries require [Fetchez #526](https://github.com/continuous-dems/fetchez/pull/526). Existing provider/argument
selectors remain valid. IDs distinguish declarations without replacing native
processed contribution identity or scientific survey groups.

For measured entry coverage, configure the normal `provenance` hook with
`source_artifact: qc-side-stack`, `group_by: recipe_id`, and the final grid `res`.
It emits a bitmap and a JSON legend. With `viz: true` on `stack_qc`, an automatic final review hook publishes a schema-v1 `products.json` alongside its HTML review. The four
default products are before/after measured elevations, the initial QC working
DEM, and the final combined Globato DEM. Pass diagnostics remain in
`review.html`; publishing a catalog does not import or run Raster Viz. See
[entry selection](../../stack-qc-entry-selection.md) for full semantics.
