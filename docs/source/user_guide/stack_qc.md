# Stack QC — source quality control before stacking

Stack QC evaluates each source's gridded elevation contributions, removes
inconsistent cells, and restores original contributions supported by the
surrounding terrain and survey evidence. Source files are unchanged;
decisions are recorded as 0/1 masks on each source's grid.

`stack_qc` is a standard collection-stage hook. It works in recipes that stack
with `side_stack` and `stack_combine` (for example the `mr-globato-ss`,
`mr-globato-crm` and `mr-globato-cudem` presets): the normal stream stage writes
each source's side-stack, `stack_qc` replaces a module's side-stacks with
filtered ones, and `stack_combine` assembles the MultiStack. It refuses a
recipe that stacks with `multi_stack`, which has already consumed the
unfiltered streams by the collection stage.

Attach it to a module, after the module's source transforms:

```yaml
- module: mbdb
  hooks:
    - name: set-srs
      args: {srs: "EPSG:4326+vdatum:msl"}
    - name: stream_data
    - name: stream_reproject
      args: {dst_srs: "EPSG:4326+3855"}
    - name: stack_qc
      args:
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
```

Or list it in `global_hooks`, before `stack_combine`, with `modules: mbdb` (or
several module names). Use the global form when global manifest hooks drop
files you do not want in the filter (module hooks run before them).

The optional `stack_qc_review` hook writes an HTML review with before/after
hillshades. Put it in `global_hooks` after the final interpolation (see
[Review pages](#review-pages)).

## Settings

Grid and working DEM:

| Setting | Default | Purpose |
| --- | --- | --- |
| `res` | `1s` | DEM grid of the accepted side-stacks; equal to `stack_combine.res` |
| `crs` | `EPSG:4326+3855` | Stack CRS; equal to `stack_combine.crs` |
| `qc_res` | none | Optional finer QC grid, `res / N` (e.g. `1/9s` for a 1s DEM) |
| `stack_strategy` | `mixed` | Working-DEM MultiStack strategy |
| `stack_weight_threshold` | `3/2/1/.75/.5/.4/.3/.2/.1` | Working-DEM MultiStack weight tiers |
| `interpolation` | CRM 1" ladder | `ms_binary_cudem` arguments for the working DEMs |

The working DEMs are the surfaces each pass compares source cells with. They
are built from the module's own sources only, in the build's own process, with
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

Execution (results do not depend on these): `threads` (source gridding),
`workers` (processes for the geometry step, default 4), `cache_dir` (default
`<module cache>/stack_qc`), `side_stack_cache`.

## Caching

Each module's results are cached under `<cache_dir>/<key>/`. The key covers
only what can change the decisions: the module's sources (side-stack identity
and checksum, which include the module's own processing hooks), the module
weight, the region, the QC grid, the detection and restoration tuning, the
working-DEM settings and the Stack QC code. Changing another dataset, its
weight, or the final DEM's tiers reuses the cached decisions. Pass results are
cached per pass, so raising `max_passes` continues from the last cached pass,
and a different `aggressiveness` or `terrain_restoration` reuses the shared
working DEMs where it can. The accepted side-stacks are cached in the same
folder.

The QC-grid side-stacks used with `qc_res` are cached as ordinary side-stacks
next to the module's other side-stacks.

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
width and height. QC uses a module copy; the ordinary final build keeps the
original recipe module.

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
written as a side-stack on the DEM grid; the entry's `side-stack` artifact then
points to it (the original is recorded as `stack_qc_original_side_stack`).
`stack_combine` reduces the accepted side-stacks of all modules as usual. One
survey may be rejected while an overlapping survey remains. Interpolated
working-DEM values never enter the stack.

A factor of nine means up to 81 times the working cells; finer QC costs more.

## Provenance

`provenance` and `source_masks` run in the stream stage, before Stack QC, so
they record each source's coverage before filtering.

## Review pages

```yaml
global_hooks:
  # ... ms_binary_cudem, focus_sink ...
  - name: stack_qc_review
    args: {dem: "%name%_%batch_name%.tif"}
```

For each Stack QC result in the build, `stack_qc_review` writes
`stack_qc_review/<module>_<key>/review.html` next to the DEM (or under
`output`): hillshades of the module's working DEM before filtering and after
each pass, removal footprints, restored terrain, and the final combined DEM,
with GeoTIFF downloads. It reads only cached results and never changes the
DEM; a failure is logged and written to `review_ERROR.txt`. Production builds
can leave it out. It needs matplotlib.

## Limitations

Decisions apply to source grid cells, not individual sonar beams. Original
-sounding validation and memory/runtime on full tiles remain important checks.
