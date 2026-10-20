# Stack QC — module-attached source quality control

Stack QC evaluates gridded elevation-source contributions, removes inconsistent cells,
and restores original contributions supported by the surrounding terrain and survey evidence.
It runs **before** Globato's normal final stacking and interpolation. Source files are
unchanged; accepted/rejected contributions are recorded as masks.

Add it to the source module's `hooks:` list after coordinate conversion:

```yaml
- name: stack_qc
  args:
    aggressiveness: 2
    terrain_restoration: 2
    max_passes: 5
    stop_when_stable: false
    review: true
```

| Setting | Values | Default | Purpose |
| --- | --- | --- | --- |
| `aggressiveness` | `1`, `2`, `3` | `2` | Strength of anomaly filtering |
| `terrain_restoration` | `0`–`3` | `2` | Restore supported original terrain, or turn off at `0` |
| `max_passes` | `1`–`10` | `4` | Maximum rejection passes |
| `stop_when_stable` | boolean | `true` | Stop when scientific changes stabilize |
| `qc_res` | Positive grid increment | DEM increment | Optional finer QC/source grid; e.g. `1/9s` for 1/9 arcsecond |
| `review` | boolean | `false` | Generate HTML, figures, measured surfaces and Perspecto GIS hillshades |

The hook reads the native `multi_stack` grid and **automatically manages
seven-band side-stacks**. No `side_stack` or `side_stack_check` YAML hooks are needed.
Cache reuse is validated against source identity and grid settings. A matching
raw-data cache does not guarantee that every upstream discovery API is offline-ready.

Examples: `examples/stack_qc/redondo_1_9.yaml` and the other three recipes
in that directory (under `examples/` in the release ZIP). Region CRS belongs to the Globato/Fetchez recipe, not Stack QC.
All four samples specify NAD83 geographic (`EPSG:4269`).

The `review.html` output includes rejection/restoration views and separate
GIS-resolution products; it does **not** alter the scientific DEM. Color hillshades
are display products. Globato's final interpolated DEM can combine multiple modules;
the module-only review surfaces must not be confused with it.

**Limitations:** decisions apply to source grid cells, not individual sonar beams.
Cell-based neighborhoods change physical size with grid resolution, so QC at
1/9″ and 1″ is not interchangeable. Memory and runtime on full quarter-degree
tiles and validation against original soundings remain important checks.

Run with `globato run recipe.yaml --shared-cache /path/to/cache --outdir /path/to/build --fail-fast`.
Use stock Fetchez; `fetchez run` does not perform Globato's collection preparation.
The native context needs one global `multi_stack` and one global `ms_binary_cudem`.
Source readers/transforms run before QC; put source consumers (stacking and
provenance) in `global_hooks` so they receive accepted contributions once.
Unknown deferred manifest hooks fail before discovery; the native side-stack
lookup runs after selection and before source reading.

QC recipes rebuild final stacking/provenance consumers, even when their normal
`resume` option is true: additive final states cannot remove an older QC decision.
Raw downloads, source side-stacks and QC contexts still reuse matching caches.
Use separate output folders for comparisons. Do not run concurrent builds into
the same final-output folder.

For Python recipe execution use `globato.qc.stack.pipeline.StackQCRecipe`.
Ordinary recipes compiled by this class return stock Fetchez's unchanged plan.
`globato.read`, `globato.build`, and direct `fetchez.recipe.Recipe` execution do
not opt in to module collection preparation; the supported entry point is
`globato run`. This PR does not change the streaming Python API.

Review publication runs after the production DEM. A display failure is recorded
in `review_ERROR.txt` and logged; inspect that file if `review.html` is missing.
It does not invalidate the DEM or change masks. A missing final DEM is an error.

## Choose the QC grid

First choose the normal final DEM grid (`multi_stack.res` and the first
`ms_binary_cudem.resolutions` increment). Then either omit `qc_res` to use that
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
The five existing controls keep their meaning.

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
width and height. Preparation uses a module copy; discovery and the ordinary
final build retain the original recipe module.

Both grids must share the same CRS and north-up orientation. The AOI reference
must match the stack's horizontal CRS. Source side-stacks must exactly match
the expected QC grid; every final DEM cell contains a whole N×N block of fine
cells. For factor two, cell **edges** nest; coarse and fine cell centres do not
coincide. The algorithm sums cell contributions and does not require shared
sample centres. It does not interpolate or warp misaligned masks to repair them.

Explicit requests are checked before provider discovery. Invalid units, ratios or AOI CRS fail with instructions
to correct the request. Actual side-stack geometry is checked before QC; an
incompatible artifact raises an error. Only floating-point serialization noise
is tolerated in the subdivision ratio; accepted increments are canonicalized
as final increment / N to avoid accumulated edge drift.

## How accepted measurements reach the DEM

QC and restoration operate on the original fine source grids. The private QC
context interpolation ladder scales by 1/N; weights, algorithms and tier count
stay configured. Its first resolution matches the QC grid exactly. Final stack
and interpolation hooks stay unchanged.

Accepted seven-band statistics are summed **per source** into complete final DEM
cells before ordinary source reduction. One survey may be rejected while an
overlapping survey remains. Fine interpolated DEM values are never averaged
back into the measured stack. Optional review measured surfaces use the QC grid
and are labelled separately from the final DEM.

A factor of nine means up to 81 times the working cells. Reads are bounded and
empty source blocks can be skipped, but finer QC still costs more. New additive
aggregation reference tests allow documented floating-point roundoff; default
QC remains unchanged. Changing qc_res can change decisions and does not prove
accuracy. Original-sounding and cached-field validation remain separate gates.

Fraction spelling is implemented for `stack_qc.args.qc_res`; other native hooks
retain their existing increment syntax.

The aligned bounds are DEM/QC **processing bounds**. This change does not add an
extra external QC buffer or change the normal final output rounding/crop policy.
An arbitrary AOI is accepted; the resolved native grid supplies the cell edges.


## Optional provenance

Put `provenance` and `source_masks` in `global_hooks` before `multi_stack`,
with their `res` equal to the final stack increment. They record accepted
measured coverage (`count > 0`) on the final grid, including sources aggregated
from a finer QC grid. They do not treat interpolated cells as observations.
`stack_provenance` belongs after `multi_stack`; it records sources surviving
native weight-tier reduction. Coverage grids must match the final stack's
horizontal CRS, bounds and resolution; incompatible raster streams fail loudly.
