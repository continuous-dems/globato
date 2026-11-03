# Stack QC examples

The repository contains four complete recipes in `examples/stack_qc/`:

| Recipe | Source QC and final DEM grid |
| --- | --- |
| `redondo_1_9.yaml` | Redondo, 1/9 arcsecond |
| `redondo_1s.yaml` | Redondo, 1 arcsecond |
| `la_canyons_1_9.yaml` | LA Canyons, 1/9 arcsecond |
| `la_canyons_1s.yaml` | LA Canyons, 1 arcsecond |

All use NAD83 geographic bounds (EPSG:4269) and the same filter settings. They
stack with side_stack + stack_combine (stack_qc filters side-stacks before they
are combined) and end with stack_qc_review. They omit optional qc_res so QC
follows each final grid. Use separate output folders:

```bash
globato run examples/stack_qc/redondo_1s.yaml \
  --outdir /path/to/redondo_1s --shared-cache /path/to/data
```

To evaluate finer QC on the 1s final grid, copy a 1s recipe and add
`qc_res: "1/9s"` to its module's stack_qc args (the interpolation ladder stays
written for the 1s grid; it is scaled down automatically). See the Stack QC user guide for units,
integer nesting, processing bounds and optional coverage consumers.

## Separate reproduction package

The review ZIP includes three standalone comparison scripts under `examples/`
and complete Linux commands in RUN.md. They are separate from production code:

| Script | Comparison |
| --- | --- |
| `compare_resolutions.py` | Independent 1s final DEM versus averaged 1/9s final DEM |
| `compare_qc_resolutions.py` | Accepted measured cells before final interpolation |
| `fine_qc_native_1s.py` | Actual production qc_res=1/9s builds, retaining final 1s stack/interpolation |

The third script derives its two recipes in the result folder and invokes
ordinary Globato. It has no independent mask reducer or interpolator. The
package also includes two local synthetic resolution smokes and a combined
review generator; these validation drivers are not installed production hooks.

Prior completed field evidence is preserved separately in that package:
direct 1s versus averaged fine DEM RMSE was 25.40m at Redondo and 13.57m at LA;
prior source-cell hybrid versus averaged fine was 3.01m and 3.53m. These measure
processing sensitivity, not absolute accuracy. The prior context ladder differs
from the new qc_res path and does not validate the new integration. Current
field results and original-sounding review remain separate scientific gates.
