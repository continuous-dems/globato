# Identity and selectors in the existing architecture

The existing DAV branch fix is in Globato commit
`6aee97fe2178ed380373065fdf4567dfbe29c44d` (PR #281): `drop_class.stack_partition`
separates class-selection streams, and MultiStack duplicate tracking respects the
branch label. That class-selection logic is retained unchanged. The inspected
Fetchez source also already provides `contribution_id`, `set_contribution_id` and
`dataset_id`; its `docs/source/contribute/entry.md` explains those contracts.

| Identity | Meaning / owner | Behavior |
| --- | --- | --- |
| Provider name (`dav`, `nos_hydro`) | Discovery/reader module type | Preserved |
| `module_id()` | Fetchez's canonical module configuration and provider cache | Reused unchanged; a friendly label does not create another download identity |
| Recipe `id` | Human-selected identity of one YAML declaration | New optional recipe metadata, carried as `module.recipe_id` |
| `contribution_id` / `dataset_id()` | Native processed-source identity | Explicit recipe IDs enter the existing contribution mapping before manifest/cache hooks; other components are preserved |
| `stack_partition` | Existing DAV class-branch label | Existing class filters and sink behavior retained |
| Source acquisition group | Scientific survey independence | Unchanged; recipe/module/class labels do not create independent survey votes |
| Coverage bitmap bit | Where a contribution has accepted measurements | `provenance.group_by: recipe_id` distinguishes YAML declarations; legend links bits to recipe IDs and provider metadata |

The only new user concept is a readable recipe label. No new identity hash,
provider alias registry, branch filter or stacking reducer is introduced. Core
processed-source serialization remains Fetchez's responsibility. Globato's
existing side-stack/QC keys consume that native identity, per-source content and
weights as before. Source masks use the same native contribution ID when present.

## YAML

```yaml
modules:
  - id: mbdb_input
    module: mbdb
    args: {weight: 1}
  - id: nos_xyz
    module: nos_hydro
    args: {datatype: xyz, weight: 0.5}
  - id: nos_bag
    module: nos_hydro
    args: {datatype: bag, survey_id: H11891, weight: 3}
```

In the global hooks, edit these lists once; the actual examples use YAML anchors
so grid production and QC select exactly the same entries:

```yaml
- name: stack_qc_grid
  args:
    modules: &filter_modules [mbdb_input, nos_xyz]
    add_ref_modules: &reference_modules [nos_bag]
    res: 1s
    qc_res: 1/9s
    crs: EPSG:4269+5703
- name: stack_qc
  args:
    modules: *filter_modules
    add_ref_modules: *reference_modules
    # retain the original science/grid/interpolation arguments
```

The target entries automatically supply context. Extra references are fixed,
unfiltered and retained in final assembly. Entries in neither list are excluded
from QC and pass through normal final stacking. The same entry cannot have both
roles. All targets share the invocation's method and pass policy.

A string matching a recipe ID selects that declaration before provider shorthand
is considered. To select all NOS invocations explicitly, use `{module: nos_hydro}`.
`{id: nos_xyz}` is an explicit label selector. Existing recipe-style argument
selectors such as `{module: dav, args: {survey_id: 4912}}` remain supported.
Misspelled labels/providers fail before stream consumption when using compiled
recipes. Spatially absent data from valid selections produce coverage warnings.

Duplicate/empty labels are rejected before bundle deduplication and after
expansion. Unnamed legacy declarations receive `entry_1`, `entry_2`, etc. as
runtime display labels; use explicit IDs for stable user selection across recipe
reordering and for intentionally separate same-file views. Only explicit IDs are
added to processed contribution identity, preserving unlabeled legacy cache and
stacking behavior. This does not authorize counting duplicate raw measurements
as independent survey evidence.

`provenance.group_by` defaults to the historical `module` behavior. The new study
uses `recipe_id`; `module_id` is also available for configuration-level grouping.
Instance bitmap modes require collection-stage `source_artifact` and support at
most 32 instances, failing rather than overflowing. `source_masks` remains the
native per-source option for larger inventories. The bitmap describes measured
coverage before final interpolation; it does not assign origin to filled cells.

## Same DAV survey, different classes

For example, declare separate entries `dav_4912_ground` and `dav_4912_bottom`,
both using `module: dav` and `survey_id: 4912`. Keep the existing `drop_class`
filters (class 2 and class 29 with `invert: true`) and their `stack_partition`
labels. Select the friendly IDs in QC. Unit coverage executes the actual native
class filter on one physical source, at the same weight, proving separate target
and reference contributions without changing the scientific acquisition group.
No lidar class logic is reimplemented.

## Code ownership

Fetchez owns recipe label validation/assignment, and the small core bridge that
uses `set_contribution_id(entry, "recipe", label)` before cache/sink hooks.
Globato owns selection, native QC preparation, per-entry coverage, and optional
review metadata. Regional recipes, result summarization and Raster Viz launchers
stay outside production hooks. Optional recipe IDs are proposed in
[Fetchez #526](https://github.com/continuous-dems/fetchez/pull/526); automatic final
publication uses [Fetchez #525](https://github.com/continuous-dems/fetchez/pull/525).
The local validation kit combines those exact PR revisions in a separate checkout.
No VDatum acquisition change is required. Resolve the supported Fetchez dependency
before merging the final Globato candidate.
