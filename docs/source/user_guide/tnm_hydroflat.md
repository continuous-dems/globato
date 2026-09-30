# TNM coastal hydroflat cleanup

The `tnm_hydroflat` hook removes synthetic water surfaces from coastal TNM
rasters before they enter a DEM stack. Exact and fuzzy hydroflats are detected
from raster evidence first. OSM is resolved only when the tile requires the
irregular-water fallback.

Run this hook **before any resampling or raster warp**. Exact encoded elevations
are the evidence used to identify hydroflats, and interpolation can destroy that
signal.

## Default behavior

The defaults implement the profile validated on Klamath and Yurok, California,
and South Coast, Oregon, 1 m TNM tiles:

1. Search the complete raster from -5 m through +5 m for repeated exact values.
2. Keep only connected components of at least 1,000 m² with at least 25 m² of
   exact-value support.
3. Preserve exact matching as the priority-one decision.
4. When exact components explain too little of the candidate surface, derive a
   residual fuzzy tolerance capped at 0.35 m and require fuzzy components to
   contact accepted exact evidence.
5. Grow accepted removal by at most 2 m to clear narrow seams between adjacent
   hydroflat artifacts.
6. If no exact hydroflat is accepted, generate an OSM topology landmask and
   remove irregular water-side returns at or below 4.5 m.

The elevation fallback never runs after a hydroflat has been accepted. This
prevents a broad elevation cutoff from clipping valid coastal terrain on a
hydro-flattened tile.

## Exact-first automatic mode

The default `flat_tolerance: auto` handles both exact Klamath-style flats and
noisy Yurok-style flats without classifying projects by filename. Automatic
mode is strictly ordered:

1. Run the validated exact-value detector with zero tolerance.
2. Measure the fraction of valid in-range raster cells already explained by the
   accepted exact components.
3. Keep the exact result unchanged when it explains at least
   `auto_exact_min_fraction` of the search surface.
4. Otherwise derive a fuzzy tolerance from the residual distribution,
   bounded by `auto_fuzzy_tolerance_max`.
5. Require each accepted fuzzy component to contact exact hydroflat evidence.
6. Use the irregular-water fallback only when no exact component qualifies.

Exact components remain priority-one evidence and retain the original option
to cross small vector offsets. Fuzzy components are priority-two evidence and
cannot replace an exact decision or propagate freely through similarly
elevated inland terrain.

Raster seed mode treats the hydroflat itself as the coastline evidence, so OSM
cannot prematurely clip exact or fuzzy components. Exact-contact topology keeps
the fuzzy tolerance from becoming an unconstrained tile-wide elevation cutoff.
The irregular-water fallback remains strictly limited to OSM water.

Set `hydroflat_seed_mode: water` to retain vector-guided discovery. In that
compatibility mode, `fuzzy_water_barrier` and `auto_fuzzy_buffer_m` can constrain
fuzzy candidates to vector water and a small collar.

The output tags include the selected mode, exact-water fraction, derived fuzzy
tolerance, and separate exact/fuzzy cell counts. The hook does not classify
projects by filename; it selects the exact, fuzzy, or fallback path from each
raster's elevation evidence.

## Recipe use

Add the hook to a TNM module before `raster_warp`:

```yaml
- module: tnm
  args:
    products: 1m
    formats: GeoTIFF
  hooks:
  - name: tnm_hydroflat
    args:
      stage: file
```

The minimal example uses the tested raster-first adaptive defaults. Their
expanded form is:

```yaml
  - name: tnm_hydroflat
    args:
      stage: file
      landmask: osm
      hydroflat_seed_mode: raster
      land_class_field: class
      land_class_value: land
      water_seed_inset_m: 0
      min_seed_area_m2: 25
      min_component_area_m2: 1000
      flat_min_elevation: -5
      flat_max_elevation: 5
      max_flat_values: 64
      flat_tolerance: auto
      auto_exact_min_fraction: 0.75
      auto_fuzzy_min_fraction: 0.25
      auto_fuzzy_tolerance_max: 0.35
      auto_fuzzy_quantile: 0.999
      auto_fuzzy_buffer_m: 2
      fuzzy_water_barrier: false
      fuzzy_require_exact_contact: true
      connectivity: 8
      clip_flats_to_water: false
      seam_cleanup_m: 2
      irregular_water_max_elevation: 4.5
```

## Direct raster test

```python
from globato.hooks.rasters.tnm_hydroflat import TNMHydroflat

src = "/path/to/USGS_1m_tile.tif"
dst = "/path/to/USGS_1m_tile_hydroflat_clean.tif"

hook = TNMHydroflat()
success = hook.process_raster(src, dst, entry={})

print("SUCCESS:", success)
print("OUTPUT:", dst)
```

The output contains diagnostic GeoTIFF tags recording whether the exact-flat or
fallback path ran, the accepted elevation values, component count, and cell
counts removed by each stage.

## Main safeguards

| Setting | Default | Purpose |
| --- | ---: | --- |
| `hydroflat_seed_mode` | `raster` | Search raster-wide; use `water` for legacy vector-seeded discovery. |
| `min_seed_area_m2` | 25 | Reject exact values with weak raster support. |
| `min_component_area_m2` | 1000 | Reject small repeated terrain features. |
| `max_flat_values` | 64 | Avoid interpreting a heavily quantized raster as many hydroflats. |
| `flat_tolerance` | `auto` | Exact-first residual fuzzy detection; a number requests fixed exact-component tolerance. |
| `auto_exact_min_fraction` | 0.75 | Preserve exact-only behavior when exact components explain this fraction of seeded water. |
| `auto_fuzzy_min_fraction` | 0.25 | Require meaningful residual support near exact targets before fuzzy cleanup. |
| `auto_fuzzy_tolerance_max` | 0.35 | Hard cap on a raster-derived fuzzy tolerance. |
| `auto_fuzzy_quantile` | 0.999 | Ignore only the extreme tail of residual deviations. |
| `auto_fuzzy_buffer_m` | 2 | Limit fuzzy cleanup beyond vector water in water seed mode. |
| `fuzzy_water_barrier` | false | Do not use the vector coastline to clip fuzzy hydroflats. |
| `fuzzy_require_exact_contact` | true | Require fuzzy components to touch accepted exact hydroflat evidence. |
| `clip_flats_to_water` | false | Preserve full confirmed plateaus. |
| `seam_cleanup_m` | 2 | Bound removal around accepted flat components. |
| `irregular_water_max_elevation` | 4.5 | OSM-water fallback used only when no exact flat is accepted. |

Set `irregular_water_max_elevation: null` to disable the irregular-water
fallback. Set `clip_flats_to_water: true` only when a strict landmask clip is
preferred over removal of the complete connected hydroflat.
