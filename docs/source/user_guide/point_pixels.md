# 📐 Point-to-Pixel Fusion State

`PointPixels` is Globato's point-to-pixel statistical reducer.

Its job is intentionally narrow:

```text
elevation observations
        ↓
PointPixels
        ↓
additive FusionState
```

It does not interpolate elevation values, choose between source datasets, or produce the final MultiStack.

Those responsibilities belong to later stages of the DEM pipeline. See [DEM Generation](dem_generation.md)

---

## Why FusionState Exists

Elevation data arrive in streams and may contain millions or billions of individual observations.

Globato therefore cannot assume that all observations for a raster cell will be available at the same time.

A useful accumulation representation must produce the same result regardless of how data are divided into chunks.

For observation groups `A` and `B`, Globato requires:

```text
FusionState(A + B)
    ==
FusionState(A) + FusionState(B)
```

This property makes FusionState:

* associative
* chunk-independent
* streamable
* cacheable
* serializable
* resumable
* mergeable

It also allows already-processed elevation sources to be stored as FusionState and later added directly to a MultiStack without reconstructing the original point observations.

---

# Preparing Points

Before accumulation, `PointPixels` maps incoming observations onto the target raster grid.

Incoming point records must contain:

```text
x
y
z
```

and may optionally contain:

```text
w
u
```

where:

* `w` is the observation weight
* `u` is the observation uncertainty

Missing weights default to `1`.

Missing uncertainties default to `0`.

Invalid or non-finite X, Y, or Z values are excluded before accumulation.

Internally, the prepared chunk records:

* filtered point coordinates
* elevation
* weight
* uncertainty
* global pixel indexes
* local pixel indexes
* flattened pixel indexes
* local source window
* local raster shape

This preparation step is represented by `PreparedPoints`.

---

# Coverage and Count Operations

Not every consumer needs a complete FusionState.

For example, provenance hooks only need to know whether observations occurred in a pixel.

`PointPixels` therefore exposes lightweight spatial operations separately from fusion accumulation.

## `count()`

```python
count, window, gt = pixels.count(points)
```

Returns the number of observations mapped to each populated pixel.

## `coverage()`

```python
coverage, window, gt = pixels.coverage(points)
```

Returns a boolean raster equivalent to:

```python
coverage = count > 0
```

This is used by provenance and source-mask generation without calculating unnecessary statistical state.

---

# FusionState Schema

Calling:

```python
state, window, gt = pixels.accumulate(points)
```

produces seven additive sufficient statistics.

| Band | Field                     | Definition  |
| ---: | ------------------------- | ----------- |
|    1 | `z_weighted_sum`          | Σ(z × w)    |
|    2 | `count`                   | N           |
|    3 | `weight_sum`              | Σw          |
|    4 | `z2_weighted_sum`         | Σ(z² × w)   |
|    5 | `weighted_uncertainty_sq` | Σ((w × u)²) |
|    6 | `x_weighted_sum`          | Σ(x × w)    |
|    7 | `y_weighted_sum`          | Σ(y × w)    |

Every value in FusionState is additive.

No weighted means, standard deviations, or finalized uncertainty estimates are stored at this stage.

---

# Source Weight

A source-level weight can be applied while accumulating:

```python
state, window, gt = pixels.accumulate(
    points,
    source_weight=0.8,
)
```

The effective observation weight is:

```text
effective_w = point_w × source_weight
```

All weighted sufficient statistics use this effective value.

Source weighting therefore remains compatible with associative accumulation.

---

# Source Uncertainty

A source-level uncertainty may also be supplied:

```python
state, window, gt = pixels.accumulate(
    points,
    source_uncertainty=2.0,
)
```

The source uncertainty is combined with per-point uncertainty before the additive uncertainty numerator is accumulated.

FusionState stores the sufficient statistic needed for later uncertainty propagation rather than a finalized uncertainty value.

This distinction is important because finalized uncertainty is generally not associative, while its underlying sufficient statistics can be.

---

# Finalizing FusionState

Derived raster values are calculated by:

```python
final = finalize_fusion_state(state)
```

The returned values include:

```text
z
x
y
count
weight_sum
mean_weight
uncertainty
stddev
valid
```

## Weighted Elevation

```text
z = Σ(z × w) / Σw
```

## Mean Weight

```text
mean_weight = Σw / N
```

## Weighted Coordinates

```text
x = Σ(x × w) / Σw
y = Σ(y × w) / Σw
```

These coordinates represent the weighted mean position of the observations contributing to the pixel.

## Elevation Dispersion

FusionState stores the weighted second moment:

```text
Σ(z² × w)
```

allowing weighted variance to be calculated during finalization:

```text
variance =
    Σ(z² × w) / Σw
    - z²
```

and:

```text
stddev = sqrt(variance)
```

The standard deviation describes the spread of observed elevations within the pixel.

## Measurement Uncertainty

For independent input uncertainties, the weighted-mean uncertainty is derived from the stored uncertainty numerator:

```text
Σ((w × u)²)
```

during finalization.

Measurement uncertainty and observed elevation dispersion remain separate outputs.

Globato does not currently force them into a single combined uncertainty metric.

---

# Associative Reduction

Two compatible FusionStates can be merged through simple addition:

```python
merged = merge_fusion_states(state_a, state_b)
```

Conceptually:

```text
merged.z_weighted_sum =
    a.z_weighted_sum + b.z_weighted_sum

merged.count =
    a.count + b.count

merged.weight_sum =
    a.weight_sum + b.weight_sum

...
```

If merging requires more complicated mathematics than element-wise addition, the FusionState contract has been violated.

That simplicity is intentional.

---

# Chunk Independence

The primary invariant of the module is:

```text
accumulate(all_points)

    ==

merge(
    accumulate(chunk_1),
    accumulate(chunk_2),
    ...
)
```

for every FusionState field.

This invariant is tested directly in Globato's test suite.

It ensures that numerical results do not depend on:

* source reader chunk size
* streaming boundaries
* thread or process scheduling
* side-stack cache usage
* persisted-state resume boundaries

---

# FusionState and MultiStack

FusionState is not the same thing as the finalized MultiStack.

## FusionState

FusionState stores sufficient statistics:

```text
Σ(z×w)
N
Σw
Σ(z²×w)
Σ((w×u)²)
Σ(x×w)
Σ(y×w)
```

It is:

* additive
* resumable
* mergeable

## MultiStack

The finalized MultiStack stores derived quantities:

```text
z
count
weight
uncertainty
stddev
x
y
```

It is intended for:

* inspection
* interpolation
* downstream raster processing

Finalization is lossy with respect to future accumulation.

For that reason, finalized MultiStack rasters must not be treated as FusionState.

---

# FusionState and `multi_stack`

`MultiStackAccumulator` accepts both raw point observations and already-generated FusionState.

For points:

```text
points
  ↓
PointPixels.accumulate()
  ↓
FusionState
  ↓
MultiStackAccumulator.update_state()
```

For pre-accumulated state:

```text
FusionState
  ↓
MultiStackAccumulator.update_state()
```

This is the basis for:

* side-stack caching
* resumable global accumulation
* incremental DEM updates

No point reconstruction is required when an exact FusionState is already available.

---

# Persistent FusionState

FusionState may be serialized to a seven-band GeoTIFF.

A persisted state includes identifying metadata such as:

```text
GLOBATO_DATATYPE = FUSION_STATE
GLOBATO_FUSION_VERSION = 1
```

and band descriptions matching the FusionState schema.

Before resuming an existing state, Globato validates:

* raster dimensions
* geotransform
* CRS
* number of bands
* FusionState schema version
* band descriptions

This prevents an arbitrary seven-band raster or finalized MultiStack from being mistaken for resumable accumulation state.

---

# `PixelsToPoints`

`PixelsToPoints` converts FusionState into one representative point per populated raster cell.

This is intentionally a **lossy finalization operation**.

The representative point contains finalized values such as:

```text
x
y
z
w
u
```

but does not retain all sufficient statistics from the original FusionState.

Therefore:

> `PixelsToPoints` must not be used to restore a FusionState cache when exact accumulation parity is required.

Its purpose is to produce representative point observations when a downstream operation specifically requires point records.

---

# Design Principle

The central rule of `point_pixels` is:

> Store sufficient statistics during accumulation and derive statistics only at finalization.

This allows Globato to preserve exact statistical state across streaming, caching, and resume boundaries.

The resulting processing model is:

```text
point observations
      ↓
PreparedPoints
      ↓
FusionState
      ↓
merge / cache / resume
      ↓
finalization
      ↓
derived raster statistics
```

That contract is the foundation for Globato's incremental and reproducible DEM accumulation workflow.
