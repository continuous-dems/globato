# 💻 Command Line Interface

The `globato` command line tool allows you generate Digital Elevation Models.

Build and optionally execute a reproducible coastal DEM workflow.

`globato build` provides a domain-focused interface for composing Globato DEM
recipes from registered Fetchez components. It combines a DEM build method,
one or more elevation sources, and optional source-specific processing into a
single executable recipe.

Components are chained from left to right.

## A Globato DEM build consists of:

* A DEM build method, such as `mr-globato`, which defines the global
stacking, interpolation, provenance, and output workflow.

* One or more elevation modules or bundles. Only sources registered as
Globato-compatible elevation streams are available through this command.

* Optional Fetchez hooks or presets placed after a source. These are
attached only to the preceding module or bundle and can be used to
customize how that source is processed before it enters the DEM workflow.

If no DEM build method is specified, `mr-globato` is used by default.

Global processing is controlled by the selected DEM build method. Additional
hooks and presets must therefore follow a source module or bundle rather than
appearing before the first source.

## Examples:

Build a DEM from the standard CRM source bundle:

```bash
globato build -R <W/E/S/N> -E 1s crm-bathy-topo
```

  Explicitly select and configure the multi-resolution build method:

```bash
globato build -R <W/E/S/N> -E 1s mr-globato --stack-strategy mixed crm-bathy-topo
```

Select only particular members of a source bundle:

```bash
globato build -R <W/E/S/N> -E 1s mr-globato glob-tnm --select products=1m/1_9as
```

Attach processing to an individual source:

```bash
globato build -R <W/E/S/N> -E 1s mr-globato tnm raster_warp --res 1s
```

Attach several hooks or presets to one source before adding another:

```bash
globato build -R <W/E/S/N> -E 1s mr-globato tnm spatial_claim audit copernicus
```

Use `globato build <DEM-method> --help` to inspect the options exposed by a
particular DEM build method.

Use the Fetchez module, bundle, hook, and preset discovery commands to inspect
the components that can be composed into a workflow.

Use `--export` to write the generated recipe to YAML instead of executing it.
The exported recipe can later be reproduced with `fetchez run`.

```{eval-rst}
.. click:: globato.cli:cli
   :nested: full
   :prog: globato
```
