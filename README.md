# 🌍 Fetchez-Globato 🤖

**Domo Arigato, Multi-Resolution Globato.**

> ⚠️ **BETA STATUS:** This project is in active development (v0.1.0).

**Globato** (*Global Bathymetry & Topography*) is the next-generation DEM generation suite for the `fetchez` ecosystem. Originally part of the [CUDEM](https://github.com/ciresdem/cudem) project, Globato unifies data discovery, download, and processing into a single, streaming pipeline.

## ❓ Why Globato?

Building Digital Elevation Models (DEMs) typically involves a "download-then-process" workflow that requires massive storage and directories full of custom scripts.

**Globato changes the paradigm.** It acts as a streaming extension to `fetchez`, allowing you to:
* **Stream, Don't Store:** Process points from remote sources (LiDAR, Multibeam, COGs) on-the-fly without saving raw files to disk.
* **Harmonize Resolution:** Seamlessly blend high-resolution multibeam with coarse global topography (hence the **M.R.** in *Mr. Globato*).
* **Standardize Metadata:**
* **ETC**

Whether you are building a quick 30m regional map or a precision 1m surface, Globato keeps your pipeline clean, reproducible, and memory-efficient.

---

## 🌎 Features

* **Streaming Gridders:**
    * **`simple_stack`**: A lightweight, memory-safe stream for generating standard Z-elevation rasters (weighted mean).
    * **`multi_stack`**: A heavy-duty statistical engine that generates 7-band GeoTIFFs containing Elevation, Weight, Count, Uncertainty, Source Uncertainty, and average X/Y locations for every pixel.
* **Provenance Tracking:** Automatically generate bitmask rasters that map exactly which datasets contributed to every pixel in your output.
* **Data Readers:**
    * **Native BAG Support:** A Bathymetric Attributed Grid reader that handles Variable Resolution (VR).
    * **COG Subsetting:** Windowed fetching for Cloud Optimized GeoTIFFs.
* **Modern Architecture:** Built on `rasterio`, `numpy`, and `fetchez`, dropping heavy legacy dependencies for a pure Python experience.
* **Declarative Projects:** Define complex, multi-sensor build pipelines in simple `yaml` files.

## 🔌 How Globato Extends Fetchez

Globato does not provide a separate CLI tool. Instead, it acts as a plugin suite that injects advanced processing capabilities directly into the fetchez engine.

When you install globato, fetchez automatically detects and registers these new capabilities, allowing you to chain them into your existing workflows using the standard --hook syntax.

***The Globato Toolkit***

Globato extends the core ecosystem by adding three types of components:

1. **Data Streams** (The Ingress) Standard fetchez downloads files. Globato turns those files into streaming point clouds.

`stream_data`: Auto-detects file types (LAS, LAZ, BAG, XYZ, OGR) and converts them into a standardized stream of x,y,z,weight,uncertainty records.

`stream_reproject`: Reprojects streaming points on-the-fly using pyproj (e.g., converting WGS84 to UTM Zone 10N in memory).

2. **Filters** (The QA/QC) Clean your data before it ever hits a grid.

`filter`: Applies algorithms like block_thin, outlierz (statistical outlier removal), or rangez to cull bad data from the stream.

3. **Stackers** (The Egress) The core of the "M.R. Globato" engine—turning streams into surfaces.

`simple_stack`: A fast, memory-safe sink for generating standard weighted-mean Elevation rasters.

`multi_stack`: The heavy-duty statistical engine. Generates 7-band GeoTIFFs (Z, Count, Weight, Uncertainty, Source Uncertainty, X-mean, Y-mean) for rigorous analysis.

`provenance`: Generates a bitmask raster tracking exactly which dataset contributed to each pixel.

4. **Specialized Modules**

`gebco_cog`: A specialized fetch module for the GEBCO global bathymetry dataset, optimized for COG subsetting.

🚀 Usage Example

Because Globato is just a set of hooks, a complex ETL job looks just like a standard fetchez command.

Example: The "M.R. Globato" Pipeline fetches multibeam data, filters outliers, reprojects to NAVD88, and grids it—without saving intermediate files.

```bash
fetchez multibeam -R -124.5/-124.0/43.0/43.5 \
    --weight 1.0 --uncertainty 0.5 \
    # Turn download into a stream
    --hook stream_data \
    # Reproject stream to WGS84/NAVD88
    --hook stream_reproject:dst_srs=EPSG:4326+5703 \
    # Filter statistical outliers (3-sigma)
    --hook filter:method=outlierz:threshold=3.0 \
    # Grid into a 7-band statistical surface
    --hook multi_stack:res=10:mode=mean:output=coos_bay_stack.tif
```