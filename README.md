# 🌍 Fetchez-Globato 🤖

**Domo Arigato, Multi-Resolution Globato.**

> ⚠️ **BETA STATUS:** This project is in active development (v0.1.0).

**Globato** (*Global Bathymetry & Topography*) is the next-generation DEM generation suite for the `fetchez` ecosystem. Originally part of the [CUDEM](https://github.com/ciresdem/cudem) project, Globato unifies data discovery, download, and processing into a single, streaming pipeline.

### Why Globato?

Building Digital Elevation Models (DEMs) typically involves a "download-then-process" workflow that requires massive storage and directories full of custom scripts.

**Globato changes the paradigm.** It acts as a streaming extension to `fetchez`, allowing you to:
* **Stream, Don't Store:** Process points from remote sources (LiDAR, Multibeam, COGs) on-the-fly without saving raw files to disk.
* **Harmonize Resolution:** Seamlessly blend high-resolution multibeam with coarse global topography (hence the **M.R.** in *Mr. Globato*).
* **Standardize Metadata:**
* **ETC***

Whether you are building a quick 30m regional map or a precision 1m surface, Globato keeps your pipeline clean, reproducible, and memory-efficient.

---

### Features

* **Streaming Gridders:**
    * **`simple_stack`**: A lightweight, memory-safe stream for generating standard Z-elevation rasters (weighted mean).
    * **`multi_stack`**: A heavy-duty statistical engine that generates 7-band GeoTIFFs containing Elevation, Weight, Count, Uncertainty, Source Uncertainty, and average X/Y locations for every pixel.
* **Provenance Tracking:** Automatically generate bitmask rasters that map exactly which datasets contributed to every pixel in your output.
* **Data Readers:**
    * **Native BAG Support:** A Bathymetric Attributed Grid reader that handles Variable Resolution (VR).
    * **COG Subsetting:** Windowed fetching for Cloud Optimized GeoTIFFs.
* **Modern Architecture:** Built on `rasterio`, `numpy`, and `fetchez`, dropping heavy legacy dependencies for a pure Python experience.
* **Declarative Projects:** Define complex, multi-sensor build pipelines in simple `yaml` files.