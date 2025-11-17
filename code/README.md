# MESA Tool (compiled distribution)

## Overview
MESA (Methods for Environmental Sensitivity Assessment) is delivered as a Windows desktop package composed of `mesa.exe` plus supporting helper executables. The bundle lets emergency response teams and planners score environmental sensitivity, run spatial analyses, and export maps without ever touching Python installs or GIS driver setups. Everything lives in one extracted folder, so simply unzip, launch, and start working.

## Why executables?
- **Self-contained delivery** - The archive ships with `input/`, `output/`, `qgis/`, `system_resources/`, logs, and configuration files so every helper can find the same datasets.
- **Consistent launch story** - Users double-click `mesa.exe` for the main dashboard or open any helper `.exe` in `tools/` for focused tasks.
- **Built-in GIS stack** - GDAL, GeoPandas, PyProj, PyArrow, and ttkbootstrap are already embedded, which explains the footprint (~1.7 GB on Windows 11, ~0.7 GB on Windows 10).
- **Automatic folder preparation** - On startup the executables ensure `input/geocode`, `input/lines`, `output`, and `qgis` exist so data ingestion is predictable.

## Getting started with the packaged build
1. Download the latest public release from Zenodo: https://zenodo.org/communities/mesatool/
2. Extract the ZIP to a writable location (for example `C:\MESA`). Keep the folder structure intact.
3. Launch `mesa.exe`. The home screen confirms which folders are detected, shows log activity, and provides buttons that open each helper `.exe`.
4. Work through the guided steps (configure parameters, import data, run processing, review outputs). Leave the folder open while working so cross-launching helpers is seamless.

## Helper executables at a glance
- **Hub (`mesa.exe`)** - Launchpad that opens every helper, reports log activity, and exposes documentation links. Use it to keep context in one window while you bounce between tasks.
- **Parameter setup (`tools\parametres_setup.exe`)** - Loads classification bins, valid ranges, and index weights from `config.ini`, enforces guard rails through dropdowns and fallback values, then writes the unified scoring matrices back to GeoParquet + JSON for all other executables to consume.
- **Data prep (`tools\data_import.exe`, `tools\geocode*.exe`, `tools\lines_*.exe`)** - Wizard-style workflows that check encodings, CRS, geometry validity, and attribute completeness before anything reaches processing. They harmonize column headers, flag duplicates, and log every change so schema drift is caught early.
- **Processing (`tools\data_process.exe`)** - The CPU/RAM-intensive engine that chunks intersections, spins up workers when safe, streams minimap snapshots, and persists both geometry and status metrics into `output/geoparquet`. Progress is logged continuously to `log.txt`, visible from the hub.
- **Reporting (`tools\data_report.exe`, `tools\atlas_*.exe`, `tools\maps_overview.exe`)** - Transform GeoParquet data into PDF summaries, tiled atlases arranged by administrative units, and Leaflet-ready overview dashboards while reusing metadata captured during import and parameterization.
- **Raster support (`tools\create_raster_tiles.exe`)** - Converts processed vectors into raster tiles or high-resolution imagery that align with the CRS/tiling scheme defined in `config.ini`, dropping outputs into `qgis/` for immediate use in GIS viewers.

All helpers can be opened directly from Windows Explorer, but most users launch them via `mesa.exe` so logs and context stay in one place.

## Typical executable workflow
1. **Drop inputs** into `input/geocode` and `input/lines`, then use the import helpers to check formats and metadata.
2. **Tune scoring** with `parametres_setup.exe`, saving vulnerability/importance settings to the shared GeoParquet store.
3. **Process data** through `data_process.exe` to compute intersections, indices, and status grids. This is the heaviest step and may run for hours on complex geographies.
4. **Generate outputs** with the atlas, report, or overview executables. PDF reports land in `output/` with timestamps.
5. **Distribute results** by copying the entire folder (including GeoParquet files) to partners or to another workstation; the executables auto-detect the structure.

## Performance considerations
The more detailed the geocode catalog, the more CPU, RAM, and SSD throughput the executables need. Expect high resource use when handling thousands of polygons or when exporting large atlases. Keeping the project on a fast SSD and ensuring ample free disk space helps `data_process.exe` maintain smooth minimap updates.

## Under the hood (for the curious)
Behind each `.exe` sits a Python 3.11 script (for example `mesa.py`, `data_process.py`, `parametres_setup.py`). PyInstaller wraps those scripts together with all third-party libraries so end users never have to install Python themselves. Power users who want to automate or customize the workflow can open the matching `.py` files in the `code/` folder and run them directly, but compiled executables remain the primary delivery format.

## Additional resources
- Method background: https://www.mesamethod.org/wiki/Main_Page
- Download portal: https://zenodo.org/communities/mesatool/
- Troubleshooting and automation tips: `code/instructions.md`
