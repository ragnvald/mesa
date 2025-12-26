# MESA Tool (compiled distribution)

## Overview
The MESA (Methods for Environmental Sensitivity Assessment) tool is a Windows desktop package built for assessing and mapping environmental sensitivity, particularly around marine pollution incidents and land-use planning. It bundles the full Python/Tkinter application plus helper executables (`mesa.exe`, `parametres_setup.exe`, `data_process.exe`, etc.) so teams can evaluate coastal assets, ecosystems, and wildlife without installing Python or GIS drivers. Everything needed to score vulnerability, process geospatial datasets, and export reports lives inside one extracted folder.

## Highlights
- Environmental sensitivity assessment: systematic scoring of assets against importance and sensitivity indices.
- Mapping capability: minimaps, atlases, and Word reports (`.docx`) support sharing and review.
- User-friendly GUI: ttkbootstrap-styled interfaces keep workflows accessible to non-technical staff.
- Compiled executable delivery: PyInstaller produces ready-to-run `.exe` files for Windows 11/10 (~1.7 GB / 0.7 GB).
- Customizable framework: configuration and outputs are stored alongside the app so projects can be adapted to new regions.

## Why executables?
- **Self-contained delivery** - The archive ships with `input/`, `output/`, `qgis/`, `system_resources/`, logs, and `config.ini` so every helper finds the same data.
- **Consistent launch story** - Double-click `mesa.exe` for the dashboard or open any helper `.exe` in `tools/` for a focused workflow.
- **Built-in GIS stack** - GDAL, GeoPandas, PyProj, PyArrow, and ttkbootstrap are already embedded, which explains the footprint (~1.7 GB on Windows 11, ~0.7 GB on Windows 10).
- **Automatic folder preparation** - At startup the executables ensure `input/geocode`, `input/lines`, `output`, and `qgis` exist so data ingestion is predictable.

## Getting started with the packaged build
1. Download the latest public release from Zenodo: https://zenodo.org/communities/mesatool/
2. Extract the ZIP to a writable location (for example `C:\MESA`). Keep the folder structure intact.
3. Launch `mesa.exe`. The home screen confirms which folders are detected, shows log activity, and provides buttons that open each helper `.exe`.
4. Work through the guided steps (configure parameters, import data, run processing, review outputs). Leave the folder open while working so cross-launching helpers is seamless.

## Helper executables at a glance
- **Hub (`mesa.exe`)** - Launchpad that opens every helper, reports log activity, and links to documentation so context stays in one place.
- **Parameter setup (`tools\parametres_setup.exe`)** - Loads classification bins, valid ranges, and index weights from `config.ini`, enforces guard rails through dropdowns and fallback values, then writes the unified scoring matrices back to GeoParquet + JSON for other executables.
- **Data prep (`tools\data_import.exe`, `tools\geocode*.exe`, `tools\lines_*.exe`)** - Wizard-style workflows that check encodings, CRS, geometry validity, and attribute completeness before anything reaches processing. They harmonize column headers, flag duplicates, and log every change.
- **Processing (`tools\data_process.exe`)** - The CPU/RAM-intensive engine that chunks intersections, spins up workers when safe, streams minimap snapshots, and persists geometry plus status metrics into `output/geoparquet`. Progress is logged continuously to `log.txt`.
- **Reporting (`tools\data_report.exe`, `tools\atlas_*.exe`, `tools\maps_overview.exe`)** - Turn GeoParquet data into Word reports (`.docx`), atlas artifacts, and Leaflet-ready overview dashboards while reusing metadata captured during import and parameterization.
- **Raster support (`tools\create_raster_tiles.exe`)** - Converts processed vectors into raster tiles or high-resolution imagery aligned with the CRS/tiling scheme defined in `config.ini`, writing outputs into `qgis/` for immediate use in GIS viewers.

All helpers can be opened directly from Windows Explorer, but most users launch them via `mesa.exe` so logs and context stay in one place.

## Typical executable workflow
1. **Drop inputs** into `input/geocode` and `input/lines`, then use the import helpers to check formats and metadata.
2. **Tune scoring** with `parametres_setup.exe`, saving vulnerability/importance settings to the shared GeoParquet store.
3. **Process data** through `data_process.exe` to compute intersections, indices, and status grids. This is the heaviest step and may run for hours on complex geographies.
4. **Generate outputs** with the atlas, report, or overview executables. Word reports land in `output/reports/` with timestamps.
5. **Distribute results** by copying the entire folder (including GeoParquet files) to partners or to another workstation; the executables auto-detect the structure.

## Processing capacity requirements
Throughput depends heavily on the input data being handled. Dense geocode catalogs with many objects increase CPU, RAM, and disk requirements during processing (`data_process.exe`) and when exporting atlases or raster tiles. Using SSD storage and keeping ample free space helps the compiled tools maintain smooth minimap updates.

## Under the hood (for the curious)
Behind each `.exe` sits a Python 3.11 script (for example `mesa.py`, `data_process.py`, `parametres_setup.py`). PyInstaller wraps those scripts with all third-party libraries so end users never have to install Python themselves. Power users who want to automate or customize the workflow can open the matching `.py` files and run them directly, but compiled executables remain the primary delivery format.

## Additional resources
- Download portal: https://zenodo.org/communities/mesatool/
