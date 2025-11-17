# MESA Tool (development workspace)

## Overview
The MESA (Methods for Environmental Sensitivity Assessment) tool is a Tkinter-based desktop application for evaluating and mapping environmental sensitivity, with a focus on marine pollution preparedness and land use planning. It bundles geospatial processing utilities (GeoPandas, Shapely, PyProj, GDAL) with a user-friendly GUI so emergency responders and planners can curate asset catalogues, run batch analyses, and export maps/reports without writing code. The same source drives the compiled `mesa.exe` that is distributed to Windows users via Zenodo; the most recent build footprint is roughly 1.7 GB on Windows 11 and 0.7 GB on Windows 10 due to the bundled GIS stack.

## Highlights
- **Environmental sensitivity assessment** - structured scoring of coastal assets, ecosystems, and wildlife against importance/sensitivity indices.
- **Integrated mapping** - Leaflet/pywebview minimaps, grid overlays, and QGIS templates offer quick visual validation of GeoParquet outputs.
- **End-to-end workflow** - helper tools cover data import, parameter setup, raster/atlas generation, ad-hoc area analysis, and PDF reporting.
- **User-friendly GUI** - ttkbootstrap-styled interfaces guide non-technical users through each processing stage and surface inline help links.
- **Compiled executable** - PyInstaller builds (see `code/build_all.py`) produce a distributable `mesa.exe` plus helper executables for field teams.
- **Customizable framework** - configuration is stored in `config.ini`, and helper tools persist user edits in `output/geoparquet` so datasets travel with the project directory.

## Repository layout
| Path | Purpose |
| --- | --- |
| `/mesa.py` | Main GUI that orchestrates helper tools, logging, and telemetry. |
| `/code` | Development sources, helper utilities, packaged resources, and this README. |
| `/code/input` | Sample input templates (`geocode`, `lines`, etc.). Users drop raw data here. |
| `/code/output` | Default location for GeoParquet layers, exported PDFs, and generated atlases. |
| `/code/tools` | Standalone helper scripts (`data_process.py`, `parametres_setup.py`, `atlas_create.py`, ...) that can run independently or under `mesa.py`. |
| `/code/system_resources` | Static assets (icons, HTML templates, docs) bundled into the executable. |
| `/config.ini` | Central knobs (paths, CRS, classification bins). Copied next to `mesa.exe` at build time. |

At runtime, `mesa.py` detects its base directory automatically (preferring `MESA_BASE_DIR`, the folder that holds `config.ini`, or the directory that contains the executable) and ensures standard sub-folders exist (`input/geocode`, `input/lines`, `output`, `qgis`).

## Getting started
### Using the packaged build
1. Download the latest release from Zenodo: https://zenodo.org/communities/mesatool/
2. Extract the archive and launch `mesa.exe`.
3. Follow the on-screen prompts to configure parameters, import data, and start processing.
4. Keep the extracted folder intact: helper executables expect `config.ini`, `input`, and `output` to live next to `mesa.exe`.

### Running from source (developers/power users)
1. Create and activate a Python 3.11 virtual environment.
2. Install dependencies: `pip install -r requirements.txt` (GDAL/OGR must be installed first on Windows).
3. Configure `config.ini` at the repository root (copy the provided template if needed).
4. Launch the main UI from either folder: `python mesa.py` (root) or `python code/mesa.py`.
5. Optional: run helper utilities directly (e.g., `python code/data_process.py --help`) for batch pipelines or scripted workflows.

## Data & tool workflow
- **Parameter setup (`parametres_setup.py`)** - define vulnerability/importance bins, valid input ranges, index weights, and default fallbacks; values are saved to GeoParquet/JSON under `output/geoparquet`.
- **Data import (`data_import.py`, `geocodes_create.py`, `lines_process.py`, `assetgroup_edit.py`, `geocodegroup_edit.py`)** - curate and QA source datasets before they enter the processing pipeline.
- **Processing (`data_process.py`)** - performs the CPU/memory-aware spatial intersections, multi-threaded minimap updates, and writes consolidated GeoParquet layers used by every other tool.
- **Analysis & reporting (`data_report.py`, `data_analysis_setup.py`, `data_analysis_presentation.py`, `maps_overview.py`, `atlas_create.py`, `atlas_edit.py`)** - create printable atlases, PDF summaries (`output/MESA_area_analysis_report_YYYY_MM_DD.pdf`), and overview maps for planners.
- **Runtime helpers** - `create_raster_tiles.py`, `lines_admin.py`, and `system_resources/` provide auxiliary content for QGIS or other downstream viewers.

All helpers can be run as Python scripts, compiled executables (`tools/*.exe` when distributed), or "slave" processes controlled by `mesa.exe`. Data is exchanged through the standard folder layout and shared GeoParquet files so that edits made in one mode are available everywhere.

## Building executables
1. Activate the virtual environment used for development.
2. Ensure PyInstaller and hooks are installed: `pip install -U pyinstaller _pyinstaller_hooks_contrib`.
3. From the repo root run `python code/build_all.py`.
4. The script cleans previous artifacts, builds every helper as a one-file executable, packages the main GUI as an onedir build, flattens it into `dist/mesa`, and copies `input`, `output`, `docs`, `qgis`, and `system_resources` alongside `mesa.exe`.

Expect long build times (tens of minutes) because GDAL, GeoPandas, and PyProj are bundled. Final artifacts are large (~1.7 GB/0.7 GB for Win 11/Win 10) but self-contained.

## Processing capacity requirements
Throughput depends heavily on the volume and complexity of the input geocodes and linework. Dense coastal datasets with many polygons and overlapping buffers increase CPU, RAM, and disk consumption during `data_process.py` runs. Use SSD storage, allocate ample RAM, and allow additional time when exporting atlases or raster tiles.

## Further reading
- Method background: https://www.mesamethod.org/wiki/Main_Page
- Support site & binary downloads: https://zenodo.org/communities/mesatool/
- For troubleshooting or automation tips, inspect `code/instructions.md`.
