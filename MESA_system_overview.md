# MESA System Overview

## 1. Purpose and Scope
The MESA (Methods for Environmental Sensitivity Assessment) desktop suite standardises how emergency planners and land-use teams ingest, score, and visualise environmental assets. It wraps Python 3.11 workflows, GIS libraries, and UI helpers inside PyInstaller-built executables so users on disconnected Windows workstations can complete full assessments without installing Python, GDAL, or QGIS separately. Every executable reads and writes to the shared project tree, giving teams a single source of truth for inputs (`input/`), intermediate GeoParquet stores (`output/geoparquet/`), cartographic resources (`qgis/`), and compiled reports (`output/reports/`).

The system’s core objectives are:
- **Consistent sensitivity scoring** – enforce `config.ini` parameters, scoring tables, and category colours when processing new regions.
- **Auditable GIS preparation** – automate schema cleaning, EPSG alignment, and geometry validation while keeping logs inside `code/log.txt` and helper-specific `tools/*.log` files.
- **Rapid situational products** – deliver atlases, PDF reports, Leaflet dashboards, and AI-styled map layers directly from processed GeoParquet data.

## 2. Distribution Layout and Runtime Contracts
A compiled delivery (for example `dist/mesa`) contains the following elements, all of which the 
executables rely on:

| Folder / File | Purpose |
| --- | --- |
| `mesa.exe` | Main launcher UI (`mesa.py`) with tabs for Statistics, Activities, Settings, About, Register. |
| `tools/` | One-file helper executables (e.g., `data_process.exe`, `map_assets.exe`). PyInstaller embeds the full GIS stack here. |
| `input/` | Raw datasets organised by theme (`input/asset`, `input/geocode`, `input/lines`, etc.). Helpers validate and copy from here. |
| `output/` | All generated deliverables: `geoparquet/` tables, `mbtiles/`, `reports/`, `tmp/`, `mesa.gpkg`. |
| `config.ini` | Global configuration for EPSG codes, OpenAI styling keys, statistic counters, theming. Shared by every helper. |
| `qgis/` | Packaged `.qgz` project and symbology resources used by GIS practitioners. |
| `system_resources/` | Icons, HTML templates, and static assets referenced by Tk/pywebview windows. |
| `secrets/` | Optional encrypted API keys (e.g., `secrets/openai.key`) resolved via `_maybe_deobfuscate_secret`. |

Runtime expectations enforced inside `mesa.py` and helper modules:
- **Working directory** – helpers set `APP_DIR` via `base_dir()` to ensure relative paths land in the same tree whether they are executed as `.py` scripts, packaged helpers, or child processes launched from `mesa.exe`.
- **Environment variables** – the hub injects `MESA_BASE_DIR` and copies the user’s locale-specific settings so PyInstaller builds behave like local Python runs.
- **Logging** – `log_event()` in each helper appends ISO timestamps to `code/log.txt` or the helper’s sibling log, enabling offline troubleshooting.

## 3. Data Lifecycle at a Glance
The Activity buttons in the launcher follow a deliberate progression. Each stage writes deterministic artefacts so downstream tools work without reconfiguration:

1. **Import (data_import.py / `Import` button)**
   - Reads shapefiles, GeoPackage layers, or CSV/Excel point tables from `input/`.
   - Harmonises schemas (field names, casing), enforces EPSG from `config.ini`, and persists to `output/geoparquet/tbl_asset_object.parquet` plus companion lookup tables.
2. **Grids (geocodes_create.py)**
   - Builds tiling schemes (square grids, hexagons, administrative references) that underpin zonal statistics. Outputs `tbl_geocode_*.parquet` and boundary layers.
3. **Define map tiles (atlas_create.py)**
   - Generates atlas footprint polygons with scale metadata for later editing/export. Results saved into GeoParquet and the QGIS project.
4. **Processing setup (parametres_setup.py)**
   - Interactive editor for sensitivity indices, weightings, and styling defaults. Writes cleaned tables plus JSON snapshots so processing and reporting stay aligned.
5. **Process areas (data_process.py)**
   - Heavy-weight intersection and scoring engine: dissolves assets per grid, calculates composite indices, renders minimap thumbnails, and writes results to `output/geoparquet` and `mesa.gpkg`.
6. **Process lines (lines_process.py)**
   - Specialised pipeline for linear infrastructure (pipelines, rivers). Buffers, segments, and scores per configuration, saving to GeoParquet and derived shapefiles.
7. **Exploration & reporting (map_assets.py, maps_overview.py, data_analysis*, data_report.py)**
   - Visualise processed data, adjust styling, configure study areas, and export PDF or web-ready artefacts.

## 4. Core Services and Technologies
- **UI Framework** – ttkbootstrap wraps Tkinter widgets for the launcher, while interactive tools (`map_assets.py`, `maps_overview.py`, `data_analysis_setup.py`) embed Leaflet maps through `pywebview` (WebView2 backend).
- **Geoprocessing stack** – GeoPandas, Shapely 2.x, PyProj, Fiona, and GDAL handle CRS transformations and geometry operations. Helper builds pass `--collect-all` for these libraries to keep the EXE self-contained.
- **Data interchange** – GeoParquet is the canonical store for intermediate layers; `mesa.gpkg` mirrors key outputs for GIS compatibility. CSV/JSON exports accompany PDF reports where needed.
- **AI Styling** – `map_assets.py` can call OpenAI Chat Completions (`DEFAULT_OPENAI_MODEL = "gpt-4o-mini"`) to propose coordinated colour palettes. Keys may be stored inline, in env vars, or obfuscated files.
- **Telemetry (optional)** – If `id_uuid_ok` is true and network access is available, the hub logs anonymised usage statistics to the configured InfluxDB bucket using `InfluxDBClient`.

## 5. Activities Tab Buttons (Primary Workflows)
The Activities tab (implemented around line 1000 of `mesa.py`) presents the operational sequence. Each button launches a helper script/executable and has specific inputs/outputs:

| Button | Underlying Script / EXE | Description & Key Details |
| --- | --- | --- |
| **Import** | `data_import.py` / `tools/data_import.exe` | Starting point for new projects. Validates raw asset layers, aligns CRS with `workingprojection_epsg`, and writes `tbl_asset_object.parquet`, `tbl_asset_group.parquet`, plus supportive lookup tables. Also increments `mesa_stat_import_assets` in `config.ini`. |
| **Grids** | `geocodes_create.py` | Generates spatial reference grids (hex, square, admin). Accepts existing shapefiles or algorithmically creates new ones. Ensures grid IDs and friendly names are stored for later selection. |
| **Define map tiles** | `atlas_create.py` | Automates creation of atlas tile polygons sized for printable layouts. Stores metadata consumed later by `atlas_edit.py` and `data_report.py`. |
| **Processing setup** | `parametres_setup.py` (fallback `params_edit`) | Wizard for sensitivity categories, weights, and style presets. Edits propagate to GeoParquet and `config.ini`, guaranteeing the processing engine uses the same assumptions as reporting. |
| **Process areas** | `data_process.py` | CPU-intensive stage. Streams intersections between assets and grids, calculates sensitivity/importance indices, exports aggregated GeoParquet tables, PNG thumbnails, and updates `mesa_stat_process`. |
| **Process lines** | `lines_process.py` | Similar to area processing but tailored to linear datasets (e.g., pipelines, rivers). Applies buffers and per-segment scoring, writing `tbl_line_segments.parquet`. |
| **Asset maps** | `map_assets.py` (`tools/map_assets.exe`) | Pywebview/Leaflet viewer for processed asset groups. Features include AI-based styling, layer hierarchy editing, drag/drop folders, and PNG export via `html2canvas`. Reads `tbl_asset_object.parquet` + `tbl_asset_group.parquet`. |
| **Analysis maps** | `maps_overview.py` | Interactive overview that combines base layers with processed assets for quick QA. Useful once processing completes to confirm coverage. |
| **Analysis setup** | `data_analysis_setup.py` | Allows users to digitise or import study polygons, tag them with metadata, and store them in `output/geoparquet/tbl_analysis_polygons.parquet`. Supports later comparison dashboards. |
| **Analysis results** | `data_analysis_presentation.py` | Consumes study polygons and processed metrics to produce comparative dashboards, charts, and PDF exports summarising multiple scenarios. |
| **Export reports** | `data_report.py` | Assembles narrative PDF reports, atlases, and tabular attachments using the processed GeoParquet store plus templates in `system_resources/`. Outputs land in `output/reports/` with timestamped filenames. |

### Launch Mechanics
- `mesa.py` resolves both the `.py` script and `.exe` path via `get_script_paths()`. When the system runs from source, it launches Python scripts. When frozen (`sys.frozen` is true), it opens the compiled helper.
- All subprocesses inherit the same `PROJECT_BASE` so relative paths work regardless of start menu shortcuts or double-click launches.

## 6. Settings Tab Buttons (Maintenance & Metadata)
The Settings tab groups tools that edit reference data rather than running the main pipeline.

| Button | Script / EXE | Focus |
| --- | --- | --- |
| **Edit config** | `edit_config.py` | ttkbootstrap GUI for `config.ini`. Handles identity fields (`id_uuid`, `id_name`), logging opt-ins, EPSG values, colour palettes, and API key hints. Writes changes back without reordering sections. |
| **Edit assets** | `assetgroup_edit.py` | Enables human-friendly titles, descriptions, and styling records for each imported asset group. Updates propagate to `tbl_asset_group.parquet`, which feeds both reports and `map_assets`. |
| **Edit geocodes** | `geocodegroup_edit.py` | Adds labels and metadata to grid cells or administrative polygons so atlases and reports can reference them clearly. |
| **Edit lines** | `lines_admin.py` | Provides per-line metadata editing plus buffer/segmentation defaults. Accepts an `--original_working_directory` parameter to ensure edits land in the correct project tree. |
| **Edit map tiles** | `atlas_edit.py` | Visual editor for atlas polygons generated earlier. Users can merge/split tiles, adjust names, and regenerate ordering before running exports. |

These helpers do not typically require the full processing stack but they do rely on up-to-date GeoParquet tables. They maintain bidirectional consistency: editing a title immediately affects both UI labels and downstream reports.

## 7. Supporting Components
- **Statistics tab** – Summarises usage counters stored in `config.ini` (e.g., `mesa_stat_process`, `mesa_stat_setup`). `update_stats()` populates a ttk LabelFrame with the latest run markers and guidance text.
- **About & Register tabs** – Provide attribution, version info, and opt-in telemetry toggles. The Register tab captures `id_name` and `id_email`; when allowed, `store_userinfo_online()` posts them to InfluxDB.
- **Logging infrastructure** – `log_to_logfile()` (not shown above) writes to `code/log.txt`, while each helper has its own `log.txt` in the same directory as the script. These logs aid offline debugging when a helper fails to launch.
- **Build pipeline** – `code/build_all.py` orchestrates PyInstaller runs: helper executables are built with `--onefile` and the full GIS stack, whereas the hub (`mesa.py`) uses `--onedir` before being flattened into `dist/mesa`. Runtime data directories (`qgis/`, `docs/`, `input/`, `output/`, `system_resources/`, `secrets/`) are copied beside `mesa.exe`.

## 8. End-to-End Usage Blueprint
1. **Initialisation** – Extract the distribution, ensure the folder is writable, and run `mesa.exe`. Confirm the Statistics tab shows available folders and zeroed counters.
2. **Registration (optional)** – On first launch, provide name/email and consent toggles so the system can generate a persistent `id_uuid` and update remote stats when online.
3. **Asset Preparation** – Click **Import** to populate GeoParquet tables. If the region needs fresh grids, follow with **Grids** and **Define map tiles**.
4. **Configuration** – Use **Processing setup** and the Settings tab tools (Edit config/assets/geocodes/lines) to ensure metadata is complete and stylistically aligned.
5. **Processing** – Run **Process areas** and (if applicable) **Process lines**. Monitor `code/log.txt` or the helper consoles for progress.
6. **QA & Adjustments** – Launch **Asset maps** and **Analysis maps** to visually inspect results. Apply AI styling or manual tweaks, and edit map tiles if necessary.
7. **Analytical Products** – Define study polygons via **Analysis setup**, evaluate them with **Analysis results**, and finally build packaged outputs with **Export reports**.
8. **Iteration** – Because every helper reads/writes the same GeoParquet store, repeating any step (e.g., re-importing revised assets) automatically feeds later stages with updated data.

## 9. Key Takeaways for Operators
- Maintain the prescribed folder layout; moving `input/` or `output/` breaks the helper assumptions baked into `base_dir()`.
- Always rerun **Processing setup** when `config.ini` changes weights or colour ranges; the processing engine does not guess new parameters.
- Prefer launching helpers from within `mesa.exe` to keep logs centralised and to make sure `--original_working_directory` is passed correctly.
- When troubleshooting compiled helpers, you can run the corresponding `.py` script from `code/` using the same project folder; both code paths share logic and relative imports.

This document should help new contributors, power users, and deployment partners understand not only what the MESA system does, but also how each launcher button maps onto a concrete subsystem and shared data contract.
