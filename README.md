# MESA Developer README

This repository contains the source for MESA, a Windows desktop system for environmental sensitivity assessment.

For developers, the important point is that MESA is not one monolithic GIS application. It is a launcher plus a set of focused helper tools that all operate on the same project folder, the same `config.ini`, and the same shared outputs under `output/`.

If you are looking for end-user packaging notes, see [code/README.md](code/README.md). This root README is the developer-facing overview.

## What The System Does

MESA helps analysts move from raw geospatial inputs to decision-ready outputs.

At a high level, the system:
- imports and normalizes area assets, line assets, geocodes, and atlas polygons
- stores shared working data in GeoParquet under `output/geoparquet`
- lets users define scoring rules, weights, study areas, and comparison groups
- runs area, line, and analysis processing to calculate sensitivity-related outputs
- opens interactive viewers for QA and storytelling
- exports reports, atlas outputs, and optional raster tiles

The intended user is not a Python developer. The system is designed so analysts can work through a guided desktop workflow with buttons and helper windows instead of scripts and command lines.

## Runtime Model

The runtime architecture is straightforward:

1. `mesa.py` opens the main desktop launcher.
2. The launcher starts helper tools for specific tasks.
3. Each helper reads and writes the same project files on disk.
4. Later helpers consume what earlier helpers produced.

This means the project folder is the system boundary. MESA is effectively a desktop workflow engine wrapped around a file-based data store.

Important consequences:
- Path handling matters more than in a typical library project.
- Backwards compatibility between `.py` runs and packaged `.exe` runs matters.
- Shared file formats matter more than in-memory APIs.
- Startup time matters, especially for frozen executables.

## Core Workflow

The launcher organizes work into four broad stages:

1. Prepare data
   - asset import
   - geocode creation/import
   - line import/edit
   - atlas creation/edit
2. Configure
   - processing parameters and index weights
   - analysis groups and study areas
3. Process
   - area, line, and analysis processing
4. Results
   - interactive asset/result maps
   - comparison dashboards
   - report generation

That sequence is reflected directly in the launcher UI in [mesa.py](mesa.py).

## Major Components

### Launcher

- [mesa.py](mesa.py)

This is the hub application. It owns:
- the main Tk/ttkbootstrap UI
- workflow buttons and tabs
- helper launch orchestration
- status summaries and activity logging
- project-relative path resolution

The launcher should stay relatively light. Heavy GIS or processing work belongs in helpers, not here.

### Data Import And Editing Helpers

- [asset_manage.py](code/asset_manage.py)
- [geocode_manage.py](code/geocode_manage.py)
- [line_manage.py](code/line_manage.py)
- [atlas_manage.py](code/atlas_manage.py)

These tools prepare source data for later processing. They usually validate input structure, normalize attributes, and persist cleaned data into the shared project store.

### Configuration And Analysis Setup

- [processing_setup.py](code/processing_setup.py)
- [analysis_setup.py](code/analysis_setup.py)

These tools define the rules that later processing uses:
- vulnerability and sensitivity settings
- weight tables
- study areas
- analysis groups

### Processing Engine

- [processing_pipeline_run.py](code/processing_pipeline_run.py)
- [processing_internal.py](code/processing_internal.py)

This is the heavy part of the system.

`processing_pipeline_run.py` is the GUI-oriented entrypoint for launching processing. `processing_internal.py` contains the heavier execution logic, multiprocessing behavior, progress reporting, and output writing.

### Results And Presentation

- [asset_map_view.py](code/asset_map_view.py)
- [map_overview.py](code/map_overview.py)
- [analysis_present.py](code/analysis_present.py)
- [report_generate.py](code/report_generate.py)
- [tiles_create_raster.py](code/tiles_create_raster.py)

These tools sit on top of processed outputs. They are for QA, presentation, export, and publishing rather than raw ingestion.

## Shared Data Model

MESA uses the filesystem as its shared state.

The important locations are:

- [config.ini](config.ini)
  - global settings
  - theme
  - processing defaults
  - version label
- `input/`
  - user-provided source data
- `output/geoparquet/`
  - shared structured outputs used by helpers
- `output/`
  - reports, logs, derived data, and other runtime products
- `qgis/`
  - QGIS-facing outputs and project assets
- `system_resources/`
  - icons and packaged UI resources

The helpers are loosely coupled through files, not through direct Python imports of each other.

## Repository Layout

- [mesa.py](mesa.py): desktop launcher
- [code/](code): helper scripts and internal modules
- [devtools/](devtools): build and developer tooling
- [docs/](docs): diagrams, presentations, and supporting documentation
- [input/](input): expected user input folders
- [output/](output): runtime outputs
- [system_resources/](system_resources): packaged assets
- [instructions.md](instructions.md): repo-specific engineering rules
- [learning.md](learning.md): accumulated local lessons from past fixes

## Development Setup

The project is Windows-first.

Recommended setup:

1. Create the virtual environments with [setup_venvs_win311.bat](devtools/setup_venvs_win311.bat).
2. Use `.venv` for normal development.
3. Use `.venv_compile` for packaging work.
4. Run the launcher with `python mesa.py`.

Important requirements files:
- [requirements_all_win311.txt](requirements_all_win311.txt): primary Windows development environment
- [requirements_compile_win311.txt](requirements_compile_win311.txt): compile/build environment
- [requirements_all.txt](requirements_all.txt): broader legacy superset

## Build Model

Windows packaging is handled by:

- [compile_win_11.bat](devtools/compile_win_11.bat)
- [build_all.py](devtools/build_all.py)

Current packaging model:
- `mesa.exe` is built as the main launcher
- helper tools are built as separate executables
- runtime resources are copied into the final distribution folder
- build output is intended to work from an extracted folder, not from an installed service layout

The local developer convention is to treat builds as full builds unless there is a specific reason to do otherwise.

## Design Constraints That Matter

When editing this codebase, the main constraints are:

- The packaged `.exe` path is a first-class runtime, not an afterthought.
- Helpers must work both standalone and when launched through `mesa.py`.
- User data in `output/` is project state and should be treated as user-owned.
- Path resolution must stay project-relative.
- Heavy imports in launcher-style tools hurt startup badly in frozen builds.
- Tk-based helpers should run as separate GUI processes rather than being embedded inside the launcher process.

If a change looks clean in source mode but breaks the packaged layout, it is not a good change.

## How To Read The Codebase

A good reading order is:

1. [README.md](README.md)
2. [instructions.md](instructions.md)
3. [mesa.py](mesa.py)
4. [MESA_system_overview.md](MESA_system_overview.md)
5. the relevant helper under [code/](code)
6. [devtools/build_all.py](devtools/build_all.py) if the change touches packaging

That path gives you the user workflow first, then the launcher, then the helper internals.

## Practical Verification

There is no comprehensive automated test suite at the moment.

Typical verification is manual and task-specific:
- run `python mesa.py` for launcher changes
- run the affected helper directly for helper changes
- use `py_compile` for quick syntax validation
- inspect `log.txt` when a GUI action appears silent
- only run full PyInstaller builds when the change actually touches packaging behavior and the cost is justified

## Related Documents

- [instructions.md](instructions.md): coding and workflow rules
- [learning.md](learning.md): prior fixes and local knowledge
- [MESA_system_overview.md](MESA_system_overview.md): user-oriented workflow explanation
- [code/README.md](code/README.md): compiled-distribution overview

## Bottom Line

For developers, MESA is best understood as:

- a Tk desktop launcher
- a set of task-specific helper applications
- a shared GeoParquet/file-based project store
- a Windows packaging pipeline that matters as much as source execution

If you preserve those four properties, your changes are likely aligned with how the system is intended to work.
