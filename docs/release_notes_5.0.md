This marks the public release of **MESA 5** — the new major line of the MESA desktop tool for preparing, processing, and publishing spatial sensitivity analysis deliverables using the [MESA method](https://www.mesamethod.org/). MESA 5 fully replaces all earlier MESA releases.

The Zenodo record for this build was published on 2026-05-01 and includes compiled Windows 11 binaries together with pre-processed example data from Mafia Island, Tanzania.

### Why now
This release is timed to support the **MESA workshop in Nairobi (5 May 2026 onward)**. Workshop participants can install directly from the Zenodo download below.

### Highlights
- MESA 5 fully replaces the earlier MESA releases.
- New end-to-end workflow from project inputs to published outputs.
- New PySide6 desktop launcher with five tabs (Workflows, Status, Manage data, Config, About).
- Reporting is now Word-first (`.docx`) for easier downstream editing; advanced cartographic layout remains supported via the bundled QGIS project.
- Built-in backup and restore for iterative project work.
- Lightweight status monitoring in the desktop workflow.
- Includes pre-processed example data from Mafia Island, Tanzania.

### Where can I find it?
Compiled versions are available here:

- [Zenodo record](https://zenodo.org/records/19958541)
- [DOI landing page](https://doi.org/10.5281/zenodo.19958541)
- [MESA project wiki](https://github.com/ragnvald/mesa/wiki)

### Getting started
Download and unzip `MESA_50_2026_05_01.zip` from the Zenodo record into a writable folder (for example `C:\MESA`). Keep the folder structure intact and launch `mesa.exe`.

### What's new in MESA 5
- **New PySide6 desktop launcher** with simpler five-tab structure; Tune processing and Publish to GeoNode are now popups.
- **Unified processing runner** with numbered stages, per-stage checkboxes, and a Normal / Advanced toggle.
- **Auto-tuning of worker counts** at runtime, plus pre-flight RAM/swap check, per-pool memory panic watchdog, and process-lifetime sentinel.
- **Three-phase flatten** (huge / large / small) with configurable sliver cleanup.
- **Smarter asset import** with optional dissolve-adjacent-polygons and automatic point/line buffering.
- **Atlas helper** that surfaces existing state on open and supports per-atlas Delete.
- **Report-engine polish** — index maps with legends, scaled-down area maps, atlas-tile maps with full-grid context, page break per atlas tile heading.
- **GeoNode 5 publishing** via OAuth2.
- **Cleaner version banner** ("MESA 5") and populated About → Your system panel.

**Full Changelog**: https://github.com/ragnvald/mesa/compare/4.3...5.0
