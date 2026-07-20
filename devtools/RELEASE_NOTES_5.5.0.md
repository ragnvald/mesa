A performance, memory and platform release. The classification engine is updated and woven into the rest of the tool, processing is measurably ~3× faster, and the whole stack moves to Python 3.14.

## What's new

### The classification engine is updated and now reaches the whole workflow

Classification arrived in 5.2 as a standalone engine you ran by hand after processing — its results stayed in that one window. In 5.5.0 the engine is updated and connected to everything around it:

- **Reports.** Classification is now a selectable report section, with per-type area charts, a classification overview, a segmentation legend strip, and interpretive prose generated from the actual numbers. (Before 5.5.0 the section could never appear at all — the toggle never reached the report generator.)
- **Maps.** Types and Certainty render as rasters, with hover-identify reading straight from `tbl_seg_mv`, no-data shown as grey rather than white holes, and a contrast-stretched certainty ramp.
- **Processing.** It runs as a stage inside Process, between Data and Tiles — no second manual pass. Setup moved to Configure.
- **QGIS.** Both raster and vector exports, with a stable `_segmv_latest` alias so QGIS always tracks the newest run.
- **Scope.** One run can now classify several geocode layers at once, or all of them.

AI-generated descriptions are optional and off by default; without them the engine falls back to deterministic naming.

### A configurable AI connection

Config gains an **AI connection** panel (OpenAI or Ollama). The token is stored in `secrets/ai_connection.parquet`, survives Clear generated data, and stays outside the default backup set. It powers classification descriptions, report prose and title-aware map styling — each with a deterministic fallback when no AI is configured. AI is off by default; nothing leaves your machine unless you configure and enable it.

### Processing is ~3× faster

Intersect — the core of processing — no longer ships the full asset layer to every worker; the parent sends a per-chunk asset subset instead. Measured on a full pipeline run over the `basic_mosaic` geocode layer (3.5M assets, Python 3.14): per-worker RAM fell from 5.76 GB to ~0.27 GB, which lifted the RAM-derived worker cap from 3 to 10 and cut intersect wall-clock from **9.6 hours to 3.02 hours (3.18×)**. Output is byte-identical to the old run — 1,387 parts / 91,083,233 rows, zero errors across all 1,992 chunks.

### basic_mosaic: graceful limits instead of out-of-memory crashes

A new pre-flight memory gate estimates peak RAM and skips `basic_mosaic` with a clear message rather than dying mid-run. It scales with the host, so high-RAM machines are never blocked. Indicative ceilings: ~1.3M assets on 16 GB, ~3M on 32 GB, ~6M on 64 GB, ~11M on 128 GB. Beyond that, use H3 or QDGC grids. We also diagnosed `basic_mosaic`'s dominant cost as process-spawn overhead rather than geometry computation (~87% of a measured 9h56m reference run on 3.5M assets) and removed the per-pair worker respawn; post-change wall-clock measurement is pending.

### Loading data is no longer just "restore backup"

**Manage data → Restore data** (renamed from *Restore backup*) loads any MESA project ZIP — one of your own backups, a demo-data package, or a project shared by a colleague. Demo-data packages now carry a short description of themselves (data sources, the credits they require, and how the sample is built); after you load one, the completion dialog offers an **Open description** button. An ordinary project has no such file, so nothing extra appears for a normal restore.

### Tile construction is unchanged

Building the local map tiles still costs what it always has. It is not a bottleneck we have attacked in this release, and it is retained deliberately: the tiles are what make the results explorable locally, without a server or a network connection.

### Fixes that prevented data loss

- Importing geocodes silently deleted every other geocode group. One import removed 13.3M H3 objects and 83k QDGC objects. Imports now merge, and import is a separate tab from manage so delete no longer sits beside it.
- Restoring data deleted `config.ini` even when the ZIP had none to put back, leaving MESA unable to start.
- Restoring data now keeps a `config.ini.bak` before overwriting your config, so an experimental restore is undoable.
- A lines- or analysis-only run wiped every map tile.
- Two "Process all" windows could run against one project, racing outputs.
- `basic_mosaic` could appear to hang indefinitely.

Plus fixes to silently-ignored settings (inline `#` comments made pre-flight values unparseable), report sections that were always skipped, map ramps dominated by a single outlier cell, and certainty maps drawn with white holes.

### Smaller refinements

- Buttons now render at a consistent size on Windows and macOS (sized in pixels rather than points).
- The Maps **Segmentation** zones table opens sorted by sensitivity (highest first), and its colour legend follows the table sort.

### Python 3.11 → 3.14

Windows and macOS both target CPython 3.14, with a refreshed, security-current scientific stack: numpy 2.5, pandas 3.0, shapely 2.1, pyogrio 0.12, PySide6 6.11, scikit-learn 1.9. `fiona` is gone — it has no cp314 wheel; `pyogrio` bundles GDAL and replaces it. If you maintain your own environment: `pyogrio ≥ 0.8` and `openpyxl ≥ 3.1.5` are hard requirements. This is invisible in the packaged build.

## Where can I find it?

The compiled Windows build is published on Zenodo:

- **Zenodo record:** https://zenodo.org/records/21455341
- **DOI:** https://doi.org/10.5281/zenodo.21455341
- **Project wiki:** https://github.com/ragnvald/mesa/wiki

### Getting started

Download and unzip the package from the Zenodo record into a writable folder, keep the folder structure intact, and launch `mesa.exe`.

**Full changelog:** https://github.com/ragnvald/mesa/compare/5.2...5.5.0
