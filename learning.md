# Learning log: UI capture and tooling operations (MESA)

Purpose: keep practical, reusable knowledge from UI screenshot work, build/package operations, and launcher behavior.
Historical notes are kept only when they still explain why a later decision exists; superseded constraints should be marked clearly so old guidance does not override current practice.

## Quick start

Use these first:

- Single active window capture:
  - `python devtools/screenshot_active_window.py --out output/screenshots --name mesa_active`
- Full wiki refresh batch:
  - `python devtools/capture_ui_active_batch.py`

Expected output target for batch runs:

- `../mesa.wiki/images/ui_*.png`

## Scope

This log currently covers:

- Desktop UI capture for `mesa.py` (`Workflows`, `Status`, `Config`, `Tune processing`, `Manage MESA data`, `About`)
- Helper UI capture for tools under `code/`
- Capture tooling placement and path conventions under `devtools/`

## Current canonical script locations

These moved from `code/` to `devtools/` and should be referenced by new docs/scripts:

- `devtools/build_all.py`
- `devtools/compile_win_11.bat`
- `devtools/capture_ui_active_batch.py`
- `devtools/screenshot_active_window.py`

Current launcher entrypoint:

- `mesa.py`

If you see old paths like `code/compile_win_11.bat` or `code/capture_ui_active_batch.py`, update them.
If you see old launcher references like `mesa_qt.py`, treat them as stale historical names and update them to `mesa.py`.

## Stable capture defaults

- Window detection timeout: `300s`
- Render wait:
  - Desktop tabs: `40-45s`
  - Map/webview-heavy helpers: `45-50s`
  - Other helpers: `30-35s`
- Wrapper command timeout: `0` (no forced timeout)

## Recommended workflow (repeatable)

1. Close existing GUI windows or run pre-flight cleanup.
2. Capture desktop tabs first, then helpers.
3. Validate framing and loaded content visually.
4. Re-run only failed/incorrect captures.
5. Update wiki narrative text after images are final.

Desktop tab file mapping:

- `Workflows` -> `ui_mesa_desktop.png`
- `Status` -> `ui_mesa_desktop_tab2.png`
- `Config` -> `ui_mesa_desktop_tab3.png`
- `Tune processing` -> `ui_mesa_desktop_tab4.png`
- `Manage MESA data` -> `ui_mesa_desktop_tab5.png`
- `About` -> `ui_mesa_desktop_tab6.png`

## Known pitfalls and fixes

- Problem: request/tool timeout (408 or long-running call failure)
  - Fix: run in smaller batches, increase wait times, avoid giant one-shot runs

- Problem: editor/background captured instead of app window
  - Fix: use active-window or process-tree-based window targeting

- Problem: right/bottom clipping or offset on Windows
  - Fix: enforce DPI-aware bounds + ensure window is fully inside the usable desktop work area, not just inside the full monitor bounds

- Problem: some `pywebview` helpers capture with the Windows taskbar or oversized empty margins
  - Fix: compare `DwmGetWindowAttribute(...EXTENDED_FRAME_BOUNDS...)` against `GetWindowRect()` and fall back to `GetWindowRect()` when the DWM rectangle is implausibly larger than the real window

- Problem: desktop tab screenshots end up identical
  - Fix: explicitly switch tab and wait before each capture

- Problem: new capture launch blocked by already running helper window or locked packaged executable
  - Fix: pre-flight cleanup of stale `mesa.py`, `mesa.exe`, and helper processes before starting captures or rebuilds

## Geocode UI lessons (recent)

- Consolidating geocode workflows into `code/geocode_manage.py` simplifies user pathing and screenshot automation.
- Edit flow with `Previous/Next/Save/Save & Next` is easier to document and test than wide row-based edit grids.
- When helper UI is restructured, keep screenshot output file names stable where wiki references depend on them.

## Build tooling lessons (recent)

- `devtools/compile_win_11.bat` is now the canonical full-build entrypoint.
- `devtools/build_all.py` must resolve `code/` relative to project root, not relative to script folder assumptions from old layout.
- Keep compile verification lightweight (`py_compile`, `--help`) unless explicit user approval is given for full builds.
- Do not start larger runs (full compile builds, long processing jobs, broad capture batches, or similar multi-minute operations) without checking with the user first, even if they are technically possible and even if a previous turn included a similar request.
- Stop any running `dist\mesa\mesa.exe` before rebuilding, or the flatten/copy stage can fail on a locked output file.

## Screenshot references used in docs

Desktop tabs:

![Workflows](../mesa.wiki/images/ui_mesa_desktop.png)

![Status](../mesa.wiki/images/ui_mesa_desktop_tab2.png)

![Config](../mesa.wiki/images/ui_mesa_desktop_tab3.png)

![Tune processing](../mesa.wiki/images/ui_mesa_desktop_tab4.png)

![Manage MESA data](../mesa.wiki/images/ui_mesa_desktop_tab5.png)

![About](../mesa.wiki/images/ui_mesa_desktop_tab6.png)

Selected helper examples:

![Processing runner](../mesa.wiki/images/ui_processing_pipeline_run.png)

![Maps overview](../mesa.wiki/images/ui_map_overview.png)

![Asset map view](../mesa.wiki/images/ui_asset_map_view.png)

![Report generator](../mesa.wiki/images/ui_report_generate.png)

## Maintenance rule

When a problem is solved, add a short entry here with:

- What failed
- Root cause
- Practical fix
- Any changed canonical path/command

## Mosaic log rendering lesson (2026-02-15)

- What failed:
  - Basic mosaic log pane in `geocode_manage.py` stayed visually empty even while `log.txt` showed correct live progress.
- Root cause:
  - GUI-side rendering/dispatch proved unreliable on some Windows/theme/thread combinations, while file logging stayed reliable.
- Practical fix:
  - Use a dedicated UI-thread tail loop that reads only new bytes from `log.txt` after pressing **Build mosaic**.
  - Track file offset at run start and append only current-run lines until `Step [Mosaic] COMPLETED|FAILED`.
- Why this helps:
  - Decouples user-visible progress from worker-thread widget updates and avoids stale full-log noise.

## Mosaic tab rendering result (final, 2026-02-15)

- What was observed:
  - `log.txt` and a popup log window both showed live Mosaic output, while the Mosaic tab log area stayed blank.
- Confirmed diagnosis:
  - Logging pipeline was healthy; issue was specific to tab-embedded text widget rendering in this layout/theme combination.
- Final stable solution:
  - Keep Mosaic log source as `log.txt` tail from run-start offset.
  - Render Mosaic tab log lines in a `ttk.Treeview` list (not Text/ScrolledText/Listbox in this pane).
  - Keep normal text log panes for `H3 codes` and `Import geocodes`.
  - Remove log pane from `Edit geocodes` to reduce UI complexity/noise.
- Practical rule for future UI work:
  - If logs write correctly to file but tab text area appears empty, verify with a temporary popup; if popup works and tab fails, switch the tab log renderer to a simpler non-text widget (`Treeview`) rather than changing backend logging again.

## Processing auto-tune tab lesson (2026-02-15)

- What was added:
  - New `Tune processing` tab in `mesa.py` with a one-click button that tunes selected processing keys in `config.ini` based on detected CPU and RAM.
- Implementation choice:
  - Keep user comments/order in `config.ini` by updating only key lines under `[DEFAULT]` (line-level replacement), instead of rewriting the full INI with `ConfigParser.write()`.
- User-facing behavior:
  - After tuning, show a plain-text explanation listing detected hardware, rationale, and old->new values for each updated key.

- Rollback extension:
  - Save a lightweight backup JSON at `output/processing_tuning_backup.json` containing only tuned keys and their pre-tune values.
  - `Restore previous tuning` applies those values back to `config.ini` using the same comment-preserving key update logic.

## Processing tune UX lesson (2026-02-15)

- What changed:
  - The `Tune processing` tab now uses a two-step flow: `Evaluate` first, then `Commit changes`.
- Why:
  - Users can review current vs advised values before writing anything to `config.ini`.
- Practical UI pattern:
  - Show side-by-side comparison columns (`Current` on the left, `Advised` on the right).
  - Highlight rows where values differ and mark them as suggested changes.
  - Keep `Commit changes` disabled until an evaluation has completed and at least one change is suggested.
- Safety behavior:
  - Only `Commit changes` writes to `config.ini` and updates backup JSON, preserving config comments/order via line-level updates.

## Desktop capture mapping update (2026-02-15)

- What changed:
  - `devtools/capture_ui_active_batch.py` desktop tab cycle now captures six desktop files:
    - `ui_mesa_desktop.png` (Workflows)
    - `ui_mesa_desktop_tab2.png` (Status)
    - `ui_mesa_desktop_tab3.png` (Config)
    - `ui_mesa_desktop_tab4.png` (Tune processing)
    - `ui_mesa_desktop_tab5.png` (Manage MESA data)
    - `ui_mesa_desktop_tab6.png` (About)
- Why:
  - Prevents tab-name drift in docs after the Tune processing and Manage tab additions.

## Venv association for Windows dev vs compile (2026-03-08)

- What changed:
  - Build entrypoint `devtools/compile_win_11.bat` now prefers `.venv_compile` first, then `.venv`, then `python` from PATH.
  - Added `devtools/setup_venvs_win311.bat` to create/update both venvs with the intended requirements files.
  - Added purpose headers in `requirements_all_win311.txt` and `requirements_compile_win311.txt`.
- Root cause:
  - Multiple requirements files caused ambiguity about which venv should be used for daily work vs packaging.
- Practical fix / decision:
  - Standardize `.venv` for development (`requirements_all_win311.txt`).
  - Use `.venv_compile` for packaging when it exists, but keep `.venv` as a supported fallback so builds still run in environments where the compile-specific venv has not been created yet.

## Webview helper packaging lesson (2026-03-23)

- What changed:
  - `devtools/build_all.py` now force-includes the webview collection step for known pywebview-based helpers such as `line_manage`, `analysis_setup`, `asset_map_view`, and `map_overview`.
- Root cause:
  - Generic import-pattern detection usually works, but packaging a UI helper should not depend only on source scanning for `import webview`.
- Practical fix / decision:
  - Keep `pywebview` in compile requirements and also maintain an explicit helper allowlist for webview collection in the build script.

## Launcher build stamp visibility (2026-03-23)

- What changed:
  - `mesa.py` now reads packaged `build_info.json` in frozen builds and shows the build timestamp in the header banner.
  - `devtools/build_all.py` now writes that metadata file into `dist\mesa` using `Europe/Oslo` local time.
- Root cause:
  - The visible launcher version string comes from `config.ini` as a manual release label, so a fresh build can still look stale if that label is not bumped.
- Practical fix / decision:
  - Keep `config.ini` as the release/version label, but also stamp each packaged build with an explicit build timestamp to reduce confusion.

## Historical note (superseded, 2026-03-23): Processing setup briefly stayed out-of-process

- What was true at the time:
  - Before the PySide6 migration, `processing_setup.py` still behaved like a heavy Tk helper and was safer to keep out-of-process.
- Why this is superseded:
  - The current helper is PySide6-based, exposes `run(base_dir, master=None)`, and is now one of the embedded Qt helpers launched in-process from `mesa.py`.
- Current rule:
  - Do not rely on this older out-of-process restriction when documenting or changing current launcher behavior.

## PySide6 launcher and helper finalisation (2026-04-12)

- What changed:
  - `mesa.py` now launches the migrated Qt helpers (`geocode_manage`, `asset_manage`, `processing_setup`, `processing_pipeline_run`, `atlas_manage`, `report_generate`, `analysis_present`) in-process through their `run()` entry points when they are bundled inside `mesa.exe`.
  - `devtools/build_all.py` now collects `PySide6` instead of `ttkbootstrap` for the main app and helper packaging profiles.
  - Shared styling is now applied through `asset_manage.apply_shared_stylesheet()` so the desktop and helper windows use the same generated checkbox/radio indicator assets and corner-button styling.
  - Legacy Tk-only bootstrap artifacts (`mesa_tk_old.py` and `code/locale_bootstrap.py`) were removed from the active code path.

- Root cause:
  - The packaged launcher still tried to start helpers like `asset_manage.exe` as standalone subprocesses even after the build started embedding those helpers as hidden imports inside `mesa.exe`.

- Practical fix / decision:
  - Standardise on PySide6 as the canonical desktop UI runtime.
  - Use explicit in-process helper launching for embedded Qt modules, while keeping subprocess launching only for tools that still need to stay standalone.
  - Remove obsolete `ttkbootstrap` configuration fields from active runtime settings where they no longer affect behaviour.

- UI refinements included in the same pass:
  - Embedded `Exit` buttons in the tab bar corner across the main desktop and helper windows.

## macOS development venv split from Windows bundle (2026-04-17)

- What changed:
  - Added `requirements_macos_dev.txt` for source-based development on macOS.
  - Bootstrapped `.venv` with Python 3.11 and installed the macOS-focused dependency set there.
- Root cause:
  - The canonical `requirements_all_win311.txt` is intentionally Windows-oriented and failed on macOS at `psycopg2==2.9.9` because `pg_config` was unavailable.
  - That Windows bundle also includes packaging-only and Windows-specific tools that are not needed for routine source runs on macOS.
- Practical fix / decision:
  - Keep the Windows requirements files unchanged as the packaging/runtime baseline.
  - Use `requirements_macos_dev.txt` on macOS for launcher/helper development and lightweight verification.
  - Smaller default window footprints for better fit on common laptop displays.
  - Consistent progress-bar and log-panel behaviour in the processing runner and internal processing UI.

## Processing runner log routing and minimap stability (2026-04-12)

- What changed:
  - `processing_pipeline_run.py` now forwards headless processing logs directly into the runner UI instead of relying on file tailing during the active run.
  - `processing_internal.py` now accepts an external log callback in `run_headless()` and resets it safely when the run completes.
  - The minimap helper now routes OpenStreetMap requests through the local MESA tile proxy and shuts that proxy down when the window closes.

- Root cause:
  - Direct UI logging and background file tailing could duplicate every processing line in the runner.
  - Direct OpenStreetMap tile access is less reliable in packaged helper windows and can be blocked or rate-limited.

- Practical fix / decision:
  - During active processing, prefer signal/callback-based log delivery to the parent window and resume file tailing only after the worker finishes.
  - Use the same local tile-proxy approach in the minimap that other MESA viewers already use.

## Root README scope split (2026-03-23)

- What changed:
  - Added a root `README.md` that explains the system for developers: workflow, architecture, helper responsibilities, shared data model, and build model.
- Root cause:
  - The repo had packaging-oriented documentation, but not a single developer-facing entry document explaining what MESA actually is as a system.
- Practical fix / decision:
  - Keep the root `README.md` developer-facing and keep `code/README.md` focused on the packaged distribution.

## UI helpers should lazy-load heavy GIS/chart stacks (2026-03-23)

- What changed:
  - `processing_setup.py`, `analysis_setup.py`, and `analysis_present.py` now defer `numpy`/`pandas`/`geopandas`/`shapely` or `matplotlib` imports until after a lightweight UI shell or API bridge is already up.
- Root cause:
  - Importing the full GIS/data stack at module import time made frozen helpers look hung before any visible window or status message appeared.
- Practical fix / decision:
  - For desktop helpers, defer heavy runtime imports until after the lightweight UI shell is ready.
  - For pywebview helpers, keep the bridge object cheap and initialise GIS/storage backends on first API use.

## Launcher source runs should self-correct to `.venv` (2026-03-24)

- What changed:
  - `mesa.py` now checks its interpreter at process start during source runs on Windows and relaunches itself with `.\.venv\Scripts\python.exe` (or `pythonw.exe` when appropriate) if it was started from the wrong Python.
- Root cause:
  - The repo already standardized `.venv` for daily development, but launching `mesa.py` from an IDE, shell, or file association could still pick a different interpreter and then fail on imports or mismatch helper behavior.
- Practical fix / decision:
  - Keep packaged `.exe` runs untouched.
  - Correct the interpreter before importing heavy GUI/runtime dependencies.
  - When relaunching, also set `VIRTUAL_ENV` and prepend the venv `Scripts` folder to `PATH` so descendant processes inherit the expected development environment.

## Embedded helper launch model (current, 2026-04-12)

- What changed:
  - `mesa.py` now lazy-imports selected helper modules and launches the embedded Qt helpers in-process through their `run(base_dir, master=None)` entry points.
  - This applies to both source runs and packaged `mesa.exe` runs for helpers that are bundled as hidden imports.
- Root cause:
  - Launching every helper as a separate process duplicated heavy imports and made the embedded-helper packaging strategy inconsistent with the runtime launcher logic.
- Practical fix / decision:
  - Use in-process launch for the embedded PySide6 helpers.
  - Keep subprocess launch only for tools that are intentionally standalone, such as the pywebview helpers and the raster-tiles helper.
  - Do not rely on old Tk-specific rules such as `tk.Toplevel(...)` or `root.after(...)` when describing the current launcher model.

## PySide6 as canonical Qt binding (2026-04-11)

- What changed:
  - The desktop launcher moved to `mesa.py`, using PySide6 (6.11.0) as the canonical Qt binding.
  - Color palette reworked from cool blue/slate to warm green/oker earth-tone palette, inspired by GRASP Desktop.
- Key API differences:
  - `app.exec_()` -> `app.exec()` (PySide6 uses the non-underscore form).
  - Enum paths like `QFont.Bold` still work in PySide6 6.x as compat aliases (resolves to `QFont.Weight.Bold`).
  - Import paths change from `PyQt5.QtWidgets` to `PySide6.QtWidgets` etc., but class/function names are identical.
- Practical fix / decision:
  - PySide6 chosen over PyQt5 for: LGPL licensing, active maintenance by Qt Company, better long-term support.
  - `mesa_qt.py` is no longer the canonical launcher name; update docs and comments to `mesa.py`.
  - Keep PyQt5 in `requirements_all_win311.txt` only where other repos/projects still depend on it; MESA itself should be documented as PySide6-based.
  - Color palette uses GRASP-inspired earth tones: background `#f3ecdf`, text `#3f3528`, accents `#715a36`/`#9b7c3d`, success `#4d7c0f`, warning `#b45309`, danger `#b02a37`.

## GitHub release posts from Zenodo (2026-04-16)

- What changed:
  - Added `devtools/github_release_from_zenodo.py` to generate a GitHub release title, tag, and markdown body directly from a Zenodo record, then optionally publish it with `gh release create`.
  - Used it to publish the GitHub prerelease for Zenodo record `19615520` as tag `5.0-beta-2026.04.16`.
- Root cause:
  - Release posts had previously been written manually on GitHub even though the canonical release metadata already existed in Zenodo.
- Practical fix / decision:
  - Use the Zenodo API (`https://zenodo.org/api/records/<id>`) as the source of truth for title, DOI, publication date, description, and bundled filename.
  - Use `gh` for the actual GitHub release creation so authentication stays with the local CLI session.
  - Generate a release preview first, then publish with `python devtools\github_release_from_zenodo.py <record-id> --publish`.

## Apple Silicon awareness for processing tuning (2026-04-23)

- What changed:
  - `_recommended_processing_tuning` in `mesa.py` now branches on `os_name == "darwin"` + `machine.startswith("arm")`. When both hold, it:
    - Reads the performance-core count via `sysctl hw.perflevel0.physicalcpu` and sizes the worker cap as `max(2, min(P-2, P))` (capped at 16), so efficiency cores do not inflate the cap.
    - Subtracts 2 GB from the RAM-tier `approx_gb_per_worker` with a floor of 2.5 GB, reflecting the more memory-efficient macOS runtime for this workload.
    - Uses `mem_target_frac = 0.70` instead of `0.85` because Apple Silicon uses unified memory shared with the GPU/WindowServer.
  - Baseline `config.ini` was also updated to match the M4 Max case: `auto_workers_max=10`, `approx_gb_per_worker=4.0`, `mem_target_frac=0.70`, `mosaic_auto_worker_max=10`, `mosaic_auto_worker_fraction=0.65`.
- Root cause:
  - The original ladder used only logical CPU count and a fixed `0.85` memory fraction. That ladder was calibrated for Windows x86_64 with discrete GPUs; on an Apple M4 Max (16c = 12 P + 4 E, 64 GB unified) it under-used P-cores and left too little headroom for the GPU/WindowServer.
- Non-regression guarantee:
  - The non-Apple-Silicon path is byte-identical to the prior behaviour (Windows ladder, `mem_target_frac=0.85`, same chunking constants). Intel Macs (`darwin` + `x86_64`) correctly fall through to that same path because the gate requires both `darwin` AND `arm*`.
  - Verified by unit-executing the function with simulated hosts: Windows 16c/64 GB, Windows 8c/16 GB, Apple M4 Max 16c/64 GB, Apple M2 base 8c/16 GB, and Intel Mac 8c/16 GB.
- Practical rule for future tuning changes:
  - Keep platform-specific adjustments additive behind a narrow gate so the Windows packaging baseline never shifts by accident.
  - If more Apple-specific tuning is needed (E-core behaviour, thermal-throttling hints), store the probe result in `cap_row` at capture time rather than re-probing inside the tuning function.

## Per-stage worker caps for skewed workloads (2026-04-24)

- Rule:
  - Stages with highly skewed per-item memory footprints need **per-stage worker caps**, not one global `max_workers`. `processing_internal.py` now exposes four independent caps in `config.ini` (all `0 = auto`):
    - `flatten_max_workers` - large-partition flatten phase (pandas groupby/merge can balloon whole partitions; keep conservative, e.g. 2 on 64 GB).
    - `flatten_small_max_workers` - small-partition flatten phase (no ballooning; can saturate CPU).
    - `backfill_max_workers` - post-flatten area_m2 backfill (I/O-bound; can run with broad parallelism).
    - `tiles_max_workers` - mbtiles raster generation (each worker holds a pickled geometry list; RAM scales with workers × group size).
- Why:
  - Original code used `max_workers` / CPU count for every stage. On an M4 Max (64 GB unified memory) a single tbl_stacked run produced 1363 partitions with median 202 KB, p95 19 MB, max 687 MB. With `max_workers=10` the flatten stage tried to allocate ~139 GB and choked. Halving to 2 workers fixed the crash but left CPU at ~15% utilisation through the other phases because the same conservative cap applied everywhere.
- How to apply:
  - On memory-constrained or unified-memory hosts (Apple Silicon), keep `flatten_max_workers` low (2-3) and let the other three caps auto-size via RAM budget.
  - On Windows/Linux workstations with ample RAM and many cores, set all four to `0` - the auto path multiplies by `mem_target_frac / flatten_approx_gb_per_worker` (or the stage-specific per-worker estimate) and clamps to CPU count, which scales up naturally.
  - Heavy-first partitioning: `flatten_tbl_stacked` splits files at `flatten_large_partition_mb` (default 50 MB) and runs the large tail first with the tight cap, then the long small tail with a wider pool. The small phase's per-worker RAM estimate is ~1/4 of the large phase's to match its actual footprint.
  - Large auxiliary DataFrames (e.g. `area_map`) are persisted to a scratch parquet (`__area_map.parquet`) before the pool spawns, so workers read it lazily instead of inheriting a pickled copy across every spawn boundary. This cuts driver-process RSS by several GB per run.
- Non-regression guarantee:
  - All four caps default to `0 = auto`. With no config changes and no psutil, the code paths fall back to the same CPU-based heuristic used previously. Existing Windows baselines are only affected if the user explicitly opts in by setting the new keys or by adopting the updated `config.ini` defaults.

## Parent-side memory in the pipeline (2026-04-26)

- Rule:
  - In `code/processing_internal.py`, the parent process must never materialise a known-large dataset (`tbl_stacked`, `tbl_flat`, similar partitioned outputs) into a single in-process `(Geo)DataFrame`. This applies most strictly to the windows *between* worker pools (e.g. between intersect and flatten), where no panic watchdog is alive.
  - For row counts: use `pyarrow.dataset.dataset(<path>).count_rows()` or reuse the count already logged by `intersect_assets_geocodes` ("tbl_stacked dataset written as folder with N parts and ~M rows").
  - For presence checks: glob the partition directory.
  - For actual data work on these tables: read inside a worker process.
- Why:
  - Three previous memory fixes on this branch (`99e5956` → `8b878c5` → `ba24a32`) all bolted guards onto worker pools - per-pool watchdog, per-stage worker caps, three-phase huge/large/small flatten split, flatten pre-flight RAM/swap check. All correct, all scoped to *inside* `Pool(...)` blocks.
  - The April-26 incident was caused by a 4-line "post-intersect cleanup" debug log added in the unguarded gap between intersect's pool and flatten's pool: `sample = read_parquet_or_empty("tbl_stacked"); log_to_gui(... f"tbl_stacked rows (sample read): {len(sample):,}")`. `read_parquet_or_empty()` resolves a partitioned directory to `gpd.read_parquet(<dir>)`, which materialises the entire dataset (1363 parts, 92,430,523 rows, geometries included) into the parent's RSS - tens of GB on top of the ~10 GB of intersect residue. None of the existing watchdogs fired because none were running in that window. The host swap-stalled; the process eventually died with no flatten log lines and no `tbl_flat`.
  - The variable was named `sample`. It was not a sample. The row count it printed was already in the log 11 seconds earlier.
- How to apply:
  - Before adding any read in `process_tbl_stacked`, `flatten_tbl_stacked`, `intersect_assets_geocodes`, or any code that runs in the parent between stages, check: "is this materialising a known-large dataset?" If yes, replace with a glob (presence), `count_rows()` (count), or move it inside a worker.
  - The new `_start_lifetime_panic_watchdog()` (started by `run_processing_pipeline`, default 90% RAM for 5s → `os._exit(137)`) is a backstop that frees the host but kills the run's progress. Treat it as a smoke alarm, not a fire suppression system - it is not a substitute for not making the allocation in the first place.
  - The docstring on `read_parquet_or_empty()` carries the same warning. Read it before calling.
- Non-regression guarantee:
  - The fix replaces the offending block with `len(list(ds_dir.glob("*.parquet")))` and adds the lifetime watchdog with conservative defaults (90%/5s) wired through `mem_lifetime_panic_percent` / `mem_lifetime_panic_grace_secs`. Existing per-pool watchdogs, per-stage caps, three-phase split, and pre-flight checks are unchanged. Windows baseline behaviour is unaffected unless `psutil` is available and pressure crosses the new lifetime threshold (which it should not under normal operation).

## QTimer.singleShot from worker threads is a no-op (2026-04-27)

- Rule:
  - Never call `QTimer.singleShot(ms, slot)` from a non-GUI thread. The static `QTimer.singleShot` binds to the calling thread's event loop; a Python `threading.Thread` worker has no Qt event loop, so the slot never fires. Route GUI updates from worker threads through a `Signal` defined on a `QObject` that lives on the GUI thread; Qt's auto-connection will queue the call onto the GUI thread.
- Why:
  - In `code/geocode_manage.py`, `_run_mosaic` started `run_mosaic` in a daemon thread via `_run_in_thread`. The mosaic step's `_after(success)` callback runs at the tail end of `run_mosaic` *inside that same worker thread*. The previous code used `QTimer.singleShot(200, self._update_mosaic_status)` to refresh the status label from "Running…" to "OK"/"REQUIRED". The log line and progress bar updated correctly (those went through `self._signals.mosaic_line.emit(...)` and `progress_update.emit(...)`, both `Signal`s, which are queued cross-thread), but the status label stayed pinned at "Running…" forever because the QTimer slot never executed.
- How to apply:
  - When you see a worker thread that needs to refresh the GUI on completion, add a dedicated `Signal()` to the window's `_Signals` `QObject`, connect it to the slot in `__init__`, and `emit()` from the worker. The fix here added `mosaic_finished = Signal()` and replaced the `QTimer.singleShot` call with `self._signals.mosaic_finished.emit()`.
  - This pattern already existed in the same file for `mosaic_line` and `task_finished`. The bug was a one-off shortcut that bypassed it.
- Non-regression guarantee:
  - The fix is a three-line change confined to mosaic completion routing: a new `Signal`, one `connect` call, and one `emit` swap. The worker function (`run_mosaic`), the thread runner (`_run_in_thread`), and `_update_mosaic_status` itself are unchanged. No new dependencies; no behaviour change on other threads or other helpers.

## Evaluate must cover every host-sensitive config family (2026-04-27)

- Rule:
  - When a config family scales with host capability (CPU count, RAM tier, unified-vs-discrete memory), `_recommended_processing_tuning` in `mesa.py` must emit values for *all* keys in that family. A key the function doesn't touch keeps whatever's in the repo default `config.ini`, which is calibrated to whoever last committed it.
- Why:
  - The mosaic boundary-extraction stage has its own worker-sizing knobs (`mosaic_auto_worker_fraction`, `mosaic_auto_worker_max`, `mosaic_extract_chunk_size`) that are independent of `max_workers` / `auto_workers_max`. The Apple Silicon tuning pass (commits `99e5956` and `ba24a32`) set those keys to M4-tight values in the committed `config.ini` (fraction=0.65, max=10, chunk_size=1000) but did not extend `_recommended_processing_tuning` to write them. Result on a 16-core / 127 GB Windows host running Evaluate → Commit: Stage 2 / flatten / backfill / tiles all moved to Windows-appropriate values, but mosaic kept the M4-tight settings silently. Mosaic ran with 10 workers (16 × 0.65 capped at 10) and coarse 1000-asset chunks, leaving 6 cores idle through the long tail.
  - The user noticed because the mosaic step felt incredibly slow and asked "did I forget to turn some knobs?". Yes — knobs Evaluate did not know about.
- How to apply:
  - When adding a new platform-sensitive config family, audit `_recommended_processing_tuning` and add the keys to its `recommendations` dict and rationale text. Test by simulating at least three hosts (e.g. Windows 16C/127GB, Apple M4 16C/64GB, Apple M2 8C/16GB) and confirm each row produces sensible values.
  - For mosaic specifically: Apple Silicon gets `fraction=0.65, max=10` (max=12 above 96 GB) because unified memory limits how much extraction parallelism is safe. Non-Apple gets `fraction=0.75, max=0` (unbounded; per-worker 1.5 GB RAM budget naturally caps it). `mosaic_extract_chunk_size` ladders 1000/500/250 by core count to improve load balance as worker count grows.
- Non-regression guarantee:
  - The change is additive: existing keys keep their values, three new keys appear in the Evaluate table. Users who never click Commit see no change. The repo-default `config.ini` is unchanged in this commit; first-run users on either platform should run Evaluate → Commit to get host-appropriate values. Future tuning passes for new config families should follow this pattern.

## Mosaic union reduction is the silent long-tail (2026-04-27)

- Rule:
  - "Reducing coverage" / "Reducing edges(final)" log lines that stretch for hours indicate `mosaic_coverage_union_batch` and `mosaic_line_union_max_partials` are too low for the host. GEOS `unary_union` is single-threaded and memory-bandwidth sensitive; many small batches produce many small unions, which then require many pairwise reduction rounds — each of which is another serial `unary_union` call. Bigger batches and a higher partials ceiling collapse the total work into fewer-but-bigger unions, which on hosts with ample RAM is dramatically faster end-to-end.
  - When the heartbeat goes silent for tens of minutes during reduction, the process is *not* hung — it is inside a single very-large `unary_union([a, b])` call that won't emit progress until it returns. Confirm liveness via `Get-Process` (CPU climbing) before considering a kill.
- Why:
  - On a Windows 16-core / 127 GB host with the M4-tight repo defaults (`mosaic_coverage_union_batch=500`, `mosaic_line_union_max_partials=16`), a single basic_mosaic run on a moderate dataset spent ~7 hours in the streaming "Reducing coverage" intermediate-reduction loop alone, then another 2+ hours in `edges(final)` round 1 with one merge stuck inside `unary_union` for 2 hours without a heartbeat. The bottleneck is per-merge cost in GEOS, which on Windows runs noticeably slower than on Apple Silicon at equivalent core counts because Apple's high single-core throughput plus shared-memory bandwidth fits the workload well.
  - `_recommended_processing_tuning` was extended in this same commit chain to emit two new keys per host:
    - Apple Silicon RAM tiers: 500/16, 1000/16, 1500/24, 2000/32 (≤16 / ≤32 / ≤64 / >64 GB).
    - Non-Apple RAM tiers: 500/16, 1000/24, 2000/32, 4000/64.
  - Apple Silicon stays a touch tighter because unified memory peaks during a single big `unary_union` call must not crowd the GPU/WindowServer.
- How to apply:
  - When a user reports slow mosaic on Windows or large-RAM Linux: check `config.ini` for `mosaic_coverage_union_batch` and `mosaic_line_union_max_partials` at low values (500/16). Run Evaluate → Commit to apply host-appropriate values.
  - Last-resort escape hatch for an extreme dataset: `mosaic_coverage_union = false` skips the whole coverage-reduction stage and falls back to STRtree for face filtering. Trade-off: slower per-face filtering, but no hours-long single-threaded merge tail.
- Non-regression guarantee:
  - Keys default to existing repo values when missing; small RAM tiers (≤16 GB) keep the prior 500/16 defaults so memory-constrained hosts see no change. Larger hosts opt in via Evaluate → Commit.
- Follow-up planned:
  - Per `cooperation.md` exchange (2026-04-28), the mosaic batching keys are placed in Evaluate as a temporary measure to unblock the 7-hour stall. Apple Silicon Claude takes a follow-up commit to migrate them into `auto_tune.py` with a `geocode_manage` call site, so all runtime worker auto-sizing lives in one place. When that lands, this section should be updated to reference `auto_tune` as the source of truth and the Evaluate emissions become a fallback or get removed.

## Dissolve must not synthesise a constant key when no uniform column exists (2026-04-28)

- Rule:
  - In `code/asset_manage.py` `_dissolve_by_attributes`, when partitioning attribute columns into uniform vs diverging produces an *empty* uniform set, **skip the dissolve and return the original gdf**. Do not fall back to a synthetic constant key. Constant-key dissolve groups every polygon in the layer together, and after `explode()` the result is one polygon per connected geometric component — i.e. the layer reduced to its natural extent. That destroys the boundary information `basic_mosaic` needs to subdivide its faces, and where multiple such layers overlap continuously, the mosaic produces one mega-face spanning huge fractions of the project area.
- Why:
  - April-28 incident: re-imported assets with smart-key dissolve. At import time, `tbl_asset_object` rows have no per-group classifying attributes — importance / susceptibility live in `tbl_asset_group`, set by `processing_setup` *after* import. So the smart-key partition saw every column as diverging (cell IDs / FIDs / per-pixel measurements) → `uniform_cols` empty → fell back to synthetic constant for layers like Wetlands_AGEMP (2.6 M cells), CRENVU_X (~321 k), Land Condition_X (~550 k). Each of those layers collapsed to a few big connected-region polygons covering the layer's natural geographic extent.
  - Measured damage in `tbl_flat`:

    | Metric | Pre-incident run | Post-incident run |
    |---|---:|---:|
    | basic_mosaic cells | 9,671,688 | 3,900,125 (-60 %) |
    | Median cell area | 223 m² | 1,000 m² (4.5×) |
    | Max cell area | 1,993 km² (one outlier) | 43,021 km² (21×) |
    | Cells > 1,000 km² | 0 | 4 cells = 47.7 % of project |
    | Code A (sens=25) share | 25.4 % | 72.0 % |

  - The 43,021 km² mega-face had 22 distinct asset groups overlapping it geometrically; `tbl_stacked` confirmed 1,123 individual asset polygons attributing to it. Surface_water (imp=5, sus=5) and CRENVU_21-31 (imp=5, sus=5) both correctly contributed sens=25 → `sensitivity_max = 25` was mechanically correct given the inputs. The bug was upstream: the layer dissolves stripped boundary detail that `basic_mosaic` needed to subdivide that 43 k km² region into smaller faces.
- How to apply:
  - The new behaviour: if `uniform_cols` is empty, skip dissolve, log `no uniform attribute column to key on; skipping dissolve, keeping N original polygons`, append `(label, n_in, n_in, "no_uniform_key")` to the per-import stats so the final import summary reflects how many layers were skipped vs really merged.
  - Layers with at least one column that has the same value on every row (e.g. one-class-per-file land-cover datasets where the file's `class_name` column is uniform) still dissolve cleanly.
  - Tradeoff acknowledged: pure raster grids without a classifier keep their per-cell granularity and produce the visual moire the smart-key dissolve was originally added to fix. Moire is *local visual noise*; the mega-face was a *quantitative* error propagating max-sensitivity across 35 % of the project area. Moire is the better failure mode.
- Non-regression guarantee:
  - The skip path fires only when *every* attribute column has cardinality > 1 (i.e. nothing to honestly key by). Layers with at least one uniform column behave exactly as in the smart-key implementation.
  - The "Dissolve adjacent polygons (recommended)" import checkbox stays default-on; it just becomes a no-op for layers without a classifying attribute, which is the safe default.

## Wiki note: data preparation guidance (pending — to fold into the user-facing guide)

When the wiki guide is rebuilt, capture this as a "Preparing your data" chapter. Operator-facing language:

- **Ecological / organic asset polygons work best.** Wetland outlines, settlement footprints, river polygons, protected-area boundaries, atlas pages — anything where the polygon shape carries real geographic information. `basic_mosaic` subdivides on those boundaries, so each mosaic face represents a meaningful "what overlaps here" region.
- **Gridded / raster-derived asset layers are the failure mode.** Every-pixel-is-a-polygon datasets — typically the result of someone running `raster_to_polygon` on a classification raster — cause two distinct problems in MESA:
  1. **Per-cell visual moire**: `sensitivity_max` paints each individual cell, and the source grid pattern shows up in the mosaic visualisation. Local cosmetic problem.
  2. **Mosaic-face collapse if the cells form large continuous coverage AND the layer has no honest classifier column**: the import-time dissolve has no way to distinguish "real per-cell variation" from "noise per-cell IDs". Either you get moire (no dissolve) or mega-faces that propagate max-sensitivity across huge regions (constant-key dissolve — the April-28 regression). After this commit, the mega-face path is closed; the moire one is the deliberate fallback.
- **Operator recommendations:**
  - **Vectorise-and-classify before import.** Group adjacent same-value raster cells into polygons at the data-prep stage (in QGIS: `r.to.vect`, then `Dissolve` keyed on the value). The result is a layer with N polygons (one per class region) and a meaningful classifying column the dissolve can honestly key on.
  - **One file per sensitivity bin** is the simpler workaround when full vectorisation is impractical. MESA already supports this (e.g. `SpeciesRichness2010_0-5.shp` / `_6-10.shp` / `_11-15.shp` etc.) — each file's importance/susceptibility is set per asset group in `processing_setup`, so within a file the attributes are uniform and the dissolve has something to key on.
  - **Any layer with millions of polygons at sub-100 m resolution** should be considered a data-prep candidate. If the user can't reduce the polygon count, expect moire visual cost, but at least no quantitative bleed.
- **Dissolve checkbox semantics** (for the UI doc): "Dissolve adjacent polygons (recommended)" means *merge polygons that share an honest classifying attribute*. Layers with no such attribute are skipped (kept as-is), with a log line stating that. This is correct conservative behaviour, not a missing feature.
