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

## Wiki note: data preparation guidance (folded into wiki Data page on 2026-04-28)

This guidance was folded into the wiki **Data** page (`mesa.wiki/Data.md`) on 2026-04-28 as the "Preparing your data" section, with the dissolve incident summarised in "Gridded / raster-derived asset layers are the failure mode". The original operator-facing language is preserved here so future maintainers can reconcile the two if either drifts:

- **Ecological / organic asset polygons work best.** Wetland outlines, settlement footprints, river polygons, protected-area boundaries, atlas pages — anything where the polygon shape carries real geographic information. `basic_mosaic` subdivides on those boundaries, so each mosaic face represents a meaningful "what overlaps here" region.
- **Gridded / raster-derived asset layers are the failure mode.** Every-pixel-is-a-polygon datasets — typically the result of someone running `raster_to_polygon` on a classification raster — cause two distinct problems in MESA:
  1. **Per-cell visual moire**: `sensitivity_max` paints each individual cell, and the source grid pattern shows up in the mosaic visualisation. Local cosmetic problem.
  2. **Mosaic-face collapse if the cells form large continuous coverage AND the layer has no honest classifier column**: the import-time dissolve has no way to distinguish "real per-cell variation" from "noise per-cell IDs". Either you get moire (no dissolve) or mega-faces that propagate max-sensitivity across huge regions (constant-key dissolve — the April-28 regression). After this commit, the mega-face path is closed; the moire one is the deliberate fallback.
- **Operator recommendations:**
  - **Vectorise-and-classify before import.** Group adjacent same-value raster cells into polygons at the data-prep stage (in QGIS: `r.to.vect`, then `Dissolve` keyed on the value). The result is a layer with N polygons (one per class region) and a meaningful classifying column the dissolve can honestly key on.
  - **One file per sensitivity bin** is the simpler workaround when full vectorisation is impractical. MESA already supports this (e.g. `SpeciesRichness2010_0-5.shp` / `_6-10.shp` / `_11-15.shp` etc.) — each file's importance/susceptibility is set per asset group in `processing_setup`, so within a file the attributes are uniform and the dissolve has something to key on.
  - **Any layer with millions of polygons at sub-100 m resolution** should be considered a data-prep candidate. If the user can't reduce the polygon count, expect moire visual cost, but at least no quantitative bleed.
- **Dissolve checkbox semantics** (for the UI doc): "Dissolve adjacent polygons (recommended)" means *merge polygons that share an honest classifying attribute*. Layers with no such attribute are skipped (kept as-is), with a log line stating that. This is correct conservative behaviour, not a missing feature.

## auto_tune mutations must reach flatten / backfill (2026-04-29)

- Rule:
  - `flatten_tbl_stacked` must use the *same* `cfg` object that `auto_tune_in_place` mutated, not re-read `config.ini` from disk. Pass `cfg=cfg` from `run_processing_pipeline`. The disk re-read path is kept only as a fallback for direct callers that don't have the in-memory cfg.
- Why:
  - `auto_tune_in_place` writes its derived values into `cfg["DEFAULT"]` *in memory only* — it does not persist to disk. `process_tbl_stacked` was always called with the auto-tuned cfg object, so intersect saw the mutations. But `flatten_tbl_stacked` was called only with `config_file` (a Path) and re-read cfg from disk inside, which gave it the un-tuned (= zero / missing) values. `_resolve_phase_workers` then fell into the auto-derive-from-RAM path, which by the time flatten / backfill runs (hours after pipeline start) sees parent-RSS bloat and tight available memory → derives 1 worker.
  - April-29 incident: pipeline started at 23:12 with `[auto-tune] backfill_max_workers = 16  (avail 34.9 GB / 1.5 GB per-backfill-worker = 16)`. ~13 hours later when backfill actually ran, `cfg_local = read_config(config_file)` re-read disk values, saw `backfill_max_workers = 0`, the resolver auto-derived against `avail ≈ 5 GB / 2.0 GB per-worker = 1`, logged `Backfill: 1363 partitions using 1 workers…`. At ~106 s/partition (12× slower than the normal 9 s/partition with parallel workers and lighter parent), the user was looking at a ~31-hour ETA on a single-threaded loop. Operator killed and chose Option C (skip backfill, keep tbl_flat already-written, run tiles, fix bug).
- How to apply:
  - Whenever a stage in `run_processing_pipeline` reads worker / partition / RAM-budget values that auto_tune manages (`max_workers`, `flatten_max_workers`, `flatten_small_max_workers`, `flatten_huge_partition_mb`, `tiles_max_workers`, `backfill_max_workers`), it must do so via the in-memory cfg object — either directly or by being passed it from the pipeline driver.
  - Same rule will apply to the planned `geocode_manage` mosaic call site for the eventual mosaic-batching auto_tune migration.
- Non-regression guarantee:
  - The `cfg` parameter on `flatten_tbl_stacked` defaults to `None`. When `None`, the old disk-read fallback runs, so direct callers (e.g. `run_flatten_only.py` if it ever returns) keep working. The fix only takes effect when the pipeline driver explicitly passes the auto-tuned cfg.

## Each materially-costly pipeline stage gets its own run flag + UI checkbox + standalone function (2026-04-29)

- Rule:
  - When a step inside the data-processing pipeline has wall-time on the order of minutes-to-hours (rather than seconds), it must be exposed as an *independent* stage:
    1. Standalone top-level function in `processing_internal.py` (e.g. `flatten_tbl_stacked`, `backfill_tbl_stacked`).
    2. Soft-validating: if a required input is missing on disk, log a clear skip line and return — do not raise.
    3. `run_X` boolean flag plumbed through `run_processing_pipeline` → `run_headless` → `run_data_process` → `ProcessPlan` → `run_selected`.
    4. Per-stage CLI flag `--no-X`.
    5. Sub-checkbox in the Advanced-mode `ProcessRunnerWindow` cascade so the user can rerun just that step.
  - Stage labels in the run log get split sub-numbers (`[Stage 3a] Flatten`, `[Stage 3b] Backfill`) so the operator sees them as distinct steps without breaking the existing "stages 1-4" mental model.
- Why:
  - April-29 incident: backfill (area_m2 enrichment of `tbl_stacked` from `tbl_flat`) was inlined inside `flatten_tbl_stacked` as "9. Parallel Backfill". Three real costs of inlining surfaced together:
    1. **Hidden wall-time.** Backfill ran ~31 h single-threaded on the user's project before the user even realised it was a separate logical phase from flatten. From the operator's perspective flatten just "took 31 hours" — the cost lived in code that didn't have its own log banner.
    2. **Hidden cfg propagation bug.** `flatten_tbl_stacked` re-read `config.ini` from disk, missing `auto_tune_in_place`'s in-memory mutations. The bug was discoverable only by digging into why backfill — *inside* flatten — was at 1 worker when auto-tune said 16. As a separate stage with its own log banner the disconnect would have been obvious immediately.
    3. **No independent rerun path.** Operator wanted to run only Tiles after killing the slow backfill (Tiles only needs `tbl_flat`, which was already on disk). Couldn't, because the only way to "skip backfill" was to skip flatten entirely, which would have skipped writing `tbl_flat`. Forced the choice between waiting 31 h or redoing 14+ h of intersect+flatten just to get to Tiles.
  - Splitting Backfill out into its own stage solved all three at once.
- How to apply:
  - When introducing a new heavy step (e.g. the planned `geocode_manage` mosaic auto-tune call site, or anything else that crosses the minutes-of-wall-time threshold), don't inline it inside an existing stage's function. Give it its own:
    - top-level function with `(config_file, *, cfg=None)` signature,
    - log banner `[Stage Nx] <stage name>: <one-line summary>`,
    - `run_X` flag with default `True`,
    - CLI `--no-X` flag,
    - Advanced-mode checkbox, with the master "Process" cascade including it.
  - Soft validation is mandatory: missing-input → log + return, never raise. The whole point of independent stages is independent rerunnability, which depends on stages being safe to start from cold.
  - The Advanced-mode UI grouping rule: data sub-stages (everything that touches `tbl_*` files) goes in the left grid column; post-data stages (Tiles / Lines / Analysis — things that consume the data tables) go in the right column. Prep / Intersect / Flatten / Backfill currently fill the left column; future data-side stages slot in there.
- Non-regression guarantee:
  - All `run_X` flags default to `True`, so a caller that doesn't know about the new flags still gets the full pipeline. The Backfill split is a pure refactor of behaviour the previous flatten function already had — same code, same outputs, just gated by its own checkbox.

## Flatten pre-flight should consider dataset size (2026-05-03)

- Rule: Flatten's pre-flight gate evaluates two independent signals and aborts only when *both* fail (system-wide `vm.percent` AND a dataset-aware headroom estimate). The swap-residue check stays a separate hard signal.
- Why: A user reported `[flatten] PRE-FLIGHT ABORT: vm.percent 63.4% > 60%` on a 31 GB Windows host with ~12 GB free running a tiny dataset (130 K rows, ~5 MB on disk). The old gate looked only at system-wide `vm.percent` against a fixed 60% limit, so any host that happened to have everyday desktop apps open before MESA started was blocked from flatten regardless of how small the actual workload was. Auto-tune already drops to `flatten_max_workers=1` in that situation, but the gate ignored that and also ignored on-disk dataset size. Estimated peak for that dataset is ~3.5 GB; available was 11.6 GB — passes the new headroom check by 3×, so flatten now proceeds.
- How to apply: Defaults are tuned to match auto-tune's per-worker model (10× parquet→geopandas amplification, 500 MB per-worker floor, 1.5 GB parent overhead, 1.25× safety on available RAM). When tweaking flatten memory behaviour, prefer `flatten_preflight_avail_safety_factor` (operator-facing) over the internal `flatten_preflight_amplification` / `_per_worker_floor_mb` / `_parent_overhead_gb` knobs. If the gate ever permits a real OOM, the *first* place to look is whether the amplification factor needs raising for that data shape — not whether the safety factor is too low.
- Non-regression guarantee: Big datasets on starved hosts still abort (both gates fail). Big datasets on idle, beefy hosts now go through (headroom passes). The old single-gate behaviour can be restored by setting `flatten_preflight_avail_safety_factor` to a number large enough that the headroom check never passes (e.g. `100`), reducing the dual gate to "vm.percent only" — but no operator should need that.

## basic_mosaic boundary-bleed in sensitivity_max (2026-05-04)

- Rule: After `gpd.sjoin(..., predicate='intersects')` between geocode cells and assets, drop result rows whose `(cell ∩ asset)` overlap is below `_BOUNDARY_ONLY_OVERLAP_MIN_M2` (currently 1 m²). Apply this regardless of geocode group; only the `predicate='contains'` (point-asset) branch can skip it.
- Why: A user spotted an H3_R10 hex coloured B (sensitivity 16) sitting inside a basic_mosaic face coloured A (sensitivity 25). Direct inspection of `tbl_flat` and `tbl_asset_object` for that location showed the basic_mosaic face was 100% covered by Coral/Algae (importance 5 × susceptibility 4 = 20 = B) and 0.0 % covered by Seagrass — yet `assets_overlap_total = 1222` and `sensitivity_max = 25` because 515 Seagrass features and 4 other groups *shared boundary edges* with the face. basic_mosaic faces are constructed by polygonising the union of all asset boundaries, so by construction every neighbouring asset's boundary is a face edge; `predicate='intersects'` is `True` for boundary-only contact, and the per-cell `max` aggregator in `_select_extreme_local` then picks up the highest neighbour's sensitivity even though it covers none of the face. Hex grids escape this by accident — H3 boundaries don't coincide with asset boundaries, so the same join produces interior overlaps. Empirical area distribution for the failing face: every Seagrass "intersection" had area below 3 µm² (max 2.87e-6 m²); the smallest real Coral/Algae overlap was the face itself. A 1 m² threshold cleanly separates precision noise from real overlap.
- How to apply: Any new sjoin that pairs polygonal geocode cells with polygonal assets must run results through `_drop_boundary_only_joins(res, asset_df)` before downstream aggregation. The helper (defined in `processing_internal.py`) reprojects to EPSG:3857 only for the area calc, then filters. `shapely.touches` and DE-9IM `'2********'` are *not* sufficient on their own — both still admit nano-area floating-point "overlaps" that pass the predicate but are noise. Use a metric area threshold. Reuse `_BOUNDARY_ONLY_OVERLAP_MIN_M2` rather than introducing a parallel knob; if a per-stage threshold is ever needed, lift it to config.ini alongside `flatten_sliver_min_area_m2`.
- Non-regression guarantee: H3 cells with real interior overlap are unaffected (smallest legitimate overlap on a tested R10 hex is the full hex area, ~17 000 m², orders of magnitude above the threshold). The filter is wrapped in `try/except` and falls back to the original join result on any failure rather than silently dropping rows. Point assets continue to use `predicate='contains'` and skip the filter.

## Parallelise the basic_mosaic reduction tree per round (2026-05-06)

- Rule: `_tree_reduce_unions` in `geocode_manage.py` now runs the independent pairwise unions of one round in a `multiprocessing.Pool` (default 4 workers, knob `mosaic_reduce_workers`). Late rounds with one or two merges automatically drop to (near-)serial because `round_workers = min(n_workers, merges_total)`. Output is bitwise-identical to the serial path; pairs within a round are independent and `unary_union` is associative+commutative, so processing order does not change the final geometry.
- Why: A user's basic_mosaic on Windows had stalled at 6+ hours into edges/coverage(final) reduction with 11+ idle cores. The function was documented as single-cored ("GEOS unary_union is typically single-threaded, so this stage can look like 'only one core' in Task Manager"). Empirical edges(final) timings on that run: per-merge cost grew ~3.8× per round (12 s → 40 s → 154 s → 617 s → 2,348 s), per-round time grew ~1.9× (335 s → 555 s → 1,077 s → 2,470 s → 4,696 s); coverage(final) round 1 alone took 4,645 s. Extrapolating coverage rounds to completion put basic_mosaic finishing 36–40 hours from the start of round 2. Round 1 with 28 independent merges was the worst single waste — running 4 in parallel turns wall-clock from 28×166 s into ⌈28/4⌉×166 s ≈ 1,162 s, a ~4× speedup.
- How to apply: For the GUI mosaic build, the value flows from `config.ini` → `_build_basic_mosaic` (`reduce_workers = _cfg_int(cfg, "mosaic_reduce_workers", 4)`) → all six `_tree_reduce_unions` callers in the builder (the early `flush_lines` / `flush_coverage` reductions, the periodic mid-build reductions, and the two `(final)` reductions). The retry path inside `_unary_union_safe` deliberately stays serial to avoid recursive Pool creation when a single union OOMs and is being chunked. `maxtasksperchild=1` keeps each worker's address space minimal — they each handle one pair, return the result via pickle, then exit and are replaced. Workers see the geometries through pickle, so they need to be importable at module level (`_reduce_pair_worker`). On a Pool failure the round falls back to the existing serial `_unary_union_safe` path so robust retry chunking is preserved.
- Non-regression guarantee: Setting `mosaic_reduce_workers = 1` restores the exact previous serial behaviour. Per-round results are deterministic regardless of worker count because pair completion order does not affect what pairs the next round sees (only that they're all `unary_union(a,b)` of the same input set). The retry path (chunked fallback inside `_unary_union_safe`) is unchanged.

## Diagnose small A–E buckets — sensitivity ↔ sensitivity_code consistency check (2026-05-04)

- Rule: When an A–E bucket (especially D or E) looks unexpectedly small, run `devtools/diag_sensitivity_consistency.py` *before* suspecting a calculation bug. The script bins each row's value into A–E and compares against the stored code; both `tbl_stacked` (per-asset `sensitivity` vs `sensitivity_code`) and `tbl_flat` (per-cell `sensitivity_max` vs `sensitivity_code_max`) must come back 0 mismatches. If they do, the small bucket is the max-shadowing dynamic of dense overlapping assets, not a code bug — look at the test setup before touching aggregation code.
- Why: `_select_extreme_local` in `processing_internal.py` picks the surviving row's `cat` field directly rather than re-deriving it from the surviving `sv`. Tie-breaking on equal `sv` is alphabetically-earliest `cat` (A wins over B…E), so a desync between per-asset `sensitivity` and `sensitivity_code` (e.g. hand-edited parquet, partial reprocess) would systematically bias upward and starve D/E. On 2026-05-04 a user reported D stuck at 0.2 % despite shifting test assets into D; the diagnostic came back 0 / 94,412 mismatches in `tbl_stacked` and 0 / 19,746 in `tbl_flat` across basic_mosaic + 5 H3 levels. The small-D was real shadowing — the D-target asset (rubble) was geographically embedded inside a higher-class geomorphic layer that always wins the per-cell max. The line/segment path (`tbl_segment_flat`) re-derives the code from the value via `apply_classification_to_gdf` and is immune to the desync risk.
- How to apply: Trigger `python devtools/diag_sensitivity_consistency.py` after any reprocess that changes asset importance/susceptibility values, or whenever a category bucket looks suspiciously thin. Bins are A:21-25, B:16-20, C:11-15, D:6-10, E:1-5. Runs in ~5 s on a 100 K-row stacked. Per-geocode-group breakdown helps localise drift if any partition is stale.
- Non-regression guarantee: n/a (diagnostic only, no code change).

## Optional Segment stage — name collision + parent-side memory (2026-06-05)

- Rule: The new optional "Segment" stage (`code/segmentation.py` + `segment_tbl_stacked` in `processing_internal.py`) writes `tbl_segmentation` / `tbl_segmentation_profiles` — NOT `tbl_segments`. `tbl_segments` already exists and belongs to the Lines pipeline (line splits: `tbl_segments` → `tbl_segment_stacked` → `tbl_segment_flat`). Do not reuse that name for geocode segmentation.
- Why: The first integration draft proposed `tbl_segments` for the geocode-segmentation output; the data model (`devtools/docs/data_model.graphml`) revealed the collision with the Lines stage's existing table. A shared name would have made the report engine and any `read_parquet_or_empty` lookups ambiguous. Renamed everything to `tbl_segmentation*` before wiring.
- How to apply: Segment runs after Backfill, default OFF (GUI checkbox "4b. Segment", CLI `--segment`, or `segment_enabled = 1`). It reads `tbl_stacked` per geocode layer with a `pyarrow.dataset` filter on `name_gis_geocodegroup` *inside a spawned Pool worker* (one layer per task) — the orchestrator never materialises a layer, honouring the parent-side memory rule. Output carries no geometry (join to `tbl_geocode_object` on `code` at render time). Worker cap is auto-tuned via `_derive_segment_max_workers` (config `segment_max_workers`, `segment_approx_gb_per_worker`). The stage uses the hard-panic watchdog only — a single geocode layer is not splittable, so the soft-throttle drain/restart loop (used by backfill/flatten) is deliberately not applied here.
- Non-regression guarantee: Stage is OFF by default everywhere (orchestrator param `run_segment=False`, GUI unchecked and kept out of the master cascade, CLI requires explicit `--segment`, config `segment_enabled=0`, report `include_segmentation=False`). With it off, no new files are written and no existing behaviour changes. Validated on the real store for layers H3_R5/H3_R6 (signatures + agglomerative-ward) through the actual `segment_tbl_stacked` spawn path; full GUI end-to-end run on basic_mosaic still to be done by the operator.

## Segmentation report: sort zones by area, default to basic_mosaic (2026-06-06)

- Rule: In the segmentation report tables, sort zones by **total area (km²) descending**, not by polygon count, and show a `total_area_km2` column. The Segment stage segments **basic_mosaic only by default** (config `segment_geocode_layer` blank → basic_mosaic; `all`/`*` → every level; or a comma-separated list); the report's "Segmentation (area types)" GUI group offers one checkbox per *already-segmented* level (basic_mosaic checked).
- Why: A tiny high-sensitivity cell would top a mean-sensitivity sort and read as noise to a report reader, while a conditional damping rule would be opaque. Area is the single intuitive ordering ("biggest zones first") and the row already carries n_polygons + sens_mean for context. Segmenting all 7 geocode levels by default produced redundant coarse-grained outputs nobody reads and cluttered the level pickers — basic_mosaic is the canonical analytical layer, so the rest are opt-in. Method (signatures vs a specific cluster method) moved from a repeated table column into the sub-heading, since each sub-table is one method.
- How to apply: Per-zone area comes from per-cell `area_m2` — primary source is the `area_m2` already in `tbl_stacked` (backfilled, constant per code → `groupby("code").max()`), falling back to `tbl_flat` filtered by layer, then to NaN → the table shows "–" and sorts by polygon count. `tbl_segmentation_profiles` gained `total_area_km2`. The report (`code/report_generate.py`) renders one sub-table per method under a method-named `heading(4)`, sorted by area desc; `generate_report` takes `segmentation_layers: list | None` (GUI multi-select) and falls back to basic_mosaic when `include_segmentation` is set with no explicit list.
- Non-regression guarantee: Feature is pre-release and still OFF by default. Validated on H3_R6 (area populated, monotonically area-sorted) and a bounded basic_mosaic area-presence check (area_m2 100% non-null in tbl_flat); the full 68M-row basic_mosaic segment run is left to the operator's GUI run to avoid loading 68M rows in a throwaway process.

## Segmentation spatial view belongs in the unified map, not a standalone window (2026-06-06)

- Rule: Do not give segmentation its own map window. Its spatial dimension is just another map layer and belongs in the planned unified Asset + Results map app (as a toggleable overlay with a level selector + signatures/clusters sub-toggle). Its analytical dimension (signature mosaic, area-sorted zone tables) lives in the Word report. Keep `segmentation.build_overview_geojson()` as the spatial renderer.
- Why: A standalone "Segmentations overview" pywebview window was built and parked the same day — it added a 4th map surface (Asset map, Results map, report, segmentation) exactly while the operator's direction is to *merge* Asset map + Results map into one map app. Fewer map surfaces, not more. The hard/valuable part (dissolve 9M cells → a few zones → cached coloured GeoJSON) is in `build_overview_geojson`, independent of any window, so parking the wrapper cost nothing.
- How to apply: When building the unified map app, add "Segmentation" as a layer fed by `build_overview_geojson(gpq_dir, layer, mode)` (cache under output/cache/segmentation_overview/). Reuse the level-selector / legend / area-sorted-zones-table UX from devtools/docs/SEGMENTATION_OVERVIEW_VIEWER_PLAN.md. Do not re-introduce a separate segmentation window.
- Non-regression guarantee: Removing `code/segmentation_overview.py` + its button left the Segment stage, the report section, and the renderer untouched; the renderer is unused until the unified map app calls it.

## Segmentation map must be MBTiles, not on-the-fly vector dissolve (2026-06-06)

- Rule: Render the segmentation map as pre-rendered raster MBTiles (like the index layers), not as a client-side vector dissolve. The vector path (`segmentation.build_overview_geojson`) is now hard-capped at `max_cells=2_000_000` and only used for small levels / legend+stats; above the cap it returns `{"too_large": True, ...}` and the viewer shows a message.
- Why: The unified map's Segmentation tab dissolved `tbl_geocode_object` filtered to the level in-process. For basic_mosaic (~9M cells, ~1 GB+ geometry) the dissolve OOM'd the pywebview subprocess and the window died **silently** — diagnosed as "is segmentation crashing the whole thing?" Yes. The Overview/Results tab was an empty stub at the time, so it was not the cause. The index layers already avoid this by using `tiles_create_raster.py` → `output/mbtiles/<group>_<kind>.mbtiles`; segmentation was the only layer doing client vector.
- How to apply: Add categorical layer modes to `tiles_create_raster.py` (`seg_signatures`, `seg_clusters`) that join `tbl_flat` (group, geometry, code) with `tbl_segmentation/<group>.parquet` (code → signature / cluster_id) and colour per `segmentation._signature_colour` (A–E) / cluster palette → `<group>_seg_signatures.mbtiles` / `<group>_seg_clusters.mbtiles`. Generate in the Tiles stage when segmentation outputs exist. Consume as raster tile layers in map_overview / combined_map; take the legend + zones table from the tiny `tbl_segmentation_profiles` (no vector needed). Then a re-run of the Tiles stage is required to produce the segmentation tiles.
- Non-regression guarantee: The size guard is the only behaviour change shipped so far; small levels still render as vector exactly as before. The MBTiles work is planned, not yet built.

## Segmentation MBTiles auto-generate in the Tiles stage (2026-06-06, Phase A)

- Rule: `tiles_create_raster.py` renders `<group>_seg_signatures.mbtiles` (always) and `<group>_seg_clusters.mbtiles` (when cluster_id present) for any group that has `tbl_segmentation/<group>.parquet`. No flag — auto-gated on the file's existence (operator asked for auto-generate). Re-run the Tiles stage to produce them after segmenting.
- Why: The segmentation map must be pre-rendered raster like every other index layer (see 2026-06-06 entry on the silent vector-dissolve crash). Categorical colours are precomputed per feature in `main()` (signature ramp via `segmentation._signature_colour`, cluster palette via `segmentation._overview_colour`) and shipped to tile workers through a new `colors_by_mode` arg → `_G_COLORS_BY_MODE`; `_render_one_tile` paints `mode.startswith("seg_")` from that per-feature RGBA. `code` was added to the optional read columns (guarded) to join tbl_flat geometry ↔ segmentation category.
- How to apply: The viewer consumes `output/mbtiles/<group>_seg_{signatures,clusters}.mbtiles` as raster tile layers (Phase B). Legend + zones come from `tbl_segmentation_profiles` (no vector). The vector `build_overview_geojson` stays only for small levels / fallback and is hard-capped at 2M cells.
- Non-regression guarantee: Seg layers only render when a segmentation partition exists; all existing index layers are unchanged. `code` is optional, so datasets without it still tile.

## combined_map tile-URL parsing + false-positive validation (2026-06-07)

- Rule: combined_map's tile route is `/tiles/<name>/<z>/<x>/<y>.png` → **5** path segments (map_overview's is `/tiles/<kind>/<cat>/<z>/<x>/<y>` → 6). Parse with `len(parts)==5` and z=parts[2], x=parts[3], y=parts[4]. And when validating a tile server, NEVER use the blank/placeholder PNG as the stored test tile — use distinct bytes, else a blank-on-miss path returns identical bytes and the test false-passes.
- Why: combined_map was copied from map_overview's handler with `len(parts)!=6` and z/x/y at parts[3..5]; every overlay tile fell through to blank, so the Overview map zoomed (basemap OK, bounds from metadata) but no layer ever drew. My synthetic test stored `BLANK_PNG` as the tile content, so "served == stored" was true even though it served the miss-blank — the bug shipped. Caught only when the operator reported "no map on any layer". Re-validated against a real mbtiles with the actual stored blob (890 bytes, matched).
- Also: read A–E `category_colour` (hex like `#bd0026`) with `inline_comment_prefixes=(';',)` only — including `'#'` strips the value to empty.
- How to apply: For any loopback tile server, assert a served tile equals the real SQLite blob (distinct, non-blank) at a known (z,x,y) with the TMS flip `tms_row=(1<<z)-1-y`. Keep combined_map's route shape in sync with the handler's segment count.
- Non-regression guarantee: n/a (bug fix). Index tiles now render; the A–E area chart (area_stats.json + config colours) is restored on the Overview tab.

## Segmentation robust to a corrupt tbl_stacked partition + combined_map Export PNG (2026-06-07)

- Rule: `segmentation._read_layer_stacked` tries the fast whole-dataset `pyarrow` read, and on failure falls back to a per-file read that **skips unreadable partitions with a named warning** (recommending re-Intersect) instead of aborting the whole Segment stage. One corrupt parquet must not kill segmentation.
- Why: The operator's Segment runs failed with "Parquet magic bytes not found in footer" — exactly 1 of 1363 `tbl_stacked` partitions (part_21506_…, 480 KB, truncated write on 2026-06-07 01:43) was corrupt, and `ds.dataset(dir).to_table()` opens all files so the bad one aborted the read → no `tbl_segmentation` → no seg tiles → blank Segmentation tab. After the fix, basic_mosaic reads 68,046,771 of 68,137,097 rows (the ~91k from the bad partition are dropped, logged). The corrupt partition is a latent landmine for any flatten/backfill re-read too — the real fix is to re-run Intersect (Stage 2); do NOT silently delete it.
- How to apply: After a corruption warning, re-run Intersect to rebuild `tbl_stacked`. combined_map also gained an Export PNG button (html2canvas of the active tab's map → `Api.save_png` Save dialog), matching map_overview / asset_map_view.
- Non-regression guarantee: Clean stores take the fast path unchanged; the per-file fallback only triggers when the whole-dataset read raises.

## Flatten pre-flight swap gate must consider dataset size (2026-06-07)

- Rule: The flatten pre-flight swap-residue gate aborts only when `swap_used > flatten_preflight_max_swap_gb` AND there is insufficient real RAM headroom (`avail < need`). A standalone absolute swap check must not block flatten on a host that has plenty of free RAM for the job.
- Why: After processing a huge dataset (EACOP, 9M cells), macOS left ~6.7 GB of stale swap. A subsequent tiny run (Zanzibar demo, tbl_stacked = 1.9 MB) then hit `[flatten] PRE-FLIGHT ABORT: swap_used 6.7 GB > 5.0 GB` even though avail was 45.9 GB and the job needed ~8.6 GB. Flatten never produced tbl_flat → Tiles skipped ("tbl_flat.parquet is missing") → both the Results map and the unified Overview/Segmentation map panes were empty, and analysis failed ("Presentation table missing"). The vm_pct/headroom gate had already been made dataset-aware earlier; the swap gate had not.
- How to apply: When a host shows high swap but maps/flatten fail on a small project, it's stale swap residue, not a real shortage. The gate now self-clears when avail RAM covers the need; otherwise raise `flatten_preflight_max_swap_gb` in config.ini or restart to drain swap. Re-run Flatten→Tiles (Advanced mode, Flatten onward — do NOT re-check Prep, which wipes tbl_stacked) to rebuild tbl_flat and the tiles.

## Lines/analysis-only run wiped Maps tiles (2026-06-28)

- Rule: Delete `output/mbtiles` at run start ONLY when a tile-feeding stage (Data or Classification) is reprocessed without re-running Tiles — gate the cleanup on `(plan.run_data or plan.run_classification) and not plan.run_tiles`, never on `not plan.run_tiles` alone.
- Why: The pre-run "stale tiles" cleanup in `processing_pipeline_run.py` wiped all mbtiles whenever Tiles was unchecked. Running only Lines (8) + Analysis (9) — which write tbl_segments*/tbl_analysis_* and never touch tbl_flat/tbl_seg_mv/tbl_segmentation — also triggered it, so an operator "catching up" on lines/areas destroyed every tile. The Maps **Overview** lists its selectable geocode layers from the mbtiles filenames (`combined_map.mbtiles_catalog`) and the **Segmentation** map needs the seg mbtiles, so both panes went empty with no default/selectable geocodes.
- How to apply: When editing the tile cleanup, keep it gated on a tile-feeding stage actually running. Keep the Tiles "will be deleted" UI warning (`_sync_tiles_warning`) gated the same way via `feeds_tiles`, and re-run it on the data/classification checkbox signals (not just the Tiles toggle).
- Non-regression guarantee: A Data/Classification run with Tiles left unchecked still wipes the now-stale tiles exactly as before.
- Non-regression guarantee: On genuinely constrained hosts (avail < need) with high swap, flatten still aborts exactly as before — the protection is unchanged for the case it was designed for.

## Unified Maps window: one helper exe replaces two viewers + lazy-tab zoom fix (2026-06-08)

- Rule: The "Maps" window (`combined_map.py`) is the single map app — Overview (results) + Segmentation + Assets tabs, Overview default. It replaces the two standalone viewers `map_overview.py` and `asset_map_view.py`, which are no longer built. In `devtools/build_all.py` the helper list ships `combined_map` (not the two old viewers), and `combined_map` is in `force_webview`; the two `.py` files stay on disk as reference only. When fitting bounds on a lazily-shown tab, call `map.invalidateSize()` immediately before `fitBounds`.
- Why: (1) Compiled-size: each helper is its own ~1–2 GB GIS-heavy PyInstaller onefile; two map exes → one is a real footprint cut (operator's stated goal). `combined_map` was also missing from the build list entirely, so a frozen build's Maps button would have launched nothing. `segmentation.py` keeps sklearn/libpysal/geopandas lazily imported, so following `import segmentation` does not drag clustering libs into the map exe. (2) Zoom: a hidden tab's Leaflet container reports a stale/zero size; the async data chain (`seg_levels→…→fitBounds`) can resolve before `showTab`'s 60 ms `invalidateSize()` timer fires, so `fitBounds` computes the wrong zoom and the map never frames the default level. Operator: "Segmentation map does not zoom to the default choice."
- How to apply: Add new map tabs/exes to the `helpers` list AND `force_webview` in build_all.py; never assume a webview helper is auto-detected. For any fit on a tab that starts hidden, `invalidateSize()` then `fitBounds` (done at the seg-raster, seg-vector, and asset fit points). Removed dead handlers `open_maps_overview`/`open_asset_layers_viewer` from mesa.py.
- Non-regression guarantee: Running from source is unaffected (helpers resolve to `.py`); the two old viewer modules are retained, just unbuilt. The `invalidateSize` add is defensive — it cannot mis-fit a correctly-sized map.

## PySide6 bundling in standalone helpers (2026-06-08)

- Rule: In `devtools/build_all.py`, `helper_collects_for()` now adds `COLLECT_PYSIDE6` only when the helper actually uses Qt (`uses_qt`: import scan for `PySide6`/`PyQt5`/`PyQt6` plus an explicit `force_qt` set), mirroring the existing `uses_gis`/`uses_webview` pattern. Previously PySide6 was collected unconditionally for every helper.
- Why: PySide6 (Qt) costs ~230 MB per frozen onefile, measured: a full build gave `analysis_setup.exe` 416.5 MB (GIS+webview+Qt) vs `combined_map.exe` 184.8 MB (GIS+webview, no Qt) — the ~231 MB delta is Qt. Of the 4 standalone onefile helpers, only `analysis_setup` renders a Qt UI (top-level `from PySide6.QtWidgets import ...`, analysis_setup.py:48). `combined_map` and `line_manage` are webview-based (pywebview → WebView2, no Qt) and `tiles_create_raster` is a pure geopandas/shapely/PIL/sqlite tile generator with no UI — all three were carrying a full unused Qt copy. Onefile exes cannot share DLLs, so each duplicate is paid in full; gating Qt cut ~690 MB total across the three. This is the low-risk slice of the larger GIS+Qt duplication across mesa.exe/combined_map.exe/tiles_create_raster.exe (the bigger lever — onedir DLL sharing — was deferred past the stabilization phase).
- How to apply: New Qt-based standalone helpers must either import PySide6 at top level (auto-detected) or be added to `force_qt`. If a webview/compute helper starts failing at frozen startup with a Qt import error, that means it gained a lazy Qt dependency — add it to `force_qt` rather than reverting the gate. Verify a build of `tiles_create_raster`/`combined_map`/`line_manage` still launches before shipping.
- Non-regression guarantee: `analysis_setup` (the only Qt helper) is in `force_qt`, so its bundle is unchanged; `MESA_HELPERS_FULL_DEPS=1` still forces PySide6 for all helpers via the `HELPERS_FULL_DEPS` branch.

## Build timing + persistent build history (2026-06-08)

- Rule: `devtools/build_all.py` now logs total wall-clock build time at the end of a run and appends one record per build to `D:/dist/build_history.log` (timestamp, scope flags, total, per-task durations). The history file lives at `DIST_FOLDER_ROOT` (D:/dist), one level above `FINAL_DIST` (D:/dist/mesa), specifically because that parent is wiped by neither a main build (`shutil.rmtree(FINAL_DIST)`) nor `MESA_BUILD_CLEAN` (`shutil.rmtree(BUILD_FOLDER_ROOT)`), so durations accumulate across runs.
- Why: There was no record of how long a build took, so "how long did the last build take?" was unanswerable. Baseline from the first full run: ~9m49s total (4 workers, 5 tasks: 4 helpers + main). Wall-clock exceeds the slowest single task because there are 5 tasks on 4 workers (main waits for a slot) plus the post-build onedir flatten + resource copy. `mesa.exe` itself is lean (~34 MB; GIS lazy-loaded from `_internal`).
- How to apply: To check past build times, read `D:/dist/build_history.log`. The history write is best-effort (wrapped in try/except) so a logging failure can never fail the build. Per-task timings are collected thread-safely via `record_timing()` since helpers build in parallel.

## MESA version is config-driven; build number is auto-stamped (2026-06-08)

- Rule: The user-facing main version is the single key `mesa_version` in `config.ini` (now 5.2), read by `mesa.py` (`config['DEFAULT'].get('mesa_version', 'MESA 5')`) and the helpers' `_mesa_version_label()`. There is no hardcoded version string anywhere else — bumping the version is a one-line config edit. The build number is NOT in config: `write_build_metadata()` auto-stamps an Oslo-time `YYYY-MM-DD HH:MM` into `build_info.json` at build time, shown to users appended after the version (`f"{version_text} Build {packaged_build_timestamp}"`, mesa.py).
- Why: Operator distinction — "build number can live in config, but the main version number is the one shown to users." In practice the main version is the only thing edited by hand; the build stamp follows automatically each compile.
- How to apply: To change the displayed version, edit `mesa_version` in `config.ini` only; do not hunt for hardcoded copies (there are none) and do not touch `build_info.json` (regenerated every build).

## Sensitivity generalisation: multivariate segmentation as a separate v2 feature (2026-06-10)

- Rule: The new multivariate "sensitivity generalisation" capability is **additive and lives in its own `tbl_seg_mv*` namespace** — it does not touch the shipped `tbl_segmentation*` feature (`code/segmentation.py`, the Segment pipeline stage, the Maps Segmentation tab, the report's "Segmentation" section). Two new standalone helpers: `code/segmentation_run.py` (heavy compute) and `code/segmentation_setup.py` (light Qt config UI). Both are subprocess-launched, NOT in `INPROCESS_HELPERS`, so the clustering stack (scikit-learn/scipy/libpysal/spopt/hdbscan) bundles into `segmentation_run.exe` only and never bloats `mesa.exe`.
- Why: A pasted greenfield spec asked to "add" segmentation and "productionise the prototype in code/devtools/test_segmentation.py" — but segmentation already shipped, the prototype was already promoted into `code/segmentation.py`, and the spec's proposed `tbl_segmentation` schema (one row per polygon×method×n_clusters×run_id) would have **broken** the Maps tab + report that read today's slim `tbl_segmentation/<layer>.parquet`. Operator chose a separate v2 rather than evolving the shipped one. Two spec assumptions were wrong about the repo and were corrected: (1) "Ollama already in the stack" — it isn't; only OpenAI is, so AI labels call Ollama first then fall back to the existing OpenAI key resolution, default OFF; (2) SKATER was a documented non-goal at scale (basic_mosaic ≈ 11k–9M polygons) — so `fit_spatial` guards on `segmv_skater_max_polys` (default 50k) and routes to KMeans + post-hoc Queen-contiguity merge above it, logging which path ran. Also discovered: sklearn/libpysal/spopt/hdbscan were entirely MISSING from `.venv`, which means the shipped `clusters` segmentation mode also silently degrades to a no-op here until they're installed.
- How to apply: When a brief says "add"/"productionise" something segmentation-shaped, first check `devtools/docs/SEGMENTATION_INTEGRATION_PLAN.md` §7 and `code/segmentation.py` — most of it exists. Memory discipline holds: `segmentation_run._read*` reads `tbl_stacked` partition-by-partition with a pyarrow `name_gis_geocodegroup` filter, never whole. New table names must avoid the `tbl_segmentation*` prefix. New config keys use the `segmv_` prefix (distinct from shipped `segment_*`). In `build_all.py`, `segmentation_run` auto-detects its lazy heavy imports but `segmentation_setup` reaches pandas/pyarrow/GIS only indirectly (via `import segmentation`), so its data stack is force-bundled while sklearn is deliberately kept out of it.
- Non-regression guarantee: Everything defaults OFF; with no `tbl_seg_mv*` tables present, `generate_report(include_segmentation_mv=...)` skips cleanly and the report is byte-identical. The shipped segmentation stage/tab/section and their tables are untouched. Reproducibility: a fixed `run_id` + fixed seeds reproduce identical cluster assignments (verified: same MD5 of `cluster_id` across re-runs on QDGC_L6).

## Stylesheet import dragged GIS into helpers (2026-06-11)

- Rule: Keep the shared Qt look-and-feel in the GIS-free `code/ui_style.py`; never import `apply_shared_stylesheet` from `asset_manage`. Supersedes the 2026-04-12 note ("Shared styling is now applied through `asset_manage.apply_shared_stylesheet()`") — the function moved; `asset_manage` now re-exports it from `ui_style`.
- Why: `asset_manage.py` imports `fiona`/`geopandas`/`shapely` at module top. Nine modules imported `apply_shared_stylesheet` *from* `asset_manage` just to set a QSS string, so every helper that wanted the look-and-feel transitively bundled the whole ~250 MB GIS stack. The Classification config UI (`segmentation_setup`) packaged at 425 MB for this reason — and excluding only its (lazy) clustering stack saved 0 MB, because GIS, not sklearn/scipy, was the bulk and GIS entered solely via the stylesheet import. Extracting `ASSET_STYLESHEET` + `_generate_indicator_stylesheet` + `apply_shared_stylesheet` into `ui_style.py` (PySide6 + stdlib only) removed every module-level path to GIS in `segmentation_setup`. Verified at runtime: everything its config UI calls (`segmentation.list_geocode_layers`, `segmentation_run.detect_pressure_columns`, `params_from_config`) reads parquet via pandas/pyarrow only, never geopandas — the GIS + clustering work runs solely in the spawned `segmentation_run.exe`. So `helper_exclude_modules["segmentation_setup"]` in `devtools/build_all.py` now drops geopandas/fiona/shapely/pyproj/pyogrio + sklearn/scipy/libpysal/spopt/hdbscan, and the GIS *force-collect* for `segmentation_setup` was removed (gated to `segmentation_run` only) so the collect+exclude no longer contradict. The exclude is still required: PyInstaller's static graph discovers the lazy imports nested inside `segmentation_run.py`'s functions and would bundle them otherwise.
- How to apply: When a light helper needs only the shared styling, import from `ui_style`. Before importing anything from `asset_manage` into a config-only helper, check whether it drags the GIS stack — prefer the lightweight shared module. After such a change, confirm with `import <helper>` that `sys.modules` has no GIS/compute libs AND that the helper's runtime-called functions don't lazy-import them, before adding those libs to `--exclude-module`. Never pair `--collect-*` and `--exclude-module` for the same lib on the same helper.
- Non-regression guarantee: `ui_style.apply_shared_stylesheet` is the same function moved verbatim (5844-char stylesheet, byte-identical QSS); `asset_manage` re-exports `ASSET_STYLESHEET`/`_generate_indicator_stylesheet`/`apply_shared_stylesheet` from `ui_style`, so any remaining `from asset_manage import apply_shared_stylesheet` still resolves. The other eight importers were redirected to `ui_style`. Only `segmentation_setup`'s exclude set changed in the build; other helpers' collect flags are untouched. The actual MB reduction is pending a rebuild+launch test of `segmentation_setup`.

## `--exclude-module scipy` + `--collect-all scipy` = half-broken scipy (2026-06-11)

- Rule: Never apply both `--exclude-module X` and `--collect-all X` to the same PyInstaller build. If a helper bundles the sklearn stack (`COLLECT_SKLEARN`, which `--collect-all scipy`), strip `scipy` from `HELPER_EXCLUDES` for that helper. `build_helper` in `devtools/build_all.py` now does this via `_helper_needs_scipy()` + `_strip_exclude_module()`.
- Why: `scipy` is in the default `HELPER_EXCLUDES` (dead weight in the GIS/UI helpers). `segmentation_run` also gets `COLLECT_SKLEARN` → `--collect-all scipy`. With both flags, `--collect-all` copies scipy's `.py` files (so `scipy/__init__.py` is physically in the bundle) but `--exclude-module scipy` prunes the package from the import graph, dropping its compiled C-extensions (`scipy/_lib/_ccallback_c.pyd`). Result: a scipy that imports far enough to run `scipy/__init__.py` then raises `ImportError: cannot import name '_ccallback_c'` → `The scipy install you are using seems to be broken`. Surfaced the first time `segmentation_run.exe` actually ran clustering (`fit_attribute` → `sklearn.cluster.KMeans` → `scipy.sparse`); it was latent before because sklearn/scipy were missing from `.venv` (see the 2026-06-10 entry) so the clustering path never executed.
- How to apply: When adding a helper that needs a compiled scientific lib already in `HELPER_EXCLUDES` (scipy, and watch for numpy/numba/etc.), add it to `_helper_needs_scipy` (or a sibling predicate) so the exclude is removed for that helper only. Verify with the per-helper flag dump (count `--exclude-module scipy` pairs = 0 and `--collect-all scipy` present for the helper that needs it; still excluded for the ones that don't). A bundled package whose `__init__.py` runs but whose `_*.pyd` extension is "cannot import name" almost always means collect+exclude fighting over the same module.
- Non-regression guarantee: The strip is gated to helpers matching `_helper_needs_scipy` (`segmentation_run`, or any whose source imports sklearn/libpysal/spopt/scipy). `segmentation_setup`, `combined_map`, and the GIS/UI helpers keep `scipy` excluded (verified). Only segmentation_run's flag set changes; it must be rebuilt to pick up a working scipy.

## Classification reworked to (importance, susceptibility) histogram clustering (2026-06-11)

- Rule: The Classification engine (`code/segmentation_run.py`) clusters cells by the *shape* of their joint (importance, susceptibility) histogram via a BIC-selected Gaussian Mixture, not by sensitivity codes. The Segment sub-stage (`segmentation.py`) is signatures-only (its `clusters` mode + libpysal/spopt/hdbscan/Ward/SKATER/KMeans code was removed). Signatures are kept as the deterministic reference the clustering is validated against (ARI/NMI).
- Why: the A–E sensitivity code is the product importance×susceptibility, so (imp 5, sus 1) and (imp 1, sus 5) are indistinguishable to any code-based clustering. Histogram clustering separates them (verified from source: two synthetic cells with identical products land in different clusters). Per-asset intersection area is NOT persisted in `tbl_stacked` (only the cell `area_m2`, constant per code; the intersect geometry is discarded — [processing_internal.py](code/processing_internal.py) sjoin), so the spec's per-asset area weighting + area-based coverage index are not buildable from existing data. Operator chose **count-based histograms + cell-area-weighted aggregation**: within a cell each overlap weighs equally; cross-cell comparability (QDGC vs basic_mosaic) is restored by area-weighting the per-type fingerprint/means; coverage index = stack depth (overlap count) as the intensity proxy. importance/susceptibility are materialised per `tbl_stacked` row (asset-group level), or backfill from `tbl_asset_group` via `ref_asset_group`.
- How to apply: bins come from `[VALID_VALUES] valid_input` (fallback 1–5). New config keys `segmv_k_range`/`segmv_transform`/`segmv_coverage_weight`; removed `segmv_n_clusters`/`segmv_method`/`segmv_features`/`segmv_skater_max_polys`. Outputs `tbl_seg_mv` (cluster_id, cluster_label, p_max, entropy, coverage_index, top_bins) + `tbl_seg_mv_profile` (25-bin fingerprint + area-weighted means). When the engine's output schema changes again, `_write_parquet_coexist` now REPLACES the file if column sets differ (prior runs from an older engine are obsolete) rather than producing a half-NaN union.
- Non-regression guarantee: a fixed `seed=42` + same `run_id` reproduce byte-identical assignments (verified: identical MD5 across two runs). GMM is deterministic with `n_init=5, random_state=42`. Signatures, the shipped `tbl_segmentation*` tables, and the report's signatures section are untouched.

## segmv_* config keys were written under [VALID_VALUES], invisible to cfg["DEFAULT"] (2026-06-11)

- Rule: `segmentation_setup._update_config` must insert NEW keys inside the `[DEFAULT]` section, not append at EOF. `params_from_config` reads `cfg["DEFAULT"].get(...)`, and configparser does NOT expose keys from other sections there.
- Why: the local `_update_config` appended unknown keys at end-of-file, which fell under the LAST section header (`[VALID_VALUES]`). So every saved `segmv_*` setting landed in `[VALID_VALUES]`, and `cfg["DEFAULT"].get("segmv_…")` returned `None` — the shipped Classification UI's saved settings were silently ignored and defaults always used (verified: `cfg['DEFAULT'].get('segmv_geocode_layer')` → None while `cfg['VALID_VALUES']` had it). The fix walks to the end of the `[DEFAULT]` block and inserts there; the repo `config.ini` keys were relocated from `[VALID_VALUES]` to `[DEFAULT]`.
- How to apply: any comment-preserving INI writer that "appends if not found" must target the intended section explicitly — appending at EOF is only correct when the target section is last. When a config value seems ignored, check which section the key physically sits in vs which section the reader queries.

## Mirror-ghost tiles: the XYZ "fallback" against a TMS mbtiles paints a vertical mirror (2026-06-11)

- Rule: When serving mbtiles to Leaflet (`tms:false`), pick the row by the file's declared `scheme` and do ONE lookup. Never "try TMS row, then fall back to the raw Y row" — for a TMS-stored db the raw-Y lookup hits the *vertical mirror* of a real row and renders a flipped ghost copy of the data on top of the correct rendering.
- Why: MESA's `tiles_create_raster.writer_process` always stores TMS (`tms_y = 2^z-1-y`, metadata `scheme=tms`). The `combined_map.py` `/tiles/` handler queried `tile_row = 2^z-1-y` (correct) and, on a miss, retried `tile_row = y`. A raw-Y query hits iff `y` ∈ stored TMS rows, i.e. iff there is data at XYZ row `2^z-1-y` — exactly the requested tile's vertical mirror. So real data drew via query A and an upside-down ghost drew via query B wherever the dataset's mirror overlapped empty rows. Reported as "speiling av data" in the Overview and Segmentation tabs (both raster-tile paths through this handler); Classification was unaffected because it renders vector GeoJSON, never `/tiles/`. The same retry-the-raw-row pattern lived in two sibling handlers — `map_overview.py` (the superseded standalone viewer) and `analysis_setup.py` (the Analysis window) — and all three were fixed in the same change so no revived window reintroduces the ghost.
- How to apply: any mbtiles→Leaflet shim must read `metadata.scheme` (default `tms`) and map XYZ→storage once: `tile_row = y if scheme=='xyz' else 2^z-1-y`. If you genuinely need to support both schemes for unknown third-party files, detect the scheme from metadata or a one-time probe — do NOT blind-fallback to the opposite row, which silently mirrors instead of failing.
- Non-regression guarantee: MESA's own tiles declare `scheme=tms`, so the new single-lookup path is byte-identical to the old query-A result for every real tile; only the spurious query-B ghosts disappear. XYZ-declared files are now served correctly instead of mirrored.

## Sample-data generator: organic + scattered, not templates rotated round the centre (2026-06-11)

- Rule: `devtools/make_sample_packages.py` must generate each asset patch as a UNIQUE procedural outline and scatter features across the whole AOI. Do not reuse a few OSM templates rotated/scaled (looks copy-pasted), and do not place every feature near the centre or build linear features by rotating a strip about the AOI centre.
- Why: the first version placed all blobs within ±7 km of centre and built rivers/strips originating at centre then `shapely.affinity.rotate(..., origin=centre)` — producing a thick central blob with radial spokes (a "star") and no outliers, from a handful of templates that read as copies. Operator rejected it: wanted creativity, randomness, and satellites. Rewrite: `organic_blob` builds the boundary radius as `1 + Σ harmonic sines` (random amplitude/phase/vertex-count + anisotropy + free rotation), so every patch differs; `organic_line` meanders along its OWN random heading (never rotated about the centre). Placement uses theme-level shared *hotspots* at mid-radius for distributed overlap plus per-group *satellites* drawn from an outer annulus (`SAT_RING`). Verified on mount_kenya: 82 objects, median 9.4 km from centre, only 10% within 4 km, 65% beyond 8 km, reaching the 13 km rim; 82/82 unique areas.
- How to apply: keep it deterministic via a fixed `SEED` + per-(theme,group) `random.Random`, NOT `Math.random`/`Date`-style sources. Overlap depth (what the Classification histogram needs) comes from groups sharing hotspots, not from piling at the centre — tune `N_HOTSPOTS`/`HOTSPOT_*`/`SAT_RING`, not a central radius. The `.gpkg`/`.xlsx` outputs are gitignored (`*.gpkg`, `*.xlsx`); only the generator, per-theme `README.md`, and `sample_data/_preview.png` are tracked, so the data is reproduced by re-running the script. `osm_shape_templates.json` + `fetch_osm_templates.py` are now unused by this generator (kept, not imported).

## Segment config offered retired modes; "# objects vs # groups" per cell (2026-06-11)

- Rule: After retiring a stage mode, prune its config + UI controls, not just the engine. The Process dialog's "Segment mode" combo (Signatures / Clustering / Both) plus the "Zones (k)" spinner kept offering algorithmic clustering after `segment_layer` went signatures-only — so selecting "Both" wrote only signatures and only one mode showed in Maps. Fixed: removed the combo + spinner ([processing_pipeline_run.py], replaced with a static note), dropped `segment_n_clusters`/`segment_spatial_method` from `config.ini` (code reads them with safe defaults), and consumers now pass `segment_mode="signatures"`, `segment_n_clusters=0`. The `ProcessPlan` fields stay for signature stability. Clustering now lives only in the Classification tool (`segmv_*`).
- Why (data model): the Overview "# asset groups" vs "# asset objects" layers are `asset_groups_total` = `nunique(ref_asset_group)` per cell vs `assets_overlap_total` = count of overlapping object rows (`value_counts` of `code`) — so `# objects ≥ # groups`, equal **only** when each group covers the cell with a single object. This is NOT a basic_mosaic property; it depends on intra-group object overlap. The new sample data deliberately self-overlaps within a group, so objects > groups there.
- How to apply: when a "mode"/"method" is retired, grep the *setup* UIs (`*_setup.py`, `processing_pipeline_run.py`) and `config.ini` for its controls/keys, not just the worker. When wiring an A–E "equal counts?" question, check `nunique` vs row-count semantics in [processing_internal.py](code/processing_internal.py) `_flatten`/`tbl_flat` build.

## Hardcoded wiki anchors break when headings are renamed (2026-06-11)

- Rule: `_InfoCircleLabel`/status-row wiki links hardcode GitHub heading anchors, which silently rot when a wiki heading changes. The Status tab linked `User-interface#prepare-data` / `#configure-analysis` / `#run-processing`, but the headings had become "Prepare data (step 1)" / "Configure (step 2)" / "Process (step 3)" → real slugs `prepare-data-step-1` / `configure-step-2` / `process-step-3`. GitHub silently scrolls to page top on a bad anchor, so it looks "fine."
- Why: GitHub slug = heading lowercased, punctuation/parentheses dropped, spaces→hyphens, backticks stripped. "Process (step 3)" → `process-step-3`. Any "(step N)"/parenthetical in a heading makes a fragile anchor. Two further links in the **superseded** `map_overview.py` point at `Indexes#importance-index-index_importance` / `#sensitivity-index-index_sensitivity`, but those indices were **removed in 5.2** — dead links to a deleted feature, not just a moved anchor.
- How to apply: validate anchors against the wiki rather than eyeballing — extract `^#{1,6}` headings from each `mesa.wiki/<Page>.md`, compute the slug, and assert every code anchor is in the set (a ~20-line script does it). Re-run after any wiki heading edit. New info-icons added to the four Workflows step cards link to the matching `User-interface#*-step-N` anchors (validated). When adding a `_InfoCircleLabel`, prefer linking a stable page over a parenthetical sub-heading. The info-icon widget is now shared as `ui_style.InfoCircleLabel` (used by `segmentation_setup`; `mesa._InfoCircleLabel` is the same painted-circle, kept to avoid churn).

## Removed the two superseded map viewers; vulture must scan the whole tree (2026-06-11)

- Rule: `code/map_overview.py` and `code/asset_map_view.py` are GONE — they were superseded by `combined_map.py` (unified Maps window) and `asset_styling.py` (the promoted AI-styling helper), were never built (not in `build_all.py` helpers), and had zero real imports (only comment/docstring mentions). ~5,200 lines removed. Don't resurrect them; extend `combined_map.py` / `asset_styling.py` instead.
- Why: a reference-graph pass (for each `code/*.py` stem, grep the tree for real `import`/`from`/launch-string refs, excluding self) found both modules referenced only inside comments. Confirmed by `grep -E '^\s*(import|from)\s+(map_overview|asset_map_view)'` → none. Also stripped 5 genuinely-unused imports (mesa.py `QPalette`/`QSize`, geocode_manage.py `Union`/`linemerge`, processing_internal.py `_traceback`).
- How to apply: run `vulture code/ mesa.py --min-confidence 90` over the WHOLE tree at once — vulture matches used names across all scanned paths, so scanning files in isolation invents false positives (e.g. `locale`/`PILImage` flagged alone but used elsewhere). Always re-verify a flagged import with a whole-tree grep before deleting. Vulture does NOT flag dead functions in pywebview `Api` classes (called from JS) or Qt overrides (`paintEvent`/`eventFilter`) — those need manual JS/signal cross-referencing, so a clean 90% report ≠ "no dead functions". A flagged unused *parameter* of a public entry point (e.g. `report_generate.launch_gui(theme=...)`) is an API signature, not dead code — keep it.

## Classification map: dissolve doesn't scale, rasterise like seg_clusters (2026-06-21)

- Rule: For big geocode layers, the Classification (segmv) map must render as a raster MBTiles, not dissolved vectors. The Tiles stage now builds `<slug>_segmv_<run_id>.mbtiles` (newest run per group) from `tbl_seg_mv`, and `combined_map.segmv_layer` returns a `raster` ref for layers over `SEGMV_MAX_FEATURES` (250k) when that file exists, else `too_large` pointing at the Tiles stage. Layers under the cap keep the per-cell vector path (Certainty/p_max colouring + click-identify) unchanged.
- Why: the "too many polygons" message on basic_mosaic (9M cells) was the per-cell feature count, NOT a clustering failure — clustering correctly produced 15 types. The intuitive fix (dissolve cells per class to 15 multipolygons) does NOT work: clustering yields salt-and-pepper patches scattered across the AOI, so the dissolved boundary is dominated by perimeter, not area. Measured on H3_R9 (1M cells): dissolve→15 parts but 858k vertices / 36 MB GeoJSON, and `simplify` bottoms out at ~170k verts / 7 MB even at 1.1 km tolerance. basic_mosaic (9M, finer grid) would be 100+ MB and minutes per map-open. Raster is O(pixels), independent of cell count and fragmentation, and matches how MESA already shows big layers (seg_signatures/seg_clusters).
- How to apply: a new categorical tile layer only needs a `colors_by_mode[<mode>]` entry (per-feature RGBA) — the worker render branch is now `elif mode in _G_COLORS_BY_MODE` (was `mode.startswith("seg_")`), so any mode key flows through. Name per-run mbtiles so each classification run is independently viewable; the Tiles stage rasterises only the newest run per group. `slugify` (tiles) and `_safe_name` (combined_map) must agree on the layer slug for the filename to round-trip — they do for current group names. The raster view has no per-cell identify (inherent at that scale); keep the vector path for layers under the cap. Certainty (p_max) parity is preserved by baking a *second* raster `<slug>_segmv_<run>_cert.mbtiles` (ramp via `cert_rgba`, which MUST mirror `combined_map._certainty_colour`/`_CERT_STOPS`); the map's "Colour by" toggle swaps tile layers (`applyClassRaster`) just as it restyles vectors. So any per-cell *continuous* field can get parity with one extra baked raster — but each doubles tile build time + storage, so add them deliberately.
- Non-regression guarantee: layers ≤ 250k cells are untouched (vector per-cell path); the change is additive (one extra mode in the Tiles stage + a raster branch in segmv_layer), so existing seg/index/sensitivity tiles and small-layer Classification rendering behave exactly as before.

## p_max saturates near 1.0; Certainty raster needs the full grid + a contrast stretch (2026-06-21)

- Rule: The Classification "Certainty (p_max)" view must (a) be rendered from the FULL geocode grid (`tbl_geocode_object`), not `tbl_flat`, and (b) contrast-stretch the ramp over `[_CERT_LO, 1.0]` (currently 0.95). No-data cells (no cluster) are painted grey `#cccccc`; the legend states the stretched range so it reads as "marginal differences near full confidence", not a 0..1 scale.
- Why: a diagonal-covariance GMM over 26 Hellinger dims is wildly overconfident — its posterior is a product over 26 dims, so p_max saturates: for basic_mosaic median p_max=1.0, 97% ≥0.999, 10.8% exactly 1.0; `entropy` (also stored) confirms it (median 0). An un-stretched ramp paints the whole map one dark green. Separately, the "white holes" the operator saw on the Certainty map were no-data cells: of 661,463 no-cluster cells, **638,777 are absent from `tbl_flat`** (cells with zero asset overlap never enter tbl_flat), so a tile renderer driven by tbl_flat geometry simply never draws them — they show the basemap through. Greying them requires rendering tbl_geocode_object geometry instead.
- How to apply: the segmv rasters are built in a DEDICATED pool in `tiles_create_raster.main()` (after the shared tbl_flat pool is freed) using `tbl_geocode_object` geometry for the group; cluster colours / p_max come from `tbl_seg_mv` keyed by `code`. Keep `_CERT_LO` identical in `tiles_create_raster` and `combined_map._certainty_colour` (raster vs vector parity) — they're two copies on purpose (importing combined_map into the tiles helper would drag in webview). If you ever want a genuinely informative per-cell uncertainty, p_max/entropy won't give it here; you'd need posterior calibration (e.g. temperature scaling) or full-covariance GMM, not a different stored column.
- Non-regression guarantee: only the segmv (Classification) rasters changed source/colour; sensitivity/index/totals/seg_* tiles still render from tbl_flat exactly as before. Layers ≤ `SEGMV_MAX_FEATURES` (250k) still use the per-cell vector path in the map.

## assetstotal ramp skew — one-colour Overview map (2026-06-24)

- Rule: The `assetstotal` raster ramp top (`vmax`) must be the **99th percentile** of `assets_overlap_total`, not the raw max, paired with a **log1p** scale. Use the shared `tiles_create_raster.assetstotal_vmax()` helper in BOTH the production `main()` minmax pass and the legacy `run_one_layer()` so render + legend metadata agree.
- Why: `assets_overlap_total` is extreme right-skewed — on basic_mosaic p50=5, p95=8, p99=48, but max=460,829 (≈9,600× p99, one cell). With `vmax=np.nanmax`, ~99% of cells (values 1–20) squash into the palest sliver of the ramp → the operator sees "one value". log1p alone is not enough (log1p(48)/log1p(460829)≈0.30, so the bulk still lives in the bottom third). p99 cap + log1p spreads the bulk across the full pale→dark range; denser cells clamp to darkest. The calculation was never wrong — purely a ramp/normalisation issue.
- How to apply: when a totals/count raster looks flat, check the percentile spread before suspecting the data (`s.quantile([.5,.95,.99,1.0])`). The production tile path is `main()` (its inner `_run_layer`), NOT `run_one_layer()` (dead) — fix the `group_minmax` computation there. Legend stops + `mesa_value_max` mbtiles metadata derive from the same `vmax`, so capping vmax fixes the legend too. The log1p scale + legend-metadata infrastructure (mesa_scale/value_min/value_max/legend keys, read by `combined_map._mbtiles_meta`) came from an out-of-scope parallel session but genuinely fit and were merged.
- Non-regression guarantee: only `assetstotal` changed (p99 + log1p); `groupstotal` stays linear raw-min/max (its range is 1–32, not skewed), and all other layers are untouched.

## Intersect per-worker memory — parent-side asset pre-filter (2026-06-24)

- Rule: The intersect/tbl_stacked stage must ship each worker only its chunk's asset SUBSET (parent-side `asset_data.iloc[sindex.intersection(chunk.total_bounds)]`), not broadcast the full asset layer to every worker. Size workers from `intersect_prefilter_worker_gb` (bounded per-chunk footprint), NOT the full-data estimate.
- Why: workers held the entire asset_df (~5.76 GB measured on basic_mosaic — ~96% of per-worker RSS; the per-chunk sjoin output is only ~0.11 GB). That made intersect memory-bound: the old `data_gb × stage2_worker_overhead_multiplier` estimate (~8 GB/worker) capped the pool to 3 workers on a 64 GB box, so 9.6 h of mostly volume÷3. The worker ALREADY filtered assets per chunk via `_POOL_ASSETS.sindex.intersection(bbox)` — only the memory was wasteful. Relocating that exact query to the parent and shipping the subset cuts per-worker RAM ~21x (max subset 164k vs 3.5M assets), making intersect CPU-bound (→ ~12-16 workers).
- How to apply: correctness is preserved exactly — same sindex bbox query, just relocated; boundary-spanning assets are included in every chunk their bbox touches, and each geocode lives in exactly one chunk, so every geocode×asset pair is computed once (proven: byte-identical output across real chunks incl. a dense 323k-row one). Platform-neutral (no fork/COW — works on Windows spawn). `stage2_worker_overhead_multiplier` is now INERT for intersect (the pre-filter bypasses the full-data estimate). New candidate bottleneck is parent task-feed (sindex+slice+pickle per chunk, ~tens of ms vs ~36 s/chunk worker time → unlikely to bind); if it does, pre-package subsets to disk (toggle), since disk is cheap (~1-1.5 GB of subset files). Correctness-proven; full-pipeline perf validation pending.
- Non-regression guarantee: a 2-tuple task (no subset) still falls back to the broadcast global, so the change is behavior-identical if the parent pre-filter is bypassed.

## Intersect pre-filter — first full-run validation (2026-06-25)

- Rule: When shipping per-chunk asset subsets to intersect workers, skip the REDUNDANT top-level sindex query. The parent already filters `pool_assets` to the chunk's bbox, so the worker's first `pool_assets.sindex.intersection(geocode_gdf.total_bounds)` rebuilds an index only to return the whole subset — pure overhead. Use the subset directly at the top level; build a sindex only for the recursive geocode/asset sub-splits that actually narrow the bbox.
- Why: the first full-scale validation (basic_mosaic, Python 3.14, 10 workers) measured per-worker throughput ~36k rows/min vs ~53k in the old 3-worker broadcast run — ~30% lower per worker. Root cause is structural: the OLD path built one big `_POOL_ASSETS.sindex` per worker PROCESS and reused it across all ~199 chunks that worker handled; the pre-filter ships raw subsets, so each worker now rebuilds a sindex PER CHUNK, and the first query of each chunk is redundant (see Rule). Net wall-clock still ~2.3x faster (10 workers × lower per-worker rate), projecting ~4–4.5 h vs the 9.6 h / 3-worker baseline — a real win, but scaling is sub-linear: partly the self-inflicted sindex rebuild, partly genuine Apple-Silicon unified-memory bandwidth contention at 10 concurrent sjoin workers.
- How to apply: when iterating the pre-filter, fix the redundant top-level query first (cheapest win toward the ~3x ideal). Also observed: with the pre-filter + bounded estimate (effective 1.5 GB), `max_workers = 12` resolved to **10** — the avail-RAM budget (momentary *available* memory at pool sizing, not total) still undercuts the configured/CPU ceiling even when total-RAM headroom is large. So `max_workers` is a ceiling the avail-budget can lower; to actually hit N workers, available RAM at sizing must be ≥ N × effective_worker_gb / mem_target. Memory stayed safe throughout (predicted peak ~51% of 64 GB, no swap), and the run produced output error-free across all 1992 chunks — confirming the byte-identical equivalence proof holds at full scale, not just the spot-check.
- Non-regression guarantee: the ~2.3x speedup and the memory safety are real regardless of the sindex fix; that fix only improves parallel efficiency further (toward ~3x). The 2-tuple fallback to the broadcast global remains, so disabling the pre-filter restores exact prior behaviour.

## Intersect pre-filter — validation COMPLETE, final numbers (2026-06-25)

- Supersedes the preliminary figures in the entry above (those were measured mid-run during a front-loaded dense band and read pessimistic).
- Rule: The parent-side per-chunk asset pre-filter is validated end-to-end — ship it. Final full-pipeline run on Python 3.14, basic_mosaic, 10 workers (max_workers=12 → 10 via the avail-RAM budget).
- Why / measured: intersect **3.02 h** (23:18→02:19) vs the old 3-worker broadcast run's **9.6 h** = **3.18x faster** — near the 10/3=3.33x ideal. Output is byte-identical at full scale: **1387 parts / 91,083,233 rows**, exactly matching the old run (and tbl_flat = 9,830,778 rows, also exact). Aggregate per-worker throughput is only ~5% below the old run (3.01 vs 3.16 M rows/h/worker) — the ~30% I measured mid-run was a dense-band artifact, so the redundant-sindex-rebuild overhead is MINOR, not the ~30% it appeared. Memory stayed ~52% all run, no swap (the 140 GB failure mode is gone). Zero errors across all 1992 chunks.
- How to apply: the optimization is proven; promote with confidence. The sindex-rebuild tidy (skip the redundant top-level query) is now a nice-to-have, not a priority. Cost skew is real and FRONT-LOADED here (dense chunks early, sparse back-half), so mid-run throughput swings wildly (1.6–55 chunks/min) — judge by the running average, not any single heartbeat.

## 3.14 subprocess launch — segmv/combined_map run on the wrong venv (2026-06-25)

- Rule: `mesa.py` must launch the Classification helper (`segmentation_run`) and the map viewer (`combined_map`) via `sys.executable`, like the processing pipeline does — not a hardcoded `.venv` path.
- Why: in the 3.14 validation run (launched on `.venv314`), intersect/flatten/tiles correctly inherited 3.14 (they spawn via `sys.executable`), but `segmentation_run` and `combined_map` were launched on the hardcoded `.venv` (Python **3.11**), whose pyogrio is ABI-broken (`numpy.core.multiarray failed to import`). segmv then threw a scary multi-frame Traceback at `export_gpkg → gdf.to_file(driver="GPKG") → import pyogrio` — NON-fatal (it recovered, wrote classification_results.gpkg, completed), but misleading. (Earlier I twice misdiagnosed this as a SKATER/hdbscan-extras gap; it is purely the wrong-interpreter launch.)
- How to apply: find the hardcoded `.venv/bin/python` (or equivalent) launch for these two helpers in `mesa.py` and switch to `sys.executable`. Then a 3.14 run is 3.14 end-to-end and the broken-3.11-pyogrio dependency disappears.

## 3.14 launcher — CORRECTION: the venv re-exec, not sys.executable (2026-06-25)

- SUPERSEDES the "segmv/combined_map run on the wrong venv" entry above, which mis-prescribed switching those launches to sys.executable. The real cause is upstream.
- Rule: to run MESA on a non-default venv (e.g. .venv314), set `MESA_SKIP_VENV_RELAUNCH=1` before launching. `mesa.py`'s `_ensure_repo_dev_venv()` (top of mesa.py, called at import) re-execs the process into the hardcoded repo `.venv` at startup unless that env var is set or `sys.prefix` already equals `.venv`.
- Why: launching `mesa.py` with `.venv314/bin/python` is silently defeated — `_ensure_repo_dev_venv` sees `sys.prefix (.venv314) != .venv` and re-execs into `.venv` (Python 3.11). So EVERY "3.14" run this session actually ran on 3.11 (the segmv `.venv/lib/python3.11/...` pyogrio traceback is the tell; that broken-3.11 pyogrio then failed the GeoPackage export, non-fatally). The processing pipeline + segmv + combined_map all spawn via `sys.executable`, which was therefore the 3.11 `.venv`. The pre-filter speedup (3.18x, byte-identical output) is unaffected — it's algorithmic and version-agnostic — but the 3.14 stack was never actually exercised by a full pipeline run.
- How to apply: `run_mesa_314.sh` now exports `MESA_SKIP_VENV_RELAUNCH=1`. A full 3.14 pipeline run is still UNVALIDATED; do one with the fixed launcher to confirm numpy-2 / pandas-3 hold end-to-end. Long-term clean fix: make `.venv` itself the 3.14 env (rebuild) so the re-exec target is 3.14 and no env var is needed.

## Intersect sindex-rebuild optimization (2026-06-25)

- Rule: in `_intersection_worker`, when the parent pre-filtered (3-tuple), call the top-level `join_geocode_assets(geocode_chunk, prefiltered=True)` so it uses `pool_assets` directly instead of a redundant top-level sindex query. Recursive sub-splits keep `prefiltered=False` (they narrow to smaller bboxes and do need the sindex).
- Why: `pool_assets` is already scoped to the chunk bbox by the parent, so the worker's first `pool_assets.sindex.intersection(chunk.total_bounds)` rebuilt an index only to re-select everything. For non-splitting chunks (the common ~420-geocode case) this skips the entire per-chunk sindex build+query.
- How to apply: proven set-identical to the full-broadcast path across 8 real chunks incl. a dense 323,140-row one (row counts + content match exactly). The gain is small (the mid-run "~30% per-worker" was a dense-band measurement artifact; aggregate was ~5%) but free and correct.
- Non-regression guarantee: the 2-tuple fallback (full broadcast) still runs the sindex query; only the prefiltered top-level skips it.

## Python 3.14 — full-pipeline validation (partial, all-green) (2026-06-25)

- Rule: 3.14 is validated for the high-risk stages; run it via `MESA_SKIP_VENV_RELAUNCH=1` (else mesa.py re-execs into the 3.11 `.venv`). Not switching the default to 3.14 yet (operator's call).
- Why / measured: a genuine 3.14 headless run (`.venv314`: numpy 2.5 / pandas 3.0 / pyogrio 0.12 / sklearn 1.9), reusing the 3.11-built `tbl_stacked` (`--no-prep --no-intersect`). Results, all 0 errors:
  - **Flatten** ran its per-partition pandas-3 groupby/merge across all 1387 partitions; the RAM-throttle safety engaged correctly (7→3→1 workers as vm crossed 60%). (Final `tbl_flat` write was skipped as already-current vs the unchanged tbl_stacked, but the per-partition compute — the pandas-3 risk — executed.)
  - **Backfill + Segment**: clean.
  - **Classification (segmv)**: sklearn GMM (k=6..10 by BIC), pyarrow writes (appended 3,383,953 cells — matches), and CRUCIALLY the **pyogrio GeoPackage export** (`export_gpkg → gdf.to_file(driver="GPKG")`) — the exact line that threw a multi-frame Traceback on 3.11's broken pyogrio — **wrote cleanly, no Traceback**. This confirms the launcher fix → genuine 3.14 → working pyogrio resolves the segmv failure.
  - **Tiles**: started clean, built segmv rasters across groups (basic_mosaic + H3_* + QDGC_*) for ~10 min before the operator stopped the run.
- How to apply: **lines + analysis were not reached** (stopped during tiles) — low risk but untested; tiles only partially ran. To finish the validation: launch on 3.14 with `MESA_SKIP_VENV_RELAUNCH=1` and let tiles/lines/analysis complete. The migration's genuine unknowns (pandas-3 flatten CoW, classification, pyogrio GPKG export) are now PASSED.
- Non-regression: validation reused the 3.11 `tbl_stacked`; only `tbl_seg_mv` (one appended run) and some segmv mbtiles were rewritten. `tbl_flat` and `tbl_stacked` untouched — the good 3.11 output is intact.

## Geocode import must merge, not overwrite (2026-06-26)

- Rule: `run_import_geocodes` must persist through `_merge_and_write_geocodes(refresh_group_names=<imported group names>)`, never blind-`to_parquet` straight onto `tbl_geocode_group/object.parquet`. Deleting geocode groups is an explicit Manage-geocodes-tab action only.
- Why: the old import wrote the imported groups directly over the geocode tables, silently wiping every other group. During a ~10 h basic_mosaic run the operator triggered an import mid-run; it deleted the H3 (13.3 M) + QDGC (83 k) objects built earlier the same day (table dropped to 5 groups / 6,530 objects). basic_mosaic survived only because its own publish does read-modify-write and re-appended itself afterwards. H3/QDGC/mosaic writers already merged; import was the lone destructive path. The "Import && manage geocodes" tab was also split into separate "Import geocodes" and "Manage geocodes" tabs so the delete control no longer sits next to import.
- How to apply: any writer of `tbl_geocode_*` goes through `_merge_and_write_geocodes` with the group names it owns in `refresh_group_names`; never `to_parquet` onto the geocode tables directly. When adding a tab between existing ones in `GeocodeManagerWindow`, update the `tab_lookup` index map (a new tab shifts every later index).
- Non-regression guarantee: re-importing the same source still refreshes those group names (identical end state for the imported groups); only the previously-deleted unrelated groups are now preserved.

## Mosaic union reduction is spawn-bound, not compute-bound (2026-06-26)

- Rule: in `_tree_reduce_unions` the per-round `maxtasksperchild` must scale with the round's merge count (high early, 1 late), not a flat 1. Respawning a spawn-context worker per pair pays a full process spawn + re-import; for early rounds (thousands of small-geometry merges) that overhead dwarfs the GEOS union itself.
- Why / measured: a 3.53 M-asset basic_mosaic took ~9 h 56 m, of which ~87 % was the edge+coverage pairwise tree-reduction (linework extraction was 6 min; polygonize ~1 h at 21.7 GB RSS). The reduce ran ~2.2 merges/s on 4 workers ≈ ~1.8 s/merge for unions that take GEOS milliseconds — dominated by process spawn+reimport (Windows AND macOS default to 'spawn'; this code forces 'spawn' on Linux too). Round 1 alone had 7,053 merges. Fix: `mtpc = clamp(merges_total // (round_workers*4), 1..64)` when `merges_total >= 4*round_workers` (small geometries → safe to persist a worker), else 1 (late rounds hold large geometries → respawn to bound RSS). Added an optional `mosaic_union_grid_size` (snap-rounding via `union_all(grid_size=)`, default 0 = off) that cuts GEOS cost, robustness retries and vertex count — safe because the mosaic always runs in a metric CRS.
- How to apply: when touching the mosaic union path keep the mtpc heuristic (many merges ⇒ small geoms ⇒ persist; few ⇒ large ⇒ respawn). The residual single-threaded `polygonize` (21.7 GB peak; would OOM a 16–32 GB host) is unchanged in *cost*, but a **pre-flight memory gate** now estimates peak from the asset count (`mosaic_preflight_gb_per_million_assets`, default 7.0) and, when it exceeds `available × mosaic_preflight_safety_frac` (0.8), skips basic_mosaic early with a clear message instead of OOM-crashing mid-run (override: `mosaic_preflight_allow_oversized`). It scales with the host — high-RAM users are never blocked (e.g. 3.53 M assets ≈ 25 GB est. proceeds on a 74 GB-available box, but is skipped on a ~12 GB-available 16 GB box; the 16 GB crossover is ~1.37 M assets). Truly bounding the peak still needs the tiled-overlay + membership-dissolve approach (speed-up review Tier 2).
- Non-regression guarantee: the spawn fix is output-identical (pairwise union is associative+commutative; verified serial==parallel on synthetic geometry). Snap-rounding defaults OFF (grid_size None ⇒ exact prior result). The removed dead code (`_mosaic_tile_worker`, `_plan_tiles_quadtree`, `_split_tile`; the `mosaic_tile_*` / `mosaic_quadtree_*` / clip / dedup / simplify config keys) was never called — no behaviour change.

## Settings store: config.ini defaults overlaid by tbl_settings (Phase 1) (2026-06-26)

- Rule: settings reads go through `read_config`, which now overlays a per-project key/value table (`output/geoparquet/tbl_settings.parquet`) on top of config.ini's `[DEFAULT]` via `mesa_shared.apply_settings_overlay(cfg, base_dir)`. config.ini stays the version-controlled defaults; the table (to be written by Tune processing in Phase 2) is the live store. **Absent table ⇒ config.ini is authoritative** — the fallback for helpers run before mesa.py has seeded anything.
- Why / shape: the operator wants tuning to live outside version control (seed from config.ini at init; carried by Data Management backups since it is a parquet under output/geoparquet). There is NO single read_config — 7 implementations across modules (mesa.py, geocode_manage, processing_internal, processing_setup, report_generate, analysis_present, mesa_shared) with different signatures — so the overlay is applied at each chokepoint, not by consolidating to one function. `mesa_shared` is intentionally stdlib-only, so pyarrow is lazy-imported *inside* the settings functions and only when the table exists (untuned projects pay just a `Path.exists()`); the overlay dict is cached per process on the file mtime.
- How to apply: Phase 1 is DONE and is a no-op in production (the table does not exist yet, so every overlay call returns config.ini unchanged; fallback + overlay + cache-invalidation verified in an isolated temp project). **Phase 2 — POSSIBLE FURTHER-DEVELOPMENT POINT, not done:** (1) seed the table from config.ini at mesa.py startup for *missing* keys only (do not clobber saved tuning); (2) make "Tune processing" Commit write via `write_settings(...)` instead of editing config.ini, and load current values from the table when the window opens. Any NEW `read_config` added later must call `apply_settings_overlay` to stay consistent.
- Non-regression guarantee: until something writes the table, every read_config returns exactly config.ini as before; each overlay call is wrapped so a failed import / unreadable table silently falls back to config.ini.

## Single-instance lock must cover the GUI path (and never os.kill on Windows) (2026-06-26)

- Rule: the pipeline single-instance lock (`<base>/output/.pipeline.lock`) must guard the **GUI "Process all"** path, not only the CLI/headless entry. Acquire atomically with `os.open(..., O_CREAT|O_EXCL)` and test holder liveness with `psutil.pid_exists` — **never `os.kill(pid, 0)` on Windows**, where a non-CTRL signal calls `TerminateProcess(handle, sig)` and *kills* the process.
- Why / observed: `_acquire_pipeline_lock` was only called from `run_headless`/`main()` (gated on `args.headless`); the GUI button ran `run_selected` in a worker thread without ever touching the lock. So two MESA windows could each start a run on the same project — they interleaved log.txt, raced the parquet outputs, and each called `_init_idle_status()` which reset the shared `__chunk_status.json` to idle (symptom: an empty minimap, and progress that oscillated 80↔81 / two alternating rows-written values instead of advancing). Two further latent bugs: `os.kill(pid,0)` would terminate the holder on Windows, and a check-then-write (`exists()` then `write_text`) TOCTOU let two processes both pass.
- How to apply: GUI acquires the lock in the Process-button handler (on the GUI thread, so the refusal QMessageBox is safe) and releases it in `_on_task_finished` — per-run, because the GUI process outlives individual runs, so atexit alone would hold the lock until MESA closes. The CLI path keeps the atexit backstop. Verified: fresh acquire / release / refuse-on-live-other-pid / reclaim-stale-dead-pid all pass.
- Non-regression guarantee: same-process re-acquire reclaims its own PID (a process never blocks itself); stale/garbage locks are reclaimed; psutil-absent falls back to a conservative "assume alive" on Windows so a live lock is never wrongly stolen.

## Config inline comments + mosaic pre-flight silent-default trap (2026-07-15)

- Rule: in config.ini, keep inline value comments to `;` only, and never put a `#` comment on the same line as a numeric key that `geocode_manage.read_config` consumes (the `mosaic_preflight_*` keys). `read_config` builds a plain `configparser.ConfigParser()` with NO `inline_comment_prefixes`, so `mosaic_preflight_safety_frac = 0.85  # note` is read verbatim as the string `"0.85  # note"`, `_cfg_float` fails to parse it, and it SILENTLY falls back to the code default (0.8) — no error logged.
- Why / observed: after a GitHub pull, "create basic mosaics" produced nothing — the new pre-flight gate skipped basic_mosaic (3.53 M assets: est 24.7 GB > budget 24 GB = 30 GB avail × 0.8). Raising `mosaic_preflight_safety_frac` to 0.85 inline (`= 0.85  # ...`) had NO effect because the inline `#` made the value unparseable → default 0.8 → still 24.0 GB budget → still skipped. The incoming commit had written every `mosaic_preflight_*` line with an inline `#` comment; they only "worked" because the written values happened to equal the code defaults (7.0, 0.8, false). Tempting fix — give `read_config` `inline_comment_prefixes=(";","#")` like `mesa_shared`/`asset_manage` — BREAKS colour reading: config.ini uses `#` as DATA in hex colours (`category_colour = #bd0026`, `report_inset_border_color = #1f1f1f`), which a `#`-stripping parser turns into empty string. So the file cannot safely treat `#` as both comment and data. Correct fix: move the notes to their own full-line `#` comments above each key; leave `read_config` as a plain parser.
- Durable fix (2026-07-15): `geocode_manage._cfg_float`/`_cfg_int` now strip any trailing inline comment via `_strip_inline_comment` (split on `#` then `;`) before parsing, and a new `_cfg_flag` does the same for the `mosaic_preflight_enabled`/`allow_oversized` booleans (previously read with a raw `.get().strip().lower()` that an inline `#` silently broke). This is done at the read site, NOT the parser, precisely so hex-colour string values (`#bd0026`, read via plain `.get()`) are never touched — numbers and flags never legitimately contain `#`/`;`, colours do. Result: the repo's inline-`#`-commented `config.ini` now loads correctly on every platform without reformatting the file.
- How to apply: read numeric config in geocode_manage via `_cfg_float`/`_cfg_int` and booleans via `_cfg_flag`, never a bare `.get()` — those tolerate inline comments; a bare `.get()` on a value line with a `; ...`/`# ...` note returns the comment too. Still prefer own-line comments for new keys as hygiene, but code no longer depends on it. Do NOT give `read_config` `inline_comment_prefixes` including `#`: it would blank the hex-colour values.
- Non-regression guarantee: `read_config` stays a plain parser, so colour reads (`#rrggbb`) and every other string value are byte-for-byte as before; the helpers only strip trailing `#`/`;`, which valid numbers/flags never contain, so all previously-correct values still parse identically.

## Standardize on Python 3.14 on both platforms; retire the 3.11 macOS req file (2026-07-15)

- Rule: macOS and Windows both target CPython 3.14. Mac dev/runtime venv = `python3.14 -m venv .venv` + `requirements_py314.txt`; Windows = `requirements_py314_win.txt` (same versions, pyobjc dropped, EdgeChromium webview). `requirements_macos_dev.txt` (the pre-migration 3.11 set) is DELETED — do not reintroduce a 3.11-specific requirements file.
- Why: this Mac checkout carried a stale 3.11 `.venv` alongside the canonical 3.14 env (in the sibling `mesa-py314` git worktree). The 3.11 venv had drifted into a broken mix: numpy 2.x + pyogrio 0.7.2 (built against numpy 1.x → "numpy.core.multiarray failed to import", which killed the Assets dialog), and openpyxl 3.1.2 + pandas 3.x (→ "Pandas requires version '3.1.5' or newer of 'openpyxl'", which killed Excel load). `requirements_macos_dev.txt` still pinned numpy 1.26.2 / pyogrio 0.7.2 / pandas 2.1.4, so a fresh install from it recreated exactly those breakages. The confusing symptom: same errors on Mac but not Windows, because Windows was already on the 3.14 set.
- How to apply: fresh Mac setup is `python3.14 -m venv .venv && .venv/bin/python -m pip install -U pip -r requirements_py314.txt`. When bumping the geo stack keep pyogrio >= 0.8 (first numpy-2 build) and openpyxl >= 3.1.5 (pandas-3 Excel I/O). The displaced 3.11 venv was moved to `.venv.old-311` (gitignored) as rollback, not deleted outright.
- Non-regression guarantee: `requirements_py314.txt` is the exact pinned set validated 2026-06-24 that ran a full basic_mosaic pipeline on CPython 3.14 (macOS arm64); the recreated `.venv` imports numpy/pandas/pyogrio/openpyxl/geopandas/shapely/PySide6/rasterio/sklearn cleanly and mesa.py + geocode_manage + asset_manage compile under 3.14.

## Classification (segmv) over multiple geocode layers + grid-area caveat (2026-07-16)

- Rule: `segmv_geocode_layer` now accepts one layer, a comma list, or `all` (every generated geocode layer: basic_mosaic, H3/QDGC levels, uploaded admin layers). Each layer is classified as a separate GMM fit under ONE shared run_id and coexists in `tbl_seg_mv` / `tbl_seg_mv_profile` keyed on (run_id, name_gis_geocodegroup). The Maps window lists each (run_id, layer) as its own selectable result.
- Why / shape: previously classification ran on a single layer only, so H3/QDGC/admin layers had no interactive segmentation (only the seg_signatures raster). Enabling multi-layer required three touch-points: (1) `segmentation_run.run_all_layers` resolves the spec and loops `run_segmentation` per layer with a shared run_id; (2) `_write_parquet_coexist` gained a `layer` arg so per-layer writes accumulate instead of clobbering the same run_id (its old filter dropped ALL rows for the run_id — a per-layer loop would keep only the last layer); (3) `combined_map.segmv_layer`/`_segmv_profile` gained a `layer` arg (a run_id can now hold several layers, so `iloc[0]` was wrong) and the JS passes `run.layer`. The tiles stage needed NO change — it already filters tbl_seg_mv per group (`name_gis_geocodegroup == gv`) and rasterises whatever layers are present.
- How to apply: area for grid layers (H3_*/QDGC_*) is the grid-CELL area, which generalizes the true asset footprint; `_area_basis(layer)` tags each profile row `grid` vs `polygon`, and the Maps stats panel shows an amber warning when any row is `grid`. basic_mosaic and uploaded polygon layers report exact area. When adding a new geocode layer type, extend `_area_basis` if its geometry is a tessellation rather than the real feature.
- Non-regression guarantee: single-layer runs are unchanged (spec `basic_mosaic` → one layer, one run_id); `_segmv_profile`/`segmv_layer` default `layer=None` to the first layer (prior behaviour); existing tbl_seg_mv without an `area_basis` column reads as `polygon`. `all` is opt-in via config; the committed default stays a single layer to avoid surprising users with 12× longer classification.

## Module-level config in mesa.py hangs spawned worker pools (2026-07-17)

- Rule: nothing at mesa.py module level may read config.ini, raise, or touch disk. Put it in `_bootstrap_config()`, called only under `if __name__ == "__main__"`. Under `spawn`, every worker child re-executes mesa.py as `__mp_main__`, so module-level code runs once per child — and a raise there kills each child before it claims its task, which `Pool` answers by respawning it forever.
- Why: the MESA demo dataset (`jinja_sample.zip`, 8 features / 494k vertices) appeared to make basic_mosaic hang: "workers=12, chunks=1" then "0/1 chunks completed; workers alive=12/12" for 9+ minutes. Measured ground truth: `_mosaic_extract_chunk_worker` on all 8 features = 8.0s; the whole `Pool(12)` + spawn + that chunk = 9.2s from a clean `__main__`. The mosaic code was never the problem. The chain: `restore_backup_archive` deleted config.ini and the demo zip had no config.ini to put back -> `geocode_manage` (an INPROCESS_HELPER, so `__main__` is mesa.py) had already loaded its cfg and ran on defaults -> each of the 12 spawn children re-exec'd mesa.py, hit the module-level `raise FileNotFoundError`, and died. Reproduced the exact log signature: 3379 child deaths in 50s, `alive` flapping 11/12 <-> 12/12, forever. Tell: the log printed `chunk_size=2,500 maxtasksperchild=4` (code defaults) while config.ini says 250/20 — proof the config was already gone before the mosaic started.
- How to apply: when adding module-level statements to mesa.py, or making a helper an INPROCESS_HELPER, ask "does this run 12 more times, in a process with no GUI and no config?" A crash-looping pool never surfaces an error — it just stops making progress, so read "0/N completed with all workers alive" as *children dying at startup*, not as slow GEOS. `runpy.run_path(mesa.py, run_name="__mp_main__")` reproduces a child exactly; warm cost is 0.08s.
- Non-regression guarantee: `_bootstrap_config()` runs at the same point in module execution as the old inline block, so the main process sees identical ordering and the same `FileNotFoundError` when config.ini is genuinely missing. Only the `__mp_main__` re-exec skips it.

## Backup restore must not delete a config.ini it cannot replace (2026-07-17)

- Rule: `restore_backup_archive` only unlinks `<base>/config.ini` when the archive actually carries one; otherwise it keeps the current settings and logs that it did. Restoring settings from a backup stays the intended behaviour — an archive WITH config.ini still overwrites.
- Why: the delete was unconditional, but re-extraction is conditional on `m == "config.ini"` being in the zip. Archives built by the demo-data project carry only `input/`, so every restore silently destroyed the project config and left MESA unstartable (and, before the fix in the entry above, hanging rather than erroring). Top-level match also means a Mac-style zip nesting everything under a folder would not restore its config either.
- How to apply: demo/sample archives are expected to ship a config.ini at the zip root; the warning is the signal that one is missing. When adding restore targets, pair every delete with proof the archive can put the file back. Note the rmtree of `input/`+`output/` is still unconditional, and `input/` holds tracked repo files (readme.txt, sample images) — a restore wipes them.
- Non-regression guarantee: archives containing config.ini behave exactly as before (delete then extract).

## First green frozen build on Python 3.14 (2026-07-17)

- Rule: the Windows frozen build works on CPython 3.14.6 with `pyinstaller==6.21.0` + `pyinstaller-hooks-contrib==2026.6` — now PINNED in `requirements_compile_win.txt`. Do not unpin without re-running a full `devtools\compile_win_11.bat` and launching the result.
- Why: the 3.14 migration had been validated from source only; `requirements_compile_win.txt` deliberately left the build toolchain unpinned ("pin once a full build_all.py run on 3.14 confirms a working version") and `plans.md` A3 carried the gap. It went green on the first attempt: full clean build (helpers+main, parallel=4) in **435.9s / 7m16s** — faster than the last 3.11 builds (525-597s in `D:\dist\build_history.log`). Per-target: special_focus 338.3s, segmentation_setup 283.5s, main:mesa 304.2s, segmentation_run 222.2s, combined_map 124.9s, tiles_create_raster 123.0s. Result: 2.05 GB / 7,855 files.
- How to apply: PyInstaller exiting 0 is NOT the test — a frozen app can build clean and die on launch. Always smoke-test `D:\dist\mesa\mesa.exe`: it must stay up, show its version in the window title, and write `log.txt` + the host-capabilities snapshot (that last one proves config read + the background thread + a parquet write all work frozen). Benign build noise: `Library not found: could not resolve 'Qt6QuickShapesDesignHelpers.dll'` and similar PySide6 QML plugins MESA never loads.
- Non-regression guarantee: pinning records the exact set that worked; the 3.11 toolchain was already retired (see "Standardize on Python 3.14 on both platforms"). What is still NOT validated is a full pipeline run inside the compiled app — the 3.14 source validation never reached lines/analysis. Tracked in `plans.md` A3.

## Developer docs must live outside docs/ (2026-07-17)

- Rule: `docs/` ships to end users — it holds the user manuals (`MESA_User_Guide_en.docx`, `_pt.docx`) and `docs/templates/` (read at runtime by `report_generate`). Everything developer-facing goes in `devtools/docs/` (design documents, capacity records, yEd `.graphml` diagram sources, the segmentation PoC) or at the repo root next to `learning.md` (`plans.md`). Do not add developer material to `docs/`.
- Why: `build_all.py:864` copies `qgis/ docs/ input/ output/ system_resources/` wholesale, and `DEVELOPER_ONLY_FILES` only strips four root-level notes by filename. The 5.5.0 build shipped six roadmap documents to end users, including `further_development.md` — literally our not-yet-done list — plus `UNIFIED_MAP_PLAN.md`, which by then was not just stale but CONTRADICTED (it states "the Asset map stays a separate window"; Assets shipped as a tab in the unified Maps window). `devtools/` is removed from the dist wholesale, so anything under it is safe by construction — no strip-list entry to remember.
- How to apply: the copy list is a folder allowlist, so a folder that is not in it can never ship — prefer that over growing `DEVELOPER_ONLY_FILES`, which is filename-matched and must be maintained by hand. Never strip by `*.md`: `docs/templates/report_about.md` is a runtime asset and would break reports. When moving a doc, `git grep` its old path — `devtools/build_segmentation_doc.py` *writes* `MESA_Segmentation_PoC.docx` and would have put it straight back into the shipped folder.
- Non-regression guarantee: the three superseded plans (UNIFIED_MAP, SEGMENTATION_INTEGRATION, SEGMENTATION_OVERVIEW_VIEWER) were moved, not deleted — their durable content was already in learning.md (2026-06-06/08 entries), verified before the move.

## Package description surfaced after restore; it must not outlive its package (2026-07-17)

- Rule: `restore_backup_archive` extracts `docs/readme_demodata.txt` by EXACT member name, and unconditionally deletes any existing copy first. `create_backup_archive` deliberately does NOT include it. The restore-complete dialog offers an "Open description" button only when the file is present.
- Why: mesa_demodata packages ship a plain-text description of themselves — synthetic vs real data, sources and the credits they require, and the package's measured asset overlap. It was silently discarded. Two traps in the obvious implementation: (1) a `docs/` prefix match would let any archive replace `MESA_User_Guide_en.docx` or `docs/templates/report_about.md`, the latter being a RUNTIME asset that `report_generate` reads for every report — `_safe_zip_member_names` blocks traversal but not a member legitimately naming an existing doc; (2) restore rmtrees `input/` and `output/` but NOT `docs/`, so a readme from an earlier restore would survive and be offered as describing the package just restored — crediting the wrong sources, which is precisely the claim the file exists to make.
- How to apply: the delete-first rule is the OPPOSITE of the config.ini rule in the entry "Backup restore must not delete a config.ini it cannot replace", and the asymmetry is principled: config.ini is infrastructure the project cannot start without, so keep the old one when the archive has none; the readme is a claim ABOUT the archive, so a wrong one is worse than none. Apply the same test to anything new a package may carry — is it infrastructure, or a claim?
- Non-regression guarantee: an ordinary project has no such file, so ordinary restores are unchanged — verified that the completion dialog shows only OK without it, and that the shared `_ArchiveProgressDialog` (also used by backup) is unaffected. Round-trip: because create_backup_archive omits it and restore deletes stale copies, the description exists exactly while it is true and falls away the moment the user backs up their own evolved project. Do not "fix" that asymmetry.

## Qt stylesheet font-size: use px, not pt (cross-platform sizing) (2026-07-18)

- Rule: in Qt stylesheets (QSS) and inline setStyleSheet strings, size fonts in `px`, never `pt`. px is DPI-logical and renders identically on Windows and macOS; pt does not.
- Why: Qt converts `pt` to pixels via the screen's logical DPI, which differs by platform — 96 on Windows, 72 on macOS. So the same `10pt` renders ~33% larger on Windows (10 x 96/72 = 13.3 px) than on macOS (10 px). A stylesheet tuned on one platform looks oversized/undersized on the other. The MESA UI base was `font-size: 10pt` with no DPI override (no `AA_Use96Dpi`), so Windows buttons and text ran a third larger than intended — a user on Windows flagged the Edit-assets buttons as "a bit large". Converted all stylesheet font-sizes to px (10pt->11px, 9pt->10px, 8pt->9px) across ui_style.py + inline styles in mesa.py and six helpers (asset/atlas/geocode_manage, processing_pipeline_run, processing_setup, segmentation_setup) — 61 sites total. Both platforms now render at the same px; Windows comes down ~17%, macOS up ~10%, converging near native body size.
- How to apply: when adding any `font-size` to a QSS/setStyleSheet string, write px. Grep `font-size:\s*\d+pt` before shipping UI changes. Note this is separate from `QFont("Segoe UI", N)` calls, whose integer is a POINT size with the same 72-vs-96 divergence — those remain in the codebase (painted headers/labels) and would need `setPixelSize()` to be fully consistent; not converted here because the request was scoped to the stylesheet. There is one deliberate existing Mac/Win compensation to mirror the spirit of: `ui_style.py` forces left-aligned QTabBar tabs because macOS centres them by default.
- Non-regression guarantee: px values were chosen to land both platforms near their native body size, so neither platform regresses to unreadable; the conversion is unit-only (no layout/padding/logic change) and every touched file still compiles.

## "Compile/build" means build only — never touch input/ or output/ (2026-07-18)

- Rule: When the operator asks to compile or build ("kjor en kompilering", "run a Windows build"), run `devtools\compile_win_11.bat` against the working tree EXACTLY as it stands. Do not clean, restore config.ini, remove demo/project data, wipe the output folder, or "prepare a clean release" — unless the operator explicitly asks to change what is bundled.
- Why: `build_all.py` copies `input/` and the output folder into the distribution on purpose. The operator has always compiled with a populated project so END USERS do not start from an empty tree — they get a ready-to-explore project (assets, processed results, tiles). Treating a plain build request as a release-hygiene task and asking "what kind of build?" / clearing input+output is both wrong and destructive. In this incident the tree held a restored zirimiti demo project (12.8 MB assets + 93.6 MB output); a cleanup command was issued and only spared because the sandbox rejected the whole command (it contained a protected output-folder deletion). Nothing was lost, but the intent was wrong.
- How to apply: a build request is not a decision point. Verify the small mechanical preconditions (no MESA process is holding a helper `.exe`; `.venv_compile` exists) and launch the build. If you genuinely believe the bundled data is wrong for the operator goal, SURFACE it as a one-line note and let them decide — do not act on it, and never reshape the tree pre-build.
- Non-regression guarantee: building against the current tree is the operator long-standing default; this entry just forbids the assistant from deviating from it.

## Restore must back up config.ini before overwriting it (2026-07-18)

- Rule: `restore_backup_archive` copies the current `config.ini` to `config.ini.bak` before it unlinks + overwrites it from the archive. Restore is otherwise irreversible, and it takes `mesa_version` from the archive.
- Why: restore replaces `config.ini` in place with no recoverable copy. A user restoring a demo/backup package into a project they care about loses their own machine-specific tuning AND has `mesa_version` silently set to whatever the archive carried — with no undo. Observed live: this repo's `config.ini` became one of the demo project's generated configs (duplicated generator banner + throttled tuning: `mem_target_frac 0.60`, `chunk_size 8000`) after a demo restore, with no `.bak` to recover the canonical file (git was the only fallback, which end users don't have). See the paired entry "Backup restore must not delete a config.ini it cannot replace".
- How to apply: `config.ini.bak` is one level deep — the config as it was immediately before THIS restore; a second restore overwrites it. That covers the common "experimental restore, undo it" case; it does not protect across two restores. It is gitignored. Nice-to-have not yet done: surface a `mesa_version A → B` note in the restore confirmation dialog when the incoming archive carries a different version.
- Non-regression guarantee: the `.bak` is only written when the archive actually carries a `config.ini` to overwrite with (so an archive without one still keeps the current config and writes no spurious `.bak`); all other restore behaviour is unchanged.

## Analysis area maps – inside/outside sensitivity (2026-07-19)

- Rule: the analysis Compare/Single area maps (`_analysis_write_area_map_png`) draw sensitivity in TWO layers from the project-wide `tbl_flat` (`context_cells_gdf`), not from the per-area `tbl_analysis_flat` cells: a faint pass across the whole frame (surroundings, alpha ≈ overlay_alpha × `outside_alpha_frac`), then the same data clipped to the study polygon at full `overlay_alpha`. The per-area `flat_cells_gdf` is only a fallback for the inside layer when `tbl_flat` is missing.
- Why: the per-area analysis cells are sparse and area-specific — measured on the zirimiti demo, Area 1 had 14 cells and Area 2 only 9, while project-wide `tbl_flat` had 48,615. Sourcing the overlay from the per-area cells made Area 2 look almost empty inside its own boundary and gave no context outside; the two Compare maps were not visually comparable. Clipping the dense project-wide layer to each polygon fills both interiors consistently and the faint full-frame pass shows the sensitivity of the surroundings, which is the point of a study-area map.
- How to apply: load `tbl_flat` ONCE per analysis section (`analysis_context_cells = load_tbl_flat(...)`) and pass it as `context_cells_gdf` to every area — do not call `load_tbl_flat` per area (it materialises the whole flat table). The inside layer is `gpd.clip(context_3857, study_polygon_union)`; the context is bbox-trimmed with `.cx[minx:maxx, miny:maxy]` before drawing to bound cost. Alphas were lowered here for general transparency (inside 0.65→0.55, outside ≈0.22) so the basemap reads through — if tuning, keep inside > outside so the study area still reads as the emphasis.
- Non-regression guarantee: when no `context_cells_gdf` is supplied the function falls back to the old per-area `flat_cells_gdf` behaviour, so callers that don't pass the project-wide source render as before; only the analysis section was wired to pass it.

## Recent activity – Processing duration reflects the last full [Process] run (2026-07-19)

- Rule: the Status tab's "Recent activity" "Processing" row is scanned with the outer `[Process] STARTED` → `[Process] COMPLETED`/`[Process] FAILED` wrapper markers ONLY. Do not add the per-stage markers (`DATA PROCESS START/COMPLETED`, `LINES PROCESS …`, `ANALYSIS PROCESS …`, `[Tiles] …`) to that scan's marker lists. Its timestamp comes from the same run's end time (fallback: tbl_flat mtime, then config `last_process_run`), so duration and stamp describe the same run.
- Why: `_scan_last_run_from_log` restarts its interval on every start marker, so mixing the sub-stage markers in made it treat each sub-phase as a separate "run" and return only the LAST micro-phase. Measured on the live log this reported Processing = **1 s** (the analysis phase 21:01:46→47) for a run whose real span was 21:01:43→21:01:47; a full data+tiles run (~3.5 min) would likewise have collapsed to its trailing ~1 s. `open_process_all()` (mesa.py) logs `[Process] STARTED` on button click and `processing_pipeline_run` logs it again ~13 s later when the pipeline actually starts; reset-on-new-start keeps the pipeline's own start, so the duration is execution time excluding exe launch — the more consistent metric. `[Process] COMPLETED` is written after data+tiles+lines+analysis, so the wrapper brackets the whole run.
- How to apply: when a log-scanned "Recent activity" row shows an implausibly tiny duration, check whether its marker list contains nested sub-phase markers that also appear as their own start/stop pairs — that fragments the interval. Bracket each row by the OUTERMOST start/stop only. When adding new pipeline stages, keep their markers out of the wrapper scan.
- Non-regression guarantee: rows that already bracket cleanly (`Step [Assets] …`, `Step [Mosaic] …`, `Report mode selected:` → `Word report created:`) were not touched; only the Processing row's marker set and timestamp source changed, and the timestamp still falls back to tbl_flat mtime so a log without any `[Process]` run behaves as before.

## Windows taskbar button icon is fixed at button creation (2026-07-20)

- Rule: to get an icon on the Windows taskbar button, set `System.AppUserModel.RelaunchIconResource` (plus the AppUserModelID) on the WINDOW's shell property store, from the window's own process, BEFORE its first `show()` — `mesa_shared.set_window_taskbar_icon(hwnd, icon_resource)`. `setWindowIcon()` is not enough. `mesa_shared.set_windows_app_user_model_id()` is still called early in each process, but it only buys taskbar *grouping/identity*, NOT the icon.
- Why: the compiled app showed a blank taskbar button while the exe icon (Explorer), the window icon and the taskbar *thumbnail* were all correct. Windows decides a button's icon when it creates the button and never re-reads the window icon afterwards; with no icon associated with the AppUserModelID the button stays blank. Diagnosis ruled out, in order: icon cache (survived `ie4uinit -show` + reboot), a malformed .ico (9 sizes, all decode, 98% opaque), a missing exe resource (all 9 images embedded; `PrivateExtractIcons` returns content at 16–256), the non-square `128x127`/`256x254` entries (Windows pads them itself), and a missing `ICON_BIG` (measured `WM_GETICON` — SMALL, BIG and both class icons were all set). The tell was that hiding+showing the window made the icon appear: that re-registers the button, forcing the shell to re-read. Confirmed the real fix by running from source (host `python.exe`, which would otherwise show the Python icon) — with RelaunchIconResource set before show, the taskbar button rendered the MESA icon, and the property read back as set.
- How to apply: any new window that gets its own taskbar button needs this before its first show; the property store is READ-ONLY cross-process (`SetValue` → `0x80070008`), so it cannot be bolted on from a helper script — it must run in-process. Pass the **.ico path** (`system_resources/mesa.ico`, which ships next to the exe) in BOTH frozen and source builds. Do not use the `"<exe>,0"` resource form: an isolated test accepted all formats, but in the frozen app that value silently failed to stick (`AppUserModel.ID` still read back as set because it falls back to the process-wide id, which masked the failure — always verify the ICON key specifically, not just the id). When a taskbar icon looks blank, test hide/show first: if that fixes it, the icon data is fine and the problem is button-creation timing, not the icon.
- Non-regression guarantee: fully guarded (try/except, Windows-only, no-op without an hwnd) so it cannot affect startup or non-Windows platforms. NOTE: `mesa.py` lives at the repo root, so `code/` is not on `sys.path` when running from source — call `_ensure_code_dir_on_syspath()` and guard the import before importing `mesa_shared` in `__main__`, or the app starts fine when frozen (PyInstaller bundles it) but dies from source with ModuleNotFoundError. That regression shipped briefly and was caught only by running from source; `py_compile` cannot see it.

## Release automation: match version markers on word boundaries (2026-07-20)

- Rule: in `devtools/github_release_from_zenodo.py`, every keyword test matches on WORD boundaries (`_PRERELEASE_RE`, `_has_word`), never with a bare `token in text`. Never hardcode a version number into generated release copy — derive it from the release being built.
- Why: three substring bugs shipped in one generator. (1) `"rc" in joined` matched *a-rc-hive* and *sou-rc-es* in the Zenodo description, so a finished release was marked `prerelease` **and** got a "This is still a beta release … should not be used for decision-making" notice — the exact disclaimer MESA 5 is not supposed to carry. (2) `"ui" in text` filed *G-ui-de for counting LOC* under "User interface". (3) A highlight line was hardcoded to `"Version 5.1 fully replaces …"` and still said 5.1 while publishing 5.5.0. All three are invisible until you read the generated output.
- How to apply: always run the generator in preview (no `--publish`) and READ the output before releasing — the tag, the prerelease flag, and every generated sentence. Check the tag especially: it is derived from the Zenodo title, so a title like "MESA tool version 5.5.0 2026.07.20" yields the tag `5.5.0-2026.07.20` while previous releases used bare `5.2`/`5.0.3`; pass `--tag` to keep the series consistent. Release plumbing (Zenodo text, publishing checklist, release notes, changelog, LOC guides) is filtered out of the changelog by `_skip_subject` — it describes how the release was made, not what it does.
- Non-regression guarantee: the word-boundary form still detects genuine markers — verified `beta`, `rc1` and `pre-release` all return True, while `archive`/`sources` no longer do. Classification keywords were widened at the same time so real user-facing commits stop falling into the discarded "Other changes" bucket.

## QGIS project raster opacity: data mosaics see-through, basemaps opaque (2026-07-20)

- Rule: in `qgis/mesa.qgz` the MESA result rasters (the `type=mbtiles`, local `output/mbtiles/` layers: sensitivity/importance/index_owa/groups/assets totals, segmentation, classification) carry `rasterrenderer opacity="0.65"` (35% see-through). The external tile basemaps (`type=xyz`: Bing, Google, OSM, Waze) stay `opacity="1"`. Both kinds use QGIS's `wms` provider, so provider does NOT distinguish them — the datasource `type=` does.
- Why: "make all raster layers 35% see-through" means the data overlays should let the basemap show; making the basemaps themselves see-through is meaningless (nothing renders behind them → white). None of these rasters carry a `<layerOpacity>` element (that element belongs to the vector layers here), so `rasterrenderer opacity` is the single authoritative value to edit — no double-multiply risk. Edited block-aware (per `<maplayer>`) so only the 8 mbtiles rasters change and the 4 xyz basemaps are left alone.
- How to apply: a `.qgz` is a zip of `mesa.qgs` (+ a styles .db with a random-prefixed name); extract, edit the `.qgs`, repackage preserving the exact member names. Classify a raster by its datasource `type=` (`mbtiles` = MESA data, `xyz` = external basemap), not by provider. Validate the rewrite by re-opening the zip and XML-parsing the `.qgs` before overwriting the source.
- Non-regression guarantee: only the 8 data rasters' opacity attribute changed; basemaps, vector layers, styles.db, and all other project settings are untouched, and the repackaged zip re-parses as valid QGIS XML.

## Self-intersecting basic_mosaic faces come from the reprojection, not polygonize (2026-07-21)

- Rule: repair mosaic faces AFTER `to_crs("EPSG:4326")`, never before. In `geocode_manage.run_mosaic`'s `_flush_faces`, the batch is reprojected first and then passed through `_repair_faces` (`_fix_valid` → `_extract_polygonal`), and `[Mosaic][Sanity]` asserts that no invalid face survives.
- Why: a 5.5.0 Uganda project published 9 of 27,983 `basic_mosaic` faces that failed `ST_IsValid` with self-intersections (pinch points — a ring touching itself at one vertex). MESA's own numbers were unaffected (area identical, `[Mosaic][Sanity]` coverage diff 0 m²), but PostGIS 3.2 (GEOS 3.10.2) aborts any `ST_Intersection` against them with `TopologyException: Input geom 1 is invalid`, and GDAL/ArcGIS reject or silently alter them; it stayed hidden only because shapely 2.x's OverlayNG is tolerant. The obvious fix — repair right after `polygonize` — logged "all 27,983 faces already valid" and changed nothing: the faces ARE valid in the metric CRS (EPSG:3857); it is the coordinate rounding in the degrees reprojection that pinches near-coincident vertices together. The sanity assertion is what exposed that, one line after the repair claimed success. Inputs were not at fault (0 invalid of 16 asset objects); `H3_*`/`QDGC_*` never pass through polygonize and were clean.
- How to apply: when a geometry defect appears only in published output, check validity in BOTH the working CRS and the published one before deciding where to repair — a repair pass in the wrong CRS is a no-op that reads as a pass. Any new geocode-publishing path needs its own post-reprojection repair; measure area deltas in metres by projecting just the repaired subset back, since the faces are in degrees at that point. Repaired faces stay one face (Polygon → MultiPolygon), keeping face count and codes stable; downstream (`processing_internal`, `tiles_create_raster`) already handles MultiPolygon.
- Non-regression guarantee: verified on `mesa_5.5.0_uganda_full.zip` rebuilt with `geocode_manage.py --nogui --mosaic`: 27,983 faces in and out, 27,974 geometries byte-identical, only the 9 repaired faces changed, total area 553,818,529.797878 → 553,818,529.791762 m² (−0.006 m², 1.1e-11 relative; max per-face 0.003 m²), 0 invalid remaining, `[Mosaic][Sanity]` coverage diff unchanged at 0 m². Asset-side quality control was NOT removed or disabled along the way: `asset_manage._validate_geometry` (make_valid, buffer(0) fallback) still runs unconditionally before dissolve, and the user-facing "Validate geometries" checkbox still drives `_apply_quality_controls`. Its `import_validate_geometries` default has been "false" since it was written (`data_import.py`, 2026-01-13; carried into `asset_manage.py` on 2026-02-13) — off by default, but never silently dropped. It would not have caught these faces anyway: they are MESA's own geometry, generated after import.

## Classification run exports: the GPKG is opt-in (2026-07-21)

- Rule: `output/segmentation_mv/<run_id>/classification_results.gpkg` is written only when `segmv_export_gpkg = 1` (or `segmentation_run.py --gpkg`), like `segmv_make_png`. `summary.md` and `params.json` are always written — they are the run's only record of the BIC table, the chosen k and the ARI/NMI validation.
- Why: the GPKG is a denormalised copy of geometry MESA already stores (`tbl_geocode_object`) joined to six columns it already stores (`tbl_seg_mv`), and NOTHING reads it back: the Maps Classification tab, the Word report section and the shipped `qgis/mesa.qgz` vector layer all read `tbl_seg_mv*.parquet`, and the QGIS rasters read `output/mbtiles/*segmv_latest*`. It cost ~770 bytes per cell per run — 21.5 MB for a 27,983-cell mosaic, ~700 MB/run at a million cells — with no retention anywhere: every run adds a folder, nothing prunes, and the only cleanup is Manage data → Clear output, which deletes all of `output/`. It also shipped in every backup (`_classify_output_file` files `output/segmentation_mv/` under "databases"). Six runs on the Uganda project = 124 MB of GPKG against 0.6 MB of parquet driving the entire UI.
- How to apply: before adding a file export to a pipeline stage, check whether any MESA feature reads it — if the answer is "a user might open it in QGIS", make it opt-in, not default. Run-keyed output directories accumulate silently; anything written per run and never read needs either a flag or a retention policy. Note the run folder is keyed on `run_id` ALONE (`segmentation_run.py` `run_segmentation`) while `run_all_layers` gives every layer the same `run_id`, so with `segmv_geocode_layer = all` each layer's `summary.md`/GPKG overwrites the previous layer's — the parquet tables are keyed `(run_id, layer)` and are unaffected. Fix `out_dir` to include the layer if that export is ever turned back on by default.
- Non-regression guarantee: only the GPKG write is gated; the parquet tables, `summary.md`, `params.json` and the optional PNGs are unchanged, and existing run folders are left alone (no deletion, no migration). Verified from source on the Uganda fixture: a default run wrote `params.json` + `summary.md` only (3 KB, was 21.5 MB), `--gpkg` still wrote `classification_results.gpkg`, and both runs produced identical parquet output.
