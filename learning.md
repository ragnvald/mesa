# Learning log: UI capture and tooling operations (MESA)

Purpose: keep practical, reusable knowledge from UI screenshot work and related tooling changes.

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

- Desktop UI capture for `mesa.py` (`Workflows`, `Status`, `Settings`, `About`)
- Helper UI capture for tools under `code/`
- Capture tooling placement and path conventions under `devtools/`

## Current canonical script locations

These moved from `code/` to `devtools/` and should be referenced by new docs/scripts:

- `devtools/build_all.py`
- `devtools/compile_win_11.bat`
- `devtools/capture_ui_active_batch.py`
- `devtools/screenshot_active_window.py`

If you see old paths like `code/compile_win_11.bat` or `code/capture_ui_active_batch.py`, update them.

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
- `Settings` -> `ui_mesa_desktop_tab3.png`
- `About` -> `ui_mesa_desktop_tab4.png`

## Known pitfalls and fixes

- Problem: request/tool timeout (408 or long-running call failure)
  - Fix: run in smaller batches, increase wait times, avoid giant one-shot runs

- Problem: editor/background captured instead of app window
  - Fix: use active-window or process-tree-based window targeting

- Problem: right/bottom clipping or offset on Windows
  - Fix: enforce DPI-aware bounds + ensure window is fully on-screen before capture

- Problem: desktop tab screenshots end up identical
  - Fix: explicitly switch tab and wait before each capture

- Problem: new capture launch blocked by already running helper window
  - Fix: pre-flight cleanup of stale `mesa.py`/helper Python processes

## Geocode UI lessons (recent)

- Consolidating geocode workflows into `code/geocode_manage.py` simplifies user pathing and screenshot automation.
- Edit flow with `Previous/Next/Save/Save & Next` is easier to document and test than wide row-based edit grids.
- When helper UI is restructured, keep screenshot output file names stable where wiki references depend on them.

## Build tooling lessons (recent)

- `devtools/compile_win_11.bat` is now the canonical full-build entrypoint.
- `devtools/build_all.py` must resolve `code/` relative to project root, not relative to script folder assumptions from old layout.
- Keep compile verification lightweight (`py_compile`, `--help`) unless explicit user approval is given for full builds.

## Screenshot references used in docs

Desktop tabs:

![Workflows](../mesa.wiki/images/ui_mesa_desktop.png)

![Status](../mesa.wiki/images/ui_mesa_desktop_tab2.png)

![Settings](../mesa.wiki/images/ui_mesa_desktop_tab3.png)

![About](../mesa.wiki/images/ui_mesa_desktop_tab4.png)

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
