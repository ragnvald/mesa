# MESA codebase instructions (initial draft)

Use this document as the first stop before editing the `mesa` repository. Update it whenever new conventions emerge so the next contributor can stay aligned without re-reading the full chat history.

## 1. Scope & goals
- Maintain the desktop launcher (`mesa.py`/`mesa.exe`) and helper tools that automate the MESA 5 workflow: import → configure → process → review/publish.
- Keep the packaged experience friendly for non-technical analysts; advanced wiring and build steps belong in developer docs, not the UI.
- Ensure every change keeps GeoParquet/MBTiles outputs reproducible from the public source.

## 1.1 Review-before-commit rule
- Never `git add`, `git commit`, `git push`, or otherwise publish changes unless the user explicitly asks for it after reviewing the proposed diffs/text.
- Default workflow: make changes locally → report what changed → wait for approval → then (and only then) stage/commit/push.

## 1.2 Keep this document current
- Whenever you edit this file, update the **Last updated** line at the bottom with the current date **and time** in the format `YYYY-MM-DD HH:MM`.

## 1.3 Time zone for logs
- Any timestamps we add to logs, status messages, docs, or build output should be in **Norway local time** (`Europe/Oslo`, i.e. CET/CEST), since the lead programmer is in Norway. He likes things to be convenient for himself.

## 2. Repository layout (high level)
| Path | Purpose |
| --- | --- |
| `mesa.py` | Tk/ttkbootstrap desktop that orchestrates all helper scripts and exposes Workflows/Status/Settings/About tabs. |
| `code/` | Individual command-line utilities (imports, processing, atlas tools, etc.). Each should run standalone, via `mesa.py`, and when compiled. |
| `input/` | Expected user-provided data (`asset`, `geocode`, `lines`, etc.). Never ship sample data here. |
| `output/` | Runtime products (GeoParquet, MBTiles, PDF reports, logs). Preserve structure when adding new exporters. |
| `config.ini` | Central settings + usage counters. Always ensure `[DEFAULT]` remains intact. |
| `system_resources/` | Icons and HTML assets bundled with the executable. |

## 3. Environment & tooling
1. Use Python 3.11+ (matches the packaged interpreter bundled in releases).
2. Create a venv inside the repo: `python -m venv .venv`.
3. Activate and install deps: `pip install -r requirements_all.txt` (or `requirements.txt` inside `code/` for lighter work).
4. Launch the UI with `python mesa.py` to test changes before compiling.
5. Keep ttkbootstrap as the primary UI framework; discuss before adding new GUI toolkits.

## 4. UI conventions (current as of MESA 5)
- **Header band**: intro text plus Exit button lives at the top. Text should clearly direct users to start in Workflows and monitor Status.
- **Aspect ratio**: the main window enforces 5:3 with sensible minimums; honor that logic when resizing or adding views.
- **Responsive layout**: Workflows grid supports 1–2 columns; sections like “Prepare data” and “Review & publish” display their actions in two sub-columns. Do nott hardcode pixel-perfect positions that would break this responsiveness.
- **Tab order**: Workflows (default) → Status → Settings → About. The Register tab is retired; keep UUID handling headless.
- **Buttons**: prefer verb-first labels (“Import data”, “Run area processing”). Pair each button with a concise helper line.
- **Styling**: stay within the current ttkbootstrap theme; introduce new bootstyles only if they harmonize with the palette.

## 5. Data & configuration rules
- Always resolve paths relative to `PROJECT_BASE`, never to the working directory.
- Keep UUID and logging updates in `config.ini`; even without the Register tab, anonymous telemetry must keep working.
- When adding new stats counters, update both the config defaults and the Status tab so users can see them.
- Treat `output/` as user-owned; do not delete files unless explicitly confirmed by the workflow.

## 6. Coding standards
- Stick to ASCII in source files unless a file already uses UTF-8 symbols for a justified reason (e.g., documentation text).
- Use clear helper functions with succinct comments when the intent is non-obvious (e.g., geometry enforcement, responsive relayout).
- Avoid expanding dependencies; prefer standard library + existing packages unless there is a compelling case.
- Maintain backwards compatibility with both the .py and compiled .exe launch paths (check `get_script_paths` usage before moving code).
- Log meaningful events via `log_to_logfile` instead of printing directly to stdout from GUI callbacks.

## 7. Testing expectations
- After UI changes, run `python mesa.py` and manually verify: window sizing, tab ordering, button commands, Status counters.
- For backend scripts, run the relevant helper (e.g., `python code/data_process.py --help`) using sample data in `input/`.
- Keep an eye on `log.txt` for errors; the GUI swallows some exceptions if they are only logged.
- No automated test suite exists yet; document manual verification steps in PRs/issues.

## 7.1 Compilation/build steps require explicit approval
- **Always ask first** before running any compilation/build steps (PyInstaller builds, `.bat`/`.ps1` build scripts, `dotnet build`, etc.).
- Default approach is **code inspection + lightweight Python checks** (read the code, `py_compile`, `--help`, minimal repro snippets) and only compile when the user explicitly confirms that it’s worth the time/compute.
- If a build is needed for confidence (packaging regressions, missing bundled assets), propose the smallest build that answers the question (e.g., a single helper rather than full distribution) and wait for approval.

## 8. Documentation & assets
- When UI elements change, update the wiki (`mesa.wiki`) and note which screenshots must be refreshed (use placeholder callouts if images are pending).
- Inline tooltips or info icons should link to stable wiki anchors.
- If a feature is removed (e.g., Register tab), scrub references in both the UI and wiki to avoid confusing users.

## 9. Build & distribution (heads-up)
- The project is packaged via PyInstaller; any new data files must be added to the spec before release.
- Keep executable launch parity: workflows triggered from `mesa.py` must still be runnable by double-clicking the compiled helper exe.
- Document new build steps or flags here if the release pipeline changes.

**Local developer workflow:** we treat builds as **full builds** (main + all helper tools). Use `code/compile_win_11.bat` as the entrypoint; do not rely on partial-build environment toggles in normal work.

---
_Last updated: 2026-01-10 00:00_
