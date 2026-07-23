# Building MESA for macOS

macOS counterpart to the Windows build (`devtools/build_all.py` +
`devtools/compile_win_11.bat`). Same principle: PyInstaller freezes the app so
end users need no Python. This document tracks the current state, what still
needs doing, and the Developer ID signing/notarization pipeline.

## Why a Mac build is feasible

Every runtime dependency (PySide6, GDAL via pyogrio, shapely, pyproj, h3,
sklearn/scipy) already has working macOS arm64 wheels — MESA runs from source
on Apple Silicon. Freezing is therefore an orchestration job, not a port.

## Phase 1 — main app (DONE, non-intrusive)

`devtools/build_mac.py` builds `mesa.app` from the main entry point. It is a
standalone mirror of `build_all.py`'s `build_main()` and touches no existing
source: the frozen app works because `mesa.py`'s `_ensure_repo_dev_venv()`
already returns early on `sys.frozen`.

macOS adaptations that live entirely in `build_mac.py`:
- `--icon system_resources/mesa.icns` (generated from `icon.png`; the Windows
  `mesa.ico` is untouched).
- `--windowed` yields a real `.app` bundle; no Windows "flatten to FINAL_DIST".
- Tk is excluded — this environment's Python has no `_tkinter`, and MESA
  renders through PySide6 (QtAgg) + Agg. The `matplotlib.backends.backend_tkagg`
  hidden import and Tcl/Tk data bundling from the Windows build are dropped.

Build and run:

```bash
.venv/bin/python3 devtools/build_mac.py --distpath dist_mac --clean
open dist_mac/mesa.app
```

Requirements for a clean compile venv: `requirements_compile_mac.txt`
(production should use a dedicated `.venv_compile`; the POC reused `.venv`).

### Known Phase-1 limitation

Only the MAIN app is frozen. The subprocess helpers (`combined_map`,
`segmentation_setup`, `special_focus`) are resolved as `.exe` in `mesa.py`
(`get_script_paths`, `_launch_helper_subprocess`), so their buttons won't
launch on macOS yet. The seven in-process helpers (geocode/asset/atlas/
processing/report/analysis) work, because they're bundled as hidden imports.

## Phase 2 + size optimization — subprocess helpers via re-exec (DONE)

The three subprocess helpers (`combined_map`, `segmentation_setup`,
`special_focus`) run on macOS by **re-exec of the main frozen binary**, not as
separate bundles — so the Python/Qt/GIS stack isn't duplicated. Result: the
signed bundle is **~1.1 GB** instead of ~3.6 GB (3 duplicated stacks removed).

- **`mesa.py` changes (guarded, Windows byte-identical):**
  - `_maybe_run_helper()` — at `__main__`, if `sys.platform == "darwin"` and
    `--run-helper <name>` is present, `runpy.run_module(name, "__main__")` and
    exit. No-op off macOS / without the flag.
  - `_launch_helper_subprocess` — on frozen macOS launches
    `[sys.executable, "--run-helper", <name>, …]` instead of a separate exe.
    The Windows/frozen and source branches are unchanged.
  - (`get_script_paths` also kept its earlier Darwin `.app` fallback, now
    unused by the re-exec path but harmless.)
- **`build_mac.py`:** the main build imports the three helpers
  (`RUN_HELPER_IMPORTS`) plus sklearn/scipy (segmentation) and webview
  (special_focus), and keeps scipy (`BASE_EXCLUDES`). No separate helper bundles
  by default (`--separate-helpers` keeps the legacy nested-`.app` path).
  `build_all.py` is untouched.

Verified end-to-end: all three helpers launch via `mesa --run-helper <name>`;
the main app launches; the bundle passes `codesign --verify --strict`.

### Runtime data layout (macOS)

A notarized `.app` is read-only, so the frozen Mac build keeps all writable data
in **`~/Documents/MESA/`** (`mesa.py` `_resolve_working_dir`), not next to the
executable. On first run `_seed_working_dir` copies the reference material
bundled in the app (config.ini, docs, qgis templates, system_resources) into it,
and `check_and_create_folders` makes the empty `input/{asset,geocode,lines,
images}` and `output` folders — mirroring what `build_all.py` stages next to
`mesa.exe` on Windows. The 1.5 GB `input/` working data is NOT bundled. Because
the app never writes into its own bundle, the signature/notarization survives
every run. On Windows/dev these paths resolve to the working dir itself, so the
seeding is a no-op.

Resources are bundled from the repo root (`system_resources/`, `docs/`,
`qgis/`), not `code/` — bundling `code/system_resources` (which doesn't exist)
silently dropped the top banner.

### .dmg

`build_mac.py --dmg` (or `make_dmg`) builds a compressed `.dmg` with a
drag-to-Applications layout and a custom volume icon (mesa.icns). Build the dmg
*after* notarizing + stapling the app so the copy inside carries the staple.

## Phase 3a — Developer ID signing (DONE)

`build_mac.py --sign-id "Developer ID Application: … (TEAMID)"` produces a
bundle that passes `codesign --verify --strict` and that `spctl` reports as a
valid (un-notarized) Developer ID signature. What it took:

- **Strip the Qt dev tools** (`strip_qt_dev_tools`): PySide6 ships Designer /
  Assistant / Linguist as **both** a `<Name>.app` symlink and the real
  `<Name>__dot__app` directory. Removing only the `.app` left dangling symlinks
  that broke `codesign --verify`; strip **both** forms (symlink → unlink, real
  dir → rmtree).
- **Hardened runtime + entitlements** (`devtools/mesa.entitlements`):
  disable-library-validation, allow-unsigned-executable-memory,
  allow-dyld-environment-variables — PyInstaller apps fail the hardened runtime
  without these.
- **One `codesign --deep` pass** per bundle (not per-file): ~2700 nested Mach-O
  timestamped individually took far too long; a single `--deep` process is the
  way. `--timestamp` (networked, slow) is applied only for the notarization
  build (`--timestamp`), off for fast local validation.

## Phase 3 — Developer ID signing + notarization

Distribution outside the App Store needs a signed, notarized, stapled bundle so
Gatekeeper opens it with no warnings. Until then, `build_mac.py` ad-hoc signs
(`codesign --sign -`) which only satisfies the local machine.

### One-time setup
1. Enrol in the Apple Developer Program — https://developer.apple.com/programs/enroll/
   Individual enrolment is enough to sign + notarize (Gatekeeper then shows the
   individual's name); Organization (needs a D-U-N-S number) only changes the
   displayed developer name. Keep the Team ID out of this repo — pass it to
   codesign / notarytool as a parameter or env var.
2. Generate a **Developer ID Application** certificate (developer.apple.com →
   Certificates) and install it in the login keychain.
3. Store notarytool credentials once:
   `xcrun notarytool store-credentials mesa-notary --apple-id <id> --team-id <TEAMID> --password <app-specific-password>`
4. `xcode-select --install` (provides `codesign`, `notarytool`, `stapler`).

### Hardened-runtime entitlements (PyInstaller needs these)
Notarization requires the hardened runtime, under which PyInstaller apps fail
without an entitlements plist. Create `devtools/mesa.entitlements`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>com.apple.security.cs.disable-library-validation</key><true/>
  <key>com.apple.security.cs.allow-unsigned-executable-memory</key><true/>
  <key>com.apple.security.cs.allow-dyld-environment-variables</key><true/>
</dict></plist>
```

### Sign → notarize → staple
```bash
codesign --deep --force --options runtime \
  --entitlements devtools/mesa.entitlements \
  --sign "Developer ID Application: <NAME> (<TEAMID>)" dist_mac/mesa.app
ditto -c -k --keepParent dist_mac/mesa.app mesa.zip
xcrun notarytool submit mesa.zip --keychain-profile mesa-notary --wait
xcrun stapler staple dist_mac/mesa.app
```

Then wrap in a `.dmg` for distribution.

## Architecture

Build **arm64-only** (Apple Silicon). Universal2 would need universal wheels
for the whole GIS stack, which are arch-specific — not worth it.
