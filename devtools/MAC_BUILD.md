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

## Phase 2 — full parity (needs existing-code changes — discuss first)

These require editing existing source, so they are intentionally NOT done:

1. **`mesa.py` helper resolution** — `get_script_paths` / helper launchers
   hard-code `.exe`. macOS onefile helpers are bare executables (no suffix).
   Needs a Darwin branch resolving `system/<name>` (and/or an executable inside
   a `.app`).
2. **`build_all.py` Darwin branches** — icon selection (`.icns`), keep `.app`
   instead of flattening, and place onefile helpers where `mesa.py` looks.
   Recommendation: build macOS helpers as **onedir inside
   `mesa.app/Contents/Resources/`** rather than onefile, so signing/notarization
   covers everything in one bundle and there's no `/var/folders` unpack step.
3. **`mesa.py` config location** — the frozen app reads `config.ini` next to the
   executable (`config_file = PROJECT_BASE/config.ini`). `build_mac.py` copies it
   into `Contents/MacOS/`, which runs, but a signed/notarized `.app` is read-only,
   so config can't be edited or written back there. Proper macOS behaviour needs a
   Darwin branch resolving config to the working dir or `~/Library/Application
   Support/MESA/`.

## Phase 3 — Developer ID signing + notarization

Distribution outside the App Store needs a signed, notarized, stapled bundle so
Gatekeeper opens it with no warnings. Until then, `build_mac.py` ad-hoc signs
(`codesign --sign -`) which only satisfies the local machine.

### One-time setup
1. Enrol in the Apple Developer Program (Organization) — https://developer.apple.com/programs/enroll/
   (needs the company D-U-N-S number; keep it out of this repo).
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
