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

## Phase 2 — subprocess-helper parity (DONE)

The three subprocess helpers (`combined_map`, `segmentation_setup`,
`special_focus`) now build and resolve on macOS.

- **`mesa.py` change (the only existing-code edit):** `get_script_paths` gained
  a `sys.platform == "darwin"` branch resolving bare executables under
  `system/<name>` / `system/<name>/<name>`. The `else` branch is the original
  `.exe` code verbatim, so Windows behaviour is byte-identical. Nothing else in
  `mesa.py` needed changing — `_launch_gui_process` already branches on
  `os.name` (`start_new_session=True` on non-Windows).
- **`build_mac.py`:** builds each helper as **onedir (not `--windowed`)** into
  `Contents/MacOS/system/<name>/`, keeping `_internal/` (and the `Python`
  framework symlink) adjacent to the executable. `build_all.py` is untouched.

Verified: `combined_map` builds, lands at the resolved path, and launches. The
other two use the same path; `segmentation_setup` additionally collects
sklearn/scipy (untested end-to-end — build all three with a full run).

Still deferred (config location): the frozen app reads `config.ini` next to the
executable. `build_mac.py` copies it into `Contents/MacOS/`, which runs, but a
signed/notarized `.app` is read-only and MESA writes config back, so this needs
a Darwin branch in `mesa.py` resolving config to the working dir or
`~/Library/Application Support/MESA/`. Folded into Phase 3.

### Signing blockers found during Phase 2 (real Phase-3 work)

Ad-hoc `codesign` of the whole bundle does **not** succeed yet — two known
PyInstaller + PySide6 issues surface, both needing Phase-3 attention:

1. **onedir helpers embedded under `Contents/MacOS/system/`** are not a valid
   nested structure for `codesign` to seal the outer bundle (it walks into
   `_internal/…/*.pyx` etc.). Options: make each helper a proper nested `.app`
   under `Contents/Resources/`, or use a helper-tool layout, then sign
   inside-out. `build_mac.py` already signs the nested Mach-O binaries
   individually; the outer seal is the remaining piece.
2. **PySide6 bundles `Designer.app` / `Assistant.app` with symlinks** that
   `codesign` rejects ("main executable … must be a regular file"). Strip the
   Qt dev-tool apps from the bundle (they're dead weight for MESA) or fix up
   their layout before signing.

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
