#!/usr/bin/env python3
"""Build a frozen macOS mesa.app with PyInstaller.

Standalone, non-intrusive counterpart to build_all.py's build_main(): it
mirrors the main-app PyInstaller invocation but targets macOS (produces a
.app bundle via --windowed, uses mesa.icns, keeps the bundle instead of the
Windows "flatten to FINAL_DIST" step). It does NOT import build_all.py and
leaves the Windows build byte-identical.

The three subprocess helpers (combined_map, segmentation_setup, special_focus)
are built as onedir bundles placed under mesa.app/Contents/MacOS/system/<name>/,
where mesa.py's get_script_paths() Darwin branch resolves them. The seven
in-process helpers are bundled into the main app as hidden imports.

The frozen app needs no source changes to launch (mesa.py's
_ensure_repo_dev_venv() bails on sys.frozen); the only existing-code change for
helper support is the Darwin branch in get_script_paths, guarded so Windows is
untouched.

Usage:
    .venv/bin/python3 devtools/build_mac.py [--distpath DIR] [--no-helpers]
        [--helpers a,b] [--no-sign] [--clean]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code"
APP_NAME = "mesa"

# The helpers mesa.py launches as subprocesses (INPROCESS_HELPERS run inside the
# main app instead). Built as separate onedir bundles inside the .app.
SUBPROCESS_HELPERS = ["combined_map", "segmentation_setup", "special_focus"]


def add_data(src: Path, dest: str) -> list[str]:
    # macOS/Linux use ':' as the --add-data separator (os.pathsep handles it).
    return ["--add-data", f"{src}{os.pathsep}{dest}"]


def resolve_icon() -> Path | None:
    icns = REPO_ROOT / "system_resources" / "mesa.icns"
    return icns if icns.is_file() else None


# Full GIS/UI collect set. tkagg / tcltk are intentionally omitted: this repo's
# macOS Python has no _tkinter, and MESA renders through PySide6 (QtAgg) + Agg.
COLLECTS = [
    "--collect-data", "pandas",
    "--collect-data", "pyarrow",
    "--collect-all", "PySide6",
    "--collect-all", "shapely",
    "--collect-all", "pyproj",
    "--collect-all", "pyogrio",
    "--collect-submodules", "geopandas",
    "--collect-all", "matplotlib",
    "--collect-all", "docx",
    "--collect-all", "h3",
    "--collect-submodules", "openpyxl",
    "--hidden-import", "et_xmlfile",
]

# Excludes shared by main + helpers. scipy is dropped here but re-added for
# helpers that need the sklearn stack (segmentation).
BASE_EXCLUDES = [
    "--exclude-module", "cupy",
    "--exclude-module", "cupy_backends",
    "--exclude-module", "numba",
    "--exclude-module", "pandas.tests",
    "--exclude-module", "pyarrow.tests",
    "--exclude-module", "matplotlib.tests",
    "--exclude-module", "pytest",
    "--exclude-module", "IPython",
    "--exclude-module", "jedi",
    "--exclude-module", "pysqlite2",
    "--exclude-module", "MySQLdb",
    "--exclude-module", "psycopg2",
    "--exclude-module", "tkinterweb",
    "--exclude-module", "influxdb_client",
    # macOS Python here has no Tk; keep it out of the graph entirely.
    "--exclude-module", "tkinter",
    "--exclude-module", "_tkinter",
]

MAIN_EXCLUDES = BASE_EXCLUDES + ["--exclude-module", "scipy"]

# setuptools/pkg_resources shims frozen builds load at runtime.
PKG_RESOURCES_HIDDEN_IMPORTS = [
    "--hidden-import", "jaraco.text",
    "--hidden-import", "jaraco.functools",
    "--hidden-import", "jaraco.context",
    "--hidden-import", "more_itertools",
    "--hidden-import", "platformdirs",
    "--hidden-import", "autocommand",
    "--hidden-import", "backports.tarfile",
]

# Helpers that run in-process inside the frozen main app (mirror of
# build_all.py MESA_INPROCESS_HIDDEN_IMPORTS, minus the Tk matplotlib backend).
MESA_INPROCESS_HIDDEN_IMPORTS = [
    "--hidden-import", "geocode_manage",
    "--hidden-import", "asset_manage",
    "--hidden-import", "atlas_manage",
    "--hidden-import", "processing_setup",
    "--hidden-import", "processing_pipeline_run",
    "--hidden-import", "report_generate",
    "--hidden-import", "analysis_present",
    "--hidden-import", "mesa_shared",
    "--hidden-import", "mesa_constants",
    "--hidden-import", "analysis_setup",
    "--hidden-import", "mesa_osm_tiles",
    "--hidden-import", "matplotlib.backends.backend_agg",
]

# On macOS the subprocess helpers run via re-exec of the main binary
# (mesa --run-helper <name>), not as separate bundles, so the main app must
# import them and their extra deps. This makes the main app a bit larger but
# removes ~3 full duplicated stacks (~2.5 GB saved). See mesa.py _maybe_run_helper.
RUN_HELPER_IMPORTS = [
    "--hidden-import", "combined_map",
    "--hidden-import", "segmentation_setup",
    "--hidden-import", "special_focus",
    # segmentation_setup needs the sklearn/scipy stack; special_focus uses pywebview.
    "--collect-all", "sklearn",
    "--collect-all", "scipy",
    "--collect-all", "webview",
]


def run_pyinstaller(args: list[str]) -> None:
    # PyInstaller's analysis recurses past Python's default 1000-frame limit on
    # this dependency graph; run it in a subprocess that lifts the limit first.
    launcher = (
        "import sys; sys.setrecursionlimit(5000); "
        "from PyInstaller.__main__ import run; run()"
    )
    cmd = [sys.executable, "-c", launcher, *args]
    print("[build_mac] PyInstaller " + " ".join(a for a in args if not a.startswith("/")))
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if completed.returncode != 0:
        sys.exit(f"[build_mac] PyInstaller failed (exit {completed.returncode}).")


def build_main(distpath: Path, clean: bool) -> Path:
    work = distpath.parent / "_mac_build"
    spec = distpath.parent / "_mac_spec"
    entry = REPO_ROOT / f"{APP_NAME}.py"
    if not entry.is_file():
        sys.exit(f"[build_mac] entry point not found: {entry}")

    flags = ["--windowed", "--noconfirm", "--log-level=WARN"]
    if clean:
        flags.append("--clean")

    data_args: list[str] = []
    # The real resources (top_graphics.png banner, mesa.ico window icon, …) live
    # at the repo root, not under code/. Bundling code/system_resources (which
    # doesn't exist) silently dropped the top banner in the frozen app.
    sysres = REPO_ROOT / "system_resources"
    if not sysres.is_dir():
        sysres = CODE_DIR / "system_resources"
    if sysres.is_dir():
        data_args += add_data(sysres, "system_resources")
    # Bundle read-only reference material as templates. mesa.py seeds these into
    # the writable working dir (~/Documents/MESA) on first run — mirrors what
    # build_all.py copies next to mesa.exe on Windows. The app is never written
    # to, so its signature/notarization stays intact. input/ (1.5 GB of working
    # data) is NOT bundled; its empty subfolders are created at runtime.
    cfg = REPO_ROOT / "config.ini"
    if cfg.is_file():
        data_args += add_data(cfg, ".")
    for name in ("docs", "qgis"):
        d = REPO_ROOT / name
        if d.is_dir():
            data_args += add_data(d, name)

    # BASE_EXCLUDES (not MAIN_EXCLUDES) keeps scipy, which segmentation_setup
    # needs now that it runs in-binary via re-exec.
    args = (
        flags + COLLECTS + RUN_HELPER_IMPORTS + BASE_EXCLUDES
        + [
            "--name", APP_NAME,
            "--distpath", str(distpath),
            "--workpath", str(work),
            "--specpath", str(spec),
            "--paths", str(CODE_DIR),
        ]
        + PKG_RESOURCES_HIDDEN_IMPORTS + MESA_INPROCESS_HIDDEN_IMPORTS + data_args
        + [str(entry)]
    )
    icon = resolve_icon()
    if icon is not None:
        args[0:0] = ["--icon", str(icon)]

    t0 = time.perf_counter()
    run_pyinstaller(args)
    app = distpath / f"{APP_NAME}.app"
    print(f"[build_mac] built {app.name} in {time.perf_counter() - t0:.1f}s")
    return app


def build_helpers(app: Path, distpath: Path, clean: bool, only: list[str] | None) -> None:
    """Build each subprocess helper as onedir into the app under
    Contents/MacOS/system/<name>/, matching mesa.py's Darwin resolution."""
    names = only if only else SUBPROCESS_HELPERS
    system_dir = app / "Contents" / "MacOS" / "system"
    system_dir.mkdir(parents=True, exist_ok=True)
    staging = distpath.parent / "_mac_helpers"

    for name in names:
        pyfile = CODE_DIR / f"{name}.py"
        if not pyfile.is_file():
            print(f"[build_mac] skip helper '{name}' (not found)")
            continue
        t0 = time.perf_counter()
        # segmentation needs a working sklearn/scipy stack; others don't.
        excludes = BASE_EXCLUDES if name == "segmentation_setup" else MAIN_EXCLUDES
        extra = ["--collect-all", "sklearn", "--collect-all", "scipy"] if name == "segmentation_setup" else []
        # --windowed onedir → a proper <name>.app bundle. codesign recognizes a
        # nested .app and seals it recursively; a bare onedir folder under
        # Contents/MacOS/ cannot be sealed (notarization blocker). mesa.py
        # resolves system/<name>.app/Contents/MacOS/<name>.
        flags = ["--windowed", "--noconfirm", "--log-level=WARN"]
        if clean:
            flags.append("--clean")
        args = (
            flags + COLLECTS + excludes + extra
            + [
                "--name", name,
                "--distpath", str(staging),
                "--workpath", str(distpath.parent / f"_mac_build_{name}"),
                "--specpath", str(distpath.parent / "_mac_spec"),
                "--paths", str(CODE_DIR),
            ]
            + PKG_RESOURCES_HIDDEN_IMPORTS + [str(pyfile)]
        )
        icon = resolve_icon()
        if icon is not None:
            args[0:0] = ["--icon", str(icon)]
        run_pyinstaller(args)

        produced = staging / f"{name}.app"  # PyInstaller --windowed bundle
        dest = system_dir / f"{name}.app"
        if dest.exists():
            shutil.rmtree(dest)
        if not produced.is_dir():
            print(f"[build_mac] WARNING: expected .app output missing: {produced}")
            continue
        shutil.copytree(produced, dest, symlinks=True)
        strip_qt_dev_tools(dest)  # drop Qt dev tools from each helper too
        print(f"[build_mac] helper '{name}' → {dest} in {time.perf_counter() - t0:.1f}s")


# PyInstaller ships each Qt dev tool twice: a "<Name>.app" symlink pointing at
# the real "<Name>__dot__app" directory (its '.'→'__dot__' name mangling).
# Strip both forms, or the real dir's dangling internal symlinks break
# codesign --verify.
QT_DEV_APPS = (
    "Assistant.app", "Designer.app", "Linguist.app",
    "Assistant__dot__app", "Designer__dot__app", "Linguist__dot__app",
)


def strip_qt_dev_tools(app: Path) -> int:
    """Remove the PySide6 developer-tool .app bundles (Qt Designer/Assistant/
    Linguist). MESA never uses them, and their symlinked Contents/MacOS breaks
    codesign ('main executable must be a regular file' / verify 'No such file').

    PyInstaller cross-links Frameworks/ and Resources/, so a match can be either
    a symlink (unlink) or the real directory (rmtree) — shutil.rmtree refuses a
    symlink, which silently no-op'd the old version. Walk without following
    symlinks and handle both. Returns count removed."""
    targets = set(QT_DEV_APPS)
    removed = 0
    for root, dirs, _files in os.walk(app, followlinks=False):
        for entry in list(dirs):
            if entry in targets:
                p = Path(root) / entry
                if p.is_symlink():
                    p.unlink()
                else:
                    shutil.rmtree(p, ignore_errors=True)
                removed += 1
                dirs.remove(entry)  # don't descend into what we just removed
    return removed


def _codesign_deep(bundle: Path, identity: str, entitlements: Path | None,
                   runtime: bool, timestamp: bool) -> tuple[int, str]:
    """One --deep pass over a bundle. --deep recurses all nested Mach-O in a
    single process (fast; no per-file network round-trips) and, because the
    helpers are now proper nested .app bundles, seals them recursively."""
    cmd = ["codesign", "--force", "--deep", "--sign", identity]
    if runtime:
        cmd += ["--options", "runtime"]
    cmd += ["--timestamp"] if timestamp else ["--timestamp=none"]
    if entitlements is not None:
        cmd += ["--entitlements", str(entitlements)]
    cmd.append(str(bundle))
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stderr.strip()


def sign_bundle(app: Path, identity: str, entitlements: Path | None,
                runtime: bool, timestamp: bool) -> None:
    """Sign each nested helper .app first (so they get hardened runtime +
    entitlements of their own), then the main bundle. Pass a real
    'Developer ID Application: …' identity for a notarizable build; '-' is a
    local ad-hoc signature. `timestamp=False` skips the (slow, networked)
    secure timestamp — use it for fast local validation, on for notarization."""
    strip_qt_dev_tools(app)
    system = app / "Contents" / "MacOS" / "system"
    helpers = sorted(system.glob("*.app")) if system.is_dir() else []
    for h in helpers:
        rc, err = _codesign_deep(h, identity, entitlements, runtime, timestamp)
        print(f"[build_mac] helper {h.name}: {'signed' if rc == 0 else 'FAILED — ' + err}")
    rc, err = _codesign_deep(app, identity, entitlements, runtime, timestamp)
    if rc == 0:
        print(f"[build_mac] signed {app.name} ({'Developer ID' if identity != '-' else 'ad-hoc'}"
              f"{', timestamped' if timestamp else ''})")
    else:
        print(f"[build_mac] bundle sign FAILED: {err}")


def adhoc_sign(app: Path) -> None:
    sign_bundle(app, "-", None, runtime=False, timestamp=False)


def make_dmg(app: Path, out: Path, volname: str) -> None:
    """Compressed .dmg with a drag-to-Applications layout and a custom volume
    icon (mesa.icns). ditto preserves the app's notarization staple."""
    icns = REPO_ROOT / "system_resources" / "mesa.icns"
    stage = out.parent / "_dmg_stage"
    rw = out.parent / "_dmg_rw.dmg"
    for p in (stage, rw, out):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()
    stage.mkdir(parents=True)
    subprocess.run(["ditto", str(app), str(stage / "MESA.app")], check=True)
    (stage / "Applications").symlink_to("/Applications")
    if icns.is_file():
        shutil.copy2(icns, stage / ".VolumeIcon.icns")
    subprocess.run(["hdiutil", "create", "-format", "UDRW", "-volname", volname,
                    "-srcfolder", str(stage), str(rw)], check=True, capture_output=True)
    r = subprocess.run(["hdiutil", "attach", str(rw), "-nobrowse", "-noverify"],
                       capture_output=True, text=True)
    mp = next((ln.split("\t")[-1].strip() for ln in r.stdout.splitlines()
               if "/Volumes/" in ln), None)
    if mp and icns.is_file():
        subprocess.run(["SetFile", "-a", "C", mp], capture_output=True)
    if mp:
        subprocess.run(["hdiutil", "detach", mp], capture_output=True)
    subprocess.run(["hdiutil", "convert", str(rw), "-format", "UDZO", "-o", str(out)],
                   check=True, capture_output=True)
    rw.unlink()
    shutil.rmtree(stage, ignore_errors=True)
    print(f"[build_mac] dmg → {out}")


def _config_version() -> str:
    try:
        for line in (REPO_ROOT / "config.ini").read_text().splitlines():
            if line.strip().startswith("mesa_version"):
                return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return "dev"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build macOS mesa.app")
    ap.add_argument("--distpath", default=str(REPO_ROOT / "dist_mac"),
                    help="output dir for mesa.app (default: <repo>/dist_mac)")
    ap.add_argument("--separate-helpers", action="store_true",
                    help="legacy: build helpers as separate nested .app bundles "
                         "instead of embedding them in the main binary (re-exec)")
    ap.add_argument("--helpers", default="", help="comma list of helpers (with --separate-helpers)")
    ap.add_argument("--no-sign", action="store_true", help="skip codesign")
    ap.add_argument("--sign-id", default="-",
                    help="codesign identity (default '-' = ad-hoc; pass "
                         "'Developer ID Application: … (TEAMID)' for a notarizable build)")
    ap.add_argument("--timestamp", action="store_true",
                    help="secure (networked) timestamp — required for notarization, slow")
    ap.add_argument("--dmg", action="store_true",
                    help="also build a compressed .dmg with a custom volume icon")
    ap.add_argument("--dmg-out", default="",
                    help="output path for the .dmg (default ~/Desktop/MESA-<version>.dmg)")
    ap.add_argument("--clean", action="store_true", help="PyInstaller --clean")
    ns = ap.parse_args()

    distpath = Path(ns.distpath).resolve()
    distpath.mkdir(parents=True, exist_ok=True)
    app = build_main(distpath, clean=ns.clean)
    if ns.separate_helpers:
        only = [h.strip() for h in ns.helpers.split(",") if h.strip()] or None
        build_helpers(app, distpath, clean=ns.clean, only=only)
    if not ns.no_sign:
        devid = ns.sign_id != "-"
        ent = (REPO_ROOT / "devtools" / "mesa.entitlements") if devid else None
        sign_bundle(app, ns.sign_id, ent if (ent and ent.is_file()) else None,
                    runtime=devid, timestamp=ns.timestamp)
    if ns.dmg:
        ver = _config_version()
        out = Path(ns.dmg_out).expanduser() if ns.dmg_out else \
            Path.home() / "Desktop" / f"MESA-{ver}.dmg"
        make_dmg(app, out, f"MESA {ver}")
    print(f"\n[build_mac] DONE → {app}")
    print("[build_mac] launch:  open " + str(app))


if __name__ == "__main__":
    main()
