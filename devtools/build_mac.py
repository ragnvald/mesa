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
    sysres = CODE_DIR / "system_resources"
    if sysres.is_dir():
        data_args += add_data(sysres, "system_resources")

    args = (
        flags + COLLECTS + MAIN_EXCLUDES
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
    _place_config(app)
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
        # onedir (NOT --windowed): produces a plain <name>/ folder with the
        # executable and _internal/ adjacent, so the Python dylib resolves and
        # the layout is codesign-clean. mesa.py resolves system/<name>/<name>.
        # The helper is launched as a subprocess, so it needs no .app wrapper.
        flags = ["--noconfirm", "--log-level=WARN"]
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
        run_pyinstaller(args)

        onedir = staging / name  # PyInstaller onedir output folder
        dest = system_dir / name
        if dest.exists():
            shutil.rmtree(dest)
        if not onedir.is_dir():
            print(f"[build_mac] WARNING: expected onedir output missing: {onedir}")
            continue
        shutil.copytree(onedir, dest, symlinks=True)
        print(f"[build_mac] helper '{name}' → {dest} in {time.perf_counter() - t0:.1f}s")


def _place_config(app: Path) -> None:
    """Copy config.ini next to the frozen executable, where mesa.py's
    _bootstrap_config() looks (PROJECT_BASE == dirname(sys.executable) when
    frozen). Mirrors build_all.py copying config.ini next to mesa.exe.

    NOTE (Phase 3): a signed/notarized bundle is read-only, so config.ini
    living inside Contents/MacOS/ can't be user-edited or written back. Proper
    macOS behaviour is config in the working dir or ~/Library/Application
    Support — that needs a config_file resolution change in mesa.py."""
    for src in (REPO_ROOT / "config.ini", CODE_DIR / "config.ini"):
        if src.is_file():
            dst = app / "Contents" / "MacOS" / "config.ini"
            shutil.copy2(src, dst)
            print(f"[build_mac] placed config.ini → {dst}")
            return
    print("[build_mac] WARNING: no config.ini found to place in bundle")


def _is_macho(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            magic = f.read(4)
    except Exception:
        return False
    # arm64 thin (cf fa ed fe) or universal/fat (ca fe ba be / be ba fe ca).
    return magic in (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca")


def _codesign(path: Path) -> tuple[int, str]:
    r = subprocess.run(
        ["codesign", "--force", "--timestamp=none", "--sign", "-", str(path)],
        capture_output=True, text=True,
    )
    return r.returncode, r.stderr.strip()


def adhoc_sign(app: Path) -> None:
    """Ad-hoc, inside-out signature so Gatekeeper lets it launch locally.
    `codesign --deep` chokes on the PyInstaller onedir helpers embedded under
    Contents/MacOS/system/, so sign the nested Mach-O binaries deepest-first,
    then the outer bundle. Replace with a Developer ID + notarization for
    distribution — see devtools/MAC_BUILD.md (Phase 3)."""
    system = app / "Contents" / "MacOS" / "system"
    if system.is_dir():
        machos = [p for p in system.rglob("*")
                  if p.is_file() and not p.is_symlink() and _is_macho(p)]
        machos.sort(key=lambda p: len(p.parts), reverse=True)  # deepest first
        failed = 0
        for m in machos:
            rc, err = _codesign(m)
            if rc != 0:
                failed += 1
        print(f"[build_mac] signed {len(machos) - failed}/{len(machos)} nested helper binaries")
    rc, err = _codesign(app)
    if rc == 0:
        print(f"[build_mac] ad-hoc signed {app.name}")
    else:
        print(f"[build_mac] ad-hoc sign warning (outer bundle): {err}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build macOS mesa.app")
    ap.add_argument("--distpath", default=str(REPO_ROOT / "dist_mac"),
                    help="output dir for mesa.app (default: <repo>/dist_mac)")
    ap.add_argument("--no-helpers", action="store_true", help="skip subprocess helpers")
    ap.add_argument("--helpers", default="", help="comma list of helpers to build (subset)")
    ap.add_argument("--no-sign", action="store_true", help="skip ad-hoc codesign")
    ap.add_argument("--clean", action="store_true", help="PyInstaller --clean")
    ns = ap.parse_args()

    distpath = Path(ns.distpath).resolve()
    distpath.mkdir(parents=True, exist_ok=True)
    app = build_main(distpath, clean=ns.clean)
    if not ns.no_helpers:
        only = [h.strip() for h in ns.helpers.split(",") if h.strip()] or None
        build_helpers(app, distpath, clean=ns.clean, only=only)
    if not ns.no_sign:
        adhoc_sign(app)  # sign last so it covers the helpers too
    print(f"\n[build_mac] DONE → {app}")
    print("[build_mac] launch:  open " + str(app))


if __name__ == "__main__":
    main()
