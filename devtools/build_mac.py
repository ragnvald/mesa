#!/usr/bin/env python3
"""Build a frozen macOS mesa.app with PyInstaller.

Standalone, non-intrusive counterpart to build_all.py's build_main(): it
mirrors the main-app PyInstaller invocation but targets macOS (produces a
.app bundle via --windowed, uses mesa.icns, keeps the bundle instead of the
Windows "flatten to FINAL_DIST" step). It does NOT import build_all.py and
does NOT touch mesa.py — the frozen app works because _ensure_repo_dev_venv()
already bails out on sys.frozen.

Scope: the MAIN app only. The subprocess helpers (combined_map,
segmentation_setup, special_focus) still resolve as `.exe` in mesa.py, so
their buttons won't launch on macOS yet — that needs an existing-code change
(see devtools/MAC_BUILD.md, "Phase 2"). Everything the main window and the
in-process helpers do works from this bundle.

Usage:
    .venv/bin/python3 devtools/build_mac.py [--distpath DIR] [--no-sign] [--clean]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code"
APP_NAME = "mesa"


def add_data(src: Path, dest: str) -> list[str]:
    # macOS/Linux use ':' as the --add-data separator (os.pathsep handles it).
    return ["--add-data", f"{src}{os.pathsep}{dest}"]


def resolve_icon() -> Path | None:
    icns = REPO_ROOT / "system_resources" / "mesa.icns"
    return icns if icns.is_file() else None


# Mirror of build_all.py MAIN_COLLECTS (the main app bundles the full GIS stack
# because the in-process helpers need it). tkagg / tcltk are intentionally
# omitted: this repo's macOS Python has no _tkinter, and MESA renders through
# PySide6 (QtAgg) + Agg, never Tk.
MAIN_COLLECTS = [
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

MAIN_EXCLUDES = [
    "--exclude-module", "cupy",
    "--exclude-module", "cupy_backends",
    "--exclude-module", "numba",
    "--exclude-module", "pandas.tests",
    "--exclude-module", "pyarrow.tests",
    "--exclude-module", "matplotlib.tests",
    "--exclude-module", "pytest",
    "--exclude-module", "scipy",
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

# The helpers that run in-process inside the frozen main app (mirror of
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


def build(distpath: Path, clean: bool) -> Path:
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
        flags
        + MAIN_COLLECTS
        + MAIN_EXCLUDES
        + [
            "--name", APP_NAME,
            "--distpath", str(distpath),
            "--workpath", str(work),
            "--specpath", str(spec),
            "--paths", str(CODE_DIR),
        ]
        + PKG_RESOURCES_HIDDEN_IMPORTS
        + MESA_INPROCESS_HIDDEN_IMPORTS
        + data_args
        + [str(entry)]
    )
    icon = resolve_icon()
    if icon is not None:
        args[0:0] = ["--icon", str(icon)]

    # PyInstaller's analysis recurses past Python's default 1000-frame limit on
    # this dependency graph; run it in a subprocess that lifts the limit first.
    launcher = (
        "import sys; sys.setrecursionlimit(5000); "
        "from PyInstaller.__main__ import run; run()"
    )
    cmd = [sys.executable, "-c", launcher, *args]
    print("[build_mac] PyInstaller " + " ".join(args))
    t0 = time.perf_counter()
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if completed.returncode != 0:
        sys.exit(f"[build_mac] PyInstaller failed (exit {completed.returncode}).")
    app = distpath / f"{APP_NAME}.app"
    print(f"[build_mac] built {app} in {time.perf_counter() - t0:.1f}s")
    _place_config(app)
    return app


def _place_config(app: Path) -> None:
    """Copy config.ini next to the frozen executable, where mesa.py's
    _bootstrap_config() looks (PROJECT_BASE == dirname(sys.executable) when
    frozen). Mirrors build_all.py copying config.ini next to mesa.exe.

    NOTE (Phase 2/3): a signed/notarized bundle is read-only, so config.ini
    living inside Contents/MacOS/ can't be user-edited or written back. Proper
    macOS behaviour is config in the working dir or ~/Library/Application
    Support — that needs a config_file resolution change in mesa.py. For the
    POC this placement is enough to run the app."""
    import shutil
    for src in (REPO_ROOT / "config.ini", CODE_DIR / "config.ini"):
        if src.is_file():
            dst = app / "Contents" / "MacOS" / "config.ini"
            shutil.copy2(src, dst)
            print(f"[build_mac] placed config.ini → {dst}")
            return
    print("[build_mac] WARNING: no config.ini found to place in bundle")


def adhoc_sign(app: Path) -> None:
    """Ad-hoc signature so Gatekeeper lets it launch locally (no Developer ID
    yet). Replace with a real Developer ID + notarization for distribution —
    see devtools/MAC_BUILD.md."""
    r = subprocess.run(
        ["codesign", "--force", "--deep", "--sign", "-", str(app)],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        print(f"[build_mac] ad-hoc signed {app.name}")
    else:
        print(f"[build_mac] ad-hoc sign warning: {r.stderr.strip()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build macOS mesa.app")
    ap.add_argument("--distpath", default=str(REPO_ROOT / "dist_mac"),
                    help="output dir for mesa.app (default: <repo>/dist_mac)")
    ap.add_argument("--no-sign", action="store_true", help="skip ad-hoc codesign")
    ap.add_argument("--clean", action="store_true", help="PyInstaller --clean")
    ns = ap.parse_args()

    distpath = Path(ns.distpath).resolve()
    distpath.mkdir(parents=True, exist_ok=True)
    app = build(distpath, clean=ns.clean)
    if not ns.no_sign:
        adhoc_sign(app)
    print(f"\n[build_mac] DONE → {app}")
    print("[build_mac] launch:  open " + str(app))


if __name__ == "__main__":
    main()
