#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)

def fail(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", flush=True)
    sys.exit(code)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent           # ...\mesa\code
PROJECT_ROOT = CODE_DIR.parent                       # ...\mesa
PARENT_DIR = PROJECT_ROOT.parent                     # ...\wingide

BUILD_FOLDER_ROOT = PARENT_DIR / "build"
DIST_FOLDER_ROOT  = PARENT_DIR / "dist"
FINAL_DIST        = DIST_FOLDER_ROOT / "mesa"
TOOLS_DIST        = FINAL_DIST / "tools"

APP_NAME = "mesa"
ONEDIR_SUBDIR = FINAL_DIST / APP_NAME  # PyInstaller onedir output folder

# ---------------------------------------------------------------------------
# Setup / Cleanup
# ---------------------------------------------------------------------------
def clean_and_prepare() -> None:
    log("Cleaning previous build/dist...")
    shutil.rmtree(BUILD_FOLDER_ROOT, ignore_errors=True)
    # Only remove our app’s dist folder (keep siblings under dist/)
    shutil.rmtree(FINAL_DIST, ignore_errors=True)

    BUILD_FOLDER_ROOT.mkdir(parents=True, exist_ok=True)
    FINAL_DIST.mkdir(parents=True, exist_ok=True)
    TOOLS_DIST.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# PyInstaller
# ---------------------------------------------------------------------------
def ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
        from PyInstaller.__main__ import run as _run  # noqa: F401
    except Exception as e:
        fail(
            "PyInstaller is missing. Activate your venv and run:\n"
            "  pip install -U pyinstaller _pyinstaller_hooks_contrib\n"
            f"Details: {e}"
        )

def run_pyinstaller(args: list[str]) -> None:
    # High recursion depth for deep dependency graphs
    sys.setrecursionlimit(20000)
    # Avoid picking up stray user-site packages (e.g., lingering CuPy)
    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    from PyInstaller.__main__ import run
    log("PyInstaller " + " ".join(args))
    run(args)

def add_data_arg(src_path: Path, dest_name: str) -> list[str]:
    """Build a --add-data argument. On Windows, separator is ';'."""
    return ["--add-data", f"{str(src_path)}{os.pathsep}{dest_name}"]

# ---------------------------------------------------------------------------
# Build profiles
#   Helpers: full GIS stack, onefile
#   Main: lean (no GIS), onedir then flattened to FINAL_DIST
# ---------------------------------------------------------------------------
HELPER_COLLECTS = [
    "--collect-all", "shapely",
    "--collect-all", "pyproj",
    "--collect-all", "fiona",
    "--collect-submodules", "geopandas",
    "--collect-data", "pandas",      # data only, avoid tests
    "--collect-data", "pyarrow",     # data only, avoid tests
    "--collect-submodules", "ttkbootstrap",
    "--collect-submodules", "tkinterweb",
]
HELPER_EXCLUDES = [
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
    "--exclude-module", "fiona._shim",
]

MAIN_COLLECTS = [
    "--collect-data", "pandas",
    "--collect-data", "pyarrow",
    "--collect-submodules", "ttkbootstrap",
    "--collect-submodules", "tkinterweb",
]
MAIN_EXCLUDES = [
    "--exclude-module", "cupy",
    "--exclude-module", "cupy_backends",
    "--exclude-module", "numba",
    "--exclude-module", "pandas.tests",
    "--exclude-module", "pyarrow.tests",
    "--exclude-module", "matplotlib",
    "--exclude-module", "matplotlib.tests",
    "--exclude-module", "pytest",
    "--exclude-module", "scipy",
    "--exclude-module", "IPython",
    "--exclude-module", "jedi",
    "--exclude-module", "geopandas",
    "--exclude-module", "fiona",
    "--exclude-module", "pyproj",
    "--exclude-module", "shapely",
    "--exclude-module", "pysqlite2",
    "--exclude-module", "MySQLdb",
    "--exclude-module", "psycopg2",
    "--exclude-module", "fiona._shim",
]

FLAGS_HELPER = [
    "--windowed",
    "--noconfirm",
    "--onefile",
    "--log-level=WARN",
    "--clean",
] + HELPER_COLLECTS + HELPER_EXCLUDES

FLAGS_MAIN = [
    "--windowed",
    "--noconfirm",
    "--log-level=WARN",
    "--clean",
] + MAIN_COLLECTS + MAIN_EXCLUDES

# ---------------------------------------------------------------------------
# Build steps
# ---------------------------------------------------------------------------
def build_helper(basename: str) -> None:
    pyfile = CODE_DIR / f"{basename}.py"
    if not pyfile.exists():
        log(f"[NOTE] Skipping helper '{basename}' (not found).")
        return

    args = FLAGS_HELPER + [
        "--name", basename,
        "--distpath", str(TOOLS_DIST),
        "--workpath", str(BUILD_FOLDER_ROOT / f"{basename}_build"),
        "--specpath", str(BUILD_FOLDER_ROOT / "helper_specs"),
        str(pyfile),
    ]
    run_pyinstaller(args)

def build_main() -> None:
    # Keep the main app lean; do not embed GIS or tools here.
    data_args: list[str] = []

    # Optional: embed system_resources into onedir (redundant but harmless)
    sysres = CODE_DIR / "system_resources"
    if sysres.exists():
        data_args += add_data_arg(sysres, "system_resources")

    args = FLAGS_MAIN + [
        "--name", APP_NAME,
        "--distpath", str(FINAL_DIST),
        "--workpath", str(BUILD_FOLDER_ROOT / f"{APP_NAME}_build"),
        "--specpath", str(BUILD_FOLDER_ROOT),
    ] + data_args + [str(CODE_DIR / f"{APP_NAME}.py")]

    run_pyinstaller(args)

def flatten_onedir_output() -> None:
    """
    PyInstaller ONEDIR outputs into FINAL_DIST/APP_NAME/.
    Move its contents up into FINAL_DIST so mesa.exe sits next to config.ini.
    """
    src_dir = ONEDIR_SUBDIR
    if not src_dir.is_dir():
        log(f"[NOTE] Expected onedir output at '{src_dir}' was not found (build may have failed earlier).")
        return

    log(f"Flattening onedir output from '{src_dir}' into '{FINAL_DIST}'...")
    for item in src_dir.iterdir():
        dest = FINAL_DIST / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
            shutil.rmtree(item, ignore_errors=True)
        else:
            if dest.exists():
                try:
                    dest.unlink()
                except Exception:
                    pass
            shutil.move(str(item), str(dest))

    # Remove now-empty onedir folder
    try:
        src_dir.rmdir()
    except OSError:
        shutil.rmtree(src_dir, ignore_errors=True)

# ---------------------------------------------------------------------------
# Copy resources next to the app (runtime layout)
# ---------------------------------------------------------------------------
def copy_resources() -> None:
    # Copy runtime dirs next to mesa.exe (recursive)
    for folder in ["qgis", "docs", "input", "output", "system_resources"]:
        src = CODE_DIR / folder
        dst = FINAL_DIST / folder
        if src.exists():
            log(f"Copying '{folder}/' ...")
            shutil.copytree(src, dst, dirs_exist_ok=True)

    # Copy config.ini next to mesa.exe
    cfg = CODE_DIR / "config.ini"
    if cfg.exists():
        shutil.copy2(cfg, FINAL_DIST / "config.ini")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log("\n==============================\n  Building Mesa\n==============================\n")
    log(f"CODE_DIR      = {CODE_DIR}")
    log(f"PROJECT_ROOT  = {PROJECT_ROOT}")
    log(f"PARENT_DIR    = {PARENT_DIR}")
    log(f"BUILD_FOLDER  = {BUILD_FOLDER_ROOT}")
    log(f"DIST_FOLDER   = {DIST_FOLDER_ROOT}")
    log(f"FINAL_DIST    = {FINAL_DIST}")
    log(f"TOOLS_DIST    = {TOOLS_DIST}\n")

    clean_and_prepare()
    ensure_pyinstaller()

    log("Building helper tools (onefile, with GIS stack)...")
    helpers = [
        "assetgroup_edit",
        "atlas_create",
        "atlas_edit",
        "create_raster_tiles",
        "data_import",
        "data_process",
        "data_report",
        "geocodegroup_edit",
        "geocodes_create",
        "lines_admin",
        "lines_process",
        "maps_overview",
        "parametres_setup",
    ]
    for h in helpers:
        build_helper(h)
    log("Helper tools built.\n")

    log("Building main app (ONEDIR, lean)...")
    build_main()
    log("Main app built.\n")

    log("Flattening main app into FINAL_DIST...")
    flatten_onedir_output()

    log("Copying runtime resources next to the app...")
    copy_resources()

    exe_path = FINAL_DIST / f"{APP_NAME}.exe"
    if exe_path.exists():
        log(f"\nBuild complete. Launch here:\n  {exe_path}\n")
    else:
        log("\n[NOTE] mesa.exe not found where expected. Listing FINAL_DIST for troubleshooting:")
        for p in sorted(FINAL_DIST.rglob("*")):
            if p.is_file():
                log(f"  - {p.relative_to(FINAL_DIST)}")

    log(f"\nDistribution ready at:\n  {FINAL_DIST}\n")
    sys.exit(0)

if __name__ == "__main__":
    main()
