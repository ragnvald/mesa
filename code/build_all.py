#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import shutil
from pathlib import Path

# --- Robust logging helpers -------------------------------------------------
def log(msg: str):
    print(msg, flush=True)

def fail(msg: str, code: int = 1):
    print(f"[FEIL] {msg}", flush=True)
    sys.exit(code)

# --- Paths ------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent                           # ...\mesa\code
PROJECT_ROOT = CODE_DIR.parent                                       # ...\mesa
PARENT_DIR = PROJECT_ROOT.parent                                     # ...\wingide

BUILD_FOLDER_ROOT = PARENT_DIR / "build"
DIST_FOLDER_ROOT  = PARENT_DIR / "dist"
FINAL_DIST        = DIST_FOLDER_ROOT / "mesa"
TOOLS_DIST        = FINAL_DIST / "tools"

# --- Safety cleanup ----------------------------------------------------------
def clean_and_prepare():
    log("Cleaning up build/dist...")
    shutil.rmtree(BUILD_FOLDER_ROOT, ignore_errors=True)
    shutil.rmtree(FINAL_DIST, ignore_errors=True)
    (BUILD_FOLDER_ROOT).mkdir(parents=True, exist_ok=True)
    (FINAL_DIST).mkdir(parents=True, exist_ok=True)
    (TOOLS_DIST).mkdir(parents=True, exist_ok=True)

# --- PyInstaller build -------------------------------------------------------
def ensure_pyinstaller():
    try:
        import PyInstaller  # noqa: F401
        from PyInstaller.__main__ import run as _run  # noqa: F401
    except Exception as e:
        fail("PyInstaller missing. Activate .venv and run: pip install pyinstaller _pyinstaller_hooks_contrib\n"
             f"Details: {e}")

def run_pyinstaller(args):
    # Øk rekursjonsgrensen tidlig for å unngå RecursionError i dype avhengigheter
    sys.setrecursionlimit(20000)
    from PyInstaller.__main__ import run
    log("PyInstaller " + " ".join(args))
    run(args)

COMMON_COLLECTS = [
    "--collect-all", "shapely",
    "--collect-all", "pyproj",
    "--collect-all", "fiona",
    "--collect-all", "geopandas",
    "--collect-all", "pandas",
    "--collect-all", "pyarrow",
    "--collect-all", "ttkbootstrap",
    "--collect-all", "tkinterweb",
]

COMMON_EXCLUDES = [
    "--exclude-module", "cupy",
    "--exclude-module", "cupy_backends",
    "--exclude-module", "numba",
]

COMMON_FLAGS = [
    "--windowed",
    "--noconfirm",
    "--onefile",
    "--log-level=WARN",
    "--hidden-import", "fiona._shim",
] + COMMON_COLLECTS + COMMON_EXCLUDES

def build_helper(basename: str):
    pyfile = CODE_DIR / f"{basename}.py"
    if not pyfile.exists():
        log(f"[MERKNAD] Skipping {basename} (not available).")
        return
    args = COMMON_FLAGS + [
        "--name", basename,
        "--distpath", str(TOOLS_DIST),
        "--workpath", str(BUILD_FOLDER_ROOT / f"{basename}_build"),
        "--specpath", str(BUILD_FOLDER_ROOT / "helper_specs"),
        str(pyfile),
    ]
    run_pyinstaller(args)

def build_main():
    args = COMMON_FLAGS + [
        "--name", "mesa",
        "--distpath", str(FINAL_DIST),
        "--workpath", str(BUILD_FOLDER_ROOT / "mesa_build"),
        "--specpath", str(BUILD_FOLDER_ROOT),
        str(CODE_DIR / "mesa.py"),
    ]
    run_pyinstaller(args)

# --- Copy resources ----------------------------------------------------------
def copy_resources():
    for folder in ["input", "output", "qgis", "system_resources"]:
        src = CODE_DIR / folder
        dst = FINAL_DIST / folder
        if src.exists():
            log(f"Kopierer {folder}/ ...")
            shutil.copytree(src, dst, dirs_exist_ok=True)
    # config.ini (hvis tilstede)
    cfg = CODE_DIR / "config.ini"
    if cfg.exists():
        shutil.copy2(cfg, FINAL_DIST / "config.ini")

def main():
    log("\n==============================\n  Compiling Python scripts...\n==============================\n")
    log(f"CODE_DIR         = {CODE_DIR}")
    log(f"PROJECT_ROOT     = {PROJECT_ROOT}")
    log(f"PARENT_DIR       = {PARENT_DIR}")
    log(f"BUILD_FOLDER     = {BUILD_FOLDER_ROOT}")
    log(f"DIST_FOLDER      = {DIST_FOLDER_ROOT}")
    log(f"FINAL_DIST       = {FINAL_DIST}")
    log(f"TOOLS_DIST       = {TOOLS_DIST}\n")

    clean_and_prepare()
    ensure_pyinstaller()

    log("Bygger hjelpeverktøy...")
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
        "parametres_setup",   # NB: matcher faktisk filnavn hos deg
        "xx_stats_use",
    ]
    for h in helpers:
        build_helper(h)
    log("Hjelpere ferdig.\n")

    log("Building main program (mesa.exe)...")
    build_main()
    log("mesa.exe completed.\n")

    log("Compiling resources...")
    copy_resources()
    log("\n=============================\n  KOMPILERING FULLFØRT\n  Distribusjon: {}\n=============================\n".format(FINAL_DIST))
    sys.exit(0)

if __name__ == "__main__":
    main()
