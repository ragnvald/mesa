#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import shutil
import time
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

# By default, keep the PyInstaller work folders to enable incremental builds.
# Set MESA_BUILD_CLEAN=1 to force a full rebuild (slower, but guarantees no stale artifacts).
CLEAN_BUILD = os.environ.get("MESA_BUILD_CLEAN", "0").strip().lower() in {"1", "true", "yes"}

# Build toggles (defaults keep current behavior)
BUILD_HELPERS = os.environ.get("MESA_BUILD_HELPERS", "1").strip().lower() not in {"0", "false", "no"}
BUILD_MAIN = os.environ.get("MESA_BUILD_MAIN", "1").strip().lower() not in {"0", "false", "no"}

# Helper dependency strategy:
# - Default: per-helper detection to keep bundles smaller.
# - Set MESA_HELPERS_FULL_DEPS=1 to bundle full dependency stacks for all helpers.
HELPERS_FULL_DEPS = os.environ.get("MESA_HELPERS_FULL_DEPS", "0").strip().lower() in {"1", "true", "yes"}

# Compression toggle for onefile helpers:
# - Compressed (default): smaller .exe, slower startup (decompression overhead)
# - Uncompressed: larger .exe, faster startup (no decompression, less AV scanning)
# Set MESA_NO_COMPRESS=1 to build uncompressed helpers for faster startup.
NO_COMPRESS = os.environ.get("MESA_NO_COMPRESS", "0").strip().lower() in {"1", "true", "yes"}

def resolve_main_script() -> Path:
    """
    Locate the mesa entry-point. Prefer code/mesa.py, but fall back to the
    repo root where the real one currently lives.
    """
    candidates = [
        CODE_DIR / f"{APP_NAME}.py",
        PROJECT_ROOT / f"{APP_NAME}.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n  - ".join(str(c) for c in candidates)
    fail(f"Could not find '{APP_NAME}.py'. Searched:\n  - {searched}")

# ---------------------------------------------------------------------------
# Setup / Cleanup
# ---------------------------------------------------------------------------
def clean_and_prepare() -> None:
    log("Preparing build folders...")

    # IMPORTANT:
    # - When building the main app (mesa.exe), we want a clean dist to avoid stale DLLs/files.
    # - When building helpers only (MESA_BUILD_MAIN=0), do NOT wipe FINAL_DIST, otherwise we
    #   unintentionally delete an existing mesa.exe distribution.
    if BUILD_MAIN:
        log(f"Cleaning previous dist for main build: {FINAL_DIST}")
        # Only remove our app’s dist folder (keep siblings under dist/)
        shutil.rmtree(FINAL_DIST, ignore_errors=True)
    else:
        log(f"[NOTE] BUILD_MAIN=0 -> preserving existing dist folder (if any): {FINAL_DIST}")

    # Optionally remove build cache to force a clean rebuild.
    if CLEAN_BUILD:
        shutil.rmtree(BUILD_FOLDER_ROOT, ignore_errors=True)

    BUILD_FOLDER_ROOT.mkdir(parents=True, exist_ok=True)
    FINAL_DIST.mkdir(parents=True, exist_ok=True)
    TOOLS_DIST.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# PyInstaller
# ---------------------------------------------------------------------------
def _pyinstaller_log_level() -> str:
    """Return a PyInstaller log level.

    Default is WARN to keep helper builds concise. Override via env var:
      MESA_PYINSTALLER_LOG_LEVEL=INFO
    """

    raw = os.environ.get("MESA_PYINSTALLER_LOG_LEVEL", "WARN").strip().upper()
    if raw == "WARNING":
        raw = "WARN"

    allowed = {"TRACE", "DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"}
    if raw not in allowed:
        log(f"[NOTE] Invalid MESA_PYINSTALLER_LOG_LEVEL='{raw}', using WARN")
        return "WARN"

    return raw

def ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
        from PyInstaller.__main__ import run as _run  # noqa: F401
    except Exception as e:
        fail(
            "PyInstaller is missing. Activate your venv and run:\n"
            "  pip install -U pyinstaller pyinstaller-hooks-contrib\n"
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

def tcltk_data_args() -> list[str]:
    """Return --add-data args to bundle Tcl/Tk runtime data.

    PyInstaller's tkinter runtime hook expects these folders to exist under
    sys._MEIPASS (onedir: typically <dist>/_internal/):
      - _tcl_data
      - _tk_data

    On some setups, the default hook detection can fail, producing:
      Failed to execute script 'pyi_rth__tkinter' ... Tk data directory ... not found
    """

    if os.environ.get("MESA_SKIP_TCLTK", "0").strip().lower() in {"1", "true", "yes"}:
        log("[NOTE] MESA_SKIP_TCLTK=1 -> skipping Tcl/Tk data collection")
        return []

    try:
        from tkinter import Tcl  # type: ignore
    except Exception as e:
        log(f"[NOTE] tkinter not available in build environment; skipping Tcl/Tk data collection. Details: {e}")
        return []

    tcl = Tcl()

    tcl_dir: Path | None = None
    tk_dir: Path | None = None

    try:
        # Typically returns .../tcl/tcl8.6
        tcl_dir = Path(tcl.eval("info library"))
    except Exception as e:
        log(f"[NOTE] Could not resolve Tcl 'info library'; skipping Tcl/Tk data collection. Details: {e}")
        return []

    try:
        # Load Tk into the interpreter (does not need to open a window).
        tcl.eval("package require Tk")
        tk_dir = Path(tcl.eval("set tk_library"))
    except Exception:
        # Fallback: infer from Tcl layout
        candidate = (tcl_dir.parent / "tk8.6")
        if candidate.exists():
            tk_dir = candidate

    args: list[str] = []

    if tcl_dir and tcl_dir.exists():
        log(f"Including Tcl data: {tcl_dir} -> _tcl_data")
        args += add_data_arg(tcl_dir, "_tcl_data")
    else:
        log(f"[NOTE] Tcl data dir not found: {tcl_dir}")

    if tk_dir and tk_dir.exists():
        log(f"Including Tk data: {tk_dir} -> _tk_data")
        args += add_data_arg(tk_dir, "_tk_data")
    else:
        log(f"[NOTE] Tk data dir not found: {tk_dir}")

    return args

# ---------------------------------------------------------------------------
# Build profiles
#   Helpers: full GIS stack, onefile
#   Main: lean (no GIS), onedir then flattened to FINAL_DIST
# ---------------------------------------------------------------------------
COLLECT_TTKBOOTSTRAP = [
    # ttkbootstrap ships theme assets (tcl/images). Use collect-all to reliably
    # bundle both code + assets into frozen builds.
    "--collect-all", "ttkbootstrap",
]

COLLECT_GIS_STACK = [
    "--collect-all", "shapely",
    "--collect-all", "pyproj",
    "--collect-all", "fiona",
    "--collect-submodules", "geopandas",
]

COLLECT_PANDAS = [
    "--collect-data", "pandas",      # data only, avoid tests
]

COLLECT_PYARROW = [
    "--collect-data", "pyarrow",     # data only, avoid tests
]

COLLECT_H3 = [
    "--collect-all", "h3",
]

COLLECT_WEBVIEW = [
    "--collect-all", "webview",
]

COLLECT_DOCX = [
    # python-docx uses package data (templates). Collect-all ensures those
    # resources are bundled into frozen helper executables.
    "--collect-all", "docx",
]

# setuptools/pkg_resources runtime deps
#
# PyInstaller may bundle pkg_resources (via transitive deps), which in newer
# setuptools versions imports jaraco.* modules at runtime. If these aren't
# bundled, frozen helpers can fail immediately with:
#   ModuleNotFoundError: No module named 'jaraco'
#
# NOTE: 'jaraco' is a namespace package; bundling the concrete subpackages is
# more reliable than trying to collect the namespace root.
PKG_RESOURCES_HIDDEN_IMPORTS: list[str] = [
    "--hidden-import", "jaraco.text",
    "--hidden-import", "jaraco.functools",
    "--hidden-import", "jaraco.context",
    "--hidden-import", "more_itertools",
    "--hidden-import", "platformdirs",
    "--hidden-import", "autocommand",
    "--hidden-import", "backports.tarfile",
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
    # Exclude Android platform to avoid jnius warnings (Windows-only app)
    "--exclude-module", "jnius",
    "--exclude-module", "webview.platforms.android",
]

MAIN_COLLECTS = [
    # Mesa.py is a UI launcher that reads parquet files and spawns helper processes.
    # GIS stack (geopandas/shapely/pyproj/fiona) is lazy-imported only when Status
    # metrics need geometry calculations, so we exclude it from the main bundle for
    # faster startup. Helpers that need GIS will bundle it themselves (onefile).
    "--collect-data", "pandas",
    "--collect-data", "pyarrow",
    # ttkbootstrap ships theme assets (tcl/images). Use collect-all to reliably
    # bundle both code + assets into frozen builds.
    "--collect-all", "ttkbootstrap",
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
    "--exclude-module", "pysqlite2",
    "--exclude-module", "MySQLdb",
    "--exclude-module", "psycopg2",
    # GIS stack excluded from main (helpers bundle it when needed)
    "--exclude-module", "geopandas",
    "--exclude-module", "shapely",
    "--exclude-module", "pyproj",
    "--exclude-module", "fiona",
    "--exclude-module", "pyogrio",
    "--exclude-module", "tkinterweb",
    "--exclude-module", "influxdb_client",
]

FLAGS_HELPER = [
    "--windowed",
    "--noconfirm",
    "--onefile",
    f"--log-level={_pyinstaller_log_level()}",
] + HELPER_EXCLUDES

# Add compression flags based on NO_COMPRESS setting
if NO_COMPRESS:
    FLAGS_HELPER.extend(["--noupx", "--no-compress"])

if CLEAN_BUILD:
    FLAGS_HELPER.insert(4, "--clean")

_HELPER_SOURCE_CACHE: dict[str, str] = {}

def _read_helper_source(basename: str) -> str:
    cached = _HELPER_SOURCE_CACHE.get(basename)
    if cached is not None:
        return cached
    pyfile = CODE_DIR / f"{basename}.py"
    try:
        text = pyfile.read_text(encoding="utf-8")
    except Exception:
        try:
            text = pyfile.read_text(encoding="latin-1")
        except Exception:
            text = ""
    _HELPER_SOURCE_CACHE[basename] = text
    return text

def _imports_any_module(source_text: str, module_names: set[str]) -> bool:
    """Best-effort detection of 'import X' / 'from X import ...' patterns."""
    if not source_text:
        return False
    # Keep it simple + fast: regex on lines. This intentionally ignores dynamic imports.
    import re

    # Precompile a single regex that matches any of the module names.
    # Examples matched:
    #   import geopandas as gpd
    #   from shapely.geometry import box
    mods = "|".join(re.escape(m) for m in sorted(module_names, key=len, reverse=True))
    pattern = re.compile(rf"^\s*(?:from|import)\s+(?:{mods})(?:\s|\.|$)", re.MULTILINE)
    return pattern.search(source_text) is not None

def _mentions_any(source_text: str, needles: set[str]) -> bool:
    if not source_text:
        return False
    lower = source_text.lower()
    return any(n.lower() in lower for n in needles)

def helper_collects_for(basename: str) -> list[str]:
    """Return collect flags for a helper tool.

    Goal: keep most helper EXEs smaller/faster to build by only bundling large
    dependency stacks when the script actually needs them.

    Some helpers lazy-import GIS (geopandas, shapely, etc.) to speed up UI init.
    We still bundle GIS if we detect top-level imports OR if the helper's purpose
    clearly requires spatial processing.
    """

    if HELPERS_FULL_DEPS:
        collects: list[str] = []
        collects += COLLECT_TTKBOOTSTRAP
        collects += PKG_RESOURCES_HIDDEN_IMPORTS
        collects += COLLECT_GIS_STACK
        collects += COLLECT_PANDAS
        collects += COLLECT_PYARROW
        collects += COLLECT_WEBVIEW
        collects += COLLECT_H3
        collects += COLLECT_DOCX
        return collects

    src = _read_helper_source(basename)

    # Helpers that don't bundle GIS (either they don't use it, or they lazy-import it)
    # - geocode_group_edit: lazy-imports geopandas only when loading/saving data
    # - config_edit: config.ini editor only
    # - backup_restore: ZIP backup/restore, no data processing
    #
    # Note: PyInstaller may still show warnings like "Datas for pyproj not found" for
    # lazy-import helpers because it scans source code, but no GIS code is actually bundled.
    never_gis = {
        "geocode_group_edit",
        "config_edit",
        "backup_restore",
    }
    uses_gis = (
        basename not in never_gis
        and _imports_any_module(src, {"geopandas", "shapely", "fiona", "pyproj"})
    )

    uses_webview = _imports_any_module(src, {"webview"})
    uses_h3 = _imports_any_module(src, {"h3"})
    uses_docx = _imports_any_module(src, {"docx"})

    uses_pandas = uses_gis or _imports_any_module(src, {"pandas"})

    # pyarrow is large; include it when the helper clearly uses parquet/arrow.
    uses_pyarrow = (
        _imports_any_module(src, {"pyarrow"})
        or _mentions_any(src, {"read_parquet", "to_parquet", ".parquet"})
    )

    collects: list[str] = []
    collects += COLLECT_TTKBOOTSTRAP
    # Always include these small runtime deps to prevent frozen-startup failures
    # caused by transitive pkg_resources imports.
    collects += PKG_RESOURCES_HIDDEN_IMPORTS

    if uses_gis:
        collects += COLLECT_GIS_STACK

    if uses_pandas:
        collects += COLLECT_PANDAS

    if uses_pyarrow:
        collects += COLLECT_PYARROW

    if uses_webview:
        collects += COLLECT_WEBVIEW

    if uses_h3:
        collects += COLLECT_H3

    if uses_docx:
        collects += COLLECT_DOCX

    return collects

FLAGS_MAIN = [
    "--windowed",
    "--noconfirm",
    f"--log-level={_pyinstaller_log_level()}",
] + MAIN_COLLECTS + MAIN_EXCLUDES

if CLEAN_BUILD:
    FLAGS_MAIN.insert(3, "--clean")

# ---------------------------------------------------------------------------
# Build steps
# ---------------------------------------------------------------------------
def build_helper(basename: str) -> None:
    pyfile = CODE_DIR / f"{basename}.py"
    if not pyfile.exists():
        log(f"[NOTE] Skipping helper '{basename}' (not found).")
        return

    start = time.perf_counter()
    log(f"[HELPER] Building '{basename}'...")

    hidden_imports: list[str] = []
    extra_collects: list[str] = []
    # processing_pipeline_run runs the area pipeline by importing the internal module.
    # Add an explicit hidden import so PyInstaller always bundles it.
    if basename == "processing_pipeline_run":
        hidden_imports += ["--hidden-import", "processing_internal"]
        # Minimap uses pywebview from the internal module.
        extra_collects += COLLECT_WEBVIEW

    # Ensure Tcl/Tk data is bundled for helpers too (many use ttkbootstrap/tkinter).
    args = FLAGS_HELPER + tcltk_data_args() + hidden_imports + helper_collects_for(basename) + extra_collects + [
        "--name", basename,
        "--distpath", str(TOOLS_DIST),
        "--workpath", str(BUILD_FOLDER_ROOT / f"{basename}_build"),
        "--specpath", str(BUILD_FOLDER_ROOT / "helper_specs"),
        str(pyfile),
    ]
    run_pyinstaller(args)
    elapsed = time.perf_counter() - start
    log(f"[HELPER] Finished '{basename}' in {elapsed:.1f}s")

def build_main() -> None:
    start = time.perf_counter()
    log("[MAIN] Building 'mesa'...")
    # Keep the main app lean; UI launcher only needs ttkbootstrap, pandas, and pyarrow.
    # GIS stack (geopandas/shapely/pyproj/fiona) is lazy-imported in mesa.py only when
    # Status metrics need geometry calculations, and will be loaded from system Python.
    data_args: list[str] = []

    # Ensure Tcl/Tk data is bundled for tkinter/ttkbootstrap.
    # (Fixes runtime error: "Tk data directory ... not found" in compiled builds.)
    data_args += tcltk_data_args()

    # Optional: embed system_resources into onedir (redundant but harmless)
    sysres = CODE_DIR / "system_resources"
    if sysres.exists():
        data_args += add_data_arg(sysres, "system_resources")

    entry_point = resolve_main_script()

    # Embedded helper modules launched in-process from mesa when frozen.
    # Bundle them into the main app so separate helper exes are not required.
    hidden_imports = [
        "--hidden-import", "backup_restore",
        "--hidden-import", "config_edit",
        "--hidden-import", "processing_setup",
    ]

    args = FLAGS_MAIN + hidden_imports + [
        "--name", APP_NAME,
        "--distpath", str(FINAL_DIST),
        "--workpath", str(BUILD_FOLDER_ROOT / f"{APP_NAME}_build"),
        "--specpath", str(BUILD_FOLDER_ROOT),
    ] + data_args + [str(entry_point)]

    run_pyinstaller(args)
    elapsed = time.perf_counter() - start
    log(f"[MAIN] Finished 'mesa' in {elapsed:.1f}s")

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
        # IMPORTANT: qgis/ must be taken from repo root (same level as mesa.py).
        # Avoid accidentally copying a stale/partial code/qgis folder.
        if folder == "qgis":
            candidates = [PROJECT_ROOT / folder]
        else:
            # Prefer repo root; fall back to code/ for legacy layouts.
            candidates = [PROJECT_ROOT / folder, CODE_DIR / folder]

        src = next((c for c in candidates if c.exists()), None)
        if not src:
            continue
        dst = FINAL_DIST / folder
        log(f"Copying '{folder}/' from {src} ...")
        shutil.copytree(src, dst, dirs_exist_ok=True)

    # Also include legacy QGIS packages if they live outside the repo (e.g. "qgis older")
    qgis_older_names = ["qgis older", "qgis_older"]
    qgis_older_roots = [CODE_DIR, PROJECT_ROOT, PARENT_DIR]
    copied_qgis_older = False
    for root in qgis_older_roots:
        for name in qgis_older_names:
            candidate = root / name
            if candidate.exists():
                dst = FINAL_DIST / name
                log(f"Copying '{name}/' from {candidate} ...")
                shutil.copytree(candidate, dst, dirs_exist_ok=True)
                copied_qgis_older = True
                break
        if copied_qgis_older:
            break
    if not copied_qgis_older:
        log("[NOTE] 'qgis older' folder not found in code/, repo root, or parent; skipping.")

    # Copy config.ini next to mesa.exe
    for cfg in [CODE_DIR / "config.ini", PROJECT_ROOT / "config.ini"]:
        if cfg.exists():
            shutil.copy2(cfg, FINAL_DIST / "config.ini")
            break

    # NOTE: Intentionally do NOT copy secrets/ into distributions.

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log("\n==============================\n  Building MESA\n==============================\n")
    log(f"CODE_DIR      = {CODE_DIR}")
    log(f"PROJECT_ROOT  = {PROJECT_ROOT}")
    log(f"PARENT_DIR    = {PARENT_DIR}")
    log(f"BUILD_FOLDER  = {BUILD_FOLDER_ROOT}")
    log(f"DIST_FOLDER   = {DIST_FOLDER_ROOT}")
    log(f"FINAL_DIST    = {FINAL_DIST}")
    log(f"TOOLS_DIST    = {TOOLS_DIST}\n")
    log(f"CLEAN_BUILD   = {CLEAN_BUILD}")
    log(f"BUILD_HELPERS = {BUILD_HELPERS}")
    log(f"BUILD_MAIN    = {BUILD_MAIN}\n")
    log(f"NO_COMPRESS   = {NO_COMPRESS} (faster startup, larger .exe)\n")

    clean_and_prepare()
    ensure_pyinstaller()

    if BUILD_HELPERS:
        log("Building helper tools (onefile, per-tool dependency profiles)...")
        helpers = [
            "asset_manage",
            "atlas_manage",
            "tiles_create_raster",
            "report_generate",
            "analysis_setup",
            "analysis_present",
            "geocode_group_edit",
            "geocode_create",
            "line_manage",
            "asset_map_view",
            "map_overview",
            "processing_pipeline_run",
        ]

        # Optional helper selection:
        # - MESA_HELPERS="a,b,c" builds only those helpers
        # - MESA_HELPERS_SKIP="a,b" skips those helpers
        only_raw = os.environ.get("MESA_HELPERS", "").strip()
        skip_raw = os.environ.get("MESA_HELPERS_SKIP", "").strip()

        if only_raw:
            only = {p.strip() for p in only_raw.split(",") if p.strip()}
            helpers = [h for h in helpers if h in only]

        if skip_raw:
            skip = {p.strip() for p in skip_raw.split(",") if p.strip()}
            helpers = [h for h in helpers if h not in skip]

        for h in helpers:
            build_helper(h)
        log("Helper tools built.\n")
    else:
        log("[NOTE] Skipping helper tools (MESA_BUILD_HELPERS=0).\n")

    if BUILD_MAIN:
        log("Building main app (ONEDIR, lean)...")
        build_main()
        log("Main app built.\n")
    else:
        log("[NOTE] Skipping main app build (MESA_BUILD_MAIN=0).\n")

    if BUILD_MAIN:
        log("Flattening main app into FINAL_DIST...")
        flatten_onedir_output()

    log("Copying runtime resources next to the app...")
    copy_resources()

    exe_path = FINAL_DIST / f"{APP_NAME}.exe"
    if BUILD_MAIN:
        if exe_path.exists():
            log(f"\nBuild complete. Launch here:\n  {exe_path}\n")
        else:
            log("\n[NOTE] mesa.exe not found where expected. Listing FINAL_DIST for troubleshooting:")
            for p in sorted(FINAL_DIST.rglob("*")):
                if p.is_file():
                    log(f"  - {p.relative_to(FINAL_DIST)}")
    else:
        # Helper-only builds are useful, but they do not produce mesa.exe.
        if exe_path.exists():
            log(f"[NOTE] BUILD_MAIN=0 -> existing mesa.exe is present and was preserved: {exe_path}")
        else:
            log(
                "[WARN] BUILD_MAIN=0 -> mesa.exe was NOT built and is not present in the dist folder. "
                "This is expected for helper-only builds; the dist will contain tools/resources only. "
                "To build mesa.exe, re-run with MESA_BUILD_MAIN=1."
            )

    log(f"\nDistribution ready at:\n  {FINAL_DIST}\n")
    sys.exit(0)

if __name__ == "__main__":
    main()
