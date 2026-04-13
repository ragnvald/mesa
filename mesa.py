import os
import locale
import sys
import warnings
import importlib
from pathlib import Path
from typing import TYPE_CHECKING

_MESA_SKIP_VENV_RELAUNCH = "MESA_SKIP_VENV_RELAUNCH"


def _preferred_repo_dev_python() -> Path | None:
    if os.name != "nt":
        return None
    try:
        scripts_dir = Path(__file__).resolve().parent / ".venv" / "Scripts"
    except Exception:
        return None
    current_name = Path(sys.executable or "").name.lower()
    preferred_names = ["pythonw.exe", "python.exe"] if current_name == "pythonw.exe" else ["python.exe", "pythonw.exe"]
    for exe_name in preferred_names:
        candidate = scripts_dir / exe_name
        if candidate.exists():
            return candidate
    return None


def _venv_env_overrides(python_exe: Path) -> dict[str, str]:
    env = os.environ.copy()
    scripts_dir = python_exe.parent
    venv_dir = scripts_dir.parent
    env["VIRTUAL_ENV"] = str(venv_dir)
    scripts_dir_str = str(scripts_dir)
    path_value = env.get("PATH", "")
    path_entries = [p for p in path_value.split(os.pathsep) if p]
    scripts_norm = os.path.normcase(os.path.abspath(scripts_dir_str))
    if not any(os.path.normcase(os.path.abspath(p)) == scripts_norm for p in path_entries):
        env["PATH"] = os.pathsep.join([scripts_dir_str, *path_entries]) if path_entries else scripts_dir_str
    return env


def _ensure_repo_dev_venv() -> None:
    if __name__ != "__main__":
        return
    if os.name != "nt" or getattr(sys, "frozen", False):
        return
    if os.environ.get(_MESA_SKIP_VENV_RELAUNCH) == "1":
        return
    preferred_python = _preferred_repo_dev_python()
    if preferred_python is None:
        return
    try:
        current_python = Path(sys.executable).resolve()
        preferred_python = preferred_python.resolve()
    except Exception:
        return
    if current_python == preferred_python:
        return

    env = _venv_env_overrides(preferred_python)
    env[_MESA_SKIP_VENV_RELAUNCH] = "1"
    argv = [str(preferred_python), str(Path(__file__).resolve()), *sys.argv[1:]]
    import subprocess

    completed = subprocess.run(
        argv,
        cwd=str(Path(__file__).resolve().parent),
        env=env,
    )
    raise SystemExit(int(completed.returncode or 0))


_ensure_repo_dev_venv()

if TYPE_CHECKING:
    import geopandas as gpd

# pyogrio (used by GeoPandas by default when installed) warns that measured (M)
# geometries are not supported and will be converted.
warnings.filterwarnings(
    "ignore",
    message=r"Measured \(M\) geometry types are not supported\..*",
    category=UserWarning,
    module=r"pyogrio\..*",
)

# =====================================================================
# Imports  (PySide6 replaces tkinter/ttkbootstrap)
# =====================================================================
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QPlainTextEdit, QTableWidget,
    QTableWidgetItem, QScrollArea, QFrame, QSizePolicy,
    QMessageBox, QFileDialog, QHeaderView, QSplitter,
)
from PySide6.QtGui import QPixmap, QIcon, QFont, QColor, QPalette, QDesktopServices
from PySide6.QtCore import Qt, QTimer, QUrl, QSize

import subprocess
import webbrowser
import pandas as pd
import configparser
import platform
import shutil
import json
import tempfile
from datetime import datetime
import threading
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import time
import struct
import ctypes
import re
import zipfile

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None

# ---------------------------------------------------------------------
# Project/base resolution (works in dev and when frozen)
# ---------------------------------------------------------------------
def _get_project_base() -> str:
    if getattr(sys, "frozen", False):
        env_dir = os.environ.get("MESA_BASE_DIR")
        if env_dir:
            return os.path.abspath(env_dir)
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))

PROJECT_BASE = _get_project_base()
RESOURCE_BASE = getattr(sys, "_MEIPASS", PROJECT_BASE)

# Qt application reference (assigned when UI initializes)
app = None
main_window = None
_launched_helper_windows = []

INPROCESS_HELPERS = {
    "geocode_manage",
    "asset_manage",
    "processing_setup",
    "processing_pipeline_run",
    "atlas_manage",
    "report_generate",
    "analysis_present",
}

# ---------------------------------------------------------------------
# Path resolver
# ---------------------------------------------------------------------
def resolve_path(rel_path: str) -> str:
    p_disk_1 = os.path.join(PROJECT_BASE, rel_path)
    if os.path.exists(p_disk_1):
        return p_disk_1
    p_disk_2 = os.path.join(PROJECT_BASE, "code", rel_path)
    if os.path.exists(p_disk_2):
        return p_disk_2
    p_res_1 = os.path.join(RESOURCE_BASE, rel_path)
    if os.path.exists(p_res_1):
        return p_res_1
    p_res_2 = os.path.join(RESOURCE_BASE, "code", rel_path)
    if os.path.exists(p_res_2):
        return p_res_2
    return p_disk_1

def _candidate_tool_dirs() -> list[str]:
    bases = [PROJECT_BASE]
    if PROJECT_BASE != RESOURCE_BASE and RESOURCE_BASE:
        bases.append(RESOURCE_BASE)
    tool_dirs = []
    for base in bases:
        tool_dirs.append(os.path.join(base, "tools"))
        tool_dirs.append(os.path.join(base, "code", "tools"))
    seen = set()
    dedup = []
    for path in tool_dirs:
        ap = os.path.abspath(path)
        if ap not in seen:
            seen.add(ap)
            dedup.append(ap)
    return dedup

def _resolve_from_tool_dirs(file_name: str) -> str | None:
    if not file_name:
        return None
    for folder in _candidate_tool_dirs():
        candidate = os.path.join(folder, file_name)
        if os.path.exists(candidate):
            return candidate
    return None

# ---------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------
def _ensure_default_header_present(path: str):
    try:
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        first = ""
        for line in txt.splitlines():
            s = line.strip()
            if not s or s.startswith(";") or s.startswith("#"):
                continue
            first = s
            break
        if first and not first.startswith("["):
            with open(path, "w", encoding="utf-8") as f:
                f.write("[DEFAULT]\n" + txt)
    except Exception:
        pass

def read_config(abs_or_rel_path: str):
    if os.path.isabs(abs_or_rel_path):
        cfg_path = abs_or_rel_path
    else:
        cfg_path = os.path.join(PROJECT_BASE, abs_or_rel_path)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Configuration not found: {cfg_path}")
    _ensure_default_header_present(cfg_path)
    cfg = configparser.ConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
    cfg.read(cfg_path, encoding="utf-8")
    if "DEFAULT" not in cfg:
        cfg["DEFAULT"] = {}
    return cfg

def check_and_create_folders():
    folders = [
        os.path.join(original_working_directory, "input", "geocode"),
        os.path.join(original_working_directory, "output"),
        os.path.join(original_working_directory, "qgis"),
        os.path.join(original_working_directory, "input", "lines"),
    ]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

def log_to_logfile(message):
    ts = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    try:
        with open(os.path.join(original_working_directory, "log.txt"), "a", encoding="utf-8") as f:
            f.write(f"{ts} - {message}\n")
    except Exception:
        pass

def _dedup_paths(paths):
    seen = set()
    unique = []
    for path in paths:
        ap = os.path.abspath(path)
        if ap not in seen:
            seen.add(ap)
            unique.append(ap)
    return unique

def _geoparquet_base_candidates():
    candidates = [
        os.path.join(original_working_directory, "output", "geoparquet"),
        os.path.join(PROJECT_BASE, "output", "geoparquet"),
        os.path.join(PROJECT_BASE, "code", "output", "geoparquet"),
    ]
    if RESOURCE_BASE and RESOURCE_BASE != PROJECT_BASE:
        candidates.extend([
            os.path.join(RESOURCE_BASE, "output", "geoparquet"),
            os.path.join(RESOURCE_BASE, "code", "output", "geoparquet"),
        ])
    return _dedup_paths(candidates)

def _detect_geoparquet_dir() -> str:
    unique_candidates = _geoparquet_base_candidates()
    sentinel_files = [
        "tbl_asset_group.parquet",
        "tbl_geocode_group.parquet",
        "tbl_lines_original.parquet",
        "tbl_flat.parquet",
        "tbl_segment_flat.parquet",
    ]

    def has_data(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        for sentinel in sentinel_files:
            if os.path.exists(os.path.join(path, sentinel)):
                return True
        try:
            for entry in os.listdir(path):
                if entry.lower().endswith(".parquet") and not entry.startswith("__"):
                    return True
        except OSError:
            pass
        return False

    def score(path: str) -> int:
        if not os.path.isdir(path):
            return -1
        hits = 0
        for sentinel in sentinel_files:
            fp = os.path.join(path, sentinel)
            if os.path.exists(fp):
                hits += 10
                continue
            alt = os.path.join(path, os.path.splitext(sentinel)[0])
            if os.path.isdir(alt):
                hits += 10
        try:
            extra = sum(
                1 for entry in os.listdir(path)
                if entry.lower().endswith(".parquet") and not entry.startswith("__")
            )
        except OSError:
            extra = 0
        return hits + extra

    for candidate in unique_candidates:
        if has_data(candidate):
            if unique_candidates and candidate != unique_candidates[0]:
                log_to_logfile(f"[status] Using geoparquet dir: {candidate}")
            return candidate

    best_path = None
    best_score = -1
    for candidate in unique_candidates:
        current = score(candidate)
        if current > best_score:
            best_score = current
            best_path = candidate

    if best_path and best_score >= 0:
        if unique_candidates and best_path != unique_candidates[0]:
            log_to_logfile(f"[status] Using geoparquet dir: {best_path}")
        return best_path

    for candidate in unique_candidates:
        if os.path.isdir(candidate):
            return candidate

    return unique_candidates[0] if unique_candidates else os.path.join(PROJECT_BASE, "output", "geoparquet")


def _geoparquet_search_paths() -> list[str]:
    detect_dir = _detect_geoparquet_dir()
    ordered = [detect_dir, *_geoparquet_base_candidates()]
    return _dedup_paths([path for path in ordered if path])


def _locate_geoparquet_file(layer_name: str) -> str | None:
    normalized = layer_name.strip()
    file_candidate = normalized if normalized.lower().endswith(".parquet") else f"{normalized}.parquet"
    dir_candidate = normalized[:-8] if normalized.lower().endswith(".parquet") else normalized
    search_names = [file_candidate]
    if dir_candidate:
        search_names.append(dir_candidate)
    for folder in _geoparquet_search_paths():
        for name in search_names:
            candidate = os.path.join(folder, name)
            if os.path.exists(candidate):
                return candidate
    return None


def _parquet_row_count(parquet_path: str | None) -> int | None:
    if not parquet_path:
        return None
    try:
        if os.path.isdir(parquet_path):
            try:
                dataset = ds.dataset(parquet_path, format="parquet")
                return dataset.count_rows()
            except Exception:
                return len(pd.read_parquet(parquet_path))
        metadata = pq.ParquetFile(parquet_path).metadata
        return metadata.num_rows if metadata else None
    except Exception:
        try:
            return len(pd.read_parquet(parquet_path))
        except Exception:
            return None

def _preferred_lines_base_dir() -> str:
    try:
        gpq_path = Path(_detect_geoparquet_dir()).resolve()
        base_candidate = gpq_path.parent.parent
        if (base_candidate / "config.ini").exists():
            return str(base_candidate)
    except Exception:
        pass
    return original_working_directory

# ---------------------------------------------------------------------
# Status helpers (reads GeoParquet)
# ---------------------------------------------------------------------
def get_status(geoparquet_dir):
    status_list = []
    gpq_dirs = _dedup_paths([geoparquet_dir, *_geoparquet_base_candidates()])

    def _table_path_candidates(layer_name: str):
        return [os.path.join(base, f"{layer_name}.parquet") for base in gpq_dirs]

    def _existing_table_path(layer_name: str):
        for fp in _table_path_candidates(layer_name):
            if os.path.exists(fp):
                return fp
        return None

    def ppath(layer_name: str) -> str:
        fp = _existing_table_path(layer_name)
        if fp:
            return fp
        candidates = _table_path_candidates(layer_name)
        return candidates[0] if candidates else os.path.join(geoparquet_dir, f"{layer_name}.parquet")

    def read_table_and_count(layer_name: str):
        fp = ppath(layer_name)
        if not os.path.exists(fp):
            log_to_logfile(f"Parquet table {layer_name} does not exist.")
            return None
        try:
            return pq.ParquetFile(fp).metadata.num_rows
        except Exception as e:
            log_to_logfile(f"Parquet metadata read failed for {layer_name}: {e}; falling back to full read.")
            try:
                return len(pd.read_parquet(fp))
            except Exception as e2:
                log_to_logfile(f"Error counting rows for {layer_name}: {e2}")
                return None

    def read_setup_status():
        fp = ppath('tbl_asset_group')
        assets_ok = False
        missing_cols_msg = ""
        try:
            if os.path.exists(fp):
                cols = ['importance', 'susceptibility', 'sensitivity']
                df = pd.read_parquet(fp, columns=cols)
                have_all = all(c in df.columns for c in cols)
                if not have_all:
                    missing = [c for c in cols if c not in df.columns]
                    missing_cols_msg = f" Missing columns in tbl_asset_group: {', '.join(missing)}."
                if have_all:
                    num = df.apply(pd.to_numeric, errors='coerce').fillna(0)
                    assets_ok = all((num[c] > 0).any() for c in cols)
        except Exception as e:
            log_to_logfile(f"Error evaluating setup status on tbl_asset_group: {e}")

        if assets_ok:
            return "+", "Set up ok. Feel free to adjust it. Remember to rerun after adjustments."

        parts = []
        if missing_cols_msg:
            parts.append(missing_cols_msg.strip())
        else:
            parts.append("importance/susceptibility/sensitivity not assigned (>0) in tbl_asset_group")
        detail = "; ".join([p for p in parts if p]) or "Incomplete setup."
        return "-", f"You need to set up the calculation. \nPress the 'Assets'-button to proceed. ({detail})"

    def append_status(symbol, message, link):
        status_list.append({'Status': symbol, 'Message': message, 'Link': link})

    try:
        asset_group_count = read_table_and_count('tbl_asset_group')
        has_asset_group_rows = asset_group_count is not None and asset_group_count > 0
        append_status("+" if has_asset_group_rows else "-",
                      f"Asset layers imported: {asset_group_count}" if has_asset_group_rows else
                      "Assets are missing.\nUse 'Assets' to import and register asset groups.",
                      "https://github.com/ragnvald/mesa/wiki/User-interface#prepare-data")

        geocode_group_count = read_table_and_count('tbl_geocode_group')
        append_status("+" if geocode_group_count is not None else "/",
                      f"Geocode layers: {geocode_group_count}" if geocode_group_count is not None else
                      "Geocodes are missing.\nImport assets by pressing the Assets button.",
                      "https://github.com/ragnvald/mesa/wiki/User-interface#prepare-data")

        lines_original_count = read_table_and_count('tbl_lines_original')
        lines_processed_count = read_table_and_count('tbl_lines')
        visible_lines_count = lines_original_count if lines_original_count is not None else lines_processed_count
        append_status("+" if visible_lines_count is not None else "/",
                  f"Lines: {visible_lines_count}" if visible_lines_count is not None else
                  "Lines are missing.\nImport or initiate lines if you want to use\nthe line feature.",
                  "https://github.com/ragnvald/mesa/wiki/User-interface#run-processing")

        symbol, message = read_setup_status()
        append_status(symbol, message, "https://github.com/ragnvald/mesa/wiki/User-interface#configure-analysis")

        flat_original_count = read_table_and_count('tbl_flat')
        append_status("+" if flat_original_count is not None else "-",
                      "Processing completed. You may choose to Show maps or open the QGIS-project file in the qgis-folder."
                      if flat_original_count is not None else
                      "Processing incomplete. Press the \nProcess area-button.",
                      "https://github.com/ragnvald/mesa/wiki/User-interface#run-processing")

        atlas_count = read_table_and_count('tbl_atlas')
        append_status("+" if atlas_count is not None else "/",
                      f"Atlas pages: {atlas_count}" if atlas_count is not None else
                      "Please create map tile.",
                      "https://github.com/ragnvald/mesa/wiki/Definitions#atlas")

        segments_flat_count = read_table_and_count('tbl_segment_flat')
        lines_count = lines_processed_count if lines_processed_count is not None else visible_lines_count
        lines_count_label = lines_count if lines_count is not None else "--"
        append_status("+" if segments_flat_count is not None else "/",
                  f"Segments are in place with {segments_flat_count} segments along {lines_count_label} lines."
                      if segments_flat_count is not None else
                      "Segments are missing.\nImport or initiate lines if you want to use\nthe line feature.",
                      "https://github.com/ragnvald/mesa/wiki/User-interface#run-processing")

        return pd.DataFrame(status_list)
    except Exception as e:
        return pd.DataFrame({'Status': ['Error'], 'Message': [f"Error accessing statistics: {e}"], 'Link': [""]})

# ---------------------------------------------------------------------
# Subprocess runner (+ unified env/cwd)
# ---------------------------------------------------------------------
def _infer_base_dir_from_cmd(cmd) -> str | None:
    try:
        if not cmd:
            return None
        if "--original_working_directory" in cmd:
            i = cmd.index("--original_working_directory")
            if i + 1 < len(cmd):
                base = str(cmd[i + 1]).strip()
                if base and os.path.isdir(base):
                    return os.path.abspath(base)
    except Exception:
        pass
    return None

def _sub_env(base_dir: str | None = None):
    env = os.environ.copy()
    env["MESA_BASE_DIR"] = os.path.abspath(base_dir) if base_dir else PROJECT_BASE
    return env

def _schedule_stats_refresh(gpkg_file):
    if not gpkg_file:
        return
    if main_window is None:
        log_to_logfile("UI not initialized; skipping stats refresh")
        return
    QTimer.singleShot(0, lambda: _do_stats_refresh(gpkg_file))

def _do_stats_refresh(gpkg_file):
    try:
        if main_window is not None:
            main_window.update_stats(gpkg_file)
    except Exception as exc:
        log_to_logfile(f"Failed to refresh stats: {exc}")

def run_subprocess(command, fallback_command, gpkg_file):
    try:
        base_dir = _infer_base_dir_from_cmd(command) or PROJECT_BASE
        log_to_logfile(f"Attempting to run command: {command}")
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=base_dir,
            env=_sub_env(base_dir)
        )
        log_to_logfile("Primary command executed successfully")
        log_to_logfile(f"stdout: {result.stdout}")
        _schedule_stats_refresh(gpkg_file)
    except subprocess.CalledProcessError as e:
        log_to_logfile(f"Primary command failed with error: {e}")
        log_to_logfile(f"Failed to execute command: {command}, error: {e.stderr}")
        try:
            if fallback_command:
                fallback_base = _infer_base_dir_from_cmd(fallback_command) or base_dir
                log_to_logfile(f"Attempting to run fallback command: {fallback_command}")
                result = subprocess.run(
                    fallback_command,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=fallback_base,
                    env=_sub_env(fallback_base)
                )
                log_to_logfile("Fallback command executed successfully")
                log_to_logfile(f"stdout: {result.stdout}")
                _schedule_stats_refresh(gpkg_file)
        except subprocess.CalledProcessError as e2:
            log_to_logfile(f"Failed to execute fallback command: {fallback_command}, error: {e2.stderr}")
        except FileNotFoundError as e2:
            log_to_logfile(f"File not found for fallback command: {fallback_command}, error: {e2}")
    except FileNotFoundError as e:
        log_to_logfile(f"File not found for command: {command}, error: {e}")

def run_subprocess_async(command, fallback_command, gpkg_file):
    primary = command[:] if command else None
    fallback = fallback_command[:] if fallback_command else None
    def _runner():
        run_subprocess(primary, fallback, gpkg_file)
    threading.Thread(target=_runner, daemon=True).start()


def _launch_gui_process(cmd: list[str], label: str):
    if not cmd:
        log_to_logfile(f"No command provided for {label}; skipping launch")
        return
    log_to_logfile(f"Launching {label}: {cmd}")

    try:
        if os.name == "nt" and cmd and isinstance(cmd[0], str):
            exe0 = os.path.abspath(cmd[0])
            if exe0.lower().endswith("python.exe"):
                candidate = exe0[:-len("python.exe")] + "pythonw.exe"
                if os.path.exists(candidate):
                    cmd = [candidate, *cmd[1:]]
    except Exception:
        pass

    base_dir = _infer_base_dir_from_cmd(cmd) or PROJECT_BASE

    stdout_target = subprocess.DEVNULL
    stderr_target = subprocess.DEVNULL
    log_handle = None
    try:
        log_path = os.path.join(base_dir, "log.txt")
        log_handle = open(log_path, "a", encoding="utf-8")
        log_handle.write(f"\n--- launch {label} {datetime.now().strftime('%Y.%m.%d %H:%M:%S')} ---\n")
        log_handle.flush()
        stdout_target = log_handle
        stderr_target = log_handle
    except Exception as exc:
        log_to_logfile(f"Unable to open child log file for {label}: {exc}")

    popen_kwargs = {
        "cwd": base_dir,
        "env": _sub_env(base_dir),
        "stdout": stdout_target,
        "stderr": stderr_target,
    }
    if os.name == "nt":
        creationflags = 0
        creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        creationflags |= getattr(subprocess, "DETACHED_PROCESS", 0)
        creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        popen_kwargs["creationflags"] = creationflags
    else:
        popen_kwargs["start_new_session"] = True
    try:
        subprocess.Popen(cmd, **popen_kwargs)
    except Exception as exc:
        log_to_logfile(f"Failed to launch {label}: {exc}")
    finally:
        try:
            if log_handle is not None:
                log_handle.close()
        except Exception:
            pass

def _resolve_tool_path(*rel_candidates: str) -> str:
    for rel in rel_candidates:
        candidate = resolve_path(rel)
        if os.path.exists(candidate):
            return candidate
        tool_candidate = _resolve_from_tool_dirs(os.path.basename(rel))
        if tool_candidate:
            return tool_candidate
    tool_fallback = _resolve_from_tool_dirs(os.path.basename(rel_candidates[0]))
    if tool_fallback:
        return tool_fallback
    return resolve_path(rel_candidates[0])

def get_script_paths(file_name: str):
    python_script = _resolve_tool_path(
        os.path.join("system", f"{file_name}.py"),
        f"{file_name}.py",
    )
    exe_file = _resolve_tool_path(
        os.path.join("system", f"{file_name}.exe"),
        f"{file_name}.exe",
    )
    log_to_logfile(f"Python script path: {python_script}")
    log_to_logfile(f"Executable file path: {exe_file}")
    return python_script, exe_file

# ---------------------------------------------------------------------
# Helper launchers
# ---------------------------------------------------------------------
def _ensure_code_dir_on_syspath() -> None:
    code_dir = os.path.join(PROJECT_BASE, "code")
    if not os.path.isdir(code_dir):
        return

    norm_target = os.path.normcase(os.path.abspath(code_dir))
    for entry in sys.path:
        try:
            if os.path.normcase(os.path.abspath(entry)) == norm_target:
                return
        except Exception:
            continue
    sys.path.insert(0, code_dir)


def _track_helper_window(window):
    if window is None:
        return None
    _launched_helper_windows.append(window)
    return window


def _launch_helper_inprocess(file_name: str, base_dir: str | None = None):
    _ensure_code_dir_on_syspath()
    launch_base = os.path.abspath(base_dir or original_working_directory or PROJECT_BASE)
    log_to_logfile(f"Launching {file_name} in-process with base_dir={launch_base}")

    module = importlib.import_module(file_name)
    run_fn = getattr(module, "run", None)
    if not callable(run_fn):
        raise AttributeError(f"Module '{file_name}' does not expose a callable run() entry point")

    # These helpers are standalone Qt windows. Do not parent them to the
    # launcher main window, or Windows can treat them as owned child windows
    # with inconsistent title-bar/taskbar icon behaviour.
    return _track_helper_window(run_fn(launch_base, master=None))


def _launch_helper_subprocess(file_name: str, extra_args: list[str] | None = None):
    """Launch a helper tool as a subprocess. Works for both source and frozen."""
    python_script, exe_file = get_script_paths(file_name)
    args = list(extra_args or [])
    if getattr(sys, "frozen", False):
        _launch_gui_process([exe_file, *args], f"{file_name} exe")
    else:
        python_exe = sys.executable or "python"
        _launch_gui_process([python_exe, python_script, *args], f"{file_name} script")


def _launch_helper(file_name: str, extra_args: list[str] | None = None, base_dir: str | None = None):
    if file_name in INPROCESS_HELPERS:
        try:
            return _launch_helper_inprocess(file_name, base_dir=base_dir)
        except Exception as exc:
            log_to_logfile(f"Failed to launch {file_name} in-process: {exc}")
            python_script, exe_file = get_script_paths(file_name)
            if os.path.exists(python_script) or os.path.exists(exe_file):
                log_to_logfile(f"Falling back to subprocess launch for {file_name}")
                return _launch_helper_subprocess(file_name, extra_args)

            if main_window is not None:
                QMessageBox.critical(
                    main_window,
                    "Launch failed",
                    f"Could not open '{file_name}'.\n\n{exc}",
                )
            return None

    return _launch_helper_subprocess(file_name, extra_args)


# ---------------------------------------------------------------------
# Button handlers
# ---------------------------------------------------------------------
def geocodes_grids():
    log_to_logfile("Launching geocode_manage")
    _launch_helper("geocode_manage", ["--original_working_directory", original_working_directory], base_dir=original_working_directory)

def open_assets():
    log_to_logfile("Launching asset_manage")
    _launch_helper("asset_manage", ["--original_working_directory", original_working_directory], base_dir=original_working_directory)

def edit_processing_setup():
    log_to_logfile("Launching processing_setup")
    _launch_helper("processing_setup", ["--original_working_directory", original_working_directory], base_dir=original_working_directory)

def open_process_all():
    log_to_logfile("[Process] STARTED")
    log_to_logfile("Launching processing_pipeline_run")
    _launch_helper("processing_pipeline_run", ["--original_working_directory", original_working_directory], base_dir=original_working_directory)

def make_atlas():
    log_to_logfile("Launching atlas_manage")
    _launch_helper("atlas_manage", ["--original_working_directory", original_working_directory], base_dir=original_working_directory)

def open_maps_overview():
    log_to_logfile("Launching map_overview subprocess")
    _launch_helper_subprocess("map_overview")

def open_asset_layers_viewer():
    log_to_logfile("Launching asset_map_view subprocess")
    _launch_helper_subprocess("asset_map_view")

def open_present_files():
    log_to_logfile("Launching report_generate")
    _launch_helper("report_generate", ["--original_working_directory", original_working_directory], base_dir=original_working_directory)

def open_data_analysis_setup():
    log_to_logfile("Launching analysis_setup subprocess")
    _launch_helper_subprocess("analysis_setup", ["--original_working_directory", original_working_directory])

def open_data_analysis_presentation():
    log_to_logfile("Launching analysis_present")
    _launch_helper("analysis_present", ["--original_working_directory", original_working_directory], base_dir=original_working_directory)

def edit_lines():
    chosen_base = _preferred_lines_base_dir()
    log_to_logfile(f"Launching line_manage with base_dir={chosen_base}")
    _launch_helper_subprocess("line_manage", ["--original_working_directory", chosen_base])


# ---------------------------------------------------------------------
# Backup / restore / clear (unchanged logic, Qt dialogs)
# ---------------------------------------------------------------------
def _iter_backup_files(root_path: Path):
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            yield Path(dirpath) / fn

def _safe_zip_member_names(names: list[str]) -> list[str]:
    safe: list[str] = []
    for member in names:
        name = str(member or "").replace("\\", "/")
        if not name or name.startswith("/"):
            continue
        parts = [p for p in name.split("/") if p]
        if any(p == ".." for p in parts):
            continue
        safe.append("/".join(parts))
    return safe

def create_backup_archive(base_dir: str, destination_folder: str) -> str:
    base = Path(base_dir)
    dest = Path(destination_folder)
    dest.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    zip_path = dest / f"mesa_backup_{ts}.zip"
    config_path = base / "config.ini"
    input_dir = base / "input"
    output_dir = base / "output"
    files_to_add: list[tuple[Path, str]] = []
    if config_path.is_file():
        files_to_add.append((config_path, "config.ini"))
    for folder in (input_dir, output_dir):
        if folder.is_dir():
            for file_path in _iter_backup_files(folder):
                if file_path.is_file():
                    arc = file_path.relative_to(base).as_posix()
                    files_to_add.append((file_path, arc))
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file_path, arcname in files_to_add:
            zf.write(file_path, arcname=arcname)
    log_to_logfile(f"Created backup archive: {zip_path}")
    return str(zip_path)

def restore_backup_archive(base_dir: str, zip_path: str) -> None:
    base = Path(base_dir)
    zip_file = Path(zip_path)
    if not zip_file.is_file():
        raise FileNotFoundError(zip_path)
    with zipfile.ZipFile(zip_file, mode="r") as zf:
        safe_members = _safe_zip_member_names(zf.namelist())
        to_extract = [
            m for m in safe_members
            if m == "config.ini" or m.startswith("input/") or m.startswith("output/")
        ]
        for folder_name in ("input", "output"):
            folder = base / folder_name
            if folder.exists() and folder.is_dir():
                shutil.rmtree(folder, ignore_errors=True)
        cfg = base / "config.ini"
        if cfg.exists() and cfg.is_file():
            try:
                cfg.unlink()
            except Exception:
                pass
        for member in to_extract:
            target = (base / Path(member)).resolve()
            base_resolved = base.resolve()
            if target != base_resolved and base_resolved not in target.parents:
                raise ValueError(f"Unsafe ZIP member: {member}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    check_and_create_folders()
    log_to_logfile(f"Restored backup archive: {zip_file}")


# ---------------------------------------------------------------------
# Host capability snapshot
# ---------------------------------------------------------------------
def _powershell_json(command: str) -> object | None:
    if os.name != "nt":
        return None
    try:
        if str(os.environ.get("MESA_NO_POWERSHELL", "")).strip().lower() in ("1", "true", "yes", "on"):
            return None
        flags = 0
        try:
            flags = subprocess.CREATE_NO_WINDOW
        except Exception:
            flags = 0
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
            stderr=subprocess.DEVNULL,
            text=True,
            creationflags=flags,
        )
        out = (out or "").strip()
        if not out:
            return None
        return json.loads(out)
    except Exception:
        return None

def _windows_global_memory_status() -> dict | None:
    if os.name != "nt":
        return None
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return None
        return {
            "ram_total_gb": round(float(stat.ullTotalPhys) / (1024 ** 3), 2),
            "ram_available_gb": round(float(stat.ullAvailPhys) / (1024 ** 3), 2),
        }
    except Exception:
        return None

def _wmic_list(command_args: list[str]) -> list[dict] | None:
    if os.name != "nt":
        return None
    try:
        if str(os.environ.get("MESA_NO_WMIC", "")).strip().lower() in ("1", "true", "yes", "on"):
            return None
        flags = 0
        try:
            flags = subprocess.CREATE_NO_WINDOW
        except Exception:
            flags = 0
        out = subprocess.check_output(
            command_args, stderr=subprocess.DEVNULL, text=True, creationflags=flags,
        )
        txt = (out or "").strip()
        if not txt:
            return None
        blocks: list[dict] = []
        cur: dict = {}
        for raw in txt.splitlines():
            line = raw.strip("\r\n")
            if not line:
                if cur:
                    blocks.append(cur)
                    cur = {}
                continue
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip()
            if k:
                cur[k] = v
        if cur:
            blocks.append(cur)
        return blocks or None
    except Exception:
        return None

def _collect_gpu_info_windows() -> dict:
    out: dict = {}
    try:
        gpu = _powershell_json(
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name, DriverVersion, AdapterRAM | ConvertTo-Json -Depth 2"
        )
        gpus: list[dict] = []
        if isinstance(gpu, list):
            gpus = [g for g in gpu if isinstance(g, dict)]
        elif isinstance(gpu, dict):
            gpus = [gpu]
        if gpus:
            out["gpu_probe_method"] = "powershell_cim"
            names = [str(g.get("Name") or "").strip() for g in gpus if str(g.get("Name") or "").strip()]
            drivers = [str(g.get("DriverVersion") or "").strip() for g in gpus if str(g.get("DriverVersion") or "").strip()]
            ram = []
            for g in gpus:
                try:
                    v = g.get("AdapterRAM")
                    if v is None:
                        continue
                    ram.append(round(float(v) / (1024 ** 3), 2))
                except Exception:
                    continue
            out["gpu_count"] = len(gpus)
            out["gpu_names"] = "; ".join(names) if names else None
            out["gpu_driver_versions"] = "; ".join(drivers) if drivers else None
            out["gpu_adapter_ram_gb"] = "; ".join(str(x) for x in ram) if ram else None
            return out
    except Exception:
        pass
    try:
        blocks = _wmic_list(["wmic", "path", "Win32_VideoController", "get", "Name,DriverVersion,AdapterRAM", "/format:list"])
        if blocks:
            out["gpu_probe_method"] = "wmic"
            names = [str(b.get("Name") or "").strip() for b in blocks if str(b.get("Name") or "").strip()]
            drivers = [str(b.get("DriverVersion") or "").strip() for b in blocks if str(b.get("DriverVersion") or "").strip()]
            ram = []
            for b in blocks:
                try:
                    v = b.get("AdapterRAM")
                    if v is None:
                        continue
                    v2 = float(re.sub(r"[^0-9.]", "", str(v)))
                    if v2:
                        ram.append(round(v2 / (1024 ** 3), 2))
                except Exception:
                    continue
            out["gpu_count"] = len(blocks)
            out["gpu_names"] = "; ".join(names) if names else None
            out["gpu_driver_versions"] = "; ".join(drivers) if drivers else None
            out["gpu_adapter_ram_gb"] = "; ".join(str(x) for x in ram) if ram else None
            return out
    except Exception:
        pass
    out["gpu_probe_method"] = "unavailable"
    return out

def _windows_friendly_release() -> str:
    """Return '11' on Windows 11 instead of Python's misleading '10'."""
    release = platform.release()
    if platform.system() != "Windows" or release != "10":
        return release
    try:
        ver = platform.version()  # e.g. "10.0.22631"
        build = int(ver.split(".")[-1])
        if build >= 22000:
            return "11"
    except Exception:
        pass
    return release


def _collect_system_capabilities() -> dict:
    now_utc = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    try:
        if ZoneInfo is not None:
            now_local = datetime.now(ZoneInfo("Europe/Oslo")).replace(microsecond=0).isoformat()
        else:
            now_local = datetime.now().replace(microsecond=0).isoformat()
    except Exception:
        now_local = datetime.now().replace(microsecond=0).isoformat()

    out: dict = {
        "collected_at_utc": now_utc,
        "collected_at_local": now_local,
        "mesa_version": mesa_version_display,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "os_name": platform.system(),
        "os_release": _windows_friendly_release(),
        "os_version": platform.version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count() or None,
        "project_base": PROJECT_BASE,
        "working_dir": original_working_directory,
    }
    try:
        import psutil
        vm = psutil.virtual_memory()
        out["ram_total_gb"] = round(float(vm.total) / (1024 ** 3), 2)
        out["ram_available_gb"] = round(float(vm.available) / (1024 ** 3), 2)
        out["cpu_count_physical"] = psutil.cpu_count(logical=False)
        try:
            out["cpu_freq_mhz"] = getattr(psutil.cpu_freq(), "current", None)
        except Exception:
            pass
    except Exception:
        try:
            m = _windows_global_memory_status()
            if m:
                out.update(m)
                out.setdefault("ram_probe_method", "ctypes_globalmemorystatusex")
        except Exception:
            pass
    try:
        usage = shutil.disk_usage(original_working_directory)
        out["disk_total_gb"] = round(float(usage.total) / (1024 ** 3), 2)
        out["disk_free_gb"] = round(float(usage.free) / (1024 ** 3), 2)
    except Exception:
        pass
    try:
        if os.name == "nt":
            out.update(_collect_gpu_info_windows())
    except Exception:
        pass
    return out

def _write_system_capabilities_parquet(geoparquet_dir: str, row: dict) -> str | None:
    try:
        os.makedirs(geoparquet_dir, exist_ok=True)
        out_path = os.path.join(geoparquet_dir, "tbl_system_capabilities.parquet")
        df = pd.DataFrame([row])
        try:
            wkb_point_0_0 = struct.pack("<BIdd", 1, 1, 0.0, 0.0)
        except Exception:
            wkb_point_0_0 = b""
        df["geometry"] = [wkb_point_0_0]
        import pyarrow as pa
        table = pa.Table.from_pandas(df, preserve_index=False)
        try:
            geo_md = {
                "version": "1.0.0",
                "primary_column": "geometry",
                "columns": {
                    "geometry": {
                        "encoding": "WKB",
                        "geometry_types": ["Point"],
                        "crs": {"id": {"authority": "EPSG", "code": 4326}},
                    }
                },
            }
            md = dict(table.schema.metadata or {})
            md[b"geo"] = json.dumps(geo_md, ensure_ascii=True).encode("utf-8")
            table = table.replace_schema_metadata(md)
        except Exception:
            pass
        pq.write_table(table, out_path)
        return out_path
    except Exception as e:
        try:
            log_to_logfile(f"[system] Failed to write system capabilities parquet: {e}")
        except Exception:
            pass
        return None

def _ensure_system_capabilities_snapshot(cfg: configparser.ConfigParser) -> None:
    try:
        present_keys = (
            "system_os", "system_ram_gb", "system_cpu", "system_cores",
            "system_gpu", "gpu_names", "cpu_count_logical",
        )
        default = cfg["DEFAULT"] if cfg and "DEFAULT" in cfg else {}
        has_info = any(str(default.get(k, "")).strip() for k in present_keys)
        if "SYSTEM" in cfg and cfg["SYSTEM"]:
            has_info = True
        geoparquet_dir = _detect_geoparquet_dir()
        out_path = os.path.join(geoparquet_dir, "tbl_system_capabilities.parquet")
        if has_info:
            return
        if os.path.exists(out_path):
            return
        row = _collect_system_capabilities()
        written = _write_system_capabilities_parquet(geoparquet_dir, row)
        if written:
            log_to_logfile(f"[system] Wrote host capabilities snapshot: {written}")
    except Exception:
        pass

def _read_system_capabilities_latest_row() -> dict | None:
    try:
        geoparquet_dir = _detect_geoparquet_dir()
        path = os.path.join(geoparquet_dir, "tbl_system_capabilities.parquet")
        if not os.path.exists(path):
            return None
        df = pd.read_parquet(path)
        if df is None or df.empty:
            return None
        row = dict(df.iloc[-1].to_dict())
        row.pop("geometry", None)
        return row
    except Exception:
        return None

def _format_system_capabilities_for_about(row: dict | None) -> str:
    if not row:
        geoparquet_dir = _detect_geoparquet_dir()
        path = os.path.join(geoparquet_dir, "tbl_system_capabilities.parquet")
        return (
            "System profile not available yet.\n\n"
            "MESA writes a one-time host capability snapshot to:\n"
            f"{path}\n\n"
            "It will be created automatically on first run."
        )

    def _g(key: str) -> str:
        v = row.get(key)
        if v is None:
            return "--"
        s = str(v).strip()
        return s if s else "--"

    lines: list[str] = []
    lines.append(f"Collected (local): {_g('collected_at_local')}")
    lines.append(f"Collected (UTC):   {_g('collected_at_utc')}")
    lines.append("")
    lines.append(f"OS:        {_g('os_name')} {_g('os_release')}")
    lines.append(f"Platform:  {_g('platform')}")
    lines.append(f"Machine:   {_g('machine')}")
    lines.append("")
    cpu = _g("processor")
    if cpu == "--":
        cpu = _g("cpu")
    lines.append(f"CPU:       {cpu}")
    lines.append(f"Cores:     physical={_g('cpu_count_physical')}  logical={_g('cpu_count_logical')}")
    lines.append(f"RAM (GB):  total={_g('ram_total_gb')}  available={_g('ram_available_gb')}")
    lines.append("")
    lines.append(f"GPU:       {_g('gpu_names')}")
    lines.append(f"GPU drv:   {_g('gpu_driver_versions')}")
    lines.append(f"GPU RAM:   {_g('gpu_adapter_ram_gb')}")
    lines.append(f"GPU probe: {_g('gpu_probe_method')}")
    lines.append("")
    lines.append(f"Disk (GB): total={_g('disk_total_gb')}  free={_g('disk_free_gb')}")
    lines.append("")
    lines.append(f"Python:    {_g('python_version')}")
    lines.append(f"Exe:       {_g('python_executable')}")
    lines.append("")
    lines.append(f"MESA:      {_g('mesa_version')}")
    if str(row.get("ram_probe_method") or "").strip():
        lines.append("")
        lines.append(f"RAM probe: {_g('ram_probe_method')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Main setup
# ---------------------------------------------------------------------
original_working_directory = PROJECT_BASE
config_file = os.path.join(PROJECT_BASE, "config.ini")
if not os.path.exists(config_file):
    raise FileNotFoundError(f"Configuration not found: {config_file}")

gpkg_file = os.path.join(original_working_directory, "output", "mesa.gpkg")
config = read_config(config_file)
mesa_version = config['DEFAULT'].get('mesa_version', 'MESA 5')


def _format_display_version(version: str) -> str:
    v = (version or "").strip() or "MESA"
    try:
        if getattr(sys, "frozen", False):
            return v
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_BASE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if sha:
            return f"{v} (dev {sha})"
    except Exception:
        pass
    return v


def _read_packaged_build_info() -> dict:
    if not getattr(sys, "frozen", False):
        return {}
    try:
        info_path = resolve_path("build_info.json")
        if not os.path.exists(info_path):
            return {}
        with open(info_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


mesa_version_display = _format_display_version(mesa_version)
packaged_build_info = _read_packaged_build_info()
packaged_build_timestamp = str(packaged_build_info.get("build_timestamp", "") or "").strip()

check_and_create_folders()


# =====================================================================
# QSS Stylesheet  (replaces ttkbootstrap theme)
# =====================================================================
MESA_STYLESHEET = """
/* =============================================================
   MESA Qt stylesheet -- warm green/oker palette
   Inspired by GRASP Desktop earth-tone design system
   ============================================================= */

/* ---- Global ---- */
QMainWindow {
    background-color: #f3ecdf;
}
QWidget#CentralWidget {
    background-color: #f3ecdf;
}
QWidget {
    color: #3f3528;
    font-family: "Segoe UI", "Inter", "Helvetica Neue", sans-serif;
    font-size: 10pt;
}

/* ---- Tab widget ---- */
QTabWidget::pane {
    border: 1px solid #cbb791;
    border-top: none;
    background: #f3ecdf;
}
QTabBar {
    background: #e6dac2;
}
QTabBar::tab {
    background: #e6dac2;
    color: #5c4a2f;
    padding: 8px 18px;
    margin-right: 2px;
    border: 1px solid #c6b089;
    border-bottom: none;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    font-weight: 500;
    font-size: 10pt;
}
QTabBar::tab:selected {
    background: #f8f3e9;
    color: #3f3528;
    font-weight: 600;
}
QTabBar::tab:!selected {
    margin-top: 2px;
}
QTabBar::tab:hover:!selected {
    background: #efe3cc;
    color: #3f3528;
}

/* ---- Group boxes (card style) ---- */
QGroupBox {
    font-weight: 600;
    font-size: 10pt;
    background: #faf6ee;
    border: 1px solid #d5c3a4;
    border-radius: 8px;
    margin-top: 12px;
    padding: 18px 16px 14px 16px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 3px 10px;
    color: #715a36;
    font-size: 10pt;
}

/* ---- Buttons ---- */
QPushButton {
    background-color: #f2e7d3;
    color: #4d4029;
    border: 1px solid #c7b18c;
    border-radius: 7px;
    padding: 6px 16px;
    font-weight: 500;
    min-width: 100px;
}
QPushButton:hover {
    background-color: #eadbbd;
    border-color: #b99763;
}
QPushButton:pressed {
    background-color: #ddc89f;
}
QPushButton:disabled {
    background-color: #eee5d7;
    color: #a28f71;
    border-color: #d4c6af;
}

/* Role-based button variants */
QPushButton[role="primary"] {
    background-color: #d9bd7d;
    color: #3f3018;
    border: 1px solid #9b7c3d;
    font-weight: 600;
}
QPushButton[role="primary"]:hover {
    background-color: #e1c78d;
    border-color: #8c6d31;
}
QPushButton[role="primary"]:pressed {
    background-color: #cfb06f;
}
QPushButton[role="success"] {
    background-color: #e6ecd8;
    color: #34482a;
    border: 1px solid #9cad83;
}
QPushButton[role="success"]:hover {
    background-color: #edf2e3;
    border-color: #899b72;
}
QPushButton[role="success"]:pressed {
    background-color: #d9e3c7;
}
QPushButton[role="danger"] {
    background-color: #efdfd5;
    color: #5c3825;
    border: 1px solid #c4a08d;
    border-radius: 7px;
}
QPushButton[role="danger"]:hover {
    background-color: #f4e4da;
    border-color: #b58672;
}
QPushButton[role="danger"]:pressed {
    background-color: #e6d2c5;
}

/* ---- Labels ---- */
QLabel {
    background: transparent;
    color: #3f3528;
}
QLabel[role="heading"] {
    font-size: 11pt;
    font-weight: 700;
    color: #3f3528;
}
QLabel[role="description"] {
    color: #6a5533;
    font-size: 9pt;
    line-height: 1.4;
}
QLabel[role="muted"] {
    color: #9a8a6e;
    font-size: 8pt;
}

/* Status bullet colors */
QLabel[status="success"] { color: #4d7c0f; }
QLabel[status="warning"] { color: #b45309; }
QLabel[status="danger"]  { color: #b02a37; }

/* ---- Text editors ---- */
QPlainTextEdit, QTextEdit {
    background: #fffdf8;
    border: 1px solid #d9cab1;
    border-radius: 6px;
    padding: 8px;
    font-family: "Cascadia Code", "Consolas", "Fira Code", monospace;
    font-size: 9pt;
    selection-background-color: #d7bb7f;
    selection-color: #2f2517;
}
QPlainTextEdit:focus, QTextEdit:focus {
    border-color: #b99763;
}

/* ---- Tables ---- */
QTableWidget {
    background: #fffdf8;
    border: 1px solid #d9cab1;
    border-radius: 6px;
    gridline-color: #e2d5bf;
    font-size: 9pt;
    alternate-background-color: #f6efdf;
}
QTableWidget::item {
    padding: 4px 8px;
}
QTableWidget::item:hover {
    background-color: #efe1bf;
}
QTableWidget::item:selected {
    background-color: #d7bb7f;
    color: #2f2517;
}
QHeaderView::section {
    background: #e7dbc4;
    color: #54462d;
    font-weight: 600;
    border: none;
    border-bottom: 1px solid #cfbc99;
    padding: 4px 6px;
}

/* ---- Checkboxes ---- */
QCheckBox {
    spacing: 6px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 3px;
}
QCheckBox::indicator:unchecked {
    background: #f5edd8;
    border: 1.5px solid #9a8260;
}
QCheckBox::indicator:unchecked:hover {
    border-color: #715a36;
    background: #efe3cc;
}
QCheckBox::indicator:checked {
    background: #715a36;
    border: 1.5px solid #513912;
}
QCheckBox::indicator:checked:hover {
    background: #8a6d3a;
}
QCheckBox::indicator:disabled {
    background: #e5dcc9;
    border-color: #c4b699;
}

/* ---- Radio buttons ---- */
QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border-radius: 8px;
}
QRadioButton::indicator:unchecked {
    background: #f5edd8;
    border: 1.5px solid #9a8260;
}
QRadioButton::indicator:unchecked:hover {
    border-color: #715a36;
    background: #efe3cc;
}
QRadioButton::indicator:checked {
    background: #715a36;
    border: 1.5px solid #513912;
}
QRadioButton::indicator:checked:hover {
    background: #8a6d3a;
}

/* ---- Scroll bars (thin, modern) ---- */
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #d0bc97;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #b99763;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: transparent;
    height: 8px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #d0bc97;
    border-radius: 4px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background: #b99763;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

/* ---- Progress dialog ---- */
QProgressDialog {
    background: #faf6ee;
}
QProgressBar {
    background: #e7dbc4;
    border: 1px solid #d3c29f;
    border-radius: 5px;
    height: 6px;
    text-align: center;
    color: #4f4129;
}
QProgressBar::chunk {
    background: #b79b67;
    border-radius: 4px;
}

/* ---- Scroll area ---- */
QScrollArea {
    border: none;
    background: transparent;
}

/* ---- Tooltips ---- */
QToolTip {
    background: #3f3528;
    color: #faf6ee;
    border: none;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 9pt;
}

/* ---- Status bar ---- */
QStatusBar {
    background: #eee4d2;
    border-top: 1px solid #cfbc99;
}

/* ---- Footer ---- */
QLabel[role="footer"] {
    color: #9a8a6e;
    font-size: 8pt;
}
"""


# =====================================================================
# Custom widgets
# =====================================================================
from PySide6.QtGui import QPainter, QPen, QBrush
from PySide6.QtWidgets import QProgressDialog, QLineEdit, QCheckBox


class _ActionCard(QFrame):
    """Clickable card with title, description, and right-arrow indicator.

    Replaces the old button+label pair with a modern card that responds
    as a whole surface to hover and click.
    """

    def __init__(self, title: str, description: str, callback, parent=None):
        super().__init__(parent)
        self._callback = callback
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("ActionCard")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(10)

        # Text column
        text_col = QVBoxLayout()
        text_col.setSpacing(2)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            "font-weight: 600; font-size: 10pt; color: #3f3528; background: transparent;"
        )
        text_col.addWidget(title_lbl)

        desc_lbl = QLabel(description)
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet(
            "font-size: 9pt; color: #6a5533; background: transparent;"
        )
        text_col.addWidget(desc_lbl)

        layout.addLayout(text_col, stretch=1)

        # Right-arrow indicator
        arrow = QLabel("\u203A")  # single right-pointing angle quotation mark
        arrow.setStyleSheet(
            "font-size: 16pt; color: #b99763; background: transparent;"
        )
        arrow.setAlignment(Qt.AlignCenter)
        arrow.setFixedWidth(20)
        layout.addWidget(arrow)

        self.setStyleSheet("""
            #ActionCard {
                background: #fffdf8;
                border: 1px solid #e2d5bf;
                border-radius: 8px;
            }
            #ActionCard:hover {
                background: #f6efdf;
                border-color: #c6b089;
            }
        """)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._callback:
            self._callback()
        super().mousePressEvent(event)


class _BannerWidget(QFrame):
    """Full-width banner that paints the image edge-to-edge with rounded
    corners, overlaying title and version text.  Falls back to a plain
    gradient strip when no image is available."""

    _RADIUS = 10

    def __init__(self, pixmap: QPixmap | None = None, parent=None):
        super().__init__(parent)
        self._source = pixmap
        h = max(72, pixmap.height()) if pixmap and not pixmap.isNull() else 72
        self.setFixedHeight(h)
        self.setStyleSheet("background: transparent;")

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        r = self.rect()
        radius = self._RADIUS

        from PySide6.QtGui import QLinearGradient, QPainterPath
        # Clip to rounded rect
        path = QPainterPath()
        path.addRoundedRect(float(r.x()), float(r.y()),
                            float(r.width()), float(r.height()),
                            radius, radius)
        p.setClipPath(path)

        if self._source and not self._source.isNull():
            # Scale image to fill the full widget width
            scaled = self._source.scaled(
                r.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            )
            p.drawPixmap(0, 0, scaled)
            # Semi-transparent overlay so text reads well
            p.fillRect(r, QColor(243, 236, 223, 140))
        else:
            # Gradient fallback
            grad = QLinearGradient(0, 0, r.width(), 0)
            grad.setColorAt(0.0, QColor("#e6dac2"))
            grad.setColorAt(1.0, QColor("#f3ecdf"))
            p.fillRect(r, grad)

        # Title left
        p.setPen(QColor("#3f3528"))
        p.setFont(QFont("Segoe UI", 16, QFont.Bold))
        fm = p.fontMetrics()
        text_y = (r.height() - fm.height()) // 2 + fm.ascent()
        p.drawText(20, text_y, "MESA tool")

        # Version right
        version_text = mesa_version_display or "unknown"
        p.setFont(QFont("Segoe UI", 9, italic=True))
        fm2 = p.fontMetrics()
        vw = fm2.horizontalAdvance(version_text)
        ver_x = r.width() - vw - 20
        if packaged_build_timestamp:
            # Two-line: version + build stamp
            ver_y = (r.height() // 2) - 2
            p.drawText(ver_x, ver_y, version_text)
            p.setPen(QColor("#6a5533"))
            p.setFont(QFont("Segoe UI", 8))
            fm3 = p.fontMetrics()
            build_text = "Build " + packaged_build_timestamp
            bw = fm3.horizontalAdvance(build_text)
            p.drawText(r.width() - bw - 20, ver_y + fm3.height() + 1, build_text)
        else:
            ver_y = (r.height() - fm2.height()) // 2 + fm2.ascent()
            p.drawText(ver_x, ver_y, version_text)

        p.end()


class _InfoCircleLabel(QLabel):
    """Small painted circle with 'i' that opens a URL on click."""

    def __init__(self, url: str, parent=None):
        super().__init__(parent)
        self._url = url
        self.setFixedSize(20, 20)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Open in browser")

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor("#9b7c3d"), 1.5))
        p.setBrush(QBrush(QColor("#faf6ee")))
        p.drawEllipse(2, 2, 15, 15)
        p.setPen(QColor("#715a36"))
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(self.rect(), Qt.AlignCenter, "i")
        p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._url:
            webbrowser.open(self._url)


# =====================================================================
# Main Window
# =====================================================================
class MesaMainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(mesa_version_display or "MESA")

        # Icon
        icon_path = resolve_path(os.path.join("system_resources", "mesa.ico"))
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # Size: 5:3 aspect ratio, large enough for all content
        self.resize(1160, 740)
        self.setMinimumSize(900, 560)

        # Central widget
        central = QWidget()
        central.setObjectName("CentralWidget")
        self.setCentralWidget(central)
        self._main_layout = QVBoxLayout(central)
        self._main_layout.setContentsMargins(12, 6, 12, 6)
        self._main_layout.setSpacing(8)

        # Build UI
        self._build_header()
        self._build_tabs()
        self._build_footer()

        # System capabilities snapshot (background thread)
        try:
            threading.Thread(
                target=_ensure_system_capabilities_snapshot,
                args=(config,),
                daemon=True,
            ).start()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    def _build_header(self):
        # --- Banner strip ---
        banner_path = resolve_path(os.path.join("system_resources", "top_graphics.png"))
        pixmap = None
        if os.path.exists(banner_path):
            px = QPixmap(banner_path)
            if not px.isNull():
                pixmap = px

        banner = _BannerWidget(pixmap)
        self._main_layout.addWidget(banner)

    # ------------------------------------------------------------------
    # Tabs (with Exit button right-aligned on the tab bar row)
    # ------------------------------------------------------------------
    def _build_tabs(self):
        self._tabs = QTabWidget()

        # Compact Exit button embedded in the tab bar's corner
        exit_btn = QPushButton("Exit")
        exit_btn.setObjectName("CornerExitButton")
        exit_btn.setFixedHeight(24)
        exit_btn.setStyleSheet("""
            QPushButton#CornerExitButton {
                background: #eadfc8; border: 1px solid #b79f73;
                border-radius: 4px; color: #453621; font-size: 8pt;
                padding: 2px 14px; margin: 2px 6px;
            }
            QPushButton#CornerExitButton:hover { background: #e1d1ae; }
            QPushButton#CornerExitButton:pressed { background: #d4c094; }
        """)
        exit_btn.clicked.connect(self.close)
        self._tabs.setCornerWidget(exit_btn, Qt.TopRightCorner)

        self._main_layout.addWidget(self._tabs, stretch=1)

        self._build_workflows_tab()
        self._build_status_tab()
        self._build_config_tab()
        self._build_tune_tab()
        self._build_manage_tab()
        self._build_geonode_tab()
        self._build_about_tab()

    # ---- Tab 1: Workflows ----
    def _build_workflows_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)

        intro = QLabel(
            "Launch the workflows grouped by phase. Pick the task that matches what you are "
            "trying to achieve, then glance at the Status tab to confirm progress and find "
            "project statistics."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        grid = QGridLayout()
        grid.setSpacing(12)
        layout.addLayout(grid, stretch=1)

        workflow_sections = [
            ("Prepare data (step 1)", "Import or create new data and generate supporting geometries.", [
                ("Assets", open_assets,
                 "Import assets and edit asset groups in one tool."),
                ("Geocodes", geocodes_grids,
                 "Create or refresh the hexagon/tile grids that support analysis."),
                ("Lines", edit_lines,
                 "Import and edit lines (transport, rivers, utilities, etc)."),
                ("Atlas", make_atlas,
                 "Create/import atlas polygons and edit atlas page metadata in one tool."),
            ]),
            ("Configure (step 2)", "Tune parameters/study areas before running heavy jobs.", [
                ("Parameters", edit_processing_setup,
                 "Adjust weights, thresholds and other processing rules."),
                ("Analysis", open_data_analysis_setup,
                 "Define analysis groups and study area polygons."),
            ]),
            ("Process (step 3)", "Execute the automated steps that build fresh outputs.", [
                ("Process", open_process_all,
                 "Runs area, line, and analysis processing."),
            ]),
            ("Results (step 4)", "Open the interactive viewers and export the deliverables.", [
                ("Asset map", open_asset_layers_viewer,
                 "Inspect layers with AI-assisted styling controls."),
                ("Results map", open_maps_overview,
                 "Review current background layers together with processed assets."),
                ("Compare study areas", open_data_analysis_presentation,
                 "Open the dashboard for comparing study groups."),
                ("Report engine", open_present_files,
                 "Create a tailor made report based on the latest results."),
            ]),
        ]

        for col_idx, (title, description, actions) in enumerate(workflow_sections):
            group = QGroupBox(title)
            group_layout = QVBoxLayout(group)
            group_layout.setSpacing(4)

            desc = QLabel(description)
            desc.setStyleSheet("font-size: 9pt; color: #6a5533; font-weight: 500;")
            desc.setWordWrap(True)
            group_layout.addWidget(desc)

            group_layout.addSpacing(6)

            for action_label, action_command, action_desc in actions:
                card = _ActionCard(action_label, action_desc, action_command)
                group_layout.addWidget(card)

            group_layout.addStretch()
            grid.addWidget(group, 0, col_idx)
            grid.setColumnStretch(col_idx, 1)

        self._tabs.addTab(tab, "Workflows")

    # ---- Tab 2: Status ----
    # ------------------------------------------------------------------
    # Status tab helpers (log parsing, timestamps, formatting)
    # ------------------------------------------------------------------
    _log_duration_cache = {"mtime": None, "durations": {}, "seconds": {}, "times": {}}
    _status_calc_runtime = {"seconds": None}
    _asset_area_cache = {"path": None, "mtime": None, "area_km2": None}
    _lines_length_cache = {"path": None, "mtime": None, "length_km": None}

    @staticmethod
    def _parse_log_timestamp(line: str) -> datetime | None:
        if not line or len(line) < 19:
            return None
        ts = line[:19]
        try:
            return datetime.strptime(ts, "%Y.%m.%d %H:%M:%S")
        except Exception:
            return None

    @staticmethod
    def _scan_last_run_from_log(log_path, start_markers, end_markers_primary,
                                end_markers_secondary=None):
        try:
            if not os.path.exists(log_path):
                return None, None
        except Exception:
            return None, None

        current_start = None
        secondary_end = None
        last_duration = None
        last_end = None

        def _has_any(haystack, needles):
            return any(n in haystack for n in needles)

        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                for raw in f:
                    line = raw.rstrip("\n")
                    ts = MesaMainWindow._parse_log_timestamp(line)
                    if ts is None:
                        continue
                    if current_start is None:
                        if _has_any(line, start_markers):
                            current_start = ts
                            secondary_end = None
                        continue
                    if _has_any(line, start_markers):
                        if secondary_end is not None:
                            last_duration = (secondary_end - current_start).total_seconds()
                            last_end = secondary_end
                        current_start = ts
                        secondary_end = None
                        continue
                    if end_markers_secondary and _has_any(line, end_markers_secondary):
                        secondary_end = ts
                    if _has_any(line, end_markers_primary):
                        last_duration = (ts - current_start).total_seconds()
                        last_end = ts
                        current_start = None
                        secondary_end = None
            if current_start is not None and secondary_end is not None:
                last_duration = (secondary_end - current_start).total_seconds()
                last_end = secondary_end
        except Exception:
            return None, None
        return last_duration, last_end

    @staticmethod
    def _fmt_timestamp(ts_val):
        if not ts_val:
            return "--"
        try:
            return datetime.fromtimestamp(ts_val).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return "--"

    @staticmethod
    def _path_mtime(path):
        try:
            return os.path.getmtime(path)
        except Exception:
            return None

    @staticmethod
    def _fmt_duration(seconds):
        if seconds is None:
            return "--"
        try:
            total = int(round(float(seconds)))
        except Exception:
            return "--"
        if total < 0:
            return "--"
        mins, sec = divmod(total, 60)
        hrs, mins = divmod(mins, 60)
        days, hrs = divmod(hrs, 24)
        if days:
            return f"{days}d {hrs}h"
        if hrs:
            return f"{hrs}h {mins:02d}m"
        if mins:
            return f"{mins}m {sec:02d}s"
        return f"{sec}s"

    @staticmethod
    def _fmt_stats_runtime(seconds):
        if seconds is None:
            return "--"
        try:
            s = float(seconds)
        except Exception:
            return "--"
        if s < 0:
            return "--"
        if s < 1.0:
            return f"{s * 1000.0:.0f} ms"
        if s < 60.0:
            return f"{s:.1f}s"
        return MesaMainWindow._fmt_duration(s)

    @staticmethod
    def _fmt_count(value):
        if value is None:
            return "--"
        try:
            return f"{int(value):,}"
        except Exception:
            return "--"

    @staticmethod
    def _fmt_km2(value):
        if value is None:
            return "--"
        try:
            return f"{float(value):,.1f}"
        except Exception:
            return "--"

    @staticmethod
    def _fmt_km(value):
        if value is None:
            return "--"
        try:
            return f"{float(value):,.1f}"
        except Exception:
            return "--"

    # ------------------------------------------------------------------
    # Status tab: build
    # ------------------------------------------------------------------
    def _build_status_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        intro = QLabel("Get on top of your project with key metrics statistics.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # ---- Top row: Status & help (left) + Recent activity (right) ----
        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        # --- Status and help ---
        self._status_group = QGroupBox("Status and help")
        self._status_group.setStyleSheet(
            "QGroupBox { border: 1px solid #c6b089; }"
            "QGroupBox::title { color: #715a36; }"
        )
        self._status_layout = QGridLayout(self._status_group)
        self._status_layout.setContentsMargins(10, 16, 10, 10)
        self._status_layout.setHorizontalSpacing(8)
        self._status_layout.setVerticalSpacing(6)
        self._status_layout.setColumnStretch(0, 0)   # bullet
        self._status_layout.setColumnStretch(1, 1)   # message
        self._status_layout.setColumnStretch(2, 0)   # info icon
        top_row.addWidget(self._status_group, stretch=1)

        # --- Recent activity ---
        timeline_group = QGroupBox("Recent activity")
        timeline_layout = QVBoxLayout(timeline_group)
        timeline_layout.setContentsMargins(10, 16, 10, 10)
        timeline_layout.setSpacing(6)

        self._timeline_entries = []
        TIMELINE_COLORS = {
            "success": "#4d7c0f",
            "info":    "#9b7c3d",
            "warning": "#b45309",
            "secondary": "#9a8a6e",
        }
        for _ in range(6):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 2, 0, 2)
            row_layout.setSpacing(6)

            color_bar = QFrame()
            color_bar.setFixedSize(4, 20)
            color_bar.setStyleSheet(f"background: {TIMELINE_COLORS['success']}; border-radius: 2px;")
            row_layout.addWidget(color_bar)

            title_lbl = QLabel("Event")
            title_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            row_layout.addWidget(title_lbl, stretch=1)

            dur_lbl = QLabel("--")
            dur_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            dur_lbl.setFixedWidth(80)
            dur_lbl.setProperty("role", "muted")
            row_layout.addWidget(dur_lbl)

            time_lbl = QLabel("--")
            time_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            time_lbl.setFixedWidth(120)
            time_lbl.setProperty("role", "muted")
            row_layout.addWidget(time_lbl)

            timeline_layout.addWidget(row_widget)
            self._timeline_entries.append((color_bar, title_lbl, dur_lbl, time_lbl))

        timeline_layout.addStretch()
        top_row.addWidget(timeline_group, stretch=1)
        layout.addLayout(top_row, stretch=1)

        # ---- Bottom row: Four insight boxes ----
        insights_row = QHBoxLayout()
        insights_row.setSpacing(10)

        def _make_insight_box(title, header_left, header_right, num_rows):
            group = QGroupBox(title)
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(10, 16, 10, 8)
            group_layout.setSpacing(0)

            table = QGridLayout()
            table.setContentsMargins(0, 0, 0, 0)
            table.setHorizontalSpacing(8)
            table.setVerticalSpacing(3)
            table.setColumnStretch(0, 1)
            table.setColumnStretch(1, 0)

            hdr_l = QLabel(header_left)
            hdr_l.setFont(QFont("Segoe UI", 9, QFont.Bold))
            table.addWidget(hdr_l, 0, 0, Qt.AlignLeft)

            hdr_r = QLabel(header_right)
            hdr_r.setFont(QFont("Segoe UI", 9, QFont.Bold))
            hdr_r.setAlignment(Qt.AlignRight)
            table.addWidget(hdr_r, 0, 1, Qt.AlignRight)

            cells = []
            for i in range(num_rows):
                k_lbl = QLabel("--")
                k_lbl.setWordWrap(True)
                v_lbl = QLabel("")
                v_lbl.setAlignment(Qt.AlignRight)
                v_lbl.setFixedWidth(70)
                table.addWidget(k_lbl, i + 1, 0, Qt.AlignLeft)
                table.addWidget(v_lbl, i + 1, 1, Qt.AlignRight)
                cells.append((k_lbl, v_lbl))

            group_layout.addLayout(table)
            group_layout.addStretch()
            return group, cells

        self._geocode_box, self._geocode_cells = _make_insight_box(
            "Objects per geocode", "Geocode", "Objects", 6)
        insights_row.addWidget(self._geocode_box)

        self._assets_box, self._assets_cells = _make_insight_box(
            "Assets overview", "Metric", "Value", 4)
        insights_row.addWidget(self._assets_box)

        self._lines_box, self._lines_cells = _make_insight_box(
            "Lines", "Metric", "Value", 4)
        insights_row.addWidget(self._lines_box)

        self._analysis_box, self._analysis_cells = _make_insight_box(
            "Results metrics", "Metric", "Count", 2)
        insights_row.addWidget(self._analysis_box)

        layout.addLayout(insights_row)

        self._tabs.addTab(tab, "Status")

        # Refresh on tab change
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # Deferred initial load
        QTimer.singleShot(500, self._refresh_status_tab)

    def _on_tab_changed(self, index):
        if self._tabs.tabText(index) == "Status":
            self._refresh_status_tab()

    def _refresh_status_tab(self):
        try:
            started = time.perf_counter()
            self.update_stats(gpkg_file)
            self._update_insight_boxes()
            self._status_calc_runtime["seconds"] = time.perf_counter() - started
            self._update_timeline()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Status tab: update status indicators
    # ------------------------------------------------------------------
    def update_stats(self, _gpkg_path):
        # Clear existing status widgets
        while self._status_layout.count():
            item = self._status_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        geoparquet_dir = _detect_geoparquet_dir()

        if not os.path.isdir(geoparquet_dir):
            self._add_status_row(0, "danger",
                                 "No data imported.\nStart with importing data.",
                                 "https://github.com/ragnvald/mesa/wiki")
        else:
            my_status = get_status(geoparquet_dir)
            if not my_status.empty and {'Status', 'Message', 'Link'}.issubset(my_status.columns):
                for idx, row in my_status.iterrows():
                    symbol = row['Status']
                    role = 'success' if symbol == "+" else 'warning' if symbol == "/" else 'danger'
                    self._add_status_row(idx, role, row['Message'], row['Link'])
            else:
                self._add_status_row(0, "danger",
                                     "To initiate the system please import assets.\n"
                                     "Press the Assets button.",
                                     "https://github.com/ragnvald/mesa/wiki")

    def _add_status_row(self, row_idx, status_role, message, link_url):
        STATUS_COLORS = {
            "success": "#4d7c0f",
            "warning": "#b45309",
            "danger":  "#b02a37",
        }
        color = STATUS_COLORS.get(status_role, "#ef4444")
        bullet = QLabel("\u2B24")  # filled circle
        bullet.setStyleSheet(f"QLabel {{ color: {color}; font-size: 16pt; }}")
        bullet.setFixedWidth(28)
        bullet.setAlignment(Qt.AlignCenter)
        self._status_layout.addWidget(bullet, row_idx, 0)

        msg = QLabel(message)
        msg.setWordWrap(True)
        self._status_layout.addWidget(msg, row_idx, 1)

        self._add_info_link(self._status_layout, link_url, row_idx, 2)

    def _add_info_link(self, grid_layout, url, row, col):
        icon = _InfoCircleLabel(url)
        grid_layout.addWidget(icon, row, col)

    # ------------------------------------------------------------------
    # Status tab: timeline
    # ------------------------------------------------------------------
    def _recent_activity_durations(self):
        log_path = os.path.join(original_working_directory, "log.txt")
        try:
            mtime = os.path.getmtime(log_path)
        except Exception:
            mtime = None
        cache = self._log_duration_cache
        if mtime and cache.get("mtime") == mtime:
            return cache.get("durations", {})

        seconds = {}
        times = {}

        seconds["Assets"] = self._scan_last_run_from_log(
            log_path,
            start_markers=["Step [Assets] STARTED"],
            end_markers_primary=["Step [Assets] COMPLETED", "Step [Assets] FAILED"],
        )[0]

        mosaic_secs, mosaic_end = self._scan_last_run_from_log(
            log_path,
            start_markers=["Step [Mosaic] STARTED"],
            end_markers_primary=["Step [Mosaic] COMPLETED", "Step [Mosaic] FAILED"],
        )
        seconds["Build basic_mosaic"] = mosaic_secs
        times["Build basic_mosaic"] = mosaic_end.strftime("%Y-%m-%d %H:%M") if mosaic_end else "--"

        seconds["Processing"] = self._scan_last_run_from_log(
            log_path,
            start_markers=[
                "[Process] STARTED", "DATA PROCESS START",
                "LINES PROCESS START", "LINES PROCESS START (Parquet)",
                "ANALYSIS PROCESS START", "Attempting to run command:",
                "[Stage 1/4] Preparing workspace",
            ],
            end_markers_primary=[
                "[Process] COMPLETED", "[Process] FAILED",
                "DATA PROCESS COMPLETED", "LINES PROCESS COMPLETED",
                "ANALYSIS PROCESS COMPLETED",
                "ERROR: data processing failed", "ERROR: lines processing failed",
                "ERROR: analysis processing failed",
                "[Tiles] Completed.",
                "[Tiles] Skipping MBTiles stage because processing exited with code",
                "[Tiles] tbl_flat not present or empty; skipping MBTiles generation.",
                "[Tiles] tiles_create_raster exited with code",
                "[Tiles] Error:", "Error during processing:",
            ],
        )[0]

        seconds["Line processing"] = self._scan_last_run_from_log(
            log_path,
            start_markers=[
                "SEGMENT PROCESS START", "LINES PROCESS START",
                "LINES PROCESS START (Parquet)",
            ],
            end_markers_primary=[
                "COMPLETED: Segment processing", "FAILED: Segment processing",
                "LINES PROCESS COMPLETED", "ERROR: lines processing failed",
            ],
        )[0]

        seconds["Newest report export"] = self._scan_last_run_from_log(
            log_path,
            start_markers=["Report mode selected:"],
            end_markers_primary=["Word report created:", "ERROR during report generation:"],
        )[0]

        durations = {k: self._fmt_duration(v) for k, v in seconds.items()}
        cache.update({"mtime": mtime, "seconds": seconds, "durations": durations, "times": times})
        return durations

    def _last_flat_timestamp(self):
        geoparquet_dir = _detect_geoparquet_dir()
        flat_path = os.path.join(geoparquet_dir, "tbl_flat.parquet")
        ts = self._path_mtime(flat_path)
        if ts:
            return self._fmt_timestamp(ts)
        return config['DEFAULT'].get('last_process_run', '--')

    def _last_asset_import_timestamp(self):
        geoparquet_dir = _detect_geoparquet_dir()
        candidates = [
            os.path.join(geoparquet_dir, "tbl_asset_group.parquet"),
            os.path.join(geoparquet_dir, "tbl_asset_object.parquet"),
        ]
        newest = None
        for p in candidates:
            ts = self._path_mtime(p)
            if ts and (newest is None or ts > newest):
                newest = ts
        return self._fmt_timestamp(newest) if newest else "--"

    def _last_line_processing_timestamp(self):
        geoparquet_dir = _detect_geoparquet_dir()
        segment_path = os.path.join(geoparquet_dir, "tbl_segment_flat.parquet")
        ts = self._path_mtime(segment_path)
        if ts:
            return self._fmt_timestamp(ts)
        return config['DEFAULT'].get('last_lines_process_run', '--')

    def _latest_report_timestamp(self):
        output_dir = os.path.join(original_working_directory, "output")
        reports_dir = os.path.join(output_dir, "reports")
        newest_ts = None

        def _consider_file(path):
            nonlocal newest_ts
            ts = self._path_mtime(path)
            if ts and (newest_ts is None or ts > newest_ts):
                newest_ts = ts

        def _scan_dir(dir_path, recursive):
            if not os.path.isdir(dir_path):
                return
            try:
                if recursive:
                    for root_dir, _dirs, files in os.walk(dir_path):
                        for name in files:
                            low = name.lower()
                            if low.endswith(".pdf") or low.endswith(".docx"):
                                _consider_file(os.path.join(root_dir, name))
                else:
                    for entry in os.scandir(dir_path):
                        if not entry.is_file():
                            continue
                        low = entry.name.lower()
                        if low.endswith(".pdf") or low.endswith(".docx"):
                            _consider_file(entry.path)
            except Exception:
                return

        _scan_dir(reports_dir, recursive=True)
        _scan_dir(output_dir, recursive=False)
        if newest_ts:
            return self._fmt_timestamp(newest_ts)
        return config['DEFAULT'].get('last_report_export', '--')

    def _update_timeline(self):
        TIMELINE_COLORS = {
            "success":   "#4d7c0f",
            "info":      "#9b7c3d",
            "warning":   "#b45309",
            "secondary": "#9a8a6e",
        }
        durations = dict(self._recent_activity_durations())
        durations["Time to calculate stats on this page"] = self._fmt_stats_runtime(
            self._status_calc_runtime.get("seconds"))
        times = self._log_duration_cache.get("times", {})

        events = [
            ("Assets", self._last_asset_import_timestamp(), "success"),
            ("Build basic_mosaic", times.get("Build basic_mosaic", "--"), "info"),
            ("Processing", self._last_flat_timestamp(), "info"),
            ("Line processing", self._last_line_processing_timestamp(), "warning"),
            ("Newest report export", self._latest_report_timestamp(), "secondary"),
            ("Time to calculate stats on this page", "", "secondary"),
        ]
        for idx, (title, timestamp, style_key) in enumerate(events):
            if idx >= len(self._timeline_entries):
                break
            color_bar, title_lbl, dur_lbl, time_lbl = self._timeline_entries[idx]
            color = TIMELINE_COLORS.get(style_key, "#9ca3af")
            color_bar.setStyleSheet(f"background: {color}; border-radius: 2px;")
            title_lbl.setText(title)
            dur_lbl.setText(durations.get(title, "--"))
            time_lbl.setText(timestamp)

    # ------------------------------------------------------------------
    # Status tab: insight boxes
    # ------------------------------------------------------------------
    def _populate_two_col_table(self, cells, rows):
        for i, (k_lbl, v_lbl) in enumerate(cells):
            if i < len(rows):
                k, v = rows[i]
                k_lbl.setText(k)
                v_lbl.setText(v)
            else:
                k_lbl.setText("")
                v_lbl.setText("")

    def _fetch_geocode_objects_summary(self):
        flat_path = _locate_geoparquet_file("tbl_flat")
        if not flat_path or not os.path.exists(flat_path):
            return [("No processing results yet.", "")]
        try:
            preferred_cols = ["geocode_category", "name_gis_geocodegroup", "ref_geocodegroup"]
            available_cols = []
            try:
                available_cols = pq.ParquetFile(flat_path).schema.names
            except Exception:
                pass
            target_col = next((col for col in preferred_cols if col in available_cols), None)
            if not target_col:
                df_all = pd.read_parquet(flat_path)
                target_col = next((col for col in preferred_cols if col in df_all.columns), None)
                if not target_col:
                    return [("No geocode identifiers found.", "")]
                data = df_all[target_col]
            else:
                data = pd.read_parquet(flat_path, columns=[target_col])[target_col]
            counts = data.value_counts(dropna=False).head(5)
            rows = []
            for idx, val in counts.items():
                key = "(missing)" if pd.isna(idx) else str(idx)
                try:
                    value = str(int(val))
                except Exception:
                    value = str(val)
                rows.append((key, value))
            if data.nunique(dropna=False) > 5:
                rows.append(("...", ""))
            return rows if rows else [("No records found.", "")]
        except Exception as exc:
            return [("Unable to read tbl_flat:", ""), (str(exc)[:160], "")]

    def _measurement_epsg(self):
        try:
            raw_area = (config["DEFAULT"].get("area_projection_epsg", "") or "").strip()
            if raw_area:
                return int(float(raw_area))
            raw_working = (config["DEFAULT"].get("working_projection_epsg", "") or "").strip()
            if raw_working:
                epsg = int(float(raw_working))
                if epsg in (4326, 4258):
                    return None
                return epsg
            return None
        except Exception:
            return None

    def _total_area_km2_from_asset_objects(self, asset_object_path):
        if not asset_object_path or not os.path.exists(asset_object_path):
            return None
        try:
            import geopandas as gpd
            mtime = self._path_mtime(asset_object_path)
            cache = self._asset_area_cache
            if mtime and cache.get("path") == asset_object_path and cache.get("mtime") == mtime:
                return cache.get("area_km2")
            epsg = self._measurement_epsg()
            gdf = gpd.read_parquet(asset_object_path, columns=["geometry"])
            if epsg:
                try:
                    if gdf.crs is None:
                        gdf = gdf.set_crs(epsg=epsg, allow_override=True)
                    elif getattr(gdf.crs, "to_epsg", lambda: None)() != epsg:
                        gdf = gdf.to_crs(epsg=epsg)
                except Exception:
                    pass
            else:
                try:
                    gdf = gdf.to_crs(epsg=3857)
                except Exception:
                    pass
            total_m2 = float(gdf.geometry.area.fillna(0).sum()) if not gdf.empty else 0.0
            km2 = total_m2 / 1_000_000.0
            cache.update({"path": asset_object_path, "mtime": mtime, "area_km2": km2})
            return km2
        except Exception:
            return None

    def _total_length_km_from_lines(self, lines_path):
        if not lines_path or not os.path.exists(lines_path):
            return None
        try:
            import geopandas as gpd
            mtime = self._path_mtime(lines_path)
            cache = self._lines_length_cache
            if mtime and cache.get("path") == lines_path and cache.get("mtime") == mtime:
                return cache.get("length_km")
            epsg = self._measurement_epsg()
            gdf = gpd.read_parquet(lines_path, columns=["geometry"])
            if epsg:
                try:
                    if gdf.crs is None:
                        gdf = gdf.set_crs(epsg=epsg, allow_override=True)
                    elif getattr(gdf.crs, "to_epsg", lambda: None)() != epsg:
                        gdf = gdf.to_crs(epsg=epsg)
                except Exception:
                    pass
            else:
                try:
                    gdf = gdf.to_crs(epsg=3857)
                except Exception:
                    pass
            total_m = float(gdf.geometry.length.fillna(0).sum()) if not gdf.empty else 0.0
            km = total_m / 1000.0
            cache.update({"path": lines_path, "mtime": mtime, "length_km": km})
            return km
        except Exception:
            return None

    def _fetch_asset_summary(self):
        asset_group_path = _locate_geoparquet_file("tbl_asset_group")
        if not asset_group_path:
            return [("Assets not imported yet.", "")]
        try:
            layers = _parquet_row_count(asset_group_path)
            objects = None
            try:
                cols = pq.ParquetFile(asset_group_path).schema.names
                if "total_asset_objects" in cols:
                    s = pd.read_parquet(asset_group_path, columns=["total_asset_objects"])["total_asset_objects"]
                    objects = int(s.fillna(0).sum())
            except Exception:
                pass
            asset_object_path = _locate_geoparquet_file("tbl_asset_object")
            total_area_km2 = self._total_area_km2_from_asset_objects(asset_object_path)
            return [
                ("Layers", self._fmt_count(layers)),
                ("Objects", self._fmt_count(objects)),
                ("Area (km2)", self._fmt_km2(total_area_km2)),
            ]
        except Exception as exc:
            return [("Unable to read assets:", ""), (str(exc)[:160], "")]

    def _fetch_lines_summary(self):
        lines_path = _locate_geoparquet_file("tbl_lines")
        if not lines_path:
            return [("Lines not imported yet.", "")]
        segments_path = _locate_geoparquet_file("tbl_segment_flat")
        try:
            lines_count = _parquet_row_count(lines_path)
            segments_count = _parquet_row_count(segments_path) if segments_path and os.path.exists(segments_path) else 0
            total_length_km = self._total_length_km_from_lines(lines_path)
            return [
                ("Lines", self._fmt_count(lines_count)),
                ("Segments", self._fmt_count(segments_count)),
                ("Length (km)", self._fmt_km(total_length_km)),
            ]
        except Exception as exc:
            return [("Unable to read lines:", ""), (str(exc)[:160], "")]

    def _fetch_analysis_summary(self):
        stacked_path = _locate_geoparquet_file("tbl_stacked")
        flat_path = _locate_geoparquet_file("tbl_flat")
        if not stacked_path and not flat_path:
            return [("Results missing.", "")]
        try:
            stacked_count = _parquet_row_count(stacked_path) if stacked_path else None
            flat_count = _parquet_row_count(flat_path) if flat_path else None
            return [
                ("Analysis layer objects", "--" if stacked_count is None else str(stacked_count)),
                ("Presentation layer objects", "--" if flat_count is None else str(flat_count)),
            ]
        except Exception as exc:
            return [("Unable to read results:", ""), (str(exc)[:160], "")]

    def _update_insight_boxes(self):
        self._populate_two_col_table(self._geocode_cells, self._fetch_geocode_objects_summary())
        self._populate_two_col_table(self._assets_cells, self._fetch_asset_summary())
        self._populate_two_col_table(self._lines_cells, self._fetch_lines_summary())
        self._populate_two_col_table(self._analysis_cells, self._fetch_analysis_summary())

    # ---- Tab 3: Config ----
    def _build_config_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)

        intro = QLabel("Edit config.ini directly. Changes are saved when you press Save.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._config_status_label = QLabel("")
        self._config_status_label.setProperty("role", "muted")
        layout.addWidget(self._config_status_label)

        self._config_editor = QPlainTextEdit()
        self._config_editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self._config_editor, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._load_config_editor)
        btn_row.addWidget(reload_btn)
        save_btn = QPushButton("Save")
        save_btn.setProperty("role", "success")
        save_btn.clicked.connect(self._save_config_editor)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

        self._tabs.addTab(tab, "Config")
        self._load_config_editor()

    def _load_config_editor(self):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                self._config_editor.setPlainText(f.read())
            self._config_status_label.setText(f"Loaded: {config_file}")
        except Exception as e:
            self._config_status_label.setText(f"Error loading config: {e}")

    def _save_config_editor(self):
        try:
            text = self._config_editor.toPlainText()
            with open(config_file, "w", encoding="utf-8") as f:
                f.write(text)
            self._config_status_label.setText(f"Saved: {config_file}")
            log_to_logfile("Config saved from editor")
        except Exception as e:
            self._config_status_label.setText(f"Error saving config: {e}")
            log_to_logfile(f"Config save error: {e}")

    # ---- Tab 4: Tune processing ----
    def _build_tune_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Top: intro + buttons side by side
        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        intro_col = QVBoxLayout()
        intro_col.setSpacing(4)
        intro = QLabel(
            "Automatically tune processing settings based on this computer's CPU and RAM.\n"
            "This updates selected keys in [DEFAULT] in config.ini."
        )
        intro.setWordWrap(True)
        intro_col.addWidget(intro)

        self._tune_status_label = QLabel(
            "Ready. Press Evaluate to compare current and advised settings."
        )
        self._tune_status_label.setProperty("role", "muted")
        self._tune_status_label.setWordWrap(True)
        intro_col.addWidget(self._tune_status_label)
        intro_col.addStretch()
        top_row.addLayout(intro_col, stretch=1)

        btn_col = QVBoxLayout()
        btn_col.setSpacing(6)

        eval_btn = QPushButton("Evaluate")
        eval_btn.setProperty("role", "primary")
        eval_btn.setFixedWidth(180)
        eval_btn.clicked.connect(self._evaluate_processing_tuning)
        btn_col.addWidget(eval_btn)

        self._commit_tune_btn = QPushButton("Commit changes")
        self._commit_tune_btn.setProperty("role", "success")
        self._commit_tune_btn.setFixedWidth(180)
        self._commit_tune_btn.setEnabled(False)
        self._commit_tune_btn.clicked.connect(self._commit_processing_tuning)
        btn_col.addWidget(self._commit_tune_btn)

        restore_btn = QPushButton("Restore previous tuning")
        restore_btn.setFixedWidth(180)
        restore_btn.clicked.connect(self._restore_previous_tuning)
        btn_col.addWidget(restore_btn)

        btn_col.addStretch()
        top_row.addLayout(btn_col)
        layout.addLayout(top_row)

        # Comparison table
        compare_group = QGroupBox("Current vs advised settings")
        compare_layout = QVBoxLayout(compare_group)
        compare_layout.setContentsMargins(10, 16, 10, 10)

        hdr_row = QHBoxLayout()
        hdr_l = QLabel("Current settings")
        hdr_l.setProperty("role", "muted")
        hdr_r = QLabel("Advised settings")
        hdr_r.setProperty("role", "muted")
        hdr_row.addWidget(hdr_l)
        hdr_row.addWidget(hdr_r)
        compare_layout.addLayout(hdr_row)

        self._tune_table = QTableWidget(0, 4)
        self._tune_table.setHorizontalHeaderLabels(["Key", "Current", "Advised", "Suggested"])
        self._tune_table.horizontalHeader().setStretchLastSection(True)
        self._tune_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._tune_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._tune_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._tune_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._tune_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._tune_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._tune_table.setAlternatingRowColors(True)
        self._tune_table.verticalHeader().setVisible(False)
        compare_layout.addWidget(self._tune_table)
        layout.addWidget(compare_group, stretch=1)

        # Explanation text
        self._tune_text = QPlainTextEdit()
        self._tune_text.setReadOnly(True)
        self._tune_text.setMaximumHeight(120)
        layout.addWidget(self._tune_text)

        # Internal state
        self._pending_tune_values: dict[str, str] = {}
        self._pending_before_values: dict[str, str] = {}
        self._pending_eval_rationale = ""
        self._tune_backup_path = os.path.join(PROJECT_BASE, "output", "processing_tuning_backup.json")

        self._tabs.addTab(tab, "Tune processing")

    # ------------------------------------------------------------------
    # Tune processing: logic
    # ------------------------------------------------------------------
    def _recommended_processing_tuning(self, cap_row: dict):
        try:
            logical = int(float(cap_row.get("cpu_count_logical") or 0))
        except Exception:
            logical = int(os.cpu_count() or 4)
        logical = max(1, logical)

        try:
            ram_total = float(cap_row.get("ram_total_gb") or 0.0)
        except Exception:
            ram_total = 0.0
        if ram_total <= 0:
            ram_total = 16.0

        if logical <= 4:
            worker_cap = 2
        elif logical <= 8:
            worker_cap = 4
        elif logical <= 12:
            worker_cap = 6
        elif logical <= 16:
            worker_cap = 8
        elif logical <= 24:
            worker_cap = 10
        else:
            worker_cap = 12

        if ram_total <= 12:
            approx_gb_per_worker = 2.5
            geocode_soft_limit = 180
            asset_soft_limit = 30000
            target_geocodes = 1800
            chunk_size = 10000
            cell_size = 7000
        elif ram_total <= 24:
            approx_gb_per_worker = 3.5
            geocode_soft_limit = 260
            asset_soft_limit = 45000
            target_geocodes = 2600
            chunk_size = 14000
            cell_size = 8000
        elif ram_total <= 48:
            approx_gb_per_worker = 4.5
            geocode_soft_limit = 340
            asset_soft_limit = 70000
            target_geocodes = 3400
            chunk_size = 18000
            cell_size = 9000
        else:
            approx_gb_per_worker = 6.0
            geocode_soft_limit = 420
            asset_soft_limit = 100000
            target_geocodes = 4200
            chunk_size = 22000
            cell_size = 10000

        recommendations = {
            "max_workers": "0",
            "auto_workers_min": "1",
            "auto_workers_max": str(worker_cap),
            "approx_gb_per_worker": f"{approx_gb_per_worker:.1f}",
            "mem_target_frac": "0.85",
            "target_geocodes_per_chunk": str(int(target_geocodes)),
            "chunk_backlog_multiplier": "3.5",
            "chunk_cells_min": "2",
            "chunk_cells_max": "72",
            "chunk_overshoot_factor": "1.15",
            "geocode_soft_limit": str(int(geocode_soft_limit)),
            "asset_soft_limit": str(int(asset_soft_limit)),
            "chunk_size": str(int(chunk_size)),
            "cell_size": str(int(cell_size)),
        }

        rationale = (
            f"Detected logical CPU: {logical}. Detected RAM: {ram_total:.1f} GB. "
            f"Using auto worker mode with an upper cap of {worker_cap} and "
            f"moderate memory headroom (mem_target_frac=0.85). "
            "Chunking is tuned toward better load balancing to reduce slow end-of-run tails."
        )
        return recommendations, rationale

    def _update_default_keys_preserve_comments(self, path, updates):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        lines = text.splitlines(keepends=True)
        newline = "\n"
        for line in lines:
            if line.endswith("\r\n"):
                newline = "\r\n"
                break
        default_start = None
        default_end = None
        for i, line in enumerate(lines):
            if line.strip().lower() == "[default]":
                default_start = i
                break
        if default_start is None:
            lines = ["[DEFAULT]" + newline, newline] + lines
            default_start = 0
        for i in range(default_start + 1, len(lines)):
            s = lines[i].strip()
            if s.startswith("[") and s.endswith("]"):
                default_end = i
                break
        if default_end is None:
            default_end = len(lines)
        found = set()
        for idx in range(default_start + 1, default_end):
            line = lines[idx]
            for key, val in updates.items():
                pat = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)(.*?)(\s*(?:[;#].*)?)$", re.IGNORECASE)
                m = pat.match(line.rstrip("\r\n"))
                if not m:
                    continue
                lines[idx] = f"{m.group(1)}{val}{m.group(3)}{newline}"
                found.add(key)
                break
        missing = [k for k in updates.keys() if k not in found]
        if missing:
            insert_at = default_end
            if insert_at > 0 and lines[insert_at - 1].strip() != "":
                lines.insert(insert_at, newline)
                insert_at += 1
            for key in missing:
                lines.insert(insert_at, f"{key:<30} = {updates[key]}{newline}")
                insert_at += 1
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8", newline="") as f:
            f.writelines(lines)
        os.replace(tmp_path, path)

    def _write_tune_backup(self, previous_values, applied_values):
        os.makedirs(os.path.dirname(self._tune_backup_path), exist_ok=True)
        payload = {
            "created_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "config_path": config_file,
            "keys": sorted(list(applied_values.keys())),
            "previous_values": previous_values,
            "applied_values": applied_values,
        }
        tmp_path = self._tune_backup_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._tune_backup_path)

    def _read_tune_backup(self):
        if not os.path.exists(self._tune_backup_path):
            return None
        try:
            with open(self._tune_backup_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    @staticmethod
    def _format_int_like(value):
        try:
            return f"{int(float(value)):,}"
        except Exception:
            return str(value)

    def _populate_tune_table(self, current_values, advised_values):
        self._tune_table.setRowCount(0)
        suggested_count = 0
        for key in sorted(advised_values.keys()):
            old_val = str(current_values.get(key, "(missing)")).strip() or "(empty)"
            new_val = str(advised_values.get(key, "")).strip() or "(empty)"
            changed = old_val != new_val
            if changed:
                suggested_count += 1
            row = self._tune_table.rowCount()
            self._tune_table.insertRow(row)
            self._tune_table.setItem(row, 0, QTableWidgetItem(key))
            self._tune_table.setItem(row, 1, QTableWidgetItem(old_val))
            self._tune_table.setItem(row, 2, QTableWidgetItem(new_val))
            self._tune_table.setItem(row, 3, QTableWidgetItem("Yes" if changed else "No"))
            if changed:
                highlight = QColor("#f6efdf")
                for col in range(4):
                    self._tune_table.item(row, col).setBackground(highlight)
        return suggested_count

    def _evaluate_processing_tuning(self):
        try:
            detected = _collect_system_capabilities() or {}
            tuning, rationale = self._recommended_processing_tuning(detected)

            cfg_before = read_config(config_file)
            before_default = cfg_before["DEFAULT"] if "DEFAULT" in cfg_before else {}
            current_values = {key: str(before_default.get(key, "(missing)")).strip() for key in tuning.keys()}

            self._pending_tune_values = {k: str(v) for k, v in tuning.items()}
            self._pending_before_values = {k: str(before_default.get(k, "")).strip() for k in tuning.keys()}
            self._pending_eval_rationale = rationale

            suggested_count = self._populate_tune_table(current_values, tuning)
            self._commit_tune_btn.setEnabled(suggested_count > 0)

            summary_lines = [
                "Evaluation completed.",
                rationale,
                f"Suggested changes: {suggested_count}",
                "Press Commit changes to apply the advised values to config.ini.",
                f"Config file: {config_file}",
            ]
            if suggested_count == 0:
                summary_lines.append("No changes suggested; config.ini already matches advised values.")
            self._tune_text.setPlainText("\n".join(summary_lines))

            if suggested_count > 0:
                self._tune_status_label.setText("Evaluation complete. Review highlighted rows, then Commit changes.")
            else:
                self._tune_status_label.setText("Evaluation complete. No changes needed.")
            log_to_logfile("Processing tuning evaluated from Tune processing tab")
        except Exception as exc:
            self._commit_tune_btn.setEnabled(False)
            self._tune_status_label.setText("Processing tuning evaluation failed")
            QMessageBox.critical(self, "Evaluate failed", f"Could not evaluate tuning:\n{exc}")

    def _commit_processing_tuning(self):
        if not self._pending_tune_values:
            QMessageBox.information(self, "Commit changes", "Run Evaluate first to generate advised settings.")
            return
        confirmed = QMessageBox.question(
            self, "Commit processing tuning",
            "Apply the advised processing settings to config.ini?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirmed != QMessageBox.Yes:
            return
        try:
            self._update_default_keys_preserve_comments(config_file, self._pending_tune_values)
            self._write_tune_backup(dict(self._pending_before_values), dict(self._pending_tune_values))

            global config
            config = read_config(config_file)
            try:
                self._load_config_editor()
            except Exception:
                pass

            changed_lines = []
            for key, new_val in self._pending_tune_values.items():
                old_val = str(self._pending_before_values.get(key, "(missing)")).strip() or "(empty)"
                if old_val != str(new_val):
                    if key in ("chunk_size", "cell_size", "target_geocodes_per_chunk",
                               "geocode_soft_limit", "asset_soft_limit", "auto_workers_max"):
                        old_disp = self._format_int_like(old_val) if old_val != "(missing)" else old_val
                        new_disp = self._format_int_like(new_val)
                    else:
                        old_disp = old_val
                        new_disp = str(new_val)
                    changed_lines.append(f"- {key}: {old_disp} -> {new_disp}")
            if not changed_lines:
                changed_lines.append("- No effective changes (values were already tuned).")

            summary = "\n".join([
                "Tune processing committed.",
                self._pending_eval_rationale,
                "Applied keys:",
                *changed_lines,
                f"Backup saved: {self._tune_backup_path}",
                f"Config file: {config_file}",
            ])
            self._tune_text.setPlainText(summary)
            self._commit_tune_btn.setEnabled(False)
            self._tune_status_label.setText("Processing tuning committed and saved to config.ini")
            log_to_logfile("Processing tuning committed from Tune processing tab")
        except Exception as exc:
            self._tune_status_label.setText("Commit processing tuning failed")
            QMessageBox.critical(self, "Commit failed", f"Could not apply advised tuning:\n{exc}")

    def _restore_previous_tuning(self):
        backup = self._read_tune_backup()
        if not backup:
            QMessageBox.information(self, "Restore tuning", "No previous tuning backup was found.")
            return
        prev = backup.get("previous_values") if isinstance(backup, dict) else None
        if not isinstance(prev, dict) or not prev:
            QMessageBox.information(self, "Restore tuning", "Backup exists but does not contain restore values.")
            return
        confirmed = QMessageBox.question(
            self, "Restore previous tuning",
            "Restore the previously saved values for tuned processing keys in config.ini?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirmed != QMessageBox.Yes:
            return
        try:
            cfg_before = read_config(config_file)
            before_default = cfg_before["DEFAULT"] if "DEFAULT" in cfg_before else {}
            restore_updates = {str(k): str(v) for k, v in prev.items()}
            self._update_default_keys_preserve_comments(config_file, restore_updates)

            global config
            config = read_config(config_file)
            try:
                self._load_config_editor()
            except Exception:
                pass

            changed_lines = []
            for key, restored_val in restore_updates.items():
                old_val = str(before_default.get(key, "(missing)")).strip()
                if old_val != str(restored_val):
                    if key in ("chunk_size", "cell_size", "target_geocodes_per_chunk",
                               "geocode_soft_limit", "asset_soft_limit", "auto_workers_max"):
                        old_disp = self._format_int_like(old_val) if old_val != "(missing)" else old_val
                        new_disp = self._format_int_like(restored_val) if str(restored_val).strip() else "(empty)"
                    else:
                        old_disp = old_val
                        new_disp = str(restored_val).strip() if str(restored_val).strip() else "(empty)"
                    changed_lines.append(f"- {key}: {old_disp} -> {new_disp}")
            if not changed_lines:
                changed_lines.append("- No effective changes (config already matched backup values).")

            created_at = str(backup.get("created_at", "--"))
            summary = "\n".join([
                "Restore previous tuning completed.",
                f"Backup timestamp: {created_at}",
                "Restored keys:",
                *changed_lines,
                f"Config file: {config_file}",
            ])
            self._tune_text.setPlainText(summary)
            self._tune_status_label.setText("Previous tuning values restored")
            log_to_logfile("Processing tuning restored from Tune processing tab")
        except Exception as exc:
            self._tune_status_label.setText("Restore previous tuning failed")
            QMessageBox.critical(self, "Restore tuning failed", f"Could not restore tuned values:\n{exc}")

    # ---- Tab 5: Manage MESA data ----
    def _build_manage_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        intro = QLabel(
            "Manage your project data with backup, restore, and cleanup operations."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._manage_status = QLabel("")
        self._manage_status.setProperty("role", "muted")
        self._manage_status.setWordWrap(True)
        layout.addWidget(self._manage_status)

        # --- Backup section ---
        backup_group = QGroupBox("Backup and restore")
        backup_layout = QVBoxLayout(backup_group)
        backup_layout.setSpacing(8)

        backup_intro = QLabel(
            "A backup saves your complete project state as a single ZIP file. "
            "It includes all imported data (input/), processed results (output/), "
            "and your configuration (config.ini). Use this before major changes "
            "or to transfer a project to another computer.\n\n"
            "Restoring a backup replaces the current project data entirely. "
            "The existing input/, output/, and config.ini are removed and "
            "replaced with the contents of the selected backup ZIP."
        )
        backup_intro.setWordWrap(True)
        backup_intro.setStyleSheet("color: #6a5533; font-size: 9pt;")
        backup_layout.addWidget(backup_intro)

        backup_btn_row = QHBoxLayout()
        backup_btn_row.setSpacing(10)
        create_btn = QPushButton("Create backup")
        create_btn.setProperty("role", "primary")
        create_btn.setFixedWidth(160)
        create_btn.clicked.connect(self._do_backup)
        backup_btn_row.addWidget(create_btn)

        restore_btn = QPushButton("Restore backup")
        restore_btn.setFixedWidth(160)
        restore_btn.clicked.connect(self._do_restore)
        backup_btn_row.addWidget(restore_btn)
        backup_btn_row.addStretch()
        backup_layout.addLayout(backup_btn_row)

        layout.addWidget(backup_group)

        # --- Clear output section ---
        clear_group = QGroupBox("Clear generated data")
        clear_layout = QVBoxLayout(clear_group)
        clear_layout.setSpacing(8)

        clear_intro = QLabel(
            "Remove all imported and processed data from the output/ folder. "
            "Your original source files in input/ and your configuration are "
            "kept intact. Use this to start fresh or free disk space after "
            "a completed project cycle."
        )
        clear_intro.setWordWrap(True)
        clear_intro.setStyleSheet("color: #6a5533; font-size: 9pt;")
        clear_layout.addWidget(clear_intro)

        clear_btn_row = QHBoxLayout()
        clear_btn = QPushButton("Clear output")
        clear_btn.setProperty("role", "danger")
        clear_btn.setFixedWidth(160)
        clear_btn.clicked.connect(self._do_clear_output)
        clear_btn_row.addWidget(clear_btn)
        clear_btn_row.addStretch()
        clear_layout.addLayout(clear_btn_row)

        layout.addWidget(clear_group)

        layout.addStretch()
        self._tabs.addTab(tab, "Manage MESA data")

    def _do_backup(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Choose backup destination folder",
            original_working_directory,
        )
        if not folder:
            return

        # Show progress dialog (backup can be slow for large projects)
        progress = QProgressDialog("Creating backup archive...", None, 0, 0, self)
        progress.setWindowTitle("Backup in progress")
        progress.setMinimumDuration(0)
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.show()
        QApplication.processEvents()

        result_path = None
        error_msg = None
        try:
            result_path = create_backup_archive(original_working_directory, folder)
        except Exception as e:
            error_msg = str(e)

        progress.close()

        if error_msg:
            self._manage_status.setText(f"Backup failed: {error_msg}")
            QMessageBox.critical(self, "Backup failed", error_msg)
        else:
            fname = os.path.basename(result_path) if result_path else "backup"
            fdir = os.path.dirname(result_path) if result_path else ""
            try:
                size_mb = os.path.getsize(result_path) / (1024 * 1024)
                size_str = f"\nSize: {size_mb:.1f} MB"
            except Exception:
                size_str = ""
            self._manage_status.setText(f"Backup created: {result_path}")
            QMessageBox.information(self, "Backup created",
                                    f"Backup saved successfully.\n\n"
                                    f"File: {fname}\n"
                                    f"Location: {fdir}{size_str}")

    def _do_restore(self):
        zip_path, _ = QFileDialog.getOpenFileName(
            self, "Select backup ZIP",
            original_working_directory,
            "ZIP files (*.zip);;All files (*.*)",
        )
        if not zip_path:
            return
        confirm = QMessageBox.question(
            self, "Confirm restore",
            "Restoring backup will delete and replace current input/, output/, and config.ini.\n\n"
            "Do you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        progress = QProgressDialog("Restoring backup archive...", None, 0, 0, self)
        progress.setWindowTitle("Restore in progress")
        progress.setMinimumDuration(0)
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.show()
        QApplication.processEvents()

        error_msg = None
        try:
            restore_backup_archive(original_working_directory, zip_path)
        except Exception as e:
            error_msg = str(e)

        progress.close()

        if error_msg:
            self._manage_status.setText(f"Restore failed: {error_msg}")
            QMessageBox.critical(self, "Restore failed", error_msg)
        else:
            self._manage_status.setText(f"Backup restored: {zip_path}")
            QMessageBox.information(self, "Restore completed",
                                    "Backup restore completed successfully.")
            try:
                self._refresh_status_tab()
            except Exception:
                pass

    def _do_clear_output(self):
        output_dir = os.path.join(original_working_directory, "output")
        if not os.path.isdir(output_dir):
            QMessageBox.information(self, "Clear output",
                                    f"Output folder does not exist:\n{output_dir}")
            return
        confirm = QMessageBox.question(
            self, "Confirm clear output",
            "This will permanently delete imported and processed data from the output/ folder.\n\n"
            "Your source files in input/ and your configuration are not affected.\n\n"
            "Do you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        def _keep_github_related(name: str) -> bool:
            lname = (name or "").strip().lower()
            return lname.startswith(".git") or lname == ".github"

        removed = 0
        kept: list[str] = []
        failed: list[str] = []
        for name in os.listdir(output_dir):
            path = os.path.join(output_dir, name)
            if _keep_github_related(name):
                kept.append(name)
                continue
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                removed += 1
            except Exception as exc:
                failed.append(f"{name}: {exc}")

        log_to_logfile(f"Clear output executed. removed={removed}, kept={len(kept)}, failed={len(failed)}")
        if failed:
            details = "\n".join(failed[:6])
            if len(failed) > 6:
                details += f"\n... and {len(failed) - 6} more"
            QMessageBox.warning(
                self, "Clear output completed with issues",
                f"Removed: {removed} items\nKept: {len(kept)} system files"
                f"\nFailed: {len(failed)}\n\n{details}",
            )
            return
        self._manage_status.setText(f"Cleared output. Removed: {removed} items, kept: {len(kept)} system files.")
        QMessageBox.information(
            self, "Clear output completed",
            f"Removed: {removed} items\nKept: {len(kept)} system files",
        )

    # ---- Tab 6: Publish to GeoNode ----
    def _build_geonode_tab(self):
        _ensure_code_dir_on_syspath()
        import queue as _queue
        self._geonode_log_queue: _queue.Queue = _queue.Queue()
        self._geonode_cancel_event = __import__("threading").Event()
        self._geonode_layer_checkboxes: dict = {}   # id → QCheckBox
        self._geonode_confirm_event = __import__("threading").Event()
        self._geonode_confirm_result: str = "skip"  # "replace" or "skip"

        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(0)

        # ── two-column body ──────────────────────────────────────────────
        body = QHBoxLayout()
        body.setSpacing(12)
        outer.addLayout(body, stretch=1)

        # ── LEFT: layer list ─────────────────────────────────────────────
        layers_group = QGroupBox("Layers to publish")
        layers_group.setFixedWidth(272)
        layers_outer = QVBoxLayout(layers_group)
        layers_outer.setContentsMargins(8, 8, 8, 8)
        layers_outer.setSpacing(0)

        # Scrollable inner widget
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner_widget = QWidget()
        inner_vbox = QVBoxLayout(inner_widget)
        inner_vbox.setContentsMargins(0, 0, 4, 0)
        inner_vbox.setSpacing(0)

        # Load layer catalogue
        try:
            from geonode_export import layer_info as _layer_info  # type: ignore[import]
            gpq_dir = _detect_geoparquet_dir()
            info = _layer_info(gpq_dir)
            sens_layers = info["sensitivity"]
            supp_layers = info["supporting"]
        except Exception:
            import geonode_export as _ge  # type: ignore[import]
            sens_layers = []
            supp_layers = [{**l, "available": False, "row_count": None}
                           for l in _ge.SUPPORTING_LAYERS]

        # Store all layer defs for use in export
        self._geonode_all_layers = sens_layers + supp_layers

        def _add_section_header(text: str):
            lbl = QLabel(text)
            lbl.setStyleSheet(
                "color: #6a5533; font-size: 8pt; font-weight: bold;"
                "padding: 4px 0px 2px 0px;"
            )
            inner_vbox.addWidget(lbl)

        def _add_layer_row(layer: dict):
            available = layer.get("available", False)
            row_count = layer.get("row_count")

            row_widget = QWidget()
            row_layout = QVBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 2, 0, 2)
            row_layout.setSpacing(1)

            cb = QCheckBox(layer["label"])
            cb.setChecked(layer["default_checked"] and available)
            cb.setEnabled(available)
            cb.setToolTip(layer["hint"])
            if not available:
                cb.setStyleSheet("color: #aaa;")
            row_layout.addWidget(cb)
            self._geonode_layer_checkboxes[layer["id"]] = cb

            detail_parts = []
            if row_count is not None:
                detail_parts.append(f"{row_count:,} features")
            elif not available:
                detail_parts.append("not yet generated")
            if layer.get("size_note"):
                detail_parts.append(layer["size_note"])
            if layer.get("sld_field"):
                detail_parts.append("A-E styled")
            if detail_parts:
                detail = QLabel("    " + "  \u00b7  ".join(detail_parts))
                detail.setStyleSheet(
                    "color: #888; font-size: 8pt;" if available
                    else "color: #bbb; font-size: 8pt;"
                )
                row_layout.addWidget(detail)

            inner_vbox.addWidget(row_widget)

        # Section 1: sensitivity (one per geocode group)
        if sens_layers:
            _add_section_header("Geocode layers (A-E styled)")
            for layer in sens_layers:
                _add_layer_row(layer)
        else:
            _add_section_header("Geocode layers")
            placeholder = QLabel("    Run processing first to generate tbl_flat.")
            placeholder.setStyleSheet("color: #aaa; font-size: 8pt;")
            inner_vbox.addWidget(placeholder)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #d4c8a8; margin: 6px 0px 2px 0px;")
        inner_vbox.addWidget(sep)

        # Section 2: supporting layers
        _add_section_header("Supporting data")
        for layer in supp_layers:
            _add_layer_row(layer)

        inner_vbox.addStretch()
        scroll.setWidget(inner_widget)
        layers_outer.addWidget(scroll)
        body.addWidget(layers_group)

        # ── RIGHT: connection + actions + log ────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(10)
        body.addLayout(right, stretch=1)

        # Connection group
        conn_group = QGroupBox("GeoNode server")
        conn_layout = QGridLayout(conn_group)
        conn_layout.setSpacing(8)
        conn_layout.setColumnMinimumWidth(0, 80)
        conn_layout.setColumnStretch(1, 1)

        conn_layout.addWidget(QLabel("Server URL"), 0, 0)
        self._geonode_url = QLineEdit()
        self._geonode_url.setPlaceholderText("http://your-geonode-server/")
        conn_layout.addWidget(self._geonode_url, 0, 1)

        conn_layout.addWidget(QLabel("Username"), 1, 0)
        self._geonode_username = QLineEdit()
        self._geonode_username.setPlaceholderText("admin")
        conn_layout.addWidget(self._geonode_username, 1, 1)

        conn_layout.addWidget(QLabel("Password"), 2, 0)
        self._geonode_password = QLineEdit()
        self._geonode_password.setEchoMode(QLineEdit.Password)
        self._geonode_password.setPlaceholderText("password")
        conn_layout.addWidget(self._geonode_password, 2, 1)

        conn_btn_row = QHBoxLayout()
        test_btn = QPushButton("Test connection")
        test_btn.setFixedWidth(140)
        test_btn.clicked.connect(self._geonode_test_connection)
        conn_btn_row.addWidget(test_btn)
        self._geonode_conn_status = QLabel("Not tested")
        self._geonode_conn_status.setProperty("role", "muted")
        conn_btn_row.addWidget(self._geonode_conn_status)
        conn_btn_row.addStretch()
        conn_layout.addLayout(conn_btn_row, 3, 0, 1, 2)
        right.addWidget(conn_group)

        # Publish / cancel buttons
        pub_row = QHBoxLayout()
        pub_row.setSpacing(8)
        self._geonode_export_btn = QPushButton("Publish selected layers")
        self._geonode_export_btn.setProperty("role", "primary")
        self._geonode_export_btn.setFixedWidth(190)
        self._geonode_export_btn.clicked.connect(self._geonode_start_export)
        pub_row.addWidget(self._geonode_export_btn)
        self._geonode_cancel_btn = QPushButton("Cancel")
        self._geonode_cancel_btn.setFixedWidth(80)
        self._geonode_cancel_btn.setEnabled(False)
        self._geonode_cancel_btn.clicked.connect(self._geonode_cancel)
        pub_row.addWidget(self._geonode_cancel_btn)
        pub_row.addStretch()
        right.addLayout(pub_row)

        # Map option
        self._geonode_create_map_cb = QCheckBox("Combine published layers into a GeoNode Map")
        self._geonode_create_map_cb.setChecked(True)
        right.addWidget(self._geonode_create_map_cb)

        # Log
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(8, 8, 8, 8)
        self._geonode_log = QPlainTextEdit()
        self._geonode_log.setReadOnly(True)
        self._geonode_log.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 9pt;"
        )
        log_layout.addWidget(self._geonode_log)
        right.addWidget(log_group, stretch=1)

        self._tabs.addTab(tab, "Publish to GeoNode")

        # Load saved connection settings
        self._geonode_load_settings()

        # Timer to drain the log queue
        self._geonode_log_timer = QTimer(self)
        self._geonode_log_timer.timeout.connect(self._geonode_drain_log)
        self._geonode_log_timer.start(200)

    def _geonode_settings_path(self) -> str:
        secrets_dir = os.path.join(original_working_directory, "secrets")
        os.makedirs(secrets_dir, exist_ok=True)
        return os.path.join(secrets_dir, "geonode.ini")

    def _geonode_load_settings(self):
        try:
            import configparser as _cp
            cfg = _cp.ConfigParser()
            cfg.read(self._geonode_settings_path(), encoding="utf-8")
            url = cfg.get("connection", "url", fallback="")
            username = cfg.get("connection", "username", fallback="")
            if url:
                self._geonode_url.setText(url)
            if username:
                self._geonode_username.setText(username)
        except Exception:
            pass

    def _geonode_save_settings(self):
        try:
            import configparser as _cp
            cfg = _cp.ConfigParser()
            cfg["connection"] = {
                "url": self._geonode_url.text().strip(),
                "username": self._geonode_username.text().strip(),
            }
            with open(self._geonode_settings_path(), "w", encoding="utf-8") as f:
                cfg.write(f)
        except Exception:
            pass

    def _geonode_test_connection(self):
        from geonode_export import test_connection as _test_conn  # type: ignore[import]
        url = self._geonode_url.text().strip()
        username = self._geonode_username.text().strip()
        password = self._geonode_password.text()
        if not url or not username or not password:
            self._geonode_conn_status.setText("Fill in URL, username, and password first.")
            return
        self._geonode_conn_status.setText("Testing …")
        QApplication.processEvents()
        ok, msg = _test_conn(url, username, password)
        self._geonode_conn_status.setText(msg)
        if ok:
            self._geonode_conn_status.setStyleSheet("color: #2a7a2a;")
            self._geonode_save_settings()
        else:
            self._geonode_conn_status.setStyleSheet("color: #b00;")

    def _geonode_start_export(self):
        from geonode_export import export_layers as _export_layers  # type: ignore[import]

        url = self._geonode_url.text().strip()
        username = self._geonode_username.text().strip()
        password = self._geonode_password.text()

        if not url or not username or not password:
            QMessageBox.warning(self, "Missing credentials",
                                "Please fill in the server URL, username, and password.")
            return

        selected = [
            lid for lid, cb in self._geonode_layer_checkboxes.items()
            if cb.isChecked()
        ]
        if not selected:
            QMessageBox.warning(self, "No layers selected",
                                "Select at least one layer to publish.")
            return

        self._geonode_log.clear()
        self._geonode_cancel_event.clear()
        self._geonode_export_btn.setEnabled(False)
        self._geonode_cancel_btn.setEnabled(True)
        self._geonode_save_settings()

        gpq_dir = _detect_geoparquet_dir()
        config_path = os.path.join(original_working_directory, "config.ini")
        styles_dir = os.path.join(original_working_directory, "output", "geonode_styles")

        def _run():
            def _log(msg: str):
                self._geonode_log_queue.put(msg)

            def _confirm_cb(layer_name: str, existing_pk: int) -> bool:
                """Ask the main thread whether to replace an existing layer."""
                self._geonode_confirm_event.clear()
                self._geonode_confirm_result = "skip"
                self._geonode_log_queue.put(
                    f"__CONFIRM__:{layer_name}:{existing_pk}"
                )
                self._geonode_confirm_event.wait(timeout=120)
                return self._geonode_confirm_result == "replace"

            try:
                _log(f"Starting export to {url}")
                _log(f"Layers: {', '.join(selected)}\n")
                results = _export_layers(
                    url, username, password, selected, gpq_dir,
                    _log, self._geonode_cancel_event,
                    config_path=config_path,
                    styles_output_dir=styles_dir,
                    all_layers=getattr(self, "_geonode_all_layers", None),
                    confirm_cb=_confirm_cb,
                    create_map=self._geonode_create_map_cb.isChecked(),
                )
                success_count = sum(1 for r in results if r["success"])
                fail_count = len(results) - success_count
                _log(f"\n------------------------------")
                _log(f"Done. {success_count} layer(s) published, {fail_count} failed.")
            except Exception as _exc:
                import traceback as _tb
                _log(f"\nUnhandled error in export thread:")
                _log(_tb.format_exc())
            finally:
                self._geonode_log_queue.put("__DONE__")

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def _geonode_cancel(self):
        self._geonode_cancel_event.set()
        self._geonode_cancel_btn.setEnabled(False)
        self._geonode_log.appendPlainText("Cancelling …")

    def _geonode_drain_log(self):
        import queue as _queue
        try:
            while True:
                msg = self._geonode_log_queue.get_nowait()
                if msg == "__DONE__":
                    self._geonode_export_btn.setEnabled(True)
                    self._geonode_cancel_btn.setEnabled(False)
                elif msg.startswith("__CONFIRM__:"):
                    # Format: __CONFIRM__:<layer_name>:<pk>
                    try:
                        parts = msg.split(":", 2)
                        layer_name = parts[1] if len(parts) > 1 else "?"
                        existing_pk = parts[2] if len(parts) > 2 else "?"
                        ans = QMessageBox.question(
                            self,
                            "Layer already exists",
                            f"'{layer_name}' already exists on GeoNode (pk={existing_pk}).\n\n"
                            "Replace it (delete and re-upload) or skip?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.No,
                        )
                        if ans == QMessageBox.StandardButton.Yes:
                            self._geonode_confirm_result = "replace"
                            self._geonode_log.appendPlainText(
                                f"  Replacing existing '{layer_name}' ..."
                            )
                        else:
                            self._geonode_confirm_result = "skip"
                            self._geonode_log.appendPlainText(
                                f"  Skipping '{layer_name}' (already exists)."
                            )
                    except Exception:
                        self._geonode_confirm_result = "skip"
                    finally:
                        self._geonode_confirm_event.set()
                elif msg.startswith("__MAP__:"):
                    map_url = msg[len("__MAP__:"):]
                    self._geonode_log.appendPlainText(f"  Map URL: {map_url}")
                else:
                    self._geonode_log.appendPlainText(msg)
        except _queue.Empty:
            pass

    # ---- Tab 7: About ----
    def _build_about_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Left column
        left_col = QVBoxLayout()
        left_col.setSpacing(12)

        # --- About MESA ---
        about_group = QGroupBox("About MESA")
        about_layout = QVBoxLayout(about_group)

        version_parts = [mesa_version_display or "MESA"]
        if packaged_build_timestamp:
            version_parts.append(f"Built: {packaged_build_timestamp}")
        about_text = QLabel("\n".join(version_parts))
        about_text.setProperty("role", "heading")
        about_text.setWordWrap(True)
        about_layout.addWidget(about_text)

        about_desc = QLabel(
            "MESA (Method for an Easy Sensitivity Assessment) is a desktop application "
            "for environmental sensitivity assessment. It guides analysts through a "
            "complete workflow: importing spatial data, configuring analysis parameters, "
            "running automated processing, and reviewing results through interactive "
            "maps and reports.\n\n"
            "The system works with GeoParquet data and produces maps, atlases, and "
            "PDF/DOCX reports. All processing is done locally on your machine \u2014 "
            "no data is uploaded to external services."
        )
        about_desc.setWordWrap(True)
        about_layout.addWidget(about_desc)

        about_layout.addStretch()
        left_col.addWidget(about_group, stretch=1)

        # --- Workflow overview ---
        workflow_group = QGroupBox("How it works")
        workflow_layout = QVBoxLayout(workflow_group)

        steps_text = QLabel(
            "\u2460  Prepare data \u2014 Import assets, geocodes, lines, and atlas polygons.\n\n"
            "\u2461  Configure \u2014 Set processing parameters and define analysis groups.\n\n"
            "\u2462  Process \u2014 Run area, line, and analysis processing to generate results.\n\n"
            "\u2463  Review \u2014 Explore results on interactive maps, compare study areas, "
            "and generate reports for distribution."
        )
        steps_text.setWordWrap(True)
        steps_text.setStyleSheet("color: #4d4029; font-size: 9pt; line-height: 1.5;")
        workflow_layout.addWidget(steps_text)

        workflow_layout.addStretch()
        left_col.addWidget(workflow_group, stretch=1)

        layout.addLayout(left_col, stretch=1)

        # Right column
        right_col = QVBoxLayout()
        right_col.setSpacing(12)

        # --- Links ---
        links_group = QGroupBox("Resources")
        links_layout = QVBoxLayout(links_group)

        wiki_row = QHBoxLayout()
        wiki_lbl = QLabel("Online documentation")
        wiki_row.addWidget(wiki_lbl)
        wiki_row.addWidget(_InfoCircleLabel("https://github.com/ragnvald/mesa/wiki"))
        wiki_row.addStretch()
        links_layout.addLayout(wiki_row)

        linkedin_row = QHBoxLayout()
        linkedin_lbl = QLabel("Lead developer (LinkedIn)")
        linkedin_row.addWidget(linkedin_lbl)
        linkedin_row.addWidget(_InfoCircleLabel("https://www.linkedin.com/in/ragnvald/"))
        linkedin_row.addStretch()
        links_layout.addLayout(linkedin_row)

        links_layout.addStretch()
        right_col.addWidget(links_group)

        # --- System info ---
        system_group = QGroupBox("Your system")
        system_layout = QVBoxLayout(system_group)

        self._system_text = QPlainTextEdit()
        self._system_text.setReadOnly(True)
        system_layout.addWidget(self._system_text)
        right_col.addWidget(system_group, stretch=1)

        layout.addLayout(right_col, stretch=1)

        self._tabs.addTab(tab, "About")

        # Populate system info
        try:
            row = _read_system_capabilities_latest_row()
            txt = _format_system_capabilities_for_about(row)
        except Exception:
            txt = "Unable to read system profile."
        self._system_text.setPlainText(txt)

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    def _build_footer(self):
        footer = QLabel(mesa_version_display or "MESA")
        footer.setProperty("role", "footer")
        footer.setContentsMargins(0, 4, 0, 4)
        self._main_layout.addWidget(footer)


# =====================================================================
# Entry point
# =====================================================================
if __name__ == "__main__":
    import multiprocessing as _mp
    _mp.freeze_support()

    app = QApplication(sys.argv)

    # Generate checkmark / radio-dot indicator images for QSS
    _indicator_dir = os.path.join(tempfile.gettempdir(), "mesa_indicators")
    os.makedirs(_indicator_dir, exist_ok=True)
    _check_path = os.path.join(_indicator_dir, "check.png")
    _dot_path = os.path.join(_indicator_dir, "dot.png")

    from PySide6.QtGui import QPainter, QPen, QPixmap, QColor
    # Checkmark (white ✓ on transparent)
    _pm = QPixmap(16, 16)
    _pm.fill(QColor(0, 0, 0, 0))
    _p = QPainter(_pm)
    _p.setRenderHint(QPainter.Antialiasing)
    _pen = QPen(QColor("#ffffff"), 2.2)
    _pen.setCapStyle(Qt.RoundCap)
    _pen.setJoinStyle(Qt.RoundJoin)
    _p.setPen(_pen)
    _p.drawLine(3, 8, 6, 12)
    _p.drawLine(6, 12, 13, 4)
    _p.end()
    _pm.save(_check_path, "PNG")

    # Radio dot (white circle on transparent)
    _pm2 = QPixmap(16, 16)
    _pm2.fill(QColor(0, 0, 0, 0))
    _p2 = QPainter(_pm2)
    _p2.setRenderHint(QPainter.Antialiasing)
    _p2.setPen(Qt.NoPen)
    _p2.setBrush(QColor("#ffffff"))
    _p2.drawEllipse(4, 4, 8, 8)
    _p2.end()
    _pm2.save(_dot_path, "PNG")

    # Inject indicator image paths into stylesheet
    _check_url = _check_path.replace("\\", "/")
    _dot_url = _dot_path.replace("\\", "/")
    _indicator_css = f"""
QCheckBox::indicator:checked {{
    image: url("{_check_url}");
}}
QRadioButton::indicator:checked {{
    image: url("{_dot_url}");
}}
"""
    app.setStyleSheet(MESA_STYLESHEET + _indicator_css)

    # Set application-wide icon
    icon_path = resolve_path(os.path.join("system_resources", "mesa.ico"))
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    main_window = MesaMainWindow()
    main_window.show()

    sys.exit(app.exec())
