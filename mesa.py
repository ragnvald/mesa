import os
import locale
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import geopandas as gpd

# pyogrio (used by GeoPandas by default when installed) warns that measured (M)
# geometries are not supported and will be converted. Shapely/GEOS does not
# preserve M anyway, so treat this as expected and keep stdout clean.
warnings.filterwarnings(
    "ignore",
    message=r"Measured \(M\) geometry types are not supported\..*",
    category=UserWarning,
    module=r"pyogrio\..*",
)

def _patch_locale_setlocale_for_windows() -> None:
    """Make locale.setlocale resilient on Windows.

    Some Windows machines/environments raise locale.Error('unsupported locale setting')
    for calls like setlocale(LC_TIME, ""). ttkbootstrap calls setlocale during import
    (e.g. DatePickerDialog), so we must not allow this to crash the app.
    """
    try:
        if os.name != "nt":
            return
        _orig = locale.setlocale

        def _safe_setlocale(category, value=None):
            try:
                if value is None:
                    return _orig(category)
                return _orig(category, value)
            except locale.Error:
                for fallback in ("", "C"):
                    try:
                        return _orig(category, fallback)
                    except Exception:
                        continue
                try:
                    return _orig(category)
                except Exception:
                    return "C"

        locale.setlocale = _safe_setlocale  # type: ignore[assignment]
    except Exception:
        pass

# Locale is surprisingly fragile on Windows:
# - Many machines do not support POSIX locale names like "en_US.UTF-8".
# - Some environments set LANG/LC_ALL to POSIX values, and `setlocale(LC_ALL, "")`
#   can raise locale.Error("unsupported locale setting").
try:
    if os.name == "nt":
        for _k in ("LC_ALL", "LC_CTYPE", "LANG"):
            _v = os.environ.get(_k)
            if _v and ("utf-8" in _v.lower()) and ("_" in _v) and ("." in _v):
                os.environ.pop(_k, None)
except Exception:
    pass

_patch_locale_setlocale_for_windows()

try:
    locale.setlocale(locale.LC_ALL, "")
except Exception:
    try:
        locale.setlocale(locale.LC_ALL, "C")
    except Exception:
        pass

import tkinter as tk
from tkinter import *
from pathlib import Path
import subprocess
import webbrowser
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import pandas as pd
import configparser
import platform
import shutil
import json
import socket
import datetime
from datetime import datetime
import threading
import sys
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import math
import time
import struct
import ctypes
import re

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

try:
    from PIL import Image as PILImage
    from PIL import ImageTk
except Exception:
    PILImage = None
    ImageTk = None

# ---------------------------------------------------------------------
# Project/base resolution (works in dev and when frozen)
# ---------------------------------------------------------------------
def _get_project_base() -> str:
    # 1) External override
    env_dir = os.environ.get("MESA_BASE_DIR")
    if env_dir:
        return os.path.abspath(env_dir)
    # 2) Compiled exe folder
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    # 3) Folder containing this mesa.py
    return os.path.dirname(os.path.abspath(__file__))

PROJECT_BASE = _get_project_base()

# Bundled resources (PyInstaller _MEIPASS) or project folder in dev
RESOURCE_BASE = getattr(sys, "_MEIPASS", PROJECT_BASE)

# Tk root reference (assigned when UI initializes)
root = None

# ---------------------------------------------------------------------
# Path resolver
#   - Prefer disk under PROJECT_BASE (live data and overrides)
#   - Then PROJECT_BASE/code (alt layout)
#   - Then bundled resources under RESOURCE_BASE (read-only)
#   - Then RESOURCE_BASE/code
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

    # Default guess (caller handles existence)
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
# Config helpers (disk alongside EXE; auto-heal headerless INI)
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
    # If an absolute path is given, use it directly
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

# ---------------------------------------------------------------------
# Networking / utils
# ---------------------------------------------------------------------
def is_connected(hostname="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((hostname, port))
        return True
    except socket.error:
        return False

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
        # If logging fails, avoid crashing the UI
        pass

def create_link_icon(parent, url, row, col, padx, pady):
    icon_size = 20
    canvas = tk.Canvas(parent, width=icon_size, height=icon_size, bd=0, highlightthickness=0)
    canvas.grid(row=row, column=col, padx=padx, pady=pady, sticky="nsew")
    canvas.create_oval(2, 2, icon_size-2, icon_size-2, fill='white', outline='blue')
    canvas.create_text(icon_size/2, icon_size/2, text="i", font=('Calibri', 10, 'bold'), fill='blue')
    canvas.bind("<Button-1>", lambda e: webbrowser.open(url))

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
    """Locate the geoparquet folder, preferring the live data set when multiple copies exist."""
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
    """Ordered list of geoparquet folders to probe for specific tables."""
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
    """Infer the base directory whose output/geoparquet folder is currently in use."""
    try:
        gpq_path = Path(_detect_geoparquet_dir()).resolve()
        base_candidate = gpq_path.parent.parent
        if (base_candidate / "config.ini").exists():
            return str(base_candidate)
    except Exception:
        pass
    return original_working_directory

# ---------------------------------------------------------------------
# Status panel (unchanged logic, reads GeoParquet)
# ---------------------------------------------------------------------
def update_stats(_unused_gpkg_path):
    for widget in info_labelframe.winfo_children():
        widget.destroy()

    geoparquet_dir = _detect_geoparquet_dir()

    if not os.path.isdir(geoparquet_dir):
        status_label = ttk.Label(info_labelframe, text='\u26AB', bootstyle='danger')
        status_label.grid(row=0, column=0, padx=5, pady=5)
        message_label = ttk.Label(info_labelframe,
                                  text="No data imported.\nStart with importing data.",
                                  wraplength=380, justify="left")
        message_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        create_link_icon(info_labelframe, "https://github.com/ragnvald/mesa/wiki", 0, 2, 5, 5)
    else:
        my_status = get_status(geoparquet_dir)
        if not my_status.empty and {'Status', 'Message', 'Link'}.issubset(my_status.columns):
            for idx, row in my_status.iterrows():
                symbol = row['Status']
                boot = 'success' if symbol == "+" else 'warning' if symbol == "/" else 'danger'
                lbl_status = ttk.Label(info_labelframe, text='\u26AB', bootstyle=boot)
                lbl_status.grid(row=idx, column=0, padx=5, pady=5)
                lbl_msg = ttk.Label(info_labelframe, text=row['Message'], wraplength=380, justify="left")
                lbl_msg.grid(row=idx, column=1, padx=5, pady=5, sticky="w")
                create_link_icon(info_labelframe, row['Link'], idx, 2, 5, 5)
        else:
            status_label = ttk.Label(info_labelframe, text='\u26AB', bootstyle='danger')
            status_label.grid(row=0, column=0, padx=5, pady=5)
            message_label = ttk.Label(info_labelframe,
                                      text="To initiate the system please import assets.\n"
                                           "Press the Import button.",
                                      wraplength=380, justify="left")
            message_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
            create_link_icon(info_labelframe, "https://github.com/ragnvald/mesa/wiki", 0, 2, 5, 5)

    root.update_idletasks()
    root.update()

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

    def _parquet_has_rows(fp: str) -> bool:
        if not os.path.exists(fp):
            return False
        try:
            return (pq.ParquetFile(fp).metadata.num_rows or 0) > 0
        except Exception:
            try:
                return len(pd.read_parquet(fp)) > 0
            except Exception:
                return False

    def ppath(layer_name: str) -> str:
        fp = _existing_table_path(layer_name)
        if fp:
            return fp
        candidates = _table_path_candidates(layer_name)
        return candidates[0] if candidates else os.path.join(geoparquet_dir, f"{layer_name}.parquet")

    def table_exists_nonempty(layer_name: str) -> bool:
        fp = ppath(layer_name)
        return _parquet_has_rows(fp)

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
        return "-", f"You need to set up the calculation. \nPress the 'Set up'-button to proceed. ({detail})"

    def append_status(symbol, message, link):
        status_list.append({'Status': symbol, 'Message': message, 'Link': link})

    try:
        asset_group_count = read_table_and_count('tbl_asset_group')
        has_asset_group_rows = asset_group_count is not None and asset_group_count > 0
        append_status("+" if has_asset_group_rows else "-",
                      f"Asset layers imported: {asset_group_count}" if has_asset_group_rows else
                      "Assets are missing.\nUse 'Set up' to register asset groups.",
                      "https://github.com/ragnvald/mesa/wiki/User-interface#prepare-data")

        geocode_group_count = read_table_and_count('tbl_geocode_group')
        append_status("+" if geocode_group_count is not None else "/",
                      f"Geocode layers: {geocode_group_count}" if geocode_group_count is not None else
                      "Geocodes are missing.\nImport assets by pressing the Import button.",
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
    if root is None:
        log_to_logfile("UI not initialized; skipping stats refresh")
        return

    def _do_refresh():
        try:
            update_stats(gpkg_file)
        except Exception as exc:
            log_to_logfile(f"Failed to refresh stats: {exc}")

    try:
        root.after(0, _do_refresh)
    except Exception as exc:
        log_to_logfile(f"Unable to schedule stats refresh: {exc}")

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

    # On Windows, launching GUI scripts via python.exe will typically open a console.
    # Prefer pythonw.exe when available to avoid the popup.
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

    # Capture output to log.txt so failures aren't silent when using pythonw.exe
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
    """Find the first existing helper path across system/, project root, or code/."""
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
    # Fall back to the first option even if it does not exist (caller handles error)
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
# Button handlers (now pass args as separate tokens; always set cwd/env)
# ---------------------------------------------------------------------
def geocodes_grids():
    python_script, exe_file = get_script_paths("geocodes_create")
    arg_tokens = ["--original_working_directory", original_working_directory]
    if getattr(sys, "frozen", False):
        log_to_logfile(f"Running bundled exe: {exe_file}")
        run_subprocess([exe_file, *arg_tokens], [], gpkg_file)
    else:
        run_subprocess([sys.executable or "python", python_script, *arg_tokens], [exe_file, *arg_tokens], gpkg_file)

def import_assets(gpkg_file):
    python_script, exe_file = get_script_paths("data_import")
    arg_tokens = ["--original_working_directory", original_working_directory]
    if getattr(sys, "frozen", False):
        log_to_logfile(f"Running bundled exe: {exe_file}")
        run_subprocess([exe_file, *arg_tokens], [], gpkg_file)
    else:
        run_subprocess(
            [sys.executable or "python", python_script, *arg_tokens],
            [exe_file, *arg_tokens],
            gpkg_file
        )

def edit_processing_setup():
    script_candidates = [
        os.path.join("system", "parametres_setup.py"),
        "parametres_setup.py",
        os.path.join("system", "params_edit.py"),
        "params_edit.py",
    ]
    exe_candidates = [
        os.path.join("system", "parametres_setup.exe"),
        "parametres_setup.exe",
        os.path.join("system", "params_edit.exe"),
        "params_edit.exe",
    ]
    python_script = _resolve_tool_path(*script_candidates)
    exe_file = _resolve_tool_path(*exe_candidates)
    log_to_logfile(f"Processing setup script path: {python_script}")
    log_to_logfile(f"Processing setup executable path: {exe_file}")
    arg_tokens = ["--original_working_directory", original_working_directory]
    if getattr(sys, "frozen", False):
        run_subprocess([exe_file, *arg_tokens], [], gpkg_file)
    else:
        run_subprocess(
            [sys.executable or "python", python_script, *arg_tokens],
            [exe_file, *arg_tokens],
            gpkg_file
        )

def open_process_all():
    # Explicit marker so Status->Recent activity can time the full run.
    log_to_logfile("[Process] STARTED")
    python_script, exe_file = get_script_paths("process_all")
    arg_tokens = ["--original_working_directory", original_working_directory]
    if getattr(sys, "frozen", False):
        _launch_gui_process([exe_file, *arg_tokens], "process_all exe")
    else:
        python_exe = sys.executable or "python"
        _launch_gui_process([python_exe, python_script, *arg_tokens], "process_all script")

def make_atlas():
    python_script, exe_file = get_script_paths("atlas_create")
    if getattr(sys, "frozen", False):
        log_to_logfile(f"Running bundled exe: {exe_file}")
        run_subprocess([exe_file], [], gpkg_file)
    else:
        run_subprocess([sys.executable or "python", python_script], [exe_file], gpkg_file)

def open_maps_overview():
    python_script, exe_file = get_script_paths("maps_overview")
    if getattr(sys, "frozen", False):
        _launch_gui_process([exe_file], "maps_overview exe")
    else:
        python_exe = sys.executable or "python"
        _launch_gui_process([python_exe, python_script], "maps_overview script")


def open_asset_layers_viewer():
    python_script, exe_file = get_script_paths("map_assets")
    if getattr(sys, "frozen", False):
        _launch_gui_process([exe_file], "map_assets exe")
    else:
        python_exe = sys.executable or "python"
        _launch_gui_process([python_exe, python_script], "map_assets script")

def open_present_files():
    python_script, exe_file = get_script_paths("data_report")
    arg_tokens = ["--original_working_directory", original_working_directory]
    if getattr(sys, "frozen", False):
        _launch_gui_process([exe_file, *arg_tokens], "data_report exe")
    else:
        python_exe = sys.executable or "python"
        _launch_gui_process([python_exe, python_script, *arg_tokens], "data_report script")


def open_create_raster_tiles():
    python_script, exe_file = get_script_paths("create_raster_tiles")
    if getattr(sys, "frozen", False):
        log_to_logfile(f"Running bundled exe: {exe_file}")
        run_subprocess_async([exe_file], [], gpkg_file)
    else:
        run_subprocess_async([sys.executable or "python", python_script], [exe_file], gpkg_file)

def open_data_analysis_setup():
    python_script, exe_file = get_script_paths("data_analysis_setup")
    arg_tokens = ["--original_working_directory", original_working_directory]
    if getattr(sys, "frozen", False):
        log_to_logfile(f"Running bundled exe: {exe_file}")
        run_subprocess([exe_file, *arg_tokens], [], gpkg_file)
    else:
        run_subprocess([sys.executable or "python", python_script, *arg_tokens], [exe_file, *arg_tokens], gpkg_file)

def open_data_analysis_presentation():
    python_script, exe_file = get_script_paths("data_analysis_presentation")
    if getattr(sys, "frozen", False):
        _launch_gui_process([exe_file], "data_analysis_presentation exe")
    else:
        python_exe = sys.executable or "python"
        arg_tokens = ["--original_working_directory", original_working_directory]
        _launch_gui_process([python_exe, python_script, *arg_tokens], "data_analysis_presentation script")

def edit_assets():
    python_script, exe_file = get_script_paths("assetgroup_edit")
    if getattr(sys, "frozen", False):
        log_to_logfile(f"Running bundled exe: {exe_file}")
        run_subprocess([exe_file], [], gpkg_file)
    else:
        run_subprocess([sys.executable or "python", python_script], [exe_file], gpkg_file)

def edit_geocodes():
    python_script, exe_file = get_script_paths("geocodegroup_edit")
    if getattr(sys, "frozen", False):
        log_to_logfile(f"Running bundled exe: {exe_file}")
        run_subprocess([exe_file], [], gpkg_file)
    else:
        run_subprocess([sys.executable or "python", python_script], [exe_file], gpkg_file)

def edit_lines():
    python_script, exe_file = get_script_paths("lines_admin")
    chosen_base = _preferred_lines_base_dir()
    arg_tokens = ["--original_working_directory", chosen_base]
    log_to_logfile(f"Launching lines_admin with base_dir={chosen_base}")
    if getattr(sys, "frozen", False):
        log_to_logfile(f"Running bundled exe: {exe_file}")
        run_subprocess([exe_file, *arg_tokens], [], gpkg_file)
    else:
        run_subprocess(
            [sys.executable or "python", python_script, *arg_tokens],
            [exe_file, *arg_tokens],
            gpkg_file
        )


def edit_main_config():
    python_script, exe_file = get_script_paths("edit_config")
    arg_tokens = ["--original_working_directory", original_working_directory]
    if getattr(sys, "frozen", False):
        log_to_logfile(f"Running bundled exe: {exe_file}")
        run_subprocess([exe_file, *arg_tokens], [], gpkg_file)
    else:
        run_subprocess(
            [sys.executable or "python", python_script, *arg_tokens],
            [exe_file, *arg_tokens],
            gpkg_file
        )

def edit_atlas():
    python_script, exe_file = get_script_paths("atlas_edit")
    if getattr(sys, "frozen", False):
        log_to_logfile(f"Running bundled exe: {exe_file}")
        run_subprocess_async([exe_file], [], gpkg_file)
    else:
        run_subprocess_async([sys.executable or "python", python_script], [exe_file], gpkg_file)

def backup_restore_data():
    python_script, exe_file = get_script_paths("backup_restore")
    arg_tokens = ["--original_working_directory", original_working_directory]
    if getattr(sys, "frozen", False):
        _launch_gui_process([exe_file, *arg_tokens], "backup_restore exe")
    else:
        python_exe = sys.executable or "python"
        _launch_gui_process([python_exe, python_script, *arg_tokens], "backup_restore script")

def exit_program():
    root.destroy()

# ---------------------------------------------------------------------
# Config updaters (preserve layout, ensure [DEFAULT])
# ---------------------------------------------------------------------
def update_config_with_values(config_file, **kwargs):
    if os.path.isabs(config_file):
        cfg_path = config_file
    else:
        cfg_path = os.path.join(PROJECT_BASE, config_file)
    _ensure_default_header_present(cfg_path)
    if not os.path.isfile(cfg_path):
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write("[DEFAULT]\n")
    with open(cfg_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not any(line.strip().startswith('[') for line in lines):
        lines.insert(0, "[DEFAULT]\n")
    for key, value in kwargs.items():
        found = False
        key_norm = key.strip().casefold()
        for i, line in enumerate(lines):
            if "=" not in line:
                continue
            left, sep, right = line.partition("=")
            if sep == "":
                continue
            if left.strip().casefold() != key_norm:
                continue
            indent = left[: len(left) - len(left.lstrip())]
            lines[i] = f"{indent}{key} = {value}\n"
            found = True
            break
        if not found:
            lines.append(f"{key} = {value}\n")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

def add_text_to_labelframe(labelframe, text):
    label = tk.Label(labelframe, text=text, justify='left')
    label.pack(padx=10, pady=10, fill='both', expand=True)
    def update_wrap(event):
        label.config(wraplength=labelframe.winfo_width() - 20)
    labelframe.bind('<Configure>', update_wrap)


# ---------------------------------------------------------------------
# Host capability snapshot (written once to output/geoparquet)
# ---------------------------------------------------------------------
def _powershell_json(command: str) -> object | None:
    if os.name != "nt":
        return None
    try:
        # Allow environments with strict policy to disable PowerShell probing.
        if str(os.environ.get("MESA_NO_POWERSHELL", "")).strip().lower() in ("1", "true", "yes", "on"):
            return None
        flags = 0
        try:
            flags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        except Exception:
            flags = 0
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command,
            ],
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
    """Best-effort RAM info on Windows using ctypes (no admin required)."""
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
    """Run WMIC and parse /format:list output into a list of dicts."""
    if os.name != "nt":
        return None
    try:
        if str(os.environ.get("MESA_NO_WMIC", "")).strip().lower() in ("1", "true", "yes", "on"):
            return None
        flags = 0
        try:
            flags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        except Exception:
            flags = 0
        out = subprocess.check_output(
            command_args,
            stderr=subprocess.DEVNULL,
            text=True,
            creationflags=flags,
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
    """GPU info collection with layered fallbacks (PowerShell -> WMIC -> none)."""
    out: dict = {}
    # 1) PowerShell (preferred)
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

    # 2) WMIC fallback (deprecated on some systems)
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
                    # Some WMIC builds return integers as strings.
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
        "os_release": platform.release(),
        "os_version": platform.version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count() or None,
        "project_base": PROJECT_BASE,
        "working_dir": original_working_directory,
    }

    # RAM / CPU core details (best effort)
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        out["ram_total_gb"] = round(float(vm.total) / (1024 ** 3), 2)
        out["ram_available_gb"] = round(float(vm.available) / (1024 ** 3), 2)
        out["cpu_count_physical"] = psutil.cpu_count(logical=False)
        try:
            out["cpu_freq_mhz"] = getattr(psutil.cpu_freq(), "current", None)
        except Exception:
            pass
    except Exception:
        # Standard-library fallback for RAM on Windows (no psutil required)
        try:
            m = _windows_global_memory_status()
            if m:
                out.update(m)
                out.setdefault("ram_probe_method", "ctypes_globalmemorystatusex")
        except Exception:
            pass

    # Disk free (on the output drive)
    try:
        usage = shutil.disk_usage(original_working_directory)
        out["disk_total_gb"] = round(float(usage.total) / (1024 ** 3), 2)
        out["disk_free_gb"] = round(float(usage.free) / (1024 ** 3), 2)
    except Exception:
        pass

    # GPU (Windows best effort via CIM)
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

        # Write as GeoParquet (WKB geometry) without requiring geopandas/shapely.
        # Use a harmless POINT(0 0) placeholder so GeoParquet readers can load it.
        df = pd.DataFrame([row])
        try:
            wkb_point_0_0 = struct.pack("<BIdd", 1, 1, 0.0, 0.0)
        except Exception:
            wkb_point_0_0 = b""
        df["geometry"] = [wkb_point_0_0]

        import pyarrow as pa  # local import: keeps top-level import surface smaller

        table = pa.Table.from_pandas(df, preserve_index=False)
        try:
            # Minimal GeoParquet metadata (v1.0 style) so tools recognize geometry.
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
            # If metadata fails for any reason, still write a normal Parquet.
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
    """If config.ini lacks system capability fields, write a snapshot parquet once."""
    try:
        # If config already includes any system capability info, treat as "present".
        present_keys = (
            "system_os",
            "system_ram_gb",
            "system_cpu",
            "system_cores",
            "system_gpu",
            "gpu_names",
            "cpu_count_logical",
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
        # Never let this block UI startup.
        pass


def _read_system_capabilities_latest_row() -> dict | None:
    """Read the latest row from tbl_system_capabilities.parquet (best-effort)."""
    try:
        geoparquet_dir = _detect_geoparquet_dir()
        path = os.path.join(geoparquet_dir, "tbl_system_capabilities.parquet")
        if not os.path.exists(path):
            return None
        df = pd.read_parquet(path)
        if df is None or df.empty:
            return None
        row = dict(df.iloc[-1].to_dict())
        # GeoParquet placeholder geometry is not meaningful for display.
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

    # Probe methods are helpful for locked-down systems.
    if str(row.get("ram_probe_method") or "").strip():
        lines.append("")
        lines.append(f"RAM probe: {_g('ram_probe_method')}")

    return "\n".join(lines)

# ---------------------------------------------------------------------
# Main setup
# ---------------------------------------------------------------------
# The directory where data/logs should live (never _MEIPASS)
original_working_directory = PROJECT_BASE

# Config path: always sit next to mesa.py / mesa.exe
config_file = os.path.join(PROJECT_BASE, "config.ini")
if not os.path.exists(config_file):
    raise FileNotFoundError(f"Configuration not found: {config_file}")

gpkg_file = os.path.join(original_working_directory, "output", "mesa.gpkg")

config                  = read_config(config_file)
ttk_bootstrap_theme     = config['DEFAULT'].get('ttk_bootstrap_theme', 'flatly')
mesa_version            = config['DEFAULT'].get('mesa_version', 'MESA 5')
workingprojection_epsg  = config['DEFAULT'].get('working_projection_epsg', '4326')


def _format_display_version(version: str) -> str:
    """Human-friendly version string.

    We treat config.ini's mesa_version as the *release* label. During development
    runs (not frozen), append a short git SHA when available so testers can
    identify the exact code revision without having to bump config.ini.
    """
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


mesa_version_display = _format_display_version(mesa_version)

check_and_create_folders()

# ---------------------------------------------------------------------
# Tk UI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    root = ttk.Window(themename=ttk_bootstrap_theme)
    root.title(mesa_version_display or "MESA")
    try:
        root.iconbitmap(resolve_path(os.path.join("system_resources", "mesa.ico")))
    except Exception:
        pass

    TARGET_ASPECT_RATIO = 5 / 3
    DEFAULT_WIDTH = 1000
    DEFAULT_HEIGHT = int(DEFAULT_WIDTH / TARGET_ASPECT_RATIO)
    MIN_WIDTH = 900
    MIN_HEIGHT = int(MIN_WIDTH / TARGET_ASPECT_RATIO)

    root.geometry(f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}")
    root.minsize(MIN_WIDTH, MIN_HEIGHT)

    # Best-effort: write a one-time system capability snapshot to output/geoparquet
    # if config.ini does not already include such info.
    try:
        threading.Thread(
            target=_ensure_system_capabilities_snapshot,
            args=(config,),
            daemon=True,
        ).start()
    except Exception:
        pass

    # Allow free resizing; keep only minimum size constraints.

    style = ttk.Style()
    style.configure(
        "Intro.TLabel",
        background="#f3f4f6",
        foreground="#1f2937",
        font=("Segoe UI", 11, "bold")
    )

    banner_image = None
    banner_image_path = resolve_path(os.path.join("system_resources", "top_graphics.png"))
    if os.path.exists(banner_image_path):
        try:
            # Render the banner a bit narrower (e.g. ~90%) to avoid right-side cutoff.
            banner_scale = 0.90
            if PILImage is not None and ImageTk is not None:
                pil_img = PILImage.open(banner_image_path)
                src_w, src_h = pil_img.size
                dst_w = max(1, int(round(src_w * banner_scale)))
                dst_h = max(1, int(round(src_h * banner_scale)))
                if (dst_w, dst_h) != (src_w, src_h):
                    resample = getattr(PILImage, "Resampling", PILImage).LANCZOS
                    pil_img = pil_img.resize((dst_w, dst_h), resample=resample)
                banner_image = ImageTk.PhotoImage(pil_img)
            else:
                banner_image = tk.PhotoImage(file=banner_image_path)
            root._header_banner_image = banner_image  # stash reference to avoid GC
        except Exception as exc:
            banner_image = None
            log_to_logfile(f"Unable to load header graphic: {exc}")

    header = ttk.Frame(root, padding=0)
    header.pack(fill="x", padx=0, pady=(6, 8))

    banner_host = ttk.Frame(header, padding=0)
    banner_host.pack(side="left", fill="x", expand=True, padx=0, pady=0)

    banner_wrap = 760
    if banner_image is not None:
        text_y_nudge = -15
        img_w = 800
        img_h = 80
        try:
            img_w = int(banner_image.width())
            img_h = int(banner_image.height())
        except Exception:
            pass

        banner_canvas = tk.Canvas(
            banner_host,
            height=max(1, img_h),
            bd=0,
            highlightthickness=0,
            relief="flat",
            bg="#f3f4f6"
        )
        banner_canvas.pack(fill="x", expand=True)
        banner_canvas.create_image(0, 0, image=banner_image, anchor="nw")

        # Title left (slightly larger)
        banner_canvas.create_text(
            24,
            max(14, img_h // 2 - 2 + text_y_nudge),
            anchor="w",
            text="MESA tool",
            fill="#0f172a",
            font=("Segoe UI", 14, "bold")
        )

        # Version right
        banner_canvas.create_text(
            max(120, img_w - 100),
            max(12, img_h // 2 + 14 + text_y_nudge),
            anchor="e",
            text="Version " + (mesa_version_display or "unknown"),
            fill="#0f172a",
            font=("Segoe UI", 9, "italic")
        )
    else:
        intro_text = "MESA tool  ·  " + (mesa_version_display or "unknown")
        intro_label = ttk.Label(
            banner_host,
            text=intro_text,
            wraplength=banner_wrap,
            justify="left",
            padding=(14, 10),
            style="Intro.TLabel"
        )
        intro_label.pack(side="left", fill="x", expand=True, padx=(12, 0))

    ttk.Button(
        header,
        text="Exit",
        command=root.destroy,
        bootstyle="danger-outline",
        width=20
    ).pack(side="right", padx=(24, 16), pady=(6, 0))

    notebook = ttk.Notebook(root, bootstyle=SECONDARY)
    notebook.pack(fill="both", expand=True, padx=12, pady=(0, 10))

    # ------------------------------------------------------------------
    # Workflows tab
    # ------------------------------------------------------------------
    workflows_tab = ttk.Frame(notebook)
    notebook.add(workflows_tab, text="Workflows")

    workflows_container = ttk.Frame(workflows_tab, padding=12)
    workflows_container.pack(fill="both", expand=True)

    ttk.Label(
        workflows_container,
        text="Launch the workflows grouped by phase. Pick the task that matches what you are trying to achieve, "
             "then glance at the Status tab to confirm progress and find project statistics.",
        justify="left",
        wraplength=780
    ).pack(anchor="w", pady=(0, 10))

    workflow_grid = ttk.Frame(workflows_container)
    workflow_grid.pack(fill="both", expand=True)
    for col_idx in range(4):
        workflow_grid.columnconfigure(col_idx, weight=1)
    workflow_grid.rowconfigure(0, weight=1)

    workflow_section_frames = []
    workflow_wrap_targets: list[tuple[ttk.Frame, list[ttk.Label]]] = []
    ACTION_COLUMNS = 1

    workflow_sections = [
        ("Prepare data (step 1)", "Import new data and generate supporting geometries.", [
            ("Area assets", lambda: import_assets(gpkg_file),
             "Start here to import area asset (wetlands, mangrove forests etc)."),
            ("Geocodes", geocodes_grids,
             "Create or refresh the hexagon/tile grids that support analysis."),
            ("Line assets", edit_lines,
             "Import and edit line assets (transport, rivers, utilities, etc)."),
            ("Atlas", make_atlas,
             "Generate atlas polygons used in the QGIS atlas and the report engine."),
        ]),
        ("Configure processing (step 2)", "Tune processing parameters/study areas before running heavy jobs.", [
            ("Area parameters", edit_processing_setup,
             "Adjust weights, thresholds and other processing rules."),
            ("Analysis design", open_data_analysis_setup,
             "Define analysis groups and study area polygons."),
            ("Edit assets", edit_assets,
             "Add titles to imported layers, plus a short descriptive text."),
            ("Edit atlas", edit_atlas,
             "Edit atlas tile titles/metadata after creating/importing atlas tiles."),
        ]),
        ("Run processing (step 3)", "Execute the automated steps that build fresh outputs.", [
            ("Process", open_process_all,
             "Runs area, line, and analysis processing."),
        ]),
        ("Review & publish (step 4)", "Open the interactive viewers and export the deliverables.", [
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

    for idx, (section_title, section_description, actions) in enumerate(workflow_sections):
        row = 0
        col = idx
        section_frame = ttk.LabelFrame(workflow_grid, text=section_title, padding=(12, 10))
        section_frame.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
        workflow_section_frames.append(section_frame)
        section_desc_label = ttk.Label(
            section_frame,
            text=section_description,
            wraplength=240,
            justify="left",
            font=("Segoe UI", 9, "bold")
        )
        section_desc_label.pack(anchor="w", fill="x", pady=(0, 6))
        actions_container = ttk.Frame(section_frame)
        actions_container.pack(fill="x", pady=(16, 0))
        for col_index in range(ACTION_COLUMNS):
            actions_container.columnconfigure(col_index, weight=1)
        action_desc_labels: list[ttk.Label] = []
        for action_index, (action_label, action_command, action_description) in enumerate(actions):
            col_idx = action_index % ACTION_COLUMNS
            row_idx = action_index // ACTION_COLUMNS
            action_block = ttk.Frame(actions_container)
            action_block.grid(row=row_idx, column=col_idx, padx=4, pady=(10, 4), sticky="nsew")
            ttk.Button(
                action_block,
                text=action_label,
                command=action_command,
                width=20
            ).pack(anchor="center")
            action_desc = ttk.Label(
                action_block,
                text=action_description,
                wraplength=240,
                justify="left"
            )
            action_desc.pack(anchor="w", fill="x", pady=(2, 0))
            action_desc_labels.append(action_desc)
        workflow_wrap_targets.append((section_frame, [section_desc_label] + action_desc_labels))

    _wrap_after_id = {"id": None}
    _wrap_cache: dict[int, int] = {}

    def _apply_workflow_wraps() -> None:
        _wrap_after_id["id"] = None
        for frame, labels in workflow_wrap_targets:
            try:
                width = max(220, frame.winfo_width() - 24)
            except Exception:
                width = 240
            key = id(frame)
            if _wrap_cache.get(key) == width:
                continue
            _wrap_cache[key] = width
            for lbl in labels:
                try:
                    lbl.configure(wraplength=width)
                except Exception:
                    pass

    def _schedule_workflow_wraps(_event=None) -> None:
        if _wrap_after_id["id"] is not None:
            try:
                workflows_container.after_cancel(_wrap_after_id["id"])
            except Exception:
                pass
        _wrap_after_id["id"] = workflows_container.after(120, _apply_workflow_wraps)

    workflows_container.bind("<Configure>", _schedule_workflow_wraps)
    _schedule_workflow_wraps()

    def resize_window_to_fit_contents():
        root.update_idletasks()
        required_height = root.winfo_reqheight()
        required_width = root.winfo_reqwidth()
        width_needed = max(DEFAULT_WIDTH, required_width)
        height_needed = max(DEFAULT_HEIGHT, required_height)
        current_width = root.winfo_width()
        current_height = root.winfo_height()
        if width_needed <= current_width and height_needed <= current_height:
            return
        root.geometry(f"{width_needed}x{height_needed}")

    root.update_idletasks()
    resize_window_to_fit_contents()

    # ------------------------------------------------------------------
    # Status tab
    # ------------------------------------------------------------------
    stats_tab = ttk.Frame(notebook)
    notebook.add(stats_tab, text="Status")

    stats_container = ttk.Frame(stats_tab, padding=12)
    stats_container.pack(fill="both", expand=True)

    ttk.Label(
        stats_container,
        text="Get on top of your project with key metrics statistics.",
        justify="left"
    ).pack(anchor="w", pady=(0, 8))

    status_columns = ttk.Frame(stats_container)
    status_columns.pack(fill="both", expand=True, pady=(0, 12))
    status_columns.columnconfigure(0, weight=1)
    status_columns.columnconfigure(1, weight=1)
    status_columns.rowconfigure(0, weight=1)

    global info_labelframe
    info_labelframe = ttk.LabelFrame(status_columns, text="Status and help", bootstyle='info')
    info_labelframe.grid(row=0, column=0, padx=(0, 8), pady=(0, 8), sticky="nsew")
    info_labelframe.grid_columnconfigure(0, weight=1)
    info_labelframe.grid_columnconfigure(1, weight=3)
    info_labelframe.grid_columnconfigure(2, weight=2)

    timeline_frame = ttk.LabelFrame(status_columns, text="Recent activity", bootstyle="secondary")
    timeline_frame.grid(row=0, column=1, padx=(8, 0), pady=(0, 8), sticky="nsew")
    timeline_canvas = ttk.Frame(timeline_frame)
    timeline_canvas.pack(fill="both", expand=True, padx=10, pady=8)
    timeline_entries = []
    for _ in range(6):
        entry = ttk.Frame(timeline_canvas)
        entry.pack(fill="x", pady=4)
        color_bar = ttk.Label(entry, text=" ", width=2, bootstyle="success")
        color_bar.grid(row=0, column=0, padx=(0, 6), sticky="ns")
        title_label = ttk.Label(entry, text="Event", justify="left")
        title_label.grid(row=0, column=1, sticky="w")
        duration_label = ttk.Label(entry, text="--", width=12, anchor="e")
        duration_label.grid(row=0, column=2, padx=(6, 0), sticky="e")
        time_label = ttk.Label(entry, text="--", width=18, anchor="e")
        time_label.grid(row=0, column=3, padx=(8, 0), sticky="e")
        entry.columnconfigure(1, weight=1)
        timeline_entries.append((color_bar, title_label, duration_label, time_label))

    # insight boxes container (below Status and help)
    insights_frame = ttk.Frame(stats_container)
    insights_frame.pack(fill="x", pady=(4, 8))
    insights_frame.columnconfigure((0, 1, 2, 3), weight=1)

    def _make_two_col_table(parent, header_left: str, header_right: str, rows: int):
        table = ttk.Frame(parent)
        table.pack(fill="x", padx=10, pady=8)
        table.columnconfigure(0, weight=1)
        table.columnconfigure(1, weight=0)

        hdr_l = ttk.Label(table, text=header_left, font=("Segoe UI", 9, "bold"), justify="left")
        hdr_r = ttk.Label(table, text=header_right, font=("Segoe UI", 9, "bold"), anchor="e", justify="right")
        hdr_l.grid(row=0, column=0, sticky="w")
        hdr_r.grid(row=0, column=1, sticky="e")

        cells = []
        for idx in range(rows):
            k = ttk.Label(table, text="--", justify="left", wraplength=180)
            v = ttk.Label(table, text="", anchor="e", justify="right", width=10)
            k.grid(row=idx + 1, column=0, sticky="w", pady=1)
            v.grid(row=idx + 1, column=1, sticky="e", pady=1)
            cells.append((k, v))
        return cells

    def _populate_two_col_table(cells, rows: list[tuple[str, str]]):
        for i, (k_lbl, v_lbl) in enumerate(cells):
            if i < len(rows):
                k, v = rows[i]
                k_lbl.config(text=k)
                v_lbl.config(text=v)
            else:
                k_lbl.config(text="")
                v_lbl.config(text="")

    geocode_box = ttk.LabelFrame(insights_frame, text="Objects per geocode", bootstyle="secondary")
    geocode_box.grid(row=0, column=0, padx=5, sticky="nsew")
    geocode_box.columnconfigure(0, weight=1)
    geocode_table = _make_two_col_table(geocode_box, "Geocode", "Objects", rows=6)

    assets_box = ttk.LabelFrame(insights_frame, text="Assets overview", bootstyle="secondary")
    assets_box.grid(row=0, column=1, padx=5, sticky="nsew")
    assets_table = _make_two_col_table(assets_box, "Metric", "Value", rows=4)

    lines_box = ttk.LabelFrame(insights_frame, text="Lines & segments", bootstyle="secondary")
    lines_box.grid(row=0, column=2, padx=5, sticky="nsew")
    lines_table = _make_two_col_table(lines_box, "Metric", "Value", rows=4)

    analysis_box = ttk.LabelFrame(insights_frame, text="Results metrics", bootstyle="secondary")
    analysis_box.grid(row=0, column=3, padx=5, sticky="nsew")
    analysis_table = _make_two_col_table(analysis_box, "Metric", "Count", rows=2)

    def _fmt_timestamp(ts: float | None) -> str:
        if not ts:
            return "--"
        try:
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return "--"

    def _path_mtime(path: str) -> float | None:
        try:
            return os.path.getmtime(path)
        except Exception:
            return None

    _log_duration_cache = {
        "mtime": None,
        "durations": {},
        "seconds": {},
        "times": {},
    }

    _status_calc_runtime = {
        "seconds": None,
    }

    def _parse_log_timestamp(line: str) -> datetime | None:
        """Parse timestamps like 'YYYY.MM.DD HH:MM:SS' at start of log.txt lines."""
        if not line or len(line) < 19:
            return None
        ts = line[:19]
        try:
            return datetime.strptime(ts, "%Y.%m.%d %H:%M:%S")
        except Exception:
            return None

    def _scan_last_run_from_log(
                                log_path: str,
                                start_markers: list[str],
                                end_markers_primary: list[str],
                                end_markers_secondary: list[str] | None = None,
                                ) -> tuple[float | None, datetime | None]:
        """Return (duration_seconds, end_timestamp) for the most recent completed run."""
        try:
            if not os.path.exists(log_path):
                return None, None
        except Exception:
            return None, None

        current_start: datetime | None = None
        secondary_end: datetime | None = None
        last_duration: float | None = None
        last_end: datetime | None = None

        def _has_any(haystack: str, needles: list[str]) -> bool:
            return any(n in haystack for n in needles)

        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                for raw in f:
                    line = raw.rstrip("\n")
                    ts = _parse_log_timestamp(line)
                    if ts is None:
                        continue

                    if current_start is None:
                        if _has_any(line, start_markers):
                            current_start = ts
                            secondary_end = None
                        continue

                    # run is active
                    # If we encounter a new start marker while a run is active, it means the
                    # previous run likely ended without a primary end marker (e.g., UI closed,
                    # crash, or run stopped early). In that case, finalize the previous run
                    # using the best available secondary end marker (if any), then begin a new run.
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

            # If we ended the file mid-run but have a secondary end marker, use it.
            if current_start is not None and secondary_end is not None:
                last_duration = (secondary_end - current_start).total_seconds()
                last_end = secondary_end
        except Exception:
            return None, None

        return last_duration, last_end

    def _scan_last_duration_from_log(log_path: str,
                                    start_markers: list[str],
                                    end_markers_primary: list[str],
                                    end_markers_secondary: list[str] | None = None) -> float | None:
        """Return duration (seconds) for the most recent completed run.

        Uses a simple state-machine:
        - start markers begin a run
        - primary end markers end a run immediately
        - secondary end markers are remembered and used only if no primary end is found
          before the next run begins or the file ends.
        """
        duration, _end = _scan_last_run_from_log(
            log_path,
            start_markers=start_markers,
            end_markers_primary=end_markers_primary,
            end_markers_secondary=end_markers_secondary,
        )
        return duration

    def _fmt_duration(seconds: float | None) -> str:
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

    def _fmt_stats_runtime(seconds: float | None) -> str:
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
        return _fmt_duration(s)

    def _recent_activity_durations() -> dict[str, str]:
        """Duration strings for the Status -> Recent activity items."""
        log_path = os.path.join(original_working_directory, "log.txt")
        try:
            mtime = os.path.getmtime(log_path)
        except Exception:
            mtime = None

        if mtime and _log_duration_cache.get("mtime") == mtime:
            return _log_duration_cache.get("durations", {})

        seconds: dict[str, float | None] = {}
        times: dict[str, str] = {}
        seconds["Import assets"] = _scan_last_duration_from_log(
            log_path,
            start_markers=["Step [Assets] STARTED"],
            end_markers_primary=["Step [Assets] COMPLETED", "Step [Assets] FAILED"],
        )

        # Mosaic geocode group build (basic_mosaic)
        mosaic_secs, mosaic_end = _scan_last_run_from_log(
            log_path,
            start_markers=["Step [Mosaic] STARTED"],
            end_markers_primary=["Step [Mosaic] COMPLETED", "Step [Mosaic] FAILED"],
        )
        seconds["Build basic_mosaic"] = mosaic_secs
        times["Build basic_mosaic"] = mosaic_end.strftime("%Y-%m-%d %H:%M") if mosaic_end else "--"

        seconds["Processing"] = _scan_last_duration_from_log(
            log_path,
            start_markers=[
                "[Process] STARTED",
                # Backward-compatible fallbacks for older logs
                "Attempting to run command:",
                "[Stage 1/4] Preparing workspace",
            ],
            end_markers_primary=[
                "[Process] COMPLETED",
                # Full pipeline completion
                "[Tiles] Completed.",
                # Explicit tiles failure/skip markers (still ends the overall attempt)
                "[Tiles] Skipping MBTiles stage because processing exited with code",
                "[Tiles] tbl_flat not present or empty; skipping MBTiles generation.",
                "[Tiles] create_raster_tiles exited with code",
                "[Tiles] Error:",
                # Core failure marker
                "Error during processing:",
            ],
            end_markers_secondary=None,
        )

        seconds["Line processing"] = _scan_last_duration_from_log(
            log_path,
            start_markers=["SEGMENT PROCESS START"],
            end_markers_primary=["COMPLETED: Segment processing", "FAILED: Segment processing"],
        )

        seconds["Newest report export"] = _scan_last_duration_from_log(
            log_path,
            start_markers=["Report mode selected:"],
            end_markers_primary=["Word report created:", "ERROR during report generation:"],
        )

        durations: dict[str, str] = {k: _fmt_duration(v) for k, v in seconds.items()}

        _log_duration_cache["mtime"] = mtime
        _log_duration_cache["seconds"] = seconds
        _log_duration_cache["durations"] = durations
        _log_duration_cache["times"] = times
        return durations

    def _last_flat_timestamp():
        geoparquet_dir = _detect_geoparquet_dir()
        flat_path = os.path.join(geoparquet_dir, "tbl_flat.parquet")
        ts = _path_mtime(flat_path)
        if ts:
            return _fmt_timestamp(ts)
        return config['DEFAULT'].get('last_process_run', '--')

    def _last_asset_import_timestamp():
        geoparquet_dir = _detect_geoparquet_dir()
        candidates = [
            os.path.join(geoparquet_dir, "tbl_asset_group.parquet"),
            os.path.join(geoparquet_dir, "tbl_asset_object.parquet"),
        ]
        newest = None
        for p in candidates:
            ts = _path_mtime(p)
            if ts and (newest is None or ts > newest):
                newest = ts
        return _fmt_timestamp(newest) if newest else "--"

    def _last_line_processing_timestamp():
        geoparquet_dir = _detect_geoparquet_dir()
        segment_path = os.path.join(geoparquet_dir, "tbl_segment_flat.parquet")
        ts = _path_mtime(segment_path)
        if ts:
            return _fmt_timestamp(ts)
        return config['DEFAULT'].get('last_lines_process_run', '--')

    def _latest_report_timestamp():
        output_dir = os.path.join(original_working_directory, "output")
        reports_dir = os.path.join(output_dir, "reports")

        newest_ts = None

        def _consider_file(path: str):
            nonlocal newest_ts
            ts = _path_mtime(path)
            if ts and (newest_ts is None or ts > newest_ts):
                newest_ts = ts

        def _scan_dir(dir_path: str, recursive: bool):
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

        # Prefer the dedicated reports folder (and any nested subfolders)
        _scan_dir(reports_dir, recursive=True)
        # Fall back to the output root (non-recursive) for legacy exports
        _scan_dir(output_dir, recursive=False)

        if newest_ts:
            return _fmt_timestamp(newest_ts)
        return config['DEFAULT'].get('last_report_export', '--')

    def update_timeline():
        durations = dict(_recent_activity_durations())
        durations["Time to calculate stats on this page"] = _fmt_stats_runtime(_status_calc_runtime.get("seconds"))
        times = _log_duration_cache.get("times", {})
        events = [
            ("Import assets", _last_asset_import_timestamp(), 'success'),
            ("Build basic_mosaic", times.get("Build basic_mosaic", "--"), 'info'),
            ("Processing", _last_flat_timestamp(), 'info'),
            ("Line processing", _last_line_processing_timestamp(), 'warning'),
            ("Newest report export", _latest_report_timestamp(), 'secondary'),
            ("Time to calculate stats on this page", "", 'secondary'),
        ]
        for idx, (title, timestamp, bootstyle) in enumerate(events):
            if idx >= len(timeline_entries):
                break
            color_bar, title_label, duration_label, time_label = timeline_entries[idx]
            color_bar.config(bootstyle=bootstyle)
            title_label.config(text=title)
            duration_label.config(text=durations.get(title, "--"))
            time_label.config(text=timestamp)

    def fetch_geocode_objects_summary() -> list[tuple[str, str]]:
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
            rows: list[tuple[str, str]] = []
            for idx, val in counts.items():
                key = "(missing)" if pd.isna(idx) else str(idx)
                try:
                    value = str(int(val))
                except Exception:
                    value = str(val)
                rows.append((key, value))
            if data.nunique(dropna=False) > 5:
                rows.append(("…", ""))
            return rows if rows else [("No records found.", "")]
        except Exception as exc:
            return [(f"Unable to read tbl_flat:", ""), (str(exc)[:160], "")]

    def _fmt_count(value: int | float | None) -> str:
        if value is None:
            return "--"
        try:
            return f"{int(value):,}"
        except Exception:
            try:
                return f"{float(value):,.0f}"
            except Exception:
                return "--"

    def _fmt_km2(value: float | None) -> str:
        if value is None:
            return "--"
        try:
            return f"{float(value):,.1f}"
        except Exception:
            return "--"

    def _fmt_km(value: float | None) -> str:
        if value is None:
            return "--"
        try:
            return f"{float(value):,.1f}"
        except Exception:
            return "--"

    def _measurement_epsg() -> int | None:
        try:
            # Prefer the explicit metric projection used for area/distance calculations.
            raw_area = (config["DEFAULT"].get("area_projection_epsg", "") or "").strip()
            if raw_area:
                return int(float(raw_area))

            raw_working = (config["DEFAULT"].get("working_projection_epsg", "") or "").strip()
            if raw_working:
                epsg = int(float(raw_working))
                # If working CRS is WGS84/latlon, it's not suitable for meters.
                if epsg in (4326, 4258):
                    return None
                return epsg

            return None
        except Exception:
            return None

    def _fallback_metric_epsg_for_wgs84(gdf: "gpd.GeoDataFrame") -> int | None:
        # A pragmatic fallback when config doesn't provide a metric EPSG.
        # EPSG:3857 is global and meter-based (approx). Only used if needed.
        try:
            if gdf.crs is None:
                return 3857
            epsg = getattr(gdf.crs, "to_epsg", lambda: None)()
            if epsg in (4326, 4258, None):
                return 3857
            return None
        except Exception:
            return 3857

    _asset_area_cache = {"path": None, "mtime": None, "area_km2": None}
    _lines_length_cache = {"path": None, "mtime": None, "length_km": None}

    def _total_area_km2_from_asset_objects(asset_object_path: str | None) -> float | None:
        if not asset_object_path or not os.path.exists(asset_object_path):
            return None
        try:
            import geopandas as gpd  # Lazy import to avoid loading GIS stack at startup
            mtime = _path_mtime(asset_object_path)
            if (
                mtime
                and _asset_area_cache.get("path") == asset_object_path
                and _asset_area_cache.get("mtime") == mtime
            ):
                return _asset_area_cache.get("area_km2")

            epsg = _measurement_epsg()
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
                fallback_epsg = _fallback_metric_epsg_for_wgs84(gdf)
                if fallback_epsg:
                    try:
                        gdf = gdf.to_crs(epsg=fallback_epsg)
                    except Exception:
                        pass
            total_m2 = float(gdf.geometry.area.fillna(0).sum()) if not gdf.empty else 0.0
            km2 = total_m2 / 1_000_000.0

            _asset_area_cache.update({"path": asset_object_path, "mtime": mtime, "area_km2": km2})
            return km2
        except Exception:
            return None

    def _total_length_km_from_lines(lines_path: str | None) -> float | None:
        if not lines_path or not os.path.exists(lines_path):
            return None
        try:
            import geopandas as gpd  # Lazy import to avoid loading GIS stack at startup
            mtime = _path_mtime(lines_path)
            if (
                mtime
                and _lines_length_cache.get("path") == lines_path
                and _lines_length_cache.get("mtime") == mtime
            ):
                return _lines_length_cache.get("length_km")

            epsg = _measurement_epsg()
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
                fallback_epsg = _fallback_metric_epsg_for_wgs84(gdf)
                if fallback_epsg:
                    try:
                        gdf = gdf.to_crs(epsg=fallback_epsg)
                    except Exception:
                        pass
            total_m = float(gdf.geometry.length.fillna(0).sum()) if not gdf.empty else 0.0
            km = total_m / 1000.0

            _lines_length_cache.update({"path": lines_path, "mtime": mtime, "length_km": km})
            return km
        except Exception:
            return None

    def fetch_asset_summary() -> list[tuple[str, str]]:
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
            total_area_km2 = _total_area_km2_from_asset_objects(asset_object_path)

            return [
                ("Layers", _fmt_count(layers)),
                ("Objects", _fmt_count(objects)),
                ("Area (km²)", _fmt_km2(total_area_km2)),
            ]
        except Exception as exc:
            return [("Unable to read assets:", ""), (str(exc)[:160], "")]

    def fetch_lines_summary() -> list[tuple[str, str]]:
        lines_path = _locate_geoparquet_file("tbl_lines")
        if not lines_path:
            return [("Lines not imported yet.", "")]
        segments_path = _locate_geoparquet_file("tbl_segment_flat")
        try:
            lines_count = _parquet_row_count(lines_path)
            segments_count = _parquet_row_count(segments_path) if segments_path and os.path.exists(segments_path) else 0
            total_length_km = _total_length_km_from_lines(lines_path)
            return [
                ("Lines", _fmt_count(lines_count)),
                ("Segments", _fmt_count(segments_count)),
                ("Length (km)", _fmt_km(total_length_km)),
            ]
        except Exception as exc:
            return [("Unable to read lines:", ""), (str(exc)[:160], "")]

    def fetch_analysis_summary() -> list[tuple[str, str]]:
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

    def update_insight_boxes():
        _populate_two_col_table(geocode_table, fetch_geocode_objects_summary())
        _populate_two_col_table(assets_table, fetch_asset_summary())
        _populate_two_col_table(lines_table, fetch_lines_summary())
        _populate_two_col_table(analysis_table, fetch_analysis_summary())

    def _on_notebook_tab_changed(_event=None):
        """Refresh Status tab when it is opened/selected."""
        try:
            current_tab = notebook.select()
            if current_tab and notebook.nametowidget(current_tab) is stats_tab:
                started = time.perf_counter()
                update_stats(gpkg_file)
                update_insight_boxes()
                _status_calc_runtime["seconds"] = time.perf_counter() - started
                update_timeline()
        except Exception:
            pass

    notebook.bind("<<NotebookTabChanged>>", _on_notebook_tab_changed)

    # Avoid doing potentially heavy IO (Parquet reads + log scanning) during startup.
    # Refresh is performed when the Status tab is selected.
    try:
        root.after(0, _on_notebook_tab_changed)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Settings tab
    # ------------------------------------------------------------------
    settings_tab = ttk.Frame(notebook)
    notebook.add(settings_tab, text="Settings")

    settings_container = ttk.Frame(settings_tab, padding=12)
    settings_container.pack(fill="both", expand=True)

    mesa_text2 = (
        "Some of the objects you already have imported or created might need some further adjustments.\n"
        "You may do this by reading up on the below suggestions."
    )
    ttk.Label(
        settings_container,
        text=mesa_text2,
        wraplength=660,
        justify="left"
    ).pack(anchor="w", pady=(0, 10))

    settings_grid = ttk.Frame(settings_container, padding=(10, 10))
    settings_grid.pack(fill="both", expand=True)
    settings_grid.columnconfigure(1, weight=1)

    settings_actions = [
        ("Edit config", edit_main_config,
         "Open the config.ini editor to review or adjust global settings."),
           ("Backup / restore", backup_restore_data,
            "Create a ZIP backup of input/, output/ and config.ini, or restore from a previous backup."),
        ("Edit geocodes", edit_geocodes,
         "Geocodes can be grid cells, hexagons or other polygons. Add titles to them here for easier reference later."),
    ]

    for row, (label, command, description) in enumerate(settings_actions):
        ttk.Button(
            settings_grid,
            text=label,
            command=command,
            bootstyle="primary",
            width=18
        ).grid(row=row, column=0, padx=5, pady=4, sticky="ew")
        ttk.Label(
            settings_grid,
            text=description,
            wraplength=500,
            justify="left"
        ).grid(row=row, column=1, padx=5, pady=4, sticky="w")

    # ------------------------------------------------------------------
    # About tab
    # ------------------------------------------------------------------
    about_tab = ttk.Frame(notebook)
    notebook.add(about_tab, text="About")

    about_container = ttk.Frame(about_tab, padding=12)
    about_container.pack(fill="both", expand=True)

    # Split view: About MESA (left) + About your system (right)
    about_grid = ttk.Frame(about_container)
    about_grid.pack(fill="both", expand=True)
    # Left pane should have more space than the system pane.
    # Use a uniform group so requested widget widths don't overpower the split.
    about_grid.columnconfigure(0, weight=8, uniform="about")
    about_grid.columnconfigure(1, weight=1, uniform="about", minsize=320)
    about_grid.rowconfigure(0, weight=1)

    about_box = ttk.LabelFrame(about_grid, text="About MESA", bootstyle='secondary')
    about_box.grid(row=0, column=0, sticky="nsew", padx=(5, 6), pady=(0, 10))
    about_box.columnconfigure(0, weight=1)
    about_text = (
        "Welcome to the MESA tool. The method is developed by UNEP-WCMC and the Norwegian Environment Agency. "
        "The tool is a reflection of ongoing methods development done by UNEP-WCMC and NEA. "
        "The software streamlines sensitivity analysis, reducing the likelihood of manual errors in GIS workflows.\n\n"
        "The project is Open Source. Although the software can be used (without guarantees), further work is pending, "
        "including final conclusions on the methods development (to be subject to hearings with stakeholders), bug fixing, "
        "and continued discussions and workshops.\n\n"
        "Timeline: the aim is to conclude the work in May 2025 at a workshop in Nairobi. Broad inclusion is planned, "
        "so not only stakeholders are welcome.\n\n"
        "Documentation and user guides are available on the MESA wiki.\n\n"
        "This release incorporates feedback from workshops with partners in Ghana, Tanzania, Uganda, Mozambique, "
        "and Kenya. Lead programmer: Ragnvald Larsen."
    )
    about_label = ttk.Label(
        about_box,
        text=about_text,
        wraplength=700,
        justify="left"
    )
    about_label.pack(fill='x', expand=False, padx=10, pady=(10, 6))

    # Keep wrapping responsive so text doesn't look cut off when the pane is narrow.
    def _update_about_wrap(_event=None):
        try:
            w = max(300, about_box.winfo_width() - 40)
            about_label.configure(wraplength=w)
        except Exception:
            pass

    about_box.bind("<Configure>", _update_about_wrap)
    _update_about_wrap()

    links_frame = ttk.Frame(about_box)
    links_frame.pack(fill='x', expand=False, padx=10, pady=(0, 10))
    links_frame.columnconfigure(0, weight=1)

    ttk.Label(links_frame, text="Open MESA wiki:", justify="left").grid(row=0, column=0, sticky="w")
    create_link_icon(links_frame, "https://github.com/ragnvald/mesa/wiki", 0, 1, 6, 0)

    ttk.Label(links_frame, text="Lead programmer (LinkedIn):", justify="left").grid(row=1, column=0, sticky="w", pady=(6, 0))
    create_link_icon(links_frame, "https://www.linkedin.com/in/ragnvald/", 1, 1, 6, 6)

    system_box = ttk.LabelFrame(about_grid, text="About your system", bootstyle='secondary')
    system_box.grid(row=0, column=1, sticky="nsew", padx=(6, 5), pady=(0, 10))
    system_box.columnconfigure(0, weight=1)
    system_box.rowconfigure(0, weight=1)

    system_text_frame = ttk.Frame(system_box)
    system_text_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 10))
    system_text_frame.columnconfigure(0, weight=1)
    system_text_frame.rowconfigure(0, weight=1)

    system_text = tk.Text(system_text_frame, wrap="word", height=18, width=36, borderwidth=0)
    system_scroll = ttk.Scrollbar(system_text_frame, orient="vertical", command=system_text.yview)
    system_text.configure(yscrollcommand=system_scroll.set)
    system_text.grid(row=0, column=0, sticky="nsew")
    system_scroll.grid(row=0, column=1, sticky="ns")

    def _refresh_system_about_text():
        try:
            row = _read_system_capabilities_latest_row()
            txt = _format_system_capabilities_for_about(row)
        except Exception:
            txt = "Unable to read system profile."
        system_text.configure(state="normal")
        system_text.delete("1.0", "end")
        system_text.insert("1.0", txt)
        system_text.configure(state="disabled")

    # Populate once on startup.
    _refresh_system_about_text()

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    footer = ttk.Frame(root, padding=(10, 5))
    footer.pack(fill='x', padx=12, pady=(0, 6))

    ttk.Label(footer, text=mesa_version, font=("Calibri", 8)).pack(side='left')

    notebook.select(0)
    root.update_idletasks()
    root.mainloop()
