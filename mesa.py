import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

import tkinter as tk
from tkinter import *
import os
from pathlib import Path
import subprocess
import webbrowser
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import pandas as pd
import geopandas as gpd
import configparser
import sqlite3
import socket
import uuid
import datetime
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient, Point, WriteOptions
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import threading
import sys
from shapely import wkb
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import math

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
                      "https://github.com/ragnvald/mesa/wiki/3-User-interface#assets")

        geocode_group_count = read_table_and_count('tbl_geocode_group')
        append_status("+" if geocode_group_count is not None else "/",
                      f"Geocode layers: {geocode_group_count}" if geocode_group_count is not None else
                      "Geocodes are missing.\nImport assets by pressing the Import button.",
                      "https://github.com/ragnvald/mesa/wiki/3-User-interface#geocodes")

        lines_original_count = read_table_and_count('tbl_lines_original')
        append_status("+" if lines_original_count is not None else "/",
                      f"Lines: {lines_original_count}" if lines_original_count is not None else
                      "Lines are missing.\nImport or initiate lines if you want to use\nthe line feature.",
                      "https://github.com/ragnvald/mesa/wiki/3-User-interface#lines")

        symbol, message = read_setup_status()
        append_status(symbol, message, "https://github.com/ragnvald/mesa/wiki/3-User-interface#setting-up-parameters")

        flat_original_count = read_table_and_count('tbl_flat')
        append_status("+" if flat_original_count is not None else "-",
                      "Processing completed. You may choose to Show maps or open the QGIS-project file in the qgis-folder."
                      if flat_original_count is not None else
                      "Processing incomplete. Press the \nProcess area-button.",
                      "https://github.com/ragnvald/mesa/wiki/3-User-interface#processing")

        atlas_count = read_table_and_count('tbl_atlas')
        append_status("+" if atlas_count is not None else "/",
                      f"Atlas pages: {atlas_count}" if atlas_count is not None else
                      "Please create map tile.",
                      "https://github.com/ragnvald/mesa/wiki/5-Definitions#atlas")

        segments_flat_count = read_table_and_count('tbl_segment_flat')
        lines_count = read_table_and_count('tbl_lines')
        append_status("+" if segments_flat_count is not None else "/",
                      f"Segments are in place with {segments_flat_count} segments along {lines_count} lines."
                      if segments_flat_count is not None else
                      "Segments are missing.\nImport or initiate lines if you want to use\nthe line feature.",
                      "https://github.com/ragnvald/mesa/wiki/3-User-interface#lines-and-segments")

        return pd.DataFrame(status_list)
    except Exception as e:
        return pd.DataFrame({'Status': ['Error'], 'Message': [f"Error accessing statistics: {e}"], 'Link': [""]})

# ---------------------------------------------------------------------
# Subprocess runner (+ unified env/cwd)
# ---------------------------------------------------------------------
def _sub_env():
    env = os.environ.copy()
    env["MESA_BASE_DIR"] = PROJECT_BASE
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
        log_to_logfile(f"Attempting to run command: {command}")
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=PROJECT_BASE,
            env=_sub_env()
        )
        log_to_logfile("Primary command executed successfully")
        log_to_logfile(f"stdout: {result.stdout}")
        _schedule_stats_refresh(gpkg_file)
    except subprocess.CalledProcessError as e:
        log_to_logfile(f"Primary command failed with error: {e}")
        log_to_logfile(f"Failed to execute command: {command}, error: {e.stderr}")
        try:
            if fallback_command:
                log_to_logfile(f"Attempting to run fallback command: {fallback_command}")
                result = subprocess.run(
                    fallback_command,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=PROJECT_BASE,
                    env=_sub_env()
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

def process_data(gpkg_file):
    python_script, exe_file = get_script_paths("data_process")
    arg_tokens = ["--original_working_directory", original_working_directory]
    if getattr(sys, "frozen", False):
        log_to_logfile(f"Running bundled exe: {exe_file}")
        run_subprocess([exe_file, *arg_tokens], [], gpkg_file)
    else:
        run_subprocess([sys.executable or "python", python_script, *arg_tokens], [exe_file, *arg_tokens], gpkg_file)

def make_atlas():
    python_script, exe_file = get_script_paths("atlas_create")
    if getattr(sys, "frozen", False):
        log_to_logfile(f"Running bundled exe: {exe_file}")
        run_subprocess([exe_file], [], gpkg_file)
    else:
        run_subprocess([sys.executable or "python", python_script], [exe_file], gpkg_file)

def process_lines():
    python_script, exe_file = get_script_paths("lines_process")
    if getattr(sys, "frozen", False):
        log_to_logfile(f"Running bundled exe: {exe_file}")
        run_subprocess([exe_file], [], gpkg_file)
    else:
        run_subprocess([sys.executable or "python", python_script], [exe_file], gpkg_file)

def open_maps_overview():
    python_script, exe_file = get_script_paths("maps_overview")
    try:
        if getattr(sys, "frozen", False):
            log_to_logfile(f"Launching maps_overview exe: {exe_file}")
            subprocess.Popen([exe_file], cwd=PROJECT_BASE, env=_sub_env())
        else:
            python_exe = sys.executable or "python"
            subprocess.Popen([python_exe, python_script], cwd=PROJECT_BASE, env=_sub_env())
    except Exception as e:
        log_to_logfile(f"Failed to open maps overview: {e}")


def open_asset_layers_viewer():
    python_script, exe_file = get_script_paths("map_assets")
    try:
        if getattr(sys, "frozen", False):
            log_to_logfile(f"Launching map_assets exe: {exe_file}")
            subprocess.Popen([exe_file], cwd=PROJECT_BASE, env=_sub_env())
        else:
            python_exe = sys.executable or "python"
            subprocess.Popen([python_exe, python_script], cwd=PROJECT_BASE, env=_sub_env())
    except Exception as e:
        log_to_logfile(f"Failed to open map_assets viewer: {e}")

def open_present_files():
    python_script, exe_file = get_script_paths("data_report")
    try:
        if getattr(sys, "frozen", False):
            log_to_logfile(f"Launching data_report exe: {exe_file}")
            subprocess.Popen([exe_file], cwd=PROJECT_BASE, env=_sub_env())
        else:
            python_exe = sys.executable or "python"
            subprocess.Popen([python_exe, python_script], cwd=PROJECT_BASE, env=_sub_env())
    except Exception as e:
        log_to_logfile(f"Failed to open data_report: {e}")

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
        log_to_logfile(f"Running bundled exe: {exe_file}")
        run_subprocess([exe_file], [], gpkg_file)
    else:
        run_subprocess([sys.executable or "python", python_script], [exe_file], gpkg_file)

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

def increment_stat_value(config_file, stat_name, increment_value):
    cfg_path = config_file if os.path.isabs(config_file) else resolve_path(config_file)
    if not os.path.isfile(cfg_path):
        log_to_logfile(f"Configuration file {cfg_path} not found.")
        return
    _ensure_default_header_present(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    updated = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f'{stat_name} ='):
            parts = line.split('=')
            if len(parts) == 2:
                current_value = parts[1].strip()
                try:
                    new_value = int(current_value) + increment_value
                    lines[i] = f"{stat_name} = {new_value}\n"
                    updated = True
                    break
                except ValueError:
                    log_to_logfile(f"Error: Current value of {stat_name} is not an integer.")
                    return
    if updated:
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

def add_text_to_labelframe(labelframe, text):
    label = tk.Label(labelframe, text=text, justify='left')
    label.pack(padx=10, pady=10, fill='both', expand=True)
    def update_wrap(event):
        label.config(wraplength=labelframe.winfo_width() - 20)
    labelframe.bind('<Configure>', update_wrap)

# ---------------------------------------------------------------------
# Influx logging helpers (unchanged)
# ---------------------------------------------------------------------
def store_logs_online(
        log_host, log_token, log_org, log_bucket, id_uuid, mesa_version,
        mesa_stat_startup, mesa_stat_process, mesa_stat_import_assets,
        mesa_stat_import_geocodes, mesa_stat_import_atlas, mesa_stat_import_lines,
        mesa_stat_setup, mesa_stat_edit_atlas, mesa_stat_create_atlas, mesa_stat_process_lines):
    if not is_connected():
        return "No network access, logs not updated"
    try:
        def write_point():
            client = InfluxDBClient(url=log_host, token=log_token, org=log_org)
            point = Point("tbl_usage") \
                .tag("uuid", id_uuid) \
                .field("mesa_version", mesa_version) \
                .field("mesa_stat_startup", int(mesa_stat_startup)) \
                .field("mesa_stat_process", int(mesa_stat_process)) \
                .field("mesa_stat_import_assets", int(mesa_stat_import_assets)) \
                .field("mesa_stat_import_geocodes", int(mesa_stat_import_geocodes)) \
                .field("mesa_stat_import_atlas", int(mesa_stat_import_atlas)) \
                .field("mesa_stat_import_lines", int(mesa_stat_import_lines)) \
                .field("mesa_stat_setup", int(mesa_stat_setup)) \
                .field("mesa_stat_edit_atlas", int(mesa_stat_edit_atlas)) \
                .field("mesa_stat_create_atlas", int(mesa_stat_create_atlas)) \
                .field("mesa_stat_process_lines", int(mesa_stat_process_lines))
            write_api = client.write_api(write_options=WriteOptions(batch_size=1))
            write_api.write(bucket=log_bucket, org=log_org, record=point)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(write_point)
            future.result(timeout=3)
    except TimeoutError:
        return "No network access, logs not updated"
    except Exception as e:
        return f"An error occurred: {str(e)}"
    return "Usage logs updated successfully"

def store_userinfo_online(log_host, log_token, log_org, log_bucket, id_uuid, id_name, id_email):
    if not is_connected():
        return "No network access, logs not updated"
    try:
        def write_point():
            client = InfluxDBClient(url=log_host, token=log_token, org=log_org)
            point = Point("tbl_user").tag("uuid", id_uuid).field("id_name", id_name).field("id_email", id_email)
            write_api = client.write_api(write_options=WriteOptions(batch_size=1))
            write_api.write(bucket=log_bucket, org=log_org, record=point)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(write_point)
            future.result(timeout=3)
    except TimeoutError:
        return "No network access, logs not updated"
    except Exception as e:
        return f"An error occurred: {str(e)}"
    return "User logs updated successfully"

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
workingprojection_epsg  = config['DEFAULT'].get('workingprojection_epsg', '4326')

log_date_initiated      = config['DEFAULT'].get('log_date_initiated', '')
log_date_lastupdate     = config['DEFAULT'].get('log_date_lastupdate', '')
log_org                 = config['DEFAULT'].get('log_org', '')
log_bucket              = config['DEFAULT'].get('log_bucket', '')
log_host                = config['DEFAULT'].get('log_host', '')
log_token               = "Xp_sTOcg-46FFiQuplxz-Fqi-jEe5YGfOZarPR7gwZ4CMTMYseUPUjdKtp2xKV9w85TlBlh5X_lnaNzKULAhog=="

mesa_stat_startup           = config['DEFAULT'].get('mesa_stat_startup', '0')
mesa_stat_process           = config['DEFAULT'].get('mesa_stat_process', '0')
mesa_stat_import_assets     = config['DEFAULT'].get('mesa_stat_import_assets', '0')
mesa_stat_import_geocodes   = config['DEFAULT'].get('mesa_stat_import_geocodes', '0')
mesa_stat_import_atlas      = config['DEFAULT'].get('mesa_stat_import_atlas', '0')
mesa_stat_import_lines      = config['DEFAULT'].get('mesa_stat_import_lines', '0')
mesa_stat_setup             = config['DEFAULT'].get('mesa_stat_setup', '0')
mesa_stat_edit_atlas        = config['DEFAULT'].get('mesa_stat_edit_atlas', '0')
mesa_stat_create_atlas      = config['DEFAULT'].get('mesa_stat_create_atlas', '0')
mesa_stat_process_lines     = config['DEFAULT'].get('mesa_stat_process_lines', '0')

id_uuid = config['DEFAULT'].get('id_uuid', '').strip()
id_name = config['DEFAULT'].get('id_name', '').strip()
id_email = config['DEFAULT'].get('id_email', '').strip()
id_uuid_ok_value = config['DEFAULT'].get('id_uuid_ok', 'False').lower() in ('true', '1', 't', 'yes')
id_personalinfo_ok_value = config['DEFAULT'].get('id_personalinfo_ok', 'False').lower() in ('true', '1', 't', 'yes')

has_run_update_stats = False

if not id_uuid:
    id_uuid = str(uuid.uuid4())
    update_config_with_values(config_file, id_uuid=id_uuid)

if not log_date_initiated:
    log_date_initiated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_config_with_values(config_file, log_date_initiated=log_date_initiated)

if not log_date_lastupdate:
    log_date_lastupdate = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_config_with_values(config_file, log_date_lastupdate=log_date_lastupdate)

now = datetime.now()
try:
    log_date_lastupdate_dt = datetime.strptime(log_date_lastupdate, "%Y-%m-%d %H:%M:%S")
except Exception:
    log_date_lastupdate_dt = now - timedelta(hours=2)

if ((now - log_date_lastupdate_dt) > timedelta(hours=1)) and (id_uuid_ok_value is True):
    log_to_logfile(store_logs_online(log_host, log_token, log_org, log_bucket, id_uuid, mesa_version,
                                     mesa_stat_startup, mesa_stat_process, mesa_stat_import_assets,
                                     mesa_stat_import_geocodes, mesa_stat_import_atlas, mesa_stat_import_lines,
                                     mesa_stat_setup, mesa_stat_edit_atlas, mesa_stat_create_atlas, mesa_stat_process_lines))
    log_to_logfile(store_userinfo_online(log_host, log_token, log_org, log_bucket, id_uuid, id_name, id_email))
    update_config_with_values(config_file, log_date_lastupdate=now.strftime("%Y-%m-%d %H:%M:%S"))

check_and_create_folders()

# ---------------------------------------------------------------------
# Tk UI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    root = ttk.Window(themename=ttk_bootstrap_theme)
    root.title(mesa_version or "MESA")
    try:
        root.iconbitmap(resolve_path(os.path.join("system_resources", "mesa.ico")))
    except Exception:
        pass

    TARGET_ASPECT_RATIO = 5 / 3
    DEFAULT_WIDTH = 1050
    DEFAULT_HEIGHT = int(DEFAULT_WIDTH / TARGET_ASPECT_RATIO)
    MIN_WIDTH = 930
    MIN_HEIGHT = int(MIN_WIDTH / TARGET_ASPECT_RATIO)

    root.geometry(f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}")
    root.minsize(MIN_WIDTH, MIN_HEIGHT)

    aspect_guard = {"active": False}

    def enforce_aspect(event):
        if event.widget is not root or aspect_guard["active"]:
            return
        width, height = max(event.width, MIN_WIDTH), max(event.height, MIN_HEIGHT)
        if height == 0:
            return
        ratio = width / height
        if abs(ratio - TARGET_ASPECT_RATIO) < 0.01:
            return
        aspect_guard["active"] = True
        if ratio > TARGET_ASPECT_RATIO:
            width = int(height * TARGET_ASPECT_RATIO)
        else:
            height = int(width / TARGET_ASPECT_RATIO)
        width = max(width, MIN_WIDTH)
        height = max(height, MIN_HEIGHT)
        root.geometry(f"{width}x{height}")
        root.after_idle(lambda: aspect_guard.update(active=False))

    root.bind("<Configure>", enforce_aspect)

    intro_text = (
        "Launch core jobs from Workflows, then review the live counters in Status to confirm imports, processing, "
        "and publishing have completed."
    )

    header = ttk.Frame(root, padding=(12, 10))
    header.pack(fill="x", padx=12, pady=(12, 6))

    intro_label = ttk.Label(
        header,
        text=intro_text,
        wraplength=760,
        justify="left",
        padding=(14, 10),
        bootstyle="inverse-primary"
    )
    intro_label.pack(side="left", fill="x", expand=True)

    ttk.Button(
        header,
        text="Exit",
        command=root.destroy,
        bootstyle="danger-outline",
        width=12
    ).pack(side="right", padx=(12, 0))

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
             "then glance at the Status tab to confirm progress.",
        justify="left",
        wraplength=780
    ).pack(anchor="w", pady=(0, 10))

    workflow_grid = ttk.Frame(workflows_container)
    workflow_grid.pack(fill="both", expand=True)
    workflow_grid.columnconfigure(0, weight=1)
    workflow_grid.columnconfigure(1, weight=1)

    workflow_section_frames = []
    SINGLE_COLUMN_BREAKPOINT = 900
    ACTION_COLUMNS = 2

    workflow_sections = [
        ("Prepare data (step 1)", "Import new sources and generate supporting geometry.", [
            ("Import data", lambda: import_assets(gpkg_file),
             "Start here when preparing a new dataset or refreshing existing inputs."),
            ("Build geocode grids", geocodes_grids,
             "Create or refresh the hexagon/tile grids that support analysis."),
            ("Define atlas tiles", make_atlas,
             "Generate atlas tile polygons used in the QGIS atlas and the report engine."),
        ]),
        ("Configure analysis (step 2)", "Tune processing parameters and study areas before running heavy jobs.", [
            ("Processing settings", edit_processing_setup,
             "Adjust weights, thresholds and other processing rules."),
            ("Define study areas", open_data_analysis_setup,
             "Launch the area analysis tool to pick the study groups."),
        ]),
        ("Run processing (step 3)", "Execute the automated steps that build fresh outputs.", [
            ("Run area processing", lambda: process_data(gpkg_file),
             "Runs the main area pipeline to refresh GeoParquet, MBTiles and stats."),
            ("Run line processing", process_lines,
             "Processes line assets (transport, rivers, utilities) into analysis-ready segments."),
        ]),
        ("Review & publish (step 4)", "Open the interactive viewers and export the deliverables.", [
            ("Asset map studio", open_asset_layers_viewer,
             "Inspect layers with AI-assisted styling controls."),
            ("Analysis map studio", open_maps_overview,
             "Review current background layers together with processed assets."),
            ("Compare study areas", open_data_analysis_presentation,
             "Open the dashboard for comparing study groups."),
            ("Export reports", open_present_files,
             "Render PDF reports based on the latest results."),
        ]),
    ]

    for idx, (section_title, section_description, actions) in enumerate(workflow_sections):
        row = idx // 2
        col = idx % 2
        section_frame = ttk.LabelFrame(workflow_grid, text=section_title, padding=(12, 10))
        section_frame.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
        workflow_section_frames.append(section_frame)
        ttk.Label(
            section_frame,
            text=section_description,
            wraplength=320,
            justify="left"
        ).pack(anchor="w", fill="x")
        actions_container = ttk.Frame(section_frame)
        actions_container.pack(fill="x", pady=(10, 0))
        for col_index in range(ACTION_COLUMNS):
            actions_container.columnconfigure(col_index, weight=1)
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
            ).pack(anchor="w")
            ttk.Label(
                action_block,
                text=action_description,
                wraplength=300,
                justify="left"
            ).pack(anchor="w", pady=(2, 0))

    def relayout_workflow_sections(event=None):
        if event is not None and event.widget is not workflow_grid:
            return
        available_width = workflow_grid.winfo_width()
        if not available_width:
            available_width = workflows_container.winfo_width() or DEFAULT_WIDTH
        columns = 2 if available_width >= SINGLE_COLUMN_BREAKPOINT else 1
        if columns <= 0:
            columns = 1
        for col_index in range(2):
            workflow_grid.columnconfigure(col_index, weight=1 if col_index < columns else 0)
        for idx, frame in enumerate(workflow_section_frames):
            row = idx // columns
            col = idx % columns
            frame.grid_configure(row=row, column=col, padx=8, pady=8, sticky="nsew")

    def resize_window_to_fit_contents():
        root.update_idletasks()
        required_height = root.winfo_reqheight()
        required_width = root.winfo_reqwidth()
        width_for_height = math.ceil(required_height * TARGET_ASPECT_RATIO)
        width_needed = max(DEFAULT_WIDTH, required_width, width_for_height)
        height_needed = int(width_needed / TARGET_ASPECT_RATIO)
        current_width = root.winfo_width()
        current_height = root.winfo_height()
        if width_needed <= current_width and height_needed <= current_height:
            return
        aspect_guard["active"] = True
        root.geometry(f"{width_needed}x{height_needed}")
        root.after_idle(lambda: aspect_guard.update(active=False))

    workflow_grid.bind("<Configure>", relayout_workflow_sections)

    root.update_idletasks()
    relayout_workflow_sections()
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
        text="Live counters and helper tips update automatically when you run any workflow.",
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

    timeline_frame = ttk.Frame(status_columns)
    timeline_frame.grid(row=0, column=1, padx=(8, 0), pady=(0, 8), sticky="nsew")
    ttk.Label(timeline_frame, text="Recent activity", font=("Segoe UI", 10, "bold"), justify="left").pack(anchor="w")
    timeline_canvas = ttk.Frame(timeline_frame)
    timeline_canvas.pack(fill="both", expand=True, pady=(6, 0))
    timeline_entries = []
    for _ in range(4):
        entry = ttk.Frame(timeline_canvas)
        entry.pack(fill="x", pady=4)
        color_bar = ttk.Label(entry, text=" ", width=2, bootstyle="success")
        color_bar.grid(row=0, column=0, padx=(0, 6), sticky="ns")
        title_label = ttk.Label(entry, text="Event", justify="left")
        title_label.grid(row=0, column=1, sticky="w")
        time_label = ttk.Label(entry, text="--", width=18, anchor="e")
        time_label.grid(row=0, column=2, sticky="e")
        entry.columnconfigure(1, weight=1)
        timeline_entries.append((color_bar, title_label, time_label))

    # insight boxes container (below Status and help)
    insights_frame = ttk.Frame(stats_container)
    insights_frame.pack(fill="x", pady=(4, 8))
    insights_frame.columnconfigure((0, 1, 2, 3), weight=1)

    geocode_box = ttk.LabelFrame(insights_frame, text="Objects per geocode", bootstyle="secondary")
    geocode_box.grid(row=0, column=0, padx=5, sticky="nsew")
    geocode_box.columnconfigure(0, weight=1)
    geocode_summary = ttk.Label(geocode_box, text="--", justify="left", wraplength=220)
    geocode_summary.pack(anchor="w", padx=10, pady=8)

    assets_box = ttk.LabelFrame(insights_frame, text="Assets overview", bootstyle="secondary")
    assets_box.grid(row=0, column=1, padx=5, sticky="nsew")
    assets_summary = ttk.Label(assets_box, text="--", justify="left", wraplength=220)
    assets_summary.pack(anchor="w", padx=10, pady=8)

    lines_box = ttk.LabelFrame(insights_frame, text="Lines & segments", bootstyle="secondary")
    lines_box.grid(row=0, column=2, padx=5, sticky="nsew")
    lines_summary = ttk.Label(lines_box, text="--", justify="left", wraplength=220)
    lines_summary.pack(anchor="w", padx=10, pady=8)

    analysis_box = ttk.LabelFrame(insights_frame, text="Analysis layer", bootstyle="secondary")
    analysis_box.grid(row=0, column=3, padx=5, sticky="nsew")
    analysis_summary = ttk.Label(analysis_box, text="--", justify="left", wraplength=220)
    analysis_summary.pack(anchor="w", padx=10, pady=8)

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

    def _last_flat_timestamp():
        geoparquet_dir = _detect_geoparquet_dir()
        flat_path = os.path.join(geoparquet_dir, "tbl_flat.parquet")
        ts = _path_mtime(flat_path)
        if ts:
            return _fmt_timestamp(ts)
        return config['DEFAULT'].get('last_process_run', '--')

    def _last_line_processing_timestamp():
        geoparquet_dir = _detect_geoparquet_dir()
        segment_path = os.path.join(geoparquet_dir, "tbl_segment_flat.parquet")
        ts = _path_mtime(segment_path)
        if ts:
            return _fmt_timestamp(ts)
        return config['DEFAULT'].get('last_lines_process_run', '--')

    def _latest_report_timestamp():
        reports_dir = os.path.join(original_working_directory, "output")
        newest_ts = None
        if os.path.isdir(reports_dir):
            for entry in os.scandir(reports_dir):
                if entry.is_file() and entry.name.lower().endswith(".pdf"):
                    ts = _path_mtime(entry.path)
                    if ts and (newest_ts is None or ts > newest_ts):
                        newest_ts = ts
        if newest_ts:
            return _fmt_timestamp(newest_ts)
        return config['DEFAULT'].get('last_report_export', '--')

    def update_timeline():
        events = [
            ("Import assets", config['DEFAULT'].get('log_date_lastupdate', '--'), 'success'),
            ("Processing", _last_flat_timestamp(), 'info'),
            ("Line processing", _last_line_processing_timestamp(), 'warning'),
            ("Newest report export", _latest_report_timestamp(), 'secondary')
        ]
        for idx, (title, timestamp, bootstyle) in enumerate(events):
            if idx >= len(timeline_entries):
                break
            color_bar, title_label, time_label = timeline_entries[idx]
            color_bar.config(bootstyle=bootstyle)
            title_label.config(text=title)
            time_label.config(text=timestamp)

    def fetch_geocode_objects_summary():
        flat_path = _locate_geoparquet_file("tbl_flat")
        if not flat_path or not os.path.exists(flat_path):
            return "No processing results yet."
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
                    return "No geocode identifiers found in tbl_flat."
                data = df_all[target_col]
            else:
                data = pd.read_parquet(flat_path, columns=[target_col])[target_col]
            counts = data.value_counts(dropna=False).head(5)
            lines = [f"{idx}: {val}" for idx, val in counts.items()]
            if data.nunique(dropna=False) > 5:
                lines.append("…")
            return "\n".join(lines) if lines else "No records found."
        except Exception as exc:
            return f"Unable to read tbl_flat: {exc}"[:200]

    def fetch_asset_summary():
        asset_group_path = _locate_geoparquet_file("tbl_asset_group")
        if not asset_group_path:
            return "Assets not imported yet."
        try:
            asset_groups = pd.read_parquet(asset_group_path)
            layers = asset_groups.shape[0]
            objects = None
            assets_path = _locate_geoparquet_file("tbl_assets")
            if assets_path:
                objects = pd.read_parquet(assets_path).shape[0]
            elif "total_asset_objects" in asset_groups.columns:
                objects = int(asset_groups["total_asset_objects"].fillna(0).sum())
            detail = f"Layers: {layers}"
            if objects is None:
                detail += "\nObjects: --"
            else:
                detail += f"\nObjects: {objects}"
            return detail
        except Exception as exc:
            return f"Unable to read assets: {exc}"[:200]

    def fetch_lines_summary():
        lines_path = _locate_geoparquet_file("tbl_lines")
        if not lines_path:
            return "Lines not processed yet."
        segments_path = _locate_geoparquet_file("tbl_segment_flat")
        try:
            lines_count = pd.read_parquet(lines_path).shape[0]
            segments_count = pd.read_parquet(segments_path).shape[0] if segments_path and os.path.exists(segments_path) else 0
            return f"Lines: {lines_count}\nSegments: {segments_count}"
        except Exception as exc:
            return f"Unable to read lines: {exc}"[:200]

    def fetch_analysis_summary():
        stacked_path = _locate_geoparquet_file("tbl_stacked")
        if not stacked_path:
            return "Analysis layer missing."
        try:
            row_count = _parquet_row_count(stacked_path)
            if row_count is None:
                return "Unable to read tbl_stacked."
            return f"Objects: {row_count}"
        except Exception as exc:
            return f"Unable to read tbl_stacked: {exc}"[:200]

    def update_insight_boxes():
        geocode_summary.config(text=fetch_geocode_objects_summary())
        assets_summary.config(text=fetch_asset_summary())
        lines_summary.config(text=fetch_lines_summary())
        analysis_summary.config(text=fetch_analysis_summary())

    update_stats(gpkg_file)
    update_timeline()
    update_insight_boxes()
    log_to_logfile("User interface, status updated.")

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
        ("Edit assets", edit_assets,
         "This is where you can add titles to the different layers you have imported. You may also add a short descriptive text."),
        ("Edit geocodes", edit_geocodes,
         "Geocodes can be grid cells, hexagons or other polygons. Add titles to them here for easier reference later."),
        ("Edit lines", edit_lines,
         "Remember to import lines before attempting to edit them. Adjust buffer/segment parameters per line as needed."),
        ("Edit map tiles", edit_atlas,
         "Remember to import or create map tiles before attempting to edit them. Map tiles are polygons highlighted in the QGIS project."),
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

    increment_stat_value(config_file, 'mesa_stat_startup', increment_value=1)

    about_box = ttk.LabelFrame(about_container, text="About MESA", bootstyle='secondary')
    about_box.pack(fill='x', expand=False, padx=5, pady=(0, 10))
    about_text = (
        "Welcome to the MESA tool. The method is developed by UNEP-WCMC and the Norwegian Environment Agency. "
        "The software streamlines sensitivity analysis, reducing the likelihood of manual errors in GIS workflows.\n\n"
        "Documentation and user guides are available on the MESA wiki: https://github.com/ragnvald/mesa/wiki\n\n"
        "This release incorporates feedback from workshops with partners in Ghana, Tanzania, Uganda, Mozambique, "
        "and Kenya. Lead programmer: Ragnvald Larsen - https://www.linkedin.com/in/ragnvald/"
    )
    ttk.Label(
        about_box,
        text=about_text,
        wraplength=700,
        justify="left"
    ).pack(fill='x', expand=False, padx=10, pady=10)

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    footer = ttk.Frame(root, padding=(10, 5))
    footer.pack(fill='x', padx=12, pady=(0, 6))

    ttk.Label(footer, text=mesa_version, font=("Calibri", 8)).pack(side='left')

    notebook.select(0)
    root.update_idletasks()
    root.mainloop()
