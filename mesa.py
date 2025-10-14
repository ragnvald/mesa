import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

import tkinter as tk
from tkinter import *
import os
from tkinterweb import HtmlFrame
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
import sys
from shapely import wkb
import pyarrow.parquet as pq

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

def _detect_geoparquet_dir() -> str:
    """Locate the geoparquet folder, preferring the live data set when multiple copies exist."""
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

    seen = set()
    unique_candidates = []
    for path in candidates:
        ap = os.path.abspath(path)
        if ap not in seen:
            seen.add(ap)
            unique_candidates.append(ap)

    sentinel_files = [
        "tbl_asset_group.parquet",
        "tbl_geocode_group.parquet",
        "tbl_lines_original.parquet",
        "tbl_flat.parquet",
        "tbl_segment_flat.parquet",
    ]

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

# ---------------------------------------------------------------------
# Status panel (unchanged logic, reads GeoParquet)
# ---------------------------------------------------------------------
def update_stats(_unused_gpkg_path):
    for widget in info_labelframe.winfo_children():
        widget.destroy()

    geoparquet_dir = _detect_geoparquet_dir()

    if not os.path.isdir(geoparquet_dir):
        status_label = ttk.Label(info_labelframe, text='\u26AB', bootstyle='danger')
        status_label.grid(row=1, column=0, padx=5, pady=5)
        message_label = ttk.Label(info_labelframe,
                                  text="No data imported.\nStart with importing data.",
                                  wraplength=380, justify="left")
        message_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        create_link_icon(info_labelframe, "https://github.com/ragnvald/mesa/wiki", 1, 2, 5, 5)
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
            status_label.grid(row=1, column=0, padx=5, pady=5)
            message_label = ttk.Label(info_labelframe,
                                      text="To initiate the system please import assets.\n"
                                           "Press the Import button.",
                                      wraplength=380, justify="left")
            message_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")
            create_link_icon(info_labelframe, "https://github.com/ragnvald/mesa/wiki", 1, 2, 5, 5)

    root.update_idletasks()
    root.update()

def get_status(geoparquet_dir):
    status_list = []

    def ppath(layer_name: str) -> str:
        return os.path.join(geoparquet_dir, f"{layer_name}.parquet")

    def table_exists_nonempty(layer_name: str) -> bool:
        fp = ppath(layer_name)
        if not os.path.exists(fp):
            return False
        try:
            return (pq.ParquetFile(fp).metadata.num_rows or 0) > 0
        except Exception:
            try:
                return len(pd.read_parquet(fp)) > 0
            except Exception:
                return False

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
        env_ok = table_exists_nonempty('tbl_env_profile')

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

        if env_ok and assets_ok:
            return "+", "Set up ok. Feel free to adjust it."
        else:
            parts = []
            if not env_ok:
                parts.append("tbl_env_profile is missing or empty")
            if not assets_ok:
                if missing_cols_msg:
                    parts.append(missing_cols_msg.strip())
                else:
                    parts.append("importance/susceptibility/sensitivity not assigned (>0) in tbl_asset_group")
            detail = "; ".join(parts) if parts else "Incomplete setup."
            return "-", f"You need to set up the calculation. \nPress the 'Set up'-button to proceed. ({detail})"

    def append_status(symbol, message, link):
        status_list.append({'Status': symbol, 'Message': message, 'Link': link})

    try:
        asset_group_count = read_table_and_count('tbl_asset_group')
        append_status("+" if asset_group_count is not None else "-",
                      f"Asset layers: {asset_group_count}" if asset_group_count is not None else
                      "Assets are missing.\nImport assets by pressing the Import button.",
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
                      "Processing completed. You may open the QGIS-project file in the output-folder."
                      if flat_original_count is not None else
                      "Processing incomplete. Press the \nprocessing button.",
                      "https://github.com/ragnvald/mesa/wiki/3-User-interface#processing")

        atlas_count = read_table_and_count('tbl_atlas')
        append_status("+" if atlas_count is not None else "/",
                      f"Atlas pages: {atlas_count}" if atlas_count is not None else
                      "Please create atlas.",
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
        update_stats(gpkg_file)
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
                update_stats(gpkg_file)
        except subprocess.CalledProcessError as e2:
            log_to_logfile(f"Failed to execute fallback command: {fallback_command}, error: {e2.stderr}")
        except FileNotFoundError as e2:
            log_to_logfile(f"File not found for fallback command: {fallback_command}, error: {e2}")
    except FileNotFoundError as e:
        log_to_logfile(f"File not found for command: {command}, error: {e}")

def get_script_paths(file_name: str):
    # Python and compiled variants live under system/
    python_script = resolve_path(os.path.join("system", f"{file_name}.py"))
    exe_file     = resolve_path(os.path.join("system", f"{file_name}.exe"))
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
        file_path = resolve_path(os.path.join("system", "geocodes_create.exe"))
        log_to_logfile(f"Running bundled exe: {file_path}")
        run_subprocess([file_path, *arg_tokens], [], gpkg_file)
    else:
        run_subprocess([sys.executable or "python", python_script, *arg_tokens], [exe_file, *arg_tokens], gpkg_file)

def import_assets(gpkg_file):
    python_script, exe_file = get_script_paths("data_import")
    if getattr(sys, "frozen", False):
        file_path = resolve_path(os.path.join("system", "data_import.exe"))
        log_to_logfile(f"Running bundled exe: {file_path}")
        run_subprocess([file_path], [], gpkg_file)
    else:
        run_subprocess([sys.executable or "python", python_script], [exe_file], gpkg_file)

def edit_processing_setup():
    python_script, exe_file = get_script_paths("params_edit")
    if getattr(sys, "frozen", False):
        file_path = resolve_path(os.path.join("system", "params_edit.exe"))
        log_to_logfile(f"Running bundled exe: {file_path}")
        run_subprocess([file_path], [], gpkg_file)
    else:
        run_subprocess([sys.executable or "python", python_script], [exe_file], gpkg_file)

def process_data(gpkg_file):
    python_script, exe_file = get_script_paths("data_process")
    arg_tokens = ["--original_working_directory", original_working_directory]
    if getattr(sys, "frozen", False):
        file_path = resolve_path(os.path.join("system", "data_process.exe"))
        log_to_logfile(f"Running bundled exe: {file_path}")
        run_subprocess([file_path, *arg_tokens], [], gpkg_file)
    else:
        run_subprocess([sys.executable or "python", python_script, *arg_tokens], [exe_file, *arg_tokens], gpkg_file)

def make_atlas():
    python_script, exe_file = get_script_paths("atlas_create")
    if getattr(sys, "frozen", False):
        file_path = resolve_path(os.path.join("system", "atlas_create.exe"))
        log_to_logfile(f"Running bundled exe: {file_path}")
        run_subprocess([file_path], [], gpkg_file)
    else:
        run_subprocess([sys.executable or "python", python_script], [exe_file], gpkg_file)

def admin_lines():
    python_script, exe_file = get_script_paths("lines_admin")
    if getattr(sys, "frozen", False):
        file_path = resolve_path(os.path.join("system", "lines_admin.exe"))
        log_to_logfile(f"Running bundled exe: {file_path}")
        run_subprocess([file_path], [], gpkg_file)
    else:
        run_subprocess([sys.executable or "python", python_script], [exe_file], gpkg_file)

def open_maps_overview():
    python_script = resolve_path(os.path.join("system", "maps_overview.py"))
    python_exe = sys.executable or "python"
    try:
        subprocess.Popen([python_exe, python_script], cwd=PROJECT_BASE, env=_sub_env())
    except Exception as e:
        log_to_logfile(f"Failed to open maps_overview.py: {e}")

def open_present_files():
    python_script = resolve_path(os.path.join("system", "data_report.py"))
    python_exe = sys.executable or "python"
    try:
        subprocess.Popen([python_exe, python_script], cwd=PROJECT_BASE, env=_sub_env())
    except Exception as e:
        log_to_logfile(f"Failed to open data_report.py: {e}")

def edit_assets():
    python_script, exe_file = get_script_paths("assetgroup_edit")
    if getattr(sys, "frozen", False):
        file_path = resolve_path(os.path.join("system", "assetgroup_edit.exe"))
        log_to_logfile(f"Running bundled exe: {file_path}")
        run_subprocess([file_path], [], gpkg_file)
    else:
        run_subprocess([sys.executable or "python", python_script], [exe_file], gpkg_file)

def edit_geocodes():
    python_script, exe_file = get_script_paths("geocodegroup_edit")
    if getattr(sys, "frozen", False):
        file_path = resolve_path(os.path.join("system", "geocodegroup_edit.exe"))
        log_to_logfile(f"Running bundled exe: {file_path}")
        run_subprocess([file_path], [], gpkg_file)
    else:
        run_subprocess([sys.executable or "python", python_script], [exe_file], gpkg_file)

def edit_lines():
    python_script, exe_file = get_script_paths("lines_edit")
    if getattr(sys, "frozen", False):
        file_path = resolve_path(os.path.join("system", "lines_edit.exe"))
        log_to_logfile(f"Running bundled exe: {file_path}")
        run_subprocess([file_path], [], gpkg_file)
    else:
        run_subprocess([sys.executable or "python", python_script], [exe_file], gpkg_file)

def edit_atlas():
    python_script, exe_file = get_script_paths("atlas_edit")
    if getattr(sys, "frozen", False):
        file_path = resolve_path(os.path.join("system", "atlas_edit.exe"))
        log_to_logfile(f"Running bundled exe: {file_path}")
        run_subprocess([file_path], [], gpkg_file)
    else:
        run_subprocess([sys.executable or "python", python_script], [exe_file], gpkg_file)

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
        for i, line in enumerate(lines):
            if line.strip().startswith(f'{key} ='):
                lines[i] = f"{key} = {value}\n"
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

# ---------------------------------------------------------------------
# Frame switchers (unchanged)
# ---------------------------------------------------------------------
def show_main_frame():
    about_frame.pack_forget()
    registration_frame.pack_forget()
    settings_frame.pack_forget()
    main_frame.pack(fill='both', expand=True, pady=10)

def show_about_frame():
    main_frame.pack_forget()
    registration_frame.pack_forget()
    settings_frame.pack_forget()
    about_frame.pack(fill='both', expand=True)

def show_registration_frame():
    main_frame.pack_forget()
    about_frame.pack_forget()
    settings_frame.pack_forget()
    registration_frame.pack(fill='both', expand=True)

def show_settings_frame():
    main_frame.pack_forget()
    about_frame.pack_forget()
    registration_frame.pack_forget()
    settings_frame.pack(fill='both', expand=True)

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

def submit_form():
    global id_name, id_email
    id_name = name_entry.get()
    id_email = email_entry.get()
    id_uuid_ok_str = str(id_uuid_ok.get())
    id_personalinfo_ok_str = str(id_personalinfo_ok.get())
    update_config_with_values(config_file,
                              id_uuid=id_uuid,
                              id_name=id_name,
                              id_email=id_email,
                              id_uuid_ok=id_uuid_ok_str,
                              id_personalinfo_ok=id_personalinfo_ok_str)

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
    root.title("MESA 4")
    try:
        root.iconbitmap(resolve_path(os.path.join("system_resources", "mesa.ico")))
    except Exception:
        pass
    root.geometry("850x540")

    button_width = 18
    button_padx  =  7
    button_pady  =  7

    main_frame = tk.Frame(root)
    main_frame.pack(fill='both', expand=True, pady=10)

    main_frame.grid_columnconfigure(0, weight=0)
    main_frame.grid_columnconfigure(1, weight=0)
    main_frame.grid_columnconfigure(2, weight=1)

    left_panel = tk.Frame(main_frame)
    left_panel.grid(row=0, column=0, sticky="nsew", padx=20)
    main_frame.grid_columnconfigure(0, minsize=220)

    import_assets_btn = ttk.Button(left_panel, text="Import",
                                   command=lambda: import_assets(gpkg_file),
                                   width=button_width, bootstyle=PRIMARY)
    import_assets_btn.grid(row=0, column=0, padx=button_padx, pady=button_pady)

    geocodes_btn = ttk.Button(left_panel, text="Geocodes/grids",
                              command=geocodes_grids, width=button_width)
    geocodes_btn.grid(row=1, column=0, padx=button_padx, pady=button_pady)

    setup_processing_btn = ttk.Button(left_panel, text="Set up",
                                      command=edit_processing_setup, width=button_width)
    setup_processing_btn.grid(row=2, column=0, padx=button_padx, pady=button_pady)

    process_data_btn = ttk.Button(left_panel, text="Process",
                                  command=lambda: process_data(gpkg_file),
                                  width=button_width, bootstyle=PRIMARY)
    process_data_btn.grid(row=3, column=0, padx=button_padx, pady=button_pady)

    admin_atlas_btn = ttk.Button(left_panel, text="Atlas",
                                 command=make_atlas, width=button_width)
    admin_atlas_btn.grid(row=4, column=0, padx=button_padx, pady=button_pady)

    admin_lines_btn = ttk.Button(left_panel, text="Segments",
                                 command=admin_lines, width=button_width)
    admin_lines_btn.grid(row=5, column=0, padx=button_padx, pady=button_pady)

    maps_overview_btn = ttk.Button(left_panel, text="Maps overview",
                                   command=open_maps_overview, width=button_width, bootstyle=PRIMARY)
    maps_overview_btn.grid(row=6, column=0, padx=button_padx, pady=button_pady)

    present_files_btn = ttk.Button(left_panel, text="Report engine",
                                   command=open_present_files, width=button_width)
    present_files_btn.grid(row=7, column=0, padx=button_padx, pady=button_pady)

    separator = ttk.Separator(main_frame, orient='vertical')
    separator.grid(row=0, column=1, sticky='ns')

    right_panel = ttk.Frame(main_frame)
    right_panel.grid(row=0, column=2, sticky="nsew", padx=5)
    right_panel.grid_rowconfigure(0, weight=1)
    right_panel.grid_columnconfigure(0, weight=1)

    global info_labelframe
    info_labelframe = ttk.LabelFrame(right_panel, text="Statistics and help", bootstyle='info')
    info_labelframe.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    info_labelframe.grid_columnconfigure(0, weight=1)
    info_labelframe.grid_columnconfigure(1, weight=3)
    info_labelframe.grid_columnconfigure(2, weight=2)

    update_stats(gpkg_file)
    log_to_logfile("User interface, statistics updated.")

    # About frame
    about_frame = ttk.Frame(root)
    increment_stat_value(config_file, 'mesa_stat_startup', increment_value=1)

    html_frame = HtmlFrame(about_frame, horizontal_scrollbar="auto", messages_enabled=False)

    userguide_path = resolve_path(os.path.join("system_resources", "userguide.html"))
    if os.path.exists(userguide_path):
        with open(userguide_path, "r", encoding="utf-8") as file:
            html_content = file.read()
        html_frame.load_html(html_content)
        html_frame.pack(fill=BOTH, expand=YES)
    else:
        # Graceful fallback if the HTML is missing
        fallback_lbl = ttk.Label(about_frame,
                                 text="User guide (system_resources/userguide.html) not found.\n"
                                      "Place it under either:\n"
                                      " - <folder with mesa.py or EXE>\\system_resources\\userguide.html, or\n"
                                      " - <folder with mesa.py or EXE>\\code\\system_resources\\userguide.html",
                                 justify="left", wraplength=700)
        fallback_lbl.pack(fill='both', expand=True, padx=10, pady=10)

    # Registration frame
    registration_frame = ttk.Frame(root)
    registration_frame.pack(fill='both', expand=True)

    id_uuid_ok = tk.BooleanVar(value=id_uuid_ok_value)
    id_personalinfo_ok = tk.BooleanVar(value=id_personalinfo_ok_value)

    about_labelframe = ttk.LabelFrame(registration_frame, text="Licensing and personal information", bootstyle='secondary')
    about_labelframe.pack(side='top', fill='both', expand=True, padx=5, pady=5)

    mesa_text = ("MESA 4.1 is open source software. It is available under the "
                 "GNU GPLv3 license. This means you can use the software for free."
                 "\n\n"
                 "In MESA, a unique random identifier (UUID) is automatically generated. "
                 "It can be used to count how many times the system has been used. It "
                 "is not associated with where you are or who you are. The UUID together "
                 "with usage information will be sent to one of our servers. You can opt "
                 "out of using this functionality by unticking the associated box below."
                 "\n\n"
                 "Additionally you can tick the box next to name and email registration "
                 "and add your name and email for our reference. This might be used "
                 "to send you questionaires and information about updates of the MESA "
                 "tool/method at a later stage."
                 "\n\n"
                 "Your email and name is also stored locally in the config.ini-file.")
    def add_text_to_labelframe(labelframe, text):
        label = tk.Label(labelframe, text=text, justify='left')
        label.pack(padx=10, pady=10, fill='both', expand=True)
        def update_wrap(event):
            label.config(wraplength=labelframe.winfo_width() - 20)
        labelframe.bind('<Configure>', update_wrap)

    add_text_to_labelframe(about_labelframe, mesa_text)

    grid_frame = ttk.Frame(registration_frame)
    grid_frame.pack(side='top', fill='both', expand=True, padx=5, pady=5)

    uuid_ok_checkbox = ttk.Checkbutton(grid_frame, text="", variable=id_uuid_ok)
    uuid_ok_checkbox.grid(row=0, column=0, padx=10, pady=5, sticky="w")

    personalinfo_ok_checkbox = ttk.Checkbutton(grid_frame, text="", variable=id_personalinfo_ok)
    personalinfo_ok_checkbox.grid(row=1, column=0, padx=10, pady=5, sticky="w")

    ttk.Label(grid_frame, text="UUID:").grid(row=0, column=1, padx=10, pady=5, sticky="w")
    ttk.Label(grid_frame, text="Name:").grid(row=1, column=1, padx=10, pady=5, sticky="w")
    ttk.Label(grid_frame, text="Email:").grid(row=2, column=1, padx=10, pady=5, sticky="w")

    ttk.Label(grid_frame, text=id_uuid).grid(row=0, column=2, padx=10, pady=5, sticky="w")

    name_entry = ttk.Entry(grid_frame)
    name_entry.grid(row=1, column=2, padx=10, pady=5, sticky="we")
    name_entry.insert(0, id_name)

    email_entry = ttk.Entry(grid_frame)
    email_entry.grid(row=2, column=2, padx=10, pady=5, sticky="we")
    email_entry.insert(0, id_email)

    submit_btn = ttk.Button(grid_frame, text="Save", command=submit_form, bootstyle=SUCCESS)
    submit_btn.grid(row=2, column=3, padx=10, pady=5, sticky="e")

    grid_frame.columnconfigure(2, weight=1)

    # Settings frame
    settings_frame = ttk.Frame(root)
    settings_frame.pack(fill='both', expand=True)

    about_labelframe = ttk.LabelFrame(settings_frame, text="Settings", bootstyle='info')
    about_labelframe.pack(side='top', fill='both', expand=True, padx=5, pady=5)

    mesa_text2 = ("Some of the objects you already have imported or created might need some further adjustments.\n"
                  "You may do this by reading up on the below suggestions.")
    add_text_to_labelframe(about_labelframe, mesa_text2)

    grid_frame2 = ttk.Frame(settings_frame)
    grid_frame2.pack(side='top', fill='both', expand=True, padx=20, pady=20)

    edit_polygons_btn = ttk.Button(grid_frame2, text="Edit assets", command=edit_assets, bootstyle="primary")
    edit_polygons_btn.grid(row=1, column=0, padx=5, pady=5, sticky='ew')
    edit_polygons_lbl = ttk.Label(grid_frame2, text="This is where you can add titles to the different layers you have imported. You may also add a short descriptive text.", wraplength=550)
    edit_polygons_lbl.grid(row=1, column=1, padx=5, pady=5, sticky='w')

    edit_geocode_btn = ttk.Button(grid_frame2, text="Edit geocodes", command=edit_geocodes, bootstyle="primary")
    edit_geocode_btn.grid(row=2, column=0, padx=5, pady=5, sticky='ew')
    edit_geocode_lbl = ttk.Label(grid_frame2, text="Geocodes can be grid cells, hexagons or other types of polygons. You may add titles to them here for easier reference later.", wraplength=550)
    edit_geocode_lbl.grid(row=2, column=1, padx=5, pady=5, sticky='w')

    edit_lines_btn = ttk.Button(grid_frame2, text="Edit lines", command=edit_lines, bootstyle="primary")
    edit_lines_btn.grid(row=3, column=0, padx=5, pady=5, sticky='ew')
    edit_lines_lbl = ttk.Label(grid_frame2, text="Remember to import lines before attempting to edit them. Lines are processed to segments with the parameters length and width. Default values are set when the lines are imported. If you want to do the processing with other segment sizes you may do so here on a per line basis. This is where you can adjust the parameters as well as their names.", wraplength=550)
    edit_lines_lbl.grid(row=3, column=1, padx=5, pady=5, sticky='w')

    edit_atlas_btn = ttk.Button(grid_frame2, text="Edit atlas", command=edit_atlas, bootstyle="primary")
    edit_atlas_btn.grid(row=4, column=0, padx=5, pady=5, sticky='ew')
    edit_atlas_lbl = ttk.Label(grid_frame2, text="Remember to import or create atlases before attempting to edit them. Atlases are polygons which will be highlighted in the QGIS project file.", wraplength=550)
    edit_atlas_lbl.grid(row=4, column=1, padx=5, pady=5, sticky='w')

    grid_frame2.columnconfigure(1, weight=1)
    grid_frame2.columnconfigure(2, weight=1)

    # Bottom nav
    bottom_frame_buttons = ttk.Frame(root)
    bottom_frame_buttons.pack(side='bottom', fill='x', padx=10, pady=5)

    main_frame_btn = ttk.Button(bottom_frame_buttons, text="MESA desktop", command=show_main_frame, bootstyle="primary")
    main_frame_btn.pack(side='left', padx=(0, 10))

    settings_frame_btn = ttk.Button(bottom_frame_buttons, text="Settings", command=show_settings_frame, bootstyle="primary")
    settings_frame_btn.pack(side='left', padx=(0, 10))

    about_frame_btn = ttk.Button(bottom_frame_buttons, text="About...", command=show_about_frame, bootstyle="primary")
    about_frame_btn.pack(side='left', padx=(0, 10))

    registration_frame_btn = ttk.Button(bottom_frame_buttons, text="Register...", command=show_registration_frame, bootstyle="primary")
    registration_frame_btn.pack(side='left', padx=(0, 10))

    center_frame = ttk.Frame(bottom_frame_buttons)
    center_frame.pack(side='left', expand=True, fill='x')

    version_label = ttk.Label(center_frame, text=mesa_version, font=("Calibri", 7))
    version_label.pack(side='left', padx=50, pady=5)

    exit_btn = ttk.Button(bottom_frame_buttons, text="Exit", command=root.destroy, bootstyle="warning")
    exit_btn.pack(side='right')

    # Default view
    show_main_frame()
    root.update_idletasks()
    root.mainloop()
