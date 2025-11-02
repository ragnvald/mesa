import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
import tkinter as tk
from tkinter import *
import os
import subprocess
import webbrowser
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import pandas as pd
import configparser
import socket
import uuid
import datetime
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient, Point, WriteOptions
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import sys
import pyarrow.parquet as pq  # <-- for GeoParquet status checks

# -------------------------------
# Launch helpers (frozen/dev)
# -------------------------------
def is_frozen() -> bool:
    """True when running inside a PyInstaller/onefile bundle."""
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')

def app_base_dir() -> str:
    """Folder where the running app lives (EXE when frozen, .py folder in dev)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def resource_path(*parts: str) -> str:
    """Find resources in both packaged and dev modes."""
    candidates = []
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        candidates.append(os.path.join(sys._MEIPASS, *parts))      # packed
    candidates.append(os.path.join(app_base_dir(), *parts))        # next to mesa.exe / mesa.py
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]

def tool_path(exe_name: str) -> str:
    """
    Locate helper EXE robustly:
    - dist\mesa\tools\<exe>   (built layout)
    - _MEIPASS\tools\<exe>     (if ever packed inside the EXE)
    """
    paths = [
        os.path.join(app_base_dir(), "tools", exe_name),
    ]
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        paths.append(os.path.join(sys._MEIPASS, "tools", exe_name))
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Helper not found; tried: {paths}")

def run_tool(name: str, args: list[str] | None = None, wait: bool = False) -> int | None:
    """
    Dev: run name.py with your interpreter.
    Frozen: run tools\\name.exe.
    CWD is set to the app base dir so helpers see input/output/qgis/system_resources.
    If wait=True, return the process' return code; otherwise return None.
    """
    import subprocess

    args = args or []
    try:
        if getattr(sys, "frozen", False):
            exe = tool_path(f"{name}.exe")
            cmd = [exe, *args]
            creationflags = 0x08000000 | 0x00000008  # CREATE_NO_WINDOW | DETACHED_PROCESS
            if wait:
                completed = subprocess.run(cmd, cwd=app_base_dir(), check=False,
                                           creationflags=creationflags, shell=False, close_fds=True)
                return completed.returncode
            else:
                subprocess.Popen(cmd, cwd=app_base_dir(), shell=False, close_fds=True,
                                 creationflags=creationflags)
                return None
        else:
            # Dev: run the .py helper sitting next to mesa.py
            script = os.path.join(app_base_dir(), f"{name}.py")
            if not os.path.exists(script):
                raise FileNotFoundError(f"Dev helper not found: {script}")
            cmd = [sys.executable, script, *args]
            if wait:
                completed = subprocess.run(cmd, cwd=os.path.dirname(script), check=False, shell=False, close_fds=True)
                return completed.returncode
            else:
                subprocess.Popen(cmd, cwd=os.path.dirname(script), shell=False, close_fds=True)
                return None
    except Exception as e:
        try:
            log_to_logfile(f"[run_tool] Failed: {cmd} :: {e}")
        except Exception:
            pass
        raise

# -------------------------------
# Read the configuration file
# -------------------------------
def read_config(file_name):
    config = configparser.ConfigParser()
    cfg_path = os.path.join(app_base_dir(), file_name)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Configuration not found: {cfg_path}")
    config.read(cfg_path)
    return config

def is_connected(hostname="8.8.8.8", port=53, timeout=3):
    """
    Hostname: 8.8.8.8 (Google DNS)
    OpenPort: 53/tcp
    Service: domain (DNS/TCP)
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((hostname, port))
        return True
    except socket.error as ex:
        print(ex)
        return False

# Function to check and create folders
def check_and_create_folders():
    folders = ["input/geocode", "output", "qgis", "input/lines"]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

def log_to_logfile(message):
    timestamp = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted_message = f"{timestamp} - {message}"
    with open("log.txt", "a") as log_file:
        log_file.write(formatted_message + "\n")

def create_link_icon(parent, url, row, col, padx, pady):
    # Create a canvas widget
    icon_size = 20  # Size of the icon
    canvas = tk.Canvas(parent, width=icon_size, height=icon_size, bd=0, highlightthickness=0)
    canvas.grid(row=row, column=col, padx=padx, pady=pady, sticky="nsew")
    
    # Draw a circle with a white fill
    canvas.create_oval(2, 2, icon_size-2, icon_size-2, fill='white', outline='blue')
    # Place the letter "i" inside the circle
    canvas.create_text(icon_size/2, icon_size/2, text="i", font=('Calibri', 10, 'bold'), fill='blue')
    # Bind the canvas to open the URL on click
    canvas.bind("<Button-1>", lambda e: webbrowser.open(url))

# This function updates the stats in the labelframe. Clear labels first, then write the updates.
def update_stats(_unused_gpkg_path):
    """
    Update the right-hand status panel based on GeoParquet files in output/geoparquet.
    (Signature kept the same so existing calls don't need to change.)
    """
    for widget in info_labelframe.winfo_children():
        widget.destroy()

    geoparquet_dir = os.path.join(original_working_directory, "output", "geoparquet")

    # If the geoparquet folder doesn't exist, prompt to import
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
                                           "Press the 'Import' button.",
                                      wraplength=380, justify="left")
            message_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")
            create_link_icon(info_labelframe, "https://github.com/ragnvald/mesa/wiki", 1, 2, 5, 5)

    # Always refresh the UI so the new lights actually appear
    root.update_idletasks()
    root.update()

def get_status(geoparquet_dir):
    """
    Build the status table by inspecting GeoParquet files in `geoparquet_dir`.
    Mirrors the old logic (counts, setup check, etc.) but for .parquet files.
    """
    status_list = []

    def ppath(layer_name: str) -> str:
        return os.path.join(geoparquet_dir, f"{layer_name}.parquet")

    def table_exists_nonempty(layer_name: str) -> bool:
        """True if the parquet file exists and has >0 rows."""
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
        """Return row count for a GeoParquet table, or None if it doesn't exist."""
        fp = ppath(layer_name)
        if not os.path.exists(fp):
            log_to_logfile(f"Parquet table {layer_name} does not exist.")
            return None
        try:
            # Fast: read metadata only
            return pq.ParquetFile(fp).metadata.num_rows
        except Exception as e:
            # Fallback: load and count (slower)
            log_to_logfile(f"Parquet metadata read failed for {layer_name}: {e}; falling back to full read.")
            try:
                return len(pd.read_parquet(fp))
            except Exception as e2:
                log_to_logfile(f"Error counting rows for {layer_name}: {e2}")
                return None

    def read_setup_status():
        """
        Determine whether processing 'setup' has been completed.

        Criteria (both must be satisfied):
        1) tbl_env_profile exists and is non-empty.
        2) tbl_asset_group has assigned (non-zero) numeric values for all of:
           'importance', 'susceptibility', and 'sensitivity'.
        """
        env_ok = table_exists_nonempty('tbl_env_profile')

        fp = ppath('tbl_asset_group')
        assets_ok = False
        missing_cols_msg = ""
        try:
            if os.path.exists(fp):
                # Only read the needed columns (fast if present)
                cols = ['importance', 'susceptibility', 'sensitivity']
                df = pd.read_parquet(fp, columns=cols)
                have_all = all(c in df.columns for c in cols)
                if not have_all:
                    missing = [c for c in cols if c not in df.columns]
                    missing_cols_msg = f" Missing columns in tbl_asset_group: {', '.join(missing)}."
                # Convert to numeric and test for at least one > 0 value in each column
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
        status_list.append({'Status': symbol, 
                            'Message': message,
                            'Link': link})

    try:
        # Check for tbl_asset_group
        asset_group_count = read_table_and_count('tbl_asset_group')
        append_status("+" if asset_group_count is not None else "-", 
                      f"Asset layers: {asset_group_count}" if asset_group_count is not None else "Assets are missing.\nImport assets by pressing the Import button.",
                      "https://github.com/ragnvald/mesa/wiki/3-User-interface#assets")

        # Check for tbl_geocode_group
        geocode_group_count = read_table_and_count('tbl_geocode_group')
        append_status("+" if geocode_group_count is not None else "/", 
                      f"Geocode layers: {geocode_group_count}" if geocode_group_count is not None else "Geocodes are missing.\nImport assets by pressing the Import button.",
                      "https://github.com/ragnvald/mesa/wiki/3-User-interface#geocodes")

        # Check for original lines
        lines_original_count = read_table_and_count('tbl_lines_original')
        append_status("+" if lines_original_count is not None else "/", 
                      f"Lines: {lines_original_count}" if lines_original_count is not None else "Lines are missing.\nImport or initiate lines if you want to use\nthe line feature.",
                      "https://github.com/ragnvald/mesa/wiki/3-User-interface#lines")

        # Setup status — requires tbl_env_profile present and tbl_asset_group having values
        symbol, message = read_setup_status()
        append_status(symbol, message, "https://github.com/ragnvald/mesa/wiki/3-User-interface#setting-up-parameters")

        # Present status for calculations on geocode objects
        flat_original_count = read_table_and_count('tbl_flat')
        append_status("+" if flat_original_count is not None else "-", 
                      "Processing completed. You may open the QGIS-project file in the output-folder." if flat_original_count is not None else "Processing incomplete. Press the \nprocessing button.",
                      "https://github.com/ragnvald/mesa/wiki/3-User-interface#processing")
        
        # Present status for atlas objects
        atlas_count = read_table_and_count('tbl_atlas')
        append_status("+" if atlas_count is not None else "/", 
                      f"Atlas pages: {atlas_count}" if atlas_count is not None else "Please create atlas.",
                      "https://github.com/ragnvald/mesa/wiki/5-Definitions#atlas")

        # Present status for calculations on segments
        segments_flat_count = read_table_and_count('tbl_segment_flat')
        lines_count = read_table_and_count('tbl_lines')
        append_status("+" if segments_flat_count is not None else "/", 
                      f"Segments are in place with {segments_flat_count} segments along {lines_count} lines." if segments_flat_count is not None else "Segments are missing.\nImport or initiate lines if you want to use\nthe line feature.",
                      "https://github.com/ragnvald/mesa/wiki/3-User-interface#lines-and-segments")

        # Convert the list of statuses to a DataFrame
        status_df = pd.DataFrame(status_list)
        return status_df

    except Exception as e:
        return pd.DataFrame({'Status': ['Error'], 'Message': [f"Error accessing statistics: {e}"], 'Link': [""]})

# -------------------------------
# Tool launchers (now using run_tool)
# -------------------------------
def geocodes_grids():
    run_tool("geocodes_create")

def import_assets(_gpkg_file):
    run_tool("data_import")

def edit_processing_setup():
    # correct helper name (matches your build list: parametres_setup.py)
    run_tool("parametres_setup")

def process_data(_gpkg_file):
    run_tool("data_process")

def make_atlas():
    run_tool("atlas_create")

def admin_lines():
    run_tool("lines_process")

def open_maps_overview():
    run_tool("maps_overview")

def open_data_analysis_setup():
    run_tool("data_analysis_setup")

def open_data_analysis_presentation():
    run_tool("data_analysis_presentation")

def open_present_files():
    run_tool("data_report")

def edit_assets():
    run_tool("assetgroup_edit")

def edit_geocodes():
    run_tool("geocodes_create")

def edit_lines():
    run_tool("lines_admin")

def edit_atlas():
    run_tool("atlas_edit")

def exit_program():
    root.destroy()

def update_config_with_values(config_file, **kwargs):
    # Read the entire config file to keep the layout and comments
    with open(config_file, 'r') as file:
        lines = file.readlines()

    # Update each key in kwargs if it exists, preserve layout
    for key, value in kwargs.items():
        found = False
        for i, line in enumerate(lines):
            if line.startswith(f'{key} ='):
                lines[i] = f"{key} = {value}\n"
                found = True
                break

    # Write the updated content back to the file
    with open(config_file, 'w') as file:
        file.writelines(lines)

def increment_stat_value(config_file, stat_name, increment_value):
    # Check if the config file exists
    if not os.path.isfile(config_file):
        log_to_logfile(f"Configuration file {config_file} not found.")
        return
    
    # Read the entire config file to preserve the layout and comments
    with open(config_file, 'r') as file:
        lines = file.readlines()
    
    # Initialize a flag to check if the variable was found and updated
    updated = False
    
    # Update the specified variable's value if it exists
    for i, line in enumerate(lines):
        if line.strip().startswith(f'{stat_name} ='):
            # Extract the current value, increment it, and update the line
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
    
    # Write the updated content back to the file if the variable was found and updated
    if updated:
        with open(config_file, 'w') as file:
            file.writelines(lines)

def add_text_to_labelframe(labelframe, text):
    label = tk.Label(labelframe, text=text, justify='left')
    label.pack(padx=10, pady=10, fill='both', expand=True)
    def update_wrap(event):
        label.config(wraplength=labelframe.winfo_width() - 20)
    labelframe.bind('<Configure>', update_wrap)

def apply_bootstyle(widget, style):
    try:
        widget.configure(bootstyle=style)
    except Exception:
        pass

def store_logs_online(
        log_host, 
        log_token, 
        log_org, 
        log_bucket, 
        id_uuid, 
        mesa_version,
        mesa_stat_startup, 
        mesa_stat_process, 
        mesa_stat_import_assets, 
        mesa_stat_import_geocodes, 
        mesa_stat_import_atlas, 
        mesa_stat_import_lines, 
        mesa_stat_setup, 
        mesa_stat_edit_atlas, 
        mesa_stat_create_atlas, 
        mesa_stat_process_lines
        ):
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

def store_userinfo_online(
        log_host, 
        log_token, 
        log_org, 
        log_bucket, 
        id_uuid, 
        id_name, 
        id_email
        ):
    if not is_connected():
        return "No network access, logs not updated"
    
    try:
        def write_point():
            client = InfluxDBClient(url=log_host, token=log_token, org=log_org)
            point = Point("tbl_user") \
                .tag("uuid", id_uuid) \
                .field("id_name", id_name) \
                .field("id_email", id_email)
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

#####################################################################################
#  Main
#
# Establish working directory that all helpers should use
original_working_directory  = os.getcwd()

# Load configuration settings
config_file                 = os.path.join(original_working_directory, "config.ini")
gpkg_file                   = os.path.join(original_working_directory, "output/mesa.gpkg")

# Setting variables
config                      = read_config(config_file)
ttk_bootstrap_theme         = config['DEFAULT']['ttk_bootstrap_theme']
mesa_version                = config['DEFAULT']['mesa_version']
workingprojection_epsg      = config['DEFAULT']['workingprojection_epsg']

log_date_initiated          = config['DEFAULT']['log_date_initiated']
log_date_lastupdate         = config['DEFAULT']['log_date_lastupdate']
log_org                     = config['DEFAULT']['log_org']
log_bucket                  = config['DEFAULT']['log_bucket']
log_host                    = config['DEFAULT']['log_host']
log_token                   = "Xp_sTOcg-46FFiQuplxz-Fqi-jEe5YGfOZarPR7gwZ4CMTMYseUPUjdKtp2xKV9w85TlBlh5X_lnaNzKULAhog=="

mesa_stat_startup           = config['DEFAULT']['mesa_stat_startup']
mesa_stat_process           = config['DEFAULT']['mesa_stat_process']
mesa_stat_import_assets     = config['DEFAULT']['mesa_stat_import_assets']
mesa_stat_import_geocodes   = config['DEFAULT']['mesa_stat_import_geocodes']
mesa_stat_import_atlas      = config['DEFAULT']['mesa_stat_import_atlas']
mesa_stat_import_lines      = config['DEFAULT']['mesa_stat_import_lines']
mesa_stat_setup             = config['DEFAULT']['mesa_stat_setup']
mesa_stat_edit_atlas        = config['DEFAULT']['mesa_stat_edit_atlas']
mesa_stat_create_atlas      = config['DEFAULT']['mesa_stat_create_atlas']
mesa_stat_process_lines     = config['DEFAULT']['mesa_stat_process_lines']

id_uuid = config['DEFAULT'].get('id_uuid', '').strip()
id_name = config['DEFAULT'].get('id_name', '').strip()
id_email = config['DEFAULT'].get('id_email', '').strip()
id_uuid_ok_value = config['DEFAULT'].get('id_uuid_ok', 'False').lower() in ('true', '1', 't')
id_personalinfo_ok_value = config['DEFAULT'].get('id_personalinfo_ok', 'False').lower() in ('true', '1', 't')

has_run_update_stats = False

# Function to handle the submission of the form
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

# Check and populate id_uuid / dates
if not id_uuid:
    id_uuid = str(uuid.uuid4())
    update_config_with_values(config_file, id_uuid=id_uuid)

if not log_date_initiated:
    log_date_initiated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_config_with_values(config_file, log_date_initiated=log_date_initiated)

if not log_date_lastupdate:
    log_date_lastupdate=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_config_with_values(config_file, log_date_lastupdate=log_date_lastupdate)

now = datetime.now()
log_date_lastupdate_dt = datetime.strptime(log_date_lastupdate, "%Y-%m-%d %H:%M:%S")

# Log if the number of hours exceeds hour limit
if ((now - log_date_lastupdate_dt) > timedelta(hours=1)) and (id_uuid_ok_value == True):
    storing_usage_message = store_logs_online(
        log_host, log_token, log_org, log_bucket, id_uuid,
        mesa_version, mesa_stat_startup, mesa_stat_process, mesa_stat_import_assets,
        mesa_stat_import_geocodes, mesa_stat_import_atlas, mesa_stat_import_lines,
        mesa_stat_setup, mesa_stat_edit_atlas, mesa_stat_create_atlas, mesa_stat_process_lines
    )
    log_to_logfile(storing_usage_message)

    storing_user_message = store_userinfo_online(log_host, log_token, log_org, log_bucket, id_uuid, id_name, id_email )
    log_to_logfile(storing_user_message)
    update_config_with_values(config_file, log_date_lastupdate=now.strftime("%Y-%m-%d %H:%M:%S"))

# Check and create folders at the beginning
check_and_create_folders()


if __name__ == "__main__":
    root = ttk.Window(themename=ttk_bootstrap_theme)
    root.title("MESA 5")
    try:
        root.iconbitmap(resource_path("system_resources", "mesa.ico"))
    except Exception:
        try:
            root.iconbitmap("system_resources/mesa.ico")
        except Exception:
            pass
    root.geometry("850x540")

    intro_text = (
        "Welcome to the MESA desktop. The Statistics tab is your live status board, while the Operations tab "
        "launches the main workflows for preparing and analysing data."
    )
    ttk.Label(root, text=intro_text, wraplength=780, justify="left", padding=(12, 8)).pack(
        fill="x", padx=12, pady=(12, 6)
    )

    notebook = ttk.Notebook(root, bootstyle=SECONDARY)
    notebook.pack(fill="both", expand=True, padx=12, pady=(0, 10))

    # ------------------------------------------------------------------
    # Statistics tab
    # ------------------------------------------------------------------
    stats_tab = ttk.Frame(notebook)
    notebook.add(stats_tab, text="Statistics")

    stats_container = ttk.Frame(stats_tab, padding=12)
    stats_container.pack(fill="both", expand=True)

    ttk.Label(
        stats_container,
        text="This overview updates automatically when you import data or run helper tools.",
        justify="left"
    ).pack(anchor="w", pady=(0, 8))

    global info_labelframe
    info_labelframe = ttk.LabelFrame(stats_container, text="Statistics and help", bootstyle='info')
    info_labelframe.pack(fill="both", expand=True)
    info_labelframe.grid_columnconfigure(0, weight=1)
    info_labelframe.grid_columnconfigure(1, weight=3)
    info_labelframe.grid_columnconfigure(2, weight=2)

    update_stats(gpkg_file)
    log_to_logfile("User interface, statistics updated.")

    # ------------------------------------------------------------------
    # Operations tab
    # ------------------------------------------------------------------
    operations_tab = ttk.Frame(notebook)
    notebook.add(operations_tab, text="Operations")

    operations_container = ttk.Frame(operations_tab, padding=12)
    operations_container.pack(fill="both", expand=True)

    ttk.Label(
        operations_container,
        text="Launch the main workflows from here. Each button has a short description of when to use it.",
        justify="left",
        wraplength=760
    ).pack(anchor="w", pady=(0, 10))

    button_width = 16
    button_padx = 6
    button_pady = 4

    operations_grid = ttk.Frame(operations_container)
    operations_grid.pack(fill="both", expand=True)
    operations_grid.columnconfigure(1, weight=1)

    operations = [
        ("Import", lambda: import_assets(gpkg_file),
         "Opens the asset and polygon importer. Start here when preparing a new dataset.", PRIMARY),
        ("Geocodes/grids", geocodes_grids,
         "Creates or refreshes geocode grids (hexagons, tiles) that support the analysis.", None),
        ("Set up", edit_processing_setup,
         "Adjust processing parameters such as buffers and thresholds before running analysis.", None),
        ("Process", lambda: process_data(gpkg_file),
         "Runs the core processing pipeline to produce the GeoParquet outputs.", PRIMARY),
        ("Atlas", make_atlas,
         "Generates atlas tiles and artefacts for map visualisations.", None),
        ("Segments", admin_lines,
         "Processes transport or utility lines into analysis segments.", None),
        ("Maps overview", open_maps_overview,
         "Opens the interactive map viewer with current background layers and assets.", PRIMARY),
        ("Analysis setup", open_data_analysis_setup,
         "Launches the area analysis tool used to define polygons and run clipping.", None),
        ("Analysis presentation", open_data_analysis_presentation,
         "Opens the comparison dashboard for analysis groups.", None),
        ("Report engine", open_present_files,
         "Builds printable reports and map packages for sharing with partners.", None),
    ]

    for idx, (label, command, description, bootstyle) in enumerate(operations):
        btn_kwargs = {"text": label, "command": command, "width": button_width}
        if bootstyle:
            btn_kwargs["bootstyle"] = bootstyle
        ttk.Button(operations_grid, **btn_kwargs).grid(
            row=idx, column=0, padx=(0, 12), pady=(button_pady, 2), sticky="w"
        )
        ttk.Label(
            operations_grid,
            text=description,
            wraplength=520,
            justify="left"
        ).grid(row=idx, column=1, padx=(0, 4), pady=(button_pady, 2), sticky="w")

    # ------------------------------------------------------------------
    # Settings tab
    # ------------------------------------------------------------------
    settings_tab = ttk.Frame(notebook)
    notebook.add(settings_tab, text="Settings")

    settings_container = ttk.Frame(settings_tab, padding=12)
    settings_container.pack(fill="both", expand=True)

    ttk.Label(
        settings_container,
        text="Some imported layers and helper artefacts can be refined from here.",
        justify="left",
        wraplength=660
    ).pack(anchor="w", pady=(0, 10))

    settings_grid = ttk.Frame(settings_container, padding=(10, 10))
    settings_grid.pack(fill="both", expand=True)
    settings_grid.columnconfigure(1, weight=1)

    settings_actions = [
        ("Edit assets", edit_assets,
         "Add titles and descriptions to imported asset layers for easier reference."),
        ("Edit geocodes", edit_geocodes,
         "Update grid cell metadata (for example H3 tessellations) used in analysis."),
        ("Edit lines", edit_lines,
         "Adjust line and segment parameters such as buffer sizes or naming conventions."),
        ("Edit atlas", edit_atlas,
         "Modify atlas polygons and their presentation settings for the QGIS project."),
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
    about_box.pack(fill='both', expand=True, padx=10, pady=10)
    about_text = (
        "Welcome to the MESA tool. The method is developed by UNEP-WCMC and the Norwegian Environment Agency. "
        "The software streamlines sensitivity analysis, reducing the likelihood of manual errors in GIS workflows."
        "\n\n"
        "Documentation and user guides are available on the MESA wiki: https://github.com/ragnvald/mesa/wiki"
        "\n\n"
        "This fourth-generation release incorporates feedback from workshops with partners in Ghana, Tanzania, "
        "Uganda, Mozambique, and Kenya. Lead programmer: Ragnvald Larsen - https://www.linkedin.com/in/ragnvald/"
    )
    add_text_to_labelframe(about_box, about_text)

    # ------------------------------------------------------------------
    # Registration tab
    # ------------------------------------------------------------------
    registration_tab = ttk.Frame(notebook)
    notebook.add(registration_tab, text="Register")

    registration_container = ttk.Frame(registration_tab, padding=12)
    registration_container.pack(fill="both", expand=True)

    id_uuid_ok = tk.BooleanVar(value=id_uuid_ok_value)
    id_personalinfo_ok = tk.BooleanVar(value=id_personalinfo_ok_value)

    registration_labelframe = ttk.LabelFrame(
        registration_container,
        text="Licensing and personal information",
        bootstyle='secondary'
    )
    registration_labelframe.pack(fill='both', expand=True, padx=5, pady=5)

    registration_text = (
        "MESA is open-source (GNU GPLv3). A random UUID helps us count anonymous usage. "
        "You may opt out of sending usage data and optionally supply your name and email for updates."
    )
    add_text_to_labelframe(registration_labelframe, registration_text)

    reg_grid = ttk.Frame(registration_container, padding=(10, 10))
    reg_grid.pack(fill='both', expand=True)
    reg_grid.columnconfigure(2, weight=1)

    uuid_ok_checkbox = ttk.Checkbutton(reg_grid, text="", variable=id_uuid_ok)
    uuid_ok_checkbox.grid(row=0, column=0, padx=10, pady=5, sticky="w")

    personalinfo_ok_checkbox = ttk.Checkbutton(reg_grid, text="", variable=id_personalinfo_ok)
    personalinfo_ok_checkbox.grid(row=1, column=0, padx=10, pady=5, sticky="w")

    ttk.Label(reg_grid, text="UUID:").grid(row=0, column=1, padx=10, pady=5, sticky="w")
    ttk.Label(reg_grid, text="Name:").grid(row=1, column=1, padx=10, pady=5, sticky="w")
    ttk.Label(reg_grid, text="Email:").grid(row=2, column=1, padx=10, pady=5, sticky="w")

    ttk.Label(reg_grid, text=id_uuid).grid(row=0, column=2, padx=10, pady=5, sticky="w")

    global name_entry, email_entry
    name_entry = ttk.Entry(reg_grid)
    name_entry.grid(row=1, column=2, padx=10, pady=5, sticky="we")
    name_entry.insert(0, id_name)

    email_entry = ttk.Entry(reg_grid)
    email_entry.grid(row=2, column=2, padx=10, pady=5, sticky="we")
    email_entry.insert(0, id_email)

    submit_btn = ttk.Button(reg_grid, text="Save", command=submit_form, bootstyle=SUCCESS)
    submit_btn.grid(row=2, column=3, padx=10, pady=5, sticky="e")

    reg_grid.columnconfigure(2, weight=1)

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    footer = ttk.Frame(root, padding=(10, 5))
    footer.pack(fill='x', padx=12, pady=(0, 6))

    ttk.Label(footer, text=mesa_version, font=("Calibri", 8)).pack(side='left')

    ttk.Button(footer, text="Exit", command=root.destroy, bootstyle="warning").pack(side='right')

    notebook.select(0)
    root.update_idletasks()
    root.mainloop()

