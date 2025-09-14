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
import pyarrow.parquet as pq  # <-- for GeoParquet status checks

# -------------------------------
# Launch helpers (frozen/dev)
# -------------------------------
def is_frozen() -> bool:
    """True when running inside a PyInstaller/onefile bundle."""
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')

def resource_path(*parts) -> str:
    """
    Resolve a path either from the PyInstaller temp dir (_MEIPASS) or the
    original working directory (set below).
    """
    base = sys._MEIPASS if is_frozen() else original_working_directory
    return os.path.join(base, *parts)

def run_tool(basename: str, *, wait: bool = True, pass_owd: bool = True):
    """
    Launch a helper either as a bundled EXE (frozen) or as a .py (dev).
    If wait=True we block and then call update_stats().
    """
    args = []
    if pass_owd:
        args.append(f'--original_working_directory={original_working_directory}')

    if is_frozen():
        exe = resource_path('tools', f'{basename}.exe')
        if not os.path.exists(exe):
            log_to_logfile(f"[run_tool] Missing bundled helper EXE: {exe}")
            return
        cmd = [exe] + args
    else:
        script = os.path.join(original_working_directory, f'{basename}.py')
        if os.path.exists(script):
            cmd = [sys.executable, script] + args
        else:
            # dev fallback: if someone compiled helpers locally into code/tools
            exe = os.path.join(original_working_directory, 'tools', f'{basename}.exe')
            if os.path.exists(exe):
                cmd = [exe] + args
            else:
                log_to_logfile(f"[run_tool] Missing helper: {script} / {exe}")
                return

    try:
        if wait:
            subprocess.run(cmd, check=True)
            update_stats(gpkg_file)
        else:
            subprocess.Popen(cmd)
    except Exception as e:
        log_to_logfile(f"[run_tool] Failed: {cmd} :: {e}")

# -------------------------------

# Read the configuration file
def read_config(file_name):
    config = configparser.ConfigParser()
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    config.read(os.path.join(base_path, file_name))
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


# This function updates the stats in the labelframe. Clear labels first,
# then write the updates.
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
                                           "Press the Import button.",
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
    run_tool("geocodes_create", wait=True)

def import_assets(gpkg_file):
    run_tool("data_import", wait=True)

def edit_processing_setup():
    # fixed name
    run_tool("parameters_setup", wait=True)

def process_data(gpkg_file):
    run_tool("data_process", wait=True)

def make_atlas():
    run_tool("atlas_create", wait=True)

def admin_lines():
    run_tool("lines_process", wait=True)

def open_maps_overview():
    run_tool("maps_overview", wait=False)

def open_present_files():
    run_tool("data_report", wait=False)

def edit_assets():
    run_tool("assetgroup_edit", wait=True)

def edit_geocodes():
    run_tool("geocodes_create", wait=True)

def edit_lines():
    run_tool("lines_admin", wait=True)

def edit_atlas():
    run_tool("atlas_edit", wait=True)


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
                    # Attempt to convert the current value to an integer and increment it
                    new_value = int(current_value) + increment_value
                    lines[i] = f"{stat_name} = {new_value}\n"
                    updated = True
                    break
                except ValueError:
                    # Handle the case where the conversion fails
                    log_to_logfile(f"Error: Current value of {stat_name} is not an integer.")
                    return
    
    # Write the updated content back to the file if the variable was found and updated
    if updated:
        with open(config_file, 'w') as file:
            file.writelines(lines)


# Define functions for showing each frame
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

    # Function to update the wraplength based on the width of the labelframe
    def update_wrap(event):
        label.config(wraplength=labelframe.winfo_width() - 20)

    # Bind the resize event of the labelframe to the update_wrap function
    labelframe.bind('<Configure>', update_wrap)


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
        # Function to execute the writing process
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

        # Using ThreadPoolExecutor to enforce a timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(write_point)
            # Adjust timeout as needed (2-3 seconds as per the requirement)
            future.result(timeout=3)

    except TimeoutError:
        return "No network access, logs not updated"
    
    except Exception as e:
        # Handle other exceptions, if needed
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
        # Function to execute the writing process
        def write_point():
            client = InfluxDBClient(url=log_host, token=log_token, org=log_org)
            point = Point("tbl_user") \
                .tag("uuid", id_uuid) \
                .field("id_name", id_name) \
                .field("id_email", id_email)

            write_api = client.write_api(write_options=WriteOptions(batch_size=1))
            write_api.write(bucket=log_bucket, org=log_org, record=point)

        # Using ThreadPoolExecutor to enforce a timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(write_point)
            # Adjust timeout as needed (2-3 seconds as per the requirement)
            future.result(timeout=3)

    except TimeoutError:
        return "No network access, logs not updated"
    except Exception as e:
        # Handle other exceptions, if needed
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
    global id_name, id_email  # If they're used globally; adjust according to your application's structure
    id_name = name_entry.get()
    id_email = email_entry.get()
    # Capture the current states of the checkboxes
    id_uuid_ok_str = str(id_uuid_ok.get())
    id_personalinfo_ok_str = str(id_personalinfo_ok.get())
    # Update the config file with these values
    update_config_with_values(config_file, 
                              id_uuid=id_uuid, 
                              id_name=id_name, 
                              id_email=id_email, 
                              id_uuid_ok=id_uuid_ok_str, 
                              id_personalinfo_ok=id_personalinfo_ok_str)
    

# Check and populate id_uuid if empty
if not id_uuid:  # if id_uuid is empty
    id_uuid = str(uuid.uuid4())  # Generate a new UUID
    update_config_with_values(config_file, id_uuid=id_uuid)  # Update the config file manually to preserve structure and comments

if not log_date_initiated:  # if log_date_initiated is empty
    log_date_initiated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_config_with_values(config_file, log_date_initiated=log_date_initiated)

if not log_date_lastupdate:  # if log_date_lastupdate is empty
    log_date_lastupdate=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_config_with_values(config_file, log_date_lastupdate=log_date_lastupdate)

now = datetime.now()
log_date_lastupdate_dt = datetime.strptime(log_date_lastupdate, "%Y-%m-%d %H:%M:%S")

# Log if the number of hours exceeds hour limit (hours =xx).
if ((now - log_date_lastupdate_dt) > timedelta(hours=1)) and (id_uuid_ok_value == True):
    # Parameters for store_logs_online function should be provided accordingly
    storing_usage_message = store_logs_online(log_host, log_token, log_org, log_bucket, id_uuid, mesa_version, mesa_stat_startup, mesa_stat_process, mesa_stat_import_assets, mesa_stat_import_geocodes, mesa_stat_import_atlas, mesa_stat_import_lines, mesa_stat_setup, mesa_stat_edit_atlas, mesa_stat_create_atlas, mesa_stat_process_lines)
    log_to_logfile(storing_usage_message)

    storing_user_message = store_userinfo_online(log_host, log_token, log_org, log_bucket, id_uuid, id_name, id_email )
    log_to_logfile(storing_user_message)
   
    # Update log_date_lastupdate with the current datetime, formatted as a string
    update_config_with_values(config_file, log_date_lastupdate=now.strftime("%Y-%m-%d %H:%M:%S"))
    
# Check and create folders at the beginning
check_and_create_folders()

if __name__ == "__main__":

    # Setup the main Tkinter window
    root = ttk.Window(themename=ttk_bootstrap_theme)
    root.title("MESA 5")
    # Make icon robust both dev/frozen (non-breaking if missing)
    try:
        root.iconbitmap(resource_path("system_resources", "mesa.ico"))
    except Exception:
        try:
            root.iconbitmap("system_resources/mesa.ico")
        except Exception:
            pass
    root.geometry("850x540")

    button_width = 18
    button_padx  =  7
    button_pady  =  7

    ###################################################
    # Main frame set up
    #
    main_frame = tk.Frame(root)
    main_frame.pack(fill='both', expand=True, pady=10)

    # Configure the grid weights
    main_frame.grid_columnconfigure(0, weight=0)  # Left panel has no weight
    main_frame.grid_columnconfigure(1, weight=0)  # Separator has no weight
    main_frame.grid_columnconfigure(2, weight=1)  # Right panel has weight

    # Left panel
    left_panel = tk.Frame(main_frame)
    left_panel.grid(row=0, column=0, sticky="nsew", padx=20)

    main_frame.grid_columnconfigure(0, minsize=220)  # Set minimum size for left panel

    # Add buttons to left panel with spacing between buttons
    import_assets_btn       = ttk.Button(left_panel, text="Import", command=lambda: import_assets(gpkg_file), width=button_width, bootstyle=PRIMARY)
    import_assets_btn.grid(row=0, column=0, padx=button_padx, pady=button_pady)
    
    geocodes_btn            = ttk.Button(left_panel, text="Geocodes/grids", command=geocodes_grids, width=button_width)
    geocodes_btn.grid(row=1, column=0, padx=button_padx, pady=button_pady)

    setup_processing_btn    = ttk.Button(left_panel, text="Set up", command=edit_processing_setup, width=button_width)
    setup_processing_btn.grid(row=2, column=0, padx=button_padx, pady=button_pady)

    process_data_btn        = ttk.Button(left_panel, text="Process", command=lambda: process_data(gpkg_file), width=button_width, bootstyle=PRIMARY)
    process_data_btn.grid(row=3, column=0, padx=button_padx, pady=button_pady)

    admin_atlas_btn         = ttk.Button(left_panel, text="Atlas", command=make_atlas, width=button_width)
    admin_atlas_btn.grid(row=4, column=0, padx=button_padx, pady=button_pady)

    admin_lines_btn         = ttk.Button(left_panel, text="Segments", command=admin_lines, width=button_width)
    admin_lines_btn.grid(row=5, column=0, padx=button_padx, pady=button_pady)

    # --- Maps Overview button ---
    maps_overview_btn       = ttk.Button(left_panel, text="Maps overview", command=open_maps_overview, width=button_width, bootstyle=PRIMARY)
    maps_overview_btn.grid(row=6, column=0, padx=button_padx, pady=button_pady)

    # --- New button for Present files (02_present_files.py) ---
    present_files_btn       = ttk.Button(left_panel, text="Report engine", command=open_present_files, width=button_width)
    present_files_btn.grid(row=7, column=0, padx=button_padx, pady=button_pady)

    # Separator
    separator = ttk.Separator(main_frame, orient='vertical')
    separator.grid(row=0, column=1, sticky='ns')

    # Right panel
    right_panel = ttk.Frame(main_frame)
    right_panel.grid(row=0, column=2, sticky="nsew", padx=5)

    # Configure the rows and columns within the right panel where widgets will be placed
    right_panel.grid_rowconfigure(0, weight=1)  # Adjust row for info_labelframe to grow
    right_panel.grid_columnconfigure(0, weight=1)  # Allow the column to grow

    # Info label frame (add this above the exit button)
    info_labelframe = ttk.LabelFrame(right_panel, text="Statistics and help", bootstyle='info')
    info_labelframe.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    info_labelframe.grid_columnconfigure(0, weight=1)  # For status symbols
    info_labelframe.grid_columnconfigure(1, weight=3)  # For messages
    info_labelframe.grid_columnconfigure(2, weight=2)  # For links
    
    update_stats(gpkg_file)

    log_to_logfile("User interface, statistics updated.")


    ###################################################
    # About frame set up
    #
    # Adjusted Content for About Page
    about_frame = ttk.Frame(root)  # This frame is for the alternate screen

    increment_stat_value(config_file, 'mesa_stat_startup', increment_value=1)

    # Create a HtmlFrame widget
    html_frame = HtmlFrame(about_frame, horizontal_scrollbar="auto", messages_enabled=False)

    # Define the path to your HTML content file
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    file_path = os.path.join(base_path, "system_resources/userguide.html")

    # Read the HTML content from the file
    with open(file_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    html_frame.load_html(html_content)

    # Pack the HtmlFrame into the ttkbootstrap window
    html_frame.pack(fill=BOTH, expand=YES)


    ###################################################
    # Registration frame set up
    #

    # Setup for the registration frame (assuming root is your Tk window)
    registration_frame = ttk.Frame(root)
    registration_frame.pack(fill='both', expand=True)

    id_uuid_ok = tk.BooleanVar(value=id_uuid_ok_value)
    id_personalinfo_ok = tk.BooleanVar(value=id_personalinfo_ok_value)

    # About label frame
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

    add_text_to_labelframe(about_labelframe, mesa_text)

    # Create a new frame for the grid layout within registration_frame
    grid_frame = ttk.Frame(registration_frame)
    grid_frame.pack(side='top', fill='both', expand=True, padx=5, pady=5)

    # Labels in the first column
    # Checkboxes in the first column
    uuid_ok_checkbox = ttk.Checkbutton(grid_frame, text="", variable=id_uuid_ok)
    uuid_ok_checkbox.grid(row=0, column=0, padx=10, pady=5, sticky="w")

    personalinfo_ok_checkbox = ttk.Checkbutton(grid_frame, text="", variable=id_personalinfo_ok)
    personalinfo_ok_checkbox.grid(row=1, column=0, padx=10, pady=5, sticky="w")

    # Labels for UUID, Name, Email in the second column
    ttk.Label(grid_frame, text="UUID:").grid(row=0, column=1, padx=10, pady=5, sticky="w")
    ttk.Label(grid_frame, text="Name:").grid(row=1, column=1, padx=10, pady=5, sticky="w")
    ttk.Label(grid_frame, text="Email:").grid(row=2, column=1, padx=10, pady=5, sticky="w")

    # UUID value, Name and Email entries in the third column
    ttk.Label(grid_frame, text=id_uuid).grid(row=0, column=2, padx=10, pady=5, sticky="w")
        
    name_entry = ttk.Entry(grid_frame)
    name_entry.grid(row=1, column=2, padx=10, pady=5, sticky="we")
    name_entry.insert(0, id_name)

    email_entry = ttk.Entry(grid_frame)
    email_entry.grid(row=2, column=2, padx=10, pady=5, sticky="we")
    email_entry.insert(0, id_email)

    # Submit button in the fourth column's bottom cell
    submit_btn = ttk.Button(grid_frame, text="Save", command=submit_form, bootstyle=SUCCESS)
    submit_btn.grid(row=2, column=3, padx=10, pady=5, sticky="e")

    # Optional: Configure the grid_frame column 2 (Entries) to take extra space
    grid_frame.columnconfigure(2, weight=1)

    # Setup for the registration frame (assuming root is your Tk window)
    settings_frame = ttk.Frame(root)
    settings_frame.pack(fill='both', expand=True)

    # About label frame
    about_labelframe = ttk.LabelFrame(settings_frame, text="Settings", bootstyle='info')
    about_labelframe.pack(side='top', fill='both', expand=True, padx=5, pady=5)

    mesa_text = ("Some of the objects you already have imported or created might need some further adjustments.\nYou may do this by reading up on the below suggestions.")

    add_text_to_labelframe(about_labelframe, mesa_text)

    # Create the bottom frame with grid layout
    grid_frame = ttk.Frame(settings_frame)
    grid_frame.pack(side='top', fill='both', expand=True, padx=20, pady=20)

    # Create buttons and labels manually

    # Edit assets button and label
    edit_polygons_btn = ttk.Button(grid_frame, text="Edit assets", command=edit_assets, bootstyle="primary")
    edit_polygons_btn.grid(row=1, column=0, padx=5, pady=5, sticky='ew')
    edit_polygons_lbl = ttk.Label(grid_frame, text="This is where you can add titles to the different layers you have imported. You may also add a short descriptive text.", wraplength=550)
    edit_polygons_lbl.grid(row=1, column=1, padx=5, pady=5, sticky='w')

    # Edit geocodes button and label
    edit_geocode_btn = ttk.Button(grid_frame, text="Edit geocodes", command=edit_geocodes, bootstyle="primary")
    edit_geocode_btn.grid(row=2, column=0, padx=5, pady=5, sticky='ew')
    edit_geocode_lbl = ttk.Label(grid_frame, text="Geocodes can be grid cells, hexagons or other types of polygons. You may add titles to them here for easier reference later.", wraplength=550)
    edit_geocode_lbl.grid(row=2, column=1, padx=5, pady=5, sticky='w')

    # Edit Lines button and label
    edit_lines_btn = ttk.Button(grid_frame, text="Edit lines", command=edit_lines, bootstyle="primary")
    edit_lines_btn.grid(row=3, column=0, padx=5, pady=5, sticky='ew')
    edit_lines_lbl = ttk.Label(grid_frame, text="Remember to import lines before attempting to edit them. Lines are processed to segments with the parameters length and width. Default values are set when the lines are imported. If you want to do the processing with other segment sizes you may do so here on a per line basis. This is where you can adjust the parameters as well as their names.", wraplength=550)
    edit_lines_lbl.grid(row=3, column=1, padx=5, pady=5, sticky='w')

    # Edit atlas button and label
    edit_atlas_btn = ttk.Button(grid_frame, text="Edit atlas", command=edit_atlas, bootstyle="primary")
    edit_atlas_btn.grid(row=4, column=0, padx=5, pady=5, sticky='ew')
    edit_atlas_lbl = ttk.Label(grid_frame, text="Remember to import or create atlases before attempting to edit them. Atlases are polygons which will be highlighted in the QGIS project file.", wraplength=550)
    edit_atlas_lbl.grid(row=4, column=1, padx=5, pady=5, sticky='w')


    # Optional: Configure the grid_frame column 1 (Labels) to take extra space
    grid_frame.columnconfigure(1, weight=1)

    # Optional: Configure the grid_frame column 2 (Entries) to take extra space
    grid_frame.columnconfigure(2, weight=1)


    ###################################################
    # Bottom frame in Main Interface for toggling to About Page
    #
    bottom_frame_buttons = ttk.Frame(root)
    bottom_frame_buttons.pack(side='bottom', fill='x', padx=10, pady=5)

    # Create buttons for each frame
    main_frame_btn = ttk.Button(bottom_frame_buttons, text="MESA desktop", command=show_main_frame, bootstyle="primary")
    main_frame_btn.pack(side='left', padx=(0, 10))

    settings_frame_btn = ttk.Button(bottom_frame_buttons, text="Settings", command=show_settings_frame, bootstyle="primary")
    settings_frame_btn.pack(side='left', padx=(0, 10))

    about_frame_btn = ttk.Button(bottom_frame_buttons, text="About...", command=show_about_frame, bootstyle="primary")
    about_frame_btn.pack(side='left', padx=(0, 10))

    registration_frame_btn = ttk.Button(bottom_frame_buttons, text="Register...", command=show_registration_frame, bootstyle="primary")
    registration_frame_btn.pack(side='left', padx=(0, 10))

    # Center frame for version label
    center_frame = ttk.Frame(bottom_frame_buttons)
    center_frame.pack(side='left', expand=True, fill='x')

    version_label = ttk.Label(center_frame, text=mesa_version, font=("Calibri", 7))
    version_label.pack(side='left', padx=50, pady=5)  # Adjust side and padding as needed

    # Continue with the Exit button and version label as before
    exit_btn = ttk.Button(bottom_frame_buttons, text="Exit", command=root.destroy, bootstyle="warning")
    exit_btn.pack(side='right')  # Assuming `root.destroy` for exiting

    show_main_frame()

    root.update_idletasks()

    root.mainloop()
