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
import uuid
import datetime
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient, Point, WriteOptions
from concurrent.futures import ThreadPoolExecutor, TimeoutError


# Read the configuration file
def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config


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
def update_stats(documentation_link):
    print(f"Update stats called with link: {documentation_link}")  # Debug output
    for widget in info_labelframe.winfo_children():
        widget.destroy()

    my_status = get_status(gpkg_file)

    if not my_status.empty and {'Status', 'Message'}.issubset(my_status.columns):
        for index, row in my_status.iterrows():
            status_label = ttk.Label(info_labelframe, text='\u26AB', justify='left', bootstyle='success' if row['Status'] == "+" else 'warning' if row['Status'] == "/" else 'danger')
            status_label.grid(row=index, column=0, sticky="nsew", padx=5, pady=5)

            message_label = ttk.Label(info_labelframe, text=row['Message'], justify='left')
            message_label.grid(row=index, column=1, sticky="nsew", padx=5, pady=5)

            if documentation_link:
                create_link_icon(info_labelframe, documentation_link, index, 2, 5, 5)

        root.update_idletasks()

    else:
        print("No status information available.")


def get_status(gpkg_file):
    if not os.path.exists(gpkg_file):
        return pd.DataFrame({'Status': ['Error'], 'Message': ["To initiate the system please import assets.\nStart doing this by pressing the Import-button.\nMake sure you have asset and geocode files\nstored in the respective folders."]})

    # Initialize an empty list to store each row of the DataFrame
    status_list = []

    try:
        # Count using an SQL-query. loading big data frames for counting them only is not
        # very efficient.
        def read_table_and_count(layer_name):
            """
            Counts the records in the specified layer of a GeoPackage.
            
            Parameters:
            - layer_name: The name of the layer to count records in.
            
            Returns:
            - The count of records in the layer, or None if an error occurs or the layer does not exist.
            """
            try:
                # Use a context manager to ensure the connection is closed automatically
                with sqlite3.connect(gpkg_file) as conn:
                    cur = conn.cursor()
                    # Check if the table exists to handle non-existent tables gracefully
                    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (layer_name,))
                    if cur.fetchone() is None:
                        log_to_logfile(f"Table {layer_name} does not exist.")
                        return None
                    # Execute a SQL query to count the records in the specified layer
                    cur.execute(f"SELECT COUNT(*) FROM {layer_name}")
                    count = cur.fetchone()[0]  # Fetch the count result
                    return count
            except sqlite3.Error as e:
                log_to_logfile(f"Error counting records in {layer_name}: {e}")
                return None

        def read_table_and_check_sensitivity(layer_name):
            try:
                table = gpd.read_file(gpkg_file, layer=layer_name)
                if table['sensitivity'].sum() > 0:
                    return "+", "Set up ok. Feel free to adjust it."
                else:
                    return "-", "You need to set up the calculation. \nPress the 'Set up'-button to proceed."
            except Exception:
                return None, None


        # Function to append status and message to the list
        def append_status(symbol, message):
            status_list.append({'Status': symbol, 'Message': message})


        # Check for tbl_asset_group
        asset_group_count = read_table_and_count('tbl_asset_group')
        append_status("+" if asset_group_count is not None else "-", f"Asset layers: {asset_group_count}" if asset_group_count is not None else "Assets are missing.\nImport assets by pressing the Import button.")

        # Check for tbl_geocode_group
        geocode_group_count = read_table_and_count('tbl_geocode_group')
        append_status("+" if geocode_group_count is not None else "/", f"Geocode layers: {geocode_group_count}" if geocode_group_count is not None else "Geocodes are missing.\nImport assets by pressing the Import button.")

        # Check for tbl_asset_group sensitivity
        symbol, message = read_table_and_check_sensitivity('tbl_asset_group')
        if symbol:
            append_status(symbol, message)

        # Check for tbl_geocode_group
        stacked_cells_count = read_table_and_count('tbl_stacked')
        flat_original_count = read_table_and_count('tbl_flat')
        append_status("+" if stacked_cells_count is not None else "-", f"Processing completed ({flat_original_count} / {stacked_cells_count})" if flat_original_count is not None else "Processing incomplete. Press the \nprocessing button.")
        
        # Check for tbl_geocode_group
        atlas_count = read_table_and_count('tbl_atlas')
        append_status("+" if atlas_count is not None else "/", f"Atlas pages: {atlas_count}" if atlas_count is not None else "Please create atlas.")

        # Check for tbl_geocode_group
        lines_original_count = read_table_and_count('tbl_lines_original')
        append_status("+" if lines_original_count is not None else "/", f"Lines in the system: {lines_original_count}" if lines_original_count is not None else "Lines are missing are missing.\nImport lines if you want to use the line feature.")

        # Convert the list of statuses to a DataFrame
        status_df = pd.DataFrame(status_list)
        
        return status_df

    except Exception as e:
        return pd.DataFrame({'Status': ['Error'], 'Message': [f"Error accessing statistics: {e}"]})


def run_subprocess(command, fallback_command):
    """ Utility function to run a subprocess with a fallback option. """
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            subprocess.run(fallback_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            log_to_logfile(f"Failed to execute command: {command}")


def import_assets():
    run_subprocess(["python", "01_import.py"], ["01_import.exe"])
    link = "https://www.mesamethod.org/wiki/Current_tool_version#Working_with_assets"
    update_stats(link)


def edit_asset_group():
    run_subprocess(["python", "04_edit_asset_group.py"], ["04_edit_asset_group.exe"])
    link = "https://www.mesamethod.org/wiki/Current_tool_version"
    update_stats(link)


def edit_geocode_group():
    run_subprocess(["python", "04_edit_geocode_group.py"], ["04_edit_geocode_group.exe"])
    link = "https://www.mesamethod.org/wiki/Current_tool_version#Geocodes"
    update_stats(link)


def edit_processing_setup():
    run_subprocess(["python", "04_edit_input.py"], ["04_edit_input.exe"])
    link = "https://www.mesamethod.org/wiki/Current_tool_version#Processing"
    update_stats(link)


def process_data():
    run_subprocess(["python", "06_process.py"], ["06_process.exe"])
    link = "https://www.mesamethod.org/wiki/Current_tool_version"
    update_stats(link)


def make_atlas():
    run_subprocess(["python", "07_make_atlas.py"], ["07_make_atlas.exe"])
    link = "https://www.mesamethod.org/wiki/Current_tool_version#Atlas"
    update_stats(link)


def edit_atlas():
    run_subprocess(["python", "07_edit_atlas.py"], ["07_edit_atlas.exe"])
    link = "https://www.mesamethod.org/wiki/Current_tool_version#Atlas"
    update_stats(link)


def admin_lines():
    run_subprocess(["python", "08_admin_lines.py"], ["08_admin_lines.exe"])
    link = "https://www.mesamethod.org/wiki/Current_tool_version#Working_with_lines"
    update_stats(link)


def edit_lines():
    run_subprocess(["python", "08_edit_lines.py"], ["08_edit_lines.exe"])
    link = "https://www.mesamethod.org/wiki/Current_tool_version#Working_with_lines"
    update_stats(link)


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
        print(f"Configuration file {config_file} not found.")
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
                    print(f"Error: Current value of {stat_name} is not an integer.")
                    return
    
    # Write the updated content back to the file if the variable was found and updated
    if updated:
        with open(config_file, 'w') as file:
            file.writelines(lines)


# Define functions for showing each frame
def show_main_frame():
    about_frame.pack_forget()
    registration_frame.pack_forget()
    main_frame.pack(fill='both', expand=True, pady=10)


def show_about_frame():
    main_frame.pack_forget()
    registration_frame.pack_forget()
    about_frame.pack(fill='both', expand=True)


def show_registration_frame():
    main_frame.pack_forget()
    about_frame.pack_forget()
    registration_frame.pack(fill='both', expand=True)


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

# Load configuration settings
config_file               = 'config.ini'
config                    = read_config(config_file)

gpkg_file                 = config['DEFAULT']['gpkg_file']
ttk_bootstrap_theme       = config['DEFAULT']['ttk_bootstrap_theme']
mesa_version              = config['DEFAULT']['mesa_version']
workingprojection_epsg    = config['DEFAULT']['workingprojection_epsg']

log_date_initiated        = config['DEFAULT']['log_date_initiated']
log_date_lastupdate       = config['DEFAULT']['log_date_lastupdate']
log_org                   = config['DEFAULT']['log_org']
log_bucket                = config['DEFAULT']['log_bucket']
log_host                  = config['DEFAULT']['log_host']
log_token                 = config['DEFAULT']['log_token']

mesa_stat_startup         = config['DEFAULT']['mesa_stat_startup']
mesa_stat_process         = config['DEFAULT']['mesa_stat_process']
mesa_stat_import_assets   = config['DEFAULT']['mesa_stat_import_assets']
mesa_stat_import_geocodes = config['DEFAULT']['mesa_stat_import_geocodes']
mesa_stat_import_atlas    = config['DEFAULT']['mesa_stat_import_atlas']
mesa_stat_import_lines    = config['DEFAULT']['mesa_stat_import_lines']
mesa_stat_setup           = config['DEFAULT']['mesa_stat_setup']
mesa_stat_edit_atlas      = config['DEFAULT']['mesa_stat_edit_atlas']
mesa_stat_create_atlas    = config['DEFAULT']['mesa_stat_create_atlas']
mesa_stat_process_lines   = config['DEFAULT']['mesa_stat_process_lines']

id_uuid                   = config['DEFAULT'].get('id_uuid', '').strip()
id_name                   = config['DEFAULT'].get('id_name', '').strip()
id_email                  = config['DEFAULT'].get('id_email', '').strip()
id_uuid_ok_value          = config['DEFAULT'].get('id_uuid_ok', 'False').lower() in ('true', '1', 't')
id_personalinfo_ok_value  = config['DEFAULT'].get('id_personalinfo_ok', 'False').lower() in ('true', '1', 't')

    
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
if ((now - log_date_lastupdate_dt) > timedelta(hours=24)) and (id_uuid_ok_value == True):
    # Parameters for store_logs_online function should be provided accordingly
    storing_usage_message = store_logs_online(log_host, log_token, log_org, log_bucket, id_uuid, mesa_version, mesa_stat_startup, mesa_stat_process, mesa_stat_import_assets, mesa_stat_import_geocodes, mesa_stat_import_atlas, mesa_stat_import_lines, mesa_stat_setup, mesa_stat_edit_atlas, mesa_stat_create_atlas, mesa_stat_process_lines)
    log_to_logfile(storing_usage_message)
    
    storing_user_message = store_userinfo_online(log_host, log_token, log_org, log_bucket, id_uuid, id_name, id_email )
    log_to_logfile(storing_user_message)
   
    # Update log_date_lastupdate with the current datetime, formatted as a string
    update_config_with_values(config_file, log_date_lastupdate=now.strftime("%Y-%m-%d %H:%M:%S"))
    
# Check and create folders at the beginning
check_and_create_folders()

# Setup the main Tkinter window
root = ttk.Window(themename=ttk_bootstrap_theme)
root.title("MESA 4")
root.geometry("800x540")

button_width = 18
button_padx  =  7
button_pady  =  7
button_text  = 'About...'

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
import_assets_btn = ttk.Button(left_panel, text="Import data", command=import_assets, width=button_width, bootstyle=PRIMARY)
import_assets_btn.grid(row=0, column=0, padx=button_padx, pady=button_pady)

edit_asset_group_btn = ttk.Button(left_panel, text="Edit asset groups", command=edit_asset_group, width=button_width, bootstyle=SECONDARY)
edit_asset_group_btn.grid(row=0, column=1, padx=button_padx, pady=button_pady)

edit_geocode_group_btn = ttk.Button(left_panel, text="Edit geocode groups", command=edit_geocode_group, width=button_width, bootstyle=SECONDARY)
edit_geocode_group_btn.grid(row=1, column=1, padx=button_padx, pady=button_pady)

edit_processing_setup_btn = ttk.Button(left_panel, text="Set up", command=edit_processing_setup, width=button_width)
edit_processing_setup_btn.grid(row=2, column=0, padx=button_padx, pady=button_pady)

process_stacked_data_btn = ttk.Button(left_panel, text="Process data", command=process_data, width=button_width)
process_stacked_data_btn.grid(row=3, column=0, padx=button_padx, pady=button_pady)

process_stacked_data_btn = ttk.Button(left_panel, text="Atlas", command=make_atlas, width=button_width)
process_stacked_data_btn.grid(row=4, column=0, padx=button_padx, pady=button_pady)

edit_asset_group_btn = ttk.Button(left_panel, text="Edit atlas", command=edit_atlas, width=button_width, bootstyle=SECONDARY)
edit_asset_group_btn.grid(row=4, column=1, padx=button_padx, pady=button_pady)

admin_lines_btn = ttk.Button(left_panel, text="Lines", command=admin_lines, width=button_width)
admin_lines_btn.grid(row=5, column=0, padx=button_padx, pady=button_pady)

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

log_to_logfile("User interface, statistics updated.")

update_stats(None)

###################################################
# About frame set up
#

# Adjusted Content for About Page
about_frame = ttk.Frame(root)  # This frame is for the alternate screen

increment_stat_value(config_file, 'mesa_stat_startup', increment_value=1)

# Create a HtmlFrame widget
html_frame = HtmlFrame(about_frame, horizontal_scrollbar="auto", messages_enabled=False)


# Define the path to your HTML content file
file_path = "system_resources/userguide.html"

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

id_uuid_ok               = tk.BooleanVar(value=id_uuid_ok_value)
id_personalinfo_ok       = tk.BooleanVar(value=id_personalinfo_ok_value)

# About label frame
about_labelframe = ttk.LabelFrame(registration_frame, text="Licensing and personal information", bootstyle='secondary')
about_labelframe.pack(side='top', fill='both', expand=True, padx=5, pady=5)

mesa_text = ("MESA 4.0 is open source software. It is available under the "
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
submit_btn = ttk.Button(grid_frame, text="Submit", command=submit_form)
submit_btn.grid(row=2, column=3, padx=10, pady=5, sticky="e")

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

# Start the GUI event loop
root.mainloop()