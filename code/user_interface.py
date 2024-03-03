import tkinter as tk
from tkinter import *
import locale
import os
import json
import platform
import getpass
from datetime import datetime
import subprocess
import webbrowser
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import pandas as pd
import geopandas as gpd
import configparser
import sqlite3


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


# This function updates the stats in the labelframe. Clear labels first,
# then write the updates.
def update_stats():
    # Clear the labels before updating
    for widget in info_labelframe.winfo_children():
        widget.destroy()

    my_status = get_status(gpkg_file)

    # Check if the DataFrame is not empty and has the expected columns
    if not my_status.empty and {'Status', 'Message'}.issubset(my_status.columns):
        for index, row in my_status.iterrows():
            if row['Status'] == "+":
                # Green is for success - data has been added.
                status_label = ttk.Label(info_labelframe, text='\u26AB', justify='left', bootstyle='success')
                status_label.grid(row=index, column=0, sticky="nsew", padx=5, pady=5)
            elif row['Status'] == "/":
                # Orange is an option where it is not necessary to have registered data.
                status_label = ttk.Label(info_labelframe, text='\u26AB', justify='left', bootstyle='warning')
                status_label.grid(row=index, column=0, sticky="nsew", padx=5, pady=5)
            else:
                # Red is for data missing.
                status_label = ttk.Label(info_labelframe, text='\u26AB', justify='left', bootstyle='danger')
                status_label.grid(row=index, column=0, sticky="nsew", padx=5, pady=5)

            message_label = ttk.Label(info_labelframe, text=row['Message'], justify='left')
            message_label.grid(row=index, column=1, sticky="nsew", padx=5, pady=5)
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
        append_status("+" if stacked_cells_count is not None else "-", f"Processing success ({flat_original_count} / {stacked_cells_count})" if flat_original_count is not None else "Processing incomplete. Press the \nprocessing button.")
        
        # Check for tbl_geocode_group
        atlas_count = read_table_and_count('tbl_atlas')
        append_status("+" if atlas_count is not None else "/", f"Atlas pages: {atlas_count}" if atlas_count is not None else "Please create atlas.")

        # Check for tbl_geocode_group
        lines_original_count = read_table_and_count('tbl_lines_original')
        append_status("+" if lines_original_count is not None else "/", f"{lines_original_count} lines in place." if lines_original_count is not None else "Lines are missing are missing.\nImport lines if you want to use the line feature.")

        # Convert the list of statuses to a DataFrame
        status_df = pd.DataFrame(status_list)

        return status_df

    except Exception as e:
        return pd.DataFrame({'Status': ['Error'], 'Message': [f"Error accessing statistics: {e}"]})


def import_assets():
    try:
        # First try running import.py
        subprocess.run(["python", "01_import.py"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # If import.py fails or is not found, try running import.exe
            subprocess.run(["01_import.exe"], check=True)
        except subprocess.CalledProcessError:
            log_to_logfile("Failed to execute import assets script.")
    update_stats()


def edit_asset_group():
    try:
        subprocess.run(["python", "04_edit_asset_group.py"], check=True)
        log_to_logfile("Opened edit asset group")
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # If import.py fails, try running import.exe
            subprocess.run(["04_edit_asset_group.exe"], check=True)
        except subprocess.CalledProcessError:
            log_to_logfile("Failed to execute edit asset group script.")
    update_stats()


def edit_geocode_group():
    try:
        subprocess.run(["python", "04_edit_geocode_group.py"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # If import.py fails, try running import.exe
            subprocess.run(["04_edit_geocode_group.exe"], check=True)
        except subprocess.CalledProcessError:
            log_to_logfile("Failed to execute edit geocode group script.")
    update_stats()


def edit_processing_setup():
    try:
        subprocess.run(["python", "04_edit_input.py"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # If import.py fails, try running import.exe
            subprocess.run(["04_edit_input.exe"], check=True)
        except subprocess.CalledProcessError:
            log_to_logfile("Failed to execute edit input script.")
    update_stats()


def process_data():
    try:
        subprocess.run(["python", "06_process.py"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # If import.py fails, try running import.exe
            subprocess.run(["06_process.exe"], check=True)
        except subprocess.CalledProcessError:
            log_to_logfile("Failed to execute processing script.")
    update_stats()


def make_atlas():
    try:
        subprocess.run(["python", "07_make_atlas.py"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # If import.py fails, try running import.exe
            subprocess.run(["07_make_atlas.exe"], check=True)
        except subprocess.CalledProcessError:
            log_to_logfile("Failed to execute make atlas script.")
    update_stats()


def edit_atlas():
    try:
        subprocess.run(["python", "07_edit_atlas.py"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # If import.py fails, try running import.exe
            subprocess.run(["07_edit_atlas.exe"], check=True)
        except subprocess.CalledProcessError:
            log_to_logfile("Failed to execute edit atlas script.")
    update_stats()


def admin_lines():
    try:
        subprocess.run(["python", "08_admin_lines.py"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # If import.py fails, try running import.exe
            subprocess.run(["08_admin_lines.exe"], check=True)
        except subprocess.CalledProcessError:
            log_to_logfile("Failed to execute admin lines.")
    update_stats()


def edit_lines():
    try:
        subprocess.run(["python", "08_edit_lines.py"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # If import.py fails, try running import.exe
            subprocess.run(["08_edit_lines.exe"], check=True)
        except subprocess.CalledProcessError:
            log_to_logfile("Failed to execute edit lines script.")
    update_stats()


def exit_program():
    root.destroy()



# Function to dynamically update wraplength of label text
#def update_wraplength(event):
#    # Subtract some padding to ensure text does not touch frame borders
#    new_width = event.width - 20
#    stats_label.config(wraplength=new_width)
def add_text_to_labelframe(labelframe, text):
    label = tk.Label(labelframe, text=text, justify='left')
    label.pack(padx=10, pady=10, fill='both', expand=True)

    # Function to update the wraplength based on the width of the labelframe
    def update_wrap(event):
        label.config(wraplength=labelframe.winfo_width() - 20)

    # Bind the resize event of the labelframe to the update_wrap function
    labelframe.bind('<Configure>', update_wrap)

#####################################################################################
#  Main
#
    
# Load configuration settings
config_file             = 'config.ini'
config                  = read_config(config_file)
gpkg_file               = config['DEFAULT']['gpkg_file']
ttk_bootstrap_theme     = config['DEFAULT']['ttk_bootstrap_theme']
workingprojection_epsg  = config['DEFAULT']['workingprojection_epsg']

# Check and create folders at the beginning
check_and_create_folders()

# Setup the main Tkinter window
root = ttk.Window(themename=ttk_bootstrap_theme)
root.title("MESA 4")
root.geometry("800x540")

button_width = 18   
button_padx  =  7
button_pady  =  7

# Main frame
main_frame = tk.Frame(root)
main_frame.pack(fill='both', expand=True, pady=10)

# Configure the grid weights
main_frame.grid_columnconfigure(0, weight=0)  # Left panel has no weight
main_frame.grid_columnconfigure(1, weight=0)  # Separator has no weight
main_frame.grid_columnconfigure(2, weight=1)  # Right panel has weight

# Left panel
left_panel = tk.Frame(main_frame)
left_panel.grid(row=0, column=0, sticky="nsew", padx=20)

# Set minimum size for left panel
main_frame.grid_columnconfigure(0, minsize=220)  # Adjust the minsize as needed

# Add buttons to left panel with spacing between buttons
import_assets_btn = ttk.Button(left_panel, text="Import", command=import_assets, width=button_width, bootstyle=PRIMARY)
import_assets_btn.grid(row=0, column=0, padx=button_padx, pady=button_pady)

edit_asset_group_btn = ttk.Button(left_panel, text="Edit asset groups", command=edit_asset_group, width=button_width, bootstyle=SECONDARY)
edit_asset_group_btn.grid(row=0, column=1, padx=button_padx, pady=button_pady)

edit_geocode_group_btn = ttk.Button(left_panel, text="Edit geocode groups", command=edit_geocode_group, width=button_width, bootstyle=SECONDARY)
edit_geocode_group_btn.grid(row=1, column=1, padx=button_padx, pady=button_pady)

edit_processing_setup_btn = ttk.Button(left_panel, text="Set up", command=edit_processing_setup, width=button_width)
edit_processing_setup_btn.grid(row=2, column=0, padx=button_padx, pady=button_pady)

process_stacked_data_btn = ttk.Button(left_panel, text="Process", command=process_data, width=button_width)
process_stacked_data_btn.grid(row=3, column=0, padx=button_padx, pady=button_pady)

process_stacked_data_btn = ttk.Button(left_panel, text="Create atlas", command=make_atlas, width=button_width)
process_stacked_data_btn.grid(row=4, column=0, padx=button_padx, pady=button_pady)

edit_asset_group_btn = ttk.Button(left_panel, text="Edit atlas", command=edit_atlas, width=button_width, bootstyle=SECONDARY)
edit_asset_group_btn.grid(row=4, column=1, padx=button_padx, pady=button_pady)

admin_lines_btn = ttk.Button(left_panel, text="Work with lines", command=admin_lines, width=button_width)
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

log_to_logfile("User interface, statistics updated.")
update_stats()

# Bind the configure event to update the wraplength of the label
#info_labelframe.bind('<Configure>', update_wraplength)

# Adjust the exit button to align it to the right
exit_btn = ttk.Button(right_panel, text="Exit", command=exit_program, width=button_width, bootstyle="warning")
exit_btn.grid(row=1, column=0, pady=button_pady, sticky='e')  # Align to the right side

# Bottom panel
bottom_panel = ttk.Frame(root)
bottom_panel.pack(fill='both', expand=True, pady=5)

# About label frame
about_labelframe = ttk.LabelFrame(bottom_panel, text="About", bootstyle='secondary')
about_labelframe.pack(side='top', fill='both', expand=True, padx=5, pady=5)

mesa_text = """This version of the MESA tool is a stand-alone desktop based version prepared for use on the Windows platform. To use it you will have to deposit spatial data for assets and geocodes (e.g., grids). The result of the processing is a sensitivity data set. To balance the resulting scores you will have to provide values for assets and their associated susceptibilities."""

add_text_to_labelframe(about_labelframe, mesa_text)

# Version label aligned bottom right
version_label = ttk.Label(bottom_panel, text="MESA version 4.0.0-alpha", font=("Calibri", 7), anchor='e')
version_label.pack(side='bottom', anchor='e', padx=10, pady=5)

log_to_logfile("User interface, main dialogue opened.")

# Start the GUI event loop
root.mainloop()