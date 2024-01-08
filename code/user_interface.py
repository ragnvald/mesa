import tkinter as tk
import locale

try:
    locale.setlocale(locale.LC_ALL, 'de_DE.utf8')  # For US English, adjust as needed
except locale.Error:
    locale.setlocale(locale.LC_ALL, '') 

from tkinter import messagebox
import subprocess
import webbrowser
import datetime
import os
import fiona
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap import LabelFrame
import geopandas as gpd
import configparser


# Read the configuration file
def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config


# Function to check and create folders
def check_and_create_folders():
    folders = ["input/geocode", "output", "qgis"]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


def log_to_logfile(message):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted_message = f"{timestamp} - {message}"
    with open("log.txt", "a") as log_file:
        log_file.write(formatted_message + "\n")


# Function to open a web link
def open_link(url):
    webbrowser.open_new_tab(url)


def update_stats():
    my_status = get_status(gpkg_file)
    stats_label.config(text=my_status)
    # Schedule the update_stats function to be called again after 30000 milliseconds (30 seconds)


def get_status(gpkg_file):
    if not os.path.exists(gpkg_file):
        return "To initiate the system please import assets. Do this by pressing the Import-button. Make sure you have asset and geocode files stored in the respective folders."

    stats_text = ""

    try:
        def read_table_and_count(layer_name):
            try:
                table = gpd.read_file(gpkg_file, layer=layer_name)
                return len(table)
            except Exception:
                return None

        def read_table_and_check_sensitivity(layer_name):
            try:
                table = gpd.read_file(gpkg_file, layer=layer_name)
                if  table['sensitivity'].sum() > 0:
                    return "Everything is set up. Ready for processing."
                else:
                    return "You need to set up the calculation. Press the 'Set up'-button to proceed."
            except Exception:
                return None

         # Check for tbl_asset_group
        asset_group_count = read_table_and_count('tbl_asset_group')
        if asset_group_count is not None:
            stats_text += f"Asset layers: {asset_group_count}\n"
        else:
            stats_text += "Assets are missing. Import assets by pressing the Import button.\n"

        # Check for tbl_geocode_group
        geocode_group_count = read_table_and_count('tbl_geocode_group')
        if geocode_group_count is not None:
            stats_text += f"Geocode layers: {geocode_group_count}\n"
        else:
            stats_text += "Geocodes are missing. Import assets by pressing the Import button.\n"

        # Check for tbl_asset_group sensitivity
        asset_group_sensitivity = read_table_and_check_sensitivity('tbl_asset_group')
        if asset_group_sensitivity:
            stats_text += f"{asset_group_sensitivity}\n"

        # Check for tbl_stacked
        stacked_cells_count = read_table_and_count('tbl_stacked')
        if stacked_cells_count is not None:
            stats_text += f"Stacked cells: {stacked_cells_count}\n"
        else:
            stats_text += "Stacked table is missing. Press button Process data to initiate.\n"

        # Check for tbl_flat
        stacked_cells_count = read_table_and_count('tbl_flat')
        if stacked_cells_count is not None:
            stats_text += f"Flat cells: {stacked_cells_count}\n"
        else:
            stats_text += "Flat table is missing. Press button Process data to initiate.\n"

        # Check for tbl_atlas
        stacked_cells_count = read_table_and_count('tbl_atlas')
        if stacked_cells_count is not None:
            stats_text += f"Atlas pages: {stacked_cells_count}\n"
        else:
            stats_text += "Atlas is missing. Press button 'Create atlas'.\n"


        return stats_text.strip()

    except Exception as e:
        return f"Error accessing statistics: {e}"



def import_assets():
    try:
        # First try running import.py
        subprocess.run(["python", "01_import.py"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # If import.py fails or is not found, try running import.exe
            subprocess.run(["01_import.exe"], check=True)
        except subprocess.CalledProcessError:
            log_to_logfile("Failed to execute import assets script")
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
            log_to_logfile("Failed to execute edit asset group script")
    update_stats()


def edit_geocode_group():
    try:
        subprocess.run(["python", "04_edit_geocode_group.py"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # If import.py fails, try running import.exe
            subprocess.run(["04_edit_geocode_group.exe"], check=True)
        except subprocess.CalledProcessError:
            log_to_logfile("Failed to execute edit geocode group script")
    update_stats()


def edit_processing_setup():
    try:
        subprocess.run(["python", "04_edit_input.py"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # If import.py fails, try running import.exe
            subprocess.run(["04_edit_input.exe"], check=True)
        except subprocess.CalledProcessError:
            log_to_logfile("Failed to execute edit input script")
    update_stats()


def process_data():
    try:
        subprocess.run(["python", "06_process.py"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # If import.py fails, try running import.exe
            subprocess.run(["06_process.exe"], check=True)
        except subprocess.CalledProcessError:
            log_to_logfile("Failed to execute processing script")
    update_stats()


def make_atlas():
    try:
        subprocess.run(["python", "07_make_atlas.py"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # If import.py fails, try running import.exe
            subprocess.run(["07_make_atlas.exe"], check=True)
        except subprocess.CalledProcessError:
            log_to_logfile("Failed to execute make atlas script")
    update_stats()


def edit_atlas():
    try:
        subprocess.run(["python", "07_edit_atlas.py"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # If import.py fails, try running import.exe
            subprocess.run(["07_edit_atlas.exe"], check=True)
        except subprocess.CalledProcessError:
            log_to_logfile("Failed to execute edit atlas script")
    update_stats()


def export_qgis():
    messagebox.showinfo("Export Package", "Export package script executed.")


def exit_program():
    root.destroy()

# Function to dynamically update wraplength of label text
def update_wraplength(event):
    # Subtract some padding to ensure text does not touch frame borders
    new_width = event.width - 20
    stats_label.config(wraplength=new_width)

def add_text_to_labelframe(labelframe, text):
    label = tk.Label(labelframe, text=text, justify='left')
    label.pack(padx=10, pady=10, fill='both', expand=True)

    # Function to update the wraplength based on the width of the labelframe
    def update_wrap(event):
        label.config(wraplength=labelframe.winfo_width() - 20)

    # Bind the resize event of the labelframe to the update_wrap function
    labelframe.bind('<Configure>', update_wrap)



# Load configuration settings
config_file = 'config.ini'
config = read_config(config_file)
gpkg_file = config['DEFAULT']['gpkg_file']
ttk_bootstrap_theme = config['DEFAULT']['ttk_bootstrap_theme']

# Check and create folders at the beginning
check_and_create_folders()

# Setup the main Tkinter window
root = ttk.Window(themename=ttk_bootstrap_theme)
root.title("MESA 4")
root.geometry("800x540")

button_width = 18   
button_padx  = 7
button_pady  = 7

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

export_qgis_btn = ttk.Button(left_panel, text="Export QGIS file", command=export_qgis, width=button_width)
export_qgis_btn.grid(row=5, column=0, padx=button_padx, pady=button_pady)

# Separator
separator = ttk.Separator(main_frame, orient='vertical')
separator.grid(row=0, column=1, sticky='ns')

# Right panel
right_panel = tk.Frame(main_frame)
right_panel.grid(row=0, column=2, sticky="nsew", padx=5)

# Configure the rows and columns within the right panel where widgets will be placed
right_panel.grid_rowconfigure(0, weight=1)  # Adjust row for info_labelframe to grow
right_panel.grid_columnconfigure(0, weight=1)  # Allow the column to grow

# Info label frame (add this above the exit button)
info_labelframe = ttk.LabelFrame(right_panel, text="Statistics and help", bootstyle='info')
info_labelframe.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)


# Get text for the stats-label
my_status = get_status(gpkg_file)

# Add the stats text inside the info_labelframe, aligned to the top
stats_label = tk.Label(info_labelframe, text=my_status, justify='left')
stats_label.pack(side='top', padx=5, pady=5, fill='x')  # Align to the top and expand horizontally

# Bind the configure event to update the wraplength of the label
info_labelframe.bind('<Configure>', update_wraplength)


# Adjust the exit button to align it to the right
exit_btn = ttk.Button(right_panel, text="Exit", command=exit_program, width=button_width, bootstyle=WARNING)
exit_btn.grid(row=1, column=0, pady=button_pady, sticky='e')  # Align to the right side



# Bottom panel
bottom_panel = tk.Frame(root)
bottom_panel.pack(fill='both', expand=True, pady=5)

# About label frame
about_labelframe = ttk.LabelFrame(bottom_panel, text="About", bootstyle='secondary')
about_labelframe.pack(side='top', fill='both', expand=True, padx=5, pady=5)

mesa_text = """This version of the MESA tool is a stand-alone desktop based version prepared for use on the Windows platform. To use it you will have to deposit spatial data for assets and geocodes (e.g., grids). The result of the processing is a sensitivity data set. To balance the resulting scores you will have to provide values for assets and their associated susceptibilities."""

add_text_to_labelframe(about_labelframe, mesa_text)

# Version label aligned bottom right
version_label = tk.Label(bottom_panel, text="MESA version 3.5 beta", font=("Calibri", 7), anchor='e')
version_label.pack(side='bottom', anchor='e', padx=10, pady=5)

log_to_logfile("User interface, main dialogue opened")

update_stats()

# Start the GUI event loop
root.mainloop()