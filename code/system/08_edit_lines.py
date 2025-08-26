import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

import tkinter as tk
import configparser
import geopandas as gpd
import datetime
import argparse
import os
import ttkbootstrap as ttk  # ttkbootstrap UI
from ttkbootstrap.constants import *

# # # # # # # # # # # # # # 
# Shared/general functions

def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config

def write_to_log(message):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted_message = f"{timestamp} - {message}"
    log_destination_file = os.path.join(original_working_directory, "log.txt")
    with open(log_destination_file, "a", encoding="utf-8") as log_file:
        log_file.write(formatted_message + "\n")

# # # # # # # # # # # # # # 
# Core functions (GeoParquet-only)

def _parquet_lines_path() -> str:
    """Standard location for the lines layer in GeoParquet."""
    return os.path.join(original_working_directory, "output", "geoparquet", "tbl_lines.parquet")

# Load data from GeoParquet
def load_data():
    global parquet_lines_file
    parquet_lines_file = _parquet_lines_path()
    if not os.path.exists(parquet_lines_file):
        raise FileNotFoundError(f"GeoParquet file not found: {parquet_lines_file}")
    return gpd.read_parquet(parquet_lines_file)

# Save data to GeoParquet
def save_data(df):
    try:
        os.makedirs(os.path.dirname(parquet_lines_file), exist_ok=True)
        df.to_parquet(parquet_lines_file, index=False)
        write_to_log("Line data saved to GeoParquet")
    except Exception as e:
        write_to_log(f"Error saving data: {e}")
        print(f"Error saving data: {e}")

# Update current record and save
def update_record(save_message=True):
    try:
        df.at[current_index, 'name_gis'] = name_gis_var.get()
        df.at[current_index, 'name_user'] = name_user_var.get()
        df.at[current_index, 'segment_length'] = segment_length_var.get()
        df.at[current_index, 'segment_width'] = segment_width_var.get()
        df.at[current_index, 'description'] = description_var.get()
        save_data(df)
        if save_message:
            write_to_log("Record updated and saved")
    except Exception as e:
        write_to_log(f"Error updating and saving record: {e}")

# Navigation
def navigate(direction):
    global current_index
    update_record(save_message=False)  # persist edits before changing row
    if direction == 'next' and current_index < len(df) - 1:
        current_index += 1
    elif direction == 'previous' and current_index > 0:
        current_index -= 1
    load_record()

# Load a record into the form
def load_record():
    record = df.iloc[current_index]
    name_gis_var.set(record.get('name_gis', ''))
    name_user_var.set(record.get('name_user', ''))
    segment_length_var.set(record.get('segment_length', ''))
    segment_width_var.set(record.get('segment_width', ''))
    description_var.set(record.get('description', ''))

# Close the application
def exit_application():
    write_to_log("Closing edit segments")
    root.destroy()

#####################################################################################
#  Main
#####################################################################################

parser = argparse.ArgumentParser(description='Edit segments (GeoParquet)')
parser.add_argument('--original_working_directory', required=False, help='Path to running folder')
args = parser.parse_args()
original_working_directory = args.original_working_directory

# Resolve working directory if not provided
if not original_working_directory:
    original_working_directory = os.getcwd()
    if "system" in str(original_working_directory):
        original_working_directory = os.path.join(os.getcwd(), '../')

# Load configuration settings
config_file = os.path.join(original_working_directory, "system/config.ini")
config = read_config(config_file)

input_folder_asset   = os.path.join(original_working_directory, config['DEFAULT'].get('input_folder_asset', 'input/assets'))
input_folder_geocode = os.path.join(original_working_directory, config['DEFAULT'].get('input_folder_geocode', 'input/geocodes'))

ttk_bootstrap_theme    = config['DEFAULT'].get('ttk_bootstrap_theme', 'flatly')
workingprojection_epsg = config['DEFAULT'].get('workingprojection_epsg', '4326')

# UI
root = ttk.Window(themename=ttk_bootstrap_theme)
root.title("Edit segments")
try:
    root.iconbitmap(os.path.join(original_working_directory, "system_resources/mesa.ico"))
except Exception:
    pass

# Layout
root.columnconfigure(0, minsize=200)
root.columnconfigure(1, weight=1)

# Load data
try:
    df = load_data()
except FileNotFoundError as e:
    print(e)
    write_to_log(str(e))
    root.destroy()
    raise SystemExit(1)

current_index = 0

# Variables for form fields
name_gis_var        = tk.StringVar()
name_user_var       = tk.StringVar()
segment_length_var  = tk.StringVar()
segment_width_var   = tk.StringVar()
description_var     = tk.StringVar()

# GIS name (read-only)
tk.Label(root, text="GIS name").grid(row=0, column=0, sticky='w')
name_gis_label = tk.Label(root, textvariable=name_gis_var, width=50, relief="sunken", anchor="w")
name_gis_label.grid(row=0, column=1, sticky='w')

# User title
tk.Label(root, text="Title").grid(row=1, column=0, sticky='w')
name_user_entry = tk.Entry(root, textvariable=name_user_var, width=50)
name_user_entry.grid(row=1, column=1, sticky='w')

# Segment length
tk.Label(root, text="Length of segments").grid(row=2, column=0, sticky='w')
segment_length_entry = tk.Entry(root, textvariable=segment_length_var, width=50)
segment_length_entry.grid(row=2, column=1, sticky='w')

# Segment width
tk.Label(root, text="Segments width (meters)").grid(row=3, column=0, sticky='w')
segment_width_entry = tk.Entry(root, textvariable=segment_width_var, width=50)
segment_width_entry.grid(row=3, column=1, sticky='w')

# Description
tk.Label(root, text="Description").grid(row=4, column=0, sticky='w')
description_entry = tk.Entry(root, textvariable=description_var, width=50)
description_entry.grid(row=4, column=1, sticky='w')

# Info
info_label_text = (
    "Registered lines will be used to create segment polygons along the designated line. "
    "You may edit the attributes here, or both attributes and line data in QGIS. "
    "Edits are saved when you click Previous or Next."
)
info_label = tk.Label(root, text=info_label_text, wraplength=600, justify="left")
info_label.grid(row=5, column=0, columnspan=3, padx=10, pady=10)

# Buttons
ttk.Button(root, text="Previous", command=lambda: navigate('previous'), bootstyle=PRIMARY)\
   .grid(row=6, column=0, padx=10, pady=10)
ttk.Button(root, text="Next", command=lambda: navigate('next'), bootstyle=PRIMARY)\
   .grid(row=6, column=4, padx=10, pady=10)

ttk.Button(root, text="Save", command=lambda: update_record(save_message=False), bootstyle=SUCCESS)\
   .grid(row=7, column=4, columnspan=1, padx=10, pady=10)
ttk.Button(root, text="Exit", command=exit_application, bootstyle=WARNING)\
   .grid(row=7, column=5, columnspan=1, padx=10, pady=10)

# Initialize first record
load_record()

root.mainloop()
