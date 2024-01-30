#
#
#  User selects input geocode group
#  List of lines is provided (tbl_line_in)
#   1) No tbl_line_input means a line should be created based on tbl_flat.
#   2) tbl_line_input has one of more lines
#  User selects lines
#  User selects:
#     - buffered width of line in meters
#     - line segment length
#  Feedback if selected combinations will overwrite existing data.
#  Analyse overlayng geocode with line segments / polygons
#     - results in tbl_line_out_flat (we might build a stacked one at a later stage...)
#
#

import geopandas as gpd
import configparser
import datetime
import locale

# Set locale
try:
    locale.setlocale(locale.LC_ALL, 'de_DE.utf8')
except locale.Error:
    locale.setlocale(locale.LC_ALL, '')

import os
import tkinter as tk
import tkinter.scrolledtext as scrolledtext
import ttkbootstrap as ttk
from ttkbootstrap import Style

# Read the configuration file
def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config

# Logging function to write to the GUI log
def log_to_gui(log_widget, message):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted_message = f"{timestamp} - {message}"
    log_widget.insert(tk.END, formatted_message + "\n")
    log_widget.see(tk.END)
    with open("log.txt", "a") as log_file:
        log_file.write(formatted_message + "\n")

def check_and_update_geopackage(gpkg_path, log_widget, geocode_table='tbl_geocode_group', line_table='tbl_line'):
    # Check if the GeoPackage file exists
    if not os.path.exists(gpkg_path):
        log_to_gui(log_widget, f"GeoPackage file {gpkg_path} not found.")
        return

    # Load the GeoPackage
    try:
        geopackage = gpd.read_file(gpkg_path, layer=None)
    except Exception as e:
        log_to_gui(log_widget, f"Failed to read GeoPackage: {e}")
        return

    # Check if tbl_line exists
    if line_table in geopackage:
        # List the lines
        lines = gpd.read_file(gpkg_path, layer=line_table)
        log_to_gui(log_widget, f"Lines in {line_table}:\n{lines.to_string()}")
    else:
        # Create a line in tbl_line based on the bounding box for all objects in tbl_geocode_group
        if geocode_table not in geopackage:
            log_to_gui(log_widget, f"{geocode_table} not found in the GeoPackage.")
            return

        geocode_data = gpd.read_file(gpkg_path, layer=geocode_table)
        bbox = geocode_data.total_bounds  # Get the bounding box
        new_line = gpd.GeoDataFrame(geometry=[bbox], columns=['id', 'name_gis', 'comment'])
        new_line.to_file(gpkg_path, layer=line_table, driver="GPKG")
        log_to_gui(log_widget, f"New line added to {line_table} based on {geocode_table} bounding box.")

def initiate_process(log_widget, config):
    gpkg_path = config['DEFAULT']['gpkg_file']
    check_and_update_geopackage(gpkg_path, log_widget)

# Load configuration settings
config_file = 'config.ini'
config = read_config(config_file)
ttk_bootstrap_theme = config['DEFAULT']['ttk_bootstrap_theme']

# Create the user interface using ttkbootstrap
style = Style(theme=ttk_bootstrap_theme)
root = style.master
root.title("Import assets")

# Create a LabelFrame for the log output
log_frame = ttk.LabelFrame(root, text="Log Output", bootstyle="info") 
log_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Create a log widget inside the LabelFrame
log_widget = scrolledtext.ScrolledText(log_frame, height=10)
log_widget.pack(fill=tk.BOTH, expand=True)

# Create a frame to hold the progress bar and the label
progress_frame = tk.Frame(root)
progress_frame.pack(pady=5)

# Create a progress bar
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=200, mode="determinate", variable=progress_var, bootstyle='info')
progress_bar.pack(side=tk.LEFT)

# Label for displaying the progress percentage
progress_label = tk.Label(progress_frame, text="0%", bg="light grey")
progress_label.pack(side=tk.LEFT, padx=5)

# Create the Initiate button
initiate_button = ttk.Button(root, text="Initiate", command=lambda: initiate_process(log_widget, config))
initiate_button.pack(pady=10)

root.mainloop()
