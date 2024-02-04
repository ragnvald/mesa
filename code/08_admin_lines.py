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
import fiona
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



def check_and_update_geopackage(gpkg_path, log_widget):

    geocode_table='tbl_geocode_group'
    line_table='tbl_line'

    if not os.path.exists(gpkg_path):
        log_to_gui(log_widget, f"GeoPackage file {gpkg_path} not found.")
        return

    try:
        # Check if the GeoPackage contains the specified tables
        layer_list = gpd.io.file.fiona.listlayers(gpkg_path)
        
        if geocode_table not in layer_list:
            log_to_gui(log_widget, f"{geocode_table} not found in the GeoPackage.")
            return

        if line_table not in layer_list:
            log_to_gui(log_widget, f"{line_table} not found in the GeoPackage.")
            return

        # Read and process lines from the line table within the GeoPackage
        lines = gpd.read_file(gpkg_path, layer=line_table)
        
        # Count and list lines, excluding the geometry attribute
        line_count = 0
        for _, line in lines.iterrows():
            line_info = line.drop('geometry').to_dict()
            line_count += 1
            log_to_gui(log_widget, f"Line {line_count}: {line_info}")

        log_to_gui(log_widget, f"Total number of lines in {line_table}: {line_count}")

        return lines

    except Exception as e:
        log_to_gui(log_widget, f"Error accessing GeoPackage: {e}")
        return



def read_lines_from_geopackage(gpkg_local):
    """
    Reads lines from a specified table within a GeoPackage and returns them as a DataFrame.
    
    Parameters:
    - gpkg_path: Path to the GeoPackage file.
    - line_table: Name of the table containing lines.
    
    Returns:
    - A DataFrame containing the lines from the specified table, excluding the geometry attribute.
      Returns None if the GeoPackage or the specified table is not found or an error occurs.
    """
    
    line_table='tbl_line'

    if not os.path.exists(gpkg_local):
        print(f"GeoPackage file {gpkg_local} not found.")
        return None

    try:
        # Check if the GeoPackage contains the specified table
        layer_list = gpd.read_file(gpkg_local,log_widget, line_table)
        
        if line_table not in layer_list:
            print(f"{line_table} not found in the GeoPackage.")
            return None

        # Read lines from the line table within the GeoPackage
        lines = gpd.read_file(gpkg_local, layer=line_table)
        
        # Exclude the geometry attribute and return as a DataFrame
        lines_df = lines.drop(columns=['geometry'])
        
        return lines_df

    except Exception as e:
        print(f"Error accessing GeoPackage: {e}")
        return None


#####################################################################################
#  Main
#

# Load configuration settings
config_file = 'config.ini'
config = read_config(config_file)
input_folder_asset = config['DEFAULT']['input_folder_asset']
input_folder_geocode = config['DEFAULT']['input_folder_geocode']
gpkg_file = config['DEFAULT']['gpkg_file']
ttk_bootstrap_theme = config['DEFAULT']['ttk_bootstrap_theme']

# Create the user interface using ttkbootstrap
style = Style(theme=ttk_bootstrap_theme)
root = style.master
root.title("Line processing")

# Create a LabelFrame for the log output
log_frame = ttk.LabelFrame(root, text="Log Output", bootstyle="info") 
log_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Create a log widget inside the LabelFrame
log_widget = scrolledtext.ScrolledText(log_frame, height=10)
log_widget.pack(fill=tk.BOTH, expand=True)


line_df = read_lines_from_geopackage(gpkg_file)
row_count = len(line_df)
print (row_count)

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
initiate_button = ttk.Button(root, text="Initiate", command=lambda: check_and_update_geopackage(gpkg_file, log_widget))
initiate_button.pack(pady=10)

root.mainloop()
