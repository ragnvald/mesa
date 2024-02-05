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
import uuid
import geopandas as gpd
import configparser
import datetime
import locale
from shapely.geometry import box, LineString, Point
from shapely.ops import unary_union
from shapely.geometry import mapping
import fiona
from fiona.crs import from_epsg

# Set locale
try:
    locale.setlocale(locale.LC_ALL, 'de_DE.utf8')
except locale.Error:
    locale.setlocale(locale.LC_ALL, '')

import os
import numpy as np
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


def load_lines_table(gpkg_file):

    # Check if the GeoPackage file exists
    if not os.path.exists(gpkg_file):
        return None  # File not found, return None without printing an error message

    try:
        # Attempt to load 'tbl_lines' from the specified GeoPackage file
        gdf = gpd.read_file(gpkg_file, layer='tbl_lines')
        
        # Check if the loaded GeoDataFrame is empty
        if gdf.empty:
            return None  # The table 'tbl_lines' is empty, return None
        else:
            return gdf
    except ValueError:
        return None  # The table 'tbl_lines' does not exist, return None


def create_lines_table_and_lines(gpkg_file):
    gdf_geocode_group = gpd.read_file(gpkg_file, layer='tbl_geocode_group', rows=1)
    if gdf_geocode_group.empty:
        print("The 'tbl_geocode_group' table is empty or does not exist.")
        return
    
    # Extract bounding box
    minx, miny, maxx, maxy = gdf_geocode_group.total_bounds
    
    # Generate three random lines within this bounding box
    lines = []
    for i in range(3):
        # Generate random start and end points within the bounding box
        start_x = np.random.uniform(minx, maxx)
        start_y = np.random.uniform(miny, maxy)
        end_x = np.random.uniform(minx, maxx)
        end_y = np.random.uniform(miny, maxy)
        line = LineString([(start_x, start_y), (end_x, end_y)])
        lines.append(line)
    
    # Create a GeoDataFrame for the lines
    gdf_lines = gpd.GeoDataFrame({
        'name_gis': [f'line_{i:03}' for i in range(1, 4)],
        'name_user': [f'line_{i:03}' for i in range(1, 4)],
        'segment_nr': [10, 10, 10],
        'segment_width': [1000, 1000, 1000],
        'description': ['another line', 'another line', 'another line'],
        'geometry': lines
    }, crs=gdf_geocode_group.crs)
    
    # Save to the GeoPackage
    gdf_lines.to_file(gpkg_file, layer='tbl_lines', if_exists='replace')


def buffer_lines(gpkg_file, line, name_gis, name_user, segment_width, segment_nr, description):

    # Ensure the line is in a GeoDataFrame and set the original CRS
    gdf = gpd.GeoDataFrame([[name_gis, name_user, segment_nr, description, line]],
                           columns=['name_gis', 'name_user', 'segment_nr', 'description', 'geometry'],
                           geometry='geometry', crs="EPSG:4326")
    
    segment_width = float(segment_width)

    # Reproject to EPSG:4087 for buffering
    gdf_projected = gdf.to_crs("EPSG:4087")
    
    # Buffer using the specified width in meters
    gdf_projected['geometry'] = gdf_projected.geometry.buffer(segment_width)
    
    # Reproject buffered geometries back to EPSG:4326
    gdf_buffered = gdf_projected.to_crs("EPSG:4326")
   
    # Prepare the GeoDataFrame for saving
    gdf_to_save = gpd.GeoDataFrame(gdf_buffered, geometry=gdf_buffered.geometry, crs=gdf_buffered.crs)
    
    # Save to GeoPackage
    gdf_to_save.to_file(gpkg_file, layer='tbl_lines_buffered', if_exists='append')


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


line_df = load_lines_table(gpkg_file)

if line_df is not None:
    log_to_gui(log_widget, "Lines exist")
    for index, row in line_df.iterrows():
        try:
    
            # Access the line geometry
            line            = row['geometry']
            name_gis        = row['name_gis'] 
            name_user       = row['name_user'] 
            segment_width   = row['segment_width'] 
            segment_nr      = row['segment_nr'] 
            description     = row['description'] 
            buffer_lines(gpkg_file, line, name_gis, name_user, segment_width, segment_nr, description)
            log_to_gui(log_widget, f"Added a buffered version of {name_gis}.")

        except Exception as e:
            log_to_gui(log_widget, f"Error processing line {index}: {e}")
else:
    log_to_gui(log_widget, "Lines do not exist. Will create three template lines.")
    create_lines_table_and_lines(gpkg_file)
    

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
