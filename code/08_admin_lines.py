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
from fiona.crs import from_epsg
from shapely.geometry import LineString, MultiLineString
from shapely.ops import transform
import pyproj
from functools import partial

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
        end_x   = np.random.uniform(minx, maxx)
        end_y   = np.random.uniform(miny, maxy)
        line    = LineString([(start_x, start_y), (end_x, end_y)])
        lines.append(line)
    
    # Create a GeoDataFrame for the lines
    gdf_lines = gpd.GeoDataFrame({
        'name_gis': [f'line_{i:03}' for i in range(1, 4)],
        'name_user': [f'line_{i:03}' for i in range(1, 4)],
        'segment_nr': [15, 30, 10],
        'segment_width': [1000, 20000, 5000],
        'description': ['another line', 'another line', 'another line'],
        'geometry': lines
    }, crs=gdf_geocode_group.crs)
    
    # Save to the GeoPackage
    gdf_lines.to_file(gpkg_file, layer='tbl_lines', if_exists='replace')


def process_and_buffer_line(gpkg_file, lines_df, crs="EPSG:4326", target_crs="EPSG:4087"):
    if lines_df is not None:
        # Initialize an empty list for storing buffered line dictionaries
        buffered_lines_data = []

        for index, row in lines_df.iterrows():
            try:
                # Extract the geometry and attributes for the current row
                fid = index
                gdf_line = row['geometry']
                name_gis = row['name_gis']
                name_user = row['name_user']
                segment_nr = row['segment_nr']
                segment_width = row['segment_width']
                description = row['description']
                
                # Create a temporary GeoDataFrame for buffering
                temp_gdf = gpd.GeoDataFrame([{'geometry': gdf_line}], geometry='geometry', crs=crs)
                
                # Reproject to target CRS for buffering
                temp_gdf_projected = temp_gdf.to_crs(target_crs)
                
                # Buffer using the specified width in meters
                temp_gdf_projected['geometry'] = temp_gdf_projected.buffer(segment_width)
                
                # Reproject buffered geometries back to original CRS
                temp_gdf_buffered = temp_gdf_projected.to_crs(crs)

                # Append the buffered geometry and attributes as a new dictionary
                buffered_lines_data.append({
                    'fid': fid, 
                    'name_gis': name_gis, 
                    'name_user': name_user, 
                    'segment_nr': segment_nr, 
                    'segment_width': segment_width, 
                    'description': description,
                    'geometry': temp_gdf_buffered.iloc[0].geometry  # Access the buffered geometry
                })

                log_to_gui(log_widget, f"Added a buffered version of {name_gis} to GeoPackage.")
                    
            except Exception as e:
                log_to_gui(log_widget, f"Error processing line {index}: {e}")

        # After the loop, create a GeoDataFrame from the list of dictionaries
        if buffered_lines_data:  # Check if there's any data to save
            all_buffered_lines_df = gpd.GeoDataFrame(buffered_lines_data, geometry='geometry', crs=crs)

            all_buffered_lines_df.to_file(gpkg_file, layer="tbl_lines_buffered", driver="GPKG")
            
            log_to_gui(log_widget, "All buffered lines added to GeoPackage.")

        else:
            log_to_gui(log_widget, "No lines were processed.")
      
    else:
        log_to_gui(log_widget, "Lines do not exist. Will create three template lines.")

        create_lines_table_and_lines(gpkg_file)


def create_perpendicular_lines(line_input, segment_width, segment_nr):
    # Define the projection transformation: EPSG:4326 to EPSG:4087 and back using the updated syntax
    transformer_to_4087 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4087", always_xy=True)
    transformer_to_4326 = pyproj.Transformer.from_crs("EPSG:4087", "EPSG:4326", always_xy=True)
    
    # Reproject line to EPSG:4087 using the updated transformer
    line_transformed = transform(transformer_to_4087.transform, line_input)
    
    # Your existing logic to calculate equal intervals along the line
    distances = [i/segment_nr * line_transformed.length for i in range(1, segment_nr + 1)]
    
    # Generate perpendicular lines
    perpendicular_lines = []
    for d in distances:
        point = line_transformed.interpolate(d)
        offset = line_transformed.parallel_offset(segment_width/2, 'left').interpolate(d)
        # Here, ensure to use the transformer_to_4326 for reprojection back to EPSG:4326
        line_coords = [transform(transformer_to_4326.transform, point), transform(transformer_to_4326.transform, offset)]
        perpendicular_line = LineString(line_coords)
        perpendicular_lines.append(perpendicular_line)
    
    # Reproject perpendicular lines back to EPSG:4326 and create a MultiLineString
    multi_line = MultiLineString(perpendicular_lines)
    
    return multi_line

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

#Load lines. If no lines three random lines will be made
lines_df = load_lines_table(gpkg_file)

#Buffer the lines and create a new table tbl_lines_buffered
process_and_buffer_line(gpkg_file,lines_df)

#ierate throuhg the lines
for index, row in lines_df.iterrows():
    line_input      = row.geometry
    segment_width   = row['segment_width']
    segment_nr      = row['segment_nr']

    perpendicular_lines = create_perpendicular_lines(line_input, segment_width, segment_nr)

    print (perpendicular_lines)
    print('- - - - - - - -')


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
