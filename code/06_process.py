# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# This is where evertthing comes together for the gridded assets.
#

import tkinter as tk
import locale
from tkinter import scrolledtext, ttk
import threading
import geopandas as gpd
import pandas as pd
import configparser
import datetime
import ttkbootstrap as ttk  # Import ttkbootstrap
from ttkbootstrap.constants import *

# # # # # # # # # # # # # # 
# Shared functions

# Read the configuration file
def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config

def read_config_classification(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    classification = {}
    for section in config.sections():
        range_str = config[section]['range']
        start, end = map(int, range_str.split('-'))
        classification[section] = range(start, end + 1)
    return classification


# # # # # # # # # # # # # # 
# Core functions


def update_progress(new_value):
    progress_var.set(new_value)
    progress_label.config(text=f"{int(new_value)}%")


# Function to close the application
def close_application(root):
    root.destroy()


# Logging function to write to the GUI log
def log_to_gui(log_widget, message):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted_message = f"{timestamp} - {message}"
    log_widget.insert(tk.END, formatted_message + "\n")
    log_widget.see(tk.END)
    with open("log.txt", "a") as log_file:
        log_file.write(formatted_message + "\n")


# # # # # # # # # # # # # # 
# Spatial functions


# Function to perform intersection with geocode data
def intersection_with_geocode_data(asset_df, geocode_df, geom_type, log_widget):
    log_to_gui(log_widget, f"Processing {geom_type} intersections")
    asset_filtered = asset_df[asset_df.geometry.geom_type == geom_type]

    if asset_filtered.empty:
        return gpd.GeoDataFrame()

    return gpd.sjoin(geocode_df, asset_filtered, how='inner', predicate='intersects')


# Function to aggregate data by code
def aggregate_data(intersected_data):
    # Group by code and calculate aggregates
    aggregation_functions = {
        'importance': ['min', 'max'],
        'sensitivity': ['min', 'max'],
        'susceptibility': ['min', 'max'],
        'ref_geocodegroup': 'first',                # Include the first reference to geocode group id
        'name_gis_geocodegroup': 'first',           # Include the first reference to geocode group id
        'name': 'first',                            # Include the first name for each group
        'geometry': 'first',                        # Keeping the first geometry for each group
        'asset_group_name': lambda x: '; '.join(x)  # Concatenating asset_group_name
    }

    # Check if asset_group_name column exists
    if 'asset_group_name' in intersected_data.columns:
        grouped = intersected_data.groupby('code').agg(aggregation_functions)
    else:
        # Remove asset_group_name aggregation if the column does not exist
        aggregation_functions.pop('asset_group_name')
        grouped = intersected_data.groupby('code').agg(aggregation_functions)

    # Flatten MultiIndex columns
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

    # Rename columns after flattening
    renamed_columns = {
        'name_first': 'geocode_name_user',
        'ref_geocodegroup_first': 'ref_geocodegroup',
        'name_gis_first': 'name_gis_geocodegroup', 
        'asset_group_name_<lambda>': 'asset_group_names'  
    }

    grouped.rename(columns=renamed_columns, inplace=True)

    # Count the total assets in each cell (code is the unique identifier)
    grouped['assets_total'] = intersected_data.groupby('code').size()

    return grouped


# Create tbl_stacked by intersecting all asset data with the geocoding data
def main_tbl_stacked(log_widget, progress_var, gpkg_file):
    log_to_gui(log_widget, "Building data mining database (tbl_stacked).")
    update_progress(10)  # Indicate start

    asset_data = gpd.read_file(gpkg_file, layer='tbl_asset_object')
    update_progress(15)  # Progress after reading asset data

    geocode_data = gpd.read_file(gpkg_file, layer='tbl_geocode_object')
    update_progress(20)  # Progress after reading geocode data

    asset_group_data = gpd.read_file(gpkg_file, layer='tbl_asset_group')
    update_progress(25)  # Progress after reading asset group data

    # Merge asset group data with asset data
    asset_data = asset_data.merge(asset_group_data[['id', 'name_gis_assetgroup', 'total_asset_objects', 'importance', 'susceptibility', 'sensitivity']], 
                                  left_on='ref_asset_group', right_on='id', how='left')
   
    update_progress(30)  # Progress after merging data

    point_intersections = intersection_with_geocode_data(asset_data, geocode_data, 'Point', log_widget)
    update_progress(35)  # Progress after point intersections

    line_intersections = intersection_with_geocode_data(asset_data, geocode_data, 'LineString', log_widget)
    update_progress(40)  # Progress after line intersections

    polygon_intersections = intersection_with_geocode_data(asset_data, geocode_data, 'Polygon', log_widget)
    update_progress(43)  # Progress after polygon intersections

    intersected_data = pd.concat([point_intersections, line_intersections, polygon_intersections])
    # To list the columns:
    columns_list = intersected_data.columns.tolist()

    # To print the list of columns:
    print(columns_list)
    
    update_progress(45)  # Progress after concatenating data
    
    # Drop the unnecessary columns
    intersected_data.drop(columns=['fid', 'id_x', 'id_y', 'total_asset_objects', 'process', 'index_right'], inplace=True)

    intersected_data.to_file(gpkg_file, layer='tbl_stacked', driver='GPKG')
    log_to_gui(log_widget, "Data processing done.")
    update_progress(50)  # Final progress


# Create tbl_flat by reading out values from tbl_stacked
def main_tbl_flat(log_widget, progress_var, gpkg_file):
    
    log_to_gui(log_widget, "Building map database (tbl_flat).")

    tbl_stacked = gpd.read_file(gpkg_file, layer='tbl_stacked')

    update_progress(60)

    # Aggregation functions
    aggregation_functions = {
        'importance': ['min', 'max'],
        'sensitivity': ['min', 'max'],
        'susceptibility': ['min', 'max'],
        'ref_geocodegroup': 'first',
        'name_gis_geocodegroup': 'first',
        'asset_group_name': lambda x: ', '.join(x.unique()),  # Joining into a comma-separated string
        'ref_asset_group': 'nunique',
        'geometry': 'first'
    }

    # Group by 'code' and aggregate
    tbl_flat = tbl_stacked.groupby('code').agg(aggregation_functions)

    # Flatten the MultiIndex columns
    tbl_flat.columns = ['_'.join(col).strip() for col in tbl_flat.columns.values]

    # Rename columns after flattening
    renamed_columns = {
        'importance_min': 'importance_min',
        'importance_max': 'importance_max',
        'sensitivity_min': 'sensitivity_min',
        'sensitivity_max': 'sensitivity_max',
        'susceptibility_min': 'susceptibility_min',
        'susceptibility_max': 'susceptibility_max',
        'ref_geocodegroup_first': 'ref_geocodegroup',
        'name_gis_geocodegroup_first': 'name_gis_geocodegroup',
        'asset_group_name_<lambda>': 'asset_group_names',
        'ref_asset_group_nunique': 'assets_total',
        'geometry_first': 'geometry'
    }

    tbl_flat.rename(columns=renamed_columns, inplace=True)

    # Convert to GeoDataFrame
    tbl_flat = gpd.GeoDataFrame(tbl_flat, geometry='geometry')

    # Reset index to make 'code' a column
    tbl_flat.reset_index(inplace=True)

    # Save tbl_flat as a new layer in the GeoPackage
    tbl_flat.to_file(gpkg_file, layer='tbl_flat', driver='GPKG')


def classify_data(log_widget, gpkg_file, process_layer, column_name, config_path):
    # Load classification configuration
    classification = read_config_classification(config_path)

    # Load geopackage data
    gdf = gpd.read_file(gpkg_file, layer=process_layer)

    # Function to classify each row
    def classify_row(row):
        for label, value_range in classification.items():
            if row[column_name] in value_range:
                return label
        return 0  # or any default value

    new_column_name = column_name + "_code"
    # Apply classification
    gdf[new_column_name] = gdf.apply(lambda row: classify_row(row), axis=1)

    log_to_gui(log_widget, f"Updated codes for: {process_layer} - {column_name} ")
    update_progress(97)

    # Save the modified geopackage
    gdf.to_file(gpkg_file, layer=process_layer, driver='GPKG')

def process_all(log_widget, progress_var, gpkg_file):
    # Process and create tbl_stacked
    main_tbl_stacked(log_widget, progress_var, gpkg_file)

    # Process and create tbl_flat
    main_tbl_flat(log_widget, progress_var, gpkg_file) 
    
    classify_data(log_widget, gpkg_file, 'tbl_flat', 'sensitivity_min', config_file)
    classify_data(log_widget, gpkg_file, 'tbl_flat', 'sensitivity_max', config_file)
    classify_data(log_widget, gpkg_file, 'tbl_stacked', 'sensitivity', config_file)

    log_to_gui(log_widget, "COMPLETED: Data processing and aggregation completed.")
    update_progress(100)


#####################################################################################
#  Main
#

# Load configuration settings
config_file             = 'config.ini'
config                  = read_config(config_file)
gpkg_file               = config['DEFAULT']['gpkg_file']
ttk_bootstrap_theme     = config['DEFAULT']['ttk_bootstrap_theme']
workingprojection_epsg  = config['DEFAULT']['workingprojection_epsg']

# Create the user interface
root = ttk.Window(themename=ttk_bootstrap_theme)
root.title("Process data")

# Create a log widget
log_widget = scrolledtext.ScrolledText(root, height=10)
log_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Create a frame to hold the progress bar and the label
progress_frame = tk.Frame(root)
progress_frame.pack(pady=5)

# Create a progress bar
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=200, mode="determinate", variable=progress_var, bootstyle='info')
progress_bar.pack(side=tk.LEFT)  # Pack the progress bar on the left side of the frame

# Label for displaying the progress percentage
progress_label = tk.Label(progress_frame, text="0%", bg="light grey")
progress_label.pack(side=tk.LEFT, padx=5)  # Pack the label on the left side, next to the progress bar

# Information text field above the buttons
info_label_text = ("This is where all assets and geocode objects (grids) are processed "
                   "using the intersect function. For each such intersection a separate "
                   "geocode object (grid cell) is established. At the same time we also "
                   "calculate the sensitivity based on input asset importance and "
                   "susceptibility. Our data model provides a rich set of attributes "
                   "which we believe can be usefull in your further analysis of the "
                   "area sensitivities.")
info_label = tk.Label(root, text=info_label_text, wraplength=500, justify="left")
info_label.pack(padx=10, pady=10)

# Create a frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=5)

# Add 'Process All' button to the button frame
process_all_btn = ttk.Button(button_frame, text="Process All", command=lambda: threading.Thread(
    target=process_all, args=(log_widget, progress_var, gpkg_file), daemon=True).start())
process_all_btn.pack(side=tk.LEFT, padx=5, expand=False, fill=tk.X)

# Add 'Close' button to the button frame
close_btn = ttk.Button(button_frame, text="Exit", command=lambda: close_application(root))
close_btn.pack(side=tk.LEFT, padx=5, expand=False, fill=tk.X)


log_to_gui(log_widget, "Opened processing subprocess.")

root.mainloop()