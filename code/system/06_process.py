# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# This is where evertything comes together for the gridded assets.
#

import tkinter as tk
import locale
from tkinter import scrolledtext, ttk
import threading
import os
import geopandas as gpd
import pandas as pd
import configparser
import argparse
import datetime
import ttkbootstrap as ttk  # Import ttkbootstrap
from ttkbootstrap.constants import *

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# This script processes geospatial data by reading assets and geocode objects,
# performing spatial intersections, and aggregating results into final tables.
# It provides a GUI for monitoring progress and logs the processing steps.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Shared functions

# Read the configuration file
def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config

# Read the classification configuration file and populate the global dictionary
def read_config_classification(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    classification.clear()
    for section in config.sections():
        if section in ['A', 'B', 'C', 'D', 'E']:  # Only handle specific sections
            range_str = config[section]['range']
            description = config[section].get('description', '')  # Safely get the description if it exists
            start, end = map(int, range_str.split('-'))
            classification[section] = {
                'range': range(start, end + 1),  # Adjust the end value to make the range inclusive
                'description': description
            }
    return classification

# Core functions

# Update progress bar and label
def update_progress(new_value):
    progress_var.set(new_value)
    progress_label.config(text=f"{int(new_value)}%")

# Close the application
def close_application(root):
    root.destroy()

# Log messages to the GUI log widget and a log file
def log_to_gui(log_widget, message):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted_message = f"{timestamp} - {message}"
    log_widget.insert(tk.END, formatted_message + "\n")
    log_widget.see(tk.END)
    log_destination_file = os.path.join(original_working_directory, "log.txt")
    with open(log_destination_file, "a") as log_file:
        log_file.write(formatted_message + "\n")

# Increment a statistical value in the configuration file
def increment_stat_value(config_file, stat_name, increment_value):
    if not os.path.isfile(config_file):
        log_to_gui(log_widget,f"Configuration file {config_file} not found.")
        return

    with open(config_file, 'r') as file:
        lines = file.readlines()

    updated = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f'{stat_name} ='):
            parts = line.split('=')
            if len(parts) == 2:
                current_value = parts[1].strip()
                try:
                    new_value = int(current_value) + increment_value
                    lines[i] = f"{stat_name} = {new_value}\n"
                    updated = True
                    break
                except ValueError:
                    log_to_gui(log_widget,f"Error: Current value of {stat_name} is not an integer.")
                    return

    if updated:
        with open(config_file, 'w') as file:
            file.writelines(lines)

# Spatial functions

# Perform intersection with geocode data
def intersection_with_geocode_data(asset_df, geocode_df, geom_type, log_widget):
    log_to_gui(log_widget, f"Processing {geom_type} intersections")
    asset_filtered = asset_df[asset_df.geometry.geom_type == geom_type]

    if asset_filtered.empty:
        log_to_gui(log_widget, "No asset data of the specified geom type.")
        return gpd.GeoDataFrame()

    if asset_filtered.crs != geocode_df.crs:
        asset_filtered = asset_filtered.to_crs(geocode_df.crs)

    intersection_result = gpd.sjoin(geocode_df, asset_filtered, how='inner', predicate='intersects')
    if intersection_result.empty:
        log_to_gui(log_widget, "No intersections found.")
    else:
        log_to_gui(log_widget, f"Found {len(intersection_result)} intersections.")
    return intersection_result

# Create tbl_stacked by intersecting all asset data with the geocoding data
def main_tbl_stacked(log_widget, progress_var, gpkg_file, workingprojection_epsg):
    log_to_gui(log_widget, "Building analysis table (tbl_stacked).")
    update_progress(10)  # Indicate start

    asset_data = gpd.read_file(gpkg_file, layer='tbl_asset_object')

    if asset_data.crs is None:
        log_to_gui(log_widget, "No CRS found, setting default CRS.")
        asset_data.set_crs(workingprojection_epsg, inplace=True)
    update_progress(15)  # Progress after reading asset data

    geocode_data = gpd.read_file(gpkg_file, layer='tbl_geocode_object')
    update_progress(20)  # Progress after reading geocode data

    asset_group_data = gpd.read_file(gpkg_file, layer='tbl_asset_group')
    update_progress(25)  # Progress after reading asset group data

    log_to_gui(log_widget, f"Asset data count before merge: {len(asset_data)}")

    # Merge asset group data with asset data   
    asset_data = asset_data.merge(
        asset_group_data[['id', 'name_gis_assetgroup', 'total_asset_objects', 'importance', 'susceptibility', 'sensitivity', 'sensitivity_code', 'sensitivity_description']], 
        left_on='ref_asset_group', right_on='id', how='left'
    )
    log_to_gui(log_widget, f"Asset data count after merge: {len(asset_data)}")
    update_progress(30)  # Progress after merging data

    # For  now these are the types. Later - consider if GeometryCollection should also be included.
    geom_types = ['Point', 'MultiPoint', 'LineString', 'MultiLineString', 'Polygon', 'MultiPolygon']
    intersections = []

    # Process intersections for each geometry type
    for geom_type in geom_types:
        try:
            intersected = intersection_with_geocode_data(asset_data, geocode_data, geom_type, log_widget)
            intersections.append(intersected)
            update_progress(progress_var.get() + 3)
        except Exception as e:
            log_to_gui(log_widget, f"Error processing intersections for {geom_type}: {e}")

    intersected_data = pd.concat(intersections, ignore_index=True)
    log_to_gui(log_widget, f"Total intersected data count: {len(intersected_data)}")
    update_progress(49)  # Progress after concatenating data

    # Drop unnecessary columns if they exist
    columns_to_drop = ['id_x', 'id_y', 'total_asset_objects', 'process', 'index_right']
    intersected_data.drop(columns=[col for col in columns_to_drop if col in intersected_data.columns], inplace=True)
    log_to_gui(log_widget, f"Total intersected data after drop function: {len(intersected_data)}")

    # Area calculation
    if intersected_data.crs.is_geographic:
        temp_data = intersected_data.copy()
        temp_data.geometry = temp_data.geometry.to_crs("EPSG:3395")
        intersected_data['area_m2'] = temp_data.geometry.area.astype('int64')
    else:
        intersected_data['area_m2'] = intersected_data.geometry.area.astype('int64')
    log_to_gui(log_widget, f"Total intersected data after area calculations: {len(intersected_data)}")
    update_progress(50)  # Progress after area calculation

    # Assign the CRS to the GeoDataFrame
    intersected_data.crs = workingprojection_epsg
    log_to_gui(log_widget, f"Total intersected data after projection to working projection: {len(intersected_data)}")

    intersected_data.to_file(gpkg_file, layer='tbl_stacked', driver='GPKG')
    log_to_gui(log_widget, "Done processing the analysis layer.")
    update_progress(50)  # Final progress

# Create tbl_flat by aggregating values from tbl_stacked
def main_tbl_flat(log_widget, progress_var, gpkg_file, workingprojection_epsg):
    log_to_gui(log_widget, "Building map database (tbl_flat).")
    tbl_stacked = gpd.read_file(gpkg_file, layer='tbl_stacked')

    if tbl_stacked.crs is None:
        log_to_gui(log_widget, "CRS not found, setting default CRS for tbl_stacked.")
        tbl_stacked.set_crs(workingprojection_epsg, inplace=True)
    update_progress(60)

    # Calculate overlap counts per 'code'
    overlap_counts = tbl_stacked['code'].value_counts().reset_index()
    overlap_counts.columns = ['code', 'assets_overlap_total']

    # Aggregation functions
    aggregation_functions = {
        'importance': ['min', 'max'],
        'sensitivity': ['min', 'max'],
        'susceptibility': ['min', 'max'],
        'ref_geocodegroup': 'first',
        'name_gis_geocodegroup': 'first',
        'asset_group_name': lambda x: ', '.join(x.unique()),
        'ref_asset_group': 'nunique',
        'geometry': 'first'
    }

    # Group by 'code' and aggregate
    tbl_flat = tbl_stacked.groupby('code').agg(aggregation_functions)
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
        'ref_asset_group_nunique': 'asset_groups_total',
        'geometry_first': 'geometry'
    }
    tbl_flat.rename(columns=renamed_columns, inplace=True)

    tbl_flat = gpd.GeoDataFrame(tbl_flat, geometry='geometry', crs=workingprojection_epsg)

    if tbl_flat.crs is None:
        log_to_gui(log_widget, "CRS not found after aggregation, setting CRS for tbl_flat.")
        tbl_flat.set_crs(workingprojection_epsg, inplace=True)

    if tbl_flat.crs.is_geographic:
        temp_tbl_flat = tbl_flat.copy()
        temp_tbl_flat.geometry = temp_tbl_flat.geometry.to_crs("EPSG:3395")
        tbl_flat['area_m2'] = temp_tbl_flat.geometry.area.astype('int64')
    else:
        tbl_flat['area_m2'] = tbl_flat.geometry.area.astype('int64')

    tbl_flat.reset_index(inplace=True)
    tbl_flat = tbl_flat.merge(overlap_counts, on='code', how='left')

    tbl_flat.to_file(gpkg_file, layer='tbl_flat', driver='GPKG')
    log_to_gui(log_widget, "tbl_flat processed and saved.")

# Classify data based on configuration
def classify_data(log_widget, gpkg_file, process_layer, column_name, config_path):
    classification = read_config_classification(config_path)
    gdf = gpd.read_file(gpkg_file, layer=process_layer)

    def classify_row(value):
        for label, info in classification.items():
            if value in info['range']:
                return label, info['description']
        return 'Unknown', 'No description available'

    base_name, *suffix = column_name.rsplit('_', 1)
    suffix = suffix[0] if suffix else ''
    new_code_col = f"{base_name}_code_{suffix}" if suffix else f"{base_name}_code"
    new_desc_col = f"{base_name}_description_{suffix}" if suffix else f"{base_name}_description"

    gdf[new_code_col], gdf[new_desc_col] = zip(*gdf[column_name].apply(classify_row))

    log_to_gui(log_widget, f"Updated classifications for {process_layer} based on {column_name}")

    gdf.to_file(gpkg_file, layer=process_layer, driver='GPKG')

    log_to_gui(log_widget, f"Data saved to {process_layer} with new fields {new_code_col} and {new_desc_col}")

# Process all steps and create final tables
def process_all(log_widget, progress_var, gpkg_file, config_file, workingprojection_epsg):
    main_tbl_stacked(log_widget, progress_var, gpkg_file, workingprojection_epsg)
    main_tbl_flat(log_widget, progress_var, gpkg_file, workingprojection_epsg) 
    update_progress(94)

    classify_data(log_widget, gpkg_file, 'tbl_flat', 'sensitivity_min', config_file)
    update_progress(95)

    classify_data(log_widget, gpkg_file, 'tbl_flat', 'sensitivity_max', config_file)
    update_progress(97)

    classify_data(log_widget, gpkg_file, 'tbl_stacked', 'sensitivity', config_file)
    update_progress(99)

    increment_stat_value(config_file, 'mesa_stat_process', increment_value=1)

    log_to_gui(log_widget, "COMPLETED: Processing of analysis and presentation layers completed.")
    update_progress(100)

#####################################################################################
#  Main
#

# original folder for the system is sent from the master executable. If the script is
# invoked this way we are fetching the address here.
parser = argparse.ArgumentParser(description='Slave script')
parser.add_argument('--original_working_directory', required=False, help='Path to running folder')
args = parser.parse_args()
original_working_directory = args.original_working_directory

# However - if this is not the case we will have to establish the root folder in 
# one of two different ways.
if original_working_directory is None or original_working_directory == '':
    original_working_directory  = os.getcwd()
    if str("system") in str(original_working_directory):
        original_working_directory = os.path.abspath(os.path.join(original_working_directory, os.pardir))

# Load configuration settings and data
config_file             = os.path.join(original_working_directory, "system/config.ini")
gpkg_file               = os.path.join(original_working_directory, "output/mesa.gpkg")

# Load configuration settings
config                  = read_config(config_file)
        
mesa_stat_process       = config['DEFAULT']['mesa_stat_process']
ttk_bootstrap_theme     = config['DEFAULT']['ttk_bootstrap_theme']
workingprojection_epsg  = f"EPSG:{config['DEFAULT']['workingprojection_epsg']}"

# Global declaration
classification = {}

if __name__ == "__main__":
    root = ttk.Window(themename=ttk_bootstrap_theme)
    root.title("Process data")
    root.iconbitmap(os.path.join(original_working_directory,"system_resources/mesa.ico"))

    log_widget = scrolledtext.ScrolledText(root, height=10)
    log_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    progress_frame = tk.Frame(root)
    progress_frame.pack(pady=5)

    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=200, mode="determinate", variable=progress_var, bootstyle='info')
    progress_bar.pack(side=tk.LEFT)

    progress_label = tk.Label(progress_frame, text="0%", bg="light grey")
    progress_label.pack(side=tk.LEFT, padx=5)

    info_label_text = ("This is where all assets and geocode objects (grids) are processed "
                    "using the intersect function. For each such intersection a separate "
                    "geocode object (grid cell) is established. At the same time we also "
                    "calculate the sensitivity based on input asset importance and "
                    "susceptibility. Our data model provides a rich set of attributes "
                    "which we believe can be useful in your further analysis of the "
                    "area sensitivities.")
    info_label = tk.Label(root, text=info_label_text, wraplength=600, justify="left")
    info_label.pack(padx=10, pady=10)

    button_frame = tk.Frame(root)
    button_frame.pack(pady=5)

    process_all_btn = ttk.Button(button_frame, text="Process", command=lambda: threading.Thread(
        target=process_all, args=(log_widget, progress_var, gpkg_file, config_file, workingprojection_epsg), daemon=True).start())
    process_all_btn.pack(side=tk.LEFT, padx=5, expand=False, fill=tk.X)

    close_btn = ttk.Button(button_frame, text="Exit", command=lambda: close_application(root), bootstyle=WARNING)
    close_btn.pack(side=tk.LEFT, padx=5, expand=False, fill=tk.X)

    log_to_gui(log_widget, "Opened processing subprocess.")

    root.mainloop()
