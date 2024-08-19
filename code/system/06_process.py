# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# This is where evertything comes together for the gridded assets.
#

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
import tkinter as tk
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
import concurrent.futures
import multiprocessing
import time

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

def intersection_with_geocode_data(asset_df, geocode_df, geom_type):
    asset_filtered = asset_df[asset_df.geometry.geom_type == geom_type]

    if asset_filtered.empty:
        return gpd.GeoDataFrame()

    if asset_filtered.crs != geocode_df.crs:
        asset_filtered = asset_filtered.to_crs(geocode_df.crs)

    intersection_result = gpd.sjoin(geocode_df, asset_filtered, how='inner', predicate='intersects')
    return intersection_result

# Perform intersection with geocode data
def test_function_for_choice(asset_data, geocode_data, geom_type):
    """A small task to test performance with threading vs multiprocessing."""
    asset_filtered = asset_data[asset_data.geometry.geom_type == geom_type]
    if asset_filtered.empty:
        return gpd.GeoDataFrame()

    if asset_filtered.crs != geocode_data.crs:
        asset_filtered = asset_filtered.to_crs(geocode_data.crs)

    intersection_result = gpd.sjoin(geocode_data, asset_filtered, how='inner', predicate='intersects')
    return intersection_result

def measure_execution_time(method, asset_data, geocode_data, geom_type):
    """Measure the execution time for threading or multiprocessing."""
    start_time = time.time()
    
    if method == 'threading':
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(test_function_for_choice, asset_data, geocode_data, geom_type)
            future.result()

    elif method == 'multiprocessing':
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future = executor.submit(test_function_for_choice, asset_data, geocode_data, geom_type)
            future.result()

    return time.time() - start_time

def choose_best_method(asset_data, geocode_data, geom_types, log_widget):
    """Choose the best method between threading and multiprocessing by testing multiple geom types."""
    threading_total_time = 0
    multiprocessing_total_time = 0

    for geom_type in ['Polygon', 'MultiPolygon']:  # Focus on the more complex geometries
        if geom_type in geom_types:
            log_to_gui(log_widget, f"Testing parallel method selection based on {geom_type} geometry.")
            threading_total_time += measure_execution_time('threading', asset_data, geocode_data, geom_type)
            multiprocessing_total_time += measure_execution_time('multiprocessing', asset_data, geocode_data, geom_type)

    if threading_total_time < multiprocessing_total_time:
        log_to_gui(log_widget, "Threading chosen based on overall performance.")
        return 'threading'
    else:
        log_to_gui(log_widget, "Multiprocessing chosen based on overall performance.")
        return 'multiprocessing'


def parallel_intersection(asset_data, geocode_data, geom_types, log_widget, progress_var, method):
    intersections = []
    
    if method == 'threading':
        Executor = concurrent.futures.ThreadPoolExecutor
    elif method == 'multiprocessing':
        Executor = concurrent.futures.ProcessPoolExecutor

    with Executor() as executor:
        future_to_geom_type = {
            executor.submit(intersection_with_geocode_data, asset_data, geocode_data, geom_type): geom_type
            for geom_type in geom_types
        }
        
        for future in concurrent.futures.as_completed(future_to_geom_type):
            geom_type = future_to_geom_type[future]
            try:
                result = future.result()
                log_to_gui(log_widget, f"Processed {geom_type} with {len(result)} intersections.")
                intersections.append(result)
                update_progress(progress_var.get() + 3)
            except Exception as e:
                log_to_gui(log_widget, f"Error processing intersections for {geom_type}: {e}")

    return pd.concat(intersections, ignore_index=True)

def main_tbl_stacked(log_widget, progress_var, gpkg_file, workingprojection_epsg):
    log_to_gui(log_widget, "Started building analysis table (tbl_stacked).")
    update_progress(10)

    asset_data = gpd.read_file(gpkg_file, layer='tbl_asset_object')
    log_to_gui(log_widget, "tbl_asset_object read from database to memory.")

    if asset_data.crs is None:
        log_to_gui(log_widget, "No CRS found, setting default CRS.")
        asset_data.set_crs(workingprojection_epsg, inplace=True)
    update_progress(15)

    geocode_data = gpd.read_file(gpkg_file, layer='tbl_geocode_object')
    log_to_gui(log_widget, "Geocodes read")
    update_progress(20)

    asset_group_data = gpd.read_file(gpkg_file, layer='tbl_asset_group')
    log_to_gui(log_widget, "Assets read")
    update_progress(25)

    log_to_gui(log_widget, f"Asset data count before merge: {len(asset_data)}")

    asset_data = asset_data.merge(
        asset_group_data[['id', 'name_gis_assetgroup', 'total_asset_objects', 'importance', 'susceptibility', 'sensitivity', 'sensitivity_code', 'sensitivity_description']], 
        left_on='ref_asset_group', right_on='id', how='left'
    )
    log_to_gui(log_widget, f"Asset data count after merge: {len(asset_data)}")
    update_progress(30)

    geom_types = ['Point', 'MultiPoint', 'LineString', 'MultiLineString', 'Polygon', 'MultiPolygon']
    
    # Choose the best method for parallel processing
    best_method = choose_best_method(asset_data, geocode_data, geom_types[0])
    log_to_gui(log_widget, f"Chosen method for parallel processing: {best_method}")
    
    # Use the chosen method for parallel_intersection
    intersected_data = parallel_intersection(asset_data, geocode_data, geom_types, log_widget, progress_var, best_method)

    log_to_gui(log_widget, f"Total intersected objects count: {len(intersected_data)}")
    update_progress(49)

    columns_to_drop = ['id_x', 'id_y', 'total_asset_objects', 'process', 'index_right']
    intersected_data.drop(columns=[col for col in columns_to_drop if col in intersected_data.columns], inplace=True)
    log_to_gui(log_widget, f"Total intersected data after drop function: {len(intersected_data)}")

    if intersected_data.crs.is_geographic:
        temp_data = intersected_data.copy()
        temp_data.geometry = temp_data.geometry.to_crs("EPSG:3395")
        intersected_data['area_m2'] = temp_data.geometry.area.astype('int64')
    else:
        intersected_data['area_m2'] = intersected_data.geometry.area.astype('int64')

    intersected_data.crs = workingprojection_epsg
    log_to_gui(log_widget, f"Total intersected data after projection to working projection: {len(intersected_data)}")

    intersected_data.to_file(gpkg_file, layer='tbl_stacked', driver='GPKG')
    log_to_gui(log_widget, "Done processing the analysis layer.")
    update_progress(50)


# Create tbl_flat by aggregating values from tbl_stacked
def main_tbl_flat(log_widget, progress_var, gpkg_file, workingprojection_epsg):
    log_to_gui(log_widget, "Building map database (tbl_flat).")
    tbl_stacked = gpd.read_file(gpkg_file, layer='tbl_stacked')

    # Log the initial state of tbl_stacked
    log_to_gui(log_widget, f"Initial tbl_stacked size: {len(tbl_stacked)}")
    
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

    # Log the initial state of the GeoDataFrame and the classification configuration
    log_to_gui(log_widget, f"Initial size of {process_layer}: {len(gdf)}")
    log_to_gui(log_widget, f"Classification configuration: {classification}")

    def classify_row(value):
        for label, info in classification.items():
            if value in info['range']:
                return label, info['description']
        return 'Unknown', 'No description available'

    base_name, *suffix = column_name.rsplit('_', 1)
    suffix = suffix[0] if suffix else ''
    new_code_col = f"{base_name}_code_{suffix}" if suffix else f"{base_name}_code"
    new_desc_col = f"{base_name}_description_{suffix}" if suffix else f"{base_name}_description"

    # Ensure the column exists
    if column_name not in gdf.columns:
        log_to_gui(log_widget, f"Column {column_name} not found in {process_layer}. Classification skipped.")
        return

    # Apply classification
    gdf[new_code_col], gdf[new_desc_col] = zip(*gdf[column_name].apply(classify_row))


    log_to_gui(log_widget, f"Updated classifications for {process_layer} based on {column_name}")

    # Save the classified data back to the file
    gdf.to_file(gpkg_file, layer=process_layer, driver='GPKG')

    # Log after saving
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
