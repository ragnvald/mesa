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
import multiprocessing
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time
from datetime import datetime, timedelta
import shapely.wkb
from shapely import wkb
from shapely.geometry import box  # Import the box function from shapely.geometry
import numpy as np

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
    timestamp = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
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

# Function to convert geometries to WKB and save to Parquet
def save_data_to_parquet(df, parquet_file, geom_col='geometry'):
    df['geometry_wkb'] = df[geom_col].apply(lambda geom: shapely.wkb.dumps(geom) if geom else None)
    df.drop(columns=[geom_col], inplace=True)
    df.to_parquet(parquet_file, index=False, compression='snappy')


def intersection_with_geocode_data(asset_df, geocode_df, geom_type):

    # Use the regular GeoPandas spatial join for regular GeoDataFrames
    asset_filtered = asset_df[asset_df.geometry.geom_type == geom_type]

    if asset_filtered.empty:
        return gpd.GeoDataFrame()

    if asset_filtered.crs != geocode_df.crs:
        asset_filtered = asset_filtered.to_crs(geocode_df.crs)

    intersection_result = gpd.sjoin(geocode_df, asset_filtered, how='inner', predicate='intersects')

    return intersection_result


def process_geocode_chunk(geocode_chunk, asset_data):
    intersections = []
    for geom_type in asset_data.geometry.geom_type.unique():
        asset_filtered = asset_data[asset_data.geometry.geom_type == geom_type]
        if asset_filtered.empty:
            continue

        # Perform the intersection
        intersection_result = intersection_with_geocode_data(asset_filtered, geocode_chunk, geom_type)
        intersections.append(intersection_result)
    
    return pd.concat(intersections, ignore_index=True)


# Additional functions for grid creation and asset assignment
def create_grid(geodata, cell_size):
    bounds = geodata.total_bounds  # Get the spatial extent (xmin, ymin, xmax, ymax)
    xmin, ymin, xmax, ymax = bounds
    x_cells = np.arange(xmin, xmax + cell_size, cell_size)
    y_cells = np.arange(ymin, ymax + cell_size, cell_size)
    
    cells = []
    for x in x_cells:
        for y in y_cells:
            cells.append((
                x, y,
                x + cell_size, y + cell_size
            ))
    
    return cells

def process_chunk(geodata_chunk, grid_gdf):
    # Spatial join for each chunk
    return gpd.sjoin(geodata_chunk, grid_gdf, how="left", predicate="intersects")


def assign_assets_to_grid(geodata, grid_cells, log_widget, max_workers):
    log_to_gui(log_widget, "Creating grid GeoDataFrame.")

    # Create a GeoDataFrame for the grid cells
    grid_gdf = gpd.GeoDataFrame({
        'grid_cell': range(len(grid_cells)),
        'geometry': [box(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax in grid_cells]
    }, crs=geodata.crs)

    log_to_gui(log_widget, "Grid GeoDataFrame created. Building spatial index for grid cells.")
    
    # Create a spatial index for the grid cells
    grid_gdf.sindex  # Just creating the index

    # Split the geodata into chunks for parallel processing
    geodata_chunks = np.array_split(geodata, max_workers)

    log_to_gui(log_widget, f"Processing {len(geodata_chunks)} chunks in parallel using {max_workers} workers.")

    # Use ThreadPoolExecutor instead of ProcessPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_chunk, geodata_chunks, [grid_gdf] * len(geodata_chunks)))

    log_to_gui(log_widget, "Combining results from all chunks.")
    
    # Combine results from all chunks
    geodata = gpd.GeoDataFrame(pd.concat(results, ignore_index=True), crs=geodata.crs)
    
    log_to_gui(log_widget, "Cleaning up the result.")
    
    return geodata.drop(columns='index_right')


def process_chunk_and_log(idx, geocode_chunk, asset_data, total_chunks):
    try:
        # Pre-indexing each chunk
        geocode_chunk.sindex
        
        result = process_geocode_chunk(geocode_chunk, asset_data)
        
        return (idx, len(result), result)  # Return necessary info for logging and processing
    except Exception as e:
        return (idx, 0, None, str(e))  # Return error info if something goes wrong
    
    
# This is perhaps the most important function in this program. The function calculates intersects
# between the assets and geocodes. It handles a substantial amount of data. This is where
# efforts should be made to look at methods for parallell computing using threading or even
# using procedures which interact with the GPU.
def intersect_asset_and_geocode(asset_data, geocode_data, log_widget, progress_var, method, workingprojection_epsg, cell_size, max_workers):
    intersections = []

    # If max_workers is set to 0, determine the number of physical CPU cores
    if max_workers == 0:
        try:
            # Get the number of physical cores
            max_workers = multiprocessing.cpu_count()
            log_to_gui(log_widget, f"Number of workers determined by system to {max_workers}.")
        except NotImplementedError:
            # Fallback to a default value in case cpu_count() is not implemented
            max_workers = 4
    else:
        log_to_gui(log_widget, f"Number of workers determinedset in config to {max_workers}.")
    
    log_to_gui(log_widget, f"Processing method is {method}.")

    Executor = ThreadPoolExecutor

    # Start timing the process
    start_time = time.time()

    # Calculate cell_size in degrees
    meters_per_degree = 111320  # Approximation at the equator
    cell_size_degrees = cell_size / meters_per_degree
    log_to_gui(log_widget, f"Cell size converted to degrees: {cell_size_degrees:.6f} degrees")

    # Create a grid cell where each of the sides is cell_size_degrees long
    log_to_gui(log_widget, "Creating analysis grid.")
    grid_cells = create_grid(geocode_data, cell_size_degrees)
    log_to_gui(log_widget, "Assigning assets to grid.")
    geocode_data = assign_assets_to_grid(geocode_data, grid_cells, log_widget, max_workers)

    log_to_gui(log_widget, f"Grouping geocodes into chunks according to cell size of {cell_size}.")
    geocode_data_chunks = [geocode_data[geocode_data['grid_cell'] == idx] for idx in geocode_data['grid_cell'].unique()]
    
    total_chunks = len(geocode_data_chunks)
    log_to_gui(log_widget, f"Processing {total_chunks} chunks with a maximum of {max_workers} workers.")

    with Executor(max_workers=max_workers) as executor:
        future_to_chunk = {}
        chunks_processed = 0
        for idx, geocode_chunk in enumerate(geocode_data_chunks, start=1):
            future = executor.submit(process_chunk_and_log, idx, geocode_chunk, asset_data, total_chunks)
            future_to_chunk[future] = idx

            if len(future_to_chunk) >= max_workers:
                done, _ = concurrent.futures.wait(future_to_chunk, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    idx, num_results, result, *error = future.result()
                    chunks_processed += 1
                    if error:
                        log_to_gui(log_widget, f"Error processing geocode chunk {idx} of {total_chunks}: {error[0]}")
                    elif result is not None:
                        log_to_gui(log_widget, f"Processed geocode chunk {idx} of {total_chunks} with {num_results} intersections.")
                        intersections.append(result)
                        update_progress(progress_var.get() + (3 / total_chunks))

                    # Estimate time remaining
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    estimated_total_time = (elapsed_time / chunks_processed) * total_chunks if chunks_processed > 0 else 0
                    estimated_time_remaining = estimated_total_time - elapsed_time

                    # Calculate estimated completion time
                    estimated_completion_time = datetime.now() + timedelta(seconds=estimated_time_remaining)
                    estimated_completion_time_str = estimated_completion_time.strftime("%H:%M:%S")

                    # Calculate the difference in days from today
                    days_diff = (estimated_completion_time.date() - datetime.now().date()).days

                    # Add a red "+N" if the timestamp is in the future
                    if days_diff > 0:
                        red_plus_n = f" (+{days_diff} days)"  # ANSI escape code for red
                        estimated_completion_time_str += red_plus_n

                    log_to_gui(log_widget, f"Core computation might conclude at {estimated_completion_time_str}.")
                    del future_to_chunk[future]

        # Process any remaining futures
        for future in concurrent.futures.as_completed(future_to_chunk):
            idx, num_results, result, *error = future.result()
            chunks_processed += 1
            if error:
                log_to_gui(log_widget, f"Error processing geocode chunk {idx} of {total_chunks}: {error[0]}")
            elif result is not None:
                log_to_gui(log_widget, f"Processed geocode chunk {idx} of {total_chunks} with {num_results} intersections.")
                intersections.append(result)
                update_progress(progress_var.get() + (3 / total_chunks))

            # Estimate time remaining for the last chunks
            current_time = time.time()
            elapsed_time = current_time - start_time
            estimated_total_time = (elapsed_time / chunks_processed) * total_chunks if chunks_processed > 0 else 0
            estimated_time_remaining = estimated_total_time - elapsed_time
            estimated_completion_time = datetime.now() + timedelta(seconds=estimated_time_remaining)

            # Calculate estimated completion time
            estimated_completion_time = datetime.now() + timedelta(seconds=estimated_time_remaining)
            estimated_completion_time_str = estimated_completion_time.strftime("%H:%M:%S")

            days_diff = (estimated_completion_time.date() - datetime.now().date()).days

            # Add a red "+N" if the timestamp is in the future
            if days_diff > 0:
                extra_days_formatted = f" {days_diff} days"  # ANSI escape code for red
                estimated_completion_time_str += extra_days_formatted

            log_to_gui(log_widget, f"Intersection analysis might conclude at {estimated_completion_time_str}.")
            
    # Calculate the total time taken
    end_time = time.time()
    total_time = end_time - start_time
    time_per_chunk = total_time / total_chunks if total_chunks > 0 else 0

    # Convert total time to hours, minutes, and seconds
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Calculate the processing rates
    asset_objects_per_second = len(asset_data) / total_time if total_time > 0 else 0
    geocode_objects_per_second = len(geocode_data) / total_time if total_time > 0 else 0

    log_to_gui(log_widget, "Core computation concluded.")
    log_to_gui(log_widget, f"Processing completed in {total_time:.2f} seconds.")
    log_to_gui(log_widget, f"Average time per chunk: {time_per_chunk:.2f} seconds.")
    log_to_gui(log_widget, f"Total asset objects processed: {len(asset_data)}.")
    log_to_gui(log_widget, f"Asset objects processed per second: {asset_objects_per_second:.2f}.")
    log_to_gui(log_widget, f"Total geocode objects processed: {len(geocode_data)}.")
    log_to_gui(log_widget, f"Geocode objects processed per second: {geocode_objects_per_second:.2f}.")
    log_to_gui(log_widget, f"Processing completed in {int(hours)}h {int(minutes)}m {int(seconds)}s - a total of {round(total_time,1)} seconds.")
   
    return gpd.GeoDataFrame(pd.concat(intersections, ignore_index=True), crs=workingprojection_epsg)

# Process the tbl_stacked table by intersecting assets and geocodes.
def process_tbl_stacked(log_widget, progress_var, gpkg_file, workingprojection_epsg, chunk_size):
    log_to_gui(log_widget, "Started building analysis table (tbl_stacked).")
    update_progress(10)

    # Reading assets to memory
    log_to_gui(log_widget, "Reading assets to memory.")
    asset_data = gpd.read_file(gpkg_file, layer='tbl_asset_object')
    log_to_gui(log_widget, "Indexing the asset data.")
    asset_data.sindex
    log_to_gui(log_widget, "Assets read to memory.")

    if asset_data.crs is None:
        log_to_gui(log_widget, "No CRS found, setting default CRS.")
        asset_data.set_crs(workingprojection_epsg, inplace=True)
    update_progress(15)

    # Reading geocodes to memory
    log_to_gui(log_widget, "Reading geocodes to memory.")
    geocode_data = gpd.read_file(gpkg_file, layer='tbl_geocode_object')
    geocode_data.sindex
    log_to_gui(log_widget, "Geocodes read to memory.")
    update_progress(20)
    log_to_gui(log_widget, f"Geocode objects count before intersection: {len(geocode_data)}")

    # Reading asset groups to memory
    log_to_gui(log_widget, "Reading asset groups to memory.")
    asset_group_data = gpd.read_file(gpkg_file, layer='tbl_asset_group')
    asset_group_data.sindex
    log_to_gui(log_widget, "Asset groups read")
    update_progress(25)

    log_to_gui(log_widget, f"Asset objects count before intersection: {len(asset_data)}")

    # Merging asset data per asset group
    log_to_gui(log_widget, "Merging asset data per asset group.")
    asset_data = asset_data.merge(
        asset_group_data[['id', 'name_gis_assetgroup', 'total_asset_objects', 'importance', 'susceptibility', 'sensitivity', 'sensitivity_code', 'sensitivity_description']], 
        left_on='ref_asset_group', right_on='id', how='left'
    )
    log_to_gui(log_widget, "Asset data merged.")
    update_progress(30)

    log_to_gui(log_widget, "Starting the intersection part of the processing. This might take some time...")
    # Perform the intersection using the updated intersect_asset_and_geocode function
    intersected_data = intersect_asset_and_geocode(asset_data, geocode_data, log_widget, progress_var, method, workingprojection_epsg, cell_size, max_workers)
    
    if intersected_data.empty:
        log_to_gui(log_widget, "No intersected data returned.")
        return

    log_to_gui(log_widget, f"Total intersected objects count: {len(intersected_data)}")
    update_progress(49)

    # Ensure geometries are valid and calculate areas
    log_to_gui(log_widget, "Qualify grid cell areas.")
    intersected_data = intersected_data[intersected_data.geometry.notna()]
    
    if intersected_data.empty:
        log_to_gui(log_widget, "No valid intersected data found.")
        return

    # Calculate area in square meters for valid geometries
    log_to_gui(log_widget, "Calculating grid cell areas.")
    if intersected_data.crs.is_geographic:
        temp_data = intersected_data.copy()
        temp_data['geometry'] = temp_data.geometry.to_crs("EPSG:3395")
        intersected_data['area_m2'] = temp_data.geometry.area.astype('float64')
    else:
        intersected_data['area_m2'] = intersected_data.geometry.area.astype('float64')

    intersected_data['area_m2'] = intersected_data['area_m2'].fillna(0).astype('int64')
    log_to_gui(log_widget, "Done area calculations.")

    log_to_gui(log_widget, f"Total intersected data after projection to working projection: {len(intersected_data)}")

    # Drop unnecessary columns before saving
    intersected_data = intersected_data.drop(columns=['geometry_wkb', 'geometry_wkb_1', 'process'], errors='ignore')

    # Write the intersected data to the geopackage
    intersected_data.to_file(gpkg_file, layer='tbl_stacked', driver='GPKG')
    log_to_gui(log_widget, "Done processing the analysis layer.")
    update_progress(50)


# Create tbl_flat by aggregating values from tbl_stacked.
def process_tbl_flat(log_widget, progress_var, gpkg_file, workingprojection_epsg):
    log_to_gui(log_widget, "Building map database (tbl_flat).")
    tbl_stacked = gpd.read_file(gpkg_file, layer='tbl_stacked')

    # Log the initial state of tbl_stacked
    log_to_gui(log_widget, f"Initial tbl_stacked size: {len(tbl_stacked)}")
    
    if 'code' not in tbl_stacked.columns:
        log_to_gui(log_widget, "Error: 'code' column is missing in tbl_stacked.")
        return
    
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
def process_all(log_widget, progress_var, gpkg_file, config_file, workingprojection_epsg,chunk_size):
    log_to_gui(log_widget, "Started processing of analysis and presentation layers.")
    process_tbl_stacked(log_widget, progress_var, gpkg_file, workingprojection_epsg,chunk_size)
    process_tbl_flat(log_widget, progress_var, gpkg_file, workingprojection_epsg) 
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
parquet_folder          = os.path.join(original_working_directory, "output/")

# Load configuration settings
config                  = read_config(config_file)
        
mesa_stat_process       = config['DEFAULT']['mesa_stat_process']
ttk_bootstrap_theme     = config['DEFAULT']['ttk_bootstrap_theme']
workingprojection_epsg  = f"EPSG:{config['DEFAULT']['workingprojection_epsg']}"

chunk_size              = int(config['DEFAULT'].get('chunk_size', '10000'))  # Default to 10000 if not set in config
method                  = str(config['DEFAULT'].get('method'))  
max_workers             = int(config['DEFAULT'].get('max_workers'))
cell_size               = int(config['DEFAULT'].get('cell_size'))

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
        target=process_all, args=(log_widget, progress_var, gpkg_file, config_file, workingprojection_epsg, chunk_size), daemon=True).start())
    process_all_btn.pack(side=tk.LEFT, padx=5, expand=False, fill=tk.X)

    close_btn = ttk.Button(button_frame, text="Exit", command=lambda: close_application(root), bootstyle=WARNING)
    close_btn.pack(side=tk.LEFT, padx=5, expand=False, fill=tk.X)

    log_to_gui(log_widget, "Opened processing subprocess.")
    
    log_to_gui(log_widget, f"Chunk size is {chunk_size}.")

    root.mainloop()
