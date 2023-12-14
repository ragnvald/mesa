import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import threading
import geopandas as gpd
import pandas as pd
import configparser
import datetime
import os
from osgeo import ogr

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

# Function to perform intersection with geocode data
def intersection_with_geocode_data(asset_df, geocode_df, geom_type, log_widget):
    log_to_gui(log_widget, f"Processing {geom_type} intersections")
    asset_filtered = asset_df[asset_df.geometry.geom_type == geom_type]

    if asset_filtered.empty:
        return gpd.GeoDataFrame()

    return gpd.sjoin(geocode_df, asset_filtered, how='inner', op='intersects')

# Function to aggregate data by code
def aggregate_data(intersected_data):
    # Group by code and calculate aggregates
    aggregation_functions = {
        'importance': ['min', 'max'],
        'sensitivity': ['min', 'max'],
        'susceptibility': ['min', 'max'],
        'ref_geocodegroup': 'first',  # Include the first reference to geocode group id
        'name': 'first',  # Include the first name for each group
        'geometry': 'first'  # Keeping the first geometry for each group
    }
    
    grouped = intersected_data.groupby('code').agg(aggregation_functions)
    
    # Flatten MultiIndex columns
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

    # Rename columns after flattening
    renamed_columns = {
        'name_first': 'name_geocodegroup',
        'ref_geocodegroup_first': 'ref_geocodegroup'
    }    
    
    grouped.rename(columns=renamed_columns, inplace=True)
    
    # Count the total assets in each group
    grouped['assets_total'] = intersected_data.groupby('code').size()

    return grouped

# Main function for processing data
def main(log_widget, progress_var, gpkg_file):
    log_to_gui(log_widget, "Starting processing...")

    # Reading 'tbl_stacked' data from the GeoPackage
    log_to_gui(log_widget, "Reading 'tbl_stacked' data...")
    asset_data = gpd.read_file(gpkg_file, layer='tbl_stacked')

    # Reading 'tbl_geocode_group' data from the GeoPackage
    log_to_gui(log_widget, "Reading 'tbl_geocode_group' data...")
    geocode_group_data = gpd.read_file(gpkg_file, layer='tbl_geocode_group')

    # Ensure 'code' column is present in 'tbl_stacked'
    if 'code' not in asset_data.columns:
        log_to_gui(log_widget, "'code' column not found in 'tbl_stacked'.")
        raise KeyError("'code' column not found in 'tbl_stacked'")

    # Merge asset_data with geocode_group_data on ref_geocodegroup
    log_to_gui(log_widget, "Merging data...")
    merged_data = asset_data.merge(geocode_group_data[['id', 'name']],
                                   left_on='ref_geocodegroup',
                                   right_on='id',
                                   how='left',
                                   suffixes=('_asset', '_geocode'))

    # Drop the unnecessary columns (id_x and id_y)
    merged_data.drop(columns=['id_asset', 'id_geocode'], inplace=True)

    # Proceed with aggregation
    log_to_gui(log_widget, "Aggregating data...")
    aggregated_data = aggregate_data(merged_data)

    # Save to GeoPackage
    aggregated_gdf = gpd.GeoDataFrame(aggregated_data, geometry='geometry_first')
    aggregated_gdf.to_file(gpkg_file, layer='tbl_flat', driver='GPKG')

    log_to_gui(log_widget, "Data processing and aggregation completed.")
    progress_var.set(100)


# Thread function to run main without freezing GUI
def run_main(log_widget, progress_var, gpkg_file):
    main(log_widget, progress_var, gpkg_file)

# Function to close the application
def close_application(root):
    root.destroy()

# Create the user interface
root = tk.Tk()
root.title("Geocode Intersection Utility")

# Create a log widget
log_widget = scrolledtext.ScrolledText(root, height=10)
log_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Create a progress bar
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root, orient="horizontal", length=200, mode="determinate", variable=progress_var)
progress_bar.pack(pady=5, fill=tk.X)

# Add buttons for operations
run_btn = ttk.Button(root, text="Run Analysis", command=lambda: threading.Thread(
    target=run_main, args=(log_widget, progress_var, gpkg_file), daemon=True).start())
run_btn.pack(pady=5, fill=tk.X)

close_btn = ttk.Button(root, text="Close", command=lambda: close_application(root))
close_btn.pack(pady=5, fill=tk.X)

# Load configuration settings
config_file = 'config.ini'
config = read_config(config_file)
gpkg_file = config['DEFAULT']['gpkg_file']

root.mainloop()
 