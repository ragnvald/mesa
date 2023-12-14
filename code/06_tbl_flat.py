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
        'importance': ['max', 'min'],
        'sensitivity': ['max', 'min'],
        'susceptibility': ['max', 'min'],
        'geometry': 'first'  # Keeping the first geometry for each group
    }

    grouped = intersected_data.groupby('code').agg(aggregation_functions)

    # Flatten MultiIndex columns
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

    # Count the total assets in each group
    grouped['assets_total'] = intersected_data.groupby('code').size()

    return grouped

# Main function for processing data
def main(log_widget, progress_var, gpkg_file):
    log_to_gui(log_widget, "Starting processing...")
    asset_data = gpd.read_file(gpkg_file, layer='tbl_asset_object')
    geocode_data = gpd.read_file(gpkg_file, layer='tbl_geocode_object')
    asset_group_data = gpd.read_file(gpkg_file, layer='tbl_asset_group')

    # Merge asset group data with asset data
    asset_data = asset_data.merge(asset_group_data[['id', 'total_asset_objects', 'importance', 'susceptibility', 'sensitivity']], 
                                  left_on='ref_asset_group', right_on='id', how='left')

    point_intersections = intersection_with_geocode_data(asset_data, geocode_data, 'Point', log_widget)
    line_intersections = intersection_with_geocode_data(asset_data, geocode_data, 'LineString', log_widget)
    polygon_intersections = intersection_with_geocode_data(asset_data, geocode_data, 'Polygon', log_widget)

    intersected_data = pd.concat([point_intersections, line_intersections, polygon_intersections])

    # Aggregate data
    aggregated_data = aggregate_data(intersected_data)

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
 