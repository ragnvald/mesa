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
        'geometry': 'first',  # Keeping the first geometry for each group
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
        'name_first': 'name_geocodegroup',
        'ref_geocodegroup_first': 'ref_geocodegroup',
        'asset_group_name_<lambda>': 'asset_group_names'  # Rename aggregated asset_group_name
    }

    grouped.rename(columns=renamed_columns, inplace=True)

    # Count the total assets in each group
    grouped['assets_total'] = intersected_data.groupby('code').size()

    return grouped


# Main function for processing data
def main(log_widget, progress_var, gpkg_file):
    log_to_gui(log_widget, "Starting processing...")
    progress_var.set(10)  # Indicate start

    # Reading 'tbl_stacked' data from the GeoPackage
    log_to_gui(log_widget, "Reading 'tbl_stacked' data...")
    asset_data = gpd.read_file(gpkg_file, layer='tbl_stacked')
    progress_var.set(25)  # Update progress after reading asset data

    # Reading 'tbl_geocode_group' data from the GeoPackage
    log_to_gui(log_widget, "Reading 'tbl_geocode_group' data...")
    geocode_group_data = gpd.read_file(gpkg_file, layer='tbl_geocode_group')
    progress_var.set(40)  # Update progress after reading geocode group data
    

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

    progress_var.set(55)  # Update progress after merging data

    # Drop the unnecessary columns (id_x and id_y)
    merged_data.drop(columns=['id_asset', 'id_geocode'], inplace=True)

    # Proceed with aggregation
    log_to_gui(log_widget, "Aggregating data...")
    aggregated_data = aggregate_data(merged_data)
    progress_var.set(70)  # Update progress after data aggregation
    

    # Save to GeoPackage
    aggregated_gdf = gpd.GeoDataFrame(aggregated_data, geometry='geometry_first')
    aggregated_gdf.to_file(gpkg_file, layer='tbl_flat', driver='GPKG')
    progress_var.set(85)  # Update progress after saving data

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

# Create a frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=5, fill=tk.X)

# Add buttons for operations within the button frame
run_btn = ttk.Button(button_frame, text="Run Analysis", command=lambda: threading.Thread(
    target=run_main, args=(log_widget, progress_var, gpkg_file), daemon=True).start())
run_btn.pack(side=tk.LEFT, padx=5, expand=True)  # Adjust button placement and padding

close_btn = ttk.Button(button_frame, text="Close", command=lambda: close_application(root))
close_btn.pack(side=tk.LEFT, padx=5, expand=True)  # Adjust button placement and padding


# Load configuration settings
config_file = 'config.ini'
config = read_config(config_file)
gpkg_file = config['DEFAULT']['gpkg_file']

root.mainloop()
 