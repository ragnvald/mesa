import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import threading
import geopandas as gpd
import pandas as pd
import configparser
from shapely.geometry import box
import datetime

# Function to read the configuration file
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

# Function to close the application
def close_application(root):
    root.destroy()

# Thread function to run main without freezing GUI
def run_main(log_widget, progress_var, gpkg_file):
    main(log_widget, progress_var, gpkg_file)

# Function to filter and update atlas geometries
def filter_and_update_atlas_geometries(atlas_geometries, tbl_flat):
    log_to_gui(log_widget, "Filtering and updating atlas geometries...")
    atlas_gdf = gpd.GeoDataFrame(atlas_geometries, columns=['id', 'name_gis', 'title_user', 'geom'])
    atlas_gdf.set_geometry('geom', inplace=True)
    filtered_indices = atlas_gdf.geometry.apply(lambda geom: tbl_flat.intersects(geom).any())
    intersecting_geometries = atlas_gdf[filtered_indices].copy()
    id_counter = 1
    for index, row in intersecting_geometries.iterrows():
        intersecting_geometries.loc[index, 'name_gis'] = f'atlas{id_counter:03}'
        intersecting_geometries.loc[index, 'title_user'] = f'Map title for {id_counter:03}'
        id_counter += 1
    return intersecting_geometries

# Function to generate atlas geometries
def generate_atlas_geometries(tbl_flat, atlas_lon_size_km, atlas_lat_size_km, atlas_overlap_percent):
    log_to_gui(log_widget, "Generating atlas geometries...")
    lon_size_deg = atlas_lon_size_km / 111
    lat_size_deg = atlas_lat_size_km / 111
    overlap = atlas_overlap_percent / 100
    bounds = tbl_flat.total_bounds
    minx, miny, maxx, maxy = bounds
    atlas_geometries = []
    id_counter = 1
    y = miny
    while y < maxy:
        x = minx
        while x < maxx:
            geom = box(x, y, x + lon_size_deg, y + lat_size_deg)
            atlas_geometries.append({'id': id_counter, 'name_gis': f'atlas{id_counter:03}', 'title_user': f'Map title for {id_counter:03}', 'geom': geom})
            id_counter += 1
            x += lon_size_deg * (1 - overlap)
        y += lat_size_deg * (1 - overlap)
    return atlas_geometries

# Main function with GUI integration
def main(log_widget, progress_var, gpkg_file):
    log_to_gui(log_widget, "Starting processing...")
    progress_var.set(10)  # Indicate start

    # Read configuration
    config = read_config('config.ini')
    atlas_lon_size_km = float(config['DEFAULT']['atlas_lon_size_km'])
    atlas_lat_size_km = float(config['DEFAULT']['atlas_lat_size_km'])
    atlas_overlap_percent = float(config['DEFAULT']['atlas_overlap_percent'])

    # Load tbl_flat from GeoPackage
    tbl_flat = gpd.read_file(gpkg_file, layer='tbl_flat')
    progress_var.set(30)

    # Generate atlas geometries
    atlas_geometries = generate_atlas_geometries(tbl_flat, atlas_lon_size_km, atlas_lat_size_km, atlas_overlap_percent)
    progress_var.set(60)

    # Filter atlas geometries and update name_gis
    updated_atlas_geometries = filter_and_update_atlas_geometries(atlas_geometries, tbl_flat)
    progress_var.set(80)

    # Save updated geometries to GeoPackage
    updated_atlas_geometries.to_file(gpkg_file, layer='tbl_atlas', driver='GPKG')
    progress_var.set(100)
    log_to_gui(log_widget, "Completed processing.")

# Create the user interface
root = tk.Tk()
root.title("Atlas Generation and Update")

# Create a log widget
log_widget = scrolledtext.ScrolledText(root, height=10)
log_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Create a progress bar
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root, orient="horizontal", length=200, mode="determinate", variable=progress_var)
progress_bar.pack(pady=5, fill=tk.X)

# Information text field above the buttons
info_label_text = ("This tool generates and updates atlas geometries based on the provided configurations.")
info_label = tk.Label(root, text=info_label_text, wraplength=500, justify="left")
info_label.pack(padx=10, pady=10)

# Create a frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=5)

# Add 'Run' button to the button frame
run_btn = ttk.Button(button_frame, text="Run", command=lambda: threading.Thread(
    target=run_main, args=(log_widget, progress_var, gpkg_file), daemon=True).start())
run_btn.pack(side=tk.LEFT, padx=5, expand=False, fill=tk.X)

# Add 'Close' button to the button frame
close_btn = ttk.Button(button_frame, text="Close", command=lambda: close_application(root))
close_btn.pack(side=tk.LEFT, padx=5, expand=False, fill=tk.X)

# Load configuration settings
config_file = 'config.ini'
config = read_config(config_file)
gpkg_file = config['DEFAULT']['gpkg_file']

root.mainloop()