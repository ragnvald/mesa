import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import threading
import geopandas as gpd
from sqlalchemy import create_engine
from shapely.geometry import box
import configparser
import datetime
import os
import glob
from osgeo import ogr

# Read the configuration file
def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config

def update_progress(new_value):
    progress_var.set(new_value)
    progress_label.config(text=f"{int(new_value)}%")

# Logging function to write to the GUI log
def log_to_gui(log_widget, message):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted_message = f"{timestamp} - {message}"
    log_widget.insert(tk.END, formatted_message + "\n")
    log_widget.see(tk.END)
    with open("log.txt", "a") as log_file:
        log_file.write(formatted_message + "\n")

# Function to read and reproject spatial data
def read_and_reproject(filepath, layer=None):
    data = gpd.read_file(filepath, layer=layer)
    if data.crs is None:
        log_to_gui(log_widget, f"Warning: No CRS found for {filepath}. Setting CRS to EPSG:4326.")
        data.set_crs(epsg=4326, inplace=True)
    elif data.crs.to_epsg() != 4326:
        data = data.to_crs(epsg=4326)
    return data

# Function to process a layer and add to groups and objects
def process_layer(data, geocode_groups, geocode_objects, group_id_counter, object_id_counter, layer_name, log_widget):
    if data.empty:
        log_to_gui(log_widget, f"No data found in layer {layer_name}")
        return group_id_counter, object_id_counter

    feature_count = len(data)
    log_to_gui(log_widget, f"{layer_name} ({feature_count} features)")

    # Calculate bounding box and add to geocode groups
    bounding_box = data.total_bounds
    bbox_geom = box(*bounding_box)
    geocode_groups.append({
        'id': group_id_counter,  # Group ID
        'name': layer_name,
        'description': f'Description for {layer_name}',
        'geom': bbox_geom
    })

    # Add geocode objects with unique IDs
    for index, row in data.iterrows():
        geom = row.geometry if 'geometry' in data.columns else None
        code = row['qdgc'] if 'qdgc' in data.columns else object_id_counter
        geocode_objects.append({
            'pk_id': object_id_counter,  # Primary Key ID
            'id': object_id_counter,  # Object ID
            'code': code,
            'ref_geocodegroup': group_id_counter,
            'geom': geom
        })
        object_id_counter += 1

    return group_id_counter + 1, object_id_counter

# Function to export to geopackage
def export_to_geopackage(geocode_groups_gdf, geocode_objects_gdf, gpkg_file, log_widget):
    engine = create_engine(f'sqlite:///{gpkg_file}')
    try:
        if not geocode_groups_gdf.empty:
            geocode_groups_gdf.to_file(gpkg_file, layer='tbl_geocode_group', driver="GPKG", if_exists='append')
            log_to_gui(log_widget, f"Exported {len(geocode_groups_gdf)} groups to {gpkg_file}")

        if not geocode_objects_gdf.empty:
            log_to_gui(log_widget, f"Attempting to export {len(geocode_objects_gdf)} objects to {gpkg_file}")
            geocode_objects_gdf.to_file(gpkg_file, layer='tbl_geocode_object', driver="GPKG", if_exists='append')
            log_to_gui(log_widget, f"Exported {len(geocode_objects_gdf)} objects to {gpkg_file}")

    except Exception as e:
        log_to_gui(log_widget, f"Error during export: {e}")

# Function to process each file
def process_file(filepath, geocode_groups, geocode_objects, group_id_counter, object_id_counter, log_widget):
    if filepath.endswith('.gpkg'):
        ds = ogr.Open(filepath)
        for i in range(ds.GetLayerCount()):
            layer = ds.GetLayerByIndex(i)
            layer_name = layer.GetName()
            data = read_and_reproject(filepath, layer=layer_name)
            group_id_counter, object_id_counter = process_layer(
                data, geocode_groups, geocode_objects, group_id_counter, object_id_counter, layer_name, log_widget)
        ds = None
    else:
        data = read_and_reproject(filepath)
        layer_name = os.path.splitext(os.path.basename(filepath))[0]
        group_id_counter, object_id_counter = process_layer(
            data, geocode_groups, geocode_objects, group_id_counter, object_id_counter, layer_name, log_widget)
    return group_id_counter, object_id_counter

# Import spatial data and export to geopackage
def import_spatial_data(input_folder_grid, log_widget, progress_var):
    geocode_groups = []
    geocode_objects = []
    group_id_counter = 1
    object_id_counter = 1
    file_patterns = ['*.shp', '*.gpkg']
    total_files = sum([len(glob.glob(os.path.join(input_folder_grid, '**', pattern), recursive=True)) for pattern in file_patterns])
    processed_files = 0
    progress_increment = 70 / total_files  # Distribute 70% of progress bar over file processing

    log_to_gui(log_widget, "Working with imports...")
    progress_var.set(10)  # Initial progress after starting
    update_progress(10)

    for pattern in file_patterns:
        for filepath in glob.glob(os.path.join(input_folder_grid, '**', pattern), recursive=True):
            log_to_gui(log_widget, f"Processing file: {filepath}")
            progress_var.set(10 + processed_files * progress_increment)  # Update progress before processing each file

            group_id_counter, object_id_counter = process_file(
                filepath, geocode_groups, geocode_objects, group_id_counter, object_id_counter, log_widget)

            processed_files += 1
            progress_var.set(10 + processed_files * progress_increment)  # Update progress after processing each file
            update_progress(10 + processed_files * progress_increment)


    geocode_groups_gdf = gpd.GeoDataFrame(geocode_groups, geometry='geom' if geocode_groups else None)
    geocode_objects_gdf = gpd.GeoDataFrame(geocode_objects, geometry='geom' if geocode_objects else None)
    
    log_to_gui(log_widget, f"Total geocodes added: {object_id_counter - 1}")
    return geocode_groups_gdf, geocode_objects_gdf

# Thread function to run import without freezing GUI
def run_import(input_folder_grid, gpkg_file, log_widget, progress_var):
    geocode_groups_gdf, geocode_objects_gdf = import_spatial_data(input_folder_grid, log_widget, progress_var)
    
    log_to_gui(log_widget, f"Preparing to export {len(geocode_objects_gdf)} geocode objects.")
    
    export_to_geopackage(geocode_groups_gdf, geocode_objects_gdf, gpkg_file, log_widget)
    log_to_gui(log_widget, "Import completed.")
    progress_var.set(100)

# Function to close the application
def close_application():
    root.destroy()

# Create the user interface
root = tk.Tk()
root.title("MESA Import Utility")

# Create a log widget
log_widget = scrolledtext.ScrolledText(root, height=10)
log_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Create a progress bar
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root, orient="horizontal", length=200, mode="determinate", variable=progress_var)
progress_bar.pack(pady=5, fill=tk.X)

# Create a label to show progress percentage on the progress bar
progress_label = tk.Label(root, text="0%", bg="light grey")
progress_label.place(in_=progress_bar, relx=0.5, rely=0.5, anchor="center")

# Information text field below the progress bar
info_label_text = ("Geocodes can be of any shape. In this context they area usually rectangular "
                   "grid cells. Geocodes are impoted per shapefile or by layers within a "
                   "geopackage file in the folder input/grid. you may adjust the geocode "
                   "group name and description at a later stage")
info_label = tk.Label(root, text=info_label_text, wraplength=500, justify="left")
info_label.pack(padx=10, pady=10)

# Create a frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=5)

# Add buttons for the different operations within the button frame
import_btn = ttk.Button(button_frame, text="Import Data", command=lambda: threading.Thread(
    target=run_import, args=(input_folder_grid, gpkg_file, log_widget, progress_var), daemon=True).start())
import_btn.pack(side=tk.LEFT, padx=10)

close_btn = ttk.Button(button_frame, text="Close", command=close_application)
close_btn.pack(side=tk.LEFT, padx=10)

# Load configuration settings
config_file = 'config.ini'
config = read_config(config_file)
input_folder_grid = config['DEFAULT']['input_folder_grid']
gpkg_file = config['DEFAULT']['gpkg_file']

root.mainloop()
