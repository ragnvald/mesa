# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Creating atlases - this can be made in at least forur different ways:
# - Importing polygons designed by the user
# - Generating based on a minimum size (rectangular x/y) by the user
# - Generating based on number of plates x and y set by the user
# - Trying to cover the asset data as best as it can not using a 
#   bounding box as the outer limits but rather a flexible polygon

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import geopandas as gpd
import configparser
from shapely.geometry import box
import datetime
import argparse
import ttkbootstrap as ttk  # Import ttkbootstrap
from ttkbootstrap.constants import *
import glob
import os


# # # # # # # # # # # # # # 
# Shared/general functions

def update_progress(new_value):
    progress_var.set(new_value)
    progress_label.config(text=f"{int(new_value)}%")


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
    log_destination_file = os.path.join(original_working_directory, "log.txt")
    with open(log_destination_file, "a") as log_file:
        log_file.write(formatted_message + "\n")


# Function to close the application
def close_application(root):
    root.destroy()


# # # # # # # # # # # # # # 
# Core functions
    
# Thread function to run main without freezing GUI
def run_create_atlas(log_widget, progress_var, gpkg_file):
    main_create_atlas(log_widget, progress_var, gpkg_file)


# Function to filter and update atlas geometries
def filter_and_update_atlas_geometries(atlas_geometries, tbl_flat):
    log_to_gui(log_widget, "Filtering and updating atlas geometries...")
    atlas_gdf = gpd.GeoDataFrame(atlas_geometries, columns=['id', 'name_gis', 'title_user', 'geom', 'description', 'image_name_1', 'image_desc_1', 'image_name_2', 'image_desc_2'])
    atlas_gdf.set_geometry('geom', inplace=True)
    filtered_indices = atlas_gdf.geometry.apply(lambda geom: tbl_flat.intersects(geom).any())
    intersecting_geometries = atlas_gdf[filtered_indices].copy()
    id_counter = 1
    for index, row in intersecting_geometries.iterrows():
        intersecting_geometries.loc[index, 'name_gis'] = f'atlas_{id_counter:03}'
        intersecting_geometries.loc[index, 'title_user'] = f'Map title for {id_counter:03}'
        id_counter += 1
    return intersecting_geometries


# Function to generate atlas geometries. This is done by making a bounding box around the gridded data
# which can be found in tbl_flat.
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
            atlas_geometries.append({
                'id': id_counter, 
                'name_gis': f'atlas{id_counter:03}', 
                'title_user': f'Map title for {id_counter:03}', 
                'geom': geom,
                'description': '',   # Default empty value
                'image_name_1': '',  # Default empty value
                'image_desc_1': '',  # Default empty value
                'image_name_2': '',  # Default empty value
                'image_desc_2': ''   # Default empty value
            })
            id_counter += 1
            x += lon_size_deg * (1 - overlap)
        y += lat_size_deg * (1 - overlap)
    return atlas_geometries


# Main function with GUI integration
def main_create_atlas(log_widget, progress_var, gpkg_file):
    log_to_gui(log_widget, "Starting processing...")
    progress_var.set(10)  # Indicate start

    # Load tbl_flat from GeoPackage
    tbl_flat = gpd.read_file(gpkg_file, layer='tbl_flat')
    update_progress(30)

    # Generate atlas geometries
    atlas_geometries = generate_atlas_geometries(tbl_flat, atlas_lon_size_km, atlas_lat_size_km, atlas_overlap_percent)
    update_progress(60)

    # Filter atlas geometries and update name_gis
    updated_atlas_geometries = filter_and_update_atlas_geometries(atlas_geometries, tbl_flat)
    update_progress(80)

    # Save updated geometries to GeoPackage
    updated_atlas_geometries.to_file(gpkg_file, layer='tbl_atlas', driver='GPKG')
    update_progress(100)
    log_to_gui(log_widget, "COMPLETED: Atlas creation done. Old ones deleted.")

    increment_stat_value(config_file, 'mesa_stat_create_atlas', increment_value=1)


# Process the spatial file. Make sure it is a single polygon, not a multipolygon or any other geometry.
def process_spatial_file(filepath, atlas_objects, atlas_id_counter, log_widget):
    try:
        # Load spatial data from file
        gdf = gpd.read_file(filepath)

        # Filter for polygon geometries
        polygons = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]

        # Iterate over each polygon feature
        for _, row in polygons.iterrows():
            atlas_object = {
                'id': atlas_id_counter, 
                'name_gis': f'atlas{atlas_id_counter:03}',
                'title_user': f'Map title for {atlas_id_counter:03}', 
                'geom': row.geometry,
                'description': '',   # Default empty value
                'image_name_1': '',  # Default empty value
                'image_desc_1': '',  # Default empty value
                'image_name_2': '',  # Default empty value
                'image_desc_2': ''   # Default empty value
            }
            atlas_objects.append(atlas_object)
            atlas_id_counter += 1
        log_to_gui(log_widget, f"Processed file: {filepath}")
    except Exception as e:
        log_to_gui(log_widget, f"Error processing file {filepath}: {e}")

    return atlas_id_counter



# Import atlas objects from the atlas folder.
def import_atlas_objects(input_folder_atlas, log_widget, progress_var):
    atlas_objects = []
    atlas_id_counter = 1
    file_patterns = ['*.shp', '*.gpkg']
    total_files = sum([len(glob.glob(os.path.join(input_folder_atlas, '**', pattern), recursive=True)) for pattern in file_patterns])
    processed_files = 0

    if total_files == 0:
        progress_increment = 70
    else:
        progress_increment = 70 / total_files  # Distribute 70% of progress bar over file processing

    log_to_gui(log_widget, "Working with imports...")
    update_progress(10)  # Initial progress after starting

    for pattern in file_patterns:
        for filepath in glob.glob(os.path.join(input_folder_atlas, '**', pattern), recursive=True):
            try:
                log_to_gui(log_widget, f"Processing file: {filepath}")
                atlas_id_counter = process_spatial_file(filepath, atlas_objects, atlas_id_counter, log_widget)
                processed_files += 1
                update_progress(10 + processed_files * progress_increment)  # Update progress after processing each file

            except Exception as e:
                log_to_gui(log_widget, f"Error processing file {filepath}: {e}")

    # Creating a GeoDataFrame from the list of atlas objects, ensuring the 'geom' column is set as the geometry
    if atlas_objects:
        atlas_objects_gdf = gpd.GeoDataFrame(atlas_objects, geometry='geom')
    else:
        atlas_objects_gdf = gpd.GeoDataFrame(columns=['id', 'name_gis', 'title_user', 'description', 'image_name_1', 'image_desc_1', 'image_name_2', 'image_desc_2', 'geom'])

    update_progress(100)

    log_to_gui(log_widget, f"Total atlas polygons added: {atlas_id_counter - 1}")

    increment_stat_value(config_file, 'mesa_stat_import_atlas', increment_value=1)

    return atlas_objects_gdf


# Thread function to run import without freezing GUI
def run_import_atlas(input_folder_atlas, gpkg_file, log_widget, progress_var):
    log_to_gui(log_widget, "Starting atlas import process...")

    atlas_objects_gdf = import_atlas_objects(input_folder_atlas, log_widget, progress_var)

    # Check if the GeoDataFrame is not empty before exporting
    if not atlas_objects_gdf.empty:  # Corrected atlas
        log_to_gui(log_widget, "Importing atlas objects to GeoPackage")
        atlas_objects_gdf.to_file(gpkg_file, layer='tbl_atlas', driver='GPKG', if_exists='replace')
    else:
        log_to_gui(log_widget, "No atlas objects to export.")

    log_to_gui(log_widget, "COMPLETED: Atlas polygons imported. Old ones deleted.")
    progress_var.set(100)


def increment_stat_value(config_file, stat_name, increment_value):
    # Check if the config file exists
    if not os.path.isfile(config_file):
        log_to_gui(log_widget, f"Configuration file {config_file} not found.")
        return

    # Read the entire config file to preserve the layout and comments
    with open(config_file, 'r') as file:
        lines = file.readlines()

    # Initialize a flag to check if the variable was found and updated
    updated = False

    # Update the specified variable's value if it exists
    for i, line in enumerate(lines):
        if line.strip().startswith(f'{stat_name} ='):
            # Extract the current value, increment it, and update the line
            parts = line.split('=')
            if len(parts) == 2:
                current_value = parts[1].strip()
                try:
                    # Attempt to convert the current value to an integer and increment it
                    new_value = int(current_value) + increment_value
                    lines[i] = f"{stat_name} = {new_value}\n"
                    updated = True
                    break
                except ValueError:
                    # Handle the case where the conversion fails
                    log_to_gui(log_widget, f"Error: Current value of {stat_name} is not an integer.")
                    return

    # Write the updated content back to the file if the variable was found and updated
    if updated:
        with open(config_file, 'w') as file:
            file.writelines(lines)


#####################################################################################
#  Main
#
# original folder for the system is sent from the master executable. If the script is
# invked this way we are fetching the adress here.
parser = argparse.ArgumentParser(description='Slave script')
parser.add_argument('--original_working_directory', required=False, help='Path to running folder')
args = parser.parse_args()
original_working_directory = args.original_working_directory

# However - if this is not the case we will have to establish the root folder in 
# one of two different ways.
if original_working_directory is None or original_working_directory == '':
    
    #if it is running as a python subprocess we need to get the originating folder.
    original_working_directory  = os.getcwd()

    # When running directly separate script we need to find out and go up one level.
    if str("system") in str(original_working_directory):
        original_working_directory = os.path.join(os.getcwd(),'../')

# Load configuration settings and data
config_file             = os.path.join(original_working_directory, "system/config.ini")
gpkg_file               = os.path.join(original_working_directory, "output/mesa.gpkg")

# Load configuration settings
config                  = read_config(config_file)

input_folder_atlas      = os.path.join(original_working_directory, "input/atlas")

ttk_bootstrap_theme     = config['DEFAULT']['ttk_bootstrap_theme']
workingprojection_epsg  = config['DEFAULT']['workingprojection_epsg']

atlas_lon_size_km       = float(config['DEFAULT']['atlas_lon_size_km'])
atlas_lat_size_km       = float(config['DEFAULT']['atlas_lat_size_km'])
atlas_overlap_percent   = float(config['DEFAULT']['atlas_overlap_percent'])

icon_file               = os.path.join(original_working_directory, "system_resources/mesa.ico")

# Create the user interface
root = ttk.Window(themename=ttk_bootstrap_theme)
root.title("Create atlas tiles")
root.iconbitmap(icon_file)

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
info_label_text = ("This is where you can import, generate and update atlas geometries. "
                   "The size of an atlas frame is set in the config.ini-file. "
                   "Earlier atlas frames and their asociated information will be deleted.")
info_label = tk.Label(root, text=info_label_text, wraplength=600, justify="left")
info_label.pack(padx=10, pady=10)

# Create a frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=5)

# Add 'Import' button to the button frame
import_atlas_btn = ttk.Button(button_frame, text="Import", command=lambda: threading.Thread(
    target=run_import_atlas, args=(input_folder_atlas, gpkg_file, log_widget, progress_var, ), daemon=True).start())
import_atlas_btn.grid(row=1, column=0, columnspan=1, padx=10, pady=5, sticky='ew')

# Add 'Run' button to the button frame  
create_atlas_btn = ttk.Button(button_frame, text="Create", command=lambda: threading.Thread(
    target=run_create_atlas, args=(log_widget, progress_var, gpkg_file), daemon=True).start())
create_atlas_btn.grid(row=1, column=2, columnspan=1, padx=10, pady=5, sticky='ew')

# Add 'Close' button to the button frame
close_btn = ttk.Button(button_frame, bootstyle=WARNING, text="Exit", command=lambda: close_application(root))
close_btn.grid(row=1, column=3, columnspan=1, padx=10, pady=5, sticky='ew')

root.mainloop()