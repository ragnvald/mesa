import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import threading
import geopandas as gpd
from sqlalchemy import create_engine
import configparser
import datetime
import glob
import os
from osgeo import ogr
import pandas as pd
from sqlalchemy import exc
from shapely.geometry import box


# Read the configuration file
def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config


# Logging function to write to the GUI log and log file
def log_to_gui(log_widget, message):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted_message = f"{timestamp} - {message}"
    log_widget.insert(tk.END, formatted_message + "\n")
    log_widget.see(tk.END)
    with open("log.txt", "a") as log_file:
        log_file.write(formatted_message + "\n")

# Get bounding box in EPSG:4326
def get_bounding_box(data):
    bbox = data.total_bounds
    bbox_geom = box(*bbox)
    return bbox_geom


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


# Function to process a layer and add to asset objects
def process_asset_layer(data, asset_objects, object_id_counter, group_id, layer_name, log_widget):
    if data.empty:
        log_to_gui(log_widget, f"No data found in layer {layer_name}")
        return object_id_counter

    # Create a temporary GeoSeries for area calculation if the CRS is geographic
    if data.crs.is_geographic:
        temp_data = data.copy()
        temp_data.geometry = temp_data.geometry.to_crs("EPSG:3395")

        # Calculate area in the temporary projected system
        area_m2_series = temp_data.geometry.area
    else:
        # Calculate area in the original system
        area_m2_series = data.geometry.area

    for index, row in data.iterrows():
        attributes = '; '.join([f"{col}: {row[col]}" for col in data.columns if col != 'geometry'])
        area_m2 = area_m2_series.iloc[index] if row.geometry.geom_type == 'Polygon' else 0

        asset_objects.append({
            'id': int(object_id_counter),
            'ref_asset_group': int(group_id),
            'asset_group_name': layer_name,
            'attributes': attributes,
            'process': True,
            'area_m2': int(area_m2),
            'geom': row.geometry  # Original geometry in EPSG:4326
        })
        object_id_counter += 1

    return object_id_counter


# Function to process a layer and add to groups and objects
def process_geocode_layer(data, geocode_groups, geocode_objects, group_id_counter, object_id_counter, layer_name, log_widget):
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
def process_geocode_file(filepath, geocode_groups, geocode_objects, group_id_counter, object_id_counter, log_widget):
    if filepath.endswith('.gpkg'):
        ds = ogr.Open(filepath)
        for i in range(ds.GetLayerCount()):
            layer = ds.GetLayerByIndex(i)
            layer_name = layer.GetName()
            data = read_and_reproject(filepath, layer=layer_name)
            group_id_counter, object_id_counter = process_geocode_layer(
                data, geocode_groups, geocode_objects, group_id_counter, object_id_counter, layer_name, log_widget)
        ds = None
    else:
        data = read_and_reproject(filepath)
        layer_name = os.path.splitext(os.path.basename(filepath))[0]
        group_id_counter, object_id_counter = process_geocode_layer(
            data, geocode_groups, geocode_objects, group_id_counter, object_id_counter, layer_name, log_widget)
    return group_id_counter, object_id_counter

# Import spatial data and export to geopackage
def import_spatial_data_grid(input_folder_grid, log_widget, progress_var):
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

            group_id_counter, object_id_counter = process_geocode_file(
                filepath, geocode_groups, geocode_objects, group_id_counter, object_id_counter, log_widget)

            processed_files += 1
            progress_var.set(10 + processed_files * progress_increment)  # Update progress after processing each file
            update_progress(10 + processed_files * progress_increment)


    geocode_groups_gdf = gpd.GeoDataFrame(geocode_groups, geometry='geom' if geocode_groups else None)
    geocode_objects_gdf = gpd.GeoDataFrame(geocode_objects, geometry='geom' if geocode_objects else None)
    
    log_to_gui(log_widget, f"Total geocodes added: {object_id_counter - 1}")
    return geocode_groups_gdf, geocode_objects_gdf

# Thread function to run import without freezing GUI
def run_import_geocode(input_folder_grid, gpkg_file, log_widget, progress_var):
    geocode_groups_gdf, geocode_objects_gdf = import_spatial_data_grid(input_folder_grid, log_widget, progress_var)
    
    log_to_gui(log_widget, f"Preparing to export geocode groups and objects.")

    if not geocode_groups_gdf.empty:
        export_to_geopackage(geocode_groups_gdf, gpkg_file, 'tbl_geocode_group', log_widget)

    if not geocode_objects_gdf.empty:
        export_to_geopackage(geocode_objects_gdf, gpkg_file, 'tbl_geocode_object', log_widget)

    log_to_gui(log_widget, "Import completed.")
    progress_var.set(100)
    update_progress(100)


# Function to import spatial data for assets
def import_spatial_data_asset(input_folder_asset, log_widget, progress_var):
    asset_objects = []
    asset_groups = {}
    group_id_counter = 1
    object_id_counter = 1
    file_patterns = ['*.shp', '*.gpkg']
    total_files = sum([len(glob.glob(os.path.join(input_folder_asset, '**', pattern), recursive=True)) for pattern in file_patterns])
    processed_files = 0

    for pattern in file_patterns:
        for filepath in glob.glob(os.path.join(input_folder_asset, '**', pattern), recursive=True):
            try:
                filename = os.path.basename(filepath)
                log_to_gui(log_widget, f"Processing file: {filename}")
                if filepath.endswith('.gpkg'):
                    ds = ogr.Open(filepath)
                    if ds is None:
                        log_to_gui(log_widget, f"No layers found in GeoPackage: {filepath}")
                        continue

                    for i in range(ds.GetLayerCount()):
                        layer = ds.GetLayerByIndex(i)
                        if layer.GetGeomType() != 0:  # Check if the layer is spatial
                            layer_name = layer.GetName()
                            data = read_and_reproject(filepath, layer_name=layer_name)
                            if layer_name not in asset_groups:
                                bbox_geom = get_bounding_box(data)
                                asset_groups[layer_name] = {
                                    'id': int(group_id_counter),
                                    'name_original': layer_name,
                                    'name_fromuser': layer_name,
                                    'date_import': datetime.datetime.now(),
                                    'bounding_box_geom': bbox_geom.wkt,
                                    'total_asset_objects': int(0),
                                    'importance': int(0),
                                    'susceptibility': int(0),
                                    'sensitivity': int(0)
                                }
                                group_id_counter += 1
                            group_id = asset_groups[layer_name]['id']
                            object_id_counter = process_asset_layer(
                                data, asset_objects, object_id_counter, group_id, layer_name, log_widget)
                    ds = None
                else:
                    data = read_and_reproject(filepath)
                    asset_group_name = os.path.splitext(os.path.basename(filepath))[0]
                    if asset_group_name not in asset_groups:
                        bbox_geom = get_bounding_box(data)
                        asset_groups[asset_group_name] = {
                            'id': int(group_id_counter),
                            'name_original': asset_group_name,
                            'name_fromuser': '',
                            'date_import': datetime.datetime.now(),
                            'bounding_box_geom': bbox_geom.wkt,
                            'total_asset_objects': int(0),
                            'importance': int(0),
                            'susceptibility': int(0),
                            'sensitivity': int(0)
                        }
                        group_id_counter += 1
                    group_id = asset_groups[asset_group_name]['id']
                    object_id_counter = process_asset_layer(
                        data, asset_objects, object_id_counter, group_id, asset_group_name, log_widget)

                processed_files += 1
                progress_var.set(processed_files / total_files * 100)
            except Exception as e:
                log_to_gui(log_widget, f"Error processing file {filepath}: {e}")
    
    # Convert dictionary to DataFrame and list of asset objects to GeoDataFrame
    asset_groups_df = pd.DataFrame(asset_groups.values())
    asset_objects_gdf = gpd.GeoDataFrame(asset_objects, geometry='geom')

    # Calculate total bounding box for all asset objects
    if not asset_objects_gdf.empty:
        total_bbox = asset_objects_gdf.geometry.unary_union.bounds
        total_bbox_geom = box(*total_bbox)
        log_to_gui(log_widget, f"Total bounding box for all assets imported.")

    # Ensure the id column is of type int64
    asset_groups_df['id'] = asset_groups_df['id'].astype('int64')
    asset_objects_gdf['id'] = asset_objects_gdf['id'].astype('int64')

    return asset_objects_gdf, asset_groups_df, total_bbox_geom


# Function to export to geopackage
def export_to_geopackage(gdf, gpkg_file, layer_name, log_widget):
    engine = create_engine(f'sqlite:///{gpkg_file}')
    gdf.to_file(gpkg_file, layer=layer_name, driver="GPKG", if_exists='append')
    log_to_gui(log_widget, f"Data exported to {gpkg_file}, layer {layer_name}")


# Function to update asset groups in geopackage
def update_asset_groups(asset_groups_df, gpkg_file, log_widget):
    engine = create_engine(f'sqlite:///{gpkg_file}')

    # Define the data types for the ID columns
    id_col = asset_groups_df.columns[asset_groups_df.dtypes == 'int64']
    asset_groups_df[id_col] = asset_groups_df[id_col].astype(int)

    try:
        asset_groups_df.to_sql('tbl_asset_group', con=engine, if_exists='replace', index=False)
        log_to_gui(log_widget, "Asset groups updated in GeoPackage.")
    except exc.SQLAlchemyError as e:
        log_to_gui(log_widget, f"Failed to update asset groups: {e}")


# Thread function to run import without freezing GUI
def run_import_asset(input_folder_asset, gpkg_file, log_widget, progress_var):
    log_to_gui(log_widget, "Starting asset import process...")
    
    asset_objects_gdf, asset_groups_df, total_bbox_geom = import_spatial_data_asset(input_folder_asset, log_widget, progress_var)

    export_to_geopackage(asset_objects_gdf, gpkg_file, 'tbl_asset_object', log_widget)
    
    update_asset_groups(asset_groups_df, gpkg_file, log_widget)
    
    log_to_gui(log_widget, "Asset import completed.")
    
    progress_var.set(100)


# Function to close the application
def close_application():
    root.destroy()

# Create the user interface
root = tk.Tk()
root.title("Import assets")


# Create a log widget
log_widget = scrolledtext.ScrolledText(root, height=10)
log_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)


# Create a progress bar
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root, orient="horizontal", length=200, mode="determinate", variable=progress_var)
progress_bar.pack(pady=5, fill=tk.X)


progress_label = tk.Label(root, text="0%", bg="light grey")
progress_label.place(in_=progress_bar, relx=0.5, rely=0.5, anchor="center")


# Information text field below the progress bar
info_label_text = ("Assets are all shapefiles or geopackage files with their layers "
                   "that are placed in the folder input/assets-folder. The features will "
                   "be placed in our database and used in the analysis. All assets will "
                   "be associated with importance and susceptibility values.")
info_label = tk.Label(root, text=info_label_text, wraplength=500, justify="left")
info_label.pack(padx=10, pady=10)


# Create a frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=5)


# Add buttons for the different operations within the button frame
import_btn = ttk.Button(button_frame, text="Import assets", command=lambda: threading.Thread(
    target=run_import_asset, args=(input_folder_asset, gpkg_file, log_widget, progress_var), daemon=True).start())
import_btn.pack(side=tk.LEFT, padx=10)

# Add buttons for the different operations within the button frame
import_btn = ttk.Button(button_frame, text="Import geocodes", command=lambda: threading.Thread(
    target=run_import_geocode, args=(input_folder_geocode, gpkg_file, log_widget, progress_var), daemon=True).start())
import_btn.pack(side=tk.LEFT, padx=10)

close_btn = ttk.Button(button_frame, text="Close", command=close_application)
close_btn.pack(side=tk.LEFT, padx=10)


# Load configuration settings
config_file = 'config.ini'
config = read_config(config_file)
input_folder_asset = config['DEFAULT']['input_folder_asset']
input_folder_geocode = config['DEFAULT']['input_folder_grid']
gpkg_file = config['DEFAULT']['gpkg_file']

root.mainloop()