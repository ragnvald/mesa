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


# Read and reproject spatial file to EPSG:4326
def read_spatial_file(filepath, layer_name=None):
    if layer_name:
        data = gpd.read_file(filepath, layer=layer_name)
    else:
        data = gpd.read_file(filepath)

    if data.crs is None:
        data.set_crs(epsg=4326, inplace=True)
    elif data.crs.to_epsg() != 4326:
        data = data.to_crs(epsg=4326)
    return data


# Get bounding box in EPSG:4326
def get_bounding_box(data):
    bbox = data.total_bounds
    bbox_geom = box(*bbox)
    return bbox_geom

# Function to process a layer and add to asset objects
def process_layer(data, asset_objects, object_id_counter, group_id, layer_name, log_widget):
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


# Function to import spatial data for assets
def import_spatial_data(input_folder_asset, log_widget, progress_var):
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
                            data = read_spatial_file(filepath, layer_name=layer_name)
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
                            object_id_counter = process_layer(
                                data, asset_objects, object_id_counter, group_id, layer_name, log_widget)
                    ds = None
                else:
                    data = read_spatial_file(filepath)
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
                    object_id_counter = process_layer(
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
def run_import(input_folder_asset, gpkg_file, log_widget, progress_var):
    log_to_gui(log_widget, "Starting asset import process...")
    
    asset_objects_gdf, asset_groups_df, total_bbox_geom = import_spatial_data(input_folder_asset, log_widget, progress_var)

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
import_btn = ttk.Button(button_frame, text="Import Assets", command=lambda: threading.Thread(
    target=run_import, args=(input_folder_asset, gpkg_file, log_widget, progress_var), daemon=True).start())
import_btn.pack(side=tk.LEFT, padx=10)

close_btn = ttk.Button(button_frame, text="Close", command=close_application)
close_btn.pack(side=tk.LEFT, padx=10)


# Load configuration settings
config_file = 'config.ini'
config = read_config(config_file)
input_folder_asset = config['DEFAULT']['input_folder_asset']
gpkg_file = config['DEFAULT']['gpkg_file']

root.mainloop()