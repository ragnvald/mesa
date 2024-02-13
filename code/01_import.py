# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# This is the most complex part of the mesa python too. It imports
# files from a set folder and places them in two tables.
#
# tbl_assets:   Information about assets like mangroves ant the likes. 
#               Data is stored as polygons, points and lines.
#
# tbl_geocodes: Information about geocodes which could be grids, hexagons
#               as well as municipalities and other.

import tkinter as tk

import locale
try:
    locale.setlocale(locale.LC_ALL, 'de_DE.utf8')
except locale.Error:
    locale.setlocale(locale.LC_ALL, '') 

from tkinter import scrolledtext, ttk

from fiona import open as fiona_open    

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
import sqlite3
from shapely.geometry import box

import ttkbootstrap as ttk  # Import ttkbootstrap
from ttkbootstrap.constants import *


# # # # # # # # # # # # # # 
# Shared/general functions

# Read the configuration file
def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config


# # # # # # # # # # # # # # 
# Core functions

# Get bounding box in EPSG:4326
def get_bounding_box(data):
    bbox = data.total_bounds
    bbox_geom = box(*bbox)
    return bbox_geom


# Update progress labl
def update_progress(new_value):
    progress_var.set(new_value)
    progress_label.config(text=f"{int(new_value)}%")


# Logging function to write to the GUI log
def log_to_gui(log_widget, message):
    timestamp           = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted_message   = f"{timestamp} - {message}"
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
# Asset objects are all objects form all asset files/geopackages. In this process we will have
# to harvest all attribute data and place them in one specific attribute. The attributes are all
# placed within one attribute (attributes). At the time of writing this I am not sure if the 
# attribute name is kept here. If not it should be placed separately in tbl_asset_group.
#
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


def process_line_layer(data, line_objects, line_id_counter, layer_name, log_widget):
    if data.empty:
        log_to_gui(log_widget, f"No data found in layer {layer_name}")
        return line_id_counter

    for index, row in data.iterrows():
        attributes = '; '.join([f"{col}: {row[col]}" for col in data.columns if col != 'geometry'])

        line_objects.append({
            'name_gis': int(index),
            'name_user': layer_name,
            'attributes': attributes,
            'geom': row.geometry  # Original geometry in EPSG:4326
        })
        line_id_counter += 1

    return line_id_counter


# Function to process a geocode layer and place it in context.
# Geocode objects are all objects form all asset files/geopackages. The attributes are all
# placed within one attribute (attributes) in the tbl_geocode_object table. At the time
# of writing this I am not sure if the attribute name is kept here. If not it should be
# placed separately in tbl_geocode_group.
#
def process_geocode_layer(data, geocode_groups, geocode_objects, group_id_counter, object_id_counter, layer_name, log_widget):
    if data.empty:
        log_to_gui(log_widget, f"No data found in layer {layer_name}")
        return group_id_counter, object_id_counter

    feature_count = len(data)
    log_to_gui(log_widget, f"  {layer_name} ({feature_count} features)")

    # Calculate bounding box and add to geocode groups
    bounding_box    = data.total_bounds
    bbox_geom       = box(*bounding_box)
    name_gis_geocodegroup = f"geocode_{group_id_counter:03d}"

    geocode_groups.append({
        'id': group_id_counter,  # Group ID
        'name': layer_name,
        'name_gis_geocodegroup': name_gis_geocodegroup, 
        'title_user': layer_name,
        'description': f'Description for {layer_name}',
        'geom': bbox_geom
    })

    # Add geocode objects with unique IDs and name_gis_geocodegroup from the group
    for index, row in data.iterrows():
        geom = row.geometry if 'geometry' in data.columns else None
        code = row['qdgc'] if 'qdgc' in data.columns else object_id_counter
        geocode_objects.append({
            'code': code,
            'ref_geocodegroup': group_id_counter,
            'name_gis_geocodegroup': name_gis_geocodegroup, 
            'geom': geom
        })
        object_id_counter += 1

    return group_id_counter + 1, object_id_counter


# Function to process each file
def process_geocode_file(filepath, geocode_groups, geocode_objects, group_id_counter, object_id_counter, log_widget):
    if filepath.endswith('.gpkg'):
        ds = ogr.Open(filepath)
        for i in range(ds.GetLayerCount()):
            layer       = ds.GetLayerByIndex(i)
            layer_name  = layer.GetName()
            data        = read_and_reproject(filepath, layer=layer_name)
            log_to_gui(log_widget, f"Importing {layer_name}")
            group_id_counter, object_id_counter = process_geocode_layer(
                data, geocode_groups, geocode_objects, group_id_counter, object_id_counter, layer_name, log_widget)
        ds = None
    elif filepath.endswith('.shp'):
        data            = read_and_reproject(filepath)
        layer_name      = os.path.splitext(os.path.basename(filepath))[0]
        log_to_gui(log_widget, f"Importing {layer_name}")
        group_id_counter, object_id_counter = process_geocode_layer(
            data, geocode_groups, geocode_objects, group_id_counter, object_id_counter, layer_name, log_widget)
    else:
        log_to_gui(log_widget, f"Unsupported file format for {filepath}")
    return group_id_counter, object_id_counter


# Function to process each file
def process_line_file(filepath, line_objects, line_id_counter, log_widget):
    if filepath.endswith('.gpkg'):
        ds = ogr.Open(filepath)
        for i in range(ds.GetLayerCount()):
            layer       = ds.GetLayerByIndex(i)
            layer_name  = layer.GetName()
            data        = read_and_reproject(filepath, layer=layer_name)
            log_to_gui(log_widget, f"Importing {layer_name}")
            line_id_counter = process_line_layer(
                data, line_objects, line_id_counter, layer_name, log_widget)
        ds = None
    elif filepath.endswith('.shp'):
        data = read_and_reproject(filepath)
        layer_name = os.path.splitext(os.path.basename(filepath))[0]
        log_to_gui(log_widget, f"Importing {layer_name}")
        line_id_counter = process_line_layer(
            data, line_objects, line_id_counter, layer_name, log_widget)
    else:
        log_to_gui(log_widget, f"Unsupported file format for {filepath}")
    return line_id_counter


# Import spatial data and export to geopackage
def import_spatial_data_geocode(input_folder_geocode, log_widget, progress_var):
    geocode_groups      = []
    geocode_objects     = []
    group_id_counter    = 1
    object_id_counter   = 1
    file_patterns       = ['*.shp', '*.gpkg']
    total_files         = sum([len(glob.glob(os.path.join(input_folder_geocode, '**', pattern), recursive=True)) for pattern in file_patterns])
    processed_files     = 0

    if total_files == 0:
        progress_increment = 70
    else:
        progress_increment = 70 / total_files  # Distribute 70% of progress bar over file processing

    log_to_gui(log_widget, "Working with imports...")
    progress_var.set(10)  # Initial progress after starting
    update_progress(10)

    for pattern in file_patterns:
        for filepath in glob.glob(os.path.join(input_folder_geocode, '**', pattern), recursive=True):
            try:
                log_to_gui(log_widget, f"Processing layer: {os.path.splitext(os.path.basename(filepath))[0]}")
                progress_var.set(10 + processed_files * progress_increment)  # Update progress before processing each file

                group_id_counter, object_id_counter = process_geocode_file(
                    filepath, geocode_groups, geocode_objects, group_id_counter, object_id_counter, log_widget)

                processed_files += 1
                progress_var.set(10 + processed_files * progress_increment)  # Update progress after processing each file
                update_progress(10 + processed_files * progress_increment)

            except Exception as e:
                log_to_gui(log_widget, f"Error processing file {filepath}: {e}")

    geocode_groups_gdf = gpd.GeoDataFrame(geocode_groups, geometry='geom' if geocode_groups else None)
    geocode_objects_gdf = gpd.GeoDataFrame(geocode_objects, geometry='geom' if geocode_objects else None)
    
    update_progress(90)

    log_to_gui(log_widget, f"Total geocodes added: {object_id_counter - 1}")
    return geocode_groups_gdf, geocode_objects_gdf


# Import line data and export to geopackage
def import_spatial_data_lines(input_folder_lines, log_widget, progress_var):
    line_objects    = []
    line_id_counter = 1
    file_patterns   = ['*.shp', '*.gpkg']
    total_files     = sum([len(glob.glob(os.path.join(input_folder_lines, '**', pattern), recursive=True)) for pattern in file_patterns])
    processed_files = 0

    if total_files == 0:
        progress_increment = 70
    else:
        progress_increment = 70 / total_files  # Distribute 70% of progress bar over file processing

    log_to_gui(log_widget, "Working with imports...")
    progress_var.set(10)  # Initial progress after starting
    update_progress(10)

    for pattern in file_patterns:
        for filepath in glob.glob(os.path.join(input_folder_lines, '**', pattern), recursive=True):
            try:
                log_to_gui(log_widget, f"Processing layer: {os.path.splitext(os.path.basename(filepath))[0]}")
                progress_var.set(10 + processed_files * progress_increment)  # Update progress before processing each file

                line_id_counter = process_line_file(filepath, line_objects, line_id_counter, log_widget)

                processed_files += 1
                progress_var.set(10 + processed_files * progress_increment)  # Update progress after processing each file
                update_progress(10 + processed_files * progress_increment)

            except Exception as e:
                log_to_gui(log_widget, f"Error processing file {filepath}: {e}")

    line_objects_gdf = gpd.GeoDataFrame(line_objects, geometry='geom' if line_objects else None)
    
    update_progress(90)

    log_to_gui(log_widget, f"Total lines added: {line_id_counter - 1}")

    return line_objects_gdf


# Thread function to run import without freezing GUI
def run_import_geocode(input_folder_geocode, gpkg_file, log_widget, progress_var):
    geocode_groups_gdf, geocode_objects_gdf = import_spatial_data_geocode(input_folder_geocode, log_widget, progress_var)

    log_to_gui(log_widget, f"Preparing import of geocode groups and objects.")

    if not geocode_groups_gdf.empty:
        export_to_geopackage(geocode_groups_gdf, gpkg_file, 'tbl_geocode_group', log_widget)

    if not geocode_objects_gdf.empty:
        export_to_geopackage(geocode_objects_gdf, gpkg_file, 'tbl_geocode_object', log_widget)

    log_to_gui(log_widget, "Import completed.")

    update_progress(100)

def initialize_empty_table(gpkg_file, dest_table, schema):
    # Create an empty GeoDataFrame with the specified schema
    empty_gdf = gpd.GeoDataFrame(columns=schema.keys(), geometry='geometry')
    for column, dtype in schema.items():
        empty_gdf[column] = pd.Series(dtype=dtype)
    
    # Save the empty GeoDataFrame to the destination table, replacing any existing content
    empty_gdf.to_file(gpkg_file, layer=dest_table, if_exists='replace')


def copy_original_lines_to_tbl_lines(gpkg_file, segment_width, segment_length):
    
    src_table   = 'tbl_lines_original'
    dest_table  = 'tbl_lines'

    # Ensure the destination table is empty
    schema = {
        'name_gis': 'str',
        'name_user': 'str',
        'segment_length': 'int',
        'segment_width': 'int',
        'description': 'str',
        'geometry': 'geometry'
    }

    initialize_empty_table(gpkg_file, dest_table, schema)
    
    # Load the source table as a GeoDataFrame
    src_gdf = gpd.read_file(gpkg_file, layer=src_table)
    
    # Transform the source data to match the destination table's schema
    dest_gdf                    = src_gdf.copy()
    dest_gdf['name_gis']        = dest_gdf.index.to_series().apply(lambda x: f"line_{x+1:03}")
    dest_gdf['name_user']       = dest_gdf['name_gis']
    dest_gdf['segment_length']  = segment_length
    dest_gdf['segment_width']   = segment_width
    dest_gdf['description']     = dest_gdf.apply(lambda row: f"{row['name_user']} + {row['attributes']}", axis=1)
    
    # Adjust to ensure only the necessary columns are included
    dest_gdf = dest_gdf[['name_gis', 'name_user', 'segment_length', 'segment_width', 'description', 'geometry']]
    
    # Save the transformed GeoDataFrame to the now-empty destination table
    dest_gdf.to_file(gpkg_file, layer=dest_table, if_exists='replace')


# Thread function to run import without freezing GUI
def run_import_lines(input_folder_lines, gpkg_file, log_widget, progress_var):
    line_objects_gdf = import_spatial_data_lines(input_folder_lines, log_widget, progress_var)

    log_to_gui(log_widget, f"Preparing import of lines.")

    if not line_objects_gdf.empty:
        export_to_geopackage(line_objects_gdf, gpkg_file, 'tbl_lines_original', log_widget)

    log_to_gui(log_widget, "Import of completed.")

    copy_original_lines_to_tbl_lines(gpkg_file, segment_width, segment_length)

    update_progress(100)


# Function imports spatial data for assets
def import_spatial_data_asset(input_folder_asset, log_widget, progress_var):
    asset_objects       = []
    asset_groups        = []
    group_id_counter    = 1
    object_id_counter   = 1
    file_patterns       = ['*.shp', '*.gpkg']
    total_files         = sum([len(glob.glob(os.path.join(input_folder_asset, '**', pattern), recursive=True)) for pattern in file_patterns])
    processed_files     = 0
    progress_increment  = 70 / total_files  # Distribute 70% of progress bar over file processing
    log_to_gui(log_widget, "Working with asset imports...")
    progress_var.set(10)  # Initial progress after starting
    update_progress(10)

    for pattern in file_patterns:
        for filepath in glob.glob(os.path.join(input_folder_asset, '**', pattern), recursive=True):
            try:
                filename = os.path.splitext(os.path.basename(filepath))[0]
                log_to_gui(log_widget, f"Processing file: {filename}")

                data = read_and_reproject(filepath)
                if not data.empty:
                    bbox_geom = get_bounding_box(data)
                    asset_groups.append({
                        'id': group_id_counter,
                        'name_original': filename,
                        'name_gis_assetgroup': f"layer_{group_id_counter:03d}",
                        'title_fromuser': filename,
                        'date_import': datetime.datetime.now(),
                        'geom': bbox_geom,
                        'total_asset_objects': int(0),
                        'importance': int(0),
                        'susceptibility': int(0),
                        'sensitivity': int(0)
                    })

                    object_id_counter = process_asset_layer(
                        data, asset_objects, object_id_counter, group_id_counter, filename, log_widget)
                    group_id_counter += 1

                processed_files += 1
                progress_var.set(10 + processed_files * progress_increment)
                update_progress(10 + processed_files * progress_increment)

            except Exception as e:
                log_to_gui(log_widget, f"Error processing file {filepath}: {e}")
    
    asset_groups_gdf = gpd.GeoDataFrame(asset_groups, geometry='geom')
    asset_objects_gdf = gpd.GeoDataFrame(asset_objects, geometry='geom')
    
    asset_groups_gdf.set_crs(epsg=4326, inplace=True)
    
    update_progress(90)

    # Calculate total bounding box for all asset objects
    if not asset_objects_gdf.empty:
        total_bbox = asset_objects_gdf.geometry.unary_union.bounds
        total_bbox_geom = box(*total_bbox)
        log_to_gui(log_widget, f"Total bounding box for all assets imported.")

    update_progress(95)

    asset_groups_gdf['id'] = asset_groups_gdf['id'].astype('int64')
    asset_objects_gdf['id'] = asset_objects_gdf['id'].astype('int64')

    update_progress(100)

    return asset_objects_gdf, asset_groups_gdf, total_bbox_geom


# Function exports data to geopackage and secures replacing relevant data.
def export_to_geopackage(gdf, gpkg_file, layer_name, log_widget):
    # Check if the GeoPackage file exists
    if not os.path.exists(gpkg_file):
        # Create a new GeoPackage file by writing the gdf with the specified layer
        gdf.to_file(gpkg_file, layer=layer_name, driver='GPKG')
        log_to_gui(log_widget, f"Created new GeoPackage file {gpkg_file} with layer {layer_name}.")
    else:
        # Overwrite the layer if it exists, otherwise create a new layer
        gdf.to_file(gpkg_file, layer=layer_name, driver='GPKG', if_exists='replace')
        log_to_gui(log_widget, f"Replaced data for layer {layer_name} in GeoPackage.")


# Function to update asset groups in geopackage
def update_asset_groups(asset_groups_df, gpkg_file, log_widget):
    engine = create_engine(f'sqlite:///{gpkg_file}')

    # Check if 'geom' column exists in the DataFrame
    if 'geom' in asset_groups_df.columns:
        # Define the data types for the ID columns
        id_col = asset_groups_df.columns[asset_groups_df.dtypes == 'int64']
        
        asset_groups_gdf = gpd.GeoDataFrame(asset_groups_df, geometry='geom')

        asset_groups_gdf[id_col] = asset_groups_gdf[id_col].astype(int)

        try:
            asset_groups_gdf.to_file(gpkg_file, layer='tbl_asset_group', driver="GPKG", if_exists='append')
            log_to_gui(log_widget, "Asset groups updated in GeoPackage.")
        except exc.SQLAlchemyError as e:
            log_to_gui(log_widget, f"Failed to update asset groups: {e}")
    else:
        log_to_gui(log_widget, "Error: 'geom' column not found in asset_groups_df.")


# Thread function to run import without freezing GUI
def run_import_asset(input_folder_asset, gpkg_file, log_widget, progress_var):
    log_to_gui(log_widget, "Starting asset import process...")
    
    asset_objects_gdf, asset_groups_df, total_bbox_geom = import_spatial_data_asset(input_folder_asset, log_widget, progress_var)

    log_to_gui(log_widget, "Importing:")
    export_to_geopackage(asset_objects_gdf, gpkg_file, 'tbl_asset_object', log_widget)
    
    update_asset_groups(asset_groups_df, gpkg_file, log_widget)

    update_asset_objects_with_name_gis(gpkg_file, log_widget)
    
    log_to_gui(log_widget, "Asset import completed.")
    
    progress_var.set(100)


# Name gis should be part of the asset objects table
def update_asset_objects_with_name_gis(db_file, log_widget):
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(db_file)

        # Load data into dataframes
        df_asset_group = pd.read_sql_query("SELECT id, name_gis_assetgroup FROM tbl_asset_group", conn)
        df_asset_object = pd.read_sql_query("SELECT * FROM tbl_asset_object", conn)

        # Check if 'fid' is part of the dataframe
        if 'fid' in df_asset_object.columns:
            df_asset_object.set_index('fid', inplace=True)

        # Join dataframes to get name_gis_assetgroup
        df_joined = df_asset_object.merge(df_asset_group, left_on='ref_asset_group', right_on='id', how='left')

        # Update tbl_asset_object dataframe with name_gis_assetgroup
        df_asset_object['ref_name_gis_assetgroup'] = df_joined['name_gis_assetgroup']

        # Write updated dataframe back to SQLite database
        df_asset_object.to_sql('tbl_asset_object', conn, if_exists='replace', index=True, index_label='fid')

        log_to_gui(log_widget, "tbl_asset_object updated with name_gis_assetgroup from tbl_asset_group.")
    except Exception as e:
        log_to_gui(log_widget, f"Error updating tbl_asset_object: {e}")
    finally:
        conn.close()


# Function to close the application
def close_application():
    root.destroy()


#####################################################################################
#  Main
#

# Load configuration settings
config_file             = 'config.ini'
config                  = read_config(config_file)
input_folder_asset      = config['DEFAULT']['input_folder_asset']
input_folder_geocode    = config['DEFAULT']['input_folder_geocode']
input_folder_lines      = config['DEFAULT']['input_folder_lines']
segment_width           = config['DEFAULT']['segment_width']
segment_length          = config['DEFAULT']['segment_length']
gpkg_file               = config['DEFAULT']['gpkg_file']
ttk_bootstrap_theme     = config['DEFAULT']['ttk_bootstrap_theme']

# Create the user interface
root = ttk.Window(themename=ttk_bootstrap_theme)  # Use ttkbootstrap Window
root.title("Import assets")

# Create a LabelFrame for the log output
log_frame = ttk.LabelFrame(root, text="Log Output", bootstyle="info") 
log_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Create a log widget inside the LabelFrame
log_widget = scrolledtext.ScrolledText(log_frame, height=10)
log_widget.pack(fill=tk.BOTH, expand=True)

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


# Information text field below the progress bar
info_label_text = ("Assets are all shapefiles or geopackage files with their layers "
                   "that are placed in the folder input/assets-folder. The features will "
                   "be placed in our database and used in the analysis. All assets will "
                   "be associated with importance and susceptibility values."
                   "Please refer to the log.txt to review the full log.")
info_label = tk.Label(root, text=info_label_text, wraplength=500, justify="left")
info_label.pack(padx=10, pady=10)

# Create a frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=5)

# Add button importing assets
import_btn = ttk.Button(button_frame, text="Import assets", bootstyle=PRIMARY, command=lambda: threading.Thread(
    target=run_import_asset, args=(input_folder_asset, gpkg_file, log_widget, progress_var), daemon=True).start())
import_btn.pack(side=tk.LEFT, padx=10)

# Add button forimporting geocodes
import_btn = ttk.Button(button_frame, text="Import geocodes", bootstyle=PRIMARY, command=lambda: threading.Thread(
    target=run_import_geocode, args=(input_folder_geocode, gpkg_file, log_widget, progress_var), daemon=True).start())
import_btn.pack(side=tk.LEFT, padx=10)

# Add button for importing lines data
import_btn = ttk.Button(button_frame, text="Import lines", bootstyle=PRIMARY, command=lambda: threading.Thread(
    target=run_import_lines, args=(input_folder_lines, gpkg_file, log_widget, progress_var), daemon=True).start())
import_btn.pack(side=tk.LEFT, padx=10)

close_btn = ttk.Button(button_frame, text="Close", command=close_application, bootstyle=WARNING)
close_btn.pack(side=tk.LEFT, padx=10)

root.mainloop()