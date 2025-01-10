# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# This is the most complex part of the mesa python too. It imports
# files from a set folder and places them in two tables.
#
# tbl_assets:   Information about assets like mangroves ant the likes. 
#               Data is stored as polygons, points and lines.
#               The assets are grouped into tbl_asset_groups. Objects are
#               kept in tbl_asset_objects.
#
# tbl_geocodes: Information about geocodes which could be grids, hexagons
#               as well as municipalities and other.
#               The geocodes are grouped into tbl_geocode_groups. Objects are
#               kept in tbl_geocode_objects.

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
import tkinter as tk
from tkinter import scrolledtext, ttk
from fiona import open as fiona_open    
import threading
import geopandas as gpd
from sqlalchemy import create_engine
import configparser
import subprocess
from collections import defaultdict
import datetime
import glob
import os
import uuid
import argparse
from osgeo import ogr
import pandas as pd
from sqlalchemy import exc
import sqlite3
from shapely.geometry import box
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from shapely.geometry import shape
import fiona

# Read the configuration file
def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config

# Update progress label
def update_progress(new_value):
    def task():
        progress_var.set(new_value)
        progress_label.config(text=f"{int(new_value)}%")
    root.after(0, task)

# Logging function to write to the GUI log
def log_to_gui(log_widget, message):
    timestamp           = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted_message   = f"{timestamp} - {message}"
    log_widget.insert(tk.END, formatted_message + "\n")
    log_widget.see(tk.END)
    log_destination_file = os.path.join(original_working_directory, "log.txt")
    with open(log_destination_file, "a") as log_file:
        log_file.write(formatted_message + "\n")

# Function to read and reproject spatial data
def read_and_reproject(filepath, layer=None, log_widget=None):
    try:
        data = gpd.read_file(filepath, layer=layer)
        original_crs = data.crs
        if data.crs is None:
            log_to_gui(log_widget, f"No CRS found for {filepath}. Setting CRS to EPSG:{workingprojection_epsg}.")
            data.set_crs(epsg=workingprojection_epsg, inplace=True)
        elif data.crs.to_epsg() != workingprojection_epsg:
            log_to_gui(log_widget, f"Reprojecting data from {original_crs} to EPSG:{workingprojection_epsg}.")
            data = data.to_crs(epsg=workingprojection_epsg)
        return data
    except Exception as e:
        log_to_gui(log_widget, f"Failed to read or reproject {filepath}: {e}")
        return gpd.GeoDataFrame()  # Return an empty GeoDataFrame on failure

# Process the lines in a layer. Add appropriate attributes.
def process_line_layer(data, line_objects, line_id_counter, layer_name, log_widget):
    if data.empty:
        log_to_gui(log_widget, f"No data found in layer {layer_name}")
        return line_id_counter

    # Temporarily reproject geometries to EPSG:3395 for length calculation
    temp_data = data.copy()
    temp_data = temp_data.to_crs(epsg=3395)

    for index, row in data.iterrows():
        length_m = int(temp_data.loc[index].geometry.length)  # Calculate length in meters and convert to integer
        attributes = '; '.join([f"{col}: {row[col]}" for col in data.columns if col != 'geometry'])

        line_objects.append({
            'name_gis': int(line_id_counter),
            'name_user': layer_name,
            'attributes': attributes,
            'length_m': length_m,  # Add the length attribute
            'geom': row.geometry  # Original geometry in workingprojection_epsg
        })

        line_id_counter += 1

    return line_id_counter

# Function to process a geocode layer and place it in context.
def process_geocode_layer(data, geocode_groups, geocode_objects, group_id_counter, object_id_counter, layer_name, log_widget):
    if data.empty:
        log_to_gui(log_widget, f"No data found in layer {layer_name}")
        return group_id_counter, object_id_counter

    feature_count = len(data)
    log_to_gui(log_widget, f"Imported {layer_name} with {feature_count} features.")

    # Calculate bounding box and add to geocode groups
    bounding_box = data.total_bounds
    bbox_geom = box(*bounding_box)
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
        code = row['qdgc'] if 'qdgc' in data.columns else str(object_id_counter)
        
        geocode_objects.append({
            'code': code,
            'ref_geocodegroup': group_id_counter,
            'name_gis_geocodegroup': name_gis_geocodegroup, 
            'geom': geom
        })
        object_id_counter += 1

    return group_id_counter + 1, object_id_counter

# Function to process each geocode file
def process_geocode_file(filepath, geocode_groups, geocode_objects, group_id_counter, object_id_counter, log_widget):
    if filepath.endswith('.gpkg'):
        try:
            # List layers in the geopackage
            layers = fiona.listlayers(filepath)
            log_to_gui(log_widget, f"Found layers in {filepath}: {layers}")
            
            # Process each layer
            for layer_name in layers:
                log_to_gui(log_widget, f"Processing geopackage layer: {layer_name}")
                
                # Explicitly specify the layer when reading the file
                data = gpd.read_file(filepath, layer=layer_name)
                
                # Reproject if necessary
                data = read_and_reproject(filepath, layer=layer_name, log_widget=log_widget)  
                
                # Process the geocode layer
                group_id_counter, object_id_counter = process_geocode_layer(
                    data, geocode_groups, geocode_objects, group_id_counter, object_id_counter, layer_name, log_widget
                )
        except Exception as e:
            log_to_gui(log_widget, f"Error processing geopackage {filepath}: {e}")
    
    elif filepath.endswith('.shp'):
        try:
            # Process shapefile directly
            data = read_and_reproject(filepath, log_widget=log_widget)
            layer_name = os.path.splitext(os.path.basename(filepath))[0]
            log_to_gui(log_widget, f"Processing shapefile layer: {layer_name}")
            
            group_id_counter, object_id_counter = process_geocode_layer(
                data, geocode_groups, geocode_objects, group_id_counter, object_id_counter, layer_name, log_widget
            )
        except Exception as e:
            log_to_gui(log_widget, f"Error processing shapefile {filepath}: {e}")
    else:
        log_to_gui(log_widget, f"Unsupported file format for {filepath}")
    
    return group_id_counter, object_id_counter




# Function to ensure unique 'code' attributes in geocode_objects
def ensure_unique_codes(geocode_objects, log_widget):
    code_counts = defaultdict(int)
    for obj in geocode_objects:
        code_counts[obj['code']] += 1

    for obj in geocode_objects:
        if code_counts[obj['code']] > 1:
            base_code = obj['code']
            new_code = f"{base_code}_{uuid.uuid4()}"
            obj['code'] = new_code
            code_counts[new_code] += 1
            log_to_gui(log_widget, f"Duplicate found and renamed to {new_code}")

# Function to process each file while reading through the filepath.
def process_line_file(filepath, line_objects, line_id_counter, log_widget):
    if filepath.endswith('.gpkg'):
        ds = ogr.Open(filepath)
        for i in range(ds.GetLayerCount()):
            layer       = ds.GetLayerByIndex(i)
            layer_name  = layer.GetName()
            data        = read_and_reproject(filepath, layer=layer_name, log_widget=log_widget)
            log_to_gui(log_widget, f"Importing geopackage layer {layer_name}.")
            line_id_counter = process_line_layer(
                data, line_objects, line_id_counter, layer_name, log_widget)
        ds = None
    elif filepath.endswith('.shp'):
        data = read_and_reproject(filepath, log_widget=log_widget)
        layer_name = os.path.splitext(os.path.basename(filepath))[0]
        log_to_gui(log_widget, f"Importing shapefile layer: {layer_name}")
        line_id_counter = process_line_layer(
            data, line_objects, line_id_counter, layer_name, log_widget)
    else:
        log_to_gui(log_widget, f"Unsupported file format for {filepath}")
    return line_id_counter


def get_file_metadata(file_path):
    """Extracts number of features from a geospatial file."""
    try:
        data = gpd.read_file(file_path)
        num_features = len(data)
        return (file_path, num_features)
    except Exception as e:
        log_to_gui(log_widget, f"Error reading {file_path}: {e}")
        return (file_path, 0)

#Sorting files by filepaths (filenames). If there is only one file no counting necessary.
def sort_files_by_feature_count(file_paths):
    """Sorts file paths by the number of features, descending."""
    if len(file_paths) == 1:
        return file_paths
    
    files_with_metadata = [get_file_metadata(fp) for fp in file_paths]
    sorted_files = sorted(files_with_metadata, key=lambda x: x[1], reverse=True)
    return [fp[0] for fp in sorted_files]


def import_spatial_data_geocode(input_folder_geocode, log_widget, progress_var):
    geocode_groups = []
    geocode_objects = []
    group_id_counter = 1
    object_id_counter = 1
    file_patterns = ['*.shp', '*.gpkg']
    
    log_to_gui(log_widget, "Working with imports...")
    update_progress(5)  # Initial progress after starting

    # Gather all file paths
    all_file_paths = [fp for pattern in file_patterns for fp in glob.glob(os.path.join(input_folder_geocode, '**', pattern), recursive=True)]
    
    log_to_gui(log_widget, "Sorting the geocode layers by increasing size")
    update_progress(10)  # Initial progress after starting

    # Sort files by the number of features
    sorted_file_paths = sort_files_by_feature_count(all_file_paths)
    total_files = len(sorted_file_paths)
    processed_files = 0

    if total_files == 0:
        progress_increment = 70
    else:
        progress_increment = 70 / total_files  # Distribute 70% of progress bar over file processing

    for filepath in sorted_file_paths:
        try:
            layer_name = os.path.splitext(os.path.basename(filepath))[0]
            log_to_gui(log_widget, f"Processing file: {filepath}")
            progress_var.set(10 + processed_files * progress_increment)  # Update progress before processing each file

            group_id_counter, object_id_counter = process_geocode_file(
                filepath, geocode_groups, geocode_objects, group_id_counter, object_id_counter, log_widget)

            processed_files += 1
            progress_var.set(10 + processed_files * progress_increment)  # Update progress after processing each file
            update_progress(10 + processed_files * progress_increment)

        except Exception as e:
            log_to_gui(log_widget, f"Error processing file {filepath}: {e}")

    ensure_unique_codes(geocode_objects, log_widget)  # Ensure unique codes after all files are processed

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
    update_progress(10)  # Initial progress after starting

    for pattern in file_patterns:
        for filepath in glob.glob(os.path.join(input_folder_lines, '**', pattern), recursive=True):
            try:
                log_to_gui(log_widget, f"Processing layer: {os.path.splitext(os.path.basename(filepath))[0]}")
                progress_var.set(10 + processed_files * progress_increment)  # Update progress before processing each file

                line_id_counter = process_line_file(filepath, line_objects, line_id_counter, log_widget)

                processed_files += 1
                update_progress(10 + processed_files * progress_increment)

            except Exception as e:
                log_to_gui(log_widget, f"Error processing file {filepath}: {e}")

    line_objects_gdf = gpd.GeoDataFrame(line_objects, geometry='geom' if line_objects else None)
    
    update_progress(90)

    log_to_gui(log_widget, f"Total lines added: {line_id_counter - 1}")

    return line_objects_gdf


# Thread function to run import of geocodes
def run_import_geocode(input_folder_geocode, gpkg_file, log_widget, progress_var):
    log_to_gui(log_widget, "Preparing to import geocode groups and objects.")

    # Delete old geocode group and object tables
    log_to_gui(log_widget, "Cleaning up old geocodes. Deleting geocode groups and objects if they exist.")
    delete_table_from_geopackage(gpkg_file, 'tbl_geocode_group', log_widget)
    delete_table_from_geopackage(gpkg_file, 'tbl_geocode_object', log_widget)

    # Import new geocode data
    geocode_groups_gdf, geocode_objects_gdf = import_spatial_data_geocode(input_folder_geocode, log_widget, progress_var)

    # Check if the GeoDataFrame is not empty before exporting
    if not geocode_groups_gdf.empty:
        log_to_gui(log_widget, "Exporting geocode groups to geopackage")
        export_to_geopackage(geocode_groups_gdf, gpkg_file, 'tbl_geocode_group', log_widget)
    
    if not geocode_objects_gdf.empty:
        log_to_gui(log_widget, "Exporting geocode objects to geopackage")
        export_to_geopackage(geocode_objects_gdf, gpkg_file, 'tbl_geocode_object', log_widget)

    save_geocodes_to_geoparquet(geocode_groups_gdf, geocode_objects_gdf, original_working_directory, log_widget)

    log_to_gui(log_widget, "COMPLETED: Geocode import done.")
    progress_var.set(100)
    update_progress(100)
    increment_stat_value(config_file, 'mesa_stat_import_geocodes', increment_value=1)



# Emtpy the destination table. Not sure if this one works 100%, but we will
# stick with it for now.
def initialize_empty_table(gpkg_file, dest_table, schema):
    # Create an empty GeoDataFrame with the specified schema
    empty_gdf = gpd.GeoDataFrame(columns=schema.keys(), geometry='geometry')
    for column, dtype in schema.items():
        empty_gdf[column] = pd.Series(dtype=dtype)
    
    # Save the empty GeoDataFrame to the destination table, replacing any existing content
    empty_gdf.to_file(gpkg_file, layer=dest_table)


# Original lines are copied to tbl_lines where the user can make
# edits to segment width and length. Furthermore name (title) and description
# can also be reviewed by the user.
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
        'length_m': 'int',  # Add the length_m field to the schema
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
    
    # Calculate the length in meters for each line
    dest_gdf = dest_gdf.to_crs(epsg=3395)
    dest_gdf['length_m'] = dest_gdf.geometry.length.astype(int)
    dest_gdf = dest_gdf.to_crs(epsg=workingprojection_epsg)  # Reproject back to the original CRS
    
    # Adjust to ensure only the necessary columns are included
    dest_gdf = dest_gdf[['name_gis', 'name_user', 'segment_length', 'segment_width', 'description', 'length_m', 'geometry']]
    
    # Save the transformed GeoDataFrame to the now-empty destination table
    dest_gdf.to_file(gpkg_file, layer=dest_table)


# Thread function to run import lines
def run_import_lines(input_folder_lines, gpkg_file, log_widget, progress_var):
    log_to_gui(log_widget, "Preparing import of lines.")
    
    # Delete old lines, if they exist
    log_to_gui(log_widget, "First deleting old lines, if they exist.")
    delete_table_from_geopackage(gpkg_file, 'tbl_lines_original', log_widget)
    delete_table_from_geopackage(gpkg_file, 'tbl_lines', log_widget)
    
    log_to_gui(log_widget, "Looking through input folder.")

    # Import new line data
    line_objects_gdf = import_spatial_data_lines(input_folder_lines, log_widget, progress_var)

    # Check if the GeoDataFrame is not empty before proceeding
    if not line_objects_gdf.empty:
        # Check if CRS is set, if not, set it to the working projection
        if line_objects_gdf.crs is None:
            log_to_gui(log_widget, f"No CRS found, setting CRS to EPSG:{workingprojection_epsg}")
            line_objects_gdf.set_crs(epsg=workingprojection_epsg, inplace=True)

        # Reproject to the working projection
        log_to_gui(log_widget, f"Reprojecting lines to EPSG:{workingprojection_epsg}")
        line_objects_gdf = line_objects_gdf.to_crs(epsg=workingprojection_epsg)

        log_to_gui(log_widget, "Exporting line objects to geopackage.")
        export_line_to_geopackage(line_objects_gdf, gpkg_file, 'tbl_lines_original', log_widget)
    
    # Copy original lines to tbl_lines for further user editing
    log_to_gui(log_widget, "Copying original lines to tbl_lines.")
    copy_original_lines_to_tbl_lines(gpkg_file, segment_width, segment_length)

    log_to_gui(log_widget, "COMPLETED: Line imports done.")
    
    # Update progress bar and stats
    progress_var.set(100)
    update_progress(100)
    increment_stat_value(config_file, 'mesa_stat_import_lines', increment_value=1)



def append_to_asset_groups(layer_name, data, asset_groups, group_id_counter):
    # Assuming data.total_bounds gives you [minx, miny, maxx, maxy]
    bbox = data.total_bounds
    bbox_polygon = box(*bbox)  # Create a Polygon geometry from the bounding box
    
    asset_groups.append({
        'id': group_id_counter,
        'name_original': layer_name, 
        'name_gis_assetgroup': f"layer_{group_id_counter:03d}",
        'title_fromuser': layer_name,
        'date_import': datetime.datetime.now(),
        'geom': bbox_polygon,
        'total_asset_objects': int(0),
        'importance': int(0),
        'susceptibility': int(0),
        'sensitivity': int(0),
        'sensitivity_code': int(0),
        'sensitivity_description': int(0)
    })
    return group_id_counter + 1


def append_to_asset_objects(data, asset_objects, object_id_counter, group_id):
    # Example logic to append to asset_objects based on processed data
    for _, row in data.iterrows():
        asset_objects.append({
            'id': object_id_counter,
            'ref_asset_group': group_id,
            'geom': row.geometry,  # Directly use geometry
            # Add other necessary attributes here
        })
        object_id_counter += 1
    return object_id_counter


def process_geopackage_layer(filepath, layer_name, asset_objects, asset_groups, object_id_counter, group_id_counter, log_widget):
    data = read_and_reproject(filepath, layer=layer_name, log_widget=log_widget)
    if not data.empty:
        group_id_counter = append_to_asset_groups(layer_name, data, asset_groups, group_id_counter)
        object_id_counter = append_to_asset_objects(data, asset_objects, object_id_counter, group_id_counter - 1)
    return group_id_counter, object_id_counter


def import_spatial_data_asset(input_folder_asset, log_widget, progress_var):
    asset_objects       = []
    asset_groups        = []
    group_id_counter    = 1
    object_id_counter   = 1
    file_patterns       = ['*.shp', '*.gpkg']
    processed_files     = 0

    progress_var.set(10)
    update_progress(10)

    total_files = sum([len(glob.glob(os.path.join(input_folder_asset, '**', pattern), recursive=True)) for pattern in file_patterns])
    if total_files > 0:
        progress_increment = 70 / total_files  # Progress increment for each file
    else:
        progress_increment = 0  # Avoid division by zero if no files found

    log_to_gui(log_widget, "Working with asset imports...")
    progress_var.set(15)  # Initial progress after starting
    update_progress(15)

    for pattern in file_patterns:
        for filepath in glob.glob(os.path.join(input_folder_asset, '**', pattern), recursive=True):
            filename = os.path.splitext(os.path.basename(filepath))[0]
            if filepath.endswith('.gpkg'):

                # Handle geopackage: iterate over each layer
                for layer_name in fiona.listlayers(filepath):
                    
                    log_to_gui(log_widget, f"Importing geopackage layer: {layer_name}")
                    
                    data = read_and_reproject(filepath, layer=layer_name, log_widget=log_widget)
                    if not data.empty:
                        bbox_polygon = box(*data.total_bounds)
                        asset_groups.append({
                            'id': group_id_counter,
                            'name_original': layer_name, 
                            'name_gis_assetgroup': f"layer_{group_id_counter:03d}",
                            'title_fromuser': filename,
                            'date_import': datetime.datetime.now(),
                            'geom': bbox_polygon,
                            'total_asset_objects': int(0),
                            'importance': int(0),
                            'susceptibility': int(0),
                            'sensitivity': int(0),
                            'sensitivity_code': '',  # Default or placeholder value
                            'sensitivity_description': ''  # Default or placeholder value
                        })
                        for _, row in data.iterrows():

                            attributes = '; '.join([f"{col}: {row[col]}" for col in data.columns if col != 'geometry'])

                            asset_objects.append({
                                'id': object_id_counter,
                                'asset_group_name': layer_name,
                                'attributes': attributes,
                                'process': True,
                                'ref_asset_group': group_id_counter,
                                'geom': row.geometry  # Original geometry in workingprojection_epsg
                            })
                            
                            object_id_counter += 1
                        group_id_counter += 1

            elif filepath.endswith('.shp'):
                
                layer_name = filename

                log_to_gui(log_widget, f"Importing shapefile layer: {layer_name}")
                
                data = read_and_reproject(filepath, log_widget=log_widget)
                if not data.empty:
                    bbox_polygon = box(*data.total_bounds)
                    asset_groups.append({
                        'id': group_id_counter,
                        'name_original': filename, 
                        'name_gis_assetgroup': f"layer_{group_id_counter:03d}",
                        'title_fromuser': filename,
                        'date_import': datetime.datetime.now(),
                        'geom': bbox_polygon,
                        'total_asset_objects': int(0),
                        'importance': int(0),
                        'susceptibility': int(0),
                        'sensitivity': int(0),
                        'sensitivity_code': '',  # Default or placeholder value
                        'sensitivity_description': '',  # Default or placeholder value
                    })

                    for index, row in data.iterrows():
                        
                        attributes = '; '.join([f"{col}: {row[col]}" for col in data.columns if col != 'geometry'])

                        asset_objects.append({
                            'id': object_id_counter,
                            'asset_group_name': layer_name,
                            'attributes': attributes,
                            'process': True,
                            'ref_asset_group': group_id_counter,
                            'geom': row.geometry  # Original geometry in workingprojection_epsg
                        })

                        object_id_counter += 1
                    group_id_counter += 1
            else:
                log_to_gui(log_widget, f"Unsupported file format for {filepath}")

            processed_files += 1

            progress_var.set(10 + processed_files * progress_increment)

            update_progress(10 + processed_files * progress_increment)

    asset_groups_gdf = gpd.GeoDataFrame(asset_groups, geometry='geom')
    asset_objects_gdf = gpd.GeoDataFrame(asset_objects, geometry='geom')
    
    if not asset_objects_gdf.empty:
        total_bbox_geom = box(*asset_objects_gdf.total_bounds)
    else:
        total_bbox_geom = None

    progress_var.set(90)
    update_progress(90)

    return asset_objects_gdf, asset_groups_gdf, total_bbox_geom

# Attempts to delete a layer from a geopacakge file. Necessary to avoid invalid
# spatial indexes when new data is added to an existing layer.
def delete_layer(gpkg_file, layer_name, log_widget):
    try:
        ds = ogr.Open(gpkg_file, update=True)  # Open in update mode
        if ds is not None:
            layer = ds.GetLayerByName(layer_name)
            if layer is not None:
                ds.DeleteLayer(layer_name)
                log_to_gui(log_widget, f"Layer {layer_name} deleted from {gpkg_file}.")
            else:
                log_to_gui(log_widget, f"Layer {layer_name} does not exist in {gpkg_file}.")
            ds = None  # Ensure the data source is closed
        else:
            log_to_gui(log_widget, f"Failed to open {gpkg_file}.")
    except Exception as e:
        log_to_gui(log_widget, f"An error occurred while attempting to delete layer {layer_name} from {gpkg_file}: {e}")


# Function exports data to geopackage and secures replacing relevant data.  
def export_to_geopackage(gdf_or_list, gpkg_file, layer_name, log_widget):
    # Check if the input is a list and convert it to a GeoDataFrame
    if isinstance(gdf_or_list, list):
        if gdf_or_list and 'geom' in gdf_or_list[0]:
            gdf = gpd.GeoDataFrame(gdf_or_list)
            gdf['geometry'] = gdf['geom'].apply(lambda x: shape(x) if not isinstance(x, gpd.geoseries.GeoSeries) else x)
            gdf.drop('geom', axis=1, inplace=True)
        else:
            log_to_gui(log_widget, "The list is empty or does not contain 'geom' key.")
            return
    elif isinstance(gdf_or_list, gpd.GeoDataFrame):
        gdf = gdf_or_list
    else:
        log_to_gui(log_widget, "Unsupported data type for exporting to GeoPackage.")
        return
    
    # Set the CRS for the GeoDataFrame if it's not already set
    if gdf.crs is None:
        gdf.set_crs(epsg=workingprojection_epsg, inplace=True)
    
    # Delete the layer first if it exists to avoid overwrite issues
    delete_layer(gpkg_file, layer_name, log_widget)
    
    # Attempt to save the GeoDataFrame to the specified layer in the GeoPackage
    try:
        gdf.to_file(gpkg_file, layer=layer_name, driver='GPKG')
        log_to_gui(log_widget, f"Data successfully exported to {layer_name} in {gpkg_file}.")
        progress_var.set(95)
        update_progress(95)
    except Exception as e:
        log_to_gui(log_widget, f"Failed to export data to GeoPackage: {str(e)}")


# Function exports data to geopackage and secures replacing relevant data.  
def export_line_to_geopackage(gdf, gpkg_file, layer_name, log_widget):
    # Check if the GeoPackage file exists
    if not os.path.exists(gpkg_file):
        # Create a new GeoPackage file by writing the gdf with the specified layer
        gdf.to_file(gpkg_file, layer=layer_name, driver='GPKG')
        log_to_gui(log_widget, f"Created new GeoPackage file {gpkg_file} with layer {layer_name}.")
    else:

        if not gdf.empty:

            delete_layer(gpkg_file, 'tbl_lines_original', log_widget)
            delete_layer(gpkg_file, 'tbl_lines', log_widget)

            gdf.to_file(gpkg_file, layer=layer_name, driver='GPKG')
            log_to_gui(log_widget, f"Data for layer {layer_name} saved in GeoPackage.")
        else:
            log_to_gui(log_widget, f"Warning: Attempted to save an empty GeoDataFrame for layer {layer_name}.")

def enable_spatialite(db_file, log_widget):
    try:
        conn = sqlite3.connect(db_file)
        conn.enable_load_extension(True)  # Allows loading of SQLite extensions
        conn.execute('SELECT load_extension("mod_spatialite");')  # Adjust the path to mod_spatialite if necessary

        log_to_gui(log_widget, "SpatiaLite extension loaded successfully.")
        return conn  # Return the connection if you want to use it further

    except sqlite3.Error as e:
        log_to_gui(log_widget, f"Failed to load SpatiaLite extension: {e}")

# Name gis should be part of the asset objects table
def update_asset_objects_with_name_gis(db_file, log_widget):
    try:
        # Load spatial data directly into GeoDataFrames
        gdf_assets = gpd.read_file(db_file, layer='tbl_asset_object')
        gdf_asset_groups = gpd.read_file(db_file, layer='tbl_asset_group')
        
        # Perform an attribute-based join
        gdf_assets = gdf_assets.merge(
            gdf_asset_groups[['id', 'name_gis_assetgroup']], 
            left_on='ref_asset_group', 
            right_on='id', 
            how="left"
        )

        # After merging, 'name_gis_assetgroup' contains the information we want to add to 'tbl_asset_object' as 'ref_name_gis_assetgroup'
        gdf_assets['ref_name_gis_assetgroup'] = gdf_assets['name_gis_assetgroup']
        
        # Drop the 'id' column from the merge operation if it exists
        if 'id' in gdf_assets.columns:
            gdf_assets.drop(columns=['id'], inplace=True)
            
        # Drop the 'id_y' column from the merge operation if it exists
        if 'id_y' in gdf_assets.columns:
            gdf_assets.drop(columns=['id_y'], inplace=True)

        # Drop the 'id_x' column from the merge operation if it exists
        if 'id_x' in gdf_assets.columns:
            gdf_assets.drop(columns=['id_x'], inplace=True)
            
        # Drop the 'name_gis_assetgroup' column from the merge operation if it exists
        if 'name_gis_assetgroup' in gdf_assets.columns:
            gdf_assets.drop(columns=['name_gis_assetgroup'], inplace=True)

        # Save the updated GeoDataFrame back to a GeoPackage
        gdf_assets.to_file(db_file, layer='tbl_asset_object', driver="GPKG")
        
        log_to_gui(log_widget, "Successfully updated 'ref_name_gis_assetgroup' in tbl_asset_object.")

    except Exception as e:
        log_to_gui(log_widget, f"Failed to update 'ref_name_gis_assetgroup': {e}")

# Thread function to run import without freezing GUI
def run_import_asset(input_folder_asset, gpkg_file, log_widget, progress_var):
    log_to_gui(log_widget, "Starting asset import process...")

    # Delete asset group and object tables
    log_to_gui(log_widget, "Cleaning up old assets. Deleting asset groups and objects if they exist.")
    delete_table_from_geopackage(gpkg_file, 'tbl_asset_group', log_widget)
    delete_table_from_geopackage(gpkg_file, 'tbl_asset_object', log_widget)

    # Import new data
    asset_objects_gdf, asset_groups_gdf, total_bbox_geom = import_spatial_data_asset(input_folder_asset, log_widget, progress_var)

    # Check if the GeoDataFrame is not empty before exporting
    if not asset_objects_gdf.empty:
        log_to_gui(log_widget, "Exporting asset objects to geopackage")
        export_to_geopackage(asset_objects_gdf, gpkg_file, 'tbl_asset_object', log_widget)
    else:
        log_to_gui(log_widget, "No asset objects to export.")
    
    if not asset_groups_gdf.empty:
        log_to_gui(log_widget, "Exporting asset groups to geopackage")
        export_to_geopackage(asset_groups_gdf, gpkg_file, 'tbl_asset_group', log_widget)
    else:
        log_to_gui(log_widget, "No asset groups to export.")
    
    save_assets_to_geoparquet(asset_objects_gdf, asset_groups_gdf, original_working_directory, log_widget)

    update_asset_objects_with_name_gis(gpkg_file, log_widget)

    log_to_gui(log_widget, "COMPLETED: Asset import done.")
    progress_var.set(100)
    update_progress(100)
    increment_stat_value(config_file, 'mesa_stat_import_assets', increment_value=1)


# Function to delete a table from a GeoPackage file
def delete_table_from_geopackage(gpkg_file, table_name, log_widget=None):
    try:
        # Open the GeoPackage in update mode
        ds = ogr.Open(gpkg_file, update=True)
        if ds is not None:
            # Check if the specified table (layer) exists
            layer = ds.GetLayerByName(table_name)
            if layer is not None:
                ds.DeleteLayer(table_name)
                message = f"Table {table_name} deleted from {gpkg_file}."
            else:
                message = f"Table {table_name} does not exist in {gpkg_file}."
        else:
            message = f"Failed to open {gpkg_file}."
        # Close the data source
        ds = None
        # Log the result
        if log_widget:
            log_to_gui(log_widget, message)
        else:
            print(message)
    except Exception as e:
        error_message = f"Error deleting table {table_name} from {gpkg_file}: {e}"
        if log_widget:
            log_to_gui(log_widget, error_message)
        else:
            print(error_message)


def increment_stat_value(config_file, stat_name, increment_value):
    # Check if the config file exists
    if not os.path.isfile(config_file):
        print(f"Configuration file {config_file} not found.")
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
                    print(f"Error: Current value of {stat_name} is not an integer.")
                    return

    # Write the updated content back to the file if the variable was found and updated
    if updated:
        with open(config_file, 'w') as file:
            file.writelines(lines)

def run_subprocess(command, fallback_command):

    """ Utility function to run a subprocess with a fallback option. """
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            subprocess.run(fallback_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            log_to_gui(f"Failed to execute command: {command}")

# Function to close the application
def close_application():
    root.destroy()

def save_to_geoparquet(gdf, file_path, log_widget):
    try:
        gdf.to_parquet(file_path, index=False)
        log_to_gui(log_widget, f"Saved to geoparquet: {file_path}")
    except Exception as e:
        log_to_gui(log_widget, f"Error saving to geoparquet: {e}")

def save_assets_to_geoparquet(asset_objects_gdf, asset_groups_gdf, original_working_directory, log_widget):
    try:
        save_to_geoparquet(asset_objects_gdf, os.path.join(original_working_directory, "output/geoparquet/assets_objects.parquet"), log_widget)
        save_to_geoparquet(asset_groups_gdf, os.path.join(original_working_directory, "output/geoparquet/assets_groups.parquet"), log_widget)
    except Exception as e:
        log_to_gui(log_widget, f"Error saving assets to geoparquet: {e}")

def save_geocodes_to_geoparquet(geocode_groups_gdf, geocode_objects_gdf, original_working_directory, log_widget):
    try:
        save_to_geoparquet(geocode_groups_gdf, os.path.join(original_working_directory, "output/geoparquet/geocodes_groups.parquet"), log_widget)
        save_to_geoparquet(geocode_objects_gdf, os.path.join(original_working_directory, "output/geoparquet/geocodes_objects.parquet"), log_widget)
    except Exception as e:
        log_to_gui(log_widget, f"Error saving geocodes to geoparquet: {e}")

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

# Load configuration settings
config_file                 = os.path.join(original_working_directory, "system/config.ini")
gpkg_file                   = os.path.join(original_working_directory, "output/mesa.gpkg")

# Load configuration settings
config                      = read_config(config_file)

input_folder_asset          = os.path.join(original_working_directory, config['DEFAULT']['input_folder_asset'])
input_folder_geocode        = os.path.join(original_working_directory, config['DEFAULT']['input_folder_geocode'])
input_folder_lines          = os.path.join(original_working_directory, config['DEFAULT']['input_folder_lines'])

segment_width               = config['DEFAULT']['segment_width']
segment_length              = config['DEFAULT']['segment_length']
ttk_bootstrap_theme         = config['DEFAULT']['ttk_bootstrap_theme']
workingprojection_epsg      = config['DEFAULT']['workingprojection_epsg']

# Create the user interface
root = ttk.Window(themename=ttk_bootstrap_theme)  # Use ttkbootstrap Window
root.title("Import assets")
root.iconbitmap(os.path.join(original_working_directory,"system_resources/mesa.ico"))

# Create a LabelFrame for the log output
log_frame = ttk.LabelFrame(root, text="Log output", bootstyle="info") 
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
info_label_text = ("On this page you can import assets, geocodes (grids) and lines. The features will "
                   "be placed in our database and used in the analysis. Assets are geopackage files with "
                   "layers or shapefiles placed in the input/assets-folder. All assets will later be associated "
                   "with importance and susceptibility values. Geocodes are usually grid cells. Lines are being used "
                   "to create segments which will have a similar calulation to the geocode areas. ")
info_label = tk.Label(root, text=info_label_text, wraplength=600, justify="left")
info_label.pack(padx=10, pady=10)

# Create a frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=5)

# Add button importing assets
import_asset_btn = ttk.Button(button_frame, text="Import assets", bootstyle=PRIMARY, command=lambda: threading.Thread(
    target=run_import_asset, args=(input_folder_asset, gpkg_file, log_widget, progress_var), daemon=True).start())
import_asset_btn.grid(row=0, column=0, padx=10, pady=5, sticky='ew')

# Add button for importing geocodes
import_geocode_btn = ttk.Button(button_frame, text="Import geocodes", bootstyle=PRIMARY, command=lambda: threading.Thread(
    target=run_import_geocode, args=(input_folder_geocode, gpkg_file, log_widget, progress_var), daemon=True).start())
import_geocode_btn.grid(row=0, column=1, padx=10, pady=5, sticky='ew')

# Add button for importing lines data
import_lines_btn = ttk.Button(button_frame, text="Import lines", bootstyle=PRIMARY, command=lambda: threading.Thread(
    target=run_import_lines, args=(input_folder_lines, gpkg_file, log_widget, progress_var), daemon=True).start())
import_lines_btn.grid(row=0, column=2, padx=10, pady=5, sticky='ew')

# Exit button for this sub-program
exit_btn = ttk.Button(button_frame, text="Exit", command=close_application, bootstyle=WARNING)
exit_btn.grid(row=0, column=3, padx=10, sticky='ew')

root.mainloop()
