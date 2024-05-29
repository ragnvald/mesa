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

import tkinter as tk
from tkinter import scrolledtext, ttk
from fiona import open as fiona_open    
import threading
import geopandas as gpd
from sqlalchemy import create_engine
import configparser
import subprocess
import datetime
import glob
import os
import sys
from osgeo import ogr
import pandas as pd
from sqlalchemy import exc
import sqlite3
from shapely.geometry import box
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from shapely.geometry import shape
import fiona
import threading


# # # # # # # # # # # # # # 
# Shared/general functions

# Read the configuration file
def read_config(file_name):
    config = configparser.ConfigParser()
    if not config.read(file_name):
        raise FileNotFoundError(f"Unable to read the config file at {file_name}")
    return config


# # # # # # # # # # # # # # 
# Core functions

# Get bounding box
def get_bounding_box(data):
    bbox = data.total_bounds
    bbox_geom = box(*bbox)
    return bbox_geom


def get_base_path():
    if hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


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
    with open("../log.txt", "a") as log_file:
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
        attributes  = '; '.join([f"{col}: {row[col]}" for col in data.columns if col != 'geometry'])
        area_m2     = area_m2_series.iloc[index] if row.geometry.geom_type == 'Polygon' else 0

        asset_objects.append({
            'id': int(object_id_counter),
            'ref_asset_group': int(group_id),
            'asset_group_name': layer_name,
            'attributes': attributes,
            'process': True,
            'area_m2': int(area_m2),
            'geom': row.geometry  # Original geometry in workingprojection_epsg
        })
        object_id_counter += 1

    return object_id_counter


# Process the lines in a layer. Add appropriate attributes.
def process_line_layer(data, line_objects, line_id_counter, layer_name, log_widget):
    if data.empty:
        log_to_gui(log_widget, f"No data found in layer {layer_name}")
        return line_id_counter

    for index, row in data.iterrows():
        attributes = '; '.join([f"{col}: {row[col]}" for col in data.columns if col != 'geometry'])

        line_objects.append({
            'name_gis': int(line_id_counter),
            'name_user': layer_name,
            'attributes': attributes,
            'geom': row.geometry  # Original geometry in workingprojection_epsg
        })

        line_id_counter += 1

    return line_id_counter


# Function to process a geocode layer and place it in context.
# Geocode objects are all objects form all asset files/geopackages. The attributes are all
# placed within one attribute (attributes) in the tbl_geocode_object table. At the time
# of writing this I am not sure if the attribute name is kept here. If not it should be
# placed separately in tbl_geocode_group.
def process_geocode_layer(data, geocode_groups, geocode_objects, group_id_counter, object_id_counter, layer_name, log_widget):
    if data.empty:
        log_to_gui(log_widget, f"No data found in layer {layer_name}")
        return group_id_counter, object_id_counter

    feature_count = len(data)
    log_to_gui(log_widget, f"Imported {layer_name} with {feature_count}.")

    # Calculate bounding box and add to geocode groups
    bounding_box            = data.total_bounds
    bbox_geom               = box(*bounding_box)
    name_gis_geocodegroup   = f"geocode_{group_id_counter:03d}"

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


# Function to process each geocode file
def process_geocode_file(filepath, geocode_groups, geocode_objects, group_id_counter, object_id_counter, log_widget):
    if filepath.endswith('.gpkg'):
        ds = ogr.Open(filepath)
        for i in range(ds.GetLayerCount()):
            layer       = ds.GetLayerByIndex(i)
            layer_name  = layer.GetName()
            data        = read_and_reproject(filepath, layer=layer_name, log_widget=log_widget)
            log_to_gui(log_widget, f"Importing geopackage layer: {layer_name}")
            group_id_counter, object_id_counter = process_geocode_layer(
                data, geocode_groups, geocode_objects, group_id_counter, object_id_counter, layer_name, log_widget)
        ds = None
    elif filepath.endswith('.shp'):
        data            = read_and_reproject(filepath, log_widget=log_widget)
        layer_name      = os.path.splitext(os.path.basename(filepath))[0]
        log_to_gui(log_widget, f"Importing shapefile layyer: {layer_name}")
        group_id_counter, object_id_counter = process_geocode_layer(
            data, geocode_groups, geocode_objects, group_id_counter, object_id_counter, layer_name, log_widget)
    else:
        log_to_gui(log_widget, f"Unsupported file format for {filepath}")
    return group_id_counter, object_id_counter


# Function to process each file while readint through the filepath.
# Supposedly being done recursively.
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


def sort_files_by_feature_count(file_paths):
    """Sorts file paths by the number of features, descending."""
    files_with_metadata = [get_file_metadata(fp) for fp in file_paths]
    sorted_files = sorted(files_with_metadata, key=lambda x: x[1], reverse=False)
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
            log_to_gui(log_widget, f"Processing the layer {layer_name}")
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

    log_to_gui(log_widget, f"Preparing import of geocode groups and objects.")

    log_to_gui(log_widget, "Cleaning up. Deleting old geocodes, if they exist.")

    delete_table_from_geopackage(gpkg_file, 'tbl_geocode_group', log_widget=None)
    delete_table_from_geopackage(gpkg_file, 'tbl_geocode_object', log_widget=None)


    log_to_gui(log_widget, f"Looking through the input folder.")
    geocode_groups_gdf, geocode_objects_gdf = import_spatial_data_geocode(input_folder_geocode, log_widget, progress_var)


    if not geocode_groups_gdf.empty:
        export_to_geopackage(geocode_groups_gdf, gpkg_file, 'tbl_geocode_group', log_widget)

    if not geocode_objects_gdf.empty:
        export_to_geopackage(geocode_objects_gdf, gpkg_file, 'tbl_geocode_object', log_widget)

    log_to_gui(log_widget, "COMPLETED: Geocoding imports done.")
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
    empty_gdf.to_file(gpkg_file, layer=dest_table, if_exists='replace')


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


# Thread function to run import lines
def run_import_lines(input_folder_lines, gpkg_file, log_widget, progress_var):

    log_to_gui(log_widget, f"Preparing import of lines.")
    
    log_to_gui(log_widget, "First deleting old lines, if they exist.")

    delete_table_from_geopackage(gpkg_file, 'tbl_lines', log_widget=None)
    
    log_to_gui(log_widget, "Looking trough input folder.")
    
    line_objects_gdf = import_spatial_data_lines(input_folder_lines, log_widget, progress_var)

    if not line_objects_gdf.empty:
        export_line_to_geopackage(line_objects_gdf, gpkg_file, 'tbl_lines_original', log_widget)

    log_to_gui(log_widget, "COMPLETED: Line imports done.")

    copy_original_lines_to_tbl_lines(gpkg_file, segment_width, segment_length)

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


def process_geopackage_layers(filepath, asset_objects, asset_groups, object_id_counter, group_id_counter, log_widget):
    """
    Process each layer in a GeoPackage file, adding each layer's data to the asset_objects list,
    and creating a new entry in asset_groups for each layer.
    """
    # Open the GeoPackage file
    with fiona.open(filepath) as src:
        layer_names = src.layer_names()
    
    # Process each layer individually
    for layer_name in layer_names:
        data = read_and_reproject(filepath, layer=layer_name, log_widget=log_widget)
        if not data.empty:
            # Process layer data and update asset_objects and asset_groups
            bbox_polygon = box(*data.total_bounds)  # Create a Polygon geometry from the bounding box
            asset_groups.append({
                'id': group_id_counter,
                'name_original': layer_name,
                'geom': bbox_polygon,  # Using the Polygon geometry here
                # Add other necessary attributes here
            })
            for _, row in data.iterrows():
                asset_objects.append({
                    'id': object_id_counter,
                    'ref_asset_group': group_id_counter,
                    'geom': row.geometry,  # Directly use geometry
                    # Add other necessary attributes here
                })
                object_id_counter += 1
            group_id_counter += 1
            log_to_gui(log_widget, f"Processed layer {layer_name} from GeoPackage {os.path.basename(filepath)}")
        else:
            log_to_gui(log_widget, f"Layer {layer_name} in GeoPackage {os.path.basename(filepath)} is empty")

    return asset_objects, asset_groups, object_id_counter, group_id_counter


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
        # Assuming each dictionary in the list has a 'geom' key with geometry data
        if gdf_or_list and 'geom' in gdf_or_list[0]:
            gdf = gpd.GeoDataFrame(gdf_or_list)
            # Convert geometries from WKT or GeoJSON to Shapely geometries if they are not already
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
    
    # Attempt to save the GeoDataFrame to the specified layer in the GeoPackage
    try:
        gdf.to_file(gpkg_file, layer=layer_name, driver='GPKG', if_exists='replace')
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

            gdf.to_file(gpkg_file, layer=layer_name, driver='GPKG', if_exists='replace')
            log_to_gui(log_widget, f"Data for layer {layer_name} saved in GeoPackage.")
        else:
            log_to_gui(log_widget, f"Warning: Attempted to save an empty GeoDataFrame for layer {layer_name}.")


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

    log_to_gui(log_widget, "Cleaning up. Deleting old assets, if they exist.")

    delete_table_from_geopackage(gpkg_file, 'tbl_asset_object', log_widget=None)

    delete_table_from_geopackage(gpkg_file, 'tbl_asset_group', log_widget=None)

    asset_objects_gdf, asset_groups_gdf, total_bbox_geom = import_spatial_data_asset(input_folder_asset, log_widget, progress_var)

    # Check if the GeoDataFrame is not empty before exporting
    if not asset_objects_gdf.empty:  # Corrected line
        log_to_gui(log_widget, "Starting export of asset objects to geopackage")
        export_to_geopackage(asset_objects_gdf, gpkg_file, 'tbl_asset_object', log_widget)
    else:
        log_to_gui(log_widget, "No asset objects to export.")
    
    
    if not asset_groups_gdf.empty:
        log_to_gui(log_widget, "Starting export of asset groups to geopackage")
        export_to_geopackage(asset_groups_gdf, gpkg_file, 'tbl_asset_group', log_widget)
    else:
        log_to_gui(log_widget, "No asset groups to export.")
        
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
            # Check if the specified table exists
            layer = ds.GetLayerByName(table_name)
            if layer is not None:
                ds.DeleteLayer(table_name)
                message = f"Table {table_name} deleted from {gpkg_file}."
                if log_widget:
                    log_to_gui(log_widget, message)
                else:
                    print(message)
            else:
                message = f"Table {table_name} does not exist in {gpkg_file}."
                if log_widget:
                    log_to_gui(log_widget, message)
                else:
                    print(message)
            ds = None  # Ensure the data source is closed
        else:
            message = f"Failed to open {gpkg_file}."
            if log_widget:
                log_to_gui(log_widget, message)
            else:
                print(message)
    except Exception as e:
        message = f"An error occurred while attempting to delete table {table_name} from {gpkg_file}: {e}"
        if log_widget:
            log_to_gui(log_widget, message)
        else:
            print(message)



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


def get_current_directory_and_file():
    # Get the current working directory
    current_directory = os.getcwd()
    
    # Get the name of the current file
    current_file = os.path.basename(sys.argv[0])
    
    return current_directory, current_file


# Function to close the application
def close_application():
    root.destroy()


#####################################################################################
#  Main
#

# Load configuration settings
config_file             = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')
config                  = read_config(config_file)

input_folder_asset      = config['DEFAULT']['input_folder_asset']
input_folder_geocode    = config['DEFAULT']['input_folder_geocode']
input_folder_lines      = config['DEFAULT']['input_folder_lines']
gpkg_file               = config['DEFAULT']['gpkg_file']

segment_width           = config['DEFAULT']['segment_width']
segment_length          = config['DEFAULT']['segment_length']
ttk_bootstrap_theme     = config['DEFAULT']['ttk_bootstrap_theme']
workingprojection_epsg  = config['DEFAULT']['workingprojection_epsg']

# Create the user interface
root = ttk.Window(themename=ttk_bootstrap_theme)  # Use ttkbootstrap Window
root.title("Import assets")

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