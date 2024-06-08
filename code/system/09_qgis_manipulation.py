# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 
# Work in progress
#
# Purpose
#
# * Create data frames based on tables tbl_geocode_group, tbl_asset_group
#   and tbl_atlas. This has been done through making a general function. 
#   We might want to filter the attributes, at least loose the geom
#   
# * Use the data frame tbl_geocode_group to set up layers for tbl_flat and 
#   tbl_stacked.
# ** tbl_flat function to establish <maplayer> under <projectlayers>. This
#    returns a text as well as the unique layer id (layer name+uuid) which 
#    will be used later in the definitions.
# ** combine data frame tbl_geocode group with maplayer info to filter (and 
#    repeat) layers. If there is just one geocode, then just one layer for
#    each of the tbl_stacked and tbl_flat tables (layers).
#
# * Set up atlases based on dataframe tbl_atlas.

import xml.etree.ElementTree as ET
import uuid
import geopandas as gpd
import pandas as pd
import os
import argparse
import pandas 
from sqlalchemy import create_engine
import configparser

#Function reads a specified layer from a GeoPackage file and prints its contents.
# - geopackage_path: Path to the GeoPackage fil
# - layer_name: Name of the layer to be read

def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config

def list_geopackage_layers(gpkg_file, tablerequest):
    engine = create_engine(f'sqlite:///{gpkg_file}')
    return pd.read_sql_table(tablerequest, engine)

def add_layer_to_qgs(qgs_file, layer_name, gpkg_path, filter_query=None):
    tree = ET.parse(qgs_file)
    root = tree.getroot()

    layer_tree_group = root.find('.//layer-tree-group')
    custom_order = layer_tree_group.find('.//custom-order')
    snapping_settings = root.find('.//snapping-settings/individual-layer-settings')

    existing_layer_id = None
    for layer in layer_tree_group.findall('layer-tree-layer'):
        if layer.attrib['name'] == f"{layer_name}":
            existing_layer_id = layer.attrib['id']
            break

    if not existing_layer_id:
        layer_id = f"{layer_name}_{str(uuid.uuid4()).replace('-', '_')}"
        
        # Define the source string with an optional filter
        source_str = f"{gpkg_path}|layername={layer_name}"
        if filter_query:
            source_str += f"|subset={filter_query}"

        # Define new layer
        new_layer = ET.SubElement(layer_tree_group, 'layer-tree-layer', {
            'providerKey': 'ogr',
            'expanded': '1',
            'legend_exp': '',
            'checked': 'Qt::Unchecked',
            'legend_split_behavior': '0',
            'id': layer_id,
            'name': f"{layer_name}",
            'source': source_str,
            'patch_size': '-1,-1'
        })

        # Add the new layer above custom_order if custom_order exists
        if custom_order is not None:
            custom_order_item = ET.Element('item')
            custom_order_item.text = layer_id
            custom_order.insert(0, custom_order_item)
            custom_order_item.tail = "\n"  # Add a newline after the item element
        else:
            # If custom_order does not exist, create it and add the new layer
            custom_order = ET.SubElement(layer_tree_group, 'custom-order')
            custom_order_item = ET.Element('item')
            custom_order_item.text = layer_id
            custom_order.insert(0, custom_order_item)
            custom_order_item.tail = "\n"  # Add a newline after the item element

        # Add new layer to snapping settings
        new_snapping_setting = ET.SubElement(snapping_settings, 'layer-setting', {
            'type': '1',
            'maxScale': '0',
            'enabled': '0',
            'units': '1',
            'minScale': '0',
            'tolerance': '12',
            'id': layer_id
        })

        new_snapping_setting.tail = "\n"  # Add a newline for layout   
        
        tree.write(qgs_file)
        return layer_id
    else:
        # Check if snapping setting exists for this layer
        setting_exists = any(setting.attrib.get('id') == existing_layer_id for setting in snapping_settings.findall('layer-setting'))
        if not setting_exists:
            # Add new layer to snapping settings
            new_snapping_setting = ET.SubElement(snapping_settings, 'layer-setting', {
                'tolerance': '12',
                'units': '1',
                'type': '1',
                'maxScale': '0',
                'id': existing_layer_id,
                'enabled': '0',
                'minScale': '0'
            })

            tree.write(qgs_file)

        return existing_layer_id
    
def read_geopackage_layer(geopackage_path, layer_name):
    try:
        # Reading the layer data using Geopandas
        layer_data = gpd.read_file(geopackage_path, layer=layer_name)

        # Print the data
        print(f"Data from layer '{layer_name}':\n{layer_data}")
        
        return gpd.read_file(geopackage_path, layer=layer_name)

    except Exception as e:
        print(f"Error reading layer '{layer_name}' from GeoPackage '{geopackage_path})': {e}")


def read_geopackage_layer_names(geopackage_path, layer_name):
    try:
        # Reading only the 'name_gis' column from the layer data using Geopandas
        layer_data = gpd.read_file(geopackage_path, layer=layer_name)[['name_gis']]

        # Print the data
        print(f"Data from layer '{layer_name}':\n{layer_data}")

        return layer_data

    except Exception as e:
        print(f"Error reading 'name_gis' from layer '{layer_name}' in GeoPackage '{geopackage_path}': {e}")


# Load configuration settings and data
config_file = 'config.ini'
config = read_config(config_file)
gpkg_file_codelevel = config['DEFAULT']['gpkg_file']
gpkg_file_qgslevel = "../../"+gpkg_file_codelevel
ttk_bootstrap_theme = config['DEFAULT']['ttk_bootstrap_theme']

qgs_file = 'qgis/mesa.qgs'

# Example usage

# Create layers for all asset groups
# This allows the user to "recreate" all assets within the QGIS setting.

table_filter        = 'tbl_asset_group'
table_output        = 'tbl_asset'
table_folder_name   = 'Original assets'

# Get a list of layers from the GeoPackage

df_posts_in_table = list_geopackage_layers(gpkg_file_codelevel,table_filter)

# Iterate over each row in the DataFrame
for index, row in df_posts_in_table.iterrows():
    # Access 'name_gis' and 'another_attribute' for each row
    name_gis = row['name_gis']
    title_fromuser = row['title_fromuser']
    group_id = row['id']

    # Initiate a new layer with arguments:
    # tbl_asset         Data table
    # name_gis          Filter names
    # title_from_user   The name which will appear for the user
    filter_query = f'\"id\"={group_id}'
    
    print(qgs_file, table_output, title_fromuser, filter_query)
    layer_id= add_layer_to_qgs(qgs_file, table_output, gpkg_file_qgslevel, filter_query)
    print("Layer ID:", layer_id)