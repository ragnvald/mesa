import pandas as pd
import geopandas as gpd
import configparser
import fiona
from datetime import datetime
from sqlalchemy import create_engine

# # # # # # # # # # # # # # 
# Shared/general functions

# Read the configuration file
def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config

# # # # # # # # # # # # # # 
# Core functions

# Function to extract metadata from geopackage layers and export to a geopackage file
def export_gpkg_metadata_to_gpkg(gpkg_path, destination_gpkg_path, table_name):
    # Read the names of all layers from the geopackage
    layer_names = fiona.listlayers(gpkg_path)

    # Prepare a list to store metadata for each layer
    layers_metadata = []

    # Collect metadata for each layer
    for layer_name in layer_names:
        # Load the layer as a GeoDataFrame
        layer = gpd.read_file(gpkg_path, layer=layer_name)

        # Extract metadata or placeholders for each column
        metadata = {
            'Datasett': layer_name,
            'asset_name': layer_name,
            'susceptibility': '',
            'importance': '',
            'sensitivity': '',
            'Latest_update_date': datetime.now().strftime('%Y%m%d'),
            'asset_type': '',
            'asset_referencenumber': '',
            'asset_date': '',
            'is_it_grid': 0,
            '_original_coordsys': layer.crs.to_string() if layer.crs else 'Unknown',
            '_count_excel_row': ''
        }

        # Append the metadata dictionary to the list
        layers_metadata.append(metadata)

    # Create a DataFrame from the metadata list
    metadata_df = pd.DataFrame(layers_metadata)

    # Create an SQLAlchemy engine
    engine = create_engine(f'sqlite:///{destination_gpkg_path}')

    # Write the metadata DataFrame to a geopackage file
    metadata_df.to_sql(table_name, con=engine, if_exists='replace', index=False)

# Load configuration settings
config_file = 'config.ini'
config = read_config(config_file)
input_folder_asset = config['DEFAULT']['input_folder_asset']
input_folder_geocode = config['DEFAULT']['input_folder_geocode']
gpkg_file = config['DEFAULT']['gpkg_file']

# Paths for the geopackage and the destination geopackage file
#destination_gpkg_path = 'output/mesa.gpkg'
table_name = 'tbl_asset_group'

# Call the function to export the geopackage metadata to another geopackage
export_gpkg_metadata_to_gpkg(gpkg_file, gpkg_file, table_name)
