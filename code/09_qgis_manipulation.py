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

import geopandas as gpd

#Function reads a specified layer from a GeoPackage file and prints its contents.
# - geopackage_path: Path to the GeoPackage fil
# - layer_name: Name of the layer to be read

def read_geopackage_layer(geopackage_path, layer_name):
    try:
        # Reading the layer data using Geopandas
        layer_data = gpd.read_file(geopackage_path, layer=layer_name)

        # Print the data
        print(f"Data from layer '{layer_name}':\n{layer_data}")
        
        return gpd.read_file(geopackage_path, layer=layer_name)

    except Exception as e:
        print(f"Error reading layer '{layer_name}' from GeoPackage '{geopackage_path}': {e}")

# Set the path to your GeoPackage file and the layer name
geopackage_file = 'output/mesa.gpkg'
layer_to_read = 'tbl_geocode_group'

# Call the function with the specified GeoPackage file and layer name
dataframe_tbl_geocode_group = read_geopackage_layer(geopackage_file, layer_to_read)

# Set the path to your GeoPackage file and the layer name
geopackage_file = 'output/mesa.gpkg'
layer_to_read = 'tbl_asset_group'

# Call the function with the specified GeoPackage file and layer name
dataframe_tbl_asset_group = read_geopackage_layer(geopackage_file, layer_to_read)


# Set the path to your GeoPackage file and the layer name
geopackage_file = 'output/mesa.gpkg'
layer_to_read = 'tbl_stacked'

# Call the function with the specified GeoPackage file and layer name
dataframe_tbl_stacked = read_geopackage_layer(geopackage_file, layer_to_read)


# Set the path to your GeoPackage file and the layer name
geopackage_file = 'output/mesa.gpkg'
layer_to_read = 'tbl_flat'

# Call the function with the specified GeoPackage file and layer name
dataframe_tbl_flat = read_geopackage_layer(geopackage_file, layer_to_read)
# Set the path to your GeoPackage file and the layer name

geopackage_file = 'output/mesa.gpkg'
layer_to_read = 'tbl_atlas'

# Call the function with the specified GeoPackage file and layer name
dataframe_tbl_atlas = read_geopackage_layer(geopackage_file, layer_to_read)