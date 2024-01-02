import geopandas as gpd

def read_geopackage_layer(geopackage_path, layer_name):
    """
    Reads a specified layer from a GeoPackage file and prints its contents.

    :param geopackage_path: Path to the GeoPackage file
    :param layer_name: Name of the layer to be read
    """
    try:
        # Reading the layer data using Geopandas
        layer_data = gpd.read_file(geopackage_path, layer=layer_name)

        # Print the data
        print(f"Data from layer '{layer_name}':\n{layer_data}")

    except Exception as e:
        print(f"Error reading layer '{layer_name}' from GeoPackage '{geopackage_path}': {e}")

# Set the path to your GeoPackage file and the layer name
geopackage_file = 'output/mesa.gpkg'
layer_to_read = 'tbl_geocode_group'

# Call the function with the specified GeoPackage file and layer name
read_geopackage_layer(geopackage_file, layer_to_read)

# Set the path to your GeoPackage file and the layer name
geopackage_file = 'output/mesa.gpkg'
layer_to_read = 'tbl_asset_group'

# Call the function with the specified GeoPackage file and layer name
read_geopackage_layer(geopackage_file, layer_to_read)