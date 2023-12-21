
import geopandas as gpd
import pandas as pd
import configparser
from shapely.geometry import box

# Function to read the configuration file
def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config

def filter_and_update_atlas_geometries(atlas_geometries, tbl_flat):
    # Convert atlas_geometries to a GeoDataFrame
    atlas_gdf = gpd.GeoDataFrame(atlas_geometries, columns=['id', 'name_gis', 'title_user', 'geom'])
    atlas_gdf.set_geometry('geom', inplace=True)

    # Check for intersection
    filtered_indices = atlas_gdf.geometry.apply(lambda geom: tbl_flat.intersects(geom).any())
    intersecting_geometries = atlas_gdf[filtered_indices].copy()

    # Recalculate name_gis
    id_counter = 1
    for index, row in intersecting_geometries.iterrows():
        intersecting_geometries.loc[index, 'name_gis']   = f'atlas{id_counter:03}'
        intersecting_geometries.loc[index, 'title_user'] = f'Map title for {id_counter:03}'
        id_counter += 1
    return intersecting_geometries

# Function to generate atlas geometries
def generate_atlas_geometries(tbl_flat, atlas_lon_size_km, atlas_lat_size_km, atlas_overlap_percent):
    # Convert the atlas sizes to degrees (approximation)
    # Assuming 1 degree ~ 111 km
    lon_size_deg = atlas_lon_size_km / 111
    lat_size_deg = atlas_lat_size_km / 111
    overlap = atlas_overlap_percent / 100

    # Find the bounding box of all geometries in tbl_flat
    bounds = tbl_flat.total_bounds
    minx, miny, maxx, maxy = bounds

    # Initialize list to hold atlas geometries
    atlas_geometries = []
    id_counter = 1

    # Create grid of boxes
    y = miny
    while y < maxy:
        x = minx
        while x < maxx:
            # Create a box and append to list
            geom = box(x, y, x + lon_size_deg, y + lat_size_deg)
            atlas_geometries.append({'id': id_counter, 'name_gis': f'atlas{id_counter:03}',  'title_user': f'Map title for {id_counter:03}', 'geom': geom})
            id_counter += 1

            # Move to the next box in x-direction, consider overlap
            x += lon_size_deg * (1 - overlap)

        # Move to the next row of boxes in y-direction, consider overlap
        y += lat_size_deg * (1 - overlap)

    return atlas_geometries

# Main function
def main():
    # Read configuration
    config = read_config('config.ini')
    atlas_lon_size_km = float(config['DEFAULT']['atlas_lon_size_km'])
    atlas_lat_size_km = float(config['DEFAULT']['atlas_lat_size_km'])
    atlas_overlap_percent = float(config['DEFAULT']['atlas_overlap_percent'])
    gpkg_file = config['DEFAULT']['gpkg_file']

    # Load tbl_flat from GeoPackage
    tbl_flat = gpd.read_file(gpkg_file, layer='tbl_flat')

    # Generate atlas geometries
    atlas_geometries = generate_atlas_geometries(tbl_flat, atlas_lon_size_km, atlas_lat_size_km, atlas_overlap_percent)

    # Filter atlas geometries and update name_gis
    updated_atlas_geometries = filter_and_update_atlas_geometries(atlas_geometries, tbl_flat)

    # Save updated geometries to GeoPackage
    updated_atlas_geometries.to_file(gpkg_file, layer='tbl_atlas', driver='GPKG')

if __name__ == "__main__":
    main()
