# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 
# Work in progress
# The plan is to create grid cells at a requested level based on the imported
# assets.
#

import geopandas as gpd
from shapely.geometry import Polygon, box
import numpy as np
import argparse

def fill_qdgc_to_geopackage(geopackage_path, area_polygon, qdgc_level):
    for level in range(1, qdgc_level + 1):
        cell_size = 1 / (2 ** level)
        grid = []

        # Generate grid cells
        for x in np.arange(area_polygon.bounds[0], area_polygon.bounds[2], cell_size):
            for y in np.arange(area_polygon.bounds[1], area_polygon.bounds[3], cell_size):
                cell = box(x, y, x + cell_size, y + cell_size)
                if cell.intersects(area_polygon):
                    grid.append(cell)

        df_grid = gpd.GeoDataFrame(grid, columns=['geometry'])
        df_grid['area_reference'] = 'Custom Area'  # Or any suitable name
        df_grid['level_qdgc'] = level

        # Calculate centroid, area, etc.
        df_grid['lon_center'] = df_grid.centroid.x
        df_grid['lat_center'] = df_grid.centroid.y
        df_grid['area_km2'] = df_grid.area
        df_grid['qdgc'] = '...' # Replace with your QDGC calculation logic

        # Write to GeoPackage
        df_grid.to_file(geopackage_path, layer=f'level_{level}', driver='GPKG', append=True)


#####################################################################################
#  Main
#

# Define the polygon for Norway (or any area of interest)
# Example: Polygon([(x1, y1), (x2, y2), ..., (xn, yn)])
norway_polygon = Polygon([...])  # Replace with the actual coordinates

# Path to your GeoPackage file
geopackage_path = "path/to/your_geopackage.gpkg"

# Example usage
fill_qdgc_to_geopackage(geopackage_path, norway_polygon, 3)
