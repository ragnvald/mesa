import geopandas as gpd
import pandas as pd
import os

# Define the paths
geopackage_file = 'output/mesa.gpkg'
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)
excel_assets_stats = os.path.join(output_folder, 'assets_stats_per_sensitivity.xlsx')
excel_overall_stats = os.path.join(output_folder, 'overall_stats_per_sensitivity.xlsx')

# Load both tables from the GeoPackage file
gdf_assets = gpd.read_file(geopackage_file, layer='tbl_asset_object')
tbl_asset_group = gpd.read_file(geopackage_file, layer='tbl_asset_group')

# Exclude geometry column from tbl_asset_group
tbl_asset_group = tbl_asset_group.drop(columns=['geometry'])

# Join the dataframes to enrich the geodataframe with sensitivity information
gdf_assets = gdf_assets.merge(tbl_asset_group, left_on='ref_asset_group', right_on='id')

# Ensure the geometry column is present and set explicitly
gdf_assets = gpd.GeoDataFrame(gdf_assets, geometry='geometry')

# Filter to only include Polygons and MultiPolygons
gdf_assets = gdf_assets[gdf_assets.geometry.type.isin(['Polygon', 'MultiPolygon'])]

# Reproject to an equal-area projection for accurate area calculations
equal_area_crs = 'ESRI:54009'  # Mollweide projection
gdf_assets = gdf_assets.to_crs(equal_area_crs)

# Ensure that our geodataframe has a valid geometry column
if not gdf_assets.geom_type.isin(['Polygon', 'MultiPolygon']).all():
    print("Warning: Not all geometries are polygons or multipolygons. Area calculations might not be accurate.")

# Calculate areas in square meters
gdf_assets['area'] = gdf_assets['geometry'].area

# Function to format areas based on magnitude
def format_area(row):
    if row['total_area'] < 1_000_000:
        return f"{row['total_area']:.0f} m²"
    else:
        return f"{row['total_area'] / 1_000_000:.2f} km²"

# Function to aggregate statistics by sensitivity category
def asset_statistics_by_sensitivity(gdf, groupby_column):
    stats = gdf.groupby(groupby_column).agg(
        total_area=('area', 'sum'),
        count=('geometry', 'size')
    ).reset_index()
    stats['total_area'] = stats.apply(format_area, axis=1)
    return stats

# Table 1: Each asset with total (summed) areas per sensitivity category
def asset_table_per_sensitivity(gdf):
    assets_stats = gdf.groupby(['asset_group_name', 'sensitivity']).agg(
        total_area=('area', 'sum'),
        count=('geometry', 'size')
    ).reset_index()
    assets_stats['total_area'] = assets_stats.apply(format_area, axis=1)
    return assets_stats

# Create a column combining sensitivity code and description
gdf_assets['sensitivity_text'] = gdf_assets['sensitivity_code'] + ' - ' + gdf_assets['sensitivity_description']

# Table 2: Total areas for all objects within each sensitivity category by descriptive text
def overall_table_per_sensitivity(gdf):
    overall_stats = asset_statistics_by_sensitivity(gdf, 'sensitivity_text')
    return overall_stats

# Generate the tables
assets_stats_table = asset_table_per_sensitivity(gdf_assets)
overall_stats_table = overall_table_per_sensitivity(gdf_assets)

# Save the results to Excel files
assets_stats_table.to_excel(excel_assets_stats, index=False)
overall_stats_table.to_excel(excel_overall_stats, index=False)

# Display tables for quick verification
print(assets_stats_table)
print(overall_stats_table)
