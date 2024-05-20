import geopandas as gpd
import pandas as pd
import os
import matplotlib.pyplot as plt
import fiona

# Define paths
geopackage_file = '../output/mesa.gpkg'
output_folder = '../output'
os.makedirs(output_folder, exist_ok=True)
excel_assets_stats = os.path.join(output_folder, 'assets_stats_per_sensitivity.xlsx')
excel_overall_stats = os.path.join(output_folder, 'overall_stats_per_sensitivity.xlsx')
bar_chart_output = os.path.join(output_folder, 'sensitivity_overall_stats_bar_chart.png')

# List all layers in the GeoPackage file using fiona
layers = fiona.listlayers(geopackage_file)
print("Available layers in GeoPackage:", layers)

# Load data from all geometry types
gdf_assets_list = []
for layer in layers:
    if 'tbl_asset_object' in layer:
        gdf = gpd.read_file(geopackage_file, layer=layer)
        gdf_assets_list.append(gdf)

# Combine all geometry types into a single GeoDataFrame
gdf_assets = pd.concat(gdf_assets_list, ignore_index=True)
print("Combined gdf_assets data:\n", gdf_assets.head())

tbl_asset_group = gpd.read_file(geopackage_file, layer='tbl_asset_group')
print("Initial tbl_asset_group data:\n", tbl_asset_group.head())

# Drop geometry column from asset group table and merge with assets
tbl_asset_group = tbl_asset_group.drop(columns=['geometry'])

# Verify unique IDs in tbl_asset_group
print("Unique IDs in tbl_asset_group:", tbl_asset_group['id'].unique())

# Verify unique ref_asset_group in gdf_assets
print("Unique ref_asset_group in gdf_assets:", gdf_assets['ref_asset_group'].unique())

# Merge operation
gdf_assets = gdf_assets.merge(tbl_asset_group, left_on='ref_asset_group', right_on='id', how='left')

# Check merged data
print("Merged gdf_assets data:\n", gdf_assets.all())

# Create GeoDataFrame and filter polygons
gdf_assets = gpd.GeoDataFrame(gdf_assets, geometry='geometry')
gdf_assets = gdf_assets[gdf_assets.geometry.type.isin(['Polygon', 'MultiPolygon'])]

# Reproject and calculate area
gdf_assets = gdf_assets.to_crs('ESRI:54009')  # Mollweide projection
gdf_assets['area'] = gdf_assets['geometry'].area

# Combine sensitivity code and description
gdf_assets['sensitivity_text'] = gdf_assets['sensitivity_code'] + ' | ' + gdf_assets['sensitivity_description']

# Verify unique sensitivity categories after processing
unique_sensitivity_categories = gdf_assets['sensitivity_text'].unique()
print("Unique Sensitivity Categories after processing:", unique_sensitivity_categories)

# Check if 'A' category is missing
if not any('A' in category for category in unique_sensitivity_categories):
    print("Warning: Category 'A' is missing from the data!")

def format_area(row):
    if row['total_area'] < 1_000_000:
        return f"{row['total_area']:.0f} m²"
    else:
        return f"{row['total_area'] / 1_000_000:.2f} km²"

def asset_statistics_by_sensitivity(gdf, groupby_column):
    stats = gdf.groupby(groupby_column).agg(
        total_area=('area', 'sum'),
        count=('geometry', 'size')
    ).reset_index()
    stats['total_area'] = stats.apply(format_area, axis=1)
    return stats

overall_stats_table = asset_statistics_by_sensitivity(gdf_assets, 'sensitivity_text')
overall_stats_table.to_excel(excel_overall_stats, index=False)
print(overall_stats_table)

def create_bar_chart(df, output_path):
    # Ensure the sort_key is handling alphabetic codes correctly
    df['sort_key'] = df['sensitivity_text'].str.extract('([A-Za-z]+)').fillna(df['sensitivity_text'])
    df_sorted = df.sort_values(by='sort_key', ascending=False)
    labels = df_sorted['sensitivity_text']
    values = df_sorted['total_area'].str.replace(' m²', '').str.replace(' km²', '').astype(float)

    # Check if the values are in square meters or square kilometers
    is_km = 'km²' in df_sorted['total_area'].iloc[0]

    # Adjust the units based on the format
    if is_km:
        ylabel = 'Total Area (km²)'
    else:
        ylabel = 'Total Area (m²)'

    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(labels, values, color='teal')
    plt.xlabel(ylabel)
    plt.ylabel('Sensitivity Category')
    plt.title('Total Areas per Sensitivity Category')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

create_bar_chart(overall_stats_table, bar_chart_output)
