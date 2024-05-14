import geopandas as gpd
import pandas as pd
import os
import matplotlib.pyplot as plt

# Define paths
geopackage_file = 'output/mesa.gpkg'
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)
excel_assets_stats = os.path.join(output_folder, 'assets_stats_per_sensitivity.xlsx')
excel_overall_stats = os.path.join(output_folder, 'overall_stats_per_sensitivity.xlsx')
bar_chart_output = os.path.join(output_folder, 'sensitivity_overall_stats_bar_chart.png')

# Load data
gdf_assets = gpd.read_file(geopackage_file, layer='tbl_asset_object')
tbl_asset_group = gpd.read_file(geopackage_file, layer='tbl_asset_group')
tbl_asset_group = tbl_asset_group.drop(columns=['geometry'])
gdf_assets = gdf_assets.merge(tbl_asset_group, left_on='ref_asset_group', right_on='id')
gdf_assets = gpd.GeoDataFrame(gdf_assets, geometry='geometry')
gdf_assets = gdf_assets[gdf_assets.geometry.type.isin(['Polygon', 'MultiPolygon'])]
gdf_assets = gdf_assets.to_crs('ESRI:54009')  # Mollweide projection
gdf_assets['area'] = gdf_assets['geometry'].area
gdf_assets['sensitivity_text'] = gdf_assets['sensitivity_code'] + ' | ' + gdf_assets['sensitivity_description']

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

