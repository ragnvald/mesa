import geopandas as gpd
import pandas as pd
import configparser
import datetime
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import os
import contextily as ctx

def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config

# Logging function to write to the GUI log
def write_to_log(message):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted_message = f"{timestamp} - {message}"
    with open("log.txt", "a") as log_file:
        log_file.write(formatted_message + "\n")

def plot_geopackage_layer(gpkg_file, layer_name, output_png, crs='EPSG:4326'):
    try:
        layer = gpd.read_file(gpkg_file, layer=layer_name)

        if layer.empty or layer.is_empty.any() or 'geometry' not in layer.columns:
            print(f"Layer {layer_name} is empty, contains invalid geometries, or has no geometry column.")
            return

        fig, ax = plt.subplots(figsize=(40, 40), dpi=200)
        
        if layer.crs.to_string() != crs:
            layer = layer.to_crs(crs)

        polygons = layer[layer.geom_type.isin(['Polygon', 'MultiPolygon'])]
        lines = layer[layer.geom_type.isin(['LineString', 'MultiLineString'])]
        points = layer[layer.geom_type.isin(['Point', 'MultiPoint'])]

        if not points.empty:
            points.plot(ax=ax, alpha=0.7, color='#99EE77')
        if not lines.empty:
            lines.plot(ax=ax, alpha=0.7, color='#99EE77')
        if not polygons.empty:
            polygons.plot(ax=ax, alpha=0.7, facecolor='#99EE77', edgecolor='#8CC7D6')

        # Add basemap
        ctx.add_basemap(ax, crs=crs, source=ctx.providers.OpenStreetMap.Mapnik)

        # Default plot limits in case of invalid bounds
        default_limits = (-180, 180, -90, 90)

        try:
            total_bounds = layer.total_bounds
            if np.all(np.isfinite(total_bounds)):
                ax.set_xlim(total_bounds[0], total_bounds[2])
                ax.set_ylim(total_bounds[1], total_bounds[3])
            else:
                ax.set_xlim(default_limits[0], default_limits[1])
                ax.set_ylim(default_limits[2], default_limits[3])
        except Exception as e:
            print(f"Error setting plot limits, using default values: {e}")
            ax.set_xlim(default_limits[0], default_limits[1])
            ax.set_ylim(default_limits[2], default_limits[3])

        plt.savefig(output_png, bbox_inches='tight')
        plt.close(fig)  # Close the figure to ensure the file is written
        print(f"Plot saved to {output_png}")

    except Exception as e:
        print(f"Error processing layer {layer_name}: {e}")

def fetch_asset_group_statistics(db_path):
    """
    Fetches the asset group statistics from the database.
    
    :param db_path: Path to the SQLite database file
    :return: A pandas DataFrame containing the asset group statistics
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    sql_query = """
    SELECT a.*, COUNT(b.ref_asset_group) AS asset_objects_nr
    FROM tbl_asset_group AS a
    LEFT JOIN tbl_asset_object AS b ON a.id = b.ref_asset_group
    GROUP BY a.sensitivity_code
    ORDER BY a.sensitivity_code;
    """
    
    cur.execute(sql_query)
    results = cur.fetchall()
    column_names = [description[0] for description in cur.description]
    conn.close()
    
    df_results = pd.DataFrame(results, columns=column_names)
    
    return df_results

def export_to_excel(data_frame, file_path):
    """
    Exports a pandas DataFrame to an Excel file.
    
    :param data_frame: The pandas DataFrame to export
    :param file_path: The file path to save the Excel file
    """
    if not file_path.endswith('.xlsx'):
        file_path += '.xlsx'
    
    data_frame.to_excel(file_path, index=False)

#####################################################################################
#  Main
#
config_file = 'config.ini'
config = read_config(config_file)

gpkg_file = os.path.join('..', config['DEFAULT']['gpkg_file'])
output_png = os.path.join('..', config['DEFAULT']['output_png'])
asset_output_png = '../output/asset.png'
flat_output_png = '../output/flat.png'
geocode_output_png = '../output/geocode.png'
excel_output = '../output/asset_group_statistics.xlsx'

workingprojection_epsg = config['DEFAULT']['workingprojection_epsg']

plot_geopackage_layer(gpkg_file, 'tbl_asset_object', asset_output_png)
write_to_log("Overview of assets exported")
plot_geopackage_layer(gpkg_file, 'tbl_flat', flat_output_png)
write_to_log("Overview of flat tables exported")

df_asset_group_statistics = fetch_asset_group_statistics(gpkg_file)
export_to_excel(df_asset_group_statistics, excel_output)
write_to_log("Excel table exported")
