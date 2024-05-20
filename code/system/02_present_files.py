import geopandas as gpd
import pandas as pd
import configparser
import datetime
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import os
import contextily as ctx
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config

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

        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        
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

        ctx.add_basemap(ax, crs=crs, source=ctx.providers.OpenStreetMap.Mapnik)

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
        plt.close(fig)
        print(f"Plot saved to {output_png}")

    except Exception as e:
        print(f"Error processing layer {layer_name}: {e}")

def fetch_asset_group_statistics(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    sql_query = """
    SELECT a.sensitivity_code, a.sensitivity_description, COUNT(b.ref_asset_group) AS asset_objects_nr
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

def format_area(row):
    if row['geometry_type'] in ['Polygon', 'MultiPolygon']:
        if row['total_area'] < 1_000_000:
            return f"{row['total_area']:.0f} m²"
        else:
            return f"{row['total_area'] / 1_000_000:.2f} km²"
    else:
        return "-"

def calculate_group_statistics(gpkg_file, layer_name):
    gdf_assets = gpd.read_file(gpkg_file, layer=layer_name)
    tbl_asset_group = gpd.read_file(gpkg_file, layer='tbl_asset_group')

    # Drop geometry column from asset group to merge with asset object table
    tbl_asset_group = tbl_asset_group.drop(columns=['geometry'])

    # Merge asset object table with asset group table
    gdf_assets = gdf_assets.merge(tbl_asset_group, left_on='ref_asset_group', right_on='id')

    # Add geometry type column
    gdf_assets['geometry_type'] = gdf_assets.geometry.type

    # Transform all geometries to Mollweide projection for consistency
    gdf_assets = gdf_assets.to_crs('ESRI:54009')

    # Separate polygons and non-polygons
    polygons = gdf_assets[gdf_assets.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    non_polygons = gdf_assets[~gdf_assets.geometry.type.isin(['Polygon', 'MultiPolygon'])]

    # Calculate area for polygons
    polygons.loc[:, 'area'] = polygons['geometry'].area

    # For non-polygons, set area to NaN
    non_polygons.loc[:, 'area'] = np.nan

    # Concatenate back together
    gdf_assets = pd.concat([polygons, non_polygons], ignore_index=True)

    # Group statistics calculation
    group_stats = gdf_assets.groupby(['title_fromuser', 'sensitivity_code', 'sensitivity_description', 'geometry_type']).agg(
        total_area=('area', 'sum'),
        object_count=('geometry', 'size')
    ).reset_index()

    # Format the area
    group_stats['total_area'] = group_stats.apply(format_area, axis=1)

    # Sort by sensitivity_code
    group_stats = group_stats.sort_values(by='sensitivity_code')
    
    return group_stats

def export_to_excel(data_frame, file_path):
    if not file_path.endswith('.xlsx'):
        file_path += '.xlsx'
    
    data_frame.to_excel(file_path, index=False)

def line_up_to_pdf(order_list):
    elements = []
    styles = getSampleStyleSheet()

    # Define custom heading styles
    heading_styles = {
        1: ParagraphStyle(name='Heading1', fontSize=18, leading=22, spaceAfter=12, alignment=TA_CENTER),
        2: ParagraphStyle(name='Heading2', fontSize=16, leading=8, spaceAfter=4),
        3: ParagraphStyle(name='Heading3', fontSize=14, leading=8, spaceAfter=4)
    }

    for item in order_list:
        item_type, item_value = item

        if item_type == 'text':
            with open(item_value, 'r') as file:
                text = file.read()
                text = text.replace("\n", "<br/>")  # Replace newlines with <br/> for HTML-style line breaks
                elements.append(Paragraph(text, styles['Normal']))
                elements.append(Spacer(1, 12))
        
        elif item_type == 'image':
            elements.append(Image(item_value, width=500, height=500))
            elements.append(Spacer(1, 12))
        
        elif item_type == 'table':
            df = pd.read_excel(item_value)
            table_data = [df.columns.tolist()] + df.values.tolist()
            table = Table(table_data)
            table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                       ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                       ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                       ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                       ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                       ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                       ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
            elements.append(table)
        
        elif item_type.startswith('heading'):
            level = int(item_type[-2])  # Extract heading level
            elements.append(Paragraph(item_value, heading_styles[level]))
            elements.append(Spacer(1, 12))
        
        elif item_type == 'new_page':
            elements.append(PageBreak())
    
    return elements

def compile_pdf(output_pdf, elements):
    doc = SimpleDocTemplate(output_pdf, pagesize=A4)  # Use A4 page size
    doc.build(elements)

#####################################################################################
#  Main
#
config_file = 'config.ini'
config = read_config(config_file)

gpkg_file = os.path.join('..', config['DEFAULT']['gpkg_file'])
output_png = os.path.join('..', config['DEFAULT']['output_png'])
tmp_dir = os.path.join('..', 'output/tmp')
os.makedirs(tmp_dir, exist_ok=True)

asset_output_png = os.path.join(tmp_dir, 'asset.png')
flat_output_png = os.path.join(tmp_dir, 'flat.png')
excel_output = os.path.join(tmp_dir, 'asset_group_statistics.xlsx')
object_stats_output = os.path.join(tmp_dir, 'asset_object_statistics.xlsx')

workingprojection_epsg = config['DEFAULT']['workingprojection_epsg']

plot_geopackage_layer(gpkg_file, 'tbl_asset_object', asset_output_png)
write_to_log("Overview of assets exported")
plot_geopackage_layer(gpkg_file, 'tbl_flat', flat_output_png)
write_to_log("Overview of flat tables exported")

df_asset_group_statistics = fetch_asset_group_statistics(gpkg_file)
export_to_excel(df_asset_group_statistics, excel_output)
write_to_log("Excel table exported")

# Calculate group statistics
group_statistics = calculate_group_statistics(gpkg_file, 'tbl_asset_object')
export_to_excel(group_statistics, object_stats_output)
write_to_log("Asset object statistics table exported")

# Define the order of objects for the PDF
order_list = [
    ('heading(1)', "MESA report"),
    ('heading(1)', "MESA report"),
    ('new_page', None),
    ('heading(2)', "Introduction"),
    ('text', '../output/tmp/introduction.txt'), 
    ('heading(2)', "Geographical Coordinates"),
    ('text', '../output/tmp/geographical_coordinates.txt'), 
    ('new_page', None),
    ('heading(2)', "Asset Data Table"),
    ('text', '../output/tmp/asset_overview.txt'),
    ('table', excel_output),
    ('new_page', None),
    ('heading(2)', "Detailed Asset Description"),
    ('text', '../output/tmp/asset_desc.txt'),
    ('image', flat_output_png),
    ('new_page', None),
    ('heading(2)', "Asset Data Overview"),
    ('heading(2)', "Map Representation"),
    ('image', asset_output_png),
    ('new_page', None),
    ('heading(2)', "Asset Object Statistics"),
    ('table', object_stats_output)
]

elements = line_up_to_pdf(order_list)

# Create PDF
output_pdf = '../output/report.pdf'
compile_pdf(output_pdf, elements)
write_to_log("PDF report created")
