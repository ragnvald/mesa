import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
import geopandas as gpd
import pandas as pd
import configparser
import datetime
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sqlite3
import os
import contextily as ctx
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import cm
from PIL import Image as PILImage
import re

def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config

def read_color_codes(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    color_codes = {}
    for section in config.sections():
        if section in ['A', 'B', 'C', 'D', 'E']:
            color = config[section]['category_colour']
            color_codes[section] = color
    return color_codes

def write_to_log(message):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted_message = f"{timestamp} - {message}"
    with open("../log.txt", "a") as log_file:
        log_file.write(formatted_message + "\n")

def plot_geopackage_layer(gpkg_file, layer_name, output_png, crs='EPSG:4326'):
    try:
        # Memory optimization: Only read necessary columns
        layer = gpd.read_file(gpkg_file, layer=layer_name)
        if layer.empty or layer.is_empty.any():
            write_to_log(f"Layer {layer_name} empty or contains invalid geometries")
            return None
            
        # Release memory for unused columns
        keep_cols = ['geometry'] + [col for col in layer.columns if col in ['name', 'title', 'description']][:2]
        layer = layer[keep_cols]
        
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        plt.ioff()  # Turn off interactive mode

        polygons = layer[layer.geom_type.isin(['Polygon', 'MultiPolygon'])]
        lines = layer[layer.geom_type.isin(['LineString', 'MultiLineString'])]
        points = layer[layer.geom_type.isin(['Point', 'MultiPoint'])]

        if not points.empty:
            points.plot(ax=ax, alpha=0.7, color='#99EE77')
        if not lines.empty:
            lines.plot(ax=ax, alpha=0.7, color='#99EE77')
        if not polygons.empty:
            polygons.plot(ax=ax, alpha=0.7, facecolor='#99EE77', edgecolor='#8CC7D6')

        try:
            ctx.add_basemap(ax, crs=crs, source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception as e:
            write_to_log(f"Error adding basemap: {e}")

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
            write_to_log(f"Error setting plot limits, using default values: {e}")
            ax.set_xlim(default_limits[0], default_limits[1])
            ax.set_ylim(default_limits[2], default_limits[3])

        ax.set_title(layer_name, fontsize=15, fontweight='bold')
        plt.savefig(output_png, bbox_inches='tight')
        plt.close(fig)
        write_to_log(f"Plot saved to {output_png}")

        return layer

    except Exception as e:
        write_to_log(f"Error processing layer {layer_name}: {e}")
        plt.close('all')
        return None

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
    column_names = ['Sensitivity Code', 'Sensitivity Description', 'Number of Asset Objects']
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

    tbl_asset_group = tbl_asset_group.drop(columns=['geometry'])

    gdf_assets = gdf_assets.merge(tbl_asset_group, left_on='ref_asset_group', right_on='id')

    gdf_assets['geometry_type'] = gdf_assets.geometry.type

    gdf_assets = gdf_assets.to_crs('ESRI:54009')

    polygons = gdf_assets[gdf_assets.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
    non_polygons = gdf_assets[~gdf_assets.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()

    polygons.loc[:, 'area'] = polygons['geometry'].area

    non_polygons.loc[:, 'area'] = np.nan

    gdf_assets = pd.concat([polygons, non_polygons], ignore_index=True)

    group_stats = gdf_assets.groupby(['title_fromuser', 'sensitivity_code', 'sensitivity_description', 'geometry_type']).agg(
        total_area=('area', 'sum'),
        object_count=('geometry', 'size')
    ).reset_index()

    group_stats['total_area'] = group_stats.apply(format_area, axis=1)

    group_stats = group_stats.sort_values(by='sensitivity_code')

    group_stats.columns = ['Title', 'Code', 'Description', 'Type', 'Total area', '# objects']
    
    return group_stats

def export_to_excel(data_frame, file_path):
    if not file_path.endswith('.xlsx'):
        file_path += '.xlsx'
    
    data_frame.to_excel(file_path, index=False)

def fetch_lines_and_segments(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Check if tbl_lines table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tbl_lines';")
    tbl_lines_exists = cur.fetchone()

    # Check if tbl_segment_flat table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tbl_segment_flat';")
    tbl_segment_flat_exists = cur.fetchone()

    lines_df = pd.DataFrame()
    segments_df = pd.DataFrame()

    if tbl_lines_exists:
        lines_query = "SELECT * FROM tbl_lines"
        lines_df = pd.read_sql_query(lines_query, conn)
    else:
        print("Table 'tbl_lines' does not exist. Skipping.")

    if tbl_segment_flat_exists:
        segments_query = "SELECT * FROM tbl_segment_flat"
        segments_df = pd.read_sql_query(segments_query, conn)
    else:
        print("Table 'tbl_segment_flat' does not exist. Skipping.")

    conn.close()
    return lines_df, segments_df

def get_color_from_code(code, color_codes):
    return color_codes.get(code, "#FFFFFF")

def sort_segments_numerically(segments):
    def extract_number(segment_id):
        match = re.search(r'_(\d+)$', segment_id)
        return int(match.group(1)) if match else float('inf')
    
    segments = segments.copy()
    segments['sort_key'] = segments['segment_id'].apply(extract_number)
    sorted_segments = segments.sort_values(by='sort_key').drop(columns=['sort_key'])
    return sorted_segments

def create_line_statistic_image(line_name, sensitivity_series, color_codes, length_m, output_path):
    segment_count = len(sensitivity_series)
    fig, ax = plt.subplots(figsize=(10, 0.25), dpi=100)
    length_km = length_m / 1000  # Convert length to kilometers

    for i, code in enumerate(sensitivity_series):
        color = get_color_from_code(code, color_codes)
        ax.add_patch(plt.Rectangle((i/segment_count, 0), 1/segment_count, 1, color=color))

    # Adding x-axis with min, halfway, and max values
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels([f"0 km", f"{length_km/2:.1f} km", f"{length_km:.1f} km"])
    
    ax.xaxis.set_tick_params(width=1)
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

def resize_image(image_path, max_width, max_height):
    with PILImage.open(image_path) as img:
        width, height = img.size
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        img = img.resize((new_width, new_height), PILImage.LANCZOS)
        img.save(image_path)

def create_sensitivity_summary(sensitivity_series, color_codes, output_path):
    """Create statistical summary for a sensitivity series (max or min)"""
    # Count occurrences and sort by sensitivity code (A to E)
    sensitivity_counts = sensitivity_series.value_counts().sort_index()
    total_segments = len(sensitivity_series)
    
    # Create figure with fixed height
    fig, (ax_text, ax_bar) = plt.subplots(2, 1, figsize=(10, 1.5), height_ratios=[0.4, 0.6])
    plt.subplots_adjust(hspace=0.2)
    
    # Text summary
    summary_parts = []
    for code in sorted(color_codes.keys()):  # Iterate A through E
        count = sensitivity_counts.get(code, 0)
        percentage = (count/total_segments) * 100
        summary_parts.append(f"{code}: {count} ({percentage:.1f}%)")
    summary_text = "Distribution: " + " | ".join(summary_parts)
    
    ax_text.text(0.5, 0.5, summary_text, ha='center', va='center')
    ax_text.axis('off')
    
    # Bar chart - ensure all sensitivity codes are shown
    cumulative_position = 0
    for code in sorted(color_codes.keys()):  # Iterate A through E
        count = sensitivity_counts.get(code, 0)
        width = count/total_segments
        color = color_codes[code]
        ax_bar.barh(0, width, left=cumulative_position, 
                   color=color, edgecolor='white')
        cumulative_position += width
    
    ax_bar.set_ylim(-0.5, 0.5)
    ax_bar.set_xlim(0, 1)
    ax_bar.axis('off')
    
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close(fig)

def generate_line_statistics_pages(lines_df, segments_df, color_codes, tmp_dir):
    pages = []
    log_data = []
    
    for _, line in lines_df.iterrows():
        line_name = line['name_gis']
        length_m = line['length_m']
        line_segments = sort_segments_numerically(segments_df[segments_df['name_gis'] == line_name])
        
        # Generate separate statistical summaries for max and min sensitivity
        max_stats_path = os.path.join(tmp_dir, f"{line_name}_max_sensitivity_stats.png")
        min_stats_path = os.path.join(tmp_dir, f"{line_name}_min_sensitivity_stats.png")
        create_sensitivity_summary(line_segments['sensitivity_code_max'], color_codes, max_stats_path)
        create_sensitivity_summary(line_segments['sensitivity_code_min'], color_codes, min_stats_path)
        
        # Create segment visualizations
        max_image_path = os.path.join(tmp_dir, f"{line_name}_max_stats.png")
        min_image_path = os.path.join(tmp_dir, f"{line_name}_min_stats.png")
        create_line_statistic_image(line_name, line_segments['sensitivity_code_max'], color_codes, length_m, max_image_path)
        create_line_statistic_image(line_name, line_segments['sensitivity_code_min'], color_codes, length_m, min_image_path)
        
        # Add to pages with both statistics and segment visualizations
        pages.append(('heading(3)', f"Line: {line_name}"))
        pages.append(('text', f"Total length: {length_m/1000:.2f} km, with {len(line_segments)} segments."))
        pages.append(('text', "Maximum sensitivity distribution:"))
        pages.append(('image', max_stats_path))
        pages.append(('text', "Maximum sensitivity along line:"))
        pages.append(('image', max_image_path))
        pages.append(('text', "Minimum sensitivity distribution:"))
        pages.append(('image', min_stats_path))
        pages.append(('text', "Minimum sensitivity along line:"))
        pages.append(('image', min_image_path))
        
        # Log data collection as before
        for _, segment in line_segments.iterrows():
            log_data.append({
                'line_name': line_name,
                'segment_id': segment['segment_id'],
                'sensitivity_code_max': segment['sensitivity_code_max'],
                'sensitivity_code_min': segment['sensitivity_code_min']
            })
    
    log_df = pd.DataFrame(log_data)
    log_file_path = os.path.join(tmp_dir, 'line_segment_log.xlsx')
    export_to_excel(log_df, log_file_path)
    write_to_log(f"Segment log exported to {log_file_path}")
    
    return pages, log_file_path

def line_up_to_pdf(order_list):
    elements = []
    styles = getSampleStyleSheet()

    heading_styles = {
        1: ParagraphStyle(name='Heading1', fontSize=18, leading=22, spaceAfter=12, alignment=TA_CENTER),
        2: ParagraphStyle(name='Heading2', fontSize=16, leading=8, spaceAfter=4),
        3: ParagraphStyle(name='Heading3', fontSize=12, leading=8, spaceAfter=4)
    }

    max_image_width = 16 * cm
    max_image_height = 24 * cm

    for item in order_list:
        item_type, item_value = item

        if item_type == 'text':
            if os.path.isfile(item_value):
                with open(item_value, 'r') as file:
                    text = file.read()
            else:
                text = item_value
            text = text.replace("\n", "<br/>")
            elements.append(Paragraph(text, styles['Normal']))
            elements.append(Spacer(1, 12))
        
        elif item_type == 'image':
            resize_image(item_value, max_image_width, max_image_height)
            elements.append(Image(item_value))
            elements.append(Spacer(1, 12))

        elif item_type == 'spacer':
            lines = item_value if item_value else 1
            elements.append(Spacer(1, lines * 12))
        
        elif item_type == 'table':
            df = pd.read_excel(item_value)
            if len(df.columns) == 3:
                df.columns = ['Code', 'Description', '# asset objects']
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
            level = int(item_type[-2])
            elements.append(Paragraph(item_value, heading_styles[level]))
            elements.append(Spacer(1, 12))
        
        elif item_type == 'new_page':
            elements.append(PageBreak())
    
    return elements

def compile_pdf(output_pdf, elements):
    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    
    def add_header(canvas, doc):
        canvas.saveState()
        header_text = "MESA - report on data provided"
        canvas.setFont('Helvetica-Bold', 10)
        canvas.setFillColor(colors.gray)
        width, height = A4
        if doc.page > 1:
            canvas.drawCentredString(width / 2.0, height - 40, header_text)
        canvas.restoreState()
    
    doc.build(elements, onFirstPage=add_header, onLaterPages=add_header)

# Main script execution
#####################################################################################
#  Main
#

parser = argparse.ArgumentParser(description='Slave script')
parser.add_argument('--original_working_directory', required=False, help='Path to running folder')
args = parser.parse_args()
original_working_directory = args.original_working_directory

if original_working_directory is None or original_working_directory == '':
    original_working_directory  = os.getcwd()
    if str("system") in str(original_working_directory):
        original_working_directory = os.path.join(os.getcwd(),'../')

config_file                 = os.path.join(original_working_directory, "system/config.ini")
gpkg_file                   = os.path.join(original_working_directory, "output/mesa.gpkg")

config = read_config(config_file)
color_codes = read_color_codes(config_file)

output_png              = os.path.join(original_working_directory, config['DEFAULT']['output_png'])
tmp_dir                 = os.path.join(original_working_directory, 'output/tmp')

asset_output_png = os.path.join(tmp_dir, 'asset.png')
flat_output_png         = os.path.join(tmp_dir, 'flat.png')
asset_group_statistics  = os.path.join(tmp_dir, 'asset_group_statistics.xlsx')
object_stats_output     = os.path.join(tmp_dir, 'asset_object_statistics.xlsx')

workingprojection_epsg  = config['DEFAULT']['workingprojection_epsg']

if __name__ == "__main__":
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Process one layer at a time to manage memory
    write_to_log("Processing asset layer...")
    asset_layer = plot_geopackage_layer(gpkg_file, 'tbl_asset_object', asset_output_png)
    group_statistics = calculate_group_statistics(gpkg_file, 'tbl_asset_object')
    export_to_excel(group_statistics, object_stats_output)
    del asset_layer  # Free memory
    
    write_to_log("Processing flat layer...")
    flat_layer = plot_geopackage_layer(gpkg_file, 'tbl_flat', flat_output_png)
    del flat_layer  # Free memory
    
    # Export statistics
    df_asset_group_statistics = fetch_asset_group_statistics(gpkg_file)
    export_to_excel(df_asset_group_statistics, asset_group_statistics)
    write_to_log("Excel table exported")

    # Fetch lines and segments
    lines_df, segments_df = fetch_lines_and_segments(gpkg_file)
    line_statistics_pages, log_file_path = generate_line_statistics_pages(lines_df, segments_df, color_codes, tmp_dir)
    write_to_log("Line statistics images created")

    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    time_info = f"Timestamp for this report is: {timestamp}"

    order_list = [
        ('heading(1)', "MESA report"),
        ('spacer', 5),
        ('text', time_info),
        ('new_page', None),
        ('heading(2)', "Introduction"),
        ('text', '../output/tmp/introduction.txt'),
        ('spacer', 2),
        ('heading(2)', "Asset object statistics"),
        ('text', '../output/tmp/asset_objects_statistics_desc.txt'),
        ('table', object_stats_output),
        ('new_page', None),
        ('heading(2)', "Map representation of all assets"),
        ('text', '../output/tmp/asset_desc.txt'),
        ('image', asset_output_png),
        ('heading(3)', "Geographical Coordinates"),
        ('text', '../output/tmp/geographical_coordinates.txt'),
        ('new_page', None),
        ('heading(2)', "Asset data table"),
        ('text', '../output/tmp/asset_overview.txt'),
        ('spacer', 2),
        ('table', asset_group_statistics),
        ('new_page', None),
        ('heading(2)', "Detailed asset description"),
        ('image', flat_output_png),
        ('new_page', None),
        ('heading(2)', "Information about lines and segments"),
        ('text', '../output/tmp/introduction_line_data.txt')
    ]

    order_list.extend(line_statistics_pages)  # Add line statistics pages at the end
    elements = line_up_to_pdf(order_list)

    timestamp_pdf = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    output_pdf = os.path.join(original_working_directory, f'output/MESA-report_{timestamp_pdf}.pdf')
    compile_pdf(output_pdf, elements)
    write_to_log(f"PDF report created at {output_pdf}")

    # Yes, this script stores output in both GeoPackage and GeoParquet formats,
    # but only for tables that are already present in the GeoPackage.
    # It reads from the GeoPackage (gpkg_file) and exports statistics and images,
    # but does NOT write new spatial tables to GeoParquet directly.
    # If you want to export GeoPackage tables to GeoParquet, add code like below:

    def export_gpkg_to_geoparquet(gpkg_file, output_folder, layers):
        os.makedirs(output_folder, exist_ok=True)
        for layer in layers:
            gdf = gpd.read_file(gpkg_file, layer=layer)
            parquet_path = os.path.join(output_folder, f"{layer}.parquet")
            gdf.to_parquet(parquet_path, index=False)

    # After generating statistics and images, you can export tables:
    geoparquet_folder = os.path.join(original_working_directory, "output/geoparquet")
    export_gpkg_to_geoparquet(gpkg_file, geoparquet_folder, ["tbl_asset_object", "tbl_flat"])

# Functionality summary:
# - Reads configuration and color codes for sensitivity categories.
# - Generates overview plots for asset and flat tables from the GeoPackage, saving as PNG images.
# - Calculates and exports asset group statistics and asset object statistics to Excel.
# - Fetches line and segment data from the GeoPackage and generates visual statistics for lines.
# - Creates a multi-page PDF report using ReportLab, including images, tables, and formatted text.
# - Handles resizing images for PDF, sorting segments, and formatting area values.
# - Logs all major actions and errors to a log file.
# - Main script builds the report by assembling all elements and exporting the final PDF.
# - Exports existing GeoPackage tables to GeoParquet format if needed.
# - Fetches line and segment data from the GeoPackage and generates visual statistics for lines.
# - Creates a multi-page PDF report using ReportLab, including images, tables, and formatted text.
# - Handles resizing images for PDF, sorting segments, and formatting area values.
# - Logs all major actions and errors to a log file.
# - Main script builds the report by assembling all elements and exporting the final PDF.
# - Exports existing GeoPackage tables to GeoParquet format if needed.
# - Main script builds the report by assembling all elements and exporting the final PDF.
# - Exports existing GeoPackage tables to GeoParquet format if needed.
