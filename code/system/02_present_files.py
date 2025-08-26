import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
import geopandas as gpd
import pandas as pd
import configparser
import datetime
import matplotlib
# Force non-interactive backend to avoid "Starting a Matplotlib GUI outside of the main thread" warnings
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import argparse
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
import tkinter as tk
from tkinter import scrolledtext
import ttkbootstrap as tb
from ttkbootstrap.constants import PRIMARY, WARNING
import threading
from pathlib import Path
import subprocess

# ---------------- GUI globals ----------------
log_widget = None
progress_var = None
progress_label = None
last_report_path = None
link_var = None  # hyperlink label StringVar

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

def write_to_log(message: str):
    """Log to file and (if GUI active) to the GUI log window."""
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted_message = f"{timestamp} - {message}"
    try:
        with open("../log.txt", "a", encoding="utf-8") as log_file:
            log_file.write(formatted_message + "\n")
    except Exception:
        pass
    try:
        if log_widget and log_widget.winfo_exists():
            log_widget.insert(tk.END, formatted_message + "\n")
            log_widget.see(tk.END)
    except Exception:
        pass

def plot_parquet_layer(parquet_path, layer_label, output_png, crs='EPSG:4326'):
    """
    Plot a single GeoParquet layer (polygons / lines / points mixed) with contextily basemap.
    """
    try:
        if not os.path.exists(parquet_path):
            write_to_log(f"Parquet file missing: {parquet_path}")
            return None
        layer = gpd.read_parquet(parquet_path)
        if layer.empty or layer.is_empty.any():
            write_to_log(f"Layer {layer_label} empty or contains invalid geometries")
            return None
            
        # keep geometry + first two descriptive columns if present
        keep_cols = ['geometry']
        for c in ['name','title','description','name_gis_geocodegroup','name_gis_assetgroup']:
            if c in layer.columns and c not in keep_cols:
                keep_cols.append(c)
            if len(keep_cols) >= 3:
                break
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

        ax.set_title(layer_label, fontsize=15, fontweight='bold')
        plt.savefig(output_png, bbox_inches='tight')
        plt.close(fig)
        write_to_log(f"Plot saved to {output_png}")

        return layer

    except Exception as e:
        write_to_log(f"Error processing layer {layer_label}: {e}")
        plt.close('all')
        return None

def fetch_asset_group_statistics(asset_group_df: gpd.GeoDataFrame, asset_object_df: gpd.GeoDataFrame):
    """
    Rebuild asset group sensitivity statistics from parquet data.
    """
    if asset_group_df.empty:
        return pd.DataFrame(columns=['Sensitivity Code','Sensitivity Description','Number of Asset Objects'])
    # Count objects per ref group
    if not asset_object_df.empty and 'ref_asset_group' in asset_object_df.columns:
        cnt = asset_object_df.groupby('ref_asset_group').size().rename('asset_objects_nr')
        merged = asset_group_df.merge(cnt, left_on='id', right_index=True, how='left')
    else:
        merged = asset_group_df.copy()
        merged['asset_objects_nr'] = 0
    merged['asset_objects_nr'] = merged['asset_objects_nr'].fillna(0).astype(int)
    cols_map = {
        'sensitivity_code': 'Sensitivity Code',
        'sensitivity_description': 'Sensitivity Description',
        'asset_objects_nr': 'Number of Asset Objects'
    }
    out = (merged.groupby(['sensitivity_code','sensitivity_description'], dropna=False)['asset_objects_nr']
                 .sum()
                 .reset_index()
                 .rename(columns=cols_map)
           )
    out = out.sort_values('Sensitivity Code')
    return out

def format_area(row):
    if row['geometry_type'] in ['Polygon', 'MultiPolygon']:
        if row['total_area'] < 1_000_000:
            return f"{row['total_area']:.0f} m²"
        else:
            return f"{row['total_area'] / 1_000_000:.2f} km²"
    else:
        return "-"

def calculate_group_statistics(asset_object_df: gpd.GeoDataFrame, asset_group_df: gpd.GeoDataFrame):
    """
    Build per asset group statistics (area/objects) from parquet frames.
    """
    if asset_object_df.empty or asset_group_df.empty:
        return pd.DataFrame(columns=['Title','Code','Description','Type','Total area','# objects'])
    tbl_asset_group = asset_group_df.drop(columns=[c for c in ['geometry'] if c in asset_group_df.columns])
    gdf_assets = asset_object_df.merge(tbl_asset_group, left_on='ref_asset_group', right_on='id', how='left')
    gdf_assets['geometry_type'] = gdf_assets.geometry.type
    try:
        gdf_assets = gdf_assets.to_crs('ESRI:54009')
    except Exception:
        pass
    polygons = gdf_assets[gdf_assets.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
    non_polygons = gdf_assets[~gdf_assets.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()

    polygons.loc[:, 'area'] = polygons['geometry'].area
    non_polygons.loc[:, 'area'] = np.nan

    gdf_assets = pd.concat([polygons, non_polygons], ignore_index=True)

    group_stats = gdf_assets.groupby(
        ['title_fromuser', 'sensitivity_code', 'sensitivity_description', 'geometry_type'],
        dropna=False
    ).agg(
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

def fetch_lines_and_segments(parquet_dir: str):
    """
    Load lines & segment flat tables from GeoParquet if present.
    """
    lines_pq    = os.path.join(parquet_dir, "tbl_lines.parquet")
    segments_pq = os.path.join(parquet_dir, "tbl_segment_flat.parquet")
    try:
        lines_df    = gpd.read_parquet(lines_pq) if os.path.exists(lines_pq) else gpd.GeoDataFrame()
    except Exception:
        lines_df = gpd.GeoDataFrame()
    try:
        segments_df = gpd.read_parquet(segments_pq) if os.path.exists(segments_pq) else gpd.GeoDataFrame()
    except Exception:
        segments_df = gpd.GeoDataFrame()
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

def set_progress(pct: float, message: str | None = None):
    """Update progress bar/label and optionally log a message."""
    try:
        pct = max(0, min(100, int(pct)))
        if progress_var is not None:
            progress_var.set(pct)
        if progress_label is not None:
            progress_label.config(text=f"{pct}%")
        if message:
            write_to_log(message)
    except Exception:
        pass

def generate_report(original_working_directory: str,
                    config_file: str,
                    color_codes: dict):
    """Run full report generation and update GUI log/progress."""
    try:
        set_progress(3, "Initializing report generation …")
        gpq_dir_path  = os.path.join(original_working_directory, "output", "geoparquet")
        cfg           = read_config(config_file)
        output_png    = os.path.join(original_working_directory, cfg['DEFAULT']['output_png'])
        tmp_dir       = os.path.join(original_working_directory, 'output', 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        set_progress(6, "Paths prepared.")

        asset_output_png       = os.path.join(tmp_dir, 'asset.png')
        flat_output_png        = os.path.join(tmp_dir, 'flat.png')
        asset_group_statistics = os.path.join(tmp_dir, 'asset_group_statistics.xlsx')
        object_stats_output    = os.path.join(tmp_dir, 'asset_object_statistics.xlsx')

        write_to_log("Starting report generation (GeoParquet)...")
        set_progress(10)
        asset_object_pq = os.path.join(gpq_dir_path, "tbl_asset_object.parquet")
        asset_group_pq  = os.path.join(gpq_dir_path, "tbl_asset_group.parquet")
        flat_pq         = os.path.join(gpq_dir_path, "tbl_flat.parquet")

        try:
            asset_objects_df = gpd.read_parquet(asset_object_pq) if os.path.exists(asset_object_pq) else gpd.GeoDataFrame()
        except Exception:
            asset_objects_df = gpd.GeoDataFrame()
        try:
            asset_groups_df  = gpd.read_parquet(asset_group_pq)  if os.path.exists(asset_group_pq)  else gpd.GeoDataFrame()
        except Exception:
            asset_groups_df = gpd.GeoDataFrame()

        write_to_log("Plotting asset layer...")
        _ = plot_parquet_layer(asset_object_pq, 'tbl_asset_object', asset_output_png)
        set_progress(20, "Asset layer plotted.")

        write_to_log("Computing asset object statistics...")
        set_progress(28, "Computing asset object statistics...")
        group_statistics = calculate_group_statistics(asset_objects_df, asset_groups_df)
        export_to_excel(group_statistics, object_stats_output)
        set_progress(34, "Asset object statistics exported.")

        write_to_log("Plotting flat layer...")
        _ = plot_parquet_layer(flat_pq, 'tbl_flat', flat_output_png)
        set_progress(42, "Flat layer plotted.")

        write_to_log("Exporting asset group sensitivity statistics...")
        set_progress(48, "Exporting asset group sensitivity statistics…")
        df_asset_group_statistics = fetch_asset_group_statistics(asset_groups_df, asset_objects_df)
        export_to_excel(df_asset_group_statistics, asset_group_statistics)
        set_progress(55, "Asset group sensitivity exported.")

        write_to_log("Loading lines and segments...")
        set_progress(58, "Loading lines and segments …")
        lines_df, segments_df = fetch_lines_and_segments(gpq_dir_path)
        required_seg_cols = {'name_gis', 'segment_id', 'sensitivity_code_max', 'sensitivity_code_min'}
        if (not lines_df.empty and not segments_df.empty and
            required_seg_cols.issubset(set(segments_df.columns)) and
            'length_m' in lines_df.columns and 'name_gis' in lines_df.columns):
            line_statistics_pages, _ = generate_line_statistics_pages(lines_df, segments_df, color_codes, tmp_dir)
            write_to_log("Line statistics generated.")
            set_progress(70, "Line statistics generated.")
        else:
            line_statistics_pages = []
            write_to_log("Skipping line statistics (missing required columns or empty datasets).")
            set_progress(70, "Skipped line statistics.")

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
        order_list.extend(line_statistics_pages)
        set_progress(72, "Composing PDF elements …")
        elements = line_up_to_pdf(order_list)
        set_progress(86, "Elements ready – building PDF …")

        timestamp_pdf = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
        output_pdf = os.path.join(original_working_directory, f'output/MESA-report_{timestamp_pdf}.pdf')
        compile_pdf(output_pdf, elements)
        set_progress(100, "Report completed.")
        global last_report_path
        last_report_path = output_pdf
        write_to_log(f"PDF report created: {output_pdf}")
        # Update link (if GUI)
        try:
            if link_var:
                link_var.set("Open report folder")
        except Exception:
            pass
    except Exception as e:
        write_to_log(f"ERROR during report generation: {e}")
        set_progress(100, "Report failed.")

def _start_report_thread(base_dir, config_file, color_codes):
    threading.Thread(target=generate_report,
                     args=(base_dir, config_file, color_codes),
                     daemon=True).start()

def launch_gui(original_working_directory: str, config_file: str, color_codes: dict, theme: str):
    global log_widget, progress_var, progress_label, link_var
    root = tb.Window(themename=theme)
    root.title("MESA – Report generator")
    try:
        ico = Path(original_working_directory) / "system_resources" / "mesa.ico"
        if ico.exists():
            root.iconbitmap(str(ico))
    except Exception:
        pass

    frame = tb.LabelFrame(root, text="Log", bootstyle="info")
    frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    log_widget = scrolledtext.ScrolledText(frame, height=14)
    log_widget.pack(fill=tk.BOTH, expand=True)

    pframe = tk.Frame(root); pframe.pack(pady=6)
    progress_var = tk.DoubleVar(value=0.0)
    pbar = tb.Progressbar(pframe, orient="horizontal", length=300, mode="determinate",
                          variable=progress_var, bootstyle="info")
    pbar.pack(side=tk.LEFT, padx=6)
    progress_label = tk.Label(pframe, text="0%", width=5, anchor="w")
    progress_label.pack(side=tk.LEFT)

    btn_frame = tk.Frame(root); btn_frame.pack(pady=4)
    tb.Button(btn_frame, text="Generate report", bootstyle=PRIMARY,
              command=lambda: _start_report_thread(original_working_directory, config_file, color_codes)).grid(row=0, column=0, padx=6)
    tb.Button(btn_frame, text="Exit", bootstyle=WARNING, command=root.destroy).grid(row=0, column=1, padx=6)

    # Live link placeholder
    def open_report_folder(event=None):
        if not last_report_path:
            return
        folder = os.path.dirname(last_report_path)
        if os.path.isdir(folder):
            try:
                os.startfile(folder)  # Windows
            except Exception:
                try:
                    subprocess.Popen(["explorer", folder])
                except Exception as ee:
                    write_to_log(f"Failed to open folder: {ee}")
        else:
            write_to_log("Report folder not found.")

    link_var = tk.StringVar(value="")
    link_label = tk.Label(root, textvariable=link_var, fg="#4ea3ff", cursor="hand2", font=("Segoe UI", 10, "underline"))
    link_label.pack(pady=(2,8))
    link_label.bind("<Button-1>", open_report_folder)

    write_to_log(f"Working directory: {original_working_directory}")
    write_to_log("Ready. Press 'Generate report' to start.")
    root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Presentation report (GeoParquet only)')
    parser.add_argument('--original_working_directory', required=False, help='Path to running folder')
    parser.add_argument('--no-gui', action='store_true', help='Run directly without GUI')
    args = parser.parse_args()

    original_working_directory = args.original_working_directory
    if not original_working_directory:
        original_working_directory = os.getcwd()
        if os.path.basename(original_working_directory).lower() == "system":
            original_working_directory = os.path.abspath(os.path.join(original_working_directory, ".."))

    config_file = os.path.join(original_working_directory, "system", "config.ini")
    config = read_config(config_file)
    color_codes = read_color_codes(config_file)
    theme = config['DEFAULT'].get('ttk_bootstrap_theme', 'flatly')

    if args.no_gui:
        generate_report(original_working_directory, config_file, color_codes)
    else:
        launch_gui(original_working_directory, config_file, color_codes, theme)
