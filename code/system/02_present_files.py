import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

import geopandas as gpd
import pandas as pd
import configparser
import datetime
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import argparse
import os
import contextily as ctx
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Image, Spacer,
    Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import cm as RL_CM
from reportlab.lib.colors import HexColor
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

# Primary UI color (Steel Blue by default)
PRIMARY_HEX = "#4682B4"  # Steel blue
LIGHT_PRIMARY_HEX = "#6fa6cf"  # slightly lighter steel blue

# ---------------- Config helpers ----------------
def read_config(file_name):
    cfg = configparser.ConfigParser()
    cfg.read(file_name)
    return cfg

def read_sensitivity_palette_and_desc(file_name):
    """
    Reads color palette and descriptions for A–E from config.ini.
    Returns: (colors: dict{A..E->hex}, desc: dict{A..E->str})
    """
    cfg = configparser.ConfigParser()
    cfg.read(file_name)
    colors_map, desc_map = {}, {}
    for code in ['A','B','C','D','E']:
        if cfg.has_section(code):
            col = cfg[code].get('category_colour', '').strip() or '#BDBDBD'
            colors_map[code] = col
            desc_map[code] = cfg[code].get('description', '').strip()
        else:
            colors_map[code] = '#BDBDBD'
            desc_map[code] = ''
    return colors_map, desc_map

# ---------------- Logging ----------------
def write_to_log(message: str):
    """Log to file and (if GUI active) to the GUI log window."""
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted = f"{timestamp} - {message}"
    try:
        with open("../log.txt", "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
    except Exception:
        pass
    try:
        if log_widget and log_widget.winfo_exists():
            log_widget.insert(tk.END, formatted + "\n")
            log_widget.see(tk.END)
    except Exception:
        pass

# ---------------- Generic plotting utils ----------------
def _safe_to_3857(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a copy of gdf in EPSG:3857 (Web Mercator) for contextily."""
    if gdf.empty:
        return gdf
    g = gdf.copy()
    try:
        if g.crs is None:
            g = g.set_crs(4326)
        if g.crs.to_epsg() != 3857:
            g = g.to_crs(3857)
    except Exception:
        try:
            g = g.set_crs(3857, allow_override=True)
        except Exception:
            pass
    return g

def _plot_basemap(ax, crs_epsg=3857):
    try:
        ctx.add_basemap(ax, crs=f"EPSG:{crs_epsg}", source=ctx.providers.OpenStreetMap.Mapnik)
    except Exception as e:
        write_to_log(f"Basemap unavailable (offline?): {e}")

def _expand_bounds(bounds, pad_ratio=0.08):
    """Expand (minx, miny, maxx, maxy) by a padding ratio."""
    minx, miny, maxx, maxy = bounds
    dx, dy = maxx - minx, maxy - miny
    if dx <= 0 or dy <= 0:
        return bounds
    pad_x, pad_y = dx * pad_ratio, dy * pad_ratio
    return (minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)

# ---------------- Per-geocode maps from tbl_flat ----------------
def load_tbl_flat(parquet_dir: str) -> gpd.GeoDataFrame:
    fp = os.path.join(parquet_dir, "tbl_flat.parquet")
    if not os.path.exists(fp):
        write_to_log("tbl_flat.parquet not found.")
        return gpd.GeoDataFrame()
    try:
        gdf = gpd.read_parquet(fp)
        if 'geometry' not in gdf.columns or gdf.geometry.is_empty.all():
            write_to_log("tbl_flat has no valid geometry.")
            return gpd.GeoDataFrame()
        return gdf
    except Exception as e:
        write_to_log(f"Failed reading tbl_flat: {e}")
        return gpd.GeoDataFrame()

def compute_fixed_bounds_3857(flat_df: gpd.GeoDataFrame):
    """Compute a single global extent (EPSG:3857) shared by all maps."""
    try:
        poly = flat_df[flat_df.geometry.type.isin(['Polygon','MultiPolygon'])].copy()
        if poly.empty:
            write_to_log("No polygons in tbl_flat to compute fixed bounds; using all geometry.")
            poly = flat_df.copy()
        g3857 = _safe_to_3857(poly)
        if g3857.empty:
            return None
        b = g3857.total_bounds  # (minx, miny, maxx, maxy)
        b = _expand_bounds(b, pad_ratio=0.08)
        return b
    except Exception as e:
        write_to_log(f"Failed computing fixed bounds: {e}")
        return None

def _legend_for_sensitivity(ax, palette: dict, desc: dict):
    """Render a small A–E legend on the current axes."""
    try:
        y0 = 0.02
        dy = 0.06
        for i, code in enumerate(['A','B','C','D','E']):
            y = y0 + i*dy
            ax.add_patch(plt.Rectangle((0.02, y), 0.04, 0.035,
                                       transform=ax.transAxes,
                                       color=palette.get(code, '#BDBDBD'),
                                       ec='white', lw=0.8))
            label = f"{code}: {desc.get(code,'')}".strip() or code
            ax.text(0.07, y+0.005, label, transform=ax.transAxes,
                    fontsize=9, color='white',
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.35, ec="none"))
    except Exception as e:
        write_to_log(f"Sensitivity legend failed: {e}")

def draw_group_map_sensitivity(gdf_group: gpd.GeoDataFrame,
                               group_name: str,
                               palette: dict,
                               desc: dict,
                               out_path: str,
                               fixed_bounds_3857=None):
    """
    Choropleth by sensitivity_code_max using A–E palette.
    Uses a fixed shared extent if provided.
    """
    try:
        g = gdf_group.copy()
        g = g[g.geometry.type.isin(['Polygon','MultiPolygon'])]
        if g.empty:
            write_to_log(f"[{group_name}] No polygon geometries for sensitivity map.")
            return False

        g = _safe_to_3857(g)
        fig, ax = plt.subplots(figsize=(10, 10), dpi=130)
        ax.set_axis_off()

        if fixed_bounds_3857:
            minx, miny, maxx, maxy = fixed_bounds_3857
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)

        facecolors = g['sensitivity_code_max'].map(lambda v: palette.get(str(v).upper(), '#BDBDBD'))
        g.plot(ax=ax, facecolor=facecolors, edgecolor='white', linewidth=0.3, alpha=0.85)

        _plot_basemap(ax, crs_epsg=3857)

        ax.set_title(f"{group_name} – Sensitivity (A–E)", fontsize=14, fontweight='bold')
        _legend_for_sensitivity(ax, palette, desc)

        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        write_to_log(f"Sensitivity map saved: {out_path}")
        return True
    except Exception as e:
        write_to_log(f"Sensitivity map failed for {group_name}: {e}")
        plt.close('all')
        return False

def draw_group_map_envindex(gdf_group: gpd.GeoDataFrame,
                            group_name: str,
                            out_path: str,
                            fixed_bounds_3857=None):
    """
    Continuous choropleth 0–100 for env_index using a yellow→red ramp and colorbar.
    Uses a fixed shared extent if provided.
    """
    try:
        if 'env_index' not in gdf_group.columns:
            write_to_log(f"[{group_name}] env_index missing; skipping env map.")
            return False

        g = gdf_group.copy()
        g = g[g.geometry.type.isin(['Polygon','MultiPolygon'])]
        if g.empty:
            write_to_log(f"[{group_name}] No polygon geometries for env_index map.")
            return False

        g['env_index'] = pd.to_numeric(g['env_index'], errors='coerce')
        g = g[np.isfinite(g['env_index'])]
        if g.empty:
            write_to_log(f"[{group_name}] env_index has no finite values.")
            return False

        g = _safe_to_3857(g)
        fig, ax = plt.subplots(figsize=(10, 10), dpi=130)
        ax.set_axis_off()

        if fixed_bounds_3857:
            minx, miny, maxx, maxy = fixed_bounds_3857
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)

        norm = mcolors.Normalize(vmin=0, vmax=100)
        cmap = cm.get_cmap('YlOrRd')

        colors_arr = cmap(norm(g['env_index'].values))
        g.plot(ax=ax, color=colors_arr, edgecolor='white', linewidth=0.25, alpha=0.95)

        _plot_basemap(ax, crs_epsg=3857)

        ax.set_title(f"{group_name} – Environment index (0–100)", fontsize=14, fontweight='bold')

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label('env_index')

        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        write_to_log(f"Env index map saved: {out_path}")
        return True
    except Exception as e:
        write_to_log(f"Env index map failed for {group_name}: {e}")
        plt.close('all')
        return False

# ---------------- Lines: context map ----------------
def draw_line_context_map(line_row: pd.Series, out_path: str, pad_ratio: float = 0.15):
    """
    Create a small context map for a single line geometry with basemap.
    Highlights the line with a halo to stand out.
    """
    try:
        if 'geometry' not in line_row or line_row['geometry'] is None:
            write_to_log(f"[{line_row.get('name_gis','?')}] Missing geometry; cannot draw context map.")
            return False

        g = gpd.GeoDataFrame([{'name_gis': line_row.get('name_gis','(line)'), 'geometry': line_row['geometry']}],
                             crs=line_row.get('geometry').crs if hasattr(line_row.get('geometry'), 'crs') else None)
        g = _safe_to_3857(g)
        if g.empty or g.geometry.is_empty.all():
            write_to_log(f"[{line_row.get('name_gis','?')}] Invalid geometry after reprojection.")
            return False

        fig, ax = plt.subplots(figsize=(10, 7.5), dpi=130)  # slightly landscape-ish
        ax.set_axis_off()

        # Bounds with padding
        b = g.total_bounds
        minx, miny, maxx, maxy = _expand_bounds(b, pad_ratio)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        _plot_basemap(ax, crs_epsg=3857)

        # Halo + line for visibility
        try:
            g.plot(ax=ax, color='none', edgecolor='white', linewidth=5.0, alpha=0.9)
        except Exception:
            pass
        g.plot(ax=ax, color='none', edgecolor=PRIMARY_HEX, linewidth=2.2, alpha=1.0)

        title_txt = f"Geographical context – {line_row.get('name_gis','(line)')}"
        ax.set_title(title_txt, fontsize=13, fontweight='bold')

        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        write_to_log(f"Line context map saved: {out_path}")
        return True
    except Exception as e:
        write_to_log(f"Context map failed for line: {e}")
        plt.close('all')
        return False

# ---------------- Existing (kept / tidied) ----------------
def get_color_from_code(code, color_codes):
    return color_codes.get(code, "#FFFFFF")

def sort_segments_numerically(segments):
    def extract_number(segment_id):
        m = re.search(r'_(\d+)$', str(segment_id))
        return int(m.group(1)) if m else float('inf')
    s = segments.copy()
    s['sort_key'] = s['segment_id'].apply(extract_number)
    return s.sort_values(by='sort_key').drop(columns=['sort_key'])

def create_line_statistic_image(line_name, sensitivity_series, color_codes, length_m, output_path):
    segment_count = max(1, len(sensitivity_series))
    fig, ax = plt.subplots(figsize=(10, 0.35), dpi=110)
    length_km = (length_m or 0) / 1000.0

    for i, code in enumerate(sensitivity_series):
        color = get_color_from_code(code, color_codes)
        ax.add_patch(plt.Rectangle((i/segment_count, 0), 1/segment_count, 1, color=color))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels([f"0 km", f"{length_km/2:.1f} km", f"{length_km:.1f} km"])
    ax.yaxis.set_visible(False)
    for sp in ['top','bottom','left','right']:
        ax.spines[sp].set_visible(False)

    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

def resize_image(image_path, max_width, max_height):
    """
    NOTE: Inputs here are in ReportLab 'points' if caller passes RL_CM-based values.
    That’s okay — we only need a reasonable downscale to fit the PDF.
    """
    try:
        with PILImage.open(image_path) as img:
            width, height = img.size
            ratio = min(float(max_width) / width, float(max_height) / height)
            if ratio <= 0:
                return
            new_w, new_h = max(1, int(width * ratio)), max(1, int(height * ratio))
            if new_w < width or new_h < height:
                img = img.resize((new_w, new_h), PILImage.LANCZOS)
                img.save(image_path)
    except Exception as e:
        write_to_log(f"Image resize failed for {image_path}: {e}")

def create_sensitivity_summary(sensitivity_series, color_codes, output_path):
    # Count and ensure A–E order
    counts = sensitivity_series.value_counts().reindex(['A','B','C','D','E']).fillna(0)
    total = max(1, len(sensitivity_series))

    fig, (ax_text, ax_bar) = plt.subplots(2, 1, figsize=(10, 1.7), height_ratios=[0.45, 0.55])
    plt.subplots_adjust(hspace=0.15)

    # Text
    parts = [f"{c}: {int(counts[c])} ({counts[c]/total*100:.1f}%)" for c in ['A','B','C','D','E']]
    summary = "Distribution: " + " | ".join(parts)
    ax_text.text(0.5, 0.5, summary, ha='center', va='center')
    ax_text.axis('off')

    # Bar (stacked)
    left = 0
    for c in ['A','B','C','D','E']:
        w = counts[c] / total
        ax_bar.barh(0, w, left=left, color=color_codes.get(c, '#BDBDBD'), edgecolor='white')
        left += w
    ax_bar.set_xlim(0, 1); ax_bar.set_ylim(-0.5, 0.5); ax_bar.axis('off')

    plt.savefig(output_path, bbox_inches='tight', dpi=110)
    plt.close(fig)

def fetch_asset_group_statistics(asset_group_df: gpd.GeoDataFrame, asset_object_df: gpd.GeoDataFrame):
    if asset_group_df.empty:
        return pd.DataFrame(columns=['Sensitivity Code','Sensitivity Description','Number of Asset Objects'])
    if not asset_object_df.empty and 'ref_asset_group' in asset_object_df.columns:
        cnt = asset_object_df.groupby('ref_asset_group').size().rename('asset_objects_nr')
        merged = asset_group_df.merge(cnt, left_on='id', right_index=True, how='left')
    else:
        merged = asset_group_df.copy()
        merged['asset_objects_nr'] = 0
    merged['asset_objects_nr'] = merged['asset_objects_nr'].fillna(0).astype(int)
    out = out = (merged.groupby(['sensitivity_code','sensitivity_description'], dropna=False)['asset_objects_nr']
                      .sum().reset_index())
    out = out.rename(columns={
        'sensitivity_code': 'Sensitivity Code',
        'sensitivity_description': 'Sensitivity Description',
        'asset_objects_nr': 'Number of Asset Objects'
    }).sort_values('Sensitivity Code')
    return out

def format_area(row):
    if row['geometry_type'] in ['Polygon', 'MultiPolygon']:
        try:
            area = float(row['total_area'])
        except Exception:
            return "-"
        if area < 1_000_000:
            return f"{area:.0f} m²"
        return f"{area/1_000_000:.2f} km²"
    return "-"

def calculate_group_statistics(asset_object_df: gpd.GeoDataFrame, asset_group_df: gpd.GeoDataFrame):
    if asset_object_df.empty or asset_group_df.empty:
        return pd.DataFrame(columns=['Title','Code','Description','Type','Total area','# objects'])
    base = asset_group_df.drop(columns=[c for c in ['geometry'] if c in asset_group_df.columns])
    gdf = asset_object_df.merge(base, left_on='ref_asset_group', right_on='id', how='left')
    gdf['geometry_type'] = gdf.geometry.type
    try:
        gdf = gdf.to_crs('ESRI:54009')
    except Exception:
        pass
    polys = gdf[gdf.geometry.type.isin(['Polygon','MultiPolygon'])].copy()
    others = gdf[~gdf.geometry.type.isin(['Polygon','MultiPolygon'])].copy()
    polys.loc[:, 'area'] = polys['geometry'].area
    others.loc[:, 'area'] = np.nan
    gdf2 = pd.concat([polys, others], ignore_index=True)

    stats = gdf2.groupby(
        ['title_fromuser','sensitivity_code','sensitivity_description','geometry_type'],
        dropna=False
    ).agg(total_area=('area','sum'), object_count=('geometry','size')).reset_index()

    stats['total_area'] = stats['total_area'].fillna(0)
    stats['total_area'] = stats.apply(format_area, axis=1)
    stats = stats.rename(columns={
        'title_fromuser':'Title',
        'sensitivity_code':'Code',
        'sensitivity_description':'Description',
        'geometry_type':'Type',
        'object_count':'# objects'
    }).sort_values('Code')
    return stats

def export_to_excel(df, fp):
    if not fp.lower().endswith('.xlsx'):
        fp += '.xlsx'
    df.to_excel(fp, index=False)

def fetch_lines_and_segments(parquet_dir: str):
    lines_pq    = os.path.join(parquet_dir, "tbl_lines.parquet")
    segments_pq = os.path.join(parquet_dir, "tbl_segment_flat.parquet")
    try:
        lines_df = gpd.read_parquet(lines_pq) if os.path.exists(lines_pq) else gpd.GeoDataFrame()
    except Exception:
        lines_df = gpd.GeoDataFrame()
    try:
        segments_df = gpd.read_parquet(segments_pq) if os.path.exists(segments_pq) else gpd.GeoDataFrame()
    except Exception:
        segments_df = gpd.GeoDataFrame()
    return lines_df, segments_df

# ---------------- PDF helpers ----------------
def line_up_to_pdf(order_list):
    elements = []
    styles = getSampleStyleSheet()

    # Build heading styles with steel blue chrome
    primary = HexColor(PRIMARY_HEX)
    light_primary = HexColor(LIGHT_PRIMARY_HEX)
    heading_styles = {
        'H1': ParagraphStyle(
            name='H1', fontSize=20, leading=24, spaceBefore=6, spaceAfter=8,
            alignment=TA_CENTER, textColor=colors.black
        ),
        'H2Bar': ParagraphStyle(
            name='H2Bar', fontSize=16, leading=20, spaceBefore=10, spaceAfter=6,
            textColor=colors.white, backColor=light_primary, leftIndent=0, rightIndent=0
        ),
        'H3': ParagraphStyle(
            name='H3', fontSize=13, leading=16, spaceBefore=4, spaceAfter=4,
            textColor=primary
        ),
        'Body': styles['Normal']
    }

    max_image_width = 16 * RL_CM
    max_image_height = 24 * RL_CM

    def _add_heading(level, text):
        if level == 1:
            elements.append(Paragraph(text, heading_styles['H1']))
            elements.append(HRFlowable(width="100%", thickness=0.8, color=primary))
        elif level == 2:
            elements.append(Paragraph(text, heading_styles['H2Bar']))
        else:
            elements.append(Paragraph(text, heading_styles['H3']))
        elements.append(Spacer(1, 6))

    for item in order_list:
        itype, ival = item

        if itype == 'text':
            if os.path.isfile(ival):
                with open(ival, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                text = str(ival)
            text = text.replace("\n", "<br/>")
            elements.append(Paragraph(text, heading_styles['Body']))
            elements.append(Spacer(1, 8))

        elif itype == 'image':
            # 'ival' is a tuple: (heading_str, path)
            heading_str, path = ival
            _add_heading(3, heading_str)     # heading directly above image
            resize_image(path, max_image_width, max_image_height)
            img = Image(path)
            img.hAlign = 'CENTER'
            elements.append(img)
            elements.append(Spacer(1, 6))

        elif itype == 'table':
            df = pd.read_excel(ival)
            data = [df.columns.tolist()] + df.values.tolist()
            table = Table(data, repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), primary),
                ('TEXTCOLOR',  (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('ALIGN',      (0, 0), (-1,-1), 'CENTER'),
                ('GRID',       (0, 0), (-1,-1), 0.5, colors.HexColor("#A9A9A9")),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.whitesmoke, colors.Color(0.97,0.97,0.97)])
            ]))
            elements.append(table)
            elements.append(Spacer(1, 8))

        elif itype.startswith('heading'):
            level = int(itype[-2])
            _add_heading(level, ival)

        elif itype == 'rule':
            elements.append(HRFlowable(width="100%", thickness=0.8, color=primary))
            elements.append(Spacer(1, 6))

        elif itype == 'spacer':
            lines = ival if ival else 1
            elements.append(Spacer(1, lines * 10))

        elif itype == 'new_page':
            elements.append(PageBreak())

    return elements

def compile_pdf(output_pdf, elements):
    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    primary = HexColor(PRIMARY_HEX)

    def add_header_footer(canvas, doc_obj):
        width, height = A4
        # Top color band (steel blue)
        canvas.saveState()
        canvas.setFillColor(primary)
        canvas.rect(0, height - 16, width, 16, fill=1, stroke=0)
        canvas.setFont('Helvetica-Bold', 10)
        canvas.setFillColor(colors.white)
        canvas.drawString(24, height - 12, "MESA – Report")
        canvas.restoreState()

        # Footer with page number and a thin rule
        canvas.saveState()
        canvas.setStrokeColor(primary)
        canvas.setLineWidth(0.5)
        canvas.line(24, 36, width - 24, 36)
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(colors.gray)
        page_num = canvas.getPageNumber()
        canvas.drawRightString(width - 24, 24, f"Page {page_num}")
        canvas.restoreState()

    doc.build(elements, onFirstPage=add_header_footer, onLaterPages=add_header_footer)

def set_progress(pct: float, message: str | None = None):
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

# ---------------- Core: generate report ----------------
def generate_report(original_working_directory: str,
                    config_file: str,
                    palette_A2E: dict,
                    desc_A2E: dict):
    try:
        set_progress(3, "Initializing report generation …")
        gpq_dir   = os.path.join(original_working_directory, "output", "geoparquet")
        cfg       = read_config(config_file)
        tmp_dir   = os.path.join(original_working_directory, 'output', 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        set_progress(6, "Paths prepared.")

        # Load key tables
        asset_object_pq = os.path.join(gpq_dir, "tbl_asset_object.parquet")
        asset_group_pq  = os.path.join(gpq_dir, "tbl_asset_group.parquet")
        flat_pq         = os.path.join(gpq_dir, "tbl_flat.parquet")

        try: asset_objects_df = gpd.read_parquet(asset_object_pq) if os.path.exists(asset_object_pq) else gpd.GeoDataFrame()
        except Exception: asset_objects_df = gpd.GeoDataFrame()
        try: asset_groups_df  = gpd.read_parquet(asset_group_pq)  if os.path.exists(asset_group_pq)  else gpd.GeoDataFrame()
        except Exception: asset_groups_df  = gpd.GeoDataFrame()

        set_progress(10, "Loaded asset tables.")

        # ---- Assets statistics ----
        write_to_log("Computing asset object statistics …")
        group_stats_df = calculate_group_statistics(asset_objects_df, asset_groups_df)
        object_stats_xlsx = os.path.join(tmp_dir, 'asset_object_statistics.xlsx')
        export_to_excel(group_stats_df, object_stats_xlsx)

        ag_stats_df = fetch_asset_group_statistics(asset_groups_df, asset_objects_df)
        ag_stats_xlsx = os.path.join(tmp_dir, 'asset_group_statistics.xlsx')
        export_to_excel(ag_stats_df, ag_stats_xlsx)
        set_progress(22, "Asset stats ready.")

        # ---- Lines & segments (grouped per line with context map first) ----
        lines_df, segments_df = fetch_lines_and_segments(gpq_dir)
        pages_lines = []
        if (not lines_df.empty and not segments_df.empty and
            {'name_gis','segment_id','sensitivity_code_max','sensitivity_code_min'}.issubset(segments_df.columns) and
            'length_m' in lines_df.columns and 'name_gis' in lines_df.columns and 'geometry' in lines_df.columns):

            log_data = []
            for _, line in lines_df.iterrows():
                ln = line['name_gis']
                length_m = float(line.get('length_m', 0) or 0)
                length_km = length_m / 1000.0
                segs = sort_segments_numerically(segments_df[segments_df['name_gis']==ln])

                # Context map
                context_img = os.path.join(tmp_dir, f"{ln}_context.png")
                draw_line_context_map(line, context_img, pad_ratio=0.15)

                # summaries
                max_stats_img = os.path.join(tmp_dir, f"{ln}_max_dist.png")
                min_stats_img = os.path.join(tmp_dir, f"{ln}_min_dist.png")
                create_sensitivity_summary(segs['sensitivity_code_max'], palette_A2E, max_stats_img)
                create_sensitivity_summary(segs['sensitivity_code_min'], palette_A2E, min_stats_img)

                # ribbons
                max_img = os.path.join(tmp_dir, f"{ln}_max_ribbon.png")
                min_img = os.path.join(tmp_dir, f"{ln}_min_ribbon.png")
                create_line_statistic_image(ln, segs['sensitivity_code_max'], palette_A2E, length_m, max_img)
                create_line_statistic_image(ln, segs['sensitivity_code_min'], palette_A2E, length_m, min_img)

                # GROUPED presentation per line: heading + paragraph + context map, then the four visuals
                pages_lines += [
                    ('heading(2)', f"Line: {ln}"),
                    ('text', f"This section summarizes sensitivity along the line <b>{ln}</b> "
                             f"(total length <b>{length_km:.2f} km</b>, segments: <b>{len(segs)}</b>). "
                             "We first provide a geographical context map, then distributions (stacked bars) "
                             "and linear ribbons for maximum and minimum sensitivity."),
                    ('image', ("Geographical context", context_img)),
                    ('image', ("Maximum sensitivity – distribution", max_stats_img)),
                    ('image', ("Maximum sensitivity – along line", max_img)),
                    ('new_page', None),

                    ('heading(2)', f"Line: {ln} (continued)"),
                    ('image', ("Minimum sensitivity – distribution", min_stats_img)),
                    ('image', ("Minimum sensitivity – along line", min_img)),
                    ('new_page', None),
                ]

                for _, seg in segs.iterrows():
                    log_data.append({
                        'line_name': ln,
                        'segment_id': seg['segment_id'],
                        'sensitivity_code_max': seg['sensitivity_code_max'],
                        'sensitivity_code_min': seg['sensitivity_code_min']
                    })

            if log_data:
                log_df = pd.DataFrame(log_data)
                log_xlsx = os.path.join(tmp_dir, 'line_segment_log.xlsx')
                export_to_excel(log_df, log_xlsx)
                write_to_log(f"Segments log exported to {log_xlsx}")
        else:
            write_to_log("Skipping line/segment pages (missing/empty).")
        set_progress(36, "Lines/segments processed.")

        # ---- Per-geocode maps from tbl_flat (same scale, one page per map) ----
        flat_df = load_tbl_flat(gpq_dir)
        geocode_pages = []
        if not flat_df.empty and 'name_gis_geocodegroup' in flat_df.columns:
            groups = (flat_df['name_gis_geocodegroup']
                      .astype('string')
                      .dropna().unique().tolist())
            groups = sorted(groups)
            write_to_log(f"Found geocode categories: {groups}")

            # Compute one shared extent for all maps:
            fixed_bounds_3857 = compute_fixed_bounds_3857(flat_df)

            done = 0
            for gname in groups:
                sub = flat_df[flat_df['name_gis_geocodegroup'] == gname].copy()

                # Sensitivity map
                sens_png = os.path.join(tmp_dir, f"map_sensitivity_{re.sub(r'[^A-Za-z0-9_-]+','_',gname)}.png")
                ok_sens = draw_group_map_sensitivity(sub, gname, palette_A2E, desc_A2E, sens_png, fixed_bounds_3857)

                # Env index map
                env_png  = os.path.join(tmp_dir, f"map_envindex_{re.sub(r'[^A-Za-z0-9_-]+','_',gname)}.png")
                ok_env   = draw_group_map_envindex(sub, gname, env_png, fixed_bounds_3857)

                # One map per page, heading right above
                if ok_sens:
                    geocode_pages += [
                        ('heading(2)', f"Geocode group: {gname}"),
                        ('image', (f"Sensitivity (A–E palette): {gname}", sens_png)),
                        ('new_page', None),
                    ]
                if ok_env:
                    geocode_pages += [
                        ('heading(2)', f"Geocode group: {gname}"),
                        ('image', (f"Environment index (0–100): {gname}", env_png)),
                        ('new_page', None),
                    ]

                done += 1
                set_progress(36 + int(30 * done/max(1,len(groups))), f"Rendered maps for '{gname}'")

            if geocode_pages:
                write_to_log("Per-geocode maps created.")
        else:
            write_to_log("tbl_flat missing or no 'name_gis_geocodegroup'; skipping per-geocode maps.")
        set_progress(70, "Per-geocode maps completed.")

        # ---- Compose PDF ----
        timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
        order_list = [
            ('heading(1)', "MESA report"),
            ('text', f"Timestamp: {timestamp}"),
            ('rule', None),

            ('heading(2)', "Contents"),
            ('text', "- Assets overview & statistics"
                     "<br/>- Per-geocode maps (Sensitivity & Environment Index, shared scale)"
                     "<br/>- Lines & segments (grouped by line, with context map)"),
            ('new_page', None),

            ('heading(2)', "Assets – overview"),
            ('text', "Asset objects by group (count and area, where applicable)."),
            ('table', os.path.join(tmp_dir, 'asset_object_statistics.xlsx')),
            ('spacer', 1),
            ('text', "Asset groups – sensitivity distribution (count of objects)."),
            ('table', os.path.join(tmp_dir, 'asset_group_statistics.xlsx')),
            ('new_page', None),

            ('heading(2)', "Per-geocode maps"),
            ('text',
             "About geocode categories: <br/><br/>"
             "<b>basic_mosaic</b> is the baseline geocoding used for area statistics in MESA. "
             "It is a consistent partition intended for repeatable reporting and comparison across runs. "
             "In contrast, <b>H3</b> geocodes are hexagonal cells from Uber’s H3 hierarchical indexing system. "
             "H3 provides multi-resolution grids (fine to coarse) that are excellent for scalable analysis and "
             "visualization. In this report, we render each available geocode category independently. "
             "All maps below share the same cartographic scale/extent for easier visual comparison."),
            ('spacer', 1),
            ('rule', None),
        ]

        order_list.extend(geocode_pages)

        if pages_lines:
            order_list.append(('heading(2)', "Lines and segments"))
            order_list.append(('text', "For each line we provide a short paragraph and a geographical context map, "
                                       "followed by four visuals: maximum sensitivity distribution, "
                                       "maximum sensitivity ribbon along the line, minimum sensitivity distribution, "
                                       "and minimum sensitivity ribbon along the line."))
            order_list.append(('rule', None))
            order_list.append(('new_page', None))
            order_list.extend(pages_lines)

        set_progress(86, "Composing PDF …")
        elements = line_up_to_pdf(order_list)

        ts_pdf = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
        output_pdf = os.path.join(original_working_directory, f'output/MESA-report_{ts_pdf}.pdf')
        compile_pdf(output_pdf, elements)
        set_progress(100, "Report completed.")

        global last_report_path
        last_report_path = output_pdf
        write_to_log(f"PDF report created: {output_pdf}")

        try:
            if link_var:
                link_var.set("Open report folder")
        except Exception:
            pass

    except Exception as e:
        write_to_log(f"ERROR during report generation: {e}")
        set_progress(100, "Report failed.")

# ---------------- GUI runner ----------------
def _start_report_thread(base_dir, config_file, palette, desc):
    threading.Thread(
        target=generate_report,
        args=(base_dir, config_file, palette, desc),
        daemon=True
    ).start()

def launch_gui(original_working_directory: str, config_file: str, palette: dict, desc: dict, theme: str):
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
              command=lambda: _start_report_thread(original_working_directory, config_file, palette, desc)
              ).grid(row=0, column=0, padx=6)
    tb.Button(btn_frame, text="Exit", bootstyle=WARNING, command=root.destroy).grid(row=0, column=1, padx=6)

    # Live link to report folder
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
    link_label = tk.Label(root, textvariable=link_var, fg="#4ea3ff", cursor="hand2",
                          font=("Segoe UI", 10, "underline"))
    link_label.pack(pady=(2, 8))
    link_label.bind("<Button-1>", open_report_folder)

    write_to_log(f"Working directory: {original_working_directory}")
    write_to_log("Ready. Press 'Generate report' to start.")
    root.mainloop()

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Presentation report (GeoParquet per geocode, same-scale maps, line context)')
    parser.add_argument('--original_working_directory', required=False, help='Path to running folder')
    parser.add_argument('--no-gui', action='store_true', help='Run directly without GUI')
    args = parser.parse_args()

    original_working_directory = args.original_working_directory
    if not original_working_directory:
        original_working_directory = os.getcwd()
        if os.path.basename(original_working_directory).lower() == "system":
            original_working_directory = os.path.abspath(os.path.join(original_working_directory, ".."))

    config_file = os.path.join(original_working_directory, "system", "config.ini")
    cfg = read_config(config_file)
    theme = cfg['DEFAULT'].get('ttk_bootstrap_theme', 'flatly')

    # Sensitivity palette + descriptions from config (A–E)
    palette_A2E, desc_A2E = read_sensitivity_palette_and_desc(config_file)

    # Allow optional override from config for brand color; default to steel blue
    PRIMARY_HEX = cfg['DEFAULT'].get('ui_primary_color', PRIMARY_HEX).strip() or PRIMARY_HEX
    # Slightly lighter variant for section bars
    try:
        # simple lighten: move 30% toward white
        def _lighten(hexcol, amt=0.30):
            hexcol = hexcol.lstrip('#')
            r = int(hexcol[0:2],16); g = int(hexcol[2:4],16); b = int(hexcol[4:6],16)
            r = int(r + (255 - r)*amt); g = int(g + (255 - g)*amt); b = int(b + (255 - b)*amt)
            return f"#{r:02x}{g:02x}{b:02x}"
        LIGHT_PRIMARY_HEX = _lighten(PRIMARY_HEX, 0.30)
    except Exception:
        LIGHT_PRIMARY_HEX = LIGHT_PRIMARY_HEX  # keep default

    if args.no_gui:
        generate_report(original_working_directory, config_file, palette_A2E, desc_A2E)
    else:
        launch_gui(original_working_directory, config_file, palette_A2E, desc_A2E, theme)
