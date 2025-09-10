#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

import geopandas as gpd
import pandas as pd
import configparser
import subprocess
import datetime
import argparse
import math
from shapely.geometry import box, LineString, Point, Polygon, MultiLineString, MultiPolygon
from shapely.ops import unary_union, split, polygonize, linemerge
from shapely.geometry import mapping
from fiona.crs import from_epsg
from shapely.ops import transform
import pyproj
from functools import partial
import os
import numpy as np
import tkinter as tk
import tkinter.scrolledtext as scrolledtext
import ttkbootstrap as ttk
from ttkbootstrap import Style
from pathlib import Path

# -------------------------------
# Config helpers
# -------------------------------
def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config

def read_config_classification(file_name):
    """
    Expects sections like:
    [A]
    range = 1-3
    description = Very low
    """
    cfg = configparser.ConfigParser()
    cfg.read(file_name)
    classification = {}
    for section in cfg.sections():
        if 'range' in cfg[section]:
            try:
                start, end = map(int, str(cfg[section]['range']).split('-'))
                classification[section] = {
                    'range': range(start, end + 1),
                    'description': cfg[section].get('description', '')
                }
            except Exception:
                # Ignore malformed ranges
                continue
    return classification

# -------------------------------
# GUI logging
# -------------------------------
def log_to_gui(log_widget, message):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted_message = f"{timestamp} - {message}"
    log_widget.insert(tk.END, formatted_message + "\n")
    log_widget.see(tk.END)
    log_destination_file = os.path.join(original_working_directory, "log.txt")
    with open(log_destination_file, "a") as log_file:
        log_file.write(formatted_message + "\n")
    # Safe root update (root may not yet exist in early calls)
    try:
        if 'root' in globals() and root.winfo_exists():
            root.update_idletasks()
    except Exception:
        pass

def update_progress(new_value):
    progress_var.set(new_value)
    progress_label.config(text=f"{int(new_value)}%")

def increment_stat_value(config_file, stat_name, increment_value):
    if not os.path.isfile(config_file):
        log_to_gui(log_widget, f"Configuration file {config_file} not found.")
        return
    with open(config_file, 'r') as file:
        lines = file.readlines()
    updated = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f'{stat_name} ='):
            parts = line.split('=')
            if len(parts) == 2:
                current_value = parts[1].strip()
                try:
                    new_value = int(current_value) + increment_value
                    lines[i] = f"{stat_name} = {new_value}\n"
                    updated = True
                    break
                except ValueError:
                    log_to_gui(log_widget, f"Error: Current value of {stat_name} is not an integer.")
                    return
    if updated:
        with open(config_file, 'w') as file:
            file.writelines(lines)

# -------------------------------
# IO helpers
# -------------------------------
def base_dir() -> Path:
    bd = Path(original_working_directory or os.getcwd())
    if bd.name.lower() == "system":
        return bd.parent
    return bd

def gpq_dir() -> Path:
    out = base_dir() / "output" / "geoparquet"
    out.mkdir(parents=True, exist_ok=True)
    return out

def parquet_path(name: str) -> Path:
    return gpq_dir() / f"{name}.parquet"

def read_parquet_or_none(name: str):
    p = parquet_path(name)
    if not p.exists():
        return None
    try:
        return gpd.read_parquet(p)
    except Exception as e:
        log_to_gui(log_widget, f"Failed reading {p.name}: {e}")
        return None

def write_parquet(name: str, gdf: gpd.GeoDataFrame):
    p = parquet_path(name)
    try:
        gdf.to_parquet(p, index=False)
        log_to_gui(log_widget, f"Saved {p.name} (rows={len(gdf)})")
    except Exception as e:
        log_to_gui(log_widget, f"Parquet write failed {p.name}: {e}")

def optional_write_gpkg(gdf: gpd.GeoDataFrame, layer: str):
    """
    Best-effort legacy write if gpkg exists. Silent if not.
    """
    if not os.path.exists(gpkg_file):
        return
    try:
        gdf.to_file(gpkg_file, layer=layer, driver="GPKG")
    except Exception as e:
        log_to_gui(log_widget, f"GPKG write skipped ({layer}): {e}")

def load_lines_table(gpkg_file):
    """
    Returns GeoDataFrame of lines (tbl_lines) from Parquet first,
    falling back to GeoPackage if it exists; otherwise None.
    """
    gdf = read_parquet_or_none("tbl_lines")
    if gdf is not None and not gdf.empty:
        return gdf
    if os.path.exists(gpkg_file):
        try:
            gdf = gpd.read_file(gpkg_file, layer='tbl_lines')
            if not gdf.empty:
                write_parquet("tbl_lines", gdf)  # cache for future
                return gdf
        except Exception:
            return None
    return None

# -------------------------------
# Geometry building
# -------------------------------
def create_lines_table_and_lines(gpkg_file, log_widget):
    # Source extent from geocode group parquet (preferred)
    geocode_group = read_parquet_or_none("tbl_geocode_group")
    if geocode_group is None or geocode_group.empty:
        # Fallback to gpkg if present
        if os.path.exists(gpkg_file):
            try:
                geocode_group = gpd.read_file(gpkg_file, layer='tbl_geocode_group')
            except Exception:
                geocode_group = None
    if geocode_group is None or geocode_group.empty:
        log_to_gui(log_widget, "Cannot derive extent (no geocode groups).")
        return
    log_to_gui(log_widget, "Creating template lines (parquet).")
    minx, miny, maxx, maxy = geocode_group.total_bounds
    lines = []
    for _ in range(3):
        sx = np.random.uniform(minx, maxx)
        sy = np.random.uniform(miny, maxy)
        ex = np.random.uniform(minx, maxx)
        ey = np.random.uniform(miny, maxy)
        lines.append(LineString([(sx, sy), (ex, ey)]))
    gdf_lines = gpd.GeoDataFrame({
        'name_gis': [f'line_{i:03d}' for i in range(1, 4)],
        'name_user': [f'line_{i:03d}' for i in range(1, 4)],
        'segment_length': [15, 30, 10],
        'segment_width': [1000, 20000, 5000],
        'description': ['auto line', 'auto line', 'auto line'],
        'geometry': lines
    }, crs=geocode_group.crs or workingprojection_epsg)
    gdf_lines = gdf_lines.set_crs(workingprojection_epsg, allow_override=True)
    write_parquet("tbl_lines", gdf_lines)
    optional_write_gpkg(gdf_lines, "tbl_lines")

def process_and_buffer_lines(gpkg_file, log_widget):
    crs        = workingprojection_epsg
    target_crs = "EPSG:4087"
    lines_df   = load_lines_table(gpkg_file)

    if lines_df is None:
        log_to_gui(log_widget, "No lines found; creating defaults.")
        create_lines_table_and_lines(gpkg_file, log_widget)
        lines_df = load_lines_table(gpkg_file)
        if lines_df is None:
            log_to_gui(log_widget, "Aborting: lines still missing.")
            return

    buffered_records = []
    for idx, row in lines_df.iterrows():
        try:
            geom = row.geometry
            seg_len  = int(row['segment_length'])
            seg_w    = int(row['segment_width'])
            name_gis = row['name_gis']
            name_usr = row['name_user']
            desc     = row.get('description', '')

            log_to_gui(log_widget, f"Buffering {name_gis}")
            tmp = gpd.GeoDataFrame([{'geometry': geom}], geometry='geometry', crs=crs).to_crs(target_crs)
            tmp['geometry'] = tmp.buffer(seg_w, cap_style=2)
            back = tmp.to_crs(crs)
            gbuf = back.iloc[0].geometry
            if not isinstance(gbuf, (Polygon, MultiPolygon)):
                log_to_gui(log_widget, f"Unexpected buffered geom type: {type(gbuf)}")
            buffered_records.append({
                'fid': idx,
                'name_gis': name_gis,
                'name_user': name_usr,
                'segment_length': seg_len,
                'segment_width': seg_w,
                'description': desc,
                'geometry': gbuf
            })
        except Exception as e:
            log_to_gui(log_widget, f"Line {idx} failed: {e}")

    if not buffered_records:
        log_to_gui(log_widget, "No buffered lines produced.")
        return

    buffered_gdf = gpd.GeoDataFrame(buffered_records, geometry='geometry', crs=crs)
    write_parquet("tbl_lines_buffered", buffered_gdf)
    optional_write_gpkg(buffered_gdf, "tbl_lines_buffered")
    log_to_gui(log_widget, "Buffered lines ready (parquet).")

def create_perpendicular_lines(line_input, segment_width, segment_length):
    if not isinstance(line_input, LineString):
        raise ValueError("line_input must be a LineString")
    segment_width  = float(segment_width)
    segment_length = float(segment_length)

    transformer_to_4087 = pyproj.Transformer.from_crs(workingprojection_epsg, "EPSG:4087", always_xy=True)
    transformer_back    = pyproj.Transformer.from_crs("EPSG:4087", workingprojection_epsg, always_xy=True)

    line_trans = transform(transformer_to_4087.transform, line_input)
    full_len = line_trans.length
    num_segments = math.ceil(full_len / segment_length)

    perpendicular_lines = []
    for i in range(num_segments + 1):
        d = min(i * segment_length, full_len)
        point = line_trans.interpolate(d)

        if d < segment_width:
            seg = LineString([line_trans.interpolate(0), line_trans.interpolate(segment_width)])
        elif d > full_len - segment_width:
            seg = LineString([line_trans.interpolate(full_len - segment_width), line_trans.interpolate(full_len)])
        else:
            seg = LineString([line_trans.interpolate(d - segment_width/2), line_trans.interpolate(d + segment_width/2)])

        dx = seg.coords[1][0] - seg.coords[0][0]
        dy = seg.coords[1][1] - seg.coords[0][1]
        angle = math.atan2(-dx, dy) if dx != 0 else (math.pi/2 if dy > 0 else -math.pi/2)
        length = (segment_width / 2) * 3
        dxp = math.cos(angle) * length
        dyp = math.sin(angle) * length

        p1 = Point(point.x - dxp, point.y - dyp)
        p2 = Point(point.x + dxp, point.y + dyp)

        p1b = transform(transformer_back.transform, p1)
        p2b = transform(transformer_back.transform, p2)
        perpendicular_lines.append(LineString([p1b, p2b]))

    return MultiLineString(perpendicular_lines)

def cut_into_segments(perpendicular_lines, buffered_line_geometry):
    if not isinstance(buffered_line_geometry, Polygon):
        raise TypeError("Second call: The buffered_line_geometry must be a Polygon.")
    if not isinstance(perpendicular_lines, MultiLineString):
        raise TypeError("The perpendicular_lines must be a MultiLineString.")

    line_list = [line for line in perpendicular_lines.geoms]
    combined_lines = unary_union([buffered_line_geometry.boundary] + line_list)
    result_polygons = list(polygonize(combined_lines))
    return gpd.GeoDataFrame(geometry=result_polygons)

def create_segments_from_buffered_lines(gpkg_file, log_widget):
    lines_df = load_lines_table(gpkg_file)
    if lines_df is None:
        log_to_gui(log_widget, "No lines for segment creation.")
        return
    buffered = read_parquet_or_none("tbl_lines_buffered")
    if (buffered is None or buffered.empty) and os.path.exists(gpkg_file):
        try:
            buffered = gpd.read_file(gpkg_file, layer='tbl_lines_buffered')
        except Exception:
            buffered = None
    if buffered is None or buffered.empty:
        log_to_gui(log_widget, "No buffered lines found; run buffering first.")
        return

    all_segments = []
    counter = {}
    for _, row in lines_df.iterrows():
        name_gis = row['name_gis']
        seg_w    = row['segment_width']
        seg_l    = row['segment_length']
        geom     = row.geometry
        if name_gis not in counter:
            counter[name_gis] = 1
        try:
            perp = create_perpendicular_lines(geom, seg_w, seg_l)
        except Exception as e:
            log_to_gui(log_widget, f"Perpendicular gen failed {name_gis}: {e}")
            continue
        blines = buffered[buffered['name_gis'] == name_gis]
        if blines.empty:
            continue
        log_to_gui(log_widget, f"Segmenting {name_gis}")
        for _, brow in blines.iterrows():
            bgeom = brow.geometry
            if not isinstance(bgeom, Polygon):
                continue
            segs = cut_into_segments(perp, bgeom)
            segs = segs[segs.is_valid]
            if segs.empty:
                continue
            segs['segment_id'] = [f"{name_gis}_{counter[name_gis]+i}" for i in range(len(segs))]
            counter[name_gis] += len(segs)
            segs['name_gis'] = name_gis
            segs['name_user'] = row['name_user']
            segs['segment_length'] = seg_l
            all_segments.append(segs)

    if not all_segments:
        log_to_gui(log_widget, "No segments produced.")
        return
    seg_all = gpd.GeoDataFrame(pd.concat(all_segments, ignore_index=True),
                               geometry='geometry', crs=lines_df.crs)
    write_parquet("tbl_segments", seg_all)
    optional_write_gpkg(seg_all, "tbl_segments")
    log_to_gui(log_widget, "Segments saved (parquet).")

# -------------------------------
# Intersections & stacks
# -------------------------------
def intersection_with_geocode_data(asset_df, segment_df, geom_type, log_widget):
    log_to_gui(log_widget, f"Processing {geom_type} intersections")
    asset_filtered = asset_df[asset_df.geometry.geom_type == geom_type]
    if asset_filtered.empty:
        return gpd.GeoDataFrame()
    return gpd.sjoin(segment_df, asset_filtered, how='inner', predicate='intersects')

def intersection_with_segments(asset_data, segment_data, log_widget):
    try:
        return gpd.overlay(asset_data, segment_data, how='intersection')
    except Exception as e:
        log_to_gui(log_widget, f"Error in intersection: {str(e)}")
        return pd.DataFrame()

# -------------------------------
# Classification (in-memory) so Parquet gets the fields too
# -------------------------------
def _coerce_int_or_none(v):
    try:
        if pd.isna(v):
            return None
        return int(round(float(v)))
    except Exception:
        return None

def apply_classification_to_gdf(gdf, column_name, classes_dict, code_suffix=""):
    """
    Adds <base>_code_<suffix> and <base>_description_<suffix> to gdf,
    where base is column_name without trailing _suffix if present.
    """
    base_name, *suffix = column_name.rsplit('_', 1)
    suffix = suffix[0] if suffix else ''
    final_suffix = suffix if not code_suffix else code_suffix  # allow override

    new_code_col = f"{base_name}_code_{final_suffix}" if final_suffix else f"{base_name}_code"
    new_desc_col = f"{base_name}_description_{final_suffix}" if final_suffix else f"{base_name}_description"

    if not classes_dict:
        # No classes—default Unknown
        gdf[new_code_col] = "Unknown"
        gdf[new_desc_col] = "No description available"
        return gdf, new_code_col, new_desc_col

    # Normalize: ensure dict like {'A': {'range': range(..), 'description': '...'}, ...}
    def classify_value(v):
        iv = _coerce_int_or_none(v)
        if iv is None:
            return "Unknown", "No description available"
        for label, info in classes_dict.items():
            rng = info.get('range', range(0))
            if iv in rng:
                return label, info.get('description', '')
        return "Unknown", "No description available"

    codes, descs = zip(*gdf[column_name].apply(classify_value))
    gdf[new_code_col] = list(codes)
    gdf[new_desc_col] = list(descs)
    return gdf, new_code_col, new_desc_col

# -------------------------------
# Build stacked & flat, with classification persisted to Parquet
# -------------------------------
def build_stacked_data(gpkg_file, log_widget):
    log_to_gui(log_widget, "Building tbl_segment_stacked (parquet)…")
    update_progress(65)

    asset_data = read_parquet_or_none("tbl_asset_object")
    group_data = read_parquet_or_none("tbl_asset_group")
    segments   = read_parquet_or_none("tbl_segments")

    if asset_data is None or asset_data.empty or segments is None or segments.empty:
        log_to_gui(log_widget, "Missing assets or segments; cannot build segment stacked.")
        return

    if group_data is not None and not group_data.empty:
        merge_cols = [c for c in [
            'id','name_gis_assetgroup','total_asset_objects',
            'importance','susceptibility','sensitivity',
            'sensitivity_code','sensitivity_description'
        ] if c in group_data.columns]
        asset_data = asset_data.merge(group_data[merge_cols],
                                      left_on='ref_asset_group', right_on='id', how='left')

    segments = segments.set_crs(workingprojection_epsg, allow_override=True)
    asset_data = asset_data.set_crs(workingprojection_epsg, allow_override=True)

    # Per geometry type intersections
    parts = []
    for gt in asset_data.geometry.geom_type.unique():
        parts.append(intersection_with_geocode_data(asset_data, segments, gt, log_widget))
    stacked = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if stacked.empty:
        log_to_gui(log_widget, "No segment intersections.")
        return

    stacked = gpd.GeoDataFrame(stacked, geometry='geometry', crs=workingprojection_epsg)
    stacked.reset_index(drop=True, inplace=True)
    stacked['fid'] = stacked.index

    # Classification
    classes = read_config_classification(config_file)
    if 'sensitivity' in stacked.columns:
        stacked, codecol, desccol = apply_classification_to_gdf(stacked, 'sensitivity', classes)
        if codecol != 'sensitivity_code':
            stacked.rename(columns={codecol: 'sensitivity_code'}, inplace=True)
        if desccol != 'sensitivity_description':
            stacked.rename(columns={desccol: 'sensitivity_description'}, inplace=True)

    write_parquet("tbl_segment_stacked", stacked)
    optional_write_gpkg(stacked, "tbl_segment_stacked")
    update_progress(75)
    log_to_gui(log_widget, f"tbl_segment_stacked rows: {len(stacked)}")

def build_flat_data(gpkg_file, log_widget):
    log_to_gui(log_widget, "Building tbl_segment_flat (parquet)…")
    stacked = read_parquet_or_none("tbl_segment_stacked")
    if stacked is None or stacked.empty:
        log_to_gui(log_widget, "No stacked segment data.")
        return
    update_progress(80)

    agg = {
        'importance': ['min','max'],
        'sensitivity': ['min','max'],
        'susceptibility': ['min','max'],
        'name_gis': 'first',
        'segment_id': 'first',
        'geometry': 'first'
    }
    flat = stacked.groupby('segment_id').agg(agg)
    flat.columns = ['_'.join(c).strip() for c in flat.columns]
    rename_map = {'name_gis_first':'name_gis','geometry_first':'geometry'}
    flat.rename(columns=rename_map, inplace=True)
    flat.reset_index(inplace=True)
    if 'segment_id_first' in flat.columns:
        flat.drop(columns=['segment_id_first'], inplace=True)

    flat = gpd.GeoDataFrame(flat, geometry='geometry', crs=stacked.crs)

    classes = read_config_classification(config_file)
    if 'sensitivity_min_min' in flat.columns:
        flat['sensitivity_min'] = flat.get('sensitivity_min', flat['sensitivity_min_min'])
        flat.drop(columns=['sensitivity_min_min'], inplace=True)
    if 'sensitivity_max_max' in flat.columns:
        flat['sensitivity_max'] = flat.get('sensitivity_max', flat['sensitivity_max_max'])
        flat.drop(columns=['sensitivity_max_max'], inplace=True)

    if 'sensitivity_min' in flat.columns:
        flat, cmin, dmin = apply_classification_to_gdf(flat, 'sensitivity_min', classes, code_suffix='min')
        if cmin != 'sensitivity_code_min':
            flat.rename(columns={cmin: 'sensitivity_code_min'}, inplace=True)
        if dmin != 'sensitivity_description_min':
            flat.rename(columns={dmin: 'sensitivity_description_min'}, inplace=True)
    if 'sensitivity_max' in flat.columns:
        flat, cmax, dmax = apply_classification_to_gdf(flat, 'sensitivity_max', classes, code_suffix='max')
        if cmax != 'sensitivity_code_max':
            flat.rename(columns={cmax: 'sensitivity_code_max'}, inplace=True)
        if dmax != 'sensitivity_description_max':
            flat.rename(columns={dmax: 'sensitivity_description_max'}, inplace=True)

    write_parquet("tbl_segment_flat", flat)
    optional_write_gpkg(flat, "tbl_segment_flat")
    update_progress(88)
    log_to_gui(log_widget, f"tbl_segment_flat rows: {len(flat)}")

# --- NEW: combine stacked + flat build (referenced earlier) ---
def build_flat_and_stacked(gpkg_file, log_widget):
    build_stacked_data(gpkg_file, log_widget)
    build_flat_data(gpkg_file, log_widget)

# -------------------------------
# Orchestrator
# -------------------------------
def process_all(gpkg_file, log_widget):
    log_to_gui(log_widget, "SEGMENT PROCESS START (Parquet first).")
    process_and_buffer_lines(gpkg_file, log_widget); update_progress(25)
    create_segments_from_buffered_lines(gpkg_file, log_widget); update_progress(55)
    build_flat_and_stacked(gpkg_file, log_widget); update_progress(95)
    increment_stat_value(config_file, 'mesa_stat_process_lines', increment_value=1)
    log_to_gui(log_widget, "COMPLETED: Segment processing (parquet).")
    update_progress(100)

# -------------------------------
# Misc
# -------------------------------
def run_subprocess(command, fallback_command):
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            subprocess.run(fallback_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            log_to_gui(f"Failed to execute command: {command}")

def edit_asset_group():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    edit_lines_py = os.path.join(current_dir, '08_edit_lines.py')
    edit_lines_exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), '08_edit_lines.exe')
    run_subprocess(["python", edit_lines_py], [edit_lines_exe])

def exit_program():
    root.destroy()

# -------------------------------
# Main (UI)
# -------------------------------
parser = argparse.ArgumentParser(description='Slave script')
parser.add_argument('--original_working_directory', required=False, help='Path to running folder')
args = parser.parse_args()
original_working_directory = args.original_working_directory

if original_working_directory is None or original_working_directory == '':
    original_working_directory  = os.getcwd()
    if str("system") in str(original_working_directory):
        original_working_directory = os.path.join(os.getcwd(),'../')

config_file             = os.path.join(original_working_directory, "system/config.ini")
gpkg_file               = os.path.join(original_working_directory, "output/mesa.gpkg")

config                  = read_config(config_file)
input_folder_asset      = os.path.join(original_working_directory, config['DEFAULT']['input_folder_asset'])
input_folder_geocode    = os.path.join(original_working_directory, config['DEFAULT']['input_folder_geocode'])

ttk_bootstrap_theme     = config['DEFAULT']['ttk_bootstrap_theme']
workingprojection_epsg  = f"EPSG:{config['DEFAULT']['workingprojection_epsg']}"
parquet_folder          = os.path.join(
    original_working_directory,
    config['DEFAULT'].get('parquet_folder', 'output/geoparquet')
)

root = ttk.Window(themename=ttk_bootstrap_theme)
root.title("Admin segments")
try:
    root.iconbitmap(os.path.join(original_working_directory,"system_resources/mesa.ico"))
except Exception:
    pass
root.geometry("750x350")

button_width = 18
button_padx  = 7
button_pady  = 7

main_frame = tk.Frame(root)
main_frame.pack(fill='both', expand=True, pady=10)

log_frame = ttk.LabelFrame(main_frame, text="Log output", bootstyle="info")
log_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

log_widget = scrolledtext.ScrolledText(log_frame, height=6)
log_widget.pack(fill=tk.BOTH, expand=True)

progress_frame = tk.Frame(main_frame)
progress_frame.pack(pady=5)

progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=200, mode="determinate", variable=progress_var, bootstyle='info')
progress_bar.pack(side=tk.LEFT)

progress_label = tk.Label(progress_frame, text="0%", bg="light grey")
progress_label.pack(side=tk.LEFT, padx=5)

buttons_frame = tk.Frame(main_frame)
buttons_frame.pack(side='left', fill='both', padx=20, pady=5)

info_label_text = "Create sensitivity values for the segments."
info_label = tk.Label(root, text=info_label_text, wraplength=600, justify="left", anchor="w")
info_label.pack(fill=tk.X, padx=10, pady=10, anchor="w")

process_button = ttk.Button(buttons_frame, text="Process segments", command=lambda: process_all(gpkg_file, log_widget), width=button_width)
process_button.grid(row=0, column=0, padx=button_padx, pady=button_pady)

process_label = tk.Label(buttons_frame, text="Create sensitivity values for the segments.", bg="light grey",  justify='left')
process_label.grid(row=0, column=1, padx=button_padx, sticky='w')

exit_btn = ttk.Button(buttons_frame, text="Exit", command=exit_program, width=button_width, bootstyle="warning")
exit_btn.grid(row=1, column=1, pady=button_pady, sticky='e')

root.mainloop()