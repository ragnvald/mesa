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
    root.update_idletasks()

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
def load_lines_table(gpkg_file):
    if not os.path.exists(gpkg_file):
        return None
    try:
        gdf = gpd.read_file(gpkg_file, layer='tbl_lines')
        return None if gdf.empty else gdf
    except ValueError:
        return None

def save_to_geoparquet(gdf, file_path, log_widget):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        gdf.to_parquet(file_path, index=False)
        log_to_gui(log_widget, f"Saved to geoparquet: {file_path}")
    except Exception as e:
        log_to_gui(log_widget, f"Error saving to geoparquet: {e}")

# -------------------------------
# Geometry building
# -------------------------------
def create_lines_table_and_lines(gpkg_file, log_widget):
    gdf_geocode_group = gpd.read_file(gpkg_file, layer='tbl_geocode_group', rows=1)
    if gdf_geocode_group.empty:
        log_to_gui(log_widget, "The 'tbl_geocode_group' table is empty or does not exist.")
        return
    log_to_gui(log_widget, "Creating table and lines")

    minx, miny, maxx, maxy = gdf_geocode_group.total_bounds
    lines = []
    for i in range(3):
        start_x = np.random.uniform(minx, maxx)
        start_y = np.random.uniform(miny, maxy)
        end_x   = np.random.uniform(minx, maxx)
        end_y   = np.random.uniform(miny, maxy)
        lines.append(LineString([(start_x, start_y), (end_x, end_y)]))

    gdf_lines = gpd.GeoDataFrame({
        'name_gis': [f'line_{i:03}' for i in range(1, 4)],
        'name_user': [f'line_{i:03}' for i in range(1, 4)],
        'segment_length': [15, 30, 10],
        'segment_width': [1000, 20000, 5000],
        'description': ['another line', 'another line', 'another line'],
        'geometry': lines
    }, crs=gdf_geocode_group.crs)

    gdf_lines.crs = workingprojection_epsg
    gdf_lines.to_file(gpkg_file, layer='tbl_lines', mode="w")

def process_and_buffer_lines(gpkg_file, log_widget):
    crs        = workingprojection_epsg
    target_crs = "EPSG:4087"
    lines_df   = load_lines_table(gpkg_file)

    if lines_df is None:
        log_to_gui(log_widget, "Lines do not exist. Will create three template lines.")
        create_lines_table_and_lines(gpkg_file, log_widget)
        lines_df = load_lines_table(gpkg_file)
        if lines_df is None:
            log_to_gui(log_widget, "Failed to create lines.")
            return

    buffered_lines_data = []
    for index, row in lines_df.iterrows():
        try:
            geom            = row['geometry']
            name_gis        = row['name_gis']
            name_user       = row['name_user']
            segment_length  = int(row['segment_length'])
            segment_width   = int(row['segment_width'])
            description     = row['description']

            log_to_gui(log_widget, f"Processing line {index}, Geometry type: {type(geom)}")

            temp_gdf = gpd.GeoDataFrame([{'geometry': geom}], geometry='geometry', crs=crs)
            temp_gdf_proj = temp_gdf.to_crs(target_crs)
            temp_gdf_proj['geometry'] = temp_gdf_proj.buffer(segment_width, cap_style=2)
            temp_gdf_buffered = temp_gdf_proj.to_crs(crs)

            if not isinstance(temp_gdf_buffered.iloc[0].geometry, (Polygon, MultiPolygon)):
                log_to_gui(log_widget, f"Buffered geometry is not a Polygon/MultiPolygon. Got: {type(temp_gdf_buffered.iloc[0].geometry)}")

            buffered_lines_data.append({
                'fid': index,
                'name_gis': name_gis,
                'name_user': name_user,
                'segment_length': segment_length,
                'segment_width': segment_width,
                'description': description,
                'geometry': temp_gdf_buffered.iloc[0].geometry
            })
            log_to_gui(log_widget, f"Added a buffered version of {name_gis} to GeoPackage.")
        except Exception as e:
            log_to_gui(log_widget, f"Error processing line {index}: {e}")

    if buffered_lines_data:
        all_buffered_lines_df = gpd.GeoDataFrame(buffered_lines_data, geometry='geometry', crs=crs)
        all_buffered_lines_df.to_file(gpkg_file, layer="tbl_lines_buffered", driver="GPKG")
        log_to_gui(log_widget, "All buffered lines added to the database.")
    else:
        log_to_gui(log_widget, "No lines were processed.")

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
    buffered_lines_gdf = gpd.read_file(gpkg_file, layer='tbl_lines_buffered')

    all_segments_gdf = gpd.GeoDataFrame(columns=['name_gis', 'name_user', 'segment_length', 'geometry'])
    segment_id_counter = {}

    for index, row in lines_df.iterrows():
        line_input     = row.geometry
        name_gis       = row['name_gis']
        name_user      = row['name_user']
        segment_width  = row['segment_width']
        segment_length = row['segment_length']

        if name_gis not in segment_id_counter:
            segment_id_counter[name_gis] = 1

        perpendicular_lines = create_perpendicular_lines(line_input, segment_width, segment_length)
        matches = buffered_lines_gdf[buffered_lines_gdf['name_gis'] == name_gis]

        log_to_gui(log_widget, f"Creating segments for: {name_gis}")

        for _, match_row in matches.iterrows():
            buffered_line_geometry = match_row.geometry
            if not isinstance(buffered_line_geometry, Polygon):
                log_to_gui(log_widget, "Geometry is not a Polygon. Skipping.")
                continue

            segments_gdf = cut_into_segments(perpendicular_lines, buffered_line_geometry)
            valid_segments_gdf = segments_gdf[segments_gdf.is_valid]
            if valid_segments_gdf.empty:
                log_to_gui(log_widget, f"No valid segments created for {name_gis}. Skipping.")
                continue

            valid_segments_gdf['segment_id'] = [f"{name_gis}_{segment_id_counter[name_gis] + i}" for i in range(len(valid_segments_gdf))]
            segment_id_counter[name_gis] += len(valid_segments_gdf)

            valid_segments_gdf['name_gis'] = name_gis
            valid_segments_gdf['name_user'] = name_user
            valid_segments_gdf['segment_length'] = segment_length

            all_segments_gdf = pd.concat([all_segments_gdf, valid_segments_gdf], ignore_index=True)

    if not all_segments_gdf.empty:
        all_segments_gdf.crs = lines_df.crs
        all_segments_gdf.to_file(gpkg_file, layer="tbl_segments", driver="GPKG")
        log_to_gui(log_widget, "All segments have been accumulated and saved to 'tbl_segments'.")
    else:
        log_to_gui(log_widget, "No segments were created.")

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
    log_to_gui(log_widget, "Building tbl_segment_stacked.")
    update_progress(10)

    asset_data = gpd.read_file(gpkg_file, layer='tbl_asset_object')
    update_progress(15)

    asset_group_data = gpd.read_file(gpkg_file, layer='tbl_asset_group')
    update_progress(25)

    # Merge attributes (assumes these columns exist in asset_group_data)
    asset_data = asset_data.merge(
        asset_group_data[['id', 'name_gis_assetgroup', 'total_asset_objects',
                          'importance', 'susceptibility', 'sensitivity',
                          'sensitivity_code', 'sensitivity_description']],
        left_on='ref_asset_group', right_on='id', how='left'
    )

    lines_data  = gpd.read_file(gpkg_file, layer='tbl_lines')
    update_progress(29)

    segments_data = gpd.read_file(gpkg_file, layer='tbl_segments')
    update_progress(35)

    lines_data_renamed = lines_data.rename(columns={'name_gis': 'lines_name_gis'})
    segments_related = segments_data.merge(
        lines_data_renamed[['lines_name_gis']],
        left_on='name_gis', right_on='lines_name_gis', how='left', suffixes=('_seg', '_line')
    ).set_crs(workingprojection_epsg, allow_override=True)

    # Intersections by geometry type
    point_intersections        = intersection_with_geocode_data(asset_data, segments_related, 'Point', log_widget);        update_progress(33)
    multipoint_intersections   = intersection_with_geocode_data(asset_data, segments_related, 'MultiPoint', log_widget);   update_progress(36)
    line_intersections         = intersection_with_geocode_data(asset_data, segments_related, 'LineString', log_widget);   update_progress(49)
    multiline_intersections    = intersection_with_geocode_data(asset_data, segments_related, 'MultiLineString', log_widget); update_progress(42)
    polygon_intersections      = intersection_with_geocode_data(asset_data, segments_related, 'Polygon', log_widget);      update_progress(45)
    multipolygon_intersections = intersection_with_geocode_data(asset_data, segments_related, 'MultiPolygon', log_widget); update_progress(47)

    intersected_data = pd.concat([
        point_intersections, multipoint_intersections,
        line_intersections, multiline_intersections,
        polygon_intersections, multipolygon_intersections
    ], ignore_index=True)

    log_to_gui(log_widget, f"Total intersected data count: {len(intersected_data)}")
    update_progress(60)

    for col in ['id_x', 'id_y', 'lines_name_gis']:
        if col in intersected_data.columns:
            intersected_data.drop(columns=[col], inplace=True)

    intersected_data.reset_index(drop=True, inplace=True)
    intersected_data['fid'] = intersected_data.index
    intersected_data = gpd.GeoDataFrame(intersected_data, geometry='geometry', crs=workingprojection_epsg)

    # ---- Classification for stacked on 'sensitivity' ----
    classes = read_config_classification(config_file)
    if 'sensitivity' in intersected_data.columns:
        intersected_data, codecol, desccol = apply_classification_to_gdf(intersected_data, 'sensitivity', classes)
        # rename to match stacked convention (no suffix)
        if codecol != 'sensitivity_code':
            intersected_data.rename(columns={codecol:'sensitivity_code'}, inplace=True)
        if desccol != 'sensitivity_description':
            intersected_data.rename(columns={desccol:'sensitivity_description'}, inplace=True)
    else:
        log_to_gui(log_widget, "Warning: 'sensitivity' column missing in intersected data; classification skipped for stacked.")

    # Save GPKG + Parquet
    intersected_data.to_file(gpkg_file, layer='tbl_segment_stacked', driver='GPKG')
    update_progress(70)
    parquet_path = os.path.join(parquet_folder, "tbl_segment_stacked.parquet")
    save_to_geoparquet(intersected_data, parquet_path, log_widget)

def build_flat_data(gpkg_file, log_widget):
    log_to_gui(log_widget, "Building tbl_segment_flat ...")
    stacked = gpd.read_file(gpkg_file, layer='tbl_segment_stacked')
    update_progress(60)

    # Aggregation
    aggregation_functions = {
        'importance': ['min', 'max'],
        'sensitivity': ['min', 'max'],
        'susceptibility': ['min', 'max'],
        'name_gis': 'first',
        'segment_id': 'first',
        # If you have a name for the asset group, make sure this column exists; else comment out next line
        # 'asset_group_name': lambda x: ', '.join(x.astype(str).unique()),
        'geometry': 'first'
    }

    tbl_segment_flat = stacked.groupby('segment_id').agg(aggregation_functions)
    tbl_segment_flat.columns = ['_'.join(col).strip() for col in tbl_segment_flat.columns.values]
    renamed = {
        'name_gis_first': 'name_gis',
        'geometry_first': 'geometry'
    }
    tbl_segment_flat.rename(columns=renamed, inplace=True)

    # Keep index as column
    tbl_segment_flat.reset_index(inplace=True)

    # Remove duplicate segment_id if present
    if 'segment_id_first' in tbl_segment_flat.columns:
        tbl_segment_flat.drop(columns=['segment_id_first'], inplace=True)

    # To GeoDataFrame
    tbl_segment_flat = gpd.GeoDataFrame(tbl_segment_flat, geometry='geometry', crs=workingprojection_epsg)

    # ---- Classification for flat on sensitivity_min/max ----
    classes = read_config_classification(config_file)
    if not classes:
        log_to_gui(log_widget, "Warning: No classification ranges found in config; codes will be 'Unknown'.")

    if 'sensitivity_min_min' in tbl_segment_flat.columns:
        # Some datasets could double-suffix after groupby; normalize: prefer 'sensitivity_min'/'sensitivity_max'
        tbl_segment_flat['sensitivity_min'] = tbl_segment_flat.get('sensitivity_min', tbl_segment_flat['sensitivity_min_min'])
        tbl_segment_flat.drop(columns=['sensitivity_min_min'], inplace=True)
    if 'sensitivity_max_max' in tbl_segment_flat.columns:
        tbl_segment_flat['sensitivity_max'] = tbl_segment_flat.get('sensitivity_max', tbl_segment_flat['sensitivity_max_max'])
        tbl_segment_flat.drop(columns=['sensitivity_max_max'], inplace=True)

    if 'sensitivity_min' in tbl_segment_flat.columns:
        tbl_segment_flat, code_min_col, desc_min_col = apply_classification_to_gdf(tbl_segment_flat, 'sensitivity_min', classes, code_suffix='min')
    else:
        log_to_gui(log_widget, "Warning: 'sensitivity_min' missing in flat; min classification skipped.")

    if 'sensitivity_max' in tbl_segment_flat.columns:
        tbl_segment_flat, code_max_col, desc_max_col = apply_classification_to_gdf(tbl_segment_flat, 'sensitivity_max', classes, code_suffix='max')
        # Ensure exact names expected by cartography: sensitivity_code_max, sensitivity_description_max
        if code_max_col != 'sensitivity_code_max':
            tbl_segment_flat.rename(columns={code_max_col:'sensitivity_code_max'}, inplace=True)
        if desc_max_col != 'sensitivity_description_max':
            tbl_segment_flat.rename(columns={desc_max_col:'sensitivity_description_max'}, inplace=True)
    else:
        log_to_gui(log_widget, "Warning: 'sensitivity_max' missing in flat; max classification skipped.")

    # Save GPKG + Parquet with the new code/desc columns INCLUDED
    tbl_segment_flat.to_file(gpkg_file, layer='tbl_segment_flat', driver='GPKG')
    log_to_gui(log_widget, "Completed flat segments with classification.")

    parquet_path = os.path.join(parquet_folder, "tbl_segment_flat.parquet")
    save_to_geoparquet(tbl_segment_flat, parquet_path, log_widget)

def build_flat_and_stacked(gpkg_file, log_widget):
    build_stacked_data(gpkg_file, log_widget)
    build_flat_data(gpkg_file, log_widget)
    log_to_gui(log_widget, "Finalising processing.")
    update_progress(100)

# -------------------------------
# Orchestrator
# -------------------------------
def process_all(gpkg_file, log_widget):
    process_and_buffer_lines(gpkg_file, log_widget)
    update_progress(30)
    create_segments_from_buffered_lines(gpkg_file, log_widget)
    update_progress(60)
    build_flat_and_stacked(gpkg_file, log_widget)
    update_progress(90)
    log_to_gui(log_widget, "COMPLETED: Data processing and aggregation.")
    update_progress(100)
    increment_stat_value(config_file, 'mesa_stat_process_lines', increment_value=1)

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
