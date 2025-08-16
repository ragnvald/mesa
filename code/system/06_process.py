# -*- coding: utf-8 -*-
# CPU-optimized, Windows-safe (spawn) multiprocessing.
# - Top-level worker functions (pickle-friendly)
# - ProcessPoolExecutor with initializer to share large data once
# - Coarser spatial chunking, specialized predicates, throttled UI logging

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

import os
import argparse
import configparser
import threading
import time
import multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta

import math
import random
import numpy as np
import pandas as pd
import geopandas as gpd

import tkinter as tk
from tkinter import scrolledtext

from shapely.geometry import box, Polygon, MultiPolygon

import ttkbootstrap as tb
from ttkbootstrap.constants import *

# ----------------------------
# Globals set in __main__
# ----------------------------
original_working_directory = None
log_widget = None
progress_var = None
progress_label = None
classification = {}

# ----------------------------
# UI / Logging / Config utils
# ----------------------------
def update_progress(new_value: float) -> None:
    try:
        v = max(0, min(100, float(new_value)))
        progress_var.set(v)
        progress_label.config(text=f"{int(v)}%")
        progress_label.update_idletasks()
    except Exception:
        pass

def close_application(root: tk.Tk) -> None:
    root.destroy()

def log_to_gui(widget: scrolledtext.ScrolledText, message: str) -> None:
    timestamp = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted = f"{timestamp} - {message}"
    try:
        if widget and widget.winfo_exists():
            widget.insert(tk.END, formatted + "\n")
            widget.see(tk.END)
    except tk.TclError:
        pass
    if original_working_directory:
        try:
            with open(os.path.join(original_working_directory, "log.txt"), "a", encoding="utf-8") as f:
                f.write(formatted + "\n")
        except Exception:
            pass

def read_config(file_name: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(file_name)
    return cfg

def increment_stat_value(config_file: str, stat_name: str, increment_value: int) -> None:
    if not os.path.isfile(config_file):
        log_to_gui(log_widget, f"Configuration file {config_file} not found.")
        return
    with open(config_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{stat_name} ="):
            parts = line.split("=", 1)
            if len(parts) == 2:
                try:
                    cur = int(parts[1].strip())
                    lines[i] = f"{stat_name} = {cur + increment_value}\n"
                except ValueError:
                    log_to_gui(log_widget, f"Error: Current value of {stat_name} is not an integer.")
            break
    with open(config_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

# --------------------
# Classification (unchanged)
# --------------------
def read_config_classification(file_name: str) -> dict:
    global classification
    cfg = configparser.ConfigParser()
    cfg.read(file_name)
    classification.clear()
    for section in cfg.sections():
        if section in ['A', 'B', 'C', 'D', 'E']:
            rng = cfg[section].get('range', '').strip()
            desc = cfg[section].get('description', '').strip()
            if "-" in rng:
                try:
                    start, end = map(int, rng.split("-"))
                    classification[section] = {"range": range(start, end + 1), "description": desc}
                except Exception as e:
                    log_to_gui(log_widget, f"Invalid range in [{section}]: {rng} ({e})")
    return classification

def classify_data(gpkg_file: str, layer: str, column_name: str, config_path: str) -> None:
    read_config_classification(config_path)
    gdf = gpd.read_file(gpkg_file, layer=layer)
    if column_name not in gdf.columns:
        log_to_gui(log_widget, f"Column '{column_name}' not in {layer}; skipping.")
        return
    def classify_value(v):
        for label, info in classification.items():
            if isinstance(v, (int, np.integer)) and v in info['range']:
                return label, info['description']
        return 'Unknown', 'No description available'
    base, *suf = column_name.rsplit('_', 1)
    suf = suf[0] if suf else ''
    code_col = f"{base}_code_{suf}" if suf else f"{base}_code"
    desc_col = f"{base}_description_{suf}" if suf else f"{base}_description"
    codes, descs = zip(*gdf[column_name].apply(classify_value))
    gdf[code_col] = codes
    gdf[desc_col] = descs
    gdf.to_file(gpkg_file, layer=layer, driver='GPKG')
    log_to_gui(log_widget, f"Saved {layer} with {code_col}, {desc_col}")

# -----------------
# Spatial utilities (top-level)
# -----------------
def intersection_with_geocode_data(asset_df: gpd.GeoDataFrame,
                                   geocode_df: gpd.GeoDataFrame,
                                   geom_type: str) -> gpd.GeoDataFrame:
    af = asset_df[asset_df.geometry.geom_type == geom_type]
    if af.empty:
        return gpd.GeoDataFrame(geometry=[], crs=geocode_df.crs)
    if af.crs != geocode_df.crs:
        af = af.to_crs(geocode_df.crs)
    _ = geocode_df.sindex; _ = af.sindex
    predicate = 'contains' if geom_type == 'Point' else 'intersects'
    return gpd.sjoin(geocode_df, af, how='inner', predicate=predicate)

def process_geocode_chunk(geocode_chunk: gpd.GeoDataFrame,
                          asset_data: gpd.GeoDataFrame,
                          asset_geom_types) -> pd.DataFrame:
    pieces = []
    for gtype in asset_geom_types:
        res = intersection_with_geocode_data(asset_data, geocode_chunk, gtype)
        if not res.empty:
            pieces.append(res)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)

def create_grid(geodata: gpd.GeoDataFrame, cell_size_deg: float) -> list:
    xmin, ymin, xmax, ymax = geodata.total_bounds
    x_edges = np.arange(xmin, xmax + cell_size_deg, cell_size_deg)
    y_edges = np.arange(ymin, ymax + cell_size_deg, cell_size_deg)
    cells = []
    for x in x_edges[:-1]:
        for y in y_edges[:-1]:
            cells.append((x, y, x + cell_size_deg, y + cell_size_deg))
    return cells

# -----------------
# assign-to-grid: move grid to child via initializer (top-level)
# -----------------
_GLOBAL_GRID_GDF = None
def _grid_pool_init(grid_gdf):
    # Called in each worker process once
    global _GLOBAL_GRID_GDF
    _GLOBAL_GRID_GDF = grid_gdf

def _process_chunk_indexed(geodata_chunk: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    _ = geodata_chunk.sindex
    _ = _GLOBAL_GRID_GDF.sindex
    return gpd.sjoin(geodata_chunk, _GLOBAL_GRID_GDF, how="left", predicate="intersects")

def calculate_optimal_chunk_size(data_size: int, max_memory_mb: int = 512) -> int:
    memory_per_row_mb = 0.001
    max_rows_per_chunk = max_memory_mb / memory_per_row_mb
    chunk_size = max(1, int(data_size / max_rows_per_chunk))
    return min(chunk_size, data_size)

def assign_assets_to_grid(geodata: gpd.GeoDataFrame,
                          grid_cells: list,
                          log_widget,
                          max_workers: int) -> gpd.GeoDataFrame:
    log_to_gui(log_widget, "Creating grid GeoDataFrame.")
    grid_gdf = gpd.GeoDataFrame(
        {'grid_cell': range(len(grid_cells)),
         'geometry': [box(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax in grid_cells]},
        geometry='geometry',
        crs=geodata.crs
    )
    log_to_gui(log_widget, "Grid built. Building spatial index…")
    _ = grid_gdf.sindex

    chunk_size = calculate_optimal_chunk_size(len(geodata))
    chunks = [geodata.iloc[i:i + chunk_size] for i in range(0, len(geodata), chunk_size)]
    log_to_gui(log_widget, f"Assigning to grid using {len(chunks)} chunks with {max_workers} workers.")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers,
                             initializer=_grid_pool_init,
                             initargs=(grid_gdf,)) as ex:
        for part in ex.map(_process_chunk_indexed, chunks):
            results.append(part)

    j = gpd.GeoDataFrame(pd.concat(results, ignore_index=True), crs=geodata.crs)
    return j.drop(columns='index_right', errors='ignore')

def make_spatial_chunks(geocode_tagged: gpd.GeoDataFrame, max_workers: int, multiplier: int = 4):
    cell_ids = geocode_tagged['grid_cell'].unique().tolist()
    random.shuffle(cell_ids)
    target_chunks = max(1, min(len(cell_ids), max_workers * multiplier))
    cells_per_chunk = math.ceil(len(cell_ids) / target_chunks)
    chunks = []
    for i in range(0, len(cell_ids), cells_per_chunk):
        sel = set(cell_ids[i:i + cells_per_chunk])
        chunks.append(geocode_tagged[geocode_tagged['grid_cell'].isin(sel)])
    return chunks

# -----------------
# intersection workers via initializer (top-level)
# -----------------
_POOL_ASSETS = None
_POOL_ASSET_GEOM_TYPES = None

def _intersect_pool_init(asset_df, asset_geom_types):
    # Called in each worker process once
    global _POOL_ASSETS, _POOL_ASSET_GEOM_TYPES
    _POOL_ASSETS = asset_df
    _POOL_ASSET_GEOM_TYPES = asset_geom_types

def _intersection_worker(args):
    idx, geocode_chunk = args
    try:
        _ = geocode_chunk.sindex
        pieces = []
        for gt in _POOL_ASSET_GEOM_TYPES:
            res = intersection_with_geocode_data(_POOL_ASSETS, geocode_chunk, gt)
            if not res.empty:
                pieces.append(res)
        if not pieces:
            return (idx, 0, None, None)
        out = pd.concat(pieces, ignore_index=True)
        return (idx, len(out), out, None)
    except Exception as e:
        return (idx, 0, None, str(e))

# --------------------------------------------
# Heavy lifter: Intersections (CPU, optimized)
# --------------------------------------------
def intersect_asset_and_geocode(asset_data, geocode_data, log_widget, progress_var, method,
                                workingprojection_epsg, cell_size, max_workers):
    intersections = []

    # Workers
    if max_workers == 0:
        try:
            max_workers = multiprocessing.cpu_count()
            log_to_gui(log_widget, f"Number of workers determined by system to {max_workers}.")
        except NotImplementedError:
            max_workers = 4
    else:
        log_to_gui(log_widget, f"Number of workers set in config to {max_workers}.")

    log_to_gui(log_widget, f"Processing method is {method}. (CPU-optimized)")
    start_time = time.time()

    # meters→degrees approx (prefer projected CRS in meters in your data upstream)
    meters_per_degree = 111320.0
    cell_size_degrees = cell_size / meters_per_degree
    log_to_gui(log_widget, f"Cell size converted to degrees: {cell_size_degrees:.6f} degrees")

    # Grid + tag geocodes
    log_to_gui(log_widget, "Creating analysis grid.")
    grid_cells = create_grid(geocode_data, cell_size_degrees)
    log_to_gui(log_widget, "Assigning geocodes to grid.")
    geocode_tagged = assign_assets_to_grid(geocode_data, grid_cells, log_widget, max_workers)

    # Build coarser chunks
    chunks = make_spatial_chunks(geocode_tagged, max_workers, multiplier=4)
    total_chunks = len(chunks)
    log_to_gui(log_widget, f"Processing {total_chunks} chunks with up to {max_workers} workers.")

    # Cache asset geometry types once
    asset_geom_types = asset_data.geometry.geom_type.unique().tolist()

    # Share ASSETS once per worker via initializer (lighter than sending every task)
    completed = 0
    next_log_tick = 0.0
    with ProcessPoolExecutor(max_workers=max_workers,
                             initializer=_intersect_pool_init,
                             initargs=(asset_data, asset_geom_types)) as ex:
        futures = {ex.submit(_intersection_worker, (i, ch)): i
                   for i, ch in enumerate(chunks, start=1)}
        for fut in concurrent.futures.as_completed(futures):
            idx, nrows, res, err = fut.result()
            completed += 1
            if err:
                log_to_gui(log_widget, f"[{completed}/{total_chunks}] Error: {err}")
            elif nrows:
                intersections.append(res)

            pct = 30.0 + (completed / max(total_chunks, 1)) * 15.0
            if pct >= next_log_tick:
                update_progress(pct)
                if completed % max(1, total_chunks // 20) == 0 or completed == total_chunks:
                    elapsed = time.time() - start_time
                    est_total = (elapsed / completed) * total_chunks if completed else 0.0
                    eta = datetime.now() + timedelta(seconds=max(0.0, est_total - elapsed))
                    ts = eta.strftime("%H:%M:%S")
                    days_diff = (eta.date() - datetime.now().date()).days
                    if days_diff > 0:
                        ts += f" (+{days_diff} days)"
                    log_to_gui(log_widget, f"Core computation might conclude at {ts}.")
                next_log_tick = pct + 1.0

    # Wrap up
    total_time = time.time() - start_time
    time_per_chunk = total_time / total_chunks if total_chunks > 0 else 0
    hours, rem = divmod(total_time, 3600); minutes, seconds = divmod(rem, 60)
    assets_per_sec = (len(asset_data) / total_time) if total_time > 0 else 0.0
    geocodes_per_sec = (len(geocode_tagged) / total_time) if total_time > 0 else 0.0

    log_to_gui(log_widget, "Core computation concluded.")
    log_to_gui(log_widget, f"Processing completed in {total_time:.2f} seconds "
                           f"({int(hours)}h {int(minutes)}m {int(seconds)}s).")
    log_to_gui(log_widget, f"Average time per chunk: {time_per_chunk:.2f} s.")
    log_to_gui(log_widget, f"Total asset objects: {len(asset_data)} (≈{assets_per_sec:.2f}/s).")
    log_to_gui(log_widget, f"Total geocode objects: {len(geocode_tagged)} (≈{geocodes_per_sec:.2f}/s).")

    if not intersections:
        return gpd.GeoDataFrame(geometry=[], crs=workingprojection_epsg)

    return gpd.GeoDataFrame(pd.concat(intersections, ignore_index=True), crs=workingprojection_epsg)

# ------------------------------------
# Table builders
# ------------------------------------
def process_tbl_stacked(gpkg_file: str,
                        workingprojection_epsg: str,
                        method: str,
                        cell_size: int,
                        max_workers: int) -> None:
    log_to_gui(log_widget, "Started building analysis table (tbl_stacked).")
    update_progress(10)

    log_to_gui(log_widget, "Reading assets…")
    asset_data = gpd.read_file(gpkg_file, layer='tbl_asset_object'); _ = asset_data.sindex
    if asset_data.crs is None:
        asset_data.set_crs(workingprojection_epsg, inplace=True)
    update_progress(15)

    log_to_gui(log_widget, "Reading geocodes…")
    geocode_data = gpd.read_file(gpkg_file, layer='tbl_geocode_object'); _ = geocode_data.sindex
    if geocode_data.crs is None:
        geocode_data.set_crs(workingprojection_epsg, inplace=True)
    update_progress(20)
    log_to_gui(log_widget, f"Geocode objects: {len(geocode_data)}")

    log_to_gui(log_widget, "Reading asset groups…")
    asset_groups = gpd.read_file(gpkg_file, layer='tbl_asset_group')
    cols = ['id', 'name_gis_assetgroup', 'total_asset_objects', 'importance',
            'susceptibility', 'sensitivity', 'sensitivity_code', 'sensitivity_description']
    keep = [c for c in cols if c in asset_groups.columns]
    asset_data = asset_data.merge(asset_groups[keep], left_on='ref_asset_group', right_on='id', how='left')
    update_progress(30)

    log_to_gui(log_widget, "Intersecting assets and geocodes (CPU)…")
    stacked = intersect_asset_and_geocode(asset_data, geocode_data, log_widget, progress_var, method,
                                          workingprojection_epsg, cell_size, max_workers)

    if stacked.empty:
        log_to_gui(log_widget, "No intersections found; tbl_stacked will be empty.")
        stacked = gpd.GeoDataFrame(geometry=[], crs=workingprojection_epsg)
        stacked.to_file(gpkg_file, layer='tbl_stacked', driver='GPKG')
        update_progress(50); return

    log_to_gui(log_widget, "Calculating areas…")
    if stacked.crs is None:
        stacked.set_crs(workingprojection_epsg, inplace=True)
    if stacked.crs.is_geographic:
        tmp = stacked.to_crs("EPSG:3395")
        stacked['area_m2'] = tmp.geometry.area.astype('float64').fillna(0).astype('int64')
    else:
        stacked['area_m2'] = stacked.geometry.area.astype('float64').fillna(0).astype('int64')

    stacked.drop(columns=['geometry_wkb', 'geometry_wkb_1', 'process'], errors='ignore', inplace=True)
    stacked.to_file(gpkg_file, layer='tbl_stacked', driver='GPKG')
    log_to_gui(log_widget, "tbl_stacked saved."); update_progress(50)

def process_tbl_flat(gpkg_file: str, workingprojection_epsg: str) -> None:
    log_to_gui(log_widget, "Building presentation table (tbl_flat).")
    tbl_stacked = gpd.read_file(gpkg_file, layer='tbl_stacked')
    log_to_gui(log_widget, f"tbl_stacked rows: {len(tbl_stacked)}")

    if tbl_stacked.empty:
        empty = gpd.GeoDataFrame(columns=[
            'code','importance_min','importance_max','sensitivity_min','sensitivity_max',
            'susceptibility_min','susceptibility_max','ref_geocodegroup','name_gis_geocodegroup',
            'asset_group_names','asset_groups_total','area_m2','assets_overlap_total','geometry'
        ], geometry='geometry', crs=workingprojection_epsg)
        empty.to_file(gpkg_file, layer='tbl_flat', driver='GPKG')
        log_to_gui(log_widget, "tbl_flat saved (empty)."); update_progress(65); return

    if 'code' not in tbl_stacked.columns:
        log_to_gui(log_widget, "Error: 'code' missing in tbl_stacked; cannot aggregate."); return
    if tbl_stacked.crs is None:
        tbl_stacked.set_crs(workingprojection_epsg, inplace=True)
    update_progress(60)

    overlap = tbl_stacked['code'].value_counts().reset_index()
    overlap.columns = ['code','assets_overlap_total']

    agg = {
        'importance':['min','max'],
        'sensitivity':['min','max'],
        'susceptibility':['min','max'],
        'ref_geocodegroup':'first',
        'name_gis_geocodegroup':'first',
        'name_gis_assetgroup':(lambda x: ', '.join(pd.Series(x).dropna().unique())),
        'ref_asset_group':'nunique',
        'geometry':'first', 'area_m2':'first'
    }
    present_agg = {k:v for k,v in agg.items() if k in tbl_stacked.columns}
    tbl_flat = tbl_stacked.groupby('code').agg(present_agg)
    tbl_flat.columns = ['_'.join(c).strip() if isinstance(c, tuple) else c for c in tbl_flat.columns]
    rename = {
        'ref_geocodegroup_first':'ref_geocodegroup',
        'name_gis_geocodegroup_first':'name_gis_geocodegroup',
        'name_gis_assetgroup_<lambda>':'asset_group_names',
        'ref_asset_group_nunique':'asset_groups_total',
        'geometry_first':'geometry','area_m2_first':'area_m2'
    }
    tbl_flat.rename(columns=rename, inplace=True, errors='ignore')
    tbl_flat = gpd.GeoDataFrame(tbl_flat, geometry='geometry', crs=workingprojection_epsg)

    if 'area_m2' not in tbl_flat.columns or tbl_flat['area_m2'].isna().all():
        if tbl_flat.crs.is_geographic:
            tmp = tbl_flat.to_crs("EPSG:3395")
            tbl_flat['area_m2'] = tmp.geometry.area.astype('int64')
        else:
            tbl_flat['area_m2'] = tbl_flat.geometry.area.astype('int64')

    tbl_flat.reset_index(inplace=True)
    if 'assets_overlap_total' not in tbl_flat.columns:
        tbl_flat = tbl_flat.merge(overlap, on='code', how='left')

    reference_columns = [
        'code','importance_min','importance_max','sensitivity_min','sensitivity_max',
        'susceptibility_min','susceptibility_max','ref_geocodegroup','name_gis_geocodegroup',
        'asset_group_names','asset_groups_total','area_m2','assets_overlap_total','geometry'
    ]
    for c in reference_columns:
        if c not in tbl_flat.columns:
            tbl_flat[c] = None
    tbl_flat = tbl_flat[reference_columns]
    tbl_flat.to_file(gpkg_file, layer='tbl_flat', driver='GPKG')
    log_to_gui(log_widget, "tbl_flat saved."); update_progress(75)

# ----------------------------
# Export helpers
# ----------------------------
def read_table_with_schema_validation(gpkg_file: str, layer: str) -> gpd.GeoDataFrame:
    try:
        import fiona
        with fiona.open(gpkg_file, layer=layer) as src:
            schema = src.schema; crs_wkt = src.crs_wkt
            log_to_gui(log_widget, f"Schema for {layer}: {schema}")
            log_to_gui(log_widget, f"CRS for {layer}: {crs_wkt}")
        gdf = gpd.read_file(gpkg_file, layer=layer)
        for field in schema.get('properties', {}):
            if field not in gdf.columns: gdf[field] = None
        return gdf
    except Exception as e:
        log_to_gui(log_widget, f"Failed to read {layer} with schema validation: {e}")
        return gpd.GeoDataFrame(geometry=[], crs=None)

def export_gpkg_tables_to_geoparquet(gpkg_file: str) -> None:
    out_dir = os.path.join(os.path.dirname(gpkg_file), "geoparquet")
    os.makedirs(out_dir, exist_ok=True)
    for layer in ["tbl_stacked","tbl_flat"]:
        try:
            gdf = read_table_with_schema_validation(gpkg_file, layer)
            if gdf.empty:
                log_to_gui(log_widget, f"{layer} is empty; writing empty parquet.")
            cols = [c for c in gdf.columns if c != gdf.geometry.name] + [gdf.geometry.name]
            gdf = gdf[cols]
            out_path = os.path.join(out_dir, f"{layer}.parquet")
            gdf.to_parquet(out_path, index=False)
            log_to_gui(log_widget, f"Wrote GeoParquet: {out_path}")
        except Exception as e:
            log_to_gui(log_widget, f"Failed to export {layer} to GeoParquet: {e}")

def cleanup_destination_files() -> None:
    parquet_folder = os.path.join(original_working_directory, "output", "geoparquet")
    os.makedirs(parquet_folder, exist_ok=True)
    for fn in ["tbl_stacked.parquet","tbl_flat.parquet","tbl_segment_stacked.parquet","tbl_segment_flat.parquet"]:
        p = os.path.join(parquet_folder, fn)
        try:
            if os.path.exists(p):
                os.remove(p)
                log_to_gui(log_widget, f"Removed: {fn}")
        except Exception as e:
            log_to_gui(log_widget, f"Error removing {fn}: {e}")

# -----------------
# Top-level process
# -----------------
def process_all(gpkg_file: str,
                config_file: str,
                workingprojection_epsg: str,
                method: str,
                cell_size: int,
                max_workers: int) -> None:
    try:
        log_to_gui(log_widget, "Started processing of analysis and presentation layers.")
        cleanup_destination_files(); update_progress(5)

        process_tbl_stacked(gpkg_file, workingprojection_epsg, method, cell_size, max_workers)
        process_tbl_flat(gpkg_file, workingprojection_epsg); update_progress(90)

        # Optional classification (uncomment if you use ranges)
        # classify_data(gpkg_file, 'tbl_flat', 'sensitivity_min', config_file); update_progress(94)
        # classify_data(gpkg_file, 'tbl_flat', 'sensitivity_max', config_file); update_progress(97)
        # classify_data(gpkg_file, 'tbl_stacked', 'sensitivity',     config_file); update_progress(99)

        log_to_gui(log_widget, "Exporting to GeoParquet…")
        export_gpkg_tables_to_geoparquet(gpkg_file)

        increment_stat_value(config_file, 'mesa_stat_process', increment_value=1)
        log_to_gui(log_widget, "COMPLETED."); update_progress(100)
    except Exception as e:
        log_to_gui(log_widget, f"Error during processing: {e}")
        increment_stat_value(config_file, 'mesa_stat_process', increment_value=1)
        raise

# -----------
# Entrypoint
# -----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU-optimized processing (Windows spawn-safe)")
    parser.add_argument('--original_working_directory', required=False, help='Path to running folder')
    args = parser.parse_args()
    original_working_directory = args.original_working_directory or os.getcwd()
    if "system" in os.path.basename(original_working_directory).lower():
        original_working_directory = os.path.abspath(os.path.join(original_working_directory, os.pardir))

    config_file = os.path.join(original_working_directory, "system", "config.ini")
    gpkg_file = os.path.join(original_working_directory, "output", "mesa.gpkg")

    cfg = read_config(config_file)
    workingprojection_epsg = f"EPSG:{cfg['DEFAULT']['workingprojection_epsg']}"
    method = str(cfg['DEFAULT'].get('method', 'cpu')).strip().lower()
    max_workers = int(cfg['DEFAULT'].get('max_workers', '0'))
    cell_size = int(cfg['DEFAULT'].get('cell_size', '18000'))
    ttk_bootstrap_theme = cfg['DEFAULT'].get('ttk_bootstrap_theme', 'flatly')

    root = tb.Window(themename=ttk_bootstrap_theme)
    root.title("Process data (CPU-optimized, spawn-safe)")
    try:
        icon_path = os.path.join(original_working_directory, "system_resources", "mesa.ico")
        if os.path.exists(icon_path): root.iconbitmap(icon_path)
    except Exception:
        pass

    log_widget = scrolledtext.ScrolledText(root, height=12)
    log_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    progress_frame = tk.Frame(root); progress_frame.pack(pady=5)
    progress_var = tk.DoubleVar(value=0.0)
    progress_bar = tb.Progressbar(progress_frame, orient="horizontal", length=260,
                                  mode="determinate", variable=progress_var, bootstyle='info')
    progress_bar.pack(side=tk.LEFT)
    progress_label = tk.Label(progress_frame, text="0%", bg="light grey"); progress_label.pack(side=tk.LEFT, padx=8)

    info_label_text = ("CPU-optimized & Windows spawn-safe. Processes use shared data via initializer. "
                       "Use a projected CRS (meters) upstream if possible for best accuracy/perf.")
    tk.Label(root, text=info_label_text, wraplength=680, justify="left").pack(padx=10, pady=10)

    btn_frame = tk.Frame(root); btn_frame.pack(pady=6)
    tb.Button(btn_frame, text="Process", bootstyle=PRIMARY,
              command=lambda: threading.Thread(target=process_all,
                                               args=(gpkg_file, config_file, workingprojection_epsg, method, cell_size, max_workers),
                                               daemon=True).start()).pack(side=tk.LEFT, padx=5)
    tb.Button(btn_frame, text="Exit", bootstyle=WARNING, command=lambda: close_application(root)).pack(side=tk.LEFT, padx=5)

    log_to_gui(log_widget, "Opened processing subprocess (CPU-optimized, spawn-safe).")
    root.mainloop()
