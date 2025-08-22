# -*- coding: utf-8 -*-
# 06_process.py — CPU-optimized (Windows spawn-safe) intersections + robust flattening to GeoParquet
# - Intersect tbl_asset_object × tbl_geocode_object -> tbl_stacked.parquet
# - Flatten by geocode code -> tbl_flat.parquet with min/max + A..E codes from config.ini
# - Compute area once per tile (equal-area CRS), then backfill to tbl_stacked
# - Heartbeat logs + ttkbootstrap GUI
#
# Inputs  (GeoParquet): output/geoparquet/{tbl_asset_object,tbl_geocode_object,tbl_asset_group}.parquet
# Outputs (GeoParquet): output/geoparquet/{tbl_stacked,tbl_flat}.parquet

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

import os, sys, math, time, random, argparse, threading, multiprocessing
import configparser
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

# --- GUI
import tkinter as tk
from tkinter import scrolledtext
import ttkbootstrap as tb
from ttkbootstrap.constants import PRIMARY, WARNING

# ----------------------------
# Globals (initialized in __main__)
# ----------------------------
original_working_directory = None
log_widget = None
progress_var = None
progress_label = None
HEARTBEAT_SECS = 30
AREA_PROJECTION_EPSG = "EPSG:3035"  # set from config at runtime

# ----------------------------
# Paths
# ----------------------------
def base_dir() -> Path:
    bd = Path(original_working_directory or os.getcwd())
    if bd.name.lower() == "system":
        return bd.parent
    return bd

def gpq_dir() -> Path:
    out = base_dir() / "output" / "geoparquet"
    out.mkdir(parents=True, exist_ok=True)
    return out

# ----------------------------
# Logging / UI helpers
# ----------------------------
def update_progress(new_value: float):
    try:
        v = max(0.0, min(100.0, float(new_value)))
        if progress_var is not None: progress_var.set(v)
        if progress_label is not None:
            progress_label.config(text=f"{int(v)}%")
            progress_label.update_idletasks()
    except Exception:
        pass

def log_to_gui(widget, message: str):
    timestamp = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted = f"{timestamp} - {message}"
    try:
        if widget and widget.winfo_exists():
            widget.insert(tk.END, formatted + "\n")
            widget.see(tk.END)
    except tk.TclError:
        pass
    try:
        with open(base_dir() / "log.txt", "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
    except Exception:
        pass
    if widget is None:
        print(formatted, flush=True)

def read_config(path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8")
    return cfg

# ----------------------------
# Config-driven A..E classification
# ----------------------------
def read_class_ranges(cfg_path: Path):
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path, encoding="utf-8")
    ranges = {}
    desc   = {}
    for code in ["A","B","C","D","E"]:
        if code not in cfg: continue
        rtxt = (cfg[code].get("range","") or "").strip()
        if "-" in rtxt:
            try:
                lo, hi = [int(x) for x in rtxt.split("-")]
                ranges[code] = range(lo, hi+1)
            except Exception:
                pass
        desc[code] = (cfg[code].get("description","") or "").strip()
    # Priority: A highest … E lowest
    order = {"A":5,"B":4,"C":3,"D":2,"E":1}
    return ranges, desc, order

def map_num_to_code(val, ranges_map: dict) -> str | None:
    if pd.isna(val): return None
    try:
        iv = int(round(float(val)))
    except Exception:
        return None
    for k, r in ranges_map.items():
        if iv in r: return k
    return None

# ----------------------------
# IO helpers (GeoParquet)
# ----------------------------
def read_parquet_or_empty(name: str) -> gpd.GeoDataFrame:
    path = gpq_dir() / f"{name}.parquet"
    try:
        return gpd.read_parquet(path) if path.exists() else gpd.GeoDataFrame(geometry=[], crs=None)
    except Exception as e:
        log_to_gui(log_widget, f"Failed to read {name}.parquet: {e}")
        return gpd.GeoDataFrame(geometry=[], crs=None)

def write_parquet(name: str, gdf: gpd.GeoDataFrame):
    path = gpq_dir() / f"{name}.parquet"
    gdf.to_parquet(path, index=False)
    log_to_gui(log_widget, f"Wrote {path}")

def cleanup_outputs():
    for fn in ["tbl_stacked.parquet","tbl_flat.parquet"]:
        p = gpq_dir() / fn
        try:
            if p.exists():
                p.unlink()
                log_to_gui(log_widget, f"Removed: {fn}")
        except Exception as e:
            log_to_gui(log_widget, f"Error removing {fn}: {e}")

# ----------------------------
# Spatial utilities for intersections
# ----------------------------
def create_grid(geodata: gpd.GeoDataFrame, cell_size_deg: float) -> list[tuple[float,float,float,float]]:
    xmin, ymin, xmax, ymax = geodata.total_bounds
    x_edges = np.arange(xmin, xmax + cell_size_deg, cell_size_deg)
    y_edges = np.arange(ymin, ymax + cell_size_deg, cell_size_deg)
    cells = []
    for x in x_edges[:-1]:
        for y in y_edges[:-1]:
            cells.append((x, y, x + cell_size_deg, y + cell_size_deg))
    return cells

_GLOBAL_GRID_GDF = None
def _grid_pool_init(grid_gdf):
    global _GLOBAL_GRID_GDF
    _GLOBAL_GRID_GDF = grid_gdf

def _process_chunk_indexed(geodata_chunk: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    _ = geodata_chunk.sindex
    _ = _GLOBAL_GRID_GDF.sindex
    return gpd.sjoin(geodata_chunk, _GLOBAL_GRID_GDF, how="left", predicate="intersects")

def calculate_rows_per_chunk(n: int, max_memory_mb: int = 512) -> int:
    # crude estimate ~1 KB/row
    rows_per_chunk = int(max_memory_mb / 0.001)
    return max(1, min(rows_per_chunk, n))

def assign_geocodes_to_grid(geodata: gpd.GeoDataFrame, meters_cell: int, max_workers: int) -> gpd.GeoDataFrame:
    # meters→degrees approx for WGS84 geocodes
    meters_per_degree = 111_320.0
    deg = meters_cell / meters_per_degree
    grid_cells = create_grid(geodata, deg)
    grid_gdf = gpd.GeoDataFrame(
        {'grid_cell': range(len(grid_cells)),
         'geometry': [box(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax in grid_cells]},
        geometry='geometry', crs=geodata.crs
    )
    _ = grid_gdf.sindex
    log_to_gui(log_widget, f"Assigning geocodes to {len(grid_cells):,} grid cells…")

    rows_per_chunk = calculate_rows_per_chunk(len(geodata))
    total_chunks = math.ceil(len(geodata)/rows_per_chunk)
    results = []
    with multiprocessing.get_context("spawn").Pool(processes=max_workers,
                                                   initializer=_grid_pool_init,
                                                   initargs=(grid_gdf,)) as pool:
        futures = []
        for i in range(0, len(geodata), rows_per_chunk):
            chunk = geodata.iloc[i:i+rows_per_chunk]
            futures.append(pool.apply_async(_process_chunk_indexed, (chunk,)))

        started_at = time.time()
        last_ping  = started_at
        done_count = 0

        for f in futures:
            part = f.get()
            results.append(part)
            done_count += 1

            now = time.time()
            if (now - last_ping) >= HEARTBEAT_SECS or done_count == total_chunks:
                elapsed = now - started_at
                pct = (done_count / total_chunks) * 100 if total_chunks else 100.0
                eta = "?"
                if done_count:
                    est_total = elapsed / done_count * total_chunks
                    eta_ts = datetime.now() + timedelta(seconds=max(0.0, est_total - elapsed))
                    eta = eta_ts.strftime("%H:%M:%S")
                    dd = (eta_ts.date() - datetime.now().date()).days
                    if dd > 0: eta += f" (+{dd}d)"
                log_to_gui(log_widget, f"[grid-assign] {done_count}/{total_chunks} chunks (~{pct:.2f}%) … ETA {eta}")
                last_ping = now

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

# intersection helpers
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

# multiprocessing (spawn) for intersections
_POOL_ASSETS = None
_POOL_TYPES  = None
def _intersect_pool_init(asset_df, geom_types):
    global _POOL_ASSETS, _POOL_TYPES
    _POOL_ASSETS = asset_df
    _POOL_TYPES  = geom_types

def _intersection_worker(args):
    idx, geocode_chunk = args
    try:
        _ = geocode_chunk.sindex
        parts = []
        for gt in _POOL_TYPES:
            res = intersection_with_geocode_data(_POOL_ASSETS, geocode_chunk, gt)
            if not res.empty: parts.append(res)
        if not parts:
            return (idx, 0, None, None)
        out = pd.concat(parts, ignore_index=True)
        return (idx, len(out), out, None)
    except Exception as e:
        return (idx, 0, None, str(e))

def intersect_assets_geocodes(asset_data, geocode_data, cell_size_m, max_workers):
    # tag geocodes with coarse grid to improve locality / chunking
    log_to_gui(log_widget, "Creating analysis grid + tagging geocodes.")
    geocode_tagged = assign_geocodes_to_grid(geocode_data, cell_size_m, max_workers)

    # chunks
    chunks = make_spatial_chunks(geocode_tagged, max_workers, multiplier=4)
    total_chunks = len(chunks)
    log_to_gui(log_widget, f"Intersecting in {total_chunks} chunks with {max_workers} workers. Heartbeat every {HEARTBEAT_SECS}s.")
    update_progress(35)

    geom_types = asset_data.geometry.geom_type.unique().tolist()
    out_parts  = []

    with multiprocessing.get_context("spawn").Pool(processes=max_workers,
                                                   initializer=_intersect_pool_init,
                                                   initargs=(asset_data, geom_types)) as pool:
        futures = [pool.apply_async(_intersection_worker, ((i, ch),)) for i, ch in enumerate(chunks, start=1)]
        started_at = time.time()
        last_ping  = started_at
        done_count = 0

        for f in futures:
            idx, nrows, res, err = f.get()
            done_count += 1
            if err:
                log_to_gui(log_widget, f"[{done_count}/{total_chunks}] Error: {err}")
            elif nrows:
                out_parts.append(res)

            now = time.time()
            if (now - last_ping) >= HEARTBEAT_SECS or done_count == total_chunks:
                elapsed = now - started_at
                pct = (done_count / total_chunks) * 100 if total_chunks else 100.0
                eta = "?"
                if done_count:
                    est_total = elapsed / done_count * total_chunks
                    eta_ts = datetime.now() + timedelta(seconds=max(0.0, est_total - elapsed))
                    eta = eta_ts.strftime("%H:%M:%S")
                    dd = (eta_ts.date() - datetime.now().date()).days
                    if dd > 0: eta += f" (+{dd}d)"
                log_to_gui(log_widget, f"[intersect] {done_count}/{total_chunks} chunks (~{pct:.2f}%) … ETA {eta}")
                update_progress(35.0 + (done_count / max(total_chunks,1)) * 15.0)
                last_ping = now

    if not out_parts:
        return gpd.GeoDataFrame(geometry=[], crs=geocode_data.crs)
    return gpd.GeoDataFrame(pd.concat(out_parts, ignore_index=True), crs=geocode_data.crs)

# ----------------------------
# PROCESS: build tbl_stacked
# ----------------------------
def process_tbl_stacked(cfg: configparser.ConfigParser,
                        working_epsg: str,
                        cell_size_m: int,
                        max_workers: int):
    log_to_gui(log_widget, "Building analysis table (tbl_stacked)…")
    update_progress(10)

    # Read inputs
    assets   = read_parquet_or_empty("tbl_asset_object")
    geocodes = read_parquet_or_empty("tbl_geocode_object")
    groups   = read_parquet_or_empty("tbl_asset_group")

    if assets.empty or geocodes.empty:
        log_to_gui(log_widget, "ERROR: Missing assets or geocodes parquet; aborting stacked build.")
        write_parquet("tbl_stacked", gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{working_epsg}"))
        return

    # Ensure CRS
    if assets.crs is None:   assets.set_crs(f"EPSG:{working_epsg}", inplace=True)
    if geocodes.crs is None: geocodes.set_crs(f"EPSG:{working_epsg}", inplace=True)

    # Merge asset-group attributes (keep only present columns)
    if not groups.empty:
        cols = ['id','name_gis_assetgroup','total_asset_objects','importance',
                'susceptibility','sensitivity','sensitivity_code','sensitivity_description']
        keep = [c for c in cols if c in groups.columns]
        assets = assets.merge(groups[keep], left_on='ref_asset_group', right_on='id', how='left')

    update_progress(20)
    _ = assets.sindex; _ = geocodes.sindex

    # Workers
    if max_workers == 0:
        try:
            max_workers = multiprocessing.cpu_count()
            log_to_gui(log_widget, f"Number of workers determined by system: {max_workers}")
        except NotImplementedError:
            max_workers = 4
    else:
        log_to_gui(log_widget, f"Number of workers set in config: {max_workers}")

    stacked = intersect_assets_geocodes(assets, geocodes, cell_size_m, max_workers)
    if stacked.empty:
        log_to_gui(log_widget, "No intersections; tbl_stacked is empty.")
        write_parquet("tbl_stacked", gpd.GeoDataFrame(geometry=[], crs=geocodes.crs))
        return

    # Keep tidy
    stacked.drop(columns=['index_right','geometry_wkb','geometry_wkb_1','process'], errors='ignore', inplace=True)

    write_parquet("tbl_stacked", stacked)
    log_to_gui(log_widget, "tbl_stacked saved.")
    update_progress(50)

# ----------------------------
# PROCESS: flatten to tbl_flat (min/max + codes + area)
# ----------------------------
def normalize_area_epsg(raw: str) -> str:
    v = (raw or "").strip().upper()
    if v.startswith("EPSG:"):
        v = v.split(":",1)[1]
    try:
        code = int(v)
        if code in (4326, 4258):  # geographic -> not for area
            return "EPSG:3035"
        return f"EPSG:{code}"
    except Exception:
        return "EPSG:3035"

def flatten_tbl_stacked(config_file: Path, working_epsg: str):
    log_to_gui(log_widget, "Building presentation table (tbl_flat)…")

    stacked = read_parquet_or_empty("tbl_stacked")
    log_to_gui(log_widget, f"tbl_stacked rows: {len(stacked):,}")

    if stacked.empty:
        empty_cols = [
            'ref_geocodegroup','name_gis_geocodegroup','code',
            'importance_min','importance_max','importance_code_min','importance_description_min','importance_code_max','importance_description_max',
            'sensitivity_min','sensitivity_max','sensitivity_code_min','sensitivity_description_min','sensitivity_code_max','sensitivity_description_max',
            'susceptibility_min','susceptibility_max','susceptibility_code_min','susceptibility_description_min','susceptibility_code_max','susceptibility_description_max',
            'asset_group_names','asset_groups_total','area_m2','assets_overlap_total','geometry'
        ]
        write_parquet("tbl_flat", gpd.GeoDataFrame(columns=empty_cols, geometry='geometry', crs=f"EPSG:{working_epsg}"))
        return

    # Ensure CRS
    if stacked.crs is None:
        stacked.set_crs(f"EPSG:{working_epsg}", inplace=True)

    # Numeric prep (no coercion to 0)
    bases = ["importance","sensitivity","susceptibility"]
    for b in bases:
        if b in stacked.columns:
            stacked[b] = pd.to_numeric(stacked[b], errors="coerce")
        else:
            stacked[b] = pd.Series(pd.NA, index=stacked.index, dtype="Float64")

    # --- groupby code ---
    keys = ["code"]
    # numeric min/max
    gnum = stacked.groupby(keys, dropna=False).agg({
        "importance": ["min","max"],
        "sensitivity": ["min","max"],
        "susceptibility": ["min","max"],
    })
    gnum.columns = [f"{c}_{s}" for c,s in gnum.columns]

    # meta
    gmeta = stacked.groupby(keys, dropna=False).agg({
        "ref_geocodegroup": "first",
        "name_gis_geocodegroup": "first",
        "geometry": "first",
        "ref_asset_group": pd.Series.nunique,
        "name_gis_assetgroup": (lambda s: ", ".join(pd.Series(s).dropna().astype(str).unique()))
    }).rename(columns={"ref_asset_group":"asset_groups_total", "name_gis_assetgroup":"asset_group_names"})

    # overlap count
    goverlap = stacked.groupby(keys, dropna=False).size().to_frame("assets_overlap_total")

    tbl_flat = pd.concat([gnum, gmeta, goverlap], axis=1).reset_index()
    tbl_flat = gpd.GeoDataFrame(tbl_flat, geometry="geometry", crs=stacked.crs)

    # --- derive A..E codes + descriptions from config ---
    ranges_map, desc_map, _order = read_class_ranges(config_file)

    def add_code_desc(df: pd.DataFrame, base: str) -> pd.DataFrame:
        cmin = df[f"{base}_min"].apply(lambda v: map_num_to_code(v, ranges_map)).astype("string")
        cmax = df[f"{base}_max"].apply(lambda v: map_num_to_code(v, ranges_map)).astype("string")
        df[f"{base}_code_min"] = cmin
        df[f"{base}_code_max"] = cmax
        df[f"{base}_description_min"] = df[f"{base}_code_min"].apply(lambda k: desc_map.get(k, None))
        df[f"{base}_description_max"] = df[f"{base}_code_max"].apply(lambda k: desc_map.get(k, None))
        return df

    for b in bases:
        tbl_flat = add_code_desc(tbl_flat, b)

    # --- area once per tile ---
    try:
        cfg = read_config(base_dir() / "system" / "config.ini")
        area_epsg = normalize_area_epsg(cfg["DEFAULT"].get("area_projection_epsg","3035"))
    except Exception:
        area_epsg = "EPSG:3035"

    try:
        metric = tbl_flat.to_crs(area_epsg)
        tbl_flat["area_m2"] = metric.geometry.area.astype("float64").round().astype("Int64")
    except Exception as e:
        log_to_gui(log_widget, f"Area computation failed in {area_epsg}: {e}; using EPSG:3035")
        metric = tbl_flat.to_crs("EPSG:3035")
        tbl_flat["area_m2"] = metric.geometry.area.astype("float64").round().astype("Int64")

    # stable column order
    preferred = [
        'ref_geocodegroup','name_gis_geocodegroup','code',
        'importance_min','importance_max','importance_code_min','importance_description_min','importance_code_max','importance_description_max',
        'sensitivity_min','sensitivity_max','sensitivity_code_min','sensitivity_description_min','sensitivity_code_max','sensitivity_description_max',
        'susceptibility_min','susceptibility_max','susceptibility_code_min','susceptibility_description_min','susceptibility_code_max','susceptibility_description_max',
        'asset_group_names','asset_groups_total','area_m2','assets_overlap_total','geometry'
    ]
    for c in preferred:
        if c not in tbl_flat.columns: tbl_flat[c] = pd.NA
    tbl_flat = tbl_flat[preferred]

    write_parquet("tbl_flat", tbl_flat)

    # backfill area to stacked
    try:
        stacked = read_parquet_or_empty("tbl_stacked")
        if not stacked.empty:
            join = tbl_flat[["code","area_m2"]]
            merged = stacked.merge(join, on="code", how="left", suffixes=("", "_flat"))
            if "area_m2_flat" in merged.columns:
                merged["area_m2"] = merged["area_m2_flat"]
                merged.drop(columns=["area_m2_flat"], inplace=True)
            merged = gpd.GeoDataFrame(merged, geometry=stacked.geometry.name, crs=stacked.crs)
            write_parquet("tbl_stacked", merged)
            log_to_gui(log_widget, "Backfilled area_m2 to tbl_stacked.")
    except Exception as e:
        log_to_gui(log_widget, f"Backfill to tbl_stacked failed: {e}")

    update_progress(90)

# ----------------------------
# Top-level process
# ----------------------------
def process_all(config_file: Path):
    try:
        cfg = read_config(config_file)
        working_epsg = str(cfg["DEFAULT"].get("workingprojection_epsg","4326")).strip()
        method = str(cfg["DEFAULT"].get("method","cpu")).strip().lower()
        max_workers = int(cfg["DEFAULT"].get("max_workers","0"))
        cell_size   = int(cfg["DEFAULT"].get("cell_size","18000"))

        cleanup_outputs(); update_progress(5)

        process_tbl_stacked(cfg, working_epsg, cell_size, max_workers)
        flatten_tbl_stacked(config_file, working_epsg); update_progress(95)

        log_to_gui(log_widget, "COMPLETED."); update_progress(100)
    except Exception as e:
        log_to_gui(log_widget, f"Error during processing: {e}")
        raise

# ----------------------------
# Entrypoint (GUI)
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process stacked/flat GeoParquet with A..E categorisation")
    parser.add_argument('--original_working_directory', required=False, help='Path to running folder')
    args = parser.parse_args()
    original_working_directory = args.original_working_directory or os.getcwd()
    if "system" in os.path.basename(original_working_directory).lower():
        original_working_directory = os.path.abspath(os.path.join(original_working_directory, os.pardir))

    cfg_path = Path(original_working_directory) / "system" / "config.ini"
    cfg = read_config(cfg_path)

    # Heartbeat from config
    try:
        HEARTBEAT_SECS = int(cfg['DEFAULT'].get('heartbeat_secs', str(HEARTBEAT_SECS)))
    except Exception:
        pass

    ttk_theme = cfg['DEFAULT'].get('ttk_bootstrap_theme', 'flatly')

    root = tb.Window(themename=ttk_theme)
    root.title("Process analysis & presentation (GeoParquet)")
    try:
        ico = base_dir() / "system_resources" / "mesa.ico"
        if ico.exists(): root.iconbitmap(str(ico))
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

    info = (f"Inputs: output/geoparquet/*.parquet  •  Heartbeat: {HEARTBEAT_SECS}s\n"
            "Flatten = min/max + A..E from config; area once per tile (backfilled to stacked).")
    tk.Label(root, text=info, wraplength=680, justify="left").pack(padx=10, pady=10)

    def _run():
        process_all(cfg_path)

    btn_frame = tk.Frame(root); btn_frame.pack(pady=6)
    tb.Button(btn_frame, text="Process", bootstyle=PRIMARY,
              command=lambda: threading.Thread(target=_run, daemon=True).start()).pack(side=tk.LEFT, padx=5)
    tb.Button(btn_frame, text="Exit", bootstyle=WARNING, command=root.destroy).pack(side=tk.LEFT, padx=5)

    log_to_gui(log_widget, "Opened processing UI (GeoParquet only).")
    root.mainloop()
