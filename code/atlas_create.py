# -*- coding: utf-8 -*-
# atlas_create.py — create/import atlas tiles (GeoParquet only)
# - Stable BASE_DIR resolution (env/CLI/script/CWD)
# - Flat config preferred (<base>/config.ini), fallback to <base>/system/config.ini
# - Uses DEFAULT.parquet_folder (defaults to output/geoparquet)
# - Writes tbl_atlas.parquet alongside other geoparquet outputs
# - Preserves existing logic; improves robustness

import locale
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except Exception:
    try:
        locale.setlocale(locale.LC_ALL, '')
    except Exception:
        pass

import os
import glob
import argparse
import datetime
import threading
import configparser
import math
from pathlib import Path

import tkinter as tk
from tkinter import scrolledtext
try:
    import ttkbootstrap as tb
    from ttkbootstrap.constants import PRIMARY, WARNING
except Exception:
    tb = None
    PRIMARY = WARNING = None

import geopandas as gpd
from shapely.geometry import box

# ----------------------------
# Globals / UI
# ----------------------------
log_widget = None
progress_var = None
progress_label = None

BASE_DIR: Path = Path(".").resolve()
_CFG: configparser.ConfigParser | None = None
_CFG_PATH: Path | None = None
original_working_directory: str | None = None

# Runtime settings (loaded from config)
atlas_lon_size_km: float = 10.0
atlas_lat_size_km: float = 10.0
atlas_overlap_percent: float = 10.0
config_file: str = ""  # path used by increment_stat_value

# UI option state
tile_mode_var = None
tile_lon_var = None
tile_lat_var = None
tile_overlap_var = None
tile_count_var = None
tile_tolerance_var = None
custom_option_entries: list = []
count_option_entries: list = []

# ----------------------------
# Base dir / Config helpers
# ----------------------------
def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

def _has_config_at(root: Path) -> bool:
    return _exists(root / "config.ini") or _exists(root / "system" / "config.ini")

def find_base_dir(cli_workdir: str | None = None) -> Path:
    """
    Choose a canonical project base folder that actually contains config.
    Priority:
      1) env MESA_BASE_DIR
      2) --original_working_directory (CLI)
      3) script folder & its parents
      4) CWD and CWD/code
    """
    candidates: list[Path] = []
    env_base = os.environ.get("MESA_BASE_DIR")
    if env_base:
        candidates.append(Path(env_base))
    if cli_workdir:
        candidates.append(Path(cli_workdir))

    here = Path(__file__).resolve()
    candidates += [here.parent, here.parent.parent, here.parent.parent.parent]

    cwd = Path(os.getcwd())
    candidates += [cwd, cwd / "code"]

    seen = set()
    uniq = []
    for c in candidates:
        try:
            r = c.resolve()
        except Exception:
            r = c
        if r not in seen:
            seen.add(r)
            uniq.append(r)

    for c in uniq:
        if _has_config_at(c):
            return c

    if here.parent.name.lower() == "system":
        return here.parent.parent
    return here.parent

def _ensure_cfg() -> configparser.ConfigParser:
    """Load config, preferring <base>/config.ini and falling back to <base>/system/config.ini."""
    global _CFG, _CFG_PATH
    if _CFG is not None:
        return _CFG

    cfg = configparser.ConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
    flat = BASE_DIR / "config.ini"
    legacy = BASE_DIR / "system" / "config.ini"
    loaded = False
    if flat.exists():
        try:
            cfg.read(flat, encoding="utf-8")
            _CFG_PATH = flat
            loaded = True
        except Exception:
            pass
    if not loaded and legacy.exists():
        try:
            cfg.read(legacy, encoding="utf-8")
            _CFG_PATH = legacy
            loaded = True
        except Exception:
            pass
    if "DEFAULT" not in cfg:
        cfg["DEFAULT"] = {}

    # sensible defaults
    d = cfg["DEFAULT"]
    d.setdefault("parquet_folder", "output/geoparquet")
    d.setdefault("ttk_bootstrap_theme", "flatly")
    d.setdefault("workingprojection_epsg", "4326")
    d.setdefault("input_folder_atlas", "input/atlas")
    d.setdefault("atlas_lon_size_km", "10")
    d.setdefault("atlas_lat_size_km", "10")
    d.setdefault("atlas_overlap_percent", "10")

    _CFG = cfg
    return _CFG

def _abs_path_like(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (BASE_DIR / p).resolve()

# ----------------------------
# Logging / Progress
# ----------------------------
def log_to_gui(widget, message: str):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted = f"{timestamp} - {message}"
    try:
        if widget and widget.winfo_exists():
            widget.insert(tk.END, formatted + "\n")
            widget.see(tk.END)
    except Exception:
        pass
    try:
        with open(BASE_DIR / "log.txt", "a", encoding="utf-8", errors="replace") as f:
            f.write(formatted + "\n")
    except Exception:
        pass
    if widget is None:
        print(formatted, flush=True)

def update_progress(new_value):
    try:
        v = max(0.0, min(100.0, float(new_value)))
        if progress_var is not None:
            progress_var.set(v)
        if progress_label is not None:
            progress_label.config(text=f"{int(v)}%")
    except Exception:
        pass

def update_tile_mode_ui():
    """Enable/disable option fields based on the selected tile mode."""
    mode = tile_mode_var.get() if tile_mode_var is not None else "config"
    custom_state = tk.NORMAL if mode == "custom" else tk.DISABLED
    count_state = tk.NORMAL if mode == "count" else tk.DISABLED
    for widget in custom_option_entries:
        try:
            widget.configure(state=custom_state)
        except Exception:
            pass
    for widget in count_option_entries:
        try:
            widget.configure(state=count_state)
        except Exception:
            pass

# ----------------------------
# GeoParquet helpers
# ----------------------------
def geoparquet_dir() -> Path:
    cfg = _ensure_cfg()
    sub = cfg["DEFAULT"].get("parquet_folder", "output/geoparquet")
    p = BASE_DIR / sub
    p.mkdir(parents=True, exist_ok=True)
    return p

def atlas_parquet_path() -> Path:
    return geoparquet_dir() / "tbl_atlas.parquet"

def read_flat_parquet() -> gpd.GeoDataFrame:
    p = geoparquet_dir() / "tbl_flat.parquet"
    if not p.exists():
        log_to_gui(log_widget, f"Missing tbl_flat.parquet at {p}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    try:
        gdf = gpd.read_parquet(p)
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        return gdf
    except Exception as e:
        log_to_gui(log_widget, f"Failed reading tbl_flat.parquet: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

def write_atlas_parquet(gdf: gpd.GeoDataFrame):
    p = atlas_parquet_path()
    gdf.to_parquet(p, index=False)
    log_to_gui(log_widget, f"Saved atlas GeoParquet -> {p} (rows={len(gdf)})")

# ----------------------------
# App helpers
# ----------------------------
def read_config(file_name):
    cfg = configparser.ConfigParser()
    try:
        cfg.read(file_name, encoding="utf-8")
    except Exception:
        cfg.read(file_name)
    return cfg

def close_application(root):
    root.destroy()

# ----------------------------
# Core logic
# ----------------------------
def run_create_atlas(log_widget_, progress_var_):
    main_create_atlas(log_widget_, progress_var_)

def filter_and_update_atlas_geometries(atlas_geometries, tbl_flat):
    log_to_gui(log_widget, "Filtering and updating atlas geometries...")
    atlas_gdf = gpd.GeoDataFrame(
        atlas_geometries,
        columns=['id', 'name_gis', 'title_user', 'geom', 'description',
                 'image_name_1', 'image_desc_1', 'image_name_2', 'image_desc_2']
    )
    atlas_gdf.set_geometry('geom', inplace=True)
    try:
        # Ensure CRS match
        if atlas_gdf.crs is None and tbl_flat.crs is not None:
            atlas_gdf.set_crs(tbl_flat.crs, inplace=True)
        elif atlas_gdf.crs != tbl_flat.crs and tbl_flat.crs is not None:
            atlas_gdf = atlas_gdf.to_crs(tbl_flat.crs)
    except Exception:
        pass

    try:
        filtered_indices = atlas_gdf.geometry.apply(lambda geom: tbl_flat.intersects(geom).any())
    except Exception:
        # Fallback: keep all if intersection fails
        filtered_indices = [True] * len(atlas_gdf)

    intersecting_geometries = atlas_gdf[filtered_indices].copy()
    id_counter = 1
    for idx in intersecting_geometries.index:
        intersecting_geometries.loc[idx, 'name_gis'] = f'atlas_{id_counter:03d}'
        intersecting_geometries.loc[idx, 'title_user'] = f'Map title for {id_counter:03d}'
        id_counter += 1

    intersecting_geometries = intersecting_geometries.rename(columns={'geom': 'geometry'}).set_geometry('geometry')
    return intersecting_geometries

def generate_atlas_geometries(tbl_flat, lon_km, lat_km, overlap_pct):
    log_to_gui(log_widget, "Generating atlas geometries...")
    # crude degree conversion; acceptable for page tiling use
    lon_size_deg = float(lon_km) / 111.0
    lat_size_deg = float(lat_km) / 111.0
    overlap = float(overlap_pct) / 100.0

    if tbl_flat is None or tbl_flat.empty:
        return []

    try:
        minx, miny, maxx, maxy = tbl_flat.total_bounds
    except Exception:
        return []

    atlas_geometries = []
    id_counter = 1
    y = miny
    step_y = max(1e-9, lat_size_deg * (1.0 - overlap))
    step_x = max(1e-9, lon_size_deg * (1.0 - overlap))

    while y < maxy:
        x = minx
        while x < maxx:
            geom = box(x, y, x + lon_size_deg, y + lat_size_deg)
            atlas_geometries.append({
                'id': id_counter,
                'name_gis': f'atlas{id_counter:03d}',
                'title_user': f'Map title for {id_counter:03d}',
                'geom': geom,
                'description': '',
                'image_name_1': '',
                'image_desc_1': '',
                'image_name_2': '',
                'image_desc_2': ''
            })
            id_counter += 1
            x += step_x
        y += step_y
    return atlas_geometries

def generate_atlas_geometries_by_count(tbl_flat, tile_count: int, tolerance_pct: float):
    log_to_gui(log_widget, f"Generating atlas geometries for {tile_count} tiles (tolerance {tolerance_pct}%).")
    if tbl_flat is None or tbl_flat.empty:
        log_to_gui(log_widget, "No data available to derive atlas extent.")
        return []

    try:
        minx, miny, maxx, maxy = tbl_flat.total_bounds
    except Exception:
        log_to_gui(log_widget, "Failed to compute bounds from data.")
        return []

    width = maxx - minx
    height = maxy - miny
    if width <= 0:
        width = 1e-6
        minx -= width / 2.0
        maxx += width / 2.0
    if height <= 0:
        height = 1e-6
        miny -= height / 2.0
        maxy += height / 2.0

    tile_count = max(2, int(tile_count))
    tolerance_pct = max(0.0, float(tolerance_pct))

    pad_x = width * (tolerance_pct / 100.0) / 2.0
    pad_y = height * (tolerance_pct / 100.0) / 2.0

    expanded_minx = minx - pad_x
    expanded_maxx = maxx + pad_x
    expanded_miny = miny - pad_y
    expanded_maxy = maxy + pad_y

    total_width = expanded_maxx - expanded_minx
    total_height = expanded_maxy - expanded_miny

    if total_width <= 0 or total_height <= 0:
        log_to_gui(log_widget, "Invalid atlas extent after tolerance adjustment.")
        return []

    ratio = total_width / total_height if total_height > 0 else 1.0
    ratio = max(ratio, 1e-6)

    cols = max(1, math.ceil(math.sqrt(tile_count * ratio)))
    rows = max(1, math.ceil(tile_count / cols))
    while cols * rows < tile_count:
        rows += 1

    tile_width = total_width / cols
    tile_height = total_height / rows

    atlas_geometries = []
    atlas_total = cols * rows
    log_to_gui(
        log_widget,
        f"Tile grid: {cols} columns x {rows} rows (total {atlas_total} tiles). Approx tile size ~{tile_width * 111.0:.2f} x {tile_height * 111.0:.2f} km."
    )

    id_counter = 1
    for row in range(rows):
        y0 = expanded_miny + row * tile_height
        y1 = expanded_miny + (row + 1) * tile_height
        if row == rows - 1:
            y1 = expanded_maxy
        for col in range(cols):
            x0 = expanded_minx + col * tile_width
            x1 = expanded_minx + (col + 1) * tile_width
            if col == cols - 1:
                x1 = expanded_maxx
            geom = box(x0, y0, x1, y1)
            atlas_geometries.append({
                'id': id_counter,
                'name_gis': f'atlas{id_counter:03d}',
                'title_user': f'Map title for {id_counter:03d}',
                'geom': geom,
                'description': '',
                'image_name_1': '',
                'image_desc_1': '',
                'image_name_2': '',
                'image_desc_2': ''
            })
            id_counter += 1

    return atlas_geometries

def main_create_atlas(log_widget_, progress_var_):

    log_to_gui(log_widget_, "Starting atlas generation (GeoParquet).")
    update_progress(10)

    tbl_flat = read_flat_parquet()
    if tbl_flat.empty:
        log_to_gui(log_widget_, "tbl_flat.parquet empty or missing; aborting.")
        update_progress(100)
        return

    update_progress(30)

    mode = tile_mode_var.get() if tile_mode_var is not None else "config"
    mode = (mode or "config").lower()

    atlas_geometries = []

    if mode == "custom":
        try:
            lon_km = float(tile_lon_var.get())
            lat_km = float(tile_lat_var.get())
            overlap = float(tile_overlap_var.get())
        except Exception:
            log_to_gui(log_widget_, "Invalid custom tile size values; please enter numeric values.")
            update_progress(100)
            return
        if lon_km <= 0 or lat_km <= 0:
            log_to_gui(log_widget_, "Tile sizes must be positive numbers.")
            update_progress(100)
            return
        overlap = max(0.0, min(overlap, 90.0))
        log_to_gui(log_widget_, f"Tile mode: custom size {lon_km} km x {lat_km} km with {overlap}% overlap.")
        atlas_geometries = generate_atlas_geometries(tbl_flat, lon_km, lat_km, overlap)

    elif mode == "count":
        if tile_count_var is None:
            log_to_gui(log_widget_, "Tile count option unavailable in this context.")
            update_progress(100)
            return
        try:
            requested_tiles = int(tile_count_var.get())
        except Exception:
            log_to_gui(log_widget_, "Invalid tile count; please enter an integer value (minimum 2).")
            update_progress(100)
            return
        requested_tiles = max(2, requested_tiles)
        try:
            tolerance = float(tile_tolerance_var.get()) if tile_tolerance_var is not None else 0.0
        except Exception:
            log_to_gui(log_widget_, "Invalid tolerance percentage; please enter a number.")
            update_progress(100)
            return
        tolerance = max(0.0, tolerance)
        log_to_gui(log_widget_, f"Tile mode: distribute {requested_tiles} tiles across data extent (tolerance {tolerance}%).")
        atlas_geometries = generate_atlas_geometries_by_count(tbl_flat, requested_tiles, tolerance)
        if atlas_geometries:
            log_to_gui(log_widget_, f"Generated {len(atlas_geometries)} tiles to cover the dataset.")

    else:
        log_to_gui(
            log_widget_,
            f"Tile mode: config defaults ({atlas_lon_size_km} km x {atlas_lat_size_km} km, overlap {atlas_overlap_percent}%)."
        )
        atlas_geometries = generate_atlas_geometries(tbl_flat, atlas_lon_size_km, atlas_lat_size_km, atlas_overlap_percent)

    if not atlas_geometries:
        log_to_gui(log_widget_, "No atlas tiles were generated; aborting.")
        update_progress(100)
        return

    update_progress(60)

    updated = filter_and_update_atlas_geometries(atlas_geometries, tbl_flat)
    update_progress(80)

    if not updated.empty:
        log_to_gui(log_widget_, f"Atlas tiles intersecting data: {len(updated)}.")
        write_atlas_parquet(updated)
    else:
        empty = gpd.GeoDataFrame(
            columns=['id','name_gis','title_user','description',
                     'image_name_1','image_desc_1','image_name_2','image_desc_2','geometry'],
            geometry='geometry', crs=tbl_flat.crs
        )
        write_atlas_parquet(empty)
        log_to_gui(log_widget_, "No intersecting atlas tiles; wrote empty atlas table.")

    update_progress(100)
    log_to_gui(log_widget_, "COMPLETED: Atlas creation saved to GeoParquet.")
    try:
        increment_stat_value(config_file, 'mesa_stat_create_atlas', increment_value=1)
    except Exception:
        pass

def process_spatial_file(filepath, atlas_objects, atlas_id_counter, log_widget_):
    try:
        gdf = gpd.read_file(filepath)
        polys = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        # Optionally reproject to tbl_flat CRS if needed later; not required for import-only
        for _, row in polys.iterrows():
            atlas_objects.append({
                'id': atlas_id_counter,
                'name_gis': f'atlas{atlas_id_counter:03d}',
                'title_user': f'Map title for {atlas_id_counter:03d}',
                'geom': row.geometry,
                'description': '',
                'image_name_1': '',
                'image_desc_1': '',
                'image_name_2': '',
                'image_desc_2': ''
            })
            atlas_id_counter += 1
        log_to_gui(log_widget_, f"Processed file: {filepath}")
    except Exception as e:
        log_to_gui(log_widget_, f"Error processing file {filepath}: {e}")
    return atlas_id_counter

def import_atlas_objects(input_folder_atlas: Path, log_widget_, progress_var_):
    atlas_objects = []
    atlas_id_counter = 1
    file_patterns = ('*.shp', '*.gpkg', '*.parquet')  # allow parquet atlas polygons too

    files = []
    for pat in file_patterns:
        files.extend(input_folder_atlas.rglob(pat))
    total_files = len(files)

    progress_span = 70.0
    step = (progress_span / total_files) if total_files else progress_span
    log_to_gui(log_widget_, f"Working with imports… ({total_files} files)")
    update_progress(10)

    processed = 0
    for fp in files:
        try:
            if fp.suffix.lower() == ".parquet":
                # read parquet as vector (CRS preserved if present)
                gdf = gpd.read_parquet(fp)
                if not gdf.empty:
                    polys = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
                    for _, row in polys.iterrows():
                        atlas_objects.append({
                            'id': atlas_id_counter,
                            'name_gis': f'atlas{atlas_id_counter:03d}',
                            'title_user': f'Map title for {atlas_id_counter:03d}',
                            'geom': row.geometry,
                            'description': '',
                            'image_name_1': '',
                            'image_desc_1': '',
                            'image_name_2': '',
                            'image_desc_2': ''
                        })
                        atlas_id_counter += 1
                    log_to_gui(log_widget_, f"Processed file: {fp}")
            else:
                log_to_gui(log_widget_, f"Processing file: {fp}")
                atlas_id_counter = process_spatial_file(str(fp), atlas_objects, atlas_id_counter, log_widget_)
        except Exception as e:
            log_to_gui(log_widget_, f"Error processing file {fp}: {e}")
        finally:
            processed += 1
            update_progress(10 + processed * step)

    if atlas_objects:
        atlas_objects_gdf = gpd.GeoDataFrame(atlas_objects, geometry='geom').rename(columns={'geom': 'geometry'}).set_geometry('geometry')
    else:
        atlas_objects_gdf = gpd.GeoDataFrame(
            columns=['id','name_gis','title_user','description',
                     'image_name_1','image_desc_1','image_name_2','image_desc_2','geometry'],
            geometry='geometry'
        )

    update_progress(100)
    log_to_gui(log_widget_, f"Total atlas polygons added: {atlas_id_counter - 1}")

    try:
        increment_stat_value(config_file, 'mesa_stat_import_atlas', increment_value=1)
    except Exception:
        pass

    return atlas_objects_gdf

def run_import_atlas(input_folder_atlas: Path, log_widget_, progress_var_):
    log_to_gui(log_widget_, "Starting atlas import (GeoParquet)…")
    atlas_objects_gdf = import_atlas_objects(input_folder_atlas, log_widget_, progress_var_)
    write_atlas_parquet(atlas_objects_gdf)
    if atlas_objects_gdf.empty:
        log_to_gui(log_widget_, "No atlas objects to export (empty table written).")
    log_to_gui(log_widget_, "COMPLETED: Atlas polygons imported (GeoParquet).")
    update_progress(100)

def increment_stat_value(cfg_path: str, stat_name: str, increment_value: int):
    if not cfg_path or not os.path.isfile(cfg_path):
        log_to_gui(log_widget, f"Configuration file {cfg_path} not found.")
        return
    try:
        with open(cfg_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        updated = False
        for i, line in enumerate(lines):
            if line.strip().startswith(f'{stat_name} ='):
                parts = line.split('=', 1)
                if len(parts) == 2:
                    current_value = parts[1].strip()
                    try:
                        new_value = int(current_value) + int(increment_value)
                        lines[i] = f"{stat_name} = {new_value}\n"
                        updated = True
                        break
                    except ValueError:
                        log_to_gui(log_widget, f"Error: Current value of {stat_name} is not an integer.")
                        return
        if updated:
            with open(cfg_path, 'w', encoding='utf-8', errors='replace') as f:
                f.writelines(lines)
    except Exception as e:
        log_to_gui(log_widget, f"Failed to update stat in config: {e}")

# ----------------------------
# Entrypoint (GUI)
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create/Import atlas tiles → GeoParquet")
    parser.add_argument('--original_working_directory', required=False, help='Path to running folder')
    args = parser.parse_args()

    # Resolve BASE_DIR (preferred) and keep original_working_directory for compatibility/logs
    BASE_DIR = find_base_dir(args.original_working_directory)
    original_working_directory = str(BASE_DIR)
    _ensure_cfg()

    # Load settings
    cfg = _ensure_cfg()
    d = cfg["DEFAULT"]
    ttk_bootstrap_theme = d.get("ttk_bootstrap_theme", "flatly")
    workingprojection_epsg = d.get("workingprojection_epsg", "4326")
    input_folder_atlas = _abs_path_like(d.get("input_folder_atlas", "input/atlas"))
    try:
        atlas_lon_size_km = float(d.get("atlas_lon_size_km", "10"))
        atlas_lat_size_km = float(d.get("atlas_lat_size_km", "10"))
        atlas_overlap_percent = float(d.get("atlas_overlap_percent", "10"))
    except Exception:
        atlas_lon_size_km, atlas_lat_size_km, atlas_overlap_percent = 10.0, 10.0, 10.0

    # Path to the config file we actually use (for increment_stat_value)
    config_file = str(_CFG_PATH if _CFG_PATH is not None else (BASE_DIR / "config.ini"))

    # GUI
    if tb is not None:
        try:
            root = tb.Window(themename=ttk_bootstrap_theme)
        except Exception:
            root = tb.Window(themename="flatly")
    else:
        root = tk.Tk()

    root.title("Create atlas tiles")

    # Optional icon
    try:
        ico = BASE_DIR / "system_resources" / "mesa.ico"
        if ico.exists() and hasattr(root, "iconbitmap"):
            root.iconbitmap(str(ico))
    except Exception:
        pass

    tile_mode_var = tk.StringVar(value="config")
    tile_lon_var = tk.StringVar(value=f"{atlas_lon_size_km:.2f}")
    tile_lat_var = tk.StringVar(value=f"{atlas_lat_size_km:.2f}")
    tile_overlap_var = tk.StringVar(value=f"{atlas_overlap_percent:.2f}")
    tile_count_var = tk.StringVar(value="4")
    tile_tolerance_var = tk.StringVar(value="5")
    custom_option_entries = []
    count_option_entries = []

    if tb is not None:
        options_frame = tb.LabelFrame(root, text="Tile generation options", bootstyle="info")
    else:
        options_frame = tk.LabelFrame(root, text="Tile generation options")
    options_frame.pack(padx=10, pady=(10, 4), fill=tk.X)

    if tb is not None:
        rb_config = tb.Radiobutton(options_frame, text="Use config.ini defaults", variable=tile_mode_var,
                                   value="config", command=update_tile_mode_ui, bootstyle="secondary")
    else:
        rb_config = tk.Radiobutton(options_frame, text="Use config.ini defaults", variable=tile_mode_var,
                                   value="config", command=update_tile_mode_ui)
    rb_config.grid(row=0, column=0, columnspan=4, sticky="w")

    if tb is not None:
        rb_custom = tb.Radiobutton(options_frame, text="Custom tile size (km)", variable=tile_mode_var,
                                   value="custom", command=update_tile_mode_ui, bootstyle="secondary")
    else:
        rb_custom = tk.Radiobutton(options_frame, text="Custom tile size (km)", variable=tile_mode_var,
                                   value="custom", command=update_tile_mode_ui)
    rb_custom.grid(row=1, column=0, columnspan=4, sticky="w")

    if tb is not None:
        lbl_width = tb.Label(options_frame, text="Width (lon km):")
    else:
        lbl_width = tk.Label(options_frame, text="Width (lon km):")
    lbl_width.grid(row=2, column=0, sticky="w", padx=(20, 4))

    entry_lon = (tb.Entry(options_frame, textvariable=tile_lon_var, width=8)
                 if tb is not None else tk.Entry(options_frame, textvariable=tile_lon_var, width=8))
    entry_lon.grid(row=2, column=1, sticky="w")

    if tb is not None:
        lbl_height = tb.Label(options_frame, text="Height (lat km):")
    else:
        lbl_height = tk.Label(options_frame, text="Height (lat km):")
    lbl_height.grid(row=2, column=2, sticky="w", padx=(10, 4))

    entry_lat = (tb.Entry(options_frame, textvariable=tile_lat_var, width=8)
                 if tb is not None else tk.Entry(options_frame, textvariable=tile_lat_var, width=8))
    entry_lat.grid(row=2, column=3, sticky="w")

    if tb is not None:
        lbl_overlap = tb.Label(options_frame, text="Overlap % (between custom tiles):")
    else:
        lbl_overlap = tk.Label(options_frame, text="Overlap % (between custom tiles):")
    lbl_overlap.grid(row=3, column=0, sticky="w", padx=(20, 4))

    entry_overlap = (tb.Entry(options_frame, textvariable=tile_overlap_var, width=8)
                     if tb is not None else tk.Entry(options_frame, textvariable=tile_overlap_var, width=8))
    entry_overlap.grid(row=3, column=1, sticky="w")
    custom_option_entries.extend([entry_lon, entry_lat, entry_overlap])

    if tb is not None:
        rb_count = tb.Radiobutton(options_frame, text="Distribute fixed number of tiles", variable=tile_mode_var,
                                  value="count", command=update_tile_mode_ui, bootstyle="secondary")
    else:
        rb_count = tk.Radiobutton(options_frame, text="Distribute fixed number of tiles", variable=tile_mode_var,
                                  value="count", command=update_tile_mode_ui)
    rb_count.grid(row=4, column=0, columnspan=4, sticky="w", pady=(6, 0))

    if tb is not None:
        lbl_tiles = tb.Label(options_frame, text="Tiles:")
    else:
        lbl_tiles = tk.Label(options_frame, text="Tiles:")
    lbl_tiles.grid(row=5, column=0, sticky="w", padx=(20, 4))

    entry_count = (tb.Entry(options_frame, textvariable=tile_count_var, width=8)
                   if tb is not None else tk.Entry(options_frame, textvariable=tile_count_var, width=8))
    entry_count.grid(row=5, column=1, sticky="w")

    if tb is not None:
        lbl_tol = tb.Label(options_frame, text="Padding % (expands extent):")
    else:
        lbl_tol = tk.Label(options_frame, text="Padding % (expands extent):")
    lbl_tol.grid(row=5, column=2, sticky="w", padx=(10, 4))

    entry_tol = (tb.Entry(options_frame, textvariable=tile_tolerance_var, width=8)
                 if tb is not None else tk.Entry(options_frame, textvariable=tile_tolerance_var, width=8))
    entry_tol.grid(row=5, column=3, sticky="w")
    count_option_entries.extend([entry_count, entry_tol])

    if tb is not None:
        note_label = tb.Label(
            options_frame,
            text="Overlap applies to custom tiles; padding expands the extent for tile-count mode.",
            bootstyle="secondary"
        )
    else:
        note_label = tk.Label(
            options_frame,
            text="Overlap applies to custom tiles; padding expands the extent for tile-count mode.",
            font=("Segoe UI", 9), fg="#555555"
        )
    note_label.grid(row=6, column=0, columnspan=4, sticky="w", padx=(20, 0), pady=(4, 0))

    tile_mode_var.trace_add("write", lambda *args: update_tile_mode_ui())
    update_tile_mode_ui()

    # Log widget
    log_widget = scrolledtext.ScrolledText(root, height=10)
    log_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Progress
    progress_frame = tk.Frame(root); progress_frame.pack(pady=5)
    progress_var = tk.DoubleVar(value=0.0)
    if tb is not None:
        progress_bar = tb.Progressbar(progress_frame, orient="horizontal", length=220,
                                      mode="determinate", variable=progress_var, bootstyle='info')
    else:
        from tkinter import ttk as _ttk
        progress_bar = _ttk.Progressbar(progress_frame, orient="horizontal", length=220,
                                        mode="determinate", variable=progress_var)
    progress_bar.pack(side=tk.LEFT)
    progress_label = tk.Label(progress_frame, text="0%", bg="light grey"); progress_label.pack(side=tk.LEFT, padx=6)

    # Info text
    info_label_text = ("Import or generate atlas geometries (GeoParquet only).\n"
                       "Choose a tile generation mode above. Previous atlas (tbl_atlas.parquet) will be replaced.")
    tk.Label(root, text=info_label_text, wraplength=620, justify="left").pack(padx=10, pady=10)

    # Buttons
    btn_frame = tk.Frame(root); btn_frame.pack(pady=6)

    def _spawn(fn):
        return lambda: threading.Thread(target=fn, daemon=True).start()

    if tb is not None:
        tb.Button(btn_frame, text="Import", bootstyle=PRIMARY,
                  command=_spawn(lambda: run_import_atlas(input_folder_atlas, log_widget, progress_var))).grid(row=0, column=0, padx=10, pady=5, sticky='ew')
        tb.Button(btn_frame, text="Create", bootstyle=PRIMARY,
                  command=_spawn(lambda: main_create_atlas(log_widget, progress_var))).grid(row=0, column=1, padx=10, pady=5, sticky='ew')
        tb.Button(btn_frame, text="Exit", bootstyle=WARNING,
                  command=lambda: close_application(root)).grid(row=0, column=2, padx=10, pady=5, sticky='ew')
    else:
        tk.Button(btn_frame, text="Import",
                  command=_spawn(lambda: run_import_atlas(input_folder_atlas, log_widget, progress_var))).grid(row=0, column=0, padx=10, pady=5, sticky='ew')
        tk.Button(btn_frame, text="Create",
                  command=_spawn(lambda: main_create_atlas(log_widget, progress_var))).grid(row=0, column=1, padx=10, pady=5, sticky='ew')
        tk.Button(btn_frame, text="Exit",
                  command=lambda: close_application(root)).grid(row=0, column=2, padx=10, pady=5, sticky='ew')

    # Diagnostics
    cfg_display = _CFG_PATH if _CFG_PATH is not None else (BASE_DIR / "config.ini")
    log_to_gui(log_widget, f"BASE_DIR: {BASE_DIR}")
    log_to_gui(log_widget, f"Config used: {cfg_display}")
    log_to_gui(log_widget, f"GeoParquet out: {geoparquet_dir()}")
    log_to_gui(log_widget, f"Atlas input folder: {input_folder_atlas}")
    log_to_gui(log_widget, f"EPSG: {workingprojection_epsg}, theme: {ttk_bootstrap_theme}")
    log_to_gui(log_widget, f"Atlas tile size (km): lon={atlas_lon_size_km}, lat={atlas_lat_size_km}, overlap={atlas_overlap_percent}%")

    root.mainloop()
