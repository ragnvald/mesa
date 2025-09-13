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

def main_create_atlas(log_widget_, progress_var_):
    log_to_gui(log_widget_, "Starting atlas generation (GeoParquet)…")
    update_progress(10)

    tbl_flat = read_flat_parquet()
    if tbl_flat.empty:
        log_to_gui(log_widget_, "tbl_flat.parquet empty or missing; aborting.")
        update_progress(100)
        return

    update_progress(30)
    atlas_geometries = generate_atlas_geometries(tbl_flat, atlas_lon_size_km, atlas_lat_size_km, atlas_overlap_percent)
    update_progress(60)

    updated = filter_and_update_atlas_geometries(atlas_geometries, tbl_flat)
    update_progress(80)

    if not updated.empty:
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
                       "Frame size set in config.ini. Previous atlas (tbl_atlas.parquet) will be replaced.")
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
