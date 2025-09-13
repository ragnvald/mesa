# -*- coding: utf-8 -*-
# atlas_edit.py — edit atlas attributes directly in GeoParquet
# - Stable BASE_DIR resolution (env/CLI/script/CWD)
# - Flat config preferred (<base>/config.ini), fallback to <base>/system/config.ini
# - Uses DEFAULT.parquet_folder (defaults to output/geoparquet)
# - Atomic writes; preserves geometry & CRS
# - ttkbootstrap optional (falls back to standard Tk widgets)

import locale
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except Exception:
    try:
        locale.setlocale(locale.LC_ALL, '')
    except Exception:
        pass

import os
import argparse
import tempfile
import configparser
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, filedialog

# Try ttkbootstrap; fall back to std ttk if missing
try:
    import ttkbootstrap as tb
    from ttkbootstrap.constants import SUCCESS, WARNING
except Exception:
    tb = None
    SUCCESS = "success"
    WARNING = "warning"
from tkinter import ttk as ttk

import pandas as pd
import geopandas as gpd

# -----------------------------
# Globals (populated at runtime)
# -----------------------------
BASE_DIR: Path = Path(".").resolve()
_CFG: configparser.ConfigParser | None = None
_CFG_PATH: Path | None = None

original_working_directory: str = ""
config_file: str = ""
atlas_file_name: str = "tbl_atlas.parquet"

# UI state (runtime)
df: pd.DataFrame | None = None
gdf: gpd.GeoDataFrame | None = None
current_index: int = 0

# Tk variables (runtime)
name_gis_var = title_user_var = description_var = None
image_name_1_var = image_desc_1_var = None
image_name_2_var = image_desc_2_var = None

# -----------------------------
# Base dir / Config helpers
# -----------------------------
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

    # defaults (non-destructive)
    d = cfg["DEFAULT"]
    d.setdefault("parquet_folder", "output/geoparquet")
    d.setdefault("ttk_bootstrap_theme", "flatly")
    d.setdefault("workingprojection_epsg", "4326")
    d.setdefault("atlas_parquet_file", "tbl_atlas.parquet")

    _CFG = cfg
    return _CFG

def _abs_path_like(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (BASE_DIR / p).resolve()

# -----------------------------
# Config utils
# -----------------------------
def read_config(file_name):
    cfg = configparser.ConfigParser()
    try:
        cfg.read(file_name, encoding="utf-8")
    except Exception:
        cfg.read(file_name)
    return cfg

def increment_stat_value(cfg_path: str, stat_name: str, increment_value: int):
    if not cfg_path or not os.path.isfile(cfg_path):
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
                        return
        if updated:
            with open(cfg_path, 'w', encoding='utf-8', errors='replace') as f:
                f.writelines(lines)
    except Exception:
        pass

# -----------------------------
# GeoParquet I/O
# -----------------------------
def geoparquet_dir(base_dir: Path) -> Path:
    cfg = _ensure_cfg()
    sub = cfg["DEFAULT"].get("parquet_folder", "output/geoparquet")
    p = base_dir / sub
    p.mkdir(parents=True, exist_ok=True)
    return p

def atlas_parquet_path(base_dir: Path, layer_file: str) -> Path:
    return geoparquet_dir(base_dir) / layer_file

def _empty_gdf() -> gpd.GeoDataFrame:
    cols = [
        'name_gis','title_user','description',
        'image_name_1','image_desc_1','image_name_2','image_desc_2','geometry'
    ]
    return gpd.GeoDataFrame(columns=cols, geometry='geometry', crs="EPSG:4326")

def load_data(base_dir: Path, layer_file: str):
    """
    Read the GeoParquet feature table (incl. geometry).
    Returns (gdf, df_no_geom, loaded_path).
    """
    gpq_path = atlas_parquet_path(base_dir, layer_file)
    if not gpq_path.exists():
        messagebox.showerror("Missing data", "Required atlas data is missing.")
        empty = _empty_gdf()
        return empty, pd.DataFrame(empty.drop(columns=['geometry'])), str(gpq_path)
    try:
        gdf_local = gpd.read_parquet(gpq_path)
        # ensure geometry
        if gdf_local.geometry.name not in gdf_local.columns:
            if 'geom' in gdf_local.columns:
                gdf_local = gdf_local.set_geometry('geom')
            else:
                raise ValueError("No geometry column present in GeoParquet.")
        if gdf_local.crs is None:
            gdf_local.set_crs("EPSG:4326", inplace=True)
        df_no_geom = pd.DataFrame(gdf_local.drop(columns=[gdf_local.geometry.name], errors='ignore'))
        return gdf_local, df_no_geom, str(gpq_path)
    except Exception:
        messagebox.showerror("Error", "Could not read atlas data.")
        empty = _empty_gdf()
        return empty, pd.DataFrame(empty.drop(columns=['geometry'])), str(gpq_path)

def atomic_write_geoparquet(gdf_in: gpd.GeoDataFrame, path: Path):
    """
    Atomically write GeoParquet (write to temp file then replace).
    Keeps geometry + CRS metadata.
    """
    out_dir = path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out_dir, suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        gdf_in.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise

def save_row_to_parquet(base_dir: Path, layer_file: str, row_dict):
    """
    Update a single row in the GeoParquet by key (name_gis).
    Read → modify attributes in-memory → write back atomically.
    Geometry is preserved.
    """
    key = str(row_dict.get("name_gis") or "").strip()
    if not key:
        messagebox.showerror("Error", "Missing name_gis for row update.")
        return

    gpq_path = atlas_parquet_path(base_dir, layer_file)
    if not gpq_path.exists():
        messagebox.showerror("Error", f"File not found:\n{gpq_path}")
        return

    gdf_local = gpd.read_parquet(gpq_path)

    if 'name_gis' not in gdf_local.columns:
        messagebox.showerror("Error", "GeoParquet does not contain a 'name_gis' column.")
        return

    idx = gdf_local.index[gdf_local['name_gis'].astype(str) == key]
    if len(idx) == 0:
        messagebox.showerror("Error", f"No record with name_gis='{key}' found.")
        return

    editable = [
        "title_user", "description",
        "image_name_1", "image_desc_1",
        "image_name_2", "image_desc_2",
    ]
    for c in editable:
        if c in gdf_local.columns and c in row_dict:
            gdf_local.loc[idx, c] = row_dict[c]

    atomic_write_geoparquet(gdf_local, gpq_path)

# -----------------------------
# UI helpers
# -----------------------------
def _images_dir() -> Path:
    return BASE_DIR / "input" / "images"

def browse_image_1():
    fp = filedialog.askopenfilename(
        initialdir=str(_images_dir()),
        title="Select file",
        filetypes=(("JPEG files", "*.jpg;*.jpeg"), ("PNG files", "*.png"), ("All files", "*.*"))
    )
    if fp:
        image_name_1_var.set(fp)

def browse_image_2():
    fp = filedialog.askopenfilename(
        initialdir=str(_images_dir()),
        title="Select file",
        filetypes=(("JPEG files", "*.jpg;*.jpeg"), ("PNG files", "*.png"), ("All files", "*.*"))
    )
    if fp:
        image_name_2_var.set(fp)

def _get(df_loc: pd.DataFrame, idx: int, col: str) -> str:
    try:
        v = df_loc.at[idx, col]
        return "" if pd.isna(v) else str(v)
    except Exception:
        return ""

# -----------------------------
# Record ops
# -----------------------------
def load_record():
    global df
    if df is None or len(df) == 0:
        name_gis_var.set(""); title_user_var.set(""); description_var.set("")
        image_name_1_var.set(""); image_desc_1_var.set("")
        image_name_2_var.set(""); image_desc_2_var.set("")
        return
    name_gis_var.set(_get(df, current_index, 'name_gis'))
    title_user_var.set(_get(df, current_index, 'title_user'))
    description_var.set(_get(df, current_index, 'description'))
    image_name_1_var.set(_get(df, current_index, 'image_name_1'))
    image_desc_1_var.set(_get(df, current_index, 'image_desc_1'))
    image_name_2_var.set(_get(df, current_index, 'image_name_2'))
    image_desc_2_var.set(_get(df, current_index, 'image_desc_2'))

def update_record(save_message=False):
    """
    Update current record in both the UI DataFrame and the GeoParquet (by key).
    """
    global df
    if df is None or len(df) == 0:
        return
    try:
        df.at[current_index, 'name_gis']      = name_gis_var.get()
        df.at[current_index, 'title_user']    = title_user_var.get()
        df.at[current_index, 'description']   = description_var.get()
        df.at[current_index, 'image_name_1']  = image_name_1_var.get()
        df.at[current_index, 'image_desc_1']  = image_desc_1_var.get()
        df.at[current_index, 'image_name_2']  = image_name_2_var.get()
        df.at[current_index, 'image_desc_2']  = image_desc_2_var.get()

        row_dict = df.iloc[current_index].to_dict()
        save_row_to_parquet(BASE_DIR, atlas_file_name, row_dict)

        if save_message:
            messagebox.showinfo("Info", "Record updated and saved to GeoParquet.")
    except Exception as e:
        messagebox.showerror("Error", f"Error updating record: {e}")

def navigate(direction):
    update_record()  # save current before move
    global current_index
    if df is None or len(df) == 0:
        return
    if direction == 'next' and current_index < len(df) - 1:
        current_index += 1
    elif direction == 'previous' and current_index > 0:
        current_index -= 1
    load_record()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Edit atlas (attributes in GeoParquet)')
    parser.add_argument('--original_working_directory', required=False, help='Path to running folder')
    parser.add_argument('--atlas_parquet_file', required=False, help='Override GeoParquet file name (default from config or tbl_atlas.parquet)')
    args = parser.parse_args()

    # Resolve BASE_DIR
    BASE_DIR = find_base_dir(args.original_working_directory)
    original_working_directory = str(BASE_DIR)
    _ensure_cfg()

    # Config / settings
    cfg = _ensure_cfg()
    d = cfg["DEFAULT"]
    ttk_bootstrap_theme    = d.get('ttk_bootstrap_theme', 'flatly')
    workingprojection_epsg = d.get('workingprojection_epsg', '4326')
    atlas_file_name        = args.atlas_parquet_file or d.get('atlas_parquet_file', 'tbl_atlas.parquet')

    # Path to the config file we actually use (for stats)
    config_file = str(_CFG_PATH if _CFG_PATH is not None else (BASE_DIR / "config.ini"))
    increment_stat_value(config_file, 'mesa_stat_edit_atlas', increment_value=1)

    # UI setup
    if tb is not None:
        try:
            root = tb.Window(themename=ttk_bootstrap_theme)
        except Exception:
            root = tb.Window(themename="flatly")
    else:
        root = tk.Tk()

    root.title("Edit atlas (GeoParquet)")
    try:
        ico = BASE_DIR / "system_resources" / "mesa.ico"
        if ico.exists() and hasattr(root, "iconbitmap"):
            root.iconbitmap(str(ico))
    except Exception:
        pass

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # ---------- Early data check: if the file does NOT exist, show Exit-only UI ----------
    expected_path = atlas_parquet_path(BASE_DIR, atlas_file_name)
    if not expected_path.is_file():
        warn_frame = (tb.Frame(root, padding="16") if tb is not None else ttk.Frame(root, padding="16"))
        warn_frame.grid(row=0, column=0, sticky="nsew")

        (tb.Label(warn_frame, text="Data is missing.", font=("", 14, "bold"))
         if tb is not None else ttk.Label(warn_frame, text="Data is missing.", font=("", 14, "bold"))
        ).grid(row=0, column=0, sticky="w", pady=(0,6))

        (tb.Label(warn_frame, text="Please close this window and try again after the atlas data has been generated.", justify="left")
         if tb is not None else ttk.Label(warn_frame, text="Please close this window and try again after the atlas data has been generated.", justify="left")
        ).grid(row=1, column=0, sticky="w")

        (tb.Button(warn_frame, text="Exit", command=root.destroy, bootstyle=WARNING)
         if tb is not None else ttk.Button(warn_frame, text="Exit", command=root.destroy)
        ).grid(row=2, column=0, sticky="e", pady=(16,0))

        root.mainloop()
        raise SystemExit(0)

    # ---------- Full editor UI (file exists) ----------
    main_frame = (tb.Frame(root, padding="10") if tb is not None else ttk.Frame(root, padding="10"))
    main_frame.grid(row=0, column=0, sticky="nsew")

    # Load data (GeoParquet → gdf / df without geometry)
    gdf, df, loaded_path = load_data(BASE_DIR, atlas_file_name)
    current_index = 0

    # Tk variables
    name_gis_var     = tk.StringVar()
    title_user_var   = tk.StringVar()
    description_var  = tk.StringVar()
    image_name_1_var = tk.StringVar()
    image_desc_1_var = tk.StringVar()
    image_name_2_var = tk.StringVar()
    image_desc_2_var = tk.StringVar()

    (tb.Label(main_frame, text=f"Editing file: {loaded_path}")
     if tb is not None else ttk.Label(main_frame, text=f"Editing file: {loaded_path}")
    ).grid(row=0, column=0, columnspan=4, sticky='w', pady=(0,8))

    # Row 1: GIS Name (read-only label)
    tk.Label(main_frame, text="GIS Name").grid(row=1, column=0, sticky='w')
    tk.Label(main_frame, textvariable=name_gis_var, width=40, relief="sunken", anchor="w")\
        .grid(row=1, column=1, sticky='w', padx=10, pady=6)

    # Row 2: Title
    tk.Label(main_frame, text="Title").grid(row=2, column=0, sticky='w')
    tk.Entry(main_frame, textvariable=title_user_var, width=40).grid(row=2, column=1, sticky='w', padx=10, pady=6)

    # Row 3: Image 1
    tk.Label(main_frame, text="Image Name 1").grid(row=3, column=0, sticky='w')
    tk.Entry(main_frame, textvariable=image_name_1_var, width=40).grid(row=3, column=1, sticky='w', padx=10, pady=6)
    (tb.Button(main_frame, text="Browse", command=browse_image_1)
     if tb is not None else ttk.Button(main_frame, text="Browse", command=browse_image_1)
    ).grid(row=3, column=2, padx=10, pady=6)

    tk.Label(main_frame, text="Image 1 description").grid(row=4, column=0, sticky='w')
    tk.Entry(main_frame, textvariable=image_desc_1_var, width=40).grid(row=4, column=1, sticky='w', padx=10, pady=6)

    # Row 5: Image 2
    tk.Label(main_frame, text="Image Name 2").grid(row=5, column=0, sticky='w')
    tk.Entry(main_frame, textvariable=image_name_2_var, width=40).grid(row=5, column=1, sticky='w', padx=10, pady=6)
    (tb.Button(main_frame, text="Browse", command=browse_image_2)
     if tb is not None else ttk.Button(main_frame, text="Browse", command=browse_image_2)
    ).grid(row=5, column=2, padx=10, pady=6)

    tk.Label(main_frame, text="Image 2 description").grid(row=6, column=0, sticky='w')
    tk.Entry(main_frame, textvariable=image_desc_2_var, width=40).grid(row=6, column=1, sticky='w', padx=10, pady=6)

    # Row 7: Description
    tk.Label(main_frame, text="Description").grid(row=7, column=0, sticky='w')
    tk.Entry(main_frame, textvariable=description_var, width=40).grid(row=7, column=1, sticky='w', padx=10, pady=6)

    # Nav + actions
    (tb.Button(main_frame, text="Previous", command=lambda: navigate('previous'))
     if tb is not None else ttk.Button(main_frame, text="Previous", command=lambda: navigate('previous'))
    ).grid(row=8, column=0, sticky='w')

    (tb.Button(main_frame, text="Next", command=lambda: navigate('next'))
     if tb is not None else ttk.Button(main_frame, text="Next", command=lambda: navigate('next'))
    ).grid(row=8, column=2, padx=10, pady=10, sticky='e')

    (tb.Button(main_frame, text="Save", command=lambda: update_record(True), bootstyle=SUCCESS)
     if tb is not None else ttk.Button(main_frame, text="Save", command=lambda: update_record(True))
    ).grid(row=9, column=2, sticky='e', padx=10, pady=6)

    (tb.Button(main_frame, text="Exit", command=root.destroy, bootstyle=WARNING)
     if tb is not None else ttk.Button(main_frame, text="Exit", command=root.destroy)
    ).grid(row=9, column=3, sticky='e', padx=10, pady=6)

    load_record()
    root.mainloop()
