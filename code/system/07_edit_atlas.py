import locale
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except Exception:
    try:
        locale.setlocale(locale.LC_ALL, '')
    except Exception:
        pass

import tkinter as tk
from tkinter import messagebox, filedialog
import configparser
import pandas as pd
import geopandas as gpd
import os
import argparse
from pathlib import Path
import tempfile
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

# -----------------------------
# Config helpers
# -----------------------------
def read_config(file_name):
    cfg = configparser.ConfigParser()
    cfg.read(file_name, encoding="utf-8")
    return cfg

def increment_stat_value(config_file, stat_name, increment_value):
    if not os.path.isfile(config_file):
        return
    try:
        with open(config_file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
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
                        return
        if updated:
            with open(config_file, 'w', encoding='utf-8', errors='replace') as f:
                f.writelines(lines)
    except Exception:
        pass

# -----------------------------
# GeoParquet I/O
# -----------------------------
def geoparquet_dir(base_dir):
    return os.path.join(base_dir, "output", "geoparquet")

def atlas_parquet_path(base_dir, layer_file):
    return os.path.join(geoparquet_dir(base_dir), layer_file)

def _empty_gdf():
    cols = [
        'name_gis','title_user','description',
        'image_name_1','image_desc_1','image_name_2','image_desc_2','geometry'
    ]
    return gpd.GeoDataFrame(columns=cols, geometry='geometry', crs="EPSG:4326")

def load_data(original_working_directory: str, layer_file: str):
    """
    Read the GeoParquet feature table (incl. geometry).
    Returns (gdf, df_no_geom, loaded_path).
    """
    gpq_path = atlas_parquet_path(original_working_directory, layer_file)
    if not os.path.exists(gpq_path):
        # Generic user message; no file paths.
        messagebox.showerror("Missing data", "Required atlas data is missing.")
        empty = _empty_gdf()
        return empty, pd.DataFrame(empty.drop(columns=['geometry'])), gpq_path
    try:
        gdf = gpd.read_parquet(gpq_path)
        if gdf.geometry.name not in gdf.columns:
            if 'geom' in gdf.columns:
                gdf = gdf.set_geometry('geom')
            else:
                raise ValueError("No geometry column present in GeoParquet.")
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        df_no_geom = pd.DataFrame(gdf.drop(columns=[gdf.geometry.name], errors='ignore'))
        return gdf, df_no_geom, gpq_path
    except Exception:
        messagebox.showerror("Error", "Could not read atlas data.")
        empty = _empty_gdf()
        return empty, pd.DataFrame(empty.drop(columns=['geometry'])), gpq_path

def atomic_write_geoparquet(gdf: gpd.GeoDataFrame, path: str):
    """
    Atomically write GeoParquet (write to temp file then replace).
    Keeps geometry + CRS metadata.
    """
    out_dir = os.path.dirname(path)
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out_dir, suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        gdf.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise

def save_row_to_parquet(original_working_directory: str, layer_file: str, row_dict):
    """
    Update a single row in the GeoParquet by key (name_gis).
    Read → modify attributes in-memory → write back atomically.
    Geometry is preserved.
    """
    key = str(row_dict.get("name_gis") or "").strip()
    if not key:
        messagebox.showerror("Error", "Missing name_gis for row update.")
        return

    gpq_path = atlas_parquet_path(original_working_directory, layer_file)
    gdf = gpd.read_parquet(gpq_path)

    if 'name_gis' not in gdf.columns:
        messagebox.showerror("Error", "GeoParquet does not contain a 'name_gis' column.")
        return

    idx = gdf.index[gdf['name_gis'].astype(str) == key]
    if len(idx) == 0:
        messagebox.showerror("Error", f"No record with name_gis='{key}' found.")
        return

    editable = [
        "title_user", "description",
        "image_name_1", "image_desc_1",
        "image_name_2", "image_desc_2",
    ]
    for c in editable:
        if c in gdf.columns and c in row_dict:
            gdf.loc[idx, c] = row_dict[c]

    atomic_write_geoparquet(gdf, gpq_path)

# -----------------------------
# UI helpers
# -----------------------------
def browse_image_1():
    initial_dir = os.path.join(original_working_directory, "input", "images")
    file_path = filedialog.askopenfilename(
        initialdir=initial_dir, title="Select file",
        filetypes=(("JPEG files", "*.jpg;*.jpeg"), ("PNG files", "*.png"), ("All files", "*.*"))
    )
    if file_path:
        image_name_1_var.set(file_path)

def browse_image_2():
    initial_dir = os.path.join(original_working_directory, "input", "images")
    file_path = filedialog.askopenfilename(
        initialdir=initial_dir, title="Select file",
        filetypes=(("JPEG files", "*.jpg;*.jpeg"), ("PNG files", "*.png"), ("All files", "*.*"))
    )
    if file_path:
        image_name_2_var.set(file_path)

def _get(df, idx, col):
    try:
        return "" if pd.isna(df.at[idx, col]) else str(df.at[idx, col])
    except Exception:
        return ""

# -----------------------------
# Record ops
# -----------------------------
def load_record():
    if len(df) == 0:
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
    if len(df) == 0:
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
        save_row_to_parquet(original_working_directory, atlas_file_name, row_dict)

        if save_message:
            messagebox.showinfo("Info", "Record updated and saved to GeoParquet.")
    except Exception as e:
        messagebox.showerror("Error", f"Error updating record: {e}")

def navigate(direction):
    update_record()  # save current before move
    global current_index
    if direction == 'next' and current_index < len(df) - 1:
        current_index += 1
    elif direction == 'previous' and current_index > 0:
        current_index -= 1
    load_record()

# -----------------------------
# Main
# -----------------------------
parser = argparse.ArgumentParser(description='Edit atlas (attributes in GeoParquet)')
parser.add_argument('--original_working_directory', required=False, help='Path to running folder')
parser.add_argument('--atlas_parquet_file', required=False, help='Override GeoParquet file name (default from config or tbl_atlas.parquet)')
args = parser.parse_args()

original_working_directory = args.original_working_directory
if not original_working_directory:
    original_working_directory = os.getcwd()
    if Path(original_working_directory).name.lower() == "system":
        original_working_directory = str(Path(original_working_directory).parent)

config_file = os.path.join(original_working_directory, "system", "config.ini")
config = read_config(config_file)

ttk_bootstrap_theme    = config['DEFAULT'].get('ttk_bootstrap_theme', 'flatly')
workingprojection_epsg = config['DEFAULT'].get('workingprojection_epsg', '4326')
atlas_file_name        = args.atlas_parquet_file or config['DEFAULT'].get('atlas_parquet_file', 'tbl_atlas.parquet')

increment_stat_value(config_file, 'mesa_stat_edit_atlas', increment_value=1)

if __name__ == "__main__":
    # UI setup
    root = ttk.Window(themename=ttk_bootstrap_theme)
    root.title("Edit atlas (GeoParquet)")
    try:
        root.iconbitmap(os.path.join(original_working_directory,"system_resources","mesa.ico"))
    except Exception:
        pass

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # ---------- Early data check: if the file does NOT exist, show Exit-only UI ----------
    expected_path = atlas_parquet_path(original_working_directory, atlas_file_name)
    if not os.path.isfile(expected_path):
        warn_frame = ttk.Frame(root, padding="16")
        warn_frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(
            warn_frame,
            text="Data is missing.",
            font=("", 14, "bold")
        ).grid(row=0, column=0, sticky="w", pady=(0,6))

        ttk.Label(
            warn_frame,
            text="Please close this window and try again after the atlas data has been generated.",
            justify="left",
        ).grid(row=1, column=0, sticky="w")

        ttk.Button(warn_frame, text="Exit", command=root.destroy, bootstyle=WARNING)\
            .grid(row=2, column=0, sticky="e", pady=(16,0))
        root.mainloop()
        raise SystemExit(0)

    # ---------- Full editor UI (file exists) ----------
    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky="nsew")

    gdf, df, loaded_path = load_data(original_working_directory, atlas_file_name)
    current_index = 0

    # Tk variables
    name_gis_var     = tk.StringVar()
    title_user_var   = tk.StringVar()
    description_var  = tk.StringVar()
    image_name_1_var = tk.StringVar()
    image_desc_1_var = tk.StringVar()
    image_name_2_var = tk.StringVar()
    image_desc_2_var = tk.StringVar()

    ttk.Label(main_frame, text=f"Editing file: {loaded_path}").grid(row=0, column=0, columnspan=3, sticky='w', pady=(0,8))

    tk.Label(main_frame, text="GIS Name").grid(row=1, column=0, sticky='w')
    tk.Label(main_frame, textvariable=name_gis_var, width=40, relief="sunken", anchor="w")\
        .grid(row=1, column=1, sticky='w', padx=10, pady=6)

    tk.Label(main_frame, text="Title").grid(row=2, column=0, sticky='w')
    tk.Entry(main_frame, textvariable=title_user_var, width=40).grid(row=2, column=1, sticky='w', padx=10, pady=6)

    tk.Label(main_frame, text="Image Name 1").grid(row=3, column=0, sticky='w')
    tk.Entry(main_frame, textvariable=image_name_1_var, width=40).grid(row=3, column=1, sticky='w', padx=10, pady=6)
    ttk.Button(main_frame, text="Browse", command=browse_image_1).grid(row=3, column=2, padx=10, pady=6)

    tk.Label(main_frame, text="Image 1 description").grid(row=4, column=0, sticky='w')
    tk.Entry(main_frame, textvariable=image_desc_1_var, width=40).grid(row=4, column=1, sticky='w', padx=10, pady=6)

    tk.Label(main_frame, text="Image Name 2").grid(row=5, column=0, sticky='w')
    tk.Entry(main_frame, textvariable=image_name_2_var, width=40).grid(row=5, column=1, sticky='w', padx=10, pady=6)
    ttk.Button(main_frame, text="Browse", command=browse_image_2).grid(row=5, column=2, padx=10, pady=6)

    tk.Label(main_frame, text="Image 2 description").grid(row=6, column=0, sticky='w')
    tk.Entry(main_frame, textvariable=image_desc_2_var, width=40).grid(row=6, column=1, sticky='w', padx=10, pady=6)

    tk.Label(main_frame, text="Description").grid(row=7, column=0, sticky='w')
    tk.Entry(main_frame, textvariable=description_var, width=40).grid(row=7, column=1, sticky='w', padx=10, pady=6)

    ttk.Button(main_frame, text="Previous", command=lambda: navigate('previous')).grid(row=8, column=0, sticky='w')
    ttk.Button(main_frame, text="Next", command=lambda: navigate('next')).grid(row=8, column=2, padx=10, pady=10, sticky='e')
    ttk.Button(main_frame, text="Save", command=lambda: update_record(True), bootstyle=SUCCESS).grid(row=9, column=2, sticky='e', padx=10, pady=6)
    ttk.Button(main_frame, text="Exit", command=root.destroy, bootstyle=WARNING).grid(row=9, column=3, sticky='e', padx=10, pady=6)

    load_record()
    root.mainloop()
