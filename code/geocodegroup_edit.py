import locale
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except Exception:
    try:
        locale.setlocale(locale.LC_ALL, '')
    except Exception:
        pass

import tkinter as tk
from tkinter import messagebox
import configparser
import argparse
import datetime
import geopandas as gpd
import ttkbootstrap as ttk  # ttkbootstrap UI
from ttkbootstrap.constants import *
import os
from pathlib import Path
import tempfile

DEFAULT_PARQUET_SUBDIR = "output/geoparquet"
GEOCODE_GROUP_FILE = "tbl_geocode_group.parquet"

# # # # # # # # # # # # # # 
# Shared/general functions

def read_config(file_name: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    try:
        cfg.read(file_name, encoding="utf-8")
    except Exception:
        pass
    if "DEFAULT" not in cfg:
        cfg["DEFAULT"] = {}
    return cfg

def write_to_log(message: str):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted_message = f"{timestamp} - {message}"
    log_destination_file = os.path.join(original_working_directory, "log.txt")
    try:
        with open(log_destination_file, "a", encoding="utf-8") as log_file:
            log_file.write(formatted_message + "\n")
    except Exception:
        # last resort
        print(formatted_message)

# # # # # # # # # # # # # # 
# Config & path helpers (flat config, GeoParquet-only)

def config_path(base_dir: str) -> str:
    """Flat config: <base>/config.ini"""
    return os.path.join(base_dir, "config.ini")

def parquet_dir_from_cfg(base_dir: str, cfg: configparser.ConfigParser, target_file: str | None = None) -> Path:
    base = Path(base_dir).resolve()
    sub = cfg["DEFAULT"].get("parquet_folder", DEFAULT_PARQUET_SUBDIR)
    candidates: list[Path]
    if os.path.isabs(sub):
        candidates = [Path(sub)]
    else:
        rel = Path(sub)
        candidates = [(base / rel).resolve()]
        if base.name.lower() != "code":
            candidates.append((base / "code" / rel).resolve())
        else:
            parent = base.parent
            if parent:
                candidates.append((parent / rel).resolve())

    if target_file:
        for cand in candidates:
            if (cand / target_file).exists():
                cand.mkdir(parents=True, exist_ok=True)
                return cand

    chosen = candidates[0]
    chosen.mkdir(parents=True, exist_ok=True)
    return chosen

def gpq_path_geocode_group(base_dir: str, cfg: configparser.ConfigParser) -> str:
    root = parquet_dir_from_cfg(base_dir, cfg, GEOCODE_GROUP_FILE)
    return str(root / GEOCODE_GROUP_FILE)

def atomic_write_geoparquet(gdf: gpd.GeoDataFrame, path: str):
    """Write GeoParquet atomically to avoid partial writes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(path), suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        gdf.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise e

# # # # # # # # # # # # # # 
# Core functions

def update_records():
    global df, records
    if df is None or len(df) == 0:
        return
    for record in records:
        row_id      = record['id']
        name        = record['name'].get()
        title_user  = record['title_user'].get()
        description = record['description'].get()

        # Update the GeoDataFrame (keep geometry intact)
        df.loc[df['id'] == row_id, 'name']        = name
        df.loc[df['id'] == row_id, 'title_user']  = title_user
        df.loc[df['id'] == row_id, 'description'] = description

def save_changes():
    update_records()
    save_spatial_data()

def load_spatial_data(gpq_file: str) -> gpd.GeoDataFrame:
    """
    Load GeoParquet → GeoDataFrame (including geometry).
    """
    if not os.path.exists(gpq_file):
        write_to_log(f"GeoParquet not found: {gpq_file}")
        messagebox.showerror("Missing file", f"GeoParquet not found:\n{gpq_file}")
        # Return an empty GeoDataFrame with expected columns so UI still renders
        cols = ['id', 'name', 'name_gis_geocodegroup', 'title_user', 'description', 'geometry']
        return gpd.GeoDataFrame(columns=cols, geometry='geometry', crs="EPSG:4326")

    try:
        gdf = gpd.read_parquet(gpq_file)
        if gdf.crs is None:
            # keep data as-is but attach default CRS if it was missing
            gdf.set_crs("EPSG:4326", inplace=True)
        write_to_log("Spatial data loaded (GeoParquet)")
        return gdf
    except Exception as e:
        write_to_log(f"Failed to read GeoParquet: {e}")
        messagebox.showerror("Error", f"Failed to read GeoParquet:\n{e}")
        cols = ['id', 'name', 'name_gis_geocodegroup', 'title_user', 'description', 'geometry']
        return gpd.GeoDataFrame(columns=cols, geometry='geometry', crs="EPSG:4326")

def save_spatial_data():
    """
    Save the in-memory GeoDataFrame back to the same GeoParquet (atomic write).
    """
    global df, gpq_file
    try:
        atomic_write_geoparquet(df, gpq_file)
        write_to_log("Spatial data saved (GeoParquet)")
        messagebox.showinfo("Saved", "Changes saved to GeoParquet.")
    except Exception as e:
        write_to_log(f"Failed to save GeoParquet: {e}")
        messagebox.showerror("Error", f"Failed to save GeoParquet:\n{e}")

def exit_application():
    write_to_log("Closing edit geocodes")
    root.destroy()

#####################################################################################
#  Main
parser = argparse.ArgumentParser(description='Edit geocodes (GeoParquet)')
parser.add_argument('--original_working_directory', required=False, help='Path to running folder')
args = parser.parse_args()
original_working_directory = args.original_working_directory

if not original_working_directory:
    original_working_directory = os.getcwd()
    # When running from /system subfolder, go up one level.
    if Path(original_working_directory).name.lower() == "system":
        original_working_directory = str(Path(original_working_directory).parent)

# Config & paths (flat)
config_file = config_path(original_working_directory)
config      = read_config(config_file)
gpq_file    = gpq_path_geocode_group(original_working_directory, config)

ttk_bootstrap_theme    = config['DEFAULT'].get('ttk_bootstrap_theme', 'flatly')
workingprojection_epsg = config['DEFAULT'].get('workingprojection_epsg', '4326')

if __name__ == "__main__":
    # UI
    root = ttk.Window(themename=ttk_bootstrap_theme)
    root.title("Edit geocodes (GeoParquet)")
    try:
        root.iconbitmap(os.path.join(original_working_directory,"system_resources","mesa.ico"))
    except Exception:
        pass

    # Load data (GeoDataFrame)
    df = load_spatial_data(gpq_file)

    # Editable grid
    edit_frame = tk.Frame(root, padx=5, pady=5)
    edit_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # Headers
    tk.Label(edit_frame, text="ID").grid(row=0, column=0, sticky='w')
    tk.Label(edit_frame, text="GIS Name").grid(row=0, column=1, sticky='w')
    tk.Label(edit_frame, text="Layer name").grid(row=0, column=2, sticky='w')
    tk.Label(edit_frame, text="User Title").grid(row=0, column=3, sticky='w')
    tk.Label(edit_frame, text="Description").grid(row=0, column=4, sticky='w')

    # Store entry references
    records = []

    # Build rows
    for idx, row in df.iterrows():
        r = idx + 1

        # immutable ID
        tk.Label(edit_frame, text=row.get('id', '')).grid(row=r, column=0, sticky='w')

        # read-only GIS name
        tk.Label(edit_frame, text=row.get('name_gis_geocodegroup', '')).grid(row=r, column=1, sticky='w')

        # editable: 'name'
        name_var = tk.StringVar(value=row.get('name', ''))
        tk.Entry(edit_frame, textvariable=name_var, width=30).grid(row=r, column=2, sticky='w')

        # editable: 'title_user'
        user_title_var = tk.StringVar(value=row.get('title_user', ''))
        tk.Entry(edit_frame, textvariable=user_title_var, width=30).grid(row=r, column=3, sticky='w')

        # editable: 'description'
        description_var = tk.StringVar(value=row.get('description', ''))
        tk.Entry(edit_frame, textvariable=description_var, width=70).grid(row=r, column=4, sticky='w')

        records.append({
            'id': row.get('id', None),
            'name': name_var,
            'title_user': user_title_var,
            'description': description_var
        })

    # Info text
    info_label_text = (
        "After importing geocodes you can adjust their names and descriptions here. "
        "These values are also used by the PDF report."
    )
    tk.Label(root, text=info_label_text, wraplength=600, justify="left").pack(padx=10, pady=10)

    # Buttons
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)

    ttk.Button(btn_frame, text="Save", command=save_changes, bootstyle=SUCCESS).pack(side=tk.LEFT, padx=10)
    ttk.Button(btn_frame, text="Exit", command=exit_application, bootstyle=WARNING).pack(side=tk.LEFT, padx=10)

    # Simple style tweak
    style = ttk.Style()
    style.configure("TButton", padding=6)

    root.mainloop()
