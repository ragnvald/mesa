# -*- coding: utf-8 -*-
# 01_import.py — Import assets / geocodes / lines → GeoParquet only
# - Inputs  : config.ini [DEFAULT] input_folder_asset|geocode|lines
#            Accepts: Shapefile (*.shp), GeoPackage (*.gpkg), GeoParquet (*.parquet)
# - CRS     : reproject to workingprojection_epsg
# - Outputs : output/geoparquet/
#             tbl_asset_object.parquet, tbl_asset_group.parquet
#             tbl_geocode_object.parquet, tbl_geocode_group.parquet
#             tbl_lines_original.parquet, tbl_lines.parquet
#
# Design goals:
#   * No GeoPackage writes at all (we read gpkg layers if provided, but we do not write gpkg).
#   * Ensure geometry column name is 'geometry' (not 'geom') to align with downstream code.
#   * Force valid GeoParquet even when empty (CRS + geometry dtype present).

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

import os, glob, uuid, argparse, threading, datetime, configparser
from collections import defaultdict
from pathlib import Path

import tkinter as tk
from tkinter import scrolledtext
import ttkbootstrap as tb
from ttkbootstrap.constants import PRIMARY, WARNING

import geopandas as gpd
import pandas as pd
import fiona
from shapely.geometry import box

# ----------------------------
# Globals / UI
# ----------------------------
original_working_directory = None
log_widget = None
progress_var = None
progress_label = None

# ----------------------------
# Logging / Progress
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

def log_to_gui(widget, message: str, level: str = "INFO"):
    ts = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    msg = f"{ts} [{level}] - {message}"
    try:
        if widget and widget.winfo_exists():
            widget.insert(tk.END, msg + "\n")
            widget.see(tk.END)
    except tk.TclError:
        pass
    try:
        with open(Path(base_dir()) / "log.txt", "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass
    if widget is None:
        print(msg, flush=True)

# ----------------------------
# Paths / Config
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

def read_config(path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8")
    return cfg

def load_settings(cfg: configparser.ConfigParser):
    # Robust defaults; align with config.ini keys
    d = cfg["DEFAULT"]
    return {
        "input_folder_asset":   d.get("input_folder_asset",   "input/asset"),
        "input_folder_geocode": d.get("input_folder_geocode", "input/geocode"),
        "input_folder_lines":   d.get("input_folder_lines",   "input/lines"),
        "segment_width":        int(d.get("segment_width", "600")),
        "segment_length":       int(d.get("segment_length","1000")),
        "ttk_theme":            d.get("ttk_bootstrap_theme", "flatly"),
        "working_epsg":         int(d.get("workingprojection_epsg","4326")),
    }

# ----------------------------
# Geo helpers
# ----------------------------
def ensure_geo_gdf(records_or_gdf, crs_str: str) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame with active geometry column named 'geometry',
    CRS set to crs_str. Works even when empty.
    """
    if isinstance(records_or_gdf, gpd.GeoDataFrame):
        gdf = records_or_gdf.copy()
        # ensure geometry column is literally named 'geometry'
        if gdf.geometry.name != "geometry":
            gdf = gdf.set_geometry(gdf.geometry.name).rename_geometry("geometry")
        if gdf.crs is None and crs_str:
            gdf.set_crs(crs_str, inplace=True)
        elif str(gdf.crs) != crs_str and crs_str:
            gdf = gdf.to_crs(crs_str)
        return gdf

    df = pd.DataFrame(records_or_gdf)
    if "geometry" not in df.columns:
        # create empty geometry series
        df["geometry"] = gpd.GeoSeries([], dtype="geometry")
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs_str)
    return gdf

def save_parquet(name: str, gdf: gpd.GeoDataFrame):
    path = gpq_dir() / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    # geopandas writes GeoParquet (with geo metadata) when geometry dtype present
    gdf.to_parquet(path, index=False)
    log_to_gui(log_widget, f"Saved {name} → {path} (rows={len(gdf)})")

def read_and_reproject(filepath: str, layer: str | None, working_epsg: int) -> gpd.GeoDataFrame:
    try:
        data = gpd.read_file(filepath, layer=layer) if layer else gpd.read_file(filepath)
        if data.crs is None:
            log_to_gui(log_widget, f"No CRS in {os.path.basename(filepath)} (layer={layer}); set EPSG:{working_epsg}")
            data.set_crs(epsg=working_epsg, inplace=True)
        elif (data.crs.to_epsg() or working_epsg) != int(working_epsg):
            data = data.to_crs(epsg=int(working_epsg))
        # ensure active geometry column is named 'geometry'
        if data.geometry.name != "geometry":
            data = data.set_geometry(data.geometry.name).rename_geometry("geometry")
        return data
    except Exception as e:
        log_to_gui(log_widget, f"Read fail {filepath} (layer={layer}): {e}", level="ERROR")
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{working_epsg}")

def read_parquet_vector(fp: str, working_epsg: int) -> gpd.GeoDataFrame:
    """Read a GeoParquet file and ensure CRS == working_epsg."""
    try:
        gdf = gpd.read_parquet(fp)
        if gdf.crs is None:
            gdf.set_crs(epsg=working_epsg, inplace=True)
        elif (gdf.crs.to_epsg() or working_epsg) != int(working_epsg):
            gdf = gdf.to_crs(epsg=int(working_epsg))
        # ensure geometry column literally named 'geometry'
        if gdf.geometry.name != "geometry":
            gdf = gdf.set_geometry(gdf.geometry.name).rename_geometry("geometry")
        return gdf
    except Exception as e:
        log_to_gui(log_widget, f"Read fail (parquet) {os.path.basename(fp)}: {e}", level="ERROR")
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{working_epsg}")

# ----------------------------
# IMPORT: Assets
# ----------------------------
def import_spatial_data_asset(input_folder_asset: str, working_epsg: int):
    asset_objects, asset_groups = [], []
    group_id, object_id = 1, 1

    files = []
    # Include GeoParquet vector layers
    for pat in ("*.shp", "*.gpkg", "*.parquet"):
        files.extend(glob.glob(os.path.join(input_folder_asset, "**", pat), recursive=True))
    log_to_gui(log_widget, f"Asset files found: {len(files)}")

    for i, fp in enumerate(files, start=1):
        update_progress(5 + 55 * (i / max(1, len(files))))
        filename = os.path.splitext(os.path.basename(fp))[0]

        if fp.lower().endswith(".gpkg"):
            try:
                for layer in fiona.listlayers(fp):
                    gdf = read_and_reproject(fp, layer, working_epsg)
                    if gdf.empty: 
                        continue
                    bbox_polygon = box(*gdf.total_bounds)
                    cnt = len(gdf)
                    asset_groups.append({
                        "id": group_id,
                        "name_original": layer,
                        "name_gis_assetgroup": f"layer_{group_id:03d}",
                        "title_fromuser": filename,
                        "date_import": datetime.datetime.now(),
                        "geometry": bbox_polygon,
                        "total_asset_objects": int(cnt),
                        "importance": int(0),
                        "susceptibility": int(0),
                        "sensitivity": int(0),
                        "sensitivity_code": "",
                        "sensitivity_description": "",
                    })
                    for _, row in gdf.iterrows():
                        attrs = "; ".join([f"{c}: {row[c]}" for c in gdf.columns if c != gdf.geometry.name])
                        asset_objects.append({
                            "id": object_id,
                            "asset_group_name": layer,
                            "attributes": attrs,
                            "process": True,
                            "ref_asset_group": group_id,
                            "geometry": row.geometry,
                        })
                        object_id += 1
                    group_id += 1
            except Exception as e:
                log_to_gui(log_widget, f"GPKG error {fp}: {e}", level="ERROR")
        else:  # SHP
            if fp.lower().endswith(".parquet"):
                layer = filename
                gdf = read_parquet_vector(fp, working_epsg)
            else:
                layer = filename
                gdf = read_and_reproject(fp, None, working_epsg)
            if gdf.empty:
                continue
            bbox_polygon = box(*gdf.total_bounds)
            cnt = len(gdf)
            asset_groups.append({
                "id": group_id,
                "name_original": layer,
                "name_gis_assetgroup": f"layer_{group_id:03d}",
                "title_fromuser": layer,
                "date_import": datetime.datetime.now(),
                "geometry": bbox_polygon,
                "total_asset_objects": int(cnt),
                "importance": int(0),
                "susceptibility": int(0),
                "sensitivity": int(0),
                "sensitivity_code": "",
                "sensitivity_description": "",
            })
            for _, row in gdf.iterrows():
                attrs = "; ".join([f"{c}: {row[c]}" for c in gdf.columns if c != gdf.geometry.name])
                asset_objects.append({
                    "id": object_id,
                    "asset_group_name": layer,
                    "attributes": attrs,
                    "process": True,
                    "ref_asset_group": group_id,
                    "geometry": row.geometry,
                })
                object_id += 1
            group_id += 1

    crs = f"EPSG:{working_epsg}"
    asset_groups_gdf  = ensure_geo_gdf(asset_groups,  crs)
    asset_objects_gdf = ensure_geo_gdf(asset_objects, crs)
    log_to_gui(log_widget, f"Assets: groups={len(asset_groups_gdf)}, objects={len(asset_objects_gdf)}")
    return asset_objects_gdf, asset_groups_gdf

# ----------------------------
# IMPORT: Geocodes
# ----------------------------
def _ensure_unique_geocode_codes(geocode_objects: list):
    counts = defaultdict(int)
    for o in geocode_objects:
        # normalize to string
        o["code"] = None if pd.isna(o.get("code")) else str(o.get("code"))
        counts[o["code"]] += 1
    for o in geocode_objects:
        if counts[o["code"]] > 1:
            new_code = f"{o['code']}_{uuid.uuid4()}"
            log_to_gui(log_widget, f"Duplicate geocode '{o['code']}' → '{new_code}'")
            o["code"] = new_code

def _process_geocode_layer(data: gpd.GeoDataFrame,
                           layer_name: str,
                           geocode_groups: list,
                           geocode_objects: list,
                           group_id: int,
                           object_id: int):
    if data.empty:
        log_to_gui(log_widget, f"Empty geocode layer: {layer_name}")
        return group_id, object_id

    bbox_polygon = box(*data.total_bounds)
    name_gis_geocodegroup = f"geocode_{group_id:03d}"

    geocode_groups.append({
        "id": group_id,
        "name": layer_name,
        "name_gis_geocodegroup": name_gis_geocodegroup,
        "title_user": layer_name,
        "description": f"Description for {layer_name}",
        "geometry": bbox_polygon
    })

    for _, row in data.iterrows():
        code = None
        if "qdgc" in data.columns and pd.notna(row.get("qdgc")):
            code = str(row["qdgc"])
        else:
            code = str(object_id)
        geocode_objects.append({
            "code": code,
            "ref_geocodegroup": group_id,
            "name_gis_geocodegroup": name_gis_geocodegroup,
            "geometry": row.geometry
        })
        object_id += 1

    return group_id + 1, object_id

def import_spatial_data_geocode(input_folder_geocode: str, working_epsg: int):
    geocode_groups, geocode_objects = [], []
    group_id, object_id = 1, 1

    files = []
    for pat in ("*.shp", "*.gpkg", "*.parquet"):
        files.extend(glob.glob(os.path.join(input_folder_geocode, "**", pat), recursive=True))
    log_to_gui(log_widget, f"Geocode files found: {len(files)}")

    # sort larger first (nice for progress feel)
    def _featcount(fp):
        try: return len(gpd.read_file(fp))
        except Exception: return 0
    files.sort(key=_featcount, reverse=True)

    for i, fp in enumerate(files, start=1):
        update_progress(5 + 55 * (i / max(1, len(files))))
        if fp.lower().endswith(".gpkg"):
            try:
                for layer in fiona.listlayers(fp):
                    gdf = read_and_reproject(fp, layer, working_epsg)
                    group_id, object_id = _process_geocode_layer(
                        gdf, layer, geocode_groups, geocode_objects, group_id, object_id
                    )
            except Exception as e:
                log_to_gui(log_widget, f"GPKG error {fp}: {e}", level="ERROR")
        else:
            layer = os.path.splitext(os.path.basename(fp))[0]
            gdf = read_and_reproject(fp, None, working_epsg)
            group_id, object_id = _process_geocode_layer(
                gdf, layer, geocode_groups, geocode_objects, group_id, object_id
            )

    _ensure_unique_geocode_codes(geocode_objects)

    crs = f"EPSG:{working_epsg}"
    geocode_groups_gdf  = ensure_geo_gdf(geocode_groups,  crs)
    geocode_objects_gdf = ensure_geo_gdf(geocode_objects, crs)
    log_to_gui(log_widget, f"Geocodes: groups={len(geocode_groups_gdf)}, objects={len(geocode_objects_gdf)}")
    return geocode_groups_gdf, geocode_objects_gdf

# ----------------------------
# IMPORT: Lines
# ----------------------------
def _process_line_layer(data: gpd.GeoDataFrame,
                        layer_name: str,
                        line_objects: list,
                        line_id: int):
    if data.empty:
        return line_id
    # length in meters using 3395, then back to working CRS later
    temp = data.to_crs(3395)
    for idx, row in data.iterrows():
        length_m = int(temp.loc[idx].geometry.length)
        attrs = "; ".join([f"{c}: {row[c]}" for c in data.columns if c != data.geometry.name])
        line_objects.append({
            "name_gis": int(line_id),
            "name_user": layer_name,
            "attributes": attrs,
            "length_m": length_m,
            "geometry": row.geometry
        })
        line_id += 1
    return line_id

def import_spatial_data_lines(input_folder_lines: str, working_epsg: int):
    line_objects = []
    line_id = 1
    files = []
    for pat in ("*.shp", "*.gpkg", "*.parquet"):
        files.extend(glob.glob(os.path.join(input_folder_lines, "**", pat), recursive=True))
    log_to_gui(log_widget, f"Line files found: {len(files)}")

    for i, fp in enumerate(files, start=1):
        update_progress(5 + 55 * (i / max(1, len(files))))
        if fp.lower().endswith(".gpkg"):
            for layer in fiona.listlayers(fp):
                gdf = read_and_reproject(fp, layer, working_epsg)
                line_id = _process_line_layer(gdf, layer, line_objects, line_id)
        else:
            layer = os.path.splitext(os.path.basename(fp))[0]
            gdf = read_and_reproject(fp, None, working_epsg)
            line_id = _process_line_layer(gdf, layer, line_objects, line_id)

    crs = f"EPSG:{working_epsg}"
    lines_original = ensure_geo_gdf(line_objects, crs)

    # Build tbl_lines (editable view)
    lines = lines_original.copy()
    if not lines.empty:
        lines["name_gis"] = lines["name_gis"].apply(lambda x: f"line_{int(x):03d}")
        lines["name_user"] = lines["name_gis"]
        # segment defaults from settings will be attached in run_import_lines()
    return lines_original, lines

def finalize_lines(lines_original: gpd.GeoDataFrame,
                   lines: gpd.GeoDataFrame,
                   working_epsg: int,
                   segment_width: int,
                   segment_length: int):
    if lines_original.crs is None:
        lines_original.set_crs(epsg=working_epsg, inplace=True)
    if lines.crs is None:
        lines.set_crs(epsg=working_epsg, inplace=True)

    # enrich editable lines table
    if not lines.empty:
        lines["segment_length"] = int(segment_length)
        lines["segment_width"]  = int(segment_width)
        lines["description"] = lines["name_user"].astype(str) + " + " + lines.get("attributes","").astype(str)

        # recompute length_m in meters precisely and store back in working CRS
        metric = lines.to_crs(3395)
        lines["length_m"] = metric.geometry.length.astype(int)
        lines = lines.to_crs(epsg=working_epsg)

    # column order
    if not lines_original.empty:
        lines_original = lines_original[["name_gis","name_user","attributes","length_m","geometry"]]
    if not lines.empty:
        lines = lines[["name_gis","name_user","segment_length","segment_width","description","length_m","geometry"]]

    return lines_original, lines

# ----------------------------
# Orchestration (thread targets)
# ----------------------------
def run_import_asset(input_folder_asset: str, working_epsg: int):
    log_to_gui(log_widget, "Importing assets…")
    a_obj, a_grp = import_spatial_data_asset(input_folder_asset, working_epsg)
    save_parquet("tbl_asset_object", a_obj)
    save_parquet("tbl_asset_group",  a_grp)
    update_progress(100)
    log_to_gui(log_widget, "Assets import complete.")

def run_import_geocode(input_folder_geocode: str, working_epsg: int):
    log_to_gui(log_widget, "Importing geocodes…")
    g_grp, g_obj = import_spatial_data_geocode(input_folder_geocode, working_epsg)
    # NOTE: processor expects tbl_geocode_object + tbl_geocode_group by these names
    save_parquet("tbl_geocode_group",  g_grp)   # has id + name_gis_geocodegroup + geometry (bbox)
    save_parquet("tbl_geocode_object", g_obj)   # has code + ref_geocodegroup + name_gis_geocodegroup + geometry
    update_progress(100)
    log_to_gui(log_widget, "Geocode import complete.")

def run_import_lines(input_folder_lines: str, working_epsg: int, segment_width: int, segment_length: int):
    log_to_gui(log_widget, "Importing lines…")
    l0, l1 = import_spatial_data_lines(input_folder_lines, working_epsg)
    l0, l1 = finalize_lines(l0, l1, working_epsg, segment_width, segment_length)
    save_parquet("tbl_lines_original", l0)
    save_parquet("tbl_lines",          l1)
    update_progress(100)
    log_to_gui(log_widget, "Line import complete.")

# ----------------------------
# Entrypoint (GUI)
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import assets/geocodes/lines → GeoParquet")
    parser.add_argument('--original_working_directory', required=False, help='Path to running folder')
    args = parser.parse_args()
    original_working_directory = args.original_working_directory or os.getcwd()
    # normalize if launched from /system
    if Path(original_working_directory).name.lower() == "system":
        original_working_directory = str(Path(original_working_directory).parent)

    cfg_path = Path(original_working_directory) / "system" / "config.ini"
    cfg = read_config(cfg_path)
    st = load_settings(cfg)

    input_folder_asset   = str(Path(original_working_directory) / st["input_folder_asset"])
    input_folder_geocode = str(Path(original_working_directory) / st["input_folder_geocode"])
    input_folder_lines   = str(Path(original_working_directory) / st["input_folder_lines"])
    segment_width        = st["segment_width"]
    segment_length       = st["segment_length"]
    ttk_theme            = st["ttk_theme"]
    working_epsg         = st["working_epsg"]

    root = tb.Window(themename=ttk_theme)
    root.title("Import to GeoParquet")
    try:
        ico = base_dir() / "system_resources" / "mesa.ico"
        if ico.exists():
            root.iconbitmap(str(ico))
    except Exception:
        pass

    # UI
    frame = tb.LabelFrame(root, text="Log", bootstyle="info")
    frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    log_widget = scrolledtext.ScrolledText(frame, height=12)
    log_widget.pack(fill=tk.BOTH, expand=True)

    pframe = tk.Frame(root); pframe.pack(pady=6)
    progress_var = tk.DoubleVar()
    pbar = tb.Progressbar(pframe, orient="horizontal", length=240, mode="determinate",
                          variable=progress_var, bootstyle="info")
    pbar.pack(side=tk.LEFT, padx=6)
    progress_label = tk.Label(pframe, text="0%", bg="light grey"); progress_label.pack(side=tk.LEFT)

    # Diagnostics
    log_to_gui(log_widget, f"Working dir: {base_dir()}")
    log_to_gui(log_widget, f"GeoParquet out: {gpq_dir()}")
    log_to_gui(log_widget, f"Assets in:   {input_folder_asset}")
    log_to_gui(log_widget, f"Geocodes in: {input_folder_geocode}")
    log_to_gui(log_widget, f"Lines in:    {input_folder_lines}")
    log_to_gui(log_widget, f"EPSG: {working_epsg}, theme: {ttk_theme}, segW/L: {segment_width}/{segment_length}")

    btns = tk.Frame(root); btns.pack(pady=8)
    tb.Button(btns, text="Import assets", bootstyle=PRIMARY,
              command=lambda: threading.Thread(target=run_import_asset,
                                               args=(input_folder_asset, working_epsg),
                                               daemon=True).start()).grid(row=0, column=0, padx=8, pady=4)
    tb.Button(btns, text="Import geocodes", bootstyle=PRIMARY,
              command=lambda: threading.Thread(target=run_import_geocode,
                                               args=(input_folder_geocode, working_epsg),
                                               daemon=True).start()).grid(row=0, column=1, padx=8, pady=4)
    tb.Button(btns, text="Import lines", bootstyle=PRIMARY,
              command=lambda: threading.Thread(target=run_import_lines,
                                               args=(input_folder_lines, working_epsg, segment_width, segment_length),
                                               daemon=True).start()).grid(row=0, column=2, padx=8, pady=4)
    tb.Button(btns, text="Exit", bootstyle=WARNING, command=root.destroy)\
      .grid(row=0, column=3, padx=8, pady=4)

    root.mainloop()
