# -*- coding: utf-8 -*-
# 01_import.py — robust imports to GeoParquet with stable BASE_DIR
# - Works when launched directly from /system or indirectly via mesa.py
# - Resolves BASE_DIR using MESA_BASE_DIR env, CLI arg, script location, and cwd
# - Reads system/config.ini reliably (theme + folders)
# - Maps input folders correctly (absolute or relative to BASE_DIR)
# - Writes ONLY GeoParquet outputs

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

import os, sys, argparse, threading, datetime, configparser
from collections import defaultdict
from pathlib import Path

import tkinter as tk
from tkinter import scrolledtext
try:
    import ttkbootstrap as tb
except Exception:
    tb = None
from ttkbootstrap.constants import PRIMARY, WARNING

import geopandas as gpd
import pandas as pd
import fiona
from shapely.geometry import box

# ----------------------------
# Base dir resolution
# ----------------------------
def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

def _has_config_at(root: Path) -> bool:
    return _exists(root / "system" / "config.ini")

def find_base_dir(cli_workdir: str | None = None) -> Path:
    """
    Choose a canonical project base folder that actually contains system/config.ini.
    Priority:
      1) env MESA_BASE_DIR
      2) --original_working_directory (CLI)
      3) script folder & its parents
      4) CWD and common variants (CWD/code)
    """
    candidates: list[Path] = []
    # 1) explicit env
    env_base = os.environ.get("MESA_BASE_DIR")
    if env_base:
        candidates.append(Path(env_base))
    # 2) cli
    if cli_workdir:
        candidates.append(Path(cli_workdir))
    # 3) script dir and parents
    here = Path(__file__).resolve()
    candidates += [
        here.parent,                # .../system
        here.parent.parent,         # .../code
        here.parent.parent.parent,  # .../repo root (if any)
    ]
    # 4) CWD and variant
    cwd = Path(os.getcwd())
    candidates += [cwd, cwd / "code"]

    # Deduplicate while preserving order
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

    # fallback: if launched inside /system, prefer its parent (code)
    if here.parent.name.lower() == "system":
        return here.parent.parent
    return here.parent

# ----------------------------
# Globals / UI
# ----------------------------
log_widget = None
progress_var = None
progress_label = None
BASE_DIR: Path = Path(".").resolve()

# ----------------------------
# Logging / Progress
# ----------------------------
def _ts() -> str:
    return datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")

def log_to_gui(message: str, level: str = "INFO"):
    ts = _ts()
    msg = f"{ts} [{level}] - {message}"
    try:
        if log_widget and log_widget.winfo_exists():
            log_widget.insert(tk.END, msg + "\n")
            log_widget.see(tk.END)
    except tk.TclError:
        pass
    # mirror to file
    try:
        with open(BASE_DIR / "log.txt", "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass
    if log_widget is None:
        print(msg, flush=True)

def update_progress(new_value: float):
    try:
        v = max(0.0, min(100.0, float(new_value)))
        if progress_var is not None: progress_var.set(v)
        if progress_label is not None:
            progress_label.config(text=f"{int(v)}%")
            progress_label.update_idletasks()
    except Exception:
        pass

# ----------------------------
# Paths / Config
# ----------------------------
def gpq_dir() -> Path:
    out = BASE_DIR / "output" / "geoparquet"
    out.mkdir(parents=True, exist_ok=True)
    return out

def read_config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
    cfg_path = BASE_DIR / "system" / "config.ini"
    if not cfg_path.exists():
        # Warn visibly; continue with defaults
        log_to_gui(f"config.ini not found at {cfg_path}", "WARN")
    else:
        cfg.read(cfg_path, encoding="utf-8")
    if "DEFAULT" not in cfg:
        cfg["DEFAULT"] = {}
    return cfg

def _abs_path_like(p: str | Path) -> Path:
    """
    If p is absolute, return as Path.
    If p is relative, resolve against BASE_DIR.
    """
    p = Path(p)
    if p.is_absolute():
        return p
    return (BASE_DIR / p).resolve()

def load_settings(cfg: configparser.ConfigParser) -> dict:
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
    if isinstance(records_or_gdf, gpd.GeoDataFrame):
        gdf = records_or_gdf.copy()
        # ensure geometry column named 'geometry'
        if gdf.geometry.name != "geometry":
            gdf = gdf.set_geometry(gdf.geometry.name).rename_geometry("geometry")
        if gdf.crs is None and crs_str:
            gdf.set_crs(crs_str, inplace=True)
        elif str(gdf.crs) != crs_str and crs_str:
            gdf = gdf.to_crs(crs_str)
        return gdf

    df = pd.DataFrame(records_or_gdf)
    if "geometry" not in df.columns:
        df["geometry"] = gpd.GeoSeries([], dtype="geometry")
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs_str)
    return gdf

def save_parquet(name: str, gdf: gpd.GeoDataFrame):
    path = gpq_dir() / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(path, index=False)
    log_to_gui(f"Saved {name} → {path} (rows={len(gdf)})")

def read_and_reproject(filepath: Path, layer: str | None, working_epsg: int) -> gpd.GeoDataFrame:
    try:
        data = gpd.read_file(filepath, layer=layer) if layer else gpd.read_file(filepath)
        if data.crs is None:
            log_to_gui(f"No CRS in {filepath.name} (layer={layer}); set EPSG:{working_epsg}")
            data.set_crs(epsg=working_epsg, inplace=True)
        elif (data.crs.to_epsg() or working_epsg) != int(working_epsg):
            data = data.to_crs(epsg=int(working_epsg))
        if data.geometry.name != "geometry":
            data = data.set_geometry(data.geometry.name).rename_geometry("geometry")
        return data
    except Exception as e:
        log_to_gui(f"Read fail {filepath} (layer={layer}): {e}", level="ERROR")
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{working_epsg}")

def read_parquet_vector(fp: Path, working_epsg: int) -> gpd.GeoDataFrame:
    try:
        gdf = gpd.read_parquet(fp)
        if gdf.crs is None:
            gdf.set_crs(epsg=working_epsg, inplace=True)
        elif (gdf.crs.to_epsg() or working_epsg) != int(working_epsg):
            gdf = gdf.to_crs(epsg=int(working_epsg))
        if gdf.geometry.name != "geometry":
            gdf = gdf.set_geometry(gdf.geometry.name).rename_geometry("geometry")
        return gdf
    except Exception as e:
        log_to_gui(f"Read fail (parquet) {fp.name}: {e}", level="ERROR")
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{working_epsg}")

# ----------------------------
# IMPORT: Assets
# ----------------------------
def _rglob_many(folder: Path, patterns: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for pat in patterns:
        files.extend(folder.rglob(pat))
    return files

def import_spatial_data_asset(input_folder_asset: Path, working_epsg: int):
    asset_objects, asset_groups = [], []
    group_id, object_id = 1, 1

    files: list[Path] = []
    if input_folder_asset.exists():
        files = _rglob_many(input_folder_asset, ("*.shp", "*.gpkg", "*.parquet"))
    else:
        log_to_gui(f"Assets folder does not exist: {input_folder_asset}", "WARN")
        # helpful hint: if BASE_DIR is repo root and code/ contains inputs
        alt = (BASE_DIR / "code" / "input" / "asset")
        if alt.exists():
            log_to_gui(f"Hint: try config input_folder_asset=code/input/asset (found {alt})", "WARN")
    log_to_gui(f"Asset files found: {len(files)}")

    for i, fp in enumerate(files, start=1):
        update_progress(5 + 55 * (i / max(1, len(files))))
        filename = fp.stem

        if fp.suffix.lower() == ".gpkg":
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
                log_to_gui(f"GPKG error {fp}: {e}", level="ERROR")
        else:
            if fp.suffix.lower() == ".parquet":
                gdf = read_parquet_vector(fp, working_epsg)
            else:
                gdf = read_and_reproject(fp, None, working_epsg)
            if gdf.empty:
                continue
            bbox_polygon = box(*gdf.total_bounds)
            cnt = len(gdf)
            layer = filename
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
    log_to_gui(f"Assets: groups={len(asset_groups_gdf)}, objects={len(asset_objects_gdf)}")
    return asset_objects_gdf, asset_groups_gdf

# ----------------------------
# IMPORT: Geocodes
# ----------------------------
def _ensure_unique_geocode_codes(geocode_objects: list):
    counts = defaultdict(int)
    for o in geocode_objects:
        o["code"] = None if pd.isna(o.get("code")) else str(o.get("code"))
        counts[o["code"]] += 1
    for o in geocode_objects:
        if counts[o["code"]] > 1:
            import uuid
            new_code = f"{o['code']}_{uuid.uuid4()}"
            log_to_gui(f"Duplicate geocode '{o['code']}' → '{new_code}'")
            o["code"] = new_code

def _process_geocode_layer(data: gpd.GeoDataFrame,
                           layer_name: str,
                           geocode_groups: list,
                           geocode_objects: list,
                           group_id: int,
                           object_id: int):
    if data.empty:
        log_to_gui(f"Empty geocode layer: {layer_name}")
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

def import_spatial_data_geocode(input_folder_geocode: Path, working_epsg: int):
    geocode_groups, geocode_objects = [], []
    group_id, object_id = 1, 1

    files: list[Path] = []
    if input_folder_geocode.exists():
        files = _rglob_many(input_folder_geocode, ("*.shp", "*.gpkg", "*.parquet"))
    else:
        log_to_gui(f"Geocode folder does not exist: {input_folder_geocode}", "WARN")
        alt = (BASE_DIR / "code" / "input" / "geocode")
        if alt.exists():
            log_to_gui(f"Hint: try config input_folder_geocode=code/input/geocode (found {alt})", "WARN")
    log_to_gui(f"Geocode files found: {len(files)}")

    # sort larger first (nice for progress feel)
    def _featcount(fp: Path):
        try: return len(gpd.read_file(fp))
        except Exception: return 0
    files.sort(key=_featcount, reverse=True)

    for i, fp in enumerate(files, start=1):
        update_progress(5 + 55 * (i / max(1, len(files))))
        if fp.suffix.lower() == ".gpkg":
            try:
                for layer in fiona.listlayers(fp):
                    gdf = read_and_reproject(fp, layer, working_epsg)
                    group_id, object_id = _process_geocode_layer(
                        gdf, layer, geocode_groups, geocode_objects, group_id, object_id
                    )
            except Exception as e:
                log_to_gui(f"GPKG error {fp}: {e}", level="ERROR")
        else:
            layer = fp.stem
            if fp.suffix.lower() == ".parquet":
                gdf = read_parquet_vector(fp, working_epsg)
            else:
                gdf = read_and_reproject(fp, None, working_epsg)
            group_id, object_id = _process_geocode_layer(
                gdf, layer, geocode_groups, geocode_objects, group_id, object_id
            )

    _ensure_unique_geocode_codes(geocode_objects)

    crs = f"EPSG:{working_epsg}"
    geocode_groups_gdf  = ensure_geo_gdf(geocode_groups,  crs)
    geocode_objects_gdf = ensure_geo_gdf(geocode_objects, crs)
    log_to_gui(f"Geocodes: groups={len(geocode_groups_gdf)}, objects={len(geocode_objects_gdf)}")
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

def import_spatial_data_lines(input_folder_lines: Path, working_epsg: int):
    line_objects: list[dict] = []
    line_id = 1
    files: list[Path] = []
    if input_folder_lines.exists():
        files = _rglob_many(input_folder_lines, ("*.shp", "*.gpkg", "*.parquet"))
    else:
        log_to_gui(f"Lines folder does not exist: {input_folder_lines}", "WARN")
        alt = (BASE_DIR / "code" / "input" / "lines")
        if alt.exists():
            log_to_gui(f"Hint: try config input_folder_lines=code/input/lines (found {alt})", "WARN")
    log_to_gui(f"Line files found: {len(files)}")

    for i, fp in enumerate(files, start=1):
        update_progress(5 + 55 * (i / max(1, len(files))))
        if fp.suffix.lower() == ".gpkg":
            for layer in fiona.listlayers(fp):
                gdf = read_and_reproject(fp, layer, working_epsg)
                line_id = _process_line_layer(gdf, layer, line_objects, line_id)
        else:
            layer = fp.stem
            if fp.suffix.lower() == ".parquet":
                gdf = read_parquet_vector(fp, working_epsg)
            else:
                gdf = read_and_reproject(fp, None, working_epsg)
            line_id = _process_line_layer(gdf, layer, line_objects, line_id)

    crs = f"EPSG:{working_epsg}"
    lines_original = ensure_geo_gdf(line_objects, crs)

    lines = lines_original.copy()
    if not lines.empty:
        lines["name_gis"] = lines["name_gis"].apply(lambda x: f"line_{int(x):03d}")
        lines["name_user"] = lines["name_gis"]
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

    if not lines.empty:
        lines["segment_length"] = int(segment_length)
        lines["segment_width"]  = int(segment_width)
        lines["description"] = lines["name_user"].astype(str) + " + " + lines.get("attributes","").astype(str)

        metric = lines.to_crs(3395)
        lines["length_m"] = metric.geometry.length.astype(int)
        lines = lines.to_crs(epsg=working_epsg)

    if not lines_original.empty:
        lines_original = lines_original[["name_gis","name_user","attributes","length_m","geometry"]]
    if not lines.empty:
        lines = lines[["name_gis","name_user","segment_length","segment_width","description","length_m","geometry"]]

    return lines_original, lines

# ----------------------------
# Orchestration (thread targets)
# ----------------------------
def run_import_asset(input_folder_asset: Path, working_epsg: int):
    log_to_gui("Importing assets…")
    a_obj, a_grp = import_spatial_data_asset(input_folder_asset, working_epsg)
    save_parquet("tbl_asset_object", a_obj)
    save_parquet("tbl_asset_group",  a_grp)
    update_progress(100)
    log_to_gui("Assets import complete.")

def run_import_geocode(input_folder_geocode: Path, working_epsg: int):
    log_to_gui("Importing geocodes…")
    g_grp, g_obj = import_spatial_data_geocode(input_folder_geocode, working_epsg)
    save_parquet("tbl_geocode_group",  g_grp)
    save_parquet("tbl_geocode_object", g_obj)
    update_progress(100)
    log_to_gui("Geocode import complete.")

def run_import_lines(input_folder_lines: Path, working_epsg: int, segment_width: int, segment_length: int):
    log_to_gui("Importing lines…")
    l0, l1 = import_spatial_data_lines(input_folder_lines, working_epsg)
    l0, l1 = finalize_lines(l0, l1, working_epsg, segment_width, segment_length)
    save_parquet("tbl_lines_original", l0)
    save_parquet("tbl_lines",          l1)
    update_progress(100)
    log_to_gui("Line import complete.")

# ----------------------------
# Entrypoint (GUI)
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import assets/geocodes/lines → GeoParquet (robust paths)")
    parser.add_argument('--original_working_directory', required=False, help='Base dir override (usually set by mesa.py)')
    args = parser.parse_args()

    # Resolve BASE_DIR
    BASE_DIR = find_base_dir(args.original_working_directory)

    cfg = read_config()
    st = load_settings(cfg)

    # Resolve input folders (absolute or relative to BASE_DIR)
    input_folder_asset   = _abs_path_like(st["input_folder_asset"])
    input_folder_geocode = _abs_path_like(st["input_folder_geocode"])
    input_folder_lines   = _abs_path_like(st["input_folder_lines"])
    segment_width        = st["segment_width"]
    segment_length       = st["segment_length"]
    ttk_theme            = st["ttk_theme"]
    working_epsg         = st["working_epsg"]

    # GUI init
    if tb is not None:
        try:
            root = tb.Window(themename=ttk_theme)
        except Exception:
            root = tb.Window(themename="flatly")
    else:
        # ttkbootstrap missing: fallback to standard Tk
        root = tk.Tk()
    root.title("Import to GeoParquet")
    try:
        ico = BASE_DIR / "system_resources" / "mesa.ico"
        if ico.exists() and hasattr(root, "iconbitmap"):
            root.iconbitmap(str(ico))
    except Exception:
        pass

    # UI
    frame = tk.LabelFrame(root, text="Log")
    if tb is not None:
        frame = tb.LabelFrame(root, text="Log", bootstyle="info")
    frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    log_widget = scrolledtext.ScrolledText(frame, height=12)
    log_widget.pack(fill=tk.BOTH, expand=True)

    pframe = tk.Frame(root); pframe.pack(pady=6)
    progress_var = tk.DoubleVar()
    if tb is not None:
        pbar = tb.Progressbar(pframe, orient="horizontal", length=240, mode="determinate",
                              variable=progress_var, bootstyle="info")
    else:
        from tkinter import ttk as _ttk
        pbar = _ttk.Progressbar(pframe, orient="horizontal", length=240, mode="determinate",
                                variable=progress_var)
    pbar.pack(side=tk.LEFT, padx=6)
    progress_label = tk.Label(pframe, text="0%", bg="light grey"); progress_label.pack(side=tk.LEFT)

    # Diagnostics
    log_to_gui(f"BASE_DIR: {BASE_DIR}")
    log_to_gui(f"Config:  {BASE_DIR / 'system' / 'config.ini'}")
    log_to_gui(f"GeoParquet out: {gpq_dir()}")
    log_to_gui(f"Assets in:   {input_folder_asset}")
    log_to_gui(f"Geocodes in: {input_folder_geocode}")
    log_to_gui(f"Lines in:    {input_folder_lines}")
    log_to_gui(f"EPSG: {working_epsg}, theme: {ttk_theme}, segW/L: {segment_width}/{segment_length}")

    btns = tk.Frame(root); btns.pack(pady=8)
    btn1 = tk.Button(btns, text="Import assets",
                     command=lambda: threading.Thread(target=run_import_asset,
                                                      args=(input_folder_asset, working_epsg),
                                                      daemon=True).start())
    btn2 = tk.Button(btns, text="Import geocodes",
                     command=lambda: threading.Thread(target=run_import_geocode,
                                                      args=(input_folder_geocode, working_epsg),
                                                      daemon=True).start())
    btn3 = tk.Button(btns, text="Import lines",
                     command=lambda: threading.Thread(target=run_import_lines,
                                                      args=(input_folder_lines, working_epsg, segment_width, segment_length),
                                                      daemon=True).start())
    btn4 = tk.Button(btns, text="Exit", command=root.destroy)

    if tb is not None:
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
    else:
        btn1.grid(row=0, column=0, padx=8, pady=4)
        btn2.grid(row=0, column=1, padx=8, pady=4)
        btn3.grid(row=0, column=2, padx=8, pady=4)
        btn4.grid(row=0, column=3, padx=8, pady=4)

    root.mainloop()
