# -*- coding: utf-8 -*-
# 01_import.py — robust imports to GeoParquet with stable BASE_DIR (flat config)
# - Works when launched directly from /system or indirectly via mesa.py
# - Resolves BASE_DIR using env/CLI/script/CWD
# - Reads <base>/config.ini (flat). Falls back to <base>/system/config.ini if missing.
# - Uses DEFAULT.parquet_folder (defaults to output/geoparquet)
# - Writes ONLY GeoParquet outputs

from mesa_locale import harden_locale_for_ttkbootstrap

harden_locale_for_ttkbootstrap()

import os, sys, argparse, threading, datetime, configparser, time
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
import pandas as pd
import fiona
from shapely.geometry import box

# ----------------------------
# Globals / UI
# ----------------------------
log_widget = None
progress_var = None
progress_label = None
BASE_DIR: Path = Path(".").resolve()
_CFG: configparser.ConfigParser | None = None
_CFG_PATH: Path | None = None  # where we actually read config from

# ----------------------------
# Base dir resolution
# ----------------------------
def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

def _has_config_at(root: Path) -> bool:
    # Accept either flat <base>/config.ini or legacy <base>/system/config.ini
    return _exists(root / "config.ini") or _exists(root / "system" / "config.ini")

def find_base_dir(cli_workdir: str | None = None) -> Path:
    """
    Choose a canonical project base folder that actually contains config.
    Priority order:
      1) env MESA_BASE_DIR
      2) --original_working_directory (CLI)
      3) running binary/interpreter folder (and parents)
      4) script folder & its parents (covers PyInstaller _MEIPASS)
      5) CWD, CWD/code, and their parents
    """
    candidates: list[Path] = []

    def _add(path_like):
        if not path_like:
            return
        try:
            candidates.append(Path(path_like))
        except Exception:
            pass

    env_base = os.environ.get('MESA_BASE_DIR')
    if env_base:
        _add(env_base)
    if cli_workdir:
        _add(cli_workdir)

    exe_path: Path | None = None
    try:
        exe_path = Path(sys.executable).resolve()
    except Exception:
        exe_path = None
    if exe_path:
        _add(exe_path.parent)
        _add(exe_path.parent.parent)
        _add(exe_path.parent.parent.parent)

    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        _add(meipass)

    here = Path(__file__).resolve()
    _add(here.parent)
    _add(here.parent.parent)
    _add(here.parent.parent.parent)

    cwd = Path.cwd()
    _add(cwd)
    _add(cwd / 'code')
    _add(cwd.parent)
    _add(cwd.parent / 'code')

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

    # fallback: prefer parent of /system, otherwise interpreter/exe dir, else script dir
    if here.parent.name.lower() == 'system':
        return here.parent.parent
    if exe_path:
        return exe_path.parent
    if env_base:
        return Path(env_base)
    return here.parent


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
# Config / Paths
# ----------------------------
def _ensure_cfg() -> configparser.ConfigParser:
    global _CFG, _CFG_PATH
    if _CFG is not None:
        return _CFG

    cfg = configparser.ConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
    # Preferred: flat <base>/config.ini
    flat_path = BASE_DIR / "config.ini"
    legacy_path = BASE_DIR / "system" / "config.ini"
    loaded = False
    if flat_path.exists():
        try:
            cfg.read(flat_path, encoding="utf-8")
            _CFG_PATH = flat_path
            loaded = True
        except Exception:
            pass
    if not loaded and legacy_path.exists():
        try:
            cfg.read(legacy_path, encoding="utf-8")
            _CFG_PATH = legacy_path
            loaded = True
        except Exception:
            pass
    if "DEFAULT" not in cfg:
        cfg["DEFAULT"] = {}

    # Defaults
    cfg["DEFAULT"].setdefault("parquet_folder", "output/geoparquet")
    cfg["DEFAULT"].setdefault("input_folder_asset",   "input/asset")
    cfg["DEFAULT"].setdefault("segment_width", "600")
    cfg["DEFAULT"].setdefault("segment_length", "1000")
    cfg["DEFAULT"].setdefault("ttk_bootstrap_theme", "flatly")
    cfg["DEFAULT"].setdefault("workingprojection_epsg", "4326")

    _CFG = cfg
    return _CFG

def gpq_dir() -> Path:
    cfg = _ensure_cfg()
    sub = cfg["DEFAULT"].get("parquet_folder", "output/geoparquet")
    out = BASE_DIR / sub
    out.mkdir(parents=True, exist_ok=True)
    return out

def _abs_path_like(p: str | Path) -> Path:
    """
    If p is absolute, return as Path.
    If p is relative, resolve against BASE_DIR.
    """
    p = Path(p)
    if p.is_absolute():
        return p
    return (BASE_DIR / p).resolve()

def _relative_to_base(p: Path) -> Path | None:
    try:
        return p.relative_to(BASE_DIR)
    except ValueError:
        return None

def _resolve_input_folder(label: str, configured_path: str | Path) -> Path:
    target = _abs_path_like(configured_path)
    if target.exists():
        return target

    rel = _relative_to_base(target)
    if rel is not None:
        for parent in ("code", "system"):
            candidate = (BASE_DIR / parent / rel).resolve()
            if candidate.exists():
                log_to_gui(
                    f"{label} folder missing at {target}. Falling back to {candidate}",
                    "WARN"
                )
                return candidate

    return target

def load_settings(cfg: configparser.ConfigParser) -> dict:
    d = cfg["DEFAULT"]
    return {
        "input_folder_asset":   d.get("input_folder_asset",   "input/asset"),
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

def _scan_for_files(label: str, folder: Path, patterns: tuple[str, ...]) -> list[Path]:
    if not folder.exists():
        log_to_gui(f"{label} folder does not exist: {folder}", "WARN")
        return []
    pat_list = ", ".join(patterns)
    log_to_gui(f"{label}: scanning {folder} for {pat_list} …")
    t0 = time.time()
    files = _rglob_many(folder, patterns)
    duration = time.time() - t0
    log_to_gui(f"{label}: scan finished in {duration:.1f}s → {len(files)} file(s).")
    if files:
        preview = ", ".join(p.name for p in files[:3])
        if len(files) > 3:
            preview += ", ..."
        log_to_gui(f"{label}: first files → {preview}")
    return files

def import_spatial_data_asset(input_folder_asset: Path, working_epsg: int):
    asset_objects, asset_groups = [], []
    group_id, object_id = 1, 1

    patterns = ("*.shp", "*.gpkg", "*.parquet")
    files = _scan_for_files("Assets", input_folder_asset, patterns)
    log_to_gui(f"Asset files found: {len(files)}")
    update_progress(2)

    for i, fp in enumerate(files, start=1):
        # Scale progress toward 90%; final 100% is set by the caller after the step finishes.
        update_progress(5 + 85 * (i / max(1, len(files))))
        if i == 1 or i % 25 == 0:
            log_to_gui(f"Assets: processing {fp.name} ({i}/{max(1, len(files))})")
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
# Orchestration (thread targets)
# ----------------------------
def run_import_asset(input_folder_asset: Path, working_epsg: int):
    update_progress(0)
    log_to_gui("Step [Assets] STARTED")
    try:
        log_to_gui("Importing assets…")
        a_obj, a_grp = import_spatial_data_asset(input_folder_asset, working_epsg)
        save_parquet("tbl_asset_object", a_obj)
        save_parquet("tbl_asset_group",  a_grp)
        log_to_gui("Step [Assets] COMPLETED")
    except Exception as e:
        log_to_gui(f"Step [Assets] FAILED: {e}", level="ERROR")
    finally:
        update_progress(100)

# ----------------------------
# Entrypoint (GUI)
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import assets → GeoParquet (robust paths, flat config)")
    parser.add_argument('--original_working_directory', required=False, help='Base dir override (usually set by mesa.py)')
    args = parser.parse_args()

    # Resolve BASE_DIR first
    BASE_DIR = find_base_dir(args.original_working_directory)
    _ensure_cfg()  # populate _CFG and _CFG_PATH

    cfg = _ensure_cfg()
    st = load_settings(cfg)

    # Resolve input folders (absolute or relative to BASE_DIR)
    input_folder_asset   = _resolve_input_folder("Assets", st["input_folder_asset"])
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
    progress_var = tk.DoubleVar(value=0.0)
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
    cfg_display = _CFG_PATH if _CFG_PATH is not None else (BASE_DIR / "config.ini")
    log_to_gui(f"BASE_DIR: {BASE_DIR}")
    log_to_gui(f"Config used: {cfg_display}")
    log_to_gui(f"GeoParquet out: {gpq_dir()}")
    log_to_gui(f"Assets in:   {input_folder_asset}")
    log_to_gui(f"EPSG: {working_epsg}, theme: {ttk_theme}")

    btns = tk.Frame(root); btns.pack(pady=8)

    def _btn(cmd):
        return lambda: threading.Thread(target=cmd, daemon=True).start()

    if tb is not None:
        tb.Button(btns, text="Import assets",   bootstyle=PRIMARY, command=_btn(lambda: run_import_asset(input_folder_asset, working_epsg))).grid(row=0, column=0, padx=8, pady=4)
        tb.Button(btns, text="Exit", bootstyle=WARNING, command=root.destroy).grid(row=0, column=1, padx=8, pady=4)
    else:
        tk.Button(btns, text="Import assets",   command=_btn(lambda: run_import_asset(input_folder_asset, working_epsg))).grid(row=0, column=0, padx=8, pady=4)
        tk.Button(btns, text="Exit",            command=root.destroy).grid(row=0, column=1, padx=8, pady=4)

    root.mainloop()
