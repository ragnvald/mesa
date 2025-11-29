# -*- coding: utf-8 -*-
# MESA – Setup & Registration (2 tabs: Start and Vulnerability)
# Persistence: GeoParquet + JSON only (GPKG removed)

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

import os
import sys
import argparse
import configparser
import datetime
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Optional
from shapely import wkb as _shapely_wkb
from shapely import wkt as _shapely_wkt

import tkinter as tk
from tkinter import messagebox, filedialog

import ttkbootstrap as tb
from ttkbootstrap.constants import *
from ttkbootstrap import ttk as ttkb  # themed ttk widgets

# Capture start-CWD before anything changes
START_CWD = Path.cwd()

# exe dir if frozen, else script dir
APP_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent

# ---- constant *relative* paths we join to a discovered base dir ----
PARQUET_ASSET_GROUP  = os.path.join("output", "geoparquet", "tbl_asset_group.parquet")
ASSET_GROUP_OVERRIDE: Optional[Path] = None

# UI grid helpers
column_widths = [35, 13, 13, 13, 13, 30]
valid_input_values: list[int] = []
classification: dict = {}
entries_vuln = []
FALLBACK_VULN = 3
INDEX_WEIGHT_DEFAULTS = {
    "importance": [1, 2, 5, 5, 10],
    "sensitivity": [1, 2, 3, 5, 10],
}
INDEX_WEIGHT_KEYS = {
    "importance": "index_importance_weights",
    "sensitivity": "index_sensitivity_weights",
}
index_weight_settings: dict[str, list[int]] = {}

# paths set in __main__
original_working_directory = ""
config_file = ""
workingprojection_epsg = "4326"

# -------------------------------
# Path helpers
# -------------------------------
def has_project_markers(p: Path) -> bool:
    """Heuristic: does 'p' look like the base?"""
    if not p or not p.exists():
        return False
    if (p / "config.ini").exists():
        return True
    if (p / "output" / "geoparquet").exists():
        return True
    return False

def resource_path(rel: str | os.PathLike) -> Path:
    """Resolve bundled resources both for frozen and source runs."""
    try:
        base = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    except Exception:
        base = APP_DIR
    return base / rel

def find_base_dir(cli_arg: str | None) -> Path:
    """
    Resolve the runtime base dir in a way that works in all three modes:
    - Direct .py
    - Called from mesa.py
    - Launched as helper under mesa.exe
    Precedence:
      1) --original_working_directory (if given and valid)
      2) MESA_BASE_DIR env (if valid)
      3) START_CWD, APP_DIR, and a few parents where markers exist
    """
    # 1) CLI
    if cli_arg:
        p = Path(cli_arg).resolve()
        if has_project_markers(p):
            return p

    # 2) Env
    env_p = os.environ.get("MESA_BASE_DIR", "").strip()
    if env_p:
        p = Path(env_p).resolve()
        if has_project_markers(p):
            return p

    # 3) Discovery list
    candidates: list[Path] = []

    # Start CWD (parent processes often set this to the true base)
    candidates.append(START_CWD)

    # The exe / script directory (for one-folder builds this is usually the base)
    candidates.append(APP_DIR)

    # Walk up a couple of levels from both START_CWD and APP_DIR
    for root in (START_CWD, APP_DIR):
        for up in [root.parent, root.parent.parent, root.parent.parent.parent]:
            if up and up != up.parent:
                candidates.append(up)

    # Prefer those with markers
    for c in candidates:
        try:
            if has_project_markers(c):
                return c.resolve()
        except Exception:
            pass

    # Fallbacks: if nothing obvious, prefer START_CWD, then APP_DIR
    return (START_CWD if START_CWD.exists() else APP_DIR).resolve()

# -------------------------------
# Config (theme/CRS + A–E bins)
# -------------------------------
def read_config(file_name: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
    cfg.read(file_name, encoding="utf-8")
    return cfg

def read_config_classification(file_name: str) -> dict:
    """Read A–E bins & descriptions from config.ini."""
    cfg = configparser.ConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
    cfg.read(file_name, encoding="utf-8")
    classification.clear()
    for section in cfg.sections():
        if section in ['A','B','C','D','E']:
            rng = cfg[section].get('range','').strip()
            desc = cfg[section].get('description','').strip()
            if '-' in rng:
                try:
                    start, end = map(int, rng.split('-'))
                    classification[section] = {
                        'range': range(start, end + 1),
                        'description': desc
                    }
                except Exception:
                    pass
    return classification

def determine_category(score_int: int) -> tuple[str, str]:
    """Return (code, description) based on classification bins."""
    for cat, info in classification.items():
        if score_int in info['range']:
            return cat, info.get('description', '')
    return '', ''

def get_valid_values(cfg) -> list[int]:
    try:
        vals = [int(x.strip()) for x in cfg['VALID_VALUES']['valid_input'].split(',')]
        vals = [v for v in vals if 0 <= v <= 9999]
        return sorted(set(vals)) or [1,2,3,4,5]
    except Exception:
        return [1,2,3,4,5]

def get_fallback_value(cfg, valid_vals: list[int]) -> int:
    try:
        v = int(cfg['DEFAULT'].get('default_fallback_value', '3'))
        if v in valid_vals: return v
    except Exception:
        pass
    return int(np.median(valid_vals))

# -------------------------------
# Logging
# -------------------------------
def log_to_file(message: str) -> None:
    ts = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    line = f"{ts} - {message}"
    try:
        dest = os.path.join(original_working_directory, "log.txt")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def _ensure_default_header_present(path: str) -> None:
    try:
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        first = ""
        for line in txt.splitlines():
            s = line.strip()
            if not s or s.startswith(";") or s.startswith("#"):
                continue
            first = s
            break
        if first and not first.startswith("["):
            with open(path, "w", encoding="utf-8") as f:
                f.write("[DEFAULT]\n" + txt)
    except Exception:
        pass

def update_config_with_values(cfg_path: str, **kwargs) -> None:
    if not os.path.isabs(cfg_path):
        raise ValueError("cfg_path must be absolute")
    _ensure_default_header_present(cfg_path)
    if not os.path.isfile(cfg_path):
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write("[DEFAULT]\n")
    with open(cfg_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not any(line.strip().startswith('[') for line in lines):
        lines.insert(0, "[DEFAULT]\n")
    for key, value in kwargs.items():
        found = False
        key_norm = key.strip().casefold()
        for i, line in enumerate(lines):
            if "=" not in line:
                continue
            left, sep, right = line.partition("=")
            if sep == "":
                continue
            if left.strip().casefold() != key_norm:
                continue
            indent = left[: len(left) - len(left.lstrip())]
            lines[i] = f"{indent}{key} = {value}\n"
            found = True
            break
        if not found:
            lines.append(f"{key} = {value}\n")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

def _parse_weight_line(text: str, default: list[int]) -> list[int]:
    try:
        if not text:
            return default.copy()
        raw = [int(x.strip()) for x in str(text).replace(";", ",").split(",") if x.strip()]
        cleaned = [max(1, v) for v in raw[:5]]
        while len(cleaned) < 5:
            cleaned.append(default[len(cleaned)])
        return cleaned
    except Exception:
        return default.copy()

def load_index_weight_settings(cfg: configparser.ConfigParser) -> dict[str, list[int]]:
    settings: dict[str, list[int]] = {}
    for key, option in INDEX_WEIGHT_KEYS.items():
        default = INDEX_WEIGHT_DEFAULTS[key]
        raw = cfg["DEFAULT"].get(option, "")
        settings[key] = _parse_weight_line(raw, default)
    return settings

def persist_index_weight_settings(cfg_path: str, settings: dict[str, list[int]]) -> None:
    payload = {}
    for key, weights in settings.items():
        safe = [max(1, int(w)) for w in weights[:5]]
        while len(safe) < 5:
            safe.append(1)
        payload[INDEX_WEIGHT_KEYS[key]] = ",".join(str(v) for v in safe)
    update_config_with_values(cfg_path, **payload)


# -------------------------------
# Type & vulnerability helpers
# -------------------------------
def coerce_valid_int(text: str, valid_vals: list[int], fallback: int) -> int:
    try:
        v = int(float(str(text).strip()))
    except Exception:
        v = fallback
    v = max(min(v, max(valid_vals)), min(valid_vals))
    return int(min(valid_vals, key=lambda vv: abs(vv - v)))

def enforce_vuln_dtypes_inplace(df: pd.DataFrame) -> None:
    for col in ('importance', 'susceptibility', 'sensitivity'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    for c in ('sensitivity_code', 'sensitivity_description'):
        if c in df.columns:
            df[c] = df[c].astype('string')

def sanitize_vulnerability(df: pd.DataFrame,
                           valid_vals: list[int],
                           fallback: int) -> pd.DataFrame:
    df = df.copy()
    for col in ['importance', 'susceptibility']:
        if col not in df.columns:
            df[col] = fallback
    for col in ['importance', 'susceptibility']:
        s = pd.to_numeric(df[col], errors='coerce').round().astype('Int64')
        s = s.where(s.notna(), fallback)
        s = s.clip(min(valid_vals), max(valid_vals))
        s = s.apply(lambda x: min(valid_vals, key=lambda vv: abs(int(x) - vv)))
        df[col] = s.astype(int)
    df['sensitivity'] = (pd.to_numeric(df['importance'], errors='coerce').fillna(fallback)
                         * pd.to_numeric(df['susceptibility'], errors='coerce').fillna(fallback)).astype(int)

    def _cls(s):
        try:
            score = int(s)
        except Exception:
            score = fallback
        code, desc = determine_category(max(1, score))
        return pd.Series([code, desc], index=['sensitivity_code', 'sensitivity_description'])
    klass = df['sensitivity'].apply(_cls)
    df['sensitivity_code'] = klass['sensitivity_code']
    df['sensitivity_description'] = klass['sensitivity_description']
    return df

# -------------------------------
# Asset group I/O (Parquet only)
# -------------------------------
def _candidate_asset_group_paths(base_dir: str) -> list[Path]:
    base = Path(base_dir).resolve()
    primary = (base / PARQUET_ASSET_GROUP).resolve()
    candidates = [primary]
    if base.name.lower() != "code":
        candidates.append((base / "code" / PARQUET_ASSET_GROUP).resolve())
    return candidates

def _set_asset_group_override(path: Path):
    global ASSET_GROUP_OVERRIDE
    ASSET_GROUP_OVERRIDE = path.resolve()

def _parquet_asset_group_path(base_dir: str) -> str:
    if ASSET_GROUP_OVERRIDE is not None:
        return str(ASSET_GROUP_OVERRIDE)
    primary = (Path(base_dir).resolve() / PARQUET_ASSET_GROUP).resolve()
    return str(primary)

def _empty_asset_group_frame() -> gpd.GeoDataFrame:
    cols = [
        'id',
        'name_original',
        'name_gis_assetgroup',
        'title_fromuser',
        'total_asset_objects',
        'importance',
        'susceptibility',
        'sensitivity',
        'sensitivity_code',
        'sensitivity_description',
        'geometry',
    ]
    return gpd.GeoDataFrame(columns=cols, geometry='geometry', crs=f"EPSG:{workingprojection_epsg}")


def _coerce_geometry_series(raw: pd.Series):
    def _to_geom(val):
        if val is None:
            return None
        if isinstance(val, float) and np.isnan(val):
            return None
        try:
            if isinstance(val, (bytes, bytearray, memoryview)):
                return _shapely_wkb.loads(bytes(val))
            text = str(val).strip()
            if not text:
                return None
            # Try WKB hex first (starts with 0/1)
            try:
                return _shapely_wkb.loads(bytes.fromhex(text))
            except Exception:
                return _shapely_wkt.loads(text)
        except Exception:
            return None

    return raw.apply(_to_geom)


def _load_asset_group_without_geo_metadata(target: Path) -> gpd.GeoDataFrame:
    try:
        df = pd.read_parquet(target)
    except Exception as err:
        log_to_file(f"Fallback pandas.read_parquet failed for tbl_asset_group: {err}")
        return _empty_asset_group_frame()

    geom_col = None
    for candidate in ("geometry", "geometry_wkb", "geometry_wkt"):
        if candidate in df.columns:
            geom_col = candidate
            break

    if geom_col is None:
        log_to_file("tbl_asset_group fallback could not find a geometry column; showing empty form.")
        return _empty_asset_group_frame()

    geoms = _coerce_geometry_series(df[geom_col])
    data = df.drop(columns=[geom_col])
    gdf = gpd.GeoDataFrame(data, geometry=geoms, crs=f"EPSG:{workingprojection_epsg}")
    try:
        gdf = gdf[gdf.geometry.notna()]
    except Exception:
        pass
    return gdf

def load_asset_group(base_dir: str) -> gpd.GeoDataFrame:
    candidates = _candidate_asset_group_paths(base_dir)
    target: Optional[Path] = None
    for idx, cand in enumerate(candidates):
        if cand.exists():
            target = cand
            if idx > 0:
                log_to_file(f"tbl_asset_group found in fallback location: {cand}")
                _set_asset_group_override(cand)
            break

    if target is None:
        target = candidates[0]
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            empty = _empty_asset_group_frame()
            empty.to_parquet(target, index=False)
            log_to_file(f"Initialized blank tbl_asset_group at {target}")
        except Exception as e:
            log_to_file(f"Failed to initialise blank asset group parquet at {target}: {e}")
        return _empty_asset_group_frame()

    try:
        gdf = gpd.read_parquet(target)
    except ValueError as err:
        msg = str(err)
        if "Missing geo metadata" in msg or "Use pandas.read_parquet" in msg:
            log_to_file("Geo metadata missing in tbl_asset_group; reconstructing geometry via pandas fallback.")
            gdf = _load_asset_group_without_geo_metadata(target)
        else:
            log_to_file(f"Failed reading asset group parquet: {err}")
            return _empty_asset_group_frame()
    except Exception as e:
        log_to_file(f"Failed reading asset group parquet: {e}")
        return _empty_asset_group_frame()

    if gdf is None or gdf.empty:
        return _empty_asset_group_frame()

    if gdf.crs is None:
        try:
            gdf.set_crs(epsg=int(workingprojection_epsg), inplace=True)
        except Exception:
            gdf.set_crs(f"EPSG:{workingprojection_epsg}", inplace=True, allow_override=True)
    gdf = sanitize_vulnerability(gdf, valid_input_values, FALLBACK_VULN)
    enforce_vuln_dtypes_inplace(gdf)
    return gdf

def save_asset_group_to_parquet(gdf: gpd.GeoDataFrame, base_dir: str):
    try:
        if gdf is None:
            return
        gdf2 = sanitize_vulnerability(gdf, valid_input_values, FALLBACK_VULN)
        enforce_vuln_dtypes_inplace(gdf2)
        out = _parquet_asset_group_path(base_dir)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        gdf2.to_parquet(out, index=False)
        log_to_file(f"Saved tbl_asset_group to GeoParquet: {out}")
    except Exception as e:
        log_to_file(f"Save asset group parquet failed: {e}")

# -------------------------------
# Excel round-trip
# -------------------------------
def save_all_to_excel(gdf: pd.DataFrame, excel_path: str):
    try:
        vuln_cols = ['id','name_original','susceptibility','importance','sensitivity','sensitivity_code','sensitivity_description']
        vcols = [c for c in vuln_cols if c in gdf.columns]
        vuln = gdf[vcols].copy() if vcols else pd.DataFrame(columns=vuln_cols)
        with pd.ExcelWriter(excel_path, engine='openpyxl') as xw:
            vuln.to_excel(xw, sheet_name='vulnerability', index=False)
        messagebox.showinfo("Saved", "Saved all tabs to Excel.")
    except Exception as e:
        log_to_file(f"Excel save failed: {e}")
        messagebox.showerror("Error", f"Failed saving to Excel:\n{e}")

def _apply_vulnerability_from_df(df_x: pd.DataFrame):
    global gdf_asset_group
    if df_x is None or df_x.empty: return
    for col in ['importance','susceptibility']:
        if col in df_x.columns:
            s = pd.to_numeric(df_x[col], errors='coerce').round().astype('Int64')
            s = s.where(s.notna(), FALLBACK_VULN)
            s = s.clip(min(valid_input_values), max(valid_input_values))
            s = s.apply(lambda v: min(valid_input_values, key=lambda vv: abs(int(v)-vv)))
            df_x[col] = s.astype(int)
    key = 'id' if 'id' in gdf_asset_group.columns else 'name_original'
    kx = 'id' if 'id' in df_x.columns else 'name_original'
    upd = df_x[[kx]+[c for c in ['importance','susceptibility'] if c in df_x.columns]].drop_duplicates(subset=[kx])
    merged = gdf_asset_group.merge(upd, left_on=key, right_on=kx, how='left', suffixes=('', '_xlsx'))
    for col in ['importance','susceptibility']:
        xc = f"{col}_xlsx"
        if xc in merged.columns:
            merged[col] = merged[xc].where(merged[xc].notna(), merged[col])
            merged.drop(columns=[xc], inplace=True, errors='ignore')
    merged = sanitize_vulnerability(merged, valid_input_values, FALLBACK_VULN)
    enforce_vuln_dtypes_inplace(merged)
    gdf_asset_group = merged

def load_all_from_excel(excel_path: str):
    try:
        x = pd.read_excel(excel_path, sheet_name=None)
        if 'vulnerability' in x: _apply_vulnerability_from_df(x['vulnerability'])
        refresh_vulnerability_grid_from_df()
        messagebox.showinfo("Loaded", "All settings and values were loaded from Excel.")
    except Exception as e:
        log_to_file(f"Excel load failed: {e}")
        messagebox.showerror("Error", f"Failed reading Excel:\n{e}")

# -------------------------------
# Start tab
# -------------------------------
def build_start_tab(parent):
    frm = ttkb.Frame(parent)
    frm.pack(fill='both', expand=True, padx=12, pady=12)

    info = (
        "Welcome to the setup utility.\n\n"
        "• Use the Vulnerability tab to register importance and susceptibility per asset; sensitivity and the A–E "
        "classification update automatically.\n"
        "• The shortcuts below let you round-trip data to Excel or persist changes back to GeoParquet.\n\n"
        "All files are written under the working directory that launched this helper."
    )
    ttkb.Label(frm, text=info, justify='left', wraplength=900).pack(anchor='w', pady=(0, 12))

    btns = ttkb.Frame(frm)
    btns.pack(anchor='w', pady=6)

    def do_save_all_excel():
        input_folder = os.path.join(original_working_directory, "input")
        os.makedirs(input_folder, exist_ok=True)
        excel_path = filedialog.asksaveasfilename(
            title="Save Excel File",
            initialdir=input_folder,
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")]
        )
        if excel_path:
            save_all_to_excel(gdf_asset_group, excel_path)

    def do_load_all_excel():
        input_folder = os.path.join(original_working_directory, "input")
        excel_path = filedialog.askopenfilename(
            title="Select Excel File",
            initialdir=input_folder,
            filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")]
        )
        if excel_path:
            load_all_from_excel(excel_path)

    def do_save_parquet():
        update_all_vuln_rows(entries_vuln, gdf_asset_group)
        enforce_vuln_dtypes_inplace(gdf_asset_group)
        gdf_ready = sanitize_vulnerability(gdf_asset_group, valid_input_values, FALLBACK_VULN)
        save_asset_group_to_parquet(gdf_ready, original_working_directory)
        messagebox.showinfo("Saved", "Saved asset-group layer to GeoParquet.")

    ttkb.Button(btns, text="Save all to Excel", command=do_save_all_excel, bootstyle=SUCCESS).pack(side='left', padx=6)
    ttkb.Button(btns, text="Load all from Excel", command=do_load_all_excel, bootstyle=INFO).pack(side='left', padx=6)
    ttkb.Button(btns, text="Save to Parquet", command=do_save_parquet, bootstyle=PRIMARY).pack(side='left', padx=6)

def build_indexes_tab(parent):
    global index_weight_settings
    frm = ttkb.Frame(parent)
    frm.pack(fill='both', expand=True, padx=12, pady=12)

    info = (
        "Configure weighting for the new importance and sensitivity indexes.\n"
        "Each value column corresponds to the input value (1–5) stored in tbl_stacked.\n"
        "Weights must be positive integers; higher weights increase the contribution when\n"
        "counting overlapping assets inside each mosaic cell."
    )
    ttkb.Label(frm, text=info, justify='left', wraplength=900).pack(anchor='w', pady=(0, 12))

    sections = [
        ("importance", "Importance index weights"),
        ("sensitivity", "Sensitivity index weights"),
    ]
    weight_vars: dict[str, list[tk.StringVar]] = {}

    for key, title in sections:
        box = ttkb.LabelFrame(frm, text=title, bootstyle="info")
        box.pack(fill='x', pady=8)
        ttkb.Label(box, text="Value", width=10, anchor='center').grid(row=0, column=0, padx=5, pady=(6, 4))
        for v in range(1, 6):
            ttkb.Label(box, text=str(v), width=6, anchor='center').grid(row=0, column=v, padx=2, pady=(6, 4))
        ttkb.Label(box, text="Weight", width=10, anchor='center').grid(row=1, column=0, padx=5, pady=4)

        vars_for_key: list[tk.StringVar] = []
        current = index_weight_settings.get(key, INDEX_WEIGHT_DEFAULTS[key])
        for idx in range(5):
            var = tk.StringVar(value=str(current[idx] if idx < len(current) else 1))
            entry = ttkb.Entry(box, width=6, justify='center', textvariable=var)
            entry.grid(row=1, column=idx + 1, padx=2, pady=4)
            vars_for_key.append(var)
        weight_vars[key] = vars_for_key

    def save_weight_settings():
        updated: dict[str, list[int]] = {}
        for key, vars_list in weight_vars.items():
            values: list[int] = []
            for idx, var in enumerate(vars_list, start=1):
                txt = var.get().strip()
                if not txt.isdigit():
                    messagebox.showerror("Indexes", f"Weight for value {idx} in {key} must be a positive integer.")
                    return
                val = int(txt)
                if val < 1:
                    messagebox.showerror("Indexes", f"Weight for value {idx} in {key} must be at least 1.")
                    return
                values.append(val)
            updated[key] = values
        try:
            persist_index_weight_settings(config_file, updated)
            index_weight_settings.update(updated)
            messagebox.showinfo("Indexes", "Index weights saved to config.ini.")
        except Exception as err:
            log_to_file(f"Failed to persist index weights: {err}")
            messagebox.showerror("Indexes", f"Could not save index weights:\n{err}")

    ttkb.Button(frm, text="Save weights", bootstyle=PRIMARY, command=save_weight_settings).pack(anchor='w', pady=(12, 0))

# -------------------------------
# Vulnerability UI
# -------------------------------
def setup_headers_vuln(frame, column_widths):
    headers = ["Dataset", "Importance", "Susceptibility", "Sensitivity", "Code", "Description"]
    for idx, header in enumerate(headers):
        label = ttkb.Label(frame, text=header, anchor='w', width=column_widths[idx])
        label.grid(row=0, column=idx, padx=5, pady=5, sticky='ew')
    return frame

def _entry_validator_vuln(P: str) -> bool:
    if P == "": return True
    return P.isdigit() and int(P) in valid_input_values

def add_vuln_row(index, row, frame, entries_list, gdf):
    vcmd = (frame.register(_entry_validator_vuln), "%P")
    e_imp = ttkb.Entry(frame, width=column_widths[1], validate="key", validatecommand=vcmd)
    e_imp.insert(0, str(getattr(row, 'importance', FALLBACK_VULN))); e_imp.grid(row=index, column=1, padx=5)
    e_sus = ttkb.Entry(frame, width=column_widths[2], validate="key", validatecommand=vcmd)
    e_sus.insert(0, str(getattr(row, 'susceptibility', FALLBACK_VULN))); e_sus.grid(row=index, column=2, padx=5)
    e_imp.bind('<KeyRelease>', lambda _e, imp=e_imp, sus=e_sus, idx=index-1: calculate_sensitivity(imp, sus, idx, entries_list, gdf))
    e_sus.bind('<KeyRelease>', lambda _e, imp=e_imp, sus=e_sus, idx=index-1: calculate_sensitivity(imp, sus, idx, entries_list, gdf))
    l_name = ttkb.Label(frame, text=getattr(row, 'name_original', ''), anchor='w', width=column_widths[0]); l_name.grid(row=index, column=0, padx=5, sticky='ew')
    l_sens = ttkb.Label(frame, text=str(getattr(row, 'sensitivity', '')), anchor='w', width=column_widths[3]); l_sens.grid(row=index, column=3, padx=5, sticky='ew')
    l_code = ttkb.Label(frame, text=str(getattr(row, 'sensitivity_code', '')), anchor='w', width=column_widths[4]); l_code.grid(row=index, column=4, padx=5, sticky='ew')
    l_desc = ttkb.Label(frame, text=str(getattr(row, 'sensitivity_description', '')), anchor='w', width=column_widths[5]); l_desc.grid(row=index, column=5, padx=5, sticky='ew')
    entries_list.append({'row_index': index-1, 'name': l_name, 'importance': e_imp, 'susceptibility': e_sus,
                         'sensitivity': l_sens, 'sensitivity_code': l_code, 'sensitivity_description': l_desc})

def create_scrollable_area(parent):
    outer = ttkb.Frame(parent)
    canvas = tk.Canvas(outer, highlightthickness=0)
    scroll_y = ttkb.Scrollbar(outer, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scroll_y.set)
    canvas.bind('<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.pack(side=tk.LEFT, fill="both", expand=True)
    scroll_y.pack(side=tk.RIGHT, fill="y")
    outer.pack(side=tk.TOP, fill="both", expand=True)
    inner = ttkb.Frame(canvas)
    canvas.create_window((0, 0), window=inner, anchor="nw")
    return canvas, inner

def calculate_sensitivity(entry_importance, entry_susceptibility, index, entries_list, gdf):
    try:
        imp = coerce_valid_int(entry_importance.get().strip(), valid_input_values, FALLBACK_VULN)
        sus = coerce_valid_int(entry_susceptibility.get().strip(), valid_input_values, FALLBACK_VULN)
        sensitivity = int(imp) * int(sus)
        code, desc = determine_category(max(1, sensitivity))
        entries_list[index]['sensitivity']['text'] = str(int(sensitivity))
        entries_list[index]['sensitivity_code']['text'] = str(code)
        entries_list[index]['sensitivity_description']['text'] = str(desc)
        gdf.at[index, 'importance'] = int(imp)
        gdf.at[index, 'susceptibility'] = int(sus)
        gdf.at[index, 'sensitivity'] = int(sensitivity)
        gdf.at[index, 'sensitivity_code'] = str(code)
        gdf.at[index, 'sensitivity_description'] = str(desc)
    except Exception as e:
        log_to_file(f"Input Error: {e}")

def refresh_vulnerability_grid_from_df():
    for entry in entries_vuln:
        name_original = entry['name'].cget("text")
        mask = (gdf_asset_group['name_original'] == name_original)
        if mask.any():
            idx = gdf_asset_group[mask].index[0]
            entry['importance'].delete(0, tk.END)
            entry['importance'].insert(0, str(int(gdf_asset_group.at[idx, 'importance'])))
            entry['susceptibility'].delete(0, tk.END)
            entry['susceptibility'].insert(0, str(int(gdf_asset_group.at[idx, 'susceptibility'])))
            entry['sensitivity']['text'] = str(int(gdf_asset_group.at[idx, 'sensitivity']))
            entry['sensitivity_code']['text'] = str(gdf_asset_group.at[idx, 'sensitivity_code'])
            entry['sensitivity_description']['text'] = str(gdf_asset_group.at[idx, 'sensitivity_description'])

def update_all_vuln_rows(entries_list, gdf):
    """Sync all UI entry values back into dataframe before saving."""
    name_map = {row['name'].cget("text"): row for row in entries_list}
    for idx, row in gdf.iterrows():
        name = row.get('name_original', '')
        ui = name_map.get(name)
        if not ui:
            continue
        imp = coerce_valid_int(ui['importance'].get().strip(), valid_input_values, FALLBACK_VULN)
        sus = coerce_valid_int(ui['susceptibility'].get().strip(), valid_input_values, FALLBACK_VULN)
        sens = imp * sus
        code, desc = determine_category(sens)
        gdf.at[idx, 'importance'] = imp
        gdf.at[idx, 'susceptibility'] = sus
        gdf.at[idx, 'sensitivity'] = sens
        gdf.at[idx, 'sensitivity_code'] = code
        gdf.at[idx, 'sensitivity_description'] = desc

# -------------------------------
# Misc helpers
# -------------------------------
def close_application():
    try:
        update_all_vuln_rows(entries_vuln, gdf_asset_group)
        enforce_vuln_dtypes_inplace(gdf_asset_group)
        gdf_ready = sanitize_vulnerability(gdf_asset_group, valid_input_values, FALLBACK_VULN)
        save_asset_group_to_parquet(gdf_ready, original_working_directory)
    finally:
        root.destroy()

# -------------------------------
# Entrypoint (single, Parquet+JSON)
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MESA – Setup & Registration (GeoParquet)')
    parser.add_argument('--original_working_directory', required=False, help='Path to running folder')
    args = parser.parse_args()

    # Resolve base dir robustly
    resolved_base = find_base_dir(args.original_working_directory)
    original_working_directory = str(resolved_base)

    # ---- tiny diagnostics (helps catch path mistakes fast)
    print("[parametres_setup] start_cwd:", START_CWD)
    print("[parametres_setup] app_dir  :", APP_DIR)
    print("[parametres_setup] base_dir :", original_working_directory)

    # Config
    config_file = os.path.join(original_working_directory, "config.ini")
    config = read_config(config_file)
    classification = read_config_classification(config_file)
    valid_input_values = get_valid_values(config)
    FALLBACK_VULN = get_fallback_value(config, valid_input_values)
    index_weight_settings = load_index_weight_settings(config)
    ttk_bootstrap_theme = config['DEFAULT'].get('ttk_bootstrap_theme', 'flatly')
    workingprojection_epsg = config['DEFAULT'].get('workingprojection_epsg', '4326')
    # Asset groups
    gdf_asset_group = load_asset_group(original_working_directory)
    print("[parametres_setup] asset grp:", _parquet_asset_group_path(original_working_directory))
    if gdf_asset_group is None:
        log_to_file("Failed to load tbl_asset_group (Parquet).")
        sys.exit(1)

    # UI
    root = tb.Window(themename=ttk_bootstrap_theme)
    root.title("MESA – Setup & Registration (GeoParquet)")
    try:
        icon_path = resource_path(Path("system_resources") / "mesa.ico")
        if icon_path.exists():
            root.iconbitmap(str(icon_path))
    except Exception:
        pass
    root.geometry("1100x860")

    nb = ttkb.Notebook(root, bootstyle=PRIMARY); nb.pack(fill='both', expand=True)

    # Start
    tab_start = ttkb.Frame(nb); nb.add(tab_start, text="Start")
    build_start_tab(tab_start)

    tab_idx = ttkb.Frame(nb); nb.add(tab_idx, text="Indexes")
    build_indexes_tab(tab_idx)

    # Vulnerability
    tab_vuln = ttkb.Frame(nb); nb.add(tab_vuln, text="Sensitivity")
    canvas_v, frame_v = create_scrollable_area(tab_vuln)
    entries_vuln = []
    setup_headers_vuln(frame_v, column_widths)
    for i, row in enumerate(gdf_asset_group.itertuples(), start=1):
        add_vuln_row(i, row, frame_v, entries_vuln, gdf_asset_group)
    frame_v.update_idletasks(); canvas_v.configure(scrollregion=canvas_v.bbox("all"))

    root.mainloop()
