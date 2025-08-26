# -*- coding: utf-8 -*-
# MESA – Setup & Registration (4 tabs)
# - EBSA per-row default = 50 (UI), stored as 0.5
# - Profiles (EBSA & Visualization) persisted to: GPKG tables + GeoParquet + JSON
# - Load priority for profiles: GPKG -> Parquet -> JSON -> defaults, then mirror to all
# - Asset-group vulnerability values (importance, susceptibility, sensitivity, A–E) are
#   always validated, recomputed from config bins and saved to BOTH stores.

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

import os
import sys
import argparse
import configparser
import datetime
import json
import sqlite3

import numpy as np
import pandas as pd
import geopandas as gpd

import tkinter as tk
from tkinter import messagebox, filedialog

import ttkbootstrap as tb
from ttkbootstrap.constants import *
from ttkbootstrap import ttk as ttkb  # themed ttk widgets

# -------------------------------
# Defaults & constants
# -------------------------------
DEFAULT_EBSA_PROFILE = {
    'w_R': 0.10, 'w_U': 0.30, 'w_V': 0.25, 'w_T': 0.30, 'w_F': 0.05,
    'gamma': 0.0, 'pnorm': 4.0, 'thr_primary': 0.80, 'thr_support': 0.60,
}
DEFAULT_ENV_PROFILE = {
    'w_sensitivity': 0.35, 'w_susceptibility': 0.25, 'w_importance': 0.20, 'w_pressure': 0.20,
    'gamma': 0.0, 'pnorm_minmax': 4.0, 'overlap_cap_q': 0.95,
    'scoring': 'linear', 'logistic_a': 8.0, 'logistic_b': 0.6,
}
EBSA_PER_ROW_DEFAULT_01 = 0.5

# --- constants (GeoParquet lives alongside other parquet files) ---
PARQUET_DIRNAME = os.path.join("output", "geoparquet")
PARQUET_ASSET_GROUP = os.path.join(PARQUET_DIRNAME, "tbl_asset_group.parquet")

# UI grid helpers
column_widths = [35, 13, 13, 13, 13, 30]
valid_input_values = []
classification = {}
entries_vuln = []
entries_ebsa = []
FALLBACK_VULN = 3

EBSA_PROFILE = {}
ENV_PROFILE  = {}

def _ensure_ebsa_profile_defaults():
    """Guarantee EBSA_PROFILE has all default keys/values for first-run UI."""
    global EBSA_PROFILE
    EBSA_PROFILE = {**DEFAULT_EBSA_PROFILE, **(EBSA_PROFILE or {})}

# paths set in __main__
original_working_directory = ""
config_file = ""
gpkg_file = ""
workingprojection_epsg = "4326"

# -------------------------------
# Config (theme/CRS + A–E bins)
# -------------------------------
def read_config(file_name: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(file_name, encoding="utf-8")
    return cfg

def read_config_classification(file_name: str) -> dict:
    """Read A–E bins & descriptions from config.ini."""
    cfg = configparser.ConfigParser()
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
                    # ignore malformed range, leave section out
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
        with open(dest, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

# -------------------------------
# Type helpers (avoid FutureWarning)
# -------------------------------
def coerce_valid_int(text: str, valid_vals: list[int], fallback: int) -> int:
    """Parse Tk entry text to an allowed int value."""
    try:
        v = int(float(str(text).strip()))
    except Exception:
        v = fallback
    v = max(min(v, max(valid_vals)), min(valid_vals))
    return int(min(valid_vals, key=lambda vv: abs(vv - v)))

def enforce_vuln_dtypes_inplace(df: pd.DataFrame) -> None:
    """Ensure integer cols are Int64; texts are string dtype."""
    for col in ('importance', 'susceptibility', 'sensitivity'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    for c in ('sensitivity_code', 'sensitivity_description'):
        if c in df.columns:
            df[c] = df[c].astype('string')

def enforce_ebsa_dtypes_inplace(df: pd.DataFrame) -> None:
    for c in ['ebsa_R','ebsa_U','ebsa_V','ebsa_T','ebsa_F']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype(float).clip(0.0, 1.0)

# -------------------------------
# Vulnerability sanitization
# -------------------------------
def sanitize_vulnerability(df: pd.DataFrame,
                           valid_vals: list[int],
                           fallback: int) -> pd.DataFrame:
    """
    Validate importance/susceptibility -> compute sensitivity,
    then classify to code + description using config.ini bins.
    """
    df = df.copy()

    # Ensure columns exist
    for col in ['importance', 'susceptibility']:
        if col not in df.columns:
            df[col] = np.nan

    # Snap to valid set
    for col in ['importance', 'susceptibility']:
        s = pd.to_numeric(df[col], errors='coerce').round().astype('Int64')
        s = s.where(s.notna(), fallback)
        s = s.clip(min(valid_vals), max(valid_vals))
        # snap to nearest allowed value
        s = s.apply(lambda x: min(valid_vals, key=lambda vv: abs(int(x) - vv)))
        df[col] = s.astype(int)

    # sensitivity = product
    df['sensitivity'] = (pd.to_numeric(df['importance'], errors='coerce').fillna(fallback)
                         * pd.to_numeric(df['susceptibility'], errors='coerce').fillna(fallback)).astype(int)

    # classify to A–E
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

def best_join_key(df: pd.DataFrame) -> str:
    return 'id' if 'id' in df.columns else 'name_original'

# -------------------------------
# EBSA per-row columns
# -------------------------------
def ensure_ebsa_columns_with_defaults(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = df.copy()
    for c in ['ebsa_R','ebsa_U','ebsa_V','ebsa_T','ebsa_F']:
        if c not in out.columns:
            out[c] = EBSA_PER_ROW_DEFAULT_01
        else:
            out[c] = pd.to_numeric(out[c], errors='coerce').where(
                lambda s: s.notna(), EBSA_PER_ROW_DEFAULT_01
            ).clip(0.0, 1.0)
    return out

# -------------------------------
# Profile storage: GPKG
# -------------------------------
def _gpkg_read_profile_table(gpkg_path: str, table_name: str) -> dict | None:
    try:
        con = sqlite3.connect(gpkg_path)
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not cur.fetchone():
            con.close(); return None
        cur.execute(f"SELECT key, value FROM {table_name}")
        rows = cur.fetchall(); con.close()
        out = {}
        for k, v in rows:
            try: out[k] = json.loads(v)
            except Exception: out[k] = v
        return out
    except Exception as e:
        log_to_file(f"GPKG read {table_name} failed: {e}")
        return None

def _gpkg_write_profile_table(gpkg_path: str, table_name: str, data: dict):
    try:
        con = sqlite3.connect(gpkg_path)
        cur = con.cursor()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        cur.execute(f"DELETE FROM {table_name};")
        for k, v in (data or {}).items():
            cur.execute(f"INSERT INTO {table_name}(key, value) VALUES (?, ?)", (k, json.dumps(v)))
        con.commit(); con.close()
    except Exception as e:
        log_to_file(f"GPKG write {table_name} failed: {e}")

# -------------------------------
# Profile storage: GeoParquet
# -------------------------------
def _profile_parquet_path(base_dir: str, name: str) -> str:
    """New location: output/geoparquet/<name>.parquet"""
    return os.path.join(base_dir, PARQUET_DIRNAME, f"{name}.parquet")

def _legacy_profile_parquet_path(base_dir: str, name: str) -> str:
    """Legacy location we still read from if present: output/geoparquet/settings/<name>.parquet"""
    return os.path.join(base_dir, PARQUET_DIRNAME, "settings", f"{name}.parquet")

def _parquet_read_profile(base_dir: str, name: str) -> dict | None:
    """
    Try new flat location first, then legacy settings/ subfolder for backward compatibility.
    """
    for candidate in (_profile_parquet_path(base_dir, name),
                      _legacy_profile_parquet_path(base_dir, name)):
        try:
            if not os.path.exists(candidate):
                continue
            df = pd.read_parquet(candidate)
            if {'key','value'}.issubset(df.columns):
                out = {}
                for _, r in df.iterrows():
                    try:
                        out[str(r['key'])] = json.loads(r['value'])
                    except Exception:
                        out[str(r['key'])] = r['value']
                return out
            if len(df) == 1:
                return df.iloc[0].to_dict()
        except Exception as e:
            log_to_file(f"Parquet read {name} failed at {candidate}: {e}")
    return None

def _parquet_write_profile(base_dir: str, name: str, data: dict):
    """Always write to the new flat location under output/geoparquet/."""
    try:
        path = _profile_parquet_path(base_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame({
            'key': list(data.keys()),
            'value': [json.dumps(v) for v in data.values()]
        })
        df.to_parquet(path, index=False)
        log_to_file(f"Wrote profile {name} to GeoParquet: {path}")
    except Exception as e:
        log_to_file(f"Parquet write {name} failed: {e}")

# -------------------------------
# Profile storage: JSON
# -------------------------------
def _json_read(path: str) -> dict | None:
    try:
        if not os.path.exists(path): return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_to_file(f"JSON read {path} failed: {e}")
        return None

def _json_write(path: str, data: dict):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log_to_file(f"JSON write {path} failed: {e}")

# -------------------------------
# Load & persist profiles (tri-store)
# -------------------------------
def load_profiles(gpkg_path: str, base_dir: str):
    """Load EBSA/ENV profiles with priority: GPKG -> Parquet -> JSON -> defaults. Then mirror to all."""
    global EBSA_PROFILE, ENV_PROFILE

    # EBSA
    ebsa = _gpkg_read_profile_table(gpkg_path, "tbl_ebsa_profile")
    if not ebsa: ebsa = _parquet_read_profile(base_dir, "tbl_ebsa_profile")
    if not ebsa:
        ebsa_json = os.path.join(base_dir, "output", "settings", "ebsa_profile.json")
        ebsa = _json_read(ebsa_json)
    EBSA_PROFILE = (DEFAULT_EBSA_PROFILE | ebsa) if ebsa else DEFAULT_EBSA_PROFILE.copy()

    # ENV
    env = _gpkg_read_profile_table(gpkg_path, "tbl_env_profile")
    if not env: env = _parquet_read_profile(base_dir, "tbl_env_profile")
    if not env:
        env_json = os.path.join(base_dir, "output", "settings", "env_index_profile.json")
        env = _json_read(env_json)
    ENV_PROFILE = (DEFAULT_ENV_PROFILE | env) if env else DEFAULT_ENV_PROFILE.copy()

    # Mirror to all stores to keep them in sync
    persist_profiles(gpkg_path, base_dir)

def persist_profiles(gpkg_path: str, base_dir: str):
    """Write EBSA/ENV profiles to GPKG tables, GeoParquet, and JSON."""
    # GPKG
    _gpkg_write_profile_table(gpkg_path, "tbl_ebsa_profile", EBSA_PROFILE)
    _gpkg_write_profile_table(gpkg_path, "tbl_env_profile",  ENV_PROFILE)
    # Parquet
    _parquet_write_profile(base_dir, "tbl_ebsa_profile", EBSA_PROFILE)
    _parquet_write_profile(base_dir, "tbl_env_profile",  ENV_PROFILE)
    # JSON
    settings_dir = os.path.join(base_dir, "output", "settings")
    _json_write(os.path.join(settings_dir, "ebsa_profile.json"), EBSA_PROFILE)
    _json_write(os.path.join(settings_dir, "env_index_profile.json"), ENV_PROFILE)

# -------------------------------
# Data I/O (GPKG / Parquet)
# -------------------------------
def _parquet_asset_group_path(base_dir: str) -> str:
    return os.path.join(base_dir, PARQUET_ASSET_GROUP)

def load_asset_group(gpkg_path: str) -> gpd.GeoDataFrame | None:
    """
    Load tbl_asset_group; prefer GPKG, but if unavailable try the Parquet mirror.
    After loading, validate & compute vulnerability columns and EBSA defaults.
    """
    gdf = None

    # 1) Try GPKG
    try:
        gdf = gpd.read_file(gpkg_path, layer="tbl_asset_group")
    except Exception as e:
        log_to_file(f"GPKG read tbl_asset_group failed: {e}")

    # 2) Fallback to Parquet
    if gdf is None or gdf.empty:
        try:
            pq = _parquet_asset_group_path(original_working_directory)
            if os.path.exists(pq):
                gdf = gpd.read_parquet(pq)
                log_to_file("Loaded tbl_asset_group from GeoParquet fallback.")
        except Exception as e:
            log_to_file(f"Parquet read tbl_asset_group failed: {e}")

    if gdf is None:
        return None

    # Geometry column normalization
    if 'geom' in gdf.columns and getattr(gdf, "geometry", None) is not None and gdf.geometry.name != 'geom':
        try:
            gdf.set_geometry('geom', inplace=True)
        except Exception:
            pass

    # Vulnerability + EBSA defaults + dtypes
    gdf = sanitize_vulnerability(gdf, valid_input_values, FALLBACK_VULN)
    gdf = ensure_ebsa_columns_with_defaults(gdf)
    enforce_vuln_dtypes_inplace(gdf)
    enforce_ebsa_dtypes_inplace(gdf)
    return gdf

def _ensure_geometry_column(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop extra geometry cols and keep a single active one named 'geometry'."""
    geom_cols = [c for c in gdf.columns if str(gdf.dtypes.get(c)) == 'geometry']
    if not geom_cols:
        return gdf
    main = geom_cols[0]
    if gdf.geometry.name != main:
        gdf.set_geometry(main, inplace=True)
    for c in geom_cols:
        if c != main:
            gdf.drop(columns=[c], inplace=True, errors='ignore')
    return gdf

def save_asset_group_to_gpkg(gdf: gpd.GeoDataFrame, gpkg_path: str):
    try:
        if gdf is None or gdf.empty:
            log_to_file("tbl_asset_group empty – nothing to save to GPKG.")
            return
        # Recompute before save (safety)
        gdf = sanitize_vulnerability(gdf, valid_input_values, FALLBACK_VULN)
        enforce_vuln_dtypes_inplace(gdf)
        enforce_ebsa_dtypes_inplace(gdf)
        gdf = _ensure_geometry_column(gdf)
        if gdf.crs is None:
            gdf.set_crs(epsg=int(workingprojection_epsg), inplace=True)
        gdf.to_file(filename=gpkg_path, layer='tbl_asset_group', driver='GPKG')
        log_to_file("Saved tbl_asset_group to GeoPackage.")
    except Exception as e:
        log_to_file(f"Save to GPKG failed: {e}")

def save_asset_group_to_parquet(gdf: gpd.GeoDataFrame, base_dir: str):
    try:
        if gdf is None or gdf.empty:
            log_to_file("tbl_asset_group empty – nothing to save to Parquet.")
            return
        # Recompute before save (safety)
        gdf = sanitize_vulnerability(gdf, valid_input_values, FALLBACK_VULN)
        enforce_vuln_dtypes_inplace(gdf)
        enforce_ebsa_dtypes_inplace(gdf)
        gdf = _ensure_geometry_column(gdf)

        out = _parquet_asset_group_path(base_dir)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        gdf.to_parquet(out, index=False)
        log_to_file(f"Saved tbl_asset_group to GeoParquet: {out}")
    except Exception as e:
        log_to_file(f"Save to Parquet failed: {e}")

# -------------------------------
# Excel round-trip (4 sheets)
# -------------------------------
def save_all_to_excel(gdf: pd.DataFrame, excel_path: str):
    try:
        vuln_cols = ['id','name_original','susceptibility','importance','sensitivity','sensitivity_code','sensitivity_description']
        vcols = [c for c in vuln_cols if c in gdf.columns]
        vuln = gdf[vcols].copy() if vcols else pd.DataFrame(columns=vuln_cols)

        ebsa_prof = pd.DataFrame([EBSA_PROFILE])

        ebsa_cols = ['id','name_original','ebsa_R','ebsa_U','ebsa_V','ebsa_T','ebsa_F']
        ecols = [c for c in ebsa_cols if c in gdf.columns]
        ebsa_scores = gdf[ecols].copy() if ecols else pd.DataFrame(columns=ebsa_cols)
        for c in ['ebsa_R','ebsa_U','ebsa_V','ebsa_T','ebsa_F']:
            if c in ebsa_scores.columns:
                ebsa_scores[c] = (pd.to_numeric(ebsa_scores[c], errors='coerce') * 100.0).round(2)

        env_prof = pd.DataFrame([ENV_PROFILE])

        with pd.ExcelWriter(excel_path, engine='openpyxl') as xw:
            vuln.to_excel(xw, sheet_name='vulnerability', index=False)
            ebsa_prof.to_excel(xw, sheet_name='ebsa_profile', index=False)
            ebsa_scores.to_excel(xw, sheet_name='ebsa_scores', index=False)
            env_prof.to_excel(xw, sheet_name='env_index', index=False)

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
    key_df = best_join_key(gdf_asset_group)
    key_x  = 'id' if 'id' in df_x.columns else 'name_original'
    upd = df_x[[key_x]+[c for c in ['importance','susceptibility'] if c in df_x.columns]].drop_duplicates(subset=[key_x])
    merged = gdf_asset_group.merge(upd, left_on=key_df, right_on=key_x, how='left', suffixes=('', '_xlsx'))
    for col in ['importance','susceptibility']:
        xc = f"{col}_xlsx"
        if xc in merged.columns:
            merged[col] = merged[xc].where(merged[xc].notna(), merged[col])
            merged.drop(columns=[xc], inplace=True, errors='ignore')
    merged = sanitize_vulnerability(merged, valid_input_values, FALLBACK_VULN)
    enforce_vuln_dtypes_inplace(merged)
    gdf_asset_group = ensure_ebsa_columns_with_defaults(merged)

def _to01_series(s):
    s = pd.to_numeric(s, errors='coerce')
    return (s/100.0).clip(0.0, 1.0)

def _apply_ebsa_scores_from_df(df_x: pd.DataFrame):
    global gdf_asset_group
    if df_x is None or df_x.empty: return
    key_df = best_join_key(gdf_asset_group)
    key_x  = 'id' if 'id' in df_x.columns else 'name_original'
    for c in ['ebsa_R','ebsa_U','ebsa_V','ebsa_T','ebsa_F']:
        if c in df_x.columns: df_x[c] = _to01_series(df_x[c])
    keep = [key_x] + [c for c in ['ebsa_R','ebsa_U','ebsa_V','ebsa_T','ebsa_F'] if c in df_x.columns]
    upd = df_x[keep].drop_duplicates(subset=[key_x])
    merged = gdf_asset_group.merge(upd, left_on=key_df, right_on=key_x, how='left', suffixes=('', '_xlsx'))
    for c in ['ebsa_R','ebsa_U','ebsa_V','ebsa_T','ebsa_F']:
        xc = f"{c}_xlsx"
        if xc in merged.columns:
            merged[c] = merged[xc].where(merged[xc].notna(), merged.get(c, np.nan))
            merged.drop(columns=[xc], inplace=True, errors='ignore')
    enforce_ebsa_dtypes_inplace(merged)
    gdf_asset_group = ensure_ebsa_columns_with_defaults(merged)

def load_all_from_excel(excel_path: str):
    try:
        x = pd.read_excel(excel_path, sheet_name=None)
        if 'vulnerability' in x: _apply_vulnerability_from_df(x['vulnerability'])
        if 'ebsa_profile' in x and not x['ebsa_profile'].empty:
            row = x['ebsa_profile'].iloc[0]
            EBSA_PROFILE.update({
                'w_R': float(row.get('w_R', EBSA_PROFILE['w_R'])),
                'w_U': float(row.get('w_U', EBSA_PROFILE['w_U'])),
                'w_V': float(row.get('w_V', EBSA_PROFILE['w_V'])),
                'w_T': float(row.get('w_T', EBSA_PROFILE['w_T'])),
                'w_F': float(row.get('w_F', EBSA_PROFILE['w_F'])),
                'gamma': float(row.get('gamma', EBSA_PROFILE['gamma'])),
                'pnorm': float(row.get('pnorm', EBSA_PROFILE['pnorm'])),
                'thr_primary': float(row.get('thr_primary', EBSA_PROFILE['thr_primary'])),
                'thr_support': float(row.get('thr_support', EBSA_PROFILE['thr_support'])),
            })
            persist_profiles(gpkg_file, original_working_directory)
        if 'ebsa_scores' in x: _apply_ebsa_scores_from_df(x['ebsa_scores'])
        if 'env_index' in x and not x['env_index'].empty:
            row = x['env_index'].iloc[0]
            scoring_val = str(row.get('scoring', ENV_PROFILE['scoring'])).strip().lower()
            if scoring_val not in ('linear','percentile','logistic'):
                scoring_val = 'linear'
            ENV_PROFILE.update({
                'w_sensitivity': float(row.get('w_sensitivity', ENV_PROFILE['w_sensitivity'])),
                'w_susceptibility': float(row.get('w_susceptibility', ENV_PROFILE['w_susceptibility'])),
                'w_importance': float(row.get('w_importance', ENV_PROFILE['w_importance'])),
                'w_pressure': float(row.get('w_pressure', ENV_PROFILE['w_pressure'])),
                'gamma': float(row.get('gamma', ENV_PROFILE['gamma'])),
                'pnorm_minmax': float(row.get('pnorm_minmax', ENV_PROFILE['pnorm_minmax'])),
                'overlap_cap_q': float(row.get('overlap_cap_q', ENV_PROFILE['overlap_cap_q'])),
                'scoring': scoring_val,
                'logistic_a': float(row.get('logistic_a', ENV_PROFILE['logistic_a'])),
                'logistic_b': float(row.get('logistic_b', ENV_PROFILE['logistic_b'])),
            })
            persist_profiles(gpkg_file, original_working_directory)

        refresh_vulnerability_grid_from_df()
        refresh_ebsa_grid_from_df()
        messagebox.showinfo("Loaded", "All settings and values were loaded from Excel.")
    except Exception as e:
        log_to_file(f"Excel load failed: {e}")
        messagebox.showerror("Error", f"Failed reading Excel:\n{e}")

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

def setup_vulnerability_tab(parent, gdf, column_widths):
    canvas, frame = create_scrollable_area(parent)
    entries = []
    setup_headers_vuln(frame, column_widths)
    if gdf is not None and not gdf.empty:
        for i, row in enumerate(gdf.itertuples(), start=1):
            add_vuln_row(i, row, frame, entries, gdf)
        frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))
    else:
        log_to_file("No data to display.")
    return canvas, frame, entries

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
    """Type-safe writeback from UI to DF."""
    for entry in entries_list:
        idx = entry['row_index']
        imp = coerce_valid_int(entry['importance'].get(), valid_input_values, FALLBACK_VULN)
        sus = coerce_valid_int(entry['susceptibility'].get(), valid_input_values, FALLBACK_VULN)
        try:
            sens_label_val = int(float(entry['sensitivity']['text']))
        except Exception:
            sens_label_val = int(imp) * int(sus)
        gdf.at[idx, 'importance'] = int(imp)
        gdf.at[idx, 'susceptibility'] = int(sus)
        gdf.at[idx, 'sensitivity'] = int(sens_label_val)
        gdf.at[idx, 'sensitivity_code'] = str(entry['sensitivity_code']['text'])
        gdf.at[idx, 'sensitivity_description'] = str(entry['sensitivity_description']['text'])

# -------------------------------
# EBSA per-row UI
# -------------------------------
def setup_headers_ebsa(frame):
    headers = ["Dataset", "R (0–100)", "U (0–100)", "V (0–100)", "T (0–100)", "F (0–100)"]
    widths  = [35, 12, 12, 12, 12, 12]
    for idx, header in enumerate(headers):
        ttkb.Label(frame, text=header, anchor='w', width=widths[idx]).grid(row=0, column=idx, padx=5, pady=5, sticky='ew')

def _clamp01_from_ui(text):
    try:
        v = float(text)
        if not np.isfinite(v): return EBSA_PER_ROW_DEFAULT_01
        v = max(0.0, min(100.0, v))
        return v / 100.0
    except Exception:
        return EBSA_PER_ROW_DEFAULT_01

def add_ebsa_row(index, row, frame, entries_list, gdf):
    def _mk_entry(init01):
        e = ttkb.Entry(frame, width=12)
        val = 50.0 if (init01 is None or pd.isna(init01)) else float(init01)*100.0
        e.insert(0, f"{round(val,2)}")
        return e
    eR = _mk_entry(getattr(row, 'ebsa_R', EBSA_PER_ROW_DEFAULT_01))
    eU = _mk_entry(getattr(row, 'ebsa_U', EBSA_PER_ROW_DEFAULT_01))
    eV = _mk_entry(getattr(row, 'ebsa_V', EBSA_PER_ROW_DEFAULT_01))
    eT = _mk_entry(getattr(row, 'ebsa_T', EBSA_PER_ROW_DEFAULT_01))
    eF = _mk_entry(getattr(row, 'ebsa_F', EBSA_PER_ROW_DEFAULT_01))
    eR.grid(row=index, column=1, padx=5); eU.grid(row=index, column=2, padx=5)
    eV.grid(row=index, column=3, padx=5); eT.grid(row=index, column=4, padx=5)
    eF.grid(row=index, column=5, padx=5)
    def _on_edit(_evt=None, idx=index-1):
        try:
            gdf.at[idx, 'ebsa_R'] = float(_clamp01_from_ui(eR.get()))
            gdf.at[idx, 'ebsa_U'] = float(_clamp01_from_ui(eU.get()))
            gdf.at[idx, 'ebsa_V'] = float(_clamp01_from_ui(eV.get()))
            gdf.at[idx, 'ebsa_T'] = float(_clamp01_from_ui(eT.get()))
            gdf.at[idx, 'ebsa_F'] = float(_clamp01_from_ui(eF.get()))
        except Exception as ex:
            log_to_file(f"EBSA input error: {ex}")
    for w in [eR,eU,eV,eT,eF]:
        w.bind('<KeyRelease>', _on_edit)
    l_name = ttkb.Label(frame, text=getattr(row, 'name_original', ''), anchor='w', width=35)
    l_name.grid(row=index, column=0, padx=5, sticky='ew')
    entries_list.append({'row_index': index-1, 'name': l_name, 'R': eR, 'U': eU, 'V': eV, 'T': eT, 'F': eF})

def setup_ebsa_scores_tab(parent, gdf):
    canvas, frame = create_scrollable_area(parent)
    entries_list = []
    setup_headers_ebsa(frame)
    if gdf is not None and not gdf.empty:
        for i, row in enumerate(gdf.itertuples(), start=1):
            add_ebsa_row(i, row, frame, entries_list, gdf)
        frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))
    else:
        log_to_file("No EBSA data to display.")
    return canvas, frame, entries_list

def refresh_ebsa_grid_from_df():
    for entry in entries_ebsa:
        name_original = entry['name'].cget("text")
        mask = (gdf_asset_group['name_original'] == name_original)
        if mask.any():
            idx = gdf_asset_group[mask].index[0]
            for key, col in [('R','ebsa_R'),('U','ebsa_U'),('V','ebsa_V'),('T','ebsa_T'),('F','ebsa_F')]:
                val01 = gdf_asset_group.at[idx, col]
                entry[key].delete(0, tk.END)
                entry[key].insert(0, f"{round((float(val01) if pd.notna(val01) else EBSA_PER_ROW_DEFAULT_01)*100.0,2)}")

# -------------------------------
# EBSA profile UI
# -------------------------------
EBSA_UI = {}
def _parse_float_entry(entry: ttkb.Entry, default: float) -> float:
    try:
        v = float(entry.get().strip()); return v if np.isfinite(v) else default
    except Exception:
        return default

def build_ebsa_profile_section(parent):
    frm = ttkb.Labelframe(parent, text="EBSA profile (global settings)")
    frm.pack(fill='x', expand=False, padx=10, pady=10)
    labels = [('Representativity (R)','w_R'), ('Uniqueness (U)','w_U'),
              ('Vulnerability (V)','w_V'), ('Threatened (T)','w_T'),
              ('Eco functions/services (F)','w_F')]
    for i,(txt,key) in enumerate(labels):
        ttkb.Label(frm, text=txt, width=26, anchor='w').grid(row=i, column=0, sticky='w', padx=6, pady=2)
        e = ttkb.Entry(frm, width=10); e.grid(row=i, column=1, sticky='w'); EBSA_UI[key] = e
    r0 = len(labels)
    ttkb.Label(frm, text="Compensation γ (0=geom, 1=arith)").grid(row=r0, column=0, sticky='w', padx=6)
    EBSA_UI['gamma'] = ttkb.Entry(frm, width=10); EBSA_UI['gamma'].grid(row=r0, column=1, sticky='w', pady=2)
    ttkb.Label(frm, text="Hotspot mixing p (min/max)").grid(row=r0+1, column=0, sticky='w', padx=6)
    EBSA_UI['pnorm'] = ttkb.Entry(frm, width=10); EBSA_UI['pnorm'].grid(row=r0+1, column=1, sticky='w', pady=2)
    ttkb.Label(frm, text="Primary threshold").grid(row=r0+2, column=0, sticky='w', padx=6)
    EBSA_UI['thr_primary'] = ttkb.Entry(frm, width=10); EBSA_UI['thr_primary'].grid(row=r0+2, column=1, sticky='w', pady=2)
    ttkb.Label(frm, text="Support threshold").grid(row=r0+3, column=0, sticky='w', padx=6)
    EBSA_UI['thr_support'] = ttkb.Entry(frm, width=10); EBSA_UI['thr_support'].grid(row=r0+3, column=1, sticky='w', pady=2)

    def normalize_weights():
        s = sum([_parse_float_entry(EBSA_UI[k], 0.0) for k in ['w_R','w_U','w_V','w_T','w_F']])
        if s <= 0:
            messagebox.showwarning("Weights", "Sum of weights is 0 – cannot normalize."); return
        for k in ['w_R','w_U','w_V','w_T','w_F']:
            v = _parse_float_entry(EBSA_UI[k], 0.0) / s
            EBSA_UI[k].delete(0, tk.END); EBSA_UI[k].insert(0, f"{v:.4f}")

    def save_profile():
        EBSA_PROFILE.update({
            'w_R': _parse_float_entry(EBSA_UI['w_R'], DEFAULT_EBSA_PROFILE['w_R']),
            'w_U': _parse_float_entry(EBSA_UI['w_U'], DEFAULT_EBSA_PROFILE['w_U']),
            'w_V': _parse_float_entry(EBSA_UI['w_V'], DEFAULT_EBSA_PROFILE['w_V']),
            'w_T': _parse_float_entry(EBSA_UI['w_T'], DEFAULT_EBSA_PROFILE['w_T']),
            'w_F': _parse_float_entry(EBSA_UI['w_F'], DEFAULT_EBSA_PROFILE['w_F']),
            'gamma': _parse_float_entry(EBSA_UI['gamma'], DEFAULT_EBSA_PROFILE['gamma']),
            'pnorm': _parse_float_entry(EBSA_UI['pnorm'], DEFAULT_EBSA_PROFILE['pnorm']),
            'thr_primary': _parse_float_entry(EBSA_UI['thr_primary'], DEFAULT_EBSA_PROFILE['thr_primary']),
            'thr_support': _parse_float_entry(EBSA_UI['thr_support'], DEFAULT_EBSA_PROFILE['thr_support']),
        })
        persist_profiles(gpkg_file, original_working_directory)
        messagebox.showinfo("EBSA", "EBSA profile saved to GPKG, GeoParquet and JSON.")

    btns = ttkb.Frame(frm); btns.grid(row=r0+4, column=0, columnspan=3, sticky='w', pady=(10,6), padx=6)
    ttkb.Button(btns, text="Normalize weights", command=normalize_weights, bootstyle=INFO).pack(side='left', padx=5)
    ttkb.Button(btns, text="Save profile", command=save_profile, bootstyle=SUCCESS).pack(side='left', padx=5)

def set_ebsa_profile_ui():
    for k, v in EBSA_PROFILE.items():
        if k in EBSA_UI:
            EBSA_UI[k].delete(0, tk.END); EBSA_UI[k].insert(0, str(v))

# -------------------------------
# Visualization profile UI
# -------------------------------
ENV_UI = {}
SCORING_OPTIONS = ('linear', 'percentile', 'logistic')

def build_env_tab(parent):
    frm = ttkb.Frame(parent); frm.pack(fill='both', expand=True, padx=10, pady=10)
    ttkb.Label(frm, text="Visualization index – weights (sum = 1)").grid(row=0, column=0, columnspan=4, sticky='w', pady=(0,6))
    labels = [('Sensitivity','w_sensitivity'), ('Susceptibility','w_susceptibility'),
              ('Importance','w_importance'), ('Pressure','w_pressure')]
    for i,(txt,key) in enumerate(labels, start=1):
        ttkb.Label(frm, text=txt, width=22, anchor='w').grid(row=i, column=0, sticky='w', padx=(0,6), pady=2)
        e = ttkb.Entry(frm, width=10); e.grid(row=i, column=1, sticky='w'); ENV_UI[key] = e
    r0 = len(labels) + 2
    ttkb.Label(frm, text="Compensation γ").grid(row=r0, column=0, sticky='w')
    ENV_UI['gamma'] = ttkb.Entry(frm, width=10); ENV_UI['gamma'].grid(row=r0, column=1, sticky='w', pady=2)
    ttkb.Label(frm, text="Hotspot mixing p (min/max)").grid(row=r0+1, column=0, sticky='w')
    ENV_UI['pnorm_minmax'] = ttkb.Entry(frm, width=10); ENV_UI['pnorm_minmax'].grid(row=r0+1, column=1, sticky='w', pady=2)
    ttkb.Label(frm, text="Overlap cap quantile (0–1)").grid(row=r0+2, column=0, sticky='w')
    ENV_UI['overlap_cap_q'] = ttkb.Entry(frm, width=10); ENV_UI['overlap_cap_q'].grid(row=r0+2, column=1, sticky='w', pady=2)

    ttkb.Label(frm, text="Scoring method").grid(row=r0+3, column=0, sticky='w')
    ENV_UI['scoring'] = ttkb.Combobox(frm, values=list(SCORING_OPTIONS), state="readonly", width=12)
    ENV_UI['scoring'].grid(row=r0+3, column=1, sticky='w', pady=2)

    ttkb.Label(frm, text="Logistic a").grid(row=r0+4, column=0, sticky='w')
    ENV_UI['logistic_a'] = ttkb.Entry(frm, width=10); ENV_UI['logistic_a'].grid(row=r0+4, column=1, sticky='w', pady=2)
    ttkb.Label(frm, text="Logistic b").grid(row=r0+5, column=0, sticky='w')
    ENV_UI['logistic_b'] = ttkb.Entry(frm, width=10); ENV_UI['logistic_b'].grid(row=r0+5, column=1, sticky='w', pady=2)

    def normalize_weights_env():
        s = sum([_parse_float_entry(ENV_UI[k], 0.0) for k in ['w_sensitivity','w_susceptibility','w_importance','w_pressure']])
        if s <= 0:
            messagebox.showwarning("Weights", "Sum of weights is 0 – cannot normalize."); return
        for k in ['w_sensitivity','w_susceptibility','w_importance','w_pressure']:
            v = _parse_float_entry(ENV_UI[k], 0.0) / s
            ENV_UI[k].delete(0, tk.END); ENV_UI[k].insert(0, f"{v:.4f}")

    def save_profile():
        scoring_val = ENV_UI['scoring'].get().strip().lower()
        if scoring_val not in SCORING_OPTIONS:
            scoring_val = 'linear'
        ENV_PROFILE.update({
            'w_sensitivity': _parse_float_entry(ENV_UI['w_sensitivity'], DEFAULT_ENV_PROFILE['w_sensitivity']),
            'w_susceptibility': _parse_float_entry(ENV_UI['w_susceptibility'], DEFAULT_ENV_PROFILE['w_susceptibility']),
            'w_importance': _parse_float_entry(ENV_UI['w_importance'], DEFAULT_ENV_PROFILE['w_importance']),
            'w_pressure': _parse_float_entry(ENV_UI['w_pressure'], DEFAULT_ENV_PROFILE['w_pressure']),
            'gamma': _parse_float_entry(ENV_UI['gamma'], DEFAULT_ENV_PROFILE['gamma']),
            'pnorm_minmax': _parse_float_entry(ENV_UI['pnorm_minmax'], DEFAULT_ENV_PROFILE['pnorm_minmax']),
            'overlap_cap_q': _parse_float_entry(ENV_UI['overlap_cap_q'], DEFAULT_ENV_PROFILE['overlap_cap_q']),
            'scoring': scoring_val,
            'logistic_a': _parse_float_entry(ENV_UI['logistic_a'], DEFAULT_ENV_PROFILE['logistic_a']),
            'logistic_b': _parse_float_entry(ENV_UI['logistic_b'], DEFAULT_ENV_PROFILE['logistic_b']),
        })
        persist_profiles(gpkg_file, original_working_directory)
        messagebox.showinfo("Visualization", "Visualization profile saved to GPKG, GeoParquet and JSON.")

    btns = ttkb.Frame(frm); btns.grid(row=r0+6, column=0, columnspan=3, sticky='w', pady=(10,0))
    ttkb.Button(btns, text="Normalize weights", command=normalize_weights_env, bootstyle=INFO).pack(side='left', padx=5)
    ttkb.Button(btns, text="Save profile", command=save_profile, bootstyle=SUCCESS).pack(side='left', padx=5)

def set_env_profile_ui():
    for k in ('w_sensitivity','w_susceptibility','w_importance','w_pressure','gamma','pnorm_minmax','overlap_cap_q','logistic_a','logistic_b'):
        if k in ENV_UI:
            ENV_UI[k].delete(0, tk.END); ENV_UI[k].insert(0, str(ENV_PROFILE.get(k, DEFAULT_ENV_PROFILE[k])))
    sc = ENV_PROFILE.get('scoring', DEFAULT_ENV_PROFILE['scoring'])
    if sc not in SCORING_OPTIONS:
        sc = 'linear'
    ENV_UI['scoring'].set(sc)

# -------------------------------
# Start tab
# -------------------------------
def build_start_tab(parent):
    frm = ttkb.Frame(parent); frm.pack(fill='both', expand=True, padx=12, pady=12)
    info = (
        "Welcome!\n\n"
        "• Vulnerability: register importance/susceptibility per asset; sensitivity & A–E are computed.\n"
        "• EBSA: global profile (weights, thresholds) + per-row scores (R,U,V,T,F). Defaults are 50 per row.\n"
        "• Visualization: profile for your 1–100 visualization index (choose scoring method from dropdown).\n\n"
        "Profiles are stored in three places: GPKG tables, GeoParquet, and JSON.\n"
        "Use the buttons below for Excel round-trips and to save layers to GPKG/Parquet."
    )
    ttkb.Label(frm, text=info, justify='left', wraplength=900).pack(anchor='w', pady=(0,10))

    btns = ttkb.Frame(frm); btns.pack(anchor='w', pady=6)

    def do_save_all_excel():
        input_folder = os.path.join(original_working_directory, "input")
        os.makedirs(input_folder, exist_ok=True)
        excel_path = filedialog.asksaveasfilename(
            title="Save Excel File", initialdir=input_folder,
            defaultextension=".xlsx", filetypes=[("Excel Files","*.xlsx"), ("All Files","*.*")]
        )
        if excel_path: save_all_to_excel(gdf_asset_group, excel_path)

    def do_load_all_excel():
        input_folder = os.path.join(original_working_directory, "input")
        excel_path = filedialog.askopenfilename(
            title="Select Excel File", initialdir=input_folder,
            filetypes=[("Excel Files","*.xlsx"), ("All Files","*.*")]
        )
        if excel_path: load_all_from_excel(excel_path)

    def do_save_gpkg_parquet():
        # Ensure UI -> DF flush, recompute, then persist to both stores
        update_all_vuln_rows(entries_vuln, gdf_asset_group)
        enforce_vuln_dtypes_inplace(gdf_asset_group)
        enforce_ebsa_dtypes_inplace(gdf_asset_group)
        # Recompute/validate right before saving (belt & braces)
        gdf_ready = sanitize_vulnerability(gdf_asset_group, valid_input_values, FALLBACK_VULN)

        save_asset_group_to_gpkg(gdf_ready, gpkg_file)
        save_asset_group_to_parquet(gdf_ready, original_working_directory)
        persist_profiles(gpkg_file, original_working_directory)
        messagebox.showinfo("Saved", "Saved asset-group layer + profiles to GPKG & GeoParquet (and JSON).")

    ttkb.Button(btns, text="Save all to Excel", command=do_save_all_excel, bootstyle=SUCCESS).pack(side='left', padx=6)
    ttkb.Button(btns, text="Load all from Excel", command=do_load_all_excel, bootstyle=INFO).pack(side='left', padx=6)
    ttkb.Button(btns, text="Save to GPKG/Parquet", command=do_save_gpkg_parquet, bootstyle=PRIMARY).pack(side='left', padx=6)

    ttkb.Button(frm, text="Exit", command=close_application, bootstyle=WARNING).pack(anchor='e', pady=(20,0))

# -------------------------------
# Misc helpers
# -------------------------------
def close_application():
    try:
        update_all_vuln_rows(entries_vuln, gdf_asset_group)
        enforce_vuln_dtypes_inplace(gdf_asset_group)
        enforce_ebsa_dtypes_inplace(gdf_asset_group)
        gdf_ready = sanitize_vulnerability(gdf_asset_group, valid_input_values, FALLBACK_VULN)
        save_asset_group_to_gpkg(gdf_ready, gpkg_file)
        save_asset_group_to_parquet(gdf_ready, original_working_directory)
        persist_profiles(gpkg_file, original_working_directory)
    finally:
        root.destroy()

# -------------------------------
# Entrypoint
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MESA – Setup & Registration')
    parser.add_argument('--original_working_directory', required=False, help='Path to running folder')
    args = parser.parse_args()
    original_working_directory = args.original_working_directory

    if not original_working_directory:
        original_working_directory = os.getcwd()
        if "system" in os.path.basename(original_working_directory).lower():
            original_working_directory = os.path.abspath(os.path.join(original_working_directory, os.pardir))

    config_file = os.path.join(original_working_directory, "system", "config.ini")
    gpkg_file   = os.path.join(original_working_directory, "output", "mesa.gpkg")

    config = read_config(config_file)
    classification = read_config_classification(config_file)
    valid_input_values = get_valid_values(config)
    FALLBACK_VULN = get_fallback_value(config, valid_input_values)
    ttk_bootstrap_theme = config['DEFAULT'].get('ttk_bootstrap_theme', 'flatly')
    workingprojection_epsg = config['DEFAULT'].get('workingprojection_epsg', '4326')

    # Load profiles (GPKG -> Parquet -> JSON -> defaults), then mirror to all
    load_profiles(gpkg_file, original_working_directory)
    # Ensure EBSA global UI fields are pre-populated with defaults on first run
    _ensure_ebsa_profile_defaults()

    # Load asset groups and ensure EBSA row defaults are present
    gdf_asset_group = load_asset_group(gpkg_file)
    if gdf_asset_group is None:
        log_to_file("Failed to load tbl_asset_group (GPKG/Parquet). Check data integrity.")
        sys.exit(1)

    # UI
    root = tb.Window(themename=ttk_bootstrap_theme)
    root.title("MESA – Setup & Registration")
    try:
        icon_path = os.path.join(original_working_directory, "system_resources", "mesa.ico")
        if os.path.exists(icon_path): root.iconbitmap(icon_path)
    except Exception:
        pass
    root.geometry("1100x860")

    nb = ttkb.Notebook(root, bootstyle=PRIMARY); nb.pack(fill='both', expand=True)

    # Start
    tab_start = ttkb.Frame(nb); nb.add(tab_start, text="Start")
    build_start_tab(tab_start)

    # Vulnerability
    tab_vuln = ttkb.Frame(nb); nb.add(tab_vuln, text="Vulnerability")
    canvas_v, frame_v = create_scrollable_area(tab_vuln)
    entries_vuln = []
    setup_headers_vuln(frame_v, column_widths)
    for i, row in enumerate(gdf_asset_group.itertuples(), start=1):
        add_vuln_row(i, row, frame_v, entries_vuln, gdf_asset_group)
    frame_v.update_idletasks(); canvas_v.configure(scrollregion=canvas_v.bbox("all"))

    # EBSA (profile + per-row)
    tab_ebsa = ttkb.Frame(nb); nb.add(tab_ebsa, text="EBSA")
    # Profile
    build_ebsa_profile_section(tab_ebsa)
    # Populate EBSA global settings UI from defaults/profile
    set_ebsa_profile_ui()
    ttkb.Label(tab_ebsa, text="EBSA scores per asset (0–100 in UI; stored 0–1). Default 50.").pack(anchor='w', padx=10, pady=(4,0))
    # Per-row
    canvas_e, frame_e = create_scrollable_area(tab_ebsa)
    entries_ebsa = []
    setup_headers_ebsa(frame_e)
    for i, row in enumerate(gdf_asset_group.itertuples(), start=1):
        add_ebsa_row(i, row, frame_e, entries_ebsa, gdf_asset_group)
    frame_e.update_idletasks(); canvas_e.configure(scrollregion=canvas_e.bbox("all"))

    # Visualization
    tab_env = ttkb.Frame(nb); nb.add(tab_env, text="Visualization")
    build_env_tab(tab_env); set_env_profile_ui()

    root.mainloop()
