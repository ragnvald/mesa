# -*- coding: utf-8 -*-
from __future__ import annotations
# MESA – Processing setup (2 tabs: Sensitivity and Index weights)
# Persistence: GeoParquet + JSON only (GPKG removed)
# PySide6 UI (migrated from ttkbootstrap).

import os
import sys
import argparse
import configparser
import datetime
import statistics
from pathlib import Path
from typing import Optional

np = None
pd = None
gpd = None
_shapely_wkb = None
_shapely_wkt = None


def _load_runtime_data_stack() -> None:
    global np, pd, gpd, _shapely_wkb, _shapely_wkt
    if all(module is not None for module in (np, pd, gpd, _shapely_wkb, _shapely_wkt)):
        return
    try:
        import numpy as _np
        import pandas as _pd
        import geopandas as _gpd
        from shapely import wkb as _wkb
        from shapely import wkt as _wkt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "processing_setup.py requires numpy, pandas, geopandas, and shapely in the runtime environment."
        ) from exc
    np = _np
    pd = _pd
    gpd = _gpd
    _shapely_wkb = _wkb
    _shapely_wkt = _wkt

# Capture start-CWD before anything changes
START_CWD = Path.cwd()

# exe dir if frozen, else script dir
APP_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent

# ---- constant *relative* paths we join to a discovered base dir ----
PARQUET_ASSET_GROUP  = os.path.join("output", "geoparquet", "tbl_asset_group.parquet")
PARQUET_ASSET_OBJECT = os.path.join("output", "geoparquet", "tbl_asset_object.parquet")
ASSET_GROUP_OVERRIDE: Optional[Path] = None

# UI grid helpers
column_widths = [35, 13, 13, 13, 13, 30]
valid_input_values: list[int] = []
classification: dict = {}
entries_vuln = []
FALLBACK_VULN = 3
gdf_asset_group = None  # set by run(); referenced by module-level functions
root = None             # set by run(); referenced by close_application()
INDEX_WEIGHT_DEFAULTS = {
    "importance": [1, 2, 5, 5, 10],
    # Sensitivity is derived as importance * susceptibility (both 1..5).
    # Only these products can occur.
    "sensitivity": [
        10, # 1
        10, # 2
        10, # 3
        10, # 4
        10, # 5
        10, # 6
        10, # 8
        10, # 9
        10, # 10
        10, # 12
        10, # 15
        10, # 16
        10, # 20
        10, # 25
    ],
}
INDEX_WEIGHT_KEYS = {
    "importance": "index_importance_weights",
    "sensitivity": "index_sensitivity_weights",
}
index_weight_settings: dict[str, list[int]] = {}
index_weight_vars: dict[str, list] = {}  # will hold QLineEdit references
status_message_var = None  # will be a QLabel instance

SENSITIVITY_PRODUCT_VALUES: list[int] = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 25]

# paths set in __main__
original_working_directory = ""
config_file = ""
workingprojection_epsg = "4326"

# PySide6 imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QPushButton, QLineEdit,
    QScrollArea, QFrame, QSizePolicy, QMessageBox, QFileDialog,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView,
)
from PySide6.QtGui import QFont, QIcon, QPainter, QPen, QBrush, QColor
from PySide6.QtCore import Qt, Signal, QObject

import webbrowser

from asset_manage import apply_shared_stylesheet


WIKI_INDEX_URLS = {
    "importance": "https://github.com/ragnvald/mesa/wiki/Indexes#importance-index-index_importance",
    "sensitivity": "https://github.com/ragnvald/mesa/wiki/Indexes#sensitivity-index-index_sensitivity",
    "owa":         "https://github.com/ragnvald/mesa/wiki/Indexes#owa-index-index_owa",
}


class _InfoCircleLabel(QLabel):
    """Small painted circle with 'i' that opens a URL on click.

    Mirrors the same widget used in mesa.py so the Parameters dialog has the
    visual vocabulary as the main app's Status pane.
    """

    def __init__(self, url: str, parent=None):
        super().__init__(parent)
        self._url = url
        self.setFixedSize(20, 20)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Open detailed description in browser")

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor("#9b7c3d"), 1.5))
        p.setBrush(QBrush(QColor("#faf6ee")))
        p.drawEllipse(2, 2, 15, 15)
        p.setPen(QColor("#715a36"))
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(self.rect(), Qt.AlignCenter, "i")
        p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._url:
            webbrowser.open(self._url)

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
    for root_dir in (START_CWD, APP_DIR):
        for up in [root_dir.parent, root_dir.parent.parent, root_dir.parent.parent.parent]:
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
# Config (theme/CRS + A-E bins)
# -------------------------------
def read_config(file_name: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
    cfg.read(file_name, encoding="utf-8")
    return cfg

def read_config_classification(file_name: str) -> dict:
    """Read A-E bins & descriptions from config.ini."""
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
    return int(statistics.median(valid_vals))

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


def _set_status_message(message: str) -> None:
    """Update the inline status label (if available) and log the message."""
    text = (message or "").strip()
    if not text:
        return
    log_to_file(text)
    try:
        if status_message_var is not None:
            status_message_var.setText(text)
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

def _is_blankish(v) -> bool:
    try:
        s = ("" if v is None else str(v)).strip().lower()
    except Exception:
        return True
    return (not s) or (s in {"nil", "none", "null", "nan"}) or (not any(ch.isdigit() for ch in s))

def _cfg_get_any(cfg: configparser.ConfigParser, option: str, fallback: str = "") -> str:
    """Get a config value regardless of section.

    Prefer [DEFAULT], then [VALID_VALUES], then any other section.
    Treat blank/"nil" values as missing so we can fall back.
    """
    for section in ("DEFAULT", "VALID_VALUES"):
        try:
            if cfg.has_option(section, option):
                v = cfg.get(section, option, fallback="")
                if not _is_blankish(v):
                    return v
        except Exception:
            pass
    for section in cfg.sections():
        if section in {"DEFAULT", "VALID_VALUES"}:
            continue
        try:
            if cfg.has_option(section, option):
                v = cfg.get(section, option, fallback="")
                if not _is_blankish(v):
                    return v
        except Exception:
            pass
    return fallback

def _parse_weight_line(text: str, default: list[int]) -> list[int]:
    try:
        if not text:
            return default.copy()
        raw = [int(x.strip()) for x in str(text).replace(";", ",").split(",") if x.strip()]
        want = len(default)
        cleaned = [max(1, v) for v in raw[:want]]
        while len(cleaned) < want:
            cleaned.append(default[len(cleaned)])
        return cleaned
    except Exception:
        return default.copy()

def load_index_weight_settings(cfg: configparser.ConfigParser) -> dict[str, list[int]]:
    settings: dict[str, list[int]] = {}
    for key, option in INDEX_WEIGHT_KEYS.items():
        default = INDEX_WEIGHT_DEFAULTS[key]
        raw = _cfg_get_any(cfg, option, fallback="")
        settings[key] = _parse_weight_line(raw, default)
    return settings

def persist_index_weight_settings(cfg_path: str, settings: dict[str, list[int]]) -> None:
    payload = {}
    for key, weights in settings.items():
        expected = len(INDEX_WEIGHT_DEFAULTS.get(key, [])) or 5
        safe = [max(1, int(w)) for w in weights[:expected]]
        while len(safe) < expected:
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

def enforce_vuln_dtypes_inplace(df) -> None:
    for col in ('importance', 'susceptibility', 'sensitivity'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    for c in ('sensitivity_code', 'sensitivity_description'):
        if c in df.columns:
            df[c] = df[c].astype('string')

def sanitize_vulnerability(df,
                           valid_vals: list[int],
                           fallback: int):
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

def _empty_asset_group_frame():
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


def _candidate_asset_object_paths(base_dir: str) -> list[Path]:
    base = Path(base_dir).resolve()
    primary = (base / PARQUET_ASSET_OBJECT).resolve()
    candidates = [primary]
    if base.name.lower() != "code":
        candidates.append((base / "code" / PARQUET_ASSET_OBJECT).resolve())
    return candidates


def _build_asset_groups_from_asset_object(base_dir: str):
    """Best-effort reconstruction of tbl_asset_group when missing.

    If tbl_asset_object exists, aggregate by ref_asset_group and derive:
    - id
    - name_gis_assetgroup
    - name_original (fallback to name_gis_assetgroup)
    - total_asset_objects
    Then seed default vulnerability values.
    """
    target: Optional[Path] = None
    for cand in _candidate_asset_object_paths(base_dir):
        if cand.exists():
            target = cand
            break
    if target is None:
        return _empty_asset_group_frame()

    try:
        a = gpd.read_parquet(target)
    except Exception:
        try:
            a = pd.read_parquet(target)
        except Exception as e:
            log_to_file(f"Failed reading tbl_asset_object for asset-group reconstruction: {e}")
            return _empty_asset_group_frame()

    if a is None or len(a) == 0:
        return _empty_asset_group_frame()

    if "ref_asset_group" not in a.columns:
        return _empty_asset_group_frame()

    ref = pd.to_numeric(a["ref_asset_group"], errors="coerce")
    if ref.notna().sum() == 0:
        return _empty_asset_group_frame()

    # Prefer a readable group name when present.
    name_col = None
    for c in ("name_gis_assetgroup", "name_original", "asset_group", "group", "layer"):
        if c in a.columns:
            name_col = c
            break

    tmp = pd.DataFrame({"ref_asset_group": ref})
    if name_col is not None:
        tmp["name_hint"] = a[name_col].astype("string")
    else:
        tmp["name_hint"] = pd.NA

    grouped = tmp.groupby("ref_asset_group", dropna=True).agg(
        total_asset_objects=("ref_asset_group", "size"),
        name_hint=("name_hint", lambda x: next((v for v in x.dropna().astype(str).tolist() if v.strip()), "")),
    ).reset_index()

    grouped.rename(columns={"ref_asset_group": "id"}, inplace=True)
    grouped["id"] = pd.to_numeric(grouped["id"], errors="coerce").astype("Int64")
    grouped = grouped[grouped["id"].notna()].copy()
    grouped["id"] = grouped["id"].astype(int)

    # Stable GIS name.
    grouped["name_gis_assetgroup"] = grouped["id"].apply(lambda i: f"layer_{int(i):03d}")
    grouped["name_original"] = grouped["name_hint"].where(grouped["name_hint"].astype(str).str.len() > 0, grouped["name_gis_assetgroup"])
    grouped["title_fromuser"] = ""

    # Geometry is optional for this helper; keep it nullable.
    grouped["geometry"] = None

    gdf = gpd.GeoDataFrame(grouped[
        [
            "id",
            "name_original",
            "name_gis_assetgroup",
            "title_fromuser",
            "total_asset_objects",
            "geometry",
        ]
    ], geometry="geometry", crs=f"EPSG:{workingprojection_epsg}")

    gdf = sanitize_vulnerability(gdf, valid_input_values, FALLBACK_VULN)
    enforce_vuln_dtypes_inplace(gdf)
    return gdf


def _coerce_geometry_series(raw):
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


def _load_asset_group_without_geo_metadata(target: Path):
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

def load_asset_group(base_dir: str):
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
            seeded = _build_asset_groups_from_asset_object(base_dir)
            if seeded is None or seeded.empty:
                seeded = _empty_asset_group_frame()
                # Ensure schema exists on disk even when no rows are available.
                seeded = sanitize_vulnerability(seeded, valid_input_values, FALLBACK_VULN)
                enforce_vuln_dtypes_inplace(seeded)
            seeded.to_parquet(target, index=False)
            log_to_file(f"Initialized tbl_asset_group at {target} (rows={len(seeded)})")
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
        # If the asset-group table exists but is empty, try seeding from tbl_asset_object.
        try:
            seeded = _build_asset_groups_from_asset_object(base_dir)
            if seeded is not None and not seeded.empty:
                try:
                    seeded.to_parquet(target, index=False)
                    log_to_file(f"Seeded tbl_asset_group from tbl_asset_object (rows={len(seeded)})")
                except Exception as e:
                    log_to_file(f"Failed writing seeded tbl_asset_group: {e}")
                return seeded
        except Exception:
            pass
        return _empty_asset_group_frame()

    if gdf.crs is None:
        try:
            gdf.set_crs(epsg=int(workingprojection_epsg), inplace=True)
        except Exception:
            gdf.set_crs(f"EPSG:{workingprojection_epsg}", inplace=True, allow_override=True)
    gdf = sanitize_vulnerability(gdf, valid_input_values, FALLBACK_VULN)
    enforce_vuln_dtypes_inplace(gdf)

    # Ensure defaults are persisted even if the user never clicks "Save".
    try:
        gdf.to_parquet(target, index=False)
    except Exception as e:
        log_to_file(f"Failed to persist sanitized tbl_asset_group defaults: {e}")
    return gdf

def save_asset_group_to_parquet(gdf, base_dir: str):
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
def save_all_to_excel(gdf, excel_path: str):
    try:
        vuln_cols = ['id','name_original','susceptibility','importance','sensitivity','sensitivity_code','sensitivity_description']
        vcols = [c for c in vuln_cols if c in gdf.columns]
        vuln = gdf[vcols].copy() if vcols else pd.DataFrame(columns=vuln_cols)
        with pd.ExcelWriter(excel_path, engine='openpyxl') as xw:
            vuln.to_excel(xw, sheet_name='vulnerability', index=False)
        _set_status_message(f"Saved Excel workbook: {excel_path}")
    except Exception as e:
        log_to_file(f"Excel save failed: {e}")
        QMessageBox.critical(None, "Error", f"Failed saving to Excel:\n{e}")

def _apply_vulnerability_from_df(df_x):
    global gdf_asset_group
    if df_x is None or df_x.empty: return

    # Tolerant column name matching: only `importance` and `susceptibility`
    # are loaded from Excel - everything else (sensitivity, code, description)
    # is recalculated by sanitize_vulnerability below. We accept the canonical
    # names plus simple case/whitespace variants so a hand-edited workbook
    # with "Importance " or "Susceptibility" still resolves.
    df_x = df_x.copy()
    canonical_targets = {
        'id': 'id',
        'name_original': 'name_original',
        'importance': 'importance',
        'susceptibility': 'susceptibility',
    }
    rename_map = {}
    for orig in list(df_x.columns):
        norm = str(orig).strip().lower().replace(' ', '_')
        if norm in canonical_targets and norm != orig:
            rename_map[orig] = canonical_targets[norm]
    if rename_map:
        df_x.rename(columns=rename_map, inplace=True)

    # Drop everything outside the canonical four; the rest is recomputed.
    keep_cols = [c for c in canonical_targets.values() if c in df_x.columns]
    if not any(c in keep_cols for c in ('importance', 'susceptibility')):
        log_to_file("Excel load: neither 'importance' nor 'susceptibility' column found; skipping.")
        return
    df_x = df_x[keep_cols]

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
        # Tolerant sheet-name lookup: accept the canonical name or simple
        # case variants ("Vulnerability", "vulnerability ", etc.). If only
        # one sheet exists, just use it.
        sheet_df = None
        norm_to_orig = {str(k).strip().lower(): k for k in x.keys()}
        if 'vulnerability' in norm_to_orig:
            sheet_df = x[norm_to_orig['vulnerability']]
        elif len(x) == 1:
            sheet_df = next(iter(x.values()))
        if sheet_df is not None:
            _apply_vulnerability_from_df(sheet_df)
        refresh_vulnerability_grid_from_df()
        # Persist to parquet immediately. Without this, the pipeline run
        # after an Excel load reads stale susceptibility/importance from
        # disk, which on a 35-row workbook produced an all-E result for one
        # user because the load only updated in-memory state.
        try:
            save_asset_group_to_parquet(gdf_asset_group, original_working_directory)
        except Exception as exc:
            log_to_file(f"Excel load: failed to persist asset group parquet: {exc}")
        _set_status_message(f"Loaded Excel workbook: {excel_path}")
    except Exception as e:
        log_to_file(f"Excel load failed: {e}")
        QMessageBox.critical(None, "Error", f"Failed reading Excel:\n{e}")


# -------------------------------
# Vulnerability UI helpers
# -------------------------------
def calculate_sensitivity(entry_importance, entry_susceptibility, index, entries_list, gdf):
    try:
        imp = coerce_valid_int(entry_importance.text().strip(), valid_input_values, FALLBACK_VULN)
        sus = coerce_valid_int(entry_susceptibility.text().strip(), valid_input_values, FALLBACK_VULN)
        sensitivity = int(imp) * int(sus)
        code, desc = determine_category(max(1, sensitivity))
        entries_list[index]['sensitivity'].setText(str(int(sensitivity)))
        entries_list[index]['sensitivity_code'].setText(str(code))
        entries_list[index]['sensitivity_description'].setText(str(desc))
        gdf.at[index, 'importance'] = int(imp)
        gdf.at[index, 'susceptibility'] = int(sus)
        gdf.at[index, 'sensitivity'] = int(sensitivity)
        gdf.at[index, 'sensitivity_code'] = str(code)
        gdf.at[index, 'sensitivity_description'] = str(desc)
    except Exception as e:
        log_to_file(f"Input Error: {e}")

def refresh_vulnerability_grid_from_df():
    # Block the table's itemChanged signal while we rewrite all six cells per
    # row. Without this guard, setting importance first triggers the live
    # recalc handler, which reads the still-old susceptibility and writes
    # both back into gdf - clobbering the freshly loaded Excel values
    # because sus hasn't been refreshed yet on the same row.
    table = None
    if entries_vuln:
        try:
            table = entries_vuln[0]['name'].tableWidget()
        except Exception:
            table = None
    prev_blocked = False
    if table is not None:
        prev_blocked = table.signalsBlocked()
        table.blockSignals(True)
    try:
        for entry in entries_vuln:
            name_original = entry['name'].text()
            mask = (gdf_asset_group['name_original'] == name_original)
            if mask.any():
                idx = gdf_asset_group[mask].index[0]
                imp_val = int(gdf_asset_group.at[idx, 'importance'])
                sus_val = int(gdf_asset_group.at[idx, 'susceptibility'])
                sens_val = int(gdf_asset_group.at[idx, 'sensitivity'])
                entry['importance'].setText(str(imp_val))
                entry['susceptibility'].setText(str(sus_val))
                entry['sensitivity'].setText(str(sens_val))
                entry['sensitivity_code'].setText(str(gdf_asset_group.at[idx, 'sensitivity_code']))
                entry['sensitivity_description'].setText(str(gdf_asset_group.at[idx, 'sensitivity_description']))
                # Refresh numeric sort keys on the items we just rewrote so
                # that clicking a numeric column header sorts by value, not
                # by the lexicographic display string.
                try:
                    entry['importance'].setData(Qt.UserRole, imp_val)
                    entry['susceptibility'].setData(Qt.UserRole, sus_val)
                    entry['sensitivity'].setData(Qt.UserRole, sens_val)
                except Exception:
                    pass
    finally:
        if table is not None:
            table.blockSignals(prev_blocked)

def update_all_vuln_rows(entries_list, gdf):
    """Sync all UI entry values back into dataframe before saving."""
    name_map = {row['name'].text(): row for row in entries_list}
    for idx, row in gdf.iterrows():
        name = row.get('name_original', '')
        ui = name_map.get(name)
        if not ui:
            continue
        imp = coerce_valid_int(ui['importance'].text().strip(), valid_input_values, FALLBACK_VULN)
        sus = coerce_valid_int(ui['susceptibility'].text().strip(), valid_input_values, FALLBACK_VULN)
        sens = imp * sus
        code, desc = determine_category(sens)
        gdf.at[idx, 'importance'] = imp
        gdf.at[idx, 'susceptibility'] = sus
        gdf.at[idx, 'sensitivity'] = sens
        gdf.at[idx, 'sensitivity_code'] = code
        gdf.at[idx, 'sensitivity_description'] = desc


def _collect_index_weight_values(strict: bool = True) -> Optional[dict[str, list[int]]]:
    """Read the current entry widgets and coerce them to positive integers."""
    if not index_weight_vars:
        return {}
    collected: dict[str, list[int]] = {}
    for key, widgets_list in index_weight_vars.items():
        defaults = INDEX_WEIGHT_DEFAULTS.get(key, INDEX_WEIGHT_DEFAULTS['importance'])
        labels = list(range(1, 6)) if key == "importance" else SENSITIVITY_PRODUCT_VALUES
        values: list[int] = []
        for idx, widget in enumerate(widgets_list, start=1):
            txt = (widget.text() or "").strip()
            if txt.isdigit():
                val = int(txt)
            else:
                if strict:
                    lab = labels[idx - 1] if 0 <= (idx - 1) < len(labels) else idx
                    QMessageBox.critical(None, "Indices", f"Weight for value {lab} in {key} must be a positive integer.")
                    return None
                try:
                    val = int(float(txt))
                except Exception:
                    fallback_idx = idx - 1
                    val = defaults[fallback_idx] if fallback_idx < len(defaults) else 1
            if val < 1:
                if strict:
                    lab = labels[idx - 1] if 0 <= (idx - 1) < len(labels) else idx
                    QMessageBox.critical(None, "Indices", f"Weight for value {lab} in {key} must be at least 1.")
                    return None
                val = 1
            values.append(val)
        collected[key] = values
    return collected


def persist_index_weights_from_ui(strict: bool = True, silent: bool = False) -> bool:
    updated = _collect_index_weight_values(strict=strict)
    if updated is None:
        return False
    if not updated:
        return True
    try:
        persist_index_weight_settings(config_file, updated)
        index_weight_settings.update(updated)
        if not silent:
            _set_status_message("Index weights saved to config.ini")
        return True
    except Exception as err:
        log_to_file(f"Failed to persist index weights: {err}")
        if strict and not silent:
            QMessageBox.critical(None, "Indexes", f"Could not save index weights:\n{err}")
        elif not silent:
            _set_status_message("Index weights could not be saved (see log).")
        return False


# =====================================================================
# PySide6 Window
# =====================================================================
class _NumericTableItem(QTableWidgetItem):
    # Sort numerically by the value stored in Qt.UserRole so "10" is larger
    # than "2" (default QTableWidgetItem sort is lexicographic on the text).
    def __lt__(self, other):
        try:
            a = self.data(Qt.UserRole)
            b = other.data(Qt.UserRole)
            if a is not None and b is not None:
                return float(a) < float(b)
        except Exception:
            pass
        return super().__lt__(other)


class SetupWindow(QMainWindow):

    def __init__(self, base_dir: str):
        super().__init__()
        self._base_dir = base_dir
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        self.setWindowTitle("MESA - Processing setup")
        self.resize(960, 620)
        self.setMinimumSize(740, 440)

        try:
            icon_path = resource_path(Path("system_resources") / "mesa.ico")
            if icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))
        except Exception:
            pass

        central = QWidget()
        central.setObjectName("CentralHost")
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # --- Tabs: Sensitivity + Indexes ---
        self._tabs = QTabWidget()
        main_layout.addWidget(self._tabs, stretch=1)

        # Tab 1 – Sensitivity
        self._view_sensitivity = QWidget()
        self._build_sensitivity_view(self._view_sensitivity)
        self._tabs.addTab(self._view_sensitivity, "Sensitivity")

        # Tab 2 – Indexes
        self._view_indexes = QWidget()
        self._build_indexes_view(self._view_indexes)
        self._tabs.addTab(self._view_indexes, "Index weights")

        # --- Right corner: data-management buttons + Exit ---
        _corner_css = """
            QPushButton {
                background: #eadfc8; border: 1px solid #b79f73;
                border-radius: 4px; color: #453621; font-size: 8pt;
                padding: 2px 8px; margin: 1px 2px; min-width: 0;
            }
            QPushButton:hover { background: #e1d1ae; }
            QPushButton:pressed { background: #d4c094; }
        """
        corner = QWidget()
        corner_lay = QHBoxLayout(corner)
        corner_lay.setContentsMargins(0, 0, 0, 0)
        corner_lay.setSpacing(2)

        btn_save_excel = QPushButton("Save to Excel")
        btn_save_excel.setStyleSheet(_corner_css)
        btn_save_excel.clicked.connect(self._do_save_all_excel)
        corner_lay.addWidget(btn_save_excel)

        btn_load_excel = QPushButton("Load from Excel")
        btn_load_excel.setStyleSheet(_corner_css)
        btn_load_excel.clicked.connect(self._do_load_all_excel)
        corner_lay.addWidget(btn_load_excel)

        btn_save_db = QPushButton("Save")
        btn_save_db.setStyleSheet(_corner_css.replace("#eadfc8", "#d9bd7d")
                                              .replace("#b79f73", "#9b7c3d")
                                              .replace("#e1d1ae", "#e1c78d")
                                              .replace("#d4c094", "#cfb06f"))
        btn_save_db.clicked.connect(self._do_save_database)
        corner_lay.addWidget(btn_save_db)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFixedHeight(18)
        sep.setStyleSheet("color: #cbb791;")
        corner_lay.addWidget(sep)

        exit_btn = QPushButton("Exit")
        exit_btn.setStyleSheet(_corner_css)
        exit_btn.clicked.connect(self._close_application)
        corner_lay.addWidget(exit_btn)

        self._tabs.setCornerWidget(corner, Qt.TopRightCorner)

        # --- Status label at bottom ---
        global status_message_var
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("color: #9a8a6e; font-size: 9pt;")
        status_message_var = self._status_label
        main_layout.addWidget(self._status_label)

    # ------------------------------------------------------------------
    def _build_indexes_view(self, parent: QWidget):
        global index_weight_vars
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        info = (
            "Index weights control how importance and sensitivity values are aggregated "
            "when multiple asset layers overlap within the same mosaic cell.\n\n"
            "Importance index — Each asset layer has an importance value from 1 (low) to 5 (high). "
            "The weight assigned here determines how much each importance level contributes to the "
            "aggregated importance score. A higher weight increases the influence of that level.\n\n"
            "Sensitivity index — Sensitivity is the product of importance × susceptibility "
            "(resulting in values like 1, 2, 3, …, 25). The weight assigned to each product value "
            "determines how much an overlap at that sensitivity level contributes to the aggregated "
            "score. Defaults are flat because the product values already encode magnitude; raise "
            "individual weights to over-emphasise particular sensitivity levels."
        )
        info_label = QLabel(info)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #5c4a2f; font-size: 9pt; padding: 4px 0;")
        layout.addWidget(info_label)

        sections = [
            ("importance", "Importance index weights (1-5)", list(range(1, 6))),
            ("sensitivity", "Sensitivity index weights (products 1-25)", SENSITIVITY_PRODUCT_VALUES),
        ]
        weight_widgets: dict[str, list] = {}

        for key, title, value_labels in sections:
            box = QGroupBox(title)
            box_outer = QHBoxLayout(box)
            box_outer.setContentsMargins(8, 6, 8, 8)
            box_outer.setSpacing(8)

            # Left-column info icon linking to the wiki section for this index.
            # Placed in a column rather than a top row so it doesn't add height
            # and squeeze the weight entries vertically.
            icon_col = QVBoxLayout()
            icon_col.setContentsMargins(0, 0, 0, 0)
            icon_col.setSpacing(0)
            icon_col.addWidget(_InfoCircleLabel(WIKI_INDEX_URLS[key]), alignment=Qt.AlignTop)
            icon_col.addStretch(1)
            box_outer.addLayout(icon_col)

            grid_holder = QWidget()
            grid = QGridLayout(grid_holder)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(4)
            grid.setVerticalSpacing(8)

            widgets_for_key: list = []
            current = index_weight_settings.get(key, INDEX_WEIGHT_DEFAULTS[key])

            # Lay out in 7 columns per line to keep the dialog readable.
            per_line = 7

            # Row 0 label headers
            lbl_value = QLabel("Value")
            lbl_value.setFixedWidth(60)
            grid.addWidget(lbl_value, 0, 0)
            lbl_weight = QLabel("Weight")
            lbl_weight.setFixedWidth(60)
            grid.addWidget(lbl_weight, 1, 0)

            for idx_v, v in enumerate(value_labels):
                block = (idx_v // per_line)
                col = (idx_v % per_line) + 1
                r0 = block * 2
                val_label = QLabel(str(v))
                val_label.setAlignment(Qt.AlignCenter)
                val_label.setFixedWidth(50)
                val_label.setMinimumHeight(22)
                grid.addWidget(val_label, r0, col)

                entry = QLineEdit(str(current[idx_v] if idx_v < len(current) else 1))
                entry.setFixedWidth(50)
                entry.setMinimumHeight(26)
                entry.setAlignment(Qt.AlignCenter)
                grid.addWidget(entry, r0 + 1, col)
                widgets_for_key.append(entry)

            # Add a visible gutter between block rows when the values wrap to a
            # second line — otherwise the previous block's entries butt up
            # against the next block's value labels.
            n_blocks = (len(value_labels) + per_line - 1) // per_line
            for block_idx in range(1, n_blocks):
                gutter_row = block_idx * 2
                grid.setRowMinimumHeight(gutter_row, 30)

            box_outer.addWidget(grid_holder, stretch=1)
            weight_widgets[key] = widgets_for_key
            layout.addWidget(box)

        tuning_note = QLabel(
            "<b>Tuning tips.</b> Weights act as multipliers, so only their <i>proportions</i> "
            "matter for the 0&ndash;100 ranking. Common patterns: set a weight to <b>0</b> to "
            "filter that level out entirely (e.g. importance class 1 = 0 hides low-value "
            "overlaps when screening for hotspots); raise a single weight to <b>emphasise</b> "
            "that level (e.g. boost sensitivity product 25 so cells with extreme overlaps rise "
            "to the top); keep the row <b>flat</b> when the input class numbers already encode "
            "magnitude (this is why sensitivity defaults flat — a product of 25 is already 25&times; "
            "a product of 1). Changes only affect future processing runs, so you can experiment "
            "freely and re-run to compare."
        )
        tuning_note.setWordWrap(True)
        tuning_note.setTextFormat(Qt.RichText)
        tuning_note.setStyleSheet(
            "color: #5c4a2f; font-size: 9pt; padding: 8px 10px; "
            "background-color: #f6efe1; border: 1px solid #d8c9a4; border-radius: 4px;"
        )
        layout.addWidget(tuning_note)

        owa_row = QHBoxLayout()
        owa_row.setContentsMargins(0, 0, 0, 0)
        owa_row.setSpacing(8)

        owa_icon_col = QVBoxLayout()
        owa_icon_col.setContentsMargins(0, 0, 0, 0)
        owa_icon_col.addWidget(_InfoCircleLabel(WIKI_INDEX_URLS["owa"]), alignment=Qt.AlignTop)
        owa_icon_col.addStretch(1)
        owa_row.addLayout(owa_icon_col)

        owa_note = QLabel(
            "<b>OWA index — no tunable input.</b> MESA also produces a third index, the OWA "
            "(Ordered Weighted Average) index, alongside the two weighted ones above. It ranks "
            "cells using a precautionary &ldquo;worst-first&rdquo; rule: a cell containing even one "
            "overlap at sensitivity 25 outranks any cell with zero overlaps at 25, regardless of "
            "the lower-sensitivity stack. There are no weights to set — the rule is fixed and "
            "uses only the sensitivity counts already produced from the table above."
        )
        owa_note.setWordWrap(True)
        owa_note.setTextFormat(Qt.RichText)
        owa_note.setStyleSheet(
            "color: #5c4a2f; font-size: 9pt; padding: 8px 10px; "
            "background-color: #f6efe1; border: 1px solid #d8c9a4; border-radius: 4px;"
        )
        owa_row.addWidget(owa_note, stretch=1)

        layout.addLayout(owa_row)

        index_weight_vars = weight_widgets
        layout.addStretch(1)

    # ------------------------------------------------------------------
    def _build_sensitivity_view(self, parent: QWidget):
        global entries_vuln
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        info = QLabel(
            "Set the importance and susceptibility for each asset layer. "
            "Sensitivity is calculated automatically as importance × susceptibility. "
            "Click a column header to sort."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #5c4a2f; font-size: 9pt; padding: 4px 0;")
        layout.addWidget(info)

        headers = ["Dataset", "Importance", "Susceptibility", "Sensitivity", "Code", "Description"]
        table = QTableWidget(0, len(headers), parent)
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )

        hdr = table.horizontalHeader()
        hdr.setSectionsClickable(True)
        hdr.setSortIndicatorShown(True)
        # Responsive column sizing: the two text columns share leftover width,
        # the four numeric columns size to their contents.
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)           # Dataset
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Importance
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Susceptibility
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Sensitivity
        hdr.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Code
        hdr.setSectionResizeMode(5, QHeaderView.Stretch)           # Description

        self._vuln_table = table
        # Guard to stop the itemChanged handler re-entering when it writes back
        # the derived sensitivity/code/description cells.
        self._vuln_suppress_itemchanged = False

        entries_vuln = []
        # Disable sorting while populating so row positions stay stable; we
        # enable sorting + apply the default A-Z sort once the table is full.
        table.setSortingEnabled(False)
        for i, row in enumerate(gdf_asset_group.itertuples()):
            self._add_vuln_row(i, row, table, entries_vuln, gdf_asset_group)
        table.setSortingEnabled(True)
        table.sortItems(0, Qt.AscendingOrder)

        table.itemChanged.connect(self._on_vuln_item_changed)

        layout.addWidget(table, stretch=1)

    # ------------------------------------------------------------------
    def _add_vuln_row(self, row_idx_0based, row, table, entries_list, gdf):
        name_val = str(getattr(row, 'name_original', ''))
        imp_val = int(coerce_valid_int(
            str(getattr(row, 'importance', FALLBACK_VULN)),
            valid_input_values, FALLBACK_VULN,
        ))
        sus_val = int(coerce_valid_int(
            str(getattr(row, 'susceptibility', FALLBACK_VULN)),
            valid_input_values, FALLBACK_VULN,
        ))
        try:
            sens_val = int(getattr(row, 'sensitivity', 0) or 0)
        except Exception:
            sens_val = 0
        if sens_val <= 0:
            sens_val = imp_val * sus_val
        code_val = str(getattr(row, 'sensitivity_code', '') or '')
        desc_val = str(getattr(row, 'sensitivity_description', '') or '')

        row_pos = table.rowCount()
        table.insertRow(row_pos)

        # Column 0 - Dataset (read-only, text sort)
        name_item = QTableWidgetItem(name_val)
        name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
        name_item.setData(Qt.UserRole, row_idx_0based)
        table.setItem(row_pos, 0, name_item)

        # Column 1 - Importance (editable, numeric sort)
        imp_item = _NumericTableItem(str(imp_val))
        imp_item.setTextAlignment(Qt.AlignCenter)
        imp_item.setData(Qt.UserRole, imp_val)
        table.setItem(row_pos, 1, imp_item)

        # Column 2 - Susceptibility (editable, numeric sort)
        sus_item = _NumericTableItem(str(sus_val))
        sus_item.setTextAlignment(Qt.AlignCenter)
        sus_item.setData(Qt.UserRole, sus_val)
        table.setItem(row_pos, 2, sus_item)

        # Column 3 - Sensitivity (read-only, numeric sort)
        sens_item = _NumericTableItem(str(sens_val))
        sens_item.setFlags(sens_item.flags() & ~Qt.ItemIsEditable)
        sens_item.setTextAlignment(Qt.AlignCenter)
        sens_item.setData(Qt.UserRole, sens_val)
        table.setItem(row_pos, 3, sens_item)

        # Column 4 - Code (read-only, text sort)
        code_item = QTableWidgetItem(code_val)
        code_item.setFlags(code_item.flags() & ~Qt.ItemIsEditable)
        code_item.setTextAlignment(Qt.AlignCenter)
        table.setItem(row_pos, 4, code_item)

        # Column 5 - Description (read-only, text sort)
        desc_item = QTableWidgetItem(desc_val)
        desc_item.setFlags(desc_item.flags() & ~Qt.ItemIsEditable)
        table.setItem(row_pos, 5, desc_item)

        entries_list.append({
            'row_index': row_idx_0based,
            'name': name_item,
            'importance': imp_item,
            'susceptibility': sus_item,
            'sensitivity': sens_item,
            'sensitivity_code': code_item,
            'sensitivity_description': desc_item,
        })

    # ------------------------------------------------------------------
    def _on_vuln_item_changed(self, item):
        if self._vuln_suppress_itemchanged:
            return
        col = item.column()
        if col not in (1, 2):
            return
        row_pos = item.row()
        name_item = self._vuln_table.item(row_pos, 0)
        if name_item is None:
            return
        idx = None
        for k, entry in enumerate(entries_vuln):
            if entry['name'] is name_item:
                idx = k
                break
        if idx is None:
            return

        self._vuln_suppress_itemchanged = True
        try:
            # Coerce user input and normalise display + numeric sort key.
            try:
                v = int(coerce_valid_int(
                    (item.text() or "").strip(),
                    valid_input_values, FALLBACK_VULN,
                ))
                item.setData(Qt.UserRole, v)
                if item.text() != str(v):
                    item.setText(str(v))
            except Exception:
                pass

            calculate_sensitivity(
                entries_vuln[idx]['importance'],
                entries_vuln[idx]['susceptibility'],
                idx,
                entries_vuln,
                gdf_asset_group,
            )

            # Refresh the sort key on the derived Sensitivity cell so numeric
            # sorting on that column reflects the new value.
            sens_item = entries_vuln[idx]['sensitivity']
            try:
                sens_item.setData(Qt.UserRole, int(sens_item.text() or 0))
            except Exception:
                pass
        finally:
            self._vuln_suppress_itemchanged = False

    # ------------------------------------------------------------------
    def _do_save_all_excel(self):
        input_folder = os.path.join(original_working_directory, "input")
        os.makedirs(input_folder, exist_ok=True)
        excel_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Excel File",
            input_folder,
            "Excel Files (*.xlsx);;All Files (*.*)",
        )
        if excel_path:
            update_all_vuln_rows(entries_vuln, gdf_asset_group)
            enforce_vuln_dtypes_inplace(gdf_asset_group)
            save_all_to_excel(gdf_asset_group, excel_path)

    def _do_load_all_excel(self):
        input_folder = os.path.join(original_working_directory, "input")
        excel_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Excel File",
            input_folder,
            "Excel Files (*.xlsx);;All Files (*.*)",
        )
        if excel_path:
            load_all_from_excel(excel_path)

    def _do_save_database(self):
        update_all_vuln_rows(entries_vuln, gdf_asset_group)
        enforce_vuln_dtypes_inplace(gdf_asset_group)
        gdf_ready = sanitize_vulnerability(gdf_asset_group, valid_input_values, FALLBACK_VULN)
        save_asset_group_to_parquet(gdf_ready, original_working_directory)
        persist_index_weights_from_ui(strict=False, silent=True)
        _set_status_message("Saved asset-group layer to GeoParquet.")

    # ------------------------------------------------------------------
    def _close_application(self):
        try:
            update_all_vuln_rows(entries_vuln, gdf_asset_group)
            enforce_vuln_dtypes_inplace(gdf_asset_group)
            gdf_ready = sanitize_vulnerability(gdf_asset_group, valid_input_values, FALLBACK_VULN)
            persist_index_weights_from_ui(strict=False, silent=True)
            save_asset_group_to_parquet(gdf_ready, original_working_directory)
        finally:
            self.close()

    def closeEvent(self, event):
        """Handle window close via X button."""
        try:
            update_all_vuln_rows(entries_vuln, gdf_asset_group)
            enforce_vuln_dtypes_inplace(gdf_asset_group)
            gdf_ready = sanitize_vulnerability(gdf_asset_group, valid_input_values, FALLBACK_VULN)
            persist_index_weights_from_ui(strict=False, silent=True)
            save_asset_group_to_parquet(gdf_ready, original_working_directory)
        except Exception:
            pass
        event.accept()


# -------------------------------
# In-process entry point (called by mesa.py via lazy import)
# -------------------------------
def run(base_dir: str, master=None) -> None:
    """Launch the processing setup GUI in-process.

    mesa.py calls this instead of spawning a subprocess.
    """
    global original_working_directory, config_file, status_message_var
    global valid_input_values, FALLBACK_VULN, gdf_asset_group, entries_vuln, root
    global index_weight_settings, classification, workingprojection_epsg

    # Resolve base dir robustly
    resolved_base = find_base_dir(base_dir)
    original_working_directory = str(resolved_base)

    # ---- tiny diagnostics (helps catch path mistakes fast)
    log_to_file(f"[processing_setup] start_cwd: {START_CWD}")
    log_to_file(f"[processing_setup] app_dir  : {APP_DIR}")
    log_to_file(f"[processing_setup] base_dir : {original_working_directory}")

    # Config
    config_file = os.path.join(original_working_directory, "config.ini")
    config = read_config(config_file)
    classification = read_config_classification(config_file)
    valid_input_values = get_valid_values(config)
    FALLBACK_VULN = get_fallback_value(config, valid_input_values)
    index_weight_settings = load_index_weight_settings(config)

    # Seed defaults when weights are missing OR present but blank/"nil".
    seed_needed = []
    try:
        for opt in INDEX_WEIGHT_KEYS.values():
            if _is_blankish(_cfg_get_any(config, opt, fallback="")):
                seed_needed.append(opt)
    except Exception:
        seed_needed = list(INDEX_WEIGHT_KEYS.values())

    if seed_needed:
        try:
            persist_index_weight_settings(config_file, index_weight_settings)
            log_to_file("Seeded default index weights in config.ini")
        except Exception as err:
            log_to_file(f"Unable to seed index weights: {err}")

    workingprojection_epsg = (
        _cfg_get_any(config, 'working_projection_epsg', fallback='')
        or _cfg_get_any(config, 'workingprojection_epsg', fallback='4326')
        or '4326'
    )

    # Load data stack
    _load_runtime_data_stack()

    # Load asset group
    gdf_asset_group = load_asset_group(original_working_directory)
    log_to_file(f"[processing_setup] asset grp: {_parquet_asset_group_path(original_working_directory)}")
    if gdf_asset_group is None:
        log_to_file("Failed to load tbl_asset_group (Parquet).")
        sys.exit(1)

    # If QApplication already exists (e.g. launched from mesa.py),
    # just create the window. Otherwise create a new app.
    app = QApplication.instance()
    own_app = False
    if app is None:
        app = QApplication([])
        apply_shared_stylesheet(app)
        own_app = True

    window = SetupWindow(original_working_directory)
    window.show()
    root = window

    if own_app:
        app.exec()
    return window


# -------------------------------
# Entrypoint (single, Parquet+JSON)
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MESA - Processing setup')
    parser.add_argument('--original_working_directory', required=False, help='Path to running folder')
    args = parser.parse_args()
    run(args.original_working_directory)
