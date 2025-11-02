# -*- coding: utf-8 -*-
"""
data_analysis.py — Interactive polygon analysis tool for MESA.

This helper lets power users digitise ad-hoc analysis polygons and run an
asset-intersection workflow that mirrors the summary tables from the
desktop data report.  The resulting PDF lives in ``output/`` and can be
shared independently of the main UI.
"""

from __future__ import annotations

import argparse
import configparser
import datetime as dt
import locale
import math
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import shape as shp_from_geojson, mapping as shp_to_geojson
from shapely.geometry.base import BaseGeometry

import tkinter as tk
from tkinter import filedialog

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

try:
    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
except Exception:
    try:
        locale.setlocale(locale.LC_ALL, "")
    except Exception:
        pass

os.environ.setdefault("PYWEBVIEW_GUI", "edgechromium")
os.environ.setdefault("PYWEBVIEW_LOG", "error")
os.environ.setdefault("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", "--disable-logging --log-level=3")

try:
    import webview

    try:
        webview.logger.disabled = True
    except Exception:
        pass
except ModuleNotFoundError as exc:
    raise SystemExit(
        "pywebview is required for data_analysis.py (pip install pywebview==4.*)"
    ) from exc

# ---------------------------------------------------------------------------
# Constants / configuration helpers
# ---------------------------------------------------------------------------

ANALYSIS_POLYGON_TABLE = "tbl_analysis_polygons.parquet"
ANALYSIS_GROUP_TABLE = "tbl_analysis_group.parquet"
ANALYSIS_STACKED_TABLE = "tbl_analysis_stacked.parquet"
ANALYSIS_FLAT_TABLE = "tbl_analysis_flat.parquet"
ASSET_OBJECT_TABLE = "tbl_asset_object.parquet"
ASSET_GROUP_TABLE = "tbl_asset_group.parquet"
DEFAULT_PARQUET_SUBDIR = "output/geoparquet"
DEFAULT_MBTILES_SUBDIR = "output/mbtiles"
REPORT_FILENAME_TEMPLATE = "MESA_area_analysis_report_{ts}.pdf"
DEFAULT_ANALYSIS_GEOCODE = "basic_mosaic"


def debug_log(base_dir: Path, message: str) -> None:
    """
    Append a timestamped message to ``log.txt`` for diagnostics.
    Fail silently to avoid disrupting the UI if the filesystem is unavailable.
    """
    try:
        ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        target = base_dir / "log.txt"
        if not target.exists():
            alt = base_dir / "code" / "log.txt"
            if alt.exists():
                target = alt
        path = target.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"[{ts}] [data_analysis] {message}\n")
    except Exception:
        pass


def resolve_base_dir(cli_path: Optional[str] = None) -> Path:
    """Mirror the directory probing logic used across other helpers."""
    candidates: List[Path] = []
    env_base = os.environ.get("MESA_BASE_DIR")
    if env_base:
        candidates.append(Path(env_base))
    if cli_path:
        candidates.append(Path(cli_path))

    here = Path(__file__).resolve()
    candidates.extend([here.parent, here.parent.parent, here.parent.parent.parent])
    cwd = Path(os.getcwd())
    candidates.extend([cwd, cwd / "code"])

    seen: set[Path] = set()
    ordered: List[Path] = []
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except Exception:
            resolved = cand
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)

    for cand in ordered:
        if (cand / "config.ini").exists():
            return cand
        if (cand / "system" / "config.ini").exists():
            return cand

    if here.parent.name.lower() == "system":
        return here.parent.parent
    return here.parent


def read_config(base_dir: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    paths = [base_dir / "config.ini", base_dir / "system" / "config.ini"]
    for path in paths:
        if path.exists():
            try:
                cfg.read(path, encoding="utf-8")
            except Exception:
                cfg.read(path)
            break
    if "DEFAULT" not in cfg:
        cfg["DEFAULT"] = {}
    return cfg


def parquet_dir(base_dir: Path, cfg: configparser.ConfigParser) -> Path:
    sub = cfg["DEFAULT"].get("parquet_folder", DEFAULT_PARQUET_SUBDIR)
    folder = (base_dir / sub).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _parquet_search_dirs(base_dir: Path, cfg: configparser.ConfigParser) -> List[Path]:
    """Return candidate directories that may contain parquet datasets."""
    sub = cfg["DEFAULT"].get("parquet_folder", DEFAULT_PARQUET_SUBDIR)
    base = base_dir.resolve()
    candidates: List[Path] = []

    def _add(path: Path) -> None:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved not in candidates:
            candidates.append(resolved)

    _add(base / sub)
    _add(base / "code" / sub)
    _add(base / "code" / "output" / "geoparquet")
    _add(base / "output" / "geoparquet")
    _add(base / "code" / "output")
    _add(base / "output")
    return candidates


def find_parquet_file(base_dir: Path, cfg: configparser.ConfigParser, filename: str) -> Optional[Path]:
    for directory in _parquet_search_dirs(base_dir, cfg):
        candidate = directory / filename
        if candidate.exists():
            return candidate
    return None


def find_dataset_dir(base_dir: Path, cfg: configparser.ConfigParser, dirname: str) -> Optional[Path]:
    for directory in _parquet_search_dirs(base_dir, cfg):
        candidate = directory / dirname
        if candidate.exists():
            return candidate
    return None


def _mbtiles_search_dirs(base_dir: Path) -> List[Path]:
    base = base_dir.resolve()
    candidates = [
        base / DEFAULT_MBTILES_SUBDIR,
        base / "code" / DEFAULT_MBTILES_SUBDIR,
        base / "code" / "output" / "mbtiles",
        base / "output" / "mbtiles",
    ]
    seen: set[Path] = set()
    ordered: List[Path] = []
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except Exception:
            resolved = cand
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    return ordered


def _norm_key(value: str) -> str:
    return (value or "").strip().lower().replace(" ", "_").replace("-", "_")


_MBTILES_LOCK = threading.Lock()
_MBTILES_INDEX: Dict[str, Dict[str, Optional[Path]]] = {}
_MBTILES_REV: Dict[str, str] = {}
_MBTILES_META_CACHE: Dict[Path, Dict[str, Any]] = {}
_MBTILES_BASE_URL: Optional[str] = None


def _scan_mbtiles_dir(directory: Path) -> tuple[Dict[str, Dict[str, Optional[Path]]], Dict[str, str]]:
    idx: Dict[str, Dict[str, Optional[Path]]] = {}
    rev: Dict[str, str] = {}
    if not directory.exists() or not directory.is_dir():
        return idx, rev

    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        name = entry.name.lower()
        if not name.endswith(".mbtiles"):
            continue
        kind = None
        base = entry.name
        if name.endswith("_sensitivity.mbtiles"):
            cat = base[: -len("_sensitivity.mbtiles")]
            kind = "sensitivity"
        elif name.endswith("_envindex.mbtiles"):
            cat = base[: -len("_envindex.mbtiles")]
            kind = "envindex"
        elif name.endswith("_groupstotal.mbtiles"):
            cat = base[: -len("_groupstotal.mbtiles")]
            kind = "groupstotal"
        elif name.endswith("_assetstotal.mbtiles"):
            cat = base[: -len("_assetstotal.mbtiles")]
            kind = "assetstotal"
        else:
            continue
        idx.setdefault(cat, {"sensitivity": None, "envindex": None, "groupstotal": None, "assetstotal": None})
        idx[cat][kind] = entry
        rev[_norm_key(cat)] = cat
    return idx, rev


class _MBTilesHandler(BaseHTTPRequestHandler):
    connections: Dict[Path, sqlite3.Connection] = {}

    def log_message(self, format: str, *args: Any) -> None:
        return

    def log_error(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        global _MBTILES_INDEX
        try:
            parts = [p for p in self.path.split("?", 1)[0].split("/") if p]
            if len(parts) != 6 or parts[0] != "tiles":
                self.send_response(404)
                self.end_headers()
                return
            _, kind, cat_enc, z_s, x_s, y_file = parts
            kind = (kind or "").lower()
            cat = unquote(cat_enc)

            disp, rec = _resolve_mbtiles(cat)
            if kind not in {"sensitivity", "envindex", "groupstotal", "assetstotal"} or not rec:
                self.send_response(404)
                self.end_headers()
                return

            db_path = rec.get(kind)
            if not db_path or not db_path.exists():
                self.send_response(404)
                self.end_headers()
                return

            z = int(z_s)
            x = int(x_s)
            y = int(y_file.rsplit(".", 1)[0])

            con = self.connections.get(db_path)
            if con is None:
                con = sqlite3.connect(db_path, check_same_thread=False)
                self.connections[db_path] = con

            tms_y = (1 << z) - 1 - y
            cur = con.cursor()
            cur.execute(
                "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
                (z, x, tms_y),
            )
            row = cur.fetchone()
            if not row:
                cur.execute(
                    "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
                    (z, x, y),
                )
                row = cur.fetchone()
            if not row:
                self.send_response(204)
                self.end_headers()
                return

            data = row[0]
            fmt = _mbtiles_meta(db_path)["format"]
            self.send_response(200)
            self.send_header("Content-Type", fmt)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            try:
                self.send_response(500)
                self.end_headers()
            except Exception:
                pass


def _start_mbtiles_server() -> Optional[str]:
    global _MBTILES_INDEX
    if not _MBTILES_INDEX:
        return None
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MBTilesHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}"


def _mbtiles_meta(path: Path) -> Dict[str, Any]:
    if path in _MBTILES_META_CACHE:
        return _MBTILES_META_CACHE[path]
    out = {"bounds": [[-85.0, -180.0], [85.0, 180.0]], "minzoom": 0, "maxzoom": 19, "format": "image/png"}
    try:
        con = sqlite3.connect(path)
        try:
            cur = con.cursor()
            cur.execute("SELECT name, value FROM metadata")
            meta = {str(k).lower(): (v if v is not None else "") for k, v in cur.fetchall()}
            fmt = (meta.get("format", "") or "").lower()
            if fmt in {"jpg", "jpeg"}:
                out["format"] = "image/jpeg"
            bounds_raw = meta.get("bounds", "")
            if bounds_raw:
                parts = [float(x) for x in str(bounds_raw).split(",")]
                if len(parts) == 4:
                    minx, miny, maxx, maxy = parts
                    out["bounds"] = [[miny, minx], [maxy, maxx]]
            try:
                out["minzoom"] = int(meta.get("minzoom", "0"))
            except Exception:
                pass
            try:
                out["maxzoom"] = int(meta.get("maxzoom", "19"))
            except Exception:
                pass
        finally:
            con.close()
    except Exception:
        pass
    _MBTILES_META_CACHE[path] = out
    return out


def _resolve_mbtiles(cat: str) -> tuple[str, Optional[Dict[str, Optional[Path]]]]:
    key = _norm_key(cat)
    if cat in _MBTILES_INDEX:
        return cat, _MBTILES_INDEX[cat]
    if key in _MBTILES_REV:
        canonical = _MBTILES_REV[key]
        return canonical, _MBTILES_INDEX.get(canonical)
    for disp in _MBTILES_INDEX.keys():
        if disp.lower() == cat.lower():
            return disp, _MBTILES_INDEX.get(disp)
    return cat, None


def _ensure_mbtiles(base_dir: Path) -> Optional[str]:
    global _MBTILES_INDEX, _MBTILES_REV, _MBTILES_BASE_URL
    with _MBTILES_LOCK:
        if _MBTILES_BASE_URL is not None:
            return _MBTILES_BASE_URL
        aggregated_index: Dict[str, Dict[str, Optional[Path]]] = {}
        aggregated_rev: Dict[str, str] = {}
        for directory in _mbtiles_search_dirs(base_dir):
            idx, rev = _scan_mbtiles_dir(directory)
            for cat, rec in idx.items():
                aggregated_index.setdefault(cat, {"sensitivity": None, "envindex": None, "groupstotal": None, "assetstotal": None})
                for kind, path in rec.items():
                    if path:
                        aggregated_index[cat][kind] = path
            aggregated_rev.update(rev)
        _MBTILES_INDEX = aggregated_index
        _MBTILES_REV = aggregated_rev
        if not _MBTILES_INDEX:
            _MBTILES_BASE_URL = None
            return None
        _MBTILES_BASE_URL = _start_mbtiles_server()
        return _MBTILES_BASE_URL


def _mbtiles_info(category: str) -> Optional[Dict[str, Any]]:
    disp, rec = _resolve_mbtiles(category)
    if not rec or not _MBTILES_BASE_URL:
        return None
    src = rec.get("sensitivity") or rec.get("envindex") or rec.get("groupstotal") or rec.get("assetstotal")
    meta = _mbtiles_meta(src) if src else {"bounds": [[-85.0, -180.0], [85.0, 180.0]], "minzoom": 0, "maxzoom": 19, "format": "image/png"}

    def build(kind: str) -> Optional[str]:
        path = rec.get(kind)
        if path and path.exists():
            return f"{_MBTILES_BASE_URL}/tiles/{kind}/{disp}/{{z}}/{{x}}/{{y}}.png"
        return None

    bounds = meta.get("bounds", [[-85.0, -180.0], [85.0, 180.0]])
    if isinstance(bounds, list) and len(bounds) == 2 and all(isinstance(b, (list, tuple)) and len(b) == 2 for b in bounds):
        min_lat, min_lon = bounds[0]
        max_lat, max_lon = bounds[1]
        bounds4 = [min_lon, min_lat, max_lon, max_lat]
    else:
        bounds4 = [-180.0, -85.0, 180.0, 85.0]

    return {
        "category": disp,
        "sensitivity_url": build("sensitivity"),
        "envindex_url": build("envindex"),
        "groupstotal_url": build("groupstotal"),
        "assetstotal_url": build("assetstotal"),
        "bounds": bounds4,
        "minzoom": meta.get("minzoom", 0),
        "maxzoom": meta.get("maxzoom", 19),
    }


def analysis_polygon_path(base_dir: Path, cfg: configparser.ConfigParser) -> Path:
    return parquet_dir(base_dir, cfg) / ANALYSIS_POLYGON_TABLE


def analysis_group_path(base_dir: Path, cfg: configparser.ConfigParser) -> Path:
    return parquet_dir(base_dir, cfg) / ANALYSIS_GROUP_TABLE


def analysis_stacked_path(base_dir: Path, cfg: configparser.ConfigParser) -> Path:
    return parquet_dir(base_dir, cfg) / ANALYSIS_STACKED_TABLE


def analysis_flat_path(base_dir: Path, cfg: configparser.ConfigParser) -> Path:
    return parquet_dir(base_dir, cfg) / ANALYSIS_FLAT_TABLE


def asset_object_path(base_dir: Path, cfg: configparser.ConfigParser) -> Path:
    existing = find_parquet_file(base_dir, cfg, ASSET_OBJECT_TABLE)
    if existing:
        return existing
    return parquet_dir(base_dir, cfg) / ASSET_OBJECT_TABLE


def asset_group_path(base_dir: Path, cfg: configparser.ConfigParser) -> Path:
    existing = find_parquet_file(base_dir, cfg, ASSET_GROUP_TABLE)
    if existing:
        return existing
    return parquet_dir(base_dir, cfg) / ASSET_GROUP_TABLE


def _km2(area_sq_m: float) -> str:
    if area_sq_m is None or not math.isfinite(area_sq_m) or area_sq_m <= 0:
        return "-"
    return f"{area_sq_m / 1_000_000.0:,.2f} km²"


def _timestamp_label(ts: Optional[dt.datetime] = None) -> str:
    return (ts or dt.datetime.now()).strftime("%Y_%m_%d")


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _is_blank(value: Any) -> bool:
    if _is_nullish(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _geo_from_wkb_column(df: pd.DataFrame, column: str = "geometry", epsg: Optional[int] = None) -> gpd.GeoDataFrame:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' missing from table.")
    geoms = gpd.GeoSeries.from_wkb(df[column].values, crs=f"EPSG:{epsg}" if epsg else None)
    gdf = gpd.GeoDataFrame(df.drop(columns=[column]), geometry=geoms)
    if epsg and gdf.crs is None:
        gdf.set_crs(epsg=epsg, inplace=True)
    return gdf


@dataclass
class AnalysisGroup:
    identifier: str
    name: str
    notes: str
    created_at: dt.datetime
    updated_at: dt.datetime
    default_geocode: Optional[str] = None

    @property
    def created_label(self) -> str:
        return self.created_at.strftime("%Y-%m-%d %H:%M")


@dataclass
class AnalysisRecord:
    identifier: str
    group_id: str
    title: str
    notes: str
    created_at: dt.datetime
    updated_at: dt.datetime
    geometry: BaseGeometry

    @property
    def created_label(self) -> str:
        return self.created_at.strftime("%Y-%m-%d %H:%M")


class AnalysisStorage:
    """Persistence layer around analysis groups and polygons (GeoParquet)."""

    POLYGON_COLUMNS = ["id", "group_id", "title", "notes", "created_at", "updated_at", "geometry"]
    GROUP_COLUMNS = ["id", "name", "notes", "created_at", "updated_at", "default_geocode"]

    def __init__(self, base_dir: Path, cfg: configparser.ConfigParser) -> None:
        self.base_dir = base_dir
        self._base_dir = base_dir
        self._base_dir = base_dir
        self.cfg = cfg
        self.polygon_path = analysis_polygon_path(base_dir, cfg)
        self.group_path = analysis_group_path(base_dir, cfg)
        self._lock = threading.Lock()
        self._existing_polygon_path = find_parquet_file(base_dir, cfg, ANALYSIS_POLYGON_TABLE)
        self._existing_group_path = find_parquet_file(base_dir, cfg, ANALYSIS_GROUP_TABLE)

        try:
            self.storage_epsg = int(str(cfg["DEFAULT"].get("workingprojection_epsg", "4326")))
        except Exception:
            self.storage_epsg = 4326

        self._groups_df = self._load_or_init_groups()
        self._polygons_gdf = self._load_or_init_polygons()
        self._active_group_id = self._groups_df["id"].astype(str).iloc[0]
        debug_log(self.base_dir, f"AnalysisStorage initialised with {len(self._groups_df)} group(s) and {len(self._polygons_gdf)} polygon(s)")

    # ------------------------------------------------------------------ internal helpers
    @staticmethod
    def _normalize_dt(value: Any) -> dt.datetime:
        if isinstance(value, dt.datetime):
            return value
        if _is_blank(value):
            return dt.datetime.utcnow()
        try:
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.isna(parsed):
                return dt.datetime.utcnow()
            return parsed.to_pydatetime()
        except Exception:
            return dt.datetime.utcnow()

    def _load_or_init_groups(self) -> pd.DataFrame:
        path = self._existing_group_path or self.group_path
        if path and path.exists():
            try:
                df = pd.read_parquet(path)
            except Exception:
                df = pd.DataFrame(columns=self.GROUP_COLUMNS)
        else:
            df = pd.DataFrame(columns=self.GROUP_COLUMNS)

        for col in self.GROUP_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        if df.empty:
            now = dt.datetime.utcnow().isoformat()
            df = pd.DataFrame(
                [
                    {
                        "id": "group_default",
                        "name": "Analyseområder",
                        "notes": "",
                        "created_at": now,
                        "updated_at": now,
                        "default_geocode": DEFAULT_ANALYSIS_GEOCODE,
                    }
                ],
                columns=self.GROUP_COLUMNS,
            )
        df = df[self.GROUP_COLUMNS].copy()
        df["id"] = df["id"].astype(str)
        debug_log(self.base_dir, f"Loaded analysis groups from {path or 'memory'} ({len(df)} rows)")
        return df

    def _load_or_init_polygons(self) -> gpd.GeoDataFrame:
        path = self._existing_polygon_path or self.polygon_path
        if not path or not path.exists():
            base = gpd.GeoDataFrame(
                {col: pd.Series(dtype="object") for col in self.POLYGON_COLUMNS if col != "geometry"},
                geometry=gpd.GeoSeries(dtype="object"),
                crs=f"EPSG:{self.storage_epsg}",
            )
            base["group_id"] = base.get("group_id", pd.Series(dtype="object"))
            debug_log(self.base_dir, f"No polygon parquet found at {path}; starting with empty table")
            return base

        gdf = gpd.read_parquet(path)
        for col in self.POLYGON_COLUMNS:
            if col not in gdf.columns:
                if col == "geometry":
                    gdf = gpd.GeoDataFrame(gdf, geometry=gpd.GeoSeries(dtype="object"), crs=gdf.crs)
                else:
                    gdf[col] = pd.NA
        if gdf.crs is None:
            gdf.set_crs(epsg=self.storage_epsg, inplace=True, allow_override=True)
        gdf = gdf[self.POLYGON_COLUMNS].copy()
        gdf["group_id"] = gdf["group_id"].astype(str)

        valid_group_ids = set(self._groups_df["id"].astype(str))
        default_group_id = next(iter(valid_group_ids))
        gdf.loc[~gdf["group_id"].isin(valid_group_ids), "group_id"] = default_group_id
        debug_log(self.base_dir, f"Loaded {len(gdf)} polygons from {path}")
        return gdf

    def _ensure_unique_group_id(self) -> str:
        existing = set(self._groups_df["id"].astype(str))
        while True:
            cand = f"group_{uuid.uuid4().hex[:6]}"
            if cand not in existing:
                return cand

    def _ensure_unique_polygon_id(self) -> str:
        existing = set(self._polygons_gdf["id"].astype(str))
        while True:
            cand = f"analysis_{uuid.uuid4().hex[:8]}"
            if cand not in existing:
                return cand

    def _group_from_row(self, row: pd.Series) -> AnalysisGroup:
        created_dt = self._normalize_dt(row.get("created_at"))
        updated_dt = self._normalize_dt(row.get("updated_at"))
        name_val = row.get("name")
        if _is_blank(name_val):
            name = str(row["id"])
        else:
            name = str(name_val)
        notes_val = row.get("notes")
        notes = "" if _is_blank(notes_val) else str(notes_val)
        geocode_val = row.get("default_geocode")
        default_geocode = None if _is_blank(geocode_val) else str(geocode_val)
        return AnalysisGroup(
            identifier=str(row["id"]),
            name=name,
            notes=notes,
            created_at=created_dt,
            updated_at=updated_dt if updated_dt else created_dt,
            default_geocode=default_geocode,
        )

    def _record_from_row(self, row: pd.Series) -> AnalysisRecord:
        created_dt = self._normalize_dt(row.get("created_at"))
        updated_dt = self._normalize_dt(row.get("updated_at"))
        title_val = row.get("title")
        if _is_blank(title_val):
            title = str(row["id"])
        else:
            title = str(title_val)
        notes_val = row.get("notes")
        notes = "" if _is_blank(notes_val) else str(notes_val)
        return AnalysisRecord(
            identifier=str(row["id"]),
            group_id=str(row["group_id"]),
            title=title,
            notes=notes,
            created_at=created_dt,
            updated_at=updated_dt if updated_dt else created_dt,
            geometry=row["geometry"],
        )

    def _resolve_group_id(self, group_id: Optional[str]) -> str:
        with self._lock:
            valid = set(self._groups_df["id"].astype(str))
            if group_id and group_id in valid:
                return group_id
            if self._active_group_id in valid:
                return self._active_group_id
            return next(iter(valid))

    def _write_groups(self) -> None:
        os.makedirs(self.group_path.parent, exist_ok=True)
        self._groups_df.to_parquet(self.group_path, index=False)
        self._existing_group_path = self.group_path
        debug_log(self.base_dir, f"Wrote analysis groups to {self.group_path} ({len(self._groups_df)} rows)")

    def _write_polygons(self) -> None:
        os.makedirs(self.polygon_path.parent, exist_ok=True)
        self._polygons_gdf.to_parquet(self.polygon_path, index=False)
        self._existing_polygon_path = self.polygon_path
        debug_log(self.base_dir, f"Wrote analysis polygons to {self.polygon_path} ({len(self._polygons_gdf)} rows)")

    # ------------------------------------------------------------------ public API
    def active_group_id(self) -> str:
        return self._resolve_group_id(self._active_group_id)

    def set_active_group(self, group_id: str) -> AnalysisGroup:
        gid = self._resolve_group_id(group_id)
        with self._lock:
            self._active_group_id = gid
            row = self._groups_df[self._groups_df["id"] == gid].iloc[0]
        return self._group_from_row(row)

    def list_groups(self) -> List[AnalysisGroup]:
        with self._lock:
            return [self._group_from_row(row) for _, row in self._groups_df.iterrows()]

    def get_group(self, identifier: Optional[str]) -> AnalysisGroup:
        gid = self._resolve_group_id(identifier)
        with self._lock:
            row = self._groups_df[self._groups_df["id"] == gid].iloc[0]
        return self._group_from_row(row)

    def add_group(self, name: str, notes: str = "", default_geocode: Optional[str] = None) -> AnalysisGroup:
        with self._lock:
            new_id = self._ensure_unique_group_id()
            now = dt.datetime.utcnow().isoformat()
            entry = {
                "id": new_id,
                "name": name or new_id,
                "notes": notes or "",
                "created_at": now,
                "updated_at": now,
                "default_geocode": default_geocode or DEFAULT_ANALYSIS_GEOCODE,
            }
            self._groups_df = pd.concat([self._groups_df, pd.DataFrame([entry])], ignore_index=True)
            self._active_group_id = new_id
            self._write_groups()
            return self._group_from_row(pd.Series(entry))

    def update_group(
        self,
        identifier: str,
        name: Optional[str] = None,
        notes: Optional[str] = None,
        default_geocode: Optional[str] = None,
    ) -> AnalysisGroup:
        gid = self._resolve_group_id(identifier)
        with self._lock:
            mask = self._groups_df["id"] == gid
            if not mask.any():
                raise KeyError(gid)
            now = dt.datetime.utcnow().isoformat()
            if name is not None:
                self._groups_df.loc[mask, "name"] = name or gid
            if notes is not None:
                self._groups_df.loc[mask, "notes"] = notes or ""
            if default_geocode is not None:
                self._groups_df.loc[mask, "default_geocode"] = default_geocode or DEFAULT_ANALYSIS_GEOCODE
            self._groups_df.loc[mask, "updated_at"] = now
            row = self._groups_df.loc[mask].iloc[0]
            self._write_groups()
        return self._group_from_row(row)

    def delete_group(self, identifier: str) -> None:
        gid = self._resolve_group_id(identifier)
        with self._lock:
            if len(self._groups_df) <= 1:
                raise ValueError("Det må finnes minst én analysegruppe.")
            self._groups_df = self._groups_df[self._groups_df["id"] != gid].reset_index(drop=True)
            self._polygons_gdf = self._polygons_gdf[self._polygons_gdf["group_id"] != gid].reset_index(drop=True)
            remaining = self._groups_df["id"].astype(str)
            self._active_group_id = remaining.iloc[0]
            self._write_groups()
            self._write_polygons()

    def list_records(self, group_id: Optional[str] = None) -> List[AnalysisRecord]:
        gid = self._resolve_group_id(group_id)
        with self._lock:
            subset = self._polygons_gdf[self._polygons_gdf["group_id"] == gid].copy()
        return [self._record_from_row(row) for _, row in subset.iterrows()]

    def get_records(self, group_id: Optional[str], identifiers: Iterable[str]) -> List[AnalysisRecord]:
        gid = self._resolve_group_id(group_id)
        wanted = {str(i) for i in identifiers}
        with self._lock:
            subset = self._polygons_gdf[
                (self._polygons_gdf["group_id"] == gid) & (self._polygons_gdf["id"].astype(str).isin(wanted))
            ].copy()
        return [self._record_from_row(row) for _, row in subset.iterrows()]

    def to_feature_collection(self, group_id: Optional[str] = None) -> Dict[str, Any]:
        gid = self._resolve_group_id(group_id)
        with self._lock:
            gdf = self._polygons_gdf[self._polygons_gdf["group_id"] == gid].copy()
        if gdf.empty:
            return {"type": "FeatureCollection", "features": []}
        gdf_4326 = gdf.to_crs(4326)
        features = []
        for _, row in gdf_4326.iterrows():
            props = {
                "id": row["id"],
                "group_id": row["group_id"],
                "title": row["title"],
                "notes": row["notes"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            features.append({"type": "Feature", "geometry": shp_to_geojson(row.geometry), "properties": props})
        return {"type": "FeatureCollection", "features": features}

    def add_polygon(self, feature: Dict[str, Any], group_id: Optional[str] = None) -> AnalysisRecord:
        geom_ll = shp_from_geojson(feature.get("geometry"))
        if geom_ll is None:
            raise ValueError("Invalid geometry payload.")
        gid = self._resolve_group_id(group_id)
        geom_storage = gpd.GeoDataFrame(geometry=[geom_ll], crs=4326).to_crs(self.storage_epsg).geometry.iloc[0]
        new_id = self._ensure_unique_polygon_id()
        now = dt.datetime.utcnow().isoformat()
        entry = {
            "id": new_id,
            "group_id": gid,
            "title": feature.get("properties", {}).get("title") or new_id,
            "notes": feature.get("properties", {}).get("notes") or "",
            "created_at": now,
            "updated_at": now,
            "geometry": geom_storage,
        }
        with self._lock:
            self._polygons_gdf = pd.concat(
                [self._polygons_gdf, gpd.GeoDataFrame([entry], geometry="geometry", crs=self._polygons_gdf.crs)],
                ignore_index=True,
            )
            self._write_polygons()
        return self._record_from_row(pd.Series(entry))

    def update_properties(self, identifier: str, group_id: Optional[str], title: str, notes: str) -> AnalysisRecord:
        gid = self._resolve_group_id(group_id)
        with self._lock:
            mask = (self._polygons_gdf["group_id"] == gid) & (self._polygons_gdf["id"].astype(str) == str(identifier))
            if not mask.any():
                raise KeyError(identifier)
            self._polygons_gdf.loc[mask, "title"] = title
            self._polygons_gdf.loc[mask, "notes"] = notes
            self._polygons_gdf.loc[mask, "updated_at"] = dt.datetime.utcnow().isoformat()
            row = self._polygons_gdf.loc[mask].iloc[0]
            self._write_polygons()
        return self._record_from_row(row)

    def update_geometry(self, identifier: str, group_id: Optional[str], geometry: Dict[str, Any]) -> AnalysisRecord:
        geom_ll = shp_from_geojson(geometry)
        if geom_ll is None:
            raise ValueError("Invalid geometry payload.")
        gid = self._resolve_group_id(group_id)
        geom_storage = gpd.GeoDataFrame(geometry=[geom_ll], crs=4326).to_crs(self.storage_epsg).geometry.iloc[0]
        with self._lock:
            mask = (self._polygons_gdf["group_id"] == gid) & (self._polygons_gdf["id"].astype(str) == str(identifier))
            if not mask.any():
                raise KeyError(identifier)
            self._polygons_gdf.loc[mask, "geometry"] = [geom_storage]
            self._polygons_gdf.loc[mask, "updated_at"] = dt.datetime.utcnow().isoformat()
            row = self._polygons_gdf.loc[mask].iloc[0]
            self._write_polygons()
        return self._record_from_row(row)

    def delete(self, identifier: str, group_id: Optional[str]) -> None:
        gid = self._resolve_group_id(group_id)
        with self._lock:
            mask = (self._polygons_gdf["group_id"] == gid) & (self._polygons_gdf["id"].astype(str) == str(identifier))
            before = len(self._polygons_gdf)
            self._polygons_gdf = self._polygons_gdf.drop(self._polygons_gdf[mask].index).reset_index(drop=True)
            if len(self._polygons_gdf) == before:
                raise KeyError(identifier)
            self._write_polygons()

    def import_file(self, path: Path, group_id: Optional[str] = None) -> List[AnalysisRecord]:
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix in {".json", ".geojson"}:
            gdf = gpd.read_file(path)
        elif suffix in {".parquet", ".pq"}:
            gdf = gpd.read_parquet(path)
            if not isinstance(gdf, gpd.GeoDataFrame):
                gdf = _geo_from_wkb_column(pd.read_parquet(path))
        else:
            gdf = gpd.read_file(path)
        gdf = gdf.to_crs(self.storage_epsg)

        gid = self._resolve_group_id(group_id)
        imported: List[AnalysisRecord] = []
        for _, row in gdf.iterrows():
            if row.geometry is None or row.geometry.is_empty:
                continue
            props = {}
            for key in ("title", "name", "label"):
                if key in row and not _is_blank(row[key]):
                    props["title"] = str(row[key])
                    break
            feature = {"type": "Feature", "geometry": shp_to_geojson(row.geometry), "properties": props}
            imported.append(self.add_polygon(feature, gid))
        return imported

    def save(self) -> None:
        with self._lock:
            self._write_groups()
            self._write_polygons()


# ---------------------------------------------------------------------------
# Asset analysis
# ---------------------------------------------------------------------------


class AssetAnalyzer:
    """Compute intersection statistics and build the PDF report."""

    def __init__(self, base_dir: Path, cfg: configparser.ConfigParser, storage_epsg: int = 4326) -> None:
        self.base_dir = base_dir
        self._base_dir = base_dir
        self.cfg = cfg
        try:
            self.working_epsg = int(str(cfg["DEFAULT"].get("workingprojection_epsg", "4326")))
        except Exception:
            self.working_epsg = 4326
        try:
            self.area_epsg = int(str(cfg["DEFAULT"].get("area_projection_epsg", "3035")))
        except Exception:
            self.area_epsg = 3035
        self.storage_epsg = storage_epsg or 4326

        self.asset_objects = self._load_asset_objects()
        self.asset_groups = self._load_asset_groups()
        self.analysis_flat_path = analysis_flat_path(base_dir, cfg)
        self.analysis_stacked_path = analysis_stacked_path(base_dir, cfg)
        self._flat_dataset: Optional[gpd.GeoDataFrame] = None
        self._stacked_dataset: Optional[gpd.GeoDataFrame] = None
        self._flat_category_cache: Dict[str, gpd.GeoDataFrame] = {}
        self._stacked_category_cache: Dict[str, gpd.GeoDataFrame] = {}
        self._canvas_cache: Dict[str, tuple[list[Dict[str, Any]], Optional[List[float]]]] = {}
        self.default_geocode = DEFAULT_ANALYSIS_GEOCODE
        self.canvas_bounds: Optional[List[float]] = None
        try:
            if not self.asset_objects.empty:
                self.asset_bounds = tuple(self.asset_objects.to_crs(4326).total_bounds)
            else:
                self.asset_bounds = None
        except Exception:
            self.asset_bounds = None
        self.mbtiles_base_url = _ensure_mbtiles(self.base_dir)
        if self.mbtiles_base_url:
            debug_log(
                self._base_dir,
                f"MBTiles server started at {self.mbtiles_base_url} with {len(_MBTILES_INDEX)} category entries",
            )
        else:
            debug_log(self.base_dir, "No MBTiles background detected; falling back to GeoParquet.")
        debug_log(
            self.base_dir,
            f"AssetAnalyzer ready: assets={len(self.asset_objects)}, groups={len(self.asset_groups)}, asset_bounds={self.asset_bounds}"
        )

    def _load_asset_objects(self) -> gpd.GeoDataFrame:
        pq_path = asset_object_path(self.base_dir, self.cfg)
        if not pq_path.exists():
            debug_log(self.base_dir, f"Asset objects parquet not found at {pq_path}")
            raise FileNotFoundError(f"Asset objects table missing: {pq_path}")
        debug_log(self.base_dir, f"Loading asset objects from {pq_path}")
        df = pd.read_parquet(pq_path)
        gdf = _geo_from_wkb_column(df, epsg=self.working_epsg)
        if gdf.crs is None:
            gdf.set_crs(epsg=self.working_epsg, inplace=True, allow_override=True)
        debug_log(self.base_dir, f"Loaded {len(gdf)} asset objects")
        return gdf

    def _load_asset_groups(self) -> pd.DataFrame:
        pq_path = asset_group_path(self.base_dir, self.cfg)
        if not pq_path.exists():
            debug_log(self.base_dir, f"Asset group parquet not found at {pq_path}")
            raise FileNotFoundError(f"Asset group table missing: {pq_path}")
        debug_log(self.base_dir, f"Loading asset groups from {pq_path}")
        df = gpd.read_parquet(pq_path)
        keep = ["id", "name_original", "name_gis_assetgroup", "title_fromuser", "sensitivity_code", "sensitivity_description"]
        result = df[keep].copy()
        debug_log(self.base_dir, f"Loaded {len(result)} asset group rows")
        return result

    # ---------------------- datasets & geocode helpers ----------------------
    def _load_flat_dataset(self) -> gpd.GeoDataFrame:
        if self._flat_dataset is not None:
            return self._flat_dataset
        path = find_parquet_file(self.base_dir, self.cfg, "tbl_flat.parquet")
        if not path:
            debug_log(self.base_dir, "tbl_flat.parquet could not be located")
            raise FileNotFoundError("Presentation table missing (tbl_flat.parquet).")
        debug_log(self.base_dir, f"Loading tbl_flat.parquet from {path}")
        gdf = gpd.read_parquet(path)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
        self._flat_dataset = gdf
        debug_log(self.base_dir, f"tbl_flat loaded with {len(gdf)} rows")
        return self._flat_dataset

    def _load_stacked_dataset(self) -> gpd.GeoDataFrame:
        if self._stacked_dataset is not None:
            return self._stacked_dataset
        path = find_dataset_dir(self.base_dir, self.cfg, "tbl_stacked")
        if not path:
            debug_log(self.base_dir, "tbl_stacked dataset could not be located")
            raise FileNotFoundError("Stacked table missing (tbl_stacked).")
        debug_log(self.base_dir, f"Loading tbl_stacked from {path}")
        gdf = gpd.read_parquet(path)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
        self._stacked_dataset = gdf
        debug_log(self.base_dir, f"tbl_stacked loaded with {len(gdf)} rows")
        return self._stacked_dataset

    def available_geocode_categories(self) -> List[str]:
        try:
            flat = self._load_flat_dataset()
        except FileNotFoundError:
            debug_log(self.base_dir, "tbl_flat.parquet missing; defaulting geocode list to basic_mosaic")
            return [DEFAULT_ANALYSIS_GEOCODE]
        has_column = "name_gis_geocodegroup" in flat.columns
        if has_column:
            column = flat["name_gis_geocodegroup"].astype(str).str.strip()
            if DEFAULT_ANALYSIS_GEOCODE not in column.values:
                debug_log(self.base_dir, f"Geocode '{DEFAULT_ANALYSIS_GEOCODE}' not present; proceeding with empty filtered dataset")
        else:
            debug_log(self.base_dir, "tbl_flat.parquet missing geocode column; using full dataset for basic_mosaic")
        return [DEFAULT_ANALYSIS_GEOCODE]

    def _ensure_category(self, category: Optional[str]) -> str:
        return DEFAULT_ANALYSIS_GEOCODE

    def _flat_for_category(self, category: str) -> gpd.GeoDataFrame:
        cat = DEFAULT_ANALYSIS_GEOCODE
        cache_key = cat
        if cache_key in self._flat_category_cache:
            return self._flat_category_cache[cache_key]
        try:
            dataset = self._load_flat_dataset()
        except FileNotFoundError:
            empty = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
            self._flat_category_cache[cache_key] = empty
            return empty
        if "name_gis_geocodegroup" in dataset.columns:
            mask = dataset["name_gis_geocodegroup"].astype(str).str.strip() == cat
            subset = dataset.loc[mask].copy()
        else:
            subset = dataset.copy()
        self._flat_category_cache[cache_key] = subset
        return subset

    def _stacked_for_category(self, category: str) -> gpd.GeoDataFrame:
        cat = DEFAULT_ANALYSIS_GEOCODE
        cache_key = cat
        if cache_key in self._stacked_category_cache:
            return self._stacked_category_cache[cache_key]
        try:
            dataset = self._load_stacked_dataset()
        except FileNotFoundError:
            empty = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
            self._stacked_category_cache[cache_key] = empty
            return empty
        if "name_gis_geocodegroup" in dataset.columns:
            mask = dataset["name_gis_geocodegroup"].astype(str).str.strip() == cat
            subset = dataset.loc[mask].copy()
        else:
            subset = dataset.copy()
        self._stacked_category_cache[cache_key] = subset
        return subset

    def _build_canvas_features(self, category: Optional[str] = None) -> tuple[list[Dict[str, Any]], Optional[List[float]]]:
        """
        Prepare a lightweight GeoJSON-ready list used as a visual canvas.
        Preference order:
            1. tbl_geocode_group (already grouped polygons)
            2. tbl_flat (raw mesh, heavily down-sampled)
        """
        base = parquet_dir(self.base_dir, self.cfg)
        max_feats = 2000
        cat = DEFAULT_ANALYSIS_GEOCODE
        debug_log(self.base_dir, f"Building canvas features for geocode '{cat}'")

        def _feature_list(gdf: gpd.GeoDataFrame, label_candidates: List[str]) -> tuple[list[Dict[str, Any]], Optional[List[float]]]:
            if gdf.empty or "geometry" not in gdf.columns:
                return [], None
            gdf = gdf.dropna(subset=["geometry"]).copy()
            if gdf.empty:
                return [], None
            if gdf.crs is None:
                gdf.set_crs(epsg=self.working_epsg, inplace=True, allow_override=True)
            else:
                gdf = gdf.to_crs(self.working_epsg)
            gdf["geometry"] = gdf.geometry.buffer(0)
            gdf = gdf[~gdf.geometry.is_empty]
            if gdf.empty:
                return [], None

            if len(gdf) > max_feats:
                step = max(1, len(gdf) // max_feats)
                gdf = gdf.iloc[::step].copy()

            gdf4326 = gdf.to_crs(4326)
            label_col = next((c for c in label_candidates if c in gdf4326.columns), None)

            features: list[Dict[str, Any]] = []
            for _, row in gdf4326.iterrows():
                label = ""
                if label_col:
                    raw = row.get(label_col)
                    if raw is not None and not (isinstance(raw, str) and raw.strip() == ""):
                        label = str(raw)
                try:
                    geom_json = shp_to_geojson(row.geometry)
                except Exception:
                    continue
                features.append({"type": "Feature", "geometry": geom_json, "properties": {"label": label}})

            bounds = gdf4326.total_bounds.tolist() if not gdf4326.empty else None
            return features, bounds

        group_path = base / "tbl_geocode_group.parquet"
        if group_path.exists():
            try:
                gdf = gpd.read_parquet(group_path)
            except Exception:
                gdf = None
            if isinstance(gdf, gpd.GeoDataFrame) and not gdf.empty:
                if "name_gis_geocodegroup" in gdf.columns:
                    gdf = gdf[gdf["name_gis_geocodegroup"].astype(str) == cat]
                feats, bounds = _feature_list(gdf, ["name_gis_geocodegroup", "name", "name_gis", "title_user"])
                if feats:
                    debug_log(self.base_dir, f"Canvas built from tbl_geocode_group ({len(feats)} features)")
                    return feats, bounds

        gdf = self._flat_for_category(cat)
        if isinstance(gdf, gpd.GeoDataFrame) and not gdf.empty and "geometry" in gdf.columns:
            subset_cols = ["geometry"] + [c for c in ["name_gis_geocodegroup"] if c in gdf.columns]
            gdf_subset = gdf[subset_cols].copy()
            feats, bounds = _feature_list(gdf_subset, ["name_gis_geocodegroup"])
            if feats:
                debug_log(self.base_dir, f"Canvas built from tbl_flat ({len(feats)} features)")
                return feats, bounds

        debug_log(self.base_dir, "Canvas generation produced no features")
        return [], None

    @staticmethod
    def _subset_by_polygon(gdf: gpd.GeoDataFrame, polygon: BaseGeometry) -> gpd.GeoDataFrame:
        if gdf.empty:
            return gdf.iloc[0:0].copy()
        try:
            sindex = gdf.sindex
            indices = list(sindex.intersection(polygon.bounds))
            if indices:
                subset = gdf.iloc[indices].copy()
            else:
                subset = gdf.iloc[0:0].copy()
        except Exception:
            subset = gdf.copy()
        if subset.empty:
            return subset
        subset = subset[subset.geometry.intersects(polygon)]
        return subset.copy()

    def _clip_flat_to_polygon(
        self,
        flat_gdf: gpd.GeoDataFrame,
        polygon: BaseGeometry,
        group: AnalysisGroup,
        record: AnalysisRecord,
        geocode: str,
        run_id: str,
        run_timestamp: str,
    ) -> gpd.GeoDataFrame:
        subset = self._subset_by_polygon(flat_gdf, polygon)
        if subset.empty:
            return subset.iloc[0:0].copy()

        clipped_geoms = []
        for geom in subset.geometry:
            try:
                inter = geom.intersection(polygon)
            except Exception:
                try:
                    inter = geom.buffer(0).intersection(polygon)
                except Exception:
                    inter = None
            if inter is None or inter.is_empty:
                clipped_geoms.append(None)
            else:
                clipped_geoms.append(inter)
        subset = subset.assign(geometry=clipped_geoms)
        subset = subset[subset.geometry.notna()]
        subset = subset[~subset.geometry.is_empty]
        if subset.empty:
            return subset.iloc[0:0].copy()

        subset = gpd.GeoDataFrame(subset, geometry="geometry", crs=flat_gdf.crs).copy()
        metric = subset.to_crs(self.area_epsg)
        subset["analysis_area_m2"] = metric.geometry.area.astype("float64")
        subset = subset[subset["analysis_area_m2"] > 0]
        if subset.empty:
            return subset.iloc[0:0].copy()

        base_area = pd.to_numeric(subset.get("area_m2"), errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            subset["analysis_area_fraction"] = np.where(
                base_area > 0, subset["analysis_area_m2"] / base_area.astype("float64"), np.nan
            )
        subset["analysis_group_id"] = group.identifier
        subset["analysis_group_name"] = group.name
        subset["analysis_polygon_id"] = record.identifier
        subset["analysis_polygon_title"] = record.title
        subset["analysis_polygon_notes"] = record.notes
        subset["analysis_geocode"] = geocode
        subset["analysis_run_id"] = run_id
        subset["analysis_timestamp"] = run_timestamp
        return subset.reset_index(drop=True)

    def _clip_stacked_to_polygon(
        self,
        stacked_gdf: gpd.GeoDataFrame,
        polygon: BaseGeometry,
        group: AnalysisGroup,
        record: AnalysisRecord,
        geocode: str,
        run_id: str,
        run_timestamp: str,
    ) -> gpd.GeoDataFrame:
        if stacked_gdf.empty:
            return stacked_gdf.iloc[0:0].copy()
        subset = self._subset_by_polygon(stacked_gdf, polygon)
        if subset.empty:
            return subset.iloc[0:0].copy()

        clipped_geoms = []
        for geom in subset.geometry:
            try:
                inter = geom.intersection(polygon)
            except Exception:
                try:
                    inter = geom.buffer(0).intersection(polygon)
                except Exception:
                    inter = None
            if inter is None or inter.is_empty:
                clipped_geoms.append(None)
            else:
                clipped_geoms.append(inter)
        subset = subset.assign(geometry=clipped_geoms)
        subset = subset[subset.geometry.notna()]
        subset = subset[~subset.geometry.is_empty]
        if subset.empty:
            return subset.iloc[0:0].copy()

        subset = gpd.GeoDataFrame(subset, geometry="geometry", crs=stacked_gdf.crs).copy()
        metric = subset.to_crs(self.area_epsg)
        subset["analysis_area_m2"] = metric.geometry.area.astype("float64")
        subset = subset[subset["analysis_area_m2"] > 0]
        if subset.empty:
            return subset.iloc[0:0].copy()

        base_area = pd.to_numeric(subset.get("area_m2"), errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            subset["analysis_area_fraction"] = np.where(
                base_area > 0, subset["analysis_area_m2"] / base_area.astype("float64"), np.nan
            )
        subset["analysis_group_id"] = group.identifier
        subset["analysis_group_name"] = group.name
        subset["analysis_polygon_id"] = record.identifier
        subset["analysis_polygon_title"] = record.title
        subset["analysis_polygon_notes"] = record.notes
        subset["analysis_geocode"] = geocode
        subset["analysis_run_id"] = run_id
        subset["analysis_timestamp"] = run_timestamp
        return subset.reset_index(drop=True)

    def _write_analysis_output(self, path: Path, group_id: str, new_gdf: gpd.GeoDataFrame) -> None:
        os.makedirs(path.parent, exist_ok=True)
        if path.exists():
            try:
                existing = gpd.read_parquet(path)
            except Exception:
                existing = gpd.GeoDataFrame(columns=new_gdf.columns, geometry="geometry", crs=new_gdf.crs)
        else:
            existing = gpd.GeoDataFrame(columns=new_gdf.columns, geometry="geometry", crs=new_gdf.crs)

        if not isinstance(existing, gpd.GeoDataFrame):
            existing = gpd.GeoDataFrame(existing, geometry="geometry", crs=getattr(existing, "crs", new_gdf.crs))

        if "analysis_group_id" in existing.columns:
            mask = existing["analysis_group_id"].astype(str).fillna("") != str(group_id)
            existing = existing.loc[mask].reset_index(drop=True)
        else:
            existing = existing.iloc[0:0].copy()

        if new_gdf is None or new_gdf.empty:
            combined = existing
        elif existing.empty:
            combined = new_gdf.copy()
        else:
            combined = pd.concat([existing, new_gdf], ignore_index=True)

        if not isinstance(combined, gpd.GeoDataFrame):
            combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=new_gdf.crs)
        combined.to_parquet(path, index=False)

    def analysis_preview_geojson(self, group_id: str, limit: int = 500) -> Dict[str, Any]:
        try:
            path = self.analysis_flat_path
            if not path.exists():
                return {"type": "FeatureCollection", "features": []}
            df = gpd.read_parquet(path)
        except Exception as exc:
            debug_log(self.base_dir, f"analysis_preview_geojson: failed to read flat table ({exc})")
            return {"type": "FeatureCollection", "features": []}

        if df.empty or "analysis_group_id" not in df.columns:
            return {"type": "FeatureCollection", "features": []}

        subset = df[df["analysis_group_id"].astype(str) == str(group_id)].copy()
        if subset.empty or "geometry" not in subset.columns:
            return {"type": "FeatureCollection", "features": []}

        subset = subset.dropna(subset=["geometry"])
        subset = subset[~subset.geometry.is_empty]
        if subset.empty:
            return {"type": "FeatureCollection", "features": []}

        if limit and len(subset) > limit:
            if "analysis_timestamp" in subset.columns:
                subset = subset.sort_values("analysis_timestamp", na_position="last").tail(limit)
            else:
                subset = subset.tail(limit)

        try:
            subset = gpd.GeoDataFrame(subset, geometry="geometry", crs=subset.geometry.crs or f"EPSG:{self.storage_epsg}")
            subset = subset.to_crs(4326)
        except Exception as exc:
            debug_log(self.base_dir, f"analysis_preview_geojson: reprojection failed ({exc})")
            return {"type": "FeatureCollection", "features": []}

        features: list[Dict[str, Any]] = []
        keep_keys = [
            "analysis_polygon_id",
            "analysis_polygon_title",
            "analysis_group_id",
            "analysis_group_name",
            "analysis_geocode",
            "analysis_area_m2",
            "analysis_area_fraction",
            "sensitivity_code",
            "sensitivity_description",
            "analysis_timestamp",
        ]

        for _, row in subset.iterrows():
            try:
                geom_json = shp_to_geojson(row.geometry)
            except Exception:
                continue
            props: Dict[str, Any] = {}
            for key in keep_keys:
                if key in row and not pd.isna(row[key]):
                    value = row[key]
                    if isinstance(value, (pd.Timestamp, dt.datetime)):
                        value = value.isoformat()
                    props[key] = value
            features.append({"type": "Feature", "geometry": geom_json, "properties": props})

        return {"type": "FeatureCollection", "features": features}

    def geocode_feature_collection(self, category: Optional[str]) -> Dict[str, Any]:
        geojson = self.canvas_geojson(category)
        if not geojson:
            cat = self._ensure_category(category)
            return {"type": "FeatureCollection", "features": [], "bounds": None, "category": cat}
        return geojson

    def run_group_analysis(
        self,
        group: AnalysisGroup,
        records: List[AnalysisRecord],
        geocode: Optional[str] = None,
    ) -> Dict[str, Any]:
        debug_log(self.base_dir, f"run_group_analysis: group={group.identifier}, polygons={len(records)}, requested_geocode={geocode}")
        if not records:
            raise ValueError("Ingen analysepolygoner valgt.")
        category = DEFAULT_ANALYSIS_GEOCODE
        flat_base = self._flat_for_category(category)
        if flat_base.empty:
            raise ValueError(f"Ingen data funnet for geokode «{category}».")
        try:
            stacked_base = self._stacked_for_category(category)
        except FileNotFoundError:
            stacked_base = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=flat_base.crs)

        run_id = uuid.uuid4().hex
        run_ts = dt.datetime.utcnow().isoformat()

        flat_results: List[gpd.GeoDataFrame] = []
        stacked_results: List[gpd.GeoDataFrame] = []
        summaries: List[Dict[str, Any]] = []
        preview_geojson: Dict[str, Any] = {"type": "FeatureCollection", "features": []}

        for record in records:
            poly_storage = gpd.GeoDataFrame([{"geometry": record.geometry}], geometry="geometry", crs=f"EPSG:{self.storage_epsg}")
            polygon = poly_storage.to_crs(flat_base.crs).geometry.iloc[0]
            area_sqkm = _km2(poly_storage.to_crs(self.area_epsg).area.iloc[0])

            clipped_flat = self._clip_flat_to_polygon(flat_base, polygon, group, record, category, run_id, run_ts)
            if not clipped_flat.empty:
                flat_results.append(clipped_flat)

            clipped_stacked = self._clip_stacked_to_polygon(stacked_base, polygon, group, record, category, run_id, run_ts)
            if not clipped_stacked.empty:
                stacked_results.append(clipped_stacked)

        summaries.append(
            {
                "analysis_polygon_id": record.identifier,
                "title": record.title,
                "notes": record.notes,
                "area_sqkm": area_sqkm,
                "flat_rows": int(len(clipped_flat)),
                "stacked_rows": int(len(clipped_stacked)),
            }
        )

        try:
            preview_frames: List[gpd.GeoDataFrame] = [gdf for gdf in flat_results if not gdf.empty]
            if preview_frames:
                preview_gdf = gpd.GeoDataFrame(pd.concat(preview_frames, ignore_index=True), geometry="geometry", crs=preview_frames[0].crs)
                preview_gdf = preview_gdf[preview_gdf.geometry.notna()]
                preview_gdf = preview_gdf[~preview_gdf.geometry.is_empty]
                if not preview_gdf.empty:
                    preview_wgs = preview_gdf.to_crs(4326)
                    keep_keys = [
                        "analysis_polygon_id",
                        "analysis_polygon_title",
                        "analysis_group_id",
                        "analysis_group_name",
                        "analysis_geocode",
                        "analysis_area_m2",
                        "analysis_area_fraction",
                        "display_title",
                        "sensitivity_code",
                        "sensitivity_description",
                    ]
                    for _, row in preview_wgs.iterrows():
                        try:
                            geom_json = shp_to_geojson(row.geometry)
                        except Exception:
                            continue
                        props: Dict[str, Any] = {}
                        for key in keep_keys:
                            if key in row:
                                value = row.get(key)
                                if _is_blank(value):
                                    continue
                                if isinstance(value, (np.generic,)):
                                    value = value.item()
                                props[key] = value
                        preview_geojson["features"].append({"type": "Feature", "geometry": geom_json, "properties": props})
        except Exception as exc:
            debug_log(self.base_dir, f"run_group_analysis: preview generation failed: {exc}")
            preview_geojson = {"type": "FeatureCollection", "features": []}

        if flat_results:
            flat_gdf = gpd.GeoDataFrame(pd.concat(flat_results, ignore_index=True), geometry="geometry", crs=flat_results[0].crs)
        else:
            additional_cols = [
                "analysis_group_id",
                "analysis_group_name",
                "analysis_polygon_id",
                "analysis_polygon_title",
                "analysis_polygon_notes",
                "analysis_geocode",
                "analysis_run_id",
                "analysis_timestamp",
                "analysis_area_m2",
                "analysis_area_fraction",
            ]
            flat_gdf = flat_base.iloc[0:0].copy()
            for col in additional_cols:
                if col not in flat_gdf.columns:
                    dtype = "float64" if col in {"analysis_area_m2", "analysis_area_fraction"} else "object"
                    flat_gdf[col] = pd.Series(dtype=dtype)
            flat_gdf = gpd.GeoDataFrame(flat_gdf, geometry="geometry", crs=flat_base.crs)
        flat_gdf = flat_gdf.reset_index(drop=True)
        self._write_analysis_output(self.analysis_flat_path, group.identifier, flat_gdf)

        if stacked_results:
            stacked_gdf = gpd.GeoDataFrame(
                pd.concat(stacked_results, ignore_index=True), geometry="geometry", crs=stacked_results[0].crs
            )
        else:
            additional_cols = [
                "analysis_group_id",
                "analysis_group_name",
                "analysis_polygon_id",
                "analysis_polygon_title",
                "analysis_polygon_notes",
                "analysis_geocode",
                "analysis_run_id",
                "analysis_timestamp",
                "analysis_area_m2",
                "analysis_area_fraction",
            ]
            stacked_gdf = stacked_base.iloc[0:0].copy()
            for col in additional_cols:
                if col not in stacked_gdf.columns:
                    dtype = "float64" if col in {"analysis_area_m2", "analysis_area_fraction"} else "object"
                    stacked_gdf[col] = pd.Series(dtype=dtype)
            stacked_gdf = gpd.GeoDataFrame(stacked_gdf, geometry="geometry", crs=stacked_base.crs)
        stacked_gdf = stacked_gdf.reset_index(drop=True)
        self._write_analysis_output(self.analysis_stacked_path, group.identifier, stacked_gdf)

        debug_log(self.base_dir, f"run_group_analysis complete: geocode={category}, flat_rows={len(flat_gdf)}, stacked_rows={len(stacked_gdf)}")
        return {
            "analysis_group_id": group.identifier,
            "analysis_group_name": group.name,
            "analysis_geocode": category,
            "flat_path": str(self.analysis_flat_path),
            "stacked_path": str(self.analysis_stacked_path),
            "flat_rows": int(len(flat_gdf)),
            "stacked_rows": int(len(stacked_gdf)),
            "summary": summaries,
            "run_id": run_id,
            "run_timestamp": run_ts,
            "preview_geojson": preview_geojson,
            "analysis_preview": preview_geojson,
        }

    def _join_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.merge(self.asset_groups, left_on="ref_asset_group", right_on="id", how="left", suffixes=("", "_group"))

    def _intersections(self, polygon: BaseGeometry) -> gpd.GeoDataFrame:
        mask = self.asset_objects.geometry.intersects(polygon)
        subset = self.asset_objects.loc[mask].copy()
        if subset.empty:
            return subset
        subset["geometry"] = subset.geometry.intersection(polygon)
        subset = subset[~subset.geometry.is_empty]
        return subset

    def analyse_polygon(self, record: AnalysisRecord) -> Dict[str, Any]:
        geo_df = gpd.GeoDataFrame([{"geometry": record.geometry}], geometry="geometry", crs=self.asset_objects.crs)
        intersections = self._intersections(geo_df.geometry.iloc[0])
        if intersections.empty:
            return {
                "record": record,
                "area_sqkm": _km2(geo_df.to_crs(self.area_epsg).area.iloc[0]),
                "detail_table": pd.DataFrame(columns=["Title", "Code", "Description", "Type", "total_area", "# objects"]),
                "summary_table": pd.DataFrame(columns=["Sensitivity Code", "Sensitivity Description", "Number of Asset Objects", "Active asset groups"]),
            }

        enriched = self._join_metadata(intersections)
        enriched["geom_type"] = enriched.geometry.geom_type

        area_df = enriched.to_crs(self.area_epsg)
        enriched["area_m2"] = area_df.geometry.area.where(
            area_df.geometry.geom_type.isin(["Polygon", "MultiPolygon"]), other=pd.NA
        )

        def _display_title(row: pd.Series) -> str:
            for col in ("title_fromuser", "name_original", "name_gis_assetgroup"):
                val = row.get(col)
                if not _is_blank(val):
                    return str(val)
            return f"asset_{row['ref_asset_group']}"

        enriched["display_title"] = enriched.apply(_display_title, axis=1)

        grouped = (
            enriched.groupby(["display_title", "sensitivity_code", "sensitivity_description", "geom_type"], dropna=False)
            .agg(
                total_area_m2=pd.NamedAgg(column="area_m2", aggfunc=lambda s: float(s.dropna().sum()) if not s.dropna().empty else None),
                object_count=pd.NamedAgg(column="id", aggfunc="nunique"),
            )
            .reset_index()
        )
        grouped["total_area"] = grouped["total_area_m2"].apply(_km2)
        grouped.rename(
            columns={
                "display_title": "Title",
                "sensitivity_code": "Code",
                "sensitivity_description": "Description",
                "geom_type": "Type",
                "object_count": "# objects",
            },
            inplace=True,
        )
        grouped = grouped[["Title", "Code", "Description", "Type", "total_area", "# objects"]].sort_values(
            ["Code", "Title", "Type"]
        )

        summary = (
            enriched.groupby(["sensitivity_code", "sensitivity_description"])
            .agg(object_count=("id", "nunique"), active_groups=("ref_asset_group", "nunique"))
            .reset_index()
        )
        summary.rename(
            columns={
                "sensitivity_code": "Sensitivity Code",
                "sensitivity_description": "Sensitivity Description",
                "object_count": "Number of Asset Objects",
                "active_groups": "Active asset groups",
            },
            inplace=True,
        )

        area_sqkm = _km2(geo_df.to_crs(self.area_epsg).area.iloc[0])
        return {
            "record": record,
            "area_sqkm": area_sqkm,
            "detail_table": grouped,
            "summary_table": summary.sort_values("Sensitivity Code"),
        }

    def canvas_geojson(self, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
        cat = self._ensure_category(category)
        if self.mbtiles_base_url:
            info = _mbtiles_info(cat)
            if info and (info.get("sensitivity_url") or info.get("envindex_url")):
                bounds = info.get("bounds")
                if bounds:
                    self.canvas_bounds = bounds
                debug_log(self.base_dir, f"canvas_geojson: using MBTiles for '{cat}'")
                return {
                    "type": "FeatureCollection",
                    "features": [],
                    "bounds": bounds,
                    "category": info.get("category", cat),
                    "mbtiles": info,
                }
        if cat not in self._canvas_cache:
            features, bounds = self._build_canvas_features(cat)
            self._canvas_cache[cat] = (features, bounds)
            if bounds:
                self.canvas_bounds = bounds
        features, bounds = self._canvas_cache.get(cat, ([], None))
        if not features:
            debug_log(self.base_dir, f"canvas_geojson: no features for '{cat}'")
            return None
        debug_log(self.base_dir, f"canvas_geojson: returning {len(features)} features for '{cat}'")
        return {"type": "FeatureCollection", "features": features, "bounds": bounds, "category": cat}

    def build_report(self, analyses: Iterable[Dict[str, Any]]) -> Path:
        analyses = list(analyses)
        if not analyses:
            raise ValueError("No analyses supplied.")

        output_dir = (self.base_dir / "output").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / REPORT_FILENAME_TEMPLATE.format(ts=_timestamp_label())

        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, title="MESA Area Analysis Report")
        styles = getSampleStyleSheet()
        heading = styles["Heading1"]
        heading.fontSize = 18
        heading.leading = 22
        sub_heading = ParagraphStyle(name="SubHeading", parent=styles["Heading2"], fontSize=14, leading=18, spaceBefore=12)
        body = styles["BodyText"]
        body.spaceBefore = 6
        body.spaceAfter = 6

        elements: List[Any] = []
        for idx, bundle in enumerate(analyses, start=1):
            record = bundle["record"]
            title = record.title or record.identifier
            elements.append(Paragraph(f"{idx}. Analysis polygon: {title}", heading))
            elements.append(Spacer(1, 6))

            meta = (
                f"<b>Polygon ID:</b> {record.identifier}<br/>"
                f"<b>Created:</b> {record.created_label}<br/>"
                f"<b>Area:</b> {bundle['area_sqkm']}"
            )
            if record.notes:
                meta += f"<br/><b>Notes:</b> {record.notes}"
            elements.append(Paragraph(meta, body))

            detail_df: pd.DataFrame = bundle["detail_table"]
            if detail_df.empty:
                elements.append(Paragraph("No asset objects intersect this polygon.", body))
            else:
                elements.append(Paragraph("Intersecting asset layers", sub_heading))
                table = Table([list(detail_df.columns)] + detail_df.values.tolist(), repeatRows=1, hAlign="LEFT")
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ]
                    )
                )
                elements.append(table)

                summary_df: pd.DataFrame = bundle["summary_table"]
                if not summary_df.empty:
                    elements.append(Spacer(1, 12))
                    elements.append(Paragraph("Sensitivity summary", sub_heading))
                    summary_table = Table([list(summary_df.columns)] + summary_df.values.tolist(), repeatRows=1, hAlign="LEFT")
                    summary_table.setStyle(
                        TableStyle(
                            [
                                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#385723")),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
                                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ]
                        )
                    )
                    elements.append(summary_table)

            if idx != len(analyses):
                elements.append(PageBreak())

        doc.build(elements)
        return pdf_path


# ---------------------------------------------------------------------------
# pywebview bridge
# ---------------------------------------------------------------------------


class WebApi:
    """Expose storage and analysis primitives to the embedded Leaflet app."""

    def __init__(self, storage: AnalysisStorage, analyzer: AssetAnalyzer, base_dir: Path) -> None:
        self._storage = storage
        self._analyzer = analyzer
        self._base_dir = base_dir
        self._dirty = False
        self._lock = threading.Lock()
        self._active_group_id = self._storage.active_group_id()
        try:
            group = self._storage.get_group(self._active_group_id)
            self._active_geocode = group.default_geocode or self._analyzer.default_geocode
        except Exception:
            self._active_geocode = self._analyzer.default_geocode
        debug_log(self._base_dir, "WebApi initialised")

    # --------------------- helpers ---------------------
    def _set_dirty(self, value: bool) -> None:
        with self._lock:
            self._dirty = value

    def _is_dirty(self) -> bool:
        with self._lock:
            return self._dirty

    def _group_payload(self, group: AnalysisGroup) -> Dict[str, Any]:
        return {
            "id": group.identifier,
            "name": group.name,
            "notes": group.notes,
            "default_geocode": group.default_geocode or self._analyzer.default_geocode,
            "created_at": group.created_at.isoformat(),
            "updated_at": group.updated_at.isoformat(),
        }

    def _current_group(self) -> AnalysisGroup:
        return self._storage.get_group(self._active_group_id)

    # --------------------- exposed API -----------------
    def bootstrap(self, _payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        features = {"features": []}
        try:
            debug_log(self._base_dir, "bootstrap: start")
            group = self._current_group()
            self._active_group_id = group.identifier
            self._active_geocode = group.default_geocode or self._analyzer.default_geocode

            groups_payload = [self._group_payload(g) for g in self._storage.list_groups()]
            features = self._storage.to_feature_collection(group.identifier)
            preview = self._analyzer.analysis_preview_geojson(group.identifier)

            bounds = None
            if features["features"]:
                try:
                    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
                    bounds = gdf.total_bounds.tolist()
                except Exception:
                    bounds = None

            layer = self._analyzer.canvas_geojson(self._active_geocode)
            if layer:
                geocode_layer = layer
                layer_bounds = layer.get("bounds")
                if layer_bounds:
                    bounds = layer_bounds
            else:
                geocode_layer = {
                    "type": "FeatureCollection",
                    "features": [],
                    "bounds": None,
                    "category": self._active_geocode,
                }
                if bounds is None and self._analyzer.canvas_bounds:
                    bounds = self._analyzer.canvas_bounds
                if bounds is None:
                    try:
                        bounds = list(self._analyzer.asset_objects.to_crs(4326).total_bounds)
                    except Exception:
                        bounds = None
                if bounds is None and self._analyzer.asset_bounds:
                    bounds = list(self._analyzer.asset_bounds)

            debug_log(
                self._base_dir,
                f"bootstrap: ok (group={group.identifier}, polygons={len(features['features'])}, geocode={self._active_geocode}, bounds={bounds})",
            )
            return {
                "ok": True,
                "groups": groups_payload,
                "active_group_id": group.identifier,
                "active_group": self._group_payload(group),
                "polygons": features,
                "geocode_layer": geocode_layer,
                "analysis_preview": preview,
                "mbtiles": geocode_layer.get("mbtiles") if isinstance(geocode_layer, dict) else None,
                "geocode_categories": self._analyzer.available_geocode_categories(),
                "asset_bounds": list(self._analyzer.asset_bounds) if self._analyzer.asset_bounds else None,
                "bounds": bounds,
                "dirty": self._is_dirty(),
            }
        except Exception as exc:
            debug_log(self._base_dir, f"bootstrap: error {exc}")
            return {"ok": False, "error": str(exc)}
        finally:
            count = 0
            try:
                count = len(features.get("features", []))
            except Exception:
                count = 0
            debug_log(
                self._base_dir,
                f"bootstrap: completed (group={self._active_group_id}, polygons={count})"
            )

    def add_polygon(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            feature = payload.get("feature")
            if not feature:
                raise ValueError("feature payload missing")
            group_id = payload.get("group_id") or self._active_group_id
            debug_log(self._base_dir, f"add_polygon: group={group_id}")
            record = self._storage.add_polygon(feature, group_id)
            self._active_group_id = record.group_id

            gdf = gpd.GeoDataFrame([{"geometry": record.geometry}], geometry="geometry", crs=self._storage.storage_epsg).to_crs(4326)
            feature_out = {
                "type": "Feature",
                "geometry": shp_to_geojson(gdf.geometry.iloc[0]),
                "properties": {
                    "id": record.identifier,
                    "group_id": record.group_id,
                    "title": record.title,
                    "notes": record.notes,
                    "created_at": record.created_at.isoformat(),
                    "updated_at": record.updated_at.isoformat(),
                },
            }
            self._set_dirty(True)
            debug_log(self._base_dir, f"add_polygon: created id={record.identifier}")
            return {"ok": True, "feature": feature_out, "dirty": True}
        except Exception as exc:
            debug_log(self._base_dir, f"add_polygon: error {exc}")
            return {"ok": False, "error": str(exc)}

    def update_geometry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            identifier = payload.get("identifier")
            geometry = payload.get("geometry")
            if not identifier or geometry is None:
                raise ValueError("identifier/geometry missing")
            group_id = payload.get("group_id") or self._active_group_id
            debug_log(self._base_dir, f"update_geometry: id={identifier}, group={group_id}")
            record = self._storage.update_geometry(identifier, group_id, geometry)
            self._active_group_id = record.group_id
            self._set_dirty(True)
            debug_log(self._base_dir, f"update_geometry: updated id={identifier}")
            return {"ok": True, "record": {"updated_at": record.updated_at.isoformat()}, "dirty": True}
        except Exception as exc:
            debug_log(self._base_dir, f"update_geometry: error {exc}")
            return {"ok": False, "error": str(exc)}

    def update_properties(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            identifier = payload.get("identifier")
            if not identifier:
                raise ValueError("identifier missing")
            title = payload.get("title", "")
            notes = payload.get("notes", "")
            group_id = payload.get("group_id") or self._active_group_id
            debug_log(self._base_dir, f"update_properties: id={identifier}, group={group_id}")
            record = self._storage.update_properties(identifier, group_id, title, notes)
            self._active_group_id = record.group_id
            self._set_dirty(True)
            debug_log(self._base_dir, f"update_properties: updated id={identifier}")
            return {
                "ok": True,
                "record": {
                    "title": record.title,
                    "notes": record.notes,
                    "updated_at": record.updated_at.isoformat(),
                },
                "dirty": True,
            }
        except Exception as exc:
            debug_log(self._base_dir, f"update_properties: error {exc}")
            return {"ok": False, "error": str(exc)}

    def delete_polygon(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            identifier = payload.get("identifier")
            if not identifier:
                raise ValueError("identifier missing")
            group_id = payload.get("group_id") or self._active_group_id
            debug_log(self._base_dir, f"delete_polygon: id={identifier}, group={group_id}")
            self._storage.delete(identifier, group_id)
            self._set_dirty(True)
            debug_log(self._base_dir, f"delete_polygon: removed id={identifier}")
            return {"ok": True, "dirty": True}
        except Exception as exc:
            debug_log(self._base_dir, f"delete_polygon: error {exc}")
            return {"ok": False, "error": str(exc)}

    def save(self) -> Dict[str, Any]:
        try:
            self._storage.save()
            self._set_dirty(False)
            debug_log(self._base_dir, "save: wrote storage to parquet")
            return {"ok": True, "dirty": False}
        except Exception as exc:
            debug_log(self._base_dir, f"save: error {exc}")
            return {"ok": False, "error": str(exc)}

    def import_file(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(
                title="Import polygon dataset",
                filetypes=[
                    ("GeoJSON", "*.geojson;*.json"),
                    ("Shapefile", "*.shp"),
                    ("GeoPackage", "*.gpkg"),
                    ("Parquet", "*.parquet;*.pq"),
                    ("All files", "*.*"),
                ],
            )
            root.destroy()
            if not path:
                return {"ok": False, "error": "Import cancelled."}

            group_id = (payload or {}).get("group_id") or self._active_group_id
            debug_log(self._base_dir, f"import_file: path={path}, group={group_id}")
            imported = self._storage.import_file(Path(path), group_id)
            features = []
            for record in imported:
                gdf = gpd.GeoDataFrame([{"geometry": record.geometry}], geometry="geometry", crs=self._storage.storage_epsg).to_crs(4326)
                features.append(
                    {
                        "type": "Feature",
                        "geometry": shp_to_geojson(gdf.geometry.iloc[0]),
                        "properties": {
                            "id": record.identifier,
                            "group_id": record.group_id,
                            "title": record.title,
                            "notes": record.notes,
                            "created_at": record.created_at.isoformat(),
                            "updated_at": record.updated_at.isoformat(),
                        },
                    }
                )
            self._set_dirty(True)
            debug_log(self._base_dir, f"import_file: imported {len(features)} polygon(s)")
            return {"ok": True, "features": features, "dirty": True}
        except Exception as exc:
            debug_log(self._base_dir, f"import_file: error {exc}")
            return {"ok": False, "error": str(exc)}

    def run_analysis(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            identifiers = payload.get("identifiers") or []
            if not identifiers:
                raise ValueError("Ingen polygoner valgt.")
            group_id = payload.get("group_id") or self._active_group_id
            debug_log(self._base_dir, f"run_analysis: group={group_id}, polygons={identifiers}, geocode={payload.get('geocode')}")
            group = self._storage.get_group(group_id)
            records = self._storage.get_records(group.identifier, identifiers)
            if not records:
                return {"ok": False, "error": "Ingen polygoner funnet for valgt gruppe."}
            geocode = DEFAULT_ANALYSIS_GEOCODE
            result = self._analyzer.run_group_analysis(group, records, geocode=geocode)
            self._active_group_id = group.identifier
            self._set_dirty(False)
            debug_log(self._base_dir, f"run_analysis: completed run_id={result.get('run_id')} flat_rows={result.get('flat_rows')} stacked_rows={result.get('stacked_rows')}")
            return {"ok": True, **result}
        except Exception as exc:
            debug_log(self._base_dir, f"run_analysis: error {exc}")
            return {"ok": False, "error": str(exc)}

    def load_canvas(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            category = DEFAULT_ANALYSIS_GEOCODE
            debug_log(self._base_dir, f"load_canvas: requested category={category}")
            layer = self._analyzer.canvas_geojson(category)
            if not layer:
                debug_log(self._base_dir, f"load_canvas: no layer for {category}")
                return {"ok": False, "error": "Canvas layer unavailable."}
            self._active_geocode = layer.get("category", category)
            debug_log(self._base_dir, f"load_canvas: delivering {len(layer.get('features', []))} features")
            return {"ok": True, "geojson": layer, "mbtiles": layer.get("mbtiles"), "bounds": layer.get("bounds"), "category": self._active_geocode}
        except Exception as exc:
            debug_log(self._base_dir, f"load_canvas: error {exc}")
            return {"ok": False, "error": str(exc)}

    def list_geocodes(self) -> Dict[str, Any]:
        try:
            return {"ok": True, "geocode_categories": self._analyzer.available_geocode_categories()}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def select_group(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            group_id = payload.get("group_id")
            debug_log(self._base_dir, f"select_group: group={group_id}")
            group = self._storage.set_active_group(group_id)
            self._active_group_id = group.identifier
            if payload.get("geocode"):
                self._active_geocode = payload["geocode"]
            else:
                self._active_geocode = group.default_geocode or self._analyzer.default_geocode
            return self.bootstrap()
        except Exception as exc:
            debug_log(self._base_dir, f"select_group: error {exc}")
            return {"ok": False, "error": str(exc)}

    def create_group(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            name = payload.get("name", "Ny analysegruppe")
            notes = payload.get("notes", "")
            geocode = payload.get("default_geocode")
            debug_log(self._base_dir, f"create_group: name={name}, geocode={geocode}")
            group = self._storage.add_group(name, notes, geocode)
            self._active_group_id = group.identifier
            self._active_geocode = group.default_geocode or self._analyzer.default_geocode
            self._set_dirty(True)
            return self.bootstrap()
        except Exception as exc:
            debug_log(self._base_dir, f"create_group: error {exc}")
            return {"ok": False, "error": str(exc)}

    def update_group(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            group_id = payload.get("group_id") or self._active_group_id
            name = payload.get("name")
            notes = payload.get("notes")
            geocode = payload.get("default_geocode")
            debug_log(self._base_dir, f"update_group: group={group_id}, name={name}, geocode={geocode}")
            group = self._storage.update_group(group_id, name=name, notes=notes, default_geocode=geocode)
            self._active_group_id = group.identifier
            if geocode:
                self._active_geocode = geocode
            else:
                self._active_geocode = group.default_geocode or self._analyzer.default_geocode
            self._set_dirty(True)
            return self.bootstrap()
        except Exception as exc:
            debug_log(self._base_dir, f"update_group: error {exc}")
            return {"ok": False, "error": str(exc)}

    def delete_group(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            group_id = payload.get("group_id")
            debug_log(self._base_dir, f"delete_group: group={group_id}")
            self._storage.delete_group(group_id)
            self._active_group_id = self._storage.active_group_id()
            self._active_geocode = self._storage.get_group(self._active_group_id).default_geocode or self._analyzer.default_geocode
            self._set_dirty(True)
            return self.bootstrap()
        except Exception as exc:
            debug_log(self._base_dir, f"delete_group: error {exc}")
            return {"ok": False, "error": str(exc)}

    def exit_app(self) -> None:
        threading.Timer(0.05, webview.destroy_window).start()


# HTML payload inserted in the pywebview window (truncated for brevity in code review).
HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MESA Area Analysis</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css" />
  <script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>
  <style>
    html, body { height:100%; margin:0; }
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; background:#f8fafc; color:#1e293b; }
    .wrap { height:100vh; display:grid; grid-template-columns: 400px 1fr; grid-template-rows: 100%; grid-template-areas: "panel map"; }
    .panel { grid-area: panel; border-right:2px solid #1f2937; padding:12px 14px; display:flex; flex-direction:column; gap:12px; overflow:auto; background:#fff; }
    .panel h1 { font-size:20px; margin:0; }
    .panel h2 { font-size:14px; margin:8px 0 4px; text-transform:uppercase; letter-spacing:0.03em; color:#334155; }
    .panel .buttons { display:flex; gap:8px; flex-wrap:wrap; }
    button { padding:6px 10px; border:1px solid #cbd5f5; border-radius:6px; background:#fff; cursor:pointer; }
    button.primary { background:#2563eb; color:#fff; border-color:#1d4ed8; }
    button:disabled { opacity:0.45; cursor:not-allowed; }
    #polygonList { width:100%; min-height:160px; border:1px solid #cbd5f5; border-radius:6px; padding:6px; overflow:auto; }
    #polygonList li { list-style:none; border-bottom:1px solid #e2e8f0; padding:6px 4px; display:flex; gap:6px; align-items:center; }
    #polygonList li:last-child { border-bottom:none; }
    #polygonList .title { font-weight:600; }
    #polygonList .meta { font-size:11px; color:#64748b; }
    .form-group { display:flex; flex-direction:column; gap:4px; }
    .form-group input, .form-group textarea { border:1px solid #cbd5f5; border-radius:6px; padding:6px 8px; font-size:13px; width:100%; }
    .form-group textarea { min-height:60px; resize:vertical; }
    .status { font-size:12px; color:#334155; }
    .status.error { color:#b91c1c; }
    .map { grid-area: map; position:relative; }
    #map { position:absolute; inset:0; }
    .grid-label { font-size:11px; font-weight:500; color:#1f2937; background:rgba(255,255,255,0.85); padding:2px 4px; border-radius:4px; }
    .background-toggle { font-size:12px; color:#475569; display:flex; align-items:center; gap:6px; margin:4px 0 8px; }
    .section-divider { border:none; border-top:1px solid #e2e8f0; margin:4px 0; }
  </style>
</head>
<body>
<div class="wrap">
  <div class="panel">
    <div>
      <h1>Area analysis</h1>
      <p class="status" id="statusText">Initialising...</p>
    </div>

    <div>
      <h2>Background</h2>
      <label class="background-toggle">
        <input type="checkbox" id="backgroundToggle" checked>
        Show background
      </label>
    </div>

    <hr class="section-divider">

    <div>
      <h2>Basemap</h2>
      <div class="form-group">
        <label for="basemapSelect">Basemap layer</label>
        <select id="basemapSelect">
          <option value="osm">OpenStreetMap</option>
          <option value="topo">OpenTopoMap</option>
          <option value="satellite">Satellite (Esri)</option>
        </select>
      </div>
    </div>

    <hr class="section-divider">

    <div>
      <h2>Analysis groups</h2>
      <div class="form-group">
        <label for="groupSelect">Group</label>
        <select id="groupSelect"></select>
      </div>
      <div class="buttons">
        <button id="newGroupBtn" title="Create a new analysis group">New group</button>
        <button id="renameGroupBtn" title="Rename selected group" disabled>Rename</button>
        <button id="deleteGroupBtn" title="Delete selected group" disabled>Delete</button>
      </div>
    </div>

    <hr class="section-divider">

    <div>
      <h2>Polygons</h2>
      <div class="buttons" style="margin-bottom:6px;">
        <button id="newBtn" class="primary" title="Draw a new polygon">New polygon</button>
        <button id="editBtn" title="Edit selected polygon" disabled>Edit</button>
        <button id="deleteBtn" title="Delete selected polygon" disabled>Delete</button>
        <button id="importBtn" title="Import polygons">Import&hellip;</button>
        <button id="saveBtn" title="Persist changes to disk" disabled>Save</button>
      </div>
      <ul id="polygonList"></ul>
    </div>

    <hr class="section-divider">

    <div>
      <h2>Details</h2>
      <div class="form-group">
        <label for="titleInput">Title</label>
        <input id="titleInput" type="text" placeholder="Area of interest">
      </div>
      <div class="form-group">
        <label for="notesInput">Notes</label>
        <textarea id="notesInput" placeholder="Optional notes or rationale"></textarea>
      </div>
      <button id="applyMetaBtn" title="Update title and notes" disabled>Update details</button>
    </div>

    <hr class="section-divider">

    <div>
      <h2>Analysis</h2>
      <p style="font-size:12px; color:#64748b;">
        Select one or more polygons to calculate results for <code>tbl_analysis_flat.parquet</code> and <code>tbl_analysis_stacked.parquet</code> in <code>output/geoparquet</code>.
      </p>
      <button id="analyseBtn" class="primary" disabled>Run analysis</button>
      <div id="analysisSummary" class="status"></div>
    </div>
  </div>

  <div class="map"><div id="map"></div></div>
</div>

<script>
let MAP = null;
let DRAW_CONTROL = null;
let EDIT_LAYER = null;
let GRID_LAYER = null;
let MBTILES_LAYER = null;
let ANALYSIS_LAYER = null;
let BASE_LAYER = null;
let POLYGON_SOURCE = {};
let SELECTED_IDS = new Set();
let DRAW_MODE = false;
let GROUPS = [];
let ACTIVE_GROUP_ID = null;
let GEO_CATEGORIES = [];
let ACTIVE_GEOCODE = null;
let LAST_ANALYSIS = null;
let HOME_BOUNDS = null;
let CURRENT_CANVAS_LAYER = null;
let SHOW_BACKGROUND = true;
let ANALYSIS_PREVIEW = { type:'FeatureCollection', features:[] };
const BASEMAPS = {
  osm: {
    url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    options: { attribution: '&copy; OpenStreetMap contributors' }
  },
  topo: {
    url: 'https://tile.opentopomap.org/{z}/{x}/{y}.png',
    options: { attribution: '&copy; OpenTopoMap (CC-BY-SA)' }
  },
  satellite: {
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    options: { attribution: 'Tiles &copy; Esri' }
  }
};
const statusText = (msg, cls='') => {
  const el = document.getElementById('statusText');
  el.textContent = msg || '';
  el.className = 'status ' + cls;
};

function notifyError(msg){ statusText(msg, 'error'); }

function setDirtyState(dirty){
  document.getElementById('saveBtn').disabled = !dirty;
}

const BACKEND_RETRY_MS = 250;

function getBackendApi(){
  if(!window.pywebview || !window.pywebview.api){
    return null;
  }
  const candidate = window.pywebview.api;
  if(typeof candidate === 'object' && typeof candidate.then === 'function'){
    return candidate;
  }
  return Promise.resolve(candidate);
}

function toLeafletBounds(bounds){
  if(!bounds){ return null; }
  if(Array.isArray(bounds) && bounds.length === 4){
    const [minx, miny, maxx, maxy] = bounds.map(Number);
    if([minx, miny, maxx, maxy].every(Number.isFinite)){
      return L.latLngBounds([[miny, minx], [maxy, maxx]]);
    }
  }
  if(Array.isArray(bounds) && bounds.length === 2 && Array.isArray(bounds[0]) && Array.isArray(bounds[1])){
    const [swLat, swLng] = bounds[0].map(Number);
    const [neLat, neLng] = bounds[1].map(Number);
    if([swLat, swLng, neLat, neLng].every(Number.isFinite)){
      return L.latLngBounds([[swLat, swLng], [neLat, neLng]]);
    }
  }
  return null;
}

function setHomeBounds(bounds){
  const leafletBounds = toLeafletBounds(bounds);
  if(leafletBounds){
    HOME_BOUNDS = leafletBounds;
  }
}

function fitHomeBounds(){
  if(MAP && HOME_BOUNDS){
    MAP.fitBounds(HOME_BOUNDS, {padding:[40,40]});
  }
}

function clearMbtilesLayer(){
  if(MAP && MBTILES_LAYER){
    MAP.removeLayer(MBTILES_LAYER);
  }
  MBTILES_LAYER = null;
}

function clearGridLayer(){
  if(MAP && GRID_LAYER){
    MAP.removeLayer(GRID_LAYER);
  }
  GRID_LAYER = null;
}

function clearAnalysisLayer(){
  if(MAP && ANALYSIS_LAYER){
    MAP.removeLayer(ANALYSIS_LAYER);
  }
  ANALYSIS_LAYER = null;
}

function analysisFeatureStyle(feature){
  const geomType = feature?.geometry?.type || '';
  const base = { color:'#f97316', weight:2, fillColor:'#fb923c', fillOpacity:0.35 };
  if(geomType.includes('Line')){
    return { color:'#f97316', weight:3 };
  }
  if(geomType.includes('Point')){
    return { radius:6, color:'#f97316', fillColor:'#fde68a', fillOpacity:0.9 };
  }
  return base;
}

function renderAnalysisLayer(){
  clearAnalysisLayer();
  if(!MAP){ return; }
  let features = Array.isArray(ANALYSIS_PREVIEW?.features) ? ANALYSIS_PREVIEW.features : [];
  if(SELECTED_IDS.size > 0){
    const selected = new Set(Array.from(SELECTED_IDS, id => String(id)));
    features = features.filter(f => selected.has(String(f?.properties?.analysis_polygon_id ?? '')));
  } else {
    features = [];
  }
  if(!features.length){ return; }
  const collection = { type:'FeatureCollection', features };
  ANALYSIS_LAYER = L.geoJSON(collection, {
    style: analysisFeatureStyle,
    pointToLayer: (feature, latlng) => L.circleMarker(latlng, analysisFeatureStyle(feature)),
    onEachFeature: (feature, layer) => {
      const props = feature?.properties || {};
      const name = props.analysis_polygon_title || props.display_title || props.analysis_polygon_id || '';
      const code = props.sensitivity_code || '';
      const area = props.analysis_area_m2 != null ? ((props.analysis_area_m2 / 1e6).toFixed(2) + ' km^2') : '';
      const lines = [name, code, area].filter(Boolean);
      if(lines.length){
        layer.bindTooltip(lines.join('<br>'), {direction:'top'});
      }
    }
  });
  ANALYSIS_LAYER.addTo(MAP);
}

function setAnalysisPreview(collection){
  if(collection && Array.isArray(collection.features)){
    ANALYSIS_PREVIEW = collection;
  } else {
    ANALYSIS_PREVIEW = { type:'FeatureCollection', features:[] };
  }
  renderAnalysisLayer();
}

function setMbtilesLayer(info){
  if(!MAP){ return; }
  clearMbtilesLayer();
  if(!info){ return; }
  const url = info.sensitivity_url || info.envindex_url || info.groupstotal_url || info.assetstotal_url;
  if(!url){ return; }
  const bounds = toLeafletBounds(info.bounds);
  if(bounds){
    HOME_BOUNDS = bounds;
  }
  const opts = {
    opacity: 1.0,
    crossOrigin: true,
    noWrap: true,
    minNativeZoom: info.minzoom ?? 0,
    maxNativeZoom: info.maxzoom ?? 19
  };
  if(bounds){ opts.bounds = bounds; }
  MBTILES_LAYER = L.tileLayer(url, opts);
  MBTILES_LAYER.addTo(MAP);
}

function applyCanvasLayer(layer){
  if(!layer){ return; }
  CURRENT_CANVAS_LAYER = layer;
  if(layer.bounds){
    setHomeBounds(layer.bounds);
  }
  if(!SHOW_BACKGROUND){
    clearMbtilesLayer();
    clearGridLayer();
    fitHomeBounds();
    return;
  }
  if(layer.mbtiles){
    setMbtilesLayer(layer.mbtiles);
  } else {
    clearMbtilesLayer();
  }
  if(layer.features && Array.isArray(layer.features) && layer.features.length){
    addGridLayer(layer.features);
  } else if(!layer.mbtiles){
    clearGridLayer();
  }
  fitHomeBounds();
}

function addGridLayer(features){
  if(!MAP){ return; }
  clearGridLayer();
  if(!features || !features.length){
    return;
  }
  const collection = { type: 'FeatureCollection', features };
  GRID_LAYER = L.geoJSON(collection, {
    style: () => ({
      color: '#1e3a8a',
      weight: 1,
      fillColor: '#3b82f6',
      fillOpacity: 0.08
    }),
    onEachFeature: (feature, layer) => {
      const label = feature?.properties?.label;
      if(label){
        layer.bindTooltip(String(label), {permanent:false, direction:'center', className:'grid-label'});
      }
    }
  });
  GRID_LAYER.addTo(MAP);
}

function setBasemap(name){
  if(!MAP){ return; }
  const key = BASEMAPS[name] ? name : 'osm';
  const spec = BASEMAPS[key];
  if(BASE_LAYER){
    MAP.removeLayer(BASE_LAYER);
    BASE_LAYER = null;
  }
  BASE_LAYER = L.tileLayer(spec.url, Object.assign({ maxZoom: 19 }, spec.options || {}));
  BASE_LAYER.addTo(MAP);
}

function polygonStyle(selected){
  return selected ? {
    color: '#d97706',
    weight: 3,
    fillColor: '#f59e0b',
    fillOpacity: 0.2
  } : {
    color: '#0f172a',
    weight: 2,
    fillColor: '#2563eb',
    fillOpacity: 0.08
  };
}

function removePolygonLayer(id){
  const existing = POLYGON_SOURCE[id];
  if(existing && EDIT_LAYER){
    EDIT_LAYER.removeLayer(existing);
  }
  delete POLYGON_SOURCE[id];
}

function setPolygonSelected(id, selected){
  const layer = POLYGON_SOURCE[id];
  if(layer && layer.setStyle){
    layer.setStyle(polygonStyle(selected));
    if(selected && layer.bringToFront){
      layer.bringToFront();
    }
  }
}

function syncSelectionHighlights(){
  const validIds = new Set(Object.keys(POLYGON_SOURCE));
  Array.from(SELECTED_IDS).forEach(id => {
    if(!validIds.has(id)){
      SELECTED_IDS.delete(id);
    }
  });

  Object.keys(POLYGON_SOURCE).forEach(id => {
    setPolygonSelected(id, SELECTED_IDS.has(id));
  });

  const list = document.getElementById('polygonList');
  if(list){
    list.querySelectorAll('input[type=\"checkbox\"]').forEach(input => {
      const id = input.dataset.id;
      input.checked = SELECTED_IDS.has(id);
    });
  }

  if(SELECTED_IDS.size === 1){
    const [onlyId] = Array.from(SELECTED_IDS);
    populateForm(onlyId);
  } else {
    populateForm(null);
  }
  updateActionButtons();
  renderAnalysisLayer();
}

function addFeatureToMap(feature){
  if(!feature || !feature.geometry){ return; }
  ensureMap();
  const id = feature?.properties?.id;
  if(!id){ return; }

  removePolygonLayer(id);
  const geoLayer = L.geoJSON(feature, {
    style: () => polygonStyle(SELECTED_IDS.has(id))
  });

  let addedLayer = null;
  geoLayer.eachLayer(layer => {
    addedLayer = layer;
  });

  if(!addedLayer){ return; }

  addedLayer.feature = feature;
  addedLayer.on('click', (evt) => {
    if(!evt?.originalEvent?.ctrlKey && !evt?.originalEvent?.metaKey){
      SELECTED_IDS.clear();
    }
    if(SELECTED_IDS.has(id)){
      SELECTED_IDS.delete(id);
    } else {
      SELECTED_IDS.add(id);
    }
    syncSelectionHighlights();
  });

  if(EDIT_LAYER){
    EDIT_LAYER.addLayer(addedLayer);
  }
  POLYGON_SOURCE[id] = addedLayer;
  if(SELECTED_IDS.has(id)){
    setPolygonSelected(id, true);
  }
}

function reloadListFromMap(){
  const records = Object.values(POLYGON_SOURCE).map(layer => {
    const props = layer?.feature?.properties || {};
    return {
      id: props.id || '',
      title: props.title || '',
      notes: props.notes || '',
      created_at: props.created_at || '',
      updated_at: props.updated_at || ''
    };
  }).filter(rec => rec.id);

  records.sort((a, b) => {
    const labelA = (a.title || a.id).toLowerCase();
    const labelB = (b.title || b.id).toLowerCase();
    if(labelA < labelB){ return -1; }
    if(labelA > labelB){ return 1; }
    return 0;
  });

  refreshList(records);
  syncSelectionHighlights();
}

function updateGroupSelect(){
  const select = document.getElementById('groupSelect');
  const renameBtn = document.getElementById('renameGroupBtn');
  const deleteBtn = document.getElementById('deleteGroupBtn');
  if(!select){ return; }

  select.innerHTML = '';
  if(!Array.isArray(GROUPS) || GROUPS.length === 0){
    select.disabled = true;
    if(renameBtn){ renameBtn.disabled = true; }
    if(deleteBtn){ deleteBtn.disabled = true; }
    return;
  }

  select.disabled = false;
  GROUPS.forEach(group => {
    const option = document.createElement('option');
    option.value = group.id;
    option.textContent = group.name || group.id;
    option.dataset.notes = group.notes || '';
    select.appendChild(option);
  });

  if(!ACTIVE_GROUP_ID || !GROUPS.some(g => g.id === ACTIVE_GROUP_ID)){
    ACTIVE_GROUP_ID = GROUPS[0].id;
  }
  select.value = ACTIVE_GROUP_ID || '';

  if(renameBtn){ renameBtn.disabled = !ACTIVE_GROUP_ID; }
  if(deleteBtn){ deleteBtn.disabled = GROUPS.length <= 1; }
}

function updateGeocodeSelect(){
  const select = document.getElementById('geocodeSelect');
  if(!select){ return; }
  select.innerHTML = '';
  GEO_CATEGORIES.forEach(cat => {
    const option = document.createElement('option');
    option.value = cat;
    option.textContent = cat;
    select.appendChild(option);
  });
  if(!ACTIVE_GEOCODE && GEO_CATEGORIES.length){
    ACTIVE_GEOCODE = GEO_CATEGORIES.includes('basic_mosaic') ? 'basic_mosaic' : GEO_CATEGORIES[0];
  }
  if(ACTIVE_GEOCODE){
    select.value = ACTIVE_GEOCODE;
  }
  select.disabled = GEO_CATEGORIES.length <= 1;
  const refreshBtn = document.getElementById('refreshGeocodeBtn');
  if(refreshBtn){
    refreshBtn.disabled = GEO_CATEGORIES.length <= 1;
  }
}

function clearPolygons(){
  Object.values(POLYGON_SOURCE).forEach(layer => {
    if(EDIT_LAYER && layer){
      EDIT_LAYER.removeLayer(layer);
    }
  });
  POLYGON_SOURCE = {};
  SELECTED_IDS.clear();
  updateActionButtons();
  populateForm(null);
  renderAnalysisLayer();
}

function loadPolygons(collection){
  clearPolygons();
  if(collection?.features){
    collection.features.forEach(addFeatureToMap);
  }
  reloadListFromMap();
}

function applyState(state){
  if(state.groups){ GROUPS = state.groups; }
  if(state.active_group_id){ ACTIVE_GROUP_ID = state.active_group_id; }
  if(Array.isArray(state.geocode_categories)){ GEO_CATEGORIES = state.geocode_categories; }
  if(state.geocode_layer?.category){ ACTIVE_GEOCODE = state.geocode_layer.category; }

  ensureMap();
  updateGroupSelect();
  updateGeocodeSelect();
  const bgToggle = document.getElementById('backgroundToggle');
  if(bgToggle){ bgToggle.checked = SHOW_BACKGROUND; }
  const basemapSelectEl = document.getElementById('basemapSelect');
  if(basemapSelectEl && BASE_LAYER){
    const current = Object.entries(BASEMAPS).find(([key, spec]) => spec.url === BASE_LAYER._url);
    if(current){ basemapSelectEl.value = current[0]; }
  }

  if(Object.prototype.hasOwnProperty.call(state, 'analysis_preview')){
    setAnalysisPreview(state.analysis_preview);
  } else {
    renderAnalysisLayer();
  }

  if(state.polygons){
    loadPolygons(state.polygons);
  }

  if(state.geocode_layer){
    applyCanvasLayer(state.geocode_layer);
  } else if(state.bounds){
    setHomeBounds(state.bounds);
    fitHomeBounds();
  }

  if(!state.summary){
    const summaryEl = document.getElementById('analysisSummary');
    if(summaryEl){ summaryEl.textContent = ''; }
  }

  if(GROUPS.length){
    fitHomeBounds();
    statusText('Ready.');
  } else {
    statusText('Ready - create a group to begin.');
  }
}

function handleAnalysisResult(result){
  LAST_ANALYSIS = result;
  const summaryEl = document.getElementById('analysisSummary');
  if(!summaryEl){ return; }
  if(!result){
    summaryEl.textContent = '';
    return;
  }

  const flatInfo = result.flat_rows != null ? `${result.flat_rows} flat rows` : 'flat rows unknown';
  const stackedInfo = result.stacked_rows != null ? `${result.stacked_rows} stacked rows` : 'stacked rows unknown';
  let html = `<strong>Analysis completed.</strong> Geocode: ${result.analysis_geocode || '-'}<br>`;
  if(result.flat_path){ html += `<span>Flat table: <code>${result.flat_path}</code></span><br>`; }
  if(result.stacked_path){ html += `<span>Stacked table: <code>${result.stacked_path}</code></span><br>`; }
  html += `<span>${flatInfo}, ${stackedInfo}</span>`;

  if(Array.isArray(result.summary) && result.summary.length){
    const items = result.summary.map(item => {
      const title = item.title || item.analysis_polygon_id;
      const area = item.area_sqkm || '-';
      const flatRows = item.flat_rows != null ? item.flat_rows : 0;
      const stackedRows = item.stacked_rows != null ? item.stacked_rows : 0;
      return `<li>${title}: area ${area}, flat ${flatRows}, stacked ${stackedRows}</li>`;
    }).join('');
    html += `<ul>${items}</ul>`;
  }
  summaryEl.innerHTML = html;
}

function updateActionButtons(){
  const hasSelection = SELECTED_IDS.size > 0;
  if(DRAW_MODE){
    ['editBtn', 'deleteBtn', 'applyMetaBtn', 'analyseBtn'].forEach(id => {
      const btn = document.getElementById(id);
      if(btn){ btn.disabled = true; }
    });
    return;
  }
  document.getElementById('editBtn').disabled = SELECTED_IDS.size !== 1;
  document.getElementById('deleteBtn').disabled = !hasSelection;
  document.getElementById('applyMetaBtn').disabled = SELECTED_IDS.size !== 1;
  document.getElementById('analyseBtn').disabled = !hasSelection;
}

function listItemHtml(rec){
  return `
    <label style="display:flex; gap:6px; align-items:flex-start;">
      <input type="checkbox" data-id="${rec.id}">
      <div>
        <div class="title">${rec.title || rec.id}</div>
        <div class="meta">ID: ${rec.id}&nbsp;&mdash;&nbsp;Created: ${rec.created_at?.substring(0,16) || ''}</div>
        <div class="meta">${rec.notes ? rec.notes : ''}</div>
      </div>
    </label>
  `;
}

function refreshList(records){
  const list = document.getElementById('polygonList');
  list.innerHTML = '';
  records.forEach(rec => {
    const li = document.createElement('li');
    li.innerHTML = listItemHtml(rec);
    const checkbox = li.querySelector('input[type="checkbox"]');
    checkbox.checked = SELECTED_IDS.has(rec.id);
    checkbox.addEventListener('change', (evt) => {
      if(evt.target.checked){
        SELECTED_IDS.add(rec.id);
      } else {
        SELECTED_IDS.delete(rec.id);
      }
      setPolygonSelected(rec.id, evt.target.checked);
      if(SELECTED_IDS.size === 1){
        const [onlyId] = Array.from(SELECTED_IDS);
        populateForm(onlyId);
      } else if(SELECTED_IDS.size === 0){
        populateForm(null);
      }
      updateActionButtons();
    });
    li.addEventListener('dblclick', () => flyToPolygon(rec.id));
    list.appendChild(li);
  });
  updateActionButtons();
}

function flyToPolygon(id){
  const layer = POLYGON_SOURCE[id];
  if(layer){
    MAP.fitBounds(layer.getBounds(), {padding:[30,30]});
  }
}

function populateForm(id){
  if(!id){ document.getElementById('titleInput').value=''; document.getElementById('notesInput').value=''; return; }
  const layer = POLYGON_SOURCE[id];
  if(layer){
    const props = layer.feature?.properties || {};
    document.getElementById('titleInput').value = props.title || '';
    document.getElementById('notesInput').value = props.notes || '';
  }
}

function ensureMap(){
  if(MAP){ return; }
  MAP = L.map('map', {zoomControl: true});

  const basemapSelect = document.getElementById('basemapSelect');
  const initialBasemap = basemapSelect ? basemapSelect.value : 'osm';
  setBasemap(initialBasemap);

  EDIT_LAYER = L.featureGroup().addTo(MAP);

  DRAW_CONTROL = new L.Control.Draw({
    draw: {
      polyline: false,
      rectangle: false,
      circle: false,
      circlemarker: false,
      marker: false,
      polygon: {
        allowIntersection: false,
        showArea: true
      }
    },
    edit: {
      featureGroup: EDIT_LAYER,
      remove: false
    }
  });
  MAP.addControl(DRAW_CONTROL);

  MAP.on(L.Draw.Event.CREATED, (evt) => {
    if(!DRAW_MODE){ return; }
    DRAW_MODE = false;
    toggleDrawButtons(false);
    const layer = evt.layer;
    const geojson = layer.toGeoJSON();
    layer.remove();
    callPython('add_polygon', {group_id: ACTIVE_GROUP_ID, feature: geojson});
  });

  MAP.on(L.Draw.Event.EDITED, (evt) => {
    evt.layers.eachLayer(layer => {
      const id = layer?.feature?.properties?.id;
      if(!id) return;
      const groupId = layer?.feature?.properties?.group_id || ACTIVE_GROUP_ID;
      callPython('update_geometry', {identifier: id, group_id: groupId, geometry: layer.toGeoJSON().geometry}, (result) => {
        if(result?.record?.updated_at){
          layer.feature.properties.updated_at = result.record.updated_at;
        }
      });
    });
    reloadListFromMap();
  });

  MAP.on(L.Draw.Event.EDITSTART, () => {
    statusText('Editing mode: adjust vertices then click the save icon in the map toolbar.');
  });

  MAP.on(L.Draw.Event.EDITSTOP, () => {
    statusText(GROUPS.length ? 'Ready.' : 'Ready - create a group to begin.');
  });

  MAP.on(L.Draw.Event.DRAWSTOP, () => {
    if(DRAW_MODE){
      toggleDrawButtons(false);
    }
  });

  MAP.setView([0,0], 2);
}

function initialiseMap(bounds){
  ensureMap();
  if(bounds){
    setHomeBounds(bounds);
  }
  fitHomeBounds();
}

function toggleDrawButtons(enable){
  ensureMap();
  const enteringDraw = !!enable;
  DRAW_MODE = enteringDraw;

  const newBtn = document.getElementById('newBtn');
  if(newBtn){
    if(enteringDraw){
      newBtn.textContent = 'Cancel drawing';
      newBtn.classList.remove('primary');
    } else {
      newBtn.textContent = 'New polygon';
      if(!newBtn.classList.contains('primary')){
        newBtn.classList.add('primary');
      }
    }
  }

  const lockDuringDraw = ['editBtn', 'deleteBtn', 'analyseBtn', 'importBtn', 'applyMetaBtn'];
  lockDuringDraw.forEach(id => {
    const btn = document.getElementById(id);
    if(!btn){ return; }
    if(enteringDraw){
      btn.dataset.prevDisabled = btn.disabled ? '1' : '0';
      btn.disabled = true;
    } else if(Object.prototype.hasOwnProperty.call(btn.dataset, 'prevDisabled')){
      btn.disabled = btn.dataset.prevDisabled === '1';
      delete btn.dataset.prevDisabled;
    }
  });

  const drawToolbar = (DRAW_CONTROL && DRAW_CONTROL._toolbars && DRAW_CONTROL._toolbars.draw) ? DRAW_CONTROL._toolbars.draw : null;
  const polygonHandler = drawToolbar && drawToolbar._modes && drawToolbar._modes.polygon && drawToolbar._modes.polygon.handler;
  if(polygonHandler){
    if(enteringDraw){
      polygonHandler.enable();
    } else {
      polygonHandler.disable();
    }
  }

  if(enteringDraw){
    statusText('Drawing mode: click on the map to sketch the polygon.');
  } else {
    statusText(GROUPS.length ? 'Ready.' : 'Ready - create a group to begin.');
    updateActionButtons();
  }
}

async function callPython(method, payload, handler){
  const scheduleRetry = (delay) => new Promise(resolve => {
    window.setTimeout(() => {
      callPython(method, payload, handler).then(resolve);
    }, delay);
  });

  const apiCandidate = getBackendApi();
  if(!apiCandidate){
    statusText('Waiting for backend...');
    return scheduleRetry(BACKEND_RETRY_MS);
  }

  let api;
  try {
    api = await Promise.resolve(apiCandidate);
  } catch(err){
    notifyError(err);
    return scheduleRetry(500);
  }

  if(!api){
    statusText('Waiting for backend...');
    return scheduleRetry(BACKEND_RETRY_MS);
  }

  const fn = api[method];
  if(typeof fn !== 'function'){
    statusText(`Backend method '${method}' unavailable...`);
    return scheduleRetry(400);
  }

  statusText('Working...');
  try {
    const args = (payload === undefined) ? [] : [payload];
    const result = await fn.apply(api, args);
    if(!result?.ok){
      if(result?.error){ notifyError(result.error); }
      return result;
    }

    if(result.dirty !== undefined){ setDirtyState(result.dirty); }
    if(result.groups || result.polygons || result.geocode_layer){
      applyState(result);
    }
    if(result.feature){
      addFeatureToMap(result.feature);
      const newId = result.feature?.properties?.id;
      if(newId){
        SELECTED_IDS.add(newId);
      }
      reloadListFromMap();
    }
    if(result.features){
      result.features.forEach(addFeatureToMap);
      reloadListFromMap();
    }
    if(result.geojson){
      applyCanvasLayer(result.geojson);
    }
    if(typeof result.category === 'string'){
      ACTIVE_GEOCODE = result.category;
      updateGeocodeSelect();
    }
    if(typeof result.analysis_geocode === 'string'){
      ACTIVE_GEOCODE = result.analysis_geocode;
      updateGeocodeSelect();
    }
    if(Object.prototype.hasOwnProperty.call(result, 'preview_geojson')){
      setAnalysisPreview(result.preview_geojson);
    }

    if(handler){
      handler(result);
    }

    if(method === 'bootstrap'){
      statusText(GROUPS.length ? 'Ready.' : 'Ready - create a group to begin.');
    } else {
      statusText('Ready.');
    }
    return result;
  } catch(err){
    notifyError(err);
    return scheduleRetry(500);
  }
}

const newPolygonBtn = document.getElementById('newBtn');
if(newPolygonBtn){
  newPolygonBtn.addEventListener('click', () => {
    if(!GROUPS.length){
      notifyError('Create or select an analysis group before drawing.');
      return;
    }
    if(DRAW_MODE){
      toggleDrawButtons(false);
    } else {
      toggleDrawButtons(true);
    }
  });
}

const editPolygonBtn = document.getElementById('editBtn');
if(editPolygonBtn){
  editPolygonBtn.addEventListener('click', () => {
    if(SELECTED_IDS.size !== 1){ return; }
    ensureMap();
    const editToolbar = (DRAW_CONTROL && DRAW_CONTROL._toolbars && DRAW_CONTROL._toolbars.edit) ? DRAW_CONTROL._toolbars.edit : null;
    const editHandler = editToolbar && editToolbar._modes && editToolbar._modes.edit && editToolbar._modes.edit.handler;
    if(editHandler){
      editHandler.enable();
    }
  });
}

const deletePolygonBtn = document.getElementById('deleteBtn');
if(deletePolygonBtn){
  deletePolygonBtn.addEventListener('click', () => {
    if(SELECTED_IDS.size === 0){ return; }
    if(!confirm('Delete selected polygon(s)?')){ return; }
    const ids = Array.from(SELECTED_IDS);
    const processNext = () => {
      if(ids.length === 0){
        reloadListFromMap();
        return;
      }
      const id = ids.shift();
      callPython('delete_polygon', {identifier: id, group_id: ACTIVE_GROUP_ID}, () => {
        removePolygonLayer(id);
        SELECTED_IDS.delete(id);
        processNext();
      });
    };
    processNext();
  });
}

const importBtn = document.getElementById('importBtn');
if(importBtn){
  importBtn.addEventListener('click', () => {
    if(!GROUPS.length){
      notifyError('Create or select an analysis group before importing polygons.');
      return;
    }
    callPython('import_file', {group_id: ACTIVE_GROUP_ID});
  });
}

const saveBtn = document.getElementById('saveBtn');
if(saveBtn){
  saveBtn.addEventListener('click', () => {
    callPython('save');
  });
}

document.getElementById('analyseBtn').addEventListener('click', () => {
  if(SELECTED_IDS.size === 0){ return; }
  const geocode = document.getElementById('geocodeSelect') ? document.getElementById('geocodeSelect').value : ACTIVE_GEOCODE;
  callPython('run_analysis', {group_id: ACTIVE_GROUP_ID, geocode, identifiers: Array.from(SELECTED_IDS)}, handleAnalysisResult);
});
document.getElementById('applyMetaBtn').addEventListener('click', () => {
  if(SELECTED_IDS.size !== 1) return;
  const id = [...SELECTED_IDS][0];
  const title = document.getElementById('titleInput').value.trim();
  const notes = document.getElementById('notesInput').value.trim();
  callPython('update_properties', {identifier: id, group_id: ACTIVE_GROUP_ID, title, notes}, (result) => {
    if(result?.record){
      const layer = POLYGON_SOURCE[id];
      if(layer){
        layer.feature.properties.title = result.record.title;
        layer.feature.properties.notes = result.record.notes;
        layer.feature.properties.updated_at = result.record.updated_at;
      }
    }
    populateForm(id);
    reloadListFromMap();
  });
});

const groupSelectEl = document.getElementById('groupSelect');
if(groupSelectEl){
  groupSelectEl.addEventListener('change', () => {
    const value = groupSelectEl.value;
    if(!value){ return; }
    ACTIVE_GROUP_ID = value;
    const geocodeSelect = document.getElementById('geocodeSelect');
    const geocode = geocodeSelect ? geocodeSelect.value : ACTIVE_GEOCODE;
    callPython('select_group', {group_id: value, geocode});
  });
}

const newGroupBtn = document.getElementById('newGroupBtn');
if(newGroupBtn){
  newGroupBtn.addEventListener('click', () => {
    const suggested = `Group ${GROUPS.length + 1}`;
    const name = prompt('Group name', suggested);
    if(name === null){ return; }
    const trimmed = name.trim();
    if(!trimmed){ return; }
    const geocodeSelect = document.getElementById('geocodeSelect');
    callPython('create_group', {name: trimmed, default_geocode: geocodeSelect ? geocodeSelect.value : ACTIVE_GEOCODE});
  });
}

const renameGroupBtn = document.getElementById('renameGroupBtn');
if(renameGroupBtn){
  renameGroupBtn.addEventListener('click', () => {
    if(!ACTIVE_GROUP_ID){ return; }
    const current = GROUPS.find(g => g.id === ACTIVE_GROUP_ID);
    const name = prompt('Rename group', current?.name || '');
    if(name === null){ return; }
    const trimmed = name.trim();
    if(!trimmed){ return; }
    callPython('update_group', {group_id: ACTIVE_GROUP_ID, name: trimmed});
  });
}

const deleteGroupBtn = document.getElementById('deleteGroupBtn');
if(deleteGroupBtn){
  deleteGroupBtn.addEventListener('click', () => {
    if(!ACTIVE_GROUP_ID){ return; }
    if(!confirm('Delete current group and its polygons?')){ return; }
    callPython('delete_group', {group_id: ACTIVE_GROUP_ID});
  });
}

const geocodeSelectEl = document.getElementById('geocodeSelect');
if(geocodeSelectEl){
  geocodeSelectEl.addEventListener('change', () => {
    ACTIVE_GEOCODE = geocodeSelectEl.value;
    loadCanvasLayer(ACTIVE_GEOCODE);
  });
}

const refreshGeocodeBtn = document.getElementById('refreshGeocodeBtn');
if(refreshGeocodeBtn){
  refreshGeocodeBtn.addEventListener('click', () => {
    loadCanvasLayer(ACTIVE_GEOCODE);
  });
}

const backgroundToggle = document.getElementById('backgroundToggle');
if(backgroundToggle){
  SHOW_BACKGROUND = backgroundToggle.checked;
  backgroundToggle.addEventListener('change', () => {
    SHOW_BACKGROUND = backgroundToggle.checked;
    if(!SHOW_BACKGROUND){
      clearMbtilesLayer();
      clearGridLayer();
    } else if(CURRENT_CANVAS_LAYER){
      applyCanvasLayer(CURRENT_CANVAS_LAYER);
    }
  });
}

const basemapSelect = document.getElementById('basemapSelect');
if(basemapSelect){
  basemapSelect.addEventListener('change', () => {
    ensureMap();
    setBasemap(basemapSelect.value);
  });
}

function startBootstrap(){
  const attempt = () => {
    const apiPromise = getBackendApi();
    if(!apiPromise){
      statusText('Waiting for backend...');
      window.setTimeout(attempt, BACKEND_RETRY_MS);
      return;
    }
    Promise.resolve(apiPromise)
      .then(api => {
        const fn = api && api.bootstrap;
        if(typeof fn !== 'function'){
          statusText("Backend method 'bootstrap' unavailable...");
          window.setTimeout(attempt, 400);
          return;
        }
        ensureMap();
        statusText('Initialising...');
        return Promise.resolve(fn.call(api))
          .then(result => {
            if(result?.ok){
              applyState(result);
            } else if(result?.error){
              notifyError(result.error);
              window.setTimeout(attempt, 500);
            } else {
              statusText('Unexpected bootstrap response.');
              window.setTimeout(attempt, 500);
            }
          })
          .catch(err => {
            notifyError(err);
            window.setTimeout(attempt, 500);
          });
      })
      .catch(err => {
        notifyError(err);
        window.setTimeout(attempt, 500);
      });
  };
  attempt();
}

function loadCanvasLayer(category){
  const target = category || ACTIVE_GEOCODE;
  if(!target){
    statusText(GROUPS.length ? 'Ready.' : 'Ready - create a group to begin.');
    return;
  }
  statusText('Loading background...');
  callPython('load_canvas', {category: target}, (res) => {
    if(res?.geojson){
      applyCanvasLayer(res.geojson);
    } else if(res?.mbtiles){
      setMbtilesLayer(res.mbtiles);
    }
    fitHomeBounds();
    statusText('Ready.');
  });
}

window.addEventListener('pywebviewready', startBootstrap);
startBootstrap();
(function waitForLeaflet(){
  if(typeof L !== 'undefined'){
    ensureMap();
  } else {
    setTimeout(waitForLeaflet, 100);
  }
})();
</script>
</body>
</html>"""


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MESA area analysis tool")
    parser.add_argument("--original_working_directory", dest="owd", help="Path to the running folder (mesa desktop passes this)")
    return parser.parse_args(args)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    base_dir = resolve_base_dir(args.owd)
    cfg = read_config(base_dir)

    storage = AnalysisStorage(base_dir, cfg)
    analyzer = AssetAnalyzer(base_dir, cfg, storage.storage_epsg)
    api = WebApi(storage, analyzer, base_dir)

    window = webview.create_window(
        title="MESA Area Analysis",
        html=HTML_TEMPLATE,
        js_api=api,
        width=1200,
        height=760,
        resizable=True,
    )
    webview.start(debug=False)


if __name__ == "__main__":
    main()

