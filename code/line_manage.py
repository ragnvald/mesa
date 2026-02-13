# -*- coding: utf-8 -*-
"""
line_manage.py — Leaflet line editor (selective edit & delete) with Save/Discard + dropdown picker

Updates in this Parquet-only, flat-layout version:
- No GeoPackage reads/writes or references.
- Config loaded from <base>/config.ini (flat next to mesa.py).
- Parquet folder defaults to output/geoparquet, overridable via config.ini: DEFAULT.parquet_folder.
- Starts with an empty, typed GeoDataFrame if tbl_lines.parquet does not exist.
- Keeps the UI/behavior: Save all, Save changes, Apply/Cancel geometry, Delete selected line.
"""

import os, sys, uuid, threading, locale, configparser, argparse, warnings, time
from pathlib import Path
from typing import Any, Dict, Optional

# --- pywebview config (quiet, Edge runtime preferred) ---
os.environ['PYWEBVIEW_GUI'] = 'edgechromium'
os.environ.setdefault('PYWEBVIEW_LOG', 'error')
os.environ.setdefault('WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS', '--disable-logging --log-level=3')

try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except Exception:
    pass

webview = None


def _require_webview():
    global webview
    if webview is not None:
        return webview
    try:
        import webview as _wv

        try:
            _wv.logger.disabled = True
        except Exception:
            pass
        webview = _wv
        return _wv
    except ModuleNotFoundError as exc:
        sys.stderr.write("ERROR: 'pywebview' not installed. pip install pywebview\n")
        raise SystemExit(1) from exc

import pandas as pd
import geopandas as gpd
from shapely.geometry import shape as shp_from_geojson, mapping as shp_to_geojson
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely import wkb

try:
  import fiona
except Exception:
  fiona = None

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- Path & config helpers ----------------
def resolve_base_dir(passed: Optional[str]) -> str:
    """
    Resolve the canonical Mesa base folder in dev and packaged modes.
    Priority:
      1) --original_working_directory argument
      2) current working directory and its parents
      3) executable folder when frozen
      4) script folder parents
    """
    candidates: list[str] = []
    if passed:
        candidates.append(passed)

    cwd = os.getcwd()
    candidates.extend([cwd, os.path.join(cwd, "code")])

    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        candidates.extend([
            exe_dir,
            os.path.join(exe_dir, "code"),
            os.path.abspath(os.path.join(exe_dir, "..")),
        ])

    here = os.path.abspath(os.path.dirname(__file__))
    candidates.extend([
        os.path.abspath(os.path.join(here, "..")),
        os.path.abspath(os.path.join(here, "..", "..")),
    ])

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.basename(candidate).lower() == "system":
            candidate = os.path.abspath(os.path.join(candidate, ".."))
        cfg = os.path.join(candidate, "config.ini")
        legacy_cfg = os.path.join(candidate, "system", "config.ini")
        if os.path.isfile(cfg) or os.path.isfile(legacy_cfg):
            return os.path.abspath(candidate)

    return os.path.abspath(os.path.join(here, ".."))

def read_config(path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    try:
        cfg.read(path, encoding="utf-8")
    except Exception:
        pass
    if "DEFAULT" not in cfg:
        cfg["DEFAULT"] = {}
    return cfg

# These are set after loading config in main()
_PARQUET_SUBDIR = "output/geoparquet"

def _parquet_dir_candidates(base_dir: str) -> list[str]:
    base = os.path.abspath(base_dir or os.getcwd())
    sub = _PARQUET_SUBDIR.strip("\\/")
    candidates: list[str] = []
    first = os.path.join(base, sub)
    candidates.append(first)
    if os.path.basename(base).lower() == "code":
        parent = os.path.abspath(os.path.join(base, ".."))
        candidates.append(os.path.join(parent, sub))
    else:
        candidates.append(os.path.join(base, "code", sub))
    seen = set()
    ordered: list[str] = []
    for cand in candidates:
        norm = os.path.abspath(cand)
        if norm not in seen:
            seen.add(norm)
            ordered.append(norm)
    return ordered

def gpq_dir(base_dir: str) -> str:
    for cand in _parquet_dir_candidates(base_dir):
        try:
            os.makedirs(cand, exist_ok=True)
        except Exception:
            continue
        return cand
    # fallback to first candidate
    first = _parquet_dir_candidates(base_dir)[0]
    os.makedirs(first, exist_ok=True)
    return first

def lines_parquet_path(base_dir: str) -> str:
    for cand in _parquet_dir_candidates(base_dir):
        path = os.path.join(cand, "tbl_lines.parquet")
        if os.path.exists(path):
            return path
    primary = os.path.join(gpq_dir(base_dir), "tbl_lines.parquet")
    return primary

def lines_original_parquet_path(base_dir: str) -> str:
    for cand in _parquet_dir_candidates(base_dir):
        path = os.path.join(cand, "tbl_lines_original.parquet")
        if os.path.exists(path):
            return path
    primary = os.path.join(gpq_dir(base_dir), "tbl_lines_original.parquet")
    return primary

def asset_group_parquet_path(base_dir: str) -> str:
    for cand in _parquet_dir_candidates(base_dir):
        path = os.path.join(cand, "tbl_asset_group.parquet")
        if os.path.exists(path):
            return path
    primary = os.path.join(gpq_dir(base_dir), "tbl_asset_group.parquet")
    return primary

def config_path(base_dir: str) -> str:
    # FLAT layout: config.ini sits next to mesa.py
    return os.path.join(base_dir, "config.ini")

# ---------------- Data IO ----------------
REQUIRED_COLUMNS = ["name_gis","name_user","segment_length","segment_width","description"]

def _ensure_schema_types(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    df = gdf.copy()
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    for col in ["name_gis","name_user","description"]:
        df[col] = df[col].astype("string").fillna("")
    for col in ["segment_length","segment_width"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    # ensure geometry column exists
    if "geometry" not in df.columns:
        df = gpd.GeoDataFrame(df, geometry="geometry", crs=df.crs)
    return df

def _geometry_valid(gdf: gpd.GeoDataFrame) -> bool:
    return isinstance(gdf, gpd.GeoDataFrame) and "geometry" in gdf.columns and getattr(gdf, "geometry", None) is not None

def _from_wkb_df(pdf: pd.DataFrame) -> Optional[gpd.GeoDataFrame]:
    cand_names = [c for c in pdf.columns if "geom" in c.lower() or "geo" in c.lower() or c.lower()=="geometry"]
    for name in cand_names:
        s = pdf[name].dropna()
        if s.empty:
            continue
        v = s.iloc[0]
        if isinstance(v, (bytes, bytearray, memoryview)):
            try:
                geom = pdf[name].apply(lambda b: wkb.loads(bytes(b)) if isinstance(b, (bytes, bytearray, memoryview)) else None)
                gdf = gpd.GeoDataFrame(pdf.drop(columns=[name]), geometry=geom, crs=None)
                return gdf
            except Exception:
                continue
    return None

def load_lines_gdf_any(pq: str, default_epsg: int) -> gpd.GeoDataFrame:
    # Start empty if file doesn't exist
    if not os.path.exists(pq):
        gdf_empty = gpd.GeoDataFrame(
            {c: pd.Series(dtype="object") for c in REQUIRED_COLUMNS},
            geometry=pd.Series(dtype="object"),
            crs=f"EPSG:{default_epsg}"
        )
        return _ensure_schema_types(gdf_empty)

    # Try GeoParquet w/ geometry
    try:
        gdf = gpd.read_parquet(pq)
        if not _geometry_valid(gdf):
            raise ValueError("GeoParquet loaded but geometry column invalid")
    except Exception:
        # Fallback to plain parquet with WKB geometry
        pdf = pd.read_parquet(pq)
        gdf = _from_wkb_df(pdf)
        if gdf is None:
            raise

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=default_epsg, allow_override=True)
    return _ensure_schema_types(gdf)

def save_lines_gdf(pq: str, gdf: gpd.GeoDataFrame) -> bool:
    try:
        gdf = _ensure_schema_types(gdf)
        os.makedirs(os.path.dirname(pq), exist_ok=True)
        gdf.to_parquet(pq, index=False)
        return True
    except Exception as e:
        print("Save error:", e, file=sys.stderr)
        return False

# ---------------- CRS helpers ----------------
def to_epsg4326(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        g = gdf.set_crs(4326, allow_override=True)
    else:
        g = gdf.to_crs(4326) if str(gdf.crs).upper() != "EPSG:4326" else gdf
    return g

def from_epsg4326_to_storage(geom: BaseGeometry, storage_epsg: int) -> BaseGeometry:
    if storage_epsg == 4326:
        return geom
    try:
        gg = gpd.GeoDataFrame(geometry=[geom], crs=4326).to_crs(storage_epsg)
        return gg.geometry.iloc[0]
    except Exception:
        return geom

# ---------------- Parse helpers ----------------
def _parse_int_or_na(text: str):
    s = (text or "").strip()
    if s == "":
        return pd.NA
    v = pd.to_numeric(s, errors="raise")
    if pd.isna(v):
        return pd.NA
    if float(v) < 0:
        raise ValueError("Negative values are not allowed.")
    return int(round(float(v)))

# ---------------- Import lines from vector files (moved from data_import.py) ----------------
def _scan_for_files(label: str, folder: Path, patterns: tuple[str, ...]) -> list[Path]:
    if not folder.exists():
        return []
    t0 = time.time()
    files: list[Path] = []
    for pat in patterns:
        files.extend(folder.rglob(pat))
    files = sorted(set(files))
    _ = label  # reserved for future diagnostics
    _ = time.time() - t0
    return files


def _read_and_reproject_vector(filepath: Path, layer: str | None, working_epsg: int) -> gpd.GeoDataFrame:
    try:
        data = gpd.read_file(filepath, layer=layer) if layer else gpd.read_file(filepath)
        if data.crs is None:
            data = data.set_crs(epsg=working_epsg)
        elif (data.crs.to_epsg() or working_epsg) != int(working_epsg):
            data = data.to_crs(epsg=int(working_epsg))
        if data.geometry.name != "geometry":
            data = data.set_geometry(data.geometry.name).rename_geometry("geometry")
        return data
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{working_epsg}")


def _read_parquet_vector(fp: Path, working_epsg: int) -> gpd.GeoDataFrame:
    try:
        gdf = gpd.read_parquet(fp)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=working_epsg)
        elif (gdf.crs.to_epsg() or working_epsg) != int(working_epsg):
            gdf = gdf.to_crs(epsg=int(working_epsg))
        if gdf.geometry.name != "geometry":
            gdf = gdf.set_geometry(gdf.geometry.name).rename_geometry("geometry")
        return gdf
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{working_epsg}")


def _finalize_lines(
    lines_original: gpd.GeoDataFrame,
    lines: gpd.GeoDataFrame,
    working_epsg: int,
    segment_width: int,
    segment_length: int,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if lines_original.crs is None:
        lines_original = lines_original.set_crs(epsg=working_epsg)
    if lines.crs is None:
        lines = lines.set_crs(epsg=working_epsg)

    if not lines.empty:
        lines["segment_length"] = int(segment_length)
        lines["segment_width"] = int(segment_width)
        if "attributes" not in lines.columns:
            lines["attributes"] = ""
        lines["description"] = lines["name_user"].astype(str) + " + " + lines["attributes"].astype(str)

    if not lines_original.empty:
        keep0 = [c for c in ["name_gis", "name_user", "attributes", "length_m", "geometry"] if c in lines_original.columns]
        lines_original = lines_original[keep0]
    if not lines.empty:
        keep1 = [c for c in ["name_gis", "name_user", "segment_length", "segment_width", "description", "length_m", "geometry"] if c in lines.columns]
        lines = lines[keep1]
    return lines_original, lines


def import_lines_from_folder(
    input_folder: Path,
    working_epsg: int,
    segment_width: int,
    segment_length: int,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    line_objects: list[dict] = []
    line_id = 1
    patterns = ("*.shp", "*.gpkg", "*.parquet")
    files = _scan_for_files("Lines", input_folder, patterns)

    def _process_layer(gdf: gpd.GeoDataFrame, layer_name: str) -> None:
        nonlocal line_id
        if gdf.empty:
            return
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=working_epsg)
        try:
            metric = gdf.to_crs(3395)
        except Exception:
            metric = gdf
        for idx, row in gdf.iterrows():
            try:
                length_m = int(metric.loc[idx].geometry.length) if metric is not None else 0
            except Exception:
                length_m = 0
            attrs = "; ".join([f"{c}: {row[c]}" for c in gdf.columns if c != gdf.geometry.name and c != "geometry"])
            line_objects.append(
                {
                    "name_gis": int(line_id),
                    "name_user": layer_name,
                    "attributes": attrs,
                    "length_m": length_m,
                    "geometry": row.geometry,
                }
            )
            line_id += 1

    for fp in files:
        if fp.suffix.lower() == ".gpkg":
            if fiona is None:
                continue
            try:
                for layer in fiona.listlayers(fp):
                    gdf = _read_and_reproject_vector(fp, layer, working_epsg)
                    _process_layer(gdf, layer)
            except Exception:
                continue
        else:
            layer = fp.stem
            gdf = (
                _read_parquet_vector(fp, working_epsg)
                if fp.suffix.lower() == ".parquet"
                else _read_and_reproject_vector(fp, None, working_epsg)
            )
            _process_layer(gdf, layer)

    crs = f"EPSG:{working_epsg}"
    lines_original = (
        gpd.GeoDataFrame(pd.DataFrame(line_objects), geometry="geometry", crs=crs)
        if line_objects
        else gpd.GeoDataFrame(geometry=[], crs=crs)
    )

    lines = lines_original.copy()
    if not lines.empty:
        lines["name_gis"] = lines["name_gis"].apply(lambda x: f"line_{int(x):03d}")
        lines["name_user"] = lines["name_gis"]
        lines["segment_length"] = int(segment_length)
        lines["segment_width"] = int(segment_width)
        lines["description"] = lines["name_user"].astype(str) + " + " + lines.get("attributes", "").astype(str)

    lines_original, lines = _finalize_lines(lines_original, lines, working_epsg, segment_width, segment_length)
    return lines_original, lines

# ---------------- API ----------------
class Api:
    __slots__ = (
        "_base_dir",
        "_pq_path",
    "_pq_orig_path",
        "_asset_pq_path",
        "_lock",
        "_gdf",
        "_storage_epsg",
        "_default_epsg",
        "_asset_home_bounds",
    "_input_folder_lines",
    "_segment_width",
    "_segment_length",
    )

    def __init__(self, base_dir: str, _cfg: configparser.ConfigParser):
        self._base_dir = base_dir
        self._pq_path = lines_parquet_path(base_dir)
        self._pq_orig_path = lines_original_parquet_path(base_dir)
        self._asset_pq_path = asset_group_parquet_path(base_dir)
        self._lock = threading.Lock()

        try:
            default_epsg = int(str(_cfg["DEFAULT"].get("workingprojection_epsg", "4326")))
        except Exception:
            default_epsg = 4326

        self._default_epsg = default_epsg
        self._asset_home_bounds = None

        try:
            inp = str(_cfg["DEFAULT"].get("input_folder_lines", "input/lines"))
        except Exception:
            inp = "input/lines"
        p = Path(inp)
        if not p.is_absolute():
            p = (Path(base_dir) / p).resolve()
        self._input_folder_lines = p

        try:
            self._segment_width = int(str(_cfg["DEFAULT"].get("segment_width", "600")))
        except Exception:
            self._segment_width = 600
        try:
            self._segment_length = int(str(_cfg["DEFAULT"].get("segment_length", "1000")))
        except Exception:
            self._segment_length = 1000

        self._gdf = load_lines_gdf_any(self._pq_path, default_epsg)
        try:
            self._storage_epsg = int(str(self._gdf.crs).split(":")[-1]) if "EPSG" in str(self._gdf.crs).upper() else default_epsg
        except Exception:
            self._storage_epsg = default_epsg

    def _home_bounds(self):
        g = to_epsg4326(self._gdf)
        g = g[g.geometry.notna()]
        try:
            mask_empty = g.geometry.is_empty
            mask_empty = mask_empty.fillna(True) if hasattr(mask_empty, "fillna") else mask_empty
            g = g[~mask_empty]
        except Exception:
            pass
        if g.empty:
            return self._asset_group_home_bounds()
        try:
            minx, miny, maxx, maxy = [float(x) for x in g.total_bounds]
        except Exception:
            u = unary_union([geom for geom in g.geometry.values if geom and not geom.is_empty])
            minx, miny, maxx, maxy = [float(x) for x in u.bounds]
        dx, dy = maxx-minx, maxy-miny
        if dx <= 0 or dy <= 0:
            pad = 0.1
            minx -= pad; maxx += pad; miny -= pad; maxy += pad
        else:
            minx -= dx*0.1; maxx += dx*0.1; miny -= dy*0.1; maxy += dy*0.1
        minx = max(-180.0, minx); maxx = min(180.0, maxx)
        miny = max(-85.0,  miny); maxy = min(85.0,  maxy)
        return [[miny, minx],[maxy, maxx]]

    def _asset_group_home_bounds(self):
        if self._asset_home_bounds is not None:
            return self._asset_home_bounds
        try:
            if not os.path.exists(self._asset_pq_path):
                self._asset_home_bounds = None
                return None

            ag = gpd.read_parquet(self._asset_pq_path)
            if not _geometry_valid(ag):
                self._asset_home_bounds = None
                return None

            if ag.crs is None:
                ag = ag.set_crs(epsg=self._default_epsg, allow_override=True)
            ag = to_epsg4326(ag)
            ag = ag[ag.geometry.notna()]
            try:
                mask_empty = ag.geometry.is_empty
                mask_empty = mask_empty.fillna(True) if hasattr(mask_empty, "fillna") else mask_empty
                ag = ag[~mask_empty]
            except Exception:
                pass
            if ag.empty:
                self._asset_home_bounds = None
                return None

            try:
                minx, miny, maxx, maxy = [float(x) for x in ag.total_bounds]
            except Exception:
                u = unary_union([geom for geom in ag.geometry.values if geom and not geom.is_empty])
                minx, miny, maxx, maxy = [float(x) for x in u.bounds]

            dx, dy = maxx - minx, maxy - miny
            if dx <= 0 or dy <= 0:
                pad = 0.1
                minx -= pad; maxx += pad; miny -= pad; maxy += pad
            else:
                minx -= dx * 0.05; maxx += dx * 0.05; miny -= dy * 0.05; maxy += dy * 0.05
            minx = max(-180.0, minx); maxx = min(180.0, maxx)
            miny = max(-85.0,  miny); maxy = min(85.0,  maxy)
            self._asset_home_bounds = [[miny, minx], [maxy, maxx]]
            return self._asset_home_bounds
        except Exception:
            self._asset_home_bounds = None
            return None

    def _gdf_to_geojson_ll(self) -> Dict[str,Any]:
        g = to_epsg4326(self._gdf)
        feats = []
        for _, row in g.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            props = {
                "name_gis": str(row.get("name_gis","") or ""),
                "name_user": str(row.get("name_user","") or ""),
                "segment_length": (None if pd.isna(row.get("segment_length", pd.NA)) else int(row.get("segment_length"))),
                "segment_width":  (None if pd.isna(row.get("segment_width",  pd.NA)) else int(row.get("segment_width"))),
                "description": str(row.get("description","") or ""),
            }
            feats.append({"type":"Feature","geometry": shp_to_geojson(geom),"properties": props})
        return {"type":"FeatureCollection","features": feats}

    def commit(self):
        with self._lock:
            ok = save_lines_gdf(self._pq_path, self._gdf)
            return {"ok": bool(ok), "path": self._pq_path, "rows": int(len(self._gdf))}

    def discard(self):
        with self._lock:
            try:
                # Reload from disk (preserve EPSG if possible)
                if os.path.exists(self._pq_path):
                    fresh = load_lines_gdf_any(self._pq_path, self._storage_epsg or 4326)
                else:
                    fresh = gpd.GeoDataFrame(columns=REQUIRED_COLUMNS, geometry="geometry", crs=f"EPSG:{self._storage_epsg or 4326}")
                self._gdf = _ensure_schema_types(fresh)
                try:
                    self._storage_epsg = int(str(self._gdf.crs).split(":")[-1]) if "EPSG" in str(self._gdf.crs).upper() else (self._storage_epsg or 4326)
                except Exception:
                    pass
                return {"ok": True, "rows": int(len(self._gdf))}
            except Exception as e:
                return {"ok": False, "error": str(e)}

    def get_state(self):
        with self._lock:
            try:
                recs = [{"name_gis": str(r.get("name_gis","") or ""), "name_user": str(r.get("name_user","") or "")}
                        for _, r in self._gdf[["name_gis","name_user"]].iterrows()]
                diag = {
                    "rows": int(len(self._gdf)),
                    "crs": str(self._gdf.crs),
                    "path": self._pq_path
                }
                return {"ok": True, "records": recs, "diag": diag}
            except Exception as e:
                return {"ok": False, "error": str(e)}

    def get_all_lines(self):
        with self._lock:
            try:
                gj = self._gdf_to_geojson_ll()
                home = self._home_bounds()
                return {"ok": True, "geojson": gj, "home_bounds": home, "count": len(gj["features"])}
            except Exception as e:
                return {"ok": False, "error": str(e)}

    def get_defaults(self):
        with self._lock:
            try:
                return {
                    "ok": True,
                    "segment_length": int(self._segment_length),
                    "segment_width": int(self._segment_width),
                }
            except Exception as e:
                return {"ok": False, "error": str(e)}

    def import_lines_from_input(self):
        with self._lock:
            try:
                lines_original, lines = import_lines_from_folder(
                    self._input_folder_lines,
                    self._default_epsg,
                    self._segment_width,
                    self._segment_length,
                )
                out_dir = gpq_dir(self._base_dir)
                os.makedirs(out_dir, exist_ok=True)
                try:
                    lines_original.to_parquet(os.path.join(out_dir, "tbl_lines_original.parquet"), index=False)
                except Exception:
                    pass

                lines = _ensure_schema_types(lines)
                lines.to_parquet(os.path.join(out_dir, "tbl_lines.parquet"), index=False)

                # Reload editor state from the freshly written table
                self._pq_path = os.path.join(out_dir, "tbl_lines.parquet")
                self._pq_orig_path = os.path.join(out_dir, "tbl_lines_original.parquet")
                self._gdf = load_lines_gdf_any(self._pq_path, self._default_epsg)
                self._asset_home_bounds = None
                return {
                    "ok": True,
                    "rows": int(len(self._gdf)),
                    "input": str(self._input_folder_lines),
                    "out": out_dir,
                }
            except Exception as e:
                return {"ok": False, "error": str(e)}

    def update_attribute(self, name_gis: str, field: str, value: Any):
        with self._lock:
            try:
                if field not in {"name_user","segment_length","segment_width","description"}:
                    return {"ok": False, "error": f"Field '{field}' is not editable."}
                sel = self._gdf["name_gis"].astype(str).str.strip() == str(name_gis).strip()
                if not sel.any():
                    return {"ok": False, "error": f"Line '{name_gis}' not found."}
                if field in ("segment_length","segment_width"):
                    try:
                        parsed = _parse_int_or_na("" if value is None else str(value))
                    except ValueError as e:
                        return {"ok": False, "error": f"Invalid number: {e}"}
                    self._gdf.loc[sel, field] = parsed
                else:
                    self._gdf.loc[sel, field] = ("" if value is None else str(value)).strip()
                self._gdf = _ensure_schema_types(self._gdf)
                return {"ok": True, "staged": True}
            except Exception as e:
                return {"ok": False, "error": str(e)}

    def add_line(self, feature: Dict[str,Any]):
        with self._lock:
            try:
                geom_ll = shp_from_geojson(feature.get("geometry", None))
                if geom_ll is None:
                    return {"ok": False, "error": "Missing geometry."}
                geom_s = from_epsg4326_to_storage(geom_ll, self._storage_epsg or 4326)
                new_id = f"line_{uuid.uuid4().hex[:8]}"
                while (self._gdf["name_gis"].astype(str) == new_id).any():
                    new_id = f"line_{uuid.uuid4().hex[:8]}"
                row = {
                    "name_gis": new_id, "name_user":"",
                    "segment_length": pd.NA, "segment_width": pd.NA,
                    "description":"", "geometry": geom_s
                }
                self._gdf = pd.concat([self._gdf, gpd.GeoDataFrame([row], geometry="geometry", crs=self._gdf.crs)], ignore_index=True)

                g = gpd.GeoDataFrame([row], geometry="geometry", crs=self._gdf.crs)
                g_ll = to_epsg4326(g)
                feat = {
                    "type":"Feature",
                    "geometry": shp_to_geojson(g_ll.geometry.iloc[0]),
                    "properties": {"name_gis": new_id, "name_user":"", "segment_length": None, "segment_width": None, "description": ""}
                }
                return {"ok": True, "feature": feat, "record": {"name_gis": new_id, "name_user": ""}, "staged": True}
            except Exception as e:
                return {"ok": False, "error": str(e)}

    def update_geometry(self, name_gis: str, geometry: Dict[str,Any]):
        with self._lock:
            try:
                geom_ll = shp_from_geojson(geometry)
                if geom_ll is None:
                    return {"ok": False, "error": "Invalid geometry."}
                geom_s = from_epsg4326_to_storage(geom_ll, self._storage_epsg or 4326)
                sel = self._gdf["name_gis"].astype(str).str.strip() == str(name_gis).strip()
                if not sel.any():
                    return {"ok": False, "error": f"Line '{name_gis}' not found."}
                self._gdf.loc[sel, "geometry"] = [geom_s] * sel.sum()
                return {"ok": True, "staged": True}
            except Exception as e:
                return {"ok": False, "error": str(e)}

    def delete_line(self, name_gis: str):
        with self._lock:
            try:
                before = len(self._gdf)
                self._gdf = self._gdf[self._gdf["name_gis"].astype(str).str.strip() != str(name_gis).strip()].reset_index(drop=True)
                if len(self._gdf) == before:
                    return {"ok": False, "error": f"Line '{name_gis}' not found."}
                return {"ok": True, "staged": True}
            except Exception as e:
                return {"ok": False, "error": str(e)}

    def exit_app(self):
        try:
            threading.Timer(0.05, _require_webview().destroy_window).start()
        except Exception:
            os._exit(0)

# ---------------- HTML/JS (unchanged behavior, minor wording tweaks) ----------------
HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Edit Line (GeoParquet)</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css" />
<script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>
<style>
  html, body { height:100%; margin:0; }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
  .wrap { height:100vh; display:grid; grid-template-columns: 1fr 2fr; grid-template-rows: 56px 1fr 26px; grid-template-areas: "bar bar" "form map" "foot foot"; }
  .bar { grid-area: bar; display:flex; gap:8px; align-items:center; padding:8px 12px; border-bottom:2px solid #2b3442; }
  .form { grid-area: form; border-right:2px solid #2b3442; padding:10px 12px; overflow:auto; font-size:14px; }
  .map { grid-area: map; position:relative; }
  #map { position:absolute; inset:0; }
  .foot { grid-area: foot; font-size:12px; color:#475569; padding:4px 10px; border-top:1px solid #e2e8f0; display:flex; justify-content:space-between; align-items:center; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .btn { padding:6px 10px; border:1px solid #ccd; background:#fff; border-radius:6px; cursor:pointer; }
  .btn:disabled { opacity:0.5; cursor:not-allowed; }
  .btn:active { transform:translateY(1px); }
  .row { margin-bottom:10px; }
  label { display:block; font-weight:600; margin-bottom:4px; font-size:13px; }
  input[type=text], input[type=number], textarea, select { width:100%; box-sizing:border-box; padding:8px; border:1px solid #cbd5e1; border-radius:6px; font-size:14px; }
  input[readonly] { background:#f8fafc; color:#64748b; }
  textarea { resize: vertical; min-height: 64px; }
  #status { min-height: 18px; margin-top:4px; }
  .status-ok { color:#166534; }
  .status-warn { color:#b45309; }
  .status-err { color:#b91c1c; }
  .hint { font-size:12px; color:#475569; margin-left:8px; }
  .toolbar { display:flex; gap:6px; margin:6px 0 10px; align-items:center; flex-wrap: wrap; }
  .kbd { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:11px; border:1px solid #cbd5e1; padding:1px 4px; border-radius:4px; background:#f8fafc; }
  .help { font-size:12px; color:#334155; border:1px dashed #cbd5e1; padding:6px 8px; border-radius:6px; display:none; background:#f8fafc; }
  .help strong { font-weight:600; }
  .badge { font-size:11px; border:1px solid #cbd5e1; border-radius:10px; padding:1px 6px; background:#f8fafc; }
  .form-actions { display:flex; gap:8px; align-items:center; margin-top:8px; }
</style>
</head>
<body>
<div class="wrap">
  <div class="bar">
    <button id="homeBtn" class="btn">Home</button>
    <button id="reloadBtn" class="btn">Reload</button>
    <button id="saveBtn" class="btn" title="Commit all staged changes to disk">Save all</button>
    <button id="discardBtn" class="btn">Discard</button>
    <button id="importBtn" class="btn" title="Import lines from input/lines into GeoParquet">Import lines</button>
    <button id="exitBtn" class="btn">Exit</button>
    <span id="dirtyBadge" class="badge" style="display:none;">Unsaved changes</span>
    <div class="hint">New: <span class="kbd">N</span> • Finish: <span class="kbd">Enter</span> or double-click • Cancel geometry: <span class="kbd">Esc</span> • Edit: <span class="kbd">E</span> • Delete selected: <span class="kbd">Del</span></div>
  </div>

  <div class="form">
    <div class="row">
      <label>Line</label>
      <select id="pickerSel">
        <option value="">Show all</option>
      </select>
    </div>

    <div class="toolbar">
      <button id="newBtn" class="btn" title="Start drawing a new line (N)">New line</button>
      <button id="editBtn" class="btn" title="Edit selected (E)" disabled>Edit</button>
      <button id="delBtn" class="btn" title="Delete the selected line (Del)" disabled>Delete</button>
      <button id="editApplyBtn" class="btn" style="display:none;" title="Apply geometry changes">Apply geometry</button>
      <button id="editCancelBtn" class="btn" style="display:none;" title="Cancel geometry changes">Cancel geometry</button>
      <button id="helpToggle" class="btn" title="Explain map/edit icons">?</button>
      <span id="countInfo" class="hint"></span>
    </div>

    <div id="helpBox" class="help">
      <strong>Map & edit</strong><br>
      • Draw (pencil) to add a new line (or <span class="kbd">N</span>).<br>
      • Edit: only the selected line is visible until you choose “Show all”. Use <strong>Apply geometry</strong> or <strong>Cancel geometry</strong>.<br>
      • Delete removes the currently selected line (you’ll be asked to confirm).<br>
      • Keys: <span class="kbd">N</span> new, <span class="kbd">E</span> edit, <span class="kbd">Del</span> delete, <span class="kbd">Esc</span> cancel geometry.
    </div>

    <div class="row"><label>GIS name</label><input id="f_name_gis" type="text" readonly></div>
    <div class="row"><label>Title</label><input id="f_name_user" type="text" placeholder="Enter a user-friendly title"></div>
    <div class="row"><label>Length of segments (m)</label><input id="f_seg_len" type="number" inputmode="numeric" placeholder="e.g., 500"></div>
    <div class="row"><label>Segment width (m)</label><input id="f_seg_wid" type="number" inputmode="numeric" placeholder="e.g., 20"></div>
    <div class="row"><label>Description</label><textarea id="f_desc" placeholder="Short description"></textarea></div>

    <div class="form-actions">
      <button id="formSaveBtn" class="btn" title="Commit all staged changes to disk from here">Save changes</button>
      <span id="status" class="status-warn">Initializing…</span>
    </div>
  </div>

  <div class="map"><div id="map"></div></div>

  <div class="foot"><div id="foot-left">Ready.</div><div id="foot-right"></div></div>
</div>

<script>
window.onerror = function(message, source, lineno, colno, error){
  const el = document.getElementById('status');
  if (el){ el.textContent = 'JS error: ' + message + ' @' + lineno + ':' + colno; el.className = 'status-err'; }
};

let MAP=null, BASE_OSM=null, BASE_SAT=null, LINES_GROUP=null, DRAW_CONTROL=null;
let HOME_BOUNDS = null;
let SELECTED_ID = null;
let LAYER_BY_ID = {};
let SELECT_GROUP = null;
let SINGLE_EDIT_HANDLER = null;
let DRAW_HANDLER = null;
let DIRTY = false;
let HIDDEN_LAYERS = {};
let SHOW_ONLY_CURRENT = false;
let DEFAULT_SEGMENT_LENGTH = 1000;
let DEFAULT_SEGMENT_WIDTH = 600;

function showStatus(m, cls){ const el = document.getElementById('status'); el.textContent = m || ''; el.className = cls || ''; }
function setFootLeft(m){ document.getElementById('foot-left').textContent = m || ''; }
function setFootRight(m){ document.getElementById('foot-right').textContent = m || ''; }
function dirtyBadge(show){ document.getElementById('dirtyBadge').style.display = show ? '' : 'none'; }
function setSaveButtonsEnabled(enabled){
    document.getElementById('saveBtn').disabled = !enabled;
    document.getElementById('formSaveBtn').disabled = !enabled;
    document.getElementById('discardBtn').disabled = !enabled;
}
function setDirty(v){ DIRTY = !!v; dirtyBadge(DIRTY); setSaveButtonsEnabled(DIRTY); }

function setFormEnabled(enabled){
    ['f_name_user','f_seg_len','f_seg_wid','f_desc'].forEach(function(id){
        const el = document.getElementById(id);
        if (el) el.disabled = !enabled;
    });
}

function fillForm(p){
  document.getElementById('f_name_gis').value = p?.name_gis || '';
  document.getElementById('f_name_user').value = p?.name_user || '';
  document.getElementById('f_seg_len').value  = (p?.segment_length ?? '');
  document.getElementById('f_seg_wid').value  = (p?.segment_width ?? '');
  document.getElementById('f_desc').value     = p?.description || '';
}

function enableEditButtons(enabled){
  document.getElementById('editBtn').disabled = !enabled;
  document.getElementById('delBtn').disabled  = !enabled;
}

function setSelected(id, fly=false){
  SELECTED_ID = id || null;
  SHOW_ONLY_CURRENT = !!id;
  const layer = id ? LAYER_BY_ID[id] : null;
  const props = (layer && layer.feature) ? layer.feature.properties : {};
  fillForm(props||{});
    setFormEnabled(!!id);
  enableEditButtons(!!id);
    const title = (props && props.name_user && String(props.name_user).trim().length) ? String(props.name_user).trim() : '';
    setFootRight(id ? (title ? `${title} — ${id}` : id) : 'No line selected');
  const pickerSel = document.getElementById('pickerSel');
  if (pickerSel && pickerSel.value !== (id||"")) pickerSel.value = (id||"");
  enforceVisibility();
  if (fly && id && LAYER_BY_ID[id]){
    try { MAP.fitBounds(LAYER_BY_ID[id].getBounds(), {padding:[20,20]}); } catch(e){}
  }
}

function updatePicker(records){
  const sel = document.getElementById('pickerSel');
  const keep = sel.value;
  sel.innerHTML = '';
  const optAll = document.createElement('option'); optAll.value=''; optAll.textContent='Show all';
  sel.appendChild(optAll);
  records.forEach(rec=>{
    const opt = document.createElement('option');
    opt.value = rec.name_gis;
    const title = (rec.name_user && rec.name_user.trim().length>0) ? rec.name_user : '(untitled)';
    opt.textContent = `${title} — ${rec.name_gis}`;
    sel.appendChild(opt);
  });
  sel.value = keep && (Array.from(sel.options).some(o=>o.value===keep)) ? keep : '';
  enableEditButtons(!!sel.value);
  document.getElementById('countInfo').textContent = records.length + ' line(s)';
}

function bindAuto(fieldId, fieldName){
  const el = document.getElementById(fieldId);
  el.addEventListener('change', function(){
    if (!SELECTED_ID) return;
    showStatus('Staging…','status-warn');
    window.pywebview.api.update_attribute(SELECTED_ID, fieldName, el.value).then(function(res){
      if (!res.ok){ showStatus(res.error || 'Update failed', 'status-err'); }
      else {
        const layer = LAYER_BY_ID[SELECTED_ID];
        if (layer && layer.feature){ layer.feature.properties[fieldName] =
          (fieldName==='segment_length'||fieldName==='segment_width') ? (el.value===''?null:Number(el.value)) : el.value; }
        if (fieldName === 'name_user'){
          const picker = document.getElementById('pickerSel');
          const opt = Array.from(picker.options).find(o=>o.value===SELECTED_ID);
          if (opt){
            const title = (el.value && el.value.trim().length>0) ? el.value : '(untitled)';
            opt.textContent = `${title} — ${SELECTED_ID}`;
          }
        }
        setDirty(true);
        showStatus('Staged (not saved). Click Save to commit.','status-ok');
      }
    }).catch(function(err){ showStatus('API error: '+err, 'status-err'); });
  });
}

function initMapOnce(){
  if (MAP) return;
  MAP = L.map('map');
  BASE_OSM = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom:19, crossOrigin:true, attribution:'© OpenStreetMap'}).addTo(MAP);
  BASE_SAT = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {maxZoom:19, crossOrigin:true, attribution:'Esri, Maxar'});
  const bases = {'OpenStreetMap': BASE_OSM, 'Satellite (Esri)': BASE_SAT};
  L.control.layers(bases, {}, {collapsed:false, position:'topright'}).addTo(MAP);
  L.control.scale({metric:true, imperial:false}).addTo(MAP);
  LINES_GROUP = L.featureGroup().addTo(MAP);

  DRAW_CONTROL = new L.Control.Draw({
    edit: { featureGroup: L.featureGroup() }, // dummy to satisfy control
    draw: { polygon:false, circle:false, rectangle:false, marker:false, circlemarker:false, polyline:{ shapeOptions:{ color:'#ff7f50', weight:4 } } }
  });
  MAP.addControl(DRAW_CONTROL);

  MAP.on('draw:drawstart', function(e){
    if (e.layerType === 'polyline'){
      DRAW_HANDLER = e.handler || (DRAW_CONTROL && DRAW_CONTROL._toolbars.draw && DRAW_CONTROL._toolbars.draw._modes.polyline && DRAW_CONTROL._toolbars.draw._modes.polyline.handler) || null;
    }
  });
  MAP.on('draw:drawstop', function(e){ DRAW_HANDLER = null; });

  MAP.on(L.Draw.Event.CREATED, function(e){
    showStatus('Staging new line…','status-warn');
    const tmp = e.layer; LINES_GROUP.addLayer(tmp);
    const gj = tmp.toGeoJSON();
    window.pywebview.api.add_line(gj).then(function(res){
      if (!res.ok){ showStatus(res.error || 'Add failed','status-err'); LINES_GROUP.removeLayer(tmp); return; }
      LINES_GROUP.removeLayer(tmp); addFeature(res.feature, true);
      const picker = document.getElementById('pickerSel');
      const opt = document.createElement('option');
      opt.value = res.record.name_gis;
      opt.textContent = `(untitled) — ${res.record.name_gis}`;
      picker.appendChild(opt);
      picker.value = res.record.name_gis;

    document.getElementById('f_seg_len').value = DEFAULT_SEGMENT_LENGTH;
    document.getElementById('f_seg_wid').value = DEFAULT_SEGMENT_WIDTH;
    window.pywebview.api.update_attribute(res.record.name_gis, 'segment_length', DEFAULT_SEGMENT_LENGTH);
    window.pywebview.api.update_attribute(res.record.name_gis, 'segment_width', DEFAULT_SEGMENT_WIDTH);
      setDirty(true);

      enableEditButtons(true);
      setSelected(res.record.name_gis, true);
      updateCountBadge();
      showStatus('New line staged.','status-ok');
    }).catch(err => { showStatus('API error: '+err, 'status-err'); LINES_GROUP.removeLayer(tmp); });
  });

  MAP.on(L.Draw.Event.EDITED, function(e){
    e.layers.eachLayer(function(layer){
      const f = layer.feature; if (!f || !f.properties || !f.properties.name_gis) return;
      window.pywebview.api.update_geometry(f.properties.name_gis, layer.toGeoJSON().geometry).then(function(res){
        if (!res.ok){ showStatus(res.error || 'Geometry update failed','status-err'); }
        else { setDirty(true); setFootLeft('Geometry staged for ' + f.properties.name_gis); showStatus('Geometry staged.','status-ok'); }
      }).catch(err => showStatus('API error: '+err,'status-err'));
    });
  });

  setTimeout(()=> MAP.invalidateSize(), 120);
}

function clearLayers(){
  if (!LINES_GROUP) return;
  LINES_GROUP.clearLayers();
  LAYER_BY_ID = {};
}

function addFeature(feature, selectAfter=false){
  const layer = L.geoJSON(feature, { style: ()=>({color:'#ff7f50', weight:4}) });
  layer.eachLayer(function(l){
    const id = feature.properties?.name_gis;
    if (!id) return;
    LAYER_BY_ID[id] = l;
    l.on('click', ()=> setSelected(id, false));
    l.addTo(LINES_GROUP);
    if (selectAfter) setSelected(id, true);
  });
}

function updateCountBadge(){
  document.getElementById('countInfo').textContent =
    Object.keys(LAYER_BY_ID).length + ' line(s)';
}

function cleanupSingleHandlers(){
  if (SINGLE_EDIT_HANDLER){ try{ SINGLE_EDIT_HANDLER.disable(); }catch(e){} SINGLE_EDIT_HANDLER = null; }
  if (SELECT_GROUP){ try{ MAP.removeLayer(SELECT_GROUP); }catch(e){} SELECT_GROUP = null; }
  setEditActionButtons(false);
  enforceVisibility();
}

function hideOthers(){
  HIDDEN_LAYERS = {};
  Object.keys(LAYER_BY_ID).forEach(id=>{
    if (id !== SELECTED_ID){
      const lyr = LAYER_BY_ID[id];
      if (lyr && MAP.hasLayer(lyr)){
        LINES_GROUP.removeLayer(lyr);
        HIDDEN_LAYERS[id] = lyr;
      }
    }
  });
}

function showOthers(){
  Object.keys(HIDDEN_LAYERS).forEach(id=>{
    const lyr = HIDDEN_LAYERS[id];
    try{ LINES_GROUP.addLayer(lyr); }catch(e){}
  });
  HIDDEN_LAYERS = {};
}

function enforceVisibility(){
  if (SHOW_ONLY_CURRENT && SELECTED_ID){ hideOthers(); } else { showOthers(); }
}

function setEditActionButtons(visible){
  document.getElementById('editApplyBtn').style.display  = visible ? '' : 'none';
  document.getElementById('editCancelBtn').style.display = visible ? '' : 'none';
}

function startNewLine(){
  cleanupSingleHandlers();
  const h = DRAW_CONTROL && DRAW_CONTROL._toolbars.draw && DRAW_CONTROL._toolbars.draw._modes.polyline && DRAW_CONTROL._toolbars.draw._modes.polyline.handler;
  if (h){ h.enable(); setFootLeft('Drawing: click to add points, double-click or Enter to finish.'); }
}

function toggleSingleEdit(){
  if (!SELECTED_ID){ showStatus('Select a line first', 'status-warn'); return; }
  if (SINGLE_EDIT_HANDLER){ cleanupSingleHandlers(); setFootLeft(''); return; }
  const layer = LAYER_BY_ID[SELECTED_ID];
  if (!layer){ showStatus('Selected line not found in map', 'status-err'); return; }
  hideOthers();
  SELECT_GROUP = L.featureGroup([layer]).addTo(MAP);
  SINGLE_EDIT_HANDLER = new L.EditToolbar.Edit(MAP, { featureGroup: SELECT_GROUP, poly:{ allowIntersection:false } });
  SINGLE_EDIT_HANDLER.enable();
  setEditActionButtons(true);
  setFootLeft('Edit mode: adjust vertices, then “Apply geometry” or “Cancel geometry”.');
}

function applyEdit(){
  if (!SINGLE_EDIT_HANDLER) return;
  try{
        if (typeof SINGLE_EDIT_HANDLER.save === 'function') SINGLE_EDIT_HANDLER.save();
        else if (typeof SINGLE_EDIT_HANDLER._save === 'function') SINGLE_EDIT_HANDLER._save();
  }catch(e){}
  cleanupSingleHandlers();
  reloadSelected();
}

function cancelEdit(){
  if (!SINGLE_EDIT_HANDLER) return;
  try{
    if (typeof SINGLE_EDIT_HANDLER.revertLayers === 'function') SINGLE_EDIT_HANDLER.revertLayers();
    else if (typeof SINGLE_EDIT_HANDLER._revertLayers === 'function') SINGLE_EDIT_HANDLER._revertLayers();
  }catch(e){}
  cleanupSingleHandlers();
  showStatus('Geometry edit canceled.','status-warn');
}

async function deleteSelected(){
  if (!SELECTED_ID){ showStatus('Select a line first', 'status-warn'); return; }
  const layer = LAYER_BY_ID[SELECTED_ID];
  const title = (layer && layer.feature && layer.feature.properties && layer.feature.properties.name_user) || '';
  const label = title && title.trim().length ? `${title} — ${SELECTED_ID}` : SELECTED_ID;
  if (!confirm(`Delete the selected line:\n\n${label}\n\nThis action will be staged and applied when you Save.`)) return;

  try{
    const res = await window.pywebview.api.delete_line(SELECTED_ID);
    if (!res.ok){ showStatus(res.error || 'Delete failed','status-err'); return; }
    if (layer){ LINES_GROUP.removeLayer(layer); delete LAYER_BY_ID[SELECTED_ID]; }
    const picker = document.getElementById('pickerSel');
    Array.from(picker.options).forEach(o=>{ if (o.value===SELECTED_ID) o.remove(); });
    setSelected('');
    setDirty(true);
    updateCountBadge();
    showStatus('Line deleted (staged). Click Save to commit.','status-ok');
    setFootLeft('Deleted (staged).');
  }catch(err){
    showStatus('API error: '+err,'status-err');
  }
}

async function reloadSelected(){
  if (!SELECTED_ID) return;
  try{
    const res = await window.pywebview.api.get_all_lines();
    if (!res.ok){ showStatus(res.error || 'Reload failed','status-err'); return; }
    const feats = (res.geojson && res.geojson.features) ? res.geojson.features : [];
    const f = feats.find(x => x.properties && x.properties.name_gis === SELECTED_ID);
    if (!f){ showStatus('Selected line not found after reload','status-err'); return; }
    const old = LAYER_BY_ID[SELECTED_ID];
    if (old){ try{ LINES_GROUP.removeLayer(old); }catch(e){} }
    delete LAYER_BY_ID[SELECTED_ID];
    addFeature(f, false);
    enforceVisibility();
    fillForm(f.properties || {});
    setFootLeft('Line reloaded.');
  } catch(err){
    showStatus('API error: '+err, 'status-err');
  }
}

async function loadAll(){
  try{
    initMapOnce();
    cleanupSingleHandlers();
    clearLayers();

    const st = await window.pywebview.api.get_state();
    if (!st.ok){ showStatus(st.error || 'State failed', 'status-err'); return; }
    updatePicker(st.records || []);

        try{
            const defaults = await window.pywebview.api.get_defaults();
            if (defaults && defaults.ok){
                if (Number.isFinite(Number(defaults.segment_length))) DEFAULT_SEGMENT_LENGTH = Number(defaults.segment_length);
                if (Number.isFinite(Number(defaults.segment_width))) DEFAULT_SEGMENT_WIDTH = Number(defaults.segment_width);
            }
        } catch(e){}

    const res = await window.pywebview.api.get_all_lines();
    if (!res.ok){ showStatus(res.error || 'Load failed', 'status-err'); return; }
    const feats = res.geojson && res.geojson.features ? res.geojson.features : [];
    feats.forEach(f => addFeature(f, false));
    HOME_BOUNDS = res.home_bounds || null;
    updateCountBadge();
    setTimeout(function(){
      if (HOME_BOUNDS) MAP.fitBounds(HOME_BOUNDS, {padding:[20,20]});
      else MAP.setView([59.9139,10.7522], 5);
    }, 80);

    enforceVisibility();
    showStatus('Ready.','status-ok');
  } catch(err){
    showStatus('API error: '+err, 'status-err');
  }
}

function boot(){
  document.getElementById('homeBtn').addEventListener('click', ()=> { if (HOME_BOUNDS) MAP.fitBounds(HOME_BOUNDS, {padding:[20,20]}); });
    document.getElementById('reloadBtn').addEventListener('click', ()=> {
        if (DIRTY && !confirm('You have unsaved changes. Reload anyway and keep staged changes in memory?')) return;
        loadAll();
    });
  document.getElementById('exitBtn').addEventListener('click', ()=> window.pywebview.api.exit_app());
  document.getElementById('newBtn').addEventListener('click', startNewLine);
  document.getElementById('editBtn').addEventListener('click', toggleSingleEdit);
  document.getElementById('delBtn').addEventListener('click', deleteSelected);
  document.getElementById('editApplyBtn').addEventListener('click', applyEdit);
  document.getElementById('editCancelBtn').addEventListener('click', cancelEdit);
  document.getElementById('helpToggle').addEventListener('click', ()=>{
    const box = document.getElementById('helpBox'); box.style.display = (box.style.display==='none'||box.style.display==='') ? 'block' : 'none';
  });
  document.getElementById('pickerSel').addEventListener('change', (e)=>{
    const id = e.target.value || '';
    setSelected(id, !!id);
  });

  async function doSave(){
        if (!DIRTY){ showStatus('No unsaved changes.','status-warn'); return; }
    showStatus('Saving…','status-warn');
    try{
      const res = await window.pywebview.api.commit();
      if (!res.ok){ showStatus('Save failed','status-err'); return; }
      setDirty(false);
      showStatus('All changes saved.','status-ok');
      if (SELECTED_ID) { reloadSelected(); }
    } catch(err){ showStatus('API error: '+err,'status-err'); }
  }

  document.getElementById('saveBtn').addEventListener('click', doSave);
  document.getElementById('formSaveBtn').addEventListener('click', doSave);

  document.getElementById('discardBtn').addEventListener('click', async ()=>{
    if (!confirm('Discard all unsaved changes and reload from disk?')) return;
    showStatus('Discarding…','status-warn');
    try{
      const res = await window.pywebview.api.discard();
      if (!res.ok){ showStatus(res.error || 'Discard failed','status-err'); return; }
      setDirty(false);
      await loadAll();
      const pickerSel = document.getElementById('pickerSel');
      setSelected(pickerSel.value || '');
      showStatus('Reverted to last saved state.','status-ok');
    } catch(err){ showStatus('API error: '+err,'status-err'); }
  });

  document.getElementById('importBtn').addEventListener('click', async ()=>{
        if (DIRTY && !confirm('Import replaces current line set. Continue and lose unsaved staged changes?')) return;
    showStatus('Importing lines…','status-warn');
    try{
      const res = await window.pywebview.api.import_lines_from_input();
      if (!res.ok){ showStatus(res.error || 'Import failed', 'status-err'); return; }
      setDirty(false);
      await loadAll();
      showStatus('Imported lines.','status-ok');
    } catch(err){
      showStatus('API error: '+err, 'status-err');
    }
  });

  bindAuto('f_name_user','name_user');
  bindAuto('f_seg_len','segment_length');
  bindAuto('f_seg_wid','segment_width');
  bindAuto('f_desc','description');
    setFormEnabled(false);
    setSaveButtonsEnabled(false);
    setFootRight('No line selected');

    document.addEventListener('keydown', function(e){
        // Don’t trigger map/edit shortcuts while typing in form fields.
        // (Otherwise typing “e” in Title triggers Edit, etc.)
        const t = e.target;
        const tag = (t && t.tagName) ? t.tagName.toLowerCase() : '';
        const inFormField = (tag === 'input' || tag === 'textarea' || tag === 'select' || (t && t.isContentEditable));

        if ((e.ctrlKey || e.metaKey) && (e.key === 's' || e.key === 'S')){
            e.preventDefault();
            doSave();
            return;
        }
        if (e.altKey) return;
        if (inFormField && e.key !== 'Escape') return;

        if (e.key === 'N' || e.key === 'n'){ startNewLine(); }
        else if (e.key === 'E' || e.key === 'e'){ if (!document.getElementById('editBtn').disabled) toggleSingleEdit(); }
        else if (e.key === 'Delete'){ if (!document.getElementById('delBtn').disabled) deleteSelected(); }
        else if (e.key === 'Escape'){ cancelEdit(); }
    });

  if (window.pywebview && window.pywebview.api) {
    loadAll();
  } else {
    showStatus('Waiting for backend…', 'status-warn');
    window.addEventListener('pywebviewready', () => { loadAll(); });
    let tries = 0;
    const tic = setInterval(()=>{
      if (window.pywebview && window.pywebview.api){
        clearInterval(tic);
        loadAll();
      } else if (++tries > 50) {
        clearInterval(tic);
        showStatus('Backend did not initialize. Are you opening the HTML directly?', 'status-err');
      }
    }, 100);
  }
}

document.addEventListener('DOMContentLoaded', boot);
</script>
</body>
</html>
"""

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_working_directory", required=False)
    args = parser.parse_args()

    base_dir = resolve_base_dir(args.original_working_directory)
    cfg_path = config_path(base_dir)
    cfg = read_config(cfg_path)

    # Respect custom parquet folder if provided
    global _PARQUET_SUBDIR
    _PARQUET_SUBDIR = cfg["DEFAULT"].get("parquet_folder", "output/geoparquet")

    api = Api(base_dir, cfg)
    global webview
    webview = _require_webview()
    window = webview.create_window(title="Edit line (GeoParquet)", html=HTML, js_api=api, width=1280, height=860)

    try:
        webview.start(gui='edgechromium', debug=False)
    except Exception as e:
        sys.stderr.write(
            "This app prefers Microsoft Edge WebView2 runtime to avoid console spam.\n"
            "Install 'Evergreen WebView2 Runtime' from Microsoft and retry if needed.\n"
            f"Underlying error: {e}\n"
        )
        sys.exit(1)

if __name__ == "__main__":
    main()
