#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, base64, configparser, locale, sqlite3, threading, json, time, io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping
from pyproj import Geod
from pathlib import Path

# ---------- Optional shapely make_valid ----------
try:
    from shapely.validation import make_valid as _make_valid
except Exception:
    _make_valid = None

# ---------- pywebview ----------
try:
    import webview  # pip install pywebview
except ModuleNotFoundError:
    sys.stderr.write(
        "ERROR: 'pywebview' is not installed in the Python environment launching maps_overview.py.\n"
        "Install it in that environment, e.g.:  pip install pywebview\n"
    )
    sys.exit(1)

# ===============================
# Locale (safe)
# ===============================
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except Exception:
    pass

# ===============================
# Paths / constants (robust for .py and .exe in tools/)
# ===============================

def base_dir() -> Path:
    """Resolve the mesa root folder whether running as .py or frozen .exe in tools/.
    Searches: original_working_directory (if provided), executable/script folder,
    current working dir; normalizes tools/system/code to parent; then climbs up
    a few levels to find a folder containing output/ and input/ or tools/ + config.ini.
    """
    candidates = []
    try:
        if 'original_working_directory' in globals() and original_working_directory:
            candidates.append(Path(original_working_directory))
    except Exception:
        pass
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent)
    else:
        if "__file__" in globals():
            candidates.append(Path(__file__).resolve().parent)
    candidates.append(Path(os.getcwd()).resolve())

    def normalize(p: Path) -> Path:
        p = p.resolve()
        if p.name.lower() in ("tools", "system", "code"):
            if not ((p / "config.ini").exists() or (p / "output").exists()):
                p = p.parent
        q = p
        for _ in range(4):
            if (q / "output").exists() and (q / "input").exists():
                return q
            if (q / "tools").exists() and (q / "config.ini").exists():
                return q
            code_candidate = q / "code"
            if code_candidate.exists() and (code_candidate / "config.ini").exists():
                return code_candidate
            q = q.parent
        if (p / "config.ini").exists():
            return p
        code_alt = p / "code"
        if code_alt.exists() and (code_alt / "config.ini").exists():
            return code_alt
        return p

    for c in candidates:
        root = normalize(c)
        if (root / "tools").exists() or ((root / "output").exists() and (root / "input").exists()):
            return root
    return normalize(candidates[0])

# Tame stdio encoding so printing unicode from pywebview handlers is safe on Windows
try:
    enc = os.environ.get("PYTHONIOENCODING") or getattr(sys.stdout, "encoding", None) or "utf-8"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding=enc, errors="replace")
        sys.stderr.reconfigure(encoding=enc, errors="replace")
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=enc, errors="replace", line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding=enc, errors="replace", line_buffering=True)
except Exception:
    pass

APP_DIR = base_dir()
os.chdir(APP_DIR)

CONFIG_FILE   = APP_DIR / "config.ini"
OUTPUT_DIR    = APP_DIR / "output"
PARQUET_DIR   = OUTPUT_DIR / "geoparquet"
MBTILES_DIR   = OUTPUT_DIR / "mbtiles"
AREA_JSON     = OUTPUT_DIR / "area_stats.json"

PARQUET_FILE  = PARQUET_DIR / "tbl_flat.parquet"
SEGMENT_FILE  = PARQUET_DIR / "tbl_segment_flat.parquet"
ASSET_FILE    = PARQUET_DIR / "tbl_asset_object.parquet"
LINES_FILE    = PARQUET_DIR / "tbl_lines.parquet"

PLOT_CRS         = "EPSG:4326"
BASIC_GROUP_NAME = "basic_mosaic"
ZOOM_THRESHOLD   = 10
STEEL_BLUE       = "#4682B4"

# ===============================
# Config / colors
# ===============================
def read_config(path: str) -> configparser.ConfigParser:
    # Allow # in color hex values; only ';' starts inline comments
    cfg = configparser.ConfigParser(inline_comment_prefixes=(';',), strict=False)
    try:
        cfg.read(path, encoding="utf-8")
    except Exception:
        pass
    return cfg

def _safe_hex(s, fb="#BDBDBD"):
    s = (s or "").strip() or fb
    return s

def get_color_mapping(cfg: configparser.ConfigParser) -> dict:
    fb = _safe_hex(cfg["DEFAULT"].get("category_colour_unknown", "#BDBDBD"))
    out = {}
    for c in "ABCDE":
        if c in cfg:
            out[c] = _safe_hex(cfg[c].get("category_colour", ""), fb)
        else:
            out[c] = fb
    return out

def get_desc_mapping(cfg: configparser.ConfigParser) -> dict:
    return {c: (cfg[c].get("description","") if c in cfg else "") for c in "ABCDE"}

# ===============================
# Geo helpers & stats
# ===============================
def load_parquet(path: str) -> gpd.GeoDataFrame:
    try:
        if not os.path.exists(path):
            return gpd.GeoDataFrame()
        return gpd.read_parquet(path)
    except Exception as e:
        print(f"Failed to read parquet {path}: {e}", file=sys.stderr)
        return gpd.GeoDataFrame()

def only_A_to_E(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty: return gdf
    keep = gdf[gdf.geometry.notna()].copy()
    if "sensitivity_code_max" in keep.columns:
        keep["sensitivity_code_max"] = (
            keep["sensitivity_code_max"].astype("string").fillna("").str.strip().str.upper()
        )
        keep = keep[keep["sensitivity_code_max"].isin(list("ABCDE"))]
    return keep

def to_plot_crs(gdf: gpd.GeoDataFrame, cfg: configparser.ConfigParser) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.set_crs(PLOT_CRS, allow_override=True) if gdf.crs is None else gdf
    if gdf.crs is None:
        try:
            epsg = int(cfg["DEFAULT"].get("workingprojection_epsg", "4326"))
            gdf = gdf.set_crs(epsg=epsg, allow_override=True)
        except Exception:
            gdf = gdf.set_crs(PLOT_CRS, allow_override=True)
    if str(gdf.crs).upper() != PLOT_CRS:
        try:
            gdf = gdf.to_crs(PLOT_CRS)
        except Exception:
            try:
                epsg = int(cfg["DEFAULT"].get("workingprojection_epsg", "4326"))
                gdf = gdf.set_crs(epsg=epsg, allow_override=True).to_crs(PLOT_CRS)
            except Exception:
                gdf = gdf.set_crs(PLOT_CRS, allow_override=True)
    return gdf

def to_epsg4326(gdf: gpd.GeoDataFrame, cfg: configparser.ConfigParser) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.set_crs(4326, allow_override=True) if gdf.crs is None else gdf.to_crs(4326)
    g = gdf.copy()
    if g.crs is None:
        try:
            epsg = int(cfg["DEFAULT"].get("workingprojection_epsg", "4326"))
            g = g.set_crs(epsg=epsg, allow_override=True)
        except Exception:
            g = g.set_crs(4326, allow_override=True)
    if str(g.crs).upper() != "EPSG:4326":
        g = g.to_crs(4326)
    return g

def bounds_to_leaflet(b):
    minx, miny, maxx, maxy = [float(x) for x in b]
    dx, dy = maxx-minx, maxy-miny
    if dx <= 0 or dy <= 0:
        pad = 0.1
        minx -= pad; maxx += pad; miny -= pad; maxy += pad
    else:
        minx -= dx*0.1; maxx += dx*0.1
        miny -= dy*0.1; maxy += dy*0.1
    minx = max(-180.0, minx); maxx = min(180.0, maxx)
    miny = max(-85.0,  miny); maxy = min(85.0,  maxy)
    return [[miny, minx],[maxy, maxx]]

def gdf_to_geojson_min(gdf: gpd.GeoDataFrame) -> dict:
    feats = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        props = {
            "sensitivity_code_max": row.get("sensitivity_code_max", None),
            "area_km2": (float(row.get("area_m2", 0.0)) / 1e6) if ("area_m2" in row) else None,
            "geocode_group": row.get("name_gis_geocodegroup", None),
        }
        if "name_asset_object" in row: props["name_asset_object"] = row.get("name_asset_object", None)
        if "id_asset_object"   in row: props["id_asset_object"]   = row.get("id_asset_object", None)
        feats.append({"type": "Feature", "geometry": mapping(geom), "properties": props})
    return {"type": "FeatureCollection", "features": feats}

def assets_to_geojson(gdf: gpd.GeoDataFrame, cfg: configparser.ConfigParser) -> dict:
    g = to_epsg4326(gdf, cfg)
    feats = []
    for _, row in g.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        try:
            area_m2 = 0.0
            if geom.geom_type in ("Polygon", "MultiPolygon"):
                a, _ = GEOD.geometry_area_perimeter(geom) if geom.geom_type == "Polygon" else (sum(abs(GEOD.geometry_area_perimeter(p)[0]) for p in geom.geoms), 0)
                area_m2 = abs(a)
            area_km2 = (area_m2 / 1e6) if area_m2 else None
        except Exception:
            area_km2 = None
        props = {
            "id_asset_object": row.get("id_asset_object", None),
            "name_asset_object": row.get("name_asset_object", None),
            "area_km2": area_km2
        }
        feats.append({"type": "Feature", "geometry": mapping(geom), "properties": props})
    return {"type": "FeatureCollection", "features": feats}

def envindex_to_geojson(gdf: gpd.GeoDataFrame) -> dict:
    feats = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        props = {
            "env_index": row.get("env_index", None),
            "area_km2": (float(row.get("area_m2", 0.0)) / 1e6) if ("area_m2" in row) else None,
            "geocode_group": row.get("name_gis_geocodegroup", None),
        }
        feats.append({"type": "Feature", "geometry": mapping(geom), "properties": props})
    return {"type": "FeatureCollection", "features": feats}

def totals_to_geojson(gdf: gpd.GeoDataFrame, total_col: str) -> dict:
    feats = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        props = {
            total_col: row.get(total_col, None),
            "geocode_group": row.get("name_gis_geocodegroup", None),
        }
        feats.append({"type": "Feature", "geometry": mapping(geom), "properties": props})
    return {"type": "FeatureCollection", "features": feats}

# --- geodesic area helpers (WGS84) ---
GEOD = Geod(ellps="WGS84")

def _valid_geom(geom):
    if geom is None or geom.is_empty:
        return geom
    try:
        if _make_valid is not None:
            return _make_valid(geom)
        g = geom.buffer(0)
        return g if (g is not None and not g.is_empty) else geom
    except Exception:
        return geom

def geodesic_area_m2(geom) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    gt = geom.geom_type
    try:
        if gt == "Polygon":
            a, _ = GEOD.geometry_area_perimeter(geom)
            return abs(a)
        elif gt == "MultiPolygon":
            return float(sum(abs(GEOD.geometry_area_perimeter(p)[0]) for p in geom.geoms))
    except Exception:
        pass
    return 0.0

def compute_stats_by_geodesic_area_from_flat_basic(df_flat: gpd.GeoDataFrame, cfg: configparser.ConfigParser) -> dict:
    labels = list("ABCDE")
    msg = f'The geocode/partition "{BASIC_GROUP_NAME}" is missing.'
    if df_flat.empty or "name_gis_geocodegroup" not in df_flat.columns:
        return {"labels": labels, "values": [0,0,0,0,0], "message": msg}

    df = df_flat.copy()
    df["name_gis_geocodegroup"] = df["name_gis_geocodegroup"].astype("string").str.strip().str.lower()
    df = df[df["name_gis_geocodegroup"] == BASIC_GROUP_NAME]
    df = only_A_to_E(df)
    if df.empty:
        return {"labels": labels, "values": [0,0,0,0,0], "message": msg}

    if "id_geocode_object" in df.columns:
        df = df.drop_duplicates(subset=["id_geocode_object"])
    else:
        try:
            df = df.assign(__wkb__=df.geometry.apply(lambda g: g.wkb if g is not None else None))
            df = df.drop_duplicates(subset=["__wkb__"]).drop(columns=["__wkb__"])
        except Exception:
            df = df.drop_duplicates()

    df = to_epsg4326(df, cfg)
    df["geometry"] = df["geometry"].apply(_valid_geom)

    out = []
    for c in labels:
        sub = df[df["sensitivity_code_max"] == c]
        if sub.empty:
            out.append(0.0)
            continue
        a_m2 = float(sub.geometry.apply(geodesic_area_m2).sum())
        out.append(a_m2 / 1e6)
    return {"labels": labels, "values": out}

# ===============================
# Area JSON reader (robust)
# ===============================
def _norm_key_local(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_").replace("-", "_")

def _to_float_safe(v) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        try:
            return float(v)
        except Exception:
            return 0.0
    try:
        s = str(v).strip()
        if s == "" or s.lower() in ("nan", "null", "none"):
            return 0.0
        s = s.replace(",", "")
        return float(s)
    except Exception:
        return 0.0

def _extract_labels_values(obj, default_units="km2"):
    labels_out = list("ABCDE")
    units = (default_units or "km2").lower()

    if isinstance(obj, dict):
        if "labels" in obj and "values" in obj:
            labs = [str(x).upper() for x in (obj.get("labels") or [])]
            vals = obj.get("values")
            if isinstance(vals, (list, tuple)):
                numbers = []
                for v in vals:
                    numbers.append(_to_float_safe(v))
                mapping = {k: numbers[i] for i, k in enumerate(labs) if k in labels_out and i < len(numbers)}
            elif isinstance(vals, dict):
                mapping = {}
                for k, v in vals.items():
                    ku = str(k).upper()
                    if ku in labels_out:
                        mapping[ku] = _to_float_safe(v)
            else:
                mapping = {}
            units = (obj.get("units") or obj.get("unit") or units).lower()
            valsA = [_to_float_safe(mapping.get(c, 0.0)) for c in labels_out]
            if units in ("m2","sqm","square_meters"):
                valsA = [v/1e6 for v in valsA]
            return labels_out, valsA

        for key, is_m2 in (("area_km2", False), ("km2", False), ("area_m2", True), ("m2", True)):
            if key in obj and isinstance(obj[key], dict):
                m = {str(k).upper(): _to_float_safe(v) for k, v in obj[key].items()}
                valsA = [_to_float_safe(m.get(c, 0.0)) for c in labels_out]
                if is_m2:
                    valsA = [v/1e6 for v in valsA]
                return labels_out, valsA

        use = {}
        for k, v in obj.items():
            ku = str(k).upper()
            if ku in labels_out:
                use[ku] = _to_float_safe(v)
        if use:
            units = (obj.get("units") or obj.get("unit") or units).lower()
            valsA = [_to_float_safe(use.get(c, 0.0)) for c in labels_out]
            if units in ("m2","sqm","square_meters"):
                valsA = [v/1e6 for v in valsA]
            return labels_out, valsA

    if isinstance(obj, (list, tuple)) and len(obj) >= 5:
        valsA = [_to_float_safe(obj[i]) for i in range(5)]
        return labels_out, valsA

    return labels_out, [0.0]*5

def _read_area_json(path: Path, group_name: str) -> dict | None:
    try:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        sys.stderr.write(f"[area_stats.json] Read/parse failed: {e}\n")
        return None

    cand = None
    if isinstance(data, dict):
        groups = data.get("groups")
        if isinstance(groups, dict):
            cand = groups.get(group_name)
            if cand is None:
                nk = _norm_key_local(group_name)
                for k in groups.keys():
                    if _norm_key_local(str(k)) == nk:
                        cand = groups[k]
                        break
        if cand is None:
            cand = data.get(group_name)
            if cand is None:
                nk = _norm_key_local(group_name)
                for k in data.keys():
                    if _norm_key_local(str(k)) == nk:
                        cand = data[k]
                        break
        if cand is None:
            cand = data

    try:
        labels, values = _extract_labels_values(cand if cand is not None else {})
        if not (isinstance(values, list) and len(values) >= 5):
            return None
        if any(not isinstance(v, (int, float)) for v in values):
            values = [_to_float_safe(v) for v in values[:5]]
        return { "labels": list("ABCDE"), "values": [float(v) for v in values[:5]] }
    except Exception as e:
        sys.stderr.write(f"[area_stats.json] Extraction failed: {e}\n")
        return None

def get_area_stats() -> dict:
    js = _read_area_json(AREA_JSON, BASIC_GROUP_NAME)
    if js is not None:
        return js
    msg = ("Area statistics JSON not available yet; using live computation from GeoParquet. "
           f"Expected at {AREA_JSON.name}.")
    fallback = compute_stats_by_geodesic_area_from_flat_basic(GDF, cfg)
    if "message" in fallback:
        return fallback
    fallback["message"] = msg
    return fallback

# ===============================
# MBTiles scanning + tiny server
# ===============================
def _norm_key(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_").replace("-", "_")

def scan_mbtiles(dir_path: str):
    idx = {}
    rev = {}
    if not os.path.isdir(dir_path):
        return idx, rev
    for fn in os.listdir(dir_path):
        if not fn.lower().endswith(".mbtiles"):
            continue
        path = os.path.join(dir_path, fn)
        lo   = fn.lower()
        kind = None
        cat = None
        if lo.endswith("_sensitivity.mbtiles"):
            cat = fn[:-len("_sensitivity.mbtiles")]
            kind = "sensitivity"
        elif lo.endswith("_envindex.mbtiles"):
            cat = fn[:-len("_envindex.mbtiles")]
            kind = "envindex"
        elif lo.endswith("_groupstotal.mbtiles"):
            cat = fn[:-len("_groupstotal.mbtiles")]
            kind = "groupstotal"
        elif lo.endswith("_assetstotal.mbtiles"):
            cat = fn[:-len("_assetstotal.mbtiles")]
            kind = "assetstotal"
        else:
            continue
        idx.setdefault(cat, {"sensitivity": None, "envindex": None, "groupstotal": None, "assetstotal": None})
        idx[cat][kind] = path
        rev[_norm_key(cat)] = cat
    return idx, rev

MBTILES_INDEX, MBTILES_REV = scan_mbtiles(MBTILES_DIR)
_MB_META_CACHE = {}

def _mb_meta(path: str):
    if path in _MB_META_CACHE:
        return _MB_META_CACHE[path]
    out = {"bounds": [[-85.0,-180.0],[85.0,180.0]], "minzoom": 0, "maxzoom": 19, "format": "image/png"}
    try:
        con = sqlite3.connect(path)
        try:
            cur = con.cursor()
            cur.execute("SELECT name, value FROM metadata")
            meta = {k.lower(): (v if v is not None else "") for k, v in cur.fetchall()}
            fmt = (meta.get("format","") or "").lower()
            if fmt in ("jpg","jpeg"): out["format"] = "image/jpeg"
            b = meta.get("bounds","")
            if b:
                parts = [float(x) for x in str(b).split(",")]
                if len(parts) == 4:
                    minx, miny, maxx, maxy = parts
                    out["bounds"] = [[miny, minx],[maxy, maxx]]
            try: out["minzoom"] = int(meta.get("minzoom","0"))
            except: pass
            try: out["maxzoom"] = int(meta.get("maxzoom","19"))
            except: pass
        finally:
            con.close()
    except Exception:
        pass
    _MB_META_CACHE[path] = out
    return out

def _resolve_cat(cat: str):
    if cat in MBTILES_INDEX:
        return cat, MBTILES_INDEX[cat]
    key = _norm_key(cat)
    if key in MBTILES_REV:
        disp = MBTILES_REV[key]
        return disp, MBTILES_INDEX.get(disp)
    for disp in MBTILES_INDEX.keys():
        if disp.lower() == cat.lower():
            return disp, MBTILES_INDEX[disp]
    return cat, None

class _MBTilesHandler(BaseHTTPRequestHandler):
    connections = {}
    def log_message(self, format, *args): return
    def log_error(self, format, *args): return
    def do_GET(self):
        try:
            parts = [p for p in self.path.split("?", 1)[0].split("/") if p]
            if len(parts) != 6 or parts[0] != "tiles":
                self.send_response(404); self.end_headers(); return
            _, kind, cat_enc, z_s, x_s, y_file = parts
            kind = (kind or "").lower()
            cat  = unquote(cat_enc)

            disp, rec = _resolve_cat(cat)
            if kind not in ("sensitivity","envindex","groupstotal","assetstotal") or not rec:
                self.send_response(404); self.end_headers(); return

            db_path = rec.get(kind)
            if not db_path or not os.path.exists(db_path):
                self.send_response(404); self.end_headers(); return

            z = int(z_s); x = int(x_s); y = int(y_file.rsplit(".",1)[0])

            con = self.connections.get(db_path)
            if con is None:
                con = sqlite3.connect(db_path, check_same_thread=False)
                self.connections[db_path] = con

            tms_y = (1 << z) - 1 - y
            cur = con.cursor()
            cur.execute("SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?", (z, x, tms_y))
            row = cur.fetchone()
            if not row:
                cur.execute("SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?", (z, x, y))
                row = cur.fetchone()
            if not row:
                self.send_response(204); self.end_headers(); return

            data = row[0]
            fmt = _mb_meta(db_path)["format"]
            self.send_response(200)
            self.send_header("Content-Type", fmt)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            try: self.send_response(500); self.end_headers()
            except Exception: pass

def start_mbtiles_server():
    if not MBTILES_INDEX:
        return None
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MBTilesHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}"

MBTILES_BASE_URL = start_mbtiles_server()

def mbtiles_info(cat: str):
    disp, rec = _resolve_cat(cat)
    if not rec or not MBTILES_BASE_URL:
        return {
            "sensitivity_url": None, "envindex_url": None,
            "groupstotal_url": None, "assetstotal_url": None,
            "bounds": [[-85,-180],[85,180]], "minzoom":0, "maxzoom":19
        }
    src = rec.get("sensitivity") or rec.get("envindex") or rec.get("groupstotal") or rec.get("assetstotal")
    m = _mb_meta(src) if src else {"bounds":[[-85,-180],[85,180]], "minzoom":0, "maxzoom":19}
    def build(kind):
        p = rec.get(kind)
        if p and os.path.exists(p):
            return f"{MBTILES_BASE_URL}/tiles/{kind}/{disp}/{{z}}/{{x}}/{{y}}.png"
        return None
    return {
        "sensitivity_url": build("sensitivity"),
        "envindex_url": build("envindex"),
        "groupstotal_url": build("groupstotal"),
        "assetstotal_url": build("assetstotal"),
        "bounds": m["bounds"],
        "minzoom": m["minzoom"],
        "maxzoom": m["maxzoom"],
    }

# ===============================
# Load datasets
# ===============================
cfg          = read_config(CONFIG_FILE)
COLS         = get_color_mapping(cfg)
DESC         = get_desc_mapping(cfg)

GDF          = load_parquet(PARQUET_FILE)
SEG_GDF      = load_parquet(SEGMENT_FILE)
ASSET_GDF    = load_parquet(ASSET_FILE)
LINES_GDF    = load_parquet(LINES_FILE)

GEOCODE_AVAILABLE   = (not GDF.empty) and ("name_gis_geocodegroup" in GDF.columns)
SEGMENTS_AVAILABLE  = (not SEG_GDF.empty) and ("geometry" in SEG_GDF.columns)
ASSETS_AVAILABLE    = (not ASSET_GDF.empty) and ("geometry" in ASSET_GDF.columns)
ENVINDEX_AVAILABLE  = GEOCODE_AVAILABLE and ("env_index" in GDF.columns)

mb_cats   = list(MBTILES_INDEX.keys())
vec_cats  = sorted(GDF["name_gis_geocodegroup"].dropna().unique().tolist()) if GEOCODE_AVAILABLE else []
CATS      = sorted(set(mb_cats) | set(vec_cats), key=lambda s: (str(s).lower()))

if SEGMENTS_AVAILABLE and "name_gis_geocodegroup" in SEG_GDF.columns:
    SEG_CATS = sorted(SEG_GDF["name_gis_geocodegroup"].dropna().unique().tolist())
else:
    SEG_CATS = []

LINE_NAMES = []
LINE_USER_TO_GIS = {}
if not LINES_GDF.empty and ("name_gis" in LINES_GDF.columns):
    name_col = "name_user" if "name_user" in LINES_GDF.columns else None
    if name_col is None:
        for c in ("name_line", "name"):
            if c in LINES_GDF.columns:
                name_col = c
                break
    if name_col:
        tmp = LINES_GDF[[name_col, "name_gis"]].dropna()
        tmp[name_col] = tmp[name_col].astype(str).str.strip()
        tmp["name_gis"] = tmp["name_gis"].astype(str).str.strip()
        for nm, ng in tmp.itertuples(index=False):
            if nm and ng:
                LINE_USER_TO_GIS.setdefault(nm, set()).add(ng)
        LINE_NAMES = sorted(LINE_USER_TO_GIS.keys())

BING_KEY = cfg["DEFAULT"].get("bing_maps_key", "").strip()

# ===============================
# API exposed to JavaScript
# ===============================
class Api:
    def js_log(self, message: str):
        try: print(f"[JS] {message}")
        except Exception: pass

    def get_state(self):
        return {
            "geocode_available": GEOCODE_AVAILABLE or bool(MBTILES_INDEX),
            "geocode_categories": CATS,
            "segment_available": SEGMENTS_AVAILABLE,
            "segment_categories": SEG_CATS if SEGMENTS_AVAILABLE else [],
            "segment_line_names": LINE_NAMES,
            "assets_available": ASSETS_AVAILABLE,
            "envindex_available": ENVINDEX_AVAILABLE or bool(MBTILES_INDEX),
            "colors": COLS,
            "descriptions": DESC,
            "initial_geocode": (CATS[0] if CATS else None),
            "has_segments": SEGMENTS_AVAILABLE,
            "bing_key": BING_KEY,
            "zoom_threshold": ZOOM_THRESHOLD
        }

    def get_geocode_layer(self, geocode_category):
        try:
            mb = mbtiles_info(geocode_category)
            stats = get_area_stats()

            if mb.get("sensitivity_url"):
                return {
                    "ok": True,
                    "geojson": {"type":"FeatureCollection","features":[]},
                    "home_bounds": mb.get("bounds"),
                    "stats": stats,
                    "mbtiles": {
                        "sensitivity_url": mb["sensitivity_url"],
                        "minzoom": mb["minzoom"],
                        "maxzoom": mb["maxzoom"]
                    }
                }
            if not GEOCODE_AVAILABLE:
                return {"ok": True, "geojson": {"type":"FeatureCollection","features":[]}, "home_bounds": [[0,0],[0,0]], "stats": stats}
            df_map = GDF[GDF["name_gis_geocodegroup"] == geocode_category].copy()
            df_map = only_A_to_E(df_map)
            df_map = to_plot_crs(df_map, cfg)
            map_bounds  = bounds_to_leaflet(df_map.total_bounds) if not df_map.empty else [[0,0],[0,0]]
            map_geojson = gdf_to_geojson_min(df_map)
            return {"ok": True, "geojson": map_geojson, "home_bounds": map_bounds, "stats": stats}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_envindex_layer(self, geocode_category):
        try:
            mb = mbtiles_info(geocode_category)
            if mb.get("envindex_url"):
                return {"ok": True, "geojson": {"type":"FeatureCollection","features":[]},
                        "mbtiles": {"envindex_url": mb["envindex_url"],
                                    "minzoom": mb["minzoom"], "maxzoom": mb["maxzoom"]}}
            if not ENVINDEX_AVAILABLE:
                return {"ok": False, "error": "env_index not available in tbl_flat."}
            df = GDF[GDF["name_gis_geocodegroup"] == geocode_category].copy()
            if df.empty:
                return {"ok": True, "geojson": {"type":"FeatureCollection","features":[]}}
            df = to_plot_crs(df, cfg)
            gj = envindex_to_geojson(df)
            return {"ok": True, "geojson": gj}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _get_totals_common(self, geocode_category, total_col):
        df = GDF[GDF["name_gis_geocodegroup"] == geocode_category].copy()
        if df.empty:
            return {"ok": True, "geojson": {"type":"FeatureCollection","features":[]}, "range": {"min": 0, "max": 0}}
        if total_col not in df.columns:
            return {"ok": False, "error": f"Column '{total_col}' not found in tbl_flat."}
        df = to_plot_crs(df, cfg)
        vals = pd.to_numeric(df[total_col], errors="coerce")
        vmin = float(vals.min()) if len(vals) else 0.0
        vmax = float(vals.max()) if len(vals) else 0.0
        gj = totals_to_geojson(df, total_col)
        return {"ok": True, "geojson": gj, "range": {"min": vmin, "max": vmax}}

    def get_groupstotal_layer(self, geocode_category):
        try:
            mb = mbtiles_info(geocode_category)
            if mb.get("groupstotal_url"):
                return {"ok": True, "geojson": {"type":"FeatureCollection","features":[]},
                        "mbtiles": {"groupstotal_url": mb["groupstotal_url"],
                                    "minzoom": mb["minzoom"], "maxzoom": mb["maxzoom"]},
                        "home_bounds": mb.get("bounds")}
            return self._get_totals_common(geocode_category, "asset_groups_total")
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_assetstotal_layer(self, geocode_category):
        try:
            mb = mbtiles_info(geocode_category)
            if mb.get("assetstotal_url"):
                return {"ok": True, "geojson": {"type":"FeatureCollection","features":[]},
                        "mbtiles": {"assetstotal_url": mb["assetstotal_url"],
                                    "minzoom": mb["minzoom"], "maxzoom": mb["maxzoom"]},
                        "home_bounds": mb.get("bounds")}
            return self._get_totals_common(geocode_category, "assets_overlap_total")
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_segment_layer(self, seg_line_name_or_all):
        try:
            if not SEGMENTS_AVAILABLE:
                return {"ok": False, "error": "Segments dataset is empty or missing."}
            df = SEG_GDF.copy()
            sel = seg_line_name_or_all
            if sel and sel not in (None, "", "__ALL__"):
                sel = str(sel).strip()
                if "name_gis" in df.columns:
                    allowed = LINE_USER_TO_GIS.get(sel, set())
                    if not allowed:
                        allowed = {sel}
                    df = df[df["name_gis"].astype(str).str.strip().isin(list(allowed))]
                else:
                    df = df.iloc[0:0]
            df = only_A_to_E(df)
            df = to_plot_crs(df, cfg)
            gj = gdf_to_geojson_min(df)
            return {"ok": True, "geojson": gj}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_assets_layer(self):
        try:
            if not ASSETS_AVAILABLE:
                return {"ok": False, "error": "Assets dataset is empty or missing."}
            df = ASSET_GDF.copy()
            df = df[df.geometry.notna()]
            if df.empty:
                return {"ok": False, "error": "No valid geometries found in assets."}
            df = to_plot_crs(df, cfg)
            gj = assets_to_geojson(df, cfg)
            return {"ok": True, "geojson": gj}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def exit_app(self):
        try:
            webview.destroy_window()
        except Exception:
            os._exit(0)

    def save_png(self, data_url: str):
        try:
            if "," in data_url:
                _, b64 = data_url.split(",", 1)
            else:
                b64 = data_url
            data = base64.b64decode(b64)
            win = webview.windows[0]
            path = win.create_file_dialog(
                webview.FileDialog.SAVE,
                save_filename="map_export.png",
                file_types=("PNG Files (*.png)",)
            )
            if not path:
                return {"ok": False, "error": "User cancelled."}
            if isinstance(path, (list, tuple)):
                path = path[0]
            with open(path, "wb") as f:
                f.write(data)
            return {"ok": True, "path": path}
        except Exception as e:
            return {"ok": False, "error": str(e)}

api = Api()

# ===============================
# HTML / JS UI
# ===============================
HTML_TEMPLATE = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Maps Overview</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1"></script>
<script src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"></script>
<style>
  html, body { height:100%; margin:0; }
  .wrap {
    height:100%;
    display:grid;
    grid-template-columns: 1fr 420px;
    grid-template-rows: 48px 1fr;
    grid-template-areas:
      "bar stats"
      "map stats";
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  }
  .bar    { grid-area: bar; display:flex; gap:12px; align-items:center; padding:8px 12px; flex-wrap:wrap; border-bottom: 2px solid #2b3442; }
  .map    { grid-area: map; position:relative; background:#ddd; }
  #map    { position:absolute; inset:0; }
  #map.exporting .leaflet-control-zoom { display: none !important; }
  .stats  { grid-area: stats; border-left: 2px solid #2b3442; display:flex; flex-direction:column; overflow:hidden; }
  #err { color:#b00; padding:6px 12px; font-size:12px; display:none; }
  .legend { padding:8px 12px; font-size:12px; }
  .info-block { padding:8px 12px; font-size:12px; line-height:1.35; border-top:1px solid #2b344211; border-bottom:1px solid #2b344211; background:#0f172a08; max-height:170px; overflow:auto; }
  .info-block p { margin:2px 0 4px 0; }
  .swatch { display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:6px; border:1px solid #8884; }
  .num { text-align:right; white-space:nowrap; }
  .btn { padding:6px 10px; border:1px solid #ccd; background:#fff; border-radius:6px; cursor:pointer; }
  .btn:active { transform:translateY(1px); }
  .slider { display:flex; align-items:center; gap:8px; }
  .slider input[type=range]{ width:160px; }

  .leaflet-control-layers { font-size: 12px; }
  .leaflet-control-layers label { display:block; line-height: 1.35; }
  .inlineSel { margin:4px 0 6px 22px; }
  .inlineChecks { margin:0 0 6px 22px; display:flex; flex-direction:column; gap:6px; align-items:flex-start; }
  .inlineChecks label { display:inline-flex; gap:6px; align-items:center; font-weight:500; }

  #chartBox { flex:1 1 auto; padding:8px 12px; position:relative; overflow:hidden; }
  #chart { position:absolute; top:8px; right:12px; bottom:8px; left:12px; display:block; }

  .leaflet-tooltip.poly-tip {
    background:#0f172a; color:#e5e7eb; border:1px solid #1f2937;
    border-radius:8px; box-shadow:0 6px 20px rgba(0,0,0,.25);
    padding:10px 12px; font-size:12px; line-height:1.25; max-width:280px;
  }
  .poly-tip .hdr { font-size:12px; font-weight:700; letter-spacing:.2px; margin-bottom:6px; display:flex; align-items:center; gap:8px; justify-content:space-between; }
  .poly-tip .chip { display:inline-block; font-weight:700; padding:2px 6px; border-radius:6px; background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.2); }
  .poly-tip .kv { display:grid; grid-template-columns:92px 1fr; gap:4px 10px; }
  .poly-tip .k { opacity:.8; }
  .poly-tip .v { text-align:right; white-space:nowrap; }

  /* Hide the checkbox for the "Geocode group" row and fix spacing */
  .leaflet-control-layers-overlays label.no-toggle { padding-left: 0; }
  .leaflet-control-layers-overlays label.no-toggle .inlineSel,
  .leaflet-control-layers-overlays label.no-toggle .inlineChecks { margin-left: 0; }
  .leaflet-control-layers-overlays label.no-toggle input.leaflet-control-layers-selector { display: none !important; }
</style>
</head>
<body>
<div class="wrap">
  <div class="bar">
    <button id="homeBtn" class="btn">Home</button>

    <div class="slider">
      <span class="label-sm">Opacity</span>
      <input id="opacity" type="range" min="0" max="100" value="80">
      <span id="opv" class="label-sm">80%</span>
    </div>

    <button id="exportBtn" class="btn" title="Export current map to PNG (~300 dpi)">Export PNG</button>
    <button id="exitBtn" class="btn">Exit</button>
  </div>

  <div class="map"><div id="map"></div></div>

  <div class="stats">
    <div id="err"></div>
    <div class="legend" id="legend"></div>
    <div id="infoText" class="info-block">
      <p>Use the <b>Geocode group</b> selector, then enable layers below it:
      <b>Sensitive areas</b>, <b>Environment index</b>, <b>Groups total</b>, or <b>Assets total</b>.</p>
      <p>The viewer prefers raster MBTiles in <code>output/mbtiles</code>. If missing, it falls back to GeoParquet.</p>
    </div>
    <div id="chartBox"><canvas id="chart"></canvas></div>
  </div>
</div>

<script>
var MAP=null, BASE=null, BASE_SOURCES=null, CHART=null;
var GEO_GROUP=null, SEG_GROUP=null, ASSET_GROUP=null, ENV_GROUP=null, GROUPSTOTAL_GROUP=null, ASSETSTOTAL_GROUP=null;
var GEO_FOLDER=null;
var LAYER=null, LAYER_SEG=null, LAYER_ASSETS=null, LAYER_ENV=null, LAYER_GROUPSTOTAL=null, LAYER_ASSETSTOTAL=null;
var HOME_BOUNDS=null, COLOR_MAP={}, DESC_MAP={};
var FILL_ALPHA = 0.8;
var BING_KEY_JS = null;
var SATELLITE_FALLBACK = null;
var ZOOM_THRESHOLD_JS = 12;
const ASSET_COLOR = "__ASSET_COLOR__";
var RENDERERS = { assets:null, geocodes:null, segments:null, env:null, groupstotal:null, assetstotal:null };

function logErr(m){ try{ window.pywebview.api.js_log(m); }catch(e){} }
function setError(msg){
  var e=document.getElementById('err');
  if (msg){ e.style.display='block'; e.textContent=msg; }
  else { e.style.display='none'; e.textContent=''; }
}
function fmtKm2(x){ return Number(x||0).toLocaleString('en-US',{maximumFractionDigits:2}); }

/* legend & chart omitted for brevity in comment—unchanged in logic */
function renderLegend(stats){
  var container=document.getElementById('legend');
  var codes=['A','B','C','D','E'], values={}, total=0;
  if (stats && Array.isArray(stats.labels) && Array.isArray(stats.values)){
    for (var i=0;i<stats.labels.length;i++){
      var code=String(stats.labels[i]||'').toUpperCase();
      var val =Number(stats.values[i]||0);
      values[code]=val; total+=val;
    }
  } else { codes.forEach(c=>values[c]=0); }
  function pct(x){ return Number(x||0).toLocaleString('en-US',{maximumFractionDigits:1}); }
  var html='<div style="font-weight:600;margin-bottom:6px;">Totals by sensitivity from basic mosaic</div>';
  html+='<table width=100%><thead><tr><th></th><th>Code</th><th>Description</th><th class="num">Area (km²)</th><th class="num">Share</th></tr></thead><tbody>';
  for (var k=0;k<codes.length;k++){
    var c=codes[k], color=(COLOR_MAP[c]||'#bdbdbd'), desc=(DESC_MAP[c]||''), km2=(values[c]||0), p=(total>0?(km2/total*100):0);
    html+='<tr><td><span class="swatch" style="background:'+color+'"></span></td><td style="width:48px;">'+c+'</td><td>'+desc+'</td><td class="num">'+fmtKm2(km2)+'</td><td class="num">'+pct(p)+'%</td></tr>';
  }
  html+='</tbody><tfoot><tr><td></td><td></td><td>Total</td><td class="num">'+fmtKm2(total)+'</td><td class="num">100%</td></tr></tfoot></table>';
  container.innerHTML=html;
}
function renderChart(stats){
  var ctx=document.getElementById('chart').getContext('2d');
  if (CHART) CHART.destroy();
  var labels=(stats&&stats.labels)?stats.labels:['A','B','C','D','E'];
  var values=(stats&&stats.values)?stats.values:[0,0,0,0,0];
  var colors=labels.map(c=>COLOR_MAP[c]||'#bdbdbd');
  CHART=new Chart(ctx,{type:'bar',data:{labels:labels,datasets:[{label:'km²',data:values,backgroundColor:colors,borderColor:'#fff',borderWidth:1}]},options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{display:false},title:{display:true,text:'Overall area by sensitivity code (km²)'}},scales:{y:{beginAtZero:true}}}});
  var r=document.getElementById('chart').getBoundingClientRect(); CHART.resize(Math.floor(r.width),Math.floor(r.height));
}

/* basemaps */
function tileXYToQuadKey(x,y,z){ var q=''; for(var i=z;i>0;i--){ var d=0,m=1<<(i-1); if((x&m)!==0)d+=1; if((y&m)!==0)d+=2; q+=d.toString(); } return q; }
var BingAerial = L.TileLayer.extend({ createTile: function(coords, done){ var img=document.createElement('img'); var zoom=this._getZoomForUrl(); if(!BING_KEY_JS){ img.onload=function(){done(null,img)}; img.onerror=function(){done(null,img)}; img.src='about:blank'; return img; } var q=tileXYToQuadKey(coords.x,coords.y,zoom); var t=(coords.x+coords.y)%4; var url='https://ecn.t'+t+'.tiles.virtualearth.net/tiles/a'+q+'.jpeg?g=1&n=z&key='+encodeURIComponent(BING_KEY_JS); img.crossOrigin='anonymous'; img.onload=function(){done(null,img)}; img.onerror=function(){done(null,img)}; img.src=url; return img; }});
function buildBaseSources(){
  var common={maxZoom:19,crossOrigin:true,tileSize:256};
  var osm = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {...common, attribution:'© OpenStreetMap'});
  var topo= L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {...common, subdomains:['a','b','c'], maxZoom:17, attribution:'© OpenStreetMap, © OpenTopoMap (CC-BY-SA)'});
  SATELLITE_FALLBACK = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {...common, attribution:'Esri, Maxar, Earthstar Geographics, and the GIS User Community'});
  var bing = new BingAerial({tileSize:256});
  return {osm:osm, topo:topo, sat_bing:bing, sat_esri:SATELLITE_FALLBACK};
}

/* opacity helper */
function _setOpacityMaybe(layer, alpha){
  if (!layer) return;
  if (typeof layer.setOpacity==='function'){ try{ layer.setOpacity(alpha); }catch(e){} }
  else if (typeof layer.setStyle==='function'){ try{ layer.setStyle({fillOpacity:alpha, opacity:alpha}); }catch(e){} }
}

/* gradients etc. */
function _hexToRgb(h){ h=h.replace('#',''); if(h.length===3){ h=h.split('').map(x=>x+x).join(''); } var n=parseInt(h,16); return {r:(n>>16)&255,g:(n>>8)&255,b:n&255}; }
function _rgbToHex(r,g,b){ return '#'+[r,g,b].map(v=>{var s=v.toString(16); return s.length===1?'0'+s:s;}).join(''); }
function _lerp(a,b,t){ return a+(b-a)*t; }
function rampBlue(t){ var light=_hexToRgb('#dbeafe'), dark=_hexToRgb('#1e3a8a'); var r=Math.round(_lerp(light.r,dark.r,t)), g=Math.round(_lerp(light.g,dark.g,t)), b=Math.round(_lerp(light.b,dark.b,t)); return _rgbToHex(r,g,b); }
function _normalize(val, vmin, vmax){ if (vmax===vmin){ return 1.0; } var t=(Number(val)-vmin)/(vmax-vmin); if (!isFinite(t)) t=0; return Math.max(0, Math.min(1, t)); }

/* loaders – unchanged */

function loadGeocodeIntoGroup(cat, preserveView){
  var prev=(preserveView && MAP)?{center:MAP.getCenter(), zoom:MAP.getZoom()}:null;
  window.pywebview.api.get_geocode_layer(cat).then(function(res){
    if (!res.ok){ setError('Failed to load geocode: '+res.error); return; }
    if (res.stats && res.stats.message){ setError(res.stats.message); } else { setError(''); }
    if (GEO_GROUP) GEO_GROUP.clearLayers();

    if (res.mbtiles && res.mbtiles.sensitivity_url){
      var opts={opacity:FILL_ALPHA, pane:'geocodePane', crossOrigin:true, noWrap:true, bounds:L.latLngBounds(res.home_bounds), minNativeZoom:(res.mbtiles.minzoom||0), maxNativeZoom:(res.mbtiles.maxzoom||19)};
      LAYER=L.tileLayer(res.mbtiles.sensitivity_url, opts); GEO_GROUP.addLayer(LAYER);
    } else if (res.geojson && res.geojson.features && res.geojson.features.length>0){
      LAYER=L.geoJSON(res.geojson, { style:function(f){ var c=(f.properties&&f.properties.sensitivity_code_max)||''; return {color:'white',weight:.5,opacity:1,fillOpacity:FILL_ALPHA,fillColor:(COLOR_MAP[c]||'#bdbdbd')}; }});
      GEO_GROUP.addLayer(LAYER);
    }

    HOME_BOUNDS = res.home_bounds || HOME_BOUNDS;
    if (!prev && HOME_BOUNDS){ MAP.fitBounds(HOME_BOUNDS, {padding:[20,20]}); }
    if (prev){ MAP.setView(prev.center, prev.zoom, {animate:false}); }

    renderLegend(res.stats); renderChart(res.stats);

    var chkEnv=document.getElementById('chkEnvIndex');
    var chkGT =document.getElementById('chkGroupsTotal');
    var chkAT =document.getElementById('chkAssetsTotal');
    var selEl=(document.getElementById('groupCatSel')||{});
    var envCat=selEl.value||cat||null;
    if (chkEnv && chkEnv.checked && envCat) loadEnvIndexIntoGroup(envCat, true);
    if (chkGT  && chkGT.checked  && envCat) loadGroupstotalIntoGroup(envCat, true);
    if (chkAT  && chkAT.checked  && envCat) loadAssetstotalIntoGroup(envCat, true);
  }).catch(function(err){ setError('API error: '+err); logErr(err); });
}

function loadEnvIndexIntoGroup(cat, preserveView){
  var prev=(preserveView && MAP)?{center:MAP.getCenter(), zoom:MAP.getZoom()}:null;
  window.pywebview.api.get_envindex_layer(cat).then(function(res){
    if (!res.ok){ setError('Failed to load env_index: '+res.error); return; }
    if (ENV_GROUP) ENV_GROUP.clearLayers();

    if (res.mbtiles && res.mbtiles.envindex_url){
      var opts={opacity:FILL_ALPHA, pane:'envPane', crossOrigin:true, noWrap:true, bounds: HOME_BOUNDS?L.latLngBounds(HOME_BOUNDS):null, minNativeZoom:(res.mbtiles.minzoom||0), maxNativeZoom:(res.mbtiles.maxzoom||19)};
      LAYER_ENV=L.tileLayer(res.mbtiles.envindex_url, opts); ENV_GROUP.addLayer(LAYER_ENV);
    } else if (res.geojson && res.geojson.features && res.geojson.features.length>0){
      function clamp(v,min,max){ return Math.max(min,Math.min(max,v)); }
      function h2r(h){ var s=h.replace('#',''); if(s.length===3)s=s.split('').map(x=>x+x).join(''); var n=parseInt(s,16); return {r:(n>>16)&255,g:(n>>8)&255,b:n&255}; }
      function r2h(r,g,b){ return '#'+[r,g,b].map(v=>{var h=v.toString(16); return h.length===1?'0'+h:h;}).join(''); }
      function lerp(a,b,t){ return a+(b-a)*t; }
      function envColor(x){ if(x==null||isNaN(x)) return '#cccccc'; var t=clamp(Number(x)/100,0,1), y=h2r('#ffff00'), r=h2r('#ff0000'); return r2h(Math.round(lerp(y.r,r.r,t)), Math.round(lerp(y.g,r.g,t)), Math.round(lerp(y.b,r.b,t))); }
      LAYER_ENV=L.geoJSON(res.geojson, { style:function(f){ var v=(f.properties&&f.properties.env_index!=null)?Number(f.properties.env_index):null; return {color:'white',weight:.5,opacity:1,fillOpacity:FILL_ALPHA,fillColor:envColor(v)}; }});
      ENV_GROUP.addLayer(LAYER_ENV);
    }
    if (prev){ MAP.setView(prev.center, prev.zoom, {animate:false}); }
  }).catch(function(err){ setError('API error: '+err); logErr(err); });
}

function loadGroupstotalIntoGroup(cat, preserveView){
  var prev=(preserveView && MAP)?{center:MAP.getCenter(), zoom:MAP.getZoom()}:null;
  window.pywebview.api.get_groupstotal_layer(cat).then(function(res){
    if (!res.ok){ setError('Failed to load groups total: '+res.error); return; }
    if (GROUPSTOTAL_GROUP) GROUPSTOTAL_GROUP.clearLayers();

    if (res.mbtiles && res.mbtiles.groupstotal_url){
      var opts={opacity:FILL_ALPHA, pane:'groupsTotalPane', crossOrigin:true, noWrap:true, bounds: HOME_BOUNDS?L.latLngBounds(HOME_BOUNDS):null, minNativeZoom:(res.mbtiles.minzoom||0), maxNativeZoom:(res.mbtiles.maxzoom||19)};
      LAYER_GROUPSTOTAL=L.tileLayer(res.mbtiles.groupstotal_url, opts); GROUPSTOTAL_GROUP.addLayer(LAYER_GROUPSTOTAL);
    } else if (res.geojson && res.geojson.features){
      var vmin=(res.range&&isFinite(res.range.min))?res.range.min:0;
      var vmax=(res.range&&isFinite(res.range.max))?res.range.max:1;
      LAYER_GROUPSTOTAL=L.geoJSON(res.geojson, {
        style:function(f){
          var v=(f.properties && f.properties.asset_groups_total!=null)?Number(f.properties.asset_groups_total):null;
          var t=(v==null||!isFinite(v))?0:_normalize(v, vmin, vmax);
          var col=rampBlue(t);
          return {color:'#ffffff', weight:.5, opacity:1, fillOpacity:FILL_ALPHA, fillColor:col};
        }
      });
      GROUPSTOTAL_GROUP.addLayer(LAYER_GROUPSTOTAL);
    }
    if (prev){ MAP.setView(prev.center, prev.zoom, {animate:false}); }
  }).catch(function(err){ setError('API error: '+err); logErr(err); });
}

function loadAssetstotalIntoGroup(cat, preserveView){
  var prev=(preserveView && MAP)?{center:MAP.getCenter(), zoom:MAP.getZoom()}:null;
  window.pywebview.api.get_assetstotal_layer(cat).then(function(res){
    if (!res.ok){ setError('Failed to load assets total: '+res.error); return; }
    if (ASSETSTOTAL_GROUP) ASSETSTOTAL_GROUP.clearLayers();

    if (res.mbtiles && res.mbtiles.assetstotal_url){
      var opts={opacity:FILL_ALPHA, pane:'assetsTotalPane', crossOrigin:true, noWrap:true, bounds: HOME_BOUNDS?L.latLngBounds(HOME_BOUNDS):null, minNativeZoom:(res.mbtiles.minzoom||0), maxNativeZoom:(res.mbtiles.maxzoom||19)};
      LAYER_ASSETSTOTAL=L.tileLayer(res.mbtiles.assetstotal_url, opts); ASSETSTOTAL_GROUP.addLayer(LAYER_ASSETSTOTAL);
    } else if (res.geojson && res.geojson.features){
      var vmin=(res.range&&isFinite(res.range.min))?res.range.min:0;
      var vmax=(res.range&&isFinite(res.range.max))?res.range.max:1;
      LAYER_ASSETSTOTAL=L.geoJSON(res.geojson, {
        style:function(f){
          var v=(f.properties && f.properties.assets_overlap_total!=null)?Number(f.properties.assets_overlap_total):null;
          var t=(v==null||!isFinite(v))?0:_normalize(v, vmin, vmax);
          var col=rampBlue(t);
          return {color:'#ffffff', weight:.5, opacity:1, fillOpacity:FILL_ALPHA, fillColor:col};
        }
      });
      ASSETSTOTAL_GROUP.addLayer(LAYER_ASSETSTOTAL);
    }
    if (prev){ MAP.setView(prev.center, prev.zoom, {animate:false}); }
  }).catch(function(err){ setError('API error: '+err); logErr(err); });
}

function loadSegmentsIntoGroup(lineNameOrAll){
  window.pywebview.api.get_segment_layer(lineNameOrAll).then(function(res){
    if (!res.ok){ setError('Failed to load segments: '+res.error); return; }
    if (!document.getElementById('err').textContent){ setError(''); }
    if (SEG_GROUP) SEG_GROUP.clearLayers();
    if (res.geojson && res.geojson.features && res.geojson.features.length>0){
      LAYER_SEG=L.geoJSON(res.geojson, { style:function(f){ var c=(f.properties&&f.properties.sensitivity_code_max)||''; return {color:'#f7f7f7',weight:.7,opacity:1,fillOpacity:FILL_ALPHA,fillColor:(COLOR_MAP[c]||'#bdbdbd')}; }});
      SEG_GROUP.addLayer(LAYER_SEG);
    }
  }).catch(function(err){ setError('API error: '+err); logErr(err); });
}
function loadAssetsIntoGroup(){
  window.pywebview.api.get_assets_layer().then(function(res){
    if (!res.ok){ setError('Failed to load assets: '+res.error); return; }
    if (ASSET_GROUP) ASSET_GROUP.clearLayers();
    if (res.geojson && res.geojson.features && res.geojson.features.length>0){
      LAYER_ASSETS=L.geoJSON(res.geojson,{ style:function(){ return {color:ASSET_COLOR,weight:.6,opacity:FILL_ALPHA,fillOpacity:FILL_ALPHA,fillColor:ASSET_COLOR}; }, pane:'assetsPane',
        pointToLayer:function(ft,latlng){ return L.circleMarker(latlng,{pane:'assetsPane',radius:3.5,color:ASSET_COLOR,weight:.8,opacity:FILL_ALPHA,fillOpacity:FILL_ALPHA,fillColor:ASSET_COLOR}); }});
      ASSET_GROUP.addLayer(LAYER_ASSETS);
    }
  }).catch(function(err){ setError('API error: '+err); logErr(err); });
}

/* helpers */
function ensureOnMap(g){ if (g && !MAP.hasLayer(g)) g.addTo(MAP); }
function ensureOffMap(g){ if (g && MAP.hasLayer(g)) MAP.removeLayer(g); }

/* ---- NEW: utility to remove the checkbox next to "Geocode group" ---- */
function hideGeocodeFolderCheckbox(ctrl){
  try{
    var ctn = ctrl.getContainer();
    var sel = ctn.querySelector('#groupCatSel');
    if (!sel) return;
    var label = sel.closest('label') || (sel.parentElement && sel.parentElement.closest && sel.parentElement.closest('label'));
    if (!label) return;
    var cb = label.querySelector('input.leaflet-control-layers-selector');
    if (cb) cb.remove();                  // remove the actual checkbox node
    label.classList.add('no-toggle');     // fix spacing via CSS
  }catch(e){ logErr('hideGeocodeFolderCheckbox: '+e); }
}

/* build layers control */
function buildLayersControl(state){
  var baseLayers={
    'OpenStreetMap': BASE_SOURCES.osm,
    'OSM Topography': BASE_SOURCES.topo,
    'Satellite': (BING_KEY_JS?BASE_SOURCES.sat_bing:SATELLITE_FALLBACK)
  };

  SEG_GROUP=L.layerGroup(); ASSET_GROUP=L.layerGroup();
  GEO_GROUP=L.layerGroup(); ENV_GROUP=L.layerGroup();
  GROUPSTOTAL_GROUP=L.layerGroup(); ASSETSTOTAL_GROUP=L.layerGroup();
  GEO_FOLDER=L.layerGroup();

  var folderLabel='Geocode group <div class="inlineSel"><select id="groupCatSel"></select></div>' +
                  '<div class="inlineChecks">' +
                  '<label><input type="checkbox" id="chkGeoAreas" checked> Sensitive areas</label>' +
                  '<label><input type="checkbox" id="chkEnvIndex"> Environment index</label>' +
                  '<label><input type="checkbox" id="chkGroupsTotal"> Groups total</label>' +
                  '<label><input type="checkbox" id="chkAssetsTotal"> Assets total</label>' +
                  '</div>';

  var overlays={};
  overlays['Sensitivity lines <div class="inlineSel"><select id="segLineSel"></select></div>']=SEG_GROUP;
  overlays[folderLabel]=GEO_FOLDER;  // we will hide its checkbox
  overlays['Assets']=ASSET_GROUP;

  var ctrl=L.control.layers(baseLayers, overlays, { collapsed:false, position:'topleft' }).addTo(MAP);

  /* Hide/strip the Geocode folder checkbox robustly (now and shortly after in case Leaflet reflows) */
  hideGeocodeFolderCheckbox(ctrl);
  setTimeout(function(){ hideGeocodeFolderCheckbox(ctrl); }, 50);
  setTimeout(function(){ hideGeocodeFolderCheckbox(ctrl); }, 300);

  /* keep basemaps at bottom (unchanged) */
  try {
    var ctn  = ctrl.getContainer();
    var form = ctn.querySelector('.leaflet-control-layers-list');
    if (form){
      var base      = form.querySelector('.leaflet-control-layers-base');
      var sep       = form.querySelector('.leaflet-control-layers-separator');
      if (base){
        if (sep && sep.parentNode === form) form.removeChild(sep);
        if (base.parentNode === form) form.removeChild(base);
        var overlaysN = form.querySelector('.leaflet-control-layers-overlays');
        if (overlaysN){ form.appendChild(overlaysN); }
        if (sep) form.appendChild(sep);
        form.appendChild(base);
        base.style.marginTop = '6px';
      }
    }
  } catch(e){ logErr('Layer reorder failed: ' + e); }

  BASE_SOURCES.osm.addTo(MAP);
  GEO_GROUP.addTo(MAP);
  SEG_GROUP.addTo(MAP);

  setTimeout(function(){
    var geocodes=(state.geocode_categories||[]);
    var sel=document.getElementById('groupCatSel');
    if (sel){
      sel.innerHTML='';
      for (var i=0;i<geocodes.length;i++){
        var o=document.createElement('option'); o.value=geocodes[i]; o.textContent=geocodes[i]; sel.appendChild(o);
      }
      if (state.initial_geocode && geocodes.indexOf(state.initial_geocode)>=0){ sel.value=state.initial_geocode; }
    }

    var initialCat=(sel && sel.value) ? sel.value : state.initial_geocode;
    if (initialCat){ loadGeocodeIntoGroup(initialCat, false); }

    var chkAreas=document.getElementById('chkGeoAreas');
    var chkEnv  =document.getElementById('chkEnvIndex');
    var chkGT   =document.getElementById('chkGroupsTotal');
    var chkAT   =document.getElementById('chkAssetsTotal');

    if (chkAreas && chkAreas.checked){ ensureOnMap(GEO_GROUP); } else { ensureOffMap(GEO_GROUP); }
    if (chkEnv   && chkEnv.checked)  { ensureOnMap(ENV_GROUP); if (initialCat) loadEnvIndexIntoGroup(initialCat, true); } else { ensureOffMap(ENV_GROUP); }
    if (chkGT    && chkGT.checked)   { ensureOnMap(GROUPSTOTAL_GROUP); if (initialCat) loadGroupstotalIntoGroup(initialCat, true); } else { ensureOffMap(GROUPSTOTAL_GROUP); }
    if (chkAT    && chkAT.checked)   { ensureOnMap(ASSETSTOTAL_GROUP); if (initialCat) loadAssetstotalIntoGroup(initialCat, true); } else { ensureOffMap(ASSETSTOTAL_GROUP); }

    sel && sel.addEventListener('change', function(){
      var cat=sel.value||initialCat;
      loadGeocodeIntoGroup(cat, true);
      if (chkEnv && chkEnv.checked) loadEnvIndexIntoGroup(cat, true);
      if (chkGT  && chkGT.checked)  loadGroupstotalIntoGroup(cat, true);
      if (chkAT  && chkAT.checked)  loadAssetstotalIntoGroup(cat, true);
    });

    var segsel=document.getElementById('segLineSel');
    if (segsel){
      segsel.innerHTML='';
      var names=state.segment_line_names||[];
      if (names.length>0){
        var all=document.createElement('option'); all.value="__ALL__"; all.textContent="All lines"; segsel.appendChild(all);
        for (var j=0;j<names.length;j++){ var so=document.createElement('option'); so.value=names[j]; so.textContent=names[j]; segsel.appendChild(so); }
      } else {
        var s=document.createElement('option'); s.value="__ALL__"; s.textContent="All segments"; segsel.appendChild(s);
        var segcats=state.segment_categories||[]; for (var k=0;k<segcats.length;k++){ var sc=document.createElement('option'); sc.value=segcats[k]; sc.textContent=segcats[k]; segsel.appendChild(sc); }
      }
      loadSegmentsIntoGroup(segsel.value||"__ALL__");
      segsel.addEventListener('change', function(){ loadSegmentsIntoGroup(segsel.value||"__ALL__"); });
    }

    function currentCat(){ return (document.getElementById('groupCatSel')||{}).value||initialCat; }
    if (chkAreas){
      chkAreas.addEventListener('change', function(){
        var cat=currentCat();
        if (chkAreas.checked){ ensureOnMap(GEO_GROUP); if (cat) loadGeocodeIntoGroup(cat, true); }
        else { ensureOffMap(GEO_GROUP); GEO_GROUP.clearLayers(); }
      });
    }
    if (chkEnv){
      chkEnv.addEventListener('change', function(){
        var cat=currentCat();
        if (chkEnv.checked){ ensureOnMap(ENV_GROUP); if (cat) loadEnvIndexIntoGroup(cat, true); }
        else { ensureOffMap(ENV_GROUP); ENV_GROUP.clearLayers(); }
      });
    }
    if (chkGT){
      chkGT.addEventListener('change', function(){
        var cat=currentCat();
        if (chkGT.checked){ ensureOnMap(GROUPSTOTAL_GROUP); if (cat) loadGroupstotalIntoGroup(cat, true); }
        else { ensureOffMap(GROUPSTOTAL_GROUP); GROUPSTOTAL_GROUP.clearLayers(); }
      });
    }
    if (chkAT){
      chkAT.addEventListener('change', function(){
        var cat=currentCat();
        if (chkAT.checked){ ensureOnMap(ASSETSTOTAL_GROUP); if (cat) loadAssetstotalIntoGroup(cat, true); }
        else { ensureOffMap(ASSETSTOTAL_GROUP); ASSETSTOTAL_GROUP.clearLayers(); }
      });
    }
  },0);

  return ctrl;
}

/* Boot */
function boot(){
  MAP = L.map('map', {
    zoomControl:false, preferCanvas:true,
    maxBounds: L.latLngBounds([[-85,-180],[85,180]]),
    maxBoundsViscosity: 1.0,
    worldCopyJump: false
  });
  L.control.zoom({ position:'topright' }).addTo(MAP);

  MAP.createPane('segmentsPane');      MAP.getPane('segmentsPane').style.zIndex=650;
  MAP.createPane('envPane');           MAP.getPane('envPane').style.zIndex=600;
  MAP.createPane('geocodePane');       MAP.getPane('geocodePane').style.zIndex=550;
  MAP.createPane('groupsTotalPane');   MAP.getPane('groupsTotalPane').style.zIndex=560;
  MAP.createPane('assetsTotalPane');   MAP.getPane('assetsTotalPane').style.zIndex=540;
  MAP.createPane('assetsPane');        MAP.getPane('assetsPane').style.zIndex=450;

  try {
    RENDERERS.segments    = L.canvas({ pane:'segmentsPane' });
    RENDERERS.env         = L.canvas({ pane:'envPane'      });
    RENDERERS.geocodes    = L.canvas({ pane:'geocodePane'  });
    RENDERERS.groupstotal = L.canvas({ pane:'groupsTotalPane' });
    RENDERERS.assetstotal = L.canvas({ pane:'assetsTotalPane' });
    RENDERERS.assets      = L.canvas({ pane:'assetsPane'   });
  } catch(e){ logErr('Canvas renderers creation failed: '+e); }

  L.control.scale({ position:'bottomleft', metric:true, imperial:false, maxWidth:200 }).addTo(MAP);

  BASE_SOURCES = buildBaseSources();
  BASE_SOURCES.osm.addTo(MAP);
  MAP.setView([0,0], 2);

  window.pywebview.api.get_state().then(function(state){
    COLOR_MAP=state.colors||{}; DESC_MAP=state.descriptions||{};
    BING_KEY_JS=(state.bing_key||'').trim()||null; ZOOM_THRESHOLD_JS=state.zoom_threshold||12;

    renderLegend(null);
    renderChart({labels:['A','B','C','D','E'], values:[0,0,0,0,0]});

    buildLayersControl(state);

    document.getElementById('homeBtn').addEventListener('click', function(){
      if (HOME_BOUNDS) MAP.fitBounds(HOME_BOUNDS, { padding:[20,20] });
    });

    document.getElementById('exportBtn').addEventListener('click', function(){
      var node=document.getElementById('map');
      node.classList.add('exporting');
      html2canvas(node,{useCORS:true,allowTaint:false,backgroundColor:'#ffffff',scale:3})
      .then(function(canvas){ node.classList.remove('exporting'); return window.pywebview.api.save_png(canvas.toDataURL('image/png')); })
      .then(function(res){ if(!res.ok){ setError('Save cancelled or failed: '+(res.error||'')); setTimeout(function(){ setError(''); },4000); } })
      .catch(function(err){ node.classList.remove('exporting'); setError('Export failed.'); logErr(err); setTimeout(function(){ setError(''); },5000); });
    });

    document.getElementById('exitBtn').addEventListener('click', function(){
      if (window.pywebview && window.pywebview.api && window.pywebview.api.exit_app){
        window.pywebview.api.exit_app();
      }
    });

    var slider=document.getElementById('opacity'), opv=document.getElementById('opv');
    slider.addEventListener('input', function(){
      var v=parseInt(slider.value,10); FILL_ALPHA=v/100; opv.textContent=v+'%';
      _setOpacityMaybe(LAYER,FILL_ALPHA);
      _setOpacityMaybe(LAYER_ENV,FILL_ALPHA);
      _setOpacityMaybe(LAYER_GROUPSTOTAL,FILL_ALPHA);
      _setOpacityMaybe(LAYER_ASSETSTOTAL,FILL_ALPHA);
      _setOpacityMaybe(LAYER_SEG,FILL_ALPHA);
      if (LAYER_ASSETS && typeof LAYER_ASSETS.setStyle==='function'){ LAYER_ASSETS.setStyle({fillOpacity:FILL_ALPHA, opacity:FILL_ALPHA}); }
    });
  }).catch(function(err){
    setError('Failed to get state: '+err); logErr('get_state failed: '+err);
  });
}

window.addEventListener('pywebviewready', function(){ boot(); });
document.addEventListener('DOMContentLoaded', function(){ if (window.pywebview && window.pywebview.api) { boot(); } });
</script>
</body>
</html>
"""

HTML = HTML_TEMPLATE.replace("__ASSET_COLOR__", STEEL_BLUE)

if __name__ == "__main__":
    window = webview.create_window(
        title="Maps overview",
        html=HTML,
        js_api=api,
        width=1300, height=800, resizable=True
    )
    webview.start(gui="edgechromium", debug=False)
