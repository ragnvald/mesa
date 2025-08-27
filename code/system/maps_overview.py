#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, base64, configparser, locale
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping

# Require pywebview in the environment that launches this script
try:
    import webview  # pip install pywebview
except ModuleNotFoundError:
    sys.stderr.write(
        "ERROR: 'pywebview' is not installed in the Python environment launching maps_overview.py.\n"
        "Install it in that environment, e.g.:  pip install pywebview\n"
    )
    sys.exit(1)

# Accurate geodesic areas for stats
from pyproj import Geod
try:
    from shapely.validation import make_valid as _make_valid
except Exception:
    _make_valid = None

# ===============================
# Locale (safe)
# ===============================
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except Exception:
    pass

# ===============================
# Paths / constants
# ===============================
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE      = os.path.join(BASE_DIR, "../system/config.ini")
PARQUET_FILE     = os.path.join(BASE_DIR, "../output/geoparquet/tbl_flat.parquet")           # stats source & map polygons
SEGMENT_FILE     = os.path.join(BASE_DIR, "../output/geoparquet/tbl_segment_flat.parquet")
ASSET_FILE       = os.path.join(BASE_DIR, "../output/geoparquet/tbl_asset_object.parquet")
LINES_FILE       = os.path.join(BASE_DIR, "../output/geoparquet/tbl_lines.parquet")          # human-readable names + name_gis
PLOT_CRS         = "EPSG:4326"
BASIC_GROUP_NAME = "basic_mosaic"   # stats always from here (in tbl_flat)
ZOOM_THRESHOLD   = 10               # tooltips for geocode/env/segments appear when map zoom >= this
STEEL_BLUE       = "#4682B4"        # assets color (semi-transparent via fillOpacity)

# ===============================
# Config / colors
# ===============================
def read_config(path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8")
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
# Data helpers
# ===============================
def load_parquet(path: str) -> gpd.GeoDataFrame:
    """Load a GeoParquet if it exists; return empty GeoDataFrame otherwise."""
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
    """Normalize to map CRS (EPSG:4326) for rendering/leaflet bounds."""
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
    """Ensure EPSG:4326 (WGS84) for geodesic area computation."""
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
    return [[miny, minx],[maxy, maxx]]  # [[south, west],[north, east]]

def gdf_to_geojson_min(gdf: gpd.GeoDataFrame) -> dict:
    """GeoJSON with compact, tooltip-friendly properties for geocodes/segments."""
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
    """GeoJSON for assets with tooltip props including geodesic area (for polygons)."""
    g = to_epsg4326(gdf, cfg)
    feats = []
    for _, row in g.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        # Compute area_km2 for polygons; for points/lines keep None
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
    """GeoJSON for env_index overlay (expects column env_index in gdf)."""
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
    """
    Input: tbl_flat (already loaded as GeoDataFrame with geometry).
    Filter to BASIC_GROUP_NAME, dedupe, compute WGS84 geodesic area, sum to km² by sensitivity.
    """
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

    # dedupe to avoid double counting overlapping rows in tbl_flat
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
        out.append(a_m2 / 1e6)  # km²
    return {"labels": labels, "values": out}

# ===============================
# Load datasets (robust, non-fatal)
# ===============================
cfg          = read_config(CONFIG_FILE)
COLS         = get_color_mapping(cfg)
DESC         = get_desc_mapping(cfg)

# GDF is tbl_flat (stats & map polygons)
GDF          = load_parquet(PARQUET_FILE)
SEG_GDF      = load_parquet(SEGMENT_FILE)
ASSET_GDF    = load_parquet(ASSET_FILE)
LINES_GDF    = load_parquet(LINES_FILE)  # names + name_gis primary key

GEOCODE_AVAILABLE   = (not GDF.empty) and ("name_gis_geocodegroup" in GDF.columns)
SEGMENTS_AVAILABLE  = (not SEG_GDF.empty) and ("geometry" in SEG_GDF.columns)
ASSETS_AVAILABLE    = (not ASSET_GDF.empty) and ("geometry" in ASSET_GDF.columns)
ENVINDEX_AVAILABLE  = GEOCODE_AVAILABLE and ("env_index" in GDF.columns)

CATS = sorted(GDF["name_gis_geocodegroup"].dropna().unique().tolist()) if GEOCODE_AVAILABLE else []
if SEGMENTS_AVAILABLE and "name_gis_geocodegroup" in SEG_GDF.columns:
    SEG_CATS = sorted(SEG_GDF["name_gis_geocodegroup"].dropna().unique().tolist())
else:
    SEG_CATS = []

# Build mapping from human-readable line name -> name_gis (PK)
LINE_NAMES = []
LINE_USER_TO_GIS = {}  # { name_user -> set(name_gis) }
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
        try:
            print(f"[JS] {message}")
        except Exception:
            pass

    def get_state(self):
        return {
            "geocode_available": GEOCODE_AVAILABLE,
            "geocode_categories": CATS if GEOCODE_AVAILABLE else [],
            "segment_available": SEGMENTS_AVAILABLE,
            "segment_categories": SEG_CATS if SEGMENTS_AVAILABLE else [],  # fallback
            "segment_line_names": LINE_NAMES,  # from tbl_lines.name_user
            "assets_available": ASSETS_AVAILABLE,
            "envindex_available": ENVINDEX_AVAILABLE,
            "colors": COLS,
            "descriptions": DESC,
            "initial_geocode": (CATS[0] if (GEOCODE_AVAILABLE and CATS) else None),
            "has_segments": SEGMENTS_AVAILABLE,
            "bing_key": BING_KEY,
            "zoom_threshold": ZOOM_THRESHOLD
        }

    def get_geocode_layer(self, geocode_category):
        """Return selected geocode layer for map + stats from tbl_flat:basic_mosaic."""
        try:
            if not GEOCODE_AVAILABLE:
                map_geojson = {"type":"FeatureCollection","features":[]}
                map_bounds  = [[0,0],[0,0]]
            else:
                df_map = GDF[GDF["name_gis_geocodegroup"] == geocode_category].copy()
                df_map = only_A_to_E(df_map)
                df_map = to_plot_crs(df_map, cfg)
                map_bounds = bounds_to_leaflet(df_map.total_bounds) if not df_map.empty else [[0,0],[0,0]]
                map_geojson = gdf_to_geojson_min(df_map)

            stats = compute_stats_by_geodesic_area_from_flat_basic(GDF, cfg)

            return {
                "ok": True,
                "geojson": map_geojson,
                "home_bounds": map_bounds,
                "stats": stats
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_envindex_layer(self, geocode_category):
        """Return env_index overlay (0–100) for a given geocode category from tbl_flat."""
        try:
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

    def get_segment_layer(self, seg_line_name_or_all):
        """
        Filter sensitivity segments (tbl_segment_flat) using tbl_lines relation:
        UI passes tbl_lines.name_user -> map to tbl_lines.name_gis; filter segments on name_gis.
        '__ALL__' => return all segments.
        """
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
                        allowed = {sel}  # tolerate direct name_gis passed through
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
        """Return assets overlay (steel blue, drawn just above basemap) with tooltips."""
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
        """Receive data URL from JS and save PNG."""
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
  .info-block {
    padding:8px 12px;
    font-size:12px;
    line-height:1.35;
    border-top:1px solid #2b344211;
    border-bottom:1px solid #2b344211;
    background:#0f172a08;
    max-height:170px;
    overflow:auto;
  }
  .info-block p { margin:2px 0 4px 0; }
  .swatch { display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:6px; border:1px solid #8884; }
  .num { text-align:right; white-space:nowrap; }
  .btn { padding:6px 10px; border:1px solid #ccd; background:#fff; border-radius:6px; cursor:pointer; }
  .btn:active { transform:translateY(1px); }
  .slider { display:flex; align-items:center; gap:8px; }
  .slider input[type=range]{ width:160px; }

  /* Leaflet Layers control tweaks and tooltips */
  .leaflet-control-layers { font-size: 12px; }
  .leaflet-control-layers label { display:block; line-height: 1.35; }
  .inlineSel { margin:4px 0 6px 22px; }
  .help {
    display:inline-flex; align-items:center; justify-content:center;
    width:15px; height:15px; border-radius:50%;
    border:1px solid #96a1b4; color:#435061; font-size:10px; line-height:1;
    cursor:default; position:relative; background:#f7fafc; margin-left:6px;
  }
  .help::after {
    content: attr(data-tip);
    position:absolute; left:20px; top:50%; transform:translateY(-50%);
    background:#0f172a; color:#e5e7eb; border:1px solid #1f2937;
    border-radius:6px; padding:6px 8px; font-size:12px; line-height:1.25;
    min-width:220px; max-width:320px; white-space:normal;
    box-shadow:0 8px 24px rgba(0,0,0,.25);
    opacity:0; pointer-events:none; transition:opacity .12s ease;
    z-index:2000;
  }
  .help:hover::after { opacity:1; }

  /* chart/tooltip */
  #chartBox { flex:1 1 auto; padding:8px 12px; position:relative; overflow:hidden; }
  #chart { position:absolute; top:8px; right:12px; bottom:8px; left:12px; display:block; }

  .leaflet-tooltip.poly-tip {
    background: #0f172a; color: #e5e7eb; border: 1px solid #1f2937;
    border-radius: 8px; box-shadow: 0 6px 20px rgba(0,0,0,.25);
    padding: 10px 12px; font-size: 12px; line-height: 1.25; max-width: 280px;
  }
  .poly-tip .hdr { font-size: 12px; font-weight: 700; letter-spacing: .2px; margin-bottom: 6px; display:flex; align-items:center; gap:8px; justify-content:space-between; }
  .poly-tip .chip { display:inline-block; font-weight:700; padding:2px 6px; border-radius:6px; background: rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.2); }
  .poly-tip .kv { display: grid; grid-template-columns: 92px 1fr; gap: 4px 10px; }
  .poly-tip .k  { opacity: .8; }
  .poly-tip .v  { text-align: right; white-space: nowrap; }
</style>
</head>
<body>
<div class="wrap">
  <div class="bar">
    <button id="homeBtn" class="btn">Home</button>

    <div class="slider">
      <span class="label-sm">Opacity</span>
      <input id="opacity" type="range" min="10" max="100" value="80">
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
      <p>1. Geocode areas are colored by sensitivity A (very high) to E (very low).</p>
      <p>2. Bar chart shows total km² per sensitivity for the baseline group 'basic_mosaic'.</p>
      <p>3. Percentages in the legend use geodesic (WGS84) area for accuracy.</p>
      <p>4. Use the two dropdowns in the Layers box to pick geocode categories for Sensitivity and Env. index independently.</p>
      <p>5. Enable 'Sensitivity lines' to view buffered line-based segment polygons.</p>
      <p>6. Enable 'Assets' to overlay individual asset geometries in steel blue (hover for details).</p>
      <p>7. Enable 'Environment index' for a yellow→red heat style based on env_index (0–100).</p>
      <p>8. Adjust the Opacity slider to reveal more basemap context beneath fills.</p>
      <p>9. Hover a polygon (zoom ≥ threshold) to see a tooltip with details. Asset tooltips show at any zoom.</p>
      <p>10. Use Home to reset the view to the active geocode category extent.</p>
      <p>11. Export PNG creates a high-resolution snapshot; try different basemaps.</p>
    </div>
    <div id="chartBox"><canvas id="chart"></canvas></div>
  </div>
</div>

<script>
var MAP=null, BASE=null, BASE_SOURCES=null, CHART=null;
var GEO_GROUP=null, SEG_GROUP=null, ASSET_GROUP=null, ENV_GROUP=null;
var LAYER=null, LAYER_SEG=null, LAYER_ASSETS=null, LAYER_ENV=null;
var HOME_BOUNDS=null, COLOR_MAP={}, DESC_MAP={};
var FILL_ALPHA = 0.8;
var BING_KEY_JS = null;
var SATELLITE_FALLBACK = null;
var ZOOM_THRESHOLD_JS = 12;  // for geocode/env/segments tooltips only
const ASSET_COLOR = "__ASSET_COLOR__";

/* Per-pane Canvas renderers (critical for correct stacking) */
var RENDERERS = { assets:null, geocodes:null, segments:null, env:null };

function logErr(m){ try { window.pywebview.api.js_log(m); } catch(e){} }
function setError(msg){
  var e = document.getElementById('err');
  if (msg){ e.style.display='block'; e.textContent = msg; }
  else { e.style.display='none'; e.textContent=''; }
}

function fmtKm2(x){ return Number(x||0).toLocaleString('en-US', {maximumFractionDigits:2}); }

/* ---------- Tooltips ---------- */
function buildTipHTML(props){
  const code = (props && props.sensitivity_code_max) ? String(props.sensitivity_code_max).toUpperCase() : '?';
  const desc = (DESC_MAP && DESC_MAP[code]) ? DESC_MAP[code] : '';
  const name = props && props.name_asset_object ? String(props.name_asset_object) : null;
  const oid  = props && props.id_asset_object   ? String(props.id_asset_object)   : null;
  const area = props && props.area_km2 != null  ? fmtKm2(props.area_km2)          : '—';
  const chipColor = (COLOR_MAP && COLOR_MAP[code]) ? COLOR_MAP[code] : '#bdbdbd';
  const chipStyle = 'background:' + chipColor + '22;border-color:' + chipColor + '55;color:#fff;text-shadow:0 1px 0 #0003;';
  return `
    <div class="poly-tip">
      <div class="hdr">
        <span>${name ? name.replace(/</g,'&lt;') : 'Polygon'}</span>
        <span class="chip" style="${chipStyle}">${code}</span>
      </div>
      <div class="kv">
        <div class="k">Sensitivity</div><div class="v">${desc || code}</div>
        ${oid ? `<div class="k">Object ID</div><div class="v">${oid}</div>` : ``}
        <div class="k">Area</div><div class="v">${area} km²</div>
      </div>
    </div>`;
}

/* Asset-specific tooltip: always show (no zoom gating) */
function buildAssetTipHTML(props){
  const name = props && props.name_asset_object ? String(props.name_asset_object) : 'Asset';
  const oid  = props && props.id_asset_object   ? String(props.id_asset_object)   : null;
  const area = props && props.area_km2 != null  ? fmtKm2(props.area_km2)          : '—';
  return `
    <div class="poly-tip">
      <div class="hdr"><span>${name.replace(/</g,'&lt;')}</span></div>
      <div class="kv">
        ${oid ? `<div class="k">Object ID</div><div class="v">${oid}</div>` : ``}
        <div class="k">Area</div><div class="v">${area} km²</div>
      </div>
    </div>`;
}

/* Env index tooltip */
function buildEnvTipHTML(props){
  const gi  = props && props.geocode_group ? String(props.geocode_group) : '';
  const ei  = (props && props.env_index != null) ? Number(props.env_index) : null;
  const area= props && props.area_km2 != null ? fmtKm2(props.area_km2) : '—';
  return `
    <div class="poly-tip">
      <div class="hdr"><span>Environment index</span><span class="chip">${(ei!=null)? ei.toFixed(1) : '—'}</span></div>
      <div class="kv">
        ${gi ? `<div class="k">Geocode</div><div class="v">${gi}</div>` : ``}
        <div class="k">Area</div><div class="v">${area} km²</div>
      </div>
    </div>`;
}

function getTT(layer){ return (layer && typeof layer.getTooltip === 'function') ? layer.getTooltip() : null; }
function ttIsOpen(layer){ const tt = getTT(layer); return !!(tt && typeof tt.isOpen === 'function' && tt.isOpen()); }
function ttCloseIfAny(layer){ if (ttIsOpen(layer)) layer.closeTooltip(); }

function attachPolygonTooltip(layer, feature){
  layer.on('mouseover', function () {
    if (MAP.getZoom() >= ZOOM_THRESHOLD_JS) {
      layer.setStyle({ weight: 1.2 });
      const html = buildTipHTML((feature && feature.properties) || {});
      if (!layer.getTooltip()){ layer.bindTooltip(html, { className:'poly-tip', sticky:true, direction:'auto', opacity:0.98 }); }
      else { layer.setTooltipContent(html); }
      layer.openTooltip();
    }
  });
  layer.on('mousemove', function () {
    if (ttIsOpen(layer) && MAP.getZoom() < ZOOM_THRESHOLD_JS) { layer.closeTooltip(); }
  });
  layer.on('mouseout', function () {
    layer.setStyle({ weight: 0.5 });
    ttCloseIfAny(layer);
  });
}

/* Always-on hover tooltips for assets (points and polygons) */
function attachAssetTooltip(layer, feature){
  layer.on('mouseover', function () {
    const html = buildAssetTipHTML((feature && feature.properties) || {});
    if (!layer.getTooltip()){ layer.bindTooltip(html, { className:'poly-tip', sticky:true, direction:'auto', opacity:0.98, pane:'assetsPane' }); }
    else { layer.setTooltipContent(html); }
    layer.openTooltip();
  });
  layer.on('mouseout', function () { ttCloseIfAny(layer); });
}

function attachEnvTooltip(layer, feature){
  layer.on('mouseover', function () {
    if (MAP.getZoom() >= ZOOM_THRESHOLD_JS) {
      const html = buildEnvTipHTML((feature && feature.properties) || {});
      if (!layer.getTooltip()){ layer.bindTooltip(html, { className:'poly-tip', sticky:true, direction:'auto', opacity:0.98 }); }
      else { layer.setTooltipContent(html); }
      layer.openTooltip();
    }
  });
  layer.on('mousemove', function () {
    if (ttIsOpen(layer) && MAP.getZoom() < ZOOM_THRESHOLD_JS) { layer.closeTooltip(); }
  });
  layer.on('mouseout', function () { ttCloseIfAny(layer); });
}

/* ---------- Legend / chart ---------- */
function renderLegend(stats) {
  var container = document.getElementById('legend');
  var codes = ['A','B','C','D','E'];
  var values = {};
  var total = 0;

  if (stats && Array.isArray(stats.labels) && Array.isArray(stats.values)) {
    for (var i=0;i<stats.labels.length;i++){
      var code = String(stats.labels[i] || '').toUpperCase();
      var val  = Number(stats.values[i] || 0);
      values[code] = val;
      total += val;
    }
  } else {
    for (var j=0;j<codes.length;j++) values[codes[j]] = 0;
  }

  function fmtPct(x){ return Number(x||0).toLocaleString('en-US', {maximumFractionDigits:1}); }

  var html = '<div style="font-weight:600; margin-bottom:6px;">Totals by sensitivity</div>';
  html += '<table><thead><tr><th></th><th>Code</th><th>Description</th><th class="num">Area (km²)</th><th class="num">Share</th></tr></thead><tbody>';

  for (var k=0;k<codes.length;k++){
    var c = codes[k];
    var color = COLOR_MAP[c] || '#bdbdbd';
    var desc  = DESC_MAP[c] || '';
    var valKm2 = values[c] || 0;
    var pct = total > 0 ? (valKm2 / total * 100.0) : 0;
    html += '<tr>' +
      '<td><span class="swatch" style="background:'+color+'"></span></td>' +
      '<td style="width:48px;">'+c+'</td>' +
      '<td>'+desc+'</td>' +
      '<td class="num">'+fmtKm2(valKm2)+'</td>' +
      '<td class="num">'+fmtPct(pct)+'%</td>' +
    '</tr>';
  }
  html += '</tbody><tfoot><tr><td></td><td></td><td>Total</td><td class="num">'+fmtKm2(total)+'</td><td class="num">100%</td></tr></tfoot></table>';

  container.innerHTML = html;
}

function renderChart(stats) {
  var ctx = document.getElementById('chart').getContext('2d');
  if (CHART) CHART.destroy();

  var labels = (stats && stats.labels) ? stats.labels : ['A','B','C','D','E'];
  var values = (stats && stats.values) ? stats.values : [0,0,0,0,0];

  var colors = [];
  for (var i=0;i<labels.length;i++){
    var code = labels[i];
    colors.push(COLOR_MAP[code] || '#bdbdbd');
  }

  CHART = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{ label: 'km²', data: values, backgroundColor: colors, borderColor: '#ffffff', borderWidth: 1 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, resizeDelay: 0, animation: false,
      plugins: { legend: { display:false }, title: { display:true, text:'Overall area by sensitivity code (km²)' } },
      scales: { y: { beginAtZero:true, ticks:{ padding:6 } }, x:{ ticks:{ padding:4 } } },
      layout: { padding: { left: 0, right:0, top:0, bottom:0 } }
    }
  });

  var canvas = document.getElementById('chart');
  var r = canvas.getBoundingClientRect();
  CHART.resize(Math.floor(r.width), Math.floor(r.height));
}

/* ---------- Resize handling ---------- */
function debounce(fn, ms){ var t; return function(){ var ctx=this, args=arguments; clearTimeout(t); t=setTimeout(function(){ fn.apply(ctx,args); }, ms||50); }; }
function resizeChartToBox(){
  if (!CHART) return;
  var canvas = document.getElementById('chart');
  var r = canvas.getBoundingClientRect();
  CHART.resize(Math.floor(r.width), Math.floor(r.height));
}
function setupResizeObservers(){
  var chartBox = document.getElementById('chartBox');
  var mapBox   = document.getElementById('map');
  var onChartResize = debounce(function(){ resizeChartToBox(); }, 25);
  var onMapResize   = debounce(function(){ if (MAP) MAP.invalidateSize(false); }, 25);
  if (typeof ResizeObserver !== 'undefined'){
    try {
      var ro1 = new ResizeObserver(onChartResize); ro1.observe(chartBox);
      var ro2 = new ResizeObserver(onMapResize);   ro2.observe(mapBox);
    } catch (e){ logErr('ResizeObserver failed: '+e); }
  }
  window.addEventListener('resize', function(){ onChartResize(); onMapResize(); });
  var lastW = -1, lastH = -1;
  setInterval(function(){
    var w = chartBox.clientWidth, h = chartBox.clientHeight;
    if (w !== lastW || h !== lastH){ lastW = w; lastH = h; onChartResize(); }
  }, 200);
}

/* ---------- Basemaps (incl. Bing Aerial) ---------- */
function tileXYToQuadKey(x, y, z){
  var quadKey = '';
  for (var i = z; i > 0; i--) {
    var digit = 0;
    var mask = 1 << (i - 1);
    if ((x & mask) !== 0) digit += 1;
    if ((y & mask) !== 0) digit += 2;
    quadKey += digit.toString();
  }
  return quadKey;
}
var BingAerial = L.TileLayer.extend({
  createTile: function(coords, done){
    var tile = document.createElement('img');
    var zoom = this._getZoomForUrl();
    if (!BING_KEY_JS){
      tile.onload = function(){ done(null, tile); };
      tile.onerror = function(){ done(null, tile); };
      tile.src = 'about:blank';
      return tile;
    }
    var q = tileXYToQuadKey(coords.x, coords.y, zoom);
    var t = (coords.x + coords.y) % 4;
    var url = 'https://ecn.t'+t+'.tiles.virtualearth.net/tiles/a' + q + '.jpeg?g=1&n=z&key=' + encodeURIComponent(BING_KEY_JS);
    tile.alt = ''; tile.setAttribute('role', 'presentation');
    tile.style.width = this.options.tileSize + 'px';
    tile.style.height = this.options.tileSize + 'px';
    tile.crossOrigin = 'anonymous';
    tile.onload = function(){ done(null, tile); };
    tile.onerror = function(){ done(null, tile); };
    tile.src = url;
    return tile;
  }
});

function buildBaseSources(){
  var common = { maxZoom: 19, crossOrigin: true, tileSize: 256 };

  var osm = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { ...common, attribution: '© OpenStreetMap' });
  var topo = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', { ...common, subdomains: ['a','b','c'], maxZoom: 17, attribution: '© OpenStreetMap, © OpenTopoMap (CC-BY-SA)' });
  SATELLITE_FALLBACK = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', { ...common, attribution: 'Esri, Maxar, Earthstar Geographics, and the GIS User Community' });

  var bing = new BingAerial({ tileSize: 256 });
  return { osm: osm, topo: topo, sat_bing: bing, sat_esri: SATELLITE_FALLBACK };
}

function setBasemap(kind){
  var center = MAP.getCenter(), zoom = MAP.getZoom();
  if (BASE){ MAP.removeLayer(BASE); BASE = null; }
  if (kind === 'osm'){ BASE = BASE_SOURCES.osm; }
  else if (kind === 'topo'){ BASE = BASE_SOURCES.topo; }
  else if (kind === 'sat'){ BASE = BING_KEY_JS ? BASE_SOURCES.sat_bing : BASE_SOURCES.sat_esri; }
  else { BASE = BASE_SOURCES.osm; }
  BASE.addTo(MAP);
  MAP.setView(center, zoom, { animate:false });
}

/* ---------- Continuous color scale for env_index (0–100): yellow -> red ---------- */
function clamp(v,min,max){ return Math.max(min, Math.min(max, v)); }
function hexToRgb(h){
  var s = h.replace('#',''); if (s.length===3) s = s.split('').map(x=>x+x).join('');
  var num = parseInt(s,16); return { r:(num>>16)&255, g:(num>>8)&255, b:num&255 };
}
function rgbToHex(r,g,b){
  return '#' + [r,g,b].map(v => { var h = v.toString(16); return h.length===1 ? '0'+h : h; }).join('');
}
function lerp(a,b,t){ return a + (b-a)*t; }
function envIndexToColor(v){
  if (v==null || isNaN(v)) return '#cccccc';
  var t = clamp(Number(v)/100.0, 0, 1); // 0 -> yellow (#ffff00), 1 -> red (#ff0000)
  var y = hexToRgb('#ffff00'), r = hexToRgb('#ff0000');
  var rr = Math.round(lerp(y.r, r.r, t)), gg = Math.round(lerp(y.g, r.g, t)), bb = Math.round(lerp(y.b, r.b, t));
  return rgbToHex(rr,gg,bb);
}

/* ---------- Layer styling ---------- */
function styleFeature(f){
  var c = (f.properties && f.properties.sensitivity_code_max) ? f.properties.sensitivity_code_max : '';
  return { color:'white', weight:0.5, opacity:1.0, fillOpacity:FILL_ALPHA, fillColor: COLOR_MAP[c] || '#bdbdbd' };
}
function styleFeatureSeg(f){
  var c = (f.properties && f.properties.sensitivity_code_max) ? f.properties.sensitivity_code_max : '';
  return { color:'#f7f7f7', weight:0.7, opacity:1.0, fillOpacity:FILL_ALPHA, fillColor: COLOR_MAP[c] || '#bdbdbd' };
}
function styleFeatureAssets(f){
  return { color: ASSET_COLOR, weight: 0.6, opacity: FILL_ALPHA, fillOpacity: FILL_ALPHA, fillColor: ASSET_COLOR };
}
function styleFeatureEnv(f){
  var v = (f.properties && f.properties.env_index != null) ? Number(f.properties.env_index) : null;
  return { color:'white', weight:0.5, opacity:1.0, fillOpacity:FILL_ALPHA, fillColor: envIndexToColor(v) };
}

/* ---------- API-backed layer loaders (fill the groups) ---------- */
function loadGeocodeIntoGroup(cat, preserveView){
  var prev = null;
  if (preserveView && MAP) prev = { center: MAP.getCenter(), zoom: MAP.getZoom() };
  window.pywebview.api.get_geocode_layer(cat).then(function(res){
    if (!res.ok){ setError('Failed to load geocode: '+res.error); return; }
    if (res.stats && res.stats.message){ setError(res.stats.message); } else { setError(''); }

    if (GEO_GROUP){ GEO_GROUP.clearLayers(); }
    if (res.geojson && res.geojson.features && res.geojson.features.length > 0){
      LAYER = L.geoJSON(res.geojson, {
        style: styleFeature,
        renderer: RENDERERS.geocodes,
        onEachFeature: function (feature, layer) { attachPolygonTooltip(layer, feature); }
      });
      GEO_GROUP.addLayer(LAYER);
    }

    if (preserveView && prev){ MAP.setView(prev.center, prev.zoom, { animate:false }); }
    else {
      HOME_BOUNDS = res.home_bounds || [[0,0],[0,0]];
      MAP.fitBounds(HOME_BOUNDS, { padding: [20,20] });
    }

    renderLegend(res.stats);
    renderChart(res.stats);

    // If ENV overlay is visible, refresh it for the current ENV dropdown selection
    if (ENV_GROUP && MAP.hasLayer(ENV_GROUP)) {
      var envSel = document.getElementById('envCatSel');
      var envCat = envSel ? envSel.value : (cat || null);
      if (envCat) loadEnvIndexIntoGroup(envCat);
    }
  }).catch(function(err){
    setError('API error: '+err); logErr('API error: '+err);
  });
}

function loadEnvIndexIntoGroup(cat){
  window.pywebview.api.get_envindex_layer(cat).then(function(res){
    if (!res.ok){ setError('Failed to load env_index: '+res.error); return; }
    if (ENV_GROUP){ ENV_GROUP.clearLayers(); }
    if (res.geojson && res.geojson.features && res.geojson.features.length > 0){
      LAYER_ENV = L.geoJSON(res.geojson, {
        style: styleFeatureEnv,
        renderer: RENDERERS.env,
        onEachFeature: function (feature, layer) { attachEnvTooltip(layer, feature); }
      });
      ENV_GROUP.addLayer(LAYER_ENV);
    }
  }).catch(function(err){
    setError('API error: '+err); logErr('API error: '+err);
  });
}

function loadSegmentsIntoGroup(lineNameOrAll){
  window.pywebview.api.get_segment_layer(lineNameOrAll).then(function(res){
    if (!res.ok){ setError('Failed to load segments: '+res.error); return; }
    if (!document.getElementById('err').textContent){ setError(''); }
    if (SEG_GROUP){ SEG_GROUP.clearLayers(); }
    if (res.geojson && res.geojson.features && res.geojson.features.length > 0){
      LAYER_SEG = L.geoJSON(res.geojson, { style: styleFeatureSeg, renderer: RENDERERS.segments });
      SEG_GROUP.addLayer(LAYER_SEG);
    }
  }).catch(function(err){
    setError('API error: '+err); logErr('API error: '+err);
  });
}

function loadAssetsIntoGroup(){
  window.pywebview.api.get_assets_layer().then(function(res){
    if (!res.ok){ setError('Failed to load assets: '+res.error); return; }
    if (ASSET_GROUP){ ASSET_GROUP.clearLayers(); }
    if (res.geojson && res.geojson.features && res.geojson.features.length > 0){
      LAYER_ASSETS = L.geoJSON(res.geojson, {
        style: styleFeatureAssets,
        renderer: RENDERERS.assets,
        pane: 'assetsPane',
        pointToLayer: function (feature, latlng) {
          // circleMarkers for point assets (explicit pane)
          var m = L.circleMarker(latlng, {
            pane: 'assetsPane',
            radius: 3.5,
            color: ASSET_COLOR,
            weight: 0.8,
            opacity: FILL_ALPHA,
            fillOpacity: FILL_ALPHA,
            fillColor: ASSET_COLOR,
            renderer: RENDERERS.assets
          });
          attachAssetTooltip(m, feature);
          return m;
        },
        onEachFeature: function (feature, layer) {
          // polygons/lines assets: attach tooltip
          attachAssetTooltip(layer, feature);
        }
      });
      ASSET_GROUP.addLayer(LAYER_ASSETS);
    }
  }).catch(function(err){
    setError('API error: '+err); logErr('API error: '+err);
  });
}

/* ---------- Export current map to PNG ---------- */
function exportMapPng(){
  var node = document.getElementById('map');
  function capture(){ return html2canvas(node, { useCORS: true, allowTaint: false, backgroundColor: '#ffffff', scale: 3 }); }
  node.classList.add('exporting');
  var baseWasOn = BASE && MAP.hasLayer(BASE);

  capture().then(function(canvas){
    node.classList.remove('exporting');
    var png = canvas.toDataURL('image/png');
    return window.pywebview.api.save_png(png);
  }).then(function(res){
    if (!res.ok){ setError('Save cancelled or failed: ' + (res.error || '')); setTimeout(function(){ setError(''); }, 4000); }
  }).catch(function(err1){
    if (baseWasOn) MAP.removeLayer(BASE);
    capture().then(function(canvas){
      node.classList.remove('exporting');
      if (baseWasOn) BASE.addTo(MAP);
      var png = canvas.toDataURL('image/png');
      return window.pywebview.api.save_png(png);
    }).then(function(res){
      if (!res.ok){ setError('Save cancelled or failed: ' + (res.error || '')); setTimeout(function(){ setError(''); }, 4000); }
    }).catch(function(err2){
      node.classList.remove('exporting');
      if (baseWasOn) BASE.addTo(MAP);
      setError('Export failed. Try OSM/Topography basemap.'); logErr(err1); logErr(err2);
      setTimeout(function(){ setError(''); }, 5000);
    });
  });
}

/* ---------- Build native layers control ---------- */
function buildLayersControl(state){
  var baseLayers = {
    'OpenStreetMap <span class="help" data-tip="Standard OSM cartography. Good for general context.">?</span>': BASE_SOURCES.osm,
    'OSM Topography <span class="help" data-tip="OpenTopoMap tiles. Elevation & relief shading; lower max zoom.">?</span>': BASE_SOURCES.topo,
    'Satellite <span class="help" data-tip="Bing Aerial if key configured; otherwise Esri World Imagery fallback.">?</span>': (BING_KEY_JS ? BASE_SOURCES.sat_bing : SATELLITE_FALLBACK)
  };

  // Groups (stacking top→bottom among overlays):
  // 1) Sensitivity lines (top)
  // 2) Environment index (new)
  // 3) Geocode areas
  // 4) Assets (lowest overlay)
  SEG_GROUP = L.layerGroup();
  ENV_GROUP = L.layerGroup();
  GEO_GROUP = L.layerGroup();
  ASSET_GROUP = L.layerGroup();

  var segLabel = 'Sensitivity lines <span class="help" data-tip="Buffered lines (A–E) selected by line name from tbl_lines.">?</span><div class="inlineSel"><select id="segLineSel"></select></div>';
  var envLabel = 'Environment index (yellow→red) <span class="help" data-tip="Continuous heat style per polygon using env_index (0–100). Choose category below.">?</span><div class="inlineSel"><select id="envCatSel"></select></div>';
  var geoLabel = 'Geocode areas <span class="help" data-tip="Polygons colored by sensitivity (A–E). Choose a geocode category below. Stats always from ‘basic_mosaic’.">?</span><div class="inlineSel"><select id="geoCatSel"></select></div>';
  var assLabel = 'Assets <span class="help" data-tip="All assets in semi-transparent steel blue. Hover for details.">?</span>';

  var overlays = {};
  overlays[segLabel] = SEG_GROUP;
  overlays[envLabel] = ENV_GROUP;
  overlays[geoLabel] = GEO_GROUP;
  overlays[assLabel] = ASSET_GROUP;

  var ctrl = L.control.layers(baseLayers, overlays, { collapsed:false, position:'topleft' }).addTo(MAP);

  // Move basemaps (base) section to bottom of control
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
  } catch(e){ logErr('Layer reorder failed (fixed path) : ' + e); }

  // Seed layers: show Sensitivity lines + Geocode areas by default
  BASE_SOURCES.osm.addTo(MAP);
  GEO_GROUP.addTo(MAP);
  SEG_GROUP.addTo(MAP);
  // ENV & ASSETS remain toggled off initially

  setTimeout(function(){
    // populate the two geocode dropdowns (they can be different)
    var geocat = document.getElementById('geoCatSel');
    var envcat = document.getElementById('envCatSel');
    var geocodes = (state.geocode_categories || []);

    function fillSelect(sel){
      if (!sel) return;
      sel.innerHTML = '';
      for (var i=0;i<geocodes.length;i++){
        var o = document.createElement('option'); o.value = geocodes[i]; o.textContent = geocodes[i];
        sel.appendChild(o);
      }
    }

    fillSelect(geocat);
    fillSelect(envcat);

    if (geocat && state.initial_geocode && geocodes.indexOf(state.initial_geocode) >= 0){ geocat.value = state.initial_geocode; }
    if (envcat && state.initial_geocode && geocodes.indexOf(state.initial_geocode) >= 0){ envcat.value = state.initial_geocode; }

    if (geocat){ loadGeocodeIntoGroup(geocat.value || state.initial_geocode, false); }
    if (envcat){ /* ENV overlay is off by default; it will load when toggled on */ }

    geocat && geocat.addEventListener('change', function(){ loadGeocodeIntoGroup(geocat.value || state.initial_geocode, true); });
    envcat && envcat.addEventListener('change', function(){
      if (ENV_GROUP && MAP.hasLayer(ENV_GROUP)) loadEnvIndexIntoGroup(envcat.value || state.initial_geocode);
    });

    // sensitivity line names (from tbl_lines.name_user)
    var segsel = document.getElementById('segLineSel');
    if (segsel){
      segsel.innerHTML = '';
      var names = state.segment_line_names || [];
      if (names.length > 0){
        var all = document.createElement('option'); all.value="__ALL__"; all.textContent="All lines"; segsel.appendChild(all);
        for (var j=0;j<names.length;j++){
          var so = document.createElement('option'); so.value = names[j]; so.textContent = names[j];
          segsel.appendChild(so);
        }
      } else {
        // fallback (legacy)
        var s = document.createElement('option'); s.value="__ALL__"; s.textContent="All segments"; segsel.appendChild(s);
        var segcats = state.segment_categories || [];
        for (var k=0;k<segcats.length;k++){
          var sc = document.createElement('option'); sc.value = segcats[k]; sc.textContent = segcats[k];
          segsel.appendChild(sc);
        }
      }
      loadSegmentsIntoGroup(segsel.value || "__ALL__");
      segsel.addEventListener('change', function(){ loadSegmentsIntoGroup(segsel.value || "__ALL__"); });
    }

    // overlay toggles
    MAP.on('overlayadd', function(e){
      if (e.layer === GEO_GROUP){
        var gc = document.getElementById('geoCatSel'); loadGeocodeIntoGroup(gc ? gc.value : state.initial_geocode, true);
      } else if (e.layer === SEG_GROUP){
        var sl = document.getElementById('segLineSel'); loadSegmentsIntoGroup(sl ? (sl.value||"__ALL__") : "__ALL__");
      } else if (e.layer === ASSET_GROUP){
        loadAssetsIntoGroup();
      } else if (e.layer === ENV_GROUP){
        var ec = document.getElementById('envCatSel'); loadEnvIndexIntoGroup(ec ? ec.value : state.initial_geocode);
      }
    });
    MAP.on('overlayremove', function(e){
      if (e.layer === GEO_GROUP){ GEO_GROUP.clearLayers(); }
      if (e.layer === SEG_GROUP){ SEG_GROUP.clearLayers(); }
      if (e.layer === ASSET_GROUP){ ASSET_GROUP.clearLayers(); }
      if (e.layer === ENV_GROUP){ ENV_GROUP.clearLayers(); }
    });
  }, 0);

  return ctrl;
}

/* ---------- Boot ---------- */
function boot(){
  MAP = L.map('map', { zoomControl:false, preferCanvas:true });
  L.control.zoom({ position:'topright' }).addTo(MAP);

  // Panes & z-indexes: TOP → BOTTOM among overlays
  MAP.createPane('segmentsPane'); MAP.getPane('segmentsPane').style.zIndex = 650; // Sensitivity lines (top)
  MAP.createPane('envPane');      MAP.getPane('envPane').style.zIndex      = 600; // Env index (between)
  MAP.createPane('geocodePane');  MAP.getPane('geocodePane').style.zIndex  = 550; // Geocode areas (middle)
  MAP.createPane('assetsPane');   MAP.getPane('assetsPane').style.zIndex   = 450; // Assets (lower)

  // Create one Canvas renderer per pane (this fixes stacking with Canvas)
  try {
    RENDERERS.segments = L.canvas({ pane: 'segmentsPane' });
    RENDERERS.env      = L.canvas({ pane: 'envPane'      });
    RENDERERS.geocodes = L.canvas({ pane: 'geocodePane'  });
    RENDERERS.assets   = L.canvas({ pane: 'assetsPane'   });
  } catch (e) {
    logErr('Canvas renderers creation failed: ' + e);
  }

  function closeTooltipsUnderZoom(){
    if (!MAP || MAP.getZoom() >= ZOOM_THRESHOLD_JS) return;
    if (LAYER && GEO_GROUP && MAP.hasLayer(GEO_GROUP)) { GEO_GROUP.eachLayer(ttCloseIfAny); }
    if (LAYER_SEG && SEG_GROUP && MAP.hasLayer(SEG_GROUP)) { SEG_GROUP.eachLayer(ttCloseIfAny); }
    if (LAYER_ENV && ENV_GROUP && MAP.hasLayer(ENV_GROUP)) { ENV_GROUP.eachLayer(ttCloseIfAny); }
    // Assets are always-on tooltips; close them as well when zooming out to reduce clutter
    if (LAYER_ASSETS && ASSET_GROUP && MAP.hasLayer(ASSET_GROUP)) { ASSET_GROUP.eachLayer(ttCloseIfAny); }
  }
  MAP.on('zoomend', closeTooltipsUnderZoom);

  L.control.scale({ position:'bottomleft', metric:true, imperial:false, maxWidth:200 }).addTo(MAP);

  BASE_SOURCES = buildBaseSources();
  MAP.setView([0,0], 2);
  setupResizeObservers();

  window.pywebview.api.get_state().then(function(state){
    COLOR_MAP        = state.colors || {};
    DESC_MAP         = state.descriptions || {};
    BING_KEY_JS      = (state.bing_key || '').trim() || null;
    ZOOM_THRESHOLD_JS= state.zoom_threshold || 12;

    renderLegend(null);
    renderChart({labels:['A','B','C','D','E'], values:[0,0,0,0,0]});

    buildLayersControl(state);

    document.getElementById('homeBtn').addEventListener('click', function(){
      if (HOME_BOUNDS) MAP.fitBounds(HOME_BOUNDS, { padding:[20,20] });
    });

    document.getElementById('exportBtn').addEventListener('click', exportMapPng);

    document.getElementById('exitBtn').addEventListener('click', function(){
      if (window.pywebview && window.pywebview.api && window.pywebview.api.exit_app){
        window.pywebview.api.exit_app();
      }
    });
    window.addEventListener('keydown', function(e){
      if (e.key === 'Escape'){
        if (window.pywebview && window.pywebview.api && window.pywebview.api.exit_app){
            window.pywebview.api.exit_app();
        }
      }
    });

    var slider = document.getElementById('opacity');
    var opv    = document.getElementById('opv');
    slider.addEventListener('input', function(){
      var v = parseInt(slider.value, 10);
      FILL_ALPHA = v / 100.0;
      opv.textContent = v + '%';
      if (LAYER)        LAYER.setStyle({ fillOpacity: FILL_ALPHA });
      if (LAYER_SEG)    LAYER_SEG.setStyle({ fillOpacity: FILL_ALPHA });
      if (LAYER_ENV)    LAYER_ENV.setStyle({ fillOpacity: FILL_ALPHA });
      if (LAYER_ASSETS) LAYER_ASSETS.setStyle({ fillOpacity: FILL_ALPHA, opacity: FILL_ALPHA });
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

# Inject steel-blue color without %-format conflicts
HTML = HTML_TEMPLATE.replace("__ASSET_COLOR__", STEEL_BLUE)

# ===============================
# Run
# ===============================
if __name__ == "__main__":
    window = webview.create_window(
        title="Maps overview",
        html=HTML,
        js_api=api,
        width=1300, height=800, resizable=True
    )
    webview.start(gui="edgechromium", debug=False)
