#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, base64, configparser, locale
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping
import webview  # pip install pywebview

# --- new imports for accurate area on WGS84 ---
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
PARQUET_FILE     = os.path.join(BASE_DIR, "../output/geoparquet/tbl_flat.parquet")           # <- stats source (tbl_flat)
SEGMENT_FILE     = os.path.join(BASE_DIR, "../output/geoparquet/tbl_segment_flat.parquet")
ASSET_FILE       = os.path.join(BASE_DIR, "../output/geoparquet/tbl_asset_object.parquet")
PLOT_CRS         = "EPSG:4326"
BASIC_GROUP_NAME = "basic_mosaic"   # stats always from here (in tbl_flat)
ZOOM_THRESHOLD   = 10               # tooltips appear when map zoom >= this
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
            # keep tooltip fast; legend uses corrected geodesic areas
            "area_km2": (float(row.get("area_m2", 0.0)) / 1e6) if ("area_m2" in row) else None,
            "geocode_group": row.get("name_gis_geocodegroup", None),
        }
        if "name_asset_object" in row: props["name_asset_object"] = row.get("name_asset_object", None)
        if "id_asset_object"   in row: props["id_asset_object"]   = row.get("id_asset_object", None)
        feats.append({"type": "Feature", "geometry": mapping(geom), "properties": props})
    return {"type": "FeatureCollection", "features": feats}

def assets_to_geojson(gdf: gpd.GeoDataFrame) -> dict:
    """Minimal GeoJSON for assets overlay (steel blue, no category colors)."""
    feats = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        props = {
            "id_asset_object": row.get("id_asset_object", None),
            "name_asset_object": row.get("name_asset_object", None),
        }
        feats.append({"type": "Feature", "geometry": mapping(geom), "properties": props})
    return {"type": "FeatureCollection", "features": feats}

# --- geodesic area helpers (WGS84) ---
GEOD = Geod(ellps="WGS84")

def _valid_geom(geom):
    """Robust validity fix without changing area materially."""
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
    """Signed geodesic area (m²) on WGS84, absolute value; supports holes."""
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
    Filter to BASIC_GROUP_NAME, de-duplicate, compute WGS84 geodesic area,
    and sum km² by sensitivity_code_max.
    """
    labels = list("ABCDE")
    msg = f'The geocode/partition "{BASIC_GROUP_NAME}" is missing.'

    if df_flat.empty or "name_gis_geocodegroup" not in df_flat.columns:
        return {"labels": labels, "values": [0,0,0,0,0], "message": msg}

    # filter to basic_mosaic
    df = df_flat.copy()
    df["name_gis_geocodegroup"] = df["name_gis_geocodegroup"].astype("string").str.strip().str.lower()
    df = df[df["name_gis_geocodegroup"] == BASIC_GROUP_NAME]
    df = only_A_to_E(df)
    if df.empty:
        return {"labels": labels, "values": [0,0,0,0,0], "message": msg}

    # deduplicate to avoid double counting
    if "id_geocode_object" in df.columns:
        df = df.drop_duplicates(subset=["id_geocode_object"])
    else:
        try:
            df = df.assign(__wkb__=df.geometry.apply(lambda g: g.wkb if g is not None else None))
            df = df.drop_duplicates(subset=["__wkb__"]).drop(columns=["__wkb__"])
        except Exception:
            df = df.drop_duplicates()

    # ensure EPSG:4326 for geodesic measure + validity fix
    df = to_epsg4326(df, cfg)
    df["geometry"] = df["geometry"].apply(_valid_geom)

    # sum by code
    out = []
    for c in labels:
        sub = df[df["sensitivity_code_max"] == c]
        if sub.empty:
            out.append(0.0)
            continue
        a_m2 = float(sub.geometry.apply(geodesic_area_m2).sum())
        out.append(a_m2 / 1e6)  # km²

    return {"labels": labels, "values": out}

# (legacy helper kept for map layer bounds; not used for stats)
def compute_stats(gdf: gpd.GeoDataFrame) -> dict:
    if gdf.empty or "area_m2" not in gdf.columns:
        return {"labels": list("ABCDE"), "values": [0,0,0,0,0]}
    grp = gdf.groupby("sensitivity_code_max")["area_m2"].sum()
    vals = [float(grp.get(c, 0.0))/1e6 for c in "ABCDE"]  # km²
    return {"labels": list("ABCDE"), "values": vals}

# ===============================
# Load datasets (robust, non-fatal)
# ===============================
cfg          = read_config(CONFIG_FILE)
COLS         = get_color_mapping(cfg)
DESC         = get_desc_mapping(cfg)

# GDF is tbl_flat (stats & geocode map)
GDF          = load_parquet(PARQUET_FILE)
SEG_GDF      = load_parquet(SEGMENT_FILE)
ASSET_GDF    = load_parquet(ASSET_FILE)

GEOCODE_AVAILABLE   = (not GDF.empty) and ("name_gis_geocodegroup" in GDF.columns)
SEGMENTS_AVAILABLE  = (not SEG_GDF.empty) and ("geometry" in SEG_GDF.columns)
ASSETS_AVAILABLE    = (not ASSET_GDF.empty) and ("geometry" in ASSET_GDF.columns)

CATS = sorted(GDF["name_gis_geocodegroup"].dropna().unique().tolist()) if GEOCODE_AVAILABLE else []
if SEGMENTS_AVAILABLE and "name_gis_geocodegroup" in SEG_GDF.columns:
    SEG_CATS = sorted(SEG_GDF["name_gis_geocodegroup"].dropna().unique().tolist())
else:
    SEG_CATS = []

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
        """Return UI state, including availability and category lists."""
        return {
            "geocode_available": GEOCODE_AVAILABLE,
            "geocode_categories": CATS if GEOCODE_AVAILABLE else [],
            "segment_available": SEGMENTS_AVAILABLE,
            "segment_categories": SEG_CATS if SEGMENTS_AVAILABLE else [],
            "assets_available": ASSETS_AVAILABLE,
            "colors": COLS,
            "descriptions": DESC,
            "initial_geocode": (CATS[0] if (GEOCODE_AVAILABLE and CATS) else None),
            "has_segments": SEGMENTS_AVAILABLE,
            "bing_key": BING_KEY,
            "zoom_threshold": ZOOM_THRESHOLD
        }

    def get_geocode_layer(self, geocode_category):
        """Return GeoJSON for the selected geocode category (map content),
        and stats computed from tbl_flat's 'basic_mosaic' only."""
        try:
            # Map layer (honor the user's selection for display)
            if not GEOCODE_AVAILABLE:
                map_geojson = {"type":"FeatureCollection","features":[]}
                map_bounds  = [[0,0],[0,0]]
            else:
                df_map = GDF[GDF["name_gis_geocodegroup"] == geocode_category].copy()
                df_map = only_A_to_E(df_map)
                df_map = to_plot_crs(df_map, cfg)
                map_bounds = bounds_to_leaflet(df_map.total_bounds) if not df_map.empty else [[0,0],[0,0]]
                map_geojson = gdf_to_geojson_min(df_map)

            # Stats: strictly from tbl_flat basic_mosaic (geodesic, deduped)
            stats = compute_stats_by_geodesic_area_from_flat_basic(GDF, cfg)
            if "message" in stats:
                # show missing-message if basic_mosaic absent
                pass

            return {
                "ok": True,
                "geojson": map_geojson,
                "home_bounds": map_bounds,
                "stats": stats
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_segment_layer(self, seg_category_or_all):
        """Return segments overlay; if unavailable, respond with error."""
        try:
            if not SEGMENTS_AVAILABLE:
                return {"ok": False, "error": "Segments dataset is empty or missing."}
            if "name_gis_geocodegroup" in SEG_GDF.columns and seg_category_or_all not in (None, "", "__ALL__"):
                df = SEG_GDF[SEG_GDF["name_gis_geocodegroup"] == seg_category_or_all].copy()
            else:
                df = SEG_GDF.copy()
            df = only_A_to_E(df)
            df = to_plot_crs(df, cfg)
            gj = gdf_to_geojson_min(df)
            return {"ok": True, "geojson": gj}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_assets_layer(self):
        """Return assets overlay (steel blue, sits just above basemap)."""
        try:
            if not ASSETS_AVAILABLE:
                return {"ok": False, "error": "Assets dataset is empty or missing."}
            df = ASSET_GDF.copy()
            df = df[df.geometry.notna()]
            if df.empty:
                return {"ok": False, "error": "No valid geometries found in assets."}
            df = to_plot_crs(df, cfg)
            gj = assets_to_geojson(df)
            return {"ok": True, "geojson": gj}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def exit_app(self):
        try:
            webview.destroy_window()
        except Exception:
            os._exit(0)

    def save_png(self, data_url: str):
        """Receive data URL from JS and save PNG via a Save dialog."""
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
    grid-template-columns: 320px 1fr 420px;
    grid-template-rows: 48px 1fr;
    grid-template-areas:
      "ctrl bar stats"
      "ctrl map stats";
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  }
  .ctrl   { grid-area: ctrl; border-right: 2px solid #2b3442; padding:10px; background:#f5f7fb; position:relative; overflow:auto; }
  .bar    { grid-area: bar; display:flex; gap:12px; align-items:center; padding:8px 12px; flex-wrap:wrap; }
  .map    { grid-area: map; position:relative; background:#ddd; }
  #map    { position:absolute; inset:0; }
  #map.exporting .leaflet-control-zoom { display: none !important; }
  .stats  { grid-area: stats; border-left: 2px solid #2b3442; display:flex; flex-direction:column; overflow:hidden; }
  #err { color:#b00; padding:6px 12px; font-size:12px; display:none; }
  .legend { padding:8px 12px; font-size:12px; }
  .legend table { width:100%; border-collapse:collapse; }
  .legend th, .legend td { padding:4px 6px; vertical-align:middle; }
  .legend thead th { border-bottom:1px solid #cdd4de; font-weight:600; }
  .legend tfoot td { border-top:1px solid #cdd4de; font-weight:600; }
  .swatch { display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:6px; border:1px solid #8884; }
  .num { text-align:right; white-space:nowrap; }
  .btn { padding:6px 10px; border:1px solid #ccd; background:#fff; border-radius:6px; cursor:pointer; }
  .btn:active { transform:translateY(1px); }
  .slider { display:flex; align-items:center; gap:8px; }
  .slider input[type=range]{ width:160px; }

  /* ----- Layers panel (outside map) ----- */
  .layers {
    background:#fff; border:1px solid #d9e0ea; border-radius:10px; padding:8px;
    box-shadow: 0 2px 8px rgba(0,0,0,.05);
  }
  .layers h2 {
    margin: 6px 4px 10px; font-size:15px; letter-spacing:.2px;
  }
  details.layer-group {
    border:1px solid #e4e9f2; border-radius:8px; margin:8px 4px; background:#fbfcfe;
  }
  details.layer-group[open] { background:#f8faff; }
  details.layer-group > summary {
    list-style:none; cursor:pointer; padding:8px 10px; font-weight:600; font-size:13px;
    display:flex; align-items:center; justify-content:space-between;
  }
  details.layer-group > summary::-webkit-details-marker { display:none; }
  .group-body { padding:6px 10px 10px; }
  .row {
    display:grid; grid-template-columns: 22px 1fr; align-items:center; gap:8px;
    padding:6px 4px; border-radius:6px;
  }
  .row:hover { background:#eef3fb; }
  .row.disabled { opacity:.55; }
  .lab { font-size:13px; }
  .select-block { margin-top:6px; }
  .select-block select { width:100%; padding:6px; }
  .hint { font-size:11px; opacity:.75; margin:8px 0 0; }

  /* chart/tooltip */
  #chartBox { flex:1 1 auto; padding:8px 12px; position:relative; }
  #chart { position:absolute; inset:8px 12px; width:calc(100% - 24px); height:calc(100% - 16px); }

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
  <div class="ctrl">
    <div class="layers">
      <h2>Layers</h2>

      <!-- Overlays FIRST -->
      <details class="layer-group" open>
        <summary>Overlays</summary>
        <div class="group-body" id="overlayGroup">

          <!-- Geocoded areas -->
          <div id="rowGeo" class="row">
            <input type="checkbox" id="geoToggle" checked>
            <label class="lab" for="geoToggle">Geocoded areas</label>
            <div class="select-block"><select id="cat"></select></div>
          </div>

          <!-- Segments (ON by default) -->
          <div id="rowSeg" class="row">
            <input type="checkbox" id="segToggle" checked>
            <label class="lab" for="segToggle">Sensitivity segments</label>
            <div class="select-block"><select id="segcat"></select></div>
          </div>

          <!-- Assets -->
          <div id="rowAssets" class="row">
            <input type="checkbox" id="assetsToggle">
            <label class="lab" for="assetsToggle">Assets</label>
          </div>

          <!-- Overlay notes -->
          <div class="hint" id="overlayNotes">
            • Assets render in semi-transparent steel blue; overlap appears denser.<br>
            • Statistics on the right always use areas from the "<b>basic_mosaic</b>" group in <b>tbl_flat</b>.
          </div>
        </div>
      </details>

      <!-- Basemaps LAST -->
      <details class="layer-group" open>
        <summary>Basemaps</summary>
        <div class="group-body" id="basemapGroup">
          <div class="row">
            <input type="radio" name="basemap" id="bm_osm" value="osm" checked>
            <label class="lab" for="bm_osm">OpenStreetMap</label>
          </div>
          <div class="row">
            <input type="radio" name="basemap" id="bm_topo" value="topo">
            <label class="lab" for="bm_topo">OSM Topography</label>
          </div>
          <div class="row">
            <input type="radio" name="basemap" id="bm_sat" value="sat">
            <label class="lab" for="bm_sat">Satellite (Bing / Esri fallback)</label>
          </div>
          <div id="basemapHint" class="hint"></div>
        </div>
      </details>
    </div>

    <button id="exitBtn" class="btn" style="margin-top:10px;">Exit</button>
  </div>

  <div class="bar">
    <button id="homeBtn" class="btn">Home</button>

    <div class="slider">
      <span class="label-sm">Opacity</span>
      <input id="opacity" type="range" min="10" max="100" value="80">
      <span id="opv" class="label-sm">80%</span>
    </div>

    <button id="exportBtn" class="btn" title="Export current map to PNG (~300 dpi)">Export PNG</button>
  </div>

  <div class="map"><div id="map"></div></div>

  <div class="stats">
    <div id="err"></div>
    <div class="legend" id="legend"></div>
    <div id="chartBox"><canvas id="chart"></canvas></div>
  </div>
</div>

<script>
var MAP=null, BASE=null, BASE_SOURCES=null, LAYER=null, LAYER_SEG=null, LAYER_ASSETS=null, COLOR_MAP={}, DESC_MAP={}, HOME_BOUNDS=null, CHART=null;
var FILL_ALPHA = 0.8; // default 80%
var BING_KEY_JS = null;
var SATELLITE_FALLBACK = null;
var ZOOM_THRESHOLD_JS = 12;
const ASSET_COLOR = "__ASSET_COLOR__";

function logErr(m){ try { window.pywebview.api.js_log(m); } catch(e){} }
function setError(msg){
  var e = document.getElementById('err');
  if (msg){ e.style.display='block'; e.textContent = msg; }
  else { e.style.display='none'; e.textContent=''; }
}

// Formatting helpers
function fmtKm2(x){ return Number(x||0).toLocaleString('en-US', {maximumFractionDigits:2}); }

// Build tooltip HTML (for polygons only)
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
    </div>
  `;
}

/* ---------- Safe tooltip helpers ---------- */
function getTT(layer){ return (layer && typeof layer.getTooltip === 'function') ? layer.getTooltip() : null; }
function ttIsOpen(layer){ const tt = getTT(layer); return !!(tt && typeof tt.isOpen === 'function' && tt.isOpen()); }
function ttEnsure(layer, feature){
  const html = buildTipHTML((feature && feature.properties) || {});
  if (!getTT(layer)){
    layer.bindTooltip(html, { className: 'poly-tip', sticky: true, direction: 'auto', opacity: 0.98 });
  } else {
    layer.setTooltipContent(html);
  }
}
function ttCloseIfAny(layer){ if (ttIsOpen(layer)) layer.closeTooltip(); }

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
      datasets: [{
        label: 'km²',
        data: values,
        backgroundColor: colors,
        borderColor: '#ffffff',
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      resizeDelay: 0,
      animation: false,
      plugins: { legend: { display:false }, title: { display:true, text:'Overall area by sensitivity code (km²)' } },
      scales: { y: { beginAtZero:true, ticks:{ padding:6 } }, x:{ ticks:{ padding:4 } } },
      layout: { padding: { left: 0, right:0, top:0, bottom:0 } }
    }
  });

  resizeChartToBox();
}

/* ---------- Resize handling ---------- */
function debounce(fn, ms){ var t; return function(){ var ctx=this, args=arguments; clearTimeout(t); t=setTimeout(function(){ fn.apply(ctx,args); }, ms||50); }; }
function resizeChartToBox(){
  if (!CHART) return;
  var box = document.getElementById('chartBox');
  var canvas = document.getElementById('chart');
  var w = Math.max(0, box.clientWidth);
  var h = Math.max(0, box.clientHeight);
  canvas.style.width  = w + 'px';
  canvas.style.height = h + 'px';
  canvas.removeAttribute('width'); canvas.removeAttribute('height');
  CHART.resize(w, h);
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
    tile.alt = '';
    tile.setAttribute('role', 'presentation');
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

  var osm = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    ...common, attribution: '© OpenStreetMap'
  });

  var topo = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
    ...common, subdomains: ['a','b','c'], maxZoom: 17, attribution: '© OpenStreetMap, © OpenTopoMap (CC-BY-SA)'
  });

  SATELLITE_FALLBACK = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    ...common, attribution: 'Esri, Maxar, Earthstar Geographics, and the GIS User Community'
  });

  var bing = new BingAerial({ tileSize: 256 });
  return { osm: osm, topo: topo, sat_bing: bing, sat_esri: SATELLITE_FALLBACK };
}

function setBasemap(kind){
  var center = MAP.getCenter(), zoom = MAP.getZoom();
  if (BASE){ MAP.removeLayer(BASE); BASE = null; }

  if (kind === 'osm'){ BASE = BASE_SOURCES.osm; }
  else if (kind === 'topo'){ BASE = BASE_SOURCES.topo; }
  else if (kind === 'sat'){
    BASE = BING_KEY_JS ? BASE_SOURCES.sat_bing : BASE_SOURCES.sat_esri;
    if (!BING_KEY_JS){
      var bh = document.getElementById('basemapHint');
      if (bh){ bh.textContent = 'Bing key not set; using Esri World Imagery as satellite fallback.'; setTimeout(function(){ bh.textContent=''; }, 4000); }
    }
  } else { BASE = BASE_SOURCES.osm; }

  BASE.addTo(MAP);
  MAP.setView(center, zoom, { animate:false });
}

/* ---------- Layer styling ---------- */
function styleFeature(f){
  var c = (f.properties && f.properties.sensitivity_code_max) ? f.properties.sensitivity_code_max : '';
  return { pane:'geocodePane', color:'white', weight:0.5, opacity:1.0, fillOpacity:FILL_ALPHA, fillColor: COLOR_MAP[c] || '#bdbdbd' };
}
function styleFeatureSeg(f){
  var c = (f.properties && f.properties.sensitivity_code_max) ? f.properties.sensitivity_code_max : '';
  return { pane:'segmentsPane', color:'#f7f7f7', weight:0.7, opacity:1.0, fillOpacity:FILL_ALPHA, fillColor: COLOR_MAP[c] || '#bdbdbd' };
}
function styleFeatureAssets(f){
  return { pane:'assetsPane', color: ASSET_COLOR, weight: 0.6, opacity: FILL_ALPHA, fillOpacity: FILL_ALPHA, fillColor: ASSET_COLOR };
}

/* ---------- Tooltips for polygons (geocodes) ---------- */
function attachPolygonTooltip(layer, feature){
  layer.on('mouseover', function () {
    if (MAP.getZoom() >= ZOOM_THRESHOLD_JS) {
      layer.setStyle({ weight: 1.2 });
      ttEnsure(layer, feature);
      layer.openTooltip();
    }
  });
  layer.on('mousemove', function () {
    if (ttIsOpen(layer) && MAP.getZoom() < ZOOM_THRESHOLD_JS) {
      layer.closeTooltip();
    }
  });
  layer.on('mouseout', function () {
    layer.setStyle({ weight: 0.5 });
    ttCloseIfAny(layer);
  });
}

/* ---------- Loading geocoded layer (map) ---------- */
function loadGeocodeLayer(cat, preserveView){
  var prev = null;
  if (preserveView && MAP) prev = { center: MAP.getCenter(), zoom: MAP.getZoom() };
  window.pywebview.api.get_geocode_layer(cat).then(function(res){
    if (!res.ok){ setError('Failed to load geocode: '+res.error); return; }

    // Show message if basic_mosaic missing
    if (res.stats && res.stats.message){ setError(res.stats.message); } else { setError(''); }

    var geoVisible = document.getElementById('geoToggle').checked;
    if (geoVisible){
      if (LAYER){ MAP.removeLayer(LAYER); LAYER = null; }
      LAYER = L.geoJSON(res.geojson, {
        style: styleFeature, pane:'geocodePane', renderer: L.canvas(),
        onEachFeature: function (feature, layer) { attachPolygonTooltip(layer, feature); }
      }).addTo(MAP);
      if (LAYER_SEG){ try { LAYER_SEG.bringToFront(); } catch(e){} }
      if (preserveView && prev){
        MAP.setView(prev.center, prev.zoom, { animate:false });
      } else {
        HOME_BOUNDS = res.home_bounds || [[0,0],[0,0]];
        MAP.fitBounds(HOME_BOUNDS, { padding: [20,20] });
      }
    } else {
      if (LAYER){ MAP.removeLayer(LAYER); LAYER = null; }
    }

    renderLegend(res.stats);
    renderChart(res.stats);
  }).catch(function(err){
    setError('API error: '+err); logErr('API error: '+err);
  });
}

/* ---------- Loading / reloading segments overlay ---------- */
function loadSegmentsLayer(segCatOrAll){
  if (!document.getElementById('segToggle').checked){
    if (LAYER_SEG){ MAP.removeLayer(LAYER_SEG); LAYER_SEG=null; }
    return;
  }
  window.pywebview.api.get_segment_layer(segCatOrAll).then(function(res){
    if (!res.ok){ setError('Failed to load segments: '+res.error); return; }
    if (!document.getElementById('err').textContent){ setError(''); }
    if (LAYER_SEG){ MAP.removeLayer(LAYER_SEG); LAYER_SEG = null; }
    LAYER_SEG = L.geoJSON(res.geojson, { style: styleFeatureSeg, pane:'segmentsPane', renderer: L.canvas() }).addTo(MAP);
    try { LAYER_SEG.bringToFront(); } catch(e){}
  }).catch(function(err){
    setError('API error: '+err); logErr('API error: '+err);
  });
}

/* ---------- Loading / toggling assets overlay (just above basemap) ---------- */
function loadAssetsLayer(){
  if (!document.getElementById('assetsToggle').checked){
    if (LAYER_ASSETS){ MAP.removeLayer(LAYER_ASSETS); LAYER_ASSETS=null; }
    return;
  }
  window.pywebview.api.get_assets_layer().then(function(res){
    if (!res.ok){ setError('Failed to load assets: '+res.error); return; }
    if (LAYER_ASSETS){ MAP.removeLayer(LAYER_ASSETS); LAYER_ASSETS = null; }

    LAYER_ASSETS = L.geoJSON(res.geojson, {
      style: styleFeatureAssets,
      pane: 'assetsPane',
      renderer: L.canvas(),
      pointToLayer: function (feature, latlng) {
        return L.circleMarker(latlng, { pane:'assetsPane', radius: 3.5, color: ASSET_COLOR, weight: 0.8, opacity: FILL_ALPHA, fillOpacity: FILL_ALPHA, fillColor: ASSET_COLOR });
      }
    }).addTo(MAP);

    // Keep visual stacking: assets under geocodes & segments
    if (LAYER){ try { LAYER.bringToFront(); } catch(e){} }
    if (LAYER_SEG){ try { LAYER_SEG.bringToFront(); } catch(e){} }
  }).catch(function(err){
    setError('API error: '+err); logErr('API error: '+err);
  });
}

/* ---------- Export current map to PNG ---------- */
function exportMapPng(){
  var node = document.getElementById('map');
  function capture(){
    return html2canvas(node, { useCORS: true, allowTaint: false, backgroundColor: '#ffffff', scale: 3 });
  }
  node.classList.add('exporting');
  var baseWasOn = BASE && MAP.hasLayer(BASE);

  capture().then(function(canvas){
    node.classList.remove('exporting');
    var png = canvas.toDataURL('image/png');
    return window.pywebview.api.save_png(png);
  }).then(function(res){
    if (!res.ok){ setError('Save cancelled or failed: ' + (res.error || '')); setTimeout(function(){ setError(''); }, 4000); }
  }).catch(function(){
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
      setError('Export failed. Try OSM/Topography basemap.'); logErr(err2);
      setTimeout(function(){ setError(''); }, 5000);
    });
  });
}

/* ---------- Boot ---------- */
function boot(){
  MAP = L.map('map', { zoomControl:true, preferCanvas:true });
  // Panes: basemap (default tiles ~200), assets (~300), geocodes (~450), segments (~650)
  MAP.createPane('assetsPane');   MAP.getPane('assetsPane').style.zIndex   = 300;
  MAP.createPane('geocodePane');  MAP.getPane('geocodePane').style.zIndex  = 450;
  MAP.createPane('segmentsPane'); MAP.getPane('segmentsPane').style.zIndex = 650;

  function closeTooltipsUnderZoom(){
    if (!MAP || MAP.getZoom() >= ZOOM_THRESHOLD_JS) return;
    if (LAYER)     LAYER.eachLayer(ttCloseIfAny);
    if (LAYER_SEG) LAYER_SEG.eachLayer(ttCloseIfAny);
  }
  MAP.on('zoomend', closeTooltipsUnderZoom);

  L.control.scale({ position:'bottomleft', metric:true, imperial:false, maxWidth:200 }).addTo(MAP);

  BASE_SOURCES = buildBaseSources();
  BASE = BASE_SOURCES.osm.addTo(MAP);

  MAP.setView([0,0], 2);
  setupResizeObservers();

  window.pywebview.api.get_state().then(function(state){
    COLOR_MAP        = state.colors || {};
    DESC_MAP         = state.descriptions || {};
    BING_KEY_JS      = (state.bing_key || '').trim() || null;
    ZOOM_THRESHOLD_JS= state.zoom_threshold || 12;

    renderLegend(null);
    renderChart({labels:['A','B','C','D','E'], values:[0,0,0,0,0]});

    // Geocodes UI
    var geocat  = document.getElementById('cat');
    var rowGeo  = document.getElementById('rowGeo');
    var geoToggle = document.getElementById('geoToggle');
    if (state.geocode_available){
      var geocodes = state.geocode_categories || [];
      geocat.innerHTML = '';
      for (var i=0;i<geocodes.length;i++){
        var o = document.createElement('option');
        o.value = geocodes[i]; o.textContent = geocodes[i];
        geocat.appendChild(o);
      }
      if (state.initial_geocode){
        geocat.value = state.initial_geocode;
        loadGeocodeLayer(state.initial_geocode, false);
      }
      geoToggle.disabled = false; geocat.disabled = false; rowGeo.classList.remove('disabled');
    } else {
      geoToggle.checked = false; geoToggle.disabled = true; geocat.disabled = true; rowGeo.classList.add('disabled');
    }

    // Segments UI (ON by default)
    var segcat   = document.getElementById('segcat');
    var rowSeg   = document.getElementById('rowSeg');
    var segToggle= document.getElementById('segToggle');
    if (state.segment_available){
      var segcats = state.segment_categories || [];
      segcat.innerHTML = '';
      if (segcats.length === 0){
        var o = document.createElement('option');
        o.value = "__ALL__"; o.textContent = "All segments";
        segcat.appendChild(o);
      } else {
        for (var i=0;i<segcats.length;i++){
          var so = document.createElement('option');
          so.value = segcats[i]; so.textContent = segcats[i];
          segcat.appendChild(so);
        }
      }
      segToggle.disabled = false; rowSeg.classList.remove('disabled');
      // load immediately since default is checked
      loadSegmentsLayer(segcat.value || "__ALL__");
    } else {
      segToggle.checked = false; segToggle.disabled = true; segcat.disabled = true; rowSeg.classList.add('disabled');
    }

    // Assets UI
    var rowAssets = document.getElementById('rowAssets');
    var assetsToggle = document.getElementById('assetsToggle');
    if (state.assets_available){
      assetsToggle.disabled = false; rowAssets.classList.remove('disabled');
    } else {
      assetsToggle.checked = false; assetsToggle.disabled = true; rowAssets.classList.add('disabled');
    }

    // Basemap radios (bottom group)
    function onBasemapChange(e){ if (e && e.target && e.target.name === 'basemap'){ setBasemap(e.target.value); } }
    var bm = document.getElementById('basemapGroup');
    bm.addEventListener('change', onBasemapChange);

    // Events: assets
    assetsToggle.addEventListener('change', function(){ loadAssetsLayer(); });

    // Events: segments
    segToggle.addEventListener('change', function(){
      segcat.disabled = !this.checked;
      if (this.checked){ loadSegmentsLayer(segcat.value || "__ALL__"); }
      else { if (LAYER_SEG){ MAP.removeLayer(LAYER_SEG); LAYER_SEG=null; } }
    });
    segcat.addEventListener('change', function(){
      if (segToggle.checked){ loadSegmentsLayer(segcat.value || "__ALL__"); }
    });

    // Events: geocode
    geoToggle.addEventListener('change', function(){
      if (this.checked){
        loadGeocodeLayer(geocat.value, true);
      } else {
        if (LAYER){ MAP.removeLayer(LAYER); LAYER = null; }
      }
    });
    geocat.addEventListener('change', function(){
      var geoVisible = geoToggle.checked;
      loadGeocodeLayer(geocat.value, geoVisible);
    });

    // Home (fit to current geocode bounds)
    document.getElementById('homeBtn').addEventListener('click', function(){
      if (HOME_BOUNDS) MAP.fitBounds(HOME_BOUNDS, { padding:[20,20] });
      if (LAYER_SEG){ try { LAYER_SEG.bringToFront(); } catch(e){} }
    });

    // Export (top bar)
    document.getElementById('exportBtn').addEventListener('click', exportMapPng);

    // Exit
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

    // Opacity slider (affects geocodes, segments, assets)
    var slider = document.getElementById('opacity');
    var opv    = document.getElementById('opv');
    slider.addEventListener('input', function(){
      var v = parseInt(slider.value, 10);
      FILL_ALPHA = v / 100.0;
      opv.textContent = v + '%';
      if (LAYER)        LAYER.setStyle({ fillOpacity: FILL_ALPHA });
      if (LAYER_SEG)    LAYER_SEG.setStyle({ fillOpacity: FILL_ALPHA });
      if (LAYER_ASSETS) LAYER_ASSETS.setStyle({ fillOpacity: FILL_ALPHA, opacity: FILL_ALPHA });
      if (LAYER_SEG){ try { LAYER_SEG.bringToFront(); } catch(e){} }
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
