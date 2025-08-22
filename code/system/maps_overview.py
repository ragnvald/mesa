#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

import os, sys, base64
import configparser
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping
import webview  # pip install pywebview

# -------------------------------
# Paths / constants
# -------------------------------
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE   = os.path.join(BASE_DIR, "../system/config.ini")
PARQUET_FILE  = os.path.join(BASE_DIR, "../output/geoparquet/tbl_flat.parquet")
SEGMENT_FILE  = os.path.join(BASE_DIR, "../output/geoparquet/tbl_segment_flat.parquet")
PLOT_CRS      = "EPSG:4326"

# -------------------------------
# Config / colors
# -------------------------------
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

# -------------------------------
# Data helpers
# -------------------------------
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
    if gdf.empty:
        return gdf.set_crs(PLOT_CRS, allow_override=True) if gdf.crs is None else gdf
    if gdf.crs is None:
        # workingprojection can be set as integer like 4326
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
    feats = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        feats.append({
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": {"sensitivity_code_max": row.get("sensitivity_code_max", None)}
        })
    return {"type": "FeatureCollection", "features": feats}

def compute_stats(gdf: gpd.GeoDataFrame) -> dict:
    if gdf.empty or "area_m2" not in gdf.columns:
        return {"labels": list("ABCDE"), "values": [0,0,0,0,0]}
    grp = gdf.groupby("sensitivity_code_max")["area_m2"].sum()
    vals = [float(grp.get(c, 0.0))/1e6 for c in "ABCDE"]  # km²
    return {"labels": list("ABCDE"), "values": vals}

# -------------------------------
# Load datasets (robust, non-fatal)
# -------------------------------
cfg          = read_config(CONFIG_FILE)
COLS         = get_color_mapping(cfg)
DESC         = get_desc_mapping(cfg)
GDF          = load_parquet(PARQUET_FILE)
SEG_GDF      = load_parquet(SEGMENT_FILE)

# Availability flags (no crashing)
GEOCODE_AVAILABLE = (not GDF.empty) and ("name_gis_geocodegroup" in GDF.columns)
SEGMENTS_AVAILABLE = (not SEG_GDF.empty) and ("geometry" in SEG_GDF.columns)

# Categories (safe)
CATS = sorted(GDF["name_gis_geocodegroup"].dropna().unique().tolist()) if GEOCODE_AVAILABLE else []
if SEGMENTS_AVAILABLE and "name_gis_geocodegroup" in SEG_GDF.columns:
    SEG_CATS = sorted(SEG_GDF["name_gis_geocodegroup"].dropna().unique().tolist())
else:
    SEG_CATS = []  # empty means "All segments" option only, if segments exist

# Optional Bing key (for aerial tiles)
BING_KEY = cfg["DEFAULT"].get("bing_maps_key", "").strip()

# -------------------------------
# API exposed to JavaScript
# -------------------------------
class Api:
    # Optional logger to silence JS debug calls
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
            "colors": COLS,
            "descriptions": DESC,
            "initial_geocode": (CATS[0] if (GEOCODE_AVAILABLE and CATS) else None),
            "has_segments": SEGMENTS_AVAILABLE,
            "bing_key": BING_KEY
        }

    def get_geocode_layer(self, geocode_category):
        """Return GeoJSON + stats for selected geocode category (or empty if unavailable)."""
        try:
            if not GEOCODE_AVAILABLE:
                # Empty geocode layer & stats
                return {
                    "ok": True,
                    "geojson": {"type":"FeatureCollection","features":[]},
                    "home_bounds": [[0,0],[0,0]],
                    "stats": {"labels": list("ABCDE"), "values": [0,0,0,0,0]}
                }
            df = GDF[GDF["name_gis_geocodegroup"] == geocode_category].copy()
            df = only_A_to_E(df)
            df = to_plot_crs(df, cfg)
            stats = compute_stats(df)
            home  = bounds_to_leaflet(df.total_bounds) if not df.empty else [[0,0],[0,0]]
            gj    = gdf_to_geojson_min(df)
            return {"ok": True, "geojson": gj, "home_bounds": home, "stats": stats}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_segment_layer(self, seg_category_or_all):
        """Return segments overlay; if unavailable, respond with error (UI will grey out anyway)."""
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

    def exit_app(self):
        try:
            webview.destroy_window()
        except Exception:
            os._exit(0)

    def save_png(self, data_url: str):
        """Receive data URL from JS and save PNG via a Save dialog (non-deprecated)."""
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

# -------------------------------
# HTML / JS UI
# -------------------------------
HTML = r"""
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
    grid-template-columns: 300px 1fr 420px;
    grid-template-rows: 48px 1fr;
    grid-template-areas:
      "ctrl bar stats"
      "ctrl map stats";
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  }
  .ctrl   { grid-area: ctrl; border-right: 2px solid #2b3442; padding:10px; background:#eef0f3; position:relative; overflow:auto; }
  .section{ margin-bottom:12px; transition: opacity .15s ease; }
  .section.disabled { opacity: 0.5; }
  .section h3 { margin:8px 0 6px; font-size:14px; display:flex; align-items:center; gap:8px; }
  .indent { padding-left:18px; }
  .bar    { grid-area: bar; display:flex; gap:12px; align-items:center; padding:8px 12px; flex-wrap:wrap; }
  .map    { grid-area: map; position:relative; background:#ddd; }
  #map    { position:absolute; inset:0; }
  /* Hide zoom control only while exporting */
  #map.exporting .leaflet-control-zoom { display: none !important; }
  .stats  { grid-area: stats; border-left: 2px solid #2b3442; display:flex; flex-direction:column; overflow:hidden; }
  .legend { padding:8px 12px; font-size:12px; }
  .legend table { width:100%; border-collapse:collapse; }
  .legend th, .legend td { padding:4px 6px; vertical-align:middle; }
  .legend thead th { border-bottom:1px solid #cdd4de; font-weight:600; }
  .legend tfoot td { border-top:1px solid #cdd4de; font-weight:600; }
  .swatch { display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:6px; border:1px solid #8884; }
  .num { text-align:right; white-space:nowrap; }
  .btn { padding:6px 10px; border:1px solid #ccd; background:#fff; border-radius:6px; cursor:pointer; }
  .btn:active { transform:translateY(1px); }
  select { padding:6px; }
  .chkrow { display:flex; align-items:center; gap:8px; }
  #chartBox { flex:1 1 auto; padding:8px 12px; position:relative; }
  #chart { position:absolute; inset:8px 12px; width:calc(100% - 24px); height:calc(100% - 16px); }
  #err { color:#b00; padding:6px 12px; font-size:12px; display:none; }
  #exitBtn { position:absolute; bottom:12px; left:10px; }
  .slider { display:flex; align-items:center; gap:8px; }
  .slider input[type=range]{ width:160px; }
  .hint { margin-top:6px; opacity:0.75; }
</style>
</head>
<body>
<div class="wrap">
  <div class="ctrl">

    <!-- Segments overlay group (top) -->
    <div class="section" id="segmentsSection">
      <h3 class="chkrow">
        <input type="checkbox" id="segToggle">
        <label for="segToggle" style="font-weight:600;">Sensitivity line segments</label>
      </h3>
      <div class="indent">
        <label for="segcat" class="label-sm">Segment category</label>
        <select id="segcat" disabled></select>
        <div id="segHint" class="label-sm hint"></div>
      </div>
    </div>

    <!-- Geocoded areas group -->
    <div class="section" id="geocodeSection">
      <h3 class="chkrow">
        <input type="checkbox" id="geoToggle" checked>
        <label for="geoToggle" style="font-weight:600;">Geocoded areas</label>
      </h3>
      <div class="indent">
        <label for="cat" class="label-sm">Category</label>
        <select id="cat"></select>
        <div class="label-sm hint" id="geoHint">Controls the statistics on the right.</div>

        <!-- Basemap selector (left pane) -->
        <div style="margin-top:10px;">
          <label for="basemapSelLeft" class="label-sm">Basemap</label>
          <select id="basemapSelLeft" title="Choose basemap">
            <option value="osm">OpenStreetMap</option>
            <option value="topo">OSM Topography</option>
            <option value="sat">Bing Aerial / Satellite</option>
          </select>
        </div>
      </div>
    </div>

    <button id="exitBtn" class="btn">Exit</button>
  </div>

  <div class="bar">
    <button id="homeBtn" class="btn">Home</button>

    <div class="slider">
      <span class="label-sm">Opacity</span>
      <input id="opacity" type="range" min="10" max="100" value="80">
      <span id="opv" class="label-sm">80%</span>
    </div>

    <!-- Single export button -->
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
var MAP=null, BASE=null, BASE_SOURCES=null, LAYER=null, LAYER_SEG=null, COLOR_MAP={}, DESC_MAP={}, HOME_BOUNDS=null, CHART=null;
var FILL_ALPHA = 0.8; // default 80%
var BING_KEY_JS = null;
var SATELLITE_FALLBACK = null;

function logErr(m){ try { window.pywebview.api.js_log(m); } catch(e){} }
function setError(msg){
  var e = document.getElementById('err');
  if (msg){ e.style.display='block'; e.textContent = msg; }
  else { e.style.display='none'; e.textContent=''; }
}

// --- Legend / totals table ---
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

  function fmtKm2(x){ return Number(x||0).toLocaleString('en-US', {maximumFractionDigits:2}); }
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

// --- Responsive resizing (map + chart) ---
function debounce(fn, ms){
  var t; return function(){ var ctx=this, args=arguments; clearTimeout(t); t=setTimeout(function(){ fn.apply(ctx,args); }, ms||50); };
}
function resizeChartToBox(){
  if (!CHART) return;
  var box = document.getElementById('chartBox');
  var canvas = document.getElementById('chart');
  var w = Math.max(0, box.clientWidth);
  var h = Math.max(0, box.clientHeight);
  canvas.style.width  = w + 'px';
  canvas.style.height = h + 'px';
  canvas.removeAttribute('width');
  canvas.removeAttribute('height');
  CHART.resize(w, h);
}
function setupResizeObservers(){
  var chartBox = document.getElementById('chartBox');
  var mapBox   = document.getElementById('map');

  var onChartResize = debounce(function(){ resizeChartToBox(); }, 25);
  var onMapResize   = debounce(function(){ if (MAP) MAP.invalidateSize(false); }, 25);

  if (typeof ResizeObserver !== 'undefined'){
    try {
      var ro1 = new ResizeObserver(onChartResize);
      ro1.observe(chartBox);
      var ro2 = new ResizeObserver(onMapResize);
      ro2.observe(mapBox);
    } catch (e){ logErr('ResizeObserver failed: '+e); }
  }
  window.addEventListener('resize', function(){ onChartResize(); onMapResize(); });

  var lastW = -1, lastH = -1;
  setInterval(function(){
    var w = chartBox.clientWidth, h = chartBox.clientHeight;
    if (w !== lastW || h !== lastH){
      lastW = w; lastH = h;
      onChartResize();
    }
  }, 200);
}

// --- Basemaps (incl. Bing Aerial) ---
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
    ...common,
    attribution: '© OpenStreetMap'
  });

  var topo = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
    ...common,
    subdomains: ['a','b','c'],
    maxZoom: 17,
    attribution: '© OpenStreetMap, © OpenTopoMap (CC-BY-SA)'
  });

  SATELLITE_FALLBACK = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    ...common,
    attribution: 'Esri, Maxar, Earthstar Geographics, and the GIS User Community'
  });

  var bing = new BingAerial({ tileSize: 256 });

  return { osm: osm, topo: topo, sat_bing: bing, sat_esri: SATELLITE_FALLBACK };
}

function setBasemap(kind){
  var center = MAP.getCenter(), zoom = MAP.getZoom();
  if (BASE){ MAP.removeLayer(BASE); BASE = null; }

  if (kind === 'osm'){
    BASE = BASE_SOURCES.osm;
  } else if (kind === 'topo'){
    BASE = BASE_SOURCES.topo;
  } else if (kind === 'sat'){
    BASE = BING_KEY_JS ? BASE_SOURCES.sat_bing : BASE_SOURCES.sat_esri;
    if (!BING_KEY_JS){
      setError('Bing key not set; using Esri World Imagery as satellite fallback.');
      setTimeout(function(){ setError(''); }, 4000);
    }
  } else {
    BASE = BASE_SOURCES.osm;
  }

  BASE.addTo(MAP);
  MAP.setView(center, zoom, { animate:false });
}

// --- Layer styling ---
function styleFeature(f){
  var c = (f.properties && f.properties.sensitivity_code_max) ? f.properties.sensitivity_code_max : '';
  return { color:'white', weight:0.5, opacity:1.0, fillOpacity:FILL_ALPHA, fillColor: COLOR_MAP[c] || '#bdbdbd' };
}
function styleFeatureSeg(f){
  var c = (f.properties && f.properties.sensitivity_code_max) ? f.properties.sensitivity_code_max : '';
  return { pane:'segmentsPane', color:'#f7f7f7', weight:0.7, opacity:1.0, fillOpacity:FILL_ALPHA, fillColor: COLOR_MAP[c] || '#bdbdbd' };
}

// --- Loading geocoded layer (controls stats) ---
function loadGeocodeLayer(cat, preserveView){
  var prev = null;
  if (preserveView && MAP) prev = { center: MAP.getCenter(), zoom: MAP.getZoom() };
  window.pywebview.api.get_geocode_layer(cat).then(function(res){
    if (!res.ok){ setError('Failed to load geocode: '+res.error); return; }
    setError('');
    HOME_BOUNDS = res.home_bounds;

    var geoVisible = document.getElementById('geoToggle').checked;
    if (geoVisible){
      if (LAYER){ MAP.removeLayer(LAYER); LAYER = null; }
      LAYER = L.geoJSON(res.geojson, { style: styleFeature, pane:'geocodePane', renderer: L.canvas() }).addTo(MAP);
      if (LAYER_SEG){ try { LAYER_SEG.bringToFront(); } catch(e){} }
      if (preserveView && prev){
        MAP.setView(prev.center, prev.zoom, { animate:false });
      } else {
        MAP.fitBounds(HOME_BOUNDS, { padding: [20,20] });
      }
    }
    renderLegend(res.stats);
    renderChart(res.stats);
  }).catch(function(err){
    setError('API error: '+err); logErr('API error: '+err);
  });
}

// --- Loading / reloading segments overlay (top) ---
function loadSegmentsLayer(segCatOrAll){
  if (!document.getElementById('segToggle').checked){
    if (LAYER_SEG){ MAP.removeLayer(LAYER_SEG); LAYER_SEG=null; }
    return;
  }
  window.pywebview.api.get_segment_layer(segCatOrAll).then(function(res){
    if (!res.ok){ setError('Failed to load segments: '+res.error); return; }
    setError('');
    if (LAYER_SEG){ MAP.removeLayer(LAYER_SEG); LAYER_SEG = null; }
    LAYER_SEG = L.geoJSON(res.geojson, { style: styleFeatureSeg, pane:'segmentsPane', renderer: L.canvas() }).addTo(MAP);
    try { LAYER_SEG.bringToFront(); } catch(e){}
  }).catch(function(err){
    setError('API error: '+err); logErr('API error: '+err);
  });
}

// --- Export current map view to PNG (hide zoom buttons during capture) ---
function exportMapPng(){
  var node = document.getElementById('map');

  function capture(){
    return html2canvas(node, {
      useCORS: true,
      allowTaint: false,
      backgroundColor: '#ffffff',
      scale: 3  // ~300 dpi on 96-dpi basis
    });
  }

  // Temporarily hide zoom buttons by adding a class
  node.classList.add('exporting');
  var baseWasOn = BASE && MAP.hasLayer(BASE);

  capture().then(function(canvas){
    node.classList.remove('exporting');
    var png = canvas.toDataURL('image/png');
    return window.pywebview.api.save_png(png);
  }).then(function(res){
    if (!res.ok){ setError('Save cancelled or failed: ' + (res.error || '')); setTimeout(function(){ setError(''); }, 4000); }
  }).catch(function(){
    // Retry without basemap (in case of CORS tainting)
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

// --- Boot
function boot(){
  MAP = L.map('map', { zoomControl:true, preferCanvas:true });
  MAP.createPane('geocodePane');  MAP.getPane('geocodePane').style.zIndex  = 450;
  MAP.createPane('segmentsPane'); MAP.getPane('segmentsPane').style.zIndex = 650;

  // Add a scale bar bottom-left
  L.control.scale({ position:'bottomleft', metric:true, imperial:false, maxWidth:200 }).addTo(MAP);

  BASE_SOURCES = buildBaseSources();
  BASE = BASE_SOURCES.osm.addTo(MAP);

  MAP.setView([0,0], 2);
  setupResizeObservers();

  // Load state
  window.pywebview.api.get_state().then(function(state){
    COLOR_MAP   = state.colors || {};
    DESC_MAP    = state.descriptions || {};
    BING_KEY_JS = (state.bing_key || '').trim() || null;

    renderLegend(null); // initial empty table
    renderChart({labels:['A','B','C','D','E'], values:[0,0,0,0,0]}); // empty chart

    // Geocode controls
    var geocodeSection = document.getElementById('geocodeSection');
    var geoToggle = document.getElementById('geoToggle');
    var geocat = document.getElementById('cat');
    var geoHint = document.getElementById('geoHint');

    if (state.geocode_available){
      var geocodes = state.geocode_categories || [];
      for (var i=0;i<geocodes.length;i++){
        var o = document.createElement('option');
        o.value = geocodes[i]; o.textContent = geocodes[i];
        geocat.appendChild(o);
      }
      if (state.initial_geocode){
        geocat.value = state.initial_geocode;
        loadGeocodeLayer(state.initial_geocode, false);
      }
      geoToggle.disabled = false;
      geocat.disabled = false;
      geocodeSection.classList.remove('disabled');
      geoHint.textContent = "Controls the statistics on the right.";
    } else {
      // Grey out and disable entire geocode group
      geoToggle.checked = false;
      geoToggle.disabled = true;
      geocat.disabled = true;
      geocodeSection.classList.add('disabled');
      geocat.innerHTML = '';
      geoHint.textContent = "Geocoded areas are not available.";
      // keep legend/chart empty (already rendered)
    }

    // Segments controls
    var segmentsSection = document.getElementById('segmentsSection');
    var segToggle = document.getElementById('segToggle');
    var segcat    = document.getElementById('segcat');
    var segHint   = document.getElementById('segHint');

    if (state.segment_available){
      var segcats = state.segment_categories || [];
      segcat.innerHTML = '';
      if (segcats.length === 0){
        var o = document.createElement('option');
        o.value = "__ALL__"; o.textContent = "All segments";
        segcat.appendChild(o);
        segHint.textContent = "Line segments only.";
      } else {
        for (var i=0;i<segcats.length;i++){
          var so = document.createElement('option');
          so.value = segcats[i]; so.textContent = segcats[i];
          segcat.appendChild(so);
        }
        segHint.textContent = "Choose an optional segments category to overlay.";
      }
      segToggle.disabled = false;
      segmentsSection.classList.remove('disabled');
    } else {
      // Grey out and disable segments group
      segToggle.checked = false;
      segToggle.disabled = true;
      segcat.disabled = true;
      segmentsSection.classList.add('disabled');
      segHint.textContent = "Segments dataset not available.";
    }

    // Events: Segments
    segToggle.addEventListener('change', function(){
      segcat.disabled = !this.checked;
      if (this.checked){
        loadSegmentsLayer(segcat.value || "__ALL__");
      } else {
        if (LAYER_SEG){ MAP.removeLayer(LAYER_SEG); LAYER_SEG=null; }
      }
    });
    segcat.addEventListener('change', function(){
      if (segToggle.checked){
        loadSegmentsLayer(segcat.value || "__ALL__");
      }
    });

    // Events: geocode
    geoToggle.addEventListener('change', function(){
      if (this.checked){
        loadGeocodeLayer(geocat.value, true);
      } else {
        if (LAYER){ MAP.removeLayer(LAYER); LAYER = null; }
        if (LAYER_SEG){ try { LAYER_SEG.bringToFront(); } catch(e){} }
        // Clear stats when layer hidden
        renderLegend({labels:['A','B','C','D','E'], values:[0,0,0,0,0]});
        renderChart({labels:['A','B','C','D','E'], values:[0,0,0,0,0]});
      }
    });
    geocat.addEventListener('change', function(){
      var geoVisible = geoToggle.checked;
      loadGeocodeLayer(geocat.value, geoVisible);
    });

    // Left-pane basemap selector
    document.getElementById('basemapSelLeft').addEventListener('change', function(){
      setBasemap(this.value);
    });

    // Home (fit to current geocode bounds)
    document.getElementById('homeBtn').addEventListener('click', function(){
      if (HOME_BOUNDS) MAP.fitBounds(HOME_BOUNDS, { padding:[20,20] });
      if (LAYER_SEG){ try { LAYER_SEG.bringToFront(); } catch(e){} }
    });

    // Export (top bar)
    document.getElementById('exportBtn').addEventListener('click', exportMapPng);

    // Exit button
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

    // Opacity slider (both layers)
    var slider = document.getElementById('opacity');
    var opv    = document.getElementById('opv');
    slider.addEventListener('input', function(){
      var v = parseInt(slider.value, 10);
      FILL_ALPHA = v / 100.0;
      opv.textContent = v + '%';
      if (LAYER)     LAYER.setStyle({ fillOpacity: FILL_ALPHA });
      if (LAYER_SEG) LAYER_SEG.setStyle({ fillOpacity: FILL_ALPHA });
      if (LAYER_SEG){ try { LAYER_SEG.bringToFront(); } catch(e){} }
    });
  }).catch(function(err){
    setError('Failed to get state: '+err); logErr('get_state failed: '+err);
  });
}

// Ensure API is ready
window.addEventListener('pywebviewready', function(){ boot(); });
document.addEventListener('DOMContentLoaded', function(){ if (window.pywebview && window.pywebview.api) { boot(); } });
</script>
</body>
</html>
"""

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    window = webview.create_window(
        title="Maps overview",
        html=HTML,
        js_api=api,
        width=1300, height=800, resizable=True
    )
    webview.start(gui="edgechromium", debug=False)
