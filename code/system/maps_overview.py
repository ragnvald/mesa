#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

import os, sys
import configparser
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping
import webview  # pip install pywebview

# -------------------------------
# Paths / constants
# -------------------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE  = os.path.join(BASE_DIR, "../system/config.ini")
PARQUET_FILE = os.path.join(BASE_DIR, "../output/geoparquet/tbl_flat.parquet")
PLOT_CRS     = "EPSG:4326"

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
# Data
# -------------------------------
def load_parquet(path: str) -> gpd.GeoDataFrame:
    try:
        return gpd.read_parquet(path)
    except Exception as e:
        print("Failed to read parquet:", e, file=sys.stderr)
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
    if gdf.crs is None:
        epsg = int(cfg["DEFAULT"].get("workingprojection_epsg", "4326"))
        gdf = gdf.set_crs(epsg=epsg, allow_override=True)
    if str(gdf.crs).upper() != PLOT_CRS:
        try:
            gdf = gdf.to_crs(PLOT_CRS)
        except Exception:
            epsg = int(cfg["DEFAULT"].get("workingprojection_epsg", "4326"))
            gdf = gdf.set_crs(epsg=epsg, allow_override=True).to_crs(PLOT_CRS)
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
# Load / state
# -------------------------------
cfg   = read_config(CONFIG_FILE)
COLS  = get_color_mapping(cfg)
DESC  = get_desc_mapping(cfg)
GDF   = load_parquet(PARQUET_FILE)
if GDF.empty or "name_gis_geocodegroup" not in GDF.columns:
    print("GeoParquet missing or lacks 'name_gis_geocodegroup'", file=sys.stderr)
    sys.exit(1)
CATS = sorted(GDF["name_gis_geocodegroup"].dropna().unique().tolist())

# -------------------------------
# API exposed to JS
# -------------------------------
class Api:
    def get_state(self):
        return {"categories": CATS, "colors": COLS, "descriptions": DESC, "initial": CATS[0] if CATS else None}

    def get_layer(self, geocode_category):
        try:
            df = GDF[GDF["name_gis_geocodegroup"] == geocode_category].copy()
            df = only_A_to_E(df)
            df = to_plot_crs(df, cfg)
            stats = compute_stats(df)
            home  = bounds_to_leaflet(df.total_bounds) if not df.empty else [[0,0],[0,0]]
            gj    = gdf_to_geojson_min(df)
            return {"ok": True, "geojson": gj, "home_bounds": home, "stats": stats}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def js_log(self, msg):
        print("JS:", msg, file=sys.stderr)

api = Api()

# -------------------------------
# HTML / JS (Leaflet + Chart.js + live opacity slider + bulletproof chart resize)
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
<style>
  html, body { height:100%; margin:0; }
  .wrap {
    height:100%;
    display:grid;
    grid-template-columns: 260px 1fr 380px;
    grid-template-rows: 44px 1fr;
    grid-template-areas:
      "ctrl bar stats"
      "ctrl map stats";
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  }
  .ctrl { grid-area: ctrl; border-right: 2px solid #2b3442; padding:10px; background:#eef0f3; position:relative; }
  .bar  { grid-area: bar; display:flex; gap:10px; align-items:center; padding:6px 10px; }
  .map  { grid-area: map; position:relative; background:#ddd; }
  #map  { position:absolute; inset:0; }
  .stats { grid-area: stats; border-left: 2px solid #2b3442; display:flex; flex-direction:column; overflow:hidden; }
  .legend { padding:8px 10px; }
  .legend table { width:100%; border-collapse:collapse; font-size:12px; }
  .legend td { padding:2px 6px; vertical-align:middle; }
  .swatch { display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:6px; border:1px solid #8884; }
  .btn { padding:6px 10px; border:1px solid #ccd; background:#fff; border-radius:6px; cursor:pointer; }
  .btn:active { transform:translateY(1px); }
  select { padding:6px; width:100%; }
  /* Chart container fills the remaining column space */
  #chartBox { flex:1 1 auto; padding:8px 10px; position:relative; }
  /* Canvas sized via JS to exact pixels; default to 100% so first paint fills */
  #chart { position:absolute; inset:8px 10px 8px 12px; width:calc(100% - 22px); height:calc(100% - 16px); }
  #err { color:#b00; padding:6px 10px; font-size:12px; display:none; }
  #exitBtn { position:absolute; bottom:12px; left:10px; }
  .slider { display:flex; align-items:center; gap:8px; }
  .slider input[type=range]{ width:160px; }
  .label-sm { font-size:12px; color:#333; }
</style>
</head>
<body>
<div class="wrap">
  <div class="ctrl">
    <div style="font-weight:600; margin-bottom:6px;">Geocode category:</div>
    <select id="cat"></select>
    <button id="exitBtn" class="btn">Exit</button>
  </div>

  <div class="bar">
    <button id="homeBtn" class="btn">Home</button>
    <button id="fitBtn"  class="btn">Fit layer</button>
    <div class="slider">
      <span class="label-sm">Opacity</span>
      <input id="opacity" type="range" min="10" max="100" value="80">
      <span id="opv" class="label-sm">80%</span>
    </div>
  </div>

  <div class="map"><div id="map"></div></div>

  <div class="stats">
    <div id="err"></div>
    <div class="legend" id="legend"></div>
    <div id="chartBox"><canvas id="chart"></canvas></div>
  </div>
</div>

<script>
var MAP=null, LAYER=null, COLOR_MAP={}, DESC_MAP={}, CURRENT_CAT=null, HOME_BOUNDS=null, CHART=null;
var FILL_ALPHA = 0.8; // default 80%

function logErr(m){ try { window.pywebview.api.js_log(m); } catch(e){} }
function setError(msg){
  var e = document.getElementById('err');
  if (msg){ e.style.display='block'; e.textContent = msg; }
  else { e.style.display='none'; e.textContent=''; }
}

function mkLegend() {
  var container = document.getElementById('legend');
  var html = '<div style="font-weight:600; margin-bottom:6px;">Area by sensitivity code</div><table>';
  var codes = ['A','B','C','D','E'];
  for (var i=0;i<codes.length;i++){
    var c = codes[i];
    var color = COLOR_MAP[c] || '#bdbdbd';
    var desc  = DESC_MAP[c] || '';
    html += '<tr><td><span class="swatch" style="background:'+color+'"></span></td><td style="width:60px;">'+c+'</td><td>'+desc+'</td></tr>';
  }
  html += '</table>';
  container.innerHTML = html;
}

function renderChart(stats) {
  var ctx = document.getElementById('chart').getContext('2d');
  if (CHART) CHART.destroy();

  var colors = [];
  for (var i=0;i<stats.labels.length;i++){
    var code = stats.labels[i];
    colors.push(COLOR_MAP[code] || '#bdbdbd');
  }

  CHART = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: stats.labels,
      datasets: [{
        label: 'km²',
        data: stats.values,
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
      plugins: { legend: { display:false }, title: { display:true, text:'Area by sensitivity code (km²)' } },
      scales: { y: { beginAtZero:true, ticks:{ padding:6 } }, x:{ ticks:{ padding:4 } } },
      layout: { padding: { left: 0, right:0, top:0, bottom:0 } }
    }
  });

  // ensure first fit to container pixels
  resizeChartToBox();
}

function styleFeature(f){
  var c = (f.properties && f.properties.sensitivity_code_max) ? f.properties.sensitivity_code_max : '';
  return { color:'white', weight:0.5, opacity:1.0, fillOpacity:FILL_ALPHA, fillColor: COLOR_MAP[c] || '#bdbdbd' };
}

function loadLayer(cat, preserveView){
  var prev = null;
  if (preserveView && MAP) prev = { center: MAP.getCenter(), zoom: MAP.getZoom() };

  window.pywebview.api.get_layer(cat).then(function(res){
    if (!res.ok){ setError('Failed to load layer: '+res.error); return; }
    setError('');
    CURRENT_CAT = cat;
    HOME_BOUNDS = res.home_bounds;

    if (LAYER){ MAP.removeLayer(LAYER); LAYER=null; }
    LAYER = L.geoJSON(res.geojson, { style: styleFeature, renderer: L.canvas() }).addTo(MAP);

    if (preserveView && prev){
      MAP.setView(prev.center, prev.zoom, { animate:false });
    } else {
      MAP.fitBounds(HOME_BOUNDS, { padding: [20,20] });
    }
    renderChart(res.stats);
  }).catch(function(err){
    setError('API error: '+err); logErr('API error: '+err);
  });
}

// ---------- Responsive resizing for MAP and CHART ----------
function debounce(fn, ms){
  var t; return function(){ var ctx=this, args=arguments; clearTimeout(t); t=setTimeout(function(){ fn.apply(ctx,args); }, ms||50); };
}

function resizeChartToBox(){
  if (!CHART) return;
  var box = document.getElementById('chartBox');
  var canvas = document.getElementById('chart');
  var w = Math.max(0, box.clientWidth);
  var h = Math.max(0, box.clientHeight);
  // Set CSS size explicitly before Chart resize
  canvas.style.width  = w + 'px';
  canvas.style.height = h + 'px';
  // Also drop width/height HTML attributes if any
  canvas.removeAttribute('width');
  canvas.removeAttribute('height');
  // Tell Chart.js to resize to these pixels
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

  // Window resize fallback
  window.addEventListener('resize', function(){ onChartResize(); onMapResize(); });

  // Last-resort polling (covers some embedded WebView edge cases)
  var lastW = -1, lastH = -1;
  setInterval(function(){
    var w = chartBox.clientWidth, h = chartBox.clientHeight;
    if (w !== lastW || h !== lastH){
      lastW = w; lastH = h;
      onChartResize();
    }
  }, 200);
}

function boot(){
  // Leaflet map
  MAP = L.map('map', { zoomControl:true, preferCanvas:true });
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
              { maxZoom:19, attribution:'© OpenStreetMap' }).addTo(MAP);
  MAP.setView([0,0], 2);

  setupResizeObservers();

  // Load state
  window.pywebview.api.get_state().then(function(state){
    COLOR_MAP = state.colors || {};
    DESC_MAP  = state.descriptions || {};
    mkLegend();

    var sel = document.getElementById('cat');
    var cats = state.categories || [];
    for (var i=0;i<cats.length;i++){
      var o = document.createElement('option');
      o.value = cats[i]; o.textContent = cats[i];
      sel.appendChild(o);
    }

    if (state.initial){
      sel.value = state.initial;
      loadLayer(state.initial, false);
    }

    // Events
    sel.addEventListener('change', function(){ loadLayer(sel.value, true); });
    document.getElementById('homeBtn').addEventListener('click', function(){
      if (HOME_BOUNDS) MAP.fitBounds(HOME_BOUNDS, { padding:[20,20] });
    });
    document.getElementById('fitBtn').addEventListener('click', function(){
      if (HOME_BOUNDS) MAP.fitBounds(HOME_BOUNDS, { padding:[20,20] });
    });
    document.getElementById('exitBtn').addEventListener('click', function(){ window.close(); });

    // Opacity slider
    var slider = document.getElementById('opacity');
    var opv    = document.getElementById('opv');
    slider.addEventListener('input', function(){
      var v = parseInt(slider.value, 10);
      FILL_ALPHA = v / 100.0;
      opv.textContent = v + '%';
      if (LAYER) LAYER.setStyle({ fillOpacity: FILL_ALPHA });
    });
  }).catch(function(err){
    setError('Failed to get state: '+err); logErr('get_state failed: '+err);
  });
}

// Ensure API is ready
window.addEventListener('pywebviewready', function(){ boot(); });
// Fallback if pywebviewready never fires
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
        title="Maps Overview",
        html=HTML,
        js_api=api,
        width=1600, height=900, resizable=True
    )
    # Force modern engine on Windows (Edge WebView2)
    webview.start(gui="edgechromium", debug=False)
