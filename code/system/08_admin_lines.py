
# -*- coding: utf-8 -*-
"""
08_edit_lines_web_v8.py — Leaflet editor (UI polish)
- Removed the Diagnostics button (as requested)
- Footer no longer prints verbose CRS JSON; shows short CRS label (e.g., "EPSG:4326")
- Adds CSS to prevent footer text wrapping into multiple lines
"""

import os, sys, uuid, threading, locale, configparser, argparse, warnings
from typing import Any, Dict, Optional

# Force Edge backend (avoid WinForms noise)
os.environ['PYWEBVIEW_GUI'] = 'edgechromium'
os.environ.setdefault('PYWEBVIEW_LOG', 'error')

try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except Exception:
    pass

try:
    import webview
    try:
        webview.logger.disabled = True
    except Exception:
        pass
except ModuleNotFoundError:
    sys.stderr.write("ERROR: 'pywebview' not installed. pip install pywebview\n")
    raise

import pandas as pd
import geopandas as gpd
import pyarrow as pa  # ensure available
from shapely.geometry import shape as shp_from_geojson, mapping as shp_to_geojson
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely import wkb

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- Path helpers ----------------
def resolve_base_dir(passed: Optional[str]) -> str:
    base = passed or os.getcwd()
    if os.path.basename(base).lower() == "system":
        base = os.path.abspath(os.path.join(base, ".."))
    return base

def gpq_dir(base_dir: str) -> str:
    d = os.path.join(base_dir, "output", "geoparquet")
    os.makedirs(d, exist_ok=True)
    return d

def lines_parquet_path(base_dir: str) -> str:
    return os.path.join(gpq_dir(base_dir), "tbl_lines.parquet")

def config_path(base_dir: str) -> str:
    return os.path.join(base_dir, "system", "config.ini")

def read_config(path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    try:
        cfg.read(path, encoding="utf-8")
    except Exception:
        pass
    if "DEFAULT" not in cfg:
        cfg["DEFAULT"] = {}
    return cfg

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
    try:
        gdf = gpd.read_parquet(pq)  # pyarrow
        if not _geometry_valid(gdf):
            raise ValueError("GeoParquet loaded but geometry column invalid")
    except Exception:
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

def crs_short_label(crs_obj) -> str:
    try:
        # GeoPandas/pyproj CRS
        to_epsg = getattr(crs_obj, "to_epsg", None)
        if callable(to_epsg):
            epsg = to_epsg()
            if epsg:
                return f"EPSG:{epsg}"
        to_string = getattr(crs_obj, "to_string", None)
        if callable(to_string):
            s = to_string()
        else:
            s = str(crs_obj)
        if "EPSG:" in s.upper():
            # Extract first occurrence
            idx = s.upper().find("EPSG:")
            return s[idx:idx+10].split()[0].strip(",;")
        # Fallback compact
        return "CRS set"
    except Exception:
        return "CRS set"

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

# ---------------- API ----------------
class Api:
    __slots__ = ("_base_dir", "_pq_path", "_lock", "_gdf", "_storage_epsg")

    def __init__(self, base_dir: str, _cfg: configparser.ConfigParser):
        self._base_dir = base_dir
        self._pq_path = lines_parquet_path(base_dir)
        self._lock = threading.Lock()

        self._gdf = load_lines_gdf_any(self._pq_path, 4326)
        try:
            self._storage_epsg = int(str(self._gdf.crs).split(":")[-1]) if "EPSG" in str(self._gdf.crs).upper() else 4326
        except Exception:
            self._storage_epsg = 4326

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
            return None
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

    def _save(self):
        ok = save_lines_gdf(self._pq_path, self._gdf)
        return {"ok": bool(ok), "count": int(len(self._gdf))}

    # ---- API methods ----
    def get_state(self):
        with self._lock:
            try:
                recs = [{"name_gis": str(r.get("name_gis","") or ""), "name_user": str(r.get("name_user","") or "")}
                        for _, r in self._gdf[["name_gis","name_user"]].iterrows()]
                diag = {
                    "rows": int(len(self._gdf)),
                    "crs": crs_short_label(self._gdf.crs),
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
                return self._save()
            except Exception as e:
                return {"ok": False, "error": str(e)}

    def add_line(self, feature: Dict[str,Any]):
        with self._lock:
            try:
                geom_ll = shp_from_geojson(feature.get("geometry", None))
                if geom_ll is None:
                    return {"ok": False, "error": "Missing geometry."}
                geom_s = from_epsg4326_to_storage(geom_ll, self._storage_epsg)
                new_id = f"line_{uuid.uuid4().hex[:8]}"
                while (self._gdf["name_gis"].astype(str) == new_id).any():
                    new_id = f"line_{uuid.uuid4().hex[:8]}"
                row = {
                    "name_gis": new_id, "name_user":"",
                    "segment_length": pd.NA, "segment_width": pd.NA,
                    "description":"", "geometry": geom_s
                }
                self._gdf = pd.concat([self._gdf, gpd.GeoDataFrame([row], geometry="geometry", crs=self._gdf.crs)], ignore_index=True)
                save_res = self._save()
                if not save_res.get("ok", False):
                    return save_res

                g = gpd.GeoDataFrame([row], geometry="geometry", crs=self._gdf.crs)
                g_ll = to_epsg4326(g)
                feat = {
                    "type":"Feature",
                    "geometry": shp_to_geojson(g_ll.geometry.iloc[0]),
                    "properties": {"name_gis": new_id, "name_user":"", "segment_length": None, "segment_width": None, "description": ""}
                }
                return {"ok": True, "feature": feat, "record": {"name_gis": new_id, "name_user": ""}}
            except Exception as e:
                return {"ok": False, "error": str(e)}

    def update_geometry(self, name_gis: str, geometry: Dict[str,Any]):
        with self._lock:
            try:
                geom_ll = shp_from_geojson(geometry)
                if geom_ll is None:
                    return {"ok": False, "error": "Invalid geometry."}
                geom_s = from_epsg4326_to_storage(geom_ll, self._storage_epsg)
                sel = self._gdf["name_gis"].astype(str).str.strip() == str(name_gis).strip()
                if not sel.any():
                    return {"ok": False, "error": f"Line '{name_gis}' not found."}
                self._gdf.loc[sel, "geometry"] = [geom_s] * sel.sum()
                return self._save()
            except Exception as e:
                return {"ok": False, "error": str(e)}

    def delete_line(self, name_gis: str):
        with self._lock:
            try:
                before = len(self._gdf)
                self._gdf = self._gdf[self._gdf["name_gis"].astype(str).str.strip() != str(name_gis).strip()].reset_index(drop=True)
                if len(self._gdf) == before:
                    return {"ok": False, "error": f"Line '{name_gis}' not found."}
                return self._save()
            except Exception as e:
                return {"ok": False, "error": str(e)}

    def exit_app(self):
        try: webview.destroy_window()
        except Exception: os._exit(0)

# ---------------- HTML/JS ----------------
HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Edit Lines / Segments (GeoParquet)</title>
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
  .form { grid-area: form; border-right:2px solid #2b3442; padding:10px 12px; overflow:auto; }
  .map { grid-area: map; position:relative; }
  #map { position:absolute; inset:0; }
  .foot { grid-area: foot; font-size:12px; color:#475569; padding:4px 10px; border-top:1px solid #e2e8f0; display:flex; justify-content:space-between; align-items:center; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .btn { padding:6px 10px; border:1px solid #ccd; background:#fff; border-radius:6px; cursor:pointer; }
  .btn:active { transform:translateY(1px); }
  .row { margin-bottom:10px; }
  label { display:block; font-weight:600; margin-bottom:4px; }
  input[type=text], input[type=number], textarea { width:100%; box-sizing:border-box; padding:8px; border:1px solid #cbd5e1; border-radius:6px; }
  input[readonly] { background:#f8fafc; color:#64748b; }
  textarea { resize: vertical; min-height: 64px; }
  #status { min-height: 18px; margin-top:4px; }
  .status-ok { color:#166534; }
  .status-warn { color:#b45309; }
  .status-err { color:#b91c1c; }
  .hint { font-size:12px; color:#475569; margin-left:8px; }
  .picker { margin-bottom:12px; border-bottom:1px dashed #cbd5e1; padding-bottom:10px; }
  .picker h3 { margin:0 0 6px 0; font-size:14px; }
  .picker input { width:100%; margin-bottom:6px; }
  .list { max-height:220px; overflow:auto; border:1px solid #e2e8f0; border-radius:6px; }
  .item { padding:6px 8px; border-bottom:1px solid #eef2f7; cursor:pointer; }
  .item:hover { background:#f8fafc; }
  .item.active { background:#e6f3ff; }
  .toolbar { display:flex; gap:6px; margin:6px 0 10px; }
  .kbd { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:11px; border:1px solid #cbd5e1; padding:1px 4px; border-radius:4px; background:#f8fafc; }
</style>
</head>
<body>
<div class="wrap">
  <div class="bar">
    <button id="homeBtn" class="btn">Home</button>
    <button id="reloadBtn" class="btn">Reload</button>
    <button id="exitBtn" class="btn">Exit</button>
    <div class="hint">New: <span class="kbd">N</span> • Finish: <span class="kbd">Enter</span> or double-click • Cancel: <span class="kbd">Esc</span> • Edit: <span class="kbd">E</span> • Delete: <span class="kbd">Del</span></div>
  </div>

  <div class="form">
    <div class="picker">
      <h3>Lines</h3>
      <input id="filter" type="text" placeholder="Filter by title or GIS id…">
      <div class="toolbar">
        <button id="newBtn" class="btn" title="Start drawing a new line (N)">New line</button>
        <button id="editBtn" class="btn" title="Toggle edit vertices (E)">Edit</button>
        <button id="delBtn" class="btn" title="Toggle delete mode (Del)">Delete</button>
      </div>
      <div id="list" class="list"></div>
    </div>

    <div class="row"><label>GIS name</label><input id="f_name_gis" type="text" readonly></div>
    <div class="row"><label>Title</label><input id="f_name_user" type="text" placeholder="Enter a user-friendly title"></div>
    <div class="row"><label>Length of segments (m)</label><input id="f_seg_len" type="number" inputmode="numeric" placeholder="e.g., 500"></div>
    <div class="row"><label>Segments width (m)</label><input id="f_seg_wid" type="number" inputmode="numeric" placeholder="e.g., 20"></div>
    <div class="row"><label>Description</label><textarea id="f_desc" placeholder="Short description"></textarea></div>
    <div id="status" class="status-warn">Initializing…</div>
  </div>

  <div class="map"><div id="map"></div></div>

  <div class="foot"><div id="foot-left">Ready.</div><div id="foot-right"></div></div>
</div>

<script>
// Report JS errors to the UI
window.onerror = function(message, source, lineno, colno, error){
  const txt = 'JS error: ' + message + ' @' + lineno + ':' + colno;
  const el = document.getElementById('status');
  if (el){ el.textContent = txt; el.className = 'status-err'; }
};

let MAP, BASE_OSM, BASE_SAT, LINES_GROUP, DRAW_CONTROL;
let HOME_BOUNDS = null;
let SELECTED_ID = null;
let LAYER_BY_ID = {};
let DRAW_HANDLER = null;
let EDIT_ACTIVE = false, DELETE_ACTIVE = false;
let DIAG = null;

function showStatus(m, cls){ const el = document.getElementById('status'); el.textContent = m || ''; el.className = cls || ''; }
function setFootLeft(m){ document.getElementById('foot-left').textContent = m || ''; }
function setFootRight(m){ document.getElementById('foot-right').textContent = m || ''; }
function clearStatus(){ showStatus('',''); }

function fillForm(p){
  document.getElementById('f_name_gis').value = p?.name_gis || '';
  document.getElementById('f_name_user').value = p?.name_user || '';
  document.getElementById('f_seg_len').value  = (p?.segment_length ?? '');
  document.getElementById('f_seg_wid').value  = (p?.segment_width ?? '');
  document.getElementById('f_desc').value     = p?.description || '';
}

function setSelected(id, fly=false){
  SELECTED_ID = id || null;
  document.querySelectorAll('.item').forEach(el => el.classList.toggle('active', el.dataset.id === (id||'')));
  const layer = id ? LAYER_BY_ID[id] : null;
  const props = (layer && layer.feature) ? layer.feature.properties : {};
  fillForm(props||{});
  if (fly && layer){
    try { MAP.fitBounds(layer.getBounds(), {padding:[20,20]}); } catch(e){}
  }
}

function updateList(records){
  const list = document.getElementById('list');
  list.innerHTML = '';
  records.forEach(rec => {
    const div = document.createElement('div');
    div.className = 'item';
    div.dataset.id = rec.name_gis;
    const title = (rec.name_user && rec.name_user.trim().length>0) ? rec.name_user : '(untitled)';
    div.textContent = `${title} — ${rec.name_gis}`;
    div.addEventListener('click', () => setSelected(rec.name_gis, true));
    list.appendChild(div);
  });
  if (SELECTED_ID){
    document.querySelectorAll('.item').forEach(el => el.classList.toggle('active', el.dataset.id === SELECTED_ID));
  }
  // Only show compact CRS label
  setFootRight(records.length + ' line(s)');
}

function bindAuto(fieldId, fieldName){
  const el = document.getElementById(fieldId);
  el.addEventListener('change', function(){
    if (!SELECTED_ID) return;
    clearStatus();
    window.pywebview.api.update_attribute(SELECTED_ID, fieldName, el.value).then(function(res){
      if (!res.ok){ showStatus(res.error || 'Save failed', 'status-err'); }
      else {
        const layer = LAYER_BY_ID[SELECTED_ID];
        if (layer && layer.feature){ layer.feature.properties[fieldName] = (fieldName==='segment_length'||fieldName==='segment_width') ? (el.value===''?null:Number(el.value)) : el.value; }
        if (fieldName === 'name_user'){
          const item = document.querySelector(`.item[data-id="${SELECTED_ID}"]`);
          if (item){
            const title = (el.value && el.value.trim().length>0) ? el.value : '(untitled)';
            item.textContent = `${title} — ${SELECTED_ID}`;
          }
        }
        showStatus('Saved', 'status-ok');
      }
    }).catch(function(err){ showStatus('API error: '+err, 'status-err'); });
  });
}

function initMap(){
  MAP = L.map('map');
  BASE_OSM = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom:19, crossOrigin:true, attribution:'© OpenStreetMap'}).addTo(MAP);
  BASE_SAT = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {maxZoom:19, crossOrigin:true, attribution:'Esri, Maxar'});
  const bases = {'OpenStreetMap': BASE_OSM, 'Satellite (Esri)': BASE_SAT};
  L.control.layers(bases, {}, {collapsed:false, position:'topright'}).addTo(MAP);
  L.control.scale({metric:true, imperial:false}).addTo(MAP);
  LINES_GROUP = L.featureGroup().addTo(MAP);

  DRAW_CONTROL = new L.Control.Draw({
    edit: { featureGroup: LINES_GROUP, poly: { allowIntersection:false } },
    draw: { polygon:false, circle:false, rectangle:false, marker:false, circlemarker:false, polyline:{ shapeOptions:{ color:'#ff7f50', weight:4 } } }
  });
  MAP.addControl(DRAW_CONTROL);

  MAP.on('draw:drawstart', function(e){ if (e.layerType === 'polyline'){ DRAW_HANDLER = e.handler || (DRAW_CONTROL && DRAW_CONTROL._toolbars.draw && DRAW_CONTROL._toolbars.draw._modes.polyline && DRAW_CONTROL._toolbars.draw._modes.polyline.handler) || null; } });
  MAP.on('draw:drawstop', function(e){ DRAW_HANDLER = null; });

  MAP.on(L.Draw.Event.CREATED, function(e){
    clearStatus();
    const tmp = e.layer; LINES_GROUP.addLayer(tmp);
    const gj = tmp.toGeoJSON();
    window.pywebview.api.add_line(gj).then(function(res){
      if (!res.ok){ showStatus(res.error || 'Add failed','status-err'); LINES_GROUP.removeLayer(tmp); return; }
      LINES_GROUP.removeLayer(tmp); addFeature(res.feature, true);
      if (res.record){ const items = Array.from(document.querySelectorAll('.item')).map(x => ({name_gis:x.dataset.id, name_user:x.textContent.split(' — ')[0]})); items.push(res.record); updateList(items); }
      setFootLeft('Added new line: ' + res.record.name_gis);
    }).catch(err => { showStatus('API error: '+err, 'status-err'); LINES_GROUP.removeLayer(tmp); });
  });

  MAP.on(L.Draw.Event.EDITED, function(e){
    clearStatus();
    e.layers.eachLayer(function(layer){
      const f = layer.feature; if (!f || !f.properties || !f.properties.name_gis) return;
      window.pywebview.api.update_geometry(f.properties.name_gis, layer.toGeoJSON().geometry).then(function(res){
        if (!res.ok){ showStatus(res.error || 'Geometry save failed','status-err'); }
        else setFootLeft('Geometry saved for ' + f.properties.name_gis);
      }).catch(err => showStatus('API error: '+err,'status-err'));
    });
  });

  MAP.on(L.Draw.Event.DELETED, function(e){
    clearStatus();
    const ids = [];
    e.layers.eachLayer(function(layer){ const f = layer.feature; if (f && f.properties && f.properties.name_gis) ids.push(f.properties.name_gis); });
    (async function(){
      for (let i=0;i<ids.length;i++){
        try{
          const res = await window.pywebview.api.delete_line(ids[i]);
          if (!res.ok){ showStatus(res.error || 'Delete failed','status-err'); }
          delete LAYER_BY_ID[ids[i]];
          const node = document.querySelector(`.item[data-id="${ids[i]}"]`);
          if (node) node.remove();
          if (SELECTED_ID===ids[i]) setSelected(null);
          setFootLeft('Deleted ' + ids[i]);
        } catch(err){ showStatus('API error: '+err,'status-err'); }
      }
      setFootRight(document.querySelectorAll('.item').length + ' line(s)');
    })();
  });

  setTimeout(()=> MAP.invalidateSize(), 120);
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

function startNewLine(){
  clearStatus();
  const h = DRAW_CONTROL && DRAW_CONTROL._toolbars.draw && DRAW_CONTROL._toolbars.draw._modes.polyline && DRAW_CONTROL._toolbars.draw._modes.polyline.handler;
  if (h){ h.enable(); setFootLeft('Drawing: click to add points, double-click or Enter to finish.'); }
}
function toggleEdit(){
  clearStatus();
  if (!DRAW_CONTROL || !DRAW_CONTROL._toolbars || !DRAW_CONTROL._toolbars.edit) return;
  const h = DRAW_CONTROL._toolbars.edit._modes.edit && DRAW_CONTROL._toolbars.edit._modes.edit.handler;
  if (!h) return;
  if (EDIT_ACTIVE){ h.disable(); EDIT_ACTIVE=false; setFootLeft(''); }
  else { h.enable(); EDIT_ACTIVE=true; DELETE_ACTIVE=false; const r = DRAW_CONTROL._toolbars.edit._modes.remove && DRAW_CONTROL._toolbars.edit._modes.remove.handler; if (r) r.disable(); setFootLeft('Edit mode: drag vertices, then click Save.'); }
}
function toggleDelete(){
  clearStatus();
  if (!DRAW_CONTROL || !DRAW_CONTROL._toolbars || !DRAW_CONTROL._toolbars.edit) return;
  const r = DRAW_CONTROL._toolbars.edit._modes.remove && DRAW_CONTROL._toolbars.edit._modes.remove.handler;
  if (!r) return;
  if (DELETE_ACTIVE){ r.disable(); DELETE_ACTIVE=false; setFootLeft(''); }
  else { r.enable(); DELETE_ACTIVE=true; EDIT_ACTIVE=false; const h = DRAW_CONTROL._toolbars.edit._modes.edit && DRAW_CONTROL._toolbars.edit._modes.edit.handler; if (h) h.disable(); setFootLeft('Delete mode: click lines to remove, then Save.'); }
}

async function loadAll(){
  try{
    const st = await window.pywebview.api.get_state();
    if (!st.ok){ showStatus(st.error || 'State failed', 'status-err'); return; }
    DIAG = st.diag || null;
    showStatus('Connected. Reading '+(DIAG?DIAG.path:'?'), 'status-ok');
    initMap();
    updateList(st.records || []);
    setFootLeft('Loaded '+ (DIAG ? DIAG.rows : '?') +' row(s) from '+ (DIAG ? DIAG.path : '?'));

    const res = await window.pywebview.api.get_all_lines();
    if (!res.ok){ showStatus(res.error || 'Load failed', 'status-err'); return; }
    const feats = res.geojson && res.geojson.features ? res.geojson.features : [];
    feats.forEach(f => addFeature(f, false));
    HOME_BOUNDS = res.home_bounds || null;
    setTimeout(function(){
      if (feats.length > 0 && HOME_BOUNDS) MAP.fitBounds(HOME_BOUNDS, {padding:[20,20]});
      else MAP.setView([59.9139,10.7522], 5);
    }, 80);
  } catch(err){
    showStatus('API error: '+err, 'status-err');
  }
}

function applyFilter(){
  const q = (document.getElementById('filter').value || '').toLowerCase();
  document.querySelectorAll('.item').forEach(el => {
    const t = el.textContent.toLowerCase();
    el.style.display = t.includes(q) ? '' : 'none';
  });
}

function boot(){
  try{
    document.getElementById('homeBtn').addEventListener('click', ()=> { if (HOME_BOUNDS) MAP.fitBounds(HOME_BOUNDS, {padding:[20,20]}); });
    document.getElementById('reloadBtn').addEventListener('click', ()=> { if (window.pywebview && window.pywebview.api) loadAll(); });
    document.getElementById('exitBtn').addEventListener('click', ()=> window.pywebview.api.exit_app());
    document.getElementById('newBtn').addEventListener('click', startNewLine);
    document.getElementById('editBtn').addEventListener('click', toggleEdit);
    document.getElementById('delBtn').addEventListener('click', toggleDelete);
    document.getElementById('filter').addEventListener('input', applyFilter);

    bindAuto('f_name_user','name_user');
    bindAuto('f_seg_len','segment_length');
    bindAuto('f_seg_wid','segment_width');
    bindAuto('f_desc','description');

    if (window.pywebview && window.pywebview.api) {
      loadAll();
    } else {
      showStatus('Waiting for backend…', 'status-warn');
      window.addEventListener('pywebviewready', () => {
        showStatus('Backend ready. Loading…', 'status-ok');
        loadAll();
      });
      let tries = 0;
      const tic = setInterval(()=>{
        if (window.pywebview && window.pywebview.api){
          clearInterval(tic);
          showStatus('Backend ready. Loading…', 'status-ok');
          loadAll();
        } else if (++tries > 50) {
          clearInterval(tic);
          showStatus('Backend did not initialize. Are you opening the HTML directly?', 'status-err');
        }
      }, 100);
    }
  } catch(e){
    showStatus('Boot error: '+e, 'status-err');
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
    cfg = read_config(config_path(base_dir))

    api = Api(base_dir, cfg)
    window = webview.create_window(title="Edit lines / segments (GeoParquet)", html=HTML, js_api=api, width=1280, height=860)

    try:
        webview.start(gui='edgechromium', debug=False)
    except Exception as e:
        sys.stderr.write(
            "This app requires Microsoft Edge WebView2 runtime to avoid console spam.\n"
            "Install 'Evergreen WebView2 Runtime' from Microsoft and retry.\n"
            f"Underlying error: {e}\n"
        )
        sys.exit(1)

if __name__ == "__main__":
    main()
