#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive asset layer viewer dedicated to per-group asset overlays."""

from __future__ import annotations

import base64
import configparser
import io
import locale
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping

try:
    import webview  # type: ignore
except ModuleNotFoundError:
    sys.stderr.write(
        "ERROR: 'pywebview' is not installed in the Python environment launching map_assets.py.\n"
        "Install it in that environment, e.g.:  pip install pywebview\n"
    )
    sys.exit(1)

try:
    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
except Exception:
    pass


def base_dir() -> Path:
    """Resolve the Mesa repo root regardless of how the script is launched."""

    candidates: List[Path] = []
    try:
      owdir = globals().get("original_working_directory")  # type: ignore[name-defined]
      if owdir:
        candidates.append(Path(owdir))
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
        if p.name.lower() in {"tools", "system", "code"}:
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

    for candidate in candidates:
        root = normalize(candidate)
        if (root / "tools").exists() or ((root / "output").exists() and (root / "input").exists()):
            return root
    return normalize(candidates[0])


SCRIPT_DIR = Path(__file__).resolve().parent
APP_DIR = base_dir()
os.chdir(APP_DIR)

CONFIG_FILE = APP_DIR / "config.ini"
OUTPUT_DIR = APP_DIR / "output"
PARQUET_DIR = OUTPUT_DIR / "geoparquet"
ASSET_OBJECT_FILE = PARQUET_DIR / "tbl_asset_object.parquet"
ASSET_GROUP_FILE = PARQUET_DIR / "tbl_asset_group.parquet"
ASSET_HIERARCHY_FILE = PARQUET_DIR / "tbl_asset_hierarchy.parquet"
LOG_FILE = SCRIPT_DIR / "log.txt"


def log_event(message: str) -> None:
    try:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        with LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass


log_event(f"map_assets.py starting (APP_DIR={APP_DIR})")


def read_config(path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";",), strict=False)
    try:
        cfg.read(path, encoding="utf-8")
    except Exception:
        pass
    return cfg


def _safe_hex(value: str | None, fallback: str = "#BDBDBD") -> str:
    return (value or "").strip() or fallback


def get_color_mapping(cfg: configparser.ConfigParser) -> Dict[str, str]:
    default_unknown = _safe_hex(cfg["DEFAULT"].get("category_colour_unknown", "#BDBDBD"))
    colors: Dict[str, str] = {}
    for code in "ABCDE":
        if cfg.has_section(code):
            colors[code] = _safe_hex(cfg[code].get("category_colour", default_unknown), default_unknown)
        else:
            colors[code] = default_unknown
    colors["UNKNOWN"] = default_unknown
    return colors


def to_epsg4326(gdf: gpd.GeoDataFrame, cfg: configparser.ConfigParser) -> gpd.GeoDataFrame:
    if gdf.empty:
        if gdf.crs is None:
            return gdf.set_crs(4326, allow_override=True)
        return gdf.to_crs(4326)
    if gdf.crs is None:
        try:
            epsg = int(cfg["DEFAULT"].get("workingprojection_epsg", "4326"))
            gdf = gdf.set_crs(epsg=epsg, allow_override=True)
        except Exception:
            gdf = gdf.set_crs(4326, allow_override=True)
    try:
        return gdf.to_crs(4326)
    except Exception:
        return gdf


def bounds_to_leaflet(bounds: Tuple[float, float, float, float]) -> List[List[float]]:
    minx, miny, maxx, maxy = [float(x) for x in bounds]
    dx, dy = maxx - minx, maxy - miny
    if dx <= 0 or dy <= 0:
        pad = 0.1
        minx -= pad
        maxx += pad
        miny -= pad
        maxy += pad
    else:
        minx -= dx * 0.1
        maxx += dx * 0.1
        miny -= dy * 0.1
        maxy += dy * 0.1
    minx = max(-180.0, minx)
    maxx = min(180.0, maxx)
    miny = max(-85.0, miny)
    maxy = min(85.0, maxy)
    return [[miny, minx], [maxy, maxx]]


def gdf_to_geojson_min(gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    features: List[Dict[str, Any]] = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        props = {
            "name_asset_object": row.get("name_asset_object"),
            "id_asset_object": row.get("id"),
            "sensitivity_code_max": row.get("sensitivity_code_max") or row.get("sensitivity_code"),
            "area_km2": (float(row.get("area_m2", 0)) / 1_000_000.0) if row.get("area_m2") is not None else None,
        }
        features.append({"type": "Feature", "geometry": mapping(geom), "properties": props})
    return {"type": "FeatureCollection", "features": features}


def _load_asset_layers(
    cfg: configparser.ConfigParser,
) -> Tuple[List[Dict[str, Any]], List[List[float]] | None, Dict[str, Dict[str, Any]]]:
    log_event("Loading asset layers from GeoParquet")
    if not ASSET_OBJECT_FILE.exists():
        sys.stderr.write("tbl_asset_object.parquet not found – nothing to display.\n")
        log_event("tbl_asset_object.parquet missing; aborting load")
        return [], None, {}

    try:
        asset_objects = gpd.read_parquet(ASSET_OBJECT_FILE)
    except Exception as exc:
        sys.stderr.write(f"Failed to read asset objects: {exc}\n")
        log_event(f"Failed to read asset objects: {exc}")
        return [], None, {}

    if asset_objects.empty or "geometry" not in asset_objects.columns:
        log_event("Asset object table empty or lacks geometry column")
        return [], None, {}

    asset_objects = to_epsg4326(asset_objects, cfg)
    asset_objects = asset_objects[asset_objects.geometry.notna()].copy()
    if asset_objects.empty:
        log_event("Asset object table empty after dropping missing geometries")
        return [], None, {}

    group_names: Dict[str, str] = {}
    if ASSET_GROUP_FILE.exists():
        try:
            groups_df = pd.read_parquet(ASSET_GROUP_FILE)
            if not groups_df.empty and "id" in groups_df.columns:
                for _, row in groups_df.iterrows():
                    key = str(row.get("id"))
                    title = (row.get("title_fromuser") or row.get("name_user") or row.get("name_gis") or key)
                    group_names[key] = str(title)
        except Exception as exc:
            sys.stderr.write(f"Failed to read asset groups (continuing without names): {exc}\n")
            log_event(f"Failed to read asset group names: {exc}")

    if "ref_asset_group" not in asset_objects.columns:
        sys.stderr.write("Column 'ref_asset_group' missing in asset objects; cannot build per-group layers.\n")
        log_event("ref_asset_group column missing; cannot build groups")
        return [], bounds_to_leaflet(tuple(asset_objects.total_bounds)), {}

    records: List[Dict[str, Any]] = []
    geojson_by_group: Dict[str, Dict[str, Any]] = {}
    for group_id, subset in asset_objects.groupby("ref_asset_group"):
        subset = subset.copy()
        if subset.empty:
            continue
        gid_key = str(group_id)
        display_name = group_names.get(gid_key) or f"Group {gid_key}"
        geojson = gdf_to_geojson_min(subset)
        records.append(
            {
                "id": gid_key,
                "name": display_name,
                "count": int(len(subset)),
                "bounds": bounds_to_leaflet(tuple(subset.total_bounds)),
            }
        )
        geojson_by_group[gid_key] = geojson

    records.sort(key=lambda r: r["name"].lower())
    home_bounds = bounds_to_leaflet(tuple(asset_objects.total_bounds))
    max_layers = 10
    if len(records) > max_layers:
        log_event(f"Limiting asset layers to first {max_layers} of {len(records)} groups for testing")
        keep_ids = {rec["id"] for rec in records[:max_layers]}
        records = records[:max_layers]
        geojson_by_group = {gid: geojson_by_group[gid] for gid in keep_ids if gid in geojson_by_group}
        filtered = asset_objects[asset_objects["ref_asset_group"].astype(str).isin(keep_ids)]
        if not filtered.empty:
            home_bounds = bounds_to_leaflet(tuple(filtered.total_bounds))

    log_event(f"Prepared {len(records)} asset layers (ids={[rec['id'] for rec in records]})")
    return records, home_bounds, geojson_by_group


def _read_hierarchy() -> List[Dict[str, Any]]:
    if not ASSET_HIERARCHY_FILE.exists():
        return []
    try:
        df = pd.read_parquet(ASSET_HIERARCHY_FILE)
    except Exception as exc:
        log_event(f"Failed to read asset hierarchy: {exc}")
        return []
    nodes: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        node_id = str(row.get("node_id") or "").strip()
        if not node_id:
            continue
        parent_id_val = row.get("parent_id")
        parent_id = str(parent_id_val).strip() if parent_id_val not in (None, "", "None") else None
        ref_group_val = row.get("ref_asset_group")
        ref_group = str(ref_group_val).strip() if ref_group_val not in (None, "", "None") else None
        node_type = (row.get("node_type") or "group").strip().lower()
        nodes.append(
            {
                "node_id": node_id,
                "parent_id": parent_id,
                "node_type": node_type if node_type in {"folder", "group"} else "folder",
                "ref_asset_group": ref_group,
                "title": row.get("title") or "",
                "sort_order": int(row.get("sort_order") or 0),
            }
        )
    return nodes


def _normalize_hierarchy(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lookup = {node["node_id"]: node for node in nodes}
    for node in nodes:
      parent_id = node.get("parent_id") or None
      if parent_id:
        parent_node = lookup.get(parent_id)
        if parent_node is None or parent_node.get("node_type") != "folder":
          node["parent_id"] = None
    children: Dict[str | None, List[Dict[str, Any]]] = {}
    for node in nodes:
        parent = node.get("parent_id") or None
        children.setdefault(parent, []).append(node)
    for parent, bucket in children.items():
        bucket.sort(key=lambda item: item.get("sort_order") or 0)
        for idx, item in enumerate(bucket):
            item["sort_order"] = idx
    return nodes


def _ensure_group_hierarchy(records: List[Dict[str, Any]], nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid_group_ids = {str(rec["id"]) for rec in records}
    nodes = [
        node
        for node in nodes
        if node["node_type"] != "group" or (node.get("ref_asset_group") in valid_group_ids)
    ]
    node_ids = {node["node_id"] for node in nodes}
    existing_group_refs = {
        node.get("ref_asset_group"): node for node in nodes if node["node_type"] == "group" and node.get("ref_asset_group")
    }
    root_count = len([node for node in nodes if not node.get("parent_id")])
    for idx, rec in enumerate(records):
        gid = str(rec["id"])
        if gid in existing_group_refs:
            existing_group_refs[gid]["title"] = rec["name"]
            continue
        node_id = f"group:{gid}"
        suffix = 1
        while node_id in node_ids:
            suffix += 1
            node_id = f"group:{gid}:{suffix}"
        node = {
            "node_id": node_id,
            "parent_id": None,
            "node_type": "group",
            "ref_asset_group": gid,
            "title": rec["name"],
            "sort_order": root_count + idx,
        }
        nodes.append(node)
        node_ids.add(node_id)
    return _normalize_hierarchy(nodes)


def _write_hierarchy(nodes: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(nodes)
    df = df[["node_id", "parent_id", "node_type", "ref_asset_group", "title", "sort_order"]]
    ASSET_HIERARCHY_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ASSET_HIERARCHY_FILE, index=False)


def _sanitize_hierarchy_payload(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError("Hierarchy payload must be a list")
    valid_group_ids = {str(rec["id"]) for rec in ASSET_LAYERS}
    nodes: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("node_id") or "").strip()
        if not node_id or node_id in seen:
            continue
        node_type = (item.get("node_type") or "folder").strip().lower()
        if node_type not in {"folder", "group"}:
            node_type = "folder"
        parent_id_val = item.get("parent_id")
        parent_id = str(parent_id_val).strip() if parent_id_val not in (None, "", "None") else None
        ref_group_val = item.get("ref_asset_group")
        ref_group = (
            str(ref_group_val).strip() if ref_group_val not in (None, "", "None") else None
        )
        if node_type == "group":
            if ref_group not in valid_group_ids:
                continue
        else:
            ref_group = None
        nodes.append(
            {
                "node_id": node_id,
                "parent_id": parent_id,
                "node_type": node_type,
                "ref_asset_group": ref_group,
                "title": (item.get("title") or "").strip(),
                "sort_order": int(item.get("sort_order") or 0),
            }
        )
        seen.add(node_id)
    nodes = _normalize_hierarchy(nodes)
    nodes = _ensure_group_hierarchy(ASSET_LAYERS, nodes)
    return nodes


cfg = read_config(CONFIG_FILE)
COLOR_MAP = get_color_mapping(cfg)
ASSET_LAYERS, HOME_BOUNDS, ASSET_GEOJSON = _load_asset_layers(cfg)
ASSET_HIERARCHY = _ensure_group_hierarchy(ASSET_LAYERS, _read_hierarchy())


class Api:
    def get_state(self) -> Dict[str, Any]:
        log_event(f"get_state called; {len(ASSET_LAYERS)} layers ready")
        return {
            "asset_layers": ASSET_LAYERS,
            "home_bounds": HOME_BOUNDS,
            "colors": COLOR_MAP,
          "hierarchy": ASSET_HIERARCHY,
        }

    def get_asset_layer(self, group_id: str | int | None = None) -> Dict[str, Any]:
        if group_id is None:
            log_event("get_asset_layer called without group id")
            return {"ok": False, "error": "Missing asset group id"}
        data = ASSET_GEOJSON.get(str(group_id))
        if data is None:
            log_event(f"get_asset_layer miss for id={group_id}")
            return {"ok": False, "error": f"No asset layer found for group {group_id}"}
        log_event(f"get_asset_layer hit for id={group_id}")
        return {"ok": True, "geojson": data}

    def exit_app(self) -> None:
        try:
            webview.destroy_window()
        except Exception:
            os._exit(0)

    def save_png(self, data_url: str) -> Dict[str, Any]:
        try:
            if "," in data_url:
                _, payload = data_url.split(",", 1)
            else:
                payload = data_url
            data = base64.b64decode(payload)
            win = webview.windows[0]
            target = win.create_file_dialog(
                webview.FileDialog.SAVE,
                save_filename="asset_map.png",
                file_types=("PNG Files (*.png)",),
            )
            if not target:
                return {"ok": False, "error": "User cancelled"}
            if isinstance(target, (list, tuple)):
                target = target[0]
            with open(target, "wb") as handle:
                handle.write(data)
            return {"ok": True, "path": target}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def save_hierarchy(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        global ASSET_HIERARCHY
        try:
            normalized = _sanitize_hierarchy_payload(nodes)
            _write_hierarchy(normalized)
            ASSET_HIERARCHY = normalized
            log_event(f"Saved asset hierarchy ({len(normalized)} nodes)")
            return {"ok": True}
        except Exception as exc:
            log_event(f"Failed to save asset hierarchy: {exc}")
            return {"ok": False, "error": str(exc)}


def _ensure_stdio_utf8() -> None:
    try:
        encoding = os.environ.get("PYTHONIOENCODING") or getattr(sys.stdout, "encoding", None) or "utf-8"
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding=encoding, errors="replace")
            sys.stderr.reconfigure(encoding=encoding, errors="replace")
        else:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=encoding, errors="replace", line_buffering=True)
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding=encoding, errors="replace", line_buffering=True)
    except Exception:
        pass


_ensure_stdio_utf8()
api = Api()


HTML_TEMPLATE = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Asset Layers</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"></script>
<style>
  html, body { height:100%; margin:0; }
  .wrap {
    height:100%;
    display:grid;
    grid-template-columns: 260px 1fr;
    grid-template-rows: 48px 1fr;
    grid-template-areas:
      "bar bar"
      "layers map";
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    background:#0f172a;
    color:#e2e8f0;
  }
  .bar { grid-area: bar; display:flex; gap:12px; align-items:center; padding:8px 12px; flex-wrap:wrap; border-bottom: 2px solid #1f2b46; background:#111f38; }
  .layers { grid-area: layers; border-right:2px solid #1f2b46; background:#0b1222; display:flex; flex-direction:column; }
  .layer-header { padding:12px 16px; font-size:14px; font-weight:600; border-bottom:1px solid #1f2b4666; }
  .layer-list { flex:1; overflow:auto; padding:6px 8px 10px; }
  .layer-item { display:flex; gap:8px; align-items:center; padding:6px 6px; cursor:pointer; margin-bottom:2px; border-radius:5px; }
  .layer-item:hover { background:#132445; }
  .layer-item input { margin:0; }
  .layer-name { flex:1; font-size:13px; }
  .layer-tree { display:flex; flex-direction:column; gap:4px; }
  .tree-node { font-size:13px; }
  .tree-row { display:flex; align-items:center; gap:6px; padding:4px 6px; border-radius:6px; position:relative; }
  .tree-node.folder .tree-row { font-weight:600; color:#f8fafc; }
  .tree-node.group .tree-row { color:#e2e8f0; }
  .tree-row:hover { background:#132445; }
  .tree-label { flex:1; }
  .tree-children { margin-left:14px; border-left:1px solid #1f2b46; padding-left:10px; }
  .drag-handle { cursor:grab; color:#9aa8d9; user-select:none; font-size:14px; }
  .drag-handle:active { cursor:grabbing; }
  .folder-toggle { width:18px; height:18px; border:none; background:transparent; color:#cbd5f5; cursor:pointer; padding:0; }
  .folder-toggle:focus { outline:none; }
  .tree-row.drop-before::before,
  .tree-row.drop-after::after { content:""; position:absolute; left:6px; right:6px; border-top:2px solid #7aa2ff; }
  .tree-row.drop-before::before { top:0; }
  .tree-row.drop-after::after { bottom:0; }
  .tree-row.drop-inside { background:#1c2e57; }
  .btn-xs { padding:4px 8px; font-size:12px; }
  .layer-footer { border-top:1px solid #1f2b46; padding:12px 14px; font-size:13px; }
  .base-option { display:flex; gap:8px; align-items:center; padding:6px 0; }
  .base-option input { margin:0; }
  .map { grid-area: map; position:relative; background:#0b1222; }
  #map { position:absolute; inset:0; }
  #map.exporting .leaflet-control-zoom { display:none !important; }
  .btn { padding:6px 10px; border:1px solid #354769; background:#1f2b46; border-radius:6px; cursor:pointer; color:#f8fafc; font-size:13px; }
  .btn:active { transform:translateY(1px); }
  .slider { display:flex; align-items:center; gap:8px; }
  .slider input[type=range]{ width:160px; }
  .leaflet-control-layers { font-size:12px; }
  .leaflet-control-layers label { line-height:1.3; }
  .leaflet-popup-content-wrapper { border-radius:10px; }
  .leaflet-popup-content { font-size:13px; color:#111827; }
  .popup strong { display:block; font-size:14px; margin-bottom:4px; }
</style>
</head>
<body>
<div class="wrap">
  <div class="bar">
    <button id="homeBtn" class="btn">Home</button>
    <div class="slider">
      <span>Opacity</span>
      <input id="opacity" type="range" min="20" max="100" value="85">
      <span id="opacityValue">85%</span>
    </div>
    <button id="exportBtn" class="btn" title="Export current map view to PNG">Export PNG</button>
    <button id="exitBtn" class="btn">Exit</button>
  </div>
  <div class="layers">
    <div class="layer-header" style="display:flex; align-items:center; justify-content:space-between; gap:8px;">
      <span>Asset Layers</span>
      <button id="addFolderBtn" class="btn btn-xs" title="Create a folder to organise layers">New Folder</button>
    </div>
    <div class="info-block" style="padding:10px 16px; font-size:12px; color:#94a3b8; border-bottom:1px solid #1f2b4666;">
      Drag layers to reorder, drop them onto folders to organise, and use checkboxes to toggle visibility. Folders are saved between sessions.
    </div>
    <div id="layerControls" class="layer-list layer-tree"></div>
    <div class="layer-footer">
      <div style="font-weight:600; margin-bottom:6px;">Base map</div>
      <div id="baseControls"></div>
    </div>
  </div>
  <div class="map"><div id="map"></div></div>
</div>

<script>
let MAP=null;
let BASE_SOURCES=null;
let CURRENT_BASE_KEY=null;
let CURRENT_BASE_LAYER=null;
let COLOR_MAP={};
let FILL_ALPHA=0.85;
let HOME_BOUNDS=null;
let ASSET_LAYERS=[];
let GROUP_LAYERS=[];
let LAYER_BY_GROUP=new Map();
let ACTIVE_GROUPS=new Set();
let HIERARCHY_FLAT=[];
let TREE_ROOTS=[];
let NODE_LOOKUP=new Map();
let DRAG_STATE={ nodeId:null, dropRow:null, dropPosition:null, dropTarget:null };
let PYWEBVIEW_API=null;
let API_PROMISE=null;
const NEUTRAL_FILL='#a3a7b1';
const NEUTRAL_STROKE='#242b38';

function waitForApi(timeoutMs=12000){
  if (PYWEBVIEW_API){
    return Promise.resolve(PYWEBVIEW_API);
  }
  if (window.pywebview && window.pywebview.api){
    PYWEBVIEW_API = window.pywebview.api;
    return Promise.resolve(PYWEBVIEW_API);
  }
  if (API_PROMISE){
    return API_PROMISE;
  }
  API_PROMISE = new Promise((resolve, reject) => {
    let settled = false;
    const onReady = () => {
      if (settled) return;
      settled = true;
      if (window.pywebview && window.pywebview.api){
        PYWEBVIEW_API = window.pywebview.api;
        resolve(PYWEBVIEW_API);
      } else {
        reject(new Error('pywebview API unavailable after ready event'));
      }
    };
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      reject(new Error('pywebview API not ready'));
    }, timeoutMs);
    window.addEventListener('pywebviewready', () => {
      clearTimeout(timer);
      onReady();
    }, { once:true });
  }).finally(() => {
    API_PROMISE = null;
  });
  return API_PROMISE;
}

function uuidv4(){
  if (window.crypto && window.crypto.randomUUID){
    return window.crypto.randomUUID();
  }
  return 'node-' + Math.random().toString(16).slice(2);
}

function escapeHtml(str){
  if (str === undefined || str === null) return '';
  return String(str).replace(/[&<>\"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

function styleFeature(){
  return {
    color: NEUTRAL_STROKE,
    weight: 0.6,
    fillColor: NEUTRAL_FILL,
    fillOpacity: FILL_ALPHA,
    opacity: 0.8,
  };
}

function bindFeature(feature, layer, groupMeta){
  const props = feature && feature.properties ? feature.properties : {};
  const title = (groupMeta && groupMeta.name) || props.name_asset_object || props.id_asset_object || 'Asset group';
  let html = '<div class="popup"><strong>'+escapeHtml(title)+'</strong>';
  if (props.sensitivity_code_max || props.sensitivity_code){
    html += '<div>Code: '+escapeHtml(props.sensitivity_code_max || props.sensitivity_code)+'</div>';
  }
  if (props.area_km2){
    html += '<div>Area: '+Number(props.area_km2).toLocaleString('en-US',{maximumFractionDigits:2})+' km²</div>';
  }
  html += '</div>';
  layer.bindPopup(html);
}

function buildBaseSources(){
  const common = { maxZoom:19, crossOrigin:true, tileSize:256 };
  const osm = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    ...common,
    attribution:'© OpenStreetMap contributors'
  });
  const topo = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
    ...common,
    subdomains:['a','b','c'],
    maxZoom:17,
    attribution:'© OpenStreetMap, © OpenTopoMap (CC-BY-SA)'
  });
  const esri = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    ...common,
    attribution:'© Esri, Maxar, Earthstar Geographics'
  });
  return { 'OpenStreetMap': osm, 'OSM Topography': topo, 'Satellite (ESRI)': esri };
}

function createAssetLayer(entry){
  const geoLayer = L.geoJSON(null, {
    pane: 'assetsPane',
    renderer: L.canvas({pane:'assetsPane'}),
    style: styleFeature,
    onEachFeature: (feature, layer) => bindFeature(feature, layer, entry),
  });
  geoLayer.__meta = entry;
  geoLayer.__loaded = false;
  geoLayer.__pending = null;
  return geoLayer;
}

function ensureLayerData(layer){
  if (!layer) return Promise.resolve(null);
  if (layer.__loaded) return Promise.resolve(layer);
  if (layer.__pending) return layer.__pending;
  const meta = layer.__meta || {};
  layer.__pending = waitForApi().then(api => {
    if (!api || !api.get_asset_layer){
      throw new Error('Asset layer API not available.');
    }
    return api.get_asset_layer(meta.id);
  }).then(res => {
    if (!res || !res.ok || !res.geojson){
      throw new Error((res && res.error) || 'Layer data missing');
    }
    layer.clearLayers();
    layer.addData(res.geojson);
    layer.__loaded = true;
    return layer;
  }).catch(err => {
    console.error('Failed to load asset layer', err);
    throw err;
  }).finally(() => {
    layer.__pending = null;
  });
  return layer.__pending;
}

function setLayerActive(layer, isActive, options){
  if (!layer) return;
  const opts = options || {};
  if (isActive){
    ensureLayerData(layer).then(() => {
      if (!MAP.hasLayer(layer)){
        layer.addTo(MAP);
      }
      if (opts.fitBounds && layer.__meta && layer.__meta.bounds){
        MAP.fitBounds(layer.__meta.bounds, { padding:[24,24] });
      }
    }).catch(err => {
      console.error('Layer activation failed', err);
    });
  } else if (MAP.hasLayer(layer)) {
    MAP.removeLayer(layer);
  }
}

function prepareLayerCollection(assetLayers){
  GROUP_LAYERS = [];
  LAYER_BY_GROUP = new Map();
  ACTIVE_GROUPS = new Set();
  (assetLayers || []).forEach(entry => {
    const layer = createAssetLayer(entry);
    GROUP_LAYERS.push(layer);
    LAYER_BY_GROUP.set(String(entry.id), layer);
  });
}

function normalizeHierarchyNodes(nodes){
  const validGroupIds = new Set((ASSET_LAYERS || []).map(layer => String(layer.id)));
  const sanitized = Array.isArray(nodes) ? nodes.map(item => ({
    node_id: String(item.node_id || uuidv4()),
    parent_id: item.parent_id ? String(item.parent_id) : null,
    node_type: item.node_type === 'folder' ? 'folder' : 'group',
    ref_asset_group: item.ref_asset_group ? String(item.ref_asset_group) : null,
    title: item.title || '',
    sort_order: Number.isFinite(item.sort_order) ? item.sort_order : 0,
    collapsed: Boolean(item.collapsed),
    children: [],
  })) : [];
  const filtered = sanitized.filter(node => node.node_type !== 'group' || validGroupIds.has(node.ref_asset_group));
  const existingGroupNodes = new Set(filtered.filter(node => node.node_type === 'group').map(node => node.ref_asset_group));
  (ASSET_LAYERS || []).forEach(entry => {
    const gid = String(entry.id);
    if (!existingGroupNodes.has(gid)){
      filtered.push({
        node_id: `group:${gid}`,
        parent_id: null,
        node_type: 'group',
        ref_asset_group: gid,
        title: entry.name,
        sort_order: filtered.length,
        collapsed: false,
        children: [],
      });
    }
  });
  return filtered;
}

function rebuildTree(){
  NODE_LOOKUP = new Map();
  TREE_ROOTS = [];
  HIERARCHY_FLAT.forEach(node => {
    node.children = [];
    NODE_LOOKUP.set(node.node_id, node);
  });
  HIERARCHY_FLAT.forEach(node => {
    const parentId = node.parent_id || null;
    if (parentId && NODE_LOOKUP.has(parentId)){
      NODE_LOOKUP.get(parentId).children.push(node);
    } else {
      node.parent_id = null;
      TREE_ROOTS.push(node);
    }
  });
  const sortChildren = list => {
    list.sort((a,b) => (a.sort_order || 0) - (b.sort_order || 0));
    list.forEach(child => sortChildren(child.children || []));
  };
  sortChildren(TREE_ROOTS);
}

function initHierarchy(nodes){
  HIERARCHY_FLAT = normalizeHierarchyNodes(nodes);
  rebuildTree();
}

function renderLayerTree(){
  const container = document.getElementById('layerControls');
  if (!container) return;
  container.innerHTML = '';
  if (!TREE_ROOTS.length){
    container.innerHTML = '<p style="font-size:13px; color:#94a3b8;">No asset layers available. Run the data pipeline to populate tbl_asset_object.parquet.</p>';
    return;
  }
  TREE_ROOTS.forEach(node => container.appendChild(renderTreeNode(node, 0)));
  ensureInitialActivation();
}

function renderTreeNode(node, depth){
  const wrapper = document.createElement('div');
  wrapper.className = 'tree-node ' + node.node_type;
  const row = document.createElement('div');
  row.className = 'tree-row';
  row.dataset.nodeId = node.node_id;
  row.style.paddingLeft = (8 + depth * 14) + 'px';
  row.addEventListener('dragover', evt => handleRowDragOver(evt, node, row));
  row.addEventListener('dragleave', () => clearRowDropState(row));
  row.addEventListener('drop', handleRowDrop);

  const handle = document.createElement('span');
  handle.className = 'drag-handle';
  handle.textContent = '≡';
  handle.title = 'Drag to reorder';
  handle.draggable = true;
  handle.addEventListener('dragstart', evt => handleDragStart(evt, node, row));
  handle.addEventListener('dragend', () => clearDragState());
  row.appendChild(handle);

  if (node.node_type === 'folder'){
    const toggle = document.createElement('button');
    toggle.className = 'folder-toggle';
    toggle.textContent = node.collapsed ? '▸' : '▾';
    toggle.addEventListener('click', evt => {
      evt.stopPropagation();
      node.collapsed = !node.collapsed;
      renderLayerTree();
    });
    row.appendChild(toggle);

    const label = document.createElement('span');
    label.className = 'tree-label';
    label.textContent = node.title || 'Folder';
    row.appendChild(label);
  } else {
    const spacer = document.createElement('span');
    spacer.style.width = '18px';
    row.appendChild(spacer);

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = ACTIVE_GROUPS.has(node.ref_asset_group);
    checkbox.addEventListener('click', evt => evt.stopPropagation());
    checkbox.addEventListener('change', () => {
      const shouldFit = checkbox.checked && !MAP.hasLayer(LAYER_BY_GROUP.get(node.ref_asset_group));
      toggleGroupLayer(node.ref_asset_group, checkbox.checked, { fitBounds: shouldFit });
    });
    row.appendChild(checkbox);

    const label = document.createElement('span');
    label.className = 'tree-label';
    const layer = LAYER_BY_GROUP.get(node.ref_asset_group);
    const meta = layer ? layer.__meta : null;
    const countText = meta ? ' (' + Number(meta.count || 0).toLocaleString('en-US') + ')' : '';
    label.textContent = (meta ? meta.name : (node.title || 'Layer')) + countText;
    row.appendChild(label);
  }

  wrapper.appendChild(row);

  if (node.node_type === 'folder' && node.children && node.children.length && !node.collapsed){
    const childWrap = document.createElement('div');
    childWrap.className = 'tree-children';
    node.children.forEach(child => childWrap.appendChild(renderTreeNode(child, depth + 1)));
    wrapper.appendChild(childWrap);
  }
  return wrapper;
}

function handleDragStart(evt, node, row){
  DRAG_STATE = { nodeId: node.node_id, dropRow:null, dropPosition:null, dropTarget:null, dragRow:row };
  evt.dataTransfer.effectAllowed = 'move';
  if (evt.dataTransfer && evt.dataTransfer.setData){
    evt.dataTransfer.setData('text/plain', node.node_id);
  }
  row.classList.add('dragging');
}

function handleRowDragOver(evt, node, row){
  if (!DRAG_STATE.nodeId || DRAG_STATE.nodeId === node.node_id){
    return;
  }
  evt.preventDefault();
  if (DRAG_STATE.dropRow && DRAG_STATE.dropRow !== row){
    clearRowDropState(DRAG_STATE.dropRow);
  }
  const rect = row.getBoundingClientRect();
  const offset = evt.clientY - rect.top;
  const height = rect.height || 1;
  let position = 'inside';
  if (offset < height * 0.25){
    position = 'before';
  } else if (offset > height * 0.75){
    position = 'after';
  } else if (node.node_type !== 'folder'){
    position = offset < height / 2 ? 'before' : 'after';
  }
  if (position === 'inside' && node.node_type !== 'folder'){
    position = 'after';
  }
  clearRowDropState(row);
  if (position === 'inside'){
    row.classList.add('drop-inside');
  } else if (position === 'before'){
    row.classList.add('drop-before');
  } else {
    row.classList.add('drop-after');
  }
  DRAG_STATE.dropRow = row;
  DRAG_STATE.dropPosition = position;
  DRAG_STATE.dropTarget = node;
}

function handleRowDrop(evt){
  if (!DRAG_STATE.nodeId) return;
  evt.preventDefault();
  if (!DRAG_STATE.dropTarget){
    clearDragState();
    return;
  }
  moveNode(DRAG_STATE.nodeId, DRAG_STATE.dropTarget.node_id, DRAG_STATE.dropPosition);
  clearDragState();
}

function handleRootDragOver(evt){
  if (!DRAG_STATE.nodeId) return;
  if (evt.target.closest && evt.target.closest('.tree-row')) return;
  evt.preventDefault();
  if (DRAG_STATE.dropRow){
    clearRowDropState(DRAG_STATE.dropRow);
    DRAG_STATE.dropRow = null;
  }
  DRAG_STATE.dropTarget = null;
  DRAG_STATE.dropPosition = 'root';
}

function handleRootDrop(evt){
  if (!DRAG_STATE.nodeId) return;
  if (evt.target.closest && evt.target.closest('.tree-row')) return;
  evt.preventDefault();
  moveNode(DRAG_STATE.nodeId, null, 'root');
  clearDragState();
}

function clearRowDropState(row){
  if (!row) return;
  row.classList.remove('drop-before', 'drop-after', 'drop-inside');
}

function clearDragState(){
  if (DRAG_STATE.dropRow){
    clearRowDropState(DRAG_STATE.dropRow);
  }
  if (DRAG_STATE.dragRow){
    DRAG_STATE.dragRow.classList.remove('dragging');
  }
  DRAG_STATE = { nodeId:null, dropRow:null, dropPosition:null, dropTarget:null, dragRow:null };
}

function removeNodeFromParent(node){
  const parentId = node.parent_id;
  const siblings = parentId ? (NODE_LOOKUP.get(parentId)?.children || []) : TREE_ROOTS;
  const idx = siblings.indexOf(node);
  if (idx >= 0){
    siblings.splice(idx, 1);
  }
}

function isDescendant(targetNode, ancestorId){
  if (!targetNode) return false;
  if (targetNode.node_id === ancestorId) return true;
  let parentId = targetNode.parent_id;
  while (parentId){
    if (parentId === ancestorId) return true;
    const parent = NODE_LOOKUP.get(parentId);
    parentId = parent ? parent.parent_id : null;
  }
  return false;
}

function moveNode(nodeId, targetId, position){
  const moving = NODE_LOOKUP.get(nodeId);
  if (!moving) return;
  if (targetId){
    const target = NODE_LOOKUP.get(targetId);
    if (!target || nodeId === targetId || isDescendant(target, nodeId)){
      return;
    }
  }
  removeNodeFromParent(moving);
  if (!targetId){
    moving.parent_id = null;
    TREE_ROOTS.push(moving);
  } else {
    const target = NODE_LOOKUP.get(targetId);
    let dropPosition = position;
    if (dropPosition === 'inside' && target.node_type !== 'folder'){
      dropPosition = 'after';
    }
    if (dropPosition === 'inside' && target.node_type === 'folder'){
      target.children = target.children || [];
      target.children.push(moving);
      moving.parent_id = target.node_id;
    } else {
      const parent = target.parent_id ? NODE_LOOKUP.get(target.parent_id) : null;
      const siblings = parent ? parent.children : TREE_ROOTS;
      let insertIndex = siblings.indexOf(target);
      if (insertIndex === -1){
        insertIndex = siblings.length;
      } else if (dropPosition === 'after'){
        insertIndex += 1;
      }
      siblings.splice(insertIndex, 0, moving);
      moving.parent_id = parent ? parent.node_id : null;
    }
  }
  updateSortOrders();
  persistHierarchy();
  renderLayerTree();
}

function updateSortOrders(){
  const apply = (list, parentId) => {
    list.forEach((node, idx) => {
      node.sort_order = idx;
      node.parent_id = parentId;
      if (node.children && node.children.length){
        apply(node.children, node.node_id);
      }
    });
  };
  apply(TREE_ROOTS, null);
}

function serializeHierarchy(){
  const acc = [];
  const walk = (nodes, parentId) => {
    nodes.forEach((node, idx) => {
      acc.push({
        node_id: node.node_id,
        parent_id: parentId,
        node_type: node.node_type,
        ref_asset_group: node.node_type === 'group' ? node.ref_asset_group : null,
        title: node.node_type === 'folder' ? (node.title || '') : (node.title || ''),
        sort_order: idx,
      });
      if (node.children && node.children.length){
        walk(node.children, node.node_id);
      }
    });
  };
  walk(TREE_ROOTS, null);
  return acc;
}

function persistHierarchy(){
  const flat = serializeHierarchy();
  HIERARCHY_FLAT = flat.map(item => ({ ...item }));
  waitForApi().then(api => {
    if (api && api.save_hierarchy){
      return api.save_hierarchy(flat);
    }
    throw new Error('Hierarchy API unavailable');
  }).then(res => {
    if (!res || !res.ok){
      console.error('Failed to save hierarchy', res && res.error);
    }
  }).catch(err => {
    console.error('Failed to save hierarchy', err);
  });
}

function toggleGroupLayer(groupId, enable, options){
  const layer = LAYER_BY_GROUP.get(String(groupId));
  if (!layer) return;
  setLayerActive(layer, enable, options);
  if (enable){
    ACTIVE_GROUPS.add(String(groupId));
  } else {
    ACTIVE_GROUPS.delete(String(groupId));
  }
}

function findFirstGroupNode(nodes){
  for (const node of nodes){
    if (node.node_type === 'group'){
      return node;
    }
    if (node.children && node.children.length){
      const nested = findFirstGroupNode(node.children);
      if (nested) return nested;
    }
  }
  return null;
}

function ensureInitialActivation(){
  if (ACTIVE_GROUPS.size) return;
  const first = findFirstGroupNode(TREE_ROOTS);
  if (first && first.ref_asset_group){
    toggleGroupLayer(first.ref_asset_group, true, { fitBounds:true });
    const checkbox = document.querySelector(`[data-node-id="${first.node_id}"] input[type=checkbox]`);
    if (checkbox){
      checkbox.checked = true;
    }
  }
}

function handleAddFolder(){
  const name = prompt('Folder name');
  if (!name) return;
  const trimmed = name.trim();
  if (!trimmed) return;
  const node = {
    node_id: `folder:${uuidv4()}`,
    parent_id: null,
    node_type: 'folder',
    ref_asset_group: null,
    title: trimmed,
    sort_order: TREE_ROOTS.length,
    collapsed: false,
    children: [],
  };
  TREE_ROOTS.push(node);
  NODE_LOOKUP.set(node.node_id, node);
  updateSortOrders();
  persistHierarchy();
  renderLayerTree();
}

function updateOpacity(){
  GROUP_LAYERS.forEach(layer => {
    if (layer && typeof layer.setStyle === 'function'){
      layer.setStyle(styleFeature);
    }
  });
}

function setBaseLayer(key){
  if (!BASE_SOURCES || !BASE_SOURCES[key] || !MAP){
    return;
  }
  const layer = BASE_SOURCES[key];
  if (CURRENT_BASE_LAYER === layer){
    return;
  }
  if (CURRENT_BASE_LAYER && MAP.hasLayer(CURRENT_BASE_LAYER)){
    MAP.removeLayer(CURRENT_BASE_LAYER);
  }
  CURRENT_BASE_LAYER = layer;
  CURRENT_BASE_KEY = key;
  layer.addTo(MAP);
  renderBaseControls();
}

function renderBaseControls(){
  const container = document.getElementById('baseControls');
  if (!container) return;
  if (!BASE_SOURCES){
    container.innerHTML = '<div style="color:#94a3b8;font-size:12px;">No base maps available.</div>';
    return;
  }
  const entries = Object.keys(BASE_SOURCES);
  if (!entries.length){
    container.innerHTML = '<div style="color:#94a3b8;font-size:12px;">No base maps available.</div>';
    return;
  }
  container.innerHTML = '';
  entries.forEach(key => {
    const label = document.createElement('label');
    label.className = 'base-option';
    const input = document.createElement('input');
    input.type = 'radio';
    input.name = 'baseMap';
    input.value = key;
    input.checked = key === CURRENT_BASE_KEY;
    const text = document.createElement('div');
    text.textContent = key;
    input.addEventListener('change', () => {
      if (input.checked){
        setBaseLayer(key);
      }
    });
    label.appendChild(input);
    label.appendChild(text);
    container.appendChild(label);
  });
}

function boot(){
  MAP = L.map('map', {
    zoomControl:false,
    preferCanvas:true,
    maxBounds: L.latLngBounds([[-85,-180],[85,180]]),
    maxBoundsViscosity: 1.0,
  });
  MAP.createPane('assetsPane');
  L.control.zoom({ position:'topright' }).addTo(MAP);
  L.control.scale({ position:'bottomleft', metric:true, imperial:false }).addTo(MAP);

  BASE_SOURCES = buildBaseSources();
  const baseKeys = Object.keys(BASE_SOURCES);
  if (baseKeys.length){
    CURRENT_BASE_KEY = baseKeys[0];
    CURRENT_BASE_LAYER = BASE_SOURCES[CURRENT_BASE_KEY];
    CURRENT_BASE_LAYER.addTo(MAP);
  }
  renderBaseControls();

  const treeContainer = document.getElementById('layerControls');
  if (treeContainer){
    treeContainer.addEventListener('dragover', handleRootDragOver);
    treeContainer.addEventListener('drop', handleRootDrop);
  }
  const addFolderBtn = document.getElementById('addFolderBtn');
  if (addFolderBtn){
    addFolderBtn.addEventListener('click', handleAddFolder);
  }

  waitForApi().then(api => {
    if (!api || !api.get_state){
      throw new Error('pywebview API missing get_state');
    }
    return api.get_state();
  }).then(state => {
    COLOR_MAP = state.colors || {};
    HOME_BOUNDS = state.home_bounds || null;
    ASSET_LAYERS = state.asset_layers || [];
    prepareLayerCollection(ASSET_LAYERS);
    initHierarchy(state.hierarchy || []);
    renderLayerTree();
    if (HOME_BOUNDS){
      MAP.fitBounds(HOME_BOUNDS, { padding:[24,24] });
    } else {
      MAP.setView([0,0], 2);
    }
  }).catch(err => {
    console.error('Failed to load state', err);
    const fallback = document.getElementById('layerControls');
    if (fallback){
      fallback.innerHTML = '<p style="color:#94a3b8;">Could not load asset layers.<br>' + escapeHtml(err && err.message ? err.message : String(err)) + '</p>';
    }
  });

  document.getElementById('homeBtn').addEventListener('click', () => {
    if (HOME_BOUNDS){
      MAP.fitBounds(HOME_BOUNDS, { padding:[24,24] });
    }
  });

  const slider = document.getElementById('opacity');
  const label = document.getElementById('opacityValue');
  slider.addEventListener('input', () => {
    FILL_ALPHA = parseInt(slider.value, 10) / 100.0;
    label.textContent = slider.value + '%';
    updateOpacity();
  });

  document.getElementById('exportBtn').addEventListener('click', () => {
    const mapNode = document.getElementById('map');
    mapNode.classList.add('exporting');
    html2canvas(mapNode, {useCORS:true, allowTaint:false, backgroundColor:'#ffffff', scale:3})
      .then(canvas => {
        mapNode.classList.remove('exporting');
        return waitForApi().then(api => {
          if (!api || !api.save_png){
            throw new Error('Save API not available');
          }
          return api.save_png(canvas.toDataURL('image/png'));
        });
      })
      .then(res => {
        if (!res.ok){
          console.error('Export failed', res.error);
        }
      })
      .catch(err => {
        mapNode.classList.remove('exporting');
        console.error('Export failed', err);
      });
  });

  document.getElementById('exitBtn').addEventListener('click', () => {
    waitForApi().then(api => {
      if (api && api.exit_app){
        api.exit_app();
      }
    }).catch(() => {});
  });
}
if (document.readyState === 'loading'){
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
</script>
</body>
</html>
"""


def main() -> None:
    window = webview.create_window(
        title="Asset Layers",
        html=HTML_TEMPLATE,
        js_api=api,
        width=1280,
        height=820,
        resizable=True,
    )
    webview.start(gui="edgechromium", debug=False)


if __name__ == "__main__":
    main()
