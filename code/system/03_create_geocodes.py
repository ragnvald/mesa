#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_create_geocodes.py  —  H3 + Basic Mosaic (GeoParquet-first)

What's new in this build (visibility + responsiveness):
- Heartbeat now shows: stage, tiles done/total, faces count, elapsed, ETA, tiles/min,
  and total CPU/RAM aggregated across pool children (if psutil available).
- Task scheduling defaults to INTERLEAVE (light ↔ heavy) so first results arrive quickly;
  configurable via config.ini (mosaic_task_order).
- Pool chunk size is configurable (mosaic_pool_chunksize, default 1) for snappy returns.
- Still buffers/simplifies inside workers, not in the parent → real parallel utilization.
- Kept: adaptive quadtree tiling + heavy-tile split, streaming face flush, robust H3, GUI/CLI.

Config (system/config.ini → [DEFAULT]) knobs (defaults shown):
  mosaic_workers = 0                         # 0 -> auto
  mosaic_quadtree_max_feats_per_tile = 800
  mosaic_quadtree_heavy_split_multiplier = 2.0
  mosaic_quadtree_max_depth = 8
  mosaic_quadtree_min_tile_m = 1000
  mosaic_simplify_tolerance_m = 0           # 0 disables simplify
  mosaic_faces_flush_batch = 250000
  mosaic_pool_maxtasksperchild = 8
  mosaic_pool_chunksize = 1                 # 1 = more responsive progress
  mosaic_task_order = interleave            # interleave | heavy_first | light_first
  heartbeat_secs = 10
"""

from __future__ import annotations

import argparse
import configparser
import datetime
import locale
import os
import time
import multiprocessing as mp
from pathlib import Path
from typing import Union, Optional, List, Tuple

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import (
    Polygon, MultiPolygon, GeometryCollection, LineString, MultiLineString, box
)
from shapely.geometry import mapping as shp_mapping
from shapely.ops import unary_union, polygonize
from shapely import wkb as shp_wkb

# Optional: make_valid (Shapely >=2)
try:
    from shapely.validation import make_valid as shapely_make_valid
except Exception:
    shapely_make_valid = None

# Optional: psutil for richer heartbeat
try:
    import psutil
except Exception:
    psutil = None

# Optional: H3
try:
    import h3  # v3 or v4
except Exception:
    h3 = None

# GUI / ttkbootstrap
import tkinter as tk
from tkinter import scrolledtext
try:
    import ttkbootstrap as ttk
    from ttkbootstrap.constants import PRIMARY, INFO, WARNING
except Exception:
    ttk = None
    PRIMARY = INFO = WARNING = None


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
BASIC_MOSAIC_GROUP = "basic_mosaic"

# H3 across-flats (km)
H3_RES_ACROSS_FLATS_KM = {
    0: 2215.425182, 1: 837.352011, 2: 316.489311, 3: 119.621716,
    4: 45.2127588, 5: 17.08881655, 6: 6.462555554, 7: 2.441259518,
    8: 0.922709368, 9: 0.348751336, 10: 0.131815614, 11: 0.049821122,
    12: 0.018831052, 13: 0.007119786, 14: 0.00269715, 15: 0.001019426
}
H3_RES_ACROSS_FLATS_M = {r: km * 1000.0 for r, km in H3_RES_ACROSS_FLATS_KM.items()}

AVG_HEX_AREA_KM2 = {
    0: 4259705, 1: 608529, 2: 86932, 3: 12419, 4: 1774,
    5: 253, 6: 36.1, 7: 5.15, 8: 0.736, 9: 0.105,
    10: 0.0150, 11: 0.00214, 12: 0.000305, 13: 0.0000436, 14: 0.00000623, 15: 0.00000089
}

# -----------------------------------------------------------------------------
# Locale
# -----------------------------------------------------------------------------
try:
    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
except locale.Error:
    pass

# -----------------------------------------------------------------------------
# GUI globals + heartbeat
# -----------------------------------------------------------------------------
root: Optional[tk.Tk] = None
log_widget: Optional[scrolledtext.ScrolledText] = None
progress_var: Optional[tk.DoubleVar] = None
progress_label: Optional[tk.Label] = None
original_working_directory: Optional[str] = None
mosaic_status_var: Optional[tk.StringVar] = None
size_levels_var: Optional[tk.StringVar] = None

HEARTBEAT_SECS = 10

class Stats:
    def __init__(self):
        self.stage = "idle"
        self.detail = ""
        self.tiles_total = 0
        self.tiles_done = 0
        self.faces_total = 0
        self.started_at = time.time()
        self.worker_started_at = None
        self.running = False
STATS = Stats()

# -----------------------------------------------------------------------------
# Config & paths
# -----------------------------------------------------------------------------
def read_config(file_name: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(file_name, encoding="utf-8")
    if "DEFAULT" not in cfg:
        cfg["DEFAULT"] = {}
    return cfg

def resolve_base_dir(cli_root: str | None) -> Path:
    if cli_root:
        return Path(cli_root)
    base = Path(os.getcwd())
    if base.name.lower() == "system":
        return base.parent
    return base

def gpq_dir(base_dir: Path) -> Path:
    p = base_dir / "output" / "geoparquet"
    p.mkdir(parents=True, exist_ok=True)
    return p

def geoparquet_path(base_dir: Path, name: str) -> Path:
    return gpq_dir(base_dir) / f"{name}.parquet"

# -----------------------------------------------------------------------------
# Logging / progress
# -----------------------------------------------------------------------------
def update_progress(new_value: float):
    if root is None or progress_var is None or progress_label is None:
        return
    def task():
        v = max(0, min(100, float(new_value)))
        progress_var.set(v)
        progress_label.config(text=f"{int(v)}%")
    try:
        root.after(0, task)
    except Exception:
        pass

def log_to_gui(message: str, level: str = "INFO"):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted = f"{timestamp} [{level}] - {message}"
    if log_widget is not None:
        try:
            log_widget.insert(tk.END, formatted + "\n")
            log_widget.see(tk.END)
        except Exception:
            pass
    if original_working_directory:
        try:
            with open(Path(original_working_directory) / "log.txt", "a", encoding="utf-8") as f:
                f.write(formatted + "\n")
        except Exception:
            pass
    if log_widget is None:
        print(formatted)


def _run_in_thread(fn, *args, **kwargs):
    """Run function in a daemon thread and surface exceptions in the GUI log."""
    import threading, traceback
    def _wrap():
        try:
            fn(*args, **kwargs)
        except Exception as e:
            log_to_gui(f"[H3] Error: {e}", "WARN")
            try:
                tb = "".join(traceback.format_exc())
                log_to_gui(tb, "WARN")
            except Exception:
                pass
    threading.Thread(target=_wrap, daemon=True).start()

def _fmt_eta(elapsed_s: float, done: int, total: int) -> str:
    if done <= 0 or total <= 0:
        return "ETA —"
    rate = done / max(elapsed_s, 1e-6)
    if rate <= 0:
        return "ETA —"
    remaining = (total - done) / rate
    # show hh:mm:ss (+days if needed)
    eta_ts = datetime.datetime.now() + datetime.timedelta(seconds=max(0, remaining))
    eta_str = eta_ts.strftime("%H:%M:%S")
    dd = (eta_ts.date() - datetime.datetime.now().date()).days
    if dd > 0:
        eta_str += f" (+{dd}d)"
    return eta_str

def start_heartbeat():
    if STATS.running:
        return
    STATS.running = True

    # Pre-prime psutil counters so first heartbeat shows meaningful CPU
    proc = psutil.Process(os.getpid()) if psutil else None
    if proc:
        try:
            _ = proc.cpu_percent(interval=None)
            for c in proc.children(recursive=True):
                try: _ = c.cpu_percent(interval=None)
                except Exception: pass
        except Exception:
            pass

    def _hb():
        last_tiles = -1
        while STATS.running:
            elapsed = time.time() - STATS.started_at
            worker_elapsed = (time.time() - STATS.worker_started_at) if STATS.worker_started_at else 0.0
            # Aggregate CPU/RAM across parent + children
            cpu_total = ""
            ram_total = ""
            workers_alive = ""
            if psutil:
                try:
                    proc = psutil.Process(os.getpid())
                    children = [c for c in proc.children(recursive=True) if c.is_running()]
                    cpu_p = proc.cpu_percent(interval=None) or 0.0
                    cpu_c = sum((c.cpu_percent(interval=None) or 0.0) for c in children)
                    cpu_total = f" • CPU {cpu_p + cpu_c:.0f}% (all)"
                    rss = proc.memory_info().rss
                    rss_c = 0
                    for c in children:
                        try: rss_c += c.memory_info().rss
                        except Exception: pass
                    ram_total = f" • RAM ~{(rss + rss_c)/(1024**3):.1f} GB"
                    workers_alive = f" • workers {len(children)}"
                except Exception:
                    pass

            pct = (STATS.tiles_done * 100.0 / max(1, STATS.tiles_total)) if STATS.tiles_total else 0.0
            tiles_per_min = (STATS.tiles_done / worker_elapsed * 60.0) if worker_elapsed > 0 and STATS.tiles_done > 0 else 0.0
            eta = _fmt_eta(worker_elapsed, STATS.tiles_done, STATS.tiles_total) if STATS.worker_started_at else "ETA —"
            trend = ""
            if last_tiles >= 0:
                delta = STATS.tiles_done - last_tiles
                if delta == 0 and STATS.tiles_total > 0:
                    trend = " • waiting on first results…" if STATS.tiles_done == 0 else " • steady…"
                elif delta > 0:
                    trend = f" • +{delta} tiles since last HB"
            last_tiles = STATS.tiles_done

            log_to_gui(
                f"[Mosaic/HB] {STATS.stage} {STATS.detail} • tiles {STATS.tiles_done}/{STATS.tiles_total} "
                f"(~{pct:.1f}%) • faces {STATS.faces_total:,} • elapsed {int(elapsed)}s • {eta} • "
                f"{tiles_per_min:.1f} tiles/min{trend}{ram_total}{cpu_total}{workers_alive}"
            )
            time.sleep(max(1, HEARTBEAT_SECS))
    import threading
    threading.Thread(target=_hb, daemon=True).start()

def stop_heartbeat():
    STATS.running = False

# -----------------------------------------------------------------------------
# Geo helpers
# -----------------------------------------------------------------------------
def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    try:
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        elif str(gdf.crs).upper() not in ("EPSG:4326", "WGS84"):
            gdf = gdf.to_crs(4326)
    except Exception:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    return gdf

def working_metric_crs_for(gdf: gpd.GeoDataFrame, cfg: configparser.ConfigParser) -> str:
    epsg = (cfg["DEFAULT"].get("workingprojection_epsg", "") if "DEFAULT" in cfg else "").strip()
    if epsg and epsg.isdigit() and epsg != "4326":
        return f"EPSG:{epsg}"
    if gdf.crs and getattr(gdf.crs, "is_projected", False):
        try:
            return gdf.crs.to_string()
        except Exception:
            pass
    return "EPSG:3857"

def area_projection(cfg: configparser.ConfigParser) -> str:
    epsg = (cfg["DEFAULT"].get("area_projection_epsg", "3035")
            if "DEFAULT" in cfg else "3035")
    return f"EPSG:{epsg}" if str(epsg).isdigit() else epsg

def _fix_valid(geom):
    if geom is None:
        return None
    try:
        if shapely_make_valid:
            return shapely_make_valid(geom)
        return geom.buffer(0)
    except Exception:
        return geom

def _wkb_hex(geom) -> Optional[str]:
    try:
        return shp_wkb.dumps(geom, hex=True)
    except Exception:
        return None

# -----------------------------------------------------------------------------
# H3 helpers
# -----------------------------------------------------------------------------
def _h3_version() -> str:
    return getattr(h3, "__version__", "unknown") if h3 else "none"

def _cell_boundary(index) -> list[tuple]:
    if h3 is None:
        return []
    b = h3.cell_to_boundary(index) if hasattr(h3, "cell_to_boundary") else h3.h3_to_geo_boundary(index)
    out = []
    for p in b:
        if isinstance(p, dict):
            lat = p.get("lat") or p.get("latitude")
            lng = p.get("lng") or p.get("lon") or p.get("longitude")
            if lat is not None and lng is not None:
                out.append((lng, lat))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            lat, lng = p[0], p[1]
            out.append((lng, lat))
    return out

def _extract_polygonal(geom):
    if geom is None:
        return None
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if not polys:
            return None
        u = unary_union(polys)
        return u.buffer(0) if not u.is_empty else None
    try:
        g0 = geom.buffer(0)
        if isinstance(g0, (Polygon, MultiPolygon)):
            return g0
        if isinstance(g0, GeometryCollection):
            polys = [g for g in g0.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
            if not polys:
                return None
            return unary_union(polys)
    except Exception:
        pass
    return None

def _polyfill_cells(poly: Polygon, res: int) -> set[str]:
    if h3 is None:
        raise RuntimeError("H3 module not available")
    gj = shp_mapping(poly)
    if hasattr(h3, "geo_to_cells"):
        try:
            return set(h3.geo_to_cells(gj, res))
        except Exception as e:
            log_to_gui(f"H3 geo_to_cells failed (R{res}): {e}", "WARN")
    if hasattr(h3, "polyfill"):
        tried = [
            lambda: h3.polyfill(gj, res, True),
            lambda: h3.polyfill(gj, res, geo_json_conformant=True),
            lambda: h3.polyfill(gj, res, geojson_conformant=True),
        ]
        for fn in tried:
            try:
                return set(fn())
            except TypeError:
                continue
        log_to_gui(f"H3 polyfill failed (R{res}) with all signatures.", "WARN")
    raise RuntimeError("No compatible H3 polyfill worked.")

def h3_from_union(union_geom, res: int) -> gpd.GeoDataFrame:
    if union_geom is None:
        return gpd.GeoDataFrame(columns=["h3_index", "geometry"], geometry="geometry", crs="EPSG:4326")
    geom_poly = _extract_polygonal(union_geom)
    if geom_poly is None or geom_poly.is_empty:
        return gpd.GeoDataFrame(columns=["h3_index", "geometry"], geometry="geometry", crs="EPSG:4326")
    polys = list(geom_poly.geoms) if geom_poly.geom_type == "MultiPolygon" else [geom_poly]
    hexes: set[str] = set()
    for poly in polys:
        try:
            hexes |= _polyfill_cells(poly, res)
        except Exception as e:
            log_to_gui(f"H3 polyfill failed (R{res}): {e}", "WARN")
    if not hexes:
        return gpd.GeoDataFrame(columns=["h3_index", "geometry"], geometry="geometry", crs="EPSG:4326")
    rows = [{"h3_index": h, "geometry": Polygon(_cell_boundary(h))} for h in hexes]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

# -----------------------------------------------------------------------------
# GeoParquet-first union for H3
# -----------------------------------------------------------------------------
def _read_parquet_gdf(path: Path, default_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    if path.exists():
        try:
            gdf = gpd.read_parquet(path)
            if gdf.crs is None:
                gdf.set_crs(default_crs, inplace=True)
            return gdf
        except Exception as e:
            log_to_gui(f"Failed reading {path.name}: {e}", "WARN")
    return gpd.GeoDataFrame(geometry=[], crs=default_crs)

def union_from_asset_groups_or_objects(base_dir: Path):
    cfg = read_config(base_dir / "system" / "config.ini")
    pq_groups = geoparquet_path(base_dir, "tbl_asset_group")
    g = _read_parquet_gdf(pq_groups); g = ensure_wgs84(g)
    if not g.empty and "geometry" in g:
        try:
            u = unary_union(g.geometry); u = _extract_polygonal(u)
            if u and not u.is_empty:
                log_to_gui("Union source: GeoParquet tbl_asset_group", "INFO")
                return u
        except Exception:
            pass
    pq_objs = geoparquet_path(base_dir, "tbl_asset_object")
    ao = _read_parquet_gdf(pq_objs); ao = ensure_wgs84(ao)
    if not ao.empty and "geometry" in ao:
        try:
            poly_mask = ao.geometry.geom_type.isin(["Polygon","MultiPolygon","GeometryCollection"])
            if poly_mask.any():
                u = unary_union(ao.loc[poly_mask, "geometry"]); u = _extract_polygonal(u)
                if u and not u.is_empty:
                    log_to_gui("Union source: GeoParquet tbl_asset_object (polygons)", "INFO")
                    return u
            try:
                buf_m = float(cfg["DEFAULT"].get("h3_union_buffer_m", "50"))
            except Exception:
                buf_m = 50.0
            metric_crs = working_metric_crs_for(ao, cfg)
            aom = ao.to_crs(metric_crs)
            aom["geometry"] = aom.geometry.buffer(max(0.01, buf_m))
            aom["geometry"] = aom.geometry.apply(_fix_valid)
            aom = aom[aom.geometry.notna() & ~aom.geometry.is_empty]
            if not aom.empty:
                u = unary_union(aom.geometry); u = _extract_polygonal(u)
                if u and not u.is_empty:
                    u_wgs84 = ensure_wgs84(gpd.GeoSeries([u], crs=aom.crs)).iloc[0]
                    log_to_gui(f"Union source: GeoParquet tbl_asset_object (buffer {buf_m} m)", "INFO")
                    return u_wgs84
        except Exception:
            pass
    return None

def estimate_cells_for(union_geom, res: int, cfg: configparser.ConfigParser) -> tuple[float,float]:
    proj = area_projection(cfg)
    gs = gpd.GeoSeries([union_geom], crs="EPSG:4326").to_crs(proj)
    area_km2 = float(gs.area.iloc[0]) / 1_000_000.0
    avg = AVG_HEX_AREA_KM2.get(int(res), None)
    if not avg or avg <= 0:
        return (area_km2, float("inf"))
    return (area_km2, area_km2 / avg)

# -----------------------------------------------------------------------------
# Geocode writers — APPEND/MERGE semantics (Parquet-only)
# -----------------------------------------------------------------------------
def _bbox_polygon_from(thing) -> Optional[Polygon]:
    try:
        if isinstance(thing, (gpd.GeoDataFrame, gpd.GeoSeries)):
            t = ensure_wgs84(thing)
            if t.empty: return None
            minx, miny, maxx, maxy = t.total_bounds
        else:
            if thing is None or getattr(thing, "is_empty", True): return None
            minx, miny, maxx, maxy = thing.bounds
        arr = np.array([minx, miny, maxx, maxy], dtype=float)
        if not np.isfinite(arr).all(): return None
        if maxx <= minx or maxy <= miny:
            dx = dy = 1e-6
            minx, miny, maxx, maxy = minx - dx, miny - dy, maxx + dx, maxy + dy
        return box(minx, miny, maxx, maxy)
    except Exception:
        return None

def _load_existing_geocodes(base_dir: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    geodir = gpq_dir(base_dir)
    pg = geodir / "tbl_geocode_group.parquet"
    po = geodir / "tbl_geocode_object.parquet"
    if pg.exists():
        try:
            g = gpd.read_parquet(pg)
            if g.crs is None: g.set_crs("EPSG:4326", inplace=True)
        except Exception:
            g = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    else:
        g = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if po.exists():
        try:
            o = gpd.read_parquet(po)
            if o.crs is None: o.set_crs("EPSG:4326", inplace=True)
        except Exception:
            o = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    else:
        o = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    g = ensure_wgs84(g); o = ensure_wgs84(o)
    if "id" not in g.columns:
        log_to_gui("Existing geocode group table lacks 'id' — treating as empty.", "WARN")
        g = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"); o = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return g, o

def load_geocode_groups(base_dir: Path) -> gpd.GeoDataFrame:
    pg = gpq_dir(base_dir) / "tbl_geocode_group.parquet"
    if pg.exists():
        try:
            g = gpd.read_parquet(pg)
            if g.crs is None: g.set_crs("EPSG:4326", inplace=True)
            return ensure_wgs84(g)
        except Exception:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

def mosaic_exists(base_dir: Path) -> bool:
    g = load_geocode_groups(base_dir)
    if g.empty or "name_gis_geocodegroup" not in g.columns:
        return False
    return BASIC_MOSAIC_GROUP in set(g["name_gis_geocodegroup"].astype(str))

def _merge_and_write_geocodes(base_dir: Path,
                              new_groups_gdf: gpd.GeoDataFrame,
                              new_objects_gdf: gpd.GeoDataFrame,
                              refresh_group_names: List[str]) -> tuple[int,int,int,int]:
    existing_g, existing_o = _load_existing_geocodes(base_dir)
    out_dir = gpq_dir(base_dir)

    if not existing_g.empty and "name_gis_geocodegroup" in existing_g and refresh_group_names:
        rm_mask = existing_g["name_gis_geocodegroup"].isin(refresh_group_names)
        rm_ids = set(existing_g.loc[rm_mask, "id"].astype(int).tolist())
        if rm_ids:
            existing_g = existing_g.loc[~rm_mask].copy()
            if not existing_o.empty and "ref_geocodegroup" in existing_o:
                existing_o = existing_o.loc[~existing_o["ref_geocodegroup"].astype(int).isin(rm_ids)].copy()
            log_to_gui(f"Refreshed existing groups removed: {len(rm_ids)}", "INFO")

    start_id = int(existing_g["id"].max()) + 1 if ("id" in existing_g.columns and not existing_g.empty) else 1
    new_groups_gdf = new_groups_gdf.copy()
    new_groups_gdf["id"] = list(range(start_id, start_id + len(new_groups_gdf)))

    name_to_id = dict(zip(new_groups_gdf["name_gis_geocodegroup"], new_groups_gdf["id"]))
    new_objects_gdf = new_objects_gdf.copy()
    new_objects_gdf["ref_geocodegroup"] = new_objects_gdf["name_gis_geocodegroup"].map(name_to_id)

    groups_out = pd.concat([existing_g, new_groups_gdf], ignore_index=True)
    objects_out = pd.concat([existing_o, new_objects_gdf], ignore_index=True)

    groups_out = ensure_wgs84(gpd.GeoDataFrame(groups_out, geometry="geometry"))
    objects_out = ensure_wgs84(gpd.GeoDataFrame(objects_out, geometry="geometry"))

    groups_out.to_parquet(out_dir / "tbl_geocode_group.parquet", index=False)
    objects_out.to_parquet(out_dir / "tbl_geocode_object.parquet", index=False)

    return len(new_groups_gdf), len(new_objects_gdf), len(groups_out), len(objects_out)

# -----------------------------------------------------------------------------
# Assets (GeoParquet)
# -----------------------------------------------------------------------------
def _load_asset_objects(base_dir: Path) -> gpd.GeoDataFrame:
    pq = geoparquet_path(base_dir, "tbl_asset_object")
    if pq.exists():
        try:
            g = gpd.read_parquet(pq)
            if g.crs is None:
                g.set_crs("EPSG:4326", inplace=True)
            return ensure_wgs84(g)
        except Exception as e:
            log_to_gui(f"Parquet read failed ({pq.name}): {e}", "WARN")
    return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

# -----------------------------------------------------------------------------
# Mosaic worker (buffer/simplify inside worker)
# -----------------------------------------------------------------------------
def _mosaic_tile_worker(task: Tuple[int, List[bytes], float, float]):
    """
    Input: (tile_index, [WKB of original geoms (metric CRS)], buffer_m, simplify_tol)
    Output: (tile_index, [WKB faces], error:str|None)
    """
    idx, wkb_list, buf_m, simp = task
    try:
        if not wkb_list:
            return (idx, [], None)

        geoms = [shp_wkb.loads(b) for b in wkb_list if b]
        buf = max(0.01, float(buf_m))
        processed = []
        for g in geoms:
            if g and not g.is_empty:
                try:
                    gb = g.buffer(buf)
                    if shapely_make_valid:
                        gb = shapely_make_valid(gb)
                    else:
                        gb = gb.buffer(0)
                    if simp and simp > 0:
                        gb = gb.simplify(simp, preserve_topology=True)
                    if gb and not gb.is_empty:
                        processed.append(gb)
                except Exception:
                    continue
        if not processed:
            return (idx, [], None)

        lines = []
        for g in processed:
            try:
                lb = g.boundary
                if lb and not lb.is_empty:
                    lines.append(lb)
            except Exception:
                continue
        if not lines:
            return (idx, [], None)

        def batched_union(seq, max_batch):
            out = None
            step = max_batch
            i = 0
            while i < len(seq):
                chunk = seq[i:i+step]
                try:
                    u = unary_union(chunk)
                except Exception:
                    if step <= 200:
                        raise
                    step = max(200, step // 2)
                    continue
                out = u if out is None else unary_union([out, u])
                i += step
            return out

        edge_net = batched_union(lines, max_batch=1500)
        faces = list(polygonize(edge_net))
        res = [shp_wkb.dumps(f) for f in faces
               if isinstance(f, (Polygon, MultiPolygon)) and not f.is_empty]
        return (idx, res, None)

    except MemoryError:
        return (idx, [], "memory")
    except Exception as e:
        return (idx, [], str(e))

# -----------------------------------------------------------------------------
# Quadtree tiler (+ heavy split)
# -----------------------------------------------------------------------------
def _plan_tiles_quadtree(a_metric: gpd.GeoDataFrame,
                         sidx,
                         minx: float, miny: float, maxx: float, maxy: float,
                         *,
                         overlap_m: float,
                         max_feats_per_tile: int = 800,
                         max_depth: int = 8,
                         min_tile_size_m: float = 0.0) -> List[Tuple[Tuple[float,float,float,float], List[int]]]:
    leaves: List[Tuple[Tuple[float,float,float,float], List[int]]] = []
    stack = [(minx, miny, maxx, maxy, 0)]
    eps = 1e-3
    steps = 0

    while stack:
        bx0, by0, bx1, by1, depth = stack.pop()
        w, h = (bx1 - bx0), (by1 - by0)
        if w <= eps or h <= eps:
            continue

        tile_poly = box(bx0, by0, bx1, by1).buffer(overlap_m)
        try:
            idxs = list(sidx.query(tile_poly, predicate="intersects"))
        except Exception:
            idxs = list(sidx.query(tile_poly))
        n = len(idxs)
        steps += 1
        if steps % 200 == 0:
            STATS.detail = f"(planner steps: {steps:,})"

        if n == 0:
            continue

        stop_for_size = (min_tile_size_m > 0.0 and (w <= min_tile_size_m and h <= min_tile_size_m))
        if n <= max_feats_per_tile or depth >= max_depth or stop_for_size:
            leaves.append(((bx0, by0, bx1, by1), idxs))
            continue

        mx = (bx0 + bx1) * 0.5
        my = (by0 + by1) * 0.5
        stack.extend([
            (bx0, by0, mx,  my,  depth + 1),
            (mx,  by0, bx1, my,  depth + 1),
            (bx0, my,  mx,  by1, depth + 1),
            (mx,  my,  bx1, by1, depth + 1),
        ])

    return leaves

def _split_tile(bounds, a_metric, sidx, overlap_m):
    (bx0, by0, bx1, by1) = bounds
    mx = (bx0 + bx1) * 0.5
    my = (by0 + by1) * 0.5
    out = []
    for (x0, y0, x1, y1) in [(bx0,by0,mx,my),(mx,by0,bx1,my),(bx0,my,mx,by1),(mx,my,bx1,by1)]:
        tp = box(x0, y0, x1, y1).buffer(overlap_m)
        try:
            idxs = list(sidx.query(tp, predicate="intersects"))
        except Exception:
            idxs = list(sidx.query(tp))
        if idxs:
            out.append(((x0,y0,x1,y1), idxs))
    return out

# -----------------------------------------------------------------------------
# Task ordering helpers (for snappy early progress)
# -----------------------------------------------------------------------------
def _order_tasks(counts: List[int], mode: str) -> List[int]:
    n = len(counts)
    idxs = np.arange(n)
    if mode == "heavy_first":
        return list(np.argsort(-np.array(counts)))
    if mode == "light_first":
        return list(np.argsort(np.array(counts)))
    # interleave: small, large, small, large …
    asc = list(np.argsort(np.array(counts)))
    i, j = 0, len(asc) - 1
    out = []
    while i <= j:
        out.append(asc[i]); i += 1
        if i <= j:
            out.append(asc[j]); j -= 1
    return out

# -----------------------------------------------------------------------------
# Mosaic builder (parallel + detailed heartbeats)
# -----------------------------------------------------------------------------
def mosaic_faces_from_assets_parallel(base_dir: Path,
                                      buffer_m: float,
                                      grid_size_m: float,
                                      workers: int) -> gpd.GeoDataFrame:
    t0 = time.time()
    cfg = read_config(base_dir / "system" / "config.ini")

    # heartbeat config
    global HEARTBEAT_SECS
    try:
        HEARTBEAT_SECS = int(cfg["DEFAULT"].get("heartbeat_secs", str(HEARTBEAT_SECS)))
    except Exception:
        pass

    # Start heartbeat immediately
    STATS.stage = "starting"
    STATS.detail = ""
    STATS.tiles_total = 0
    STATS.tiles_done = 0
    STATS.faces_total = 0
    STATS.started_at = time.time()
    STATS.worker_started_at = None
    start_heartbeat()

    # Load assets
    STATS.stage = "loading assets"
    assets = _load_asset_objects(base_dir)
    log_to_gui(f"[Mosaic] Loaded assets: {len(assets):,} rows; CRS={assets.crs}")
    if assets.empty:
        stop_heartbeat()
        log_to_gui("[Mosaic] No asset objects found; mosaic skipped.", "WARN")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # CRS
    STATS.stage = "reprojecting"
    metric_crs = working_metric_crs_for(assets, cfg)
    t = time.time()
    a = assets.to_crs(metric_crs).copy()
    log_to_gui(f"[Mosaic] Reprojected to {metric_crs} in {time.time()-t:.2f}s.")

    # Spatial index
    STATS.stage = "building sindex"
    t = time.time()
    try:
        sidx = a.sindex
        log_to_gui(f"[Mosaic] Built spatial index on {len(a):,} geoms in {time.time()-t:.2f}s.")
    except Exception:
        stop_heartbeat()
        log_to_gui("[Mosaic] Spatial index not available; aborting.", "WARN")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Tiler params
    try: max_feats_per_tile = int(cfg["DEFAULT"].get("mosaic_quadtree_max_feats_per_tile", "800"))
    except Exception: max_feats_per_tile = 800
    try: heavy_mult = float(cfg["DEFAULT"].get("mosaic_quadtree_heavy_split_multiplier", "2.0"))
    except Exception: heavy_mult = 2.0
    heavy_threshold = max(1, int(max_feats_per_tile * heavy_mult))
    try: max_depth = int(cfg["DEFAULT"].get("mosaic_quadtree_max_depth", "8"))
    except Exception: max_depth = 8
    try: min_tile_size_m = float(cfg["DEFAULT"].get("mosaic_quadtree_min_tile_m", str(grid_size_m)))
    except Exception: min_tile_size_m = max(0.0, float(grid_size_m))
    try: simplify_tol = float(cfg["DEFAULT"].get("mosaic_simplify_tolerance_m", "0"))
    except Exception: simplify_tol = 0.0
    try: flush_batch = int(cfg["DEFAULT"].get("mosaic_faces_flush_batch", "250000"))
    except Exception: flush_batch = 250000
    try: maxtasks = int(cfg["DEFAULT"].get("mosaic_pool_maxtasksperchild", "8"))
    except Exception: maxtasks = 8
    task_order = cfg["DEFAULT"].get("mosaic_task_order", "interleave").strip().lower()
    try:
        chunksize = int(cfg["DEFAULT"].get("mosaic_pool_chunksize", "1"))
    except Exception:
        chunksize = 1

    # Plan tiles
    STATS.stage = "planning tiles"
    t_plan0 = time.time()
    minx, miny, maxx, maxy = a.total_bounds
    leaves = _plan_tiles_quadtree(
        a_metric=a, sidx=sidx, minx=minx, miny=miny, maxx=maxx, maxy=maxy,
        overlap_m=buffer_m, max_feats_per_tile=max_feats_per_tile,
        max_depth=max_depth, min_tile_size_m=max(0.0, float(min_tile_size_m))
    )
    log_to_gui(f"[Mosaic] Planned {len(leaves):,} tiles (first pass); {time.time()-t_plan0:.2f}s.")

    # Heavy split
    STATS.stage = "splitting heavy tiles"
    balanced: List[Tuple[Tuple[float,float,float,float], List[int]]] = []
    heavy_count = 0
    for (bds, idxs) in leaves:
        if len(idxs) > heavy_threshold and max_depth > 0:
            st = _split_tile(bds, a, sidx, overlap_m=buffer_m)
            if st:
                balanced.extend(st)
                heavy_count += 1
            else:
                balanced.append((bds, idxs))
        else:
            balanced.append((bds, idxs))
    leaves = balanced
    log_to_gui(f"[Mosaic] Heavy tiles split: {heavy_count}; final tiles: {len(leaves):,}.")

    # Prepare tasks (WKB of ORIGINAL geoms; workers will buffer+fix+simplify)
    STATS.stage = "packing tasks"
    counts = []
    tasks: List[Tuple[int, List[bytes], float, float]] = []
    tick = time.time()
    for i, (_bounds, idxs) in enumerate(leaves, start=1):
        sub = a.iloc[idxs]
        counts.append(len(sub))
        wkb_list = [shp_wkb.dumps(g) for g in sub.geometry]
        tasks.append((i-1, wkb_list, float(buffer_m), float(simplify_tol)))
        if i % 200 == 0 or (time.time() - tick) >= 5:
            STATS.detail = f"(packed {i}/{len(leaves)})"
            tick = time.time()

    order_idx = _order_tasks(counts, task_order)
    tasks = [tasks[i] for i in order_idx]
    counts = [counts[i] for i in order_idx]

    n_tasks = len(tasks)
    STATS.tiles_total = n_tasks
    STATS.detail = f"(min/med/p95/max feats per tile = {min(counts) if counts else 0}/" \
                   f"{int(np.percentile(counts,50)) if counts else 0}/" \
                   f"{int(np.percentile(counts,95)) if counts else 0}/" \
                   f"{max(counts) if counts else 0})"
    log_to_gui(f"[Mosaic] Prepared {n_tasks:,} tasks; order={task_order}; chunksize={chunksize}; {STATS.detail}")

    # Determine workers
    if workers is None or workers <= 0:
        try: workers = max(1, mp.cpu_count())
        except Exception: workers = 4
    log_to_gui(f"[Mosaic] Parallel polygonize; workers={workers}, maxtasksperchild={maxtasks}.")

    # Streaming parts dir
    tmp_dir = gpq_dir(base_dir) / "__mosaic_faces_tmp"
    try:
        if tmp_dir.exists():
            for p in tmp_dir.glob("*.parquet"):
                try: p.unlink()
                except Exception: pass
        else:
            tmp_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    face_batches: List[bytes] = []
    part_files: List[Path] = []

    def _flush_batch_to_parquet(metric_crs: str):
        nonlocal face_batches, part_files
        if not face_batches:
            return
        gseries = gpd.GeoSeries([shp_wkb.loads(b) for b in face_batches], crs=metric_crs)
        gdf = gpd.GeoDataFrame(geometry=gseries).to_crs("EPSG:4326")
        part = tmp_dir / f"faces_part_{len(part_files):05d}.parquet"
        gdf.to_parquet(part, index=False)
        part_files.append(part)
        STATS.faces_total += len(gdf)
        face_batches = []

    # Process tasks
    STATS.stage = "polygonize (workers)"
    STATS.worker_started_at = time.time()
    ctx = mp.get_context("spawn")
    try:
        if workers == 1 or n_tasks == 1:
            for (idx, wkb_list, bufm, simp) in tasks:
                _, res, err = _mosaic_tile_worker((idx, wkb_list, bufm, simp))
                STATS.tiles_done += 1
                if err:
                    log_to_gui(f"[Mosaic] Tile {idx+1}/{n_tasks} error: {err}", "WARN")
                else:
                    face_batches.extend(res)
                    if len(face_batches) >= flush_batch:
                        _flush_batch_to_parquet(metric_crs)
                update_progress(5 + STATS.tiles_done * (80 / max(1, n_tasks)))
        else:
            with ctx.Pool(processes=workers, maxtasksperchild=maxtasks) as pool:
                # NOTE: small chunksize -> quicker first results / better progress feedback
                for (idx, res, err) in pool.imap_unordered(_mosaic_tile_worker, tasks, chunksize=max(1, chunksize)):
                    STATS.tiles_done += 1
                    if err:
                        log_to_gui(f"[Mosaic] Tile {idx+1}/{n_tasks} error: {err}", "WARN")
                    else:
                        face_batches.extend(res)
                        if len(face_batches) >= flush_batch:
                            _flush_batch_to_parquet(metric_crs)
                    update_progress(5 + STATS.tiles_done * (80 / max(1, n_tasks)))
    finally:
        # final flush
        _flush_batch_to_parquet(metric_crs)
        log_to_gui(f"[Mosaic] Worker stage done: tiles {STATS.tiles_done}/{STATS.tiles_total}, faces so far {STATS.faces_total:,}.")

    # Assemble faces from parts
    STATS.stage = "assembling parts"
    parts_gdfs = []
    for pf in part_files:
        try:
            parts_gdfs.append(gpd.read_parquet(pf))
        except Exception as e:
            log_to_gui(f"[Mosaic] Skipping part {pf.name}: {e}", "WARN")
    if face_batches:
        gseries = gpd.GeoSeries([shp_wkb.loads(b) for b in face_batches], crs=metric_crs)
        parts_gdfs.append(gpd.GeoDataFrame(geometry=gseries).to_crs("EPSG:4326"))
        STATS.faces_total += len(gseries)

    if not parts_gdfs:
        stop_heartbeat()
        log_to_gui("[Mosaic] No valid face parts to assemble.", "WARN")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    faces = pd.concat(parts_gdfs, ignore_index=True)
    faces = ensure_wgs84(gpd.GeoDataFrame(faces, geometry="geometry"))

    # Deduplicate
    STATS.stage = "deduplicating"
    t_ded = time.time()
    try:
        wkb_hex = faces.geometry.apply(_wkb_hex)
        keep = ~pd.Series(wkb_hex).duplicated()
        faces = faces.loc[keep.values].reset_index(drop=True)
    except Exception:
        pass
    log_to_gui(f"[Mosaic] Dedup complete in {time.time()-t_ded:.2f}s; faces={len(faces):,}.")
    STATS.faces_total = len(faces)

    stop_heartbeat()
    log_to_gui(f"[Mosaic] Completed in {time.time()-t0:.2f}s. Final faces: {len(faces):,}.")
    return faces

# -----------------------------------------------------------------------------
# Publish mosaic as geocode
# -----------------------------------------------------------------------------
def publish_mosaic_as_geocode(base_dir: Path, faces: gpd.GeoDataFrame) -> int:
    if faces is None or faces.empty:
        log_to_gui("No mosaic faces to publish.", "WARN")
        return 0

    group_name = BASIC_MOSAIC_GROUP

    try:
        cfg = read_config(base_dir / "system" / "config.ini")
        metric_crs = working_metric_crs_for(faces, cfg)
        cent = faces.to_crs(metric_crs).geometry.centroid
        order_idx = np.lexsort((cent.y.values, cent.x.values))
        faces = faces.iloc[order_idx].reset_index(drop=True)
    except Exception:
        faces = faces.reset_index(drop=True)

    codes = [f"{group_name}_{i:06d}" for i in range(1, len(faces) + 1)]

    obj = faces.copy()
    obj["code"] = codes
    obj["name_gis_geocodegroup"] = group_name
    obj["attributes"] = None
    obj = obj[["code", "name_gis_geocodegroup", "attributes", "geometry"]]

    bbox_poly = _bbox_polygon_from(faces)
    groups = gpd.GeoDataFrame([{
        "name": group_name,
        "name_gis_geocodegroup": group_name,
        "title_user": "Basic mosaic",
        "description": "Atomic faces derived from buffered assets (polygonize).",
        "geometry": bbox_poly
    }], geometry="geometry", crs="EPSG:4326")

    added_g, added_o, tot_g, tot_o = _merge_and_write_geocodes(
        base_dir, groups, obj, refresh_group_names=[group_name]
    )
    log_to_gui(
        f"Published mosaic geocode '{group_name}' → "
        f"added objects: {added_o:,}; totals => groups: {tot_g}, objects: {tot_o:,}"
    )
    return added_o

# -----------------------------------------------------------------------------
# H3 writers
# -----------------------------------------------------------------------------
def write_h3_levels(base_dir: Path, levels: List[int]) -> int:
    if not levels:
        log_to_gui("No H3 levels selected.", "WARN")
        return 0
    if h3 is None:
        log_to_gui("H3 Python package not available. Install with: pip install h3", "WARN")
        return 0
    cfg = read_config(base_dir / "system" / "config.ini")
    max_cells = float(cfg["DEFAULT"].get("h3_max_cells", "1200000")) if "DEFAULT" in cfg else 1_200_000.0
    union_geom = union_from_asset_groups_or_objects(base_dir)
    if union_geom is None:
        log_to_gui("No polygonal AOI found in tbl_asset_group/tbl_asset_object (consider polygons or set [DEFAULT] h3_union_buffer_m).", "WARN")
        return 0
    log_to_gui(f"H3 version: {_h3_version()}", "INFO")

    groups_rows = []
    objects_parts = []
    levels_sorted = sorted(set(int(r) for r in levels))
    steps = max(1, len(levels_sorted))
    bbox_poly = _bbox_polygon_from(union_geom)
    if bbox_poly is None:
        raise RuntimeError("Failed to compute bbox for H3 group.")

    for i, r in enumerate(levels_sorted):
        update_progress(5 + i * (80 / steps))
        area_km2, approx_cells = estimate_cells_for(union_geom, r, cfg)
        if approx_cells > max_cells:
            log_to_gui(
                f"Skipping H3 R{r}: AOI ~{area_km2:,.1f} km² → ~{approx_cells:,.0f} cells exceeds cap ({max_cells:,.0f}).",
                "WARN",
            )
            continue
        group_name = f"H3_R{r}"
        gdf = h3_from_union(union_geom, r)
        if gdf.empty:
            log_to_gui(f"No H3 cells produced for resolution {r}.", "WARN")
            continue
        gdf = gdf.rename(columns={"h3_index": "code"})
        if "attributes" not in gdf.columns: gdf["attributes"] = None
        gdf["name_gis_geocodegroup"] = group_name
        gdf = gdf[["code", "name_gis_geocodegroup", "attributes", "geometry"]]
        objects_parts.append(gdf)
        log_to_gui(f"H3 R{r}: prepared {len(gdf):,} cells.")
        groups_rows.append({
            "name": group_name, "name_gis_geocodegroup": group_name,
            "title_user": f"H3 resolution {r}",
            "description": f"H3 hexagons at resolution {r}",
            "geometry": bbox_poly
        })

    if not groups_rows or not objects_parts:
        log_to_gui("No H3 output generated (all levels skipped/empty).", "WARN")
        out_dir = gpq_dir(base_dir)
        if not (out_dir / "tbl_geocode_group.parquet").exists():
            gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_parquet(out_dir / "tbl_geocode_group.parquet", index=False)
        if not (out_dir / "tbl_geocode_object.parquet").exists():
            gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_parquet(out_dir / "tbl_geocode_object.parquet", index=False)
        return 0

    new_groups = gpd.GeoDataFrame(groups_rows, geometry="geometry", crs="EPSG:4326")
    new_objects = gpd.GeoDataFrame(pd.concat(objects_parts, ignore_index=True), geometry="geometry", crs="EPSG:4326")

    added_g, added_o, tot_g, tot_o = _merge_and_write_geocodes(
        base_dir, new_groups, new_objects, refresh_group_names=[r["name_gis_geocodegroup"] for r in groups_rows]
    )
    log_to_gui(
        f"Merged GeoParquet geocodes → {gpq_dir(base_dir)}  "
        f"(added groups: {added_g}, added objects: {added_o:,}; totals => groups: {tot_g}, objects: {tot_o:,})"
    )
    return added_o

# -----------------------------------------------------------------------------
# Mosaic runner
# -----------------------------------------------------------------------------
def run_mosaic(base_dir: Path, buffer_m: float, grid_size_m: float, on_done=None):
    cfg = read_config(base_dir / "system" / "config.ini")
    try:
        workers = int(cfg["DEFAULT"].get("mosaic_workers", "0"))
    except Exception:
        workers = 0
    faces = mosaic_faces_from_assets_parallel(base_dir, buffer_m, grid_size_m, workers)
    if faces.empty:
        log_to_gui("Mosaic produced no faces to publish.", "WARN")
        update_progress(100)
        if on_done:
            try: on_done(False)
            except Exception: pass
        return
    n = publish_mosaic_as_geocode(base_dir, faces)
    log_to_gui(f"Mosaic published as geocode group '{BASIC_MOSAIC_GROUP}' with {n:,} objects.")
    update_progress(100)
    if on_done:
        try: on_done(True)
        except Exception: pass

# -----------------------------------------------------------------------------
# GUI helpers for H3 level suggestions
# -----------------------------------------------------------------------------
def suggest_h3_levels_by_size(min_km: float, max_km: float) -> list[int]:
    out = []
    for res, size in H3_RES_ACROSS_FLATS_KM.items():
        if min_km <= size <= max_km:
            out.append(res)
    return out

def format_level_size_list(levels: list[int]) -> str:
    if not levels:
        return "(none)"
    return ", ".join(f"R{r} ({H3_RES_ACROSS_FLATS_M[r]:,.0f} m)" for r in levels)

# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------
def build_gui(base: Path, cfg: configparser.ConfigParser):
    global root, log_widget, progress_var, progress_label, original_working_directory, mosaic_status_var, size_levels_var
    original_working_directory = str(base)

    # heartbeat secs
    try:
        global HEARTBEAT_SECS
        HEARTBEAT_SECS = int(cfg["DEFAULT"].get("heartbeat_secs", str(HEARTBEAT_SECS)))
    except Exception:
        pass

    theme = cfg["DEFAULT"].get("ttk_bootstrap_theme", "flatly") if ttk else None
    root = ttk.Window(themename=theme) if ttk else tk.Tk()
    root.title("Create geocodes (H3 / Mosaic)")

    frame = ttk.LabelFrame(root, text="Log", bootstyle="info") if ttk else tk.LabelFrame(root, text="Log")
    frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    log_widget = scrolledtext.ScrolledText(frame, height=12)
    log_widget.pack(fill=tk.BOTH, expand=True)

    pframe = tk.Frame(root); pframe.pack(pady=6)
    progress_var = tk.DoubleVar()
    pbar = (ttk.Progressbar(pframe, orient="horizontal", length=240, mode="determinate",
                            variable=progress_var, bootstyle="info")
            if ttk else tk.Scale(pframe, orient="horizontal", length=240, from_=0, to=100))
    pbar.pack(side=tk.LEFT, padx=6)
    progress_label = tk.Label(pframe, text="0%"); progress_label.pack(side=tk.LEFT)

    log_to_gui(f"Base dir: {base}")
    log_to_gui(f"GeoParquet out: {gpq_dir(base)}")
    log_to_gui("GeoParquet-first; outputs go ONLY to tbl_geocode_*.")
    log_to_gui(f"Mosaic geocode group is fixed to '{BASIC_MOSAIC_GROUP}'.")

    size_frame = tk.LabelFrame(root, text="H3 size-based selection")
    size_frame.pack(padx=10, pady=(4,6), fill=tk.X)
    tk.Label(size_frame, text="Min m:").grid(row=0, column=0, padx=4, pady=2, sticky="e")
    tk.Label(size_frame, text="Max m:").grid(row=0, column=2, padx=4, pady=2, sticky="e")
    min_var = tk.StringVar(value="50"); max_var = tk.StringVar(value="50000")
    tk.Entry(size_frame, textvariable=min_var, width=10).grid(row=0, column=1, padx=4, pady=2)
    tk.Entry(size_frame, textvariable=max_var, width=10).grid(row=0, column=3, padx=4, pady=2)

    size_levels_var = tk.StringVar(value="(none)")
    tk.Label(size_frame, text="Matching levels:").grid(row=1, column=0, padx=4, pady=2, sticky="e")
    tk.Label(size_frame, textvariable=size_levels_var, anchor="w").grid(row=1, column=1, columnspan=3, padx=4, pady=2, sticky="w")

    def _suggest_levels():
        try:
            min_m = float(min_var.get()); max_m = float(max_var.get())
            if min_m <= 0 or max_m <= 0 or max_m < min_m:
                raise ValueError
        except Exception:
            log_to_gui("Enter valid positive meter values (min <= max).", "WARN")
            return
        min_km, max_km = min_m/1000.0, max_m/1000.0
        levels = suggest_h3_levels_by_size(min_km, max_km)
        size_levels_var.set(format_level_size_list(levels))

        def _generate_size_based():
            if not levels:
                log_to_gui("No suggested levels to generate.", "WARN")
                return
            mp.get_start_method(allow_none=True)
            _run_in_thread(write_h3_levels, base, levels)

        gen_btn.config(command=_generate_size_based, state=("normal" if levels else "disabled"))
        log_to_gui(f"Suggested H3 levels: {levels}" if levels else "No H3 levels for that size range.", "INFO")

    sugg_btn = (ttk.Button(size_frame, text="Suggest H3", width=16, bootstyle=PRIMARY, command=_suggest_levels)
                if ttk else tk.Button(size_frame, text="Suggest H3", width=16, command=_suggest_levels))
    gen_btn = (ttk.Button(size_frame, text="Generate H3", width=16, bootstyle=PRIMARY, state="disabled")
               if ttk else tk.Button(size_frame, text="Generate H3", width=16, state="disabled"))
    sugg_btn.grid(row=2, column=2, padx=4, pady=4, sticky="e")
    gen_btn.grid(row=2, column=3, padx=4, pady=4, sticky="w")

    if h3 is None:
        size_levels_var.set("H3 library missing (pip install h3)")
        try:
            gen_btn.config(state="disabled")
            sugg_btn.config(state="disabled")
        except Exception:
            pass


    mosaic_frame = tk.LabelFrame(root, text="Basic mosaic")
    mosaic_frame.pack(padx=10, pady=(0,6), fill=tk.X)
    global mosaic_status_var
    mosaic_status_var = tk.StringVar(value="")
    tk.Label(mosaic_frame, text="Status:").grid(row=0, column=0, padx=4, pady=2, sticky="w")
    status_label = tk.Label(mosaic_frame, textvariable=mosaic_status_var, width=18, anchor="w")
    status_label.grid(row=0, column=1, padx=(0,10), pady=2, sticky="w")
    mosaic_frame.grid_columnconfigure(1, weight=1)

    def _update_mosaic_status():
        exists = mosaic_exists(base)
        if mosaic_status_var.get() not in ("Running…","Completed","No faces"):
            mosaic_status_var.set("OK" if exists else "REQUIRED")
        color = "#55aa55" if exists else "#cc5555"
        try: status_label.config(fg=color)
        except Exception: pass

    def _run_mosaic_inline():
        try: buf = float(cfg["DEFAULT"].get("mosaic_buffer_m","25"))
        except Exception: buf = 25.0
        try: grid = float(cfg["DEFAULT"].get("mosaic_grid_size_m","1000"))
        except Exception: grid = 1000.0
        mosaic_status_var.set("Running…")
        def _after(success):
            def _ui():
                _update_mosaic_status()
                mosaic_status_var.set("Completed" if success else "No faces")
            try: root.after(100, _ui)
            except Exception: pass
        mp.get_start_method(allow_none=True)
        import threading
        threading.Thread(target=run_mosaic, args=(base, buf, grid, _after), daemon=True).start()

    start_btn = (ttk.Button(mosaic_frame, text="Build mosaic", width=16, bootstyle=PRIMARY, command=_run_mosaic_inline)
                 if ttk else tk.Button(mosaic_frame, text="Build mosaic", width=16, command=_run_mosaic_inline))
    start_btn.grid(row=0, column=2, padx=4, pady=2, sticky="e")

    exit_frame = tk.Frame(root); exit_frame.pack(fill=tk.X, pady=6, padx=10)
    (ttk.Button(exit_frame, text="Exit", bootstyle=WARNING, command=root.destroy) if ttk
     else tk.Button(exit_frame, text="Exit", command=root.destroy)).pack(side=tk.RIGHT)

    _update_mosaic_status()
    root.mainloop()

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Create geocodes (H3/Mosaic)")
    parser.add_argument("--nogui", action="store_true", help="Run in CLI mode")
    parser.add_argument("--original_working_directory", required=False, help="Path to running folder")

    parser.add_argument("--h3", action="store_true", help="Generate H3 by range")
    parser.add_argument("--h3-from", dest="h3_from", type=int, default=3)
    parser.add_argument("--h3-to", dest="h3_to", type=int, default=6)
    parser.add_argument("--h3-levels", dest="h3_levels", type=str, default="", help="Comma-separated list, e.g. 5,6,7")

    parser.add_argument("--mosaic", action="store_true", help="Generate basic mosaic and publish as geocode")
    parser.add_argument("--buffer-m", dest="buffer_m", type=float, default=25.0)
    parser.add_argument("--grid-size-m", dest="grid_size_m", type=float, default=1000.0)

    args = parser.parse_args()
    base = resolve_base_dir(args.original_working_directory)
    cfg = read_config(base / "system" / "config.ini")

    global original_working_directory
    original_working_directory = str(base)

    # heartbeat secs
    try:
        global HEARTBEAT_SECS
        HEARTBEAT_SECS = int(cfg["DEFAULT"].get("heartbeat_secs", str(HEARTBEAT_SECS)))
    except Exception:
        pass

    if args.nogui:
        if args.h3_levels:
            levels = [int(x.strip()) for x in args.h3_levels.split(",") if x.strip().isdigit()]
            write_h3_levels(base, levels)
        elif args.h3:
            write_h3_levels(base, list(range(args.h3_from, args.h3_to + 1)))
        elif args.mosaic:
            run_mosaic(base, args.buffer_m, args.grid_size_m)
        else:
            log_to_gui("Nothing to do. Use --h3, --h3-levels, or --mosaic.", "WARN")
    else:
        build_gui(base, cfg)

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=False)  # Windows-safe
    except RuntimeError:
        pass
    main()
