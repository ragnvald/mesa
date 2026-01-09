#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_create_geocodes.py  —  H3 + Basic Mosaic (GeoParquet-first, flat config)

What's new in this build (identity-preserving speedups):
- **Clip-before-buffer** per tile: each worker clips inputs to the tile’s envelope
  expanded by (buffer_m + epsilon) before buffering. This is mathematically equivalent
  inside the tile (no geometry change) and cuts vertex counts massively.
- Optional dedup of identical input geometries (safe; does not change mosaic).
- Kept: adaptive quadtree tiling + heavy-tile split, streaming face flush, robust H3, GUI/CLI.
- Heartbeat still shows: stage, tiles done/total, faces count, elapsed, ETA, tiles/min,
  and total CPU/RAM aggregated across pool children (if psutil available).
- Task scheduling defaults to INTERLEAVE for snappy early progress.

Config (<base>/config.ini → [DEFAULT]) knobs (defaults shown):
  mosaic_workers = 0                         # 0 -> auto
  mosaic_quadtree_max_feats_per_tile = 800
  mosaic_quadtree_heavy_split_multiplier = 2.0
  mosaic_quadtree_max_depth = 8
  mosaic_quadtree_min_tile_m = 1000
  mosaic_simplify_tolerance_m = 0           # keep 0 for identical output
  mosaic_faces_flush_batch = 250000
  mosaic_pool_maxtasksperchild = 8
  mosaic_pool_chunksize = 1
  mosaic_task_order = interleave            # interleave | heavy_first | light_first
  heartbeat_secs = 10

    # Auto worker sizing (used when mosaic_workers <= 0)
    mosaic_auto_worker_fraction = 0.75        # fraction of detected CPUs
    mosaic_auto_worker_min = 1
    mosaic_auto_worker_max = 0                # 0 -> no upper bound
    mosaic_auto_worker_mem_gb = 1.5           # approx GB per worker before capping

  # New identity-preserving speed knobs:
  mosaic_clip_before_buffer = true
  mosaic_clip_margin_m = 0.05               # tiny epsilon on top of buffer distance
  mosaic_dedup_assets = true                # drop exact duplicate geometries up-front

  # I/O:
  parquet_folder = output/geoparquet        # optional override for GeoParquet dir
"""

from __future__ import annotations

import argparse
import configparser
import datetime
import locale
import os
import sys
import time
import shutil
import multiprocessing as mp
import traceback
from pathlib import Path
from typing import Union, Optional, List, Tuple

import numpy as np
import geopandas as gpd
import pandas as pd
import uuid
from shapely.geometry import (
    Polygon, MultiPolygon, GeometryCollection, LineString, MultiLineString, box
)
from shapely.geometry import mapping as shp_mapping
from shapely.ops import unary_union, polygonize
from shapely import wkb as shp_wkb
from shapely.prepared import prep

try:
    import fiona
except Exception:
    fiona = None

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


def _progress_lerp(floor_v: float, ceil_v: float, frac: float) -> float:
    try:
        f = float(frac)
    except Exception:
        f = 0.0
    f = max(0.0, min(1.0, f))
    return float(floor_v) + (float(ceil_v) - float(floor_v)) * f


def _progress_saturating(floor_v: float, ceil_v: float, n_done: int, half_n: int) -> float:
    """Monotone progress that approaches ceil_v as n_done grows.

    Useful when total work is unknown (e.g., polygonize output size).
    """
    try:
        done = max(0.0, float(n_done))
        half = max(1.0, float(half_n))
        frac = done / (done + half)
        return _progress_lerp(floor_v, ceil_v, frac)
    except Exception:
        return float(floor_v)

# -----------------------------------------------------------------------------
# Config & paths (flat config)
# -----------------------------------------------------------------------------
# Global Parquet subdir (overridden in main from config.ini if provided)
_PARQUET_SUBDIR = "output/geoparquet"
_PARQUET_OVERRIDE: Path | None = None

def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

def _has_config_at(root: Path) -> bool:
    # Flat layout: config.ini lives at project root
    return _exists(root / "config.ini")

def find_base_dir(cli_workdir: str | None = None) -> Path:
    """Choose a canonical project base folder that contains config.ini.
    Priority order:
      1) env MESA_BASE_DIR (honored immediately when valid)
      2) --original_working_directory (CLI)
      3) running binary/interpreter folder (and parents)
      4) this script's folder and parents (covers PyInstaller _MEIPASS)
      5) CWD, CWD/code, and their parents
    """
    def _maybe_return(path_like: Path) -> Path | None:
        try:
            resolved = Path(path_like).resolve()
        except Exception:
            resolved = Path(path_like)
        return resolved if _has_config_at(resolved) else None

    env_base = os.environ.get("MESA_BASE_DIR")
    if env_base:
        env_hit = _maybe_return(env_base)
        if env_hit:
            return env_hit

    if cli_workdir:
        cli_hit = _maybe_return(cli_workdir)
        if cli_hit:
            return cli_hit

    candidates: list[Path] = []
    if env_base:
        candidates.append(Path(env_base))
    if cli_workdir:
        candidates.append(Path(cli_workdir))

    exe_path: Path | None = None
    try:
        exe_path = Path(sys.executable).resolve()
    except Exception:
        exe_path = None
    if exe_path:
        candidates += [exe_path.parent, exe_path.parent.parent, exe_path.parent.parent.parent]

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass))

    here = Path(__file__).resolve()
    candidates += [here.parent, here.parent.parent, here.parent.parent.parent]

    cwd = Path.cwd()
    candidates += [cwd, cwd / "code", cwd.parent, cwd.parent / "code"]

    seen = set()
    uniq = []
    for c in candidates:
        try:
            r = c.resolve()
        except Exception:
            r = c
        if r not in seen:
            seen.add(r)
            uniq.append(r)

    preferred = None
    fallback = None
    for c in uniq:
        if _has_config_at(c):
            if fallback is None:
                fallback = c
            if c.name.lower() not in {"code", "system"}:
                preferred = c
                break

    if preferred:
        return preferred
    if fallback:
        return fallback

    if here.parent.name.lower() == "system":
        return here.parent.parent
    if exe_path:
        return exe_path.parent
    if env_base:
        return Path(env_base)
    return here.parent

def read_config(file_name: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(file_name, encoding="utf-8")
    if "DEFAULT" not in cfg:
        cfg["DEFAULT"] = {}
    return cfg

def _safe_rmtree(p: Path):
    """Best-effort recursive delete; ignore if already gone or busy."""
    try:
        if p and p.exists():
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass

def resolve_base_dir(cli_root: str | None) -> Path:
    return find_base_dir(cli_root)

def config_path(base_dir: Path) -> Path:
    # FLAT: <base>/config.ini
    return base_dir / "config.ini"

def _primary_geoparquet_dir(base_dir: Path) -> Path:
    sub = Path(_PARQUET_SUBDIR)
    if sub.is_absolute():
        return sub.resolve()
    return (base_dir / sub).resolve()

def gpq_dir(base_dir: Path) -> Path:
    global _PARQUET_OVERRIDE
    if _PARQUET_OVERRIDE is None:
        _PARQUET_OVERRIDE = _primary_geoparquet_dir(base_dir)
    _PARQUET_OVERRIDE.mkdir(parents=True, exist_ok=True)
    return _PARQUET_OVERRIDE

def _existing_parquet_path(base_dir: Path, name: str) -> Path | None:
    primary = (_primary_geoparquet_dir(base_dir) / f"{name}.parquet").resolve()
    if primary.exists():
        return primary

    rel = Path(_PARQUET_SUBDIR)
    if not rel.is_absolute() and base_dir.name.lower() != "code":
        code_dir = (base_dir / "code" / rel).resolve()
        alt = code_dir / f"{name}.parquet"
        if alt.exists():
            log_to_gui(f"Using fallback GeoParquet copy for {name}: {alt}", "WARN")
            return alt
    return None

def geoparquet_path(base_dir: Path, name: str) -> Path:
    existing = _existing_parquet_path(base_dir, name)
    if existing is not None:
        return existing
    target = gpq_dir(base_dir) / f"{name}.parquet"
    return target

def _auto_worker_count(cfg: configparser.ConfigParser | None) -> tuple[int, str]:
    defaults = cfg["DEFAULT"] if (cfg is not None and "DEFAULT" in cfg) else {}

    def _get_float(key: str, fallback: float) -> float:
        try:
            return float(defaults.get(key, str(fallback)))
        except Exception:
            return fallback

    def _get_int(key: str, fallback: int) -> int:
        try:
            return int(defaults.get(key, str(fallback)))
        except Exception:
            return fallback

    try:
        cpu_total = max(1, mp.cpu_count())
    except Exception:
        cpu_total = 1

    frac = min(1.0, max(0.1, _get_float("mosaic_auto_worker_fraction", 0.75)))
    min_workers = max(1, _get_int("mosaic_auto_worker_min", 1))
    max_workers = max(0, _get_int("mosaic_auto_worker_max", 0))
    approx_mem_gb = max(0.1, _get_float("mosaic_auto_worker_mem_gb", 1.5))

    cpu_based = max(1, int(cpu_total * frac))
    target = max(min_workers, cpu_based)

    reason_bits = [f"cpu={cpu_total}", f"fraction={frac:.2f}"]
    if min_workers > 1:
        reason_bits.append(f"min={min_workers}")
    if max_workers > 0:
        target = min(target, max_workers)
        reason_bits.append(f"max={max_workers}")

    avail_gb = None
    mem_cap = 0
    if psutil:
        try:
            avail_gb = psutil.virtual_memory().available / (1024 ** 3)
            mem_cap = max(1, int(avail_gb // approx_mem_gb))
        except Exception:
            avail_gb = None
            mem_cap = 0
    if mem_cap:
        if target > mem_cap:
            reason_bits.append(f"mem_cap={mem_cap}")
        target = min(target, mem_cap)
        if avail_gb is not None:
            reason_bits.append(f"mem≈{avail_gb:.1f}GB/{approx_mem_gb:.1f}GB per worker")

    return max(1, target), ", ".join(reason_bits)

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


# -----------------------------------------------------------------------------
# Import geocodes from vector files (moved from data_import.py)
# -----------------------------------------------------------------------------
def _rglob_many(folder: Path, patterns: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for pat in patterns:
        files.extend(folder.rglob(pat))
    return files


def _scan_for_files(label: str, folder: Path, patterns: tuple[str, ...]) -> list[Path]:
    if not folder.exists():
        log_to_gui(f"{label} folder does not exist: {folder}", "WARN")
        return []
    pat_list = ", ".join(patterns)
    log_to_gui(f"{label}: scanning {folder} for {pat_list} …")
    t0 = time.time()
    files = _rglob_many(folder, patterns)
    log_to_gui(f"{label}: scan finished in {time.time() - t0:.1f}s → {len(files)} file(s).")
    return files


def _read_and_reproject_vector(filepath: Path, layer: str | None, working_epsg: int) -> gpd.GeoDataFrame:
    try:
        data = gpd.read_file(filepath, layer=layer) if layer else gpd.read_file(filepath)
        if data.crs is None:
            log_to_gui(f"No CRS in {filepath.name} (layer={layer}); set EPSG:{working_epsg}", "WARN")
            data.set_crs(epsg=working_epsg, inplace=True)
        elif (data.crs.to_epsg() or working_epsg) != int(working_epsg):
            data = data.to_crs(epsg=int(working_epsg))
        if data.geometry.name != "geometry":
            data = data.set_geometry(data.geometry.name).rename_geometry("geometry")
        return data
    except Exception as e:
        log_to_gui(f"Read fail {filepath} (layer={layer}): {e}", "ERROR")
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{working_epsg}")


def _read_parquet_vector(fp: Path, working_epsg: int) -> gpd.GeoDataFrame:
    try:
        gdf = gpd.read_parquet(fp)
        if gdf.crs is None:
            gdf.set_crs(epsg=working_epsg, inplace=True)
        elif (gdf.crs.to_epsg() or working_epsg) != int(working_epsg):
            gdf = gdf.to_crs(epsg=int(working_epsg))
        if gdf.geometry.name != "geometry":
            gdf = gdf.set_geometry(gdf.geometry.name).rename_geometry("geometry")
        return gdf
    except Exception as e:
        log_to_gui(f"Read fail (parquet) {fp.name}: {e}", "ERROR")
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{working_epsg}")


def _ensure_unique_geocode_codes(rows: list[dict]) -> None:
    counts: dict[str | None, int] = {}
    for r in rows:
        v = r.get("code")
        code = None if pd.isna(v) else str(v)
        r["code"] = code
        counts[code] = counts.get(code, 0) + 1
    for r in rows:
        code = r.get("code")
        if counts.get(code, 0) > 1:
            new_code = f"{code}_{uuid.uuid4()}"
            log_to_gui(f"Duplicate geocode '{code}' → '{new_code}'")
            r["code"] = new_code


def import_geocodes_from_folder(input_folder: Path, working_epsg: int) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    groups: list[dict] = []
    objects: list[dict] = []
    group_id = 1
    object_id = 1

    patterns = ("*.shp", "*.gpkg", "*.parquet")
    files = _scan_for_files("Geocodes", input_folder, patterns)
    log_to_gui(f"Geocode files found: {len(files)}")
    update_progress(2)

    def _process_layer(gdf: gpd.GeoDataFrame, layer_name: str):
        nonlocal group_id, object_id
        if gdf.empty:
            log_to_gui(f"Empty geocode layer: {layer_name}", "WARN")
            return

        bbox_polygon = box(*gdf.total_bounds)
        name_gis_geocodegroup = f"geocode_{group_id:03d}"
        groups.append({
            "id": group_id,
            "name": layer_name,
            "name_gis_geocodegroup": name_gis_geocodegroup,
            "title_user": layer_name,
            "description": "",
            "geometry": bbox_polygon,
        })

        for _, row in gdf.iterrows():
            code = None
            if "qdgc" in gdf.columns and pd.notna(row.get("qdgc")):
                code = str(row.get("qdgc"))
            elif "code" in gdf.columns and pd.notna(row.get("code")):
                code = str(row.get("code"))
            else:
                code = str(object_id)

            attrs = "; ".join(
                [f"{c}: {row[c]}" for c in gdf.columns if c != gdf.geometry.name and c != "geometry"]
            )
            objects.append({
                "code": code,
                "ref_geocodegroup": group_id,
                "name_gis_geocodegroup": name_gis_geocodegroup,
                "attributes": attrs,
                "geometry": row.geometry,
            })
            object_id += 1

        group_id += 1

    for i, fp in enumerate(files, start=1):
        update_progress(5 + 85 * (i / max(1, len(files))))
        if i == 1 or i % 20 == 0:
            log_to_gui(f"Geocodes: processing {fp.name} ({i}/{max(1, len(files))})")

        if fp.suffix.lower() == ".gpkg":
            if fiona is None:
                log_to_gui("Cannot list layers: fiona not available. Install fiona to import .gpkg.", "ERROR")
                continue
            try:
                for layer in fiona.listlayers(fp):
                    gdf = _read_and_reproject_vector(fp, layer, working_epsg)
                    _process_layer(gdf, layer)
            except Exception as e:
                log_to_gui(f"GPKG error {fp}: {e}", "ERROR")
        else:
            layer = fp.stem
            gdf = _read_parquet_vector(fp, working_epsg) if fp.suffix.lower() == ".parquet" else _read_and_reproject_vector(fp, None, working_epsg)
            _process_layer(gdf, layer)

    _ensure_unique_geocode_codes(objects)

    crs = f"EPSG:{working_epsg}"
    groups_gdf = gpd.GeoDataFrame(pd.DataFrame(groups), geometry="geometry", crs=crs) if groups else gpd.GeoDataFrame(geometry=[], crs=crs)
    objects_gdf = gpd.GeoDataFrame(pd.DataFrame(objects), geometry="geometry", crs=crs) if objects else gpd.GeoDataFrame(geometry=[], crs=crs)
    log_to_gui(f"Geocodes: groups={len(groups_gdf)}, objects={len(objects_gdf)}")
    return groups_gdf, objects_gdf


def run_import_geocodes(base_dir: Path, cfg: configparser.ConfigParser):
    update_progress(0)
    log_to_gui("Step [Import geocodes] STARTED")
    try:
        working_epsg = int(str(cfg["DEFAULT"].get("workingprojection_epsg", "4326")))
    except Exception:
        working_epsg = 4326

    inp = cfg["DEFAULT"].get("input_folder_geocode", "input/geocode")
    input_folder = Path(inp)
    if not input_folder.is_absolute():
        input_folder = (base_dir / input_folder).resolve()

    try:
        log_to_gui(f"Importing geocodes from: {input_folder}")
        g_grp, g_obj = import_geocodes_from_folder(input_folder, working_epsg)
        out_dir = gpq_dir(base_dir)
        ensure_wgs84(g_grp).to_parquet(out_dir / "tbl_geocode_group.parquet", index=False)
        ensure_wgs84(g_obj).to_parquet(out_dir / "tbl_geocode_object.parquet", index=False)
        log_to_gui(f"Saved tbl_geocode_* → {out_dir}")
        log_to_gui("Step [Import geocodes] COMPLETED")
    except Exception as e:
        log_to_gui(f"Step [Import geocodes] FAILED: {e}", "ERROR")
        raise
    finally:
        update_progress(100)

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


# -----------------------------------------------------------------------------
# Mosaic (basic_mosaic) — linework-based (no tiling)
# -----------------------------------------------------------------------------
def _cfg_float(cfg: configparser.ConfigParser, key: str, default: float) -> float:
    try:
        raw = (cfg["DEFAULT"].get(key, str(default)) if cfg is not None and "DEFAULT" in cfg else str(default))
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _cfg_int(cfg: configparser.ConfigParser, key: str, default: int) -> int:
    try:
        raw = (cfg["DEFAULT"].get(key, str(default)) if cfg is not None and "DEFAULT" in cfg else str(default))
        return int(float(str(raw).strip()))
    except Exception:
        return int(default)


def _geom_to_polygonal_metric(geom, *, line_buf_m: float, point_buf_m: float):
    """Convert a metric-CRS geometry into polygonal geometry.

    Constraints:
    - No buffering is applied to polygon inputs.
    - Line/point assets are buffered using config defaults so they can participate as polygons.
    """
    if geom is None:
        return None
    try:
        if geom.is_empty:
            return None
    except Exception:
        return None

    gt = getattr(geom, "geom_type", "")
    if gt in ("Polygon", "MultiPolygon"):
        return geom
    if gt in ("LineString", "MultiLineString"):
        try:
            return geom.buffer(max(0.0, float(line_buf_m)))
        except Exception:
            return None
    if gt in ("Point", "MultiPoint"):
        try:
            return geom.buffer(max(0.0, float(point_buf_m)))
        except Exception:
            return None
    if gt == "GeometryCollection":
        polys = []
        try:
            for g in getattr(geom, "geoms", []):
                pg = _geom_to_polygonal_metric(g, line_buf_m=line_buf_m, point_buf_m=point_buf_m)
                if pg is None:
                    continue
                ex = _extract_polygonal(pg)
                if ex is not None and not ex.is_empty:
                    polys.append(ex)
        except Exception:
            return None
        if not polys:
            return None
        try:
            return unary_union(polys)
        except Exception:
            return polys[0]
    return None


def _iter_boundary_lines(polyish):
    """Yield LineString boundary parts from a polygonal geometry."""
    if polyish is None:
        return
    try:
        if polyish.is_empty:
            return
    except Exception:
        return

    try:
        b = polyish.boundary
    except Exception:
        return
    if b is None:
        return
    try:
        if b.is_empty:
            return
    except Exception:
        return

    gt = getattr(b, "geom_type", "")
    if gt == "LineString":
        yield b
        return
    if gt == "MultiLineString":
        try:
            for seg in b.geoms:
                if seg is not None and not seg.is_empty:
                    yield seg
        except Exception:
            return


def _unary_union_safe(geoms: list, *, label: str, min_step: int = 200):
    """Attempt unary_union; on memory/GEOS errors, retry with smaller chunks."""
    if not geoms:
        return None
    try:
        return unary_union(geoms)
    except MemoryError:
        log_to_gui(f"[Mosaic] MemoryError during unary_union ({label}) for n={len(geoms):,}; retrying smaller chunks…", "WARN")
    except Exception as e:
        log_to_gui(f"[Mosaic] unary_union failed ({label}) for n={len(geoms):,}: {e}; retrying smaller chunks…", "WARN")

    step = max(min_step, len(geoms) // 4)
    out = None
    while step >= min_step:
        ok = True
        out = None
        for i in range(0, len(geoms), step):
            chunk = geoms[i:i+step]
            try:
                cu = unary_union(chunk)
            except Exception:
                ok = False
                break
            out = cu if out is None else unary_union([out, cu])
        if ok:
            return out
        step = max(min_step, step // 2)
    return None


def _tree_reduce_unions(
    unions: list,
    *,
    max_partials: int,
    label: str = "unions",
    heartbeat_s: float = 10.0,
) -> list:
    """Repeatedly merge unions in a tree-like pairwise fashion.

    Note: GEOS unary_union is typically single-threaded, so this stage can look
    like "only one core" in Task Manager. We emit throttled progress logs so
    long merges don't appear stalled.
    """
    max_partials = max(2, int(max_partials))
    label = (label or "unions").strip()
    hb = max(1.0, float(heartbeat_s or 10.0))

    started = time.time()
    last_log = started
    round_idx = 0

    while len(unions) > max_partials:
        round_idx += 1
        n_in = len(unions)
        merges_total = (n_in + 1) // 2

        now = time.time()
        if now - last_log >= hb:
            elapsed = now - started
            log_to_gui(
                f"[Mosaic] Reducing {label}: round {round_idx} starting (n={n_in:,} -> <= {max_partials:,}); merges={merges_total:,}; elapsed {elapsed:.0f}s…",
                "INFO",
            )
            last_log = now

        merged = []
        it = iter(unions)
        merge_i = 0
        for a in it:
            b = next(it, None)
            merge_i += 1
            if b is None:
                merged.append(a)
            else:
                u = _unary_union_safe([a, b], label=f"partial_merge:{label}", min_step=2)
                merged.append(u if u is not None else a)

            now = time.time()
            if now - last_log >= hb:
                elapsed = now - started
                pct = (merge_i / max(1, merges_total)) * 100.0
                log_to_gui(
                    f"[Mosaic] Reducing {label}: round {round_idx} merge {merge_i:,}/{merges_total:,} ({pct:.0f}%) • elapsed {elapsed:.0f}s",
                    "INFO",
                )
                last_log = now

        unions = merged
        now = time.time()
        if now - last_log >= hb:
            elapsed = now - started
            log_to_gui(
                f"[Mosaic] Reducing {label}: round {round_idx} complete; n={len(unions):,}; elapsed {elapsed:.0f}s",
                "INFO",
            )
            last_log = now

    return unions


def _mosaic_extract_chunk_worker(args: tuple[list[bytes], float, float]) -> dict:
    """Worker: extract boundary linework + coverage polygons for a chunk.

    Returns WKB for a locally-unioned edge net and coverage polygon union to keep
    IPC payloads small.
    """
    (wkbs, line_buf_m, point_buf_m) = args
    boundary_parts = 0
    skipped = 0
    union_retries = 0
    boundary_lines = []
    coverage_polys = []

    # IMPORTANT: Do not call log_to_gui() from worker processes.
    # Return lightweight log messages to the parent, which can safely forward
    # them to the GUI and/or log.txt.
    logs: list[tuple[str, str]] = []  # (level, message)

    def _union_local(geoms: list):
        if not geoms:
            return None
        if len(geoms) == 1:
            return geoms[0]
        try:
            return unary_union(geoms)
        except Exception:
            # Retry in smaller pieces; avoid logging from workers.
            nonlocal union_retries
            union_retries += 1
            step = max(50, len(geoms) // 4)
            out = None
            while step >= 50:
                ok = True
                out = None
                for i in range(0, len(geoms), step):
                    chunk = geoms[i:i + step]
                    try:
                        cu = unary_union(chunk)
                    except Exception:
                        ok = False
                        break
                    out = cu if out is None else unary_union([out, cu])
                if ok:
                    return out
                step = max(50, step // 2)
            return None

    try:
        for b in wkbs:
            if not b:
                skipped += 1
                continue
            try:
                geom = shp_wkb.loads(b)
            except Exception:
                skipped += 1
                continue

            polyish = _geom_to_polygonal_metric(geom, line_buf_m=line_buf_m, point_buf_m=point_buf_m)
            if polyish is None:
                skipped += 1
                continue
            polyish = _fix_valid(polyish)
            ex = _extract_polygonal(polyish)
            if ex is None or getattr(ex, "is_empty", False):
                skipped += 1
                continue

            coverage_polys.append(ex)
            for seg in _iter_boundary_lines(ex):
                boundary_lines.append(seg)
                boundary_parts += 1

        edge_u = _union_local(boundary_lines)
        cov_u = _union_local(coverage_polys)

        if union_retries:
            # Keep this quiet: only surface when it happens.
            logs.append(("WARN", f"Worker union retries={union_retries} (boundary_parts={boundary_parts:,}, skipped={skipped:,})."))

        return {
            "boundary_parts": int(boundary_parts),
            "skipped": int(skipped),
            "edge_wkb": shp_wkb.dumps(edge_u) if edge_u is not None else b"",
            "cov_wkb": shp_wkb.dumps(cov_u) if cov_u is not None else b"",
            "logs": logs,
            "err": None,
        }
    except Exception as e:
        try:
            tb = traceback.format_exc(limit=5)
            logs.append(("WARN", f"Worker exception traceback (truncated):\n{tb}"))
        except Exception:
            pass
        return {
            "boundary_parts": int(boundary_parts),
            "skipped": int(skipped),
            "edge_wkb": b"",
            "cov_wkb": b"",
            "logs": logs,
            "err": f"{type(e).__name__}: {e}",
        }


def _build_linework_and_coverage(
    a_metric: gpd.GeoDataFrame,
    *,
    cfg: configparser.ConfigParser,
    workers: int | None = None,
    progress_floor: float | None = None,
    progress_ceiling: float | None = None,
) -> tuple[object | None, object | None, dict]:
    """Build noded linework and polygon coverage union.

    - No tiling, no clipping.
    - Polygons contribute boundaries without buffering.
    - Lines/points are buffered using default_*_buffer_m so they contribute as polygons.
    - Memory safety: batched unions + tree reduction.
    """
    line_buf_m = _cfg_float(cfg, "default_line_buffer_m", 10.0)
    point_buf_m = _cfg_float(cfg, "default_point_buffer_m", 10.0)
    batch_size = max(200, _cfg_int(cfg, "mosaic_line_union_batch", 4000))
    max_partials = max(2, _cfg_int(cfg, "mosaic_line_union_max_partials", 16))
    cov_batch_size = max(50, _cfg_int(cfg, "mosaic_coverage_union_batch", 500))

    stats = {
        "assets": int(len(a_metric)),
        "line_buf_m": float(line_buf_m),
        "point_buf_m": float(point_buf_m),
        "batch_size": int(batch_size),
        "max_partials": int(max_partials),
        "boundary_parts": 0,
        "union_batches": 0,
        "skipped": 0,
    }

    boundary_batch: list = []
    line_partials: list = []
    cov_batch: list = []
    cov_partials: list = []

    def flush_boundary():
        if not boundary_batch:
            return
        u = _unary_union_safe(boundary_batch, label="boundary_batch")
        if u is not None:
            line_partials.append(u)
            stats["union_batches"] += 1
            if len(line_partials) > max_partials:
                line_partials[:] = _tree_reduce_unions(line_partials, max_partials=max_partials, label="edges")
        boundary_batch.clear()

    def flush_coverage():
        if not cov_batch:
            return
        u = _unary_union_safe(cov_batch, label="coverage_batch")
        if u is not None:
            cov_partials.append(u)
            if len(cov_partials) > max_partials:
                cov_partials[:] = _tree_reduce_unions(cov_partials, max_partials=max_partials, label="coverage")
        cov_batch.clear()

    # Prefer parallel extraction/union when workers>1. This stage is the bottleneck.
    use_parallel = False
    try:
        v = str(cfg["DEFAULT"].get("mosaic_parallel_extract", "true")).strip().lower()
        use_parallel = v in ("1", "true", "yes", "on")
    except Exception:
        use_parallel = True
    if workers is None:
        workers = 1
    if workers <= 1:
        use_parallel = False

    if use_parallel:
        # Chunk size: tradeoff between overhead and per-worker memory.
        chunk_size = max(200, _cfg_int(cfg, "mosaic_extract_chunk_size", 2500))
        # Backward compatibility: earlier docs used mosaic_pool_* keys.
        maxtasks = max(1, _cfg_int(cfg, "mosaic_extract_maxtasksperchild", _cfg_int(cfg, "mosaic_pool_maxtasksperchild", 4)))
        pool_chunksize = max(1, _cfg_int(cfg, "mosaic_extract_pool_chunksize", _cfg_int(cfg, "mosaic_pool_chunksize", 1)))

        # Serialize geometries to WKB once in the parent to avoid GeoPandas objects
        # crossing processes.
        wkbs: list[bytes] = []
        for geom in a_metric.geometry:
            try:
                wkbs.append(shp_wkb.dumps(geom) if geom is not None else b"")
            except Exception:
                wkbs.append(b"")

        chunks = [wkbs[i:i + chunk_size] for i in range(0, len(wkbs), chunk_size)]

        # Order tasks to reduce tail-end stragglers (which otherwise makes CPU
        # look "idle" near the end when only a few heavy chunks remain).
        try:
            order_mode = str(cfg["DEFAULT"].get("mosaic_task_order", "interleave")).strip().lower()
        except Exception:
            order_mode = "interleave"
        if order_mode not in ("interleave", "heavy_first", "light_first"):
            order_mode = "interleave"
        try:
            weights = [sum((len(b) for b in c if b), 0) for c in chunks]
            order = _order_tasks(weights, order_mode)
            chunks = [chunks[i] for i in order]
        except Exception:
            order_mode = "(default)"

        # Hook heartbeat/progress to real work units.
        STATS.tiles_total = int(len(chunks))
        STATS.tiles_done = 0
        STATS.worker_started_at = time.time()
        STATS.detail = "extracting boundaries"

        # Most user-visible waiting is here. Let extraction own most of the linework band.
        if progress_floor is not None and progress_ceiling is not None:
            extract_floor = float(progress_floor)
            extract_ceiling = _progress_lerp(progress_floor, progress_ceiling, 0.90)
        else:
            extract_floor = extract_ceiling = None

        log_to_gui(
            f"[Mosaic] Parallel boundary extraction: workers={workers}, chunks={len(chunks):,}, chunk_size={chunk_size:,}, maxtasksperchild={maxtasks}, task_order={order_mode}",
            "INFO",
        )

        ctx = mp.get_context("spawn")
        done = 0
        last_ui = 0.0
        last_progress_ts = time.time()
        last_wait_log_ts = time.time()
        try:
            with ctx.Pool(processes=int(workers), maxtasksperchild=int(maxtasks)) as pool:
                it = pool.imap_unordered(
                    _mosaic_extract_chunk_worker,
                    [(c, float(line_buf_m), float(point_buf_m)) for c in chunks],
                    chunksize=int(pool_chunksize),
                )
                # Use iterator timeouts so we can log “still working” heartbeats
                # during long tails (few heavy chunks) or when a worker hangs.
                hb_secs = None
                try:
                    hb_secs = float(cfg["DEFAULT"].get("heartbeat_secs", "10"))
                except Exception:
                    hb_secs = None
                next_timeout = max(5.0, min(120.0, float(hb_secs or 10.0)))

                while done < len(chunks):
                    try:
                        # multiprocessing.pool.IMapIterator supports next(timeout=...).
                        res = it.next(timeout=next_timeout)  # type: ignore[attr-defined]
                    except Exception as e:
                        # Most commonly multiprocessing.TimeoutError.
                        if type(e).__name__ == "TimeoutError":
                            now = time.time()
                            if (now - last_wait_log_ts) >= next_timeout:
                                last_wait_log_ts = now
                                alive = None
                                try:
                                    alive = sum(1 for p in getattr(pool, "_pool", []) if p is not None and p.is_alive())
                                except Exception:
                                    alive = None
                                stalled_s = now - last_progress_ts
                                msg = (
                                    f"[Mosaic] Still working… {done:,}/{len(chunks):,} chunks completed; "
                                    f"no completion for {stalled_s/60.0:.1f} min"
                                )
                                if alive is not None:
                                    msg += f"; workers alive={alive}/{int(workers)}"
                                log_to_gui(msg, "INFO")
                            continue
                        raise

                    done += 1
                    last_progress_ts = time.time()

                    STATS.tiles_done = int(done)
                    if extract_floor is not None and extract_ceiling is not None:
                        now = time.time()
                        if done <= 2 or done >= len(chunks) or (now - last_ui) >= 0.25:
                            last_ui = now
                            update_progress(_progress_lerp(extract_floor, extract_ceiling, done / max(1, len(chunks))))

                    # Forward worker logs in the parent process only (UI-safe).
                    wlogs = res.get("logs") or []
                    if isinstance(wlogs, list) and wlogs:
                        for item in wlogs[:8]:
                            try:
                                if isinstance(item, (tuple, list)) and len(item) == 2:
                                    lvl, msg = item
                                    log_to_gui(f"[Mosaic] {msg}", str(lvl) or "INFO")
                                else:
                                    log_to_gui(f"[Mosaic] {item}", "INFO")
                            except Exception:
                                pass

                    if res.get("err"):
                        log_to_gui(f"[Mosaic] Worker chunk failed: {res['err']}", "WARN")
                    stats["boundary_parts"] += int(res.get("boundary_parts", 0))
                    stats["skipped"] += int(res.get("skipped", 0))

                    eb = res.get("edge_wkb") or b""
                    cb = res.get("cov_wkb") or b""
                    if eb:
                        try:
                            line_partials.append(shp_wkb.loads(eb))
                            stats["union_batches"] += 1
                        except Exception:
                            pass
                    if cb:
                        try:
                            cov_partials.append(shp_wkb.loads(cb))
                        except Exception:
                            pass

                    if done % max(1, (len(chunks) // 20)) == 0:
                        log_to_gui(f"[Mosaic] Boundary extraction (parallel): {done:,}/{len(chunks):,} chunks…", "INFO")
        except Exception as e:
            # Fall back to the serial path (keeps correctness)
            log_to_gui(f"[Mosaic] Parallel extraction failed ({type(e).__name__}: {e}). Falling back to single-process.", "WARN")
            use_parallel = False

        # Reduce partials once, after all chunk results are collected.
        if use_parallel and line_partials:
            STATS.detail = "reducing unions"
            if progress_floor is not None and progress_ceiling is not None:
                update_progress(_progress_lerp(progress_floor, progress_ceiling, 0.93))
            log_to_gui(
                f"[Mosaic] Reducing partial unions: edges={len(line_partials):,}, coverage={len(cov_partials):,}…",
                "INFO",
            )
            if len(line_partials) > max_partials:
                line_partials[:] = _tree_reduce_unions(line_partials, max_partials=max_partials, label="edges")
            if cov_partials and len(cov_partials) > max_partials:
                cov_partials[:] = _tree_reduce_unions(cov_partials, max_partials=max_partials, label="coverage")
            if progress_floor is not None and progress_ceiling is not None:
                update_progress(_progress_lerp(progress_floor, progress_ceiling, 0.97))

    if not use_parallel:
        # Serial path: treat assets as work units.
        STATS.tiles_total = int(len(a_metric))
        STATS.tiles_done = 0
        STATS.worker_started_at = time.time()
        STATS.detail = "extracting boundaries"
        last_ui = 0.0

        if progress_floor is not None and progress_ceiling is not None:
            extract_floor = float(progress_floor)
            extract_ceiling = _progress_lerp(progress_floor, progress_ceiling, 0.90)
        else:
            extract_floor = extract_ceiling = None

        for i, geom in enumerate(a_metric.geometry, start=1):
            if i % 5000 == 0:
                log_to_gui(f"[Mosaic] Boundary extraction: {i:,}/{len(a_metric):,} assets…", "INFO")

            STATS.tiles_done = int(i)
            if extract_floor is not None and extract_ceiling is not None:
                now = time.time()
                if i <= 2 or i >= len(a_metric) or (now - last_ui) >= 0.25:
                    last_ui = now
                    update_progress(_progress_lerp(extract_floor, extract_ceiling, i / max(1, len(a_metric))))

            polyish = _geom_to_polygonal_metric(geom, line_buf_m=line_buf_m, point_buf_m=point_buf_m)
            if polyish is None:
                stats["skipped"] += 1
                continue
            # Validity (can change topology slightly; but avoids GEOS failures)
            polyish = _fix_valid(polyish)
            ex = _extract_polygonal(polyish)
            if ex is None or getattr(ex, "is_empty", False):
                stats["skipped"] += 1
                continue

            cov_batch.append(ex)
            if len(cov_batch) >= cov_batch_size:
                flush_coverage()

            for seg in _iter_boundary_lines(ex):
                boundary_batch.append(seg)
                stats["boundary_parts"] += 1
                if len(boundary_batch) >= batch_size:
                    flush_boundary()

        flush_boundary()
        flush_coverage()

    if not line_partials:
        return None, None, stats

    # Final noded linework
    STATS.detail = "final noding"
    if progress_floor is not None and progress_ceiling is not None:
        update_progress(_progress_lerp(progress_floor, progress_ceiling, 0.985))
    line_partials = _tree_reduce_unions(line_partials, max_partials=2, label="edges(final)")
    edge_net = _unary_union_safe(line_partials, label="linework_final", min_step=2)
    if edge_net is None:
        edge_net = line_partials[0]

    coverage = None
    if cov_partials:
        cov_partials = _tree_reduce_unions(cov_partials, max_partials=2, label="coverage(final)")
        coverage = _unary_union_safe(cov_partials, label="coverage_final", min_step=2)
        if coverage is None:
            coverage = cov_partials[0]

    if progress_floor is not None and progress_ceiling is not None:
        update_progress(float(progress_ceiling))

    return edge_net, coverage, stats

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
            # Fallback for missing geo metadata (common in some compiled/mixed envs)
            try:
                df = pd.read_parquet(path)
                if "geometry" in df.columns:
                    # Attempt to recover geometry from WKB bytes if present
                    if not df.empty and isinstance(df["geometry"].iloc[0], bytes):
                        df["geometry"] = df["geometry"].apply(lambda b: shp_wkb.loads(b) if b else None)
                    
                    gdf = gpd.GeoDataFrame(df, geometry="geometry")
                    if gdf.crs is None:
                        gdf.set_crs(default_crs, inplace=True)
                    return gdf
            except Exception:
                pass
            log_to_gui(f"Failed reading {path.name}: {e}", "WARN")
    return gpd.GeoDataFrame(geometry=[], crs=default_crs)

def union_from_asset_groups_or_objects(base_dir: Path):
    cfg = read_config(config_path(base_dir))
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
    pg = _existing_parquet_path(base_dir, "tbl_geocode_group")
    if pg is None:
        pg = geodir / "tbl_geocode_group.parquet"
    po = _existing_parquet_path(base_dir, "tbl_geocode_object")
    if po is None:
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


def _clear_geocode_groups(base_dir: Path, group_names: list[str]) -> None:
    names = sorted({(n or "").strip() for n in group_names if (n or "").strip()})
    if not names:
        return
    existing_g, existing_o = _load_existing_geocodes(base_dir)
    if existing_g.empty or "name_gis_geocodegroup" not in existing_g.columns:
        log_to_gui(f"No existing geocode data to clear for {', '.join(names)}.")
        return

    mask = existing_g["name_gis_geocodegroup"].astype(str).isin(names)
    if not mask.any():
        log_to_gui(f"No matching geocode groups to clear for {', '.join(names)}.")
        return

    removed_groups = existing_g.loc[mask].copy()
    remaining_groups = existing_g.loc[~mask].copy()
    remaining_objects = existing_o.copy()

    removed_ids: set[int] = set()
    if "id" in removed_groups.columns:
        try:
            removed_ids = set(removed_groups["id"].astype(int).tolist())
        except Exception:
            removed_ids = set()

    if not remaining_objects.empty:
        if removed_ids and "ref_geocodegroup" in remaining_objects.columns:
            try:
                remaining_objects = remaining_objects.loc[
                    ~remaining_objects["ref_geocodegroup"].astype(int).isin(removed_ids)
                ].copy()
            except Exception:
                pass
        if "name_gis_geocodegroup" in remaining_objects.columns:
            remaining_objects = remaining_objects.loc[
                ~remaining_objects["name_gis_geocodegroup"].astype(str).isin(names)
            ].copy()

    out_dir = gpq_dir(base_dir)
    ensure_wgs84(remaining_groups).to_parquet(out_dir / "tbl_geocode_group.parquet", index=False)
    ensure_wgs84(remaining_objects).to_parquet(out_dir / "tbl_geocode_object.parquet", index=False)

    log_to_gui(
        f"Cleared existing geocode groups: {', '.join(names)} (removed {len(removed_groups)} group record(s)).",
        "INFO",
    )

def _list_existing_h3_group_names(base_dir: Path) -> list[str]:
    g, _ = _load_existing_geocodes(base_dir)
    if g.empty or "name_gis_geocodegroup" not in g.columns:
        return []
    try:
        names = (
            g["name_gis_geocodegroup"]
            .astype(str)
            .map(lambda s: s.strip())
            .tolist()
        )
    except Exception:
        return []
    return sorted({n for n in names if n and n.startswith("H3_R")})

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
# Mosaic worker (clip-before-buffer; no geometry changes)
# -----------------------------------------------------------------------------
def _mosaic_tile_worker(task: Tuple[int, List[bytes], float, float, bytes]):
    """
    Input:
      (tile_index, [WKB of original geoms (metric CRS)], buffer_m, simplify_tol_m, clip_wkb)
    Output:
      (tile_index, [WKB faces], error:str|None)
    Notes:
      - Clip-before-buffer is mathematically exact inside the tile’s region expanded by buffer_m.
      - Keep simplify_tol_m = 0 for identity; nonzero only if you accept approximation.
    """
    idx, wkb_list, buf_m, simp, clip_wkb = task
    try:
        if not wkb_list:
            return (idx, [], None)

        geoms = [shp_wkb.loads(b) for b in wkb_list if b]
        clip_poly = shp_wkb.loads(clip_wkb) if clip_wkb else None
        buf = max(0.01, float(buf_m))

        processed = []
        for g in geoms:
            if not g or g.is_empty:
                continue
            g_local = g
            if clip_poly is not None:
                try:
                    g_local = g.intersection(clip_poly)
                    if not g_local or g_local.is_empty:
                        continue
                except Exception:
                    g_local = g

            try:
                gb = g_local.buffer(buf)
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
def mosaic_faces_from_assets_parallel(
    base_dir: Path,
    buffer_m: float,
    grid_size_m: float,
    workers: int | None = None,
) -> gpd.GeoDataFrame:
    """Build basic_mosaic faces from global linework.

    Requirements:
    - No tiling and no polygon buffering.
    - Line and point assets are buffered to polygons using config.ini defaults
      (default_line_buffer_m / default_point_buffer_m).
    - Uses batched unions + tree reduction to reduce peak memory.
    - Streams faces to disk during polygonize.

        Notes:
        - buffer_m and grid_size_m are kept for UI/CLI compatibility, but are not used
            by this algorithm.
        - workers is used for parallel boundary extraction (map-reduce) before the
            final global noding + polygonize.
    """
    cfg = read_config(config_path(base_dir))
    update_progress(0)
    log_to_gui(
        f"[Mosaic] Linework mosaic (no tiling). Params: buffer={float(buffer_m):.2f} m (unused), grid={float(grid_size_m):.2f} m (unused), workers={workers}",
        "INFO",
    )

    a = _load_asset_objects(base_dir)
    if a.empty:
        log_to_gui("[Mosaic] No assets loaded; nothing to do.", "WARN")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Reduce memory: only geometry column.
    a = gpd.GeoDataFrame(a[["geometry"]].copy(), geometry="geometry", crs=a.crs)

    metric_crs = working_metric_crs_for(a, cfg)
    t0 = time.time()
    a_metric = a.to_crs(metric_crs)
    log_to_gui(f"[Mosaic] Loaded {len(a_metric):,} assets; projected to {metric_crs} in {time.time()-t0:.2f}s.", "INFO")
    update_progress(10)

    STATS.stage = "linework"
    STATS.faces_total = 0
    edge_net, coverage, st = _build_linework_and_coverage(
        a_metric,
        cfg=cfg,
        workers=workers,
        progress_floor=10.0,
        progress_ceiling=60.0,
    )
    if edge_net is None:
        log_to_gui("[Mosaic] No linework produced; cannot polygonize.", "WARN")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    log_to_gui(
        f"[Mosaic] Linework ready: boundary_parts={st['boundary_parts']:,}, union_batches={st['union_batches']:,}, skipped={st['skipped']:,}, "
        f"line_buf_m={st['line_buf_m']:.2f}, point_buf_m={st['point_buf_m']:.2f}.",
        "INFO",
    )
    prepared_cov = None
    cov_area = None
    if coverage is not None:
        try:
            prepared_cov = prep(coverage)
            cov_area = float(getattr(coverage, "area", 0.0))
        except Exception:
            prepared_cov = None
            cov_area = None

    flush_batch = max(10_000, _cfg_int(cfg, "mosaic_faces_flush_batch", 250_000))
    tmp_dir = gpq_dir(base_dir) / "__mosaic_faces_tmp"
    try:
        if tmp_dir.exists():
            _safe_rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    STATS.stage = "polygonize"
    update_progress(60)
    log_to_gui("[Mosaic] polygonize(edge_net) started…", "INFO")

    face_wkb_batch: List[bytes] = []
    part_files: List[Path] = []
    faces_kept = 0
    faces_total = 0
    last_ui = 0.0

    def _flush_faces():
        nonlocal face_wkb_batch, faces_kept
        if not face_wkb_batch:
            return
        gseries = gpd.GeoSeries([shp_wkb.loads(b) for b in face_wkb_batch], crs=metric_crs)
        gdf = gpd.GeoDataFrame(geometry=gseries, crs=metric_crs).to_crs("EPSG:4326")
        part = tmp_dir / f"faces_part_{len(part_files):05d}.parquet"
        gdf.to_parquet(part, index=False)
        part_files.append(part)
        faces_kept += len(gdf)
        face_wkb_batch = []
        log_to_gui(f"[Mosaic] Flushed {len(gdf):,} faces to {part.name}; kept so far {faces_kept:,}.", "INFO")

    try:
        for poly in polygonize(edge_net):
            faces_total += 1
            # Surface progress while polygonize runs. Total is unknown, so use a saturating curve.
            if faces_total <= 2 or (time.time() - last_ui) >= 0.25:
                last_ui = time.time()
                STATS.faces_total = int(faces_total)
                update_progress(_progress_saturating(60.0, 85.0, faces_total, 300_000))
            if poly is None or poly.is_empty:
                continue
            if not isinstance(poly, (Polygon, MultiPolygon)):
                continue
            if prepared_cov is not None:
                try:
                    rp = poly.representative_point()
                    if not prepared_cov.contains(rp):
                        continue
                except Exception:
                    pass
            try:
                face_wkb_batch.append(shp_wkb.dumps(poly))
            except Exception:
                continue
            if len(face_wkb_batch) >= flush_batch:
                _flush_faces()
            if faces_total % 200_000 == 0:
                log_to_gui(f"[Mosaic] polygonize progress: produced {faces_total:,} faces; kept {faces_kept + len(face_wkb_batch):,}", "INFO")
    finally:
        _flush_faces()

    if not part_files:
        log_to_gui("[Mosaic] No faces produced; mosaic empty.", "WARN")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    update_progress(85)
    STATS.stage = "assembling"
    parts = []
    part_files_sorted = sorted(part_files)
    for i, p in enumerate(part_files_sorted, start=1):
        try:
            parts.append(gpd.read_parquet(p))
        except Exception as e:
            log_to_gui(f"[Mosaic] Failed to read part {p.name}: {e}", "WARN")
        update_progress(_progress_lerp(85.0, 90.0, i / max(1, len(part_files_sorted))))
    if not parts:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    faces = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs="EPSG:4326")
    log_to_gui(f"[Mosaic] Assembled {len(faces):,} faces from {len(part_files)} part(s).", "INFO")

    # Sanity check: compare face area vs coverage area (metric CRS)
    rel_tol = max(0.0, _cfg_float(cfg, "mosaic_sanity_area_rel_tol", 0.002))
    abs_tol_m2 = max(0.0, _cfg_float(cfg, "mosaic_sanity_area_abs_m2", 1.0e7))
    try:
        faces_area = float(faces.to_crs(metric_crs).geometry.area.sum())
        if cov_area is not None and cov_area > 0:
            diff = abs(faces_area - cov_area)
            ok = diff <= max(abs_tol_m2, rel_tol * cov_area)
            log_to_gui(
                f"[Mosaic][Sanity] coverage_area={cov_area:,.0f} m²; faces_area={faces_area:,.0f} m²; diff={diff:,.0f} m²; "
                f"tol=max({abs_tol_m2:,.0f}, {rel_tol:.4f}×coverage) => {'OK' if ok else 'WARN'}",
                "INFO" if ok else "WARN",
            )
        else:
            log_to_gui(f"[Mosaic][Sanity] faces_area={faces_area:,.0f} m² (coverage unavailable)", "INFO")
    except Exception as e:
        log_to_gui(f"[Mosaic][Sanity] area computation failed: {e}", "WARN")

    # Leave room for publishing (run_mosaic/publish_mosaic_as_geocode will take it to 100).
    update_progress(90)
    return faces


# -----------------------------------------------------------------------------
# Publish mosaic as geocode
# -----------------------------------------------------------------------------
def publish_mosaic_as_geocode(base_dir: Path, faces: gpd.GeoDataFrame) -> int:
    if faces is None or faces.empty:
        log_to_gui("No mosaic faces to publish.", "WARN")
        return 0

    group_name = BASIC_MOSAIC_GROUP

    STATS.stage = "publishing"
    STATS.detail = "preparing geocodes"
    update_progress(92)

    cfg: configparser.ConfigParser | None = None
    try:
        cfg = read_config(config_path(base_dir))
        metric_crs = working_metric_crs_for(faces, cfg)
        cent = faces.to_crs(metric_crs).geometry.centroid
        order_idx = np.lexsort((cent.y.values, cent.x.values))
        faces = faces.iloc[order_idx].reset_index(drop=True)
    except Exception:
        faces = faces.reset_index(drop=True)
    else:
        log_to_gui(
            f"[Mosaic] Publishing {len(faces):,} faces sorted by centroid in {metric_crs}.",
            "INFO",
        )

    codes = [f"{group_name}_{i:06d}" for i in range(1, len(faces) + 1)]

    obj = faces.copy()
    obj["code"] = codes
    obj["name_gis_geocodegroup"] = group_name
    obj["attributes"] = None
    obj = obj[["code", "name_gis_geocodegroup", "attributes", "geometry"]]

    bbox_poly = _bbox_polygon_from(faces)
    bbox_area_km2 = None
    if cfg is not None:
        try:
            if bbox_poly is not None:
                bbox_area_km2 = (
                    gpd.GeoSeries([bbox_poly], crs="EPSG:4326").to_crs(area_projection(cfg)).area.iloc[0]
                    / 1_000_000.0
                )
        except Exception:
            bbox_area_km2 = None

    if bbox_area_km2 is not None:
        log_to_gui(f"[Mosaic] Bounding box area ≈ {bbox_area_km2:,.1f} km².", "INFO")

    groups = gpd.GeoDataFrame([{
        "name": group_name,
        "name_gis_geocodegroup": group_name,
        "title_user": "Basic mosaic",
        "description": "Atomic faces derived from buffered assets (polygonize).",
        "geometry": bbox_poly
    }], geometry="geometry", crs="EPSG:4326")

    STATS.detail = "writing GeoParquet"
    update_progress(95)

    added_g, added_o, tot_g, tot_o = _merge_and_write_geocodes(
        base_dir, groups, obj, refresh_group_names=[group_name]
    )
    update_progress(99)
    log_to_gui(
        f"Published mosaic geocode '{group_name}' → "
        f"added objects: {added_o:,}; totals => groups: {tot_g}, objects: {tot_o:,}"
    )
    return added_o

# -----------------------------------------------------------------------------
# H3 writers
# -----------------------------------------------------------------------------
def write_h3_levels(base_dir: Path, levels: List[int], clear_existing: bool = False) -> int:
    update_progress(0)
    log_to_gui("Step [H3] STARTED")
    status_detail = None
    failed = False
    try:
        if not levels:
            status_detail = "No H3 levels selected."
            log_to_gui(status_detail, "WARN")
            return 0
        if h3 is None:
            status_detail = "H3 Python package not available. Install with: pip install h3"
            log_to_gui(status_detail, "WARN")
            return 0

        cfg = read_config(config_path(base_dir))
        max_cells = float(cfg["DEFAULT"].get("h3_max_cells", "1200000")) if "DEFAULT" in cfg else 1_200_000.0
        union_geom = union_from_asset_groups_or_objects(base_dir)
        if union_geom is None:
            status_detail = "No polygonal AOI found in tbl_asset_group/tbl_asset_object (consider polygons or set [DEFAULT] h3_union_buffer_m)."
            log_to_gui(status_detail, "WARN")
            return 0
        log_to_gui(f"H3 version: {_h3_version()}", "INFO")

        groups_rows = []
        objects_parts = []
        levels_sorted = sorted(set(int(r) for r in levels))
        if clear_existing:
            existing_h3 = _list_existing_h3_group_names(base_dir)
            if existing_h3:
                log_to_gui(
                    f"[H3] Delete existing enabled → clearing {len(existing_h3)} group(s): {', '.join(existing_h3)}",
                    "INFO",
                )
                _clear_geocode_groups(base_dir, existing_h3)
            else:
                log_to_gui("[H3] Delete existing enabled → no existing H3 groups found to delete.", "INFO")
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
            status_detail = "No H3 output generated (all levels skipped/empty)."
            log_to_gui(status_detail, "WARN")
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
        status_detail = f"Generated H3 levels: {', '.join(str(r) for r in levels_sorted)} (objects added: {added_o:,})"
        return added_o
    except Exception as e:
        failed = True
        log_to_gui(f"Step [H3] FAILED: {e}", "ERROR")
        return 0
    finally:
        update_progress(100)
        if not failed:
            if status_detail:
                log_to_gui(f"Step [H3] COMPLETED ({status_detail})")
            else:
                log_to_gui("Step [H3] COMPLETED")

# -----------------------------------------------------------------------------
# Mosaic runner
# -----------------------------------------------------------------------------
def run_mosaic(base_dir: Path, buffer_m: float, grid_size_m: float, on_done=None):
    update_progress(0)
    log_to_gui("Step [Mosaic] STARTED")
    success = False
    status_detail = None
    try:
        _clear_geocode_groups(base_dir, [BASIC_MOSAIC_GROUP])
        cfg = read_config(config_path(base_dir))
        # Optional force-serial (via config or ENV)
        force_serial = False
        try:
            v = str(cfg["DEFAULT"].get("mosaic_force_serial", "false")).strip().lower()
            force_serial = v in ("1", "true", "yes", "on")
        except Exception:
            pass
        if os.environ.get("MESA_FORCE_SERIAL", "").strip() in ("1", "true", "yes", "on"):
            force_serial = True

        try:
            configured_workers = int(cfg["DEFAULT"].get("mosaic_workers", "0"))
        except Exception:
            configured_workers = 0

        auto_reason = None
        workers = configured_workers
        if workers <= 0:
            workers, auto_reason = _auto_worker_count(cfg)
        if force_serial:
            workers = 1
            if auto_reason:
                auto_reason = f"force_serial overrides auto ({auto_reason})"
            else:
                auto_reason = "force_serial overrides auto"

        cfg_label = configured_workers if configured_workers > 0 else "auto"
        log_msg = (
            f"[Mosaic] Parameters ⇒ buffer={float(buffer_m):.2f} m, grid={float(grid_size_m):.2f} m, "
            f"force_serial={force_serial}, configured_workers={cfg_label}, effective_workers={workers}"
        )
        if auto_reason:
            log_msg += f" [{auto_reason}]"
        log_to_gui(log_msg, "INFO")

        faces = mosaic_faces_from_assets_parallel(base_dir, buffer_m, grid_size_m, workers)
        if faces.empty:
            status_detail = "No faces produced to publish."
            log_to_gui(status_detail, "WARN")
            if on_done:
                try: on_done(False)
                except Exception: pass
            return

        log_to_gui(
            f"[Mosaic] Received {len(faces):,} assembled faces; publishing group '{BASIC_MOSAIC_GROUP}'.",
            "INFO",
        )

        n = publish_mosaic_as_geocode(base_dir, faces)
        status_detail = f"Mosaic published as geocode group '{BASIC_MOSAIC_GROUP}' with {n:,} objects."
        log_to_gui(status_detail)
        success = True
        if on_done:
            try: on_done(True)
            except Exception: pass
    except Exception as e:
        log_to_gui(f"Step [Mosaic] FAILED: {e}", "ERROR")
        if on_done:
            try: on_done(False)
            except Exception: pass
    finally:
        update_progress(100)
        if success:
            log_to_gui("Step [Mosaic] COMPLETED")
        elif status_detail:
            log_to_gui(f"Step [Mosaic] COMPLETED ({status_detail})")

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
            if ttk else tk.Scale(pframe, orient="horizontal", length=240, from_=0, to=100, variable=progress_var, showvalue=0))
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

    clear_h3_var = tk.BooleanVar(value=False)
    clear_h3_chk = (
        ttk.Checkbutton(size_frame, text="Delete existing H3 before generating", variable=clear_h3_var, bootstyle=INFO)
        if ttk else tk.Checkbutton(size_frame, text="Delete existing H3 before generating", variable=clear_h3_var)
    )
    clear_h3_chk.grid(row=2, column=0, columnspan=2, padx=4, pady=2, sticky="w")

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
            _run_in_thread(write_h3_levels, base, levels, clear_existing=bool(clear_h3_var.get()))

        gen_btn.config(command=_generate_size_based, state=("normal" if levels else "disabled"))
        log_to_gui(f"Suggested H3 levels: {levels}" if levels else "No H3 levels for that size range.", "INFO")
        log_to_gui("Step [Suggest H3] COMPLETED")

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
    (ttk.Button(exit_frame, text="Import geocodes", bootstyle=PRIMARY, command=lambda: _run_in_thread(run_import_geocodes, base, cfg)) if ttk
     else tk.Button(exit_frame, text="Import geocodes", command=lambda: _run_in_thread(run_import_geocodes, base, cfg))).pack(side=tk.LEFT)
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
    cfg = read_config(config_path(base))

    # Respect custom parquet folder if provided
    global _PARQUET_SUBDIR
    _PARQUET_SUBDIR = cfg["DEFAULT"].get("parquet_folder", "output/geoparquet")

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
    # Important for PyInstaller child processes on Windows:
    try:
        mp.freeze_support()
    except Exception:
        pass
    main()

