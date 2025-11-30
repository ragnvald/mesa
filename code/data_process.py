# -*- coding: utf-8 -*-
# 06_process.py — memory-aware, CPU-optimized (Windows spawn-safe) intersections + robust flattening to GeoParquet
# UI: two panes (left: logs/progress/buttons; right: minimap launcher).
# Minimap opens in a separate, low-priority helper process (pywebview + Leaflet + OSM).

import locale
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except Exception:
    pass

import os, sys, math, re, time, random, argparse, threading, multiprocessing, json, shutil, uuid, gc, importlib.util, subprocess, ast
import configparser
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

# NEW: geodesic area + validity helpers
from pyproj import Geod
try:
    from shapely.validation import make_valid as _make_valid  # type: ignore
except Exception:
    _make_valid = None

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

# GUI
import tkinter as tk
from tkinter import scrolledtext
import ttkbootstrap as tb
from ttkbootstrap.constants import PRIMARY, WARNING

# ----------------------------
# Globals
# ----------------------------
original_working_directory = None
log_widget = None
progress_var = None
progress_label = None
progress_stage_text = "Preparations"
_progress_value = 0.0
HEARTBEAT_SECS = 60

def _mp_allowed() -> bool:
    """Multiprocessing Pool is unsafe from a non-main thread in frozen builds (PyInstaller/Windows).
    Return True only when it's safe to create a Pool.
    """
    try:
        if getattr(sys, 'frozen', False):
            import threading
            return threading.current_thread() is threading.main_thread()
        return True
    except Exception:
        return True

# Faster heartbeat in frozen builds so the minimap shows life quickly
if getattr(sys, 'frozen', False):
    try:
        HEARTBEAT_SECS = min(HEARTBEAT_SECS, 10)
    except Exception:
        pass


_PARTS_DIR = None  # worker output folder

# minimap status (tiny JSON snapshots)
MINIMAP_STATUS_PATH = None
MINIMAP_LOCK = threading.Lock()

# minimap helper process
_MAP_PROC = None

# processing worker process (to keep GUI responsive and allow true multiprocessing in frozen builds)
_PROC = None

# Memory logging snapshot state
_LAST_MEM_LOG_TS = 0.0
_LAST_MEM_RSS_GB: float | None = None

# grid meta for minimap
_GRID_BBOX_MAP: dict[int, list[float]] = {}   # grid_cell -> [S,W,N,E]

# Config cache (flat config at <base>/config.ini)
_CFG: configparser.ConfigParser | None = None

# Geodesic engine (WGS84) for area stats
_GEOD = Geod(ellps="WGS84")

# Default basic mosaic group name (can be overridden in config.ini [DEFAULT] basic_group_name)
_DEFAULT_BASIC_GROUP_NAME = "basic_mosaic"

INDEX_WEIGHT_DEFAULTS = {
    "importance": [1, 1, 2, 3, 3],
    "sensitivity": [1, 1, 2, 3, 3],
}
INDEX_WEIGHT_KEYS = {
    "importance": "index_importance_weights",
    "sensitivity": "index_sensitivity_weights",
}

# ----------------------------
# Paths & config helpers
# ----------------------------
def base_dir() -> Path:
    """
    Resolve the mesa root folder in all modes:
    - dev .py, compiled helper .exe (tools\), launched from mesa.exe, or double-clicked in tools\
    """
    candidates: list[Path] = []

    # 1) explicit hint (mesa.exe usually passes this)
    try:
        if 'original_working_directory' in globals() and original_working_directory:
            candidates.append(Path(original_working_directory))
    except Exception:
        pass

    # 2) frozen exe location
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent)
    else:
        # 3) script location (dev)
        if "__file__" in globals():
            candidates.append(Path(__file__).resolve().parent)

    # 4) last resort: CWD
    candidates.append(Path(os.getcwd()).resolve())

    def normalize(p: Path) -> Path:
        p = p.resolve()
        # Normalize typical subfolders to the mesa root
        if p.name.lower() in ("tools", "system", "code"):
            p = p.parent
        # Try climbing up a few levels to find a folder that looks like mesa root
        q = p
        for _ in range(4):
            if (q / "output").exists() and (q / "input").exists():
                return q
            if (q / "tools").exists() and (q / "config.ini").exists():
                return q
            q = q.parent
        return p

    for c in candidates:
        root = normalize(c)
        if (root / "tools").exists() or ((root / "output").exists() and (root / "input").exists()):
            return root

    # Fallback: normalized first candidate
    return normalize(candidates[0])



def _ensure_cfg() -> configparser.ConfigParser:
    """
    Return a ConfigParser with DEFAULT.parquet_folder available.
    Lazily loads <base>/config.ini if _CFG wasn't set yet.
    """
    global _CFG
    if _CFG is None:
        _CFG = configparser.ConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
        try:
            _CFG.read(base_dir() / "config.ini", encoding="utf-8")
        except Exception:
            pass
        if "DEFAULT" not in _CFG:
            _CFG["DEFAULT"] = {}
    if "parquet_folder" not in _CFG["DEFAULT"]:
        _CFG["DEFAULT"]["parquet_folder"] = "output/geoparquet"
    return _CFG

def _basic_group_name() -> str:
    try:
        cfg = _ensure_cfg()
        name = (cfg["DEFAULT"].get("basic_group_name", _DEFAULT_BASIC_GROUP_NAME) or "").strip()
        return name if name else _DEFAULT_BASIC_GROUP_NAME
    except Exception:
        return _DEFAULT_BASIC_GROUP_NAME

def set_global_cfg(cfg: configparser.ConfigParser):
    """Install a pre-read config for use across helpers."""
    global _CFG
    _CFG = cfg
    if "DEFAULT" not in _CFG:
        _CFG["DEFAULT"] = {}
    _CFG["DEFAULT"].setdefault("parquet_folder", "output/geoparquet")

def gpq_dir() -> Path:
    cfg = _ensure_cfg()
    sub = cfg["DEFAULT"].get("parquet_folder", "output/geoparquet")
    out = base_dir() / sub
    out.mkdir(parents=True, exist_ok=True)
    return out

def output_dir() -> Path:
    out = base_dir() / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out

def _dataset_dir(name: str) -> Path:
    return gpq_dir() / name

# --- SAFE CONFIG PARSING (flat config) ---
def read_config(path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
    try:
        cfg.read(path, encoding="utf-8")
    except Exception:
        pass
    if "DEFAULT" not in cfg:
        cfg["DEFAULT"] = {}
    cfg["DEFAULT"].setdefault("parquet_folder", "output/geoparquet")
    return cfg

def _strip_inline_comments(s: str) -> str:
    if s is None:
        return ""
    for sep in (';', '#'):
        if sep in s:
            s = s.split(sep, 1)[0]
    return s.strip()

def cfg_get_int(cfg: configparser.ConfigParser, key: str, default: int) -> int:
    try:
        raw = cfg['DEFAULT'].get(key, str(default))
        raw = _strip_inline_comments(raw)
        return int(raw)
    except Exception:
        return int(default)

def cfg_get_float(cfg: configparser.ConfigParser, key: str, default: float) -> float:
    try:
        raw = cfg['DEFAULT'].get(key, str(default))
        raw = _strip_inline_comments(raw)
        return float(raw)
    except Exception:
        return float(default)

# ----------------------------
# Logging / UI helpers
# ----------------------------
def _current_progress_display() -> str:
    stage = (progress_stage_text or "").strip()
    pct_text = f"{int(_progress_value)}%"
    return f"{stage}: {pct_text}" if stage else pct_text


def _refresh_progress_label():
    try:
        if progress_label is not None:
            progress_label.config(text=_current_progress_display())
            progress_label.update_idletasks()
    except Exception:
        pass


def set_progress_stage(stage_name: str):
    """Set the descriptive stage text that prefixes the percentage label."""
    global progress_stage_text
    try:
        progress_stage_text = (stage_name or "").strip()
        _refresh_progress_label()
    except Exception:
        pass


def _stage_from_phase(phase: str) -> str:
    p = (phase or "").strip().lower()
    if p == "intersect":
        return "Processing"
    if p in {"flatten_pending", "flatten", "writing"}:
        return "Cleanup"
    if p in {"done"}:
        return "Finalizing datasets"
    if p in {"tiles", "tiling", "mbtiles"}:
        return "Creating map tiles"
    if p in {"tiles_finalizing"}:
        return "Finalizing map tiles"
    if p in {"completed"}:
        return "Completed"
    if p in {"error"}:
        return "Attention required"
    return "Preparations"


def update_progress(new_value: float):
    global _progress_value
    try:
        v = max(0.0, min(100.0, float(new_value)))
        _progress_value = v
        if progress_var is not None:
            progress_var.set(v)
        _refresh_progress_label()
    except Exception:
        pass

def log_to_gui(widget, message: str):
    timestamp = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted = f"{timestamp} - {message}"
    try:
        if widget and widget.winfo_exists():
            widget.insert(tk.END, formatted + "\n")
            widget.see(tk.END)
    except tk.TclError:
        pass
    try:
        with open(base_dir() / "log.txt", "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
    except Exception:
        pass
    if widget is None:
        print(formatted, flush=True)


def _log_memory_snapshot(context: str, extra: dict | None = None, force: bool = False):
    """Emit a throttled log line with process/system memory stats."""
    global _LAST_MEM_LOG_TS, _LAST_MEM_RSS_GB
    if psutil is None:
        return
    try:
        now = time.time()
        proc = psutil.Process()
        mem_info = proc.memory_info()
        rss_gb = mem_info.rss / (1024 ** 3)
        vms_gb = mem_info.vms / (1024 ** 3)
        delta = None if _LAST_MEM_RSS_GB is None else rss_gb - _LAST_MEM_RSS_GB
        if not force and _LAST_MEM_LOG_TS and (now - _LAST_MEM_LOG_TS) < 30:
            if delta is None or abs(delta) < 0.25:
                return

        vm = psutil.virtual_memory()
        avail_gb = vm.available / (1024 ** 3)
        used_pct = vm.percent

        msg = f"[mem] {context}: proc RSS ~{rss_gb:.2f} GB"
        if delta is not None:
            msg += f" (Δ{delta:+.2f} GB)"
        msg += f" • VMS ~{vms_gb:.2f} GB • System avail ~{avail_gb:.2f} GB ({used_pct:.0f}% used)"

        if extra:
            try:
                extra_bits = ", ".join(f"{k}={v}" for k, v in extra.items())
                if extra_bits:
                    msg += f" • {extra_bits}"
            except Exception:
                pass

        log_to_gui(log_widget, msg)
        _LAST_MEM_LOG_TS = now
        _LAST_MEM_RSS_GB = rss_gb
    except Exception:
        pass

# ----------------------------
# Raster tiles integration helpers
# ----------------------------
def _tbl_flat_path() -> Path:
    return gpq_dir() / "tbl_flat.parquet"

def _has_big_polygon_group(threshold: int = 50000, group_col: str = "name_gis_geocodegroup") -> tuple[bool, dict]:
    """
    Returns (bool, counts_per_group) where bool indicates any polygonal group >= threshold.
    """
    try:
        pq = _tbl_flat_path()
        if not pq.exists():
            return (False, {})
        cols = [group_col, "geometry"]
        try:
            gdf = gpd.read_parquet(pq, columns=cols)
        except TypeError:
            gdf = gpd.read_parquet(pq)
        try:
            if gdf.crs is None:
                pass
        except Exception:
            pass
        try:
            gt = gdf.geometry.geom_type
            poly_mask = gt.isin(["Polygon", "MultiPolygon"])
        except Exception:
            poly_mask = gdf.geometry.notna()
        gpoly = gdf[poly_mask & gdf.geometry.notna()]
        if gpoly.empty:
            return (False, {})
        counts = gpoly.groupby(group_col).size().to_dict()
        big = any(v >= threshold for v in counts.values())
        return (big, counts)
    except Exception as e:
        log_to_gui(log_widget, f"[Tiles] Eligibility check failed: {e}")
        return (False, {})

def _find_tiles_script() -> Path | None:
    """
    Backward-compatible search for the raster-tiles helper:
      1) <base>/create_raster_tiles.py
      2) <base>/system/create_raster_tiles.py
      3) <base>/code/create_raster_tiles.py
    """
    cand1 = base_dir() / "create_raster_tiles.py"
    cand2 = base_dir() / "system" / "create_raster_tiles.py"
    cand3 = base_dir() / "code" / "create_raster_tiles.py"
    for cand in (cand1, cand2, cand3):
        if cand.exists():
            return cand
    return None

def _spawn_tiles_subprocess(minzoom: int|None=None, maxzoom: int|None=None):
    script_path = _find_tiles_script()
    if not script_path:
        log_to_gui(log_widget, f"[Tiles] Missing script: create_raster_tiles.py (looked in base and system/)")
        return None

    args = [sys.executable, str(script_path)]
    if isinstance(minzoom, int):
        args += ["--minzoom", str(minzoom)]
    if isinstance(maxzoom, int):
        args += ["--maxzoom", str(maxzoom)]

    env = dict(os.environ)
    # ensure UTF-8 stdout/stderr inside the child process
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # Pin the mesa root so helper scripts never confuse code/output with the real output folder
    try:
        env["MESA_BASE_DIR"] = str(base_dir())
    except Exception:
        env.setdefault("MESA_BASE_DIR", os.getcwd())

    return subprocess.Popen(
        args,
        cwd=str(script_path.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",   # decode our pipe as UTF-8
        bufsize=1,
        env=env,
    )

def _run_tiles_stream_to_gui(minzoom=None, maxzoom=None):
    """
    Run tiles subprocess, stream its output to the GUI, and update the shared progress bar.
    """
    tile_ceiling = 99.0  # keep headroom for explicit "completed" phase to own 100%
    try:
        layers_per_group = 4  # sensitivity, env, groupstotal, assetstotal
        ok, counts = _has_big_polygon_group()
        total_groups = len([k for k,v in counts.items() if v>0])
        total_steps = max(1, total_groups * layers_per_group)
        done_steps = 0
        tile_floor = max(95.0, min(_progress_value, 100.0))
        tile_span = max(1.0, tile_ceiling - tile_floor)
        update_progress(tile_floor)
        log_to_gui(log_widget, "[Tiles] Stage 4/4 - integrating MBTiles build (create_raster_tiles)...")

        proc = _spawn_tiles_subprocess(minzoom=minzoom, maxzoom=maxzoom)

        if proc is None:
            log_to_gui(log_widget, "[Tiles] Unable to start create_raster_tiles subprocess.")
            return

        re_groups = re.compile(r"^Groups:\s*\[(.*)\]\s*$")
        re_building = re.compile(r"\u2192\s*building|building\s+.+\.\.\.")
        re_done = re.compile(r"All done\.")

        for line in proc.stdout:
            line = line.rstrip("\n")
            if not line:
                continue
            log_to_gui(log_widget, f"[Tiles] {line}")

            m = re_groups.search(line)
            if m:
                inside = m.group(1)
                try:
                    content = inside
                    if content and not content.strip().startswith("["):
                        content = "[" + content + "]"
                    groups_list = ast.literal_eval(content)
                    if isinstance(groups_list, (list, tuple)):
                        total_groups = len(groups_list)
                        total_steps = max(1, total_groups*layers_per_group)
                except Exception:
                    total_groups = max(total_groups, len([s for s in inside.split(",") if s.strip()]))
                    total_steps = max(1, total_groups*layers_per_group)

            if re_building.search(line):
                done_steps += 1
                pct = tile_floor + (done_steps / max(total_steps, 1)) * tile_span
                update_progress(min(tile_ceiling, pct))

            if re_done.search(line):
                update_progress(tile_ceiling)
                log_to_gui(log_widget, "[Tiles] Rendering finished; finalizing MBTiles (can take a few minutes)…")
                try:
                    _update_status_phase("tiles_finalizing")
                except Exception:
                    pass

        ret = proc.wait()
        if ret != 0:
            log_to_gui(log_widget, f"[Tiles] create_raster_tiles exited with code {ret}")
        else:
            log_to_gui(log_widget, "[Tiles] Completed.")
    except Exception as e:
        log_to_gui(log_widget, f"[Tiles] Error: {e}")
    finally:
        update_progress(tile_ceiling)
        # Tail the shared worker log file into the GUI while the worker runs
        try:
            _start_log_tailer(root)
        except Exception:
            pass

def _auto_run_tiles_stage(minzoom, maxzoom):
    """
    Trigger the MBTiles helper as part of the main processing workflow.
    """
    try:
        set_progress_stage("Creating map tiles")
        _update_status_phase("tiles")
        ok, counts = _has_big_polygon_group(threshold=0)
        if not counts:
            log_to_gui(log_widget, "[Tiles] tbl_flat not present or empty; skipping MBTiles generation.")
            update_progress(100.0)
            return
        log_to_gui(log_widget, f"[Tiles] Preparing MBTiles for {len(counts)} geocode group(s).")
        _run_tiles_stream_to_gui(minzoom=minzoom, maxzoom=maxzoom)
    except Exception as e:
        log_to_gui(log_widget, f"[Tiles] Skipped due to error: {e}")
        update_progress(100.0)
    finally:
        _update_status_phase("completed")

# ----------------------------
# A..E classification helpers
# ----------------------------
def read_class_ranges(cfg_path: Path):
    cfg = configparser.ConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
    try:
        cfg.read(cfg_path, encoding="utf-8")
    except Exception:
        pass
    ranges, desc = {}, {}
    for code in ["A","B","C","D","E"]:
        if code not in cfg:
            continue
        rtxt = (cfg[code].get("range","") or "").strip()
        if "-" in rtxt:
            try:
                lo, hi = [int(x) for x in rtxt.split("-")]
                ranges[code] = range(lo, hi+1)
            except Exception:
                pass
        desc[code] = (cfg[code].get("description","") or "").strip()
    order = {"A":5,"B":4,"C":3,"D":2,"E":1}
    return ranges, desc, order

def map_num_to_code(val, ranges_map: dict) -> str | None:
    if pd.isna(val): return None
    try:
        iv = int(round(float(val)))
    except Exception:
        return None
    for k, r in ranges_map.items():
        if iv in r:
            return k
    return None

# ----------------------------
# IO helpers
# ----------------------------
def read_parquet_or_empty(name: str) -> gpd.GeoDataFrame:
    """
    Tries (in order):
      1) <base>/<parquet_folder>/<name>.parquet
      2) <base>/<parquet_folder>/<name>/  (partitioned)
      3) <base>/input/geoparquet/<name>.parquet
      4) <base>/input/geoparquet/<name>/  (partitioned)
    Returns empty GeoDataFrame if none found.
    """
    out_root = gpq_dir()
    in_root  = base_dir() / "input" / "geoparquet"

    file_path = out_root / f"{name}.parquet"
    dir_path  = out_root / name
    alt_file  = in_root / f"{name}.parquet"
    alt_dir   = in_root / name

    try:
        if file_path.exists():
            return gpd.read_parquet(file_path)
        if dir_path.exists() and dir_path.is_dir():
            return gpd.read_parquet(str(dir_path))
        if alt_file.exists():
            return gpd.read_parquet(alt_file)
        if alt_dir.exists() and alt_dir.is_dir():
            return gpd.read_parquet(str(alt_dir))
        log_to_gui(log_widget, f"[read] Not found: {name} under {out_root} or {in_root}")
        return gpd.GeoDataFrame(geometry=[], crs=None)
    except Exception as e:
        log_to_gui(log_widget, f"Failed to read {name}: {e}")
        return gpd.GeoDataFrame(geometry=[], crs=None)


def write_parquet(name: str, gdf: gpd.GeoDataFrame):
    path = gpq_dir() / f"{name}.parquet"
    gdf.to_parquet(path, index=False)
    log_to_gui(log_widget, f"Wrote {path}")

def _rm_rf(path: Path):
    try:
        if not path.exists(): return
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except Exception as e:
        log_to_gui(log_widget, f"Error removing {path.name}: {e}")

def cleanup_outputs():
    for fn in ["tbl_stacked.parquet","tbl_flat.parquet"]:
        _rm_rf(gpq_dir() / fn)
    for d in ["tbl_stacked", "__stacked_parts", "__grid_assign_in", "__grid_assign_out"]:
        _rm_rf(_dataset_dir(d))

# ----------------------------
# Grid & chunking
# ----------------------------
from uuid import uuid4

def create_grid(geodata: gpd.GeoDataFrame, cell_size_deg: float):
    xmin, ymin, xmax, ymax = geodata.total_bounds
    if not np.isfinite([xmin, ymin, xmax, ymax]).all() or xmax <= xmin or ymax <= ymin:
        return []
    x_edges = np.arange(xmin, xmax + cell_size_deg, cell_size_deg)
    y_edges = np.arange(ymin, ymax + cell_size_deg, cell_size_deg)
    cells = []
    for x in x_edges[:-1]:
        for y in y_edges[:-1]:
            cells.append((x, y, x + cell_size_deg, y + cell_size_deg))
    return cells

_GRID_OUT_DIR = None
_GRID_GDF     = None

def _grid_pool_init2(grid_gdf, out_dir_str: str):
    global _GRID_OUT_DIR, _GRID_GDF
    _GRID_OUT_DIR = Path(out_dir_str)
    _GRID_OUT_DIR.mkdir(parents=True, exist_ok=True)
    _GRID_GDF = grid_gdf
    try: _ = _GRID_GDF.sindex
    except Exception: pass
    try:
        if psutil is not None:
            p = psutil.Process()
            if os.name == "nt":
                p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:
                os.nice(5)
    except Exception:
        pass

def _grid_worker(input_path: str) -> str:
    g = gpd.read_parquet(input_path)
    try: _ = g.sindex
    except Exception: pass
    j = gpd.sjoin(g, _GRID_GDF, how="left", predicate="intersects")
    j.drop(columns=["index_right"], inplace=True, errors="ignore")
    try: j = j[j.geometry.notna() & ~j.geometry.is_empty]
    except Exception: pass
    out = _GRID_OUT_DIR / f"grid_tag_{os.getpid()}_{uuid.uuid4().hex}.parquet"
    j.to_parquet(out, index=False)
    del g, j
    gc.collect()
    return str(out)

def _rmrf_dir(p: Path):
    try:
        if p.exists():
            shutil.rmtree(p)
    except Exception:
        pass

def _mk_empty_gdf_like(crs) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[], crs=crs)

def calculate_rows_per_chunk(n: int, max_memory_mb: int = 256, est_bytes_per_row: int = 1800, hard_cap_rows: int = 100_000) -> int:
    try:
        rows = int((max_memory_mb * 1024 * 1024) / max(256, est_bytes_per_row))
    except Exception:
        rows = 50_000
    rows = max(5_000, min(rows, hard_cap_rows))
    return max(1, min(rows, n))

# ----------------------------
# Minimap status helpers (tiny JSON snapshots)
# ----------------------------
def _status_path() -> Path:
    global MINIMAP_STATUS_PATH
    if MINIMAP_STATUS_PATH is None:
        MINIMAP_STATUS_PATH = gpq_dir() / "__chunk_status.json"
    return MINIMAP_STATUS_PATH

def _write_status_atomic(payload: dict):
    p = _status_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    try:
        with MINIMAP_LOCK:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, p)
    except Exception:
        try:
            if tmp.exists(): tmp.unlink(missing_ok=True)
        except Exception:
            pass

def _init_idle_status():
    now = datetime.utcnow().isoformat() + "Z"
    payload = {
        "phase": "idle",
        "updated_at": now,
        "chunks_total": 0,
        "done": 0,
        "running": [],
        "cells": [],      # list of {id, state, n, bbox:[s,w,n,e]}
        "home_bounds": None
    }
    _write_status_atomic(payload)


def _update_status_phase(new_phase: str):
    """Adjust the shared status JSON to reflect a new high-level phase."""
    try:
        existing = {}
        p = _status_path()
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                existing = json.load(f) or {}
    except Exception:
        existing = {}
    try:
        payload = {
            "phase": (new_phase or "").strip().lower() or "idle",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "chunks_total": int(existing.get("chunks_total", 0) or 0),
            "done": int(existing.get("done", 0) or 0),
            "running": existing.get("running", []),
            "cells": existing.get("cells", []),
            "home_bounds": existing.get("home_bounds"),
        }
        _write_status_atomic(payload)
    except Exception:
        pass

# ----------------------------
# Assign grid + tag and return tagged geocodes
# ----------------------------
def assign_geocodes_to_grid(geodata: gpd.GeoDataFrame, meters_cell: int, max_workers: int) -> gpd.GeoDataFrame:
    global _GRID_BBOX_MAP
    if geodata is None or geodata.empty:
        return _mk_empty_gdf_like(geodata.crs if geodata is not None else None)

    meters_per_degree = 111_320.0
    cell_deg = meters_cell / meters_per_degree

    grid_cells = create_grid(geodata, cell_deg)
    if not grid_cells:
        log_to_gui(log_widget, "Grid creation produced no cells; skipping tagging.")
        return geodata.assign(grid_cell=pd.Series([0] * len(geodata), index=geodata.index))

    grid_gdf = gpd.GeoDataFrame(
        {"grid_cell": range(len(grid_cells)),
         "geometry": [box(x0, y0, x1, y1) for (x0, y0, x1, y1) in grid_cells]},
        geometry="geometry", crs=geodata.crs
    )
    # store bbox for minimap tiles
    _GRID_BBOX_MAP = {i: [float(y0), float(x0), float(y1), float(x1)] for i, (x0,y0,x1,y1) in enumerate(grid_cells)}
    try: _ = grid_gdf.sindex
    except Exception: pass
    log_to_gui(log_widget, f"Assigning geocodes to {len(grid_cells):,} grid cells…")

    tmp_in  = _dataset_dir("__grid_assign_in")
    tmp_out = _dataset_dir("__grid_assign_out")
    _rmrf_dir(tmp_in); _rmrf_dir(tmp_out)
    tmp_in.mkdir(parents=True, exist_ok=True)
    tmp_out.mkdir(parents=True, exist_ok=True)

    rows_per_chunk_est = calculate_rows_per_chunk(len(geodata))
    rows_per_chunk = rows_per_chunk_est
    chunk_size_cfg = None
    try:
        cfg = _ensure_cfg()
        chunk_size_cfg = cfg_get_int(cfg, "chunk_size", 0)
        if chunk_size_cfg <= 0:
            chunk_size_cfg = None
    except Exception:
        chunk_size_cfg = None
    if chunk_size_cfg is not None:
        rows_per_chunk = max(1, min(rows_per_chunk, chunk_size_cfg))

    cfg_chunk_display = f"{chunk_size_cfg:,}" if chunk_size_cfg else "auto"
    log_to_gui(
        log_widget,
        f"[grid-assign] rows_per_chunk={rows_per_chunk:,} (config chunk_size={cfg_chunk_display}, est_limit={rows_per_chunk_est:,})",
    )
    _log_memory_snapshot(
        "grid-assign",
        {
            "geocodes": f"{len(geodata):,}",
            "grid_cells": f"{len(grid_cells):,}",
            "rows_per_chunk": f"{rows_per_chunk:,}",
        },
        force=True,
    )

    total_chunks = int(math.ceil(len(geodata) / rows_per_chunk))
    input_parts = []
    for i in range(0, len(geodata), rows_per_chunk):
        part = geodata.iloc[i:i+rows_per_chunk]
        p = tmp_in / f"geo_{i:09d}.parquet"
        part.to_parquet(p, index=False)
        input_parts.append(str(p))

    started_at = time.time(); last_ping = started_at
    done = 0; out_files = []

    if _mp_allowed() and max_workers > 1:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=max(1, max_workers),
                      initializer=_grid_pool_init2,
                      initargs=(grid_gdf, str(tmp_out))) as pool:
            for out_path in pool.imap_unordered(_grid_worker, input_parts, chunksize=1):
                out_files.append(out_path)
                done += 1
                now = time.time()
                if (now - last_ping) >= HEARTBEAT_SECS or done == total_chunks:
                    pct = (done / total_chunks) * 100 if total_chunks else 100.0
                    elapsed = now - started_at
                    eta = "?"
                    if done:
                        est_total = elapsed / done * total_chunks
                        eta_ts = datetime.now() + timedelta(seconds=max(0.0, est_total - elapsed))
                        eta = eta_ts.strftime("%H:%M:%S")
                        dd = (eta_ts.date() - datetime.now().date()).days
                        if dd > 0: eta += f" (+{dd}d)"
                    log_to_gui(log_widget, f"[grid-assign] {done}/{total_chunks} chunks (~{pct:.2f}%) • ETA {eta}")
                    last_ping = now
    else:
        # Safe serial fallback in frozen builds when called from a non-main thread
        _grid_pool_init2(grid_gdf, str(tmp_out)) 
        for in_p in input_parts:
            out_path = _grid_worker(in_p)
            out_files.append(out_path)
            done += 1
            now = time.time()
            if (now - last_ping) >= HEARTBEAT_SECS or done == total_chunks:
                pct = (done / total_chunks) * 100 if total_chunks else 100.0
                elapsed = now - started_at
                eta = "?"
                if done:
                    est_total = elapsed / done * total_chunks
                    eta_ts = datetime.now() + timedelta(seconds=max(0.0, est_total - elapsed))
                    eta = eta_ts.strftime("%H:%M:%S")
                    dd = (eta_ts.date() - datetime.now().date()).days
                    if dd > 0: eta += f" (+{dd}d)"
                log_to_gui(log_widget, f"[grid-assign] {done}/{total_chunks} chunks (~{pct:.2f}%) • ETA {eta}")
                last_ping = now

    if not out_files:
        log_to_gui(log_widget, "Grid-assign produced no parts; continuing without grid_cell.")
        _rmrf_dir(tmp_in); _rmrf_dir(tmp_out)
        return geodata

    parts = []
    for p in out_files:
        try:
            parts.append(gpd.read_parquet(p))
        except Exception as e:
            log_to_gui(log_widget, f"[grid-assign] Skipping part {Path(p).name}: {e}")

    if not parts:
        _rmrf_dir(tmp_in); _rmrf_dir(tmp_out)
        return geodata

    tagged = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=geodata.crs)

    try:
        if 'id_geocode_object' in tagged.columns:
            tagged = tagged.drop_duplicates(subset=['id_geocode_object'], keep='first')
        elif 'code' in tagged.columns and 'name_gis_geocodegroup' in tagged.columns:
            tagged = tagged.drop_duplicates(subset=['code','name_gis_geocodegroup'], keep='first')
        elif 'code' in tagged.columns:
            tagged = tagged.drop_duplicates(subset=['code'], keep='first')
        else:
            tagged = tagged.assign(__wkb__=tagged.geometry.apply(lambda g: g.wkb if g is not None else None))
            tagged = tagged.drop_duplicates(subset=['__wkb__']).drop(columns=['__wkb__'])
    except Exception:
        pass

    _rmrf_dir(tmp_in); _rmrf_dir(tmp_out)
    return tagged

# ---- space-filling curve helper (Morton / Z-order, 16-bit per axis)
def _morton16(ix: int, iy: int) -> int:
    def _part1by1(n: int) -> int:
        n &= 0xFFFF
        n = (n | (n << 8)) & 0x00FF00FF
        n = (n | (n << 4)) & 0x0F0F0F0F
        n = (n | (n << 2)) & 0x33333333
        n = (n | (n << 1)) & 0x55555555
        return n
    return (_part1by1(ix) << 1) | _part1by1(iy)

def make_spatial_chunks(geocode_tagged: gpd.GeoDataFrame, max_workers: int, multiplier: int = 18):
    """Group tagged geocodes into spatially coherent chunks sized by config."""
    if geocode_tagged is None or geocode_tagged.empty or 'grid_cell' not in geocode_tagged.columns:
        return [geocode_tagged]

    cfg = _ensure_cfg()
    target_geocodes = cfg_get_int(cfg, "target_geocodes_per_chunk", 5000)
    chunk_cells_min = max(1, cfg_get_int(cfg, "chunk_cells_min", 3))
    chunk_cells_max = max(chunk_cells_min, cfg_get_int(cfg, "chunk_cells_max", 90))
    backlog_multiplier = max(1.0, cfg_get_float(cfg, "chunk_backlog_multiplier", 2.5))
    overshoot_factor = max(1.0, cfg_get_float(cfg, "chunk_overshoot_factor", 1.25))

    # One representative centroid per grid_cell (fast via bounds midpoints)
    try:
        b = geocode_tagged.geometry.bounds  # DataFrame with minx, miny, maxx, maxy
        cx = (b['minx'] + b['maxx']) * 0.5
        cy = (b['miny'] + b['maxy']) * 0.5
        centroids = pd.DataFrame({
            'grid_cell': geocode_tagged['grid_cell'].values,
            '__cx': cx.values,
            '__cy': cy.values
        })
    except Exception:
        # Fallback: shapely centroid (slower, but robust)
        centroids = geocode_tagged[['grid_cell']].copy()
        cc = geocode_tagged.geometry.centroid
        centroids['__cx'] = cc.x.values
        centroids['__cy'] = cc.y.values

    # Deduplicate per cell
    centroids = centroids.dropna(subset=['__cx','__cy']).drop_duplicates(subset=['grid_cell'])

    if centroids.empty:
        cell_ids = sorted(geocode_tagged['grid_cell'].unique().tolist())
    else:
        xmin, ymin = float(centroids['__cx'].min()), float(centroids['__cy'].min())
        xmax, ymax = float(centroids['__cx'].max()), float(centroids['__cy'].max())
        dx = max(1e-9, xmax - xmin)
        dy = max(1e-9, ymax - ymin)

        xi = ((centroids['__cx'] - xmin) / dx * 65535.0).clip(0, 65535).astype('uint16')
        yi = ((centroids['__cy'] - ymin) / dy * 65535.0).clip(0, 65535).astype('uint16')
        centroids['__z'] = [_morton16(int(x), int(y)) for x, y in zip(xi.values, yi.values)]
        cell_ids = centroids.sort_values('__z')['grid_cell'].tolist()

    # Historical fallback when user disables target chunk sizing
    if target_geocodes <= 0:
        target_chunks = max(1, min(len(cell_ids), max_workers * multiplier))
        cells_per_chunk = max(1, math.ceil(len(cell_ids) / target_chunks))
        return [geocode_tagged[geocode_tagged['grid_cell'].isin(set(cell_ids[i:i + cells_per_chunk]))]
                for i in range(0, len(cell_ids), cells_per_chunk)]

    cell_counts = geocode_tagged.groupby('grid_cell').size().to_dict()
    total_geocodes = int(sum(cell_counts.get(cid, 0) for cid in cell_ids))
    if total_geocodes <= 0:
        return [geocode_tagged]

    min_chunks = max(1, max_workers or 1)
    chunks_from_target = max(1, math.ceil(total_geocodes / max(target_geocodes, 1)))
    backlog_chunks = max(chunks_from_target, int(chunks_from_target * backlog_multiplier))
    max_reasonable_chunks = max(min_chunks, math.ceil(total_geocodes / max(int(target_geocodes * 0.33), 1)))
    target_chunks = min(len(cell_ids), max(min_chunks, min(backlog_chunks, max_reasonable_chunks)))
    target_cells_per_chunk = max(
        chunk_cells_min,
        min(chunk_cells_max, math.ceil(len(cell_ids) / max(target_chunks, 1)))
    )

    chunks = []
    chunk_sizes = []
    current_cells = []
    current_geocode_count = 0
    overshoot_limit = max(target_geocodes, int(target_geocodes * overshoot_factor))

    def _flush(reason: str):
        nonlocal current_cells, current_geocode_count
        if not current_cells:
            return
        sel = set(current_cells)
        chunk = geocode_tagged[geocode_tagged['grid_cell'].isin(sel)]
        chunks.append(chunk)
        chunk_sizes.append(len(chunk))
        current_cells = []
        current_geocode_count = 0

    for idx, cid in enumerate(cell_ids):
        current_cells.append(cid)
        current_geocode_count += int(cell_counts.get(cid, 0))

        remaining_cells = len(cell_ids) - (idx + 1)
        chunks_remaining = max(1, target_chunks - len(chunks))
        must_flush = False

        if current_geocode_count >= overshoot_limit:
            must_flush = True
        elif current_geocode_count >= target_geocodes and len(current_cells) >= chunk_cells_min:
            must_flush = True
        elif len(current_cells) >= chunk_cells_max:
            must_flush = True
        elif len(current_cells) >= target_cells_per_chunk and len(current_cells) >= chunk_cells_min:
            must_flush = True

        # Ensure we can still produce the desired number of chunks with remaining cells
        if not must_flush and remaining_cells <= max(0, chunks_remaining - 1):
            must_flush = True

        if must_flush:
            _flush("limit")

    if current_cells:
        _flush("tail")

    if not chunks:
        chunks = [geocode_tagged]
        chunk_sizes = [len(geocode_tagged)]

    try:
        avg_chunk = total_geocodes / max(1, len(chunks))
        log_to_gui(
            log_widget,
            (
                f"[chunks] planned {len(chunks):,} chunks; avg geocodes ~{avg_chunk:,.0f}; "
                f"max chunk {max(chunk_sizes or [0]):,}; backlog x{backlog_multiplier:.1f}"
            ),
        )
    except Exception:
        pass

    return chunks


# ----------------------------
# Intersection workers
# ----------------------------
_POOL_ASSETS = None
_POOL_TYPES  = None
_ASSET_SOFT_LIMIT = 200_000
_GEOCODE_SOFT_LIMIT = 160

def _intersect_pool_init(asset_df, geom_types, parts_dir, asset_soft_limit, geocode_soft_limit):
    global _POOL_ASSETS, _POOL_TYPES, _PARTS_DIR, _ASSET_SOFT_LIMIT, _GEOCODE_SOFT_LIMIT
    _POOL_ASSETS = asset_df
    _POOL_TYPES  = geom_types
    _PARTS_DIR   = Path(parts_dir)
    _ASSET_SOFT_LIMIT = int(asset_soft_limit)
    _GEOCODE_SOFT_LIMIT = int(geocode_soft_limit)
    try:
        if psutil is not None:
            p = psutil.Process()
            if os.name == "nt":
                p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:
                os.nice(5)
    except Exception:
        pass

def _intersection_worker(args):
    idx, geocode_chunk = args
    logs = []

    def write_parts(gdf):
        nonlocal logs
        try:
            if 'geometry' in gdf.columns:
                try:
                    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
                except Exception:
                    pass
            fname = f"part_{os.getpid()}_{uuid.uuid4().hex}.parquet"
            path = _PARTS_DIR / fname
            gdf.to_parquet(path, index=False)
            gc.collect()
            return [str(path)]
        except Exception:
            if len(gdf) <= 1:
                raise
            logs.append(f"Chunk {idx}: output write failed for {len(gdf)} rows; splitting into halves.")
            mid = len(gdf) // 2
            gdf1 = gdf.iloc[:mid].reset_index(drop=True)
            gdf2 = gdf.iloc[mid:].reset_index(drop=True)
            return write_parts(gdf1) + write_parts(gdf2)

    def join_with_asset_subset(geocode_gdf, asset_df, predicate):
        try:
            res = gpd.sjoin(geocode_gdf, asset_df, how='inner', predicate=predicate)
            res.drop(columns=[c for c in ['index_right','geometry_wkb','geometry_wkb_1','process'] if c in res.columns],
                     inplace=True, errors='ignore')
            try:
                res = res[~res.geometry.is_empty & res.geometry.notna()]
            except Exception:
                pass
            return res
        except Exception as e:
            msg = str(e).lower()
            if any(w in msg for w in ["realloc", "memory", "failed", "bad_alloc", "invalid argument"]) and len(asset_df) > 1:
                logs.append(f"Chunk {idx}: join failed for asset subset size {len(asset_df)}; splitting assets.")
                mid = len(asset_df) // 2
                subset1 = asset_df.iloc[:mid]
                subset2 = asset_df.iloc[mid:]
                res1 = join_with_asset_subset(geocode_gdf, subset1, predicate)
                res2 = join_with_asset_subset(geocode_gdf, subset2, predicate)
                if res1 is None and res2 is None:
                    return None
                if res1 is None or res1.empty:
                    return res2
                if res2 is None or res2.empty:
                    return res1
                out = pd.concat([res1, res2], ignore_index=True)
                gc.collect()
                return out
            else:
                raise

    def join_geocode_assets(geocode_gdf):
        if geocode_gdf.empty:
            return gpd.GeoDataFrame(geometry=[], crs=_POOL_ASSETS.crs)

        minx, miny, maxx, maxy = geocode_gdf.total_bounds
        candidate_idx = list(_POOL_ASSETS.sindex.intersection((minx, miny, maxx, maxy)))
        asset_subset = _POOL_ASSETS.iloc[candidate_idx] if candidate_idx else _POOL_ASSETS.iloc[:0]

        if len(asset_subset) > _ASSET_SOFT_LIMIT and len(geocode_gdf) > 1:
            logs.append(f"Chunk {idx}: {len(asset_subset):,} candidate assets for {len(geocode_gdf)} geocodes — splitting geocodes pre-emptively.")
            mid = len(geocode_gdf) // 2
            left = geocode_gdf.iloc[:mid].reset_index(drop=True)
            right = geocode_gdf.iloc[mid:].reset_index(drop=True)
            res_left = join_geocode_assets(left)
            res_right = join_geocode_assets(right)
            parts = []
            if res_left is not None and not res_left.empty:
                parts.append(res_left)
            if res_right is not None and not res_right.empty:
                parts.append(res_right)
            if not parts:
                return gpd.GeoDataFrame(geometry=[], crs=_POOL_ASSETS.crs)
            out = pd.concat(parts, ignore_index=True)
            gc.collect()
            return gpd.GeoDataFrame(out, geometry='geometry', crs=_POOL_ASSETS.crs)

        if len(geocode_gdf) > _GEOCODE_SOFT_LIMIT:
            logs.append(f"Chunk {idx}: geocode subset size {len(geocode_gdf)} > {_GEOCODE_SOFT_LIMIT}; splitting.")
            mid = len(geocode_gdf) // 2
            left = geocode_gdf.iloc[:mid].reset_index(drop=True)
            right = geocode_gdf.iloc[mid:].reset_index(drop=True)
            res_left = join_geocode_assets(left)
            res_right = join_geocode_assets(right)
            parts = []
            if res_left is not None and not res_left.empty:
                parts.append(res_left)
            if res_right is not None and not res_right.empty:
                parts.append(res_right)
            if not parts:
                return gpd.GeoDataFrame(geometry=[], crs=_POOL_ASSETS.crs)
            out = pd.concat(parts, ignore_index=True)
            gc.collect()
            return gpd.GeoDataFrame(out, geometry='geometry', crs=_POOL_ASSETS.crs)

        results_list = []
        for gt in _POOL_TYPES:
            af = asset_subset[asset_subset.geometry.geom_type == gt]
            if af.empty:
                continue
            predicate = 'contains' if gt == 'Point' else 'intersects'
            try:
                res = gpd.sjoin(geocode_gdf, af, how='inner', predicate=predicate)
                res.drop(columns=[c for c in ['index_right','geometry_wkb','geometry_wkb_1','process'] if c in res.columns],
                         inplace=True, errors='ignore')
                try:
                    res = res[~res.geometry.is_empty & res.geometry.notna()]
                except Exception:
                    pass
                if not res.empty:
                    results_list.append(res)
            except Exception as e:
                msg = str(e).lower()
                if any(w in msg for w in ["realloc", "memory", "failed", "bad_alloc", "invalid argument"]):
                    if len(geocode_gdf) > 1:
                        logs.append(f"Chunk {idx}: join failed for {len(geocode_gdf)} geocodes; splitting geocodes.")
                        mid = len(geocode_gdf) // 2
                        left = geocode_gdf.iloc[:mid].reset_index(drop=True)
                        right = geocode_gdf.iloc[mid:].reset_index(drop=True)
                        res_left = join_geocode_assets(left)
                        res_right = join_geocode_assets(right)
                        if res_left is not None and not res_left.empty:
                            results_list.append(res_left)
                        if res_right is not None and not res_right.empty:
                            results_list.append(res_right)
                        continue
                    elif len(af) > 1:
                        logs.append(f"Chunk {idx}: join failed for 1 geocode with {len(af)} assets; splitting assets.")
                        mid = len(af) // 2
                        a1 = af.iloc[:mid]
                        a2 = af.iloc[mid:]
                        res1 = join_with_asset_subset(geocode_gdf, a1, predicate)
                        res2 = join_with_asset_subset(geocode_gdf, a2, predicate)
                        if res1 is not None and not res1.empty:
                            results_list.append(res1)
                        if res2 is not None and not res2.empty:
                            results_list.append(res2)
                        continue
                    else:
                        raise
                else:
                    raise

        if not results_list:
            return gpd.GeoDataFrame(geometry=[], crs=_POOL_ASSETS.crs)
        combined = pd.concat(results_list, ignore_index=True)
        gc.collect()
        return gpd.GeoDataFrame(combined, geometry='geometry', crs=_POOL_ASSETS.crs)

    try:
        final_gdf = join_geocode_assets(geocode_chunk)
        if final_gdf is None or final_gdf.empty:
            return (idx, 0, None, None, logs)
        paths = write_parts(final_gdf)
        total_rows = len(final_gdf)
        if logs and paths and len(paths) > 1:
            logs.append(f"Chunk {idx}: output split into {len(paths)} files due to size.")
        return (idx, total_rows, paths, None, logs)
    except Exception as e:
        err_msg = str(e)
        if len(geocode_chunk) == 1:
            try:
                code_val = geocode_chunk.iloc[0].get('code') or geocode_chunk.iloc[0].get('id') or ''
                if code_val:
                    err_msg = f"Geocode {code_val}: {err_msg}"
            except Exception:
                pass
        return (idx, 0, None, err_msg, logs)

# ----------------------------
# Memory heuristics
# ----------------------------
def _estimate_worker_memory_gb(asset_df: gpd.GeoDataFrame,
                               geocode_df: gpd.GeoDataFrame) -> tuple[float | None, float | None]:
    """
    Approximate memory footprint for one worker when dataframes are pickled
    to child processes (Windows spawn). Returns (~GB per worker, ~GB of data).
    """
    def _estimate_gdf(gdf: gpd.GeoDataFrame) -> float:
        if gdf is None or len(gdf) == 0:
            return 0.0
        try:
            raw_bytes = float(gdf.memory_usage(deep=True).sum())
        except Exception:
            raw_bytes = 0.0
        try:
            geom = gdf.geometry
        except Exception:
            geom = None
        if geom is not None:
            try:
                n = len(geom)
                if n:
                    sample = geom.head(min(n, 500))
                    sizes = []
                    for g in sample:
                        try:
                            sizes.append(len(g.wkb))
                        except Exception:
                            pass
                    if sizes:
                        avg = sum(sizes) / len(sizes)
                        raw_bytes += avg * n  # scale sampled WKB sizes to full column
            except Exception:
                pass
        return raw_bytes / (1024 ** 3)

    try:
        total_gb = _estimate_gdf(asset_df) + _estimate_gdf(geocode_df)
    except Exception:
        total_gb = 0.0

    if total_gb <= 0:
        return None, None

    # Each worker holds full assets+geocodes; add overhead for spatial index/geometry copies
    per_worker_gb = max(0.75, total_gb * 1.5)
    return per_worker_gb, total_gb

# ----------------------------
# PROCESS: build tbl_stacked
# ----------------------------
def process_tbl_stacked(cfg: configparser.ConfigParser,
                        working_epsg: str,
                        cell_size_m: int,
                        max_workers: int,
                        approx_gb_per_worker: float,
                        mem_target_frac: float,
                        asset_soft_limit: int,
                        geocode_soft_limit: int):
    log_to_gui(log_widget, f"GeoParquet folder: {gpq_dir()}")
    log_to_gui(log_widget, "Building analysis table (tbl_stacked)…")
    update_progress(10)

    assets   = read_parquet_or_empty("tbl_asset_object")
    geocodes = read_parquet_or_empty("tbl_geocode_object")
    groups   = read_parquet_or_empty("tbl_asset_group")

    if assets.empty:
        log_to_gui(log_widget, "ERROR: Missing or empty tbl_asset_object.parquet; aborting stacked build.")
        return
    if geocodes.empty:
        log_to_gui(log_widget, "ERROR: Missing or empty tbl_geocode_object.parquet; aborting stacked build.")
        return

    _log_memory_snapshot(
        "tbl_stacked:start",
        {"assets": f"{len(assets):,}", "geocodes": f"{len(geocodes):,}"},
        force=True,
    )

    if assets.crs is None:
        assets.set_crs(f"EPSG:{working_epsg}", inplace=True)
    if geocodes.crs is None:
        geocodes.set_crs(f"EPSG:{working_epsg}", inplace=True)

    if not groups.empty:
        cols = ['id','name_gis_assetgroup','total_asset_objects','importance',
                'susceptibility','sensitivity','sensitivity_code','sensitivity_description']
        keep = [c for c in cols if c in groups.columns]
        if keep:
            assets = assets.merge(groups[keep], left_on='ref_asset_group', right_on='id', how='left')

    est_worker_gb, est_data_gb = _estimate_worker_memory_gb(assets, geocodes)
    effective_worker_gb = max(0.5, approx_gb_per_worker)
    if est_worker_gb:
        effective_worker_gb = max(effective_worker_gb, est_worker_gb)
        try:
            log_to_gui(log_widget, f"Memory estimate: data ~{est_data_gb:.2f} GB; using ~{effective_worker_gb:.2f} GB/worker cap.")
        except Exception:
            pass

    update_progress(20)
    _ = assets.sindex; _ = geocodes.sindex

    cpu_cap = 4
    try:
        cpu_cap = max(1, multiprocessing.cpu_count())
    except Exception:
        pass

    if max_workers == 0:
        max_workers = cpu_cap
        log_to_gui(log_widget, f"Number of workers determined by system: {max_workers}")
    else:
        max_workers = max(1, max_workers)
        log_to_gui(log_widget, f"Number of workers set in config: {max_workers}")

    auto_min_workers = max(1, cfg_get_int(cfg, "auto_workers_min", 1))
    auto_max_override = cfg_get_int(cfg, "auto_workers_max", 0)
    auto_max_workers = max(auto_min_workers, auto_max_override) if auto_max_override > 0 else max_workers

    target_chunk_rows = cfg_get_int(cfg, "target_geocodes_per_chunk", 5000)
    backlog_multiplier = max(1.0, cfg_get_float(cfg, "chunk_backlog_multiplier", 2.5))
    est_chunk_count = None
    if target_chunk_rows > 0:
        est_chunk_count = max(1, math.ceil(len(geocodes) / max(target_chunk_rows, 1)))
        est_chunk_count = max(est_chunk_count, int(est_chunk_count * backlog_multiplier))

    try:
        if psutil is not None:
            vm = psutil.virtual_memory()
            avail_gb = vm.available / (1024**3)
            headroom_gb = max(0.5, cfg_get_float(cfg, "mem_headroom_gb", 1.5))
            avail_after_headroom = max(0.5, avail_gb - headroom_gb)
            budget_gb = max(0.5, avail_after_headroom * mem_target_frac)
            allowed = max(1, int(budget_gb // max(0.5, effective_worker_gb)))
            log_to_gui(
                log_widget,
                f"[workers] RAM avail ~{avail_gb:.1f} GB -> budget ~{budget_gb:.1f} GB (headroom {headroom_gb:.1f} GB, {effective_worker_gb:.1f} GB/worker) => {allowed} workers",
            )
            max_workers = min(max_workers, allowed)
    except Exception:
        pass

    if est_chunk_count is not None:
        max_workers = min(max_workers, max(est_chunk_count, auto_min_workers))

    max_workers = min(max_workers, auto_max_workers)
    if max_workers < auto_min_workers and (auto_max_override == 0 or auto_min_workers <= cpu_cap):
        max_workers = auto_min_workers

    max_workers = max(1, min(max_workers, cpu_cap))
    log_to_gui(log_widget, f"[workers] final count: {max_workers}")

    _ = intersect_assets_geocodes(assets, geocodes, cell_size_m, max_workers,
                                  asset_soft_limit, geocode_soft_limit)

    try:
        sample = read_parquet_or_empty("tbl_stacked")
        log_to_gui(log_widget, f"tbl_stacked rows (sample read): {len(sample):,}")
    except Exception as e:
        log_to_gui(log_widget, f"tbl_stacked read check failed: {e}")

    update_progress(50)

# ----------------------------
# PROCESS: flatten to tbl_flat
# ----------------------------
def normalize_area_epsg(raw: str) -> str:
    v = (raw or "").strip().upper()
    if v.startswith("EPSG:"):
        v = v.split(":",1)[1]
    try:
        code = int(v)
        if code in (4326, 4258):
            return "EPSG:3035"
        return f"EPSG:{code}"
    except Exception:
        return "EPSG:3035"

# --- NEW: tiny helpers for geodesic area stats (WGS84) ---
def _to_epsg4326(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return gdf
    g = gdf.copy()
    if g.crs is None:
        try:
            cfg = _ensure_cfg()
            epsg = int((cfg["DEFAULT"].get("workingprojection_epsg","4326") or "4326"))
            g = g.set_crs(epsg=epsg, allow_override=True)
        except Exception:
            g = g.set_crs(4326, allow_override=True)
    if str(g.crs).upper() != "EPSG:4326":
        g = g.to_crs(4326)
    return g

def _valid_geom(geom):
    if geom is None or getattr(geom, "is_empty", False):
        return geom
    try:
        if _make_valid is not None:
            return _make_valid(geom)
        g = geom.buffer(0)
        return g if (g is not None and not g.is_empty) else geom
    except Exception:
        return geom

def _geodesic_area_m2(geom) -> float:
    if geom is None or getattr(geom, "is_empty", False):
        return 0.0
    gt = getattr(geom, "geom_type", "")
    try:
        if gt == "Polygon":
            a, _ = _GEOD.geometry_area_perimeter(geom)
            return abs(a)
        elif gt == "MultiPolygon":
            return float(sum(abs(_GEOD.geometry_area_perimeter(p)[0]) for p in geom.geoms))
    except Exception:
        pass
    return 0.0

def _compute_area_stats_from_tbl_flat(tbl_flat: gpd.GeoDataFrame) -> dict:
    """
    Mirror of the UI's compute_stats_by_geodesic_area_from_flat_basic, but done offline.
    Aggregates A–E geodesic areas (km²) for the configured BASIC group.
    """
    labels = list("ABCDE")
    basic = (_basic_group_name() or _DEFAULT_BASIC_GROUP_NAME).strip().lower()
    msg = f'The geocode/partition "{basic}" is missing.'

    if tbl_flat is None or tbl_flat.empty or "name_gis_geocodegroup" not in tbl_flat.columns:
        return {"labels": labels, "values": [0,0,0,0,0], "message": msg}

    df = tbl_flat.copy()
    df["name_gis_geocodegroup"] = df["name_gis_geocodegroup"].astype("string").str.strip().str.lower()
    df = df[df["name_gis_geocodegroup"] == basic]

    if "sensitivity_code_max" in df.columns:
        df["sensitivity_code_max"] = (
            df["sensitivity_code_max"].astype("string").fillna("").str.strip().str.upper()
        )
        df = df[df["sensitivity_code_max"].isin(list("ABCDE"))]
    else:
        df = df.iloc[0:0]

    if df.empty:
        return {"labels": labels, "values": [0,0,0,0,0], "message": msg}

    # Deduplicate by stable id if available, else by geometry WKB
    if "id_geocode_object" in df.columns:
        df = df.drop_duplicates(subset=["id_geocode_object"])
    else:
        try:
            df = df.assign(__wkb__=df.geometry.apply(lambda g: g.wkb if g is not None else None))
            df = df.drop_duplicates(subset=["__wkb__"]).drop(columns=["__wkb__"])
        except Exception:
            df = df.drop_duplicates()

    # Normalize CRS, make valid
    df = _to_epsg4326(df)
    df["geometry"] = df["geometry"].apply(_valid_geom)

    out = []
    for c in labels:
        sub = df[df["sensitivity_code_max"] == c]
        if sub.empty:
            out.append(0.0)
            continue
        a_m2 = float(sub.geometry.apply(_geodesic_area_m2).sum())
        out.append(a_m2 / 1e6)  # km²
    return {"labels": labels, "values": out}

def _write_area_stats_json(stats: dict):
    """
    Writes area stats JSON to <base>/output/area_stats.json (NOT inside the GeoParquet subfolder).
    """
    try:
        out_p = output_dir() / "area_stats.json"
        tmp = out_p.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        os.replace(tmp, out_p)
        log_to_gui(log_widget, f"Wrote {out_p}")
    except Exception as e:
        log_to_gui(log_widget, f"Failed to write area_stats.json: {e}")

def _parse_index_weight_line(text: str, default: list[int]) -> list[int]:
    try:
        if not text:
            return default.copy()
        parts = [int(x.strip()) for x in str(text).replace(";", ",").split(",") if x.strip()]
        values = [max(1, v) for v in parts[:5]]
        while len(values) < 5:
            values.append(default[len(values)])
        return values
    except Exception:
        return default.copy()

def _load_index_weight_settings(cfg: configparser.ConfigParser) -> dict[str, list[int]]:
    settings: dict[str, list[int]] = {}
    defaults = cfg.defaults()
    for key, option in INDEX_WEIGHT_KEYS.items():
        base = INDEX_WEIGHT_DEFAULTS[key]
        raw = defaults.get(option, "")
        settings[key] = _parse_index_weight_line(raw, base)
    return settings

def _compute_index_scores_from_stacked(df: pd.DataFrame, value_col: str, weights: list[int]) -> pd.Series:
    if df.empty or value_col not in df.columns or "code" not in df.columns:
        return pd.Series(dtype="float64")
    tmp = df[["code", value_col]].copy()
    tmp = tmp.dropna(subset=["code", value_col])
    if tmp.empty:
        return pd.Series(dtype="float64")
    vals = pd.to_numeric(tmp[value_col], errors="coerce").round()
    tmp = tmp.assign(value=vals)
    tmp = tmp.dropna(subset=["value"])
    if tmp.empty:
        return pd.Series(dtype="float64")
    max_bucket = len(weights)
    tmp["value"] = tmp["value"].clip(1, max_bucket).astype(int)
    grouped = tmp.groupby(["code", "value"]).size().unstack(fill_value=0)
    scores = pd.Series(0.0, index=grouped.index, dtype="float64")
    for bucket, weight in enumerate(weights, start=1):
        if weight <= 0:
            continue
        col = grouped.get(bucket)
        if col is None:
            continue
        scores += bucket * weight * col
    return scores

def _scale_index_scores(scores: pd.Series, labels: pd.Series | None = None) -> pd.Series:
    if scores.empty:
        return scores
    df = pd.DataFrame({"score": scores})
    if labels is not None:
        df["label"] = labels.reindex(scores.index).fillna("__all__")
    else:
        df["label"] = "__all__"
    scaled = pd.Series(0.0, index=scores.index, dtype="float64")
    for label, group in df.groupby("label"):
        sc = group["score"]
        max_val = float(sc.max())
        if not np.isfinite(max_val) or max_val <= 0:
            scaled.loc[group.index] = 0.0
            continue
        vals = (sc / max_val) * 100.0
        vals = vals.where(sc > 0, 0.0)
        vals = vals.round()
        vals = vals.where(sc == 0, vals.clip(lower=1.0))
        vals = vals.clip(upper=100.0)
        scaled.loc[group.index] = vals
    return scaled

def flatten_tbl_stacked(config_file: Path, working_epsg: str):
    log_to_gui(log_widget, "Building presentation table (tbl_flat)…")

    stacked = read_parquet_or_empty("tbl_stacked")
    log_to_gui(log_widget, f"tbl_stacked rows: {len(stacked):,}")

    if stacked.empty:
        empty_cols = [
            'ref_geocodegroup','name_gis_geocodegroup','code',
            'importance_min','importance_max','importance_code_min','importance_description_min','importance_code_max','importance_description_max',
            'sensitivity_min','sensitivity_max','sensitivity_code_min','sensitivity_description_min','sensitivity_code_max','sensitivity_description_max',
            'susceptibility_min','susceptibility_max','susceptibility_code_min','susceptibility_description_min','susceptibility_code_max','susceptibility_description_max',
            'asset_group_names','asset_groups_total','area_m2','assets_overlap_total',
            'geometry'
        ]
        gdf_empty = gpd.GeoDataFrame(columns=empty_cols, geometry='geometry', crs=f"EPSG:{working_epsg}")
        write_parquet("tbl_flat", gdf_empty)

        # NEW: produce a zeroed area_stats.json as well
        try:
            stats = {"labels": list("ABCDE"), "values": [0,0,0,0,0], "message": f'The geocode/partition "{_basic_group_name()}" is missing.'}
            log_to_gui(log_widget, "Computing area stats (empty dataset)…")
            update_progress(87)
            _write_area_stats_json(stats)
            update_progress(89)
        except Exception as e:
            log_to_gui(log_widget, f"Area stats (empty) failed: {e}")
        return

    if stacked.crs is None:
        stacked.set_crs(f"EPSG:{working_epsg}", inplace=True)

    # Ensure numeric bases exist
    bases = ["importance","sensitivity","susceptibility"]
    for b in bases:
        if b in stacked.columns:
            stacked[b] = pd.to_numeric(stacked[b], errors="coerce")
        else:
            stacked[b] = pd.Series(pd.NA, index=stacked.index, dtype="Float64")

    if "name_gis_assetgroup" not in stacked.columns:
        # Keep downstream aggregations happy even when legacy datasets lack the name column
        stacked["name_gis_assetgroup"] = pd.Series(pd.NA, index=stacked.index, dtype="string")

    # --------- Category/Rank helpers for choosing winners ---------
    ranges_map, desc_map, _order = read_class_ranges(config_file)

    def _pick_base_category(df: pd.DataFrame, base: str) -> pd.Series:
        """
        Preferred category source (per-row):
          1) base-specific code column: f"{base}_code"
          2) generic 'category' column
          3) derive from numeric via ranges map
        """
        if f"{base}_code" in df.columns:
            cat = df[f"{base}_code"].astype("string").str.strip().str.upper()
        elif "category" in df.columns:
            cat = df["category"].astype("string").str.strip().str.upper()
        else:
            cat = df[base].apply(lambda v: map_num_to_code(v, ranges_map)).astype("string").str.upper()
        return cat

    def _pick_rank(df: pd.DataFrame, base: str) -> pd.Series:
        """
        Tie-break rank (higher is better), only if available:
          1) per-base rank f"{base}_category_rank"
          2) global 'category_rank'
        If none exist -> all zeros (so we fall back to alphabetical category).
        """
        if f"{base}_category_rank" in df.columns:
            rnk = pd.to_numeric(df[f"{base}_category_rank"], errors="coerce")
        elif "category_rank" in df.columns:
            rnk = pd.to_numeric(df["category_rank"], errors="coerce")
        else:
            rnk = pd.Series(0.0, index=df.index, dtype="float64")
        return rnk.fillna(0.0)

    def _select_extreme(df: pd.DataFrame, base: str, pick: str) -> pd.DataFrame:
        val = pd.to_numeric(df[base], errors="coerce")
        cat = _pick_base_category(df, base)
        rnk = _pick_rank(df, base)
        tmp = pd.DataFrame({
            "code": df["code"],
            "val":  val,
            "cat":  cat,
            "rnk":  rnk
        })
        if pick == "max":
            tmp["sv"] = tmp["val"].fillna(-np.inf)  # ignore NaNs for maxima
            tmp = tmp.sort_values(["code","sv","rnk","cat"],
                                  ascending=[True, False, False, True],
                                  kind="mergesort")
        else:  # "min"
            tmp["sv"] = tmp["val"].fillna(np.inf)   # ignore NaNs for minima
            tmp = tmp.sort_values(["code","sv","rnk","cat"],
                                  ascending=[True, True, False, True],
                                  kind="mergesort")
        win = tmp.drop_duplicates(subset=["code"], keep="first")[["code","val","cat"]].copy()
        return win.rename(columns={"val": f"{base}_{pick}", "cat": f"{base}_code_{pick}"})

    # --------- Per-tile metadata (sum/concat-like behavior stays) ---------
    keys = ["code"]
    gmeta = stacked.groupby(keys, dropna=False).agg({
        "ref_geocodegroup": "first",
        "name_gis_geocodegroup": "first",
        "geometry": "first",
        "ref_asset_group": pd.Series.nunique,
        "name_gis_assetgroup": (lambda s: ", ".join(pd.Series(s).dropna().astype(str).unique()))
    }).rename(columns={"ref_asset_group":"asset_groups_total",
                       "name_gis_assetgroup":"asset_group_names"}).reset_index()

    goverlap = stacked.groupby(keys, dropna=False).size().to_frame("assets_overlap_total").reset_index()

    # --------- Build winners for each base and extreme ---------
    pieces = [gmeta, goverlap]
    for b in bases:
        pieces.append(_select_extreme(stacked, b, "min"))
        pieces.append(_select_extreme(stacked, b, "max"))

    # Merge to single row per tile
    tbl_flat = pieces[0]
    for p in pieces[1:]:
        tbl_flat = tbl_flat.merge(p, on="code", how="left")

    # Descriptions from chosen categories
    for b in bases:
        tbl_flat[f"{b}_description_min"] = tbl_flat[f"{b}_code_min"].map(lambda k: desc_map.get(k, None))
        tbl_flat[f"{b}_description_max"] = tbl_flat[f"{b}_code_max"].map(lambda k: desc_map.get(k, None))

    # GeoDataFrame and area
    tbl_flat = gpd.GeoDataFrame(tbl_flat, geometry="geometry", crs=stacked.crs)

    # read area projection from the same flat config we were given
    try:
        cfg_local = read_config(config_file)
    except Exception:
        cfg_local = configparser.ConfigParser()
    try:
        area_epsg = normalize_area_epsg(cfg_local["DEFAULT"].get("area_projection_epsg","3035"))
    except Exception:
        area_epsg = "EPSG:3035"

    try:
        metric = tbl_flat.to_crs(area_epsg)
        tbl_flat["area_m2"] = metric.geometry.area.astype("float64").round().astype("Int64")
    except Exception as e:
        log_to_gui(log_widget, f"Area computation failed in {area_epsg}: {e}; using EPSG:3035")
        metric = tbl_flat.to_crs("EPSG:3035")
        tbl_flat["area_m2"] = metric.geometry.area.astype("float64").round().astype("Int64")

    for col in ("asset_groups_total", "assets_overlap_total"):
        if col not in tbl_flat.columns:
            tbl_flat[col] = 1

    # New importance/sensitivity indexes based on tbl_stacked weights
    try:
        index_weights = _load_index_weight_settings(cfg_local)
    except Exception:
        index_weights = INDEX_WEIGHT_DEFAULTS.copy()
    importance_scores = _compute_index_scores_from_stacked(stacked, "importance", index_weights["importance"])
    sensitivity_scores = _compute_index_scores_from_stacked(stacked, "sensitivity", index_weights["sensitivity"])
    group_map = None
    try:
        if "code" in tbl_flat.columns and "name_gis_geocodegroup" in tbl_flat.columns:
            unique_groups = tbl_flat[["code", "name_gis_geocodegroup"]].drop_duplicates(subset=["code"])
            group_map = unique_groups.set_index("code")["name_gis_geocodegroup"].astype("string")
    except Exception:
        group_map = None
    importance_norm = _scale_index_scores(importance_scores, group_map).rename("index_importance")
    sensitivity_norm = _scale_index_scores(sensitivity_scores, group_map).rename("index_sensitivity")
    if not importance_norm.empty:
        tbl_flat = tbl_flat.merge(importance_norm, left_on="code", right_index=True, how="left")
    else:
        tbl_flat["index_importance"] = 0
    if not sensitivity_norm.empty:
        tbl_flat = tbl_flat.merge(sensitivity_norm, left_on="code", right_index=True, how="left")
    else:
        tbl_flat["index_sensitivity"] = 0
    for col in ("index_importance", "index_sensitivity"):
        tbl_flat[col] = pd.to_numeric(tbl_flat[col], errors="coerce").fillna(0).round().astype("Int64")

    preferred = [
        'ref_geocodegroup','name_gis_geocodegroup','code',
        'importance_min','importance_max','importance_code_min','importance_description_min','importance_code_max','importance_description_max',
        'sensitivity_min','sensitivity_max','sensitivity_code_min','sensitivity_description_min','sensitivity_code_max','sensitivity_description_max',
        'susceptibility_min','susceptibility_max','susceptibility_code_min','susceptibility_description_min','susceptibility_code_max','susceptibility_description_max',
        'asset_group_names','asset_groups_total','area_m2','assets_overlap_total',
        'index_importance','index_sensitivity',
        'geometry'
    ]
    for c in preferred:
        if c not in tbl_flat.columns:
            tbl_flat[c] = pd.NA
    tbl_flat = tbl_flat[preferred]

    write_parquet("tbl_flat", tbl_flat)
    log_to_gui(log_widget, f"tbl_flat saved with {len(tbl_flat):,} rows.")

    # NEW: area stats JSON (geodesic, WGS84) for the map UI
    try:
        log_to_gui(log_widget, "Computing geodesic area stats for basic mosaic (A–E)…")
        update_progress(87)
        stats = _compute_area_stats_from_tbl_flat(tbl_flat)
        _write_area_stats_json(stats)
        update_progress(89)
    except Exception as e:
        log_to_gui(log_widget, f"Area stats computation failed: {e}")

    # Backfill area_m2 to stacked parts
    try:
        area_map = tbl_flat[['code','area_m2']].dropna().drop_duplicates(subset=['code'])
        area_map['code'] = area_map['code'].astype(str)
        ds_dir = _dataset_dir("tbl_stacked")
        if ds_dir.exists() and ds_dir.is_dir():
            part_files = sorted([p for p in ds_dir.iterdir() if p.suffix.lower()=='.parquet'])
            touched = 0
            started_at = time.time()
            for i, pp in enumerate(part_files, start=1):
                try:
                    part = gpd.read_parquet(pp)
                    if 'code' not in part.columns:
                        continue
                    part['code'] = part['code'].astype(str)
                    merged = part.merge(area_map, on='code', how='left', suffixes=('','_flat'))
                    if 'area_m2_flat' in merged.columns:
                        merged['area_m2'] = merged['area_m2_flat']
                        merged.drop(columns=['area_m2_flat'], inplace=True)
                    gpd.GeoDataFrame(merged, geometry=part.geometry.name, crs=part.crs).to_parquet(pp, index=False)
                    touched += len(part)
                except Exception as e:
                    log_to_gui(log_widget, f"[Backfill] Part {pp.name} skipped: {e}")
                if i % 25 == 0 or i == len(part_files):
                    elapsed = time.time() - started_at
                    log_to_gui(log_widget, f"[Backfill] {i}/{len(part_files)} parts updated • rows seen: {touched:,} • elapsed {elapsed/60:.1f} min")
            log_to_gui(log_widget, f"Backfilled area_m2 to tbl_stacked dataset ({len(part_files)} parts).")
    except Exception as e:
        log_to_gui(log_widget, f"Backfill to tbl_stacked failed: {e}")

    update_progress(90)
    try:
        # Keep the last snapshot of cells/home_bounds so the map still shows tiles after finishing
        prev = {}
        try:
            with open(_status_path(), "r", encoding="utf-8") as f:
                prev = json.load(f) or {}
        except Exception:
            prev = {}

        chunks_total = int(prev.get("chunks_total", 0) or 0)
        done = int(prev.get("done", chunks_total) or 0)

        _write_status_atomic({
            "phase": "done",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "chunks_total": chunks_total,
            "done": done,
            "running": [],
            "cells": prev.get("cells", []),           # <-- preserve
            "home_bounds": prev.get("home_bounds")    # <-- preserve
        })
    except Exception:
        pass


# ----------------------------
# Top-level process
# ----------------------------
def process_all(config_file: Path):
    try:
        cfg = read_config(config_file)
        set_global_cfg(cfg)  # <-- make parquet_folder available to gpq_dir()

        try:
            global HEARTBEAT_SECS
            HEARTBEAT_SECS = cfg_get_int(cfg, "heartbeat_secs", HEARTBEAT_SECS)
        except Exception:
            pass

        working_epsg = str(cfg["DEFAULT"].get("workingprojection_epsg","4326")).strip()
        max_workers = cfg_get_int(cfg, "max_workers", 0)
        cell_size   = cfg_get_int(cfg, "cell_size", 18000)
        chunk_size  = cfg_get_int(cfg, "chunk_size", 40000)

        approx_gb_per_worker = cfg_get_float(cfg, "approx_gb_per_worker", 8.0)
        mem_target_frac      = cfg_get_float(cfg, "mem_target_frac", 0.75)
        asset_soft_limit     = cfg_get_int(cfg, "asset_soft_limit", 200000)
        geocode_soft_limit   = cfg_get_int(cfg,  "geocode_soft_limit", 160)

        chunk_display = f"{chunk_size:,}" if chunk_size > 0 else "auto"
        worker_display = "auto" if max_workers == 0 else f"{max_workers}"
        log_to_gui(log_widget,
             ("Config snapshot -> cell_size=%s m, chunk_size=%s rows, max_workers=%s, "
                "approx_gb_per_worker=%.2f, mem_target_frac=%.2f, asset_soft_limit=%s, geocode_soft_limit=%s") %
               (f"{cell_size:,}", chunk_display, worker_display, approx_gb_per_worker, mem_target_frac,
                f"{asset_soft_limit:,}", f"{geocode_soft_limit:,}"))
        _log_memory_snapshot("config-loaded",
                     {"cell_m": cell_size,
                      "chunk_size": chunk_display,
                      "max_workers": worker_display,
                      "approx_gb_per_worker": f"{approx_gb_per_worker:.2f}"},
                     force=True)

        log_to_gui(log_widget, "[Stage 1/4] Preparing workspace and status files…")
        cleanup_outputs(); update_progress(5)
        _init_idle_status()

        log_to_gui(log_widget, "[Stage 2/4] Building stacked dataset (intersections & classification)…")
        process_tbl_stacked(cfg, working_epsg, cell_size, max_workers,
                            approx_gb_per_worker, mem_target_frac,
                            asset_soft_limit, geocode_soft_limit)

        log_to_gui(log_widget, "[Stage 3/4] Flattening outputs, computing stats, and refreshing status…")
        flatten_tbl_stacked(config_file, working_epsg); update_progress(95)

        for temp_dir in ["__stacked_parts", "__grid_assign_in", "__grid_assign_out"]:
            _rm_rf(_dataset_dir(temp_dir))
        
        log_to_gui(log_widget, "Core processing (stages 1-3) finished. Preparing raster tiles stage…")
    except Exception as e:
        log_to_gui(log_widget, f"Error during processing: {e}")
        raise

# ----------------------------
# Minimap (Leaflet in pywebview) — helper process
# ----------------------------
MAP_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Chunk minimap</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body { height:100%; margin:0; }
  #map { height:100%; width:100%; }
  .topbar {
    position:absolute; top:0; left:0; right:0; z-index:1000;
    background:rgba(255,255,255,0.92);
    border-bottom:1px solid #e5e7eb;
    padding:6px 10px;
    font:12px/1.35 system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
    color:#0f172a;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  }
  .legend {
    position:absolute; bottom:10px; left:10px;
    background:rgba(255,255,255,0.9); padding:6px 8px; border-radius:6px;
    font:12px/1.3 system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
  }
  .swatch { display:inline-block; width:12px; height:12px; margin-right:6px; vertical-align:middle; border:1px solid #999; }
</style>
</head>
<body>
<div id="map"></div>

<!-- Stats line -->
<div class="topbar" id="facts">Loading…</div>

<!-- Legend -->
<div class="legend">
  <div><span class="swatch" style="background: rgba(34,197,94,0.22); border-color: transparent;"></span>Done</div>
    <div><span class="swatch" style="background: rgba(255,138,0,0.28); border-color: transparent;"></span>Running</div>
    <div><span class="swatch" style="background: transparent; border-color: #ff8c00;"></span>Queued</div>
</div>

<script>
let MAP, GROUP;

function init(){
  try {
    MAP = L.map('map', { zoomSnap: 0.25 });
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom:19, attribution:'© OpenStreetMap'}).addTo(MAP);
    GROUP = L.featureGroup().addTo(MAP);
    MAP.setView([59.9,10.75], 4); // harmless default
    refresh();
    setInterval(refresh, 5000); //
  } catch(e){
    safeSetFacts("Init error: " + (e && e.message ? e.message : e));
  }
}

function safeSetFacts(text){
  try{
    var f = document.getElementById('facts');
    if (f) f.textContent = String(text || "");
  }catch(_){}
}

function boundsFromSWNE(swne){
  if (!Array.isArray(swne) || swne.length!==4) return null;
  const s = Number(swne[0]); const w = Number(swne[1]); const n = Number(swne[2]); const e = Number(swne[3]);
  if (![s,w,n,e].every(Number.isFinite)) return null;
  return L.latLngBounds([s,w],[n,e]);
}

function styleFor(state){
  if (state === 'done') {
    return { stroke:false, fill:true,  fillColor:'#22c55e', fillOpacity:0.25 };
  }
  if (state === 'running') {
        return { stroke:false, fill:true,  fillColor:'#ff8c00', fillOpacity:0.9 };
  }
    return { stroke:true, color:'#ff8c00', weight:1, opacity:0.85, fill:false };
}

function summarize(status){
  const cells = (status && Array.isArray(status.cells)) ? status.cells : [];
  let queued=0, running=0, done=0;
  for (const c of cells){
    const st = (c && c.state) ? String(c.state) : '';
    if (st === 'done') done++;
    else if (st === 'running') running++;
    else queued++;
  }
  const total = cells.length;
  const tsRaw = status && status.updated_at ? status.updated_at : null;
  let ts = tsRaw;
  try { if (tsRaw) ts = new Date(tsRaw).toLocaleString(); } catch(_){}
  return { total, queued, running, done, ts };
}

function updateFacts(status){
  const s = summarize(status || {});
  const parts = [
    (s.ts ? `Updated: ${s.ts}` : 'Updated: —'),
    `Cells: ${s.total}`,
    `queued ${s.queued}`,
    `processing ${s.running}`,
    `done ${s.done}`
  ];
  safeSetFacts(parts.join('  •  '));
}

function render(status){
  try{
    GROUP.clearLayers();
  }catch(_){}
  updateFacts(status);

  const hb = status && status.home_bounds;
  const cells = (status && Array.isArray(status.cells)) ? status.cells : [];

  if (cells.length){
    for (let i=0; i<cells.length; i++){
      const c = cells[i] || {};
      const b = boundsFromSWNE(c.bbox);
      if (!b) continue;
      const r = L.rectangle(b, styleFor(c.state));
      const human = (c.state === 'running') ? 'processing' : (c.state || 'queued');
      const nrows = (c.n!=null && isFinite(Number(c.n))) ? String(c.n) : '';
      const tip = `Cell #${String(c.id ?? '')}<br>State: <b>${human}</b>${nrows ? ('<br>Rows: '+nrows) : ''}`;
      try { r.bindTooltip(tip, {sticky:true}); } catch(_){}
      try { r.addTo(GROUP); } catch(_){}
    }
    try { MAP.fitBounds(GROUP.getBounds().pad(0.08)); } catch(_){}
  } else if (Array.isArray(hb) && hb.length===2){
    try { MAP.fitBounds(hb, {padding:[20,20]}); } catch(_){}
  }
}

function refresh(){
  try{
    if (!window.pywebview || !window.pywebview.api){
      safeSetFacts("Waiting for pywebview bridge…");
      return;
    }
    window.pywebview.api.get_status()
      .then(function(st){ render(st || {}); })
      .catch(function(e){ safeSetFacts("Status read error: " + (e && e.message ? e.message : e)); });
  }catch(e){
    safeSetFacts("Refresh error: " + (e && e.message ? e.message : e));
  }
}

document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>
"""


def _map_process_entry(status_path_str: str):
    # Low priority so it never contends with workers
    try:
        if psutil is not None:
            p = psutil.Process()
            if os.name == "nt":
                try: p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                except Exception: p.nice(psutil.IDLE_PRIORITY_CLASS)
            else:
                os.nice(10)
    except Exception:
        pass

    try:
        import webview  # type: ignore
        try: webview.logger.disabled = True
        except Exception: pass
    except Exception:
        return

    class MapApi:
        def __init__(self, status_path: Path):
            self.status_path = str(status_path)
        def get_status(self):
            try:
                with open(self.status_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {"phase":"idle","chunks_total":0,"done":0,"running":[],"cells":[],"home_bounds":None}

    api = MapApi(Path(status_path_str))
    webview.create_window(title="Minimap — chunks", html=MAP_HTML, js_api=api, width=600, height=600)
    try:
        webview.start(gui='edgechromium', debug=False)
    except Exception:
        webview.start(debug=False)

def open_minimap_window():
    """Spawn the map helper process (single instance)."""
    global _MAP_PROC
    if importlib.util.find_spec("webview") is None:
        log_to_gui(log_widget, "Minimap requires 'pywebview' (Edge WebView2). Install it to use the map.")
        return
    if _MAP_PROC is not None and _MAP_PROC.is_alive():
        log_to_gui(log_widget, "Minimap is already open.")
        return
    ctx = multiprocessing.get_context("spawn")
    _MAP_PROC = ctx.Process(target=_map_process_entry, args=(str(_status_path()),), daemon=True)
    _MAP_PROC.start()
    log_to_gui(log_widget, "Opening minimap (separate process)…")

"""
"""

# ------------------------------------------------------------
# Processing worker (separate process) + GUI progress polling
# ------------------------------------------------------------
def _processing_worker_entry(cfg_path_str: str) -> None:
    """Entry point for heavy processing in a separate process.
    Runs process_all without touching GUI state so child can safely create Pools.
    """
    try:
        # Ensure spawn-friendly bootstrap in child as well
        import multiprocessing as _mp
        try:
            _mp.freeze_support()
        except Exception:
            pass
        try:
            _mp.set_start_method("spawn", force=False)
        except Exception:
            pass
    except Exception:
        pass

    try:
        process_all(Path(cfg_path_str))
    except Exception:
        # Child errors are reflected in status/log files; just exit.
        pass

def _start_processing_worker(cfg_path: Path) -> None:
    """Launch processing in a child process so Pools can be used even in frozen GUI runs."""
    global _PROC
    ctx = multiprocessing.get_context("spawn")
    # Important: do NOT daemonize; daemonic processes cannot create child processes (Pools)
    _PROC = ctx.Process(target=_processing_worker_entry, args=(str(cfg_path),))
    _PROC.start()
    try:
        log_to_gui(log_widget, f"Started processing worker (PID {_PROC.pid})")
    except Exception:
        pass

def _poll_progress_periodically(root_obj: tk.Misc, interval_ms: int = 1000) -> None:
    """Periodically read the status JSON (same as minimap) and reflect progress in the GUI bar."""
    def _poll():
        try:
            p = _status_path()
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    st = json.load(f)
                total = int(st.get("chunks_total", 0) or 0)
                done = int(st.get("done", 0) or 0)
                phase = (st.get("phase", "") or "").lower()
                set_progress_stage(_stage_from_phase(phase))
                # Map intersect progress to ~35..50% like the worker side uses
                pct: float | None
                if total > 0 and phase == "intersect":
                    pct = 35.0 + (done / max(total, 1)) * 15.0
                elif phase in ("flatten_pending", "flatten", "writing"):
                    pct = 80.0
                elif phase == "done":
                    pct = 95.0
                elif phase in ("tiles", "tiling", "mbtiles", "tiles_finalizing"):
                    pct = None  # tile stage drives its own progress updates
                elif phase == "completed":
                    pct = 100.0
                else:
                    pct = 10.0
                if pct is not None:
                    update_progress(pct)
        except Exception:
            pass
        finally:
            try:
                # keep polling while worker is alive; run a few extra ticks after
                if _PROC is not None and _PROC.is_alive():
                    root_obj.after(interval_ms, _poll)
                else:
                    # one last delayed refresh
                    root_obj.after(2 * interval_ms, _poll)
            except Exception:
                pass

    try:
        root_obj.after(interval_ms, _poll)
    except Exception:
        pass


def _start_log_tailer(root_obj: tk.Misc,
                      log_path: Path | None = None,
                      interval_ms: int = 750) -> None:
    """
    Periodically tail base_dir()/log.txt (written by the worker process) and
    append new lines to the GUI log widget. This restores live log updates
    when the heavy processing runs in a separate process.
    """
    # Candidate log files: root/log.txt (new) and code/log.txt (legacy)
    try:
        root_log = (log_path if isinstance(log_path, Path) else None) or (base_dir() / "log.txt")
    except Exception:
        root_log = None
    try:
        code_log = base_dir() / "code" / "log.txt"
    except Exception:
        code_log = None

    candidates: list[Path] = []
    if root_log is not None:
        candidates.append(root_log)
    if code_log is not None and code_log != root_log:
        candidates.append(code_log)

    if not candidates:
        return

    # Start reading from current EOF to avoid dumping old logs
    state: dict[str, int] = {}
    for p in candidates:
        try:
            state[str(p)] = p.stat().st_size if p.exists() else 0
        except Exception:
            state[str(p)] = 0

    def _gui_append(line: str) -> None:
        try:
            if log_widget and log_widget.winfo_exists():
                log_widget.insert(tk.END, line + "\n")
                log_widget.see(tk.END)
        except Exception:
            try:
                print(line, flush=True)
            except Exception:
                pass

    def _tail_once():
        try:
            # Read and append any new lines from each candidate file
            for p in candidates:
                try:
                    if not p.exists():
                        continue
                    with open(p, "r", encoding="utf-8", errors="replace") as f:
                        key = str(p)
                        pos = state.get(key, 0)
                        try:
                            f.seek(pos)
                        except Exception:
                            pos = 0
                            f.seek(0)
                        data = f.read()
                        state[key] = f.tell()
                    if data:
                        for line in data.splitlines():
                            if line.strip():
                                _gui_append(line)
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            # Keep tailing while worker is alive; a few extra ticks after
            try:
                if _PROC is not None and _PROC.is_alive():
                    root_obj.after(interval_ms, _tail_once)
                else:
                    root_obj.after(2 * interval_ms, _tail_once)
            except Exception:
                pass

    # Announce attachment once
    try:
        for p in candidates:
            _gui_append(f"[tail] Attached: {p}")
    except Exception:
        pass
    try:
        _tail_once()
    except Exception:
        pass

def intersect_assets_geocodes(asset_data: gpd.GeoDataFrame,
                              geocode_data: gpd.GeoDataFrame,
                              meters_cell: int,
                              max_workers: int,
                              asset_soft_limit: int,
                              geocode_soft_limit: int) -> gpd.GeoDataFrame:
    if asset_data is None or asset_data.empty:
        log_to_gui(log_widget, "No assets; skipping intersection.")
        return gpd.GeoDataFrame(geometry=[], crs=geocode_data.crs if geocode_data is not None else None)
    if geocode_data is None or geocode_data.empty:
        log_to_gui(log_widget, "No geocodes; skipping intersection.")
        return gpd.GeoDataFrame(geometry=[], crs=asset_data.crs)
    try: _ = asset_data.sindex
    except Exception: pass
    try: _ = geocode_data.sindex
    except Exception: pass
    meters_cell = int(max(100, meters_cell))
    log_to_gui(
        log_widget,
        f"[intersect] Assets={len(asset_data):,}, geocodes={len(geocode_data):,}, cell_size={meters_cell:,} m, "
        f"asset_soft_limit={asset_soft_limit:,}, geocode_soft_limit={geocode_soft_limit:,}"
    )
    _log_memory_snapshot(
        "intersect:init",
        {
            "assets": f"{len(asset_data):,}",
            "geocodes": f"{len(geocode_data):,}",
            "cell_m": meters_cell,
            "workers": max_workers,
        },
        force=True,
    )
    tagged = assign_geocodes_to_grid(geocode_data, meters_cell, max_workers)
    if tagged is None or tagged.empty or "grid_cell" not in tagged.columns:
        log_to_gui(log_widget, "Grid tagging failed or returned empty; aborting intersection.")
        return gpd.GeoDataFrame(geometry=[], crs=geocode_data.crs)
    chunks = make_spatial_chunks(tagged, max_workers=max_workers, multiplier=18)
    total_chunks = len(chunks)
    log_to_gui(log_widget, f"Intersecting in {total_chunks} chunks with {max_workers} workers. Heartbeat every {HEARTBEAT_SECS}s.")
    update_progress(35.0)
    chunk_cells = {}
    for i, ch in enumerate(chunks, start=1):
        try: ids = set(int(x) for x in ch['grid_cell'].dropna().unique().tolist())
        except Exception: ids = set()
        chunk_cells[i] = ids
    try:
        g4326 = geocode_data
        if g4326.crs is None: g4326 = g4326.set_crs(4326, allow_override=True)
        if str(g4326.crs).upper() != "EPSG:4326": g4326 = g4326.to_crs(4326)
        minx, miny, maxx, maxy = g4326.total_bounds
        home_bounds = [float(minx), float(miny), float(maxx), float(maxy)]
    except Exception:
        home_bounds = None
    cells_meta = []
    try:
        for cid, bbox in _GRID_BBOX_MAP.items():
            cells_meta.append({"id": int(cid), "state": "queued", "n": 0, "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]})
    except Exception: pass
    tmp_parts = _dataset_dir("__stacked_parts"); _rm_rf(tmp_parts); tmp_parts.mkdir(parents=True, exist_ok=True)
    try: geom_types = sorted(set(asset_data.geometry.geom_type.dropna().unique().tolist()))
    except Exception: geom_types = ["Polygon","MultiPolygon","LineString","Point"]
    progress_state = {"done": 0, "rows": 0, "started_at": time.time()}
    hb_stop = threading.Event()
    def _heartbeat():
        while not hb_stop.wait(HEARTBEAT_SECS):
            try:
                vm_used = None
                rss_gb = None
                if psutil is not None:
                    vm = psutil.virtual_memory(); vm_used = f"{int(vm.percent)}%"
                    try:
                        rss_gb = psutil.Process().memory_info().rss / (1024 ** 3)
                    except Exception:
                        rss_gb = None
                pct = (progress_state["done"] / max(1, total_chunks)) * 100.0
                active_workers = min(max_workers, max(0, total_chunks - progress_state["done"]))
                msg = (
                    f"[heartbeat] {progress_state['done']}/{total_chunks} chunks (~{pct:.2f}%)"
                    f" • rows written: {progress_state['rows']:,}"
                    f" • active workers {active_workers}/{max_workers}"
                )
                if vm_used: msg += f" • RAM used {vm_used}"
                if rss_gb is not None:
                    msg += f" • proc RSS ~{rss_gb:.2f} GB"
                msg += " • ETA ?"
                log_to_gui(log_widget, msg)
            except Exception: pass
    hb_thread = threading.Thread(target=_heartbeat, daemon=True); hb_thread.start()
    try:
        _write_status_atomic({"phase":"intersect","updated_at": datetime.utcnow().isoformat()+"Z","chunks_total": total_chunks,"done":0,"running": list(range(1, min(max_workers, total_chunks)+1)),"cells": cells_meta,"home_bounds": home_bounds})
    except Exception: pass
    def _update_status(done_count:int):
        try:
            running_chunk_ids = list(range(done_count+1, min(done_count+max_workers, total_chunks)+1))
            running_cells = set().union(*(chunk_cells.get(i,set()) for i in running_chunk_ids)) if running_chunk_ids else set()
            done_cells = set().union(*(chunk_cells.get(i,set()) for i in range(1, done_count+1))) if done_count else set()
            id_to_idx = {c["id"]: i for i, c in enumerate(cells_meta)}
            for cid in done_cells:
                i = id_to_idx.get(int(cid))
                if i is not None: cells_meta[i]["state"] = "done"
            for cid in running_cells:
                i = id_to_idx.get(int(cid))
                if i is not None and cells_meta[i]["state"] != "done": cells_meta[i]["state"] = "running"
            _write_status_atomic({"phase":"intersect","updated_at": datetime.utcnow().isoformat()+"Z","chunks_total": total_chunks,"done": done_count,"running": running_chunk_ids,"cells": cells_meta,"home_bounds": home_bounds})
        except Exception: pass
    def _tick_progress(done_count:int, written:int, started_at:float):
        try:
            now = time.time(); elapsed = now - started_at; pct = (done_count / max(1,total_chunks))*100.0; eta = "?"
            if done_count>0:
                est_total = elapsed / done_count * total_chunks
                eta_ts = datetime.now() + timedelta(seconds=max(0.0, est_total - elapsed)); eta = eta_ts.strftime("%H:%M:%S")
                dd = (eta_ts.date() - datetime.now().date()).days
                if dd>0: eta += f" (+{dd}d)"
            active_workers = min(max_workers, max(0, total_chunks - done_count))
            log_to_gui(
                log_widget,
                f"[intersect] {done_count}/{total_chunks} chunks (~{pct:.2f}%)"
                f" • rows written: {written:,}"
                f" • active workers {active_workers}/{max_workers}"
                f" • ETA {eta}",
            )
            update_progress(35.0 + (done_count / max(1,total_chunks)) * 15.0)
            _log_memory_snapshot(
                "intersect:progress",
                {"done": done_count, "total": total_chunks, "rows": f"{written:,}"},
                force=False,
            )
        except Exception: pass
    files = []; written = 0; error_msg = None; iterable = ((i, ch) for i, ch in enumerate(chunks, start=1))
    if _mp_allowed() and max_workers > 1:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=max_workers, initializer=_intersect_pool_init, initargs=(asset_data, geom_types, str(tmp_parts), asset_soft_limit, geocode_soft_limit)) as pool:
            for (idx, nrows, paths, err, logs) in pool.imap_unordered(_intersection_worker, iterable):
                progress_state["done"] += 1; done_count = progress_state["done"]
                if logs:
                    for line in logs: log_to_gui(log_widget, line)
                if err:
                    error_msg = err; log_to_gui(log_widget, f"[intersect] Chunk {idx} failed: {err}")
                    try: pool.terminate()
                    except Exception: pass
                    break
                written += int(nrows or 0); progress_state["rows"] = written
                if paths:
                    if isinstance(paths, list): files.extend(paths)
                    else: files.append(paths)
                _update_status(done_count); _tick_progress(done_count, written, progress_state["started_at"])
                if done_count % 8 == 0: gc.collect()
    else:
        _intersect_pool_init(asset_data, geom_types, str(tmp_parts), asset_soft_limit, geocode_soft_limit)
        for args in iterable:
            (idx, nrows, paths, err, logs) = _intersection_worker(args)
            progress_state["done"] += 1; done_count = progress_state["done"]
            if logs:
                for line in logs: log_to_gui(log_widget, line)
            if err:
                error_msg = err; log_to_gui(log_widget, f"[intersect] Chunk {idx} failed: {err}"); break
            written += int(nrows or 0); progress_state["rows"] = written
            if paths:
                if isinstance(paths, list): files.extend(paths)
                else: files.append(paths)
            _update_status(done_count); _tick_progress(done_count, written, progress_state["started_at"])
            if done_count % 8 == 0: gc.collect()
    try:
        hb_stop.set(); hb_thread.join(timeout=1.5)
    except Exception: pass
    try:
        for c in cells_meta: c["state"] = "done"
        _write_status_atomic({"phase": "flatten_pending" if error_msg is None else "error","updated_at": datetime.utcnow().isoformat()+"Z","chunks_total": total_chunks,"done": progress_state["done"],"running": [],"cells": cells_meta,"home_bounds": home_bounds})
    except Exception: pass
    if error_msg: raise RuntimeError(error_msg)
    if not files:
        log_to_gui(log_widget, "No intersections; tbl_stacked is empty.")
        return gpd.GeoDataFrame(geometry=[], crs=geocode_data.crs)
    final_ds = _dataset_dir("tbl_stacked"); _rm_rf(final_ds)
    try: Path(tmp_parts).rename(final_ds)
    except Exception:
        final_ds.mkdir(parents=True, exist_ok=True)
        for f in Path(tmp_parts).glob("*.parquet"): shutil.move(str(f), str(final_ds / f.name))
        _rm_rf(tmp_parts)
    log_to_gui(log_widget, f"tbl_stacked dataset written as folder with {len(files)} parts and ~{written:,} rows: {final_ds}")
    return gpd.GeoDataFrame(geometry=[], crs=geocode_data.crs)






if __name__ == "__main__":
    # Windows + PyInstaller: ensure child processes bootstrap correctly.
    # This allows the minimap helper process (spawn) to start in frozen builds.
    try:
        import multiprocessing as _mp
        try:
            _mp.freeze_support()
        except Exception:
            pass
        try:
            # Use spawn consistently; ignore if already set by the environment
            _mp.set_start_method("spawn", force=False)
        except Exception:
            pass
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Process stacked/flat GeoParquet with A..E categorisation")
    parser.add_argument('--original_working_directory', required=False, help='Path to running folder')
    parser.add_argument('--headless', action='store_true', help='Run without GUI (CLI mode)')
    args = parser.parse_args()
    original_working_directory = args.original_working_directory or os.getcwd()
    if "system" in os.path.basename(original_working_directory).lower():
        original_working_directory = os.path.abspath(os.path.join(original_working_directory, os.pardir))

    # Flat config
    cfg_path = base_dir() / "config.ini"
    cfg = read_config(cfg_path)
    set_global_cfg(cfg)  # make DEFAULT.parquet_folder effective everywhere

    # Zooms for raster tiles (from config.ini)
    tiles_minzoom = cfg_get_int(cfg, "tiles_minzoom", 6)
    tiles_maxzoom = cfg_get_int(cfg, "tiles_maxzoom", 12)

    # Sanitize: clamp to [0, 22] and ensure min <= max
    tiles_minzoom = max(0, min(tiles_minzoom, 22))
    tiles_maxzoom = max(tiles_minzoom, min(tiles_maxzoom, 22))

    try:
        HEARTBEAT_SECS = cfg_get_int(cfg, "heartbeat_secs", HEARTBEAT_SECS)
    except Exception:
        pass

    MINIMAP_STATUS_PATH = gpq_dir() / "__chunk_status.json"
    _init_idle_status()  # tiny baseline

    if args.headless:
        try:
            process_all(cfg_path)
        except Exception:
            sys.exit(1)
        sys.exit(0)

    ttk_theme = cfg['DEFAULT'].get('ttk_bootstrap_theme', 'flatly')

    root = tb.Window(themename=ttk_theme)
    root.title("Process analysis & presentation (GeoParquet)")
    try:
        # Keep backward-compat path for the icon
        ico = (base_dir() / "system_resources" / "mesa.ico")
        # Tail the shared worker log file into the GUI while the worker runs
        try:
            _start_log_tailer(root)
        except Exception:
            pass

        if ico.exists():
            root.iconbitmap(str(ico))
    except Exception:
        pass

    # panes
    paned = tk.PanedWindow(root, orient=tk.HORIZONTAL)
    paned.pack(fill=tk.BOTH, expand=True)
    left = tk.Frame(paned); right = tk.Frame(paned, width=660)
    paned.add(left); paned.add(right)

    # left
    log_widget = scrolledtext.ScrolledText(left, height=14)
    log_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    progress_frame = tk.Frame(left); progress_frame.pack(pady=5)
    progress_var = tk.DoubleVar(value=0.0)
    progress_bar = tb.Progressbar(progress_frame, orient="horizontal", length=260,
                                  mode="determinate", variable=progress_var, bootstyle='info')
    progress_bar.pack(side=tk.LEFT)
    progress_label = tk.Label(progress_frame, text=_current_progress_display(), bg="light grey")
    progress_label.pack(side=tk.LEFT, padx=8)

    info = (f"This is where all calculations are made. Information is provided every {HEARTBEAT_SECS} seconds.\n"
            "To start the processing press 'Process'. Then 'Progress map' to keep an eye on\n"
            "the progress of your calculations. Many assets and/or detailed geocodes will\n"
            "increase the processing time. All resulting data is saved in the GeoParquet\n"
            "vector data format. Raster MBTiles are generated automatically at the end for\n"
            "faster viewing of your data in the map viewer.")
    tk.Label(left, text=info, wraplength=680, justify="left").pack(padx=10, pady=10)

    def _await_processing_then_tiles(proc_handle):
        try:
            proc_handle.join()
        except Exception:
            return
        exit_code = getattr(proc_handle, "exitcode", None)
        if exit_code not in (0, None):
            log_to_gui(log_widget, f"[Tiles] Skipping MBTiles stage because processing exited with code {exit_code}.")
            update_progress(100.0)
            return
        _auto_run_tiles_stage(tiles_minzoom, tiles_maxzoom)

    def _run():
        set_progress_stage("Preparations")
        update_progress(0.0)
        _start_processing_worker(cfg_path)
        # Kick off periodic GUI polling of the shared status file for progress updates
        try:
            _poll_progress_periodically(root)
        except Exception:
            pass
        proc_snapshot = _PROC
        if proc_snapshot is not None:
            threading.Thread(target=lambda: _await_processing_then_tiles(proc_snapshot), daemon=True).start()

    btn_frame = tk.Frame(left); btn_frame.pack(pady=6)
    tb.Button(btn_frame, text="Process", bootstyle=PRIMARY,
              command=_run).pack(side=tk.LEFT, padx=5)
    tb.Button(btn_frame, text="Exit", bootstyle=WARNING, command=root.destroy).pack(side=tk.LEFT, padx=5)

    # right
    tk.Label(right, text="Processing progress map", font=("Segoe UI", 11, "bold")).pack(padx=10, pady=(16,6), anchor="w")
    tk.Label(right, text=("For more complex calculations this will give the user a better \n"
                          "understanding of the progress of when the calculation\n"),
             justify="left").pack(padx=10, anchor="w")
    tb.Button(right, text="Progress map", command=open_minimap_window).pack(padx=10, pady=10, anchor="w")

    log_to_gui(log_widget, "Opened processing UI (GeoParquet only).")
    root.mainloop()

def _process_chunks(chunks, max_workers, asset_data, geom_types, tmp_parts,
                    asset_soft_limit, geocode_soft_limit,
                    chunk_cells, cells_meta, home_bounds, total_chunks,
                    progress_state, files, written):
    """
    Run intersection either with a multiprocessing Pool (safe contexts)
    or serially (when running from a background thread in frozen builds).
    Returns (done_count, written, files, error_msg).
    """
    done_count = progress_state.get('done', 0)
    error_msg = None
    started_at = progress_state.get('started_at', 0.0)
    last_ping = started_at if started_at else 0.0

    iterable = ((i, ch) for i, ch in enumerate(chunks, start=1))

    def _update_status():
        try:
            running_chunk_ids = list(range(done_count+1, min(done_count + max_workers, total_chunks) + 1))
            running_cells = set().union(*(chunk_cells.get(i, set()) for i in running_chunk_ids)) if running_chunk_ids else set()
            done_cells = set().union(*(chunk_cells.get(i, set()) for i in range(1, done_count+1))) if done_count else set()

            id_to_idx = {c["id"]: i for i, c in enumerate(cells_meta)}
            for cid in done_cells:
                i = id_to_idx.get(int(cid))
                if i is not None:
                    cells_meta[i]["state"] = "done"
            for cid in running_cells:
                i = id_to_idx.get(int(cid))
                if i is not None and cells_meta[i]["state"] != "done":
                    cells_meta[i]["state"] = "running"

            _write_status_atomic({
                "phase": "intersect",
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "chunks_total": total_chunks,
                "done": done_count,
                "running": running_chunk_ids,
                "cells": cells_meta,
                "home_bounds": home_bounds
            })
        except Exception:
            pass

    def _tick_progress():
        nonlocal last_ping, written, done_count
        now = time.time()
        if (now - last_ping) >= HEARTBEAT_SECS or done_count == total_chunks:
            elapsed = now - started_at if started_at else 0.0
            pct = (done_count / total_chunks) * 100 if total_chunks else 100.0
            eta = "?"
            if done_count and started_at:
                est_total = elapsed / max(done_count, 1) * total_chunks
                eta_ts = datetime.now() + timedelta(seconds=max(0.0, est_total - elapsed))
                eta = eta_ts.strftime("%H:%M:%S")
                dd = (eta_ts.date() - datetime.now().date()).days
                if dd > 0:
                    eta += f" (+{dd}d)"
            log_to_gui(log_widget, f"[intersect] {done_count}/{total_chunks} chunks (~{pct:.2f}%) • rows written: {written:,} • ETA {eta}")
            update_progress(35.0 + (done_count / max(total_chunks, 1)) * 15.0)
            last_ping = now

    # Pool path (safe in dev or main-thread contexts)
    if _mp_allowed() and max_workers > 1:
        with multiprocessing.get_context("spawn").Pool(
                processes=max_workers,
                initializer=_intersect_pool_init,
                initargs=(asset_data, geom_types, str(tmp_parts), asset_soft_limit, geocode_soft_limit)) as pool:

            for (idx, nrows, paths, err, logs) in pool.imap_unordered(_intersection_worker, iterable):
                done_count += 1
                progress_state['done'] = done_count

                if logs:
                    for line in logs:
                        log_to_gui(log_widget, line)

                if err:
                    log_to_gui(log_widget, f"[intersect] Chunk {idx} failed: {err}")
                    error_msg = err
                    pool.terminate()
                    break

                written += nrows
                progress_state['rows'] = written
                if paths:
                    if isinstance(paths, list):
                        files.extend(paths)
                    else:
                        files.append(paths)

                _update_status()
                _tick_progress()

                if done_count % 8 == 0:
                    gc.collect()
    else:
        # Serial fallback (initialize globals like pool initializer does)
        _intersect_pool_init(asset_data, geom_types, str(tmp_parts), asset_soft_limit, geocode_soft_limit)
        for args in iterable:
            (idx, nrows, paths, err, logs) = _intersection_worker(args)

            done_count += 1
            progress_state['done'] = done_count

            if logs:
                for line in logs:
                    log_to_gui(log_widget, line)

            if err:
                log_to_gui(log_widget, f"[intersect] Chunk {idx} failed: {err}")
                error_msg = err
                break

            written += nrows
            progress_state['rows'] = written
            if paths:
                if isinstance(paths, list):
                    files.extend(paths)
                else:
                    files.append(paths)

            _update_status()
            _tick_progress()

            if done_count % 8 == 0:
                gc.collect()

    return done_count, written, files, error_msg


