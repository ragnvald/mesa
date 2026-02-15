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
import threading
import time
import shutil
import multiprocessing as mp
import traceback
import warnings
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
from shapely.ops import linemerge, polygonize, unary_union
from shapely import wkb as shp_wkb
from shapely.prepared import prep

try:
    from shapely import force_2d as _shp_force_2d  # Shapely >= 2
except Exception:
    _shp_force_2d = None

# pyogrio (used by GeoPandas by default when installed) warns that measured (M)
# geometries are not supported and will be converted. Shapely/GEOS does not
# preserve M anyway, so treat this as expected and keep stdout clean.
warnings.filterwarnings(
    "ignore",
    message=r"Measured \(M\) geometry types are not supported\..*",
    category=UserWarning,
    module=r"pyogrio\..*",
)


def _force_2d_geom(geom):
    """Drop Z/M from a single geometry, returning a 2D geometry."""
    if geom is None:
        return None
    try:
        if getattr(geom, "is_empty", False):
            return geom
    except Exception:
        return geom

    if _shp_force_2d is not None:
        try:
            return _shp_force_2d(geom)
        except Exception:
            pass

    try:
        return shp_wkb.loads(shp_wkb.dumps(geom, output_dimension=2))
    except Exception:
        return geom

# Optional: faster union_all (Shapely >= 2)
try:
    from shapely import union_all as shapely_union_all
except Exception:
    shapely_union_all = None

# Optional: STRtree for fast point-in-coverage checks (used when skipping global coverage union)
try:
    from shapely.strtree import STRtree as ShapelySTRtree
except Exception:
    ShapelySTRtree = None

# -----------------------------------------------------------------------------
# Locale
# -----------------------------------------------------------------------------
# Avoid forcing "en_US.UTF-8" on Windows: many machines don't have that locale.

def _patch_locale_setlocale_for_windows() -> None:
    """Make locale.setlocale resilient on Windows.

    ttkbootstrap calls locale.setlocale during import (DatePickerDialog). On some
    Windows machines, setlocale(LC_TIME, "") can raise locale.Error.
    """
    try:
        if os.name != "nt":
            return
        _orig = locale.setlocale

        def _safe_setlocale(category, value=None):
            try:
                if value is None:
                    return _orig(category)
                return _orig(category, value)
            except locale.Error:
                for fallback in ("", "C"):
                    try:
                        return _orig(category, fallback)
                    except Exception:
                        continue
                try:
                    return _orig(category)
                except Exception:
                    return "C"

        locale.setlocale = _safe_setlocale  # type: ignore[assignment]
    except Exception:
        pass


_patch_locale_setlocale_for_windows()

try:
    if os.name == "nt":
        for _k in ("LC_ALL", "LC_CTYPE", "LANG"):
            _v = os.environ.get(_k)
            if _v and ("utf-8" in _v.lower()) and ("_" in _v) and ("." in _v):
                os.environ.pop(_k, None)
except Exception:
    pass
try:
    locale.setlocale(locale.LC_ALL, "")
except Exception:
    try:
        locale.setlocale(locale.LC_ALL, "C")
    except Exception:
        pass

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
from tkinter import messagebox
import json
from pathlib import Path


def _import_ttkbootstrap():
    """Import ttkbootstrap with a locale-safe fallback.

    On some Windows machines, a forced/unsupported locale can cause ttkbootstrap (or its
    import chain) to raise locale.Error('unsupported locale setting'). In that case,
    retry after switching to a safe locale.
    """
    try:
        import ttkbootstrap as _ttk
        from ttkbootstrap.constants import PRIMARY as _PRIMARY, INFO as _INFO, WARNING as _WARNING
        return _ttk, _PRIMARY, _INFO, _WARNING, None, None
    except Exception as e:
        err1 = repr(e)
        tb1 = traceback.format_exc()

        try:
            is_locale_problem = isinstance(e, locale.Error) or ("unsupported locale" in str(e).lower())
        except Exception:
            is_locale_problem = False

        if is_locale_problem:
            try:
                # Prefer user default; fall back to the C locale.
                for loc in ("", "C"):
                    try:
                        locale.setlocale(locale.LC_ALL, loc)
                        break
                    except Exception:
                        continue
            except Exception:
                pass

            try:
                # Clear partially imported modules before retry.
                for name in list(sys.modules.keys()):
                    if name == "ttkbootstrap" or name.startswith("ttkbootstrap."):
                        sys.modules.pop(name, None)
            except Exception:
                pass

            try:
                import ttkbootstrap as _ttk
                from ttkbootstrap.constants import PRIMARY as _PRIMARY, INFO as _INFO, WARNING as _WARNING
                return _ttk, _PRIMARY, _INFO, _WARNING, None, None
            except Exception as e2:
                err2 = f"{err1} | retry: {repr(e2)}"
                tb2 = tb1 + "\n--- retry ---\n" + traceback.format_exc()
                return None, None, None, None, err2, tb2

        return None, None, None, None, err1, tb1


ttk, PRIMARY, INFO, WARNING, _TTKBOOTSTRAP_IMPORT_ERROR, _TTKBOOTSTRAP_IMPORT_TRACE = _import_ttkbootstrap()


def _ttkbootstrap_diagnostics() -> dict:
    """Best-effort diagnostics for ttkbootstrap import failures in frozen builds."""
    d: dict = {}
    try:
        d["frozen"] = bool(getattr(sys, "frozen", False))
    except Exception:
        d["frozen"] = False
    try:
        d["executable"] = sys.executable
    except Exception:
        pass
    try:
        d["cwd"] = os.getcwd()
    except Exception:
        pass
    try:
        meipass = getattr(sys, "_MEIPASS", None)
        d["_MEIPASS"] = meipass
        if meipass:
            root = Path(meipass)
            d["meipass_exists"] = root.exists()
            d["meipass_ttkbootstrap_dir_exists"] = (root / "ttkbootstrap").exists()
    except Exception:
        pass
    try:
        d["ttkbootstrap_import_error"] = _TTKBOOTSTRAP_IMPORT_ERROR
    except Exception:
        pass
    try:
        if _TTKBOOTSTRAP_IMPORT_TRACE:
            d["ttkbootstrap_import_trace_tail"] = _TTKBOOTSTRAP_IMPORT_TRACE[-1200:]
    except Exception:
        pass
    return d


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
BASIC_MOSAIC_GROUP = "basic_mosaic"
GEOCODE_ORIGIN_IMPORTED = "imported"
GEOCODE_ORIGIN_GENERATED = "generated"
GEOCODE_ORIGIN_BASIC_MOSAIC = "basic_mosaic"


def _normalize_geocode_origin(raw_origin: object, group_name: object) -> str:
    origin = str(raw_origin or "").strip().lower()
    if origin in {GEOCODE_ORIGIN_IMPORTED, GEOCODE_ORIGIN_GENERATED, GEOCODE_ORIGIN_BASIC_MOSAIC}:
        return origin

    gname = str(group_name or "").strip()
    if gname == BASIC_MOSAIC_GROUP:
        return GEOCODE_ORIGIN_BASIC_MOSAIC
    if gname.upper().startswith("H3_R"):
        return GEOCODE_ORIGIN_GENERATED
    return GEOCODE_ORIGIN_IMPORTED

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
# GUI globals + heartbeat
# -----------------------------------------------------------------------------
root: Optional[tk.Tk] = None
log_widget: Optional[tk.Widget] = None
log_widgets: list[tk.Widget] = []
mosaic_log_widget: Optional[tk.Widget] = None
mosaic_log_from_file_mode: bool = False
log_main_thread_id: Optional[int] = None
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
            1) --original_working_directory (CLI)
            2) env MESA_BASE_DIR (honored immediately when valid in frozen mode)
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

    if cli_workdir:
        cli_hit = _maybe_return(cli_workdir)
        if cli_hit:
            return cli_hit

    env_base = os.environ.get("MESA_BASE_DIR")
    if env_base and getattr(sys, "frozen", False):
        env_hit = _maybe_return(env_base)
        if env_hit:
            return env_hit

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


def _append_formatted_to_log_targets(formatted: str) -> None:
    targets: list[tk.Widget] = []
    if log_widget is not None:
        targets = [log_widget]
    elif mosaic_log_widget is not None and not mosaic_log_from_file_mode:
        targets = [mosaic_log_widget]
    elif log_widgets:
        targets = [log_widgets[0]]
    for target in targets:
        try:
            if isinstance(target, ttk.Treeview):
                iid = target.insert("", tk.END, values=(formatted,))
                target.see(iid)
                children = target.get_children()
                if len(children) > 2500:
                    for old_iid in children[:500]:
                        target.delete(old_iid)
                continue
            try:
                target.configure(state="normal")
            except Exception:
                pass
            target.insert(tk.END, formatted + "\n")
            target.see(tk.END)
            try:
                target.update_idletasks()
            except Exception:
                pass
        except Exception:
            pass


def _tail_text_file(path: Path, max_lines: int = 20) -> list[str]:
    try:
        if max_lines <= 0 or not path.exists() or not path.is_file():
            return []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return [ln.rstrip("\r\n") for ln in lines[-max_lines:]]
    except Exception:
        return []

def log_to_gui(message: str, level: str = "INFO"):
    global log_widgets, log_main_thread_id
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted = f"{timestamp} [{level}] - {message}"

    if root is not None and log_main_thread_id is not None:
        if threading.get_ident() == log_main_thread_id:
            _append_formatted_to_log_targets(formatted)
        else:
            try:
                root.after(0, lambda f=formatted: _append_formatted_to_log_targets(f))
            except Exception:
                pass
    else:
        _append_formatted_to_log_targets(formatted)

    if original_working_directory:
        try:
            with open(Path(original_working_directory) / "log.txt", "a", encoding="utf-8") as f:
                f.write(formatted + "\n")
        except Exception:
            pass
    if log_widget is None:
        try:
            print(formatted)
        except UnicodeEncodeError:
            # Some Windows consoles/pipes use legacy encodings (e.g. cp1252) and will
            # raise on characters like '≈' or '²'. Never let logging crash the tool.
            try:
                s = formatted + "\n"
                if hasattr(sys.stdout, "buffer") and sys.stdout.buffer is not None:
                    sys.stdout.buffer.write(s.encode("utf-8", errors="replace"))
                    sys.stdout.buffer.flush()
                else:
                    safe = formatted.encode("ascii", errors="backslashreplace").decode("ascii")
                    print(safe)
            except Exception:
                pass


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
        # Enforce 2D geometries (drop Z/M)
        try:
            data["geometry"] = data.geometry.apply(_force_2d_geom)
        except Exception:
            pass
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
        # Enforce 2D geometries (drop Z/M)
        try:
            gdf["geometry"] = gdf.geometry.apply(_force_2d_geom)
        except Exception:
            pass
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
            "geocode_origin": GEOCODE_ORIGIN_IMPORTED,
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
            objects.append({
                "code": code,
                "ref_geocodegroup": group_id,
                "name_gis_geocodegroup": name_gis_geocodegroup,
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


def _explode_polygons(geom) -> list[Polygon]:
    """Explode polygonal geometry into individual Polygon parts (no MultiPolygon output)."""
    g = _extract_polygonal(geom)
    if g is None:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return [p for p in g.geoms if isinstance(p, Polygon) and not p.is_empty]
    return []


def _default_dissolve_group_columns(df: pd.DataFrame) -> list[str]:
    """Pick non-geometry columns likely useful for dissolve grouping.

    Goal: ignore obvious unique identifiers so that "same attributes" can match.
    """
    import re

    cols = [c for c in df.columns if c != "geometry"]
    if not cols:
        return []

    # Common identifier patterns (keep conservative; users can still pre-clean inputs).
    ignore = re.compile(r"(^id$|.*_id$|^id_.*|uuid|guid|objectid|fid)", re.IGNORECASE)
    group_cols = [c for c in cols if not ignore.search(str(c))]
    return group_cols


def _prepare_assets_polygonal_metric(a_metric: gpd.GeoDataFrame, cfg: configparser.ConfigParser) -> gpd.GeoDataFrame:
    """Convert asset geometries into polygonal metric geometries (buffering lines/points)."""
    line_buf_m = _cfg_float(cfg, "default_line_buffer_m", 10.0)
    point_buf_m = _cfg_float(cfg, "default_point_buffer_m", 10.0)

    out = a_metric.copy()
    polys = []
    for geom in out.geometry:
        polyish = _geom_to_polygonal_metric(geom, line_buf_m=line_buf_m, point_buf_m=point_buf_m)
        if polyish is None:
            polys.append(None)
            continue
        polyish = _fix_valid(polyish)
        polyish = _extract_polygonal(polyish)
        if polyish is None or getattr(polyish, "is_empty", True):
            polys.append(None)
            continue
        polys.append(polyish)
    out["geometry"] = polys
    out = out[out["geometry"].notna()].copy()
    try:
        out = out[~out.geometry.is_empty].copy()
    except Exception:
        pass
    return out


def _dissolve_assets_by_attributes(a_poly: gpd.GeoDataFrame, cfg: configparser.ConfigParser) -> gpd.GeoDataFrame:
    hb_secs = _cfg_float(cfg, "heartbeat_secs", 60.0)
    last_log = time.time()
    last_ui = 0.0

    group_cols = _default_dissolve_group_columns(a_poly)
    if not group_cols:
        log_to_gui("[Mosaic] Dissolve-by-attributes: no suitable non-ID columns found; skipping dissolve.", "WARN")
        return a_poly[["geometry"]].copy()

    try:
        ng = int(a_poly.groupby(group_cols, dropna=False, sort=False).ngroups)
    except Exception:
        ng = 0
    log_to_gui(
        f"[Mosaic] Dissolve-by-attributes: grouping by {len(group_cols)} column(s){f' across {ng:,} group(s)' if ng else ''}…",
        "INFO",
    )

    rows = []
    grouped = a_poly.groupby(group_cols, dropna=False, sort=False)
    i = 0
    for key, sub in grouped:
        i += 1
        now = time.time()
        if (now - last_log) >= max(10.0, hb_secs):
            last_log = now
            try:
                log_to_gui(f"[Mosaic] Dissolve-by-attributes: processed {i:,}/{ng:,} group(s)…", "INFO")
            except Exception:
                log_to_gui(f"[Mosaic] Dissolve-by-attributes: processed {i:,} group(s)…", "INFO")
        # Provide a gentle progress tick during dissolve without conflicting with later stages.
        if ng > 0 and (now - last_ui) >= 0.25:
            last_ui = now
            update_progress(_progress_lerp(10.0, 12.0, i / max(1, ng)))

        geoms = [g for g in sub.geometry.tolist() if g is not None and not g.is_empty]
        if not geoms:
            continue

        merged = _unary_union_safe(geoms, label="dissolve_attrs")
        parts = _explode_polygons(merged)
        if not parts:
            continue

        # key is scalar for 1-col groupby; tuple otherwise
        if len(group_cols) == 1:
            key_vals = (key,)
        else:
            key_vals = tuple(key)

        base_row = dict(zip(group_cols, key_vals))
        for p in parts:
            r = dict(base_row)
            r["geometry"] = p
            rows.append(r)

    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=a_poly.crs)
    # Avoid MultiPolygon rows: explode already ensures Polygon parts.
    return out if not out.empty else gpd.GeoDataFrame(geometry=[], crs=a_poly.crs)


def _dissolve_assets_aggressive_with_attrs_json(a_poly: gpd.GeoDataFrame, cfg: configparser.ConfigParser) -> gpd.GeoDataFrame:
    hb_secs = _cfg_float(cfg, "heartbeat_secs", 60.0)
    last_log = time.time()
    last_ui = 0.0

    attr_cols = [c for c in a_poly.columns if c != "geometry"]

    geoms = [g for g in a_poly.geometry.tolist() if g is not None and not g.is_empty]
    if not geoms:
        return gpd.GeoDataFrame(geometry=[], crs=a_poly.crs)

    merged = _unary_union_safe(geoms, label="dissolve_aggressive")
    parts = _explode_polygons(merged)
    if not parts:
        return gpd.GeoDataFrame(geometry=[], crs=a_poly.crs)

    log_to_gui(f"[Mosaic] Dissolve-aggressive: produced {len(parts):,} polygon(s); attaching source attributes as JSON…", "INFO")

    sidx = None
    try:
        sidx = a_poly.sindex
    except Exception:
        sidx = None

    rows = []
    for i, p in enumerate(parts, start=1):
        now = time.time()
        if (now - last_log) >= max(10.0, hb_secs):
            last_log = now
            log_to_gui(f"[Mosaic] Dissolve-aggressive: processed {i:,}/{len(parts):,} polygon(s)…", "INFO")
        if (now - last_ui) >= 0.25:
            last_ui = now
            update_progress(_progress_lerp(10.0, 12.0, i / max(1, len(parts))))

        attrs_list = []
        try:
            if sidx is not None:
                try:
                    cand = list(sidx.query(p, predicate="intersects"))
                except Exception:
                    cand = list(sidx.query(p))
                sub = a_poly.iloc[cand]
            else:
                sub = a_poly

            try:
                mask = sub.intersects(p)
                sub = sub.loc[mask]
            except Exception:
                pass

            if not sub.empty and attr_cols:
                for _, r0 in sub[attr_cols].iterrows():
                    d = {}
                    for c in attr_cols:
                        v = r0.get(c, None)
                        if pd.isna(v):
                            v = None
                        d[str(c)] = v
                    attrs_list.append(d)
        except Exception:
            attrs_list = []

        rows.append({
            "geometry": p,
            "dissolve_source_attributes_json": json.dumps(attrs_list, ensure_ascii=False, default=str),
        })

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=a_poly.crs)


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
        # Shapely 2.x: union_all is typically faster but equivalent.
        if shapely_union_all is not None:
            return shapely_union_all(geoms)
        return unary_union(geoms)
    except MemoryError:
        log_to_gui(f"[Mosaic] MemoryError during unary_union ({label}) for n={len(geoms):,}; retrying smaller chunks…", "WARN")
    except Exception as e:
        log_to_gui(f"[Mosaic] unary_union failed ({label}) for n={len(geoms):,}: {e}; retrying smaller chunks…", "WARN")

    # Retry with chunking + balanced (pyramid) reduction to reduce peak memory.
    step = max(min_step, len(geoms) // 4)
    while step >= min_step:
        ok = True
        partials = []
        for i in range(0, len(geoms), step):
            chunk = geoms[i:i+step]
            try:
                if shapely_union_all is not None:
                    cu = shapely_union_all(chunk)
                else:
                    cu = unary_union(chunk)
            except Exception:
                ok = False
                break
            if cu is not None:
                partials.append(cu)
        if ok:
            if not partials:
                return None
            if len(partials) == 1:
                return partials[0]
            partials = _tree_reduce_unions(partials, max_partials=1, label=f"{label}:retry", heartbeat_s=10.0)
            return partials[0] if partials else None
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

    # Total rounds is known up-front for this pairwise reduction strategy.
    total_rounds = 0
    n_tmp = len(unions)
    while n_tmp > max_partials:
        total_rounds += 1
        n_tmp = (n_tmp + 1) // 2

    while len(unions) > max_partials:
        round_idx += 1
        n_in = len(unions)
        merges_total = (n_in + 1) // 2

        now = time.time()
        if now - last_log >= hb:
            elapsed = now - started
            log_to_gui(
                f"[Mosaic] Reducing {label}: round {round_idx}/{max(1, total_rounds)} starting (n={n_in:,} -> <= {max_partials:,}); merges={merges_total:,}; elapsed {elapsed:.0f}s…",
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
                    f"[Mosaic] Reducing {label}: round {round_idx}/{max(1, total_rounds)} merge {merge_i:,}/{merges_total:,} ({pct:.0f}%) • elapsed {elapsed:.0f}s",
                    "INFO",
                )
                last_log = now

        unions = merged
        now = time.time()
        if now - last_log >= hb:
            elapsed = now - started
            log_to_gui(
                f"[Mosaic] Reducing {label}: round {round_idx}/{max(1, total_rounds)} complete; n={len(unions):,}; elapsed {elapsed:.0f}s",
                "INFO",
            )
            last_log = now

    return unions


def _maybe_sort_partials_before_reduction(
    unions: list,
    *,
    cfg: configparser.ConfigParser,
    kind: str,
) -> list:
    """Optionally sort partial geometries by a cheap size proxy before reduction.

    Motivation: union performance can be highly order-dependent. Sorting so that
    smaller geometries are merged first often reduces intermediate complexity.

    Controlled by config: mosaic_reduce_sort_partials (default: false).
    """
    try:
        v = str(cfg["DEFAULT"].get("mosaic_reduce_sort_partials", "false")).strip().lower()
        enabled = v in ("1", "true", "yes", "on")
    except Exception:
        enabled = False
    if not enabled or not unions or len(unions) < 3:
        return unions

    kind = (kind or "").strip().lower()

    def key_fn(g):
        if g is None:
            return float("inf")
        try:
            if kind == "coverage":
                return float(getattr(g, "area", 0.0) or 0.0)
            # edges/linework (and default)
            return float(getattr(g, "length", 0.0) or 0.0)
        except Exception:
            return float("inf")

    try:
        return sorted(unions, key=key_fn)
    except Exception:
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
            if shapely_union_all is not None:
                return shapely_union_all(geoms)
            return unary_union(geoms)
        except Exception:
            # Retry in smaller pieces; avoid logging from workers.
            nonlocal union_retries
            union_retries += 1
            step = max(50, len(geoms) // 4)
            while step >= 50:
                ok = True
                partials = []
                for i in range(0, len(geoms), step):
                    chunk = geoms[i:i + step]
                    try:
                        if shapely_union_all is not None:
                            cu = shapely_union_all(chunk)
                        else:
                            cu = unary_union(chunk)
                    except Exception:
                        ok = False
                        break
                    if cu is not None:
                        partials.append(cu)
                if ok:
                    if not partials:
                        return None
                    # Quiet balanced reduction (no GUI logging from worker).
                    while len(partials) > 1:
                        merged = []
                        it = iter(partials)
                        for a in it:
                            b = next(it, None)
                            if b is None:
                                merged.append(a)
                            else:
                                try:
                                    if shapely_union_all is not None:
                                        u = shapely_union_all([a, b])
                                    else:
                                        u = unary_union([a, b])
                                except Exception:
                                    u = a
                                merged.append(u)
                        partials = merged
                    return partials[0]
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

    # Coverage union is used for: (1) filtering polygonize faces to covered areas,
    # and (2) sanity-checking area. The full global union can be expensive.
    # If disabled, we keep coverage as a list of partial unions and do
    # point-in-coverage checks via STRtree in the caller.
    try:
        v = str(cfg["DEFAULT"].get("mosaic_coverage_union", "true")).strip().lower()
        coverage_union_enabled = v in ("1", "true", "yes", "on")
    except Exception:
        coverage_union_enabled = True

    def flush_boundary():
        if not boundary_batch:
            return
        u = _unary_union_safe(boundary_batch, label="boundary_batch")
        if u is not None:
            line_partials.append(u)
            stats["union_batches"] += 1
            if len(line_partials) > max_partials:
                line_partials[:] = _maybe_sort_partials_before_reduction(line_partials, cfg=cfg, kind="edges")
                line_partials[:] = _tree_reduce_unions(line_partials, max_partials=max_partials, label="edges")
        boundary_batch.clear()

    def flush_coverage():
        if not cov_batch:
            return
        u = _unary_union_safe(cov_batch, label="coverage_batch")
        if u is not None:
            cov_partials.append(u)
            if len(cov_partials) > max_partials:
                cov_partials[:] = _maybe_sort_partials_before_reduction(cov_partials, cfg=cfg, kind="coverage")
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

        # Hint for very large asset datasets: smaller extract chunks tend to reduce
        # “long tail” stalls (a few heavy chunks keeping the whole pool alive).
        # We only log here; the user can tune via config.ini.
        try:
            asset_count = int(len(a_metric))
            if asset_count >= 500_000 and chunk_size > 250:
                log_to_gui(
                    f"[Mosaic] Large asset dataset detected ({asset_count:,} features). Consider setting mosaic_extract_chunk_size=250 (current={chunk_size:,}) for better load balancing.",
                    "INFO",
                )
            elif asset_count >= 200_000 and chunk_size > 500:
                log_to_gui(
                    f"[Mosaic] Large asset dataset detected ({asset_count:,} features). Consider setting mosaic_extract_chunk_size=500 (or 250) (current={chunk_size:,}) for better load balancing.",
                    "INFO",
                )
        except Exception:
            pass
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
                line_partials = _maybe_sort_partials_before_reduction(line_partials, cfg=cfg, kind="edges")
                line_partials[:] = _tree_reduce_unions(line_partials, max_partials=max_partials, label="edges")
            if cov_partials and len(cov_partials) > max_partials:
                cov_partials = _maybe_sort_partials_before_reduction(cov_partials, cfg=cfg, kind="coverage")
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
    line_partials = _maybe_sort_partials_before_reduction(line_partials, cfg=cfg, kind="edges")
    line_partials = _tree_reduce_unions(line_partials, max_partials=2, label="edges(final)")
    edge_net = _unary_union_safe(line_partials, label="linework_final", min_step=2)
    if edge_net is None:
        edge_net = line_partials[0]

    coverage = None
    if cov_partials:
        if coverage_union_enabled:
            cov_partials = _maybe_sort_partials_before_reduction(cov_partials, cfg=cfg, kind="coverage")
            cov_partials = _tree_reduce_unions(cov_partials, max_partials=2, label="coverage(final)")
            coverage = _unary_union_safe(cov_partials, label="coverage_final", min_step=2)
            if coverage is None:
                coverage = cov_partials[0]
        else:
            # Keep partials as-is (already reduced to <=max_partials earlier).
            coverage = list(cov_partials)

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
    g = _read_parquet_gdf(pq_groups)
    g = ensure_wgs84(g)
    if not g.empty and "geometry" in g:
        try:
            u = unary_union(g.geometry); u = _extract_polygonal(u)
            if u and not u.is_empty:
                log_to_gui("Union source: GeoParquet tbl_asset_group", "INFO")
                return u
        except Exception as e:
            log_to_gui(f"H3 union: tbl_asset_group union failed: {e}", "WARN")
    pq_objs = geoparquet_path(base_dir, "tbl_asset_object")
    ao = _read_parquet_gdf(pq_objs)
    ao = ensure_wgs84(ao)
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
        except Exception as e:
            log_to_gui(f"H3 union: tbl_asset_object union failed: {e}", "WARN")

        # Last-resort fallback: use bbox of whatever geometry we have. This is less precise
        # than the union, but it prevents H3 from being blocked by union/topology issues.
        try:
            minx, miny, maxx, maxy = ao.total_bounds
            arr = np.array([minx, miny, maxx, maxy], dtype=float)
            if np.isfinite(arr).all() and maxx > minx and maxy > miny:
                log_to_gui("Union source: tbl_asset_object bbox (fallback)", "WARN")
                return box(minx, miny, maxx, maxy)
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

    # Geocode objects should not carry source attributes.
    if not existing_o.empty and "attributes" in existing_o.columns:
        try:
            existing_o = existing_o.drop(columns=["attributes"]).copy()
        except Exception:
            pass
    if new_objects_gdf is not None and "attributes" in new_objects_gdf.columns:
        try:
            new_objects_gdf = new_objects_gdf.drop(columns=["attributes"]).copy()
        except Exception:
            pass

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

    def _series_or_default(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
        if col in df.columns:
            try:
                return df[col]
            except Exception:
                pass
        return pd.Series([default] * len(df), index=df.index, dtype="object")

    if "geocode_origin" not in existing_g.columns:
        existing_g["geocode_origin"] = _series_or_default(existing_g, "name_gis_geocodegroup", "").apply(
            lambda v: _normalize_geocode_origin("", v)
        )
    else:
        existing_g["geocode_origin"] = [
            _normalize_geocode_origin(o, n)
            for o, n in zip(existing_g["geocode_origin"], _series_or_default(existing_g, "name_gis_geocodegroup", ""))
        ]

    if "geocode_origin" not in new_groups_gdf.columns:
        new_groups_gdf["geocode_origin"] = _series_or_default(new_groups_gdf, "name_gis_geocodegroup", "").apply(
            lambda v: _normalize_geocode_origin("", v)
        )
    else:
        new_groups_gdf["geocode_origin"] = [
            _normalize_geocode_origin(o, n)
            for o, n in zip(new_groups_gdf["geocode_origin"], _series_or_default(new_groups_gdf, "name_gis_geocodegroup", ""))
        ]

    name_to_id = dict(zip(new_groups_gdf["name_gis_geocodegroup"], new_groups_gdf["id"]))
    new_objects_gdf = new_objects_gdf.copy()
    new_objects_gdf["ref_geocodegroup"] = new_objects_gdf["name_gis_geocodegroup"].map(name_to_id)

    groups_out = pd.concat([existing_g, new_groups_gdf], ignore_index=True)
    objects_out = pd.concat([existing_o, new_objects_gdf], ignore_index=True)

    groups_out = ensure_wgs84(gpd.GeoDataFrame(groups_out, geometry="geometry"))
    objects_out = ensure_wgs84(gpd.GeoDataFrame(objects_out, geometry="geometry"))

    def _atomic_to_parquet(gdf: gpd.GeoDataFrame, final_path: Path) -> None:
        tmp_path = final_path.with_name(final_path.name + f".tmp_{uuid.uuid4().hex}")
        gdf.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, final_path)

    # Windows can hold file locks briefly (AV scans, other readers). Retry a few times.
    last_err: Exception | None = None
    for delay_s in (0.0, 0.25, 0.75, 1.5):
        if delay_s:
            time.sleep(delay_s)
        try:
            _atomic_to_parquet(groups_out, out_dir / "tbl_geocode_group.parquet")
            _atomic_to_parquet(objects_out, out_dir / "tbl_geocode_object.parquet")
            last_err = None
            break
        except Exception as e:
            last_err = e
    if last_err is not None:
        raise last_err

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

    # Always drop non-geometry columns: basic mosaic is geometry-only.
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
    cov_tree = None
    cov_parts = None
    if coverage is not None:
        # coverage can be either a single polygonal geometry (union) or a list of
        # partial unions (when mosaic_coverage_union=false).
        if isinstance(coverage, list):
            cov_parts = [g for g in coverage if g is not None and not getattr(g, "is_empty", False)]
            if cov_parts and ShapelySTRtree is not None:
                try:
                    cov_tree = ShapelySTRtree(cov_parts)
                    log_to_gui(f"[Mosaic] Coverage union disabled; using STRtree over {len(cov_parts):,} coverage part(s) for face filtering.", "INFO")
                except Exception:
                    cov_tree = None
            elif cov_parts:
                log_to_gui(
                    f"[Mosaic] Coverage union disabled but STRtree unavailable; falling back to linear scan over {len(cov_parts):,} coverage part(s).",
                    "WARN",
                )
        else:
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

    polygonize_error: Exception | None = None
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
                    # Use covers() instead of contains() to avoid edge/precision exclusions.
                    if not prepared_cov.covers(rp):
                        continue
                except Exception:
                    pass
            elif cov_parts is not None:
                try:
                    rp = poly.representative_point()
                    inside = False
                    if cov_tree is not None:
                        try:
                            # Avoid STRtree predicate operand-order ambiguity across Shapely versions.
                            # Query by bbox only, then explicitly validate with covers().
                            hits = cov_tree.query(rp)
                            if hits is None or len(hits) == 0:
                                inside = False
                            elif isinstance(hits[0], (int, np.integer)):
                                inside = any(cov_parts[int(i)].covers(rp) for i in hits)
                            else:
                                inside = any(getattr(g, "covers")(rp) for g in hits)
                        except TypeError:
                            # No predicate support
                            hits = cov_tree.query(rp)
                            if hits is None or len(hits) == 0:
                                inside = False
                            elif isinstance(hits[0], (int, np.integer)):
                                inside = any(cov_parts[int(i)].covers(rp) for i in hits)
                            else:
                                inside = any(g.covers(rp) for g in hits)
                    else:
                        # Linear scan fallback
                        inside = any(g.covers(rp) for g in cov_parts)
                    if not inside:
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
    except Exception as e:
        polygonize_error = e
        raise
    finally:
        _flush_faces()

        if polygonize_error is None:
            log_to_gui(
                f"[Mosaic] polygonize(edge_net) completed: produced {faces_total:,} faces; kept {faces_kept:,} (flush_batch={flush_batch:,}).",
                "INFO",
            )
        else:
            log_to_gui(
                f"[Mosaic] polygonize(edge_net) failed after producing {faces_total:,} faces; kept {faces_kept:,}. Error: {polygonize_error}",
                "ERROR",
            )

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
    obj = obj[["code", "name_gis_geocodegroup", "geometry"]]

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
        "geocode_origin": GEOCODE_ORIGIN_BASIC_MOSAIC,
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
            gdf["name_gis_geocodegroup"] = group_name
            gdf = gdf[["code", "name_gis_geocodegroup", "geometry"]]
            objects_parts.append(gdf)
            log_to_gui(f"H3 R{r}: prepared {len(gdf):,} cells.")
            groups_rows.append({
                "name": group_name, "name_gis_geocodegroup": group_name,
                "geocode_origin": GEOCODE_ORIGIN_GENERATED,
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
        # Do NOT clear existing basic_mosaic up-front.
        # publish_mosaic_as_geocode() refreshes the group during the merge-write; pre-clearing risks
        # leaving the project with no basic_mosaic if a later step fails (e.g. file lock during write).
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

        faces = mosaic_faces_from_assets_parallel(
            base_dir,
            buffer_m,
            grid_size_m,
            workers,
        )
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
def build_gui(base: Path, cfg: configparser.ConfigParser, start_tab: str = ""):
    global root, log_widget, mosaic_log_widget, mosaic_log_from_file_mode, log_main_thread_id, progress_var, progress_label, original_working_directory, mosaic_status_var, size_levels_var
    original_working_directory = str(base)

    try:
        global HEARTBEAT_SECS
        HEARTBEAT_SECS = int(cfg["DEFAULT"].get("heartbeat_secs", str(HEARTBEAT_SECS)))
    except Exception:
        pass

    theme = cfg["DEFAULT"].get("ttk_bootstrap_theme", "flatly") if ttk else None
    root = ttk.Window(themename=theme) if ttk else tk.Tk()
    log_main_thread_id = threading.get_ident()
    root.title("Geocode manage (Mosaic / H3 / Import / Edit)")

    notebook = ttk.Notebook(root) if ttk else None
    if notebook is None:
        messagebox.showerror("Missing UI dependency", "ttkbootstrap is required for geocode_manage GUI.")
        root.destroy()
        return
    notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

    tab_mosaic = ttk.Frame(notebook)
    tab_h3 = ttk.Frame(notebook)
    tab_import = ttk.Frame(notebook)
    tab_edit = ttk.Frame(notebook)
    notebook.add(tab_mosaic, text="Basic mosaic")
    notebook.add(tab_h3, text="H3 codes")
    notebook.add(tab_import, text="Import geocodes")
    notebook.add(tab_edit, text="Edit geocodes")

    tab_lookup = {
        "": 0,
        "mosaic": 0,
        "basic": 0,
        "h3": 1,
        "other": 1,
        "import": 2,
        "bin": 2,
        "edit": 3,
        "group": 3,
    }
    try:
        notebook.select(tab_lookup.get(str(start_tab).strip().lower(), 0))
    except Exception:
        pass

    mosaic_log = None
    h3_log = None
    import_log = None
    # Mosaic tab log uses a Treeview list for robust rendering across themes.

    global log_widgets
    log_widgets = []
    mosaic_log_widget = mosaic_log
    mosaic_log_from_file_mode = True

    def _set_log_target(widget):
        global log_widget
        log_widget = widget

    def _bind_treeview_copy(tv) -> None:
        def _copy_selected(_event=None):
            try:
                selected = tv.selection()
                if not selected:
                    return "break"
                lines: list[str] = []
                for iid in selected:
                    vals = tv.item(iid, "values")
                    if not vals:
                        continue
                    lines.append(" | ".join(str(v) for v in vals))
                text = "\n".join(lines).strip()
                if not text:
                    return "break"
                root.clipboard_clear()
                root.clipboard_append(text)
                return "break"
            except Exception:
                return "break"

        try:
            tv.bind("<Control-c>", _copy_selected)
            tv.bind("<Control-C>", _copy_selected)
        except Exception:
            pass

    def _sync_log_target_with_tab(_event=None):
        try:
            selected = notebook.select()
        except Exception:
            selected = ""
        if selected == str(tab_h3):
            _set_log_target(h3_log)
        elif selected == str(tab_import):
            _set_log_target(import_log)
        else:
            _set_log_target(None)

    # ---------------- Tab 1: Mosaic ----------------
    ttk.Label(tab_mosaic, text="Create/update the basic mosaic geocode group.").pack(anchor="w", padx=8, pady=(8, 4))

    mosaic_frame = ttk.LabelFrame(tab_mosaic, text="Mosaic action", bootstyle="secondary")
    mosaic_frame.pack(fill=tk.X, padx=8, pady=(2, 6))
    mosaic_status_var = tk.StringVar(value="")
    ttk.Label(mosaic_frame, text="Status:").grid(row=0, column=0, padx=4, pady=4, sticky="w")
    status_label = ttk.Label(mosaic_frame, textvariable=mosaic_status_var, width=18)
    status_label.grid(row=0, column=1, padx=(0, 10), pady=4, sticky="w")

    def _update_mosaic_status():
        exists = mosaic_exists(base)
        if mosaic_status_var.get() not in ("Running…", "Completed", "No faces"):
            mosaic_status_var.set("OK" if exists else "REQUIRED")
        try:
            status_label.configure(bootstyle=("success" if exists else "danger"))
        except Exception:
            pass

    def _run_mosaic_inline():
        _set_log_target(None)
        _clear_mosaic_log_view()
        try:
            _append_mosaic_line("--- Mosaic run started ---")
        except Exception:
            pass
        try:
            lp = Path(original_working_directory or str(base)) / "log.txt"
            if lp.exists():
                _mosaic_tail_state["offset"] = int(lp.stat().st_size)
            else:
                _mosaic_tail_state["offset"] = 0
        except Exception:
            _mosaic_tail_state["offset"] = 0
        _mosaic_tail_state["carry"] = ""
        _mosaic_tail_state["active"] = True
        log_to_gui("[Mosaic] Build requested.", "INFO")
        try:
            buf = float(cfg["DEFAULT"].get("mosaic_buffer_m", "25"))
        except Exception:
            buf = 25.0
        try:
            grid = float(cfg["DEFAULT"].get("mosaic_grid_size_m", "1000"))
        except Exception:
            grid = 1000.0
        mosaic_status_var.set("Running…")

        def _after(success):
            def _ui():
                _update_mosaic_status()
                mosaic_status_var.set("Completed" if success else "No faces")
                log_to_gui(
                    f"[Mosaic] Build finished with status: {'Completed' if success else 'No faces'}.",
                    "INFO",
                )
            try:
                root.after(100, _ui)
            except Exception:
                pass

        _run_in_thread(run_mosaic, base, buf, grid, _after)

    ttk.Button(mosaic_frame, text="Build mosaic", width=18, bootstyle=PRIMARY, command=_run_mosaic_inline).grid(
        row=0, column=2, padx=4, pady=4, sticky="e"
    )

    lf_mosaic_log = ttk.LabelFrame(tab_mosaic, text="Log", bootstyle="info")
    lf_mosaic_log.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
    mosaic_log_frame = ttk.Frame(lf_mosaic_log)
    mosaic_log_frame.pack(fill=tk.BOTH, expand=True)
    mosaic_log_frame.columnconfigure(0, weight=1)
    mosaic_log_frame.rowconfigure(0, weight=1)
    mosaic_log = ttk.Treeview(mosaic_log_frame, columns=("line",), show="headings", height=3)
    mosaic_log.heading("line", text="Mosaic log")
    mosaic_log.column("line", anchor="w", stretch=True, width=880)
    mosaic_log.grid(in_=mosaic_log_frame, row=0, column=0, sticky="nsew")
    mosaic_scroll = ttk.Scrollbar(mosaic_log_frame, orient="vertical", command=mosaic_log.yview)
    mosaic_scroll.grid(row=0, column=1, sticky="ns")
    mosaic_log.configure(yscrollcommand=mosaic_scroll.set)
    _bind_treeview_copy(mosaic_log)

    # ---------------- Tab 2: H3 ----------------
    ttk.Label(tab_h3, text="Generate H3 geocodes from existing asset/geocode coverage.").pack(anchor="w", padx=8, pady=(8, 4))

    size_frame = ttk.LabelFrame(tab_h3, text="H3 generation", bootstyle="secondary")
    size_frame.pack(fill=tk.X, padx=8, pady=(2, 6))
    size_frame.columnconfigure(5, weight=1)

    ttk.Label(size_frame, text="Min m:").grid(row=0, column=0, padx=4, pady=4, sticky="e")
    ttk.Label(size_frame, text="Max m:").grid(row=0, column=2, padx=4, pady=4, sticky="e")
    min_var = tk.StringVar(value="50")
    max_var = tk.StringVar(value="50000")
    ttk.Entry(size_frame, textvariable=min_var, width=12).grid(row=0, column=1, padx=4, pady=4, sticky="w")
    ttk.Entry(size_frame, textvariable=max_var, width=12).grid(row=0, column=3, padx=4, pady=4, sticky="w")

    size_levels_var = tk.StringVar(value="(none)")
    ttk.Label(size_frame, text="Matching levels:").grid(row=1, column=0, padx=4, pady=2, sticky="e")
    ttk.Label(size_frame, textvariable=size_levels_var, anchor="w").grid(row=1, column=1, columnspan=5, padx=4, pady=2, sticky="w")

    clear_h3_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(
        size_frame,
        text="Delete existing H3 before generating",
        variable=clear_h3_var,
        bootstyle=INFO,
    ).grid(row=2, column=0, columnspan=4, padx=4, pady=(2, 6), sticky="w")

    def _suggest_levels():
        _set_log_target(h3_log)
        log_to_gui("[H3] Suggest requested.", "INFO")
        try:
            min_m = float(min_var.get())
            max_m = float(max_var.get())
            if min_m <= 0 or max_m <= 0 or max_m < min_m:
                raise ValueError
        except Exception:
            log_to_gui("Enter valid positive meter values (min <= max).", "WARN")
            return

        min_km, max_km = min_m / 1000.0, max_m / 1000.0
        levels = suggest_h3_levels_by_size(min_km, max_km)
        size_levels_var.set(format_level_size_list(levels))

        def _generate_size_based():
            _set_log_target(h3_log)
            if not levels:
                log_to_gui("No suggested levels to generate.", "WARN")
                return
            log_to_gui(f"[H3] Generate requested for levels: {levels}", "INFO")
            _run_in_thread(write_h3_levels, base, levels, clear_existing=bool(clear_h3_var.get()))

        gen_btn.configure(command=_generate_size_based, state=("normal" if levels else "disabled"))
        log_to_gui(f"Suggested H3 levels: {levels}" if levels else "No H3 levels for that size range.", "INFO")

    btn_h3 = ttk.Frame(size_frame)
    btn_h3.grid(row=0, column=5, rowspan=3, padx=4, pady=4, sticky="e")
    sugg_btn = ttk.Button(btn_h3, text="Suggest H3", width=16, bootstyle=PRIMARY, command=_suggest_levels)
    gen_btn = ttk.Button(btn_h3, text="Generate H3", width=16, bootstyle=PRIMARY, state="disabled")
    sugg_btn.pack(side=tk.TOP, padx=2, pady=(0, 4), anchor="e")
    gen_btn.pack(side=tk.TOP, padx=2, pady=0, anchor="e")

    if h3 is None:
        size_levels_var.set("H3 library missing (pip install h3)")
        sugg_btn.configure(state="disabled")
        gen_btn.configure(state="disabled")

    lf_h3_log = ttk.LabelFrame(tab_h3, text="Log", bootstyle="info")
    lf_h3_log.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
    h3_log_frame = ttk.Frame(lf_h3_log)
    h3_log_frame.pack(fill=tk.BOTH, expand=True)
    h3_log_frame.columnconfigure(0, weight=1)
    h3_log_frame.rowconfigure(0, weight=1)
    h3_log = ttk.Treeview(h3_log_frame, columns=("line",), show="headings", height=4)
    h3_log.heading("line", text="H3 log")
    h3_log.column("line", anchor="w", stretch=True, width=860)
    h3_log.grid(in_=h3_log_frame, row=0, column=0, sticky="nsew")
    h3_scroll = ttk.Scrollbar(h3_log_frame, orient="vertical", command=h3_log.yview)
    h3_scroll.grid(row=0, column=1, sticky="ns")
    h3_log.configure(yscrollcommand=h3_scroll.set)
    _bind_treeview_copy(h3_log)

    # ---------------- Tab 3: Import geocodes ----------------
    ttk.Label(tab_import, text="Import geocode datasets and manage existing geocode groups.").pack(anchor="w", padx=8, pady=(8, 4))

    import_actions = ttk.LabelFrame(tab_import, text="Import", bootstyle="secondary")
    import_actions.pack(fill=tk.X, padx=8, pady=(2, 6))
    ttk.Label(import_actions, text="Import geocode datasets from input folder into GeoParquet tables.").pack(side=tk.LEFT, padx=8, pady=6)

    group_frame = ttk.LabelFrame(tab_import, text="Geocode groups", bootstyle="secondary")
    group_frame.pack(fill=tk.X, expand=False, padx=8, pady=(0, 6))
    group_frame.columnconfigure(0, weight=1)
    group_frame.rowconfigure(0, weight=1)

    group_tree = ttk.Treeview(
        group_frame,
        columns=("id", "gis", "name", "origin", "title", "objects"),
        show="headings",
        selectmode="extended",
        height=4,
    )
    group_tree.heading("id", text="ID")
    group_tree.heading("gis", text="GIS group")
    group_tree.heading("name", text="Layer name")
    group_tree.heading("origin", text="Origin")
    group_tree.heading("title", text="User title")
    group_tree.heading("objects", text="Objects")
    group_tree.column("id", width=70, anchor="w", stretch=False)
    group_tree.column("gis", width=220, anchor="w")
    group_tree.column("name", width=220, anchor="w")
    group_tree.column("origin", width=110, anchor="w", stretch=False)
    group_tree.column("title", width=180, anchor="w")
    group_tree.column("objects", width=90, anchor="e", stretch=False)
    group_tree.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=6)

    group_scroll = ttk.Scrollbar(group_frame, orient="vertical", command=group_tree.yview)
    group_tree.configure(yscrollcommand=group_scroll.set)
    group_scroll.grid(row=0, column=1, sticky="ns", padx=(0, 6), pady=6)

    import_status_var = tk.StringVar(value="")
    ttk.Label(tab_import, textvariable=import_status_var, bootstyle="secondary").pack(fill=tk.X, padx=8, pady=(0, 4))

    import_group_actions = ttk.Frame(tab_import)
    import_group_actions.pack(fill=tk.X, padx=8, pady=(0, 6))

    # ---------------- Tab 4: Edit geocodes ----------------
    ttk.Label(tab_edit, text="Edit geocode names, user titles and descriptions.").pack(anchor="w", padx=8, pady=(8, 4))

    read_path = geoparquet_path(base, "tbl_geocode_group")
    write_path = gpq_dir(base) / "tbl_geocode_group.parquet"

    edit_state_var = tk.StringVar(value="")
    edit_counter_var = tk.StringVar(value="0 / 0")
    edit_info_var = tk.StringVar(value="")
    edit_id_var = tk.StringVar(value="")
    edit_gis_var = tk.StringVar(value="")
    edit_name_var = tk.StringVar(value="")
    edit_title_var = tk.StringVar(value="")

    geocode_df = None
    edit_idx = 0

    ttk.Label(tab_edit, textvariable=edit_state_var).pack(fill=tk.X, padx=8, pady=(0, 6))

    edit_meta = ttk.LabelFrame(tab_edit, text="Current geocode", bootstyle="secondary")
    edit_meta.pack(fill=tk.X, padx=8, pady=(0, 6))
    ttk.Label(edit_meta, text="ID:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    ttk.Label(edit_meta, textvariable=edit_id_var).grid(row=0, column=1, sticky="w", padx=2, pady=4)
    ttk.Label(edit_meta, text="GIS name:").grid(row=0, column=2, sticky="w", padx=(16, 6), pady=4)
    ttk.Label(edit_meta, textvariable=edit_gis_var).grid(row=0, column=3, sticky="w", padx=2, pady=4)

    edit_form = ttk.LabelFrame(tab_edit, text="Editable fields", bootstyle="secondary")
    edit_form.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
    edit_form.columnconfigure(1, weight=1)

    ttk.Label(edit_form, text="Layer name").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    ttk.Entry(edit_form, textvariable=edit_name_var, width=44).grid(row=0, column=1, sticky="ew", padx=6, pady=4)
    ttk.Label(edit_form, text="User title").grid(row=1, column=0, sticky="w", padx=6, pady=4)
    ttk.Entry(edit_form, textvariable=edit_title_var, width=44).grid(row=1, column=1, sticky="ew", padx=6, pady=4)

    ttk.Label(edit_form, text="Description").grid(row=2, column=0, sticky="nw", padx=6, pady=4)
    edit_desc_txt = scrolledtext.ScrolledText(edit_form, height=7, wrap="word")
    edit_desc_txt.grid(row=2, column=1, sticky="nsew", padx=6, pady=4)
    edit_form.rowconfigure(2, weight=1)

    ttk.Label(tab_edit, textvariable=edit_info_var, bootstyle="secondary").pack(fill=tk.X, padx=8, pady=(0, 6))

    def _load_geocode_group_df():
        cols = ["id", "name", "name_gis_geocodegroup", "geocode_origin", "title_user", "description", "geometry"]
        try:
            gdf, _ = _load_existing_geocodes(base)
            for c in cols:
                if c not in gdf.columns:
                    gdf[c] = ""
            return gdf
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to read geocode group file:\n{exc}")
            return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs="EPSG:4326")

    def _refresh_group_list():
        try:
            existing_g, existing_o = _load_existing_geocodes(base)
        except Exception as exc:
            import_status_var.set(f"Failed to load geocode groups: {exc}")
            return

        obj_counts: dict[int, int] = {}
        if not existing_o.empty and "ref_geocodegroup" in existing_o.columns:
            try:
                s = existing_o["ref_geocodegroup"].dropna().astype(int).value_counts()
                obj_counts = {int(k): int(v) for k, v in s.items()}
            except Exception:
                obj_counts = {}

        for item in group_tree.get_children():
            group_tree.delete(item)

        if existing_g.empty:
            import_status_var.set("No geocode groups found.")
            return

        try:
            rows = existing_g.sort_values(by=["name_gis_geocodegroup", "id"], ascending=[True, True], na_position="last")
        except Exception:
            rows = existing_g

        for idx, row in rows.iterrows():
            rid_raw = row.get("id", "")
            try:
                rid = int(rid_raw)
            except Exception:
                rid = -1
            gis_name = str(row.get("name_gis_geocodegroup", "") or "")
            layer_name = str(row.get("name", "") or "")
            origin = _normalize_geocode_origin(row.get("geocode_origin", ""), gis_name)
            title_user = str(row.get("title_user", "") or "")
            object_count = obj_counts.get(rid, 0) if rid >= 0 else 0

            item_id = f"grp_{idx}_{rid}_{gis_name}" if gis_name else f"grp_{idx}_{rid}"
            group_tree.insert(
                "",
                tk.END,
                iid=item_id,
                values=(rid if rid >= 0 else "", gis_name, layer_name, origin, title_user, object_count),
            )

        import_status_var.set(f"Groups: {len(existing_g)} | Total objects: {len(existing_o)}")

    def _delete_selected_groups():
        _set_log_target(import_log)
        log_to_gui("[Import] Delete selected requested.", "INFO")
        selected = group_tree.selection()
        if not selected:
            messagebox.showinfo("Delete geocodes", "Select one or more geocode groups to delete.")
            return

        names_imported: list[str] = []
        skipped_non_imported: list[str] = []
        for item in selected:
            vals = group_tree.item(item, "values")
            if len(vals) >= 4:
                name = str(vals[1] or "").strip()
                origin = _normalize_geocode_origin(vals[3], name)
                if name and origin == GEOCODE_ORIGIN_IMPORTED:
                    names_imported.append(name)
                elif name:
                    skipped_non_imported.append(name)
        names_imported = sorted(set(names_imported))
        skipped_non_imported = sorted(set(skipped_non_imported))
        if not names_imported:
            message = "Only imported geocode groups can be deleted from this tab."
            if skipped_non_imported:
                preview_skip = ", ".join(skipped_non_imported[:4])
                if len(skipped_non_imported) > 4:
                    preview_skip += f" (+{len(skipped_non_imported) - 4} more)"
                message += f"\n\nSkipped: {preview_skip}"
            messagebox.showwarning("Delete geocodes", message)
            return

        preview = ", ".join(names_imported[:4])
        if len(names_imported) > 4:
            preview += f" (+{len(names_imported) - 4} more)"
        skip_line = ""
        if skipped_non_imported:
            skip_preview = ", ".join(skipped_non_imported[:4])
            if len(skipped_non_imported) > 4:
                skip_preview += f" (+{len(skipped_non_imported) - 4} more)"
            skip_line = f"\n\nWill be skipped (not imported):\n{skip_preview}"
        ok = messagebox.askyesno(
            "Confirm delete",
            f"Delete selected imported geocode groups?\n\n{preview}{skip_line}\n\nThis also removes linked geocode objects.",
        )
        if not ok:
            return

        _set_log_target(import_log)
        _clear_geocode_groups(base, names_imported)
        log_to_gui(f"[Import] Deleted imported groups: {', '.join(names_imported)}", "INFO")
        _refresh_group_list()
        _refresh_edit_data()

    def _run_import_and_refresh():
        _set_log_target(import_log)
        log_to_gui("[Import] Import requested.", "INFO")

        def _job():
            run_import_geocodes(base, cfg)
            try:
                root.after(0, _refresh_group_list)
                root.after(0, _refresh_edit_data)
            except Exception:
                pass

        _run_in_thread(_job)

    def _update_counter():
        nonlocal geocode_df, edit_idx
        total = len(geocode_df) if geocode_df is not None else 0
        current = (edit_idx + 1) if total else 0
        edit_counter_var.set(f"{current} / {total}")

    def _clear_editor():
        edit_id_var.set("")
        edit_gis_var.set("")
        edit_name_var.set("")
        edit_title_var.set("")
        edit_desc_txt.delete("1.0", tk.END)
        _update_counter()

    def _load_record():
        nonlocal geocode_df, edit_idx
        if geocode_df is None or len(geocode_df) == 0:
            _clear_editor()
            edit_state_var.set("No geocode groups found. Build/import geocodes first.")
            return
        edit_idx = max(0, min(edit_idx, len(geocode_df) - 1))
        row = geocode_df.iloc[edit_idx]
        edit_id_var.set(str(row.get("id", "") or ""))
        edit_gis_var.set(str(row.get("name_gis_geocodegroup", "") or ""))
        edit_name_var.set(str(row.get("name", "") or ""))
        edit_title_var.set(str(row.get("title_user", "") or ""))
        edit_desc_txt.delete("1.0", tk.END)
        edit_desc_txt.insert(tk.END, str(row.get("description", "") or ""))
        _update_counter()
        edit_state_var.set(f"Record {edit_idx + 1} of {len(geocode_df)}")

    def _write_back_to_df():
        nonlocal geocode_df, edit_idx
        if geocode_df is None or len(geocode_df) == 0:
            return
        geocode_df.at[edit_idx, "name"] = (edit_name_var.get() or "").strip()
        geocode_df.at[edit_idx, "title_user"] = (edit_title_var.get() or "").strip()
        geocode_df.at[edit_idx, "description"] = edit_desc_txt.get("1.0", tk.END).strip()

    def _save_current() -> bool:
        nonlocal geocode_df
        if geocode_df is None or len(geocode_df) == 0:
            messagebox.showinfo("Nothing to save", "There are no geocode groups to save.")
            return False

        _write_back_to_df()
        write_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = write_path.with_suffix(".tmp.parquet")
        try:
            geocode_df.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, write_path)
            edit_state_var.set("Saved.")
            return True
        except Exception as exc:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            messagebox.showerror("Error", f"Failed to save geocode edits:\n{exc}")
            return False

    def _save_and_next():
        if _save_current():
            _navigate(+1)

    def _navigate(step: int):
        nonlocal geocode_df, edit_idx
        if geocode_df is None or len(geocode_df) == 0:
            return
        _write_back_to_df()
        edit_idx = max(0, min(edit_idx + int(step), len(geocode_df) - 1))
        _load_record()

    def _refresh_edit_data():
        nonlocal geocode_df, edit_idx
        geocode_df = _load_geocode_group_df()
        edit_idx = min(edit_idx, max((len(geocode_df) if geocode_df is not None else 0) - 1, 0))
        _load_record()
        total = len(geocode_df) if geocode_df is not None else 0
        actual_path = _existing_parquet_path(base, "tbl_geocode_group") or write_path
        edit_info_var.set(f"Geocode group file: {actual_path} | rows: {total}")

    def _delete_current_group():
        nonlocal geocode_df, edit_idx
        if geocode_df is None or len(geocode_df) == 0:
            messagebox.showinfo("Delete geocode", "There is no geocode group to delete.")
            return

        gis_name = (edit_gis_var.get() or "").strip()
        if not gis_name:
            messagebox.showerror("Delete geocode", "Current record has no GIS group name.")
            return

        ok = messagebox.askyesno(
            "Confirm delete",
            f"Delete geocode group '{gis_name}'?\n\nThis also removes linked geocode objects.",
        )
        if not ok:
            return

        _clear_geocode_groups(base, [gis_name])
        _refresh_group_list()
        _refresh_edit_data()
        edit_state_var.set(f"Deleted geocode group: {gis_name}")

    ttk.Button(import_actions, text="Import geocodes", bootstyle=PRIMARY, command=_run_import_and_refresh).pack(side=tk.RIGHT, padx=8, pady=6)
    ttk.Button(import_group_actions, text="Refresh list", command=_refresh_group_list).pack(side=tk.LEFT)
    ttk.Button(import_group_actions, text="Delete selected", bootstyle="danger", command=_delete_selected_groups).pack(side=tk.LEFT, padx=(6, 0))

    import_log_frame = ttk.LabelFrame(tab_import, text="Log", bootstyle="info")
    import_log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
    import_log_inner = ttk.Frame(import_log_frame)
    import_log_inner.pack(fill=tk.BOTH, expand=True)
    import_log_inner.columnconfigure(0, weight=1)
    import_log_inner.rowconfigure(0, weight=1)
    import_log = ttk.Treeview(import_log_inner, columns=("line",), show="headings", height=3)
    import_log.heading("line", text="Import log")
    import_log.column("line", anchor="w", stretch=True, width=860)
    import_log.grid(in_=import_log_inner, row=0, column=0, sticky="nsew")
    import_scroll = ttk.Scrollbar(import_log_inner, orient="vertical", command=import_log.yview)
    import_scroll.grid(row=0, column=1, sticky="ns")
    import_log.configure(yscrollcommand=import_scroll.set)
    _bind_treeview_copy(import_log)

    log_widgets = [h3_log, import_log]

    edit_actions = ttk.Frame(tab_edit)
    edit_actions.pack(fill=tk.X, padx=8, pady=(0, 8))
    ttk.Label(edit_actions, textvariable=edit_counter_var).pack(side=tk.LEFT)
    ttk.Button(edit_actions, text="Previous", command=lambda: _navigate(-1)).pack(side=tk.LEFT, padx=(12, 0))
    ttk.Button(edit_actions, text="Next", command=lambda: _navigate(1)).pack(side=tk.LEFT, padx=(6, 0))
    ttk.Button(edit_actions, text="Reload", command=_refresh_edit_data).pack(side=tk.RIGHT, padx=(0, 6))
    ttk.Button(edit_actions, text="Delete", bootstyle="danger", command=_delete_current_group).pack(side=tk.RIGHT, padx=(0, 6))
    ttk.Button(edit_actions, text="Save & Next", bootstyle="primary", command=_save_and_next).pack(side=tk.RIGHT, padx=(0, 6))
    ttk.Button(edit_actions, text="Save", bootstyle="success", command=_save_current).pack(side=tk.RIGHT)

    _refresh_group_list()
    _refresh_edit_data()

    # ---------------- Shared footer ----------------
    footer = ttk.Frame(root)
    footer.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 8))
    ttk.Button(footer, text="Exit", bootstyle=WARNING, command=root.destroy).pack(side=tk.RIGHT)

    progress_var = tk.DoubleVar()
    pbar = ttk.Progressbar(footer, orient="horizontal", mode="determinate", variable=progress_var, bootstyle="info")
    pbar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

    progress_label = ttk.Label(footer, text="0%")
    progress_label.place(in_=pbar, relx=0.5, rely=0.5, anchor="center")

    try:
        root.resizable(True, True)
    except Exception:
        pass

    try:
        notebook.bind("<<NotebookTabChanged>>", _sync_log_target_with_tab)
    except Exception:
        pass
    _sync_log_target_with_tab()

    startup_line = f"{datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S')} [INFO] - Log window initialized."
    for _lw in (h3_log, import_log):
        try:
            if isinstance(_lw, ttk.Treeview):
                iid = _lw.insert("", tk.END, values=(startup_line,))
                _lw.see(iid)
            else:
                _lw.insert(tk.END, startup_line + "\n")
                _lw.see(tk.END)
        except Exception:
            pass
    try:
        mosaic_log.insert("", tk.END, values=(startup_line,))
    except Exception:
        pass

    _mosaic_tail_state: dict[str, object] = {
        "offset": 0,
        "carry": "",
        "active": False,
    }

    def _clear_mosaic_log_view() -> None:
        try:
            for iid in mosaic_log.get_children():
                mosaic_log.delete(iid)
        except Exception:
            pass

    def _append_mosaic_line(line: str) -> None:
        try:
            iid = mosaic_log.insert("", tk.END, values=(line,))
            mosaic_log.see(iid)
            children = mosaic_log.get_children()
            if len(children) > 2500:
                for old_iid in children[:500]:
                    mosaic_log.delete(old_iid)
        except Exception:
            pass

    def _refresh_mosaic_log_from_file() -> None:
        try:
            if root is None or not root.winfo_exists():
                return
        except Exception:
            return

        try:
            if not bool(_mosaic_tail_state.get("active", False)):
                try:
                    lp = Path(original_working_directory or str(base)) / "log.txt"
                    if lp.exists():
                        _mosaic_tail_state["offset"] = int(lp.stat().st_size)
                except Exception:
                    pass
            else:
                lp = Path(original_working_directory or str(base)) / "log.txt"
                if lp.exists() and lp.is_file():
                    start_offset = int(_mosaic_tail_state.get("offset", 0) or 0)
                    carry = str(_mosaic_tail_state.get("carry", "") or "")
                    with open(lp, "r", encoding="utf-8", errors="replace") as f:
                        f.seek(max(0, start_offset))
                        chunk = f.read()
                        _mosaic_tail_state["offset"] = int(f.tell())

                    if chunk:
                        text = carry + chunk
                        lines = text.splitlines()
                        if text and not text.endswith(("\n", "\r")):
                            _mosaic_tail_state["carry"] = lines.pop() if lines else text
                        else:
                            _mosaic_tail_state["carry"] = ""

                        for ln in lines:
                            _append_mosaic_line(ln)
                            if "Step [Mosaic] COMPLETED" in ln or "Step [Mosaic] FAILED" in ln:
                                _mosaic_tail_state["active"] = False
        except Exception:
            pass

        try:
            root.after(350, _refresh_mosaic_log_from_file)
        except Exception:
            pass

    try:
        root.after(350, _refresh_mosaic_log_from_file)
    except Exception:
        pass

    # NOTE: Keep mosaic log as direct in-memory stream from log_to_gui.
    # This mirrors previous working behavior and avoids file-tail overwrite races.

    log_to_gui(f"Base dir: {base}")
    log_to_gui(f"GeoParquet out: {gpq_dir(base)}")
    log_to_gui("Geocode manage ready (Basic mosaic / H3 geocodes / Import geocodes / Edit geocodes).")
    _update_mosaic_status()
    root.mainloop()

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Manage geocodes (Mosaic/H3/Edit)")
    parser.add_argument("--nogui", action="store_true", help="Run in CLI mode")
    parser.add_argument("--original_working_directory", required=False, help="Path to running folder")

    parser.add_argument("--h3", action="store_true", help="Generate H3 by range")
    parser.add_argument("--h3-from", dest="h3_from", type=int, default=3)
    parser.add_argument("--h3-to", dest="h3_to", type=int, default=6)
    parser.add_argument("--h3-levels", dest="h3_levels", type=str, default="", help="Comma-separated list, e.g. 5,6,7")

    parser.add_argument("--mosaic", action="store_true", help="Generate basic mosaic and publish as geocode")
    parser.add_argument("--buffer-m", dest="buffer_m", type=float, default=25.0)
    parser.add_argument("--grid-size-m", dest="grid_size_m", type=float, default=1000.0)
    parser.add_argument("--import-geocodes", action="store_true", help="Import geocodes in CLI mode")
    parser.add_argument("--start-tab", default="", help="GUI startup tab: mosaic|h3|import|edit")

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
        elif args.import_geocodes:
            run_import_geocodes(base, cfg)
        else:
            log_to_gui("Nothing to do. Use --h3, --h3-levels, --mosaic, or --import-geocodes.", "WARN")
    else:
        build_gui(base, cfg, start_tab=args.start_tab)

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

