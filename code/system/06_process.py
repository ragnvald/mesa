# -*- coding: utf-8 -*-
# 06_process.py — memory-aware, CPU-optimized (Windows spawn-safe) intersections + robust flattening to GeoParquet
# - Intersect tbl_asset_object × tbl_geocode_object -> tbl_stacked (folder dataset of Parquet parts)
# - Flatten by geocode code -> tbl_flat.parquet with min/max + A..E codes from config.ini
# - Compute area once per tile (equal-area CRS), then backfill to tbl_stacked (streaming over parts)
# - Compute visualization ENV index (1–100) + optional components: env_imp, env_sens, env_susc, env_press
# - Heartbeat logs (progress + memory) + adaptive splitting to avoid OOM + lower process priority
#
# Inputs  (GeoParquet): output/geoparquet/{tbl_asset_object,tbl_geocode_object,tbl_asset_group}.parquet
# Outputs (GeoParquet): output/geoparquet/{tbl_stacked/ (dataset), tbl_flat.parquet}

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

import os, sys, math, time, random, argparse, threading, multiprocessing, json, shutil, uuid, gc
import configparser
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

# Optional system introspection (recommended)
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

# --- GUI
import tkinter as tk
from tkinter import scrolledtext
import ttkbootstrap as tb
from ttkbootstrap.constants import PRIMARY, WARNING

# ----------------------------
# Globals (initialized in __main__)
# ----------------------------
original_working_directory = None
log_widget = None
progress_var = None
progress_label = None
HEARTBEAT_SECS = 30

# spill-to-disk for intersections
_PARTS_DIR = None  # folder where workers write chunk parquet parts

# ----------------------------
# Paths
# ----------------------------
def base_dir() -> Path:
    bd = Path(original_working_directory or os.getcwd())
    if bd.name.lower() == "system":
        return bd.parent
    return bd

def gpq_dir() -> Path:
    out = base_dir() / "output" / "geoparquet"
    out.mkdir(parents=True, exist_ok=True)
    return out

def _dataset_dir(name: str) -> Path:
    # e.g. .../geoparquet/tbl_stacked  (folder dataset)
    return gpq_dir() / name

# ----------------------------
# Logging / UI helpers
# ----------------------------
def update_progress(new_value: float):
    try:
        v = max(0.0, min(100.0, float(new_value)))
        if progress_var is not None:
            progress_var.set(v)
        if progress_label is not None:
            progress_label.config(text=f"{int(v)}%")
            progress_label.update_idletasks()
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

def read_config(path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8")
    if "DEFAULT" not in cfg:
        cfg["DEFAULT"] = {}
    return cfg

# ----------------------------
# Config-driven A..E classification
# ----------------------------
def read_class_ranges(cfg_path: Path):
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path, encoding="utf-8")
    ranges = {}
    desc   = {}
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
    # Priority: A highest … E lowest
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
# IO helpers (GeoParquet + dataset support)
# ----------------------------
def read_parquet_or_empty(name: str) -> gpd.GeoDataFrame:
    """Read a single GeoParquet file OR a partitioned dataset folder; return empty GDF if missing."""
    file_path = gpq_dir() / f"{name}.parquet"
    dir_path  = _dataset_dir(name)
    try:
        if file_path.exists():
            return gpd.read_parquet(file_path)
        if dir_path.exists() and dir_path.is_dir():
            # read dataset folder (pyarrow dataset)
            return gpd.read_parquet(str(dir_path))
        return gpd.GeoDataFrame(geometry=[], crs=None)
    except Exception as e:
        log_to_gui(log_widget, f"Failed to read {name}: {e}")
        return gpd.GeoDataFrame(geometry=[], crs=None)

def write_parquet(name: str, gdf: gpd.GeoDataFrame):
    """Write a single GeoParquet file."""
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
    # remove single-file outputs
    for fn in ["tbl_stacked.parquet","tbl_flat.parquet"]:
        _rm_rf(gpq_dir() / fn)
    # remove datasets / temp parts
    for d in ["tbl_stacked", "__stacked_parts"]:
        _rm_rf(_dataset_dir(d))

# ----------------------------
# Spatial utilities for intersections (robust, spill-to-disk; Windows-safe)
# ----------------------------
from uuid import uuid4

def create_grid(geodata: gpd.GeoDataFrame, cell_size_deg: float):
    xmin, ymin, xmax, ymax = geodata.total_bounds
    # guard against empty/degenerate bounds
    if not np.isfinite([xmin, ymin, xmax, ymax]).all() or xmax <= xmin or ymax <= ymin:
        return []
    x_edges = np.arange(xmin, xmax + cell_size_deg, cell_size_deg)
    y_edges = np.arange(ymin, ymax + cell_size_deg, cell_size_deg)
    cells = []
    for x in x_edges[:-1]:
        for y in y_edges[:-1]:
            cells.append((x, y, x + cell_size_deg, y + cell_size_deg))
    return cells

# --- grid-tagging workers (write parts, return file paths only) ---
_GRID_OUT_DIR = None
_GRID_GDF     = None

def _grid_pool_init2(grid_gdf, out_dir_str: str):
    """Initializer for grid-tag workers. Avoids sending big frames back to parent."""
    global _GRID_OUT_DIR, _GRID_GDF
    _GRID_OUT_DIR = Path(out_dir_str)
    _GRID_OUT_DIR.mkdir(parents=True, exist_ok=True)
    _GRID_GDF = grid_gdf
    # warm up spatial index to avoid races
    try:
        _ = _GRID_GDF.sindex
    except Exception:
        pass
    # lower worker priority to keep desktop responsive
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
    """Read one geocode chunk parquet, tag to grid, write tagged parquet, return its path."""
    g = gpd.read_parquet(input_path)
    # build sindex to force spatial index creation inside worker
    try:
        _ = g.sindex
    except Exception:
        pass
    j = gpd.sjoin(g, _GRID_GDF, how="left", predicate="intersects")
    j.drop(columns=["index_right"], inplace=True, errors="ignore")
    # simple clean
    try:
        j = j[j.geometry.notna() & ~j.geometry.is_empty]
    except Exception:
        pass
    out = _GRID_OUT_DIR / f"grid_tag_{os.getpid()}_{uuid4().hex}.parquet"
    j.to_parquet(out, index=False)
    # free memory early
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

def calculate_rows_per_chunk(n: int,
                             max_memory_mb: int = 256,
                             est_bytes_per_row: int = 1800,
                             hard_cap_rows: int = 100_000) -> int:
    """Heuristic: ~1.8KB/row default, cap chunks to keep IPC small."""
    try:
        rows = int((max_memory_mb * 1024 * 1024) / max(256, est_bytes_per_row))
    except Exception:
        rows = 50_000
    rows = max(5_000, min(rows, hard_cap_rows))
    return max(1, min(rows, n))

def assign_geocodes_to_grid(geodata: gpd.GeoDataFrame, meters_cell: int, max_workers: int) -> gpd.GeoDataFrame:
    """
    Tag every geocode with a coarse grid cell id, using spill-to-disk to avoid
    sending large GeoDataFrames over multiprocessing pipes (Windows-safe).
    """
    if geodata is None or geodata.empty:
        return _mk_empty_gdf_like(geodata.crs if geodata is not None else None)

    # meters→degrees (approx) for WGS84
    meters_per_degree = 111_320.0
    cell_deg = meters_cell / meters_per_degree

    # build grid
    grid_cells = create_grid(geodata, cell_deg)
    if not grid_cells:
        log_to_gui(log_widget, "Grid creation produced no cells; skipping tagging.")
        return geodata.assign(grid_cell=pd.Series([0] * len(geodata), index=geodata.index))

    grid_gdf = gpd.GeoDataFrame(
        {"grid_cell": range(len(grid_cells)),
         "geometry": [box(x0, y0, x1, y1) for (x0, y0, x1, y1) in grid_cells]},
        geometry="geometry", crs=geodata.crs
    )
    # kick sindex
    try:
        _ = grid_gdf.sindex
    except Exception:
        pass
    log_to_gui(log_widget, f"Assigning geocodes to {len(grid_cells):,} grid cells…")

    # temporary folders (input chunks, output parts)
    tmp_in  = _dataset_dir("__grid_assign_in")
    tmp_out = _dataset_dir("__grid_assign_out")
    _rmrf_dir(tmp_in); _rmrf_dir(tmp_out)
    tmp_in.mkdir(parents=True, exist_ok=True)
    tmp_out.mkdir(parents=True, exist_ok=True)

    # chunk & write geocodes to input parts
    rows_per_chunk = calculate_rows_per_chunk(len(geodata))
    total_chunks = int(math.ceil(len(geodata) / rows_per_chunk))
    input_parts = []
    for i in range(0, len(geodata), rows_per_chunk):
        part = geodata.iloc[i:i+rows_per_chunk]
        p = tmp_in / f"geo_{i:09d}.parquet"
        part.to_parquet(p, index=False)
        input_parts.append(str(p))

    # process with pool; workers only exchange small strings (paths)
    started_at = time.time()
    last_ping = started_at
    done = 0
    out_files = []

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
                    if dd > 0:
                        eta += f" (+{dd}d)"
                log_to_gui(log_widget, f"[grid-assign] {done}/{total_chunks} chunks (~{pct:.2f}%) • ETA {eta}")
                last_ping = now

    # assemble output
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
    # cleanup temp dirs
    _rmrf_dir(tmp_in); _rmrf_dir(tmp_out)
    return tagged


def make_spatial_chunks(geocode_tagged: gpd.GeoDataFrame, max_workers: int, multiplier: int = 18):
    # finer chunking → smaller per-chunk memory footprint (grid size unchanged)
    cell_ids = geocode_tagged['grid_cell'].unique().tolist()
    random.shuffle(cell_ids)
    target_chunks = max(1, min(len(cell_ids), max_workers * multiplier))
    cells_per_chunk = max(1, math.ceil(len(cell_ids) / target_chunks))
    chunks = []
    for i in range(0, len(cell_ids), cells_per_chunk):
        sel = set(cell_ids[i:i + cells_per_chunk])
        chunks.append(geocode_tagged[geocode_tagged['grid_cell'].isin(sel)])
    return chunks

# ----------------------------
# Adaptive intersection helpers (workers)
# ----------------------------
_POOL_ASSETS = None
_POOL_TYPES  = None
_ASSET_SOFT_LIMIT = 200_000   # can be overridden from config via env in initializer
_GEOCODE_SOFT_LIMIT = 160

def _intersect_pool_init(asset_df, geom_types, parts_dir, asset_soft_limit, geocode_soft_limit):
    """Worker initializer: capture shared data and init per-process state."""
    global _POOL_ASSETS, _POOL_TYPES, _PARTS_DIR, _ASSET_SOFT_LIMIT, _GEOCODE_SOFT_LIMIT
    _POOL_ASSETS = asset_df
    _POOL_TYPES  = geom_types
    _PARTS_DIR   = Path(parts_dir)
    _ASSET_SOFT_LIMIT = int(asset_soft_limit)
    _GEOCODE_SOFT_LIMIT = int(geocode_soft_limit)
    # Lower priority for workers (keeps desktop responsive)
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
    """Adaptive worker: spatially filters assets to chunk bbox, then joins per geometry type with recursive splitting on failure/pre-checks."""
    idx, geocode_chunk = args
    logs = []

    def write_parts(gdf):
        """Write (possibly very large) results by splitting if write fails."""
        nonlocal logs
        try:
            # sanitize geometries that might be empty/invalid
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
        """Join a single geocode subset with a subset of assets; split assets recursively on failure."""
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
        """Perform joins for a subset of geocodes; split geocode subset on failure or pre-empt if too many candidate assets."""
        if geocode_gdf.empty:
            return gpd.GeoDataFrame(geometry=[], crs=_POOL_ASSETS.crs)

        # Spatially filter global assets to the bbox of this geocode subset
        minx, miny, maxx, maxy = geocode_gdf.total_bounds
        candidate_idx = list(_POOL_ASSETS.sindex.intersection((minx, miny, maxx, maxy)))
        asset_subset = _POOL_ASSETS.iloc[candidate_idx] if candidate_idx else _POOL_ASSETS.iloc[:0]

        # PRE-EMPTIVE split if candidate set obviously too big
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

        # Also keep geocode subset sizes reasonable
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
                    # 1) Split geocode subset if multiple rows
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
                    # 2) If one geocode row, split assets
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

def intersect_assets_geocodes(asset_data, geocode_data, cell_size_m, max_workers,
                              asset_soft_limit, geocode_soft_limit):
    # tag geocodes with coarse grid to improve locality / chunking (grid size unchanged)
    log_to_gui(log_widget, "Creating analysis grid + tagging geocodes.")
    geocode_tagged = assign_geocodes_to_grid(geocode_data, cell_size_m, max_workers)

    # chunks
    chunks = make_spatial_chunks(geocode_tagged, max_workers, multiplier=18)
    total_chunks = len(chunks)
    log_to_gui(log_widget, f"Intersecting in {total_chunks} chunks with {max_workers} workers. Heartbeat every {HEARTBEAT_SECS}s.")
    update_progress(35)

    geom_types = asset_data.geometry.geom_type.unique().tolist()

    # spill directory (temporary)
    tmp_parts = _dataset_dir("__stacked_parts")
    _rm_rf(tmp_parts)  # in case of previous crash
    tmp_parts.mkdir(parents=True, exist_ok=True)

    written = 0
    files   = []
    error_msg = None
    started_at = time.time()
    last_ping  = started_at
    done_count = 0

    # Heartbeat thread (parent)
    hb_stop = threading.Event()
    progress_state = {'done': 0, 'total': total_chunks, 'rows': 0, 'started_at': started_at}
    def heartbeat():
        while not hb_stop.wait(HEARTBEAT_SECS):
            mem_txt = ""
            try:
                if psutil is not None:
                    vm = psutil.virtual_memory()
                    mem_txt = f" • RAM used {vm.percent:.0f}%"
            except Exception:
                pass
            d = progress_state['done']; t = progress_state['total']; r = progress_state['rows']
            elapsed = time.time() - progress_state['started_at']
            pct = (d / t) * 100 if t else 100.0
            eta = "?"
            if d:
                est_total = elapsed / d * t
                eta_ts = datetime.now() + timedelta(seconds=max(0.0, est_total - elapsed))
                eta = eta_ts.strftime("%H:%M:%S")
                dd = (eta_ts.date() - datetime.now().date()).days
                if dd > 0: eta += f" (+{dd}d)"
            log_to_gui(log_widget, f"[heartbeat] {d}/{t} chunks (~{pct:.2f}%) • rows written: {r:,}{mem_txt} • ETA {eta}")

    hb_thread = threading.Thread(target=heartbeat, daemon=True)
    hb_thread.start()

    iterable = ((i, ch) for i, ch in enumerate(chunks, start=1))
    with multiprocessing.get_context("spawn").Pool(
            processes=max_workers,
            initializer=_intersect_pool_init,
            initargs=(asset_data, geom_types, str(tmp_parts), asset_soft_limit, geocode_soft_limit)) as pool:

        for (idx, nrows, paths, err, logs) in pool.imap_unordered(_intersection_worker, iterable):
            done_count += 1
            progress_state['done'] = done_count

            # Propagate worker-side informative logs (splits, etc.)
            if logs:
                for line in logs:
                    log_to_gui(log_widget, line)

            if err:
                log_to_gui(log_widget, f"[intersect] Chunk {idx} failed: {err}")
                error_msg = err
                pool.terminate()  # stop other workers immediately
                break

            written += nrows
            progress_state['rows'] = written
            if paths:
                if isinstance(paths, list):
                    files.extend(paths)
                else:
                    files.append(paths)

            now = time.time()
            if (now - last_ping) >= HEARTBEAT_SECS or done_count == total_chunks:
                elapsed = now - started_at
                pct = (done_count / total_chunks) * 100 if total_chunks else 100.0
                eta = "?"
                if done_count:
                    est_total = elapsed / max(done_count, 1) * total_chunks
                    eta_ts = datetime.now() + timedelta(seconds=max(0.0, est_total - elapsed))
                    eta = eta_ts.strftime("%H:%M:%S")
                    dd = (eta_ts.date() - datetime.now().date()).days
                    if dd > 0:
                        eta += f" (+{dd}d)"
                log_to_gui(log_widget, f"[intersect] {done_count}/{total_chunks} chunks (~{pct:.2f}%) • rows written: {written:,} • ETA {eta}")
                update_progress(35.0 + (done_count / max(total_chunks, 1)) * 15.0)
                last_ping = now

            # Help GC in parent loop
            if done_count % 8 == 0:
                gc.collect()

    hb_stop.set()
    hb_thread.join(timeout=1.5)

    if error_msg:
        # Abort processing on first failure (no partial finalize)
        raise RuntimeError(error_msg)

    if not files:
        log_to_gui(log_widget, "No intersections; tbl_stacked is empty.")
        return gpd.GeoDataFrame(geometry=[], crs=geocode_data.crs)

    # Finalize: move temp parts to dataset folder "tbl_stacked"
    final_ds = _dataset_dir("tbl_stacked")
    _rm_rf(final_ds)
    tmp_parts.rename(final_ds)
    log_to_gui(log_widget, f"tbl_stacked dataset written as folder with {len(files)} parts and ~{written:,} rows: {final_ds}")
    return gpd.GeoDataFrame(geometry=[], crs=geocode_data.crs)

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

    # Read inputs
    assets   = read_parquet_or_empty("tbl_asset_object")
    geocodes = read_parquet_or_empty("tbl_geocode_object")
    groups   = read_parquet_or_empty("tbl_asset_group")

    if assets.empty:
        log_to_gui(log_widget, "ERROR: Missing or empty tbl_asset_object.parquet; aborting stacked build.")
        return
    if geocodes.empty:
        log_to_gui(log_widget, "ERROR: Missing or empty tbl_geocode_object.parquet; aborting stacked build.")
        return

    # Ensure CRS
    if assets.crs is None:
        assets.set_crs(f"EPSG:{working_epsg}", inplace=True)
    if geocodes.crs is None:
        geocodes.set_crs(f"EPSG:{working_epsg}", inplace=True)

    # Merge asset-group attributes (keep only present columns)
    if not groups.empty:
        cols = ['id','name_gis_assetgroup','total_asset_objects','importance',
                'susceptibility','sensitivity','sensitivity_code','sensitivity_description']
        keep = [c for c in cols if c in groups.columns]
        if keep:
            assets = assets.merge(groups[keep], left_on='ref_asset_group', right_on='id', how='left')

    update_progress(20)
    _ = assets.sindex; _ = geocodes.sindex

    # Workers (memory-aware)
    if max_workers == 0:
        try:
            max_workers = multiprocessing.cpu_count()
            log_to_gui(log_widget, f"Number of workers determined by system: {max_workers}")
        except NotImplementedError:
            max_workers = 4
    else:
        log_to_gui(log_widget, f"Number of workers set in config: {max_workers}")

    try:
        if psutil is not None:
            vm = psutil.virtual_memory()
            avail_gb = vm.available / (1024**3)
            budget_gb = max(1.0, avail_gb * mem_target_frac)  # keep some headroom
            allowed = max(1, int(budget_gb // max(0.5, approx_gb_per_worker)))
            if allowed < max_workers:
                log_to_gui(log_widget, f"Reducing workers from {max_workers} to {allowed} based on RAM (avail≈{avail_gb:.1f} GB, budget≈{budget_gb:.1f} GB, ~{approx_gb_per_worker:.1f} GB/worker).")
                max_workers = allowed
    except Exception:
        pass

    # Intersections → dataset folder
    _ = intersect_assets_geocodes(assets, geocodes, cell_size_m, max_workers,
                                  asset_soft_limit, geocode_soft_limit)

    # Sanity read
    try:
        sample = read_parquet_or_empty("tbl_stacked")
        log_to_gui(log_widget, f"tbl_stacked rows (sample read): {len(sample):,}")
    except Exception as e:
        log_to_gui(log_widget, f"tbl_stacked read check failed: {e}")

    update_progress(50)

# ----------------------------
# PROCESS: flatten to tbl_flat (min/max + codes + area + env_index)
# ----------------------------
def normalize_area_epsg(raw: str) -> str:
    v = (raw or "").strip().upper()
    if v.startswith("EPSG:"):
        v = v.split(":",1)[1]
    try:
        code = int(v)
        if code in (4326, 4258):  # geographic -> not for area
            return "EPSG:3035"
        return f"EPSG:{code}"
    except Exception:
        return "EPSG:3035"

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
            'env_index','env_imp','env_sens','env_susc','env_press',
            'geometry'
        ]
        gdf_empty = gpd.GeoDataFrame(columns=empty_cols, geometry='geometry', crs=f"EPSG:{working_epsg}")
        write_parquet("tbl_flat", gdf_empty)
        return

    # Ensure CRS
    if stacked.crs is None:
        stacked.set_crs(f"EPSG:{working_epsg}", inplace=True)

    # Numeric prep (no coercion to 0)
    bases = ["importance","sensitivity","susceptibility"]
    for b in bases:
        if b in stacked.columns:
            stacked[b] = pd.to_numeric(stacked[b], errors="coerce")
        else:
            stacked[b] = pd.Series(pd.NA, index=stacked.index, dtype="Float64")

    # --- groupby code ---
    keys = ["code"]
    # numeric min/max
    gnum = stacked.groupby(keys, dropna=False).agg({
        "importance": ["min","max"],
        "sensitivity": ["min","max"],
        "susceptibility": ["min","max"],
    })
    gnum.columns = [f"{c}_{s}" for c,s in gnum.columns]

    # meta
    gmeta = stacked.groupby(keys, dropna=False).agg({
        "ref_geocodegroup": "first",
        "name_gis_geocodegroup": "first",
        "geometry": "first",
        "ref_asset_group": pd.Series.nunique,
        "name_gis_assetgroup": (lambda s: ", ".join(pd.Series(s).dropna().astype(str).unique()))
    }).rename(columns={"ref_asset_group":"asset_groups_total", "name_gis_assetgroup":"asset_group_names"})

    # overlap count
    goverlap = stacked.groupby(keys, dropna=False).size().to_frame("assets_overlap_total")

    tbl_flat = pd.concat([gnum, gmeta, goverlap], axis=1).reset_index()
    tbl_flat = gpd.GeoDataFrame(tbl_flat, geometry="geometry", crs=stacked.crs)

    # --- derive A..E codes + descriptions from config ---
    ranges_map, desc_map, _order = read_class_ranges(config_file)

    def add_code_desc(df: pd.DataFrame, base: str) -> pd.DataFrame:
        cmin = df[f"{base}_min"].apply(lambda v: map_num_to_code(v, ranges_map)).astype("string")
        cmax = df[f"{base}_max"].apply(lambda v: map_num_to_code(v, ranges_map)).astype("string")
        df[f"{base}_code_min"] = cmin
        df[f"{base}_code_max"] = cmax
        df[f"{base}_description_min"] = df[f"{base}_code_min"].apply(lambda k: desc_map.get(k, None))
        df[f"{base}_description_max"] = df[f"{base}_code_max"].apply(lambda k: desc_map.get(k, None))
        return df

    for b in bases:
        tbl_flat = add_code_desc(tbl_flat, b)

    # --- area once per tile ---
    try:
        cfg = read_config(base_dir() / "system" / "config.ini")
        area_epsg = normalize_area_epsg(cfg["DEFAULT"].get("area_projection_epsg","3035"))
    except Exception:
        area_epsg = "EPSG:3035"

    try:
        metric = tbl_flat.to_crs(area_epsg)
        tbl_flat["area_m2"] = metric.geometry.area.astype("float64").round().astype("Int64")
    except Exception as e:
        log_to_gui(log_widget, f"Area computation failed in {area_epsg}: {e}; using EPSG:3035")
        metric = tbl_flat.to_crs("EPSG:3035")
        tbl_flat["area_m2"] = metric.geometry.area.astype("float64").round().astype("Int64")

    # ---- ENV index (visualization) -------------------------------------------------
    DEFAULT_ENV_PROFILE = {
        'w_sensitivity': 0.35, 'w_susceptibility': 0.25, 'w_importance': 0.20, 'w_pressure': 0.20,
        'gamma': 0.0, 'pnorm_minmax': 4.0, 'overlap_cap_q': 0.95,
        'scoring': 'linear', 'logistic_a': 8.0, 'logistic_b': 0.6,
    }

    def _load_env_profile():
        # 1) GeoParquet profile (preferred)
        p = gpq_dir() / "tbl_env_profile.parquet"
        try:
            if p.exists():
                dfp = pd.read_parquet(p)
                if {'key','value'}.issubset(dfp.columns):
                    prof = {}
                    for k, v in zip(dfp['key'], dfp['value']):
                        try:
                            prof[str(k)] = json.loads(v) if isinstance(v, str) else v
                        except Exception:
                            prof[str(k)] = v
                    return {**DEFAULT_ENV_PROFILE, **prof}
        except Exception as e:
            log_to_gui(log_widget, f"ENV profile parquet read failed: {e}")
        # 2) JSON fallback
        j = base_dir() / "output" / "settings" / "env_index_profile.json"
        try:
            if j.exists():
                with open(j, "r", encoding="utf-8") as f:
                    prof = json.load(f)
                return {**DEFAULT_ENV_PROFILE, **prof}
        except Exception as e:
            log_to_gui(log_widget, f"ENV profile JSON read failed: {e}")
        return DEFAULT_ENV_PROFILE.copy()

    def _minmax01(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        lo, hi = np.nanmin(s.values), np.nanmax(s.values)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return pd.Series(np.zeros(len(s)), index=s.index, dtype="float64")
        return (s - lo) / (hi - lo)

    def _percentile01(s: pd.Series) -> pd.Series:
        return s.rank(pct=True, method="average").astype("float64")

    def _logistic01(x01: pd.Series, a: float, b: float) -> pd.Series:
        y = 1.0 / (1.0 + np.exp(-a * (x01.astype(float) - b)))
        return _minmax01(pd.Series(y, index=x01.index))  # re-stretch til [0,1]

    def _score01(s: pd.Series, method: str, a: float, b: float) -> pd.Series:
        if method == "percentile":
            return _percentile01(s)
        x01 = _minmax01(s)  # linear/logistic use min–max first
        if method == "logistic":
            return _logistic01(x01, a, b)
        return x01  # linear

    def _pnorm_pair(s_min: pd.Series, s_max: pd.Series, p: float) -> pd.Series:
        a = pd.to_numeric(s_min, errors="coerce")
        b = pd.to_numeric(s_max, errors="coerce")
        arr = np.vstack([a.values, b.values]).astype(float)
        p = max(1e-6, float(p))
        vals = (np.nanmean(np.abs(arr) ** p, axis=0)) ** (1.0 / p)
        return pd.Series(vals, index=s_min.index)

    def _power_mean(components: dict, weights: dict, gamma: float) -> pd.Series:
        keys = list(components.keys())
        w = np.array([float(weights.get(k, 0.0)) for k in keys], dtype=float)
        sw = np.nansum(w)
        if not np.isfinite(sw) or sw <= 0:
            w[:] = 1.0 / max(1, len(keys))   # equal weights
        else:
            w /= sw
        X = np.vstack([components[k].astype(float).values for k in keys])
        if abs(gamma) < 1e-9:  # geometric mean
            Xc = np.clip(X, 1e-12, 1.0)
            y = np.exp(np.nansum(np.log(Xc) * w[:, None], axis=0))
        else:
            y = (np.nansum((w[:, None]) * (X ** gamma), axis=0)) ** (1.0 / gamma)
        return pd.Series(y, index=next(iter(components.values())).index)

    # ---- build components ----
    prof = _load_env_profile()
    p = float(prof.get("pnorm_minmax", 4.0))
    scoring = str(prof.get("scoring","linear")).lower()
    a = float(prof.get("logistic_a", 8.0))
    b = float(prof.get("logistic_b", 0.6))

    imp_raw  = _pnorm_pair(tbl_flat.get("importance_min", pd.Series(index=tbl_flat.index)),
                           tbl_flat.get("importance_max", pd.Series(index=tbl_flat.index)), p)
    sens_raw = _pnorm_pair(tbl_flat.get("sensitivity_min", pd.Series(index=tbl_flat.index)),
                           tbl_flat.get("sensitivity_max", pd.Series(index=tbl_flat.index)), p)
    susc_raw = _pnorm_pair(tbl_flat.get("susceptibility_min", pd.Series(index=tbl_flat.index)),
                           tbl_flat.get("susceptibility_max", pd.Series(index=tbl_flat.index)), p)

    # pressure = overlaps per km², capped at quantile
    eps = 1e-9
    dens = tbl_flat["assets_overlap_total"].astype(float) / (tbl_flat["area_m2"].astype(float) / 1_000_000.0 + eps)
    cap_q = float(prof.get("overlap_cap_q", 0.95))
    try:
        cap = np.nanquantile(dens.values, min(max(cap_q, 0.0), 1.0)) if len(dens) else np.nan
    except Exception:
        cap = np.nan
    press_raw = np.minimum(dens, cap) if np.isfinite(cap) else dens

    # normalize to [0,1]
    imp_n   = _score01(imp_raw,  scoring, a, b)
    sens_n  = _score01(sens_raw, scoring, a, b)
    susc_n  = _score01(susc_raw, scoring, a, b)
    press_n = _score01(press_raw, scoring, a, b)

    components = {"w_importance": imp_n, "w_sensitivity": sens_n, "w_susceptibility": susc_n, "w_pressure": press_n}
    weights    = {k: float(prof.get(k, 0.0)) for k in components.keys()}
    gamma      = float(prof.get("gamma", 0.0))
    env01 = _power_mean(components, weights, gamma).fillna(0.0)
    tbl_flat["env_index"] = (env01 * 100.0).round(2)

    # optional components
    tbl_flat["env_imp"]   = (imp_n * 100).round(1)
    tbl_flat["env_sens"]  = (sens_n * 100).round(1)
    tbl_flat["env_susc"]  = (susc_n * 100).round(1)
    tbl_flat["env_press"] = (press_n * 100).round(1)

    # stable column order
    preferred = [
        'ref_geocodegroup','name_gis_geocodegroup','code',
        'importance_min','importance_max','importance_code_min','importance_description_min','importance_code_max','importance_description_max',
        'sensitivity_min','sensitivity_max','sensitivity_code_min','sensitivity_description_min','sensitivity_code_max','sensitivity_description_max',
        'susceptibility_min','susceptibility_max','susceptibility_code_min','susceptibility_description_min','susceptibility_code_max','susceptibility_description_max',
        'asset_group_names','asset_groups_total','area_m2','assets_overlap_total',
        'env_index','env_imp','env_sens','env_susc','env_press',
        'geometry'
    ]
    for c in preferred:
        if c not in tbl_flat.columns:
            tbl_flat[c] = pd.NA
    tbl_flat = tbl_flat[preferred]

    write_parquet("tbl_flat", tbl_flat)
    log_to_gui(log_widget, f"tbl_flat saved with {len(tbl_flat):,} rows.")

    # --- streaming backfill of area_m2 to tbl_stacked parts (no big reads) ---
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

# ----------------------------
# Top-level process
# ----------------------------
def process_all(config_file: Path):
    try:
        cfg = read_config(config_file)
        working_epsg = str(cfg["DEFAULT"].get("workingprojection_epsg","4326")).strip()
        max_workers = int(cfg["DEFAULT"].get("max_workers","0"))
        cell_size   = int(cfg["DEFAULT"].get("cell_size","18000"))

        # memory guard knobs (configurable)
        approx_gb_per_worker = float(cfg["DEFAULT"].get("approx_gb_per_worker","8"))   # bigger -> fewer workers
        mem_target_frac      = float(cfg["DEFAULT"].get("mem_target_frac","0.75"))     # use up to 75% of available
        asset_soft_limit     = int(cfg["DEFAULT"].get("asset_soft_limit","200000"))    # pre-emptive split threshold
        geocode_soft_limit   = int(cfg["DEFAULT"].get("geocode_soft_limit","160"))

        cleanup_outputs(); update_progress(5)

        process_tbl_stacked(cfg, working_epsg, cell_size, max_workers,
                            approx_gb_per_worker, mem_target_frac,
                            asset_soft_limit, geocode_soft_limit)
        flatten_tbl_stacked(config_file, working_epsg); update_progress(95)

        log_to_gui(log_widget, "COMPLETED."); update_progress(100)
    except Exception as e:
        log_to_gui(log_widget, f"Error during processing: {e}")
        raise

# ----------------------------
# Entrypoint (GUI)
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process stacked/flat GeoParquet with A..E categorisation + ENV index")
    parser.add_argument('--original_working_directory', required=False, help='Path to running folder')
    args = parser.parse_args()
    original_working_directory = args.original_working_directory or os.getcwd()
    if "system" in os.path.basename(original_working_directory).lower():
        original_working_directory = os.path.abspath(os.path.join(original_working_directory, os.pardir))

    cfg_path = Path(original_working_directory) / "system" / "config.ini"
    cfg = read_config(cfg_path)

    # Heartbeat from config
    try:
        HEARTBEAT_SECS = int(cfg['DEFAULT'].get('heartbeat_secs', str(HEARTBEAT_SECS)))
    except Exception:
        pass

    ttk_theme = cfg['DEFAULT'].get('ttk_bootstrap_theme', 'flatly')

    root = tb.Window(themename=ttk_theme)
    root.title("Process analysis & presentation (GeoParquet)")
    try:
        ico = base_dir() / "system_resources" / "mesa.ico"
        if ico.exists():
            root.iconbitmap(str(ico))
    except Exception:
        pass

    log_widget = scrolledtext.ScrolledText(root, height=12)
    log_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    progress_frame = tk.Frame(root); progress_frame.pack(pady=5)
    progress_var = tk.DoubleVar(value=0.0)
    progress_bar = tb.Progressbar(progress_frame, orient="horizontal", length=260,
                                  mode="determinate", variable=progress_var, bootstyle='info')
    progress_bar.pack(side=tk.LEFT)
    progress_label = tk.Label(progress_frame, text="0%", bg="light grey"); progress_label.pack(side=tk.LEFT, padx=8)

    info = (f"Inputs: output/geoparquet/*.parquet  •  Heartbeat: {HEARTBEAT_SECS}s\n"
            "Adaptive intersections (pre-emptive splitting) + memory-aware worker scaling.\n"
            "Flatten = min/max + A..E from config; area once per tile (stream-backfilled to stacked).\n"
            "ENV index (1–100) + components: env_imp, env_sens, env_susc, env_press.\n"
            "Tuning: approx_gb_per_worker, mem_target_frac, asset_soft_limit, geocode_soft_limit.")
    tk.Label(root, text=info, wraplength=680, justify="left").pack(padx=10, pady=10)

    def _run():
        process_all(cfg_path)

    btn_frame = tk.Frame(root); btn_frame.pack(pady=6)
    tb.Button(btn_frame, text="Process", bootstyle=PRIMARY,
              command=lambda: threading.Thread(target=_run, daemon=True).start()).pack(side=tk.LEFT, padx=5)
    tb.Button(btn_frame, text="Exit", bootstyle=WARNING, command=root.destroy).pack(side=tk.LEFT, padx=5)

    log_to_gui(log_widget, "Opened processing UI (GeoParquet only).")
    root.mainloop()
