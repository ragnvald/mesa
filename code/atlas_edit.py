# -*- coding: utf-8 -*-
# atlas_edit.py — edit atlas attributes directly in GeoParquet
# - Stable BASE_DIR resolution (env/CLI/script/CWD)
# - Flat config preferred (<base>/config.ini), fallback to <base>/system/config.ini
# - Uses DEFAULT.parquet_folder (defaults to output/geoparquet)
# - Atomic writes; preserves geometry & CRS
# - ttkbootstrap optional (falls back to standard Tk widgets)

import locale
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except Exception:
    try:
        locale.setlocale(locale.LC_ALL, '')
    except Exception:
        pass

import os
import sys
import argparse
import tempfile
import configparser
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, filedialog

# Try ttkbootstrap; fall back to std ttk if missing
try:
    import ttkbootstrap as tb
    from ttkbootstrap.constants import SUCCESS, WARNING
except Exception:
    tb = None
    SUCCESS = "success"
    WARNING = "warning"
from tkinter import ttk as ttk

import pandas as pd
import geopandas as gpd
import math
import io
import urllib.request
from PIL import Image as PILImage

try:
    import contextily as ctx
except Exception:
    ctx = None

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -----------------------------
# Globals (populated at runtime)
# -----------------------------
BASE_DIR: Path = Path(".").resolve()
_CFG: configparser.ConfigParser | None = None
_CFG_PATH: Path | None = None
_PARQUET_SUBDIR = "output/geoparquet"
_PARQUET_OVERRIDE: Path | None = None

original_working_directory: str = ""
config_file: str = ""
atlas_file_name: str = "tbl_atlas.parquet"

# UI state (runtime)
df: pd.DataFrame | None = None
gdf: gpd.GeoDataFrame | None = None
gdf_plot: gpd.GeoDataFrame | None = None
current_index: int = 0

# Tk variables (runtime)
name_gis_var = title_user_var = description_var = None
image_name_1_var = image_desc_1_var = None
image_name_2_var = image_desc_2_var = None

# Map state
map_canvas = None
map_ax = None
map_default_limits = None
map_pan_state = None
map_view_center = None
map_view_span = None
MAP_TILE_LIMIT = 400
MAP_USER_AGENT = "Mozilla/5.0 (compatible; AtlasEditor/1.0)"

# -----------------------------
# Base dir / Config helpers
# -----------------------------
def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

def _has_config_at(root: Path) -> bool:
    return _exists(root / "config.ini") or _exists(root / "system" / "config.ini")

def find_base_dir(cli_workdir: str | None = None) -> Path:
    """
    Choose a canonical project base folder that actually contains config.
    Priority:
      1) env MESA_BASE_DIR
      2) --original_working_directory (CLI)
      3) folder of the running binary / interpreter (and its parents)
      4) script folder & its parents (handles PyInstaller _MEIPASS)
      5) CWD, CWD/code and their parents
    """
    candidates: list[Path] = []

    def _add(path_like):
        if not path_like:
            return
        try:
            candidates.append(Path(path_like))
        except Exception:
            pass

    env_base = os.environ.get("MESA_BASE_DIR")
    if env_base:
        _add(env_base)
    if cli_workdir:
        _add(cli_workdir)

    exe_path: Path | None = None
    try:
        exe_path = Path(sys.executable).resolve()
    except Exception:
        exe_path = None
    if exe_path:
        _add(exe_path.parent)
        _add(exe_path.parent.parent)
        _add(exe_path.parent.parent.parent)

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        _add(meipass)

    here = Path(__file__).resolve()
    _add(here.parent)
    _add(here.parent.parent)
    _add(here.parent.parent.parent)

    cwd = Path.cwd()
    _add(cwd)
    _add(cwd / "code")
    _add(cwd.parent)
    _add(cwd.parent / "code")

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

    for c in uniq:
        if _has_config_at(c):
            return c

    if here.parent.name.lower() == "system":
        return here.parent.parent
    if exe_path:
        return exe_path.parent
    if env_base:
        return Path(env_base)
    return here.parent

def _ensure_cfg() -> configparser.ConfigParser:
    """Load config, preferring <base>/config.ini and falling back to <base>/system/config.ini."""
    global _CFG, _CFG_PATH
    if _CFG is not None:
        return _CFG

    cfg = configparser.ConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
    flat = BASE_DIR / "config.ini"
    legacy = BASE_DIR / "system" / "config.ini"

    loaded = False
    if flat.exists():
        try:
            cfg.read(flat, encoding="utf-8")
            _CFG_PATH = flat
            loaded = True
        except Exception:
            pass
    if not loaded and legacy.exists():
        try:
            cfg.read(legacy, encoding="utf-8")
            _CFG_PATH = legacy
            loaded = True
        except Exception:
            pass

    if "DEFAULT" not in cfg:
        cfg["DEFAULT"] = {}

    # defaults (non-destructive)
    d = cfg["DEFAULT"]
    d.setdefault("parquet_folder", _PARQUET_SUBDIR)
    d.setdefault("ttk_bootstrap_theme", "flatly")
    d.setdefault("workingprojection_epsg", "4326")
    d.setdefault("atlas_parquet_file", "tbl_atlas.parquet")

    _CFG = cfg
    return _CFG

def _abs_path_like(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (BASE_DIR / p).resolve()

# -----------------------------
# Config utils
# -----------------------------
def read_config(file_name):
    cfg = configparser.ConfigParser()
    try:
        cfg.read(file_name, encoding="utf-8")
    except Exception:
        cfg.read(file_name)
    return cfg

def increment_stat_value(cfg_path: str, stat_name: str, increment_value: int):
    if not cfg_path or not os.path.isfile(cfg_path):
        return
    try:
        with open(cfg_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        updated = False
        for i, line in enumerate(lines):
            if line.strip().startswith(f'{stat_name} ='):
                parts = line.split('=', 1)
                if len(parts) == 2:
                    current_value = parts[1].strip()
                    try:
                        new_value = int(current_value) + int(increment_value)
                        lines[i] = f"{stat_name} = {new_value}\n"
                        updated = True
                        break
                    except ValueError:
                        return
        if updated:
            with open(cfg_path, 'w', encoding='utf-8', errors='replace') as f:
                f.writelines(lines)
    except Exception:
        pass

# -----------------------------
# GeoParquet I/O
# -----------------------------
def _parquet_dir_candidates(base_dir: Path) -> list[Path]:
    cfg = _ensure_cfg()
    sub = cfg["DEFAULT"].get("parquet_folder", _PARQUET_SUBDIR)
    if os.path.isabs(sub):
        return [Path(sub)]
    rel = Path(sub)
    cands = [(base_dir / rel).resolve()]
    if base_dir.name.lower() != "code":
        cands.append((base_dir / "code" / rel).resolve())
    else:
        parent = base_dir.parent
        if parent:
            cands.append((parent / rel).resolve())
    return cands

def geoparquet_dir(base_dir: Path, target_file: str | None = None) -> Path:
    global _PARQUET_OVERRIDE
    if _PARQUET_OVERRIDE is None:
        candidates = _parquet_dir_candidates(base_dir)
        chosen = None
        if target_file:
            for cand in candidates:
                if (cand / target_file).exists():
                    chosen = cand
                    break
        if chosen is None:
            chosen = candidates[0]
        chosen.mkdir(parents=True, exist_ok=True)
        _PARQUET_OVERRIDE = chosen
    return _PARQUET_OVERRIDE

def atlas_parquet_path(base_dir: Path, layer_file: str) -> Path:
    return geoparquet_dir(base_dir, layer_file) / layer_file

def _empty_gdf() -> gpd.GeoDataFrame:
    cols = [
        'name_gis','title_user','description',
        'image_name_1','image_desc_1','image_name_2','image_desc_2','geometry'
    ]
    return gpd.GeoDataFrame(columns=cols, geometry='geometry', crs="EPSG:4326")

def load_data(base_dir: Path, layer_file: str):
    """
    Read the GeoParquet feature table (incl. geometry).
    Returns (gdf, df_no_geom, loaded_path).
    """
    gpq_path = atlas_parquet_path(base_dir, layer_file)
    if not gpq_path.exists():
        messagebox.showerror("Missing data", "Required atlas data is missing.")
        empty = _empty_gdf()
        return empty, pd.DataFrame(empty.drop(columns=['geometry'])), str(gpq_path)
    try:
        gdf_local = gpd.read_parquet(gpq_path)
        # ensure geometry
        if gdf_local.geometry.name not in gdf_local.columns:
            if 'geom' in gdf_local.columns:
                gdf_local = gdf_local.set_geometry('geom')
            else:
                raise ValueError("No geometry column present in GeoParquet.")
        if gdf_local.crs is None:
            gdf_local.set_crs("EPSG:4326", inplace=True)
        df_no_geom = pd.DataFrame(gdf_local.drop(columns=[gdf_local.geometry.name], errors='ignore'))
        return gdf_local, df_no_geom, str(gpq_path)
    except Exception:
        messagebox.showerror("Error", "Could not read atlas data.")
        empty = _empty_gdf()
        return empty, pd.DataFrame(empty.drop(columns=['geometry'])), str(gpq_path)

def atomic_write_geoparquet(gdf_in: gpd.GeoDataFrame, path: Path):
    """
    Atomically write GeoParquet (write to temp file then replace).
    Keeps geometry + CRS metadata.
    """
    out_dir = path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out_dir, suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        gdf_in.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise

def save_row_to_parquet(base_dir: Path, layer_file: str, row_dict):
    """
    Update a single row in the GeoParquet by key (name_gis).
    Read → modify attributes in-memory → write back atomically.
    Geometry is preserved.
    """
    key = str(row_dict.get("name_gis") or "").strip()
    if not key:
        messagebox.showerror("Error", "Missing name_gis for row update.")
        return

    gpq_path = atlas_parquet_path(base_dir, layer_file)
    if not gpq_path.exists():
        messagebox.showerror("Error", f"File not found:\n{gpq_path}")
        return

    gdf_local = gpd.read_parquet(gpq_path)

    if 'name_gis' not in gdf_local.columns:
        messagebox.showerror("Error", "GeoParquet does not contain a 'name_gis' column.")
        return

    idx = gdf_local.index[gdf_local['name_gis'].astype(str) == key]
    if len(idx) == 0:
        messagebox.showerror("Error", f"No record with name_gis='{key}' found.")
        return

    editable = [
        "title_user", "description",
        "image_name_1", "image_desc_1",
        "image_name_2", "image_desc_2",
    ]
    for c in editable:
        if c in gdf_local.columns and c in row_dict:
            gdf_local.loc[idx, c] = row_dict[c]

    atomic_write_geoparquet(gdf_local, gpq_path)

# -----------------------------
# UI helpers
# -----------------------------
def _images_dir() -> Path:
    return BASE_DIR / "input" / "images"

def browse_image_1():
    fp = filedialog.askopenfilename(
        initialdir=str(_images_dir()),
        title="Select file",
        filetypes=(("JPEG files", "*.jpg;*.jpeg"), ("PNG files", "*.png"), ("All files", "*.*"))
    )
    if fp:
        image_name_1_var.set(fp)

def browse_image_2():
    fp = filedialog.askopenfilename(
        initialdir=str(_images_dir()),
        title="Select file",
        filetypes=(("JPEG files", "*.jpg;*.jpeg"), ("PNG files", "*.png"), ("All files", "*.*"))
    )
    if fp:
        image_name_2_var.set(fp)

def _get(df_loc: pd.DataFrame, idx: int, col: str) -> str:
    try:
        v = df_loc.at[idx, col]
        return "" if pd.isna(v) else str(v)
    except Exception:
        return ""

# -----------------------------
# Map helpers
# -----------------------------
def _map_clamp_limits(new_min: float, new_max: float, bounds: tuple[float, float] | None) -> tuple[float, float]:
    if bounds is None:
        return new_min, new_max
    bounds_min, bounds_max = bounds
    span = new_max - new_min
    total = bounds_max - bounds_min
    if span <= 0 or total <= 0:
        return new_min, new_max
    if span >= total:
        return bounds_min, bounds_max
    if new_min < bounds_min:
        new_max = bounds_min + span
        new_min = bounds_min
    if new_max > bounds_max:
        new_min = bounds_max - span
        new_max = bounds_max
    return new_min, new_max


def _map_prepare_plot_gdf():
    """Prepare a projected GeoDataFrame (EPSG:3857 when possible) for plotting."""
    global gdf_plot, map_default_limits, map_view_center, map_view_span
    if gdf is None or len(gdf) == 0:
        gdf_plot = None
        map_default_limits = None
        map_view_center = None
        map_view_span = None
        return
    try:
        g = gdf.copy()
        if g.crs is None:
            g.set_crs("EPSG:4326", inplace=True)
        g_projected = g.to_crs(3857)
    except Exception:
        g_projected = g.copy()
    gdf_plot = g_projected
    map_default_limits = None
    map_view_center = None
    map_view_span = None

def _map_apply_view(center_x: float, center_y: float, width: float, height: float, update_state: bool = True):
    """Apply axis limits based on desired center/span, clamped to defaults."""
    global map_view_center, map_view_span
    if map_ax is None:
        return None
    width = max(float(width), 1e-6)
    height = max(float(height), 1e-6)
    x0 = center_x - width / 2.0
    x1 = center_x + width / 2.0
    y0 = center_y - height / 2.0
    y1 = center_y + height / 2.0

    if map_default_limits is not None:
        (minx, maxx), (miny, maxy) = map_default_limits
        total_x = maxx - minx
        total_y = maxy - miny
        span_x = x1 - x0
        span_y = y1 - y0
        if span_x >= total_x:
            x0, x1 = minx, maxx
        else:
            x0 = max(minx, min(x0, maxx - span_x))
            x1 = x0 + span_x
        if span_y >= total_y:
            y0, y1 = miny, maxy
        else:
            y0 = max(miny, min(y0, maxy - span_y))
            y1 = y0 + span_y

    map_ax.set_xlim(x0, x1)
    map_ax.set_ylim(y0, y1)

    actual_center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    actual_span = (x1 - x0, y1 - y0)
    if update_state:
        map_view_center = actual_center
        map_view_span = actual_span
    return (x0, x1), (y0, y1)


def _merc_to_lonlat(x: float, y: float) -> tuple[float, float]:
    R = 6378137.0
    lon = (x / R) * 180.0 / math.pi
    lat = (2.0 * math.atan(math.exp(y / R)) - math.pi / 2.0) * 180.0 / math.pi
    return lon, lat


def _lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_clamped = max(-85.05112878, min(85.05112878, float(lat)))
    lat_rad = math.radians(lat_clamped)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _tile_bounds_lonlat(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    n = 2.0 ** z
    minlon = x / n * 360.0 - 180.0
    maxlon = (x + 1) / n * 360.0 - 180.0

    def tiley_to_lat(t: int) -> float:
        Y = t / n
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * Y)))
        return math.degrees(lat_rad)

    maxlat = tiley_to_lat(y)
    minlat = tiley_to_lat(y + 1)
    return (minlon, minlat, maxlon, maxlat)


def _lonlat_to_merc(lon: float, lat: float) -> tuple[float, float]:
    R = 6378137.0
    x = lon * math.pi / 180.0 * R
    lat_clamped = max(-85.05112878, min(85.05112878, float(lat)))
    y = math.log(math.tan((90.0 + lat_clamped) * math.pi / 360.0)) * R
    return x, y


def _map_fetch_osm_tiles(xlim: tuple[float, float], ylim: tuple[float, float]):
    try:
        minx, maxx = xlim
        miny, maxy = ylim
    except Exception:
        return None

    lon_left, lat_bottom = _merc_to_lonlat(minx, miny)
    lon_right, lat_top = _merc_to_lonlat(maxx, maxy)

    lon_min, lon_max = sorted((lon_left, lon_right))
    lat_min, lat_max = sorted((lat_bottom, lat_top))

    def _calc_tile_range(z_level: int) -> tuple[int, int, int, int]:
        tx0, ty0 = _lonlat_to_tile(lon_min, lat_min, z_level)
        tx1, ty1 = _lonlat_to_tile(lon_max, lat_max, z_level)
        return (min(tx0, tx1), max(tx0, tx1), min(ty0, ty1), max(ty0, ty1))

    target_px = 512
    zoom = 3
    for test in range(3, 20):
        xmin, xmax, ymin, ymax = _calc_tile_range(test)
        w = (abs(xmax - xmin) + 1) * 256
        h = (abs(ymax - ymin) + 1) * 256
        if w >= target_px and h >= target_px:
            zoom = test
            break

    xmin, xmax, ymin, ymax = _calc_tile_range(zoom)
    tile_span = (xmax - xmin + 1) * (ymax - ymin + 1)
    while tile_span > MAP_TILE_LIMIT and zoom > 3:
        zoom -= 1
        xmin, xmax, ymin, ymax = _calc_tile_range(zoom)
        tile_span = (xmax - xmin + 1) * (ymax - ymin + 1)

    if tile_span > MAP_TILE_LIMIT:
        return None

    TILE = 256
    W, H = (xmax - xmin + 1) * TILE, (ymax - ymin + 1) * TILE
    mosaic = PILImage.new("RGB", (W, H), (240, 240, 240))

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", MAP_USER_AGENT)]
    urllib.request.install_opener(opener)

    for xi, x in enumerate(range(xmin, xmax + 1)):
        for yi, y in enumerate(range(ymin, ymax + 1)):
            try:
                url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
                with urllib.request.urlopen(url, timeout=5) as resp:
                    data = resp.read()
                img = PILImage.open(io.BytesIO(data)).convert("RGB")
                mosaic.paste(img, (xi * TILE, yi * TILE))
            except Exception:
                pass

    west, _, _, north = _tile_bounds_lonlat(zoom, xmin, ymin)
    east = _tile_bounds_lonlat(zoom, xmax, ymax)[2]
    south = _tile_bounds_lonlat(zoom, xmax, ymax)[1]

    lx, north_y = _lonlat_to_merc(west, north)
    rx, south_y = _lonlat_to_merc(east, south)

    return mosaic, [lx, rx, south_y, north_y]


def _map_draw_basemap(ax, xlim: tuple[float, float], ylim: tuple[float, float]):
    ax.set_facecolor("#e9edf4")
    if gdf_plot is None:
        return
    crs_epsg = None
    try:
        crs_epsg = gdf_plot.crs.to_epsg()
    except Exception:
        crs_epsg = None

    if crs_epsg == 3857 and ctx is not None:
        try:
            ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.OpenStreetMap.Mapnik, reset_extent=False)
            return
        except Exception:
            pass

    if crs_epsg != 3857:
        return

    fetched = _map_fetch_osm_tiles(xlim, ylim)
    if fetched is None:
        return
    mosaic, extent = fetched
    ax.imshow(mosaic, extent=extent, origin="upper", interpolation="bilinear", zorder=0)

def _map_compute_default_limits():
    """Calculate padded bounds for the overview map."""
    global map_default_limits
    source = gdf_plot if gdf_plot is not None else gdf
    if source is None or len(source) == 0:
        map_default_limits = None
        return
    try:
        minx, miny, maxx, maxy = source.total_bounds
    except Exception:
        map_default_limits = None
        return
    if not all(math.isfinite(v) for v in (minx, miny, maxx, maxy)):
        map_default_limits = None
        return
    if maxx - minx <= 0:
        maxx = minx + 1.0
    if maxy - miny <= 0:
        maxy = miny + 1.0
    dx = max(maxx - minx, 1e-9)
    dy = max(maxy - miny, 1e-9)
    pad_x = dx * 0.05
    pad_y = dy * 0.05
    map_default_limits = ((minx - pad_x, maxx + pad_x), (miny - pad_y, maxy + pad_y))


def _map_reset_view():
    if map_ax is None or map_canvas is None:
        return
    refresh_map(reset_view=True)


def refresh_map(reset_view: bool = False):
    """Redraw the atlas overview map and highlight the current feature."""
    global map_ax, map_canvas, map_default_limits, map_view_center, map_view_span
    if map_ax is None or map_canvas is None:
        return

    if gdf is None or len(gdf) == 0:
        map_ax.clear()
        map_ax.set_title("Atlas overview", fontsize=10)
        map_ax.text(0.5, 0.5, "No atlas data available.", transform=map_ax.transAxes,
                    ha="center", va="center")
        map_ax.set_xticks([])
        map_ax.set_yticks([])
        map_canvas.draw_idle()
        map_view_center = None
        map_view_span = None
        return

    if (gdf_plot is None or len(gdf_plot) == 0) and len(gdf) > 0:
        _map_prepare_plot_gdf()

    plot_source = gdf_plot if gdf_plot is not None and len(gdf_plot) == len(gdf) else gdf
    if plot_source is None or len(plot_source) == 0:
        map_ax.clear()
        map_ax.set_title("Atlas overview", fontsize=10)
        map_ax.text(0.5, 0.5, "No atlas data available.", transform=map_ax.transAxes,
                    ha="center", va="center")
        map_ax.set_xticks([])
        map_ax.set_yticks([])
        map_canvas.draw_idle()
        map_view_center = None
        map_view_span = None
        return

    if map_default_limits is None or reset_view:
        _map_compute_default_limits()
        map_view_center = None
        map_view_span = None

    map_ax.clear()
    map_ax.set_title("Atlas overview", fontsize=10)
    map_ax.set_xticks([])
    map_ax.set_yticks([])
    map_ax.set_aspect("equal", adjustable="box")

    if map_view_center is None or map_view_span is None:
        if map_default_limits is not None:
            base_xlim, base_ylim = map_default_limits
        else:
            try:
                minx, miny, maxx, maxy = plot_source.total_bounds
            except Exception:
                minx, miny, maxx, maxy = (0.0, 0.0, 1.0, 1.0)
            base_xlim = (minx, maxx)
            base_ylim = (miny, maxy)
        width = max(base_xlim[1] - base_xlim[0], 1e-6)
        height = max(base_ylim[1] - base_ylim[0], 1e-6)
        center_x = (base_xlim[0] + base_xlim[1]) / 2.0
        center_y = (base_ylim[0] + base_ylim[1]) / 2.0
    else:
        center_x, center_y = map_view_center
        width, height = map_view_span

    view = _map_apply_view(center_x, center_y, width, height, update_state=True)
    if view is None:
        map_canvas.draw_idle()
        return
    xlim, ylim = view

    _map_draw_basemap(map_ax, xlim, ylim)

    try:
        plot_source.plot(ax=map_ax, edgecolor="#6b8aa1", facecolor="#d5e2f0",
                         linewidth=0.5, alpha=0.85, zorder=10)
    except Exception:
        map_ax.text(0.5, 0.5, "Unable to render atlas polygons.", transform=map_ax.transAxes,
                    ha="center", va="center")
    else:
        if 0 <= current_index < len(plot_source):
            geom = plot_source.iloc[current_index].geometry
            if geom is not None and not geom.is_empty:
                try:
                    gpd.GeoSeries([geom], crs=plot_source.crs).plot(
                        ax=map_ax,
                        edgecolor="#e74c3c",
                        facecolor="#f8b4aa",
                        linewidth=1.2,
                        alpha=0.8,
                        zorder=11
                    )
                except Exception:
                    pass

    if map_view_center is not None and map_view_span is not None:
        _map_apply_view(map_view_center[0], map_view_center[1],
                        map_view_span[0], map_view_span[1], update_state=False)

    map_canvas.draw_idle()


def _map_on_scroll(event):
    global map_ax, map_canvas, map_view_center, map_view_span
    if map_ax is None or map_canvas is None or event.inaxes != map_ax:
        return
    if map_view_span is not None:
        cur_width, cur_height = map_view_span
    else:
        cur_xlim = map_ax.get_xlim()
        cur_ylim = map_ax.get_ylim()
        cur_width = cur_xlim[1] - cur_xlim[0]
        cur_height = cur_ylim[1] - cur_ylim[0]
    scale = 1.2
    factor = (1 / scale) if event.button == "up" else scale
    new_width = cur_width * factor
    new_height = cur_height * factor
    if map_view_center is not None:
        default_center_x, default_center_y = map_view_center
    else:
        cur_xlim = map_ax.get_xlim()
        cur_ylim = map_ax.get_ylim()
        default_center_x = (cur_xlim[0] + cur_xlim[1]) / 2.0
        default_center_y = (cur_ylim[0] + cur_ylim[1]) / 2.0
    center_x = event.xdata if event.xdata is not None else default_center_x
    center_y = event.ydata if event.ydata is not None else default_center_y
    if _map_apply_view(center_x, center_y, new_width, new_height, update_state=True):
        map_canvas.draw_idle()


def _map_on_press(event):
    global map_pan_state
    if map_ax is None or event.inaxes != map_ax or event.xdata is None or event.ydata is None:
        return
    if event.dblclick and event.button == 1:
        _map_reset_view()
        map_pan_state = None
        return
    if event.button != 1:
        return
    if map_view_center is not None:
        center_x, center_y = map_view_center
    else:
        xlim = map_ax.get_xlim()
        ylim = map_ax.get_ylim()
        center_x = (xlim[0] + xlim[1]) / 2.0
        center_y = (ylim[0] + ylim[1]) / 2.0
    if map_view_span is not None:
        width, height = map_view_span
    else:
        xlim = map_ax.get_xlim()
        ylim = map_ax.get_ylim()
        width = xlim[1] - xlim[0]
        height = ylim[1] - ylim[0]
    map_pan_state = {
        "ref_x": event.xdata,
        "ref_y": event.ydata,
        "center_x": center_x,
        "center_y": center_y,
        "width": width,
        "height": height,
    }


def _map_on_motion(event):
    global map_pan_state, map_canvas
    if map_pan_state is None or map_canvas is None or event.inaxes != map_ax:
        return
    if event.xdata is None or event.ydata is None:
        return
    dx = event.xdata - map_pan_state["ref_x"]
    dy = event.ydata - map_pan_state["ref_y"]
    new_center_x = map_pan_state["center_x"] - dx
    new_center_y = map_pan_state["center_y"] - dy
    if _map_apply_view(new_center_x, new_center_y,
                       map_pan_state["width"], map_pan_state["height"], update_state=True):
        map_canvas.draw_idle()


def _map_on_release(event):
    global map_pan_state
    map_pan_state = None


def initialize_map(parent_frame):
    """Create the embedded matplotlib map inside the provided frame."""
    global map_canvas, map_ax, map_default_limits, map_pan_state
    map_default_limits = None
    map_pan_state = None

    fig = Figure(figsize=(6.0, 5.2), dpi=100)
    map_ax = fig.add_subplot(111)
    map_ax.set_title("Atlas overview", fontsize=10)
    map_ax.set_xticks([])
    map_ax.set_yticks([])
    map_ax.set_aspect("equal", adjustable="box")

    map_canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    canvas_widget = map_canvas.get_tk_widget()
    canvas_widget.grid(row=0, column=0, sticky="nsew")

    info_text = "Scroll to zoom | Drag to pan | Double-click to reset"
    if tb is not None:
        info_label = tb.Label(parent_frame, text=info_text)
    else:
        info_label = ttk.Label(parent_frame, text=info_text)
    info_label.grid(row=1, column=0, sticky="ew", pady=(6, 0))

    parent_frame.rowconfigure(0, weight=1)
    parent_frame.columnconfigure(0, weight=1)

    map_canvas.mpl_connect("scroll_event", _map_on_scroll)
    map_canvas.mpl_connect("button_press_event", _map_on_press)
    map_canvas.mpl_connect("motion_notify_event", _map_on_motion)
    map_canvas.mpl_connect("button_release_event", _map_on_release)

# -----------------------------
# Record ops
# -----------------------------
def load_record(reset_map: bool = False):
    global df
    if df is None or len(df) == 0:
        name_gis_var.set(""); title_user_var.set(""); description_var.set("")
        image_name_1_var.set(""); image_desc_1_var.set("")
        image_name_2_var.set(""); image_desc_2_var.set("")
        refresh_map(reset_view=True)
        return
    name_gis_var.set(_get(df, current_index, 'name_gis'))
    title_user_var.set(_get(df, current_index, 'title_user'))
    description_var.set(_get(df, current_index, 'description'))
    image_name_1_var.set(_get(df, current_index, 'image_name_1'))
    image_desc_1_var.set(_get(df, current_index, 'image_desc_1'))
    image_name_2_var.set(_get(df, current_index, 'image_name_2'))
    image_desc_2_var.set(_get(df, current_index, 'image_desc_2'))
    refresh_map(reset_view=reset_map)

def update_record(save_message=False):
    """
    Update current record in both the UI DataFrame and the GeoParquet (by key).
    """
    global df
    if df is None or len(df) == 0:
        return
    try:
        df.at[current_index, 'name_gis']      = name_gis_var.get()
        df.at[current_index, 'title_user']    = title_user_var.get()
        df.at[current_index, 'description']   = description_var.get()
        df.at[current_index, 'image_name_1']  = image_name_1_var.get()
        df.at[current_index, 'image_desc_1']  = image_desc_1_var.get()
        df.at[current_index, 'image_name_2']  = image_name_2_var.get()
        df.at[current_index, 'image_desc_2']  = image_desc_2_var.get()

        row_dict = df.iloc[current_index].to_dict()
        save_row_to_parquet(BASE_DIR, atlas_file_name, row_dict)

        if save_message:
            messagebox.showinfo("Info", "Record updated and saved to GeoParquet.")
    except Exception as e:
        messagebox.showerror("Error", f"Error updating record: {e}")

def navigate(direction):
    update_record()  # save current before move
    global current_index
    if df is None or len(df) == 0:
        return
    if direction == 'next' and current_index < len(df) - 1:
        current_index += 1
    elif direction == 'previous' and current_index > 0:
        current_index -= 1
    load_record()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Edit atlas (attributes in GeoParquet)')
    parser.add_argument('--original_working_directory', required=False, help='Path to running folder')
    parser.add_argument('--atlas_parquet_file', required=False, help='Override GeoParquet file name (default from config or tbl_atlas.parquet)')
    args = parser.parse_args()

    # Resolve BASE_DIR
    BASE_DIR = find_base_dir(args.original_working_directory)
    original_working_directory = str(BASE_DIR)
    _ensure_cfg()

    # Config / settings
    cfg = _ensure_cfg()
    d = cfg["DEFAULT"]
    ttk_bootstrap_theme    = d.get('ttk_bootstrap_theme', 'flatly')
    workingprojection_epsg = d.get('workingprojection_epsg', '4326')
    atlas_file_name        = args.atlas_parquet_file or d.get('atlas_parquet_file', 'tbl_atlas.parquet')

    # Path to the config file we actually use (for stats)
    config_file = str(_CFG_PATH if _CFG_PATH is not None else (BASE_DIR / "config.ini"))
    increment_stat_value(config_file, 'mesa_stat_edit_atlas', increment_value=1)

    # UI setup
    if tb is not None:
        try:
            root = tb.Window(themename=ttk_bootstrap_theme)
        except Exception:
            root = tb.Window(themename="flatly")
    else:
        root = tk.Tk()

    root.title("Edit atlas (GeoParquet)")
    try:
        ico = BASE_DIR / "system_resources" / "mesa.ico"
        if ico.exists() and hasattr(root, "iconbitmap"):
            root.iconbitmap(str(ico))
    except Exception:
        pass

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # ---------- Early data check: if the file does NOT exist, show Exit-only UI ----------
    expected_path = atlas_parquet_path(BASE_DIR, atlas_file_name)
    if not expected_path.is_file():
        warn_frame = (tb.Frame(root, padding="16") if tb is not None else ttk.Frame(root, padding="16"))
        warn_frame.grid(row=0, column=0, sticky="nsew")

        (tb.Label(warn_frame, text="Data is missing.", font=("", 14, "bold"))
         if tb is not None else ttk.Label(warn_frame, text="Data is missing.", font=("", 14, "bold"))
        ).grid(row=0, column=0, sticky="w", pady=(0,6))

        (tb.Label(warn_frame, text="Please close this window and try again after the atlas data has been generated.", justify="left")
         if tb is not None else ttk.Label(warn_frame, text="Please close this window and try again after the atlas data has been generated.", justify="left")
        ).grid(row=1, column=0, sticky="w")

        (tb.Button(warn_frame, text="Exit", command=root.destroy, bootstyle=WARNING)
         if tb is not None else ttk.Button(warn_frame, text="Exit", command=root.destroy)
        ).grid(row=2, column=0, sticky="e", pady=(16,0))

        root.mainloop()
        raise SystemExit(0)

    # ---------- Full editor UI (file exists) ----------
    main_frame = (tb.Frame(root, padding="10") if tb is not None else ttk.Frame(root, padding="10"))
    main_frame.grid(row=0, column=0, sticky="nsew")
    main_frame.columnconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=2)
    main_frame.rowconfigure(0, weight=1)
    main_frame.rowconfigure(1, weight=0)

    # Load data (GeoParquet -> gdf / df without geometry)
    gdf, df, loaded_path = load_data(BASE_DIR, atlas_file_name)
    gdf_plot = None
    current_index = 0
    map_default_limits = None
    map_view_center = None
    map_view_span = None
    map_pan_state = None

    # Tk variables
    name_gis_var     = tk.StringVar()
    title_user_var   = tk.StringVar()
    description_var  = tk.StringVar()
    image_name_1_var = tk.StringVar()
    image_desc_1_var = tk.StringVar()
    image_name_2_var = tk.StringVar()
    image_desc_2_var = tk.StringVar()

    form_frame = (tb.Frame(main_frame) if tb is not None else ttk.Frame(main_frame))
    form_frame.grid(row=0, column=0, sticky="nsew")
    form_frame.columnconfigure(1, weight=1)
    for r in range(1, 8):
        form_frame.rowconfigure(r, weight=0)
    form_frame.rowconfigure(7, weight=1)

    map_container = (tb.Frame(main_frame) if tb is not None else ttk.Frame(main_frame))
    map_container.grid(row=0, column=1, sticky="nsew", padx=(20, 0))
    map_container.columnconfigure(0, weight=1)
    map_container.rowconfigure(1, weight=1)

    (tb.Label(form_frame, text=f"Editing file: {loaded_path}")
     if tb is not None else ttk.Label(form_frame, text=f"Editing file: {loaded_path}")
    ).grid(row=0, column=0, columnspan=3, sticky='w', pady=(0,8))

    # Row 1: GIS Name (read-only label)
    tk.Label(form_frame, text="GIS Name").grid(row=1, column=0, sticky='w')
    tk.Label(form_frame, textvariable=name_gis_var, width=40, relief="sunken", anchor="w")\
        .grid(row=1, column=1, sticky='w', padx=10, pady=6)

    # Row 2: Title
    tk.Label(form_frame, text="Title").grid(row=2, column=0, sticky='w')
    tk.Entry(form_frame, textvariable=title_user_var, width=40).grid(row=2, column=1, sticky='w', padx=10, pady=6)

    # Row 3: Image 1
    tk.Label(form_frame, text="Image Name 1").grid(row=3, column=0, sticky='w')
    tk.Entry(form_frame, textvariable=image_name_1_var, width=40).grid(row=3, column=1, sticky='w', padx=10, pady=6)
    (tb.Button(form_frame, text="Browse", command=browse_image_1)
     if tb is not None else ttk.Button(form_frame, text="Browse", command=browse_image_1)
    ).grid(row=3, column=2, padx=10, pady=6, sticky='w')

    tk.Label(form_frame, text="Image 1 description").grid(row=4, column=0, sticky='w')
    tk.Entry(form_frame, textvariable=image_desc_1_var, width=40).grid(row=4, column=1, sticky='w', padx=10, pady=6)

    # Row 5: Image 2
    tk.Label(form_frame, text="Image Name 2").grid(row=5, column=0, sticky='w')
    tk.Entry(form_frame, textvariable=image_name_2_var, width=40).grid(row=5, column=1, sticky='w', padx=10, pady=6)
    (tb.Button(form_frame, text="Browse", command=browse_image_2)
     if tb is not None else ttk.Button(form_frame, text="Browse", command=browse_image_2)
    ).grid(row=5, column=2, padx=10, pady=6, sticky='w')

    tk.Label(form_frame, text="Image 2 description").grid(row=6, column=0, sticky='w')
    tk.Entry(form_frame, textvariable=image_desc_2_var, width=40).grid(row=6, column=1, sticky='w', padx=10, pady=6)

    # Row 7: Description
    tk.Label(form_frame, text="Description").grid(row=7, column=0, sticky='w')
    tk.Entry(form_frame, textvariable=description_var, width=40).grid(row=7, column=1, sticky='w', padx=10, pady=6)

    (tb.Label(map_container, text="Atlas map", font=("", 11, "bold"))
     if tb is not None else ttk.Label(map_container, text="Atlas map", font=("", 11, "bold"))
    ).grid(row=0, column=0, sticky='w', pady=(0, 6))

    map_frame = (tb.Frame(map_container) if tb is not None else ttk.Frame(map_container))
    map_frame.grid(row=1, column=0, sticky="nsew")
    initialize_map(map_frame)

    controls_frame = (tb.Frame(main_frame) if tb is not None else ttk.Frame(main_frame))
    controls_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))
    for col in range(4):
        controls_frame.columnconfigure(col, weight=1)

    # Nav + actions
    (tb.Button(controls_frame, text="Previous", command=lambda: navigate('previous'))
     if tb is not None else ttk.Button(controls_frame, text="Previous", command=lambda: navigate('previous'))
    ).grid(row=0, column=0, sticky='w')

    (tb.Button(controls_frame, text="Next", command=lambda: navigate('next'))
     if tb is not None else ttk.Button(controls_frame, text="Next", command=lambda: navigate('next'))
    ).grid(row=0, column=1, sticky='w', padx=(10, 0))

    (tb.Button(controls_frame, text="Save", command=lambda: update_record(True), bootstyle=SUCCESS)
     if tb is not None else ttk.Button(controls_frame, text="Save", command=lambda: update_record(True))
    ).grid(row=0, column=2, sticky='e', padx=(0, 10))

    (tb.Button(controls_frame, text="Exit", command=root.destroy, bootstyle=WARNING)
     if tb is not None else ttk.Button(controls_frame, text="Exit", command=root.destroy)
    ).grid(row=0, column=3, sticky='e')

    load_record(reset_map=True)
    root.mainloop()
