# -*- coding: utf-8 -*-
"""
Raster MBTiles generator (PNG) from tbl_flat.parquet — no GDAL/Tippecanoe.

Per group in name_gis_geocodegroup, produces six MBTiles:
  <group>_sensitivity.mbtiles   (colors from config.ini [A]..[E], uses sensitivity_code_max with numeric fallback)
  <group>_envindex.mbtiles      (yellow->red ramp from env_index 1..100)
  <group>_groupstotal.mbtiles   (light->dark blue, linear ramp of asset_groups_total per group)
  <group>_assetstotal.mbtiles   (light->dark blue, linear ramp of assets_overlap_total per group)
  <group>_importance_index.mbtiles   (1..100 gradient from the importance index)
  <group>_sensitivity_index.mbtiles  (1..100 gradient from the sensitivity index)

- EPSG:4326 input expected.
- Transparent background, polygon fill + optional stroke.
- Multiprocessing: worker pool renders tiles, single writer process inserts into SQLite.
- Dependencies: geopandas, shapely, pandas, numpy, pillow (lightweight), sqlite3 (stdlib).

Usage examples:
  python create_raster_tiles.py --minzoom 6 --maxzoom 12
  python create_raster_tiles.py --only-groups "geocode_001,H3_R8"
  python create_raster_tiles.py --procs 8 --stroke-alpha 0.6
"""

import argparse, math, os, sqlite3, re, multiprocessing as mp, sys, io
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Iterable

import pyarrow.parquet as pq

# Ensure stdout/stderr can handle Unicode across environments (Windows cp1252, pipes, etc.)
def _tame_stdio():
    try:
        # Prefer UTF-8 when requested; otherwise keep current encoding but allow replacement
        enc = None
        if os.environ.get("PYTHONUTF8", "").strip() == "1":
            enc = "utf-8"
        if not enc:
            enc = os.environ.get("PYTHONIOENCODING") or getattr(sys.stdout, "encoding", None) or "utf-8"
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding=enc, errors="replace")
            sys.stderr.reconfigure(encoding=enc, errors="replace")
        else:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=enc, errors="replace", line_buffering=True)
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding=enc, errors="replace", line_buffering=True)
    except Exception:
        # Never fail due to console encoding issues
        pass

_tame_stdio()

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from PIL import Image, ImageDraw  # Pillow

# ----------------------- Paths -----------------------
def base_dir() -> Path:
    cwd = Path(os.getcwd())
    return cwd.parent if cwd.name.lower() == "system" else cwd

def gpq_dir() -> Path:
    out = base_dir() / "output" / "geoparquet"
    out.mkdir(parents=True, exist_ok=True)
    return out

def tbl_flat_path() -> Path:
    return gpq_dir() / "tbl_flat.parquet"

def settings_dir() -> Path:
    return base_dir() / "system"

def mbtiles_dir() -> Path:
    out = base_dir() / "output" / "mbtiles"
    out.mkdir(parents=True, exist_ok=True)
    return out

# ----------------------- Logging -----------------------
def log(msg: str):
    # Safe print that won’t crash on non-encodable characters
    try:
        print(str(msg), flush=True)
    except Exception:
        try:
            sys.stdout.write(str(msg) + "\n")
            sys.stdout.flush()
        except Exception:
            # Last resort: replace unknown characters
            try:
                sys.stdout.write(str(msg).encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8") + "\n")
                sys.stdout.flush()
            except Exception:
                pass

# ----------------------- Tile math -----------------------
TILE_SIZE = 256
WEBMERCATOR_LAT_MAX = 85.05112878

def clamp_lat(lat: float) -> float:
    return max(-WEBMERCATOR_LAT_MAX, min(WEBMERCATOR_LAT_MAX, float(lat)))

def lonlat_to_tile_xy(lon: float, lat: float, z: int) -> Tuple[int, int]:
    n = 2 ** z
    x = int(math.floor((lon + 180.0) / 360.0 * n))
    lat = clamp_lat(lat)
    lat_rad = math.radians(lat)
    y = int(math.floor((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n))
    return x, y

def tile_bounds_lonlat(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    n = 2.0 ** z
    minlon = x / n * 360.0 - 180.0
    maxlon = (x + 1) / n * 360.0 - 180.0
    def tiley_to_lat(t: float) -> float:
        y = t / n
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y)))
        return math.degrees(lat_rad)
    maxlat = tiley_to_lat(y)
    minlat = tiley_to_lat(y + 1)
    return (minlon, minlat, maxlon, maxlat)

def tiles_covering_bounds(bounds: Tuple[float, float, float, float], z: int) -> Tuple[int, int, int, int]:
    minlon, minlat, maxlon, maxlat = bounds
    minx, miny = lonlat_to_tile_xy(minlon, maxlat, z)
    maxx, maxy = lonlat_to_tile_xy(maxlon, minlat, z)
    n = (2 ** z) - 1
    minx = max(0, min(minx, n)); maxx = max(0, min(maxx, n))
    miny = max(0, min(miny, n)); maxy = max(0, min(maxy, n))
    return minx, miny, maxx, maxy

def lonlat_to_tile_px(lon: float, lat: float, z: int, x_tile: int, y_tile: int) -> Tuple[float, float]:
    lat = clamp_lat(lat)
    n = 2.0 ** z
    x_float = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y_float = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    dx = (x_float - x_tile) * TILE_SIZE
    dy = (y_float - y_tile) * TILE_SIZE
    return dx, dy

# ----------------------- Styling -----------------------
_HEX_RE = re.compile(r'^\#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$')

def hex_to_rgba(hex_color: Optional[str], alpha: float) -> Tuple[int, int, int, int]:
    """Accepts #RGB or #RRGGBB (with/without leading #). Falls back to gray if invalid."""
    if not hex_color:
        r, g, b = 180, 180, 180
    else:
        s = str(hex_color).strip().strip('"').strip("'")
        m = _HEX_RE.match(s)
        if not m:
            r, g, b = 180, 180, 180
        else:
            hc = m.group(1)
            if len(hc) == 3:
                hc = ''.join(c*2 for c in hc)
            r = int(hc[0:2], 16)
            g = int(hc[2:4], 16)
            b = int(hc[4:6], 16)
    a = int(max(0.0, min(1.0, float(alpha))) * 255)
    return (r, g, b, a)

def env_index_color(v: Optional[float], alpha: float=0.70) -> Tuple[int,int,int,int]:
    """Yellow (low) -> Red (high) for 1..100. Missing -> light gray."""
    if v is None or not np.isfinite(v):
        return (200, 200, 200, int(0.55*255))
    t = max(0.0, min(1.0, (float(v) - 1.0) / 99.0))  # 1..100 -> 0..1
    # yellow (255,215,0) -> red (200,0,0)
    r = int(255 + t * (200 - 255))
    g = int(215 + t * (0   - 215))
    b = 0
    return (r, g, b, int(alpha*255))

def blue_ramp_rgba(v: Optional[float], vmin: float, vmax: float, alpha: float=0.70) -> Tuple[int,int,int,int]:
    """Linear light->dark blue; missing -> transparent."""
    if v is None or not np.isfinite(v):
        return (0, 0, 0, 0)
    if vmax <= vmin:
        t = 1.0
    else:
        t = max(0.0, min(1.0, (float(v) - vmin) / (vmax - vmin)))
    # light #DCEBFF (220,235,255) -> dark #08306B (8,48,107)
    r = int(round(220 + t * (8   - 220)))
    g = int(round(235 + t * (48  - 235)))
    b = int(round(255 + t * (107 - 255)))
    return (r, g, b, int(max(0.0, min(1.0, alpha))*255))

def build_index_gradient_from_palette(palette: Dict[str, Tuple[int,int,int,int]],
                                      steps: int = 25) -> List[Tuple[int,int,int,int]]:
    """
    Build a 25-step gradient (1..100) using the sensitivity palette colors
    ordered from low (E) to high (A).
    """
    order = ["E", "D", "C", "B", "A"]
    stops = []
    fallback = (255, 255, 178, 255)  # light yellow default
    for code in order:
        stops.append(palette.get(code, fallback))
    gradient: List[Tuple[int,int,int,int]] = []
    if steps <= 1:
        gradient.append(stops[-1])
        return gradient
    for i in range(steps):
        t = i / (steps - 1)
        pos = t * (len(stops) - 1)
        idx = int(pos)
        frac = pos - idx
        if idx >= len(stops) - 1:
            gradient.append(stops[-1])
        else:
            c1 = stops[idx]
            c2 = stops[idx + 1]
            interp = tuple(int(round(c1[k] + (c2[k] - c1[k]) * frac)) for k in range(4))
            gradient.append(interp)
    return gradient

def index_layer_color(v: Optional[float], gradient: List[Tuple[int,int,int,int]]) -> Tuple[int,int,int,int]:
    if v is None or not np.isfinite(v) or not gradient:
        return (0, 0, 0, 0)
    ratio = (float(v) - 1.0) / 99.0
    ratio = max(0.0, min(1.0, ratio))
    idx = int(round(ratio * (len(gradient) - 1)))
    idx = max(0, min(len(gradient) - 1, idx))
    return gradient[idx]

def read_sensitivity_palette_from_config(cfg_path: Path, alpha: float = 1.0) -> Dict[str, Tuple[int,int,int,int]]:
    import configparser
    # IMPORTANT: don't treat '#' as comment (colors like #ff0000)
    cfg = configparser.ConfigParser(inline_comment_prefixes=(';',), strict=False)
    cfg.read(cfg_path, encoding="utf-8")

    defaults = {"A": "#bd0026", "B": "#f03b20", "C": "#fd8d3c", "D": "#fecc5c", "E": "#ffffb2"}
    pal: Dict[str, Tuple[int,int,int,int]] = {}
    for code in ["A","B","C","D","E"]:
        hexc = None
        if code in cfg:
            hexc = cfg[code].get("category_colour", None) or cfg[code].get("category_color", None)
        pal[code] = hex_to_rgba(hexc or defaults[code], alpha)
    return pal

def read_ranges_map(cfg_path: Path) -> Dict[str, range]:
    import configparser
    cfg = configparser.ConfigParser(inline_comment_prefixes=(';',), strict=False)
    cfg.read(cfg_path, encoding="utf-8")
    ranges: Dict[str, range] = {}
    for code in ["A","B","C","D","E"]:
        if code not in cfg:
            continue
        rtxt = (cfg[code].get("range","") or "").strip()
        if "-" in rtxt:
            try:
                lo, hi = [int(x) for x in rtxt.split("-", 1)]
                ranges[code] = range(lo, hi+1)
            except Exception:
                pass
    return ranges

def build_code_from_numeric_if_missing(val, ranges_map: Dict[str, range]) -> Optional[str]:
    try:
        if val is None or pd.isna(val):
            return None
    except Exception:
        if val is None:
            return None
    try:
        v_float = float(val)
    except Exception:
        return None
    if not np.isfinite(v_float):
        return None
    try:
        v = int(round(v_float))
    except Exception:
        return None
    for k, rr in ranges_map.items():
        if v in rr:
            return k
    return None

# ----------------------- Geometry helpers -----------------------
def is_polygon_like_gdf(gdf: gpd.GeoDataFrame) -> pd.Series:
    try:
        t = gdf.geometry.geom_type
        return t.isin(["Polygon", "MultiPolygon"])
    except Exception:
        return pd.Series([True] * len(gdf), index=gdf.index)

def polygon_rings_lonlat(geom) -> Iterable[Iterable[Tuple[float,float]]]:
    """Yield exterior rings as lists of (lon,lat). Interiors ignored for speed."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [list(geom.exterior.coords)]
    elif isinstance(geom, MultiPolygon):
        return [list(p.exterior.coords) for p in geom.geoms]
    return []

def ring_lonlat_to_pixels(ring: Iterable[Tuple[float,float]], z: int, x: int, y: int, tol: float=0.3) -> List[Tuple[float,float]]:
    out: List[Tuple[float,float]] = []
    last = None
    for lon, lat in ring:
        px, py = lonlat_to_tile_px(lon, lat, z, x, y)
        if last is None or (abs(px - last[0]) > tol or abs(py - last[1]) > tol):
            out.append((px, py))
            last = (px, py)
    if len(out) >= 2 and (abs(out[0][0]-out[-1][0]) <= tol and abs(out[0][1]-out[-1][1]) <= tol):
        out[-1] = out[0]
    return out

# ----------------------- MBTiles writer (single process) -----------------------
def mbt_init(dbpath: Path, name: str, minzoom: int, maxzoom: int, bounds: Tuple[float,float,float,float]):
    if dbpath.exists():
        dbpath.unlink()
    con = sqlite3.connect(str(dbpath))
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=OFF;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA locking_mode=EXCLUSIVE;")
    cur.execute("CREATE TABLE metadata (name TEXT, value TEXT);")
    cur.execute("CREATE TABLE tiles (zoom_level INTEGER, tile_column INTEGER, tile_row INTEGER, tile_data BLOB);")
    cur.execute("CREATE UNIQUE INDEX tile_index ON tiles (zoom_level, tile_column, tile_row);")
    minlon, minlat, maxlon, maxlat = bounds
    md = {
        "name": name,
        "type": "baselayer",
        "version": "1",
        "description": name,
        "format": "png",
        "minzoom": str(minzoom),
        "maxzoom": str(maxzoom),
        "bounds": f"{minlon},{minlat},{maxlon},{maxlat}",
        "scheme": "tms"
    }
    cur.executemany("INSERT INTO metadata (name, value) VALUES (?,?)", list(md.items()))
    con.commit()
    return con

def writer_process(dbpath: str, in_q: mp.Queue, done_q: mp.Queue):
    con = None
    try:
        # First item must be ("__INIT__", name, minzoom, maxzoom, bounds)
        tag, name, minz, maxz, bounds = in_q.get()
        assert tag == "__INIT__"
        con = mbt_init(Path(dbpath), name, minz, maxz, bounds)
        cur = con.cursor()
        batch = []
        BATCH_SIZE = 300

        while True:
            item = in_q.get()
            if item == "__CLOSE__":
                if batch:
                    cur.executemany(
                        "INSERT OR REPLACE INTO tiles (zoom_level, tile_column, tile_row, tile_data) VALUES (?,?,?,?)",
                        batch
                    )
                    con.commit()
                break
            if item is None:
                continue
            # (z,x,y, png_bytes)
            z, x, y, data = item
            # MBTiles is TMS: flip y
            tms_y = (2 ** z - 1) - y
            batch.append((z, x, tms_y, sqlite3.Binary(data)))
            if len(batch) >= BATCH_SIZE:
                cur.executemany(
                    "INSERT OR REPLACE INTO tiles (zoom_level, tile_column, tile_row, tile_data) VALUES (?,?,?,?)",
                    batch
                )
                con.commit()
                batch.clear()
        done_q.put(("ok",))
    except Exception as e:
        try:
            done_q.put(("error", str(e)))
        except Exception:
            pass
    finally:
        try:
            if con: con.close()
        except Exception:
            pass

# ----------------------- Worker globals & worker -----------------------
_G_GEOMS: List = []
_G_SENS_CODES: List[Optional[str]] = []
_G_NUMVALS: List[Optional[float]] = []   # used by env/groupstotal/assetstotal
_G_FILL_MODE: str = "sensitivity"        # "sensitivity" | "env" | "groupstotal" | "assetstotal" | "importance_index" | "sensitivity_index"
_G_PALETTE: Dict[str, Tuple[int,int,int,int]] = {}
_G_STROKE_RGBA: Tuple[int,int,int,int] = (0,0,0,0)
_G_STROKE_W: float = 0.0
_G_NUM_MIN: float = 0.0
_G_NUM_MAX: float = 1.0

def _worker_init(geoms, sens_codes, numvals, fill_mode, palette, stroke_rgba, stroke_w, num_min, num_max):
    global _G_GEOMS, _G_SENS_CODES, _G_NUMVALS, _G_FILL_MODE, _G_PALETTE, _G_STROKE_RGBA, _G_STROKE_W, _G_NUM_MIN, _G_NUM_MAX
    _G_GEOMS = geoms
    _G_SENS_CODES = sens_codes
    _G_NUMVALS = numvals
    _G_FILL_MODE = fill_mode
    _G_PALETTE = palette
    _G_STROKE_RGBA = stroke_rgba
    _G_STROKE_W = float(stroke_w)
    _G_NUM_MIN = float(num_min)
    _G_NUM_MAX = float(num_max)

def _render_one_tile(task) -> Optional[Tuple[int,int,int, bytes]]:
    """
    task = (z, x, y, cand_idx_list)
    Uses globals for data and styling; returns (z,x,y,png_bytes) or None if empty.
    """
    try:
        z, x, y, cand_idx = task
        img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0,0,0,0))  # transparent background
        draw = ImageDraw.Draw(img, "RGBA")
        painted = False

        for i in cand_idx:
            geom = _G_GEOMS[i]
            if geom is None or geom.is_empty:
                continue

            if _G_FILL_MODE == "sensitivity":
                code = _G_SENS_CODES[i]
                fill_rgba = _G_PALETTE.get(code, (0,0,0,0)) if code is not None else (0,0,0,0)
            elif _G_FILL_MODE == "env":
                alpha = _G_PALETTE.get("env_alpha", 0.70)
                fill_rgba = env_index_color(_G_NUMVALS[i], alpha=alpha)
            elif _G_FILL_MODE in ("groupstotal", "assetstotal"):
                alpha = _G_PALETTE.get("blue_alpha", 0.70)
                fill_rgba = blue_ramp_rgba(_G_NUMVALS[i], _G_NUM_MIN, _G_NUM_MAX, alpha=alpha)
            elif _G_FILL_MODE in ("importance_index", "sensitivity_index"):
                gradient = _G_PALETTE.get("gradient", [])
                fill_rgba = index_layer_color(_G_NUMVALS[i], gradient)
            else:
                continue

            for ring in polygon_rings_lonlat(geom):
                path = ring_lonlat_to_pixels(ring, z, x, y, tol=0.35 if z>=9 else 0.6)
                if len(path) < 3:
                    continue
                if fill_rgba[3] > 0:
                    try:
                        draw.polygon(path, fill=fill_rgba)
                        painted = True
                    except Exception:
                        pass
                if _G_STROKE_RGBA[3] > 0 and _G_STROKE_W > 0:
                    try:
                        draw.line(path + [path[0]], fill=_G_STROKE_RGBA, width=int(max(1, round(_G_STROKE_W))))
                        painted = True
                    except Exception:
                        pass

        if not painted:
            return None

        # Encode PNG
        from io import BytesIO
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return (z, x, y, buf.getvalue())
    except Exception:
        return None

# ----------------------- Planning & helpers -----------------------
def slugify(value: str | int) -> str:
    s = str(value).strip().replace(" ", "_")
    s = re.sub(r"[^\w\-]+", "", s)
    return s[:120] if len(s) > 120 else s

def plan_tile_tasks(bounds: Tuple[float,float,float,float], minz: int, maxz: int,
                    sindex, gdf: gpd.GeoDataFrame) -> List[Tuple[int,int,int, List[int]]]:
    """
    Build full task list and print schedule summary for z in [minz..maxz].
    Each task: (z, x, y, candidate_indexes)
    """
    minx, miny, maxx, maxy = bounds
    tasks: List[Tuple[int,int,int, List[int]]] = []
    total = 0
    for z in range(minz, maxz+1):
        tx0, ty0, tx1, ty1 = tiles_covering_bounds((minx, miny, maxx, maxy), z)
        count = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
        log(f"  z{z}: scheduling tiles ({tx0},{ty0})–({tx1},{ty1}) → {count:,}")
        for x in range(tx0, tx1+1):
            for y in range(ty0, ty1+1):
                tminlon, tminlat, tmaxlon, tmaxlat = tile_bounds_lonlat(z, x, y)
                idx = list(sindex.intersection((tminlon, tminlat, tmaxlon, tmaxlat))) if sindex is not None else list(range(len(gdf)))
                if not idx:
                    continue
                tasks.append((z, x, y, idx))
                total += 1
    log(f"  total scheduled tiles with candidates: {total:,}")
    return tasks

# ----------------------- Core -----------------------
def run_one_layer(group_name: str,
                  gdf: gpd.GeoDataFrame,
                  layer_mode: str,      # "sensitivity" | "env" | "groupstotal" | "assetstotal"
                  palette: Dict[str, Tuple[int,int,int,int]],
                  ranges_map: Dict[str, range],
                  out_dir: Path,
                  minzoom: int,
                  maxzoom: int,
                  stroke_rgba: Tuple[int,int,int,int],
                  stroke_w: float,
                  procs: int):
    """
    Prepare MBTiles writer, enumerate tiles, use worker pool to render, stream to writer.
    """
    # Attributes & output naming
    if layer_mode == "sensitivity":
        sens_codes = gdf.get("sensitivity_code_max", pd.Series(index=gdf.index, dtype="string")).astype("string").str.strip().str.upper()
        if sens_codes.isna().any() or (sens_codes == "<NA>").any():
            fallback = []
            sens_num = pd.to_numeric(gdf.get("sensitivity_max", pd.Series(index=gdf.index, dtype="float")), errors="coerce")
            for i in gdf.index:
                c = None
                if pd.notna(sens_codes.at[i]) and sens_codes.at[i] != "<NA>":
                    c = str(sens_codes.at[i]).strip().str.upper()
                else:
                    c = build_code_from_numeric_if_missing(sens_num.at[i], ranges_map)
                fallback.append(c)
            sens_codes = pd.Series(fallback, index=gdf.index, dtype="object")
        numvals = pd.Series([None]*len(gdf), index=gdf.index, dtype="object")
        mbt_name = f"{group_name}_sensitivity"
        out_path = out_dir / f"{mbt_name}.mbtiles"

        vmin = 0.0; vmax = 1.0  # unused for sensitivity
    elif layer_mode == "env":
        numvals = pd.to_numeric(gdf.get("env_index", pd.Series([None]*len(gdf), index=gdf.index)), errors="coerce")
        sens_codes = pd.Series([None]*len(gdf), index=gdf.index, dtype="object")
        mbt_name = f"{group_name}_envindex"
        out_path = out_dir / f"{mbt_name}.mbtiles"
        vmin = 1.0; vmax = 100.0
    elif layer_mode == "groupstotal":
        numvals = pd.to_numeric(gdf.get("asset_groups_total", pd.Series([None]*len(gdf), index=gdf.index)), errors="coerce")
        sens_codes = pd.Series([None]*len(gdf), index=gdf.index, dtype="object")
        mbt_name = f"{group_name}_groupstotal"
        out_path = out_dir / f"{mbt_name}.mbtiles"
        vmin = float(np.nanmin(numvals.values)) if len(numvals) else 0.0
        vmax = float(np.nanmax(numvals.values)) if len(numvals) else 1.0
        if not np.isfinite(vmin): vmin = 0.0
        if not np.isfinite(vmax): vmax = 1.0
    elif layer_mode == "assetstotal":
        numvals = pd.to_numeric(gdf.get("assets_overlap_total", pd.Series([None]*len(gdf), index=gdf.index)), errors="coerce")
        sens_codes = pd.Series([None]*len(gdf), index=gdf.index, dtype="object")
        mbt_name = f"{group_name}_assetstotal"
        out_path = out_dir / f"{mbt_name}.mbtiles"
        vmin = float(np.nanmin(numvals.values)) if len(numvals) else 0.0
        vmax = float(np.nanmax(numvals.values)) if len(numvals) else 1.0
        if not np.isfinite(vmin): vmin = 0.0
        if not np.isfinite(vmax): vmax = 1.0
    elif layer_mode == "importance_index":
        numvals = pd.to_numeric(gdf.get("index_importance", pd.Series([None]*len(gdf), index=gdf.index)), errors="coerce")
        sens_codes = pd.Series([None]*len(gdf), index=gdf.index, dtype="object")
        mbt_name = f"{group_name}_importance_index"
        out_path = out_dir / f"{mbt_name}.mbtiles"
        vmin = 1.0; vmax = 100.0
    elif layer_mode == "sensitivity_index":
        numvals = pd.to_numeric(gdf.get("index_sensitivity", pd.Series([None]*len(gdf), index=gdf.index)), errors="coerce")
        sens_codes = pd.Series([None]*len(gdf), index=gdf.index, dtype="object")
        mbt_name = f"{group_name}_sensitivity_index"
        out_path = out_dir / f"{mbt_name}.mbtiles"
        vmin = 1.0; vmax = 100.0
    else:
        raise ValueError(f"Unknown layer_mode: {layer_mode}")

    # Geometry & bounds
    minx, miny, maxx, maxy = gdf.total_bounds
    miny = clamp_lat(miny); maxy = clamp_lat(maxy)
    bounds = (float(minx), float(miny), float(maxx), float(maxy))

    # Spatial index
    try:
        sindex = gdf.sindex
    except Exception:
        sindex = None

    # Writer process
    in_q = mp.Queue(maxsize=1000)
    done_q = mp.Queue()
    wp = mp.Process(target=writer_process, args=(str(out_path), in_q, done_q), daemon=True)
    wp.start()
    in_q.put(("__INIT__", mbt_name, int(minzoom), int(maxzoom), bounds))

    # Prepare worker pool globals
    geoms   = list(gdf.geometry.values)
    senslst = list(sens_codes.values)
    numlst  = list(numvals.values)
    init_args = (geoms, senslst, numlst, layer_mode, palette, stroke_rgba, stroke_w, vmin, vmax)

    # Plan tasks (also prints per-zoom schedule summary)
    tasks = plan_tile_tasks(bounds, minzoom, maxzoom, sindex, gdf)
    total_tiles = len(tasks)
    written = 0

    procs = max(1, int(procs))
    with mp.get_context("spawn").Pool(processes=procs, initializer=_worker_init, initargs=init_args) as pool:
        for out in pool.imap_unordered(_render_one_tile, tasks, chunksize=64):
            if out is not None:
                in_q.put(out)
                written += 1

    # Close writer and confirm
    in_q.put("__CLOSE__")
    status = done_q.get()
    if status[0] != "ok":
        raise RuntimeError(f"Writer failed: {status[1] if len(status)>1 else 'unknown'}")

    log(f"    {mbt_name}: tiles written {written:,} / scheduled {total_tiles:,} → {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Raster MBTiles (PNG) from tbl_flat.parquet per group (sensitivity, env_index, totals, and indexes).")
    ap.add_argument("--group-col", default="name_gis_geocodegroup", help="Grouping column (default: name_gis_geocodegroup)")
    ap.add_argument("--minzoom", type=int, default=6)
    ap.add_argument("--maxzoom", type=int, default=12)
    ap.add_argument("--only-groups", default=None, help="Comma-separated allow-list of groups")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 8)//2), help="Worker processes (default ~= half cores)")
    ap.add_argument("--stroke", default="#000000", help="Stroke color hex")
    ap.add_argument("--stroke-alpha", type=float, default=0.0, help="Stroke alpha 0..1")
    # Optional fill alpha overrides (0..1). If not provided, values are read from config.ini or default to 1.0
    ap.add_argument("--fill-alpha", type=float, default=None, help="Fill alpha for sensitivity tiles (0..1)")
    ap.add_argument("--env-alpha", type=float, default=None, help="Fill alpha for envindex tiles (0..1)")
    ap.add_argument("--blue-alpha", type=float, default=None, help="Fill alpha for *_total tiles (0..1)")
    ap.add_argument("--index-alpha", type=float, default=None, help="Fill alpha for index tiles (0..1)")
    ap.add_argument("--stroke-width", type=float, default=0.0, help="Stroke width px")
    args = ap.parse_args()

    parquet = tbl_flat_path()
    if not parquet.exists():
        raise SystemExit(f"Missing: {parquet}")

    # Load minimal columns (optional layers probed dynamically)
    required_cols = [
        args.group_col, "geometry",
        "sensitivity_code_max", "sensitivity_max",
    ]
    optional_cols = [
        "env_index",
        "asset_groups_total", "assets_overlap_total",
        "index_importance", "index_sensitivity",
    ]
    try:
        schema_cols = set(pq.ParquetFile(parquet).schema.names)
    except Exception:
        schema_cols = None

    if schema_cols is not None:
        missing_required = [c for c in required_cols if c not in schema_cols]
        if missing_required:
            raise SystemExit(f"tbl_flat missing required columns for tiles: {missing_required}")
        cols_to_read = required_cols + [c for c in optional_cols if c in schema_cols]
    else:
        cols_to_read = required_cols + optional_cols
    seen = set()
    ordered_cols = []
    for col in cols_to_read:
        if col not in seen:
            ordered_cols.append(col)
            seen.add(col)

    try:
        gdf_all = gpd.read_parquet(parquet, columns=ordered_cols)
    except (TypeError, ValueError):
        gdf_all = gpd.read_parquet(parquet)

    # EPSG:4326 expected
    try:
        if gdf_all.crs is None or str(gdf_all.crs).upper() not in ("EPSG:4326", "WGS84", "EPSG: 4326"):
            gdf_all = gdf_all.to_crs("EPSG:4326")
    except Exception:
        pass

    # Groups
    all_groups = [str(x) for x in gdf_all[args.group_col].dropna().unique().tolist()]
    all_groups.sort()
    if args.only_groups:
        allow = set([s.strip() for s in args.only_groups.split(",") if s.strip()])
        groups = [g for g in all_groups if g in allow]
        if not groups:
            raise SystemExit("No groups matched --only-groups.")
        log(f"Restricting to groups: {groups}")
    else:
        groups = all_groups

    # Styling from config.ini
    cfg_path = settings_dir() / "config.ini"
    # Alpha settings: CLI > config.ini > default(1.0)
    import configparser
    cfg = configparser.ConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
    try:
        cfg.read(cfg_path, encoding="utf-8")
    except Exception:
        pass
    def _get_alpha(key: str, cli_val: Optional[float], default: float = 1.0) -> float:
        try:
            if cli_val is not None:
                return max(0.0, min(1.0, float(cli_val)))
        except Exception:
            pass
        try:
            v = cfg["DEFAULT"].get(key, None)
            if v is not None:
                return max(0.0, min(1.0, float(str(v).strip())))
        except Exception:
            pass
        return default

    sens_alpha  = _get_alpha("tiles_fill_alpha", args.fill_alpha, 1.0)
    env_alpha   = _get_alpha("tiles_env_alpha", args.env_alpha, 1.0)
    blue_alpha  = _get_alpha("tiles_blue_alpha", args.blue_alpha, 1.0)
    index_alpha = _get_alpha("tiles_index_alpha", args.index_alpha, sens_alpha)

    sensitivity_palette = read_sensitivity_palette_from_config(cfg_path, alpha=sens_alpha)
    index_palette_src   = read_sensitivity_palette_from_config(cfg_path, alpha=index_alpha)
    index_gradient      = build_index_gradient_from_palette(index_palette_src, steps=25)
    index_palette       = {"gradient": index_gradient}
    ranges_map = read_ranges_map(cfg_path)
    stroke_rgba = hex_to_rgba(args.stroke, args.stroke_alpha)
    env_available = "env_index" in gdf_all.columns
    groups_total_available = "asset_groups_total" in gdf_all.columns
    assets_total_available = "assets_overlap_total" in gdf_all.columns
    importance_index_available = "index_importance" in gdf_all.columns
    sensitivity_index_available = "index_sensitivity" in gdf_all.columns
    blue_palette = {"blue_alpha": blue_alpha}

    out_dir = mbtiles_dir()
    log(f"Input:  {parquet}")
    log(f"Output: {out_dir}")
    log(f"Groups: {groups}")

    for gv in groups:
        slug = slugify(gv)
        gdf = gdf_all[gdf_all[args.group_col] == gv]
        gdf = gdf[is_polygon_like_gdf(gdf)]
        gdf = gdf[gdf.geometry.notna()]
        if gdf.empty:
            log(f"[{gv}] no polygonal features — skipping.")
            continue

        minx, miny, maxx, maxy = gdf.total_bounds
        log(f"[prep] group='{gv}' bounds=({minx:.3f},{miny:.3f},{maxx:.3f},{maxy:.3f}) rows={len(gdf):,}")

        # SENSITIVITY layer
        log(f"  → building {slug}_sensitivity.mbtiles …")
        run_one_layer(
            group_name=slug,
            gdf=gdf,
            layer_mode="sensitivity",
            palette=sensitivity_palette,
            ranges_map=ranges_map,
            out_dir=out_dir,
            minzoom=args.minzoom,
            maxzoom=args.maxzoom,
            stroke_rgba=stroke_rgba,
            stroke_w=args.stroke_width,
            procs=args.procs
        )

        # ENV INDEX layer
        if env_available:
            log(f"  → building {slug}_envindex.mbtiles …")
            env_palette = {"env_alpha": env_alpha}
            run_one_layer(
                group_name=slug,
                gdf=gdf,
                layer_mode="env",
                palette=env_palette,
                ranges_map=ranges_map,   # unused here, harmless
                out_dir=out_dir,
                minzoom=args.minzoom,
                maxzoom=args.maxzoom,
                stroke_rgba=stroke_rgba,
                stroke_w=args.stroke_width,
                procs=args.procs
            )
        else:
            log("  → skipping envindex tiles (env_index column missing)")

        # ASSET GROUPS TOTAL (light->dark blue, per-group min/max)
        if groups_total_available:
            log(f"  → building {slug}_groupstotal.mbtiles …")
            run_one_layer(
                group_name=slug,
                gdf=gdf,
                layer_mode="groupstotal",
                palette=blue_palette,
                ranges_map=ranges_map,  # unused
                out_dir=out_dir,
                minzoom=args.minzoom,
                maxzoom=args.maxzoom,
                stroke_rgba=stroke_rgba,
                stroke_w=args.stroke_width,
                procs=args.procs
            )
        else:
            log("  → skipping groupstotal tiles (asset_groups_total column missing)")

        # ASSETS OVERLAP TOTAL (light->dark blue, per-group min/max)
        if assets_total_available:
            log(f"  → building {slug}_assetstotal.mbtiles …")
            run_one_layer(
                group_name=slug,
                gdf=gdf,
                layer_mode="assetstotal",
                palette=blue_palette,
                ranges_map=ranges_map,  # unused
                out_dir=out_dir,
                minzoom=args.minzoom,
                maxzoom=args.maxzoom,
                stroke_rgba=stroke_rgba,
                stroke_w=args.stroke_width,
                procs=args.procs
            )
        else:
            log("  → skipping assetstotal tiles (assets_overlap_total column missing)")

        if importance_index_available:
            log(f"  → building {slug}_importance_index.mbtiles …")
            run_one_layer(
                group_name=slug,
                gdf=gdf,
                layer_mode="importance_index",
                palette=index_palette,
                ranges_map=ranges_map,
                out_dir=out_dir,
                minzoom=args.minzoom,
                maxzoom=args.maxzoom,
                stroke_rgba=stroke_rgba,
                stroke_w=args.stroke_width,
                procs=args.procs
            )
        else:
            log("  → skipping importance_index tiles (index_importance column missing)")

        if sensitivity_index_available:
            log(f"  → building {slug}_sensitivity_index.mbtiles …")
            run_one_layer(
                group_name=slug,
                gdf=gdf,
                layer_mode="sensitivity_index",
                palette=index_palette,
                ranges_map=ranges_map,
                out_dir=out_dir,
                minzoom=args.minzoom,
                maxzoom=args.maxzoom,
                stroke_rgba=stroke_rgba,
                stroke_w=args.stroke_width,
                procs=args.procs
            )
        else:
            log("  → skipping sensitivity_index tiles (index_sensitivity column missing)")

    log("All done.")

if __name__ == "__main__":
    # Windows spawn safety
    mp.freeze_support()
    main()
