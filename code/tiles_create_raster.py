# -*- coding: utf-8 -*-
"""
Raster MBTiles generator (PNG) from tbl_flat.parquet — no GDAL/Tippecanoe.

Per group in name_gis_geocodegroup, produces seven MBTiles:
    <group>_sensitivity_max.mbtiles    (colors from config.ini [A]..[E], uses sensitivity_code_max with numeric fallback)
  <group>_groupstotal.mbtiles        (light->dark blue, linear ramp of asset_groups_total per group)
  <group>_assetstotal.mbtiles        (light->dark blue, log ramp of assets_overlap_total per group)
  <group>_importance_max.mbtiles     (discrete green ramp for importance_max 1..5)
    <group>_index_owa.mbtiles          (1..100 gradient from the OWA index)

- EPSG:4326 input expected.
- Transparent background, polygon fill + optional stroke.
- Multiprocessing: worker pool renders tiles, single writer process inserts into SQLite.
- Optimization: Calculates tile tasks once per group and reuses the plan across all 7 layers to reduce overhead.
- Dependencies: geopandas, shapely, pandas, numpy, pillow (lightweight), sqlite3 (stdlib).

Usage examples:
    python tiles_create_raster.py --minzoom 6 --maxzoom 12
    python tiles_create_raster.py --only-groups "geocode_001,H3_R08"
    python tiles_create_raster.py --procs 8 --stroke-alpha 0.6
"""

INDEX_GRADIENT_STEPS = 100  # number of discrete colors for 1..100 index layers

import argparse, gc, json, math, mmap, os, sqlite3, re, multiprocessing as mp, sys, io
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
import mesa_shared
from mesa_shared import ensure_bundled_geo_data as _ensure_geo_data
_ensure_geo_data()  # repair a machine-wide PROJ/GDAL env before the GIS stack loads
import geopandas as gpd
import shapely
from shapely.geometry import Polygon, MultiPolygon
from PIL import Image, ImageDraw  # Pillow

# Candidate geometries are decoded from WKB in batches of this many per tile, so a
# worker's live shapely footprint stays bounded by the batch rather than by the
# group size. See learning.md "Tiles worker payload".
WKB_DECODE_BATCH = 20000


def _rm_quiet(path) -> None:
    """Delete a scratch file without ever failing the render for it."""
    try:
        Path(path).unlink()
    except Exception:
        pass


def spill_wkb(geoms, out_path: Path, chunk: int = 500_000) -> np.ndarray:
    """Write a geometry array to one WKB file on disk and return int64 offsets.

    The pool runs under 'spawn', so anything passed through initargs is pickled
    once per worker and each worker keeps a private copy. On basic_mosaic
    (17.6M features) the blob is ~3.2 GB, so eight workers cost ~26 GB of
    identical bytes — which is what made a 17M-cell mosaic unrenderable.
    Written to a file and mapped read-only in each worker, the operating system
    keeps one physical copy in the page cache no matter how many workers there
    are. See learning.md "Tiles worker payload".

    Encoding runs in chunks so the intermediate per-geometry bytes objects are
    never all live at once, and nothing accumulates in memory at all.
    """
    arr = np.asarray(geoms, dtype=object)
    n = len(arr)
    offsets = np.zeros(n + 1, dtype=np.int64)
    with open(out_path, "wb") as fh:
        for start in range(0, n, chunk):
            parts = shapely.to_wkb(arr[start:start + chunk])
            stop = start + len(parts)
            lengths = np.fromiter((len(p) for p in parts), dtype=np.int64, count=len(parts))
            np.cumsum(lengths, out=offsets[start + 1:stop + 1])
            offsets[start + 1:stop + 1] += offsets[start]
            fh.write(b"".join(parts))
            del parts, lengths
    return offsets


def _swap_xy_geom(geom):
    if geom is None:
        return None
    try:
        from shapely.ops import transform as _transform
    except Exception:
        return geom
    try:
        return _transform(lambda x, y, z=None: (y, x), geom)
    except TypeError:
        try:
            return _transform(lambda x, y: (y, x), geom)
        except Exception:
            return geom
    except Exception:
        return geom


def maybe_swap_xy(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Optionally swap X/Y coordinates.

    IMPORTANT: This is opt-in only. Your current Mafia Island/Tanzania data is
    already in lon/lat order (x≈39, y≈-8) and must NOT be swapped.

    Enable only if you *know* the data is lat/lon (x≈-8, y≈39):
      set MESA_SWAP_XY=1
    """
    if os.environ.get("MESA_SWAP_XY", "").strip().lower() not in {"1", "true", "yes"}:
        return gdf
    if gdf is None or not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty:
        return gdf
    out = gdf.copy()
    out["geometry"] = out.geometry.apply(_swap_xy_geom)
    try:
        out = out.set_crs(out.crs, allow_override=True)
    except Exception:
        pass
    return out

# ----------------------- Paths -----------------------
def base_dir() -> Path:
    """Resolve the mesa root folder in dev, frozen, or tools/ launches."""
    candidates: list[Path] = []

    env_hint = os.environ.get("MESA_BASE_DIR")
    if env_hint:
        candidates.append(Path(env_hint))

    try:
        owd = globals().get("original_working_directory")
        if owd:
            candidates.append(Path(owd))
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
        if p.name.lower() in ("tools", "system", "code"):
            p = p.parent
        q = p
        for _ in range(4):
            if (q / "output").exists() and (q / "input").exists():
                return q
            if (q / "tools").exists() and (q / "config.ini").exists():
                return q
            q = q.parent
        return p

    for cand in candidates:
        root = normalize(cand)
        if (root / "output").exists() and (root / "input").exists():
            return root
        if (root / "tools").exists() and (root / "config.ini").exists():
            return root

    return normalize(candidates[0])

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

# Certainty (p_max) ramp — MUST match combined_map._certainty_colour / _CERT_STOPS / _CERT_LO
# so the Classification map's Certainty raster and vector views are coloured identically.
# Contrast stretch: diag-GMM posteriors saturate near 1.0 (97%+ of cells are ≥0.999), so the
# ramp is spread over [_CERT_LO, 1.0] — only the top band carries visible structure. Cells below
# _CERT_LO clamp to the light end. The legend states the stretched range so this reads as
# "marginal differences near full confidence", not a full 0..1 certainty scale.
_CERT_STOPS = [(0.0, (255, 255, 229)), (0.5, (120, 198, 121)), (1.0, (0, 90, 50))]
_CERT_LO = 0.95

def cert_rgba(p: Optional[float], alpha: float) -> Tuple[int, int, int, int]:
    """RGBA for a posterior certainty p_max, contrast-stretched over [_CERT_LO,1.0]; grey when in-run but unknown."""
    a = int(round(max(0.0, min(1.0, float(alpha))) * 255))
    if p is None or (isinstance(p, float) and p != p):
        return (204, 204, 204, a)
    t = (max(_CERT_LO, min(1.0, float(p))) - _CERT_LO) / (1.0 - _CERT_LO)
    for (t0, c0), (t1, c1) in zip(_CERT_STOPS, _CERT_STOPS[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            r, g, b = (int(round(x + (y - x) * f)) for x, y in zip(c0, c1))
            return (r, g, b, a)
    return (0, 90, 50, a)

def blue_ramp_rgba(v: Optional[float], vmin: float, vmax: float, alpha: float=0.70,
                   scale: str="linear") -> Tuple[int,int,int,int]:
    """Light->dark blue with linear or log1p normalization; missing -> transparent."""
    if v is None or not np.isfinite(v):
        return (0, 0, 0, 0)
    if vmax <= vmin:
        t = 1.0
    elif scale == "log1p":
        lo = math.log1p(max(0.0, float(vmin)))
        hi = math.log1p(max(0.0, float(vmax)))
        cur = math.log1p(max(0.0, float(v)))
        t = max(0.0, min(1.0, (cur - lo) / (hi - lo))) if hi > lo else 1.0
    else:
        t = max(0.0, min(1.0, (float(v) - vmin) / (vmax - vmin)))
    # light #DCEBFF (220,235,255) -> dark #08306B (8,48,107)
    r = int(round(220 + t * (8   - 220)))
    g = int(round(235 + t * (48  - 235)))
    b = int(round(255 + t * (107 - 255)))
    return (r, g, b, int(max(0.0, min(1.0, alpha))*255))


def blue_ramp_legend(vmin: float, vmax: float, scale: str="linear", steps: int=5) -> list[dict]:
    """Return raw-value/color stops matching ``blue_ramp_rgba`` for MBTiles metadata."""
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return []
    count = max(2, int(steps))
    values = []
    for i in range(count):
        t = i / (count - 1)
        if vmax <= vmin:
            raw = float(vmax)
        elif scale == "log1p":
            lo = math.log1p(max(0.0, float(vmin)))
            hi = math.log1p(max(0.0, float(vmax)))
            raw = math.expm1(lo + t * (hi - lo))
        else:
            raw = float(vmin) + t * (float(vmax) - float(vmin))
        value = int(round(raw))
        if not values or value != values[-1]:
            values.append(value)
    out = []
    for value in values:
        rgba = blue_ramp_rgba(value, vmin, vmax, alpha=1.0, scale=scale)
        out.append({"value": value, "color": "#%02x%02x%02x" % rgba[:3]})
    return out


def assetstotal_vmax(values) -> float:
    """Ramp top for assets_overlap_total. The column is extreme right-skewed
    (max can be ~10,000x p99), so a raw-max top squashes the bulk into one colour.
    Cap at the 99th percentile; denser cells clamp to the darkest blue. Paired with
    the log1p scale. See learning.md "assetstotal ramp skew"."""
    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return 1.0
    v = float(np.nanpercentile(arr, 99.0))
    return v if np.isfinite(v) else 1.0

def build_index_gradient_from_palette(
    palette: Dict[str, Tuple[int,int,int,int]],
    steps: int = 25,
    order: Optional[List[str | int]] = None,
    fallback: Optional[Tuple[int,int,int,int]] = None,
) -> List[Tuple[int,int,int,int]]:
    """Build a gradient sampled along the provided palette order."""
    if order is None:
        order = ["E", "D", "C", "B", "A"]
    if fallback is None:
        fallback = (255, 255, 178, 255)  # light yellow default
    stops: List[Tuple[int,int,int,int]] = []
    for key in order:
        value = palette.get(key)
        if value is None and isinstance(key, str):
            # allow case-insensitive lookups and numeric strings
            value = palette.get(key.upper()) or palette.get(key.lower())
        if value is None:
            # int↔str fallbacks
            if isinstance(key, int):
                value = palette.get(str(key))
            else:
                try:
                    ikey = int(key)
                    value = palette.get(ikey)
                except (ValueError, TypeError):
                    value = None
        stops.append(value or fallback)

    if steps <= 1:
        return [stops[-1]]

    gradient: List[Tuple[int,int,int,int]] = []
    max_idx = len(stops) - 1
    if max_idx <= 0:
        return [stops[0]] * steps

    for i in range(steps):
        t = i / (steps - 1)
        pos = t * max_idx
        idx = int(pos)
        frac = pos - idx
        if idx >= max_idx:
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

IMPORTANCE_MAX_HEX = {
    1: "#7fbf7f",
    2: "#5fbf5f",
    3: "#3e9f4e",
    4: "#1f7f3d",
    5: "#0f5f2c",
}

def build_importance_max_palette(alpha: float = 0.8) -> Dict[str, Tuple[int,int,int,int]]:
    pal: Dict[str, Tuple[int,int,int,int]] = {}
    for level, hx in IMPORTANCE_MAX_HEX.items():
        pal[level] = hex_to_rgba(hx, alpha)
    pal["fallback"] = hex_to_rgba("#7fbf7f", alpha)
    return pal

def importance_max_color(val: Optional[float], palette: Dict) -> Tuple[int,int,int,int]:
    # Missing values arrive as NaN from the float64 value arrays; guard explicitly
    # so the common case does not go through the exception path below.
    if val is None or not np.isfinite(val):
        return (0, 0, 0, 0)
    try:
        v = int(round(float(val)))
    except Exception:
        return (0, 0, 0, 0)
    return palette.get(v, palette.get("fallback", (0, 0, 0, 0)))

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

def polygon_rings_lonlat(geom) -> List[np.ndarray]:
    """Exterior rings as (N,2) lon/lat arrays.

    Interiors are still dropped, so a polygon with a hole renders solid. That is
    wrong for imported admin layers with enclaves, but punching the hole here
    would mean erasing pixels a *neighbouring* cell may legitimately have filled
    — ImageDraw has no per-feature mask. The fix belongs with the index-raster
    rewrite, where "which cell covers this pixel" answers it by construction.
    """
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [shapely.get_coordinates(geom.exterior)]
    if isinstance(geom, MultiPolygon):
        return [shapely.get_coordinates(p.exterior) for p in geom.geoms]
    return []

def project_lonlat(coords: np.ndarray, z: int, x: int, y: int) -> Tuple[np.ndarray, np.ndarray]:
    """Project an (N,2) lon/lat array to tile pixel space in one pass.

    Call this over ALL of a tile's candidate vertices at once, never per ring.
    numpy's per-call overhead is what matters at this size: a 38-vertex ring —
    the average basic_mosaic cell — costs about as much to project one at a time
    as the Python loop it replaces, and the first attempt at this measured 1.1x
    for exactly that reason. Batched over a whole tile it is worth ~19x.
    See learning.md "Tile projection is the hot loop".
    """
    n = 2.0 ** z
    lat_rad = np.radians(np.clip(coords[:, 1], -WEBMERCATOR_LAT_MAX, WEBMERCATOR_LAT_MAX))
    px = ((coords[:, 0] + 180.0) / 360.0 * n - x) * TILE_SIZE
    py = ((1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / np.pi) / 2.0 * n - y) * TILE_SIZE
    return px, py


def decimate_ring(px: np.ndarray, py: np.ndarray, tol: float) -> List[Tuple[float,float]]:
    """Drop vertices that do not move more than `tol` from the last kept one.

    Greedy against the last *kept* vertex, so it cannot be written as a diff
    over neighbours: a long run of sub-tol steps must collapse to one point, not
    survive as many. The caller has already screened the two common cases
    (everything moves, nothing moves), so the loop here is the rare path.
    """
    keep = np.ones(len(px), dtype=bool)
    lx, ly = px[0], py[0]
    for i in range(1, len(px)):
        if abs(px[i] - lx) > tol or abs(py[i] - ly) > tol:
            lx, ly = px[i], py[i]
        else:
            keep[i] = False
    return _close_ring(px[keep], py[keep], tol)


def _close_ring(px: np.ndarray, py: np.ndarray, tol: float) -> List[Tuple[float,float]]:
    out = list(zip(px.tolist(), py.tolist()))
    if len(out) >= 2 and (abs(out[0][0]-out[-1][0]) <= tol and abs(out[0][1]-out[-1][1]) <= tol):
        out[-1] = out[0]
    return out


def ring_lonlat_to_pixels(ring, z: int, x: int, y: int, tol: float=0.3) -> List[Tuple[float,float]]:
    """Single-ring convenience wrapper. The renderer batches instead."""
    pts = np.asarray(ring, dtype="float64")
    if pts.ndim != 2 or len(pts) == 0:
        return []
    px, py = project_lonlat(pts, z, x, y)
    if len(px) > 1:
        dx = np.abs(np.diff(px)); dy = np.abs(np.diff(py))
        if np.all((dx > tol) | (dy > tol)):
            return _close_ring(px, py, tol)
        if (px.max() - px.min() <= tol) and (py.max() - py.min() <= tol):
            return _close_ring(px[:1], py[:1], tol)
        return decimate_ring(px, py, tol)
    return _close_ring(px, py, tol)

# ----------------------- MBTiles writer (single process) -----------------------
def mbt_init(dbpath: Path, name: str, minzoom: int, maxzoom: int,
             bounds: Tuple[float,float,float,float], extra_metadata: Optional[Dict[str, str]]=None):
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
        # Tiles are stored using the MBTiles default (TMS/Y origin at the south)
        "scheme": "tms"
    }
    if extra_metadata:
        md.update({str(k): str(v) for k, v in extra_metadata.items()})
    cur.executemany("INSERT INTO metadata (name, value) VALUES (?,?)", list(md.items()))
    con.commit()
    return con

def writer_process(dbpath: str, in_q: mp.Queue, done_q: mp.Queue):
    con = None
    try:
        # First item is ("__INIT__", name, minzoom, maxzoom, bounds[, metadata]).
        init = in_q.get()
        tag, name, minz, maxz, bounds = init[:5]
        extra_metadata = init[5] if len(init) > 5 else None
        assert tag == "__INIT__"
        con = mbt_init(Path(dbpath), name, minz, maxz, bounds, extra_metadata)
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
# Everything here is pickled once per worker under 'spawn', so all per-feature
# state is held as compact buffers (numpy arrays) rather than Python objects —
# and the geometry, which dwarfs the rest, is not passed at all: workers map the
# WKB file written by spill_wkb(). See learning.md "Tiles worker payload".
_G_WKB_FH = None                                          # keeps the mapping alive
_G_WKB: "mmap.mmap | bytes" = b""
_G_WKB_OFF: np.ndarray = np.zeros(1, dtype=np.int64)
_G_SENS_CODES: np.ndarray = np.zeros(0, dtype=np.int16)   # -1 = missing
_G_SENS_CATS: List[Optional[str]] = []                     # code -> label
_G_VALS_BY_MODE: Dict[str, np.ndarray] = {}                # mode -> float64, NaN = missing
_G_PALETTES_BY_MODE: Dict[str, Dict] = {}
_G_STROKE_RGBA: Tuple[int,int,int,int] = (0,0,0,0)
_G_STROKE_W: float = 0.0
# Per-feature precomputed RGBA for categorical layers (seg_signatures, seg_clusters):
# mode -> uint8 (N,4) array aligned to feature index.
_G_COLORS_BY_MODE: Dict[str, np.ndarray] = {}
# Per-mode draw-order key: higher wins a contested pixel. Kept separate from
# _G_VALS_BY_MODE because they differ where it matters — the sensitivity layer
# colours from a category code whose alphabetical order (A..E) runs opposite to
# severity, so ordering by it would hand the pixel to the *least* sensitive cell.
_G_ORDER_BY_MODE: Dict[str, np.ndarray] = {}

def _worker_init(
    wkb_path,
    wkb_offsets,
    sens_codes,
    sens_cats,
    vals_by_mode,
    palettes_by_mode,
    stroke_rgba,
    stroke_w,
    colors_by_mode=None,
    order_by_mode=None,
):
    global _G_WKB, _G_WKB_OFF, _G_WKB_FH, _G_SENS_CODES, _G_SENS_CATS, _G_VALS_BY_MODE
    global _G_PALETTES_BY_MODE, _G_STROKE_RGBA, _G_STROKE_W, _G_COLORS_BY_MODE, _G_ORDER_BY_MODE
    # Read-only mapping: every worker sees the same physical pages, so the
    # geometry costs one copy for the whole pool rather than one per process.
    _G_WKB_FH = open(wkb_path, "rb")
    _G_WKB = mmap.mmap(_G_WKB_FH.fileno(), 0, access=mmap.ACCESS_READ)
    _G_WKB_OFF = wkb_offsets
    _G_SENS_CODES = sens_codes
    _G_SENS_CATS = list(sens_cats or [])
    _G_VALS_BY_MODE = vals_by_mode or {}
    _G_PALETTES_BY_MODE = palettes_by_mode or {}
    _G_STROKE_RGBA = stroke_rgba
    _G_STROKE_W = float(stroke_w)
    _G_COLORS_BY_MODE = colors_by_mode or {}
    _G_ORDER_BY_MODE = order_by_mode or {}

def encode_tile_png(img: "Image.Image") -> bytes:
    """PNG-encode a tile.

    `optimize=True` was costing 1.92 ms a tile against 0.66 ms here, to save
    6.5% of the bytes — measured over the 195 tiles of a real project, not a
    synthetic one.

    Palettising was tried and dropped. It looked like a large win on a dense
    synthetic tile, but real tiles are mostly transparent, and RGBA with PNG's
    filters compresses long transparent runs better than an index plane plus a
    palette: measured 4.40 ms and 357 KB against 0.66 ms and 384 KB. The
    synthetic benchmark had simply been unrepresentative.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _render_one_tile(task) -> Optional[Tuple[int,int,int, bytes]]:
    """
    task = (z, x, y, cand_idx_list, mode, num_min, num_max)
    Uses globals for data and styling; returns (z,x,y,png_bytes) or None if empty.
    """
    try:
        z, x, y, cand_idx, mode, num_min, num_max = task
        img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0,0,0,0))  # transparent background
        draw = ImageDraw.Draw(img, "RGBA")
        painted = False

        palette = _G_PALETTES_BY_MODE.get(mode, {})
        numvals = _G_VALS_BY_MODE.get(mode)
        colors = _G_COLORS_BY_MODE.get(mode)

        # Paint low values first so that where several cells land on the same
        # pixel — the normal case at coarse zoom, where one pixel can cover
        # hundreds of cells — the highest value survives. That makes the tile a
        # max-aggregation of what is under it instead of "whichever cell came
        # last out of the index", which is arbitrary and order-dependent.
        order = _G_ORDER_BY_MODE.get(mode)
        if order is not None and len(cand_idx) > 1:
            v = order[cand_idx]
            cand_idx = cand_idx[np.argsort(np.where(np.isnan(v), -np.inf, v), kind="stable")]

        # Decode candidates from the WKB blob in batches: one vectorised
        # shapely call per batch, and never more than WKB_DECODE_BATCH live
        # geometries at a time even when a coarse tile covers the whole AOI.
        off = _G_WKB_OFF
        blob = _G_WKB
        tol = 0.35 if z >= 9 else 0.6
        for start in range(0, len(cand_idx), WKB_DECODE_BATCH):
            batch = cand_idx[start:start + WKB_DECODE_BATCH]
            geoms = shapely.from_wkb([blob[off[i]:off[i + 1]] for i in batch])
            rings: List[np.ndarray] = []
            ring_fill: List[Tuple[int,int,int,int]] = []

            for geom, i in zip(geoms, batch):
                if geom is None or geom.is_empty:
                    continue

                if mode == "sensitivity":
                    c = _G_SENS_CODES[i]
                    code = _G_SENS_CATS[c] if c >= 0 else None
                    fill_rgba = palette.get(code, (0,0,0,0)) if code is not None else (0,0,0,0)
                elif mode in ("groupstotal", "assetstotal"):
                    alpha = palette.get("blue_alpha", 0.70)
                    v = numvals[i] if numvals is not None else None
                    scale = "log1p" if mode == "assetstotal" else "linear"
                    fill_rgba = blue_ramp_rgba(v, float(num_min), float(num_max), alpha=alpha, scale=scale)
                elif mode == "importance_max":
                    imp_pal = palette.get("importance_max_colors", {})
                    v = numvals[i] if numvals is not None else None
                    fill_rgba = importance_max_color(v, imp_pal)
                elif mode == "index_owa":
                    gradient = palette.get("gradient", [])
                    v = numvals[i] if numvals is not None else None
                    fill_rgba = index_layer_color(v, gradient)
                elif colors is not None:
                    # Categorical layers (seg_signatures, seg_clusters, segmv): per-feature
                    # RGBA precomputed in main() (signature ramp / cluster palette). Cells
                    # with no value get a transparent colour and are skipped.
                    fill_rgba = tuple(colors[i].tolist())
                else:
                    continue

                for ring in polygon_rings_lonlat(geom):
                    if len(ring) >= 3:
                        rings.append(ring)
                        ring_fill.append(fill_rgba)

            if not rings:
                continue

            # Project every vertex of every ring in this batch in one call.
            # Projecting ring by ring costs more in numpy call overhead than it
            # saves — measured 1.1x that way against ~19x batched, because the
            # average cell is only 38 vertices. See project_lonlat.
            lens = np.fromiter((len(r) for r in rings), dtype=np.int64, count=len(rings))
            starts = np.concatenate(([0], np.cumsum(lens)))
            px_all, py_all = project_lonlat(np.concatenate(rings), z, x, y)

            # Per-ring extents and "does every step clear tol", both vectorised,
            # so the decimation loop only runs for the rings that need it.
            heads = starts[:-1]
            xmin = np.minimum.reduceat(px_all, heads); xmax = np.maximum.reduceat(px_all, heads)
            ymin = np.minimum.reduceat(py_all, heads); ymax = np.maximum.reduceat(py_all, heads)
            outside = (xmax < 0.0) | (xmin > float(TILE_SIZE)) | (ymax < 0.0) | (ymin > float(TILE_SIZE))
            collapses = (xmax - xmin <= tol) & (ymax - ymin <= tol)
            step = np.ones(len(px_all), dtype=bool)
            step[1:] = (np.abs(np.diff(px_all)) > tol) | (np.abs(np.diff(py_all)) > tol)
            step[heads] = True
            keeps_all = np.minimum.reduceat(step.astype(np.int8), heads).astype(bool)

            for r_i in range(len(rings)):
                # Fast reject: ring's pixel bbox entirely off this tile. Kept
                # because bbox-based candidate selection over-selects.
                if outside[r_i] or collapses[r_i]:
                    continue
                a, b = starts[r_i], starts[r_i + 1]
                if keeps_all[r_i]:
                    path = _close_ring(px_all[a:b], py_all[a:b], tol)
                else:
                    path = decimate_ring(px_all[a:b], py_all[a:b], tol)
                if len(path) < 3:
                    continue

                fill_rgba = ring_fill[r_i]
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

        return (z, x, y, encode_tile_png(img))
    except Exception:
        return None

# ----------------------- Planning & helpers -----------------------
def slugify(value: str | int) -> str:
    s = str(value).strip().replace(" ", "_")
    s = re.sub(r"[^\w\-]+", "", s)
    return s[:120] if len(s) > 120 else s

def _feature_tile_ranges(bboxes: np.ndarray, z: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tile index range each feature bbox spans at zoom z, computed arithmetically."""
    n = 2.0 ** z
    minx, miny, maxx, maxy = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    tx0 = np.floor((minx + 180.0) / 360.0 * n)
    tx1 = np.floor((maxx + 180.0) / 360.0 * n)
    lat_hi = np.radians(np.clip(maxy, -WEBMERCATOR_LAT_MAX, WEBMERCATOR_LAT_MAX))
    lat_lo = np.radians(np.clip(miny, -WEBMERCATOR_LAT_MAX, WEBMERCATOR_LAT_MAX))
    ty0 = np.floor((1.0 - np.log(np.tan(lat_hi) + 1.0 / np.cos(lat_hi)) / np.pi) / 2.0 * n)
    ty1 = np.floor((1.0 - np.log(np.tan(lat_lo) + 1.0 / np.cos(lat_lo)) / np.pi) / 2.0 * n)
    hi = int(n) - 1
    return (np.clip(tx0, 0, hi).astype(np.int64), np.clip(ty0, 0, hi).astype(np.int64),
            np.clip(tx1, 0, hi).astype(np.int64), np.clip(ty1, 0, hi).astype(np.int64))


def plan_tile_tasks(bounds: Tuple[float,float,float,float], minz: int, maxz: int,
                    bboxes: Optional[np.ndarray], n_features: int) -> List[Tuple[int,int,int, np.ndarray]]:
    """
    Build full task list and print schedule summary for z in [minz..maxz].
    Each task: (z, x, y, candidate_indexes as an int32 array)

    Candidates are kept as int32 arrays, not Python lists: a coarse tile over a
    large group can hold millions of them, and a list costs ~64 B per entry
    against 4 B here (and pickles ~5x larger per task).

    Assignment is by arithmetic on the feature bounding boxes rather than one
    spatial-index query per tile. Web Mercator tile indices are a closed-form
    function of lon/lat, so every feature's tile range is a vectorised
    calculation, and features are bucketed with a single argsort — O(N log N)
    over features instead of O(tiles) index queries in the parent process,
    which was single-threaded and grew with the tile count.

    The result is the same candidate set: both forms select features whose bbox
    intersects the tile's bbox.
    """
    minx, miny, maxx, maxy = bounds
    tasks: List[Tuple[int,int,int, np.ndarray]] = []
    total = 0
    for z in range(minz, maxz+1):
        tx0, ty0, tx1, ty1 = tiles_covering_bounds((minx, miny, maxx, maxy), z)
        count = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
        log(f"  z{z}: scheduling tiles ({tx0},{ty0})–({tx1},{ty1}) → {count:,}")

        if bboxes is None or len(bboxes) == 0:
            for x in range(tx0, tx1+1):
                for y in range(ty0, ty1+1):
                    tasks.append((z, x, y, np.arange(n_features, dtype=np.int32)))
                    total += 1
            continue

        fx0, fy0, fx1, fy1 = _feature_tile_ranges(bboxes, z)
        # Expand each feature into the (x,y) tiles its bbox touches. Features
        # spanning many tiles are rare and the spans are small, so building the
        # pairs directly is cheaper than any index.
        spans_x = (fx1 - fx0 + 1)
        spans_y = (fy1 - fy0 + 1)
        counts = spans_x * spans_y
        feat = np.repeat(np.arange(len(bboxes), dtype=np.int64), counts)
        within = np.arange(counts.sum(), dtype=np.int64) - np.repeat(
            np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
        sx = np.repeat(spans_x, counts)
        gx = np.repeat(fx0, counts) + (within % sx)
        gy = np.repeat(fy0, counts) + (within // sx)

        keep = (gx >= tx0) & (gx <= tx1) & (gy >= ty0) & (gy <= ty1)
        gx, gy, feat = gx[keep], gy[keep], feat[keep]
        if feat.size == 0:
            continue

        tile_key = gx * (1 << 32) + gy
        order = np.argsort(tile_key, kind="stable")
        tile_key, feat = tile_key[order], feat[order]
        edges = np.flatnonzero(np.diff(tile_key)) + 1
        for lo, hi in zip(np.concatenate(([0], edges)), np.concatenate((edges, [len(tile_key)]))):
            k = int(tile_key[lo])
            tasks.append((z, k >> 32, k & 0xFFFFFFFF, feat[lo:hi].astype(np.int32)))
            total += 1
    log(f"  total scheduled tiles with candidates: {total:,}")
    return tasks

# ----------------------- Core -----------------------
def main():
    ap = argparse.ArgumentParser(description="Raster MBTiles (PNG) from tbl_flat.parquet per group (sensitivity, totals, and indexes).")
    ap.add_argument("--group-col", default="name_gis_geocodegroup", help="Grouping column (default: name_gis_geocodegroup)")
    ap.add_argument("--minzoom", type=int, default=6)
    ap.add_argument("--maxzoom", type=int, default=12)
    ap.add_argument("--only-groups", default=None, help="Comma-separated allow-list of groups")
    def _default_procs() -> int:
        try:
            cpu = int(os.cpu_count() or 8)
        except Exception:
            cpu = 8
        # Frozen builds pay a heavy cost per worker start on Windows.
        if getattr(sys, "frozen", False):
            return max(1, min(4, cpu // 4))
        return max(1, cpu // 2)

    ap.add_argument("--procs", type=int, default=_default_procs(), help="Worker processes (default ~= half cores; frozen builds cap lower)")
    ap.add_argument("--no-max-aggregate", action="store_true",
                    help="Draw candidates in index order instead of value order. Restores the "
                         "pre-5.7 behaviour where an arbitrary cell wins a contested pixel.")
    ap.add_argument("--stroke", default="#000000", help="Stroke color hex")
    ap.add_argument("--stroke-alpha", type=float, default=0.0, help="Stroke alpha 0..1")
    # Optional fill alpha overrides (0..1). If not provided, values are read from config.ini or default to 1.0
    ap.add_argument("--fill-alpha", type=float, default=None, help="Fill alpha for sensitivity tiles (0..1)")
    ap.add_argument("--blue-alpha", type=float, default=None, help="Fill alpha for *_total tiles (0..1)")
    ap.add_argument("--index-alpha", type=float, default=None, help="Fill alpha for index tiles (0..1)")
    ap.add_argument("--importance-max-alpha", type=float, default=None, help="Fill alpha for importance_max tiles (0..1)")
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
        "code",  # needed to join segmentation categories; optional so old datasets still tile
        "asset_groups_total", "assets_overlap_total",
        "index_owa",
        "importance_max",
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

    # Enumerate groups from the group column alone. Reading the whole of tbl_flat
    # here used to hold ~18 GB of geometry for the entire run on top of the
    # per-group copy below; each group is now read on demand inside the loop.
    gcol_tbl = pq.read_table(parquet, columns=[args.group_col])
    all_groups = sorted(
        str(x) for x in gcol_tbl[args.group_col].unique().to_pylist() if x is not None
    )
    del gcol_tbl
    if args.only_groups:
        allow = set([s.strip() for s in args.only_groups.split(",") if s.strip()])
        groups = [g for g in all_groups if g in allow]
        if not groups:
            raise SystemExit("No groups matched --only-groups.")
        log(f"Restricting to groups: {groups}")
    else:
        groups = all_groups

    # Styling from config.ini
    # Prefer the root config.ini; fall back to legacy system/config.ini.
    cfg_path = base_dir() / "config.ini"
    if not cfg_path.exists():
        cfg_path = settings_dir() / "config.ini"
    # Alpha settings: CLI > config.ini > default(1.0)
    import configparser
    cfg = configparser.ConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
    try:
        cfg.read(cfg_path, encoding="utf-8")
    except Exception:
        pass

    def _get_int(key: str, default: int) -> int:
        try:
            v = cfg["DEFAULT"].get(key, None)
            if v is None:
                return int(default)
            s = str(v)
            for sep in (';', '#'):
                if sep in s:
                    s = s.split(sep, 1)[0]
            s = s.strip()
            if not s:
                return int(default)
            return int(float(s))
        except Exception:
            return int(default)
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
    blue_alpha  = _get_alpha("tiles_blue_alpha", args.blue_alpha, 1.0)
    index_alpha = _get_alpha("tiles_index_alpha", args.index_alpha, sens_alpha)

    # Tile progress heartbeat (log every N tiles per layer).
    tiles_progress_every = max(1, _get_int("tiles_progress_every", 1000))

    sensitivity_palette = read_sensitivity_palette_from_config(cfg_path, alpha=sens_alpha)
    index_palette_src   = read_sensitivity_palette_from_config(cfg_path, alpha=index_alpha)
    sensitivity_index_gradient = build_index_gradient_from_palette(
        index_palette_src,
        steps=INDEX_GRADIENT_STEPS,
    )
    sensitivity_index_palette = {"gradient": sensitivity_index_gradient}
    importance_max_alpha = _get_alpha("tiles_importance_max_alpha", args.importance_max_alpha, sens_alpha)
    importance_max_palette = {"importance_max_colors": build_importance_max_palette(importance_max_alpha)}
    ranges_map = read_ranges_map(cfg_path)
    stroke_rgba = hex_to_rgba(args.stroke, args.stroke_alpha)
    max_aggregate = not bool(args.no_max_aggregate)
    groups_total_available = "asset_groups_total" in ordered_cols
    assets_total_available = "assets_overlap_total" in ordered_cols
    index_owa_available = "index_owa" in ordered_cols
    importance_max_available = "importance_max" in ordered_cols
    blue_palette = {"blue_alpha": blue_alpha}

    out_dir = mbtiles_dir()
    log(f"Input:  {parquet}")
    log(f"Output: {out_dir}")
    log(f"Groups: {groups}")

    ctx = mp.get_context("spawn")

    for gv in groups:
        slug = slugify(gv)
        # Read just this group. basic_mosaic alone is ~18 GB as a GeoDataFrame, so
        # nothing but the current group may be resident, and it is dropped again
        # before the worker pool starts (see the del below).
        try:
            gdf = gpd.read_parquet(parquet, columns=ordered_cols,
                                   filters=[(args.group_col, "=", gv)])
        except (TypeError, ValueError):
            gdf = gpd.read_parquet(parquet, filters=[(args.group_col, "=", gv)])

        # EPSG:4326 expected
        try:
            if gdf.crs is None or str(gdf.crs).upper() not in ("EPSG:4326", "WGS84", "EPSG: 4326"):
                gdf = gdf.to_crs("EPSG:4326")
        except Exception:
            pass

        # Optional debug escape hatch: swap X/Y only if explicitly enabled.
        try:
            gdf = maybe_swap_xy(gdf)
        except Exception:
            pass

        gdf = gdf[is_polygon_like_gdf(gdf)]
        gdf = gdf[gdf.geometry.notna()]
        if gdf.empty:
            log(f"[{gv}] no polygonal features — skipping.")
            del gdf
            continue

        minx, miny, maxx, maxy = gdf.total_bounds
        log(f"[prep] group='{gv}' bounds=({minx:.3f},{miny:.3f},{maxx:.3f},{maxy:.3f}) rows={len(gdf):,}")

        # Pre-calculate tasks for this group (shared across all layers)
        miny_c = clamp_lat(miny); maxy_c = clamp_lat(maxy)
        bounds_c = (float(minx), float(miny_c), float(maxx), float(maxy_c))
        
        # Feature bounding boxes, not a spatial index: tile assignment is
        # closed-form arithmetic, so the ~1.8 GB STRtree basic_mosaic used to
        # build here is not needed at all.
        try:
            bboxes = shapely.bounds(gdf.geometry.values)
        except Exception:
            bboxes = None

        tasks = plan_tile_tasks(bounds_c, args.minzoom, args.maxzoom, bboxes, len(gdf))
        log(f"  → planned {len(tasks):,} tiles (shared across layers)")
        del bboxes

        # Precompute per-feature values for all supported layers once per group.
        sens_codes = gdf.get("sensitivity_code_max", pd.Series(index=gdf.index, dtype="string")).astype("string").str.strip().str.upper()
        if sens_codes.isna().any() or (sens_codes == "<NA>").any():
            fallback = []
            sens_num = pd.to_numeric(gdf.get("sensitivity_max", pd.Series(index=gdf.index, dtype="float")), errors="coerce")
            for i in gdf.index:
                c = None
                if pd.notna(sens_codes.at[i]) and sens_codes.at[i] != "<NA>":
                    c = str(sens_codes.at[i]).strip().upper()
                else:
                    c = build_code_from_numeric_if_missing(sens_num.at[i], ranges_map)
                fallback.append(c)
            sens_codes = pd.Series(fallback, index=gdf.index, dtype="object")
        # Ship sensitivity as categorical codes + a small label table rather than
        # one Python string reference per feature.
        sens_cat = pd.Categorical(sens_codes)
        sens_code_arr = sens_cat.codes.astype(np.int16)          # -1 = missing
        sens_cats = [None if pd.isna(c) else str(c) for c in sens_cat.categories]
        del sens_codes, sens_cat

        # float64 with NaN for missing: 8 B/feature against ~44 B for a list of
        # float/None, and it pickles as one contiguous buffer per worker.
        vals_by_mode: Dict[str, np.ndarray] = {}
        group_minmax: Dict[str, Tuple[float, float]] = {}

        if groups_total_available:
            s = pd.to_numeric(gdf.get("asset_groups_total", pd.Series([None]*len(gdf), index=gdf.index)), errors="coerce")
            vals_by_mode["groupstotal"] = s.to_numpy(dtype=np.float64, na_value=np.nan)
            try:
                vmin = float(np.nanmin(s.values)) if len(s) else 0.0
                vmax = float(np.nanmax(s.values)) if len(s) else 1.0
            except Exception:
                vmin, vmax = 0.0, 1.0
            if not np.isfinite(vmin):
                vmin = 0.0
            if not np.isfinite(vmax):
                vmax = 1.0
            group_minmax["groupstotal"] = (vmin, vmax)

        if assets_total_available:
            s = pd.to_numeric(gdf.get("assets_overlap_total", pd.Series([None]*len(gdf), index=gdf.index)), errors="coerce")
            vals_by_mode["assetstotal"] = s.to_numpy(dtype=np.float64, na_value=np.nan)
            try:
                vmin = float(np.nanmin(s.values)) if len(s) else 0.0
                vmax = assetstotal_vmax(s.values)   # p99 cap for skew; see helper
            except Exception:
                vmin, vmax = 0.0, 1.0
            if not np.isfinite(vmin):
                vmin = 0.0
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1.0
            group_minmax["assetstotal"] = (vmin, vmax)

        if importance_max_available:
            s = pd.to_numeric(gdf.get("importance_max", pd.Series([None]*len(gdf), index=gdf.index)), errors="coerce")
            vals_by_mode["importance_max"] = s.to_numpy(dtype=np.float64, na_value=np.nan)

        if index_owa_available:
            s = pd.to_numeric(gdf.get("index_owa", pd.Series([None]*len(gdf), index=gdf.index)), errors="coerce")
            vals_by_mode["index_owa"] = s.to_numpy(dtype=np.float64, na_value=np.nan)

        palettes_by_mode: Dict[str, Dict] = {
            "sensitivity": sensitivity_palette,
            "groupstotal": blue_palette,
            "assetstotal": blue_palette,
            "importance_max": importance_max_palette,
            "index_owa": sensitivity_index_palette,
        }

        # Draw-order keys (higher wins a contested pixel). The numeric layers
        # order by the value they are coloured from; sensitivity needs its own
        # key because it is coloured from a category code, not a number.
        order_by_mode: Dict[str, np.ndarray] = {}
        if max_aggregate:
            for _m in ("groupstotal", "assetstotal", "importance_max", "index_owa"):
                if _m in vals_by_mode:
                    order_by_mode[_m] = vals_by_mode[_m]
            if "sensitivity_max" in gdf.columns:
                order_by_mode["sensitivity"] = pd.to_numeric(
                    gdf["sensitivity_max"], errors="coerce"
                ).to_numpy(dtype=np.float64, na_value=np.nan)

        # Segmentation categorical layers — auto when tbl_segmentation/<slug>.parquet
        # exists. Per-feature RGBA is precomputed here (signature ramp / cluster
        # palette) and shipped to workers via colors_by_mode; cells with no
        # segmentation get a transparent colour so they are not painted.
        colors_by_mode: Dict[str, np.ndarray] = {}
        seg_part = gpq_dir() / "tbl_segmentation" / f"{slug}.parquet"
        if seg_part.exists() and "code" in gdf.columns:
            try:
                import segmentation as _segmod
                seg_df = pd.read_parquet(seg_part)
                seg_df["code"] = seg_df["code"].astype(str)
                codes = gdf["code"].astype(str).values
                a = int(round(255 * sens_alpha))
                sig_by_code = dict(zip(seg_df["code"], seg_df["signature"].fillna("")))
                # uint8 (N,4) rather than a list of tuples: 4 B/feature against ~80 B.
                sig_colors = np.zeros((len(codes), 4), dtype=np.uint8)
                sig_cache: Dict[str, Tuple[int,int,int,int]] = {}
                for n, c in enumerate(codes):
                    sig = sig_by_code.get(c, "")
                    if not sig:
                        continue
                    rgba = sig_cache.get(sig)
                    if rgba is None:
                        r, g, b = _segmod._signature_colour(sig)
                        rgba = (int(r), int(g), int(b), a)
                        sig_cache[sig] = rgba
                    sig_colors[n] = rgba
                colors_by_mode["seg_signatures"] = sig_colors
                if "cluster_id" in seg_df.columns and seg_df["cluster_id"].notna().any():
                    clu_by_code = dict(zip(seg_df["code"], seg_df["cluster_id"]))
                    clu_colors = np.zeros((len(codes), 4), dtype=np.uint8)
                    clu_cache: Dict[int, Tuple[int,int,int,int]] = {}
                    for n, c in enumerate(codes):
                        cid = clu_by_code.get(c)
                        if cid is None or (isinstance(cid, float) and cid != cid):
                            continue
                        cid = int(cid)
                        rgba = clu_cache.get(cid)
                        if rgba is None:
                            rgba = hex_to_rgba(_segmod._overview_colour(f"cluster {cid}", "clusters"), sens_alpha)
                            clu_cache[cid] = rgba
                        clu_colors[n] = rgba
                    colors_by_mode["seg_clusters"] = clu_colors
                log(f"  → segmentation tiles enabled for '{gv}' "
                    f"({'signatures' + ('+clusters' if 'seg_clusters' in colors_by_mode else '')})")
            except Exception as exc:
                log(f"  → segmentation tiles skipped for '{gv}': {exc}")
                colors_by_mode = {}

        # Shared worker pool for this group (reused across all layers). Spill the
        # geometry to a WKB file and drop the GeoDataFrame *before* spawning: the
        # parent then holds neither the GeoDataFrame nor the blob, and the workers
        # map the file instead of each unpickling a copy of it.
        wkb_path = out_dir / f"__wkb_{slug}.bin"
        wkb_off = spill_wkb(gdf.geometry.values, wkb_path)
        del gdf
        gc.collect()
        log(f"  → payload {wkb_path.stat().st_size/1024**3:.2f} GB WKB mapped by every worker "
            f"(one physical copy)")

        init_args = (str(wkb_path), wkb_off, sens_code_arr, sens_cats, vals_by_mode,
                     palettes_by_mode, stroke_rgba, args.stroke_width, colors_by_mode,
                     order_by_mode)

        procs = max(1, int(args.procs))
        with ctx.Pool(processes=procs, initializer=_worker_init, initargs=init_args) as pool:
            def _run_layer(mbt_name: str, mode: str, num_min: float, num_max: float):
                out_path = out_dir / f"{mbt_name}.mbtiles"
                in_q = mp.Queue(maxsize=1000)
                done_q = mp.Queue()
                wp = mp.Process(target=writer_process, args=(str(out_path), in_q, done_q), daemon=True)
                wp.start()
                # A coarse tile is an aggregation, and until now nothing said
                # which one. `mesa_aggregation` names the rule that decided a
                # contested pixel, so a viewer can tell the analyst rather than
                # leave them to assume the tile shows a typical value.
                #   max  — highest value under the pixel wins
                #   none — no rule; whichever candidate was drawn last
                # Reserved for the value-raster work: mode, mean, count.
                metadata = {
                    "mesa_aggregation": "max" if mode in order_by_mode else "none",
                }
                if mode in ("groupstotal", "assetstotal"):
                    scale = "log1p" if mode == "assetstotal" else "linear"
                    metadata.update({
                        "mesa_scale": scale,
                        "mesa_value_min": num_min,
                        "mesa_value_max": num_max,
                        "mesa_legend": json.dumps(
                            blue_ramp_legend(num_min, num_max, scale=scale),
                            separators=(",", ":"),
                        ),
                    })
                in_q.put(("__INIT__", mbt_name, int(args.minzoom), int(args.maxzoom), bounds_c, metadata))

                layer_tasks = [(z, x, y, idx, mode, num_min, num_max) for (z, x, y, idx) in tasks]
                total_tiles = len(layer_tasks)
                written = 0

                for i, out in enumerate(pool.imap_unordered(_render_one_tile, layer_tasks, chunksize=64), 1):
                    if out is not None:
                        in_q.put(out)
                        written += 1
                    if i % tiles_progress_every == 0 or i == total_tiles:
                        log(f"[progress] {mbt_name}: {i}/{total_tiles}")

                in_q.put("__CLOSE__")
                status = done_q.get()
                if status[0] != "ok":
                    raise RuntimeError(f"Writer failed: {status[1] if len(status)>1 else 'unknown'}")
                log(f"    {mbt_name}: tiles written {written:,} / scheduled {total_tiles:,} → {out_path}")

            # SENSITIVITY layer
            log(f"  → building {slug}_sensitivity_max.mbtiles …")
            _run_layer(f"{slug}_sensitivity_max", "sensitivity", 0.0, 1.0)

            if groups_total_available:
                log(f"  → building {slug}_groupstotal.mbtiles …")
                vmin, vmax = group_minmax.get("groupstotal", (0.0, 1.0))
                _run_layer(f"{slug}_groupstotal", "groupstotal", vmin, vmax)
            else:
                log("  → skipping groupstotal tiles (asset_groups_total column missing)")

            if assets_total_available:
                log(f"  → building {slug}_assetstotal.mbtiles …")
                vmin, vmax = group_minmax.get("assetstotal", (0.0, 1.0))
                _run_layer(f"{slug}_assetstotal", "assetstotal", vmin, vmax)
            else:
                log("  → skipping assetstotal tiles (assets_overlap_total column missing)")

            if importance_max_available:
                log(f"  → building {slug}_importance_max.mbtiles …")
                _run_layer(f"{slug}_importance_max", "importance_max", 1.0, 5.0)
            else:
                log("  → skipping importance_max tiles (importance_max column missing)")

            if index_owa_available:
                log(f"  → building {slug}_index_owa.mbtiles …")
                _run_layer(f"{slug}_index_owa", "index_owa", 1.0, 100.0)
            else:
                log("  → skipping index_owa tiles (index_owa column missing)")

            # Segmentation categorical layers (auto when tbl_segmentation exists).
            if "seg_signatures" in colors_by_mode:
                log(f"  → building {slug}_seg_signatures.mbtiles …")
                _run_layer(f"{slug}_seg_signatures", "seg_signatures", 0.0, 0.0)
            if "seg_clusters" in colors_by_mode:
                log(f"  → building {slug}_seg_clusters.mbtiles …")
                _run_layer(f"{slug}_seg_clusters", "seg_clusters", 0.0, 0.0)

        # The pool is closed, so no worker is still mapping the spill file.
        _rm_quiet(wkb_path)

        # --- Classification (segmv) rasters: rendered from the FULL geocode grid ---
        # tbl_seg_mv covers every geocode cell, but tbl_flat (the gdf above) omits cells
        # with no asset overlap (~7% of basic_mosaic). To paint those no-data cells grey on
        # the Certainty raster — instead of leaving white holes — segmv is rendered from
        # tbl_geocode_object geometry in a dedicated pool, after the shared pool is freed.
        # Two mbtiles per run (Types + a p_max Certainty ramp), named per run_id so each run
        # is independently viewable. The salt-and-pepper cluster patches make a dissolved
        # vector boundary explode to 100+ MB of GeoJSON, so raster is the only viable form.
        segmv_part = gpq_dir() / "tbl_seg_mv.parquet"
        go_part = gpq_dir() / "tbl_geocode_object.parquet"
        if segmv_part.exists() and go_part.exists():
            try:
                import segmentation as _segmod
                scols = ["run_id", "name_gis_geocodegroup", "code", "cluster_id"]
                has_pmax = "p_max" in pq.ParquetFile(segmv_part).schema_arrow.names
                if has_pmax:
                    scols.append("p_max")
                mv = pd.read_parquet(segmv_part, columns=scols)
                mv = mv[(mv["name_gis_geocodegroup"].astype(str) == gv) & mv["cluster_id"].notna()]
                if not mv.empty:
                    run_id = str(mv["run_id"].astype(str).max())   # newest run (timestamp)
                    mv = mv[mv["run_id"].astype(str) == run_id]
                    clu_by_code = dict(zip(mv["code"].astype(str), mv["cluster_id"]))
                    pmax_by_code = dict(zip(mv["code"].astype(str), mv["p_max"])) if has_pmax else {}
                    # Tolerant read: tbl_geocode_object written by the streamed
                    # merge lacks the schema-level GeoParquet header, and
                    # gpd.read_parquet refuses it outright — which silently cost
                    # this project its Classification rasters. See learning.md
                    # "GeoParquet metadata lives in two places".
                    sgeo = mesa_shared.read_geoparquet(
                        go_part, columns=["code", "geometry"],
                        filters=[("name_gis_geocodegroup", "=", gv)])
                    if sgeo.crs is not None and str(sgeo.crs).upper() not in ("EPSG:4326", "WGS84", "EPSG: 4326"):
                        sgeo = sgeo.to_crs("EPSG:4326")
                    sgeo = maybe_swap_xy(sgeo)
                    sgeo = sgeo[sgeo.geometry.notna()]
                    nodata_cert = hex_to_rgba("#cccccc", sens_alpha)
                    s_codes = sgeo["code"].astype(str).values
                    # uint8 (N,4) per layer; this grid is the largest payload in the
                    # script (basic_mosaic has ~18.5M cells here, more than tbl_flat).
                    clu_colors = np.zeros((len(s_codes), 4), dtype=np.uint8)
                    cert_colors = np.zeros((len(s_codes), 4), dtype=np.uint8)
                    clu_cache: Dict[int, Tuple[int,int,int,int]] = {}
                    for n, c in enumerate(s_codes):
                        cid = clu_by_code.get(c)
                        if cid is None or (isinstance(cid, float) and cid != cid):
                            cert_colors[n] = nodata_cert       # Certainty: grey (no data)
                            continue                           # Types: transparent (no cluster)
                        cid = int(cid)
                        rgba = clu_cache.get(cid)
                        if rgba is None:
                            rgba = hex_to_rgba(_segmod._overview_colour(f"cluster {cid}", "clusters"), sens_alpha)
                            clu_cache[cid] = rgba
                        clu_colors[n] = rgba
                        cert_colors[n] = cert_rgba(pmax_by_code.get(c), sens_alpha)
                    s_modes = [(f"{slug}_segmv_{run_id}", "segmv", clu_colors)]
                    if has_pmax:
                        s_modes.append((f"{slug}_segmv_{run_id}_cert", "segmv_cert", cert_colors))
                    s_colors = {mode: cols for (_, mode, cols) in s_modes}
                    log(f"  → classification tiles enabled for '{gv}' (run {run_id}, full grid {len(sgeo):,} cells)")
                    del mv, clu_by_code, pmax_by_code, s_codes
                    s_minx, s_miny, s_maxx, s_maxy = sgeo.total_bounds
                    s_bounds = (float(s_minx), float(clamp_lat(s_miny)), float(s_maxx), float(clamp_lat(s_maxy)))
                    try:
                        s_bboxes = shapely.bounds(sgeo.geometry.values)
                    except Exception:
                        s_bboxes = None
                    s_tasks = plan_tile_tasks(s_bounds, args.minzoom, args.maxzoom, s_bboxes, len(sgeo))
                    del s_bboxes
                    s_wkb_path = out_dir / f"__wkb_{slug}_segmv.bin"
                    s_off = spill_wkb(sgeo.geometry.values, s_wkb_path)
                    del sgeo
                    gc.collect()
                    s_init = (str(s_wkb_path), s_off, np.zeros(0, dtype=np.int16), [], {}, {},
                              stroke_rgba, args.stroke_width, s_colors)
                    with ctx.Pool(processes=procs, initializer=_worker_init, initargs=s_init) as spool:
                        for nm, mode, _cols in s_modes:
                            out_path = out_dir / f"{nm}.mbtiles"
                            in_q = mp.Queue(maxsize=1000); done_q = mp.Queue()
                            wp = mp.Process(target=writer_process, args=(str(out_path), in_q, done_q), daemon=True)
                            wp.start()
                            # Classification rasters are categorical: "highest
                            # cluster id" is not a statement anyone wants.
                            in_q.put(("__INIT__", nm, int(args.minzoom), int(args.maxzoom), s_bounds,
                                      {"mesa_aggregation": "none"}))
                            ltasks = [(z, x, y, idx, mode, 0.0, 0.0) for (z, x, y, idx) in s_tasks]
                            written = 0
                            log(f"  → building {nm}.mbtiles …")
                            for i, out in enumerate(spool.imap_unordered(_render_one_tile, ltasks, chunksize=64), 1):
                                if out is not None:
                                    in_q.put(out); written += 1
                                if i % tiles_progress_every == 0 or i == len(ltasks):
                                    log(f"[progress] {nm}: {i}/{len(ltasks)}")
                            in_q.put("__CLOSE__")
                            status = done_q.get()
                            if status[0] != "ok":
                                raise RuntimeError(f"Writer failed: {status[1] if len(status)>1 else 'unknown'}")
                            log(f"    {nm}: tiles written {written:,} / scheduled {len(ltasks):,} → {out_path}")
                            # Stable alias (<slug>_segmv_latest[_cert].mbtiles) so static
                            # consumers like qgis/mesa.qgz need not track the per-run
                            # timestamp. Best-effort: must never break the Tiles stage.
                            try:
                                import shutil
                                alias = out_dir / (f"{slug}_segmv_latest_cert.mbtiles"
                                                   if mode == "segmv_cert"
                                                   else f"{slug}_segmv_latest.mbtiles")
                                shutil.copyfile(out_path, alias)
                                log(f"    {nm}: stable alias → {alias.name}")
                            except Exception as exc:
                                log(f"    {nm}: alias copy skipped: {exc}")
                    _rm_quiet(s_wkb_path)
            except Exception as exc:
                log(f"  → classification tiles skipped for '{gv}': {exc}")
                try:
                    _rm_quiet(s_wkb_path)
                except NameError:
                    pass

    log("All done.")

if __name__ == "__main__":
    # Windows spawn safety
    mp.freeze_support()
    main()
