#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_create_geocodes.py

Create geocodes: H3 or Basic Mosaic (polygonize-based)
------------------------------------------------------

- GUI (ttkbootstrap) with log panel + progress bar. Also supports CLI via --nogui.
- H3: robust across h3 v3/v4 (uses geo_to_cells with fallbacks).
- Mosaic: adaptive quadtree tiling + parallel polygonize (Windows spawn-safe) with detailed logging.
- **GeoParquet-first**: reads assets/groups from output/geoparquet; writes ONLY to:
    * output/geoparquet/tbl_geocode_group.parquet
    * output/geoparquet/tbl_geocode_object.parquet
- **Append/refresh semantics** for BOTH H3 and Mosaic:
    * Adding a group creates/extends the geocode tables.
    * Re-running the same group name replaces only that group.

IMPORTANT: basic_mosaic is published as a geocode with fixed group name:
- Group name: "basic_mosaic" (not configurable)
- Codes: "basic_mosaic_000001", "basic_mosaic_000002", ...

Config (optional) in system/config.ini [DEFAULT]:
- workingprojection_epsg = 5973
- area_projection_epsg   = 3035
- h3_union_buffer_m      = 50
- h3_max_cells           = 4000000
- h3_from = 3
- h3_to   = 6
- mosaic_buffer_m   = 25
- mosaic_grid_size_m = 1000
- mosaic_workers = 0               # 0/absent = auto (cpu_count)
- mosaic_quadtree_max_feats_per_tile = 800
- mosaic_quadtree_max_depth = 8
- mosaic_quadtree_min_tile_m = 1000
- ttk_bootstrap_theme = flatly
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

# make_valid (Shapely >=2), optional
try:
    from shapely.validation import make_valid as shapely_make_valid
except Exception:
    shapely_make_valid = None

# --- H3 optional ---
try:
    import h3  # v3 or v4
except Exception:
    h3 = None

# --- GUI / ttkbootstrap ---
import tkinter as tk
from tkinter import scrolledtext
try:
    import ttkbootstrap as ttk
    from ttkbootstrap.constants import PRIMARY, INFO, WARNING
except Exception:
    ttk = None  # guarded
    PRIMARY = INFO = WARNING = None  # placeholders to avoid NameErrors


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
# Same values in meters (for UI display)
H3_RES_ACROSS_FLATS_M = {r: km * 1000.0 for r, km in H3_RES_ACROSS_FLATS_KM.items()}

# Approx average hex areas by resolution (km^2), order-of-magnitude
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
# GUI globals
# -----------------------------------------------------------------------------
root: Optional[tk.Tk] = None
log_widget: Optional[scrolledtext.ScrolledText] = None
progress_var: Optional[tk.DoubleVar] = None
progress_label: Optional[tk.Label] = None
original_working_directory: Optional[str] = None

# H3 helper state in GUI
suggested_h3_levels: List[int] = []
mosaic_status_var: Optional[tk.StringVar] = None
size_levels_var: Optional[tk.StringVar] = None
generate_size_btn = None
suggest_levels_btn = None


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
# Logging
# -----------------------------------------------------------------------------
def update_progress(new_value: float):
    if root is None or progress_var is None or progress_label is None:
        return
    def task():
        v = max(0, min(100, float(new_value)))
        progress_var.set(v)
        if progress_label is not None:
            progress_label.config(text=f"{int(v)}%")
    try:
        root.after(0, task)
    except Exception:
        pass

def log_to_gui(message: str, level: str = "INFO"):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted = f"{timestamp} [{level}] - {message}"
    if log_widget is not None:
        log_widget.insert(tk.END, formatted + "\n")
        log_widget.see(tk.END)
    if original_working_directory:
        try:
            with open(Path(original_working_directory) / "log.txt", "a", encoding="utf-8") as f:
                f.write(formatted + "\n")
        except Exception:
            pass
    if log_widget is None:
        print(formatted)


# -----------------------------------------------------------------------------
# Geo helpers
# -----------------------------------------------------------------------------
def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.set_crs("EPSG:4326", allow_override=True)
    try:
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        elif str(gdf.crs).upper() not in ("EPSG:4326", "WGS84"):
            gdf = gdf.to_crs(4326)
    except Exception:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    return gdf

def working_metric_crs_for(gdf: gpd.GeoDataFrame, cfg: configparser.ConfigParser) -> str:
    try:
        epsg = cfg["DEFAULT"].get("workingprojection_epsg", "").strip()
    except Exception:
        epsg = ""
    if epsg and epsg.isdigit() and epsg != "4326":
        return f"EPSG:{epsg}"
    if gdf.crs and getattr(gdf.crs, "is_projected", False):
        try:
            return gdf.crs.to_string()
        except Exception:
            pass
    return "EPSG:3857"  # reasonable metric default

def area_projection(cfg: configparser.ConfigParser) -> str:
    epsg = (cfg["DEFAULT"].get("area_projection_epsg", "3035")
            if "DEFAULT" in cfg else "3035")
    if str(epsg).isdigit():
        return f"EPSG:{epsg}"
    return epsg

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
# H3 helpers (robust across v3/v4)
# -----------------------------------------------------------------------------
def _h3_version() -> str:
    return getattr(h3, "__version__", "unknown") if h3 else "none"

def _cell_boundary(index) -> list[tuple]:
    """
    Return boundary ring as [(lng, lat), ...] for a cell index across h3 v3/v4.
    """
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
            lat, lng = p[0], p[1]  # v4 returns (lat, lng)
            out.append((lng, lat))
    return out

def _extract_polygonal(geom):
    """Return Polygon or MultiPolygon from any geometry; None if nothing polygonal."""
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
    """
    Preferred v4 path: h3.geo_to_cells(geo_interface, res).
    Fallbacks: v3 h3.polyfill(geojson, res, True/geo_json_conformant=True).
    """
    if h3 is None:
        raise RuntimeError("H3 module not available")
    gj = shp_mapping(poly)  # proper GeoJSON (lon, lat)
    # v4 preferred
    if hasattr(h3, "geo_to_cells"):
        try:
            return set(h3.geo_to_cells(gj, res))
        except Exception as e:
            log_to_gui(f"H3 geo_to_cells failed (R{res}): {e}", "WARN")
    # v3 fallback
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
    """Build hex polygons covering union_geom at given resolution."""
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
# GeoParquet-first union source for H3
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
    """
    Prefer union from GeoParquet (tbl_asset_group/object).
    If only points/lines exist, buffer them (configurable meters) to build an area.
    Always returns WGS84 polygonal geometry or None.
    """
    cfg = read_config(base_dir / "system" / "config.ini")

    # 1) Try GeoParquet groups
    pq_groups = geoparquet_path(base_dir, "tbl_asset_group")
    g = _read_parquet_gdf(pq_groups)
    g = ensure_wgs84(g)
    if not g.empty and "geometry" in g:
        try:
            u = unary_union(g.geometry)
            u = _extract_polygonal(u)
            if u and not u.is_empty:
                log_to_gui("Union source: GeoParquet tbl_asset_group", "INFO")
                return u
        except Exception:
            pass

    # 2) Try GeoParquet asset objects
    pq_objs = geoparquet_path(base_dir, "tbl_asset_object")
    ao = _read_parquet_gdf(pq_objs)
    ao = ensure_wgs84(ao)
    if not ao.empty and "geometry" in ao:
        try:
            # If any polygons exist, union them
            poly_mask = ao.geometry.geom_type.isin(["Polygon","MultiPolygon","GeometryCollection"])
            if poly_mask.any():
                u = unary_union(ao.loc[poly_mask, "geometry"])
                u = _extract_polygonal(u)
                if u and not u.is_empty:
                    log_to_gui("Union source: GeoParquet tbl_asset_object (polygons)", "INFO")
                    return u

            # Else: buffer non-polygons to get area
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
                u = unary_union(aom.geometry)
                u = _extract_polygonal(u)
                if u and not u.is_empty:
                    u_wgs84 = ensure_wgs84(gpd.GeoSeries([u], crs=aom.crs)).iloc[0]
                    log_to_gui(f"Union source: GeoParquet tbl_asset_object (buffer {buf_m} m)", "INFO")
                    return u_wgs84
        except Exception:
            pass

    return None


# -----------------------------------------------------------------------------
# Estimators to prevent runaway H3 at very high resolutions
# -----------------------------------------------------------------------------
def estimate_cells_for(union_geom, res: int, cfg: configparser.ConfigParser) -> tuple[float,float]:
    """Return (area_km2, approx_cells) for given AOI at res."""
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
def _bbox_polygon_from(
    thing: Union[gpd.GeoDataFrame, gpd.GeoSeries, Polygon, MultiPolygon, GeometryCollection]
) -> Optional[Polygon]:
    try:
        if isinstance(thing, (gpd.GeoDataFrame, gpd.GeoSeries)):
            t = ensure_wgs84(thing)
            if t.empty:
                return None
            minx, miny, maxx, maxy = t.total_bounds
        else:
            if thing is None or getattr(thing, "is_empty", True):
                return None
            minx, miny, maxx, maxy = thing.bounds
        arr = np.array([minx, miny, maxx, maxy], dtype=float)
        if not np.isfinite(arr).all():
            return None
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
            if g.crs is None:
                g.set_crs("EPSG:4326", inplace=True)
        except Exception:
            g = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    else:
        g = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if po.exists():
        try:
            o = gpd.read_parquet(po)
            if o.crs is None:
                o.set_crs("EPSG:4326", inplace=True)
        except Exception:
            o = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    else:
        o = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    g = ensure_wgs84(g)
    o = ensure_wgs84(o)
    if "id" not in g.columns:
        log_to_gui("Existing geocode group table lacks 'id' — treating as empty.", "WARN")
        g = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        o = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return g, o

def load_geocode_groups(base_dir: Path) -> gpd.GeoDataFrame:
    """Light wrapper to read only group parquet (empty if missing)."""
    pg = gpq_dir(base_dir) / "tbl_geocode_group.parquet"
    if pg.exists():
        try:
            g = gpd.read_parquet(pg)
            if g.crs is None:
                g.set_crs("EPSG:4326", inplace=True)
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
    """Append/refresh geocode groups/objects and write to parquet. Returns counts."""
    existing_g, existing_o = _load_existing_geocodes(base_dir)
    out_dir = gpq_dir(base_dir)

    # Remove pre-existing by name
    if not existing_g.empty and "name_gis_geocodegroup" in existing_g and refresh_group_names:
        rm_mask = existing_g["name_gis_geocodegroup"].isin(refresh_group_names)
        rm_ids = set(existing_g.loc[rm_mask, "id"].astype(int).tolist())
        if rm_ids:
            existing_g = existing_g.loc[~rm_mask].copy()
            if not existing_o.empty and "ref_geocodegroup" in existing_o:
                existing_o = existing_o.loc[~existing_o["ref_geocodegroup"].astype(int).isin(rm_ids)].copy()
            log_to_gui(f"Refreshed existing groups removed: {len(rm_ids)}", "INFO")

    # Assign ids to new groups
    start_id = int(existing_g["id"].max()) + 1 if ("id" in existing_g.columns and not existing_g.empty) else 1
    new_groups_gdf = new_groups_gdf.copy()
    new_groups_gdf["id"] = list(range(start_id, start_id + len(new_groups_gdf)))

    # Map to objects
    name_to_id = dict(zip(new_groups_gdf["name_gis_geocodegroup"], new_groups_gdf["id"]))
    new_objects_gdf = new_objects_gdf.copy()
    new_objects_gdf["ref_geocodegroup"] = new_objects_gdf["name_gis_geocodegroup"].map(name_to_id)

    # Merge and write
    groups_out = pd.concat([existing_g, new_groups_gdf], ignore_index=True)
    objects_out = pd.concat([existing_o, new_objects_gdf], ignore_index=True)

    groups_out = ensure_wgs84(gpd.GeoDataFrame(groups_out, geometry="geometry"))
    objects_out = ensure_wgs84(gpd.GeoDataFrame(objects_out, geometry="geometry"))

    groups_out.to_parquet(out_dir / "tbl_geocode_group.parquet", index=False)
    objects_out.to_parquet(out_dir / "tbl_geocode_object.parquet", index=False)

    return len(new_groups_gdf), len(new_objects_gdf), len(groups_out), len(objects_out)


# -----------------------------------------------------------------------------
# H3 writers (Parquet-only, append/refresh)
# -----------------------------------------------------------------------------
def write_h3_levels(base_dir: Path, levels: List[int]) -> int:
    if not levels:
        log_to_gui("No H3 levels selected.", "WARN")
        return 0

    cfg = read_config(base_dir / "system" / "config.ini")
    max_cells = float(cfg["DEFAULT"].get("h3_max_cells", "1200000")) if "DEFAULT" in cfg else 1_200_000.0

    union_geom = union_from_asset_groups_or_objects(base_dir)
    if union_geom is None:
        raise RuntimeError("No polygonal geometry found in tbl_asset_group or tbl_asset_object.")

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
        if "attributes" not in gdf.columns:
            gdf["attributes"] = None
        gdf["name_gis_geocodegroup"] = group_name
        gdf = gdf[["code", "name_gis_geocodegroup", "attributes", "geometry"]]
        objects_parts.append(gdf)
        log_to_gui(f"H3 R{r}: prepared {len(gdf):,} cells.")

        groups_rows.append({
            "name": group_name,
            "name_gis_geocodegroup": group_name,
            "title_user": f"H3 resolution {r}",
            "description": f"H3 hexagons at resolution {r}",
            "geometry": bbox_poly
        })

    if not groups_rows or not objects_parts:
        log_to_gui("No H3 output generated (all levels skipped/empty).", "WARN")
        # ensure tables exist
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
# Asset loading (GeoParquet)
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
# Mosaic worker (memory-safe polygonize per tile)
# -----------------------------------------------------------------------------
def _mosaic_tile_worker(task: Tuple[int, List[bytes]]):
    """
    Input: (tile_index, [WKB of buffered geometries for this tile])
    Output: (tile_index, [WKB of polygon faces], error:str|None)
    """
    idx, wkb_list = task
    try:
        if not wkb_list:
            return (idx, [], None)

        # WKB -> shapely, then boundaries
        geoms = [shp_wkb.loads(b) for b in wkb_list if b]
        lines = []
        for g in geoms:
            if g and not g.is_empty:
                try:
                    lb = g.boundary
                    if lb and not lb.is_empty:
                        lines.append(lb)
                except Exception:
                    continue
        if not lines:
            return (idx, [], None)

        # Progressive (batched) union to cap peak memory
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
# Adaptive quadtree tiler (skips empties)
# -----------------------------------------------------------------------------
def _plan_tiles_quadtree(a_metric: gpd.GeoDataFrame,
                         sidx,
                         minx: float, miny: float, maxx: float, maxy: float,
                         *,
                         overlap_m: float,
                         max_feats_per_tile: int = 800,
                         max_depth: int = 8,
                         min_tile_size_m: float = 0.0) -> List[Tuple[Tuple[float,float,float,float], List[int]]]:
    """
    Return list of (bounds, row_indices) for leaf tiles. Skips empty tiles.
    - a_metric: assets (buffered) in a metric CRS
    - sidx: a_metric.sindex
    - overlap_m: used for index query window (we still send full geoms; overlap avoids seams)
    """
    leaves: List[Tuple[Tuple[float,float,float,float], List[int]]] = []
    stack = [(minx, miny, maxx, maxy, 0)]
    eps = 1e-3

    while stack:
        bx0, by0, bx1, by1, depth = stack.pop()
        w, h = (bx1 - bx0), (by1 - by0)
        if w <= eps or h <= eps:
            continue

        # query with overlap
        tile_poly = box(bx0, by0, bx1, by1).buffer(overlap_m)
        try:
            idxs = list(sidx.query(tile_poly, predicate="intersects"))
        except Exception:
            idxs = list(sidx.query(tile_poly))
        n = len(idxs)

        if n == 0:
            continue  # empty tile → drop

        # stop conditions
        stop_for_size = (min_tile_size_m > 0.0 and (w <= min_tile_size_m and h <= min_tile_size_m))
        if n <= max_feats_per_tile or depth >= max_depth or stop_for_size:
            leaves.append(((bx0, by0, bx1, by1), idxs))
            continue

        # split into quadrants
        mx = (bx0 + bx1) * 0.5
        my = (by0 + by1) * 0.5
        stack.extend([
            (bx0, by0, mx,  my,  depth + 1),
            (mx,  by0, bx1, my,  depth + 1),
            (bx0, my,  mx,  by1, depth + 1),
            (mx,  my,  bx1, by1, depth + 1),
        ])

    return leaves


# -----------------------------------------------------------------------------
# Mosaic builder (adaptive + parallel + detailed logging)
# -----------------------------------------------------------------------------
def mosaic_faces_from_assets_parallel(base_dir: Path,
                                      buffer_m: float,
                                      grid_size_m: float,
                                      workers: int) -> gpd.GeoDataFrame:
    """
    Build mosaic faces with an adaptive quadtree tiler (fewer, smarter tiles),
    parallel polygonize, detailed logging, and memory-safe batching.
    Returns WGS84 polygons.
    """
    t0 = time.time()

    # Load assets
    assets = _load_asset_objects(base_dir)
    if assets.empty:
        log_to_gui("[Mosaic] No asset objects found; mosaic skipped.", "WARN")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    log_to_gui(f"[Mosaic] Loaded assets: {len(assets):,} rows; CRS={assets.crs}")

    # Config / CRS
    cfg = read_config(base_dir / "system" / "config.ini")
    metric_crs = working_metric_crs_for(assets, cfg)

    # Parameters (with sensible defaults, overridable in config.ini)
    try:
        max_feats_per_tile = int(cfg["DEFAULT"].get("mosaic_quadtree_max_feats_per_tile", "800"))
    except Exception:
        max_feats_per_tile = 800
    try:
        max_depth = int(cfg["DEFAULT"].get("mosaic_quadtree_max_depth", "8"))
    except Exception:
        max_depth = 8
    try:
        # use grid_size_m as a minimum tile size if provided; 0 → no hard size stop
        min_tile_size_m = float(cfg["DEFAULT"].get("mosaic_quadtree_min_tile_m", str(grid_size_m)))
    except Exception:
        min_tile_size_m = max(0.0, float(grid_size_m))

    # Buffer & clean once in metric CRS
    t_buf0 = time.time()
    a = assets.to_crs(metric_crs).copy()
    buf = max(0.01, float(buffer_m))
    a["geometry"] = a.geometry.buffer(buf)
    a["geometry"] = a.geometry.apply(_fix_valid)
    a = a[a.geometry.notna() & ~a.geometry.is_empty]
    log_to_gui(f"[Mosaic] Buffered & cleaned: {len(a):,} geoms in {metric_crs}  "
               f"(took {time.time()-t_buf0:.2f}s, buffer={buf:g} m)")
    if a.empty:
        log_to_gui("[Mosaic] Buffered assets empty; mosaic skipped.", "WARN")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Spatial index (main process)
    try:
        sidx = a.sindex
        log_to_gui(f"[Mosaic] Built spatial index on {len(a):,} geoms.")
    except Exception:
        sidx = None
        log_to_gui("[Mosaic] Spatial index not available; aborting (required for quadtree).", "WARN")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Quadtree planning (skips empties)
    t_plan0 = time.time()
    minx, miny, maxx, maxy = a.total_bounds
    leaves = _plan_tiles_quadtree(
        a_metric=a,
        sidx=sidx,
        minx=minx, miny=miny, maxx=maxx, maxy=maxy,
        overlap_m=buf,
        max_feats_per_tile=max_feats_per_tile,
        max_depth=max_depth,
        min_tile_size_m=max(0.0, float(min_tile_size_m))
    )
    n_tiles = len(leaves)
    log_to_gui(f"[Mosaic] Planned quadtree tiles: {n_tiles:,}  "
               f"(max_feats_per_tile={max_feats_per_tile}, max_depth={max_depth}, "
               f"min_tile_size_m={min_tile_size_m:g})  "
               f"(planning took {time.time()-t_plan0:.2f}s)")

    if n_tiles == 0:
        log_to_gui("[Mosaic] No tiles intersect data; mosaic skipped.", "WARN")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Pack tasks → WKB (skip empties; already skipped in planner)
    t_pack0 = time.time()
    tasks: List[Tuple[int, List[bytes]]] = []
    counts = []
    for i, (_bounds, idxs) in enumerate(leaves):
        if not idxs:
            continue
        sub = a.iloc[idxs]
        counts.append(len(sub))
        wkb_list = [shp_wkb.dumps(g) for g in sub.geometry]
        tasks.append((i, wkb_list))

    n_tasks = len(tasks)
    if n_tasks == 0:
        log_to_gui("[Mosaic] Planner produced only empty tiles; mosaic skipped.", "WARN")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    def _pct(q):
        return int(np.percentile(counts, q)) if counts else 0

    log_to_gui(
        f"[Mosaic] Prepared {n_tasks:,} tile tasks; features per tile "
        f"(min/med/p95/max) = {min(counts) if counts else 0}/"
        f"{_pct(50)}/"
        f"{_pct(95)}/"
        f"{max(counts) if counts else 0}.  "
        f"(pack time {time.time()-t_pack0:.2f}s)"
    )

    # Decide workers
    if workers is None or workers <= 0:
        try:
            workers = max(1, mp.cpu_count())
        except Exception:
            workers = 4
    log_to_gui(f"[Mosaic] Parallel polygonize across {n_tasks:,} tiles; workers={workers}.")

    # Chunk size: reduce overhead for many tiles
    chunk = max(1, min(64, n_tasks // max(1, workers * 4)))

    # Process tiles
    t_poly0 = time.time()
    faces_wkb_all: List[bytes] = []
    processed = 0
    last_log = time.time()
    step = max(1, n_tasks // 10)  # log ~10% steps

    if workers == 1 or n_tasks == 1:
        for task in tasks:
            idx, res, err = _mosaic_tile_worker(task)
            processed += 1
            if err:
                log_to_gui(f"[Mosaic] Tile {idx+1}/{n_tasks} error: {err}", "WARN")
            else:
                faces_wkb_all.extend(res)
            if (processed % step == 0) or (time.time() - last_log >= 5) or processed == n_tasks:
                pct = processed * 100.0 / max(1, n_tasks)
                log_to_gui(f"[Mosaic] {processed}/{n_tasks} tiles done (~{pct:.1f}%) — "
                           f"faces so far: {len(faces_wkb_all):,}")
                last_log = time.time()
            update_progress(5 + processed * (80 / max(1, n_tasks)))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for idx, res, err in pool.imap_unordered(_mosaic_tile_worker, tasks, chunksize=chunk):
                processed += 1
                if err:
                    log_to_gui(f"[Mosaic] Tile {idx+1}/{n_tasks} error: {err}", "WARN")
                else:
                    faces_wkb_all.extend(res)
                if (processed % step == 0) or (time.time() - last_log >= 5) or processed == n_tasks:
                    pct = processed * 100.0 / max(1, n_tasks)
                    log_to_gui(f"[Mosaic] {processed}/{n_tasks} tiles done (~{pct:.1f}%) — "
                               f"faces so far: {len(faces_wkb_all):,}")
                    last_log = time.time()
                update_progress(5 + processed * (80 / max(1, n_tasks)))

    log_to_gui(f"[Mosaic] Polygonize stage finished: faces (pre-dedup) = {len(faces_wkb_all):,}  "
               f"(took {time.time()-t_poly0:.2f}s)")

    if not faces_wkb_all:
        log_to_gui("[Mosaic] Polygonize produced no faces.", "WARN")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # WKB→geom, dedup across overlaps by WKB hex
    t_dedup0 = time.time()
    faces = gpd.GeoSeries([shp_wkb.loads(b) for b in faces_wkb_all], crs=metric_crs)
    kept_before = len(faces)
    try:
        wkb_hex = faces.apply(_wkb_hex)
        keep = ~pd.Series(wkb_hex).duplicated()
        faces = faces[keep.values]
    except Exception:
        pass
    kept_after = len(faces)
    log_to_gui(f"[Mosaic] Dedup: kept {kept_after:,}, dropped {kept_before - kept_after:,}  "
               f"(took {time.time()-t_dedup0:.2f}s)")

    faces = faces.to_frame(name="geometry")
    faces = ensure_wgs84(gpd.GeoDataFrame(faces, geometry="geometry"))

    log_to_gui(f"[Mosaic] Completed in {time.time()-t0:.2f}s. Final faces: {len(faces):,}.")
    return faces


# -----------------------------------------------------------------------------
# Publish mosaic as geocode
# -----------------------------------------------------------------------------
def publish_mosaic_as_geocode(base_dir: Path, faces: gpd.GeoDataFrame) -> int:
    """Append a new geocode group + objects from mosaic faces directly to tbl_geocode_*."""
    if faces is None or faces.empty:
        log_to_gui("No mosaic faces to publish.", "WARN")
        return 0

    group_name = BASIC_MOSAIC_GROUP

    # Stable ordering → stable codes (use metric CRS to avoid warnings)
    try:
        cfg = read_config(base_dir / "system" / "config.ini")
        metric_crs = working_metric_crs_for(faces, cfg)
        cent = faces.to_crs(metric_crs).geometry.centroid
        order_idx = np.lexsort((cent.y.values, cent.x.values))
        faces = faces.iloc[order_idx].reset_index(drop=True)
    except Exception:
        faces = faces.reset_index(drop=True)

    prefix = group_name
    codes = [f"{prefix}_{i:06d}" for i in range(1, len(faces) + 1)]

    obj = faces.copy()
    obj["code"] = codes
    obj["name_gis_geocodegroup"] = group_name
    obj["attributes"] = None
    obj = obj[["code", "name_gis_geocodegroup", "attributes", "geometry"]]

    # Group bbox
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
# CLI runners
# -----------------------------------------------------------------------------
def run_h3_range(base_dir: Path, r_from: int, r_to: int):
    total = write_h3_levels(base_dir, list(range(r_from, r_to + 1)))
    log_to_gui(f"Total H3 cells added (range): {total:,}")
    update_progress(100)

def run_h3_levels(base_dir: Path, levels: List[int]):
    total = write_h3_levels(base_dir, levels)
    log_to_gui(f"Total H3 cells added (selected): {total:,}")
    update_progress(100)

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
# H3 helpers (UI)
# -----------------------------------------------------------------------------
def suggest_h3_levels_by_size(min_km: float, max_km: float) -> list[int]:
    """Return list of resolutions whose across-flat size lies within [min_km, max_km]."""
    out = []
    for res, size in H3_RES_ACROSS_FLATS_KM.items():
        if min_km <= size <= max_km:
            out.append(res)
    return out

def format_level_size_list(levels: list[int]) -> str:
    """Show levels with across-flats size in meters."""
    if not levels:
        return "(none)"
    return ", ".join(f"R{r} ({H3_RES_ACROSS_FLATS_M[r]:,.0f} m)" for r in levels)


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------
def build_gui(base: Path, cfg: configparser.ConfigParser):
    global root, log_widget, progress_var, progress_label, original_working_directory
    global mosaic_status_var, size_levels_var, generate_size_btn, suggest_levels_btn, suggested_h3_levels

    original_working_directory = str(base)
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

    # Diagnostics
    log_to_gui(f"Base dir: {base}")
    log_to_gui(f"GeoParquet out: {gpq_dir(base)}")
    log_to_gui("GeoParquet-first; outputs go ONLY to tbl_geocode_*.")
    log_to_gui(f"Mosaic geocode group is fixed to '{BASIC_MOSAIC_GROUP}'.")

    # --- H3 size-based selector ---
    size_frame = tk.LabelFrame(root, text="H3 size-based selection")
    size_frame.pack(padx=10, pady=(4,6), fill=tk.X)
    tk.Label(size_frame, text="Min m:").grid(row=0, column=0, padx=4, pady=2, sticky="e")
    tk.Label(size_frame, text="Max m:").grid(row=0, column=2, padx=4, pady=2, sticky="e")
    # Defaults: 50 m to 50 000 m
    min_var = tk.StringVar(value="50")
    max_var = tk.StringVar(value="50000")
    tk.Entry(size_frame, textvariable=min_var, width=10).grid(row=0, column=1, padx=4, pady=2)
    tk.Entry(size_frame, textvariable=max_var, width=10).grid(row=0, column=3, padx=4, pady=2)

    size_levels_var = tk.StringVar(value="(none)")
    tk.Label(size_frame, text="Matching levels:").grid(row=1, column=0, padx=4, pady=2, sticky="e")
    tk.Label(size_frame, textvariable=size_levels_var, anchor="w")\
        .grid(row=1, column=1, columnspan=3, padx=4, pady=2, sticky="w")

    def _suggest_levels():
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
        suggested_h3_levels[:] = levels  # mutate in place
        size_levels_var.set(format_level_size_list(levels))
        if levels:
            generate_size_btn.config(state="normal")
            log_to_gui(f"Suggested H3 levels (m range {min_m:g}–{max_m:g}): {levels}")
        else:
            generate_size_btn.config(state="disabled")
            log_to_gui("No H3 levels fit the specified size range (meters).", "WARN")

    def _generate_size_based():
        if not suggested_h3_levels:
            log_to_gui("No suggested levels to generate.", "WARN")
            return
        mp.get_start_method(allow_none=True)  # ensure spawn on Windows
        from threading import Thread
        Thread(target=run_h3_levels, args=(base, suggested_h3_levels.copy()), daemon=True).start()

    # Buttons (consistent width)
    btn_w = 18
    size_btn_frame = tk.Frame(size_frame)
    size_btn_frame.grid(row=2, column=0, columnspan=5, sticky="e", padx=4, pady=2)
    suggest_levels_btn = (ttk.Button(size_btn_frame, text="Suggest H3", width=btn_w, bootstyle=PRIMARY,
                                     command=_suggest_levels)
                          if ttk else tk.Button(size_btn_frame, text="Suggest H3", width=btn_w,
                                                command=_suggest_levels))
    generate_size_btn = (ttk.Button(size_btn_frame, text="Generate H3", width=btn_w, bootstyle=PRIMARY,
                                    command=_generate_size_based, state="disabled")
                         if ttk else tk.Button(size_btn_frame, text="Generate H3", width=btn_w, state="disabled",
                                               command=_generate_size_based))
    suggest_levels_btn.pack(side=tk.LEFT, padx=3)
    generate_size_btn.pack(side=tk.LEFT, padx=3)

    # --- Basic mosaic frame (separate) ---
    mosaic_frame = tk.LabelFrame(root, text="Basic mosaic")
    mosaic_frame.pack(padx=10, pady=(0,6), fill=tk.X)
    mosaic_status_var = tk.StringVar(value="")
    tk.Label(mosaic_frame, text="Status:").grid(row=0, column=0, padx=4, pady=2, sticky="w")
    mosaic_status_label = tk.Label(mosaic_frame, textvariable=mosaic_status_var, width=18, anchor="w")
    mosaic_status_label.grid(row=0, column=1, padx=(0,10), pady=2, sticky="w")
    mosaic_frame.grid_columnconfigure(1, weight=1)

    def _update_mosaic_status():
        exists = mosaic_exists(base)
        if mosaic_status_var.get() not in ("Running…","Completed","No faces"):
            mosaic_status_var.set("OK" if exists else "REQUIRED")
        color = "#55aa55" if exists else "#cc5555"
        try:
            mosaic_status_label.config(fg=color)
        except Exception:
            pass

    def _run_mosaic_inline():
        try:
            buf = float(cfg["DEFAULT"].get("mosaic_buffer_m","25"))
        except Exception:
            buf = 25.0
        try:
            grid = float(cfg["DEFAULT"].get("mosaic_grid_size_m","1000"))
        except Exception:
            grid = 1000.0
        mosaic_status_var.set("Running…")
        def _after(success):
            def _ui():
                _update_mosaic_status()
                if success:
                    mosaic_status_var.set("Completed")
                elif mosaic_status_var.get() == "Running…":
                    mosaic_status_var.set("No faces")
            try: root.after(100, _ui)
            except Exception: pass
        mp.get_start_method(allow_none=True)  # ensure spawn on Windows
        from threading import Thread
        Thread(target=run_mosaic, args=(base, buf, grid, _after), daemon=True).start()

    mosaic_btn = (ttk.Button(mosaic_frame, text="Build mosaic", width=18, bootstyle=PRIMARY, command=_run_mosaic_inline)
                  if ttk else tk.Button(mosaic_frame, text="Build mosaic", width=18, command=_run_mosaic_inline))
    mosaic_btn.grid(row=0, column=2, padx=4, pady=2, sticky="e")

    # Exit button
    exit_frame = tk.Frame(root); exit_frame.pack(fill=tk.X, pady=6, padx=10)
    (ttk.Button(exit_frame, text="Exit", bootstyle=WARNING, command=root.destroy) if ttk
     else tk.Button(exit_frame, text="Exit", command=root.destroy)).pack(side=tk.RIGHT)

    # Initial status
    _update_mosaic_status()
    root.mainloop()


# -----------------------------------------------------------------------------
# Entrypoint (single, clean)
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Create geocodes (H3/Mosaic)")
    parser.add_argument("--nogui", action="store_true", help="Run in CLI mode")
    parser.add_argument("--original_working_directory", required=False, help="Path to running folder")

    # H3 options (range)
    parser.add_argument("--h3", action="store_true", help="Generate H3 by range")
    parser.add_argument("--h3-from", dest="h3_from", type=int, default=3)
    parser.add_argument("--h3-to", dest="h3_to", type=int, default=6)

    # H3 options (explicit levels)
    parser.add_argument("--h3-levels", dest="h3_levels", type=str, default="",
                        help="Comma-separated list of H3 resolutions to generate, e.g. 5,6,7")

    # Mosaic options (always publishes to tbl_geocode_* with fixed name)
    parser.add_argument("--mosaic", action="store_true", help="Generate basic mosaic and publish as geocode")
    parser.add_argument("--buffer-m", dest="buffer_m", type=float, default=25.0)
    parser.add_argument("--grid-size-m", dest="grid_size_m", type=float, default=1000.0)

    args = parser.parse_args()
    base = resolve_base_dir(args.original_working_directory)
    cfg = read_config(base / "system" / "config.ini")

    # set for logfile even in CLI mode
    global original_working_directory
    original_working_directory = str(base)

    if args.nogui:
        if args.h3_levels:
            levels = [int(x.strip()) for x in args.h3_levels.split(",") if x.strip().isdigit()]
            run_h3_levels(base, levels)
        elif args.h3:
            run_h3_range(base, args.h3_from, args.h3_to)
        elif args.mosaic:
            run_mosaic(base, args.buffer_m, args.grid_size_m)
        else:
            log_to_gui("Nothing to do. Use --h3, --h3-levels, or --mosaic.", "WARN")
    else:
        build_gui(base, cfg)

if __name__ == "__main__":
    # On Windows, prefer spawn
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass
    main()
