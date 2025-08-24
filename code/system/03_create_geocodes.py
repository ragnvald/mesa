#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_create_geocodes.py

Create geocodes: H3 or Basic Mosaic (polygonize-based)
------------------------------------------------------

- GUI (ttkbootstrap) with log panel + progress bar. Also supports CLI via --nogui.
- H3: robust across h3 v3/v4 (uses geo_to_cells with fallbacks).
- Mosaic: buffer → per-feature boundary union → polygonize (tiling-aware).
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
- h3_max_cells           = 1200000
- h3_from = 3
- h3_to   = 6
- mosaic_buffer_m   = 25
- mosaic_grid_size_m = 1000
- ttk_bootstrap_theme = flatly
"""

from __future__ import annotations

import argparse
import configparser
import datetime
import locale
import os
import threading
from pathlib import Path
from typing import Tuple, Union, Optional, List

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


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
BASIC_MOSAIC_GROUP = "basic_mosaic"


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
root = None
log_widget = None
progress_var = None
progress_label = None
original_working_directory = None


# -----------------------------------------------------------------------------
# Config & paths
# -----------------------------------------------------------------------------
def read_config(file_name: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(file_name, encoding="utf-8")
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

    # 1) Try GeoParquet groups (bbox polygons from importer)
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
# Approx average hex areas by resolution (km^2), order-of-magnitude
AVG_HEX_AREA_KM2 = {
    0: 4259705, 1: 608529, 2: 86932, 3: 12419, 4: 1774,
    5: 253, 6: 36.1, 7: 5.15, 8: 0.736, 9: 0.105,
    10: 0.0150, 11: 0.00214, 12: 0.000305, 13: 0.0000436, 14: 0.00000623, 15: 0.00000089
}

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
# Geocode (H3 & Mosaic) writers — APPEND/MERGE semantics
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
# Mosaic (polygonize-based) — tiling-aware → publish directly into geocode tables
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

def _explode_lines(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        out = []
        for g in geom.geoms:
            out.extend(_explode_lines(g))
        return out
    if isinstance(geom, (Polygon, MultiPolygon)):
        return _explode_lines(geom.boundary)
    return []

def _generate_grid(bounds: tuple[float,float,float,float], size_m: float) -> List[Polygon]:
    minx, miny, maxx, maxy = bounds
    if size_m <= 0 or maxx <= minx or maxy <= miny:
        return [box(minx, miny, maxx, maxy)]
    xs = np.arange(minx, maxx, size_m)
    ys = np.arange(miny, maxy, size_m)
    tiles = []
    for x in xs:
        for y in ys:
            xr = min(x + size_m, maxx)
            yr = min(y + size_m, maxy)
            tiles.append(box(x, y, xr, yr))
    return tiles

def mosaic_faces_from_assets(base_dir: Path, buffer_m: float, grid_size_m: float) -> gpd.GeoDataFrame:
    """Return polygon faces only (no edges), WGS84."""
    assets = _load_asset_objects(base_dir)
    if assets.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    cfg = read_config(base_dir / "system" / "config.ini")
    metric = working_metric_crs_for(assets, cfg)

    a = assets.to_crs(metric)
    buf = max(0.01, float(buffer_m))
    a["geometry"] = a.geometry.buffer(buf)
    a["geometry"] = a.geometry.apply(_fix_valid)
    a = a[a.geometry.notna() & ~a.geometry.is_empty]
    if a.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Spatial index if available
    try:
        sidx = a.sindex
    except Exception:
        sidx = None

    bounds = a.total_bounds
    tiles = _generate_grid(tuple(bounds), float(grid_size_m))
    overlap = buf  # reduce seams

    faces_parts: List[gpd.GeoDataFrame] = []

    for i, tpoly in enumerate(tiles):
        tpoly_ov = tpoly.buffer(overlap)
        if sidx:
            idx = list(sidx.query(tpoly_ov, predicate="intersects"))
            sub = a.iloc[idx] if idx else a.iloc[[]]
        else:
            sub = a[a.intersects(tpoly_ov)]
        if sub.empty:
            continue

        try:
            boundaries = [g.boundary for g in sub.geometry if g is not None and not g.is_empty]
            if not boundaries:
                continue
            edge_net = unary_union(boundaries)
        except Exception as e:
            log_to_gui(f"[Mosaic] Tile {i+1}/{len(tiles)} union failed: {e}", "WARN")
            continue

        try:
            faces_list = list(polygonize(edge_net))
        except Exception as e:
            log_to_gui(f"[Mosaic] Tile {i+1}/{len(tiles)} polygonize failed: {e}", "WARN")
            continue

        if faces_list:
            faces_list = [p for p in faces_list if isinstance(p, (Polygon, MultiPolygon)) and not p.is_empty]
            if faces_list:
                faces_parts.append(gpd.GeoDataFrame({"geometry": faces_list}, geometry="geometry", crs=a.crs))

        update_progress(5 + (i + 1) * (80 / max(1, len(tiles))))

    if not faces_parts:
        log_to_gui("Polygonize produced no faces.", "WARN")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    faces = pd.concat(faces_parts, ignore_index=True)

    # Dedup across overlaps by WKB
    try:
        faces["__wkb__"] = faces.geometry.apply(_wkb_hex)
        faces = faces.drop_duplicates(subset="__wkb__").drop(columns="__wkb__", errors="ignore")
    except Exception:
        pass

    faces = gpd.GeoDataFrame(faces, geometry="geometry", crs=a.crs)
    faces = ensure_wgs84(faces)
    return faces

def publish_mosaic_as_geocode(base_dir: Path, faces: gpd.GeoDataFrame) -> int:
    """Append a new geocode group + objects from mosaic faces directly to tbl_geocode_*."""
    if faces is None or faces.empty:
        log_to_gui("No mosaic faces to publish.", "WARN")
        return 0

    group_name = BASIC_MOSAIC_GROUP

    # Stable ordering → stable codes
    try:
        cent = faces.geometry.centroid
        order_idx = np.lexsort((cent.y.values, cent.x.values))
        faces = faces.iloc[order_idx].reset_index(drop=True)
    except Exception:
        faces = faces.reset_index(drop=True)

    # Codes unique across table by prefixing group name
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
# CLI / GUI glue
# -----------------------------------------------------------------------------
def run_h3_range(base_dir: Path, r_from: int, r_to: int):
    total = write_h3_levels(base_dir, list(range(r_from, r_to + 1)))
    log_to_gui(f"Total H3 cells added (range): {total:,}")
    update_progress(100)

def run_h3_levels(base_dir: Path, levels: List[int]):
    total = write_h3_levels(base_dir, levels)
    log_to_gui(f"Total H3 cells added (selected): {total:,}")
    update_progress(100)

def run_mosaic(base_dir: Path, buffer_m: float, grid_size_m: float):
    faces = mosaic_faces_from_assets(base_dir, buffer_m, grid_size_m)
    if faces.empty:
        log_to_gui("Mosaic produced no faces to publish.", "WARN")
        update_progress(100)
        return
    n = publish_mosaic_as_geocode(base_dir, faces)
    log_to_gui(f"Mosaic published as geocode group '{BASIC_MOSAIC_GROUP}' with {n:,} objects.")
    update_progress(100)

def main_cli(args):
    base = resolve_base_dir(args.original_working_directory)
    if args.h3_levels:
        levels = [int(x.strip()) for x in args.h3_levels.split(",") if x.strip().isdigit()]
        run_h3_levels(base, levels)
    elif args.h3:
        run_h3_range(base, args.h3_from, args.h3_to)
    elif args.mosaic:
        run_mosaic(base, args.buffer_m, args.grid_size_m)
    else:
        log_to_gui("Nothing to do. Use --h3, --h3-levels, or --mosaic.", "WARN")

def build_gui(base: Path, cfg: configparser.ConfigParser):
    global root, log_widget, progress_var, progress_label, original_working_directory
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

    # --- H3 level selectors (checkboxes 0..15)
    levels_frame = tk.LabelFrame(root, text="H3 levels (pick any)")
    levels_frame.pack(padx=10, pady=(4,6), fill=tk.X)

    h3_vars = [tk.IntVar(value=0) for _ in range(16)]  # 0..15
    for r in range(16):
        cb = (ttk.Checkbutton(levels_frame, text=str(r), variable=h3_vars[r])
              if ttk else tk.Checkbutton(levels_frame, text=str(r), variable=h3_vars[r]))
        cb.grid(row=0 + (r // 8), column=r % 8, padx=4, pady=2, sticky="w")

    def _selected_levels():
        return [r for r, v in enumerate(h3_vars) if v.get() == 1]

    # --- Mosaic info (no options for name)
    mosaic_frame = tk.LabelFrame(root, text="Basic mosaic (writes to tbl_geocode_*)")
    mosaic_frame.pack(padx=10, pady=(4,6), fill=tk.X)
    tk.Label(mosaic_frame, text=f"Group name: {BASIC_MOSAIC_GROUP}").grid(row=0, column=0, padx=6, pady=4, sticky="w")
    tk.Label(mosaic_frame, text="(Buffer & tiling from config.ini)").grid(row=0, column=1, padx=6, pady=4, sticky="w")

    # Buttons
    btns = tk.Frame(root); btns.pack(pady=8)

    def _run_h3_range_btn():
        r_from = int(cfg["DEFAULT"].get("h3_from","3"))
        r_to   = int(cfg["DEFAULT"].get("h3_to","6"))
        threading.Thread(target=run_h3_range, args=(base, r_from, r_to), daemon=True).start()

    def _run_h3_selected_btn():
        lvls = _selected_levels()
        if not lvls:
            log_to_gui("Select at least one H3 level.", "WARN")
            return
        threading.Thread(target=run_h3_levels, args=(base, lvls), daemon=True).start()

    def _run_mosaic_btn():
        buf = float(cfg["DEFAULT"].get("mosaic_buffer_m","25"))
        grid = float(cfg["DEFAULT"].get("mosaic_grid_size_m","1000"))
        threading.Thread(target=run_mosaic, args=(base, buf, grid), daemon=True).start()

    (ttk.Button(btns, text="Generate H3 (range)", bootstyle=PRIMARY, command=_run_h3_range_btn) if ttk
     else tk.Button(btns, text="Generate H3 (range)", command=_run_h3_range_btn)).grid(row=0, column=0, padx=8, pady=4)

    (ttk.Button(btns, text="Generate H3 (selected)", bootstyle=PRIMARY, command=_run_h3_selected_btn) if ttk
     else tk.Button(btns, text="Generate H3 (selected)", command=_run_h3_selected_btn)).grid(row=0, column=1, padx=8, pady=4)

    (ttk.Button(btns, text="Build mosaic", bootstyle=PRIMARY, command=_run_mosaic_btn) if ttk
     else tk.Button(btns, text="Build mosaic", command=_run_mosaic_btn)).grid(row=0, column=2, padx=8, pady=4)

    (ttk.Button(btns, text="Exit", bootstyle=WARNING, command=root.destroy) if ttk
     else tk.Button(btns, text="Exit", command=root.destroy)).grid(row=0, column=3, padx=8, pady=4)

    root.mainloop()


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
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
