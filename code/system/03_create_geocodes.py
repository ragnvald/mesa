#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create geocodes: H3 or Basic Mosaic (polygonize-based)
------------------------------------------------------

GUI (ttkbootstrap) with log panel + progress bar. Also supports CLI via --nogui.

- H3: robust across h3 v3/v4 (uses geo_to_cells with fallbacks).
- Mosaic: buffer → boundary unary_union → polygonize → robust coverage filter
  (validates polygons, avoids global union, and uses safe point-in-poly checks).
- Groups (tbl_geocode_group) store a SINGLE bounding-box Polygon (WGS84).
"""

from __future__ import annotations

import argparse
import configparser
import datetime
import locale
import os
import sys
import threading
from pathlib import Path
from typing import Tuple, List, Union, Optional

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import (
    Polygon, MultiPolygon, GeometryCollection, Point, box
)
from shapely.geometry import mapping as shp_mapping
from shapely.ops import unary_union, polygonize
from shapely.prepared import prep

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

# STRtree (shapely>=2)
try:
    from shapely.strtree import STRtree
except Exception:
    STRtree = None  # guarded

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
    cfg.read(file_name)
    return cfg

def resolve_base_dir(cli_root: str | None) -> Path:
    if cli_root:
        return Path(cli_root)
    base = Path(os.getcwd())
    if base.name.lower() == "system":
        return base.parent
    return base

def gpkg_path(base_dir: Path) -> Path:
    return base_dir / "output" / "mesa.gpkg"

def gpq_dir(base_dir: Path) -> Path:
    p = base_dir / "output" / "geoparquet"
    p.mkdir(parents=True, exist_ok=True)
    return p

def asset_object_parquet(base_dir: Path) -> Path:
    return base_dir / "output" / "geoparquet" / "tbl_asset_object.parquet"

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def update_progress(new_value: float):
    if root is None or progress_var is None or progress_label is None:
        return
    def task():
        v = max(0, min(100, float(new_value)))
        progress_var.set(v)
        progress_label.config(text=f"{int(v)}%")
    root.after(0, task)

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
        return gdf.crs.to_string()
    return "EPSG:3857"  # reasonable metric default

def ensure_gpkg_tables(gpkg: Path):
    import fiona
    layers = set()
    if gpkg.exists():
        try:
            layers = set(fiona.listlayers(gpkg))
        except Exception:
            layers = set()
    if "tbl_geocode_group" not in layers:
        g = gpd.GeoDataFrame(
            {"id": pd.Series(dtype="int"),
             "name": pd.Series(dtype="str"),
             "name_gis_geocodegroup": pd.Series(dtype="str"),
             "title_user": pd.Series(dtype="str"),
             "description": pd.Series(dtype="str")},
            geometry=gpd.GeoSeries([], dtype="geometry"),
            crs="EPSG:4326"
        )
        g.to_file(gpkg, layer="tbl_geocode_group", driver="GPKG")
    if "tbl_geocode_object" not in layers:
        o = gpd.GeoDataFrame(
            {"code": pd.Series(dtype="str"),
             "ref_geocodegroup": pd.Series(dtype="int"),
             "name_gis_geocodegroup": pd.Series(dtype="str"),
             "attributes": pd.Series(dtype="str")},
            geometry=gpd.GeoSeries([], dtype="geometry"),
            crs="EPSG:4326"
        )
        o.to_file(gpkg, layer="tbl_geocode_object", driver="GPKG")

def read_gpkg_layer(gpkg: Path, layer: str) -> gpd.GeoDataFrame:
    try:
        return gpd.read_file(gpkg, layer=layer)
    except Exception as e:
        log_to_gui(f"Failed reading layer {layer}: {e}", "WARN")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

def save_groups_objects_parquet(base_dir: Path):
    gpkg = gpkg_path(base_dir)
    geodir = gpq_dir(base_dir)
    try:
        g = read_gpkg_layer(gpkg, "tbl_geocode_group")
        o = read_gpkg_layer(gpkg, "tbl_geocode_object")
        ensure_wgs84(g).to_parquet(geodir / "tbl_geocode_group.parquet", index=False)
        ensure_wgs84(o).to_parquet(geodir / "tbl_geocode_object.parquet", index=False)
        log_to_gui("Updated GeoParquet mirrors for geocodes.")
    except Exception as e:
        log_to_gui(f"Parquet sync failed: {e}", "WARN")

def purge_group_and_objects(gpkg: Path, group_name: str) -> Tuple[int, int]:
    removed_g = removed_o = 0
    try:
        g = read_gpkg_layer(gpkg, "tbl_geocode_group")
        if not g.empty and "name_gis_geocodegroup" in g:
            keep = g[g["name_gis_geocodegroup"] != group_name]
            removed_g = len(g) - len(keep)
            keep = ensure_wgs84(keep)
            keep.to_file(gpkg, layer="tbl_geocode_group", driver="GPKG")
    except Exception as e:
        log_to_gui(f"Purge groups failed: {e}", "WARN")
    try:
        o = read_gpkg_layer(gpkg, "tbl_geocode_object")
        if not o.empty and "name_gis_geocodegroup" in o:
            keep = o[o["name_gis_geocodegroup"] != group_name]
            removed_o = len(o) - len(keep)
            keep = ensure_wgs84(keep)
            keep.to_file(gpkg, layer="tbl_geocode_object", driver="GPKG")
    except Exception as e:
        log_to_gui(f"Purge objects failed: {e}", "WARN")
    return removed_g, removed_o

# --- NEW: bbox helper --------------------------------------------------------
def _bbox_polygon_from(
    thing: Union[gpd.GeoDataFrame, gpd.GeoSeries, Polygon, MultiPolygon, GeometryCollection]
) -> Optional[Polygon]:
    """
    Produce a single bounding-box Polygon (WGS84) for the given geometry or Geo(Data)Frame.
    Returns None if bounds cannot be determined.
    """
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

def write_group(gpkg: Path, name: str, title: str, description: str, geom_bbox_wgs84: Polygon) -> int:
    """
    Write/append a single group row where geometry is a **bounding-box Polygon** in WGS84.
    """
    if geom_bbox_wgs84 is None or getattr(geom_bbox_wgs84, "is_empty", True):
        raise ValueError("write_group: bbox polygon is empty or None.")

    groups = read_gpkg_layer(gpkg, "tbl_geocode_group")
    next_id = int(groups["id"].max() + 1) if ("id" in groups.columns and not groups.empty) else 1

    row = {
        "id": next_id,
        "name": name,
        "name_gis_geocodegroup": name,
        "title_user": title,
        "description": description,
        "geometry": geom_bbox_wgs84,
    }
    newg = pd.concat([groups, gpd.GeoDataFrame([row], geometry="geometry", crs="EPSG:4326")], ignore_index=True)
    newg = ensure_wgs84(newg)
    newg.to_file(gpkg, layer="tbl_geocode_group", driver="GPKG")
    return next_id

def append_objects(gpkg: Path, objects_gdf: gpd.GeoDataFrame):
    objects_gdf = ensure_wgs84(objects_gdf)
    existing = read_gpkg_layer(gpkg, "tbl_geocode_object")

    # Align schemas
    for col in objects_gdf.columns:
        if col not in existing.columns:
            existing[col] = None
    for col in existing.columns:
        if col not in objects_gdf.columns:
            objects_gdf[col] = None

    merged = pd.concat([existing[objects_gdf.columns], objects_gdf], ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")
    merged.to_file(gpkg, layer="tbl_geocode_object", driver="GPKG")

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
        if g0.geom_type == "GeometryCollection":
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
    gj = shp_mapping(poly)  # proper GeoJSON (lon, lat)
    # --- v4 preferred ---
    if hasattr(h3, "geo_to_cells"):
        try:
            return set(h3.geo_to_cells(gj, res))
        except Exception as e:
            log_to_gui(f"H3 geo_to_cells failed (R{res}): {e}", "WARN")
    # --- v3 fallback ---
    if hasattr(h3, "polyfill"):
        try:
            return set(h3.polyfill(gj, res, True))
        except TypeError:
            pass
        for kw in ({"geo_json_conformant": True}, {"geojson_conformant": True}):
            try:
                return set(h3.polyfill(gj, res, **kw))
            except TypeError:
                continue
        log_to_gui(f"H3 polyfill failed (R{res}) with all signatures.", "WARN")
    raise RuntimeError("No compatible H3 polyfill signature worked.")

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


def union_from_asset_groups_or_objects(gpkg: Path):
    """
    Prefer union from tbl_asset_group if it has area geometry; else fall back to tbl_asset_object.
    Always returns WGS84 polygonal geometry or None.
    """
    g = read_gpkg_layer(gpkg, "tbl_asset_group")
    g = ensure_wgs84(g)
    g = g[g.geometry.notna()] if "geometry" in g else g
    union_geom = None
    if not g.empty and "geometry" in g:
        try:
            u = unary_union(g.geometry)
            u = _extract_polygonal(u)
            if u and not u.is_empty:
                union_geom = u
        except Exception:
            union_geom = None

    if union_geom is None:
        ao = read_gpkg_layer(gpkg, "tbl_asset_object")
        ao = ensure_wgs84(ao)
        if not ao.empty and "geometry" in ao:
            try:
                u = unary_union(ao.geometry)
                u = _extract_polygonal(u)
                if u and not u.is_empty:
                    union_geom = u
            except Exception:
                union_geom = None
    return union_geom

def purge_all_h3(gpkg: Path):
    try:
        groups = read_gpkg_layer(gpkg, "tbl_geocode_group")
        names = groups["name_gis_geocodegroup"].dropna().astype(str).tolist() if "name_gis_geocodegroup" in groups else []
        for nm in names:
            if nm.startswith("H3_R"):
                purge_group_and_objects(gpkg, nm)
    except Exception:
        pass

def write_h3_range(base_dir: Path, r_from: int, r_to: int) -> int:
    gpkg = gpkg_path(base_dir)
    ensure_gpkg_tables(gpkg)
    union_geom = union_from_asset_groups_or_objects(gpkg)
    if union_geom is None:
        raise RuntimeError("No polygonal geometry found in tbl_asset_group or tbl_asset_object.")

    purge_all_h3(gpkg)
    total = 0
    log_to_gui(f"H3 version: { _h3_version() }", "INFO")
    for r in range(r_from, r_to + 1):
        update_progress(5 + (r - r_from) * (80 / max(1, (r_to - r_from + 1))))
        group_name = f"H3_R{r}"

        # Group geometry = a single bbox polygon
        bbox_poly = _bbox_polygon_from(union_geom)
        if bbox_poly is None:
            raise RuntimeError("Failed to compute bbox for H3 group.")
        gid = write_group(
                gpkg,
                name=group_name,
                title=f"H3 resolution {r}",
                description=f"H3 hexagons at resolution {r}",
                geom_bbox_wgs84=bbox_poly,
        )

        gdf = h3_from_union(union_geom, r)
        if gdf.empty:
            log_to_gui(f"No H3 cells produced for resolution {r}.", "WARN")
            continue
        gdf = gdf.rename(columns={"h3_index": "code"})
        gdf["ref_geocodegroup"] = gid
        gdf["name_gis_geocodegroup"] = group_name
        if "attributes" not in gdf.columns:
            gdf["attributes"] = None
        gdf = gdf[["code", "ref_geocodegroup", "name_gis_geocodegroup", "attributes", "geometry"]]
        append_objects(gpkg, gdf)
        total += len(gdf)
        log_to_gui(f"H3 R{r}: wrote {len(gdf)} cells.")
    save_groups_objects_parquet(base_dir)
    return total

# -----------------------------------------------------------------------------
# Basic mosaic (polygonize-based)
# -----------------------------------------------------------------------------
def load_asset_objects(base_dir: Path) -> gpd.GeoDataFrame:
    pq = asset_object_parquet(base_dir)
    gpkg = gpkg_path(base_dir)
    if pq.exists():
        try:
            gdf = gpd.read_parquet(pq)
            if "geometry" in gdf and not gdf.empty:
                return gdf
        except Exception as e:
            log_to_gui(f"Failed reading {pq.name}: {e}", "WARN")
    try:
        return gpd.read_file(gpkg, layer="tbl_asset_object")
    except Exception as e:
        log_to_gui(f"Fallback read tbl_asset_object failed: {e}", "WARN")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

def _fix_valid(g):
    try:
        if shapely_make_valid is not None:
            gg = shapely_make_valid(g)
            if gg is not None and not gg.is_empty:
                return gg
    except Exception:
        pass
    try:
        return g.buffer(0)
    except Exception:
        return g

def explode_polygons(geoms: List) -> List[Polygon]:
    out = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        gt = g.geom_type
        if gt == "Polygon":
            out.append(g)
        elif gt == "MultiPolygon":
            out.extend(list(g.geoms))
        elif gt == "GeometryCollection":
            for sub in g.geoms:
                if sub.geom_type == "Polygon":
                    out.append(sub)
                elif sub.geom_type == "MultiPolygon":
                    out.extend(list(sub.geoms))
    return out

def _safe_covers(poly: Polygon, pt: Point) -> bool:
    """Robust covers test with fallbacks to avoid GEOS topology exceptions."""
    try:
        return poly.covers(pt)
    except Exception:
        try:
            pp = _fix_valid(poly)
            return pp.covers(pt)
        except Exception:
            try:
                return poly.contains(pt)
            except Exception:
                try:
                    pp = _fix_valid(poly)
                    return pp.contains(pt)
                except Exception:
                    return False

def _covered_faces_via_strtree(originals: List[Polygon], faces: List[Polygon]) -> List[Polygon]:
    """
    STRtree-based coverage without query predicate (robust).
    Query by bbox first, then do safe point-in-poly checks.
    """
    if not originals or STRtree is None:
        return []
    tree = STRtree(originals)
    covered = []

    for f in faces:
        try:
            pt = f.representative_point()
        except Exception:
            # Fall back to centroid if needed
            try:
                pt = f.centroid
            except Exception:
                continue

        try:
            cands = tree.query(pt)  # bbox candidates, no predicate
        except Exception:
            cands = []

        keep = False
        if isinstance(cands, np.ndarray):
            # cands are indices
            for idx in cands.tolist():
                geom = originals[int(idx)]
                if _safe_covers(geom, pt):
                    keep = True
                    break
        else:
            # cands might be geometries already
            for geom in cands:
                if isinstance(geom, (int, np.integer)):
                    geom = originals[int(geom)]
                if _safe_covers(geom, pt):
                    keep = True
                    break

        if keep:
            covered.append(f)

    return covered

def _covered_faces_via_union(originals: List[Polygon], faces: List[Polygon]) -> List[Polygon]:
    """
    Fallback: build a single prepared union and test representative points.
    More robust; cost is a single unary_union + cheap point tests.
    """
    if not originals:
        return []
    try:
        # Validate originals before union to avoid topology errors
        originals_valid = [_fix_valid(p) for p in originals]
        mask = unary_union(originals_valid)
        if mask.is_empty:
            return []
        pmask = prep(mask)
    except Exception as e:
        log_to_gui(f"Prepared-union fallback failed to build: {e}", "WARN")
        return []

    out = []
    for f in faces:
        try:
            pt = f.representative_point()
        except Exception:
            try:
                pt = f.centroid
            except Exception:
                continue
        try:
            if pmask.covers(pt):
                out.append(f)
        except Exception:
            try:
                if mask.covers(pt):
                    out.append(f)
            except Exception:
                pass
    return out

def build_basic_mosaic_polygonize(base_dir: Path, cfg: configparser.ConfigParser,
                                  line_buffer_m: float, point_buffer_m: float):
    """Polygonize-based mosaic (fast): buffer → edges → polygonize → robust coverage filter."""
    assets = load_asset_objects(base_dir)
    if assets.empty or "geometry" not in assets:
        return None, gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    # Honor process flag if present
    if "process" in assets.columns:
        assets = assets[assets["process"].fillna(True)]

    metric_crs = working_metric_crs_for(assets, cfg)
    if assets.crs is None:
        assets = assets.set_crs("EPSG:4326", allow_override=True)
    if assets.crs.to_string() != metric_crs:
        assets = assets.to_crs(metric_crs)

    gt = assets.geometry.geom_type
    pts   = assets[gt.isin(["Point", "MultiPoint"])].copy()
    lns   = assets[gt.isin(["LineString", "MultiLineString"])].copy()
    polys = assets[gt.isin(["Polygon", "MultiPolygon", "GeometryCollection"])].copy()

    pieces = []

    if not pts.empty:
        buf_p = max(0.01, float(point_buffer_m))
        log_to_gui(f"Buffering {len(pts)} points by {buf_p} m …")
        pts["geometry"] = pts.buffer(buf_p).apply(_fix_valid)
        pieces.append(pts.geometry)

    if not lns.empty:
        buf_l = max(0.01, float(line_buffer_m))
        log_to_gui(f"Buffering {len(lns)} lines by {buf_l} m …")
        lns["geometry"] = lns.buffer(buf_l).apply(_fix_valid)
        pieces.append(lns.geometry)

    if not polys.empty:
        log_to_gui(f"Fixing validity for {len(polys)} polygons …")
        polys["geometry"] = polys.geometry.apply(_fix_valid)
        pieces.append(polys.geometry)

    if not pieces:
        return None, gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    update_progress(35)
    all_polys_series = gpd.GeoSeries(pd.concat(pieces, ignore_index=True), crs=metric_crs)
    all_polys_series = all_polys_series[~all_polys_series.is_empty]

    # Additional defensive validity on the merged set
    all_polys_series = all_polys_series.apply(_fix_valid)
    all_polys_series = all_polys_series[~all_polys_series.is_empty]

    # --- polygonize approach ---
    log_to_gui("Building edge network (unary_union of boundaries) …")
    try:
        boundaries = [g.boundary for g in all_polys_series.geometry]
        edge_net = unary_union(boundaries)
    except Exception as e:
        log_to_gui(f"Boundary union failed: {e}", "ERROR")
        return None, gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    update_progress(50)
    log_to_gui("Polygonizing edge network into atomic faces …")
    try:
        faces = list(polygonize(edge_net))
    except Exception as e:
        log_to_gui(f"Polygonize failed: {e}", "ERROR")
        return None, gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    if not faces:
        log_to_gui("Polygonize produced no faces.", "WARN")
        return None, gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    update_progress(60)
    log_to_gui(f"Polygonize produced {len(faces)} candidate faces. Filtering to covered faces …")

    originals = explode_polygons(list(all_polys_series.geometry))
    # Validate originals again (cheap if already valid)
    originals = [_fix_valid(p) for p in originals if p and not p.is_empty]

    # 1) Try STRtree-based coverage (robust, no predicate)
    covered = _covered_faces_via_strtree(originals, faces)

    # 2) If nothing survived, robust fallback: prepared union
    if not covered:
        log_to_gui("STRtree coverage returned 0 faces; falling back to prepared-union coverage …", "WARN")
        covered = _covered_faces_via_union(originals, faces)

    if not covered:
        log_to_gui("No covered faces remain after robust filtering.", "WARN")
        return None, gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    update_progress(75)
    gdf_metric = gpd.GeoDataFrame([{"geometry": p} for p in covered], geometry="geometry", crs=metric_crs)
    gdf_wgs = gdf_metric.to_crs(4326)

    # We no longer build a global union (avoids topology errors and heavy ops).
    return None, gdf_wgs

def write_basic_mosaic(base_dir: Path, _union_geom_wgs_unused, polys_wgs: gpd.GeoDataFrame) -> int:
    gpkg = gpkg_path(base_dir)
    ensure_gpkg_tables(gpkg)

    purge_group_and_objects(gpkg, "basic_mosaic")

    # Group geometry = bbox over mosaic faces
    bbox_poly = _bbox_polygon_from(polys_wgs)
    if bbox_poly is None:
        raise RuntimeError("Failed to compute bbox for basic_mosaic group.")

    gid = write_group(
        gpkg,
        name="basic_mosaic",
        title="Basic mosaic",
        description="Unique polygons representing all overlapping polygons in the assets. Make sure the objects refer to the right geocode group.",
        geom_bbox_wgs84=bbox_poly,
    )

    out = polys_wgs.copy()
    out["code"] = [f"MOSAIC_{i+1:06d}" for i in range(len(out))]
    out["ref_geocodegroup"] = gid
    out["name_gis_geocodegroup"] = "basic_mosaic"
    if "attributes" not in out.columns:
        out["attributes"] = None
    out = out[["code", "ref_geocodegroup", "name_gis_geocodegroup", "attributes", "geometry"]]
    append_objects(gpkg, out)
    save_groups_objects_parquet(base_dir)
    return len(out)

# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------
def start_h3_thread(base_dir: Path, r_from: int, r_to: int):
    def worker():
        try:
            if h3 is None:
                log_to_gui("Package 'h3' is not installed; H3 mode unavailable.", "WARN")
                update_progress(0)
                return
            log_to_gui(f"Creating H3 cells R{r_from}..R{r_to} …")
            update_progress(3)
            n = write_h3_range(base_dir, r_from, r_to)
            update_progress(100)
            log_to_gui(f"COMPLETED: wrote {n} H3 cells.")
        except Exception as e:
            log_to_gui(f"ERROR in H3 generation: {e}", "ERROR")
            update_progress(0)
    threading.Thread(target=worker, daemon=True).start()

def start_mosaic_thread(base_dir: Path, cfg: configparser.ConfigParser, line_m: float, point_m: float):
    def worker():
        try:
            log_to_gui(f"Building Basic mosaic (line buffer {line_m} m, point buffer {point_m} m) …")
            update_progress(5)
            union_wgs_unused, polys = build_basic_mosaic_polygonize(base_dir, cfg, line_m, point_m)
            if polys.empty:
                log_to_gui("No geometry produced for mosaic.", "WARN")
                update_progress(0)
                return
            update_progress(90)
            n = write_basic_mosaic(base_dir, union_wgs_unused, polys)
            update_progress(100)
            log_to_gui(f"COMPLETED: Basic mosaic wrote {n} polygons.")
        except Exception as e:
            log_to_gui(f"ERROR in mosaic generation: {e}", "ERROR")
            update_progress(0)
    threading.Thread(target=worker, daemon=True).start()

def run_gui(base_dir: Path, cfg: configparser.ConfigParser):
    global root, log_widget, progress_var, progress_label, original_working_directory
    original_working_directory = str(base_dir)

    root = ttk.Window(themename=cfg["DEFAULT"].get("ttk_bootstrap_theme", "flatly"))
    root.title("Create geocodes")
    try:
        root.iconbitmap(os.path.join(original_working_directory, "system_resources/mesa.ico"))
    except Exception:
        pass

    log_frame = ttk.LabelFrame(root, text="Log output", bootstyle="info")
    log_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    log_widget = scrolledtext.ScrolledText(log_frame, height=12)
    log_widget.pack(fill=tk.BOTH, expand=True)

    pf = tk.Frame(root); pf.pack(pady=6)
    progress_var = tk.DoubleVar(value=0)
    pb = ttk.Progressbar(pf, orient="horizontal", length=260, mode="determinate",
                         variable=progress_var, bootstyle='info')
    pb.pack(side=tk.LEFT)
    progress_label = tk.Label(pf, text="0%", bg="light grey"); progress_label.pack(side=tk.LEFT, padx=6)

    info_text = (
        "Choose H3 hexagons or a Basic mosaic built from all asset objects.\n"
        "Mosaic uses polygonize + robust coverage filter. Group geometries are bbox (WGS84)."
    )
    ttk.Label(root, text=info_text, wraplength=680, justify="left").pack(padx=10, pady=8)

    controls = ttk.Frame(root); controls.pack(padx=10, pady=4, fill=tk.X)

    # H3
    h3_box = ttk.Labelframe(controls, text="H3", bootstyle=INFO)
    h3_box.grid(row=0, column=0, sticky="ew", padx=(0, 8))
    ttk.Label(h3_box, text="Resolution from").grid(row=0, column=0, padx=6, pady=6, sticky="w")
    res_from = tk.IntVar(value=6); res_to = tk.IntVar(value=7)
    cb_from = ttk.Combobox(h3_box, textvariable=res_from, values=list(range(3, 16)), width=6, state="readonly")
    cb_to   = ttk.Combobox(h3_box, textvariable=res_to,   values=list(range(3, 16)), width=6, state="readonly")
    cb_from.grid(row=0, column=1, padx=4, pady=6); ttk.Label(h3_box, text="to").grid(row=0, column=2, padx=2, pady=6)
    cb_to.grid(row=0, column=3, padx=4, pady=6)
    ttk.Button(h3_box, text="Create H3", bootstyle=PRIMARY,
               command=lambda: start_h3_thread(base_dir, res_from.get(), res_to.get())).grid(row=0, column=4, padx=8, pady=6)

    # Mosaic
    m_box = ttk.Labelframe(controls, text="Basic mosaic", bootstyle=INFO)
    m_box.grid(row=0, column=1, sticky="ew", padx=(8, 0))
    ttk.Label(m_box, text="Line buffer (m)").grid(row=0, column=0, padx=6, pady=6, sticky="w")
    line_buf = tk.DoubleVar(value=25.0); ttk.Entry(m_box, textvariable=line_buf, width=10).grid(row=0, column=1, padx=4, pady=6)
    ttk.Label(m_box, text="Point buffer (m)").grid(row=0, column=2, padx=6, pady=6, sticky="w")
    point_buf = tk.DoubleVar(value=10.0); ttk.Entry(m_box, textvariable=point_buf, width=10).grid(row=0, column=3, padx=4, pady=6)
    ttk.Button(m_box, text="Create mosaic", bootstyle=PRIMARY,
               command=lambda: start_mosaic_thread(base_dir, cfg, line_buf.get(), point_buf.get())).grid(row=0, column=4, padx=8, pady=6)

    btns = ttk.Frame(root); btns.pack(pady=8, fill=tk.X)
    ttk.Button(btns, text="Exit", bootstyle=WARNING, command=root.destroy).pack(side=tk.RIGHT, padx=6)

    log_to_gui("Ready. Pick a mode and run.")
    root.mainloop()

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Create geocodes (H3 or polygonize-based Basic mosaic)")
    parser.add_argument("--nogui", action="store_true", help="Run without GUI")
    parser.add_argument("--original_working_directory", "-o", default=None, help="Root folder")
    parser.add_argument("--mode", choices=["h3", "mosaic"], default="h3")
    parser.add_argument("--from", "-f", dest="start_res", type=int, default=6, help="H3: start resolution")
    parser.add_argument("--to", "-t", dest="end_res", type=int, default=7, help="H3: end resolution")
    parser.add_argument("--line_buffer_m", type=float, default=25.0, help="Mosaic: line buffer (m)")
    parser.add_argument("--point_buffer_m", type=float, default=10.0, help="Mosaic: point buffer (m)")
    args = parser.parse_args()

    base = resolve_base_dir(args.original_working_directory)
    cfg = read_config(base / "system" / "config.ini")

    if not args.nogui:
        if ttk is None:
            print("ttkbootstrap not available; run with --nogui or install ttkbootstrap.")
            sys.exit(1)
        run_gui(base, cfg); return

    global original_working_directory
    original_working_directory = str(base)

    gpkg = gpkg_path(base); ensure_gpkg_tables(gpkg)

    if args.mode == "h3":
        if h3 is None:
            print("ERROR: h3 not installed (pip install h3)"); sys.exit(2)
        if args.start_res > args.end_res:
            print("ERROR: --from must be <= --to"); sys.exit(3)
        n = write_h3_range(base, args.start_res, args.end_res)
        print(f"Done — wrote {n} H3 cells.")
    else:
        union_wgs_unused, polys = build_basic_mosaic_polygonize(base, cfg, args.line_buffer_m, args.point_buffer_m)
        if polys.empty:
            print("No geometry produced for mosaic."); sys.exit(4)
        n = write_basic_mosaic(base, union_wgs_unused, polys)
        print(f"Done — wrote {n} mosaic polygons.")

if __name__ == "__main__":
    main()
