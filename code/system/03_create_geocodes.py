#!/usr/bin/env python3
"""
H3 Geocode Generator – resilient across python-h3 versions
==========================================================

Converts *tbl_asset_group* geometries inside a GeoPackage into H3 hexagons
and appends them to *tbl_geocode_object*.

- Prefers H3 4.x high-level APIs:
    * geo_to_cells(geo, res)         ← accepts Shapely Polygon/MultiPolygon
    * cell_to_boundary(h, geo_json=True)
- Falls back to legacy 3.x:
    * polyfill(polygon_ring, res, geo_json=True)
    * h3_to_geo_boundary(h, geo_json=True)

CLI:
    python 03_create_qdgc.py --gpkg /path/to/mesa.gpkg
If you omit --gpkg it defaults to ../output/mesa.gpkg relative to this script.
"""
from __future__ import annotations

import argparse
import locale
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import geopandas as gpd
import h3
import pandas as pd  # ensure pandas is imported
from shapely.geometry import Polygon

# ----------------------------------------------------------------------
# H3 group management
# ----------------------------------------------------------------------
def purge_h3_groups_from_gpkg(gpkg: Path | str) -> int:
    """Remove all H3 groups from tbl_geocode_group; return number removed."""
    try:
        gdf = gpd.read_file(gpkg, layer="tbl_geocode_group")
        mask = gdf["name_gis_geocodegroup"].str.startswith("H3")
        to_drop = gdf[mask]
        if to_drop.empty:
            return 0
        kept = gdf[~mask]
        kept.to_file(gpkg, layer="tbl_geocode_group", driver="GPKG", mode="w")
        return len(to_drop)
    except Exception:
        return 0

def append_h3_group_to_gpkg(gpkg: Path | str, group_name: str, geom, res: int) -> int:
    """Append a new H3 group entry and return its generated ID."""
    gdf = gpd.read_file(gpkg, layer="tbl_geocode_group")
    ids = gdf["id"].astype(int) if not gdf.empty else []
    new_id = int(ids.max() + 1) if len(ids) else 1

    # Use 'geometry' as the column name to match GeoDataFrame expectations
    row = {
        "id": new_id,
        "name": group_name,
        "name_gis_geocodegroup": group_name,
        "title_user": f"H3 resolution {res}",
        "description": f"H3 hexagons at resolution {res}",
        "geometry": geom
    }
    # Create a single-row GeoDataFrame
    row_gdf = gpd.GeoDataFrame([row], geometry="geometry", crs=gdf.crs)

    # Concatenate and ensure only one geometry column exists
    new_gdf = pd.concat([gdf, row_gdf], ignore_index=True)
    new_gdf = gpd.GeoDataFrame(new_gdf, geometry="geometry", crs=gdf.crs)

    new_gdf.to_file(gpkg, layer="tbl_geocode_group", driver="GPKG", mode="w")
    return new_id

# ---------------------------------------------------------------------------
#  Locale – force English formatting for the log window
# ---------------------------------------------------------------------------
try:
    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
except locale.Error:
    pass

# ---------------------------------------------------------------------------
#  Geometry helpers
# ---------------------------------------------------------------------------

def to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure *gdf* is in EPSG:4326."""
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        raise ValueError("Layer has no CRS – cannot re-project.")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf


def read_asset_union(gpkg: Path | str):
    """Return dissolved asset geometry as a single (Multi)Polygon in WGS-84."""
    try:
        gdf = gpd.read_file(gpkg, layer="tbl_asset_group")
        if gdf.empty:
            return None
        gdf = to_wgs84(gdf)
        union_geom = (
            gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union
        )
        if union_geom.is_empty:
            return None
        return union_geom.buffer(0)
    except Exception as err:  # noqa: BLE001
        print(f"Error reading asset layer: {err}", file=sys.stderr)
        return None

# ---------------------------------------------------------------------------
#  H3 – small cross-version shim
# ---------------------------------------------------------------------------

def _cells_from_polygon(poly, res: int):
    """
    Return iterable of H3 indexes for a single Shapely Polygon using the best API.
    - On h3 4.x: use geo_to_cells(poly, res)
    - On h3 3.x: use polyfill(ring, res, geo_json=True) with (lon,lat) coords
    """
    ver = getattr(h3, "__version__", "")
    if ver.startswith("4."):
        # h3 4.x: accepts Shapely polygon directly (via __geo_interface__)
        return h3.geo_to_cells(poly, res)

    # Legacy 3.x fallback
    ring = list(poly.exterior.coords)           # (lon, lat)
    try:
        return h3.polyfill(ring, res, geo_json=True)
    except TypeError:
        # Older signatures may not accept the keyword; try positional flag
        return h3.polyfill(ring, res, True)


def _boundary_func():
    """
    Return a function that yields boundary coordinates in (lon, lat) order.
    Handles h3 4.x variants that:
      - accept geo_json kwarg,
      - accept geojson kwarg,
      - accept only a positional boolean,
      - or return (lat, lon) when no flag is provided.
    """
    ver = getattr(h3, "__version__", "")
    if ver.startswith("4."):
        def _b(idx):
            # 1) Try kwarg geo_json=True
            try:
                return h3.cell_to_boundary(idx, geo_json=True)
            except TypeError:
                pass
            # 2) Some wheels use 'geojson' (no underscore)
            try:
                return h3.cell_to_boundary(idx, geojson=True)  # type: ignore
            except TypeError:
                pass
            # 3) Positional flag (True means GeoJSON/lon-lat in most 4.x builds)
            try:
                return h3.cell_to_boundary(idx, True)
            except TypeError:
                pass
            # 4) No flag available → assume (lat, lon) and swap to (lon, lat)
            pts = h3.cell_to_boundary(idx)
            return [(p[1], p[0]) for p in pts]
        return _b

    # Legacy 3.x
    return lambda idx: h3.h3_to_geo_boundary(idx, geo_json=True)



_BOUNDARY = None


# ---------------------------------------------------------------------------
#  Convert geometry → hexagons
# ---------------------------------------------------------------------------

def h3_from_geom(geom, res: int):
    """Return a GeoDataFrame of H3 hexagons covering *geom* at resolution *res*."""
    if geom is None or geom.is_empty or not geom.is_valid:
        return gpd.GeoDataFrame(columns=["h3_index", "geometry"], geometry="geometry", crs="EPSG:4326")

    global _BOUNDARY  # pylint: disable=global-statement
    if _BOUNDARY is None:
        _BOUNDARY = _boundary_func()
    boundary_of = _BOUNDARY

    hexes: set[str] = set()
    polys = geom.geoms if getattr(geom, "geom_type", "") == "MultiPolygon" else [geom]

    for poly in polys:
        try:
            hexes |= set(_cells_from_polygon(poly, res))
        except Exception as err:
            print(f"polyfill R{res} failed: {err}", file=sys.stderr)
            continue

    if not hexes:
        return gpd.GeoDataFrame(columns=["h3_index", "geometry"], geometry="geometry", crs="EPSG:4326")

    rows = []
    for h in hexes:
        boundary = boundary_of(h)               # [(lon, lat), ...]
        polygon = Polygon(boundary)             # shapely expects (x,y)==(lon,lat)
        rows.append({"h3_index": h, "geometry": polygon})

    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


# ---------------------------------------------------------------------------
#  GeoPackage helpers
# ---------------------------------------------------------------------------

def append_h3_to_gpkg(gpkg: Path | str, hex_gdf: gpd.GeoDataFrame,
                      ref_group_id: int, group_name: str) -> int:
    """Append hexagons to tbl_geocode_object, return number written."""
    if hex_gdf.empty:
        return 0
    gdf = hex_gdf.copy()
    gdf["code"] = gdf["h3_index"]
    gdf["ref_geocodegroup"] = ref_group_id
    gdf["name_gis_geocodegroup"] = group_name
    gdf.to_file(gpkg, layer="tbl_geocode_object", driver="GPKG", mode="a")
    return len(gdf)


def purge_h3_from_gpkg(gpkg: Path | str) -> int:
    """Remove all rows with name_gis_geocodegroup == 'H3'."""
    gdf = gpd.read_file(gpkg, layer="tbl_geocode_object")
    to_drop = gdf[gdf["name_gis_geocodegroup"] == "H3"]
    if to_drop.empty:
        return 0
    keep = gdf[gdf["name_gis_geocodegroup"] != "H3"]
    keep.to_file(gpkg, layer="tbl_geocode_object", driver="GPKG", mode="w")
    return len(to_drop)


# ---------------------------------------------------------------------------
#  Tkinter GUI
# ---------------------------------------------------------------------------

def run_gui(gpkg: Path):
    root = tk.Tk()
    root.title("H3 Geocode Generator")
    root.geometry("560x420")

    union_geom = read_asset_union(gpkg)
    if union_geom is None:
        messagebox.showerror("Error", "tbl_asset_group missing or empty.")
        root.destroy()
        return

    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    ttk.Label(frame, text="H3 resolution range (coarse → fine)").pack(anchor="w")
    var_from, var_to = tk.IntVar(value=6), tk.IntVar(value=7)
    box_from = ttk.Combobox(frame, textvariable=var_from, values=list(range(6, 16)), state="readonly", width=5)
    box_to = ttk.Combobox(frame, textvariable=var_to, values=list(range(6, 16)), state="readonly", width=5)
    box_from.pack(anchor="w")
    ttk.Label(frame, text="to").pack(anchor="w")
    box_to.pack(anchor="w", pady=(0, 8))

    log = tk.Text(frame, height=12, width=65)
    log.pack(fill=tk.BOTH, expand=True)

    def log_msg(txt: str):
        log.insert(tk.END, txt + "\n")
        log.see(tk.END)

    def create():
        log.delete("1.0", tk.END)
        f, t = var_from.get(), var_to.get()
        if f > t:
            log_msg("Error: 'from' must be ≤ 'to'.")
            return
        # purge existing
        purge_h3_from_gpkg(gpkg)
        purge_h3_groups_from_gpkg(gpkg)
        total = 0
        for r in range(f, t + 1):
            log_msg(f"Generating R{r} …")
            group_name = f"H3_R{r}"
            gid = append_h3_group_to_gpkg(gpkg, group_name, union_geom, r)
            gdf = h3_from_geom(union_geom, r)
            n = append_h3_to_gpkg(gpkg, gdf, gid, group_name)
            log_msg(f"  {n} cells written.")
            total += n
        log_msg(f"✔ Finished – {total} cells added.")

    def purge():
        removed = purge_h3_from_gpkg(gpkg)
        log_msg(f"✖ Removed {removed} H3 cells.")

    btns = ttk.Frame(frame)
    btns.pack(fill=tk.X, pady=4)
    ttk.Button(btns, text="Create", command=create).pack(side=tk.LEFT, padx=4)
    ttk.Button(btns, text="Delete all H3", command=purge).pack(side=tk.LEFT, padx=4)
    ttk.Button(btns, text="Exit", command=root.destroy).pack(side=tk.RIGHT, padx=4)

    root.mainloop()


# ---------------------------------------------------------------------------
#  CLI entry-point
# ---------------------------------------------------------------------------

def cli():
    """Generate H3 hexagons for asset geometries in a GeoPackage."""
    # Quick capability check
    has_v4 = hasattr(h3, "geo_to_cells") and hasattr(h3, "cell_to_boundary")
    if not has_v4 and not hasattr(h3, "polyfill"):
        sys.exit("Error: No compatible H3 functions found. Install h3>=3.7")

    parser = argparse.ArgumentParser(
        description="Generate H3 geocodes for asset geometries in a GeoPackage."
    )
    # climb up from .../code/system → .../code → .../mesa, then into output/
    default_pkg = Path(__file__).resolve().parents[2] / "output" / "mesa.gpkg"

    parser.add_argument(
        "--gpkg", "-g", metavar="FILE",
        help="Path to GeoPackage",
        default=str(default_pkg)
    )
    parser.add_argument(
        "--from", "-f", dest="start_res", type=int,
        help="H3 resolution start (coarse)", default=6
    )
    parser.add_argument(
        "--to", "-t", dest="end_res", type=int,
        help="H3 resolution end (fine)", default=7
    )
    parser.add_argument(
        "--nogui", action="store_true",
        help="Run without launching the GUI"
    )
    args = parser.parse_args()

    gpkg = Path(args.gpkg)
    if not gpkg.exists():
        sys.exit(f"Error: GeoPackage not found: {gpkg}")

    # Read & buffer union of asset-group geometries
    union_geom = read_asset_union(gpkg)
    if union_geom is None:
        sys.exit("Error: tbl_asset_group missing or empty.")

    # GUI mode?
    if not args.nogui:
        run_gui(gpkg)
        return

    # CLI mode: loop resolutions
    if args.start_res > args.end_res:
        sys.exit("Error: --from must be <= --to")

    # Purge old H3 groups and objects
    purge_h3_from_gpkg(gpkg)
    purge_h3_groups_from_gpkg(gpkg)

    # Loop resolutions
    total = 0
    for res in range(args.start_res, args.end_res + 1):
        print(f"Generating H3 @ R{res} …")
        group_name = f"H3_R{res}"
        # create group entry
        group_id = append_h3_group_to_gpkg(gpkg, group_name, union_geom, res)
        # create and append objects referencing the new group
        hex_gdf = h3_from_geom(union_geom, res)
        n = append_h3_to_gpkg(gpkg, hex_gdf, group_id, group_name)
        print(f"  {n} cells written.")
        total += n
    print(f"Done – {total} cells added.")


if __name__ == "__main__":
    # Uncomment for quick env sanity check while debugging:
    # import sys as _sys; import h3 as _h3
    # print("python:", _sys.executable, "h3:", getattr(_h3, "__version__", "unknown"))
    cli()
    cli()
