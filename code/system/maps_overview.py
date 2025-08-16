#!/usr/bin/env python3
# Maps Overview — dynamic basemap refresh on pan/zoom, fills pane, A–E only, 10% chart border, clean exit

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

import os
import sys
import configparser
import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

import contextily as ctx
from ttkbootstrap.constants import WARNING

# ----------------------------------------------------------
# Small cache of solid-colour squares for the stats table
# ----------------------------------------------------------
_color_icons = {}  # (hex, size) -> PhotoImage
def get_color_icon(color_hex: str, size: int = 20) -> tk.PhotoImage:
    key = (color_hex, size)
    if key not in _color_icons:
        img = tk.PhotoImage(width=size, height=size)
        img.put(color_hex, to=(0, 0, size, size))
        _color_icons[key] = img
    return _color_icons[key]

# ----------------------------------------------------------
# Config helpers
# ----------------------------------------------------------
def read_config(path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8")
    return cfg

def _safe_rgba(col: str, fallback: str) -> str:
    try:
        to_rgba(col)
        return col
    except Exception:
        return fallback

def get_color_mapping(cfg: configparser.ConfigParser) -> dict:
    fallback = cfg["DEFAULT"].get("category_colour_unknown", "#BDBDBD").strip()
    fallback = _safe_rgba(fallback, "#BDBDBD")
    colour_map = {}
    for sec in ("A", "B", "C", "D", "E"):
        if sec in cfg:
            raw = cfg[sec].get("category_colour", "").strip()
            colour_map[sec] = _safe_rgba(raw or fallback, fallback)
    return colour_map

def get_descriptions(cfg: configparser.ConfigParser) -> dict:
    return {sec: cfg[sec].get("description", "") for sec in ("A","B","C","D","E") if sec in cfg}

# ----------------------------------------------------------
# I/O
# ----------------------------------------------------------
def load_geoparquet(path: str) -> gpd.GeoDataFrame:
    try:
        return gpd.read_parquet(path)
    except Exception as exc:
        print("Error reading geoparquet:", exc, file=sys.stderr)
        return gpd.GeoDataFrame()

# ----------------------------------------------------------
# Globals set at runtime
# ----------------------------------------------------------
config = None
tbl_flat_data = None
colour_map = None
desc_map = None
FULL_EXTENT = None

root = None
fig = None
ax = None
canvas = None
toolbar = None

fig_stats = None
ax_stats = None
stats_canvas = None
stats_table = None

geocode_var = None

# Basemap refresh state
BASEMAP_IMG = None
CURRENT_CRS_STR = None
_basemap_after_id = None  # debounce timer id

# Track current nav mode (so we don’t poke private toolbar attrs)
CURRENT_MODE = "NONE"   # NONE | PAN | ZOOM

# ----------------------------------------------------------
# Map drawing
# ----------------------------------------------------------
def _only_A_to_E(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    code_col = "sensitivity_code_max"
    if code_col not in df.columns:
        return df.iloc[0:0]
    out = df.copy()
    out = out[out.geometry.notna()]
    out[code_col] = out[code_col].astype("string").str.strip().str.upper()
    out = out[out[code_col].isin(list("ABCDE"))]
    return out

def _apply_map_axes_layout():
    # Fill the map pane: tiny visual edge (1%)
    ax.set_position([0.01, 0.01, 0.98, 0.98])

def _remove_old_basemap():
    global BASEMAP_IMG
    try:
        if BASEMAP_IMG is not None:
            BASEMAP_IMG.remove()
    except Exception:
        pass
    BASEMAP_IMG = None

def _add_basemap_if_crs(crs_str: str | None):
    """Add base map for current view; do not change extent."""
    global BASEMAP_IMG
    if not crs_str:
        return
    try:
        _remove_old_basemap()
        # reset_extent=False keeps current zoom/pan
        BASEMAP_IMG = ctx.add_basemap(
            ax,
            crs=crs_str,
            source=ctx.providers.OpenStreetMap.Mapnik,
            reset_extent=False
        )
    except Exception as exc:
        print("Basemap error:", exc, file=sys.stderr)

def _fit_bounds(bounds, pad_frac=0.05):
    xmin, ymin, xmax, ymax = bounds
    dx = xmax - xmin
    dy = ymax - ymin
    if dx == 0 or dy == 0:
        pad_x = pad_y = 1.0
    else:
        pad_x = dx * pad_frac
        pad_y = dy * pad_frac
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

def update_map(geocode_category: str) -> None:
    """Redraw the layer for selected category with A–E only, basemap included."""
    global CURRENT_CRS_STR
    try:
        filtered = tbl_flat_data[
            tbl_flat_data["name_gis_geocodegroup"] == geocode_category
        ].copy()
        filtered = _only_A_to_E(filtered)

        ax.clear()
        _apply_map_axes_layout()
        CURRENT_CRS_STR = filtered.crs.to_string() if getattr(filtered, "crs", None) else None

        if not filtered.empty:
            # Draw polygons on top of basemap
            col = filtered["sensitivity_code_max"].map(colour_map)
            # Add basemap first so polygons sit above it
            _fit_bounds(filtered.total_bounds, pad_frac=0.05)
            _add_basemap_if_crs(CURRENT_CRS_STR)
            filtered.plot(ax=ax, color=col.tolist(), alpha=0.80, linewidth=0.2, edgecolor='white', zorder=10)
            ax.set_aspect("equal")
        else:
            _remove_old_basemap()
            ax.text(0.5, 0.5, "No A–E data for selected category",
                    ha="center", va="center", transform=ax.transAxes)

        canvas.draw_idle()
    except Exception as exc:
        print("Error updating map:", exc, file=sys.stderr)

def refresh_basemap_now():
    """Re-add basemap for the current view (debounced externally)."""
    if CURRENT_CRS_STR:
        _add_basemap_if_crs(CURRENT_CRS_STR)
        canvas.draw_idle()

def schedule_basemap_refresh(delay_ms: int = 120):
    """Debounce basemap refreshing so we don’t hammer tiles while dragging."""
    global _basemap_after_id
    if root is None:
        return
    if _basemap_after_id is not None:
        try:
            root.after_cancel(_basemap_after_id)
        except Exception:
            pass
    _basemap_after_id = root.after(delay_ms, refresh_basemap_now)

# ----------------------------------------------------------
# Nav controls (Home/Fit/Pan/Zoom/Save)
# ----------------------------------------------------------
def _ensure_toolbar():
    if toolbar is not None:
        toolbar.update()

def nav_home():
    global FULL_EXTENT
    if FULL_EXTENT is not None:
        ax.clear()
        _apply_map_axes_layout()
        if geocode_var.get():
            update_map(geocode_var.get())
        _fit_bounds(FULL_EXTENT, pad_frac=0.05)
        schedule_basemap_refresh(10)
    _ensure_toolbar()

def nav_fit_layer():
    if geocode_var.get():
        update_map(geocode_var.get())
    schedule_basemap_refresh(10)
    _ensure_toolbar()

def nav_set_mode(mode: str):
    global CURRENT_MODE
    if toolbar is None:
        return
    if mode == "PAN":
        toolbar.pan()
        CURRENT_MODE = "PAN"
    elif mode == "ZOOM":
        toolbar.zoom()
        CURRENT_MODE = "ZOOM"
    else:
        # toggle off if already on
        if CURRENT_MODE == "PAN":
            toolbar.pan()
        elif CURRENT_MODE == "ZOOM":
            toolbar.zoom()
        CURRENT_MODE = "NONE"
    _ensure_toolbar()

def nav_save_png():
    try:
        fname = filedialog.asksaveasfilename(
            title="Save map as PNG",
            defaultextension=".png",
            filetypes=[("PNG Image","*.png")]
        )
        if fname:
            fig.savefig(fname, dpi=150, bbox_inches="tight")
    except Exception as exc:
        print("Save error:", exc, file=sys.stderr)

# ----------------------------------------------------------
# Statistics (table + bar chart with 10% border)
# ----------------------------------------------------------
def _apply_stats_axes_layout():
    ax_stats.set_position([0.10, 0.10, 0.80, 0.80])

def update_statistics(geocode_category: str) -> None:
    """Recalculate area per sensitivity class (A–E only) and update UI."""
    try:
        filtered = tbl_flat_data[
            tbl_flat_data["name_gis_geocodegroup"] == geocode_category
        ].copy()
        filtered = _only_A_to_E(filtered)

        if "area_m2" not in filtered.columns:
            return

        codes = list("ABCDE")
        grp = (filtered.groupby("sensitivity_code_max")["area_m2"].sum())
        area_km2 = pd.Series({c: grp.get(c, 0.0)/1e6 for c in codes}, name="area_km2")
        stats = pd.DataFrame({
            "code": codes,
            "description": [desc_map.get(c, "") for c in codes],
            "area_km2": area_km2.values
        })

        # Table
        stats_table.delete(*stats_table.get_children())
        for _, row in stats.iterrows():
            code = row["code"]
            col = colour_map.get(code, "#BDBDBD")
            stats_table.insert(
                "", "end",
                image=get_color_icon(col, size=18),
                values=(code, row["description"], f"{row['area_km2']:.2f}")
            )

        # Bars
        ax_stats.clear()
        bar_cols = [colour_map[c] for c in stats["code"]]
        ax_stats.bar(stats["code"], stats["area_km2"], color=bar_cols)
        ax_stats.set_title("Area by sensitivity code (km²)")
        ax_stats.set_xlabel("Sensitivity code")
        ax_stats.set_ylabel("Area (km²)")
        ax_stats.margins(x=0.10, y=0.10)  # 10% data padding
        _apply_stats_axes_layout()
        stats_canvas.draw_idle()

    except Exception as exc:
        print("Error updating statistics:", exc, file=sys.stderr)

def update_map_and_statistics(geocode_category: str) -> None:
    update_map(geocode_category)
    update_statistics(geocode_category)

# ----------------------------------------------------------
# Clean exit
# ----------------------------------------------------------
def close_application():
    try:
        plt.close(fig)
        plt.close(fig_stats)
    except Exception:
        pass
    try:
        root.quit()
        root.destroy()
    finally:
        os._exit(0)

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
if __name__ == "__main__":
    # ---- File locations -----------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(base_dir, "../system/config.ini")
    parquet_file = os.path.join(base_dir, "../output/geoparquet/tbl_flat.parquet")

    # ---- Load resources -----------------------------------
    config = read_config(config_file)
    colour_map = get_color_mapping(config)
    desc_map = get_descriptions(config)

    tbl_flat_data = load_geoparquet(parquet_file)
    if tbl_flat_data.empty:
        print("Fatal: tbl_flat.parquet is empty or missing.", file=sys.stderr)
        sys.exit(1)
    if "name_gis_geocodegroup" not in tbl_flat_data.columns:
        print("Fatal: Column 'name_gis_geocodegroup' missing in tbl_flat.", file=sys.stderr)
        sys.exit(1)

    only_valid = _only_A_to_E(tbl_flat_data)
    FULL_EXTENT = only_valid.total_bounds if not only_valid.empty else tbl_flat_data.total_bounds
    geocode_categories = sorted(tbl_flat_data["name_gis_geocodegroup"].dropna().unique().tolist())

    # ---- Tk layout ----------------------------------------
    root = tk.Tk()
    root.title("Maps Overview")
    root.geometry("1600x900")
    root.protocol("WM_DELETE_WINDOW", close_application)

    pw = tk.PanedWindow(root, orient=tk.HORIZONTAL, sashrelief='raised', sashwidth=8,
                        sashcursor='sb_h_double_arrow')
    pw.pack(fill=tk.BOTH, expand=True)

    # Left controls
    control_frame = ttk.Frame(pw, width=220)
    ttk.Label(control_frame, text="Geocode category:").pack(anchor="w", pady=(8,2), padx=8)
    geocode_var = tk.StringVar()
    cb = ttk.Combobox(control_frame, textvariable=geocode_var,
                      values=geocode_categories, state="readonly")
    cb.pack(anchor="w", pady=(0,8), padx=8, fill=tk.X)
    exit_btn = ttk.Button(control_frame, text="Exit", command=close_application, bootstyle=WARNING)
    exit_btn.pack(side="bottom", anchor="sw", padx=8, pady=8)

    # Middle: map + header toolbar
    map_frame = ttk.Frame(pw)
    header = ttk.Frame(map_frame)
    header.pack(side="top", fill="x")

    fig = Figure(figsize=(16, 9), dpi=100, constrained_layout=False)
    ax = fig.add_subplot(111)
    _apply_map_axes_layout()
    canvas = FigureCanvasTkAgg(fig, master=map_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)

    # Matplotlib toolbar (used by our custom buttons)
    toolbar_container = ttk.Frame(map_frame)  # not packed visually
    toolbar = NavigationToolbar2Tk(canvas, toolbar_container)
    toolbar.update()

    # Header buttons
    ttk.Button(header, text="Home",    command=nav_home).pack(side="left", padx=6, pady=6)
    ttk.Button(header, text="Fit layer", command=nav_fit_layer).pack(side="left", padx=6, pady=6)
    ttk.Button(header, text="Pan ⊛",  command=lambda: nav_set_mode("PAN")).pack(side="left", padx=6, pady=6)
    ttk.Button(header, text="Zoom ⊞", command=lambda: nav_set_mode("ZOOM")).pack(side="left", padx=6, pady=6)
    ttk.Button(header, text="Save",   command=nav_save_png).pack(side="left", padx=6, pady=6)

    # Hook pan/zoom events -> debounced basemap refresh
    def _on_release(event):
        # Mouse release after pan or zoom rectangle
        schedule_basemap_refresh(120)

    def _on_scroll(event):
        # If you enable wheel zoom elsewhere, keep tiles in sync
        schedule_basemap_refresh(120)

    fig.canvas.mpl_connect("button_release_event", _on_release)
    fig.canvas.mpl_connect("scroll_event", _on_scroll)

    # Right: stats (table + bar chart)
    stats_frame = ttk.Frame(pw, width=360)
    stats_table_label = ttk.Label(stats_frame, text="Area by sensitivity code",
                                  anchor="center", font=("TkDefaultFont", 10, "bold"))
    stats_table_label.pack(pady=6)

    stats_table_frame = ttk.Frame(stats_frame)
    stats_table_frame.pack(fill=tk.X, expand=False, padx=8, pady=(0, 6))
    stats_table = ttk.Treeview(
        stats_table_frame,
        columns=("Sensitivity code", "Description", "Area (km²)"),
        show="tree headings",
        height=8
    )
    stats_table.column("#0", width=36, anchor="center")
    stats_table.heading("#0", text="")
    stats_table.heading("Sensitivity code", anchor="center", text="Sensitivity code")
    stats_table.heading("Description", anchor="center", text="Description")
    stats_table.heading("Area (km²)", anchor="center", text="Area (km²)")
    stats_table.column("Sensitivity code", width=120, anchor="center")
    stats_table.column("Description", width=160, anchor="w")
    stats_table.column("Area (km²)", width=100, anchor="e")
    stats_table.pack(fill=tk.X, expand=False, padx=0, pady=0)

    stats_bar_frame = ttk.Frame(stats_frame)
    stats_bar_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(6, 8))
    fig_stats = Figure(figsize=(4, 3), dpi=100, constrained_layout=False)
    ax_stats = fig_stats.add_subplot(111)
    ax_stats.margins(x=0.10, y=0.10)  # 10% data padding
    ax_stats.set_position([0.10, 0.10, 0.80, 0.80])  # 10% figure border
    stats_canvas = FigureCanvasTkAgg(fig_stats, master=stats_bar_frame)
    stats_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

    # Add panes
    pw.add(control_frame, minsize=200, stretch='never')
    pw.add(map_frame, stretch='always')
    pw.add(stats_frame, minsize=320, stretch='never')

    # Combobox selection -> redraw
    def _on_select(event=None):
        if geocode_var.get():
            update_map_and_statistics(geocode_var.get())
    cb.bind("<<ComboboxSelected>>", _on_select)

    # Initial draw
    if geocode_categories:
        geocode_var.set(geocode_categories[0])
        update_map_and_statistics(geocode_categories[0])

    root.mainloop()
