#!/usr/bin/env python3
"""
Maps Overview – Tkinter/GeoPandas/Matplotlib demo
Displays polygons coloured by sensitivity class, lets the user
filter by geocode category, and shows a side panel with an area
table + bar chart.

Only dependency‐free standard widgets are used; ttk.Treeview is
coaxed into showing per-row colour squares by exposing the hidden
tree column.
"""

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

import os
import sys
import configparser
import tkinter as tk
from tkinter import ttk

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import contextily as ctx  # Basemap provider
from ttkbootstrap.constants import WARNING

# ----------------------------------------------------------------------
# Helper: one tiny coloured square (PhotoImage) per hex colour,
# cached so it survives garbage collection and re-use.
# ----------------------------------------------------------------------
color_icons = {}  # Keeps the PhotoImages alive


def get_color_icon(color_hex: str, size: int = 20) -> tk.PhotoImage:
    """Return (cached) solid-colour PhotoImage square."""
    key = (color_hex, size)
    if key not in color_icons:
        img = tk.PhotoImage(width=size, height=size)
        img.put(color_hex, to=(0, 0, size, size))
        color_icons[key] = img
    return color_icons[key]


# ----------------------------------------------------------------------
# Configuration helpers
# ----------------------------------------------------------------------
def read_config(file_name: str) -> configparser.ConfigParser:
    """Read .ini file and return ConfigParser object."""
    cfg = configparser.ConfigParser()
    cfg.read(file_name, encoding="utf-8")
    return cfg


def get_color_mapping(cfg: configparser.ConfigParser) -> dict[str, str]:
    """Extract hex colours for sensitivity codes A-E from config."""
    return {sec: cfg[sec]["category_colour"]
            for sec in cfg.sections() if sec in "ABCDE"}


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------
def load_geoparquet(path: str) -> gpd.GeoDataFrame:
    """Wrapper around GeoPandas parquet loader with error handling."""
    try:
        return gpd.read_parquet(path)
    except Exception as exc:
        print("Error reading geoparquet:", exc, file=sys.stderr)
        return gpd.GeoDataFrame()  # empty


# ----------------------------------------------------------------------
# Map drawing
# ----------------------------------------------------------------------
def update_map(geocode_category: str) -> None:
    """Redraw the GeoPandas layer filtered on geocode category."""
    try:
        filtered = tbl_flat_data[
            tbl_flat_data["name_gis_geocodegroup"] == geocode_category
        ].copy()

        ax.clear()
        if not filtered.empty:
            colour_map = get_color_mapping(config)
            if "sensitivity_code_max" in filtered.columns:
                filtered["__colour"] = filtered["sensitivity_code_max"].map(colour_map)
                filtered.plot(ax=ax, color=filtered["__colour"], alpha=0.7)
            else:
                filtered.plot(ax=ax, alpha=0.7)

            ctx.add_basemap(ax, crs=filtered.crs.to_string(),
                            source=ctx.providers.OpenStreetMap.Mapnik)
            ax.set_aspect("equal")
        else:
            ax.text(0.5, 0.5, "No data for selected category",
                    ha="center", va="center", transform=ax.transAxes)
        canvas.draw()
    except Exception as exc:
        print("Error updating map:", exc, file=sys.stderr)


# ----------------------------------------------------------------------
# Basemap refresh after pan/zoom
# ----------------------------------------------------------------------
def refresh_basemap() -> None:
    try:
        ctx.add_basemap(ax, crs=tbl_flat_data.crs.to_string(),
                        source=ctx.providers.OpenStreetMap.Mapnik)
        canvas.draw()
    except Exception as exc:
        print("Error refreshing basemap:", exc, file=sys.stderr)


# ----------------------------------------------------------------------
# Enabling mouse interaction (pan + scroll zoom)
# ----------------------------------------------------------------------
def enable_pan_and_zoom() -> None:
    """Connect Matplotlib mouse events to keep basemap in sync."""

    def on_press(event):  # left-button press
        if event.button == 1:
            ax.start_pan(event.x, event.y, event.button)

    def on_release(event):
        if event.button == 1:
            ax.end_pan()
            refresh_basemap()

    def on_motion(event):
        if event.button == 1:
            ax.drag_pan(1, None, event.x, event.y)

    def on_scroll(event):
        scale = 1.2 if event.step > 0 else 1 / 1.2
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        xs, ys = event.xdata, event.ydata
        ax.set_xlim(xs - (xs - xlim[0]) * scale, xs + (xlim[1] - xs) * scale)
        ax.set_ylim(ys - (ys - ylim[0]) * scale, ys + (ylim[1] - ys) * scale)
        refresh_basemap()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("scroll_event", on_scroll)


# ----------------------------------------------------------------------
# Statistics (table + bar chart)
# ----------------------------------------------------------------------
def update_statistics(geocode_category: str) -> None:
    """Recalculate area per sensitivity class for the selected geocode category and update UI."""
    try:
        # Filter for the selected geocode category
        filtered = tbl_flat_data[
            tbl_flat_data["name_gis_geocodegroup"] == geocode_category
        ].copy()

        if {"sensitivity_code_max", "area_m2"} <= set(filtered.columns):
            stats = (filtered
                     .groupby(["sensitivity_code_max",
                               "sensitivity_description_max"])["area_m2"]
                     .sum()
                     .reset_index())
            stats["area_km2"] = stats["area_m2"] / 1e6
            stats.sort_values("sensitivity_code_max", inplace=True)

            colour_map = get_color_mapping(config)

            # ------------- Treeview (table) ----------------
            stats_table.delete(*stats_table.get_children())  # clear table

            for _, row in stats.iterrows():
                code = row["sensitivity_code_max"]
                stats_table.insert(
                    "", "end",
                    image=get_color_icon(colour_map.get(code, "#000000"), size=20),
                    values=(
                        code,
                        row["sensitivity_description_max"],
                        f"{row['area_km2']:.2f}",
                    )
                )

            # ------------- Bar chart ------------------------
            ax_stats.clear()
            bars = stats["sensitivity_code_max"]
            bar_cols = bars.map(colour_map)
            ax_stats.bar(bars, stats["area_km2"], color=bar_cols)
            ax_stats.set_title("Area by sensitivity code (km²)")
            ax_stats.set_xlabel("Sensitivity code")
            ax_stats.set_ylabel("Area (km²)")
            ax_stats.tick_params(axis="x", rotation=45)
            stats_canvas.draw()
    except Exception as exc:
        print("Error updating statistics:", exc, file=sys.stderr)


def update_map_and_statistics(geocode_category: str) -> None:
    """Helper called from UI – refreshes both panels."""
    update_map(geocode_category)
    update_statistics(geocode_category)


# ----------------------------------------------------------------------
# Main block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        # ---- File locations (adjust to suit your project) -------------
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(base_dir, "../system/config.ini")
        parquet_file = os.path.join(base_dir, "../output/geoparquet/tbl_flat.parquet")

        # ---- Load resources -------------------------------------------
        config = read_config(config_file)
        tbl_flat_data = load_geoparquet(parquet_file)
        if tbl_flat_data.empty:
            raise RuntimeError("tbl_flat.parquet is empty or missing")

        geocode_categories = sorted(
            tbl_flat_data["name_gis_geocodegroup"].unique().tolist()
        )

        # ---- Tk root window -------------------------------------------
        root = tk.Tk()
        root.title("Maps Overview")
        root.geometry("1600x900")

        # ---- Paned layout (controls | map | stats) --------------------
        pw = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.Frame(pw, width=200)
        map_frame = ttk.Frame(pw)
        stats_frame = ttk.Frame(pw, width=300)

        pw.add(control_frame, weight=0)
        pw.add(map_frame,    weight=1)
        pw.add(stats_frame,  weight=0)

        # ---- Matplotlib figure for map --------------------------------
        fig, ax = plt.subplots(figsize=(16, 9))
        canvas = FigureCanvasTkAgg(fig, master=map_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ---- Statistics: table + bar chart ----------------------------
        stats_table_label = ttk.Label(stats_frame, text="Area by sensitivity code",
                                     anchor="center", font=("TkDefaultFont", 10, "bold"))
        stats_table_label.pack(pady=5)

        # Use a frame to separate table and bar chart
        stats_table_frame = ttk.Frame(stats_frame)
        stats_table_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=(5, 0))

        stats_table = ttk.Treeview(
            stats_table_frame,
            columns=("Sensitivity code", "Description", "Area (km²)"),
            show="tree headings",
            height=10
        )
        stats_table.column("#0", width=40, anchor="center")
        stats_table.heading("#0", text="")

        stats_table.heading("Sensitivity code", anchor="center", text="Sensitivity code")
        stats_table.heading("Description", anchor="center", text="Description")
        stats_table.heading("Area (km²)", anchor="center", text="Area (km²)")

        stats_table.column("Sensitivity code", width=140, anchor="center")
        stats_table.column("Description", width=180, anchor="w")
        stats_table.column("Area (km²)", width=100, anchor="e")
        stats_table.pack(fill=tk.X, expand=False, padx=0, pady=0)

        # Bar chart gets the remaining space below the table
        stats_bar_frame = ttk.Frame(stats_frame)
        stats_bar_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        fig_stats = Figure(figsize=(4, 3), dpi=100)
        ax_stats = fig_stats.add_subplot(111)
        stats_canvas = FigureCanvasTkAgg(fig_stats, master=stats_bar_frame)
        stats_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # ---- Controls (left pane) ------------------------------------
        geocode_var = tk.StringVar()
        ttk.Label(control_frame, text="Geocode category:").pack(
            anchor="w", pady=2, padx=5)

        cb = ttk.Combobox(control_frame, textvariable=geocode_var,
                          values=geocode_categories, state="readonly")
        cb.pack(anchor="w", pady=2, padx=5)

        def _on_select(event=None):
            update_map_and_statistics(geocode_var.get())

        cb.bind("<<ComboboxSelected>>", _on_select)

        def close_application():
            root.destroy()
            # Consider removing system exit when this part is integrated in the full system.
            sys.exit(0)

        # ---- Exit button (lower left) --------------------------------
        exit_btn_frame = ttk.Frame(control_frame)
        exit_btn_frame.pack(side="bottom", fill="x", expand=False)
        close_btn = ttk.Button(exit_btn_frame, text="Exit", command=close_application, bootstyle=WARNING)
        close_btn.pack(side="left", anchor="sw", padx=10, pady=10)

        # ---- Initial draw --------------------------------------------
        if geocode_categories:
            geocode_var.set(geocode_categories[0])
            update_map_and_statistics(geocode_categories[0])

        enable_pan_and_zoom()
        root.mainloop()

    except Exception as err:
        print("Fatal error:", err, file=sys.stderr)
