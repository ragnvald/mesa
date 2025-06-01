# Code makes an initial map based on the contents in a geoparquet.
# The data sets are in a file called flat_parquet. It must be 
# filtered based on geocode category. The map is then displayed in 
# a window with controls. One main control is a dropdown list that 
# allows the user to select a geocode category. The map is then updated.
# Optionally the user can switch to showing lines instead of polygons.
# The user can also select a specific asset group to display. 
# Furthermore the user can choose to add attributes from the asset.

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
import tkinter as tk
from tkinter import ttk
import geopandas as gpd
import pandas as pd
import configparser
import os
import sys
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import contextily as ctx  # Add this import for basemap support
from matplotlib.figure import Figure

color_icons = {}       # husker PhotoImage per farge (unngå GC)

def get_color_icon(color_hex, size=8):
    """Returnerer (og cacher) et lite kvadrat-ikon i gitt farge."""
    if color_hex not in color_icons:
        img = tk.PhotoImage(width=size, height=size)
        img.put(color_hex, to=(0, 0, size, size))
        color_icons[color_hex] = img
    return color_icons[color_hex]

# Read the configuration file
def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config

# Load geoparquet data
def load_geoparquet(file_path):
    try:
        return gpd.read_parquet(file_path)
    except Exception as e:
        print(f"Error loading geoparquet file: {e}")
        return gpd.GeoDataFrame()

# Get color mapping from the configuration file
def get_color_mapping(config):
    color_mapping = {}
    for section in config.sections():
        if section in ['A', 'B', 'C', 'D', 'E']:
            color_mapping[section] = config[section]['category_colour']
    return color_mapping

# Update the map using only tbl_flat_data with color codes from config.ini
def update_map(geocode_category):
    try:
        # Explicitly create a copy of the filtered data
        filtered_data = tbl_flat_data[tbl_flat_data['name_gis_geocodegroup'] == geocode_category].copy()
        ax.clear()
        if not filtered_data.empty:
            color_mapping = get_color_mapping(config)
            if 'sensitivity_code_max' in filtered_data.columns:
                filtered_data['color'] = filtered_data['sensitivity_code_max'].map(color_mapping)
                filtered_data.plot(ax=ax, color=filtered_data['color'], alpha=0.7)  # Add transparency for better visibility
            else:
                filtered_data.plot(ax=ax, alpha=0.7)
            
            # Add OpenStreetMap basemap
            ctx.add_basemap(ax, crs=filtered_data.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
            
            ax.set_aspect('equal')  # Ensure aspect ratio is equal
        else:
            ax.text(0.5, 0.5, 'No data available for the selected category',
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        canvas.draw()
    except Exception as e:
        print(f"Error updating map: {e}")

# Refresh the basemap after panning or zooming
def refresh_basemap():
    try:
        ctx.add_basemap(ax, crs=tbl_flat_data.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
        canvas.draw()
    except Exception as e:
        print(f"Error refreshing basemap: {e}")

# Enable panning and zooming with basemap refresh
def enable_pan_and_zoom():
    def on_press(event):
        if event.button == 1:  # Left mouse button for panning
            ax.start_pan(event.x, event.y, event.button)  # Initialize panning

    def on_release(event):
        if event.button == 1:  # Left mouse button for panning
            ax.end_pan()  # End panning
            refresh_basemap()  # Refresh basemap after releasing the mouse button

    def on_motion(event):
        if event.button == 1:  # Left mouse button for panning
            ax.drag_pan(1, None, event.x, event.y)  # Continue panning

    def on_scroll(event):
        # Zoom in or out based on the scroll direction
        base_scale = 1.2
        scale_factor = base_scale if event.step > 0 else 1 / base_scale

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xdata = event.xdata  # Get mouse x position in data coordinates
        ydata = event.ydata  # Get mouse y position in data coordinates

        # Calculate new limits
        new_xlim = [
            xdata - (xdata - xlim[0]) * scale_factor,
            xdata + (xlim[1] - xdata) * scale_factor,
        ]
        new_ylim = [
            ydata - (ydata - ylim[0]) * scale_factor,
            ydata + (ylim[1] - ydata) * scale_factor,
        ]

        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        refresh_basemap()  # Refresh basemap after zooming

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('scroll_event', on_scroll)  # Connect scroll event for zooming

# Update the statistics table and graph
def update_statistics():
    try:
        # Calculate area per sensitivity_code_max category using area_m2
        if 'sensitivity_code_max' in tbl_flat_data.columns and 'area_m2' in tbl_flat_data.columns:
            stats_data = tbl_flat_data.copy()
            stats_summary = stats_data.groupby(['sensitivity_code_max', 'sensitivity_description_max'])['area_m2'].sum().reset_index()
            stats_summary['area_km2'] = stats_summary['area_m2'] / 1e6  # Convert area to km²
            stats_summary = stats_summary.sort_values(by='sensitivity_code_max')  # Order alphabetically

            # Get color mapping once
            color_mapping = get_color_mapping(config)

            # Update the statistics table
            for row in stats_table.get_children():
                stats_table.delete(row)
            for _, r in stats_summary.iterrows():
                code  = r['sensitivity_code_max']
                color = color_mapping.get(code, "#000000")

                item_id = stats_table.insert(
                    '', 'end',
                    image=get_color_icon(color),             #  bildet i første kolonne
                    values=(
                        '',                                 # første "tekst-celle" er tom – bildet vises her
                        code,
                        r['sensitivity_description_max'],
                        f"{r['area_km2']:.2f}"
                    )
                )
                # Set all text to black first
                stats_table.item(item_id, tags=(f"row_{item_id}",))
                stats_table.tag_configure(f"row_{item_id}", foreground="black")
                # Now color just the square in the first column
                # This is a Tkinter Treeview limitation: per-cell coloring is not natively supported.
                # Workaround: set the square character to the correct color and pad with a black square for the rest.
                # So, set the value for the first column to a colored square, and the rest to black text.
                # This requires using a custom font or image for true per-cell coloring, but for text, we can only color the whole row.
                # Therefore, as a workaround, set the square to the correct color, and the rest of the row to black.
                # This will color the whole row, but the square will be visible in the correct color.
                # If you want only the square colored, you must use a custom widget or a third-party table library.
                stats_table.set(item_id, "Color", f"\u25A0")  # Unicode black square
                stats_table.tag_configure(f"square_{item_id}", foreground=color)
                # Apply both tags: first for the square, second for the rest of the row
                stats_table.item(item_id, tags=(f"square_{item_id}", f"row_{item_id}"))

            # Update the bar chart
            ax_stats.clear()
            bar_colors = stats_summary['sensitivity_code_max'].map(color_mapping)
            ax_stats.bar(stats_summary['sensitivity_code_max'], stats_summary['area_km2'], color=bar_colors)
            ax_stats.set_title("Area by sensitivity code (km²)")
            ax_stats.set_xlabel("Sensitivity code")
            ax_stats.set_ylabel("Area (km²)")
            ax_stats.tick_params(axis='x', rotation=45)
            stats_canvas.draw()
    except Exception as e:
        print(f"Error updating statistics: {e}")

# Update the map and statistics when a geocode category is selected
def update_map_and_statistics(geocode_category):
    update_map(geocode_category)
    update_statistics()

# Geocode category selection handler
def on_geocode_category_selected(event):
    update_map_and_statistics(geocode_category_var.get())

# Main function
if __name__ == "__main__":
    try:
        # Load configuration and tbl_flat geoparquet data only
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../system/config.ini")
        config = read_config(config_file)
        tbl_flat_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../output/geoparquet/tbl_flat.parquet")
        tbl_flat_data = load_geoparquet(tbl_flat_file)
        if tbl_flat_data.empty:
            raise ValueError("No data loaded from tbl_flat geoparquet file.")
        
        # Sort geocode categories alphabetically
        geocode_categories = sorted(tbl_flat_data['name_gis_geocodegroup'].unique().tolist())

        # Create main Tkinter window
        root = tk.Tk()
        root.title("Maps Overview")
        root.geometry("1600x900")
        
        # Create a horizontal PanedWindow as the main container
        pw = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)
        
        # Left pane: Controls (fixed width)
        control_frame = ttk.Frame(pw, width=200)
        # Center pane: Map (expandable)
        map_frame = ttk.Frame(pw)
        # Right pane: Statistics
        stats_frame = ttk.Frame(pw, width=300)
        
        pw.add(control_frame, weight=0)
        pw.add(map_frame, weight=1)
        pw.add(stats_frame, weight=0)

        # Create figure and canvas in the map pane
        fig, ax = plt.subplots(figsize=(16, 9))
        canvas = FigureCanvasTkAgg(fig, master=map_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Statistics table
        stats_table_label = ttk.Label(stats_frame, text="Area by sensitivity code", anchor="center", font=("TkDefaultFont", 10, "bold"))
        stats_table_label.pack(pady=5)

        # Create a style for the Treeview headings and rows
        style = ttk.Style()
        style.configure("Treeview.Heading", font=("TkDefaultFont", 10, "bold"))  # Set bold font for headings
        style.configure("Treeview", rowheight=40)  # Increase row height to make squares larger

        stats_table = ttk.Treeview(stats_frame, columns=("Color", "Sensitivity code", "Description", "Area (km²)"), show="headings", height=10)
        stats_table.heading("Color", text="", anchor="center")
        stats_table.heading("Sensitivity code", text="Sensitivity code", anchor="center")
        stats_table.heading("Description", text="Description", anchor="center")
        stats_table.heading("Area (km²)", text="Area (km²)", anchor="center")
        stats_table.column("Color", anchor="center", width=40)  # Increase the width of the color square column
        stats_table.column("Sensitivity code", anchor="center", width=150)
        stats_table.column("Description", anchor="center", width=200)
        stats_table.column("Area (km²)", anchor="center", width=100)
        stats_table.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Statistics bar chart
        fig_stats = Figure(figsize=(4, 3), dpi=100)
        ax_stats = fig_stats.add_subplot(111)
        stats_canvas = FigureCanvasTkAgg(fig_stats, master=stats_frame)
        stats_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # =======================
        # Left pane controls
        # =======================
        geocode_category_var = tk.StringVar()
        geocode_category_label = ttk.Label(control_frame, text="Geocode category:")
        geocode_category_label.pack(anchor="w", pady=2, padx=5)
        geocode_category_dropdown = ttk.Combobox(control_frame, textvariable=geocode_category_var)
        geocode_category_dropdown['values'] = geocode_categories  # Use sorted categories
        geocode_category_dropdown.pack(anchor="w", pady=2, padx=5)
        geocode_category_dropdown.bind("<<ComboboxSelected>>", on_geocode_category_selected)
        
        # Set default geocode category if available
        if geocode_categories:
            default_geocode_category = geocode_categories[0]
            geocode_category_var.set(default_geocode_category)
            update_map_and_statistics(default_geocode_category)

        # Enable panning and zooming with basemap refresh
        enable_pan_and_zoom()

        root.mainloop()
    except Exception as e:
        print(f"An error occurred: {e}")