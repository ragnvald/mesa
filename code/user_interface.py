import tkinter as tk
import locale

try:
    locale.setlocale(locale.LC_ALL, 'de_DE.utf8')  # For US English, adjust as needed
except locale.Error:
    locale.setlocale(locale.LC_ALL, '') 

from tkinter import messagebox, scrolledtext, Label
import subprocess
import webbrowser
import datetime
import os
import fiona
from PIL import Image, ImageTk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap import LabelFrame
import geopandas as gpd


def get_geopackage_stats(geopackage):
    # List available layers in the geopackage
    layer_names = fiona.listlayers(geopackage)
    print("Available layers in the geopackage:", layer_names)

    # Load different tables from the geopackage
    tbl_asset_group = gpd.read_file(geopackage, layer='tbl_asset_group')
    tbl_asset_object = gpd.read_file(geopackage, layer='tbl_asset_object')
    tbl_geocode_group = gpd.read_file(geopackage, layer='tbl_geocode_group')
    tbl_geocode_object = gpd.read_file(geopackage, layer='tbl_geocode_object')

    # Calculate the required statistics
    asset_layer_count = len(tbl_asset_group)
    asset_feature_count = len(tbl_asset_object)
    geocode_layer_count = len(tbl_geocode_group)
    geocode_object_count = len(tbl_geocode_object)

    # Create stats string
    stats_string = [
        ["", "Layers", "Features"],
        ["Assets", f"{asset_layer_count}", f"{asset_feature_count}"],
        ["Geocode", f"{geocode_layer_count}", f"{geocode_object_count}"]
    ]
    return stats_string


# Function to check and create folders
def check_and_create_folders():
    folders = ["input/geocode", "output", "qgis"]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

def log_to_logfile(message):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted_message = f"{timestamp} - {message}"
    with open("log.txt", "a") as log_file:
        log_file.write(formatted_message + "\n")

# Function to open a web link
def open_link(url):
    webbrowser.open_new_tab(url)

# Function to load and display the image
def display_image(bottom_panel):
    image_path = 'system_resources/mesa_illustration.png'
    original_image = Image.open(image_path)

    # Function to resize and update the image
    def resize_image(event):
        new_width = event.height
        new_height = event.height
        image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo  # keep a reference!

    # Create and place the label
    photo = ImageTk.PhotoImage(original_image)
    label = Label(bottom_panel, image=photo)
    label.image = photo  # keep a reference!
    label.pack(side='bottom', fill='both', expand=True)

    # Bind the resize function to the label's configure event
    bottom_panel.bind("<Configure>", resize_image)

def import_assets():
    try:
        # First try running import.py
        subprocess.run(["python", "01_import.py"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # If import.py fails or is not found, try running import.exe
            subprocess.run(["01_import.exe"], check=True)
        except subprocess.CalledProcessError:
            log_to_logfile("Failed to execute import assets script")

def edit_asset_group():
    try:
        subprocess.run(["python", "04_edit_asset_group.py"], check=True)
        log_to_logfile("Opened edit asset group")
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # If import.py fails, try running import.exe
            subprocess.run(["04_edit_asset_group.exe"], check=True)
        except subprocess.CalledProcessError:
            log_to_logfile("Failed to execute edit asset group script")

def edit_geocode_group():
    try:
        subprocess.run(["python", "04_edit_geocode_group.py"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # If import.py fails, try running import.exe
            subprocess.run(["04_edit_geocode_group.exe"], check=True)
        except subprocess.CalledProcessError:
            log_to_logfile("Failed to execute edit geocode group script")

def edit_processing_setup():
    try:
        subprocess.run(["python", "04_edit_input.py"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # If import.py fails, try running import.exe
            subprocess.run(["04_edit_input.exe"], check=True)
        except subprocess.CalledProcessError:
            log_to_logfile("Failed to execute edit input script")

def process_data():
    try:
        subprocess.run(["python", "06_process.py"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # If import.py fails, try running import.exe
            subprocess.run(["06_process.exe"], check=True)
        except subprocess.CalledProcessError:
            log_to_logfile("Failed to execute processing script")

def make_atlas():
    try:
        subprocess.run(["python", "07_make_atlas.py"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # If import.py fails, try running import.exe
            subprocess.run(["07_make_atlas.exe"], check=True)
        except subprocess.CalledProcessError:
            log_to_logfile("Failed to execute make atlas script")

def edit_atlas():
    try:
        subprocess.run(["python", "07_edit_atlas.py"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # If import.py fails, try running import.exe
            subprocess.run(["07_edit_atlas.exe"], check=True)
        except subprocess.CalledProcessError:
            log_to_logfile("Failed to execute edit atlas script")

def export_qgis():
    messagebox.showinfo("Export Package", "Export package script executed.")

def exit_program():
    root.destroy()

# Function to adjust the wrap length of the label
def adjust_wrap(event):
    mesa_label.config(wraplength=right_panel.winfo_width())

# Check and create folders at the beginning
check_and_create_folders()

# Setup the main Tkinter window
root = ttk.Window(themename='superhero')
root.title("MESA 4")
root.geometry("800x540")

button_width = 20
button_padx  = 20
button_pady  = 10

# Main frame
main_frame = tk.Frame(root)
main_frame.pack(fill='both', expand=True, pady=20)

# Configure the grid weights
main_frame.grid_columnconfigure(0, weight=0)  # Left panel has no weight
main_frame.grid_columnconfigure(1, weight=0)  # Separator has no weight
main_frame.grid_columnconfigure(2, weight=1)  # Right panel has weight

# Left panel
left_panel = tk.Frame(main_frame)
left_panel.grid(row=0, column=0, sticky="nsew", padx=20)

# Set minimum size for left panel
main_frame.grid_columnconfigure(0, minsize=250)  # Adjust the minsize as needed

# Separator
separator = ttk.Separator(main_frame, orient='vertical')
separator.grid(row=0, column=1, sticky='ns')

# Right panel
right_panel = tk.Frame(main_frame)
right_panel.grid(row=0, column=2, sticky="nsew", padx=20)
right_panel.grid_rowconfigure(0, weight=1)

# Bind the adjust_wrap function to right_panel's configure event
right_panel.bind("<Configure>", adjust_wrap)

# Label with text on the right panel
mesa_text = """This version of the MESA tool is a stand-alone desktop based version prepared for use on the Windwos platform. To use it you will have to deposit spatial data for assets and geocodes (eg grids). The result of the processing is a sensitivity data set. To balance the resulting scores you will have to provide values for assets and their associated  suceptibilities. 

Contact ragnvald@mindland.com for further information.

Python code is available at https://github.com/ragnvald/mesa """

mesa_label = tk.Label(right_panel, text=mesa_text, justify="left", anchor="nw")
mesa_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# Exit button
exit_btn = ttk.Button(right_panel, text="Exit", command=exit_program, width=button_width, bootstyle=WARNING)
exit_btn.grid(row=6, column=0, columnspan=2, pady=button_pady)

bottom_panel = tk.Frame(root)
bottom_panel.pack(fill='x', expand=True)

# Add buttons to left panel with spacing between buttons
import_assets_btn = ttk.Button(left_panel, text="Import", command=import_assets, width=button_width, bootstyle=PRIMARY)
import_assets_btn.grid(row=0, column=0, padx=button_padx, pady=button_pady)

edit_asset_group_btn = ttk.Button(left_panel, text="Edit asset groups", command=edit_asset_group, width=button_width, bootstyle=SECONDARY)
edit_asset_group_btn.grid(row=0, column=1, padx=button_padx, pady=button_pady)

edit_geocode_group_btn = ttk.Button(left_panel, text="Edit geocode groups", command=edit_geocode_group, width=button_width, bootstyle=SECONDARY)
edit_geocode_group_btn.grid(row=1, column=1, padx=button_padx, pady=button_pady)

edit_processing_setup_btn = ttk.Button(left_panel, text="Set up processing", command=edit_processing_setup, width=button_width)
edit_processing_setup_btn.grid(row=2, column=0, padx=button_padx, pady=button_pady)

process_stacked_data_btn = ttk.Button(left_panel, text="Process data", command=process_data, width=button_width)
process_stacked_data_btn.grid(row=3, column=0, padx=button_padx, pady=button_pady)

process_stacked_data_btn = ttk.Button(left_panel, text="Make atlas", command=make_atlas, width=button_width)
process_stacked_data_btn.grid(row=4, column=0, padx=button_padx, pady=button_pady)

edit_asset_group_btn = ttk.Button(left_panel, text="Edit atlas", command=edit_atlas, width=button_width, bootstyle=SECONDARY)
edit_asset_group_btn.grid(row=4, column=1, padx=button_padx, pady=button_pady)

export_qgis_btn = ttk.Button(left_panel, text="Export QGIS file", command=export_qgis, width=button_width)
export_qgis_btn.grid(row=5, column=0, padx=button_padx, pady=button_pady)

# Create a labelframe on the left side of the bottom frame
labelframe = ttk.LabelFrame(bottom_panel, text="System status", bootstyle='primary')
labelframe.pack(side='left', fill='y', expand=False, padx=10, pady=10)

# Set a fixed width for the labelframe
labelframe.config(width=200)

# Prevent the labelframe from growing with the content
labelframe.pack_propagate(False)

geopackage_path = "output/mesa.gpkg"  # Ensure this path is correct
table_data = get_geopackage_stats(geopackage_path)

for row_index, row in enumerate(table_data):
    for col_index, cell in enumerate(row):
        label = ttk.Label(labelframe, text=cell, borderwidth=0)
        label.grid(row=row_index, column=col_index, sticky="nsew", padx=4, pady=4)
        labelframe.grid_columnconfigure(col_index, weight=1)
        
# Adjust row weight for all rows
for row_index in range(len(table_data)):
    labelframe.grid_rowconfigure(row_index, weight=1)

# Call the function to display the image in the bottom frame
display_image(bottom_panel)

log_to_logfile("User interface, main dialogue opened")

# Start the GUI event loop
root.mainloop()