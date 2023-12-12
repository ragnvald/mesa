import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import threading
import configparser
from osgeo import gdal
import datetime

def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config

def log_to_gui_and_file(log_widget, message, log_file_path):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted_message = f"{timestamp} - {message}"
    log_widget.insert(tk.END, formatted_message + "\n")
    log_widget.see(tk.END)

    with open(log_file_path, "a") as log_file:
        log_file.write(formatted_message + "\n")

def empty_geopackage(gpkg_file, log_widget, log_file_path):
    try:
        gdal.PushErrorHandler('CPLQuietErrorHandler')
        ds = gdal.OpenEx(gpkg_file, gdal.OF_VECTOR | gdal.OF_UPDATE)

        if ds is None:
            log_to_gui_and_file(log_widget, f"Could not open the GeoPackage file at {gpkg_file}", log_file_path)
            return

        layer_names = [ds.GetLayerByIndex(i).GetName() for i in range(ds.GetLayerCount())]

        for layer_name in layer_names:
            ds.DeleteLayer(layer_name)
            log_to_gui_and_file(log_widget, f"Deleted layer: {layer_name}", log_file_path)

        ds = None
        log_to_gui_and_file(log_widget, f"All layers removed from {gpkg_file}", log_file_path)
    except Exception as e:
        log_to_gui_and_file(log_widget, f"Error emptying GeoPackage: {e}", log_file_path)
    finally:
        gdal.PopErrorHandler()

def run_delete(gpkg_file, log_widget, log_file_path):
    empty_geopackage(gpkg_file, log_widget, log_file_path)

def create_delete_window(log_file_path, gpkg_file):
    delete_window = tk.Tk()
    delete_window.title("Delete GeoPackage Data")

    log_widget = scrolledtext.ScrolledText(delete_window, height=10)
    log_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    delete_btn = tk.Button(delete_window, text="Delete GeoPackage Data",
                           command=lambda: threading.Thread(
                               target=run_delete, 
                               args=(gpkg_file, log_widget, log_file_path), 
                               daemon=True).start())
    delete_btn.pack(pady=5, fill=tk.X)

    close_btn = tk.Button(delete_window, text="Close", command=delete_window.destroy)
    close_btn.pack(pady=5, fill=tk.X)

    delete_window.mainloop()

# Script execution starts here
config_file = 'config.ini'
config = read_config(config_file)
gpkg_file = config['DEFAULT']['gpkg_file']
log_file_path = "log.txt"  # Define the path for the log file

create_delete_window(log_file_path, gpkg_file)
