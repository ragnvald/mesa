
# Combined Python Script

# Importing necessary modules
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import threading
import geopandas as gpd
from sqlalchemy import create_engine, exc
from sqlalchemy.types import Integer, String, DateTime
from shapely.geometry import box
from osgeo import ogr
import configparser
import datetime
import glob
import os
import pandas as pd
import subprocess
import webbrowser
from PIL import Image, ImageTk

# Common functions
def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config

def log_to_gui(log_widget, message):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted_message = f"{timestamp} {message}\n"
    log_widget.config(state=tk.NORMAL)
    log_widget.insert(tk.END, formatted_message)
    log_widget.yview(tk.END)
    log_widget.config(state=tk.DISABLED)

# Functions from 01_import_asset_objects.py
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import threading
import geopandas as gpd
from sqlalchemy import create_engine
import configparser
import datetime
import glob
import os
from osgeo import ogr
import pandas as pd
from sqlalchemy import exc
from shapely.geometry import box

# Read the configuration file
def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config

# Logging function to write to the GUI log and log file
def log_to_gui(log_widget, message):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted_message = f"{timestamp} - {message}"
    log_widget.insert(tk.END, formatted_message + "\n")
    log_widget.see(tk.END)
    with open("log.txt", "a") as log_file:
        log_file.write(formatted_message + "\n")

# Read and reproject spatial file to EPSG:4326
def read_spatial_file(filepath, layer_name=None):
    if layer_name:
        data = gpd.read_file(filepath, layer=layer_name)
    else:
        data = gpd.read_file(filepath)

    if data.crs is None:
        data.set_crs(epsg=4326, inplace=True)
    elif data.crs.to_epsg() != 4326:
        data = data.to_crs(epsg=4326)
    return data


# Get bounding box in EPSG:4326
def get_bounding_box(data):
    bbox = data.total_bounds
    bbox_geom = box(*bbox)
    return bbox_geom

# Function to process a layer and add to asset objects
def process_layer(data, asset_objects, object_id_counter, group_id, layer_name, log_widget):
    if data.empty:
        log_to_gui(log_widget, f"No data found in layer {layer_name}")
        return object_id_counter

    for index, row in data.iterrows():
        attributes = '; '.join([f"{col}: {row[col]}" for col in data.columns if col != 'geometry'])
        area_m2 = row.geometry.area if row.geometry else 0
        asset_objects.append({
            'id': int(object_id_counter),
            'ref_asset_group': int(group_id),
            'asset_group_name': layer_name,
            'attributes': attributes,
            'process': True,
            'area_m2': int(area_m2),
            'geom': row.geometry
        })
        object_id_counter += 1

    return object_id_counter

# Function to import spatial data for assets
def import_spatial_data(input_folder_asset, log_widget, progress_var):
    asset_objects = []
    asset_groups = {}
    group_id_counter = 1
    object_id_counter = 1
    file_patterns = ['*.shp', '*.gpkg']
    total_files = sum([len(glob.glob(os.path.join(input_folder_asset, '**', pattern), recursive=True)) for pattern in file_patterns])
    processed_files = 0

    for pattern in file_patterns:
        for filepath in glob.glob(os.path.join(input_folder_asset, '**', pattern), recursive=True):
            try:
                log_to_gui(log_widget, f"Processing file: {filepath}")
                if filepath.endswith('.gpkg'):
                    ds = ogr.Open(filepath)
                    if ds is None:
                        log_to_gui(log_widget, f"No layers found in GeoPackage: {filepath}")
                        continue

                    for i in range(ds.GetLayerCount()):
                        layer = ds.GetLayerByIndex(i)
                        if layer.GetGeomType() != 0:  # Check if the layer is spatial
                            layer_name = layer.GetName()
                            data = read_spatial_file(filepath, layer_name=layer_name)
                            if layer_name not in asset_groups:
                                bbox_geom = get_bounding_box(data)
                                asset_groups[layer_name] = {
                                    'id': int(group_id_counter),
                                    'name_original': layer_name,
                                    'name_fromuser': layer_name,
                                    'date_import': datetime.datetime.now(),
                                    'bounding_box_geom': bbox_geom.wkt,
                                    'total_asset_objects': int(0),
                                    'importance': int(0),
                                    'susceptibility': int(0),
                                    'sensitivity': int(0)
                                }
                                group_id_counter += 1
                            group_id = asset_groups[layer_name]['id']
                            object_id_counter = process_layer(
                                data, asset_objects, object_id_counter, group_id, layer_name, log_widget)
                    ds = None
                else:
                    data = read_spatial_file(filepath)
                    asset_group_name = os.path.splitext(os.path.basename(filepath))[0]
                    if asset_group_name not in asset_groups:
                        bbox_geom = get_bounding_box(data)
                        asset_groups[asset_group_name] = {
                            'id': int(group_id_counter),
                            'name_original': asset_group_name,
                            'name_fromuser': '',
                            'date_import': datetime.datetime.now(),
                            'bounding_box_geom': bbox_geom.wkt,
                            'total_asset_objects': int(0),
                            'importance': int(0),
                            'susceptibility': int(0),
                            'sensitivity': int(0)
                        }
                        group_id_counter += 1
                    group_id = asset_groups[asset_group_name]['id']
                    object_id_counter = process_layer(
                        data, asset_objects, object_id_counter, group_id, asset_group_name, log_widget)

                processed_files += 1
                progress_var.set(processed_files / total_files * 100)
            except Exception as e:
                log_to_gui(log_widget, f"Error processing file {filepath}: {e}")
    
    # Convert dictionary to DataFrame
    asset_groups_df = pd.DataFrame(asset_groups.values())
    asset_objects_gdf = gpd.GeoDataFrame(asset_objects, geometry='geom')

    # Ensure the id column is of type int64
    asset_groups_df['id'] = asset_groups_df['id'].astype('int64')
    asset_objects_gdf['id'] = asset_objects_gdf['id'].astype('int64')

    
    return gpd.GeoDataFrame(asset_objects, geometry='geom'), asset_groups_df


# Function to export to geopackage
def export_to_geopackage(gdf, gpkg_file, layer_name, log_widget):
    engine = create_engine(f'sqlite:///{gpkg_file}')
    gdf.to_file(gpkg_file, layer=layer_name, driver="GPKG", if_exists='append')
    log_to_gui(log_widget, f"Data exported to {gpkg_file}, layer {layer_name}")

# Function to update asset groups in geopackage
def update_asset_groups(asset_groups_df, gpkg_file, log_widget):
    engine = create_engine(f'sqlite:///{gpkg_file}')

    # Define the data types for the ID columns
    id_col = asset_groups_df.columns[asset_groups_df.dtypes == 'int64']
    asset_groups_df[id_col] = asset_groups_df[id_col].astype(int)

    try:
        asset_groups_df.to_sql('tbl_asset_group', con=engine, if_exists='replace', index=False)
        log_to_gui(log_widget, "Asset groups updated in GeoPackage.")
    except exc.SQLAlchemyError as e:
        log_to_gui(log_widget, f"Failed to update asset groups: {e}")

# Thread function to run import without freezing GUI
def run_import(input_folder_asset, gpkg_file, log_widget, progress_var):
    log_to_gui(log_widget, "Starting asset import process...")
    asset_objects_gdf, asset_groups_df = import_spatial_data(input_folder_asset, log_widget, progress_var)
    export_to_geopackage(asset_objects_gdf, gpkg_file, 'tbl_asset_object', log_widget)
    update_asset_groups(asset_groups_df, gpkg_file, log_widget)
    log_to_gui(log_widget, "Asset import completed.")
    progress_var.set(100)

# Function to close the application
def close_application():
    root.destroy()

# Create the user interface
root = tk.Tk()
root.title("Asset Import Utility")

# Create a log widget
log_widget = scrolledtext.ScrolledText(root, height=10)
log_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Create a progress bar
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root, orient="horizontal", length=200, mode="determinate", variable=progress_var)
progress_bar.pack(pady=5, fill=tk.X)

# Add buttons for the different operations
import_btn = tk.Button(root, text="Import Assets", command=lambda: threading.Thread(
    target=run_import, args=(input_folder_asset, gpkg_file, log_widget, progress_var), daemon=True).start())
import_btn.pack(pady=5, fill=tk.X)

close_btn = tk.Button(root, text="Close", command=close_application)
close_btn.pack(pady=5, fill=tk.X)

# Load configuration settings
config_file = 'config.ini'
config = read_config(config_file)
input_folder_asset = config['DEFAULT']['input_folder_asset']
gpkg_file = config['DEFAULT']['gpkg_file']

root.mainloop()

# Functions from 01_import_geocode_objects.py
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import threading
import geopandas as gpd
from sqlalchemy import create_engine
from shapely.geometry import box
from osgeo import ogr
import configparser
import datetime
import glob
import os

# Read the configuration file
def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config

# Logging function to write to the GUI log
def log_to_gui(log_widget, message):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted_message = f"{timestamp} - {message}"
    log_widget.insert(tk.END, formatted_message + "\n")
    log_widget.see(tk.END)

    with open("log.txt", "a") as log_file:
        log_file.write(formatted_message + "\n")

# Function to read and reproject spatial data
def read_and_reproject(filepath, layer=None):
    data = gpd.read_file(filepath, layer=layer)
    if data.crs is None:
        data.set_crs(epsg=4326, inplace=True)
    elif data.crs.to_epsg() != 4326:
        data = data.to_crs(epsg=4326)
    return data

# Function to process a layer and add to groups and objects
def process_layer(data, geocode_groups, geocode_objects, group_id_counter, object_id_counter, layer_name, log_widget):
    if data.empty:
        log_to_gui(log_widget, f"No data found in layer {layer_name}")
        return group_id_counter, object_id_counter

    feature_count = len(data)
    log_to_gui(log_widget, f"{layer_name} ({feature_count} features)")

    # Calculate bounding box and add to geocode groups
    bounding_box = data.total_bounds
    bbox_geom = box(*bounding_box)
    geocode_groups.append({
        'id': group_id_counter,
        'name': layer_name,
        'description': f'Description for {layer_name}',
        'geom': bbox_geom
    })

    # Add geocode objects
    for index, row in data.iterrows():
        geom = row.geometry if 'geometry' in data.columns else None
        code = row['QDGC'] if 'QDGC' in data.columns else object_id_counter
        geocode_objects.append({
            'id': object_id_counter,
            'code': code,
            'ref_geocodegroup': group_id_counter,
            'geom': geom
        })
        object_id_counter += 1

    return group_id_counter + 1, object_id_counter

# Function to export to geopackage
def export_to_geopackage(geocode_groups_gdf, geocode_objects_gdf, gpkg_file, log_widget):
    engine = create_engine(f'sqlite:///{gpkg_file}')
    try:
        if not geocode_groups_gdf.empty:
            geocode_groups_gdf.to_file(gpkg_file, layer='tbl_geocode_group', driver="GPKG", if_exists='append')
            log_to_gui(log_widget, f"Exported {len(geocode_groups_gdf)} groups to {gpkg_file}")

        if not geocode_objects_gdf.empty:
            log_to_gui(log_widget, f"Attempting to export {len(geocode_objects_gdf)} objects to {gpkg_file}")
            geocode_objects_gdf.to_file(gpkg_file, layer='tbl_geocode_object', driver="GPKG", if_exists='append')
            log_to_gui(log_widget, f"Exported {len(geocode_objects_gdf)} objects to {gpkg_file}")

    except Exception as e:
        log_to_gui(log_widget, f"Error during export: {e}")


# Import spatial data and export to geopackage
def import_spatial_data(input_folder_grid, log_widget, progress_var):
    geocode_groups = []
    geocode_objects = []
    group_id_counter = 1
    object_id_counter = 1
    file_patterns = ['*.shp', '*.gpkg']
    total_files = sum([len(glob.glob(os.path.join(input_folder_grid, '**', pattern), recursive=True)) for pattern in file_patterns])
    processed_files = 0

    for pattern in file_patterns:
        for filepath in glob.glob(os.path.join(input_folder_grid, '**', pattern), recursive=True):
            if filepath.endswith('.gpkg'):
                ds = ogr.Open(filepath)
                if ds is None:
                    log_to_gui(log_widget, f"No layers found in GeoPackage: {filepath}")
                    continue

                for i in range(ds.GetLayerCount()):
                    layer = ds.GetLayerByIndex(i)
                    layer_name = layer.GetName()
                    data = gpd.read_file(filepath, layer=layer_name)
                    group_id_counter, object_id_counter = process_layer(
                        data, geocode_groups, geocode_objects, group_id_counter,
                        object_id_counter, layer_name, log_widget)
                ds = None
            else:
                data = read_and_reproject(filepath)
                original_layer_name = os.path.splitext(os.path.basename(filepath))[0]
                group_id_counter, object_id_counter = process_layer(
                    data, geocode_groups, geocode_objects, group_id_counter,
                    object_id_counter, original_layer_name, log_widget)

            processed_files += 1
            progress_var.set(processed_files / total_files * 100)

    geocode_groups_gdf = gpd.GeoDataFrame(geocode_groups, geometry='geom' if geocode_groups else None)
    geocode_objects_gdf = gpd.GeoDataFrame(geocode_objects, geometry='geom' if geocode_objects else None)
    
    log_to_gui(log_widget, f"Total geocodes added: {object_id_counter - 1}")
    return geocode_groups_gdf, geocode_objects_gdf

# Thread function to run import without freezing GUI
def run_import(input_folder_grid, gpkg_file, log_widget, progress_var):
    geocode_groups_gdf, geocode_objects_gdf = import_spatial_data(input_folder_grid, log_widget, progress_var)
    
    log_to_gui(log_widget, f"Preparing to export {len(geocode_objects_gdf)} geocode objects.")
    
    export_to_geopackage(geocode_groups_gdf, geocode_objects_gdf, gpkg_file, log_widget)
    log_to_gui(log_widget, "Import and export completed.")
    progress_var.set(100)


# Function to close the application
def close_application():
    root.destroy()

# Create the user interface
root = tk.Tk()
root.title("MESA Import Utility")

# Create a log widget
log_widget = scrolledtext.ScrolledText(root, height=10)
log_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Create a progress bar
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root, orient="horizontal", length=200, mode="determinate", variable=progress_var)
progress_bar.pack(pady=5, fill=tk.X)

# Add buttons for the different operations
import_btn = tk.Button(root, text="Import Data", command=lambda: threading.Thread(
    target=run_import, args=(input_folder_grid, gpkg_file, log_widget, progress_var), daemon=True).start())
import_btn.pack(pady=5, fill=tk.X)

close_btn = tk.Button(root, text="Close", command=close_application)
close_btn.pack(pady=5, fill=tk.X)

# Load configuration settings
config_file = 'config.ini'
config = read_config(config_file)
input_folder_grid = config['DEFAULT']['input_folder_grid']
gpkg_file = config['DEFAULT']['gpkg_file']

root.mainloop()

# Functions from 04_edit_input.py
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import pandas as pd
from sqlalchemy import create_engine, exc
from sqlalchemy.types import Integer, String, DateTime

# Function to validate integer input
def validate_integer(P):
    if P.isdigit() or P == "":
        return True
    return False

# Function to update sensitivity
def calculate_sensitivity(row_index):
    try:
        susceptibility = int(entries[row_index]['susceptibility'].get())
        importance = int(entries[row_index]['importance'].get())
        sensitivity = susceptibility * importance
        entries[row_index]['sensitivity'].config(text=str(sensitivity))
        df.at[row_index, 'susceptibility'] = susceptibility
        df.at[row_index, 'importance'] = importance
        #df.at[row_index, 'sensitivity'] = sensitivity
        df.at[row_index, 'sensitivity'] = susceptibility * importance
    except ValueError:
        entries[row_index]['sensitivity'].config(text="")

    # Enforce data type after update
    df['susceptibility'] = df['susceptibility'].astype('int64', errors='ignore')
    df['importance'] = df['importance'].astype('int64', errors='ignore')
    df['sensitivity'] = df['sensitivity'].astype('int64', errors='ignore')

# Function to load and refresh data from geopackage
def load_data():
    global df, entries
    engine = create_engine(f'sqlite:///{gpkg_path}')
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", con=engine)
        print("Column names in the DataFrame:", df.columns)  # Print the column names

        # Set the data types for specific columns
        df['susceptibility'] = df['susceptibility'].astype('int64', errors='ignore')
        df['importance'] = df['importance'].astype('int64', errors='ignore')
        df['sensitivity'] = df['sensitivity'].astype('int64', errors='ignore')

    except exc.SQLAlchemyError as e:
        messagebox.showerror("Database Error", f"Failed to load data: {e}")
        return

    for widget in frame.winfo_children():
        widget.destroy()

    add_column_labels()
    entries = []
    for i, row in df.iterrows():
        add_data_row(i, row)

# Function to add column labels
def add_column_labels():
    ttk.Label(frame, text="Dataset").grid(row=0, column=0, padx=5)
    ttk.Label(frame, text="Susceptibility").grid(row=0, column=1, padx=5)
    ttk.Label(frame, text="Importance").grid(row=0, column=2, padx=5)
    ttk.Label(frame, text="Sensitivity").grid(row=0, column=3, padx=5)

# Function to add a data row
def add_data_row(i, row):
    global entries
    ttk.Label(frame, text=row['name_original']).grid(row=i+1, column=0, padx=5)  # Corrected column name

    susceptibility_entry = ttk.Entry(frame, width=5, validate='key', validatecommand=vcmd)
    susceptibility_entry.insert(0, row['susceptibility'])
    susceptibility_entry.grid(row=i+1, column=1, padx=5)
    susceptibility_entry.bind('<KeyRelease>', lambda event, index=i: calculate_sensitivity(index))

    importance_entry = ttk.Entry(frame, width=5, validate='key', validatecommand=vcmd)
    importance_entry.insert(0, row['importance'])
    importance_entry.grid(row=i+1, column=2, padx=5)
    importance_entry.bind('<KeyRelease>', lambda event, index=i: calculate_sensitivity(index))

    sensitivity_label = ttk.Label(frame, text=str(row['sensitivity']))
    sensitivity_label.grid(row=i+1, column=3, padx=5)

    entries.append({
        'susceptibility': susceptibility_entry,
        'importance': importance_entry,
        'sensitivity': sensitivity_label
    })

# Function to save changes to the geopackage
# Function to save changes to the geopackage
def save_to_gpkg():
    engine = create_engine(f'sqlite:///{gpkg_path}')
    try:
        # Convert 'date_import' to datetime format
        df['date_import'] = pd.to_datetime(df['date_import'], errors='coerce')

        # Handle null values if necessary
        # df['date_import'] = df['date_import'].fillna(some_default_date)

        print("Data types before saving:", df.dtypes)  # Check data types before saving

        # Specify SQLAlchemy data types for columns
        data_types = {
            'id': Integer,
            'name_original': String,
            'name_fromuser': String,
            'date_import': DateTime,
            'bounding_box_geom': String,
            'total_asset_objects': Integer,
            'susceptibility': Integer,
            'importance': Integer,
            'sensitivity': Integer
            # Add other columns if needed
        }

        df.to_sql(table_name, con=engine, if_exists='replace', index=False, dtype=data_types)
        messagebox.showinfo("Success", "Data successfully saved to GeoPackage.")
    except exc.SQLAlchemyError as e:
        messagebox.showerror("Database Error", f"Failed to save data: {e}")

# Function to close the application
def close_application():
    root.destroy()

# Initialize the main window with a larger size
root = tk.Tk()
root.title("Data Editor")
root.geometry("600x1000")  # Set window size (width x height)

# Register the validation command
vcmd = (root.register(validate_integer), '%P')

# Create a frame for the data rows
frame = ttk.Frame(root)
frame.pack(padx=10, pady=10, fill="both", expand=True)

# Paths for the geopackage and the table name
gpkg_path = 'output/mesa.gpkg'
table_name = 'tbl_asset_group'

# Load data and populate UI
load_data()

# Add buttons for saving data and closing the application
save_button = ttk.Button(root, text="Save to Geopackage", command=save_to_gpkg)
save_button.pack(side='left', padx=10, pady=10)

refresh_button = ttk.Button(root, text="Refresh Data", command=load_data)
refresh_button.pack(side='left', padx=10, pady=10)

close_button = ttk.Button(root, text="Close", command=close_application)
close_button.pack(side='right', padx=10, pady=10)

# Start the GUI event loop
root.mainloop()

# Functions from 06_tbl_mesa_stacked.py
import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import geopandas as gpd
import pandas as pd
import configparser
import datetime
import os

# Read the configuration file
def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config

# Logging function to write to the GUI log
def log_to_gui(log_widget, message):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted_message = f"{timestamp} - {message}"
    log_widget.insert(tk.END, formatted_message + "\n")
    log_widget.see(tk.END)

    with open("log.txt", "a") as log_file:
        log_file.write(formatted_message + "\n")

# Function to perform intersection with geocode data
def intersection_with_geocode_data(asset_df, geocode_df, geom_type, log_widget):
    log_to_gui(log_widget, f"Processing {geom_type} intersections")
    asset_filtered = asset_df[asset_df.geometry.geom_type == geom_type]

    if asset_filtered.empty:
        return gpd.GeoDataFrame()

    return gpd.sjoin(geocode_df, asset_filtered, how='inner', op='intersects')

# Main function for processing data
def main(log_widget, progress_var, gpkg_file):
    asset_data = gpd.read_file(gpkg_file, layer='tbl_asset_object')
    geocode_data = gpd.read_file(gpkg_file, layer='tbl_geocode_object')
    asset_group_data = gpd.read_file(gpkg_file, layer='tbl_asset_group')

    # Merge asset group data with asset data
    asset_data = asset_data.merge(asset_group_data[['id', 'total_asset_objects', 'importance', 'susceptibility', 'sensitivity']], 
                                  left_on='ref_asset_group', right_on='id', how='left')

    point_intersections = intersection_with_geocode_data(asset_data, geocode_data, 'Point', log_widget)
    line_intersections = intersection_with_geocode_data(asset_data, geocode_data, 'LineString', log_widget)
    polygon_intersections = intersection_with_geocode_data(asset_data, geocode_data, 'Polygon', log_widget)

    intersected_data = pd.concat([point_intersections, line_intersections, polygon_intersections])

    intersected_data.to_file(gpkg_file, layer='tbl_stacked', driver='GPKG')
    log_to_gui(log_widget, "Data processing completed.")
    progress_var.set(100)

# Thread function to run main without freezing GUI
def run_main(log_widget, progress_var, gpkg_file):
    main(log_widget, progress_var, gpkg_file)

# Function to close the application
def close_application(root):
    root.destroy()

# Create the user interface
root = tk.Tk()
root.title("Geocode Intersection Utility")

# Create a log widget
log_widget = scrolledtext.ScrolledText(root, height=10)
log_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Create a progress bar
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root, orient="horizontal", length=200, mode="determinate", variable=progress_var)
progress_bar.pack(pady=5, fill=tk.X)

# Add buttons for operations
run_btn = tk.Button(root, text="Run Analysis", command=lambda: threading.Thread(
    target=run_main, args=(log_widget, progress_var, gpkg_file), daemon=True).start())
run_btn.pack(pady=5, fill=tk.X)

close_btn = tk.Button(root, text="Close", command=lambda: close_application(root))
close_btn.pack(pady=5, fill=tk.X)

# Load configuration settings
config_file = 'config.ini'
config = read_config(config_file)
gpkg_file = config['DEFAULT']['gpkg_file']

root.mainloop()

# Functions from user_interface.py
import tkinter as tk
from tkinter import messagebox
import subprocess
import webbrowser  # Import the webbrowser module
from PIL import Image, ImageTk

# Function to open a web link
def open_link(url):
    webbrowser.open_new_tab(url)

# Function to load and display the image
def display_image(bottom_frame):
    # Load the image from the specified path
    image_path = 'data_resources/mesa_illustration.png'  # Update this path if necessary
    image = Image.open(image_path)
    
    # Resize the image to 200x200 pixels using LANCZOS filter (previously ANTIALIAS)
    image = image.resize((200, 200), Image.Resampling.LANCZOS)
    
    # Convert the image to a PhotoImage object
    photo = ImageTk.PhotoImage(image)
    
    # Create a Label widget to display the image and center it in the bottom frame
    label = tk.Label(bottom_frame, image=photo)
    label.image = photo  # Keep a reference to the image object
    label.pack(side='bottom', pady=10)  # Center the image

def import_assets():
    try:
        subprocess.run(["python", "01_import_asset_objects.py"], check=True)
        messagebox.showinfo("Import assets", "Import assets script executed.")
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Failed to execute import assets script.")

def import_grids():
    try:
        subprocess.run(["python", "01_import_geocode_objects.py"], check=True)
        messagebox.showinfo("Import grids (geocode objects)", "Import grids script executed.")
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Failed to execute import grids script.")

def edit_susceptibilitiesandimportance():
    try:
        subprocess.run(["python", "04_edit_input.py"], check=True)
        messagebox.showinfo("Edit asset data", "Edit susceptibilities and importance.")
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Failed to edit.")
    
def view_statistics():
    messagebox.showinfo("View Statistics", "View statistics script executed.")

def process_data():
    try:
        subprocess.run(["python", "06_tbl_mesa_stacked.py"], check=True)
        messagebox.showinfo("Process data", "Spatial data processing")
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Failed to process.")

def export_package():
    messagebox.showinfo("Export Package", "Export package script executed.")

def exit_program():
    root.destroy()

# Setup the main Tkinter window
root = tk.Tk()
root.title("MESA 4")

# Set the initial size of the window (width x height)
root.geometry("540x540")  # Adjust the width and height as needed

# Create a top frame for text and links
top_frame = tk.Frame(root)
top_frame.pack(fill='x', expand=False, pady=10)

# Add text and link to the top frame
info_text = tk.Label(top_frame, text="Read more about the MESA method and tools", font=("Calibri", 10))
info_text.pack(side='left')

link_text = tk.Label(top_frame, text="here ", font=("Calibri", 10, "underline"), fg="blue", cursor="hand2")
link_text.pack(side='left')
link_text.bind("<Button-1>", lambda e: open_link("https://www.mesamethod.org/wiki/Main_Page"))


# Main frame
main_frame = tk.Frame(root)
main_frame.pack(fill='both', expand=True, pady=20)
main_frame.grid_columnconfigure(0, weight=1)
main_frame.grid_columnconfigure(1, weight=1)


# Left panel
left_panel = tk.Frame(main_frame)
left_panel.grid(row=0, column=0, sticky="ew", padx=20)
left_panel.grid_rowconfigure(0, weight=1)
left_panel.grid_rowconfigure(1, weight=1)
left_panel.grid_rowconfigure(2, weight=1)
left_panel.grid_rowconfigure(3, weight=1)  # Extra row for spacing

# Right panel
right_panel = tk.Frame(main_frame)
right_panel.grid(row=0, column=1, sticky="ew", padx=20)
right_panel.grid_rowconfigure(0, weight=1)
right_panel.grid_rowconfigure(1, weight=1)
right_panel.grid_rowconfigure(2, weight=1)
right_panel.grid_rowconfigure(3, weight=1)  # Extra row for spacing to make it visually apealing

bottom_frame = tk.Frame(root)
bottom_frame.pack(fill='x', expand=False)

# Button width
button_width = 20  # Adjust as needed
button_padx = 20  # Padding around buttons on x-axis
button_pady = 10  # Padding around buttons on y-axis

# Add buttons to left panel with spacing between buttons
import_assets_btn = tk.Button(left_panel, text="Import Assets", command=import_assets, width=button_width)
import_assets_btn.grid(row=0, column=0, padx=button_padx, pady=button_pady)

import_grids_btn = tk.Button(left_panel, text="Import Grids", command=import_grids, width=button_width)
import_grids_btn.grid(row=1, column=0, padx=button_padx, pady=button_pady)

view_statistics_btn = tk.Button(left_panel, text="Set up processing", command=edit_susceptibilitiesandimportance, width=button_width)
view_statistics_btn.grid(row=2, column=0, padx=button_padx, pady=button_pady)

# Call the function to display the image in the bottom frame
display_image(bottom_frame)

# Add buttons to right panel with spacing between buttons
process_data_btn = tk.Button(right_panel, text="Process Data", command=process_data, width=button_width)
process_data_btn.grid(row=0, column=0, padx=button_padx, pady=button_pady)

export_package_btn = tk.Button(right_panel, text="Export Package", command=export_package, width=button_width)
export_package_btn.grid(row=1, column=0, padx=button_padx, pady=button_pady)

# Exit button
exit_btn = tk.Button(main_frame, text="Exit", command=exit_program, width=button_width)
exit_btn.grid(row=1, column=0, columnspan=2, pady=button_pady)

# Start the GUI event loop
root.mainloop()

# Main Execution Logic (if any specific main logic is required)

