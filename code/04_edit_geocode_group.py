import tkinter as tk

import locale
try:
    locale.setlocale(locale.LC_ALL, 'de_DE.utf8')  # For US English, adjust as needed
except locale.Error:
    locale.setlocale(locale.LC_ALL, '') 

from tkinter import scrolledtext, ttk
from tkinter import messagebox, scrolledtext, ttk
import configparser
import geopandas as gpd
from sqlalchemy import create_engine

import ttkbootstrap as ttk  # Import ttkbootstrap
from ttkbootstrap.constants import *

# # # # # # # # # # # # # # 
# Shared/general functions

# Read the configuration file
def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config

# # # # # # # # # # # # # # 
# Core functions

def update_records():
    global df, records
    for record in records:
        row_id      = record['id']
        name        = record['name'].get()
        title_user  = record['title_user'].get()
        description = record['description'].get()

        # Update the DataFrame
        df.loc[df['id'] == row_id, 'name'] = name
        df.loc[df['id'] == row_id, 'title_user'] = title_user
        df.loc[df['id'] == row_id, 'description'] = description

def save_changes():
    update_records()
    save_spatial_data()

# Function to load spatial data from the database
def load_spatial_data(gpkg_file):
    engine = create_engine(f'sqlite:///{gpkg_file}')  # Adjust as per your database
    # Use Geopandas to load a GeoDataFrame
    gdf = gpd.read_file(gpkg_file, layer='tbl_geocode_group')
    return gdf


# Function to save spatial data to the database
def save_spatial_data():
    global df  # Access the global DataFrame
    try:
        engine = create_engine(f'sqlite:///{gpkg_file}')  # Adjust as per your database
        # Use Geopandas to save the GeoDataFrame
        df.to_file(gpkg_file, layer='tbl_geocode_group', driver='GPKG', if_exists='replace')
        messagebox.showinfo("Success", "Spatial data saved successfully")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save spatial data: {e}")


# Your existing code for creating the Tkinter interface would remain the same
# Just ensure to call load_spatial_data() instead of load_data()
# and save_spatial_data(gdf) instead of save_data()


# Function to close the application
def exit_application():
    root.destroy()

# Create the user interface
root = ttk.Window(themename='superhero')  # Use ttkbootstrap Window
root.title("Import assets")


# Load configuration settings
config_file = 'config.ini'
config = read_config(config_file)
input_folder_asset = config['DEFAULT']['input_folder_asset']
input_folder_geocode = config['DEFAULT']['input_folder_geocode']
gpkg_file = config['DEFAULT']['gpkg_file']

# Load data
df = load_spatial_data(gpkg_file)

# Create a frame for the editable fields
edit_frame = tk.Frame(root, padx=5, pady=5)
edit_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Column headers
tk.Label(edit_frame, text="ID").grid(row=0, column=0, sticky='w')
tk.Label(edit_frame, text="GIS Name").grid(row=0, column=1, sticky='w')
tk.Label(edit_frame, text="Layer name").grid(row=0, column=2, sticky='w')
tk.Label(edit_frame, text="User Title").grid(row=0, column=3, sticky='w')
tk.Label(edit_frame, text="Description").grid(row=0, column=4, sticky='w')


# Store references to the variables
records = []

# Create labels and entry widgets for each record
for idx, row in df.iterrows():
    row_number = idx + 1  # Adjust for the header row

    tk.Label(edit_frame, text=row['id']).grid(row=row_number, column=0, sticky='w')
    
    # Read-only field for 'name_gis'
    tk.Label(edit_frame, text=row['name_gis']).grid(row=row_number, column=1, sticky='w')
    
    
    # Editable field for 'user title'
    name_var = tk.StringVar(value=row.get('name', ''))
    tk.Entry(edit_frame, textvariable=name_var, width=30).grid(row=row_number, column=2, sticky='w')
    
    # Editable field for 'user title'
    user_title_var = tk.StringVar(value=row.get('title_user', ''))
    tk.Entry(edit_frame, textvariable=user_title_var, width=30).grid(row=row_number, column=3, sticky='w')
    
    # Editable field for 'description'
    description_var = tk.StringVar(value=row['description'])
    tk.Entry(edit_frame, textvariable=description_var, width=70).grid(row=row_number, column=4, sticky='w')

    records.append({'id': row['id'], 'name': name_var, 'title_user': user_title_var, 'description': description_var})



# Information text field
info_label_text = ("After you have imported the geocodes you might want to "
                   "adjust the name of the geocode. You will find their names "
                   "when you open QGIS after having initiated the data processing "
                   "to both the name and the description. You will also find the "
                   "name in the PDF report which this system will generate.")
info_label = tk.Label(root, text=info_label_text, wraplength=600, justify="left")
info_label.pack(padx=10, pady=10)

# Button frame
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# Save button
save_button = ttk.Button(button_frame, text="Save Data", command=save_changes, bootstyle=PRIMARY)
save_button.pack(side=tk.LEFT, padx=10)

# Exit button
exit_button = ttk.Button(button_frame, text="Exit", command=exit_application, bootstyle=WARNING)
exit_button.pack(side=tk.LEFT, padx=10)

# Styling buttons (rounded corners)
style = ttk.Style()
style.configure("TButton", padding=6, relief="flat", background="#ccc")

root.mainloop()
