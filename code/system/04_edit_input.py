import tkinter as tk
import locale
from tkinter import messagebox, ttk
import configparser
import pandas as pd
import geopandas as gpd
import datetime
import ttkbootstrap as ttk  # Import ttkbootstrap
from ttkbootstrap.constants import *
import os
from shapely import wkb
import binascii
import sys


# Setting variables
#
# Define fixed widths for each column
column_widths = [35, 13, 13, 13, 13, 30]

# Define global variable for valid input values
valid_input_values = []

# Global declaration
classification = {}

# Shared/general functions
def read_config(file_name):
    global valid_input_values
    config = configparser.ConfigParser()
    config.read(file_name)
    # Convert valid input values from config to a list of integers
    valid_input_values = list(map(int, config['VALID_VALUES']['valid_input'].split(',')))
    return config


def read_config_classification(file_name):

    config = configparser.ConfigParser()
    config.read(file_name)
    # Clear the existing global classification dictionary before populating it
    classification.clear()
    for section in config.sections():
        if section in ['A', 'B', 'C', 'D', 'E']:  # Make sure we're only dealing with your classification sections
            range_str = config[section]['range']
            description = config[section].get('description', '')  # Safely get the description if it exists
            start, end = map(int, range_str.split('-'))
            classification[section] = {
                'range': range(start, end + 1),  # Adjust the end value to make the range inclusive
                'description': description
            }
    return classification


# Updated validation function
def validate_input_value(P):
    if P.isdigit() and (int(P) in valid_input_values or P == ""):
        return True
    return False


def determine_category(sensitivity):
    for category, info in classification.items():
        if sensitivity in info['range']:
            return category, info['description']
    return '', ''  


# Function to safely load WKB or indicate error
def load_wkb_or_flag(wkb_data):
    if wkb_data is None or wkb_data == '':
        log_to_file("No WKB data found.")
        return None
    try:
        # Assuming the WKB data might be in hexadecimal string format
        if isinstance(wkb_data, str):
            wkb_data = binascii.unhexlify(wkb_data)
        return wkb.loads(wkb_data)
    except binascii.Error as e:
        log_to_file(f"Failed to convert hex: {e}")
        return None
    except Exception as e:
        log_to_file(f"Failed to load WKB: {e}")
        return None


def calculate_sensitivity(entry_susceptibility, entry_importance, index, entries, df_assetgroup):
    try:
        susceptibility  = int(entry_susceptibility.get())
        importance      = int(entry_importance.get())
        sensitivity     = susceptibility * importance
        sensitivity_code, sensitivity_description = determine_category(sensitivity)

        # Update the entries list and DataFrame
        entries[index]['sensitivity']['text']               = str(sensitivity)
        entries[index]['sensitivity_code']['text']          = sensitivity_code
        entries[index]['sensitivity_description']['text']   = sensitivity_description

        df_assetgroup.at[index, 'susceptibility']           = susceptibility
        df_assetgroup.at[index, 'importance']               = importance
        df_assetgroup.at[index, 'sensitivity']              = sensitivity
        df_assetgroup.at[index, 'sensitivity_code']         = sensitivity_code
        df_assetgroup.at[index, 'sensitivity_description']  = sensitivity_description

    except ValueError:
        messagebox.showerror("Input Error", "Enter valid integers for susceptibility and importance.")


def update_all_rows_immediately(entries, df_assetgroup):
    for entry in entries:
        try:
            # Extract susceptibility and importance values from the UI and convert to integers
            susceptibility  = int(entry['susceptibility'].get())
            importance      = int(entry['importance'].get())
            
            # Calculate new sensitivity
            sensitivity = susceptibility * importance
            sensitivity_code, sensitivity_description = determine_category(sensitivity)
            
            # Update the DataFrame
            index = entry['row_index']
            df_assetgroup.at[index, 'susceptibility']           = susceptibility
            df_assetgroup.at[index, 'importance']               = importance
            df_assetgroup.at[index, 'sensitivity']              = sensitivity
            df_assetgroup.at[index, 'sensitivity_code']         = sensitivity_code
            df_assetgroup.at[index, 'sensitivity_description']  = sensitivity_description
            
            # Update geometry if it's included in the entry dictionary
            if 'geom' in entry:
                df_assetgroup.at[index, 'geom'] = entry['geom']

            # Update UI elements with new values
            entry['sensitivity']['text']                = str(sensitivity)
            entry['sensitivity_code']['text']           = sensitivity_code
            entry['sensitivity_description']['text']    = sensitivity_description

        except ValueError as e:
            log_to_file(f"Input Error: {e}")
            continue  # Skip this entry and continue with the next


def load_data(gpkg_file):
    try:
        # Specify the layer name if your GeoPackage contains multiple layers
        layer_name = "tbl_asset_group"
        
        # Read the specified layer directly into a GeoDataFrame
        df_assetgroup = gpd.read_file(gpkg_file, layer=layer_name)
        
        # Verify and set the geometry column if not automatically recognized
        if 'geom' in df_assetgroup.columns and df_assetgroup.geometry.name != 'geom':
            df_assetgroup.set_geometry('geom', inplace=True)
        
        # Initialize columns if they don't exist or if they are completely null
        for col in ['susceptibility', 'importance', 'sensitivity']:
            if col not in df_assetgroup.columns or df_assetgroup[col].isnull().all():
                df_assetgroup[col] = 0
        
        # Check for any geometries that failed to load (if geometries are invalid or missing)
        if df_assetgroup.geometry.isnull().any():
            log_to_file("Some geometries failed to load or are invalid.")
        
        return df_assetgroup

    except Exception as e:
        log_to_file("Failed to load data:", e)
        log_to_file("Database file is missing.")

        return None


def add_data_row(index, row, frame, column_widths, entries, df_assetgroup):
    entry_susceptibility = ttk.Entry(frame, width=column_widths[1])
    entry_susceptibility.insert(0, str(getattr(row, 'susceptibility', '')))
    entry_susceptibility.grid(row=index, column=1, padx=5)

    entry_importance = ttk.Entry(frame, width=column_widths[2])
    entry_importance.insert(0, str(getattr(row, 'importance', '')))
    entry_importance.grid(row=index, column=2, padx=5)

    # Ensure all parameters are correctly included in the lambda function call
    entry_susceptibility.bind('<KeyRelease>', lambda event, ent=entry_susceptibility, imp=entry_importance, idx=index-1: calculate_sensitivity(ent, imp, idx, entries, df_assetgroup))
    entry_importance.bind('<KeyRelease>', lambda event, ent=entry_susceptibility, imp=entry_importance, idx=index-1: calculate_sensitivity(ent, imp, idx, entries, df_assetgroup))

    geom = getattr(row, 'geom', None)

    label_name = ttk.Label(frame, text=getattr(row, 'name_original', ''), anchor='w', width=column_widths[0])
    label_name.grid(row=index, column=0, padx=5, sticky='ew')

    label_sensitivity = ttk.Label(frame, text=str(getattr(row, 'sensitivity', '')), anchor='w', width=column_widths[3])
    label_sensitivity.grid(row=index, column=3, padx=5, sticky='ew')

    label_code = ttk.Label(frame, text=str(getattr(row, 'sensitivity_code', '')), anchor='w', width=column_widths[4])
    label_code.grid(row=index, column=4, padx=5, sticky='ew')

    label_description = ttk.Label(frame, text=str(getattr(row, 'sensitivity_description', '')), anchor='w', width=column_widths[5])
    label_description.grid(row=index, column=5, padx=5, sticky='ew')

    entries.append({
        'row_index': index-1,
        'name': label_name,
        'susceptibility': entry_susceptibility,
        'importance': entry_importance,
        'sensitivity': label_sensitivity,
        'sensitivity_code': label_code,
        'sensitivity_description': label_description,
        'geom': geom
    })


def save_to_gpkg(df_assetgroup, gpkg_file):
    try:
        if not df_assetgroup.empty:
            # Find all geometry columns in the DataFrame
            geom_cols = df_assetgroup.columns[df_assetgroup.dtypes.apply(lambda dtype: dtype == 'geometry')].tolist()
            
            # Set the main geometry column; use the first geometry column found
            if geom_cols:
                main_geom_col = geom_cols[0]  # Use the first geometry column as the main one
            else:
                raise ValueError("No geometry column found in the DataFrame.")
            
            # If there are multiple geometry columns, process them
            if len(geom_cols) > 1:
                log_to_file(f"Warning: Multiple geometry columns found. Converting all but '{main_geom_col}' to WKT.")
                for col in geom_cols:
                    if col != main_geom_col:
                        df_assetgroup[col] = df_assetgroup[col].apply(lambda geom: geom.to_wkt() if geom else None)
                        df_assetgroup.drop(columns=[col], inplace=True)
            
            # Ensure 'main_geom_col' is set as the geometry column
            if df_assetgroup.geometry.name != main_geom_col:
                df_assetgroup.set_geometry(main_geom_col, inplace=True)
            
            # Set CRS if not already set
            if df_assetgroup.crs is None:
                df_assetgroup.set_crs(epsg=workingprojection_epsg, inplace=True)  # Adjust CRS as necessary
            
            # Save to GeoPackage
            df_assetgroup.to_file(filename=gpkg_file, layer='tbl_asset_group', driver='GPKG')
            log_to_file("Data saved successfully to GeoPackage.")
        else:
            log_to_file("GeoDataFrame is empty or missing a geometry column.")
    except Exception as e:
        log_to_file("Failed to save GeoDataFrame:", e)


# Application closes without saving. Not sure if this is the way or if I should add default save on exit.
def close_application():
    save_to_gpkg(df_assetgroup, gpkg_file)
    root.destroy()

def setup_headers(frame, column_widths):
    headers = ["Dataset", "Susceptibility", "Importance", "Sensitivity", "Code", "Description"]
    for idx, header in enumerate(headers):
        label = ttk.Label(frame, text=header, anchor='w', width=column_widths[idx])
        label.grid(row=0, column=idx, padx=5, pady=5, sticky='ew')
    return frame


def update_df_assetgroup(entries):
    for entry in entries:
        index = entry['row_index']
        # Ensure other data is updated
        df_assetgroup.at[index, 'susceptibility']           = entry['susceptibility'].get()
        df_assetgroup.at[index, 'importance']               = entry['importance'].get()
        df_assetgroup.at[index, 'sensitivity']              = entry['sensitivity']['text']
        df_assetgroup.at[index, 'sensitivity_code']         = entry['sensitivity_code']['text']
        df_assetgroup.at[index, 'sensitivity_description']  = entry['sensitivity_description']['text']
        # Update the geometry
        df_assetgroup.at[index, 'geom'] = entry['geom']


def create_scrollable_area(root):
    # Create a new frame to contain the canvas and the scrollbar
    scrollable_frame = ttk.Frame(root)

    # Create the canvas and scrollbar
    canvas = tk.Canvas(scrollable_frame)
    scrollbar_y = ttk.Scrollbar(scrollable_frame, orient="vertical", command=canvas.yview)

    # Configure the scrollbar with the canvas
    canvas.configure(yscrollcommand=scrollbar_y.set)

    # Bind the configure event to adjust the scroll region
    canvas.bind('<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox("all")))

    # Pack the scrollbar and canvas in the frame
    canvas.pack(side=tk.LEFT, fill="both", expand=True)
    scrollbar_y.pack(side=tk.RIGHT, fill="y")

    # Pack the scrollable_frame in the root window
    scrollable_frame.pack(side=tk.TOP, fill="both", expand=True)

    # Create a frame inside the canvas which will contain the actual widgets
    frame = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor="nw")

    return canvas, frame


def setup_ui_elements(root, df_assetgroup, column_widths):
    canvas, frame = create_scrollable_area(root)
    entries = []  # Initialize here
    setup_headers(frame, column_widths)

    if df_assetgroup is not None and not df_assetgroup.empty:
        for i, row in enumerate(df_assetgroup.itertuples(), start=1):
            add_data_row(i, row, frame, column_widths, entries, df_assetgroup)  # Ensure entries are passed and used correctly
        frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))
    else:
        log_to_file("No data to display.")

    return canvas, frame, entries  # Return entries too


def increment_stat_value(config_file, stat_name, increment_value):
    # Check if the config file exists
    if not os.path.isfile(config_file):
        log_to_file(f"Configuration file {config_file} not found.")
        return
    
    # Read the entire config file to preserve the layout and comments
    with open(config_file, 'r') as file:
        lines = file.readlines()
    
    # Initialize a flag to check if the variable was found and updated
    updated = False
    
    # Update the specified variable's value if it exists
    for i, line in enumerate(lines):
        if line.strip().startswith(f'{stat_name} ='):
            # Extract the current value, increment it, and update the line
            parts = line.split('=')
            if len(parts) == 2:
                current_value = parts[1].strip()
                try:
                    # Attempt to convert the current value to an integer and increment it
                    new_value = int(current_value) + increment_value
                    lines[i] = f"{stat_name} = {new_value}\n"
                    updated = True
                    break
                except ValueError:
                    # Handle the case where the conversion fails
                    log_to_file(f"Error: Current value of {stat_name} is not an integer.")
                    return
    
    # Write the updated content back to the file if the variable was found and updated
    if updated:
        with open(config_file, 'w') as file:
            file.writelines(lines)


# Logging function to write to the GUI log
def log_to_file(message):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted_message = f"{timestamp} - {message}"
    with open("log.txt", "a") as log_file:
        log_file.write(formatted_message + "\n")

#####################################################################################
#  Main
#

# Load configuration settings and data
config_file             = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')
config                  = read_config(config_file)

gpkg_file               = config['DEFAULT']['gpkg_file']

ttk_bootstrap_theme     = config['DEFAULT']['ttk_bootstrap_theme']
workingprojection_epsg  = config['DEFAULT']['workingprojection_epsg']
                                       
valid_input_values      = list(map(int, config['VALID_VALUES']['valid_input'].split(',')))
classification          = read_config_classification(config_file)

increment_stat_value(config_file, 'mesa_stat_setup', increment_value=1)

if __name__ == "__main__":
    root = ttk.Window(themename=ttk_bootstrap_theme)
    root.title("Set up processing")
    root.geometry("900x800")

    df_assetgroup = load_data(gpkg_file)

    if df_assetgroup is None:
        log_to_file("Failed to load the GeoDataFrame. Check the GeoPackage file and the data integrity.")
        sys.exit(1)  # Exit if the data could not be loaded, adjust handling as needed

    canvas, frame, entries = setup_ui_elements(root, df_assetgroup, column_widths)  # Ensure entries are received here
    
    update_all_rows_immediately(entries, df_assetgroup)  # Now pass entries to this function to update all rows

    # Setup the rest of the UI components
    info_text = "This is where you register values for susceptibility and importance. Ensure all values are correctly filled."
    info_label = tk.Label(root, text=info_text, wraplength=600, justify="center")
    info_label.pack(padx=10, pady=10)

    save_button = ttk.Button(root, text="Save", command=lambda: save_to_gpkg(df_assetgroup, gpkg_file), bootstyle=PRIMARY)
    save_button.pack(side='left', padx=10, pady=10)

    close_button = ttk.Button(root, text="Exit", command=close_application, bootstyle=WARNING)
    close_button.pack(side='right', padx=10, pady=10)

    root.mainloop()
