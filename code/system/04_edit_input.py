import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import configparser
import pandas as pd
import geopandas as gpd
import datetime
import argparse
import ttkbootstrap as ttk  # Import ttkbootstrap
from ttkbootstrap.constants import *
import os
import sys

# Setting variables
column_widths = [35, 13, 13, 13, 13, 30]
valid_input_values = []
classification = {}

# Shared/general functions
def read_config(file_name):
    global valid_input_values
    config = configparser.ConfigParser()
    config.read(file_name)
    valid_input_values = list(map(int, config['VALID_VALUES']['valid_input'].split(',')))
    return config

def read_config_classification(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    classification.clear()
    for section in config.sections():
        if section in ['A', 'B', 'C', 'D', 'E']:  
            range_str = config[section]['range']
            description = config[section].get('description', '')  
            start, end = map(int, range_str.split('-'))
            classification[section] = {
                'range': range(start, end + 1),  
                'description': description
            }
    return classification

def validate_input_value(P):
    if P.isdigit() and (int(P) in valid_input_values or P == ""):
        return True
    return False

def determine_category(sensitivity):
    for category, info in classification.items():
        if sensitivity in info['range']:
            return category, info['description']
    return '', ''  

def calculate_sensitivity(entry_importance, entry_susceptibility, index, entries, df_assetgroup):
    try:
        importance = int(entry_importance.get())
        susceptibility = int(entry_susceptibility.get())
        sensitivity = importance * susceptibility
        sensitivity_code, sensitivity_description = determine_category(sensitivity)

        entries[index]['sensitivity']['text'] = str(sensitivity)
        entries[index]['sensitivity_code']['text'] = sensitivity_code
        entries[index]['sensitivity_description']['text'] = sensitivity_description

        df_assetgroup.at[index, 'importance'] = importance
        df_assetgroup.at[index, 'susceptibility'] = susceptibility
        df_assetgroup.at[index, 'sensitivity'] = sensitivity
        df_assetgroup.at[index, 'sensitivity_code'] = sensitivity_code
        df_assetgroup.at[index, 'sensitivity_description'] = sensitivity_description

    except ValueError:
        messagebox.showerror("Input Error", "Enter valid integers for susceptibility and importance.")

def update_all_rows_immediately(entries, df_assetgroup):
    for entry in entries:
        try:
            importance = int(entry['importance'].get())
            susceptibility = int(entry['susceptibility'].get())
            
            sensitivity = importance * susceptibility
            sensitivity_code, sensitivity_description = determine_category(sensitivity)
            
            index = entry['row_index']
            df_assetgroup.at[index, 'importance'] = importance
            df_assetgroup.at[index, 'susceptibility'] = susceptibility
            df_assetgroup.at[index, 'sensitivity'] = sensitivity
            df_assetgroup.at[index, 'sensitivity_code'] = sensitivity_code
            df_assetgroup.at[index, 'sensitivity_description'] = sensitivity_description
            
            if 'geom' in entry:
                df_assetgroup.at[index, 'geom'] = entry['geom']

            entry['sensitivity']['text'] = str(sensitivity)
            entry['sensitivity_code']['text'] = sensitivity_code
            entry['sensitivity_description']['text'] = sensitivity_description

        except ValueError as e:
            log_to_file(f"Input Error: {e}")
            continue

def load_data(gpkg_file):
    try:
        layer_name = "tbl_asset_group"
        df_assetgroup = gpd.read_file(gpkg_file, layer=layer_name)
        
        if 'geom' in df_assetgroup.columns and df_assetgroup.geometry.name != 'geom':
            df_assetgroup.set_geometry('geom', inplace=True)
        
        for col in ['importance', 'susceptibility', 'sensitivity']:
            if col not in df_assetgroup.columns or df_assetgroup[col].isnull().all():
                df_assetgroup[col] = 0
        
        if df_assetgroup.geometry.isnull().any():
            log_to_file("Some geometries failed to load or are invalid.")
        
        return df_assetgroup

    except Exception as e:
        log_to_file(f"Failed to load data: {e}")
        return None

def add_data_row(index, row, frame, column_widths, entries, df_assetgroup):
    entry_importance = ttk.Entry(frame, width=column_widths[1])
    entry_importance.insert(0, str(getattr(row, 'importance', '')))
    entry_importance.grid(row=index, column=1, padx=5)

    entry_susceptibility = ttk.Entry(frame, width=column_widths[2])
    entry_susceptibility.insert(0, str(getattr(row, 'susceptibility', '')))
    entry_susceptibility.grid(row=index, column=2, padx=5)

    entry_importance.bind('<KeyRelease>', lambda event, imp=entry_importance, sus=entry_susceptibility, idx=index-1: calculate_sensitivity(imp, sus, idx, entries, df_assetgroup))
    entry_susceptibility.bind('<KeyRelease>', lambda event, imp=entry_importance, sus=entry_susceptibility, idx=index-1: calculate_sensitivity(imp, sus, idx, entries, df_assetgroup))

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
        'importance': entry_importance,
        'susceptibility': entry_susceptibility,
        'sensitivity': label_sensitivity,
        'sensitivity_code': label_code,
        'sensitivity_description': label_description,
        'geom': geom
    })

def save_to_gpkg(df_assetgroup, gpkg_file):
    try:
        if not df_assetgroup.empty:
            geom_cols = df_assetgroup.columns[df_assetgroup.dtypes.apply(lambda dtype: dtype == 'geometry')].tolist()
            
            if geom_cols:
                main_geom_col = geom_cols[0]  
            else:
                raise ValueError("No geometry column found in the DataFrame.")
            
            if len(geom_cols) > 1:
                log_to_file(f"Warning: Multiple geometry columns found. Converting all but '{main_geom_col}' to WKT.")
                for col in geom_cols:
                    if col != main_geom_col:
                        df_assetgroup[col] = df_assetgroup[col].apply(lambda geom: geom.to_wkt() if geom else None)
                        df_assetgroup.drop(columns=[col], inplace=True)
            
            if df_assetgroup.geometry.name != main_geom_col:
                df_assetgroup.set_geometry(main_geom_col, inplace=True)
            
            if df_assetgroup.crs is None:
                df_assetgroup.set_crs(epsg=workingprojection_epsg, inplace=True)
            
            df_assetgroup.to_file(filename=gpkg_file, layer='tbl_asset_group', driver='GPKG')
            log_to_file("Data saved successfully to GeoPackage.")
        else:
            log_to_file("GeoDataFrame is empty or missing a geometry column.")
    except Exception as e:
        log_to_file(f"Failed to save GeoDataFrame: {e}")

def save_to_excel(df_assetgroup, excel_file):
    try:
        columns_to_save = ['name_original', 'susceptibility', 'importance']
        df_to_save = df_assetgroup[columns_to_save]
        df_to_save.to_excel(excel_file, index=False)
        log_to_file("Selected data saved successfully to Excel.")
    except Exception as e:
        log_to_file(f"Failed to save data to Excel: {e}")

def load_from_excel(excel_file, df_assetgroup):
    try:
        df_excel = pd.read_excel(excel_file)
        log_to_file("Data loaded successfully from Excel.")
        
        for _, row in df_excel.iterrows():
            name_original = row['name_original']
            if name_original in df_assetgroup['name_original'].values:
                idx = df_assetgroup[df_assetgroup['name_original'] == name_original].index[0]
                
                df_assetgroup.loc[idx, 'susceptibility'] = row['susceptibility']
                df_assetgroup.loc[idx, 'importance'] = row['importance']
                
                importance = int(df_assetgroup.at[idx, 'importance'])
                susceptibility = int(df_assetgroup.at[idx, 'susceptibility'])
                sensitivity = importance * susceptibility
                sensitivity_code, sensitivity_description = determine_category(sensitivity)
                
                df_assetgroup.at[idx, 'sensitivity'] = sensitivity
                df_assetgroup.at[idx, 'sensitivity_code'] = sensitivity_code
                df_assetgroup.at[idx, 'sensitivity_description'] = sensitivity_description
            else:
                log_to_file(f"Warning: {name_original} not found in the database. Skipping this row.")
        
        df_assetgroup.fillna({'susceptibility': 0, 'importance': 0, 'sensitivity': 0}, inplace=True)
        
        return df_assetgroup

    except Exception as e:
        log_to_file(f"Failed to load data from Excel: {e}")
        return df_assetgroup

def close_application():
    save_to_gpkg(df_assetgroup, gpkg_file)
    root.destroy()

def setup_headers(frame, column_widths):
    headers = ["Dataset", "Importance", "Susceptibility", "Sensitivity", "Code", "Description"]
    for idx, header in enumerate(headers):
        label = ttk.Label(frame, text=header, anchor='w', width=column_widths[idx])
        label.grid(row=0, column=idx, padx=5, pady=5, sticky='ew')
    return frame

def update_df_assetgroup(entries):
    for entry in entries:
        index = entry['row_index']
        df_assetgroup.at[index, 'importance'] = entry['importance'].get()
        df_assetgroup.at[index, 'susceptibility'] = entry['susceptibility'].get()
        df_assetgroup.at[index, 'sensitivity'] = entry['sensitivity']['text']
        df_assetgroup.at[index, 'sensitivity_code'] = entry['sensitivity_code']['text']
        df_assetgroup.at[index, 'sensitivity_description'] = entry['sensitivity_description']['text']
        df_assetgroup.at[index, 'geom'] = entry['geom']

def create_scrollable_area(root):
    scrollable_frame = ttk.Frame(root)
    canvas = tk.Canvas(scrollable_frame)
    scrollbar_y = ttk.Scrollbar(scrollable_frame, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar_y.set)
    canvas.bind('<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.pack(side=tk.LEFT, fill="both", expand=True)
    scrollbar_y.pack(side=tk.RIGHT, fill="y")
    scrollable_frame.pack(side=tk.TOP, fill="both", expand=True)
    frame = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor="nw")
    return canvas, frame

def setup_ui_elements(root, df_assetgroup, column_widths):
    canvas, frame = create_scrollable_area(root)
    entries = []
    setup_headers(frame, column_widths)

    if df_assetgroup is not None and not df_assetgroup.empty:
        for i, row in enumerate(df_assetgroup.itertuples(), start=1):
            add_data_row(i, row, frame, column_widths, entries, df_assetgroup)
        frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))
    else:
        log_to_file("No data to display.")

    return canvas, frame, entries

def increment_stat_value(config_file, stat_name, increment_value):
    if not os.path.isfile(config_file):
        log_to_file(f"Configuration file {config_file} not found.")
        return
    with open(config_file, 'r') as file:
        lines = file.readlines()
    updated = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f'{stat_name} ='):
            parts = line.split('=')
            if len(parts) == 2:
                current_value = parts[1].strip()
                try:
                    new_value = int(current_value) + increment_value
                    lines[i] = f"{stat_name} = {new_value}\n"
                    updated = True
                    break
                except ValueError:
                    log_to_file(f"Error: Current value of {stat_name} is not an integer.")
                    return
    if updated:
        with open(config_file, 'w') as file:
            file.writelines(lines)

def log_to_file(message):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted_message = f"{timestamp} - {message}"
    log_destination_file = os.path.join(original_working_directory, "log.txt")
    with open(log_destination_file, "a") as log_file:
        log_file.write(formatted_message + "\n")

def handle_load_from_excel():
    global df_assetgroup
    input_folder = os.path.join(original_working_directory, "input")
    excel_file = filedialog.askopenfilename(title="Select Excel File", initialdir=input_folder, filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")])
    if excel_file:
        df_assetgroup = load_from_excel(excel_file, df_assetgroup)  # Load data and update df_assetgroup
        
        for entry in entries:
            name_original = entry['name'].cget("text")
            if name_original in df_assetgroup['name_original'].values:
                index = df_assetgroup[df_assetgroup['name_original'] == name_original].index[0]
                
                entry['importance'].delete(0, tk.END)
                entry['importance'].insert(0, str(df_assetgroup.at[index, 'importance']))
                
                entry['susceptibility'].delete(0, tk.END)
                entry['susceptibility'].insert(0, str(df_assetgroup.at[index, 'susceptibility']))

                importance = int(df_assetgroup.at[index, 'importance'])
                susceptibility = int(df_assetgroup.at[index, 'susceptibility'])
                sensitivity = importance * susceptibility
                sensitivity_code, sensitivity_description = determine_category(sensitivity)
                
                entry['sensitivity']['text'] = str(sensitivity)
                entry['sensitivity_code']['text'] = sensitivity_code
                entry['sensitivity_description']['text'] = sensitivity_description
            else:
                log_to_file(f"Warning: {name_original} not found in the Excel file. Skipping this entry.")
        
        root.update_idletasks()
        log_to_file("Data loaded and UI updated from Excel.")

def handle_save_to_excel():
    global df_assetgroup
    input_folder = os.path.join(original_working_directory, "input")
    excel_file = filedialog.asksaveasfilename(title="Save Excel File", initialdir=input_folder, defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")])
    if excel_file:
        save_to_excel(df_assetgroup, excel_file)

def handle_save_to_gpkg():
    global df_assetgroup
    input_folder = os.path.join(original_working_directory, "input")
    gpkg_file = filedialog.asksaveasfilename(title="Save GeoPackage File", initialdir=input_folder, defaultextension=".gpkg", filetypes=[("GeoPackage Files", "*.gpkg"), ("All Files", "*.*")])
    if gpkg_file:
        save_to_gpkg(df_assetgroup, gpkg_file)

#####################################################################################
#  Main
#

parser = argparse.ArgumentParser(description='Slave script')
parser.add_argument('--original_working_directory', required=False, help='Path to running folder')
args = parser.parse_args()
original_working_directory = args.original_working_directory

if original_working_directory is None or original_working_directory == '':
    original_working_directory = os.getcwd()
    if str("system") in str(original_working_directory):
        original_working_directory = os.path.join(os.getcwd(),'../')

config_file = os.path.join(original_working_directory, "system/config.ini")
gpkg_file = os.path.join(original_working_directory, "output/mesa.gpkg")
excel_file = os.path.join(original_working_directory, "input/settings.xlsx")

config = read_config(config_file)
ttk_bootstrap_theme = config['DEFAULT']['ttk_bootstrap_theme']
workingprojection_epsg = config['DEFAULT']['workingprojection_epsg']
                                       
valid_input_values = list(map(int, config['VALID_VALUES']['valid_input'].split(',')))
classification = read_config_classification(config_file)

increment_stat_value(config_file, 'mesa_stat_setup', increment_value=1)

if __name__ == "__main__":
    root = ttk.Window(themename=ttk_bootstrap_theme)
    root.title("Set up processing")
    root.iconbitmap(os.path.join(original_working_directory,"system_resources/mesa.ico"))
    root.geometry("900x800")

    global df_assetgroup
    df_assetgroup = load_data(gpkg_file)

    if df_assetgroup is None:
        log_to_file("Failed to load the GeoDataFrame. Check the GeoPackage file and the data integrity.")
        sys.exit(1)

    canvas, frame, entries = setup_ui_elements(root, df_assetgroup, column_widths)
    update_all_rows_immediately(entries, df_assetgroup)

    info_text = "This is where you register values for importance and susceptibility. Ensure all values are correctly filled."
    info_label = tk.Label(root, text=info_text, wraplength=600, justify="center")
    info_label.pack(padx=10, pady=10)

    close_button = ttk.Button(root, text="Exit", command=close_application, bootstyle=WARNING)
    close_button.pack(side='right', padx=10, pady=10)

    save_button = ttk.Button(root, text="Save to GeoPackage", command=handle_save_to_gpkg, bootstyle=SUCCESS)
    save_button.pack(side='right', padx=10, pady=10)

    save_excel_button = ttk.Button(root, text="Save to Excel", command=handle_save_to_excel, bootstyle=INFO)
    save_excel_button.pack(side='right', padx=10, pady=10)

    load_excel_button = ttk.Button(root, text="Load from Excel", command=handle_load_from_excel, bootstyle=PRIMARY)
    load_excel_button.pack(side='right', padx=10, pady=10)

    root.mainloop()
