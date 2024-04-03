import tkinter as tk
import locale
from tkinter import messagebox, ttk
import configparser
import pandas as pd
from sqlalchemy import create_engine, exc
from sqlalchemy.types import Integer, String, DateTime
import ttkbootstrap as ttk  # Import ttkbootstrap
from ttkbootstrap.constants import *
import os

# Setting variables
#
# Define fixed widths for each column
column_widths = [35, 13, 13, 13]

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
    global classification  # This line is crucial
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


# Updated validation function
def validate_input_value(P):
    if P.isdigit() and int(P) in valid_input_values or P == "":
        return True
    return False


def determine_category(sensitivity):
    for category, info in classification.items():
        if sensitivity in info['range']:
            return category, info['description']
    print("No match found.")
    return '', ''  


def calculate_sensitivity(row_index):
    try:
        susceptibility = int(entries[row_index]['susceptibility'].get())
        importance = int(entries[row_index]['importance'].get())
        sensitivity = susceptibility * importance
        code, description = determine_category(sensitivity)  # Function to implement

        # Update the GUI elements
        entries[row_index]['sensitivity'].config(text=str(sensitivity))
        entries[row_index]['code'].config(text=code)
        entries[row_index]['description'].config(text=description)

        # Update DataFrame if necessary
        df.at[row_index, 'sensitivity'] = sensitivity
        df.at[row_index, 'code'] = code
        df.at[row_index, 'description'] = description

    except ValueError:
        entries[row_index]['sensitivity'].config(text="")
        entries[row_index]['code'].config(text="")
        entries[row_index]['description'].config(text="")

    df['susceptibility'] = df['susceptibility'].astype('int64', errors='ignore')
    df['importance'] = df['importance'].astype('int64', errors='ignore')
    df['sensitivity'] = df['sensitivity'].astype('int64', errors='ignore')


def load_data():
    global df, entries
    engine = create_engine(f'sqlite:///{gpkg_file}')
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", con=engine)

        df['susceptibility'] = df['susceptibility'].astype('int64', errors='ignore')
        df['importance'] = df['importance'].astype('int64', errors='ignore')
        df['sensitivity'] = df['sensitivity'].astype('int64', errors='ignore')

    except exc.SQLAlchemyError as e:
        messagebox.showerror("Database Error", f"Failed to load data: {e}")
        return

    for widget in frame.winfo_children():
        widget.destroy()

    entries = []

    # Add text labels as the first row in the frame
    ttk.Label(frame, text="Dataset", anchor='w').grid(row=0, column=0, padx=5, sticky='ew')
    ttk.Label(frame, text="Susceptibility", anchor='w').grid(row=0, column=1, padx=5, sticky='ew')
    ttk.Label(frame, text="Importance", anchor='w').grid(row=0, column=2, padx=5, sticky='ew')
    ttk.Label(frame, text="Sensitivity", anchor='w').grid(row=0, column=3, padx=5, sticky='ew')
    ttk.Label(frame, text="Code", anchor='w').grid(row=0, column=4, padx=5, sticky='ew')
    ttk.Label(frame, text="Description", anchor='w').grid(row=0, column=5, padx=5, sticky='ew')

    for i, row in enumerate(df.itertuples(), start=1):  # Adjust to use itertuples() for efficiency
        add_data_row(i, row)

    frame.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))

def add_data_row(i, row):
    global entries
    ttk.Label(frame, text=getattr(row, 'name_original', ''), anchor='e').grid(row=i, column=0, padx=5, sticky='ew')

    susceptibility_entry = ttk.Entry(frame, width=column_widths[1], validate='key', validatecommand=vcmd)
    susceptibility_entry.insert(0, getattr(row, 'susceptibility', ''))
    susceptibility_entry.grid(row=i, column=1, padx=5)
    susceptibility_entry.bind('<KeyRelease>', lambda event, index=i-1: calculate_sensitivity(index))  # Adjusted index for zero-based

    importance_entry = ttk.Entry(frame, width=column_widths[2], validate='key', validatecommand=vcmd)
    importance_entry.insert(0, getattr(row, 'importance', ''))
    importance_entry.grid(row=i, column=2, padx=5)
    importance_entry.bind('<KeyRelease>', lambda event, index=i-1: calculate_sensitivity(index))  # Adjusted index for zero-based

    sensitivity_label = ttk.Label(frame, text=str(getattr(row, 'sensitivity', '')), width=column_widths[3])
    sensitivity_label.grid(row=i, column=3, padx=5)
    
    code_label = ttk.Label(frame, width=5)
    code_label.grid(row=i, column=4, padx=5)
    
    description_label = ttk.Label(frame, width=30)
    description_label.grid(row=i, column=5, padx=5)

    entries.append({
        'susceptibility': susceptibility_entry,
        'importance': importance_entry,
        'sensitivity': sensitivity_label,
        'code': code_label,
        'description': description_label
    })

def save_to_gpkg():
    engine = create_engine(f'sqlite:///{gpkg_file}')
    try:
        df['date_import'] = pd.to_datetime(df['date_import'], errors='coerce')

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
        }

        df.to_sql(table_name, con=engine, if_exists='replace', index=False, dtype=data_types)
    except exc.SQLAlchemyError as e:
        messagebox.showerror("Database Error", f"Failed to save data: {e}")

def close_application():
    root.destroy()

def create_scrollable_area(root):
    # Create a new frame to contain the canvas and the scrollbar
    scrollable_frame = ttk.Frame(root)

    # Create the canvas and scrollbar
    canvas = tk.Canvas(scrollable_frame)
    scrollbar_y = ttk.Scrollbar(scrollable_frame, orient="vertical", command=canvas.yview)

    canvas.configure(yscrollcommand=scrollbar_y.set)
    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    # Pack canvas and scrollbar in the scrollable_frame
    canvas.pack(side=tk.LEFT, fill="both", expand=True)
    scrollbar_y.pack(side=tk.RIGHT, fill="y")

    # Pack the scrollable_frame in the root
    scrollable_frame.pack(side=tk.TOP, fill="both", expand=True)

    return canvas

def increment_stat_value(config_file, stat_name, increment_value):
    # Check if the config file exists
    if not os.path.isfile(config_file):
        print(f"Configuration file {config_file} not found.")
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
                    print(f"Error: Current value of {stat_name} is not an integer.")
                    return
    
    # Write the updated content back to the file if the variable was found and updated
    if updated:
        with open(config_file, 'w') as file:
            file.writelines(lines)


def update_all_rows_immediately():
    for index, entry in enumerate(entries):
        calculate_sensitivity(index)

#####################################################################################
#  Main
#

# Load configuration settings and data
config_file             = 'config.ini'
config                  = read_config(config_file)
gpkg_file               = config['DEFAULT']['gpkg_file']
table_name              = 'tbl_asset_group'
ttk_bootstrap_theme     = config['DEFAULT']['ttk_bootstrap_theme']
workingprojection_epsg  = config['DEFAULT']['workingprojection_epsg']

read_config_classification(config_file)

increment_stat_value(config_file, 'mesa_stat_setup', increment_value=1)

# Initialize the main window
root = ttk.Window(themename=ttk_bootstrap_theme)
root.title("Set up processing")
root.geometry("900x800")

vcmd = (root.register(validate_input_value), '%P')

# Create scrollable area below the header
canvas = create_scrollable_area(root)
frame = ttk.Frame(canvas)
canvas.create_window((0, 0), window=frame, anchor="nw")

load_data()

update_all_rows_immediately()

# Text panel and buttons below the scrollable area
info_text = "This is where you register values for susceptibility and importance. Tabulate through the table to make sure sensitivity is calulated properly."
info_label = tk.Label(root, text=info_text, wraplength=600, justify="center")
info_label.pack(padx=10, pady=10)

save_button = ttk.Button(root, text="Save", command=save_to_gpkg, bootstyle=PRIMARY)
save_button.pack(side='left', padx=10, pady=10)

close_button = ttk.Button(root, text="Exit", command=close_application, bootstyle=WARNING)
close_button.pack(side='right', padx=10, pady=10)

root.mainloop()