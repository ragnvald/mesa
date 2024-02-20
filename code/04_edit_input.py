import tkinter as tk
import locale

locale.setlocale(locale.LC_ALL, 'C') 

from tkinter import messagebox, ttk
import configparser
import pandas as pd
from sqlalchemy import create_engine, exc
from sqlalchemy.types import Integer, String, DateTime
import ttkbootstrap as ttk  # Import ttkbootstrap
from ttkbootstrap.constants import *

# Setting variables
#
# Define fixed widths for each column
column_widths = [35, 13, 13, 13]

# Shared/general functions
def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config

# Core functions
def validate_integer(P):
    if P.isdigit() or P == "":
        return True
    return False

def calculate_sensitivity(row_index):
    try:
        susceptibility = int(entries[row_index]['susceptibility'].get())
        importance = int(entries[row_index]['importance'].get())
        sensitivity = susceptibility * importance
        entries[row_index]['sensitivity'].config(text=str(sensitivity))
        df.at[row_index, 'susceptibility'] = susceptibility
        df.at[row_index, 'importance'] = importance
        df.at[row_index, 'sensitivity'] = susceptibility * importance
    except ValueError:
        entries[row_index]['sensitivity'].config(text="")

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
    for i, row in df.iterrows():
        add_data_row(i, row)

    frame.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))

def add_data_row(i, row):
    global entries
    ttk.Label(frame, text=row['name_original'], anchor='e').grid(row=i+1, column=0, padx=5, sticky='ew')  # Align right

    susceptibility_entry = ttk.Entry(frame, width=column_widths[1], validate='key', validatecommand=vcmd)
    susceptibility_entry.insert(0, row['susceptibility'])
    susceptibility_entry.grid(row=i+1, column=1, padx=5)
    susceptibility_entry.bind('<KeyRelease>', lambda event, index=i: calculate_sensitivity(index))

    importance_entry = ttk.Entry(frame, width=column_widths[2], validate='key', validatecommand=vcmd)
    importance_entry.insert(0, row['importance'])
    importance_entry.grid(row=i+1, column=2, padx=5)
    importance_entry.bind('<KeyRelease>', lambda event, index=i: calculate_sensitivity(index))

    sensitivity_label = ttk.Label(frame, text=str(row['sensitivity']), width=column_widths[3])
    sensitivity_label.grid(row=i+1, column=3, padx=5)

    entries.append({
        'susceptibility': susceptibility_entry,
        'importance': importance_entry,
        'sensitivity': sensitivity_label
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


#####################################################################################
#  Main
#

# Load configuration settings and data
config_file = 'config.ini'
config = read_config(config_file)
gpkg_file = config['DEFAULT']['gpkg_file']
table_name = 'tbl_asset_group'
ttk_bootstrap_theme = config['DEFAULT']['ttk_bootstrap_theme']

# Initialize the main window
root = ttk.Window(themename=ttk_bootstrap_theme)
root.title("Set up processing")
root.geometry("700x700")

vcmd = (root.register(validate_integer), '%P')

# Header frame setup
header_frame = ttk.Frame(root)
header_frame.pack(side=tk.TOP, fill="x")

# Create and grid header labels with fixed widths
headers = ["Dataset", "Susceptibility", "Importance", "Sensitivity"]
for i, (header, width) in enumerate(zip(headers, column_widths)):
    label = ttk.Label(header_frame, text=header, width=width)
    label.grid(row=0, column=i, padx=5, sticky='ew')

# Add column labels in the header_frame with centered alignment
ttk.Label(header_frame, text="Dataset").grid(row=0, column=0, padx=5, sticky='nsew')
ttk.Label(header_frame, text="Susceptibility").grid(row=0, column=1, padx=5, sticky='nsew')
ttk.Label(header_frame, text="Importance").grid(row=0, column=2, padx=5, sticky='nsew')
ttk.Label(header_frame, text="Sensitivity").grid(row=0, column=3, padx=5, sticky='nsew')

# Configure the column weights to ensure they expand and allow centering
header_frame.grid_columnconfigure(0, weight=1)
header_frame.grid_columnconfigure(1, weight=1)
header_frame.grid_columnconfigure(2, weight=1)
header_frame.grid_columnconfigure(3, weight=1)

# Create scrollable area below the header
canvas = create_scrollable_area(root)
frame = ttk.Frame(canvas)
canvas.create_window((0, 0), window=frame, anchor="nw")

load_data()

# Text panel and buttons below the scrollable area
info_text = "This is where you register values for susceptibility and importance."
info_label = tk.Label(root, text=info_text, wraplength=400, justify="center")
info_label.pack(padx=10, pady=10)

save_button = ttk.Button(root, text="Save", command=save_to_gpkg, bootstyle=PRIMARY)
save_button.pack(side='left', padx=10, pady=10)

close_button = ttk.Button(root, text="Close", command=close_application, bootstyle=WARNING)
close_button.pack(side='right', padx=10, pady=10)

root.mainloop()