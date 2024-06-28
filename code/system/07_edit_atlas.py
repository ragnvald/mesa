import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import configparser
import pandas as pd
from sqlalchemy import create_engine
import os
import argparse
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
# Database management functions

# Function to load data from the database
def load_data():
    engine = create_engine(f'sqlite:///{gpkg_file}')
    return pd.read_sql_table('tbl_atlas', engine)

# Function to save data to the database
def save_data(df):
    try:
        engine = create_engine(f'sqlite:///{gpkg_file}')
        df.to_sql('tbl_atlas', con=engine, if_exists='replace', index=False)
    except Exception as e:
        messagebox.showerror("Error", f"Error saving data: {e}")

# Function to update record in the DataFrame and save to the database
def update_record(save_message=False):
    try:
        df.at[current_index, 'name_gis'] = name_gis_var.get()
        df.at[current_index, 'title_user'] = title_user_var.get()
        df.at[current_index, 'description'] = description_var.get()
        df.at[current_index, 'image_name_1'] = image_name_1_var.get()
        df.at[current_index, 'image_desc_1'] = image_desc_1_var.get()
        df.at[current_index, 'image_name_2'] = image_name_2_var.get()
        df.at[current_index, 'image_desc_2'] = image_desc_2_var.get()
        save_data(df)  # Save changes to the database
        if save_message:
            messagebox.showinfo("Info", "Record updated and saved")
    except Exception as e:
        messagebox.showerror("Error", f"Error updating and saving record: {e}")

# Navigate through records
def navigate(direction):
    update_record()
    global current_index
    if direction == 'next' and current_index < len(df) - 1:
        current_index += 1
    elif direction == 'previous' and current_index > 0:
        current_index -= 1
    load_record()

# Load a record into the form
def load_record():
    record = df.iloc[current_index]
    name_gis_var.set(record['name_gis'])
    title_user_var.set(record['title_user'])
    description_var.set(record['description'])
    image_name_1_var.set(record['image_name_1'])
    image_desc_1_var.set(record['image_desc_1'])
    image_name_2_var.set(record['image_name_2'])
    image_desc_2_var.set(record['image_desc_2'])

# Function to browse for image file for image_name_1
def browse_image_1():
    initial_dir = os.path.join(os.getcwd(), "input", "images")
    file_path = filedialog.askopenfilename(initialdir=initial_dir, title="Select file", filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")))
    if file_path:
        image_name_1_var.set(file_path)

# Function to browse for image file for image_name_2
def browse_image_2():
    initial_dir = os.path.join(os.getcwd(), "input", "images")
    file_path = filedialog.askopenfilename(initialdir=initial_dir, title="Select file", filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")))
    if file_path:
        image_name_2_var.set(file_path)

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

#####################################################################################
#  Main
#
# original folder for the system is sent from the master executable. If the script is
# invked this way we are fetching the adress here.
parser = argparse.ArgumentParser(description='Slave script')
parser.add_argument('--original_working_directory', required=False, help='Path to running folder')
args = parser.parse_args()
original_working_directory = args.original_working_directory

# However - if this is not the case we will have to establish the root folder in 
# one of two different ways.
if original_working_directory is None or original_working_directory == '':
    
    #if it is running as a python subprocess we need to get the originating folder.
    original_working_directory  = os.getcwd()

    # When running directly separate script we need to find out and go up one level.
    if str("system") in str(original_working_directory):
        original_working_directory = os.path.join(os.getcwd(),'../')

# Load configuration settings and data
config_file             = os.path.join(original_working_directory, "system/config.ini")
gpkg_file               = os.path.join(original_working_directory, "output/mesa.gpkg")

# Load configuration settings
config                  = read_config(config_file)

ttk_bootstrap_theme     = config['DEFAULT']['ttk_bootstrap_theme']
workingprojection_epsg  = config['DEFAULT']['workingprojection_epsg']

increment_stat_value(config_file, 'mesa_stat_edit_atlas', increment_value=1)

if __name__ == "__main__":
    
    # Initialize the main window
    root = ttk.Window(themename=ttk_bootstrap_theme)
    root.title("Edit atlas")
    root.iconbitmap(os.path.join(original_working_directory,"system_resources/mesa.ico"))
    
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Configure column widths
    root.columnconfigure(0, minsize=200)
    root.columnconfigure(1, weight=1)

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky="nsew")

    df = load_data()
    current_index = 0

    # Variables for form fields
    name_gis_var     = tk.StringVar()
    title_user_var   = tk.StringVar()
    description_var  = tk.StringVar()
    image_name_1_var = tk.StringVar()
    image_desc_1_var = tk.StringVar()
    image_name_2_var = tk.StringVar()
    image_desc_2_var = tk.StringVar()

    # GIS name (read-only)
    tk.Label(main_frame, text="GIS Name").grid(row=0, column=0, sticky='w')
    name_gis_label = tk.Label(main_frame, textvariable=name_gis_var, width=40, relief="sunken", anchor="w")
    name_gis_label.grid(row=0, column=1, sticky='w', padx=10, pady=10)

    # Title User Entry
    tk.Label(main_frame, text="Title").grid(row=1, column=0, sticky='w')
    title_user_entry = tk.Entry(main_frame, textvariable=title_user_var, width=40)
    title_user_entry.grid(row=1, column=1, sticky='w', padx=10, pady=10)

    # Image Name 1 Entry and Browse Button
    tk.Label(main_frame, text="Image Name 1").grid(row=2, column=0, sticky='w')
    image_name_1_entry = tk.Entry(main_frame, textvariable=image_name_1_var, width=40)
    image_name_1_entry.grid(row=2, column=1, sticky='w', padx=10, pady=10)

    browse_btn_1 = ttk.Button(main_frame, text="Browse", command=browse_image_1)
    browse_btn_1.grid(row=2, column=2, padx=10, pady=10)

    # Image Description 1 Entry
    tk.Label(main_frame, text="Image 1 description").grid(row=3, column=0, sticky='w')
    image_desc_1_entry = tk.Entry(main_frame, textvariable=image_desc_1_var, width=40)
    image_desc_1_entry.grid(row=3, column=1, sticky='w', padx=10, pady=10)

    # Image Name 2 Entry and Browse Button
    tk.Label(main_frame, text="Image Name 2").grid(row=4, column=0, sticky='w')
    image_name_2_entry = tk.Entry(main_frame, textvariable=image_name_2_var, width=40)
    image_name_2_entry.grid(row=4, column=1, sticky='w', padx=10, pady=10)

    browse_btn_2 = ttk.Button(main_frame, text="Browse", command=browse_image_2)
    browse_btn_2.grid(row=4, column=2, padx=10, pady=10)

    # Image Description 1 Entry
    tk.Label(main_frame, text="Image description").grid(row=5, column=0, sticky='w')
    image_desc_1_entry = tk.Entry(main_frame, textvariable=image_desc_2_var, width=40)
    image_desc_1_entry.grid(row=5, column=1, sticky='w', padx=10, pady=10)

    # Description Entry
    tk.Label(main_frame, text="Description").grid(row=6, column=0, sticky='w')
    description_entry = tk.Entry(main_frame, textvariable=description_var, width=40)
    description_entry.grid(row=6, column=1, sticky='w', padx=10, pady=10)

    # Navigation and Update buttons
    ttk.Button(main_frame, text="Previous", command=lambda: navigate('previous')).grid(row=7, column=0, sticky='w')
    ttk.Button(main_frame, text="Next", command=lambda: navigate('next')).grid(row=7, column=2, padx=10, pady=10, sticky='e')

    # Update button
    ttk.Button(main_frame, text="Save", command=update_record, bootstyle=SUCCESS).grid(row=8, column=2, sticky='e', padx=10, pady=10)

    # Exit button
    ttk.Button(main_frame, text="Exit", command=root.destroy, bootstyle=WARNING).grid(row=8, column=3, sticky='e', padx=10, pady=10)

    # Load the first record
    load_record()

    root.mainloop()
