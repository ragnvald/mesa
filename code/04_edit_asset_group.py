import tkinter as tk

import locale
try:
    locale.setlocale(locale.LC_ALL, 'de_DE.utf8')  # For US English, adjust as needed
except locale.Error:
    locale.setlocale(locale.LC_ALL, '') 

from tkinter import ttk
import configparser
import pandas as pd
import datetime
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


# Logging function to write to the GUI log
def write_to_log( message):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted_message = f"{timestamp} - {message}"
    with open("log.txt", "a") as log_file:
        log_file.write(formatted_message + "\n")

# # # # # # # # # # # # # # 
# Core functions

# Function to load data from the database
def load_data():
    engine = create_engine(f'sqlite:///{gpkg_file}')
    return pd.read_sql_table('tbl_asset_group', engine)

# Function to save data to the database
def save_data(df):
    try:
        engine = create_engine(f'sqlite:///{gpkg_file}')
        df.to_sql('tbl_asset_group', con=engine, if_exists='replace', index=False)
        write_to_log("Asset group data saved")
    except Exception as e:
        write_to_log(f"Error saving data: {e}")
        print(f"Error saving data: {e}")

# Function to update record in the DataFrame and save to the database
def update_record(save_message=True):
    try:
        df.at[current_index, 'name_original'] = name_original_var.get()
        df.at[current_index, 'name_gis_assetgroup'] = name_gis_var.get()
        df.at[current_index, 'title_fromuser'] = title_fromuser_var.get()
        save_data(df)  # Save changes to the database
        if save_message:
            write_to_log("Record updated and saved")
    except Exception as e:
        write_to_log(f"Error updating and saving record: {e}")

# Navigate through records
def navigate(direction):
    global current_index
    update_record(save_message=False)  # Save current edits without showing a message
    if direction == 'next' and current_index < len(df) - 1:
        current_index += 1
    elif direction == 'previous' and current_index > 0:
        current_index -= 1
    load_record()


# Load a record into the form
def load_record():
    record = df.iloc[current_index]
    name_original_var.set(record['name_original'])
    name_gis_var.set(record['name_gis_assetgroup'])
    title_fromuser_var.set(record['title_fromuser'])


# Function to close the application
def exit_application():
    write_to_log("Closing edit assets")
    root.destroy()

#####################################################################################
#  Main
#

# Load configuration settings
config_file = 'config.ini'
config = read_config(config_file)
input_folder_asset = config['DEFAULT']['input_folder_asset']
input_folder_geocode = config['DEFAULT']['input_folder_geocode']
gpkg_file = config['DEFAULT']['gpkg_file']
ttk_bootstrap_theme = config['DEFAULT']['ttk_bootstrap_theme']

# Create the user interface
root = ttk.Window(themename=ttk_bootstrap_theme)  # Use ttkbootstrap Window
root.title("Edit assets")

# Configure column widths
root.columnconfigure(0, minsize=200)  # Configure the size of the first column
root.columnconfigure(1, weight=1)     # Make the second column expandable

df = load_data()
current_index = 0

# Variables for form fields
name_original_var = tk.StringVar()
name_gis_var = tk.StringVar()
title_fromuser_var = tk.StringVar()

# GIS name is internal to the system. Can not be edited.
tk.Label(root, text="GIS name").grid(row=0, column=0, sticky='w')
name_gis_label = tk.Label(root, textvariable=name_gis_var, width=50, relief="sunken", anchor="w")
name_gis_label.grid(row=0, column=1, sticky='w')

# Original Name Entry
tk.Label(root, text="Original name").grid(row=1, column=0, sticky='w')
name_original_entry = tk.Entry(root, textvariable=name_original_var, width=50)
name_original_entry.grid(row=1, column=1, sticky='w')

# Title Entry
tk.Label(root, text="Title").grid(row=2, column=0, sticky='w')
title_fromuser_entry = tk.Entry(root, textvariable=title_fromuser_var, width=50)
title_fromuser_entry.grid(row=2, column=1, sticky='w')

# Information text field above the "Update and Save Record" button
info_label_text = ("All assets that are imported are associated with a file "
                   " or table name. This table name is the original name. If "
                   "you want to use a different name in presenting the analysis "
                   "we suggest that you add that name here.")
info_label = tk.Label(root, text=info_label_text, wraplength=400, justify="left")
info_label.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

# Navigation and Update buttons
ttk.Button(root, text="Previous", command=lambda: navigate('previous'), bootstyle=PRIMARY).grid(row=4, column=0, padx=5, pady=5)
ttk.Button(root, text="Save", command=update_record, bootstyle=PRIMARY).grid(row=4, column=1, padx=5, pady=5)
ttk.Button(root, text="Next", command=lambda: navigate('next'), bootstyle=PRIMARY).grid(row=4, column=2, padx=5, pady=5)

# Exit button
ttk.Button(root, text="Exit", command=exit_application, bootstyle='warning').grid(row=5, column=0, columnspan=3, pady=5)

# Load the first record
load_record()

root.mainloop()