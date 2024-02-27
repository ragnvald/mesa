# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# This code allows the user to edit tbl_lines using a simple user interface
# allowing the user to move between the different line objects. 
# It does not allow for editing name_gis which will be used for filtering
# in QGIS.
#

import tkinter as tk
import locale

locale.setlocale(locale.LC_ALL, 'C') 

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
    return pd.read_sql_table('tbl_lines', engine)

# Function to save data to the database
def save_data(df):
    try:
        engine = create_engine(f'sqlite:///{gpkg_file}')
        df.to_sql('tbl_lines', con=engine, if_exists='replace', index=False)
        write_to_log("Line data saved")
    except Exception as e:
        write_to_log(f"Error saving data: {e}")
        print(f"Error saving data: {e}")

# Function to update record in the DataFrame and save to the database
def update_record(save_message=True):
    try:
        df.at[current_index, 'name_gis']        = name_gis_var.get()
        df.at[current_index, 'name_user']       = name_user_var.get()
        df.at[current_index, 'segment_length']  = segment_length_var.get()
        df.at[current_index, 'segment_width']   = segment_width_var.get()
        df.at[current_index, 'description']     = description_var.get()
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
    name_gis_var.set(record['name_gis'])
    name_user_var.set(record['name_user'])
    segment_length_var.set(record['segment_length'])
    segment_width_var.set(record['segment_width'])
    description_var.set(record['description'])


# Function to close the application
def exit_application():
    write_to_log("Closing edit assets")
    root.destroy()

#####################################################################################
#  Main
#

# Load configuration settings
config_file             = 'config.ini'
config                  = read_config(config_file)
input_folder_asset      = config['DEFAULT']['input_folder_asset']
input_folder_geocode    = config['DEFAULT']['input_folder_geocode']
gpkg_file               = config['DEFAULT']['gpkg_file']
ttk_bootstrap_theme     = config['DEFAULT']['ttk_bootstrap_theme']
workingprojection_epsg  = config['DEFAULT']['workingprojection_epsg']

# Create the user interface
root = ttk.Window(themename=ttk_bootstrap_theme)  # Use ttkbootstrap Window
root.title("Edit assets")

# Configure column widths
root.columnconfigure(0, minsize=200)  # Configure the size of the first column
root.columnconfigure(1, weight=1)     # Make the second column expandable

df = load_data()
current_index = 0

# Variables for form fields
name_gis_var = tk.StringVar()
name_user_var = tk.StringVar()
segment_length_var = tk.StringVar()
segment_width_var = tk.StringVar()
description_var = tk.StringVar()

# GIS name is internal to the system. Can not be edited.
tk.Label(root, text="GIS name").grid(row=0, column=0, sticky='w')
name_gis_label = tk.Label(root, textvariable=name_gis_var, width=50, relief="sunken", anchor="w")
name_gis_label.grid(row=0, column=1, sticky='w')

# Users reference to the line
tk.Label(root, text="Title").grid(row=1, column=0, sticky='w')
name_user_entry = tk.Entry(root, textvariable=name_user_var, width=50)
name_user_entry.grid(row=1, column=1, sticky='w')

# Write number of segments
tk.Label(root, text="Length of segments").grid(row=2, column=0, sticky='w')
segment_length_entry = tk.Entry(root, textvariable=segment_length_var, width=50)
segment_length_entry.grid(row=2, column=1, sticky='w')

# Width of the segments
tk.Label(root, text="Segments width (meters)").grid(row=3, column=0, sticky='w')
segment_width_entry = tk.Entry(root, textvariable=segment_width_var, width=50)
segment_width_entry.grid(row=3, column=1, sticky='w')

# Users descriptive text of the line
tk.Label(root, text="Description").grid(row=4, column=0, sticky='w')
description_entry = tk.Entry(root, textvariable=description_var, width=50)
description_entry.grid(row=4, column=1, sticky='w')

# Information text field above the "Update and Save Record" button
info_label_text = ("Registred lines will be used to create segmente polygons "
                   "along the designated line. You may edit the attributes "
                   "here, or both attributes and line data in QGIS. Edits "
                   "are saved when you click on the buttons Previous or "
                   "next.")

info_label = tk.Label(root, text=info_label_text, wraplength=400, justify="left")
info_label.grid(row=5, column=0, columnspan=3, padx=10, pady=10)

# Navigation and Update buttons
ttk.Button(root, text="Previous", command=lambda: navigate('previous'), bootstyle=PRIMARY).grid(row=6, column=0, padx=5, pady=5)

ttk.Button(root, text="Next", command=lambda: navigate('next'), bootstyle=PRIMARY).grid(row=6, column=1, padx=5, pady=5)

# Exit button
ttk.Button(root, text="Exit", command=exit_application, bootstyle='warning').grid(row=6, column=2, columnspan=3, pady=5)

# Load the first record
load_record()

root.mainloop()