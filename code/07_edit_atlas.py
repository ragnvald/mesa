import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk, filedialog
import configparser
import pandas as pd
from sqlalchemy import create_engine
import os

# Read the configuration file
def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config

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
def update_record(save_message=True):
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

# Initialize the main window
root = tk.Tk()
root.title("Edit Atlas Records")

# Configure column widths
root.columnconfigure(0, minsize=200)
root.columnconfigure(1, weight=1)

# Load configuration settings
config_file = 'config.ini'
config = read_config(config_file)
gpkg_file = config['DEFAULT']['gpkg_file']

df = load_data()
current_index = 0

# Variables for form fields
name_gis_var = tk.StringVar()
title_user_var = tk.StringVar()
description_var = tk.StringVar()
image_name_1_var = tk.StringVar()
image_desc_1_var = tk.StringVar()
image_name_2_var = tk.StringVar()
image_desc_2_var = tk.StringVar()

# GIS name (read-only)
tk.Label(root, text="GIS Name").grid(row=0, column=0, sticky='w')
name_gis_label = tk.Label(root, textvariable=name_gis_var, width=50, relief="sunken", anchor="w")
name_gis_label.grid(row=0, column=1, sticky='w')

# Title User Entry
tk.Label(root, text="Title").grid(row=1, column=0, sticky='w')
title_user_entry = tk.Entry(root, textvariable=title_user_var, width=50)
title_user_entry.grid(row=1, column=1, sticky='w')

# Description Entry
tk.Label(root, text="Description").grid(row=2, column=0, sticky='w')
description_entry = tk.Entry(root, textvariable=description_var, width=50)
description_entry.grid(row=2, column=1, sticky='w')

# Image Name 1 Entry and Browse Button
tk.Label(root, text="Image Name 1").grid(row=3, column=0, sticky='w')
image_name_1_entry = tk.Entry(root, textvariable=image_name_1_var, width=40)
image_name_1_entry.grid(row=3, column=1, sticky='w')
browse_btn_1 = ttk.Button(root, text="Browse", command=browse_image_1)
browse_btn_1.grid(row=3, column=2)

# Image Description 1 Entry
tk.Label(root, text="Image Description 1").grid(row=4, column=0, sticky='w')
image_desc_1_entry = tk.Entry(root, textvariable=image_desc_1_var, width=50)
image_desc_1_entry.grid(row=4, column=1, sticky='w')

# Image Name 2 Entry and Browse Button
tk.Label(root, text="Image Name 2").grid(row=5, column=0, sticky='w')
image_name_2_entry = tk.Entry(root, textvariable=image_name_2_var, width=40)
image_name_2_entry.grid(row=5, column=1, sticky='w')
browse_btn_2 = ttk.Button(root, text="Browse", command=browse_image_2)
browse_btn_2.grid(row=5, column=2)

# Image Description 2 Entry
tk.Label(root, text="Image Description 2").grid(row=6, column=0, sticky='w')
image_desc_2_entry = tk.Entry(root, textvariable=image_desc_2_var, width=50)
image_desc_2_entry.grid(row=6, column=1, sticky='w')

# Navigation and Update buttons
ttk.Button(root, text="Previous", command=lambda: navigate('previous')).grid(row=7, column=0, padx=5, pady=5)
ttk.Button(root, text="Save", command=update_record).grid(row=7, column=1, padx=5, pady=5)
ttk.Button(root, text="Next", command=lambda: navigate('next')).grid(row=7, column=2, padx=5, pady=5)

# Exit button
ttk.Button(root, text="Exit", command=root.destroy).grid(row=8, column=0, columnspan=3, pady=5)

# Load the first record
load_record()

root.mainloop()