import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import pandas as pd
from sqlalchemy import create_engine

# Function to load data from the database
def load_data():
    engine = create_engine(f'sqlite:///{gpkg_file}')
    return pd.read_sql_table('tbl_asset_group', engine)

# Function to save data to the database
def save_data(df):
    try:
        engine = create_engine(f'sqlite:///{gpkg_file}')
        df.to_sql('tbl_asset_group', con=engine, if_exists='replace', index=False)
    except Exception as e:
        print(f"Error saving data: {e}")

# Update record in the DataFrame and save to the database
def update_record():
    try:
        df.at[current_index, 'name_original'] = name_original_var.get()
        df.at[current_index, 'name_fromuser'] = name_fromuser_var.get()
        save_data(df)  # Save changes to the database
    except Exception as e:
        print(f"Error updating and saving record: {e}")

# Navigate through records
def navigate(direction):
    global current_index
    if direction == 'next' and current_index < len(df) - 1:
        current_index += 1
    elif direction == 'previous' and current_index > 0:
        current_index -= 1
    load_record()

# Load a record into the form
def load_record():
    record = df.iloc[current_index]
    name_original_var.set(record['name_original'])
    name_fromuser_var.set(record['name_fromuser'])

# Initialize the main window
root = tk.Tk()
root.title("Edit Asset Groups")

gpkg_file = 'output/mesa.gpkg'
df = load_data()
current_index = 0

# Variables for form fields
name_original_var = tk.StringVar()
name_fromuser_var = tk.StringVar()

# Form fields with larger entry widgets
tk.Label(root, text="Name Original").grid(row=0, column=0, sticky='w')
name_original_entry = tk.Entry(root, textvariable=name_original_var, width=50)
name_original_entry.grid(row=0, column=1, sticky='e')

tk.Label(root, text="Name From User").grid(row=1, column=0, sticky='w')
name_fromuser_entry = tk.Entry(root, textvariable=name_fromuser_var, width=50)
name_fromuser_entry.grid(row=1, column=1, sticky='e')

# Navigation and Update buttons
ttk.Button(root, text="Previous", command=lambda: navigate('previous')).grid(row=2, column=0, padx=5, pady=5)
ttk.Button(root, text="Update and Save Record", command=update_record).grid(row=2, column=1, padx=5, pady=5)
ttk.Button(root, text="Next", command=lambda: navigate('next')).grid(row=2, column=2, padx=5, pady=5)

# Exit button
ttk.Button(root, text="Exit", command=root.destroy).grid(row=3, column=0, columnspan=3, pady=5)

# Load the first record
load_record()

root.mainloop()
