import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sqlalchemy import create_engine

# Function to load data from the database
def load_data():
    engine = create_engine(f'sqlite:///{gpkg_file}')  # Adjust as per your database
    return pd.read_sql_table('tbl_geocode_group', engine)

# Function to save data to the database
def save_data(df):
    try:
        engine = create_engine(f'sqlite:///{gpkg_file}')  # Adjust as per your database
        df.to_sql('tbl_geocode_group', con=engine, if_exists='replace', index=False)
        messagebox.showinfo("Success", "Data saved successfully")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save data: {e}")

# Update record in the DataFrame
def update_record():
    try:
        # Only update name and description fields
        df.at[current_index, 'name'] = geocode_name_var.get()
        df.at[current_index, 'description'] = description_text.get("1.0", tk.END).strip()  # Retrieve text from Text widget
        messagebox.showinfo("Success", "Record updated")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to update record: {e}")

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
    geocode_name_var.set(record['name'])
    description_text.delete("1.0", tk.END)  # Clear existing text
    description_text.insert(tk.END, record['description'])  # Insert new text

# Initialize the main window
root = tk.Tk()
root.title("Edit Geocode Groups")

gpkg_file = 'output/mesa.gpkg'
table_name = 'tbl_geocode_group'

df = load_data()
current_index = 0

# Variables for form fields
geocode_name_var = tk.StringVar()

# Form fields with larger entry widgets for geocode name
tk.Label(root, text="Geocode Name").grid(row=0, column=0, sticky='w')
geocode_name_entry = tk.Entry(root, textvariable=geocode_name_var, width=50)
geocode_name_entry.grid(row=0, column=1, sticky='e')

# Text widget for description with scrollbar
tk.Label(root, text="Description").grid(row=1, column=0, sticky='nw')
description_text = tk.Text(root, width=50, height=4, wrap='word')
description_text.grid(row=1, column=1, sticky='e')
scroll = tk.Scrollbar(root, command=description_text.yview)
scroll.grid(row=1, column=2, sticky='nsew')
description_text['yscrollcommand'] = scroll.set

# Navigation buttons
tk.Button(root, text="Previous", command=lambda: navigate('previous')).grid(row=2, column=0)
tk.Button(root, text="Next", command=lambda: navigate('next')).grid(row=2, column=1)

# Save and Exit buttons
tk.Button(root, text="Save Changes", command=lambda: save_data(df)).grid(row=3, column=1)
tk.Button(root, text="Exit", command=root.destroy).grid(row=4, column=0, columnspan=2)

# Load the first record
load_record()

root.mainloop()
