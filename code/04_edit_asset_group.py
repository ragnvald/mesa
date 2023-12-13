import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sqlalchemy import create_engine

# Function to load data from the database
def load_data():
    engine = create_engine(f'sqlite:///{gpkg_file}')  # Adjust as per your database
    return pd.read_sql_table('tbl_asset_group', engine)

# Function to save data to the database
def save_data(df):
    try:
        engine = create_engine(f'sqlite:///{gpkg_file}')  # Adjust as per your database
        df.to_sql('tbl_asset_group', con=engine, if_exists='replace', index=False)
        messagebox.showinfo("Success", "Data saved successfully")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save data: {e}")

# Update record in the DataFrame
def update_record():
    try:
        # Only update name_original and name_fromuser fields
        df.at[current_index, 'name_original'] = name_original_var.get()
        df.at[current_index, 'name_fromuser'] = name_fromuser_var.get()
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
    name_original_var.set(record['name_original'])
    name_fromuser_var.set(record['name_fromuser'])
    # Load other fields similarly

# Initialize the main window
root = tk.Tk()
root.title("Edit Asset Groups")

gpkg_file = 'output/mesa.gpkg'
table_name = 'tbl_asset_group'

df = load_data()
current_index = 0

# Variables for form fields
name_original_var = tk.StringVar()
name_fromuser_var = tk.StringVar()
# Define other variables similarly

# Form fields with larger entry widgets
tk.Label(root, text="Name Original").grid(row=0, column=0, sticky='w')
name_original_entry = tk.Entry(root, textvariable=name_original_var, width=50)
name_original_entry.grid(row=0, column=1, sticky='e')

tk.Label(root, text="Name From User").grid(row=1, column=0, sticky='w')
name_fromuser_entry = tk.Entry(root, textvariable=name_fromuser_var, width=50)
name_fromuser_entry.grid(row=1, column=1, sticky='e')
# Create other fields similarly

# Navigation buttons
tk.Button(root, text="Previous", command=lambda: navigate('previous')).grid(row=2, column=0)
tk.Button(root, text="Next", command=lambda: navigate('next')).grid(row=2, column=1)

# Save and Exit buttons
tk.Button(root, text="Update Record", command=update_record).grid(row=3, column=0)
tk.Button(root, text="Save Changes", command=lambda: save_data(df)).grid(row=3, column=1)
tk.Button(root, text="Exit", command=root.destroy).grid(row=4, column=0, columnspan=2)

# Load the first record
load_record()

root.mainloop()
