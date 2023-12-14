import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
from sqlalchemy import create_engine

# Function to load data from the database
def load_data():
    engine = create_engine(f'sqlite:///{gpkg_file}')  # Adjust as per your database
    return pd.read_sql_table('tbl_geocode_group', engine)

# Function to save data to the database
def save_data():
    try:
        engine = create_engine(f'sqlite:///{gpkg_file}')  # Adjust as per your database
        for record in records:
            df.at[record['id']-1, 'name'] = record['name_var'].get()
            df.at[record['id']-1, 'description'] = record['desc_var'].get()
        df.to_sql('tbl_geocode_group', con=engine, if_exists='replace', index=False)
        messagebox.showinfo("Success", "Data saved successfully")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save data: {e}")

# Function to close the application
def exit_application():
    root.destroy()

# Initialize the main window
root = tk.Tk()
root.title("Edit Geocode Groups")

# Load data
gpkg_file = 'output/mesa.gpkg'
df = load_data()

# Create a frame for the editable fields
edit_frame = tk.Frame(root, padx=5, pady=5)
edit_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Store references to the variables
records = []

# Create labels and entry widgets for each record
for idx, row in df.iterrows():
    tk.Label(edit_frame, text=row['id']).grid(row=idx, column=0, sticky='w')
    
    name_var = tk.StringVar(value=row['name'])
    tk.Entry(edit_frame, textvariable=name_var, width=20).grid(row=idx, column=1, sticky='w')
    
    desc_var = tk.StringVar(value=row['description'])
    tk.Entry(edit_frame, textvariable=desc_var, width=80).grid(row=idx, column=2, sticky='w')

    records.append({'id': row['id'], 'name_var': name_var, 'desc_var': desc_var})

# Information text field
info_label_text = ("This is where I inform the user about relevant stuff. "
                   "It could be 5 sentences long. Here's some important information "
                   "you need to know before using the geocode group editor.")
info_label = tk.Label(root, text=info_label_text, wraplength=300, justify="left")
info_label.pack(padx=10, pady=10)

# Button frame
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# Save button
save_button = ttk.Button(button_frame, text="Save Data", command=save_data)
save_button.pack(side=tk.LEFT, padx=10)

# Exit button
exit_button = ttk.Button(button_frame, text="Exit", command=exit_application)
exit_button.pack(side=tk.LEFT, padx=10)

# Styling buttons (rounded corners)
style = ttk.Style()
style.configure("TButton", padding=6, relief="flat", background="#ccc")

root.mainloop()
