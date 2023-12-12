import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import pandas as pd
from sqlalchemy import create_engine, exc
from sqlalchemy.types import Integer, String, DateTime

# Function to validate integer input
def validate_integer(P):
    if P.isdigit() or P == "":
        return True
    return False

# Function to update sensitivity
def calculate_sensitivity(row_index):
    try:
        susceptibility = int(entries[row_index]['susceptibility'].get())
        importance = int(entries[row_index]['importance'].get())
        sensitivity = susceptibility * importance
        entries[row_index]['sensitivity'].config(text=str(sensitivity))
        df.at[row_index, 'susceptibility'] = susceptibility
        df.at[row_index, 'importance'] = importance
        #df.at[row_index, 'sensitivity'] = sensitivity
        df.at[row_index, 'sensitivity'] = susceptibility * importance
    except ValueError:
        entries[row_index]['sensitivity'].config(text="")

    # Enforce data type after update
    df['susceptibility'] = df['susceptibility'].astype('int64', errors='ignore')
    df['importance'] = df['importance'].astype('int64', errors='ignore')
    df['sensitivity'] = df['sensitivity'].astype('int64', errors='ignore')

# Function to load and refresh data from geopackage
def load_data():
    global df, entries
    engine = create_engine(f'sqlite:///{gpkg_path}')
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", con=engine)
        print("Column names in the DataFrame:", df.columns)  # Print the column names

        # Set the data types for specific columns
        df['susceptibility'] = df['susceptibility'].astype('int64', errors='ignore')
        df['importance'] = df['importance'].astype('int64', errors='ignore')
        df['sensitivity'] = df['sensitivity'].astype('int64', errors='ignore')

    except exc.SQLAlchemyError as e:
        messagebox.showerror("Database Error", f"Failed to load data: {e}")
        return

    for widget in frame.winfo_children():
        widget.destroy()

    add_column_labels()
    entries = []
    for i, row in df.iterrows():
        add_data_row(i, row)

# Function to add column labels
def add_column_labels():
    ttk.Label(frame, text="Dataset").grid(row=0, column=0, padx=5)
    ttk.Label(frame, text="Susceptibility").grid(row=0, column=1, padx=5)
    ttk.Label(frame, text="Importance").grid(row=0, column=2, padx=5)
    ttk.Label(frame, text="Sensitivity").grid(row=0, column=3, padx=5)

# Function to add a data row
def add_data_row(i, row):
    global entries
    ttk.Label(frame, text=row['name_original']).grid(row=i+1, column=0, padx=5)  # Corrected column name

    susceptibility_entry = ttk.Entry(frame, width=5, validate='key', validatecommand=vcmd)
    susceptibility_entry.insert(0, row['susceptibility'])
    susceptibility_entry.grid(row=i+1, column=1, padx=5)
    susceptibility_entry.bind('<KeyRelease>', lambda event, index=i: calculate_sensitivity(index))

    importance_entry = ttk.Entry(frame, width=5, validate='key', validatecommand=vcmd)
    importance_entry.insert(0, row['importance'])
    importance_entry.grid(row=i+1, column=2, padx=5)
    importance_entry.bind('<KeyRelease>', lambda event, index=i: calculate_sensitivity(index))

    sensitivity_label = ttk.Label(frame, text=str(row['sensitivity']))
    sensitivity_label.grid(row=i+1, column=3, padx=5)

    entries.append({
        'susceptibility': susceptibility_entry,
        'importance': importance_entry,
        'sensitivity': sensitivity_label
    })

# Function to save changes to the geopackage
# Function to save changes to the geopackage
def save_to_gpkg():
    engine = create_engine(f'sqlite:///{gpkg_path}')
    try:
        # Convert 'date_import' to datetime format
        df['date_import'] = pd.to_datetime(df['date_import'], errors='coerce')

        # Handle null values if necessary
        # df['date_import'] = df['date_import'].fillna(some_default_date)

        print("Data types before saving:", df.dtypes)  # Check data types before saving

        # Specify SQLAlchemy data types for columns
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
            # Add other columns if needed
        }

        df.to_sql(table_name, con=engine, if_exists='replace', index=False, dtype=data_types)
        messagebox.showinfo("Success", "Data successfully saved to GeoPackage.")
    except exc.SQLAlchemyError as e:
        messagebox.showerror("Database Error", f"Failed to save data: {e}")

# Function to close the application
def close_application():
    root.destroy()

# Initialize the main window with a larger size
root = tk.Tk()
root.title("Data Editor")
root.geometry("600x1000")  # Set window size (width x height)

# Register the validation command
vcmd = (root.register(validate_integer), '%P')

# Create a frame for the data rows
frame = ttk.Frame(root)
frame.pack(padx=10, pady=10, fill="both", expand=True)

# Paths for the geopackage and the table name
gpkg_path = 'output/mesa.gpkg'
table_name = 'tbl_asset_group'

# Load data and populate UI
load_data()

# Add buttons for saving data and closing the application
save_button = ttk.Button(root, text="Save to Geopackage", command=save_to_gpkg)
save_button.pack(side='left', padx=10, pady=10)

refresh_button = ttk.Button(root, text="Refresh Data", command=load_data)
refresh_button.pack(side='left', padx=10, pady=10)

close_button = ttk.Button(root, text="Close", command=close_application)
close_button.pack(side='right', padx=10, pady=10)

# Start the GUI event loop
root.mainloop()