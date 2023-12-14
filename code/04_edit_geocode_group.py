import tkinter as tk
from tkinter import messagebox, ttk
import geopandas as gpd
from sqlalchemy import create_engine

def update_records():
    global df, records
    for record in records:
        row_id = record['id']
        name = record['name_var'].get()
        description = record['desc_var'].get()

        # Update the DataFrame
        df.loc[df['id'] == row_id, 'name'] = name
        df.loc[df['id'] == row_id, 'description'] = description

def save_changes():
    update_records()
    save_spatial_data()

# Function to load spatial data from the database
def load_spatial_data(gpkg_file):
    engine = create_engine(f'sqlite:///{gpkg_file}')  # Adjust as per your database
    # Use Geopandas to load a GeoDataFrame
    gdf = gpd.read_file(gpkg_file, layer='tbl_geocode_group')
    return gdf


# Function to save spatial data to the database
def save_spatial_data():
    global df  # Access the global DataFrame
    try:
        engine = create_engine(f'sqlite:///{gpkg_file}')  # Adjust as per your database
        # Use Geopandas to save the GeoDataFrame
        df.to_file(gpkg_file, layer='tbl_geocode_group', driver='GPKG', if_exists='replace')
        messagebox.showinfo("Success", "Spatial data saved successfully")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save spatial data: {e}")


# Your existing code for creating the Tkinter interface would remain the same
# Just ensure to call load_spatial_data() instead of load_data()
# and save_spatial_data(gdf) instead of save_data()


# Function to close the application
def exit_application():
    root.destroy()

# Initialize the main window
root = tk.Tk()
root.title("Edit Geocode Groups")

# Load data
gpkg_file = 'output/mesa.gpkg'
df = load_spatial_data(gpkg_file)

# Create a frame for the editable fields
edit_frame = tk.Frame(root, padx=5, pady=5)
edit_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Store references to the variables
records = []

# Create labels and entry widgets for each record
for idx, row in df.iterrows():
    tk.Label(edit_frame, text=row['id']).grid(row=idx, column=0, sticky='w')
    
    name_var = tk.StringVar(value=row['name'])
    tk.Entry(edit_frame, textvariable=name_var, width=30).grid(row=idx, column=1, sticky='w')
    
    desc_var = tk.StringVar(value=row['description'])
    tk.Entry(edit_frame, textvariable=desc_var, width=70).grid(row=idx, column=2, sticky='w')

    records.append({'id': row['id'], 'name_var': name_var, 'desc_var': desc_var})

# Information text field
info_label_text = ("After you have imported the geocodes you might want to "
                   "adjust the name of the geocode. Here you can make changes "
                   "to both the name and the description. The name will be "
                   "used in exports to QGIS. Name and description will probably "
                   "be used in a PDF report which this system will generate. ")
info_label = tk.Label(root, text=info_label_text, wraplength=600, justify="left")
info_label.pack(padx=10, pady=10)

# Button frame
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# Save button
save_button = ttk.Button(button_frame, text="Save Data", command=save_changes)
save_button.pack(side=tk.LEFT, padx=10)

# Exit button
exit_button = ttk.Button(button_frame, text="Exit", command=exit_application)
exit_button.pack(side=tk.LEFT, padx=10)

# Styling buttons (rounded corners)
style = ttk.Style()
style.configure("TButton", padding=6, relief="flat", background="#ccc")

root.mainloop()
