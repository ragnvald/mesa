import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import subprocess
import webbrowser
from PIL import Image, ImageTk

# Function to open a web link
def open_link(url):
    webbrowser.open_new_tab(url)

# Function to load and display the image
def display_image(bottom_frame):
    image_path = 'system_resources/mesa_illustration.png'
    image = Image.open(image_path)
    image = image.resize((200, 200), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(image)
    label = tk.Label(bottom_frame, image=photo)
    label.image = photo
    label.pack(side='bottom', pady=10)

def import_assets():
    try:
        subprocess.run(["python", "01_import_asset_objects.py"], check=True)
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Failed to execute import assets script.")

def import_grids():
    try:
        subprocess.run(["python", "01_import_geocode_objects.py"], check=True)
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Failed to execute import grids script.")

def edit_asset_group():
    try:
        subprocess.run(["python", "04_edit_asset_group.py"], check=True)
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Failed to execute edit asset group script.")

def edit_geocode_group():
    try:
        subprocess.run(["python", "04_edit_geocode_group.py"], check=True)
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Failed to execute edit geocode group script.")

def edit_susceptibilitiesandimportance():
    try:
        subprocess.run(["python", "04_edit_input.py"], check=True)
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Failed to edit.")

def view_statistics():
    messagebox.showinfo("View Statistics", "View statistics script executed.")

def process_data():
    try:
        subprocess.run(["python", "06_tbl_mesa_stacked.py"], check=True)
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Failed to process.")

def export_package():
    messagebox.showinfo("Export Package", "Export package script executed.")

def exit_program():
    root.destroy()

# Setup the main Tkinter window
root = tk.Tk()
root.title("MESA 4")
root.geometry("700x540")

# Create a top frame for text and links
top_frame = tk.Frame(root)
top_frame.pack(fill='x', expand=False, pady=10)

# Add text and link to the top frame
info_text = tk.Label(top_frame, text="Read more about the MESA method and tools", font=("Calibri", 10))
info_text.pack(side='left')
link_text = tk.Label(top_frame, text="here ", font=("Calibri", 10, "underline"), fg="blue", cursor="hand2")
link_text.pack(side='left')
link_text.bind("<Button-1>", lambda e: open_link("https://www.mesamethod.org/wiki/Main_Page"))

# Main frame
main_frame = tk.Frame(root)
main_frame.pack(fill='both', expand=True, pady=20)
main_frame.grid_columnconfigure(0, weight=1)
main_frame.grid_columnconfigure(1, weight=1)

# Left panel
left_panel = tk.Frame(main_frame)
left_panel.grid(row=0, column=0, sticky="ew", padx=20)
left_panel.grid_rowconfigure(0, weight=1)
left_panel.grid_rowconfigure(1, weight=1)
left_panel.grid_rowconfigure(2, weight=1)
left_panel.grid_rowconfigure(3, weight=1)

# Right panel
right_panel = tk.Frame(main_frame)
right_panel.grid(row=0, column=1, sticky="ew", padx=20)
right_panel.grid_rowconfigure(0, weight=1)
right_panel.grid_rowconfigure(1, weight=1)
right_panel.grid_rowconfigure(2, weight=1)
right_panel.grid_rowconfigure(3, weight=1)

bottom_frame = tk.Frame(root)
bottom_frame.pack(fill='x', expand=False)

button_width = 20
button_padx = 20
button_pady = 10

# Add buttons to left panel with spacing between buttons
import_assets_btn = ttk.Button(left_panel, text="Import assets", command=import_assets, width=button_width)
import_assets_btn.grid(row=0, column=0, padx=button_padx, pady=button_pady)
edit_asset_group_btn = ttk.Button(left_panel, text="Edit asset groups", command=edit_asset_group, width=button_width)
edit_asset_group_btn.grid(row=0, column=1, padx=button_padx, pady=button_pady)

import_grids_btn = ttk.Button(left_panel, text="Import geocodes", command=import_grids, width=button_width)
import_grids_btn.grid(row=1, column=0, padx=button_padx, pady=button_pady)
edit_geocode_group_btn = ttk.Button(left_panel, text="Edit geocode groups", command=edit_geocode_group, width=button_width)
edit_geocode_group_btn.grid(row=1, column=1, padx=button_padx, pady=button_pady)

view_statistics_btn = ttk.Button(left_panel, text="Set up processing", command=edit_susceptibilitiesandimportance, width=button_width)
view_statistics_btn.grid(row=2, column=0, padx=button_padx, pady=button_pady)

# Call the function to display the image in the bottom frame
display_image(bottom_frame)

# Add buttons to right panel with spacing between buttons
process_data_btn = ttk.Button(right_panel, text="Process data", command=process_data, width=button_width)
process_data_btn.grid(row=0, column=0, padx=button_padx, pady=button_pady)

export_package_btn = ttk.Button(right_panel, text="Export package", command=export_package, width=button_width)
export_package_btn.grid(row=1, column=0, padx=button_padx, pady=button_pady)

# Exit button
exit_btn = ttk.Button(main_frame, text="Exit", command=exit_program, width=button_width)
exit_btn.grid(row=1, column=0, columnspan=2, pady=button_pady)

# Start the GUI event loop
root.mainloop()
