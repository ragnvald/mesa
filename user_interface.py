import tkinter as tk
from tkinter import messagebox
import subprocess
import webbrowser  # Import the webbrowser module
from PIL import Image, ImageTk

# Function to open a web link
def open_link(url):
    webbrowser.open_new_tab(url)

# Function to load and display the image
def display_image(bottom_frame):
    # Load the image from the specified path
    image_path = 'data_resources/mesa_illustration.png'  # Update this path if necessary
    image = Image.open(image_path)
    
    # Resize the image to 200x200 pixels using LANCZOS filter (previously ANTIALIAS)
    image = image.resize((200, 200), Image.Resampling.LANCZOS)
    
    # Convert the image to a PhotoImage object
    photo = ImageTk.PhotoImage(image)
    
    # Create a Label widget to display the image and center it in the bottom frame
    label = tk.Label(bottom_frame, image=photo)
    label.image = photo  # Keep a reference to the image object
    label.pack(side='bottom', pady=10)  # Center the image

def import_assets():
    try:
        subprocess.run(["python", "01_import_asset_objects.py"], check=True)
        messagebox.showinfo("Import assets", "Import assets script executed.")
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Failed to execute import assets script.")

def import_grids():
    try:
        subprocess.run(["python", "01_import_geocode_objects.py"], check=True)
        messagebox.showinfo("Import grids (geocode objects)", "Import grids script executed.")
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Failed to execute import grids script.")

def edit_susceptibilitiesandimportance():
    try:
        subprocess.run(["python", "04_edit_input.py"], check=True)
        messagebox.showinfo("Edit asset data", "Edit susceptibilities and importance.")
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Failed to edit.")
    
def view_statistics():
    messagebox.showinfo("View Statistics", "View statistics script executed.")

def process_data():
    try:
        subprocess.run(["python", "06_tbl_mesa_stacked.py"], check=True)
        messagebox.showinfo("Process data", "Spatial data processing")
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Failed to process.")

def export_package():
    messagebox.showinfo("Export Package", "Export package script executed.")

def exit_program():
    root.destroy()

# Setup the main Tkinter window
root = tk.Tk()
root.title("MESA 4")

# Set the initial size of the window (width x height)
root.geometry("540x540")  # Adjust the width and height as needed

# Create a top frame for text and links
top_frame = tk.Frame(root)
top_frame.pack(fill='x', expand=False, pady=10)

# Add text and link to the top frame
info_text = tk.Label(top_frame, text="Read more about the MESA method and tools", font=("Arial", 10))
info_text.pack(side='left')

link_text = tk.Label(top_frame, text="here ", font=("Arial", 10, "underline"), fg="blue", cursor="hand2")
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
left_panel.grid_rowconfigure(3, weight=1)  # Extra row for spacing

# Right panel
right_panel = tk.Frame(main_frame)
right_panel.grid(row=0, column=1, sticky="ew", padx=20)
right_panel.grid_rowconfigure(0, weight=1)
right_panel.grid_rowconfigure(1, weight=1)
right_panel.grid_rowconfigure(2, weight=1)
right_panel.grid_rowconfigure(3, weight=1)  # Extra row for spacing

bottom_frame = tk.Frame(root)
bottom_frame.pack(fill='x', expand=False)

# Button width
button_width = 20  # Adjust as needed
button_padx = 20  # Padding around buttons on x-axis
button_pady = 10  # Padding around buttons on y-axis

# Add buttons to left panel with spacing between buttons
import_assets_btn = tk.Button(left_panel, text="Import Assets", command=import_assets, width=button_width)
import_assets_btn.grid(row=0, column=0, padx=button_padx, pady=button_pady)

import_grids_btn = tk.Button(left_panel, text="Import Grids", command=import_grids, width=button_width)
import_grids_btn.grid(row=1, column=0, padx=button_padx, pady=button_pady)

view_statistics_btn = tk.Button(left_panel, text="Set up processing", command=edit_susceptibilitiesandimportance, width=button_width)
view_statistics_btn.grid(row=2, column=0, padx=button_padx, pady=button_pady)

# Call the function to display the image in the bottom frame
display_image(bottom_frame)

# Add buttons to right panel with spacing between buttons
process_data_btn = tk.Button(right_panel, text="Process Data", command=process_data, width=button_width)
process_data_btn.grid(row=0, column=0, padx=button_padx, pady=button_pady)

export_package_btn = tk.Button(right_panel, text="Export Package", command=export_package, width=button_width)
export_package_btn.grid(row=1, column=0, padx=button_padx, pady=button_pady)

# Exit button
exit_btn = tk.Button(main_frame, text="Exit", command=exit_program, width=button_width)
exit_btn.grid(row=1, column=0, columnspan=2, pady=button_pady)

# Start the GUI event loop
root.mainloop()