import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import subprocess
import webbrowser
import os
from PIL import Image, ImageTk

# Function to check and create folders
def check_and_create_folders():
    folders = ["input/code", "input/grid", "output"]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

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
        subprocess.run(["python", "01_import.py"], check=True)
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Failed to execute import assets script.")

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
        subprocess.run(["python", "06_process.py"], check=True)
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Failed to process.")


def export_package():
    messagebox.showinfo("Export Package", "Export package script executed.")

def exit_program():
    root.destroy()

# Check and create folders at the beginning
check_and_create_folders()

# Setup the main Tkinter window
root = tk.Tk()
root.title("MESA 4")
root.geometry("700x540")

# Main frame
main_frame = tk.Frame(root)
main_frame.pack(fill='both', expand=True, pady=20)
main_frame.grid_columnconfigure(0, weight=1)
main_frame.grid_columnconfigure(1, weight=0)  # Adjust for the separator
main_frame.grid_columnconfigure(2, weight=1)

# Left panel
left_panel = tk.Frame(main_frame)
left_panel.grid(row=0, column=0, sticky="nsew", padx=20)
left_panel.grid_rowconfigure(0, weight=1)
left_panel.grid_rowconfigure(1, weight=1)
left_panel.grid_rowconfigure(2, weight=1)
left_panel.grid_rowconfigure(3, weight=1)

# Separator
separator = ttk.Separator(main_frame, orient='vertical')
separator.grid(row=0, column=1, sticky='ns')

# Right panel
right_panel = tk.Frame(main_frame)
right_panel.grid(row=0, column=2, sticky="nsew", padx=20)
right_panel.grid_rowconfigure(0, weight=1)

# Label with text on the right panel
mesa_text = "This is the MESA system. It is an implementation of the MESA method."
mesa_label = tk.Label(right_panel, text=mesa_text, justify="left", anchor="nw")
mesa_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

bottom_frame = tk.Frame(root)
bottom_frame.pack(fill='x', expand=False)

button_width = 20
button_padx = 20
button_pady = 10

# Add buttons to left panel with spacing between buttons
import_assets_btn = ttk.Button(left_panel, text="Import", command=import_assets, width=button_width)
import_assets_btn.grid(row=0, column=0, padx=button_padx, pady=button_pady)

edit_asset_group_btn = ttk.Button(left_panel, text="Edit asset groups", command=edit_asset_group, width=button_width)
edit_asset_group_btn.grid(row=0, column=1, padx=button_padx, pady=button_pady)

edit_geocode_group_btn = ttk.Button(left_panel, text="Edit geocode groups", command=edit_geocode_group, width=button_width)
edit_geocode_group_btn.grid(row=1, column=1, padx=button_padx, pady=button_pady)

view_statistics_btn = ttk.Button(left_panel, text="Set up processing", command=edit_susceptibilitiesandimportance, width=button_width)
view_statistics_btn.grid(row=2, column=0, padx=button_padx, pady=button_pady)

process_stacked_data_btn = ttk.Button(left_panel, text="Process data", command=process_data, width=button_width)
process_stacked_data_btn.grid(row=3, column=0, padx=button_padx, pady=button_pady)

export_package_btn = ttk.Button(left_panel, text="Export QGIS file", command=export_package, width=button_width)
export_package_btn.grid(row=4, column=0, padx=button_padx, pady=button_pady)


# Exit button
exit_btn = ttk.Button(left_panel, text="Exit", command=exit_program, width=button_width)
exit_btn.grid(row=5, column=0, columnspan=2, pady=button_pady)

# Call the function to display the image in the bottom frame
display_image(bottom_frame)

# Start the GUI event loop
root.mainloop()
